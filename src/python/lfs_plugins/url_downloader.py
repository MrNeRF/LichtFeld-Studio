# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""URL downloader with cloud provider support."""

from __future__ import annotations

import io
import logging
import os
import re
import shutil
import tarfile
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from .http import urlopen

logger = logging.getLogger(__name__)

HTTP_USER_AGENT = "LichtFeld-AssetManager/1.0"


def _raise_if_cancelled(should_cancel: Optional[Callable[[], bool]]) -> None:
    """Raise when the caller requested cancellation."""
    if should_cancel and should_cancel():
        raise InterruptedError("Download cancelled")


def _strip_all_extensions(filename: str) -> str:
    """Strip all extensions from filename (e.g., data.tar.gz -> data)."""
    name = filename
    # Keep stripping extensions until no more
    while "." in name:
        new_name = os.path.splitext(name)[0]
        if new_name == name or not new_name:
            break
        name = new_name
    return name or filename

# Optional imports (cloud providers)
try:
    import boto3
    from botocore.config import Config
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False
    boto3 = None  # type: ignore

try:
    from google.cloud import storage
    HAS_GCS = True
except ImportError:
    HAS_GCS = False
    storage = None  # type: ignore

try:
    import py7zr
    HAS_PY7ZR = True
except ImportError:
    HAS_PY7ZR = False
    py7zr = None  # type: ignore


class URLDownloadError(Exception):
    """Raised when a URL download fails."""
    pass


class UnsupportedURLError(URLDownloadError):
    """Raised when the URL type is not supported."""
    pass


class ExtractError(Exception):
    """Raised when archive extraction fails."""
    pass


def detect_url_type(url: str) -> str:
    """Detect the type of URL.
    
    Returns one of: 's3', 'gcs', 'r2', 'dropbox', 'huggingface', 'github', 'http', 'manifest'
    
    Args:
        url: The URL to analyze
        
    Returns:
        String identifier for the URL type
    """
    url_lower = url.lower().strip()
    
    # S3 URLs
    if url_lower.startswith("s3://"):
        return "s3"
    if ".s3.amazonaws.com" in url_lower or ".s3." in url_lower:
        return "s3"
    
    # R2 (Cloudflare) URLs - S3-compatible
    if ".r2.cloudflarestorage.com" in url_lower:
        return "r2"
    
    # GCS URLs
    if url_lower.startswith("gs://"):
        return "gcs"
    if "storage.googleapis.com" in url_lower:
        return "gcs"
    
    # Dropbox
    if "dropbox.com" in url_lower or "dropboxusercontent.com" in url_lower:
        return "dropbox"
    
    # HuggingFace
    if "huggingface.co" in url_lower or "hf.co" in url_lower:
        return "huggingface"
    
    # GitHub releases
    if "github.com" in url_lower and "/releases/download/" in url_lower:
        return "github"
    
    # Manifest files (JSON/YAML that describe downloads)
    if url_lower.endswith((".json", ".yaml", ".yml")):
        return "manifest"
    
    # Default to HTTP
    return "http"


# Archive extensions
_ARCHIVE_EXTENSIONS = {
    '.zip', '.tar.gz', '.tgz', '.tar.bz2', '.tbz2', '.tbz', 
    '.tar.xz', '.txz', '.tar', '.7z'
}


def is_archive_url(url: str) -> bool:
    """Check if URL points to an archive file based on extension.
    
    Args:
        url: The URL to check
        
    Returns:
        True if URL ends with an archive extension
    """
    url_lower = url.lower().strip()
    # Remove query parameters for extension check
    url_path = url_lower.split('?')[0]
    return any(url_path.endswith(ext) for ext in _ARCHIVE_EXTENSIONS)


def get_url_info(url: str) -> Dict[str, Any]:
    """Get information about a URL.
    
    Returns dict with: name, size (if available), type, supports_resume
    
    Args:
        url: The URL to analyze
        
    Returns:
        Dictionary with URL information
    """
    url_type = detect_url_type(url)
    info = {
        "name": "",
        "size": None,
        "type": url_type,
        "supports_resume": False,
    }
    
    # Extract filename from URL
    parsed = urllib.parse.urlparse(url)
    path = parsed.path
    if path:
        info["name"] = _strip_all_extensions(os.path.basename(path)) or "download"
    else:
        info["name"] = "download"
    
    # Try to get size from HTTP headers for HTTP-based URLs
    if url_type in ("http", "github", "dropbox", "huggingface"):
        try:
            req = urllib.request.Request(url, method="HEAD")
            req.add_header("User-Agent", HTTP_USER_AGENT)
            with urlopen(req, timeout=30) as resp:
                if "Content-Length" in resp.headers:
                    info["size"] = int(resp.headers["Content-Length"])
                if "Accept-Ranges" in resp.headers:
                    info["supports_resume"] = resp.headers["Accept-Ranges"] == "bytes"
                # Update name from Content-Disposition if available
                cd = resp.headers.get("Content-Disposition", "")
                if "filename=" in cd:
                    fname = cd.split("filename=")[-1].strip('"\'')
                    if fname:
                        info["name"] = fname
        except Exception:
            pass  # Size unknown is OK
    
    # S3-specific info
    if url_type == "s3" and HAS_BOTO3:
        try:
            s3_info = _get_s3_object_info(url)
            if s3_info:
                info["size"] = s3_info.get("size")
                info["name"] = s3_info.get("name", info["name"])
                info["supports_resume"] = True
        except Exception:
            pass
    
    # GCS-specific info
    if url_type == "gcs" and HAS_GCS:
        try:
            gcs_info = _get_gcs_object_info(url)
            if gcs_info:
                info["size"] = gcs_info.get("size")
                info["name"] = gcs_info.get("name", info["name"])
        except Exception:
            pass
    
    return info


def _get_s3_object_info(url: str) -> Optional[Dict[str, Any]]:
    """Get S3 object info using boto3."""
    if not HAS_BOTO3:
        return None
    
    try:
        if url.startswith("s3://"):
            # s3://bucket/key format
            parts = url[5:].split("/", 1)
            bucket = parts[0]
            key = parts[1] if len(parts) > 1 else ""
        else:
            # https://bucket.s3.amazonaws.com/key format
            parsed = urllib.parse.urlparse(url)
            host = parsed.netloc
            if ".s3.amazonaws.com" in host:
                bucket = host.replace(".s3.amazonaws.com", "")
            elif ".s3." in host:
                bucket = host.split(".s3.")[0]
            else:
                return None
            key = parsed.path.lstrip("/")
        
        s3 = boto3.client("s3")
        response = s3.head_object(Bucket=bucket, Key=key)
        return {
            "size": response.get("ContentLength"),
            "name": _strip_all_extensions(os.path.basename(key)),
        }
    except Exception as exc:
        logger.debug("Failed to get S3 object info: %s", exc)
        return None


def _get_gcs_object_info(url: str) -> Optional[Dict[str, Any]]:
    """Get GCS object info using google-cloud-storage."""
    if not HAS_GCS:
        return None
    
    try:
        if url.startswith("gs://"):
            # gs://bucket/object format
            parts = url[5:].split("/", 1)
            bucket_name = parts[0]
            blob_name = parts[1] if len(parts) > 1 else ""
        else:
            # https://storage.googleapis.com/bucket/object format
            parsed = urllib.parse.urlparse(url)
            path = parsed.path.lstrip("/")
            parts = path.split("/", 1)
            bucket_name = parts[0]
            blob_name = parts[1] if len(parts) > 1 else ""
        
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.get_blob(blob_name)
        if blob:
            return {
                "size": blob.size,
                "name": _strip_all_extensions(os.path.basename(blob_name)),
            }
        return None
    except Exception as exc:
        logger.debug("Failed to get GCS object info: %s", exc)
        return None


def _transform_dropbox_url(url: str) -> str:
    """Transform Dropbox URL to direct download URL."""
    parsed = urllib.parse.urlparse(url)
    query = urllib.parse.parse_qs(parsed.query)
    
    # Ensure dl=1 for direct download
    query["dl"] = ["1"]
    
    new_query = urllib.parse.urlencode(query, doseq=True)
    return urllib.parse.urlunparse(parsed._replace(query=new_query))


def _transform_huggingface_url(url: str) -> str:
    """Transform HuggingFace URL to use /resolve/ endpoint."""
    # If already a resolve URL, return as-is
    if "/resolve/" in url:
        return url
    
    # Convert /blob/ to /resolve/
    if "/blob/" in url:
        return url.replace("/blob/", "/resolve/")
    
    # For hf.co shorthand, expand to full URL
    if url.startswith("hf.co/"):
        url = "https://huggingface.co/" + url[6:]
    
    return url


def _download_with_progress(
    resp,
    dest_path: Path,
    total_size: Optional[int],
    on_progress: Optional[Callable[[float, str], None]],
    start_time: float,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> None:
    """Download with progress reporting."""
    downloaded = 0
    last_report_time = start_time
    min_report_interval = 0.5  # Report at most every 0.5 seconds
    
    with open(dest_path, "wb") as f:
        while True:
            _raise_if_cancelled(should_cancel)
            chunk = resp.read(8192)
            if not chunk:
                break
            
            f.write(chunk)
            downloaded += len(chunk)
            
            current_time = time.time()
            if on_progress and (current_time - last_report_time >= min_report_interval):
                last_report_time = current_time
                
                if total_size and total_size > 0:
                    percent = downloaded / total_size
                    elapsed = current_time - start_time
                    speed = downloaded / elapsed if elapsed > 0 else 0
                    
                    # Calculate ETA
                    remaining = total_size - downloaded
                    eta_seconds = remaining / speed if speed > 0 else 0
                    
                    speed_str = _format_bytes(speed) + "/s"
                    eta_str = _format_time(eta_seconds) if eta_seconds > 0 else ""
                    
                    status = f"Downloading... {int(percent * 100)}% ({_format_bytes(downloaded)} / {_format_bytes(total_size)}) {speed_str}"
                    if eta_str:
                        status += f" ETA: {eta_str}"
                    
                    on_progress(min(percent, 0.99), status)
                else:
                    # Unknown size
                    elapsed = current_time - start_time
                    speed = downloaded / elapsed if elapsed > 0 else 0
                    status = f"Downloading... {_format_bytes(downloaded)} ({_format_bytes(speed)}/s)"
                    on_progress(-1.0, status)
    
    if on_progress:
        on_progress(1.0, f"Download complete: {_format_bytes(downloaded)}")


def _format_bytes(size: float) -> str:
    """Format bytes to human readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} PB"


def _format_time(seconds: float) -> str:
    """Format seconds to human readable time string."""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def _download_http(
    url: str,
    dest_path: Path,
    on_progress: Optional[Callable[[float, str], None]],
    on_warning: Optional[Callable[[str], None]],
    timeout: int,
    headers: Optional[Dict[str, str]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> None:
    """Download from HTTP(S) URL."""
    if on_progress:
        on_progress(0.0, "Connecting...")
    
    req_headers = {"User-Agent": HTTP_USER_AGENT}
    if headers:
        req_headers.update(headers)
    
    req = urllib.request.Request(url, headers=req_headers)
    
    start_time = time.time()
    
    try:
        _raise_if_cancelled(should_cancel)
        with urlopen(req, timeout=timeout) as resp:
            # Get total size if available
            total_size = None
            if "Content-Length" in resp.headers:
                try:
                    total_size = int(resp.headers["Content-Length"])
                except ValueError:
                    pass
            
            if on_progress:
                if total_size:
                    on_progress(0.0, f"Downloading... 0% (0 / {_format_bytes(total_size)})")
                else:
                    on_progress(0.0, "Downloading... (size unknown)")
            
            _download_with_progress(
                resp,
                dest_path,
                total_size,
                on_progress,
                start_time,
                should_cancel,
            )
    
    except InterruptedError:
        raise
    except urllib.error.HTTPError as exc:
        raise URLDownloadError(f"HTTP {exc.code}: {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise URLDownloadError(f"URL error: {exc.reason}") from exc
    except Exception as exc:
        raise URLDownloadError(f"Download failed: {exc}") from exc


def _download_s3(
    url: str,
    dest_path: Path,
    on_progress: Optional[Callable[[float, str], None]],
    on_warning: Optional[Callable[[str], None]],
    timeout: int,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> None:
    """Download from S3 URL using boto3 or HTTP fallback."""
    if HAS_BOTO3:
        try:
            _download_s3_boto3(url, dest_path, on_progress, timeout, should_cancel)
            return
        except InterruptedError:
            raise
        except Exception as exc:
            if on_warning:
                on_warning(f"S3 boto3 download failed, falling back to HTTP: {exc}")
            logger.warning("S3 boto3 download failed, using HTTP fallback: %s", exc)
    
    # Fallback to HTTP
    http_url = _s3_to_http_url(url)
    _download_http(http_url, dest_path, on_progress, on_warning, timeout, should_cancel=should_cancel)


def _s3_to_http_url(url: str) -> str:
    """Convert S3 URL to HTTP URL."""
    if url.startswith("s3://"):
        parts = url[5:].split("/", 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ""
        return f"https://{bucket}.s3.amazonaws.com/{key}"
    return url


def _download_s3_boto3(
    url: str,
    dest_path: Path,
    on_progress: Optional[Callable[[float, str], None]],
    timeout: int,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> None:
    """Download from S3 using boto3 with progress."""
    if not HAS_BOTO3 or boto3 is None:
        raise URLDownloadError("boto3 not available")
    
    # Parse S3 URL
    if url.startswith("s3://"):
        parts = url[5:].split("/", 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ""
    else:
        parsed = urllib.parse.urlparse(url)
        host = parsed.netloc
        if ".s3.amazonaws.com" in host:
            bucket = host.replace(".s3.amazonaws.com", "")
        elif ".s3." in host:
            bucket = host.split(".s3.")[0]
        else:
            raise URLDownloadError(f"Cannot parse S3 URL: {url}")
        key = parsed.path.lstrip("/")
    
    if on_progress:
        on_progress(0.0, "Connecting to S3...")
    
    config = Config(connect_timeout=timeout, read_timeout=timeout)
    s3 = boto3.client("s3", config=config)
    
    # Get object info
    try:
        _raise_if_cancelled(should_cancel)
        head_response = s3.head_object(Bucket=bucket, Key=key)
        total_size = head_response.get("ContentLength", 0)
    except Exception:
        total_size = 0
    
    if on_progress:
        if total_size:
            on_progress(0.0, f"Downloading from S3... 0% (0 / {_format_bytes(total_size)})")
        else:
            on_progress(0.0, "Downloading from S3...")
    
    # Download with progress callback
    downloaded = 0
    start_time = time.time()
    last_report_time = start_time
    
    def progress_callback(bytes_transferred):
        nonlocal downloaded, last_report_time
        _raise_if_cancelled(should_cancel)
        downloaded = bytes_transferred
        
        current_time = time.time()
        if on_progress and (current_time - last_report_time >= 0.5):
            last_report_time = current_time
            
            if total_size > 0:
                percent = downloaded / total_size
                elapsed = current_time - start_time
                speed = downloaded / elapsed if elapsed > 0 else 0
                
                remaining = total_size - downloaded
                eta_seconds = remaining / speed if speed > 0 else 0
                
                speed_str = _format_bytes(speed) + "/s"
                eta_str = _format_time(eta_seconds) if eta_seconds > 0 else ""
                
                status = f"Downloading from S3... {int(percent * 100)}% ({_format_bytes(downloaded)} / {_format_bytes(total_size)}) {speed_str}"
                if eta_str:
                    status += f" ETA: {eta_str}"
                
                on_progress(min(percent, 0.99), status)
    
    _raise_if_cancelled(should_cancel)
    s3.download_file(bucket, key, str(dest_path), Callback=progress_callback)
    
    if on_progress:
        on_progress(1.0, f"S3 download complete: {_format_bytes(downloaded)}")


def _download_r2(
    url: str,
    dest_path: Path,
    on_progress: Optional[Callable[[float, str], None]],
    on_warning: Optional[Callable[[str], None]],
    timeout: int,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> None:
    """Download from Cloudflare R2 (S3-compatible)."""
    if HAS_BOTO3:
        try:
            _download_r2_boto3(url, dest_path, on_progress, timeout, should_cancel)
            return
        except InterruptedError:
            raise
        except Exception as exc:
            if on_warning:
                on_warning(f"R2 boto3 download failed, falling back to HTTP: {exc}")
            logger.warning("R2 boto3 download failed, using HTTP fallback: %s", exc)
    
    # Fallback to HTTP
    _download_http(url, dest_path, on_progress, on_warning, timeout, should_cancel=should_cancel)


def _download_r2_boto3(
    url: str,
    dest_path: Path,
    on_progress: Optional[Callable[[float, str], None]],
    timeout: int,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> None:
    """Download from R2 using boto3 with S3-compatible API."""
    if not HAS_BOTO3 or boto3 is None:
        raise URLDownloadError("boto3 not available")
    
    parsed = urllib.parse.urlparse(url)
    host = parsed.netloc
    
    # Extract endpoint and bucket from R2 URL
    # Format: https://<account>.r2.cloudflarestorage.com/<bucket>/<key>
    # or: https://<bucket>.<account>.r2.cloudflarestorage.com/<key>
    
    if ".r2.cloudflarestorage.com" not in host:
        raise URLDownloadError(f"Not a valid R2 URL: {url}")
    
    # Determine endpoint URL
    endpoint = f"https://{host}"
    
    # Parse bucket and key from path
    path_parts = parsed.path.lstrip("/").split("/", 1)
    if len(path_parts) < 2:
        raise URLDownloadError(f"Cannot parse R2 URL path: {url}")
    
    bucket = path_parts[0]
    key = path_parts[1]
    
    if on_progress:
        on_progress(0.0, "Connecting to R2...")
    
    # R2 uses S3-compatible API
    config = Config(connect_timeout=timeout, read_timeout=timeout)
    
    # Try to get credentials from environment
    access_key = os.environ.get("R2_ACCESS_KEY_ID") or os.environ.get("AWS_ACCESS_KEY_ID")
    secret_key = os.environ.get("R2_SECRET_ACCESS_KEY") or os.environ.get("AWS_SECRET_ACCESS_KEY")
    
    if access_key and secret_key:
        s3 = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=config,
        )
    else:
        # Try without explicit credentials (may use IAM role, etc.)
        s3 = boto3.client("s3", endpoint_url=endpoint, config=config)
    
    # Get object info
    try:
        _raise_if_cancelled(should_cancel)
        head_response = s3.head_object(Bucket=bucket, Key=key)
        total_size = head_response.get("ContentLength", 0)
    except Exception:
        total_size = 0
    
    if on_progress:
        if total_size:
            on_progress(0.0, f"Downloading from R2... 0% (0 / {_format_bytes(total_size)})")
        else:
            on_progress(0.0, "Downloading from R2...")
    
    # Download with progress
    downloaded = 0
    start_time = time.time()
    last_report_time = start_time
    
    def progress_callback(bytes_transferred):
        nonlocal downloaded, last_report_time
        _raise_if_cancelled(should_cancel)
        downloaded = bytes_transferred
        
        current_time = time.time()
        if on_progress and (current_time - last_report_time >= 0.5):
            last_report_time = current_time
            
            if total_size > 0:
                percent = downloaded / total_size
                elapsed = current_time - start_time
                speed = downloaded / elapsed if elapsed > 0 else 0
                
                remaining = total_size - downloaded
                eta_seconds = remaining / speed if speed > 0 else 0
                
                speed_str = _format_bytes(speed) + "/s"
                eta_str = _format_time(eta_seconds) if eta_seconds > 0 else ""
                
                status = f"Downloading from R2... {int(percent * 100)}% ({_format_bytes(downloaded)} / {_format_bytes(total_size)}) {speed_str}"
                if eta_str:
                    status += f" ETA: {eta_str}"
                
                on_progress(min(percent, 0.99), status)
    
    _raise_if_cancelled(should_cancel)
    s3.download_file(bucket, key, str(dest_path), Callback=progress_callback)
    
    if on_progress:
        on_progress(1.0, f"R2 download complete: {_format_bytes(downloaded)}")


def _download_gcs(
    url: str,
    dest_path: Path,
    on_progress: Optional[Callable[[float, str], None]],
    on_warning: Optional[Callable[[str], None]],
    timeout: int,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> None:
    """Download from GCS URL using google-cloud-storage or HTTP fallback."""
    if HAS_GCS:
        try:
            _download_gcs_client(url, dest_path, on_progress, timeout, should_cancel)
            return
        except InterruptedError:
            raise
        except Exception as exc:
            if on_warning:
                on_warning(f"GCS client download failed, falling back to HTTP: {exc}")
            logger.warning("GCS client download failed, using HTTP fallback: %s", exc)
    
    # Fallback to HTTP
    http_url = _gcs_to_http_url(url)
    _download_http(http_url, dest_path, on_progress, on_warning, timeout, should_cancel=should_cancel)


def _gcs_to_http_url(url: str) -> str:
    """Convert GCS URL to HTTP URL."""
    if url.startswith("gs://"):
        parts = url[5:].split("/", 1)
        bucket = parts[0]
        object_name = parts[1] if len(parts) > 1 else ""
        return f"https://storage.googleapis.com/{bucket}/{object_name}"
    return url


def _download_gcs_client(
    url: str,
    dest_path: Path,
    on_progress: Optional[Callable[[float, str], None]],
    timeout: int,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> None:
    """Download from GCS using google-cloud-storage with progress."""
    if not HAS_GCS or storage is None:
        raise URLDownloadError("google-cloud-storage not available")
    
    # Parse GCS URL
    if url.startswith("gs://"):
        parts = url[5:].split("/", 1)
        bucket_name = parts[0]
        blob_name = parts[1] if len(parts) > 1 else ""
    else:
        parsed = urllib.parse.urlparse(url)
        path = parsed.path.lstrip("/")
        parts = path.split("/", 1)
        bucket_name = parts[0]
        blob_name = parts[1] if len(parts) > 1 else ""
    
    if on_progress:
        on_progress(0.0, "Connecting to GCS...")
    
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    _raise_if_cancelled(should_cancel)
    blob = bucket.blob(blob_name)
    
    # Get size
    _raise_if_cancelled(should_cancel)
    blob.reload()
    total_size = blob.size or 0
    
    if on_progress:
        if total_size:
            on_progress(0.0, f"Downloading from GCS... 0% (0 / {_format_bytes(total_size)})")
        else:
            on_progress(0.0, "Downloading from GCS...")
    
    # Download with progress
    downloaded = 0
    start_time = time.time()
    last_report_time = start_time
    
    class ProgressCallback:
        def __init__(self, total):
            self.total = total
            self.downloaded = 0
            self.start_time = time.time()
            self.last_report = self.start_time
        
        def __call__(self, chunk):
            _raise_if_cancelled(should_cancel)
            self.downloaded += len(chunk)
            current_time = time.time()
            
            if on_progress and (current_time - self.last_report >= 0.5):
                self.last_report = current_time
                
                if self.total > 0:
                    percent = self.downloaded / self.total
                    elapsed = current_time - self.start_time
                    speed = self.downloaded / elapsed if elapsed > 0 else 0
                    
                    remaining = self.total - self.downloaded
                    eta_seconds = remaining / speed if speed > 0 else 0
                    
                    speed_str = _format_bytes(speed) + "/s"
                    eta_str = _format_time(eta_seconds) if eta_seconds > 0 else ""
                    
                    status = f"Downloading from GCS... {int(percent * 100)}% ({_format_bytes(self.downloaded)} / {_format_bytes(self.total)}) {speed_str}"
                    if eta_str:
                        status += f" ETA: {eta_str}"
                    
                    on_progress(min(percent, 0.99), status)
    
    callback = ProgressCallback(total_size)
    _raise_if_cancelled(should_cancel)
    blob.download_to_filename(str(dest_path), progress_callback=callback)
    
    if on_progress:
        on_progress(1.0, f"GCS download complete: {_format_bytes(callback.downloaded)}")


def download_url(
    url: str,
    dest_path: Path,
    on_progress: Optional[Callable[[float, str], None]] = None,
    on_warning: Optional[Callable[[str], None]] = None,
    timeout: int = 300,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> None:
    """Download URL to destination with progress callbacks.
    
    Args:
        url: Source URL (HTTP, S3, GCS, etc.)
        dest_path: Where to save the file
        on_progress: Callback(percent: float, status: str) - percent 0.0-1.0
        on_warning: Callback(warning_msg: str) - for non-fatal issues
        timeout: Download timeout in seconds
        
    Raises:
        URLDownloadError: If the download fails
        UnsupportedURLError: If the URL type is not supported
    """
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    url_type = detect_url_type(url)
    
    # Transform URLs as needed
    if url_type == "dropbox":
        url = _transform_dropbox_url(url)
    elif url_type == "huggingface":
        url = _transform_huggingface_url(url)
    
    # Track if we need cleanup on failure
    temp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")
    
    try:
        if url_type == "s3":
            _download_s3(url, temp_path, on_progress, on_warning, timeout, should_cancel)
        elif url_type == "r2":
            _download_r2(url, temp_path, on_progress, on_warning, timeout, should_cancel)
        elif url_type == "gcs":
            _download_gcs(url, temp_path, on_progress, on_warning, timeout, should_cancel)
        elif url_type in ("http", "github", "dropbox", "huggingface", "manifest"):
            _download_http(url, temp_path, on_progress, on_warning, timeout, should_cancel=should_cancel)
        else:
            raise UnsupportedURLError(f"Unsupported URL type: {url_type}")
        
        # Move temp file to final destination
        temp_path.replace(dest_path)
        
    except Exception:
        # Clean up temp file on failure
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass
        raise


def extract_archive(
    archive_path: Path,
    dest_dir: Path,
    on_progress: Optional[Callable[[float, str], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> None:
    """Extract zip/tar archives with progress.
    
    Supports: .zip, .tar.gz, .tgz, .tar.bz2, .tbz2, .tbz, .tar.xz, .txz, .tar, .7z
    
    Args:
        archive_path: Path to the archive file
        dest_dir: Directory to extract to
        on_progress: Callback(percent: float, status: str)
        
    Raises:
        ExtractError: If extraction fails
    """
    archive_path = Path(archive_path)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    if not archive_path.exists():
        raise ExtractError(f"Archive not found: {archive_path}")
    
    # Check for compound extensions first
    name_lower = archive_path.name.lower()

    # Determine archive type by checking compound extensions first
    is_tar = False
    is_zip = False
    is_7z = False
    if name_lower.endswith(('.tar.gz', '.tgz')):
        is_tar = True
    elif name_lower.endswith(('.tar.bz2', '.tbz2', '.tbz')):
        is_tar = True
    elif name_lower.endswith(('.tar.xz', '.txz')):
        is_tar = True
    elif name_lower.endswith('.tar'):
        is_tar = True
    elif name_lower.endswith('.zip'):
        is_zip = True
    elif name_lower.endswith('.7z'):
        is_7z = True
    else:
        # Fall back to checking by content
        is_zip = zipfile.is_zipfile(archive_path)
        is_tar = tarfile.is_tarfile(archive_path)
        is_7z = name_lower.endswith('.7z')

    try:
        if is_zip:
            _extract_zip(archive_path, dest_dir, on_progress, should_cancel)
        elif is_tar:
            _extract_tar(archive_path, dest_dir, on_progress, should_cancel)
        elif is_7z:
            _extract_7z(archive_path, dest_dir, on_progress, should_cancel)
        else:
            raise ExtractError(f"Unsupported archive format: {archive_path}")
    except InterruptedError:
        raise
    except ExtractError:
        raise
    except Exception as exc:
        raise ExtractError(f"Extraction failed: {exc}") from exc


def _copy_stream(
    src,
    dst,
    should_cancel: Optional[Callable[[], bool]],
    chunk_size: int = 1024 * 1024,
) -> None:
    """Copy a stream in chunks so large extractions can be cancelled."""
    while True:
        _raise_if_cancelled(should_cancel)
        chunk = src.read(chunk_size)
        if not chunk:
            break
        dst.write(chunk)


def _extract_zip(
    archive_path: Path,
    dest_dir: Path,
    on_progress: Optional[Callable[[float, str], None]],
    should_cancel: Optional[Callable[[], bool]],
) -> None:
    """Extract ZIP archive with progress."""
    with zipfile.ZipFile(archive_path, "r") as zf:
        members = zf.infolist()
        total = len(members)
        
        for i, member in enumerate(members):
            _raise_if_cancelled(should_cancel)
            if on_progress and i % 10 == 0:  # Report every 10 files
                percent = i / total if total > 0 else 0
                on_progress(percent, f"Extracting... {i}/{total} files")
            
            # Security: Check for path traversal
            target_path = dest_dir / member.filename
            try:
                target_path.relative_to(dest_dir)
            except ValueError:
                logger.warning("Skipping suspicious path in zip: %s", member.filename)
                continue
            
            if member.is_dir():
                target_path.mkdir(parents=True, exist_ok=True)
            else:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src, open(target_path, "wb") as dst:
                    _copy_stream(src, dst, should_cancel)
        
        if on_progress:
            on_progress(1.0, f"Extraction complete: {total} files")


def _extract_tar(
    archive_path: Path,
    dest_dir: Path,
    on_progress: Optional[Callable[[float, str], None]],
    should_cancel: Optional[Callable[[], bool]],
) -> None:
    """Extract TAR archive with progress."""
    with tarfile.open(archive_path, "r:*") as tf:
        members = tf.getmembers()
        total = len(members)
        
        for i, member in enumerate(members):
            _raise_if_cancelled(should_cancel)
            if on_progress and i % 10 == 0:  # Report every 10 files
                percent = i / total if total > 0 else 0
                on_progress(percent, f"Extracting... {i}/{total} files")
            
            # Security: Check for path traversal
            target_path = dest_dir / member.name
            try:
                target_path.relative_to(dest_dir)
            except ValueError:
                logger.warning("Skipping suspicious path in tar: %s", member.name)
                continue
            
            if member.isdir():
                target_path.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target_path.parent.mkdir(parents=True, exist_ok=True)
                with tf.extractfile(member) as src, open(target_path, "wb") as dst:
                    if src:
                        _copy_stream(src, dst, should_cancel)
            # Skip symlinks and other special files for security
        
        if on_progress:
            on_progress(1.0, f"Extraction complete: {total} files")


def _extract_7z(
    archive_path: Path,
    dest_dir: Path,
    on_progress: Optional[Callable[[float, str], None]],
    should_cancel: Optional[Callable[[], bool]],
) -> None:
    """Extract 7z archive with progress."""
    if not HAS_PY7ZR or py7zr is None:
        raise ExtractError("py7zr not available for .7z extraction")
    
    with py7zr.SevenZipFile(archive_path, mode="r") as sz:
        # Get list of files
        file_list = sz.getnames()
        total = len(file_list)
        
        _raise_if_cancelled(should_cancel)
        if on_progress:
            on_progress(0.0, f"Extracting 7z archive... 0/{total} files")
        
        # Extract all
        _raise_if_cancelled(should_cancel)
        sz.extractall(path=dest_dir)
        
        if on_progress:
            on_progress(1.0, f"Extraction complete: {total} files")


def download_and_extract(
    url: str,
    dest_dir: Path,
    on_progress: Optional[Callable[[float, str], None]] = None,
    on_warning: Optional[Callable[[str], None]] = None,
    timeout: int = 300,
    extract: bool = True,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> Path:
    """Download URL and optionally extract archive.
    
    Args:
        url: Source URL
        dest_dir: Directory to extract to (or parent for single files)
        on_progress: Callback(percent: float, status: str)
        on_warning: Callback(warning_msg: str)
        timeout: Download timeout in seconds
        extract: Whether to extract if URL is an archive
        
    Returns:
        Path to downloaded file or extraction directory
        
    Raises:
        URLDownloadError: If download fails
        ExtractError: If extraction fails
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Get URL info to determine filename
    info = get_url_info(url)
    filename = info.get("name") or "download"
    
    # Create temp directory for download
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / filename
        
        # Download
        download_url(url, tmp_path, on_progress, on_warning, timeout, should_cancel)
        
        # Check if it's an archive and should be extracted
        if extract and _is_archive(tmp_path):
            if on_progress:
                on_progress(0.0, "Extracting archive...")
            
            extract_dir = dest_dir / filename.rsplit(".", 1)[0]
            extract_dir.mkdir(parents=True, exist_ok=True)
            
            extract_archive(tmp_path, extract_dir, on_progress, should_cancel)
            return extract_dir
        else:
            # Move to destination
            final_path = dest_dir / filename
            shutil.move(str(tmp_path), str(final_path))
            return final_path


def _is_archive(path: Path) -> bool:
    """Check if file is an archive."""
    if zipfile.is_zipfile(path):
        return True
    if tarfile.is_tarfile(path):
        return True
    if path.suffix.lower() == ".7z":
        return HAS_PY7ZR
    return False


def download_with_retry(
    url: str,
    dest_path: Path,
    on_progress: Optional[Callable[[float, str], None]] = None,
    on_warning: Optional[Callable[[str], None]] = None,
    timeout: int = 300,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> None:
    """Download URL with automatic retry on transient errors.
    
    Args:
        url: Source URL
        dest_path: Where to save the file
        on_progress: Callback(percent: float, status: str)
        on_warning: Callback(warning_msg: str)
        timeout: Download timeout in seconds
        max_retries: Maximum number of retry attempts
        retry_delay: Initial delay between retries (doubles each retry)
        
    Raises:
        URLDownloadError: If all retries fail
    """
    last_error = None
    
    for attempt in range(max_retries):
        try:
            download_url(url, dest_path, on_progress, on_warning, timeout, should_cancel)
            return
        except InterruptedError:
            raise
        except (urllib.error.HTTPError, urllib.error.URLError) as exc:
            last_error = exc
            
            # Don't retry on 4xx errors (client errors)
            if isinstance(exc, urllib.error.HTTPError) and exc.code < 500:
                raise URLDownloadError(f"HTTP {exc.code}: {exc.reason}") from exc
            
            if on_warning:
                on_warning(f"Download attempt {attempt + 1} failed: {exc}. Retrying...")
            
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (2 ** attempt))
        except Exception as exc:
            last_error = exc
            if on_warning:
                on_warning(f"Download attempt {attempt + 1} failed: {exc}. Retrying...")
            
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (2 ** attempt))
    
    raise URLDownloadError(f"Download failed after {max_retries} attempts: {last_error}") from last_error

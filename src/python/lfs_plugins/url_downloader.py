# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Minimal HTTP(S) downloader with zip/tar archive extraction."""

from __future__ import annotations

import logging
import os
import shutil
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .http import urlopen

logger = logging.getLogger(__name__)

HTTP_USER_AGENT = "LichtFeld-AssetManager/1.0"


def _raise_if_cancelled(should_cancel: Optional[Callable[[], bool]]) -> None:
    """Raise when the caller requested cancellation."""
    if should_cancel and should_cancel():
        raise InterruptedError("Download cancelled")


def _strip_archive_suffix(name: str) -> str:
    """Strip archive suffix from filename (e.g., data.tar.gz -> data)."""
    for suffix in (".tar.gz", ".tar.bz2", ".tar.xz", ".zip", ".tar"):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return name


class URLDownloadError(Exception):
    """Raised when a URL download fails."""
    pass


class UnsupportedURLError(URLDownloadError):
    """Raised when the URL type is not supported."""
    pass


class ExtractError(Exception):
    """Raised when archive extraction fails."""
    pass


def normalize_url(url: str) -> str:
    """Normalize an HTTP(S) URL.
    
    - Validates scheme is http or https.
    - Rewrites Dropbox URLs to ensure dl=1.
    - Rewrites HuggingFace /blob/ to /resolve/ and expands hf.co.
    
    Args:
        url: The URL to normalize
        
    Returns:
        Normalized URL string
        
    Raises:
        UnsupportedURLError: If scheme is not http/https
    """
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise UnsupportedURLError(
            "Only http:// and https:// URLs are supported. "
            "For S3/GCS/R2 assets, provide a public or signed HTTPS URL."
        )
    
    # Dropbox
    if "dropbox.com" in parsed.netloc.lower() or "dropboxusercontent.com" in parsed.netloc.lower():
        url = _transform_dropbox_url(url)
    
    # HuggingFace
    if "huggingface.co" in parsed.netloc.lower() or "hf.co" in parsed.netloc.lower():
        url = _transform_huggingface_url(url)
    
    return url


# Archive extensions
_ARCHIVE_EXTENSIONS = {
    '.zip', '.tar.gz', '.tgz', '.tar.bz2', '.tbz2', '.tbz',
    '.tar.xz', '.txz', '.tar'
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
    
    Returns dict with: name, size (if available), type
    
    Args:
        url: The URL to analyze
        
    Returns:
        Dictionary with URL information
    """
    url = normalize_url(url)
    parsed = urllib.parse.urlparse(url)
    name = os.path.basename(parsed.path) or "download"
    size = None
    
    try:
        req = urllib.request.Request(url, method="HEAD")
        req.add_header("User-Agent", HTTP_USER_AGENT)
        with urlopen(req, timeout=30) as resp:
            if "Content-Length" in resp.headers:
                size = int(resp.headers["Content-Length"])
            cd = resp.headers.get("Content-Disposition", "")
            if "filename=" in cd:
                fname = Path(cd.split("filename=")[-1].strip('"\' \t')).name
                if fname and fname not in (".", ".."):
                    name = fname
    except Exception:
        pass  # Size unknown is OK
    
    return {"name": name, "size": size, "type": "http"}


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


def download_url(
    url: str,
    dest_path: Path,
    on_progress: Optional[Callable[[float, str], None]] = None,
    on_warning: Optional[Callable[[str], None]] = None,
    timeout: int = 300,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> None:
    """Download an HTTP(S) URL to a local file with progress callbacks.
    
    Args:
        url: Source URL (HTTP, HTTPS)
        dest_path: Where to save the file
        on_progress: Callback(percent: float, status: str) - percent 0.0-1.0
        on_warning: Callback(warning_msg: str) - for non-fatal issues
        timeout: Download timeout in seconds
        
    Raises:
        URLDownloadError: If the download fails
        UnsupportedURLError: If the URL type is not supported
    """
    url = normalize_url(url)
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    temp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")
    
    try:
        _download_http(url, temp_path, on_progress, on_warning, timeout, should_cancel=should_cancel)
        temp_path.replace(dest_path)
    except Exception:
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
    
    Supports: .zip, .tar.gz, .tgz, .tar.bz2, .tbz2, .tbz, .tar.xz, .txz, .tar
    
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
    else:
        # Fall back to checking by content
        is_zip = zipfile.is_zipfile(archive_path)
        is_tar = tarfile.is_tarfile(archive_path)

    try:
        if is_zip:
            _extract_zip(archive_path, dest_dir, on_progress, should_cancel)
        elif is_tar:
            _extract_tar(archive_path, dest_dir, on_progress, should_cancel)
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
            target_path = (dest_dir / member.filename).resolve()
            try:
                target_path.relative_to(dest_dir.resolve())
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
            target_path = (dest_dir / member.name).resolve()
            try:
                target_path.relative_to(dest_dir.resolve())
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


def download_and_extract(
    url: str,
    dest_dir: Path,
    on_progress: Optional[Callable[[float, str], None]] = None,
    on_warning: Optional[Callable[[str], None]] = None,
    timeout: int = 300,
    extract: bool = True,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> Path:
    """Download an HTTP(S) URL and optionally extract zip/tar archives.
    
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
    
    parsed = urllib.parse.urlparse(url)
    filename = os.path.basename(parsed.path) or "download"
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / filename
        
        download_url(url, tmp_path, on_progress, on_warning, timeout, should_cancel)
        
        if extract and _is_archive(tmp_path):
            if on_progress:
                on_progress(0.0, "Extracting archive...")
            
            extract_dir = dest_dir / _strip_archive_suffix(filename)
            extract_dir.mkdir(parents=True, exist_ok=True)
            
            extract_archive(tmp_path, extract_dir, on_progress, should_cancel)
            return extract_dir
        else:
            final_path = dest_dir / filename
            shutil.move(str(tmp_path), str(final_path))
            return final_path


def _is_archive(path: Path) -> bool:
    """Check if file is an archive."""
    if zipfile.is_zipfile(path):
        return True
    if tarfile.is_tarfile(path):
        return True
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
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code < 500:
                raise URLDownloadError(f"HTTP {exc.code}: {exc.reason}") from exc
            if on_warning:
                on_warning(f"Download attempt {attempt + 1} failed: {exc}. Retrying...")
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (2 ** attempt))
        except urllib.error.URLError as exc:
            last_error = exc
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

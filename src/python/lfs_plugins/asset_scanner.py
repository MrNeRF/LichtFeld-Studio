# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""AssetScanner module for detecting file types and extracting metadata.

This module provides the AssetScanner class which is used by the Asset Manager
to detect file types, infer asset roles, and extract metadata from various
file formats including Gaussian splats, checkpoints, datasets, videos, and more.
"""

import json
import logging
import os
import re
import struct
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import lichtfeld as lf

_logger = logging.getLogger(__name__)

# File extension to type mapping
_EXTENSION_TYPE_MAP = {
    ".ply": "ply",
    ".rad": "rad",
    ".sog": "sog",
    ".spz": "spz",
    ".ckpt": "checkpoint",
    ".mp4": "video",
    ".mov": "video",
    ".usd": "usd",
    ".usdz": "usd",
    ".html": "html",
    ".json": "json",
}

# Header signatures for file type detection (first few bytes)
_HEADER_SIGNATURES = {
    b"ply\n": "ply",
    b"ply\r": "ply",
    b"SPZ\x00": "spz",  # SPZ magic number
    b"SPZ\x01": "spz",
}

# Role detection patterns
_ROLE_PATTERNS = {
    "source_dataset": [
        r"dataset",
        r"input",
        r"source",
        r"images",
        r"colmap",
    ],
    "training_checkpoint": [
        r"checkpoint",
        r"ckpt",
        r"iteration[_-]?\d+",
        r"iter[_-]?\d+",
        r"chkpnt",
    ],
    "trained_output": [
        r"output",
        r"trained",
        r"result",
        r"final",
        r"point_cloud",
        r"iteration[_-]?\d+",
    ],
    "preview": [
        r"preview",
        r"thumb",
        r"thumbnail",
        r"render",
    ],
    "export": [
        r"export",
        r"output",
        r"saved",
        r"published",
    ],
    "reference": [
        r"ref",
        r"reference",
        r"guide",
        r"example",
    ],
}

# Image extensions for dataset scanning
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp", ".exr"}
_DATASET_EXCLUDED_DIRS = {
    "masks",
    "mask",
    "sparse",
    "dense",
    "stereo",
    "depth",
    "depths",
    "images_reconstruction",
    "reconstruction",
    "__pycache__",
}


class AssetScanner:
    """Scanner for detecting file types and extracting metadata from assets.

    The AssetScanner provides methods to:
    - Detect file types from extensions and headers
    - Infer asset roles from filenames and paths
    - Extract metadata from various file formats
    - Validate datasets and checkpoints

    Example:
        >>> scanner = AssetScanner()
        >>> file_type = scanner.detect_type("/path/to/model.ply")
        >>> metadata = scanner.scan_file("/path/to/model.ply")
    """

    def __init__(self):
        """Initialize the AssetScanner."""
        self._extension_map = _EXTENSION_TYPE_MAP.copy()
        self._header_signatures = _HEADER_SIGNATURES.copy()
        self._role_patterns = _ROLE_PATTERNS.copy()

    # ═══════════════════════════════════════════════════════════════════════════════
    # Type Detection
    # ═══════════════════════════════════════════════════════════════════════════════

    def detect_type(self, path: str) -> Optional[str]:
        """Detect the file type from extension and/or header sniffing.

        Args:
            path: Path to the file to analyze.

        Returns:
            One of: "ply", "rad", "sog", "spz", "checkpoint", "dataset",
            "video", "usd", "html", "json", or None if type cannot be determined.

        Example:
            >>> scanner.detect_type("model.ply")
            'ply'
            >>> scanner.detect_type("checkpoint.ckpt")
            'checkpoint'
        """
        path_obj = Path(path)

        # Check if it's a directory (dataset)
        if path_obj.is_dir():
            return "dataset" if self._looks_like_dataset(path_obj) else None

        # Check extension first
        ext = path_obj.suffix.lower()
        if ext in self._extension_map:
            detected = self._extension_map[ext]
            # Validate header for certain formats
            if detected in ("ply", "spz"):
                header_type = self._sniff_header(path)
                if header_type:
                    return header_type
            return detected

        # Try header sniffing for files without recognized extension
        header_type = self._sniff_header(path)
        if header_type:
            return header_type

        return None

    def _sniff_header(self, path: str) -> Optional[str]:
        """Sniff file header to determine type.

        Args:
            path: Path to the file.

        Returns:
            Detected type from header signature, or None.
        """
        try:
            with open(path, "rb") as f:
                header = f.read(16)
                for signature, file_type in self._header_signatures.items():
                    if header.startswith(signature):
                        return file_type

                # Check for PLY format (more flexible)
                if header.startswith(b"ply"):
                    return "ply"

                # Check for RAD format (typically starts with comment or vertex data)
                header_str = header.decode("ascii", errors="ignore")
                if "# RAD" in header_str or header_str.startswith("#"):
                    return "rad"

        except (IOError, OSError, PermissionError):
            pass

        return None

    def _looks_like_dataset(self, path: Path) -> bool:
        """Check if a directory looks like a valid dataset.

        Args:
            path: Path object to the directory.

        Returns:
            True if directory contains dataset-like structure.
        """
        try:
            has_images = bool(self._collect_dataset_images(path))
            has_sparse = (path / "sparse").is_dir()
            has_colmap_db = (path / "database.db").exists()

            return has_images or has_sparse or has_colmap_db

        except (OSError, PermissionError):
            return False

    def _get_dataset_image_root(self, path: Path) -> Path:
        images_dir = path / "images"
        return images_dir if images_dir.is_dir() else path

    def _is_excluded_dataset_image(self, image_root: Path, candidate: Path) -> bool:
        try:
            rel_parent_parts = candidate.relative_to(image_root).parts[:-1]
        except ValueError:
            rel_parent_parts = candidate.parts[:-1]
        return any(part.lower() in _DATASET_EXCLUDED_DIRS for part in rel_parent_parts)

    def _collect_dataset_images(self, path: Path) -> list[Path]:
        image_root = self._get_dataset_image_root(path)
        image_paths: dict[str, Path] = {}

        try:
            for item in image_root.rglob("*"):
                if not item.is_file() or item.suffix.lower() not in _IMAGE_EXTENSIONS:
                    continue
                if self._is_excluded_dataset_image(image_root, item):
                    continue
                resolved = str(item.resolve())
                image_paths[resolved] = item
        except (OSError, PermissionError):
            return []

        return sorted(image_paths.values(), key=lambda item: str(item))

    def _count_image_files(self, path: Path) -> int:
        try:
            return sum(
                1
                for item in path.rglob("*")
                if item.is_file() and item.suffix.lower() in _IMAGE_EXTENSIONS
            )
        except (OSError, PermissionError):
            return 0

    def _get_directory_size_bytes(self, path: Path) -> int:
        total_size = 0
        try:
            for item in path.rglob("*"):
                if not item.is_file():
                    continue
                try:
                    total_size += item.stat().st_size
                except (OSError, PermissionError):
                    continue
        except (OSError, PermissionError):
            return 0
        return total_size

    # ═══════════════════════════════════════════════════════════════════════════════
    # Role Detection
    # ═══════════════════════════════════════════════════════════════════════════════

    def detect_role(self, path: str, context: Optional[dict] = None) -> str:
        """Detect the asset role from filename, path, and context.

        Args:
            path: Path to the file or directory.
            context: Optional dictionary with additional context (e.g., {"parent": "training"}).

        Returns:
            One of: "source_dataset", "training_checkpoint", "trained_output",
            "preview", "export", "reference", or "unknown".

        Example:
            >>> scanner.detect_role("checkpoint_iteration_3000.ckpt")
            'training_checkpoint'
            >>> scanner.detect_role("output/point_cloud.ply")
            'trained_output'
        """
        path_obj = Path(path)
        name = path_obj.name.lower()
        parent = path_obj.parent.name.lower() if path_obj.parent else ""

        # Check context hints first
        if context:
            parent_hint = context.get("parent", "").lower()
            if "train" in parent_hint:
                if "input" in parent_hint or "source" in parent_hint:
                    return "source_dataset"
                if "output" in parent_hint or "result" in parent_hint:
                    return "trained_output"
                if "checkpoint" in parent_hint:
                    return "training_checkpoint"

        # Check filename against role patterns
        for role, patterns in self._role_patterns.items():
            for pattern in patterns:
                if re.search(pattern, name, re.IGNORECASE):
                    return role

        # Check parent directory for hints
        for role, patterns in self._role_patterns.items():
            for pattern in patterns:
                if re.search(pattern, parent, re.IGNORECASE):
                    return role

        # Default based on file type
        file_type = self.detect_type(path)
        if file_type == "dataset":
            return "source_dataset"
        elif file_type == "checkpoint":
            return "training_checkpoint"
        elif file_type in ("ply", "rad", "sog", "spz"):
            return "trained_output"
        elif file_type == "video":
            return "preview"

        return "unknown"

    # ═══════════════════════════════════════════════════════════════════════════════
    # Metadata Extraction
    # ═══════════════════════════════════════════════════════════════════════════════

    def scan_file(self, path: str) -> dict:
        """Perform a full metadata scan of a file or directory.

        Args:
            path: Path to the file or directory to scan.

        Returns:
            Dictionary containing comprehensive metadata including:
            - path: Canonical path
            - name: File name
            - type: Detected file type
            - role: Detected asset role
            - size_bytes: File size in bytes
            - created: Creation timestamp
            - modified: Last modification timestamp
            - format_specific: Format-specific metadata (varies by type)

        Example:
            >>> metadata = scanner.scan_file("model.ply")
            >>> print(metadata["type"], metadata["size_bytes"])
        """
        path_obj = Path(path)
        result = {
            "path": str(path_obj.resolve()),
            "name": path_obj.name,
            "type": None,
            "role": "unknown",
            "size_bytes": 0,
            "created": None,
            "modified": None,
            "format_specific": {},
        }

        # Detect type and role
        result["type"] = self.detect_type(path)
        result["role"] = self.detect_role(path)

        # Get file system info
        try:
            stat = path_obj.stat()
            result["size_bytes"] = (
                stat.st_size
                if not path_obj.is_dir()
                else self._get_directory_size_bytes(path_obj)
            )
            result["created"] = datetime.fromtimestamp(stat.st_ctime).isoformat()
            result["modified"] = datetime.fromtimestamp(stat.st_mtime).isoformat()
        except (OSError, PermissionError) as e:
            _logger.debug(f"Could not stat file {path}: {e}")

        # Extract format-specific metadata
        if result["type"] in ("ply", "rad", "sog", "spz"):
            result["format_specific"] = self.extract_geometry_metadata(path) or {}
            # For PLY files, distinguish between 3DGS and regular point cloud
            if result["type"] == "ply" and result["format_specific"]:
                is_3dgs = result["format_specific"].get("is_3dgs", False)
                result["type"] = "ply_3dgs" if is_3dgs else "ply_pcl"
        elif result["type"] == "checkpoint":
            result["format_specific"] = self.extract_checkpoint_metadata(path) or {}
        elif result["type"] == "dataset":
            result["format_specific"] = self.extract_dataset_metadata(path) or {}
        elif result["type"] == "video":
            result["format_specific"] = self.extract_video_metadata(path) or {}
        elif result["type"] == "json":
            result["format_specific"] = self._extract_json_metadata(path) or {}

        return result

    def extract_geometry_metadata(self, path: str) -> Optional[dict]:
        """Extract metadata from geometry files (PLY, RAD, SOG, SPZ).

        Args:
            path: Path to the geometry file.

        Returns:
            Dictionary with metadata including:
            - gaussian_count: Number of gaussians (if readable)
            - sh_degree: Spherical harmonics degree (if detectable)
            - compressed: True for SPZ files
            - format: File format variant
            Returns empty dict if file cannot be read.

        Example:
            >>> meta = scanner.extract_geometry_metadata("model.ply")
            >>> print(meta.get("gaussian_count"))
        """
        path_obj = Path(path)
        ext = path_obj.suffix.lower()
        result = {
            "gaussian_count": None,
            "sh_degree": None,
            "compressed": ext == ".spz",
            "format": ext.lstrip("."),
        }

        try:
            if ext == ".ply":
                ply_meta = self._parse_ply_header(path)
                if ply_meta:
                    result["gaussian_count"] = ply_meta.get("vertex_count")
                    result["sh_degree"] = ply_meta.get("sh_degree")
                    result["has_normals"] = ply_meta.get("has_normals", False)
                    result["has_colors"] = ply_meta.get("has_colors", False)
                    result["is_3dgs"] = ply_meta.get("is_3dgs", False)

            elif ext == ".spz":
                spz_meta = self._parse_spz_header(path)
                if spz_meta:
                    result["gaussian_count"] = spz_meta.get("vertex_count")
                    result["sh_degree"] = spz_meta.get("sh_degree")
                    result["version"] = spz_meta.get("version")

            elif ext in (".rad", ".sog"):
                # RAD/SOG formats - try to count lines or parse header
                rad_meta = self._parse_rad_header(path)
                if rad_meta:
                    result["gaussian_count"] = rad_meta.get("vertex_count")
                    result["format"] = rad_meta.get("format", ext.lstrip("."))

        except Exception as e:
            _logger.debug(f"Error extracting geometry metadata from {path}: {e}")

        return result

    def _parse_ply_header(self, path: str) -> Optional[dict]:
        """Parse PLY file header to extract metadata.

        Args:
            path: Path to the PLY file.

        Returns:
            Dictionary with vertex_count, sh_degree, has_normals, has_colors, is_3dgs.
        """
        try:
            with open(path, "rb") as f:
                header_lines = []
                while True:
                    line = f.readline()
                    if not line:
                        break
                    line_str = line.decode("ascii", errors="ignore").strip()
                    header_lines.append(line_str)
                    if line_str == "end_header":
                        break

                vertex_count = 0
                sh_degree = None
                has_normals = False
                has_colors = False
                sh_coeffs_count = 0
                f_rest_count = 0

                for line in header_lines:
                    if line.startswith("element vertex"):
                        parts = line.split()
                        if len(parts) >= 3:
                            vertex_count = int(parts[2])
                    elif line.startswith("property float nx") or line.startswith(
                        "property float n_x"
                    ):
                        has_normals = True
                    elif line.startswith("property uchar red") or line.startswith(
                        "property float red"
                    ):
                        has_colors = True
                    elif "f_dc" in line or "sh" in line.lower():
                        # Count SH coefficients
                        sh_coeffs_count += 1
                    elif "f_rest" in line:
                        # Count f_rest properties for SH degree detection
                        f_rest_count += 1

                # Use C++ function to detect if this is a 3DGS PLY file
                # (checks for opacity, scale_0, rot_0 properties)
                is_3dgs = lf.io.is_gaussian_splat_ply(path)

                # Infer SH degree from f_rest coefficient count
                # SH degree 0: 0 rest coefficients (only DC)
                # SH degree 1: 9 rest coefficients (3 channels × 3 coefficients)
                # SH degree 2: 24 rest coefficients (3 channels × 8 coefficients)
                # SH degree 3: 45 rest coefficients (3 channels × 15 coefficients)
                if f_rest_count > 0:
                    if f_rest_count >= 45:
                        sh_degree = 3
                    elif f_rest_count >= 24:
                        sh_degree = 2
                    elif f_rest_count >= 9:
                        sh_degree = 1
                    else:
                        sh_degree = 0
                elif sh_coeffs_count > 0:
                    # Fallback to old detection method
                    if sh_coeffs_count >= 16:
                        sh_degree = 3
                    elif sh_coeffs_count >= 9:
                        sh_degree = 2
                    elif sh_coeffs_count >= 4:
                        sh_degree = 1
                    else:
                        sh_degree = 0

                return {
                    "vertex_count": vertex_count if vertex_count > 0 else 0,
                    "sh_degree": sh_degree,
                    "has_normals": has_normals,
                    "has_colors": has_colors,
                    "is_3dgs": is_3dgs,
                }

        except Exception as e:
            _logger.debug(f"Error parsing PLY header: {e}")

        return None

    def _parse_spz_header(self, path: str) -> Optional[dict]:
        """Parse SPZ file header to extract metadata.

        SPZ format:
        - Magic: "SPZ" + version byte
        - 4 bytes: vertex count (uint32)
        - 1 byte: SH degree

        Args:
            path: Path to the SPZ file.

        Returns:
            Dictionary with vertex_count, sh_degree, version.
        """
        try:
            with open(path, "rb") as f:
                header = f.read(8)
                if len(header) < 8:
                    return None

                magic = header[:3]
                if magic != b"SPZ":
                    return None

                version = header[3]
                vertex_count = struct.unpack("<I", header[4:8])[0]

                # Read SH degree if available
                sh_degree = None
                if len(header) > 8:
                    sh_degree = header[8]

                return {
                    "vertex_count": vertex_count,
                    "sh_degree": sh_degree,
                    "version": version,
                }

        except Exception as e:
            _logger.debug(f"Error parsing SPZ header: {e}")

        return None

    def _parse_rad_header(self, path: str) -> Optional[dict]:
        """Parse RAD/SOG file header.

        Args:
            path: Path to the RAD/SOG file.

        Returns:
            Dictionary with vertex_count and format info.
        """
        path_obj = Path(path)
        ext = path_obj.suffix.lower()

        # SOG files are binary bundles - can't parse as text
        if ext == ".sog":
            # Return empty metadata for SOG - actual loading is done by C++ loader
            return {
                "vertex_count": 0,  # Unknown without parsing binary
                "format": "sog",
            }

        # RAD files are text-based
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()

                # Count non-comment, non-empty lines as vertices
                vertex_count = 0
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        vertex_count += 1

                return {
                    "vertex_count": vertex_count if vertex_count > 0 else 0,
                    "format": "rad",
                }

        except Exception as e:
            _logger.debug(f"Error parsing RAD header: {e}")

        return None

    def extract_checkpoint_metadata(self, path: str) -> Optional[dict]:
        """Extract metadata from checkpoint files.

        Args:
            path: Path to the checkpoint file.

        Returns:
            Dictionary with metadata including:
            - iteration: Training iteration (from filename or metadata)
            - num_gaussians: Number of gaussians if available
            - training_params: Training parameters (empty dict for now)
            Returns empty dict if file cannot be read.

        Example:
            >>> meta = scanner.extract_checkpoint_metadata("chkpnt_3000.ckpt")
            >>> print(meta.get("iteration"))
            3000
        """
        path_obj = Path(path)
        result = {
            "iteration": None,
            "num_gaussians": None,
            "training_params": {},
        }

        # Try to extract iteration from filename
        name = path_obj.stem
        iteration_match = re.search(
            r"(?:iteration|iter|chkpnt)[_-]?(\d+)", name, re.IGNORECASE
        )
        if iteration_match:
            result["iteration"] = int(iteration_match.group(1))

        # Try to read checkpoint metadata if it's a JSON or contains metadata
        try:
            # Check if it's a JSON checkpoint
            if path_obj.suffix.lower() == ".json":
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if "iteration" in data:
                        result["iteration"] = data["iteration"]
                    if "num_gaussians" in data:
                        result["num_gaussians"] = data["num_gaussians"]
                    if "training_params" in data:
                        result["training_params"] = data["training_params"]

            # For binary checkpoints, we can't easily extract more info
            # without knowing the exact format

        except (json.JSONDecodeError, IOError, PermissionError) as e:
            _logger.debug(f"Could not read checkpoint metadata from {path}: {e}")

        return result

    def extract_dataset_metadata(self, path: str) -> Optional[dict]:
        """Extract metadata from dataset directories.

        Args:
            path: Path to the dataset directory.

        Returns:
            Dictionary with metadata including:
            - image_count: Number of images in the dataset
            - has_masks: Whether masks subfolder exists
            - sparse_model: Whether sparse/ folder exists
            - camera_count: Number of cameras if detectable
            Returns empty dict if directory cannot be read.

        Example:
            >>> meta = scanner.extract_dataset_metadata("/path/to/dataset")
            >>> print(meta.get("image_count"))
        """
        path_obj = Path(path)
        result = {
            "image_count": 0,
            "has_masks": False,
            "mask_count": 0,
            "sparse_model": False,
            "camera_count": None,
            "database_present": False,
            "image_root": "",
        }

        try:
            image_root = self._get_dataset_image_root(path_obj)
            image_files = self._collect_dataset_images(path_obj)
            result["image_count"] = len(image_files)
            try:
                result["image_root"] = str(image_root.relative_to(path_obj))
            except ValueError:
                result["image_root"] = str(image_root)

            # Check for masks
            masks_dir = path_obj / "masks"
            result["mask_count"] = self._count_image_files(masks_dir) if masks_dir.is_dir() else 0
            result["has_masks"] = result["mask_count"] > 0

            # Check for sparse model (COLMAP)
            sparse_dir = path_obj / "sparse"
            result["sparse_model"] = sparse_dir.is_dir() and any(sparse_dir.iterdir())
            result["database_present"] = (path_obj / "database.db").exists()

            # Try to count cameras and points from COLMAP files
            if result["sparse_model"]:
                cameras_file = next(sparse_dir.rglob("cameras.bin"), None)
                if cameras_file and cameras_file.exists():
                    result["camera_count"] = self._count_colmap_cameras(cameras_file)
                # Count initial points from points3D.bin
                points_file = next(sparse_dir.rglob("points3D.bin"), None)
                if points_file and points_file.exists():
                    result["initial_points"] = self._count_colmap_points(points_file)

        except (OSError, PermissionError) as e:
            _logger.debug(f"Could not scan dataset {path}: {e}")

        return result

    def _count_colmap_cameras(self, cameras_file: Path) -> Optional[int]:
        """Count cameras in COLMAP cameras.bin file.

        Args:
            cameras_file: Path to cameras.bin file.

        Returns:
            Number of cameras or None if cannot read.
        """
        try:
            with open(cameras_file, "rb") as f:
                # COLMAP binary format: uint64 num_cameras, then camera records
                num_cameras_data = f.read(8)
                if len(num_cameras_data) == 8:
                    num_cameras = struct.unpack("<Q", num_cameras_data)[0]
                    return num_cameras
        except Exception as e:
            _logger.debug(f"Could not read COLMAP cameras file: {e}")

        return None

    def _count_colmap_points(self, points_file: Path) -> Optional[int]:
        """Count points in COLMAP points3D.bin file.

        Args:
            points_file: Path to points3D.bin file.

        Returns:
            Number of points or None if cannot read.
        """
        try:
            with open(points_file, "rb") as f:
                # COLMAP binary format: uint64 num_points, then point records
                num_points_data = f.read(8)
                if len(num_points_data) == 8:
                    num_points = struct.unpack("<Q", num_points_data)[0]
                    return num_points
        except Exception as e:
            _logger.debug(f"Could not read COLMAP points file: {e}")

        return None

    def extract_video_metadata(self, path: str) -> Optional[dict]:
        """Extract metadata from video files.

        Args:
            path: Path to the video file.

        Returns:
            Dictionary with metadata including:
            - duration: Video duration in seconds (if available)
            - resolution: Resolution as "WIDTHxHEIGHT" (if available)
            - fps: Frames per second (if available)
            Returns empty dict if file cannot be read.

        Example:
            >>> meta = scanner.extract_video_metadata("flythrough.mp4")
            >>> print(meta.get("duration"))
        """
        result = {
            "duration": None,
            "resolution": None,
            "fps": None,
        }

        # Try to use ffprobe if available
        try:
            import subprocess

            cmd = [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-show_entries",
                "stream=width,height,r_frame_rate",
                "-of",
                "json",
                path,
            ]

            output = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if output.returncode == 0:
                data = json.loads(output.stdout)

                # Extract duration
                if "format" in data and "duration" in data["format"]:
                    try:
                        result["duration"] = float(data["format"]["duration"])
                    except ValueError:
                        pass

                # Extract resolution and fps from first video stream
                if "streams" in data and data["streams"]:
                    stream = data["streams"][0]
                    width = stream.get("width")
                    height = stream.get("height")
                    if width and height:
                        result["resolution"] = f"{width}x{height}"

                    fps_str = stream.get("r_frame_rate", "")
                    if fps_str and "/" in fps_str:
                        num, den = fps_str.split("/")
                        try:
                            result["fps"] = float(num) / float(den)
                        except ValueError:
                            pass

        except (
            subprocess.SubprocessError,
            FileNotFoundError,
            json.JSONDecodeError,
        ) as e:
            _logger.debug(f"Could not extract video metadata with ffprobe: {e}")

        # Fallback: try to get basic info from file size (very rough estimate)
        if result["duration"] is None:
            try:
                size = os.path.getsize(path)
                # Rough estimate: 5MB per minute for typical compressed video
                estimated_duration = (size / (5 * 1024 * 1024)) * 60
                if estimated_duration > 0:
                    result["estimated_duration"] = estimated_duration
            except OSError:
                pass

        return result

    def _extract_json_metadata(self, path: str) -> Optional[dict]:
        """Extract metadata from JSON files.

        Args:
            path: Path to the JSON file.

        Returns:
            Dictionary with metadata including:
            - keys: Top-level keys in the JSON
            - type_hint: Inferred type based on content
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            result = {
                "keys": list(data.keys()) if isinstance(data, dict) else [],
                "type_hint": None,
                "is_array": isinstance(data, list),
            }

            # Try to infer type from content
            if isinstance(data, dict):
                if "gaussians" in data or "point_cloud" in data:
                    result["type_hint"] = "gaussian_data"
                elif "iteration" in data and "training_params" in data:
                    result["type_hint"] = "checkpoint_metadata"
                elif "cameras" in data or "images" in data:
                    result["type_hint"] = "scene_data"

            return result

        except (json.JSONDecodeError, IOError, PermissionError) as e:
            _logger.debug(f"Could not read JSON metadata from {path}: {e}")

        return None

    # ═══════════════════════════════════════════════════════════════════════════════
    # Validation
    # ═══════════════════════════════════════════════════════════════════════════════

    def validate_dataset(self, path: str) -> dict:
        """Validate if a directory looks like a valid dataset.

        Args:
            path: Path to the directory to validate.

        Returns:
            Dictionary with validation results:
            - is_valid: Boolean indicating if it looks like a valid dataset
            - has_images: Whether images were found
            - has_sparse: Whether COLMAP sparse model exists
            - has_masks: Whether masks exist
            - image_count: Number of images found
            - issues: List of issues or warnings

        Example:
            >>> result = scanner.validate_dataset("/path/to/dataset")
            >>> if result["is_valid"]:
            ...     print(f"Found {result['image_count']} images")
        """
        path_obj = Path(path)
        result = {
            "is_valid": False,
            "has_images": False,
            "has_sparse": False,
            "has_masks": False,
            "image_count": 0,
            "issues": [],
        }

        if not path_obj.is_dir():
            result["issues"].append("Path is not a directory")
            return result

        try:
            image_count = len(self._collect_dataset_images(path_obj))
            result["image_count"] = image_count
            result["has_images"] = image_count > 0

            # Check for COLMAP structure
            sparse_dir = path_obj / "sparse"
            result["has_sparse"] = sparse_dir.is_dir() and any(sparse_dir.iterdir())

            # Check for masks
            masks_dir = path_obj / "masks"
            result["has_masks"] = masks_dir.is_dir() and any(masks_dir.iterdir())

            # Validate
            if result["has_images"]:
                result["is_valid"] = True
            elif result["has_sparse"]:
                result["is_valid"] = True
                result["issues"].append("No images found, but sparse model exists")
            else:
                result["issues"].append("No images or sparse model found")

            # Check for minimum image count
            if result["has_images"] and image_count < 3:
                result["issues"].append(
                    f"Very few images ({image_count}), may not reconstruct well"
                )

        except (OSError, PermissionError) as e:
            result["issues"].append(f"Error reading directory: {e}")

        return result

    def is_checkpoint(self, path: str) -> bool:
        """Quick check if a file is a checkpoint.

        Args:
            path: Path to the file.

        Returns:
            True if the file appears to be a checkpoint.

        Example:
            >>> if scanner.is_checkpoint("model.ckpt"):
            ...     print("This is a checkpoint")
        """
        path_obj = Path(path)

        # Check extension
        if path_obj.suffix.lower() == ".ckpt":
            return True

        # Check filename patterns
        name = path_obj.name.lower()
        checkpoint_patterns = [r"checkpoint", r"chkpnt", r"iter[_-]?\d+"]
        for pattern in checkpoint_patterns:
            if re.search(pattern, name, re.IGNORECASE):
                return True

        return False

    def is_gaussian_splat(self, path: str) -> bool:
        """Check if a file is a Gaussian splat (PLY, RAD, SOG, SPZ).

        Args:
            path: Path to the file.

        Returns:
            True if the file is a recognized Gaussian splat format.

        Example:
            >>> if scanner.is_gaussian_splat("model.ply"):
            ...     print("This is a Gaussian splat")
        """
        file_type = self.detect_type(path)
        return file_type in ("ply", "rad", "sog", "spz")

    # ═══════════════════════════════════════════════════════════════════════════════
    # Utilities
    # ═══════════════════════════════════════════════════════════════════════════════

    def get_file_size(self, path: str) -> int:
        """Get file size in bytes.

        Args:
            path: Path to the file.

        Returns:
            File size in bytes, or 0 if file cannot be accessed.

        Example:
            >>> size = scanner.get_file_size("model.ply")
            >>> print(f"Size: {size / (1024*1024):.2f} MB")
        """
        try:
            return os.path.getsize(path)
        except (OSError, PermissionError):
            return 0

    def get_file_size_formatted(self, path: str) -> str:
        """Get formatted file size string.

        Args:
            path: Path to the file.

        Returns:
            Human-readable file size (e.g., "1.5 MB", "2.3 GB").

        Example:
            >>> print(scanner.get_file_size_formatted("model.ply"))
            '150.5 MB'
        """
        size_bytes = self.get_file_size(path)

        if size_bytes == 0:
            return "0 B"

        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0

        return f"{size_bytes:.1f} PB"

    def get_timestamps(self, path: str) -> dict:
        """Get file creation and modification timestamps.

        Args:
            path: Path to the file.

        Returns:
            Dictionary with:
            - created: ISO format creation timestamp
            - modified: ISO format modification timestamp
            - accessed: ISO format last access timestamp

        Example:
            >>> timestamps = scanner.get_timestamps("model.ply")
            >>> print(timestamps["modified"])
        """
        try:
            stat = os.stat(path)
            return {
                "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "accessed": datetime.fromtimestamp(stat.st_atime).isoformat(),
            }
        except (OSError, PermissionError):
            return {
                "created": None,
                "modified": None,
                "accessed": None,
            }

    def get_canonical_path(self, path: str) -> str:
        """Get canonical (absolute, normalized) path.

        Args:
            path: Path to normalize.

        Returns:
            Absolute, normalized path with symlinks resolved.

        Example:
            >>> canonical = scanner.get_canonical_path("./model.ply")
            >>> print(canonical)
            '/home/user/project/model.ply'
        """
        return str(Path(path).resolve())

    def scan_directory(self, path: str, recursive: bool = False) -> list[dict]:
        """Scan a directory for assets and return metadata for all files.

        Args:
            path: Path to the directory to scan.
            recursive: Whether to scan subdirectories recursively.

        Returns:
            List of metadata dictionaries for all detected assets.

        Example:
            >>> assets = scanner.scan_directory("/path/to/assets")
            >>> for asset in assets:
            ...     print(asset["name"], asset["type"])
        """
        path_obj = Path(path)
        results = []

        if not path_obj.is_dir():
            return results

        try:
            if recursive:
                files = path_obj.rglob("*")
            else:
                files = path_obj.iterdir()

            for item in files:
                if item.is_file():
                    metadata = self.scan_file(str(item))
                    if metadata["type"] is not None:
                        results.append(metadata)
                elif item.is_dir() and self._looks_like_dataset(item):
                    metadata = self.scan_file(str(item))
                    results.append(metadata)

        except (OSError, PermissionError) as e:
            _logger.warning(f"Error scanning directory {path}: {e}")

        return results


# ═══════════════════════════════════════════════════════════════════════════════════
# Module-level convenience functions
# ═══════════════════════════════════════════════════════════════════════════════════

_default_scanner: Optional[AssetScanner] = None


def get_scanner() -> AssetScanner:
    """Get the default AssetScanner instance.

    Returns:
        Shared AssetScanner instance.
    """
    global _default_scanner
    if _default_scanner is None:
        _default_scanner = AssetScanner()
    return _default_scanner


def detect_type(path: str) -> Optional[str]:
    """Module-level convenience function to detect file type.

    Args:
        path: Path to the file.

    Returns:
        Detected file type or None.
    """
    return get_scanner().detect_type(path)


def detect_role(path: str, context: Optional[dict] = None) -> str:
    """Module-level convenience function to detect asset role.

    Args:
        path: Path to the file.
        context: Optional context dictionary.

    Returns:
        Detected asset role.
    """
    return get_scanner().detect_role(path, context)


def scan_file(path: str) -> dict:
    """Module-level convenience function to scan a file.

    Args:
        path: Path to the file.

    Returns:
        File metadata dictionary.
    """
    return get_scanner().scan_file(path)


def validate_dataset(path: str) -> dict:
    """Module-level convenience function to validate a dataset.

    Args:
        path: Path to the dataset directory.

    Returns:
        Validation results dictionary.
    """
    return get_scanner().validate_dataset(path)


def is_checkpoint(path: str) -> bool:
    """Module-level convenience function to check if file is a checkpoint.

    Args:
        path: Path to the file.

    Returns:
        True if file is a checkpoint.
    """
    return get_scanner().is_checkpoint(path)


def is_gaussian_splat(path: str) -> bool:
    """Module-level convenience function to check if file is a Gaussian splat.

    Args:
        path: Path to the file.

    Returns:
        True if file is a Gaussian splat.
    """
    return get_scanner().is_gaussian_splat(path)

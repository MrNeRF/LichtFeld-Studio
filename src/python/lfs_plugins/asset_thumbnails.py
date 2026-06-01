# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Asset thumbnail generation and storage for the Asset Manager."""

from __future__ import annotations

import asyncio
import logging
import re
import struct
import time
from pathlib import Path
from typing import Any, Set

_RENDERED_STEM_RE = re.compile(r"^(?P<asset_id>.+)\.render(?:\.\d+)?$")
_DATASET_STEM_RE = re.compile(r"^(?P<asset_id>.+)\.dataset$")

_logger = logging.getLogger(__name__)

# Color mapping for different asset types
ASSET_TYPE_COLORS: dict[str, str] = {
    # Gaussian splats - Blue
    "ply": "#4A90D9",
    "rad": "#4A90D9",
    "sog": "#4A90D9",
    "spz": "#4A90D9",
    # Checkpoints - Green
    "checkpoint": "#5CB85C",
    # Datasets - Orange
    "dataset": "#F0AD4E",
    # USD files - Red
    "usd": "#D9534F",
    "usdz": "#D9534F",
    # JSON - Gray
    "json": "#777777",
}

# Default color for unknown types
DEFAULT_COLOR = "#999999"

# Thumbnail dimensions. Keep this ratio aligned with .asset-card-thumb so RmlUI
# image decorators fill the gallery slot without visibly stretching the source.
THUMB_WIDTH = 512
THUMB_HEIGHT = 224
MAX_RENDERED_PREVIEW_FILE_BYTES = 2 * 1024 * 1024 * 1024

RENDERABLE_PREVIEW_TYPES = {
    "checkpoint",
    "mesh",
    "ply_3dgs",
    "ply_pcl",
    "ply",
    "rad",
    "sog",
    "spz",
}
DATASET_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".tiff",
    ".tif",
    ".bmp",
    ".webp",
    ".exr",
}
DATASET_EXCLUDED_DIRS = {
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


class AssetThumbnails:
    """Manages thumbnail generation and storage for the Asset Manager.

    This class handles creation of placeholder thumbnails for different asset types,
    storage management, and retrieval of thumbnail paths.

    All generation and I/O methods are async so callers never block the UI thread.

    Args:
        thumbnails_dir: Directory path where thumbnails will be stored.

    Example:
        >>> thumbnails = AssetThumbnails(Path("/path/to/thumbnails"))
        >>> thumb_path = await thumbnails.generate_placeholder("ply", "model_01")
        >>> print(thumbnails.get_thumbnail_path("model_01"))
    """

    def __init__(self, thumbnails_dir: Path) -> None:
        """Initialize the AssetThumbnails manager.

        Args:
            thumbnails_dir: Directory path for storing thumbnail files.
        """
        self._thumbnails_dir = Path(thumbnails_dir)
        self._ensure_directory_exists()
        self._missing_thumbnail_path: Path | None = None
        self._warned_once: Set[str] = set()

    def _warn_once(self, asset_id: str, level: str, msg: str, *args) -> None:
        """Log a message only once per asset to avoid console spam."""
        if asset_id in self._warned_once:
            return
        self._warned_once.add(asset_id)
        if level == "error":
            _logger.error(msg, *args)
        elif level == "warning":
            _logger.warning(msg, *args)
        else:
            _logger.info(msg, *args)

    def _ensure_directory_exists(self) -> None:
        """Create the thumbnails directory if it doesn't exist."""
        self._thumbnails_dir.mkdir(parents=True, exist_ok=True)

    def _hex_to_rgb(self, hex_color: str) -> tuple[int, int, int]:
        """Convert hex color string to RGB tuple.

        Args:
            hex_color: Hex color string (e.g., "#4A90D9")

        Returns:
            Tuple of (r, g, b) values (0-255)
        """
        hex_color = hex_color.lstrip("#")
        return (
            int(hex_color[0:2], 16),
            int(hex_color[2:4], 16),
            int(hex_color[4:6], 16),
        )

    def _create_png_data(
        self, width: int, height: int, color: str, label: str
    ) -> bytes:
        """Create a minimal PNG image with the given color and label.

        Uses a pure-Python PNG encoder that doesn't require external dependencies.
        Creates a solid color rectangle with the type label centered.

        Args:
            width: Image width in pixels
            height: Image height in pixels
            color: Hex color string
            label: Text label to display on the image

        Returns:
            PNG image data as bytes
        """
        rgb = self._hex_to_rgb(color)

        # Create a simple PNG with the given color
        # PNG signature
        png_signature = b"\x89PNG\r\n\x1a\n"

        def png_chunk(chunk_type: bytes, data: bytes) -> bytes:
            """Create a PNG chunk with length, type, data, and CRC."""
            chunk = chunk_type + data
            crc = self._crc32(chunk) & 0xFFFFFFFF
            return struct.pack(">I", len(data)) + chunk + struct.pack(">I", crc)

        # IHDR chunk
        ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
        ihdr_chunk = png_chunk(b"IHDR", ihdr_data)

        # Create image data (RGB, no filter)
        raw_data = bytearray()
        for _ in range(height):
            raw_data.append(0)  # Filter byte (0 = no filter)
            for _ in range(width):
                raw_data.extend(rgb)

        # Compress image data
        try:
            import zlib

            compressed = zlib.compress(bytes(raw_data), level=6)
        except ImportError:
            # Fallback: uncompressed (not valid PNG, but won't happen in practice)
            compressed = bytes(raw_data)

        idat_chunk = png_chunk(b"IDAT", compressed)

        # IEND chunk
        iend_chunk = png_chunk(b"IEND", b"")

        return png_signature + ihdr_chunk + idat_chunk + iend_chunk

    def _crc32(self, data: bytes) -> int:
        """Calculate CRC32 for PNG chunks.

        Args:
            data: Data to calculate CRC for

        Returns:
            CRC32 value
        """
        try:
            import zlib

            return zlib.crc32(data)
        except ImportError:
            # Pure Python fallback (rarely needed)
            crc_table = []
            for i in range(256):
                c = i
                for _ in range(8):
                    if c & 1:
                        c = 0xEDB88320 ^ (c >> 1)
                    else:
                        c >>= 1
                crc_table.append(c)

            crc = 0xFFFFFFFF
            for byte in data:
                crc = crc_table[(crc ^ byte) & 0xFF] ^ (crc >> 8)
            return crc ^ 0xFFFFFFFF

    def _create_thumbnail(
        self, width: int, height: int, color: str, label: str
    ) -> bytes:
        """Create thumbnail image data using the pure-Python PNG encoder.

        Args:
            width: Image width in pixels
            height: Image height in pixels
            color: Hex color string
            label: Text label to display on the image

        Returns:
            PNG image data as bytes
        """
        return self._create_png_data(width, height, color, label)

    async def generate_placeholder(self, asset_type: str, asset_id: str) -> Path:
        """Generate a placeholder thumbnail for an asset.

        Creates a color-coded placeholder image based on the asset type.
        The image is saved to the thumbnails directory.

        Args:
            asset_type: Type of asset (e.g., "ply", "checkpoint")
            asset_id: Unique identifier for the asset

        Returns:
            Path to the generated thumbnail file
        """
        # Get color for this asset type
        color = ASSET_TYPE_COLORS.get(asset_type.lower(), DEFAULT_COLOR)

        # Create thumbnail filename
        thumb_path = self._thumbnails_dir / f"{asset_id}.png"

        # Generate thumbnail image data
        try:
            image_data = self._create_thumbnail(
                THUMB_WIDTH, THUMB_HEIGHT, color, asset_type.upper()
            )
        except Exception as exc:
            _logger.error(
                "Placeholder thumbnail creation failed for %s (type=%s): _create_thumbnail raised %s: %s",
                asset_id,
                asset_type,
                type(exc).__name__,
                exc,
            )
            raise

        # Write to file
        try:
            await asyncio.to_thread(thumb_path.write_bytes, image_data)
        except Exception as exc:
            _logger.error(
                "Placeholder thumbnail write failed for %s (type=%s): cannot write to %s: %s",
                asset_id,
                asset_type,
                thumb_path,
                exc,
            )
            raise

        return thumb_path

    def get_rendered_thumbnail_path(self, asset_id: str) -> Path:
        """Get the cached rendered-preview path for a splat asset."""
        return self._thumbnails_dir / f"{asset_id}.render.png"

    def _get_timestamped_rendered_thumbnail_path(self, asset_id: str) -> Path:
        """Get a unique rendered-preview path so RmlUI reloads the texture."""
        timestamp = int(time.time())
        return self._thumbnails_dir / f"{asset_id}.render.{timestamp}.png"

    async def _cleanup_old_rendered_thumbnails(
        self, asset_id: str, keep: Path | None = None
    ) -> None:
        """Remove stale rendered thumbnails for an asset, optionally keeping one."""
        pattern = f"{asset_id}.render.*.png"

        def _do_cleanup() -> None:
            for old in self._thumbnails_dir.glob(pattern):
                if keep is not None and old == keep:
                    continue
                try:
                    old.unlink()
                except Exception as exc:
                    _logger.debug("Failed to remove stale thumbnail %s: %s", old, exc)

        await asyncio.to_thread(_do_cleanup)

    def has_rendered_thumbnail(self, asset_id: str) -> bool:
        """Return whether any rendered thumbnail exists for this asset."""
        pattern = f"{asset_id}.render.*.png"
        return any(self._thumbnails_dir.glob(pattern))

    def get_dataset_thumbnail_path(self, asset_id: str) -> Path:
        """Get the cached dataset-image thumbnail path for a dataset asset."""
        return self._thumbnails_dir / f"{asset_id}.dataset.png"

    async def _find_first_dataset_image(
        self,
        dataset_path: Path,
        dataset_metadata: dict[str, Any] | None = None,
    ) -> Path | None:
        """Find the first real image in a dataset using AssetScanner-compatible rules."""

        def _do_find() -> Path | None:
            if not dataset_path.is_dir():
                return None

            image_root_value = (dataset_metadata or {}).get("image_root", "")
            image_root = None
            if image_root_value:
                candidate = Path(str(image_root_value)).expanduser()
                image_root = candidate if candidate.is_absolute() else dataset_path / candidate
            if image_root is None or not image_root.is_dir():
                images_dir = dataset_path / "images"
                image_root = images_dir if images_dir.is_dir() else dataset_path

            image_paths: dict[str, Path] = {}
            try:
                for item in image_root.rglob("*"):
                    if not item.is_file() or item.suffix.lower() not in DATASET_IMAGE_EXTENSIONS:
                        continue
                    try:
                        rel_parent_parts = item.relative_to(image_root).parts[:-1]
                    except ValueError:
                        rel_parent_parts = item.parts[:-1]
                    if any(part.lower() in DATASET_EXCLUDED_DIRS for part in rel_parent_parts):
                        continue
                    image_paths[str(item.resolve())] = item
            except (OSError, PermissionError):
                return None

            if not image_paths:
                return None
            return sorted(image_paths.values(), key=lambda item: str(item))[0]

        return await asyncio.to_thread(_do_find)

    async def generate_dataset_preview(
        self,
        asset_type: str,
        asset_id: str,
        dataset_path: str | Path,
        dataset_metadata: dict[str, Any] | None = None,
    ) -> Path | None:
        """Generate a thumbnail from the first dataset image.

        Returns the source image path directly so the UI can still show a real
        dataset image. PIL is not available in this environment, so no resizing
        or format conversion is performed.
        """
        if asset_type.lower() != "dataset" or not dataset_path:
            return None

        first_image = await self._find_first_dataset_image(
            Path(dataset_path).expanduser(),
            dataset_metadata,
        )
        if first_image is None:
            return None

        return first_image if first_image.exists() else None

    async def _generate_rendered_preview(
        self,
        asset_type: str,
        asset_id: str,
        asset_path: str | Path,
        render_preview: Any,
        save_image: Any,
        **render_kwargs: Any,
    ) -> Path | None:
        """Shared helper for rendered thumbnail generation."""
        if asset_type.lower() not in RENDERABLE_PREVIEW_TYPES:
            self._warn_once(
                asset_id, "error",
                "Thumbnail render skipped for %s: asset type '%s' is not in RENDERABLE_PREVIEW_TYPES (%s)",
                asset_id,
                asset_type,
                RENDERABLE_PREVIEW_TYPES,
            )
            return None
        if not asset_path:
            self._warn_once(asset_id, "error", "Thumbnail render skipped for %s: asset_path is empty", asset_id)
            return None
        if not callable(render_preview):
            self._warn_once(
                asset_id, "error",
                "Thumbnail render skipped for %s: render_preview function is not available (lichtfeld.render_asset_preview may be missing)",
                asset_id,
            )
            return None
        if not callable(save_image):
            self._warn_once(
                asset_id, "error",
                "Thumbnail render skipped for %s: save_image function is not available (lichtfeld.io.save_image may be missing)",
                asset_id,
            )
            return None

        path = Path(asset_path).expanduser()
        try:
            if path.is_file() and path.stat().st_size > MAX_RENDERED_PREVIEW_FILE_BYTES:
                self._warn_once(
                    asset_id, "error",
                    "Thumbnail render skipped for %s: file size %d bytes exceeds %d MiB budget",
                    asset_id,
                    path.stat().st_size,
                    MAX_RENDERED_PREVIEW_FILE_BYTES // (1024 * 1024),
                )
                return None
        except OSError as exc:
            self._warn_once(
                asset_id, "error",
                "Thumbnail render skipped for %s: cannot stat file %s: %s",
                asset_id,
                path,
                exc,
            )
            return None

        try:
            image = await asyncio.to_thread(
                render_preview,
                str(path),
                width=THUMB_WIDTH,
                height=THUMB_HEIGHT,
                **render_kwargs,
            )
        except Exception as exc:
            self._warn_once(
                asset_id, "error",
                "Thumbnail render failed for %s: render_preview(%s) raised %s: %s",
                asset_id,
                path,
                type(exc).__name__,
                exc,
            )
            return None

        if image is None:
            self._warn_once(
                asset_id, "error",
                "Thumbnail render failed for %s: render_preview(%s) returned None (renderer could not load or render the file)",
                asset_id,
                path,
            )
            return None

        thumb_path = self._get_timestamped_rendered_thumbnail_path(asset_id)
        try:
            await asyncio.to_thread(save_image, str(thumb_path), image)
        except Exception as exc:
            self._warn_once(
                asset_id, "error",
                "Thumbnail render failed for %s: save_image(%s) raised %s: %s",
                asset_id,
                thumb_path,
                type(exc).__name__,
                exc,
            )
            return None

        if thumb_path.exists():
            self._warned_once.discard(asset_id)
            await self._cleanup_old_rendered_thumbnails(asset_id, keep=thumb_path)
            return thumb_path

        self._warn_once(
            asset_id, "error",
            "Thumbnail render failed for %s: save_image wrote to %s but file does not exist after save",
            asset_id,
            thumb_path,
        )
        return None

    async def generate_rendered_preview(
        self,
        asset_type: str,
        asset_id: str,
        asset_path: str | Path,
    ) -> Path | None:
        """Generate a rendered thumbnail for a splat asset using the app renderer."""
        try:
            import lichtfeld as lf

            render_fn = getattr(lf, "render_asset_preview", None)
            save_fn = getattr(getattr(lf, "io", None), "save_image", None)
            if render_fn is None:
                _logger.error(
                    "Thumbnail render unavailable for %s: lichtfeld.render_asset_preview is not exposed in the Python API",
                    asset_id,
                )
            if save_fn is None:
                _logger.error(
                    "Thumbnail render unavailable for %s: lichtfeld.io.save_image is not exposed in the Python API",
                    asset_id,
                )
            return await self._generate_rendered_preview(
                asset_type,
                asset_id,
                asset_path,
                render_fn,
                save_fn,
            )
        except Exception as exc:
            _logger.error(
                "Thumbnail render failed for %s: unexpected error in generate_rendered_preview: %s: %s",
                asset_id,
                type(exc).__name__,
                exc,
            )
            return None

    async def generate_rendered_preview_from_camera(
        self,
        asset_type: str,
        asset_id: str,
        asset_path: str | Path,
        eye: tuple[float, float, float],
        target: tuple[float, float, float],
        up: tuple[float, float, float] = (0.0, 1.0, 0.0),
    ) -> Path | None:
        """Generate a rendered thumbnail from a custom camera pose."""
        try:
            import lichtfeld as lf

            render_fn = getattr(lf, "render_asset_preview_from_camera", None)
            save_fn = getattr(getattr(lf, "io", None), "save_image", None)
            if render_fn is None:
                _logger.error(
                    "Thumbnail render from camera unavailable for %s: lichtfeld.render_asset_preview_from_camera is not exposed",
                    asset_id,
                )
            if save_fn is None:
                _logger.error(
                    "Thumbnail render from camera unavailable for %s: lichtfeld.io.save_image is not exposed",
                    asset_id,
                )
            return await self._generate_rendered_preview(
                asset_type,
                asset_id,
                asset_path,
                render_fn,
                save_fn,
                eye=eye,
                target=target,
                up=up,
            )
        except Exception as exc:
            _logger.error(
                "Thumbnail render from camera failed for %s: unexpected error: %s: %s",
                asset_id,
                type(exc).__name__,
                exc,
            )
            return None

    def get_thumbnail_path(self, asset_id: str) -> Path:
        """Get the path to a thumbnail for the given asset.

        Args:
            asset_id: Unique identifier for the asset

        Returns:
            Path to the thumbnail file (may not exist)
        """
        return self._thumbnails_dir / f"{asset_id}.png"

    async def get_missing_thumbnail(self) -> Path:
        """Get the path to the fallback thumbnail for missing/corrupt thumbnails.

        Creates the missing thumbnail if it doesn't exist.

        Returns:
            Path to the missing thumbnail file
        """
        if self._missing_thumbnail_path is None:
            missing_path = self._thumbnails_dir / "_missing.png"
            if not missing_path.exists():
                # Create a gray placeholder with "?" label
                image_data = self._create_thumbnail(
                    THUMB_WIDTH, THUMB_HEIGHT, DEFAULT_COLOR, "?"
                )
                await asyncio.to_thread(missing_path.write_bytes, image_data)
            self._missing_thumbnail_path = missing_path

        return self._missing_thumbnail_path

    def thumbnail_exists(self, asset_id: str) -> bool:
        """Check if a thumbnail exists for the given asset.

        Args:
            asset_id: Unique identifier for the asset

        Returns:
            True if the thumbnail file exists, False otherwise
        """
        return self.get_thumbnail_path(asset_id).exists()

    async def invalidate(self, asset_id: str) -> None:
        """Invalidate (remove) a thumbnail for the given asset.

        This removes the existing thumbnail file. A new placeholder can be
        generated by calling generate_placeholder().

        Args:
            asset_id: Unique identifier for the asset
        """
        thumb_path = self.get_thumbnail_path(asset_id)
        if thumb_path.exists():
            await asyncio.to_thread(thumb_path.unlink)

    async def cleanup_orphans(self, known_asset_ids: Set[str]) -> list[Path]:
        """Remove thumbnails for assets that no longer exist.

        Args:
            known_asset_ids: Set of valid asset IDs that should have thumbnails

        Returns:
            List of paths that were removed
        """
        removed: list[Path] = []

        def _do_cleanup() -> list[Path]:
            _removed: list[Path] = []
            for thumb_file in self._thumbnails_dir.glob("*.png"):
                # Skip special files
                if thumb_file.name.startswith("_"):
                    continue

                stem = thumb_file.stem
                rendered_match = _RENDERED_STEM_RE.match(stem)
                dataset_match = _DATASET_STEM_RE.match(stem)
                if rendered_match:
                    asset_id = rendered_match.group("asset_id")
                elif dataset_match:
                    asset_id = dataset_match.group("asset_id")
                else:
                    asset_id = stem

                if asset_id not in known_asset_ids:
                    thumb_file.unlink()
                    _removed.append(thumb_file)
            return _removed

        removed = await asyncio.to_thread(_do_cleanup)
        return removed

    async def get_thumbnail_for_type(self, asset_type: str) -> Path:
        """Get a generic thumbnail for a specific asset type.

        This creates or returns a shared thumbnail that represents the type
        rather than a specific asset.

        Args:
            asset_type: Type of asset (e.g., "ply", "checkpoint")

        Returns:
            Path to the type thumbnail file
        """
        type_thumb_path = self._thumbnails_dir / f"_type_{asset_type.lower()}.png"

        if not type_thumb_path.exists():
            color = ASSET_TYPE_COLORS.get(asset_type.lower(), DEFAULT_COLOR)
            image_data = self._create_thumbnail(
                THUMB_WIDTH, THUMB_HEIGHT, color, asset_type.upper()
            )
            await asyncio.to_thread(type_thumb_path.write_bytes, image_data)

        return type_thumb_path

    @property
    def thumbnails_dir(self) -> Path:
        """Get the thumbnails directory path."""
        return self._thumbnails_dir

    def get_all_thumbnail_paths(self) -> list[Path]:
        """Get all thumbnail file paths in the thumbnails directory.

        Returns:
            List of paths to all thumbnail files
        """
        return list(self._thumbnails_dir.glob("*.png"))

    async def clear_all(self) -> int:
        """Remove all thumbnails from the directory.

        Returns:
            Number of thumbnails removed
        """
        count = 0

        def _do_clear() -> int:
            _count = 0
            for thumb_file in self._thumbnails_dir.glob("*.png"):
                thumb_file.unlink()
                _count += 1
            return _count

        return await asyncio.to_thread(_do_clear)

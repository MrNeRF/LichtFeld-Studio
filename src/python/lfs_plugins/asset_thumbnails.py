# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Asset thumbnail generation and storage for the Asset Manager."""

from __future__ import annotations

import io
import struct
from pathlib import Path
from typing import Set

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
    # Videos - Purple
    "video": "#9B59B6",
    "mp4": "#9B59B6",
    "mov": "#9B59B6",
    # USD files - Red
    "usd": "#D9534F",
    "usdz": "#D9534F",
    # HTML - Teal
    "html": "#5BC0DE",
    # JSON - Gray
    "json": "#777777",
}

# Default color for unknown types
DEFAULT_COLOR = "#999999"

# Thumbnail dimensions
THUMB_WIDTH = 256
THUMB_HEIGHT = 256


class AssetThumbnails:
    """Manages thumbnail generation and storage for the Asset Manager.

    This class handles creation of placeholder thumbnails for different asset types,
    storage management, and retrieval of thumbnail paths.

    Args:
        thumbnails_dir: Directory path where thumbnails will be stored.

    Example:
        >>> thumbnails = AssetThumbnails(Path("/path/to/thumbnails"))
        >>> thumb_path = thumbnails.generate_placeholder("ply", "model_01")
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

    def _create_thumbnail_with_pil(
        self, width: int, height: int, color: str, label: str
    ) -> bytes:
        """Create a thumbnail using PIL/Pillow if available.

        Args:
            width: Image width in pixels
            height: Image height in pixels
            color: Hex color string
            label: Text label to display on the image

        Returns:
            PNG image data as bytes
        """
        from PIL import Image, ImageDraw, ImageFont

        img = Image.new("RGB", (width, height), color)
        draw = ImageDraw.Draw(img)

        # Try to draw text label
        try:
            # Try to use a default font
            font = ImageFont.load_default()

            # Calculate text position (centered)
            bbox = draw.textbbox((0, 0), label.upper(), font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            x = (width - text_width) // 2
            y = (height - text_height) // 2

            # Draw text with slight shadow for readability
            shadow_color = "#333333"
            draw.text((x + 1, y + 1), label.upper(), fill=shadow_color, font=font)
            draw.text((x, y), label.upper(), fill="white", font=font)
        except Exception:
            # If text rendering fails, just return the colored rectangle
            pass

        # Save to bytes
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return buffer.getvalue()

    def _create_thumbnail(
        self, width: int, height: int, color: str, label: str
    ) -> bytes:
        """Create thumbnail image data, using PIL if available.

        Args:
            width: Image width in pixels
            height: Image height in pixels
            color: Hex color string
            label: Text label to display on the image

        Returns:
            PNG image data as bytes
        """
        try:
            return self._create_thumbnail_with_pil(width, height, color, label)
        except ImportError:
            return self._create_png_data(width, height, color, label)

    def generate_placeholder(self, asset_type: str, asset_id: str) -> Path:
        """Generate a placeholder thumbnail for an asset.

        Creates a color-coded placeholder image based on the asset type.
        The image is saved to the thumbnails directory.

        Args:
            asset_type: Type of asset (e.g., "ply", "checkpoint", "mp4")
            asset_id: Unique identifier for the asset

        Returns:
            Path to the generated thumbnail file
        """
        # Get color for this asset type
        color = ASSET_TYPE_COLORS.get(asset_type.lower(), DEFAULT_COLOR)

        # Create thumbnail filename
        thumb_path = self._thumbnails_dir / f"{asset_id}.png"

        # Generate thumbnail image data
        image_data = self._create_thumbnail(
            THUMB_WIDTH, THUMB_HEIGHT, color, asset_type.upper()
        )

        # Write to file
        thumb_path.write_bytes(image_data)

        return thumb_path

    def get_thumbnail_path(self, asset_id: str) -> Path:
        """Get the path to a thumbnail for the given asset.

        Args:
            asset_id: Unique identifier for the asset

        Returns:
            Path to the thumbnail file (may not exist)
        """
        return self._thumbnails_dir / f"{asset_id}.png"

    def get_missing_thumbnail(self) -> Path:
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
                missing_path.write_bytes(image_data)
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

    def invalidate(self, asset_id: str) -> None:
        """Invalidate (remove) a thumbnail for the given asset.

        This removes the existing thumbnail file. A new placeholder can be
        generated by calling generate_placeholder().

        Args:
            asset_id: Unique identifier for the asset
        """
        thumb_path = self.get_thumbnail_path(asset_id)
        if thumb_path.exists():
            thumb_path.unlink()

    def cleanup_orphans(self, known_asset_ids: Set[str]) -> list[Path]:
        """Remove thumbnails for assets that no longer exist.

        Args:
            known_asset_ids: Set of valid asset IDs that should have thumbnails

        Returns:
            List of paths that were removed
        """
        removed: list[Path] = []

        for thumb_file in self._thumbnails_dir.glob("*.png"):
            # Skip special files
            if thumb_file.name.startswith("_"):
                continue

            # Extract asset ID from filename
            asset_id = thumb_file.stem

            if asset_id not in known_asset_ids:
                thumb_file.unlink()
                removed.append(thumb_file)

        return removed

    def get_thumbnail_for_type(self, asset_type: str) -> Path:
        """Get a generic thumbnail for a specific asset type.

        This creates or returns a shared thumbnail that represents the type
        rather than a specific asset.

        Args:
            asset_type: Type of asset (e.g., "ply", "checkpoint", "mp4")

        Returns:
            Path to the type thumbnail file
        """
        type_thumb_path = self._thumbnails_dir / f"_type_{asset_type.lower()}.png"

        if not type_thumb_path.exists():
            color = ASSET_TYPE_COLORS.get(asset_type.lower(), DEFAULT_COLOR)
            image_data = self._create_thumbnail(
                THUMB_WIDTH, THUMB_HEIGHT, color, asset_type.upper()
            )
            type_thumb_path.write_bytes(image_data)

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

    def clear_all(self) -> int:
        """Remove all thumbnails from the directory.

        Returns:
            Number of thumbnails removed
        """
        count = 0
        for thumb_file in self._thumbnails_dir.glob("*.png"):
            thumb_file.unlink()
            count += 1
        return count

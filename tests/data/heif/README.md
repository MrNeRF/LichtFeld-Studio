# Synthetic HEIF/HEIC decoder fixtures

Original procedural RGB/RGBA ramps, licensed GPL-3.0-or-later with this repository.
No photographs, camera data or third-party binary samples are included.

Regenerate with `uv run --no-project --with pillow-heif==1.8.1 python tests/data/heif/generate.py`.
The generation environment is not an application or CI dependency. Tests read
committed HEVC fixtures and reference pixels without requiring any encoder.
`manifest.json` records the actual decoded size, precision, color tags and file hashes.

Reference RGB is decoded with pillow-heif's bundled libheif, independently of
the application's pinned dependency. High precision reference samples are
left-aligned little-endian; the core UINT16 contract expands them to full range.
Color assertions additionally use a scalar Display P3 to sRGB matrix/reference.
The rotation fixture contains a container transform; an EXIF label alone would
not verify that transform. PQ is explicitly tagged and exercises raw-preserving
read versus unsupported implicit SDR display conversion.

The RGB8 fixture is a 128x96 grid of 64-pixel tiles, exercising a partial bottom
tile; other fixtures are single images. Only the primary image is decoded.

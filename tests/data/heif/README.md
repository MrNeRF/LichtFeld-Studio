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
The original Display P3 ICC profile is also tested against contradictory sRGB
NCLX tags to establish ICC precedence. SDR BT.709/linear transfers have scalar
references. The untagged fixture is monochrome. YUV420 uses lossy coding, a
nonidentity matrix and explicitly verified limited-range tags. High precision
fixtures originate from adjacent native codes inside one RGB8 bucket, including
RGBA12; they are not enlarged RGB8 inputs. Generated ICC creation dates are fixed
so repeated regeneration does not change the fixture just because time passed.
The rotation fixture contains a container transform; an EXIF label alone would
not verify that transform. PQ is explicitly tagged and exercises raw-preserving
read versus unsupported implicit SDR display conversion.

The RGB8 fixture is a 128x96 grid of 64-pixel tiles, exercising a partial bottom
tile; other fixtures are single images. Only the primary image is decoded.
The small single-image fixtures contain clean-aperture (`clap`) crop properties
for padded coded dimensions; the decoded dimensions/pixels test that crop along
with rotation/mirroring. The grid case separately tests partial tile assembly.

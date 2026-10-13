# HEIF/HEIC image import

Development branch: import only. HEIF is a container family; this build enables
HEVC still-image decoding through libheif/libde265. Other codecs inside HEIF
are not implicitly supported. No HEIC writer or AVIF decoder is enabled.

The primary image is selected, with container crop/rotation/mirror applied by
libheif. Other top-level images, depth/disparity, gain maps, portrait mattes,
sequences and telemetry are not exposed in this increment. These are planned
separate capabilities, not flattened into RGB or inferred from an extension.

Dataset discovery recognizes `.heic` and `.heif` case-insensitively. Like EXR
and WebP companions, same-stem HEIF files do not inflate legacy dataset counts
or displace legacy preview images. Explicit COLMAP references still select
the named image. Existing JPEG/PNG/TIFF/EXR readers and defaults are unchanged.

Raw codec decode and `load_image_float` retain encoded RGB values (after the
codec's YUV-to-RGB conversion and geometric transforms). 9–16 bit samples use
native-endian UINT16 expanded to [0,65535], without an intermediate RGB8 image.
These values are not automatically linear light or metric data.

Display/image loaders use the RGB ICC profile when present, otherwise supported
SDR NCLX primaries/transfer, and convert to the existing bounded sRGB contract
before requested quantization/resizing. Alpha is preserved; associated alpha
is unpremultiplied for display. Out-of-gamut colors are clipped by that bounded
contract. Untagged RGB follows the existing sRGB assumption. Unknown or invalid
profiles produce an error rather than a silently guessed color conversion.
PQ/HLG display import requires an explicit HDR policy and is currently rejected;
raw decoding remains available. A higher precision training path is not enabled.

Dependencies are pinned by the existing vcpkg baseline: libheif 1.21.2,
libde265 and Little CMS. The default encoder features are disabled. CPU decode
is the portable provider; this increment makes no hardware acceleration claim.
License texts are shipped in `resources/licenses` and listed in
`THIRD_PARTY_LICENSES.md`. Windows runtime dependency copying includes these DLLs.

Tests use committed procedural fixtures, generated independently of the
application's libraries. No optional encoder is required to execute tests.
See `tests/data/heif/README.md` for regeneration and reference conventions.
Cross-platform application and distribution qualification must be recorded
before treating this development increment as release-ready.

## Targeted qualification

Windows, Linux CPU and macOS pass the codec, dataset/image-loader, exports and
TIFF regression CTest groups. Codec results: Windows 256 passed; Linux 258
passed; macOS 256 passed and 2 existing `/dev/full` output-failure cases skipped
because that device is absent. Dataset/image-loader coverage has 4 passing cases. Fixtures include a grid with a partial tile and concurrent reads.
The Windows CMake stub check matches committed stubs. These results do not claim
hosted CI, installer or full root-suite qualification, nor a real iPhone corpus.

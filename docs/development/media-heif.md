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
raw decoding remains available. The existing `decode_16bit` option preserves
source precision through image loading; its default and training policy are unchanged.

File, memory and caller-allocated target entrypoints share this decoder. Memory
input stays alive for the duration of decoding; target decode stages the decoded
image before copying/converting it, and is not a zero-copy claim. Bounded header
recognition is advisory; full decoding still validates the container and pixels.

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

Windows and macOS pass six targeted CTest groups, including asynchronous/cache
consumers on CUDA/Vulkan and Metal/Vulkan respectively. Linux CPU passes four
groups. Codec results: Windows 258 passed; Linux 260 passed; macOS 258 passed
with two existing `/dev/full` output-failure skips. Dataset/image-loader results:
Windows/macOS six passed; Linux five passed with the actual Blender camera
construction skipped because the validation container has no GPU. Header and
memory tests still execute there. CMake-generated Python stubs match committed stubs.

Fixtures cover native adjacent 10/12-bit codes, high-bitdepth alpha, monochrome,
lossy limited-range YUV420, nonidentity matrices, SDR transfers, nontrivial ICC,
ICC/NCLX precedence, HDR rejection, geometric transforms and partial grid tiles.
Cold decoded caches and asynchronous loaders preserve original samples exactly.
The existing CUDA derived JPEG 2000 cache bounds and rounds resized float samples
to UINT16; its test uses that explicit contract rather than byte equality.
The audit also fixed the cache writer's ambiguous CHW layout for tiny images by
passing HWC explicitly at that call site.

A private 4032x3024 iPhone 13 photo with an 8x6 grid, partial edge tiles and a
Display P3 ICC profile decodes identically on all three platforms and with the
independent decoder. Apple ImageIO is not pixel-identical: mean absolute RGB8
error is 0.626, with a maximum of 62; 99.68% of channel samples differ by at most
eight levels. Applying the same LCMS transform to ImageIO's original-space RGB
retains the difference, whereas Apple's color conversion alone differs by at
most one level. This records a decoder/provider difference, not an equivalence
claim or a widened fixture-test tolerance. The photo is not redistributed.
Two private alternatives were measured and rejected: forced bilinear chroma
upsampling barely lowered mean error (0.619) but increased the maximum to 96
and nearly doubled decode time; UINT16 display intermediates increased mean
error to 1.223. Neither experimental setting is enabled in this revision.

A Windows consumer runs with resolved DLLs from an isolated directory and a
system-only PATH. The three new decoder/color DLLs total 2,135,040 bytes. This
does not establish full installer, hosted CI or root-suite qualification.
Recorded photo decode measurements are diagnostic, not acceleration claims.

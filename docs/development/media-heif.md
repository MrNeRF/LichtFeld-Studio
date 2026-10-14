# AVIF / AV1-based HEIF image import

This provider imports AV1-coded still images through libheif and dav1d.
HEVC-coded HEIC is unsupported with an explicit diagnostic, including when its
extension is changed. HEIF is a container family, not a guarantee of support
for every codec, auxiliary image or sequence inside it.

The distribution policy excludes libde265 and every HEVC path from this
libheif provider. `.github/overlays/libheif` retains the exact 1.21.2 vcpkg
baseline sources/patches, replaces the unconditional libde265 dependency with
dav1d, and disables other codecs, encoders and dynamic plugin loading. dav1d is
already used by the video stack. This policy is distinct from source licenses
and is not a general statement about patent obligations.

Primary-image AV1 items, including derived grid/identity references, are
validated before decoding. libheif applies container transforms and assembles
partial grid tiles. Secondary images, sequences, depth/disparity, gain maps,
portrait mattes and full metadata are outside this increment.

Dataset discovery recognizes `.avif` and `.heif` case-insensitively. Same-stem
companions do not inflate legacy counts or displace legacy previews. Explicit
references select the named file. `.heic` decoder dispatch is retained for clear
unsupported-codec diagnostics, not advertised as a supported dataset extension.

File, memory and caller-allocated targets use the existing codec. Filesystem
reads support native Unicode paths without another complete compressed-file
copy. Memory input stays alive during decoding. Target decode stages pixels
before conversion/copying and is not zero-copy. Bounded-header recognition is
advisory; full probe/decode validates the primary image and codec. Existing
JPEG/PNG/TIFF/EXR/WebP readers are unchanged.

Raw decode and `load_image_float` preserve encoded RGB samples after YUV
conversion and geometric transforms. Higher-precision samples use native-endian
UINT16 expanded to `[0,65535]`; they are not automatically linear or metric.
Existing training precision flags/defaults are unchanged.

Display uses a valid RGB ICC first, including over contradictory PQ/HLG NCLX.
Without ICC, supported SDR NCLX is converted to bounded sRGB; PQ/HLG fallback is
rejected pending explicit HDR policy. Invalid ICC is not silently ignored.
Untagged RGB follows the existing sRGB assumption. Alpha is preserved and
associated alpha is unpremultiplied for display.

LCMS setup reuses one cached transform per thread. Keys compare full ICC bytes
or resolved NCLX chromaticities/transfer, plus RGB/RGBA layout and precision.
Profile changes replace the entry; failures are not cached. Mutable transforms
are not shared between workers and there is no unbounded global cache.

Original AV1 fixtures and independent AOM-decoded references cover native
adjacent 10/12-bit codes, RGBA12, monochrome, limited-range lossy YUV420,
nonidentity matrices, SDR transfers, explicit ICC/NCLX precedence, invalid ICC,
PQ/HLG, primary selection, rotation/mirroring and a partial-tile AV1 grid.
One original HEVC grid is kept only for rejection tests. Runtime tests require
dav1d and verify that HEVC decoding and AV1/HEVC encoding are unavailable.
Test execution requires no encoder. See `tests/data/heif/README.md`.

Cache/queue tests preserve original samples exactly. The existing CUDA derived
JPEG 2000 cache clamps and rounds resized floats to UINT16; its test checks that
contract. A tiny-image CHW/HWC ambiguity is fixed at the cache-write call site
by passing HWC explicitly.

Validation must use this AV1-only configuration. Earlier HEIC photo comparisons
and timings concern the retired provider and do not qualify AVIF or promise
equivalence with ImageIO. Targeted tests do not establish hosted CI, complete
installer qualification or a full root-suite run.

Future HEIC import requires separately qualified OS-provided decoders (Apple
ImageIO/Core Image or Windows WIC with the HEIF extension), availability checks
and explicit unsupported behavior elsewhere. No such provider, image writer or
implicit HDR display conversion is added here.

System-provider qualification and its measured limitations are tracked in
[`media-heic-system-providers.md`](media-heic-system-providers.md). The diagnostic
probes do not enable HEIC in this provider or change its distribution policy.

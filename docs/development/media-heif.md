# HEIF image import

The existing image codec reads AV1-coded AVIF/HEIF with libheif and dav1d.
HEVC-coded HEIC selects Microsoft WIC on Windows or ImageIO on macOS from the
actual coded primary/derived item, independently of the filename.

| Provider | Supported subset | Explicit limitations |
| --- | --- | --- |
| AV1/dav1d, all platforms | Qualified RGB/RGBA 8/10/12-bit, primary items and derived grids, container transforms, raw/display reads | Auxiliary depth, gain maps, sequences and full metadata are not imported. |
| Windows WIC | Qualified RGB8, including YUV 4:2:0 | Requires HEVC Video Extensions and HEIF Image Extensions. Higher precision, alpha, RGB grids and combined rotation/mirroring are rejected. |
| Apple ImageIO | Qualified RGB8 and packed RGB101010 normalized to UINT16 | Alpha, 12-bit, unknown native layouts, repeated/late clean aperture, and ICC plus contradictory HDR NCLX are rejected. |
| Linux HEVC | None | HEVC input reports unsupported before display-policy validation. |

Windows checks for an HEVC Media Foundation decoder once per process with
`MFTEnumEx`, before both metadata probing and pixel decoding. Install missing
extensions and restart the app to refresh that check. Missing HEVC and WIC
component errors name both extensions and include a hexadecimal HRESULT.
Availability is a capability check; malformed files and later decode errors
still fail explicitly.

Dataset discovery recognizes `.avif` and `.heif` case-insensitively, plus `.heic`
on Windows/macOS. Same-stem companions do not inflate legacy counts or displace
legacy previews. Explicit references select the named file. Recognizing an
extension does not promise that every codec, precision or auxiliary image in
that container is supported.

File, borrowed-memory and caller-allocated targets reuse the existing reader.
Target decode stages pixels before copying and is not zero-copy. Bounded-header
recognition is advisory; full probe/decode validates the primary item and
provider. Existing JPEG/PNG/TIFF/EXR/WebP readers are unchanged.

Raw reads preserve qualified encoded RGB samples after YUV conversion and
container transforms. Higher-precision samples use native-endian UINT16 expanded
to `[0,65535]`; they are not automatically linear or metric. Training precision
defaults are unchanged. ImageIO applies ordered container rotations/mirrors once
without applying EXIF orientation again; a single preceding clean aperture is
supported. Windows rejects unqualified combined rotation/mirroring.

Display reads use valid RGB ICC before supported SDR NCLX, with one bounded LCMS
transform cache per worker. Invalid ICC is not silently ignored. AV1 raw HDR
remains readable while implicit SDR display conversion is rejected. Native HEVC
HDR without usable ICC is unsupported; Apple also cannot decode ICC plus
contradictory PQ/HLG NCLX. No native HDR raw fallback is promised. Qualified
Windows ICC-over-HDR-NCLX inputs retain ICC precedence. HEVC alpha is unsupported;
the existing AV1 display path preserves alpha and unpremultiplies associated RGB.

The bundled libheif overlay retains the pinned sources and patches, uses dav1d,
and disables HEVC decoders, encoders and dynamic plugin loading. OS codecs are
not bundled. This distribution policy is distinct from source licenses and is
not a general statement about patent obligations. No HEIC writer or Linux
external-decoder fallback is provided.

Committed fixture tests require no encoder. They cover precision, transforms,
ICC/NCLX, primary selection, native 4:2:0 with an independent tolerance reference,
unavailable providers, memory/targets and cache/queue consumers. See
[`tests/data/heif/README.md`](../../tests/data/heif/README.md).

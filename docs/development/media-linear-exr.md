# Linear SDR frames and EXR output

`lfs_media` provides `RGBFloat32` and `RGBAFloat32` CPU frame views and immutable
owning surfaces. `RGB8` remains the default for ingest, preview and PNG/JPEG.
`FrameColor` uses the existing ISO color enums; float samples are native-endian
32-bit floats, interleaved RGB/RGBA, with a byte row stride. Views need not be
float-aligned. Visible samples are copied without reading row padding, and owned
padding is zeroed. The last row needs only its visible bytes.

`FrameInfo::origin` distinguishes decoded, rendered and external frames.
`source_timestamp` and `output_timestamp` are independent signed-tick rationals.
Rendered/external writers do not require source frame ordinals or fake PTS.
Optional `source_component_depth` describes decoded source precision rather than
the output representation.

## Qualified extraction profile

Set `IngestRequest::output_format` to `RGBFloat32`. The software decoder feeds
integer RGB or planar/semiplanar YUV of 8–16 component bits, including 4:2:0,
4:2:2 and 4:4:4. Components are unpacked directly at source precision; YUV uses
chroma-location-aware sampling and an unclipped float matrix with
declared BT.709, BT.601, SMPTE 240M or BT.2020 NCL and full/limited range. Inverse transfer
supports Linear, sRGB and BT.709. Output retains BT.709 or BT.2020 primaries;
it does not convert primaries or claim physical luminance units.

Missing transfer/primaries require explicit `input_color` overrides. Overrides
are recorded as assumptions; they do not enable PQ, HLG or Dolby Vision inputs.
HDR/LOG, alpha video, floating-point video and packed YUV layouts return
Unsupported. A change of transfer/primaries within extraction is rejected.
`float_sdr_profile` describes these limits in C++, CLI, Python and MCP capabilities.

Resize uses area filtering for reduction and pixel-center bilinear interpolation
for enlargement in linear light; quarter-turn
rotation preserves samples. Sharpness selection uses a clipped RGB8 proxy only
for ranking; accepted master pixels remain float. Negative and above-one values
are preserved. This profile intentionally selects CPU decoding even when Studio
has registered hardware adapters. Existing RGB8 hardware paths remain available.

Studio calls the Tensor library's `video_to_linear_rgb` operation, which accepts
contiguous UInt8 component storage and returns HWC Float32 on the input device
and backend. It validates component extents, strides and numerical parameters
before dispatch. Its C++/Slang sampler serves CPU, CUDA, Vulkan and Metal; the
leaf media/CLI uses the same C++ sampler without requiring Tensor or a GPU.
HDR tone mapping shares its plane reader and Lanczos chroma sampler in the
Tensor library, preserving the existing HDR filter and FP16 intermediates.
CPU conversion decodes all RGB channels together and parallelizes rows with
the existing TBB dependency. Matrix/range/chroma rules and sRGB/BT.709 curves
reuse the existing color services.

Sharpness windows retain only the running best frame; equal scores preserve the
first maximum. Working frames are checked for layout overflow and addressability
rather than a fixed 256 MiB cap. `MemoryFrameSink` retains its independent
caller-selected payload/frame budgets. Decoder allocations and metadata are
separate from these budgets.

## EXR image output and ownership

`ImageOutput::writeExr(path, view, options)` is a synchronous output service for
decoded, rendered or external linear float RGB/RGBA. It accepts BT.709/BT.2020,
HALF or FLOAT precision, and ZIP or uncompressed scanlines. The default is HALF,
ZIP, and no replacement of an existing target. HALF quantizes samples; magnitudes
above 65504 return InvalidArgument with coordinates and advise selecting FLOAT.
NaN/infinity and alpha outside [0,1] are rejected. RGB may be negative or above one.
Straight RGBA is premultiplied for EXR; already premultiplied RGBA is not multiplied
again. RGB requires alpha mode None. No display transform is applied by the writer.

OpenEXRCore is linked only in `lfs_image_codecs`, using the existing dependency.
The codec encodes to a caller-owned stream. The shared `lfs_file_io` service owns
Unicode path handling, exclusive same-directory temporaries and atomic commit;
existing settings/JSON, EXR frames and manifests use that same implementation.
No-replace uses native Windows, macOS and Linux rename APIs, with a safe hard-link
fallback when the platform lacks them. If both are unsupported, it reserves the
destination with `O_CREAT|O_EXCL`, then renames the completed temporary over its
own placeholder. The reserved name can briefly refer to an empty file; a failed
rename removes the owned placeholder. POSIX files use mode 0666 subject to umask.
Failed/cancelled writes
remove their temporary and preserve an existing target. No-replace commit remains
atomic under concurrent producers; replacement requires `overwrite=true`.
Cancellation is checked before work, between chunks and before commit. It cannot
interrupt an arbitrary filesystem call. Chunk scratch is bounded to 64 MiB;
provenance is bounded to 1 MiB. No OpenImageIO dependency is introduced.

Headers record chromaticities, linear transfer, relative units, alpha convention,
optional provenance, origin, source/output timestamps and source component depth.
`FileFrameSink` still names decoded frames by the existing source ordinal.
EXR extraction requires float output and always creates schema-3 metadata with
input transfer/overrides, output precision/compression and per-frame source facts.
Completed images remain on a later frame or manifest failure; the failure retains
the accepted count. PNG/JPEG pixels, schema-2 metadata and overwrite policy retain
their existing behavior.

## Entry points

```sh
media-ingest extract clip.mkv --output frames --interval 1 --format exr \
  --exr-precision float --exr-compression zip --quiet
# For known but untagged SDR sources, explicitly supply assumptions:
media-ingest extract clip.nut --output frames --format exr \
  --input-transfer srgb --input-primaries bt709
```

Python exposes these options in `lichtfeld.media`. `FrameSurface.from_bytes`
constructs an owning surface from interleaved bytes (at most 256 MiB); callers may
use `struct.pack` or their existing array library. Specify `FrameColor` explicitly
for EXR. `ImageOutput.write_exr(path, surface, options, cancelled=...)` releases the
GIL during native work and propagates original Python cancellation exceptions.
`MediaIngest.extract` and `extract_files` expose the same float request/options.

The `media.extract` MCP job accepts `format=exr`, `exr_precision=half|float`,
`exr_compression=zip|none`, `overwrite` and `input_transfer`/`input_primaries`.
Wrong-format options, JPEG quality and HDR-to-SDR with EXR are rejected before
starting a job. Job lifecycle, cancellation, event routing and accepted counts
are shared with PNG/JPEG. The Studio dialog offers EXR-specific controls and
source-metadata defaults, hides JPEG quality/HDR conversion for EXR, and shows a
preflight reason beside the EXR controls while disabling Start for unsupported
sources. Its preview remains a display image. Confirmed EXR replacement enables
overwrite and removes stale generated frames only after success. Discovery uses
the current pattern and safe filenames from previous extraction metadata, so
changing the pattern still detects the old extraction. PNG/JPEG replacement
retains the legacy extension-wide PNG/JPG/JPEG cleanup; EXR preserves unrelated
images. Old metadata is removed when metadata output is disabled; newly generated
metadata remains owned by the extractor regardless of filesystem timestamp
resolution. Failure/cancellation keeps the old files; symlinks and files changed
during extraction are preserved.

## Verification

Existing root `media_contracts` includes CPU unit, independent reference, public
C++ consumer, real binding-group and MCP job tests. No new CI job is required.
`MediaFloatExrReferenceContracts` creates deterministic FFV1 RGB16/YUV fixtures,
verifies their source codes independently, reads uncompressed EXR bytes without
the production reader, and checks HALF/ZIP through FFmpeg. Analytic transfer,
matrix, range, chroma location, superwhite, resize and rotation references qualify
numeric accuracy, including H.264 4:2:0 and HEVC 10-bit 4:2:0 input.
Writer units cover stride/ownership, alpha, invalid samples, overflow, budgets,
atomic collision/replacement, cancellation and temporary cleanup. The public
consumer links only `lfs_media`, with no private codec/FFmpeg include dependency.
`MediaAtomicFileContracts` covers native no-replace races, binary writes, errors,
cancellation and POSIX permissions. `MediaLinearVideo_*` exercises the shared
Tensor program against independent analytic references on available native GPUs.

Hosted Windows/Linux tests are CPU-only. Native GPU regression qualification must
run separately on physical hardware. See `tests/media/README.md` for root test
invocation and fixture/tool requirements; synthetic tests need no media downloads.

## Color and backend contract matrix

This inventory describes existing behavior. A format being decodable does not
imply that its transfer, gamut or mastering metadata can be preserved by every
output path. Decoder availability, color conversion and image encoding are
separate capabilities. Matrix/range and transfer/primaries are independent.

| Input or operation | Current contract | Reference or boundary test |
| --- | --- | --- |
| Integer SDR RGB/YUV, 8–16 bits to float | Linear, sRGB and BT.709 inverse transfer; BT.709 or BT.2020 primaries retained | `test_transfers_and_linear_resize_rotation_reference`, `test_yuv_depths_and_subsampled_reference` |
| YUV limited/full range | Explicit range required; negative and superwhite RGB retained | `test_yuv_nominal_range_superwhites_and_matrix_analytic` |
| YUV matrix | BT.709, BT.470BG/SMPTE 170M, SMPTE 240M, BT.2020 NCL supported | Independent raw-code analytic reference, five matrices × three curves × two ranges |
| Missing YUV matrix/range or BT.2020 CL | Unsupported for float extraction; no resolution-based guess or substitution of NCL | Public preflight units and actual tagged-file rejection; transfer/primaries overrides cannot supply matrix/range |
| Missing transfer/primaries | Explicit supported overrides required for float output | `test_unknown_hdr_and_yuv_colorimetry` |
| PQ, HLG, Log, LogSqrt to float/EXR | Unsupported; an SDR override cannot relabel a known HDR/LOG source | `test_float_color_profile_rejects_unsupported_tags_without_relabeling_hdr` |
| Other transfer or primaries | Unsupported unless covered by an explicit permitted override; no implicit gamut conversion | BT.470M transfer and BT.470BG primaries rejection fixtures |
| Chroma subsampling and location | 4:2:0/4:2:2/4:4:4; location-aware reconstruction | Raw-code and FFprobe siting references; source codes verified before comparison |
| Float resize/rotation | Area reduction, bilinear enlargement in linear light, sample-preserving quarter turns | Analytic resize/rotation and checkerboard references |
| Linear float RGB/RGBA to EXR | HALF/FLOAT, BT.709/BT.2020 chromaticities, relative units; alpha policy explicit | Independent EXR binary reader, FFmpeg HALF/ZIP reader and writer units |
| RGB8 PNG/JPEG and video export | Existing display/quantization contracts; not a linear HDR master path | Extraction pixel contracts, frozen CPU producer and native video round trips |
| HDR to SDR | Existing host renderer, explicit request; output is dithered RGB8 sRGB/BT.709 | Existing HDR renderer tests; not evidence of HDR-preserving EXR support |
| Dolby Vision | Existing format-specific host handling where metadata is available; excluded from float SDR profile | Existing HDR contracts; no generic vendor/model or enhancement-layer claim |

`validateFloatSdrSource` is the public metadata/layout preflight also used by
Studio. Extraction validates decoded frames as well: a frame-level rejection
after sink initialization can leave an empty output directory, but must publish
no image or manifest. Tests distinguish that from a successful conversion.
FFV1 may supply nominal range when no range was tagged; absent-range validation
therefore uses the public preflight directly rather than a mislabeled fixture.

| Layer | Headless media/CLI | Studio host | Remaining boundary |
| --- | --- | --- | --- |
| Float video decode | Software FFmpeg | Also deliberately software for this profile | Native decoded-GPU float import is not implemented |
| Linear conversion | Shared C++ sampler, CPU/TBB | Existing Tensor adapter, selected CUDA/Vulkan/Metal when available | Adapter uploads host planes and reads float output back to CPU; device-resident ingest is not implied |
| No linear GPU renderer available | Shared CPU sampler | Factory can return no renderer, using that CPU sampler | Failure after choosing a renderer is propagated; no universal automatic GPU-error fallback |
| HDR display conversion | Unsupported without a registered HDR host | Vulkan builds use `libplacebo_vulkan`; other builds use the existing Tensor HDR implementation | Compile-time provider choice, not an automatic runtime fallback chain; renderer initialization can fail |
| Image output | CPU PNG/JPEG/EXR services | Existing optional CUDA JPEG encoder; EXR remains CPU | No GPU EXR encoder; resume pins its selected JPEG provider |

Runtime registration is not proof that a device or renderer can initialize.
Capability discovery must not be used as a numerical color reference. Native
backend tests assert the backend actually exercised; CPU results and hardware
skips do not qualify another backend.

### Gates before extending HDR or LOG

1. Define the output contract first: PQ has an absolute luminance interpretation,
   HLG requires a declared rendering/viewing policy, and scene-referred LOG needs
   the actual curve and gamut. Existing relative-linear SDR EXR must not silently
   change units or acquire a display tone map.
2. Decide how source mastering/content-light metadata, unknown values and any
   assumptions are represented. The existing HDR detector can use mastering
   metadata heuristics; detection alone is not an inverse-transfer policy.
3. Specify preservation versus conversion of primaries, out-of-gamut and negative
   samples, clipping, precision, alpha and resize order. Do not infer a camera
   LOG curve or vendor calibration from file extension or component depth.
4. Add independent numerical references and representative real inputs before
   claiming a new transfer/gamut combination. Current synthetic SDR tests do not
   qualify all HDR mastering combinations, LOG vendors or Dolby Vision profiles.
5. Qualify each selected native backend and its explicit CPU/error policy on
   Windows, Linux and macOS. Reuse existing color math, Tensor programs, writer
   and backend registration; introduce only the missing contract or operation.

These gates do not require adding new codecs first. HEVC/HEIF/AVIF availability,
license/build inventory and CPU/hardware codec fallback remain a separate
increment. GPU import and batching require measured benefit and explicit device,
ownership and synchronization contracts.

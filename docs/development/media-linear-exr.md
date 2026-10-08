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
declared BT.709, BT.601 or BT.2020 NCL and full/limited range. Inverse transfer
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

Studio converts through the Tensor backend with one shared C++/Slang sampling
program for CUDA, Vulkan and Metal. The leaf media/CLI keeps a CPU implementation
of the same program without requiring a GPU. Matrix/range/chroma rules and
sRGB/BT.709 curves reuse the existing color services.

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
fallback when the platform lacks them. POSIX files use mode 0666 subject to umask.
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
overwrite and removes stale generated frames matching the filename pattern only
after success. Failure/cancellation keeps the old frames; unrelated files,
symlinks and files changed during extraction are preserved.

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

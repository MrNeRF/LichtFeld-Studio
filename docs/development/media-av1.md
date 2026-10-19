# AV1 video export

AV1 export uses the existing `VideoEncodeSession` and Tensor video producer.
H.264 remains the default; the existing H.264 and HEVC provider policies are
unchanged. Select AV1 explicitly through the C++ codec enum or Python:

```python
lf.ui.export_video(1920, 1080, 30, 18, "render.mp4", codec="av1")
```

This uses the existing viewer export job and saved reconstruction selection.
The Sequencer codec chooser is a separate UI change. The image extraction CLI
does not become a video transcoder through this addition.

## Providers and distribution

The FFmpeg 9 build enables its `aom` feature, reusing the libaom dependency
already used for AVIF. No new vcpkg overlay, codec download at runtime or global
installation is required. Existing FFmpeg features and version overrides remain
in place. The generated distribution notices include libaom and FFmpeg.

| Execution path | AV1 encoder | Behavior |
| --- | --- | --- |
| CUDA with a working AV1 NVENC encoder | `av1_nvenc` | Native CUDA frames, same shared producer |
| CUDA without usable AV1 NVENC | `libaom-av1` | Explicit same-codec CPU fallback |
| CPU, Metal or Vulkan producer | `libaom-av1` | Existing CPU-plane producer path |
| VideoToolbox preference | `libaom-av1` | FFmpeg 9 has no AV1 VideoToolbox encoder |

Registering `av1_nvenc` in FFmpeg does not establish hardware support. Opening
the actual encoder decides availability for the device, driver, dimensions and
requested quality. If both providers fail, export reports AV1 unavailable;
it never silently writes H.264. This increment does not enable AMF, VAAPI or QSV.
It does not change AV1 reading, which continues to select dav1d explicitly.

The distribution driver requirements documented for the existing FFmpeg/NVENC
build still apply. Codec support additionally depends on the GPU model.
See [FFmpeg codec documentation](https://ffmpeg.org/ffmpeg-codecs.html) for the
libaom options and [the FFmpeg 9 VideoToolbox AV1 decoder source](https://www.ffmpeg.org/doxygen/9.0/videotoolbox__av1_8c_source.html).

## Output and quality contract

Output remains MP4, with the `av01` tag and 8-bit 4:2:0 samples. Existing matrix,
range, spherical-video and provenance handling is reused. No new HDR/10-bit
output, audio, WebM or Matroska contract is introduced.

The public quality range remains 0..51, with lower values requesting higher
quality. libaom receives that value as CRF, with constant-quality mode
(`bit_rate = 0`), `cpu-used = 8`, row multithreading and eight lookahead frames.
Its worker count is clamped to 1..8 per session. NVENC AV1 maps the public range
to its native QP range 0..255; H.264/HEVC retain their previous mapping. Matching
numbers across encoders do not imply matching size, speed or distortion.

Quality zero can be rejected by native hardware. That failure uses the same AV1
CPU fallback; lossless YUV planes do not imply preservation of an original RGB
image through RGB-to-YUV conversion and chroma subsampling.

## Lifetime, failures and validation

Delayed frames are drained on close before the MP4 trailer is written. Rejected
or throwing frame writers do not consume presentation timestamps. Existing
ownership, reentrancy and reuse contracts also apply to AV1. Provider absence or
rejected encoder options are checked before opening the output file. Once
writing has started, the existing video session can leave an incomplete file on
failure or interruption; this is not the atomic/resumable image writer.
Cancellation cannot preempt an individual libaom encoding call. Per-session
worker/lookahead settings do not establish an aggregate RAM or concurrency cap.

CPU contracts check delayed-frame draining, all decoded timestamps and luma
values, quality zero, full/limited range, Unicode comments, spherical metadata,
writer rejection/exceptions, provider/option absence, trailer failure and session
recovery. The provider seam uses the production session and real libaom output.
CUDA contracts exercise the actual Tensor producer, distinct queued frames,
quality boundaries and a two-frame 8K smoke test. AV1-native hardware cases are
opted into with `LFS_MEDIA_TEST_NVENC_AV1=1`; they fail rather than silently
accepting CPU output. A separate quality-zero test accepts either actual AV1
backend and verifies the produced file. The 8K test establishes correctness on
the tested hardware, not photographic quality or throughput.

Existing H.264/HEVC and image-output regression suites remain applicable.
Sanitizer runs instrument project session code; release provider dependencies
are not fully instrumented. Hosted CI and untested GPU families require their
own qualification.

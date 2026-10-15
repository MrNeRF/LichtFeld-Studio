# HEVC video export

Video export defaults to H.264. The Python API accepts an explicit HEVC choice:

```python
lichtfeld.ui.export_video(1920, 1080, 30, 23, path="output.mp4", codec="hevc")
```

The existing encode session uses `hevc_nvenc` when the CUDA tensor backend is active and `hevc_videotoolbox` on macOS. No additional software encoder is installed or bundled. VideoToolbox can retry through the operating system’s software implementation when available. If the selected provider cannot encode HEVC, export fails explicitly; it does not substitute H.264. The generic software backend does not provide HEVC.

The current contract is MP4 with the `hvc1` sample entry and 8-bit SDR frames, using the existing YUV420P/NV12 producers. Selecting HEVC does not enable HDR or 10-bit output. Resolution support depends on the actual provider and device; encoder initialization remains authoritative.

This affects video export. HEVC input decoding and the P010 extraction pipeline have separate capabilities.

## NVIDIA driver requirements

The manifest pins FFmpeg 9.0.2 and NVIDIA codec headers (`ffnvcodec`)
13.0.19.0. For this build, NVENC H.264 and HEVC export require NVIDIA driver
**570.0 or newer on Windows and Linux**, plus a GPU that supports the requested
codec and resolution. Updating FFmpeg does not update the installed driver or
add hardware capabilities. macOS uses VideoToolbox and has no NVIDIA requirement.

The driver minimum follows the NVENC API used to compile FFmpeg, not just the
FFmpeg release number. Custom builds with newer codec headers can require newer
drivers; see the [pinned header requirements](https://github.com/FFmpeg/nv-codec-headers/blob/n13.0.19.0/README)
and [FFmpeg's driver checks](https://github.com/FFmpeg/FFmpeg/blob/n9.0.2/libavcodec/nvenc.c).
The CUDA runtime/toolkit and GPU generation can impose additional requirements.
In WSL, update the NVIDIA driver on the Windows host.

NVENC initialization remains the capability check. H.264 retains its existing
software fallback; HEVC reports an error when its selected provider is unavailable.
The pinned headers preserve the existing NVENC API 13.0 driver requirement;
this FFmpeg update does not itself raise it.

NVENC retains preset `p4`, tuning `hq` and constant-QP rate control. Requested
options are checked for errors. The existing quality-to-QP mapping adds three
and now caps the result at 51, so public quality values 49–51 cannot silently
leave NVENC's QP at its default or cause an out-of-range option failure.

The Sequencer GUI keeps its existing H.264 default; exposing codec choices there is deferred to the media GUI redesign.

On macOS, the current quality setting maps to the same target bitrate for H.264
and HEVC. Selecting HEVC therefore does not request a smaller file at that
setting. Codec-specific quality controls belong to the planned media GUI work.

Native Mac validation can enable `LFS_MEDIA_TEST_VIDEOTOOLBOX_8K=1` when running `MediaExtractionContracts`. It exercises hardware HEVC at small, 4K and 8K extents, spherical metadata and a test-only hardware-open failure followed by the real Apple software encoder and hardware recovery. No fault injection is compiled into the production media library.

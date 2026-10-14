# HEVC video export

Video export defaults to H.264. The Python API accepts an explicit HEVC choice:

```python
lichtfeld.ui.export_video(1920, 1080, 30, 23, path="output.mp4", codec="hevc")
```

The existing encode session uses `hevc_nvenc` when the CUDA tensor backend is active and `hevc_videotoolbox` on macOS. No additional software encoder is installed or bundled. VideoToolbox can retry through the operating system’s software implementation when available. If the selected provider cannot encode HEVC, export fails explicitly; it does not substitute H.264. The generic software backend does not provide HEVC.

The current contract is MP4 with the `hvc1` sample entry and 8-bit SDR frames, using the existing YUV420P/NV12 producers. Selecting HEVC does not enable HDR or 10-bit output. Resolution support depends on the actual provider and device; encoder initialization remains authoritative.

This affects video export. HEVC input decoding and the P010 extraction pipeline have separate capabilities.

The Sequencer GUI keeps its existing H.264 default; exposing codec choices there is deferred to the media GUI redesign.

On macOS, the current quality setting maps to the same target bitrate for H.264
and HEVC. Selecting HEVC therefore does not request a smaller file at that
setting. Codec-specific quality controls belong to the planned media GUI work.

Native Mac validation can enable `LFS_MEDIA_TEST_VIDEOTOOLBOX_8K=1` when running `MediaExtractionContracts`. It exercises hardware HEVC at small, 4K and 8K extents, spherical metadata and a test-only hardware-open failure followed by the real Apple software encoder and hardware recovery. No fault injection is compiled into the production media library.

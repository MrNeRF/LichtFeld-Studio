# Generated AV1 CPU fixtures

These project-owned synthetic clips contain four 64x64 frames at 4 FPS, with
changing luma and constant chroma. There is no third-party source imagery.
They are test data under GPL-3.0-or-later, like the surrounding test code.

Generated with FFmpeg 9.0.2/libaom, for each PIXFMT in yuv420p and yuv420p10le:

```sh
ffmpeg -v error -f lavfi \
  -i "nullsrc=size=64x64:rate=4:duration=1,geq=lum='32+N*32':cb=90:cr=180" \
  -pix_fmt PIXFMT -c:v libaom-av1 -cpu-used 8 -crf 30 av1-cpu-PIXFMT.mp4
```

The CPU extraction/preview contract reads these checked-in fixtures regardless
of the fixture-generation FFmpeg executable's available encoders. Missing files
are a test failure, not a skip. RGB references still come from an independent
FFmpeg decoder, with constant chroma avoiding scaler-version edge differences.

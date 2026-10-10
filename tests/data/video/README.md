# VP9 timestamp fixture

`vp9_timestamps.webm` contains 185 synthetic 128 x 128 frames at 30 fps, with a 1/1000 stream time base. It exercises NVDEC timestamp handling without requiring a VP9 encoder at test runtime. Generated with FFmpeg 8.0.1 and libvpx:

```sh
ffmpeg -f lavfi -i testsrc2=size=128x128:rate=30 -frames:v 185 \
  -c:v libvpx-vp9 -b:v 80k -deadline realtime -cpu-used 8 vp9_timestamps.webm
```

The regression compares every extracted image against FFmpeg software decoding of this fixture. The minimum height for VP9 NVDEC on the test device is 128 pixels; smaller images fall back to software and do not exercise this bug.

`vp9_small.webm` uses the same command with `size=64x64` and `-frames:v 10`. It guards software fallback for frames below NVDEC's supported minimum size. Both fixtures contain only generated test patterns.

# VP9 timestamp fixture

`vp9_timestamps.webm` contains 185 synthetic 128 x 128 frames at 30 fps, with a 1/1000 stream time base. It exercises NVDEC timestamp handling without requiring a VP9 encoder at test runtime. Generated with FFmpeg 8.0.1 and libvpx:

```sh
ffmpeg -f lavfi -i testsrc2=size=128x128:rate=30 -frames:v 185 \
  -c:v libvpx-vp9 -b:v 80k -deadline realtime -cpu-used 8 vp9_timestamps.webm
```

The regression compares every extracted image against FFmpeg software decoding of this fixture. The minimum height for VP9 NVDEC on the test device is 128 pixels; smaller images fall back to software and do not exercise this bug.

`vp9_small.webm` uses the same command with `size=64x64` and `-frames:v 10`. It guards software fallback for frames below NVDEC's supported minimum size. Both fixtures contain only generated test patterns.

# Video range fixtures

Three identical 256x256 YUV420P frames at 10 fps. Each row is a grayscale
ramp: Y=x for full range, Y=round(16+x*219/255) for limited range;
U=V=128. Dimensions satisfy the NVDEC minimums. No source images are used.

Encoded losslessly with FFmpeg 8 using rawvideo input and explicit matching
input/output `-color_range pc` or `tv`, with `-colorspace bt470bg`:

- VP9: `-c:v libvpx-vp9 -lossless 1`
- AV1: `-c:v libaom-av1 -crf 0 -cpu-used 8`
- H.264: `-c:v libx264 -qp 0`
- HEVC: `-c:v libx265 -x265-params lossless=1`

Full-range decoded RGB must reproduce x, and limited-range RGB must be within
two levels of x. The tests exercise native preview and both PNG/JPEG extraction.
Small synthetic decoder fixtures (five frames each), generated with FFmpeg:

```sh
ffmpeg -f lavfi -i testsrc=size=64x48:rate=10 -frames:v 5 -c:v libx264 -pix_fmt yuv444p h264_444.mp4
ffmpeg -f lavfi -i testsrc=size=65x49:rate=10 -frames:v 5 -c:v libx264 -pix_fmt yuv444p h264_444_odd.mp4
ffmpeg -f lavfi -i testsrc=size=64x48:rate=10 -frames:v 5 -c:v libx264 -pix_fmt yuv420p h264_420.mp4
```

The 4:4:4 fixtures exercise software fallback when the hardware decoder rejects the chroma format. The 4:2:0 fixture guards the existing hardware decode path. They also run with software decoding on machines without CUDA.

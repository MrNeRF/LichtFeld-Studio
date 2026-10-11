Small synthetic decoder fixtures (five frames each), generated with FFmpeg:

```sh
ffmpeg -f lavfi -i testsrc=size=64x48:rate=10 -frames:v 5 -c:v libx264 -pix_fmt yuv444p h264_444.mp4
ffmpeg -f lavfi -i testsrc=size=65x49:rate=10 -frames:v 5 -c:v libx264 -pix_fmt yuv444p h264_444_odd.mp4
ffmpeg -f lavfi -i testsrc=size=64x48:rate=10 -frames:v 5 -c:v libx264 -pix_fmt yuv420p h264_420.mp4
```

The 4:4:4 fixtures exercise software fallback when the hardware decoder rejects the chroma format. The 4:2:0 fixture guards the existing hardware decode path. They also run with software decoding on machines without CUDA.

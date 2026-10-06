# Offline media fixtures

This baseline block creates a reproducible corpus without using LichtFeld's
encoder or a GPU. The preparation suite verifies the corpus itself; the separate
`extractor` target compiles and exercises the real VideoFrameExtractor and image
writers. HDR, automatic rotation metadata, telemetry, actual lens calibration and
stitching are not covered yet. This is the first step towards Media Ingest.

## Run

Python 3.10+, CMake 3.24+, FFmpeg and ffprobe with FFV1/rawvideo/NUT support are
required. Missing tools fail preparation rather than silently skipping it.

```sh
cmake -S tests/media -B build-media-fixtures
ctest --test-dir build-media-fixtures --output-on-failure
python scripts/prepare_media_fixtures.py --output /path/to/new-corpus
```

Use `--ffmpeg`/`--ffprobe` for executable paths outside PATH. Tests accept the
`LFS_MEDIA_TEST_FFMPEG`/`LFS_MEDIA_TEST_FFPROBE` environment variables. No downloads,
Python packages, GUI state changes or CUDA initialization are needed. Generated
data stays in temporary directories during tests; manual output requires a new
destination and never overwrites an existing directory.

## Contracts

| ID | Contents | Reference |
|---|---|---|
| cfr-asymmetric | Five lossless 64×32 RGB frames with asymmetric corners and per-frame center IDs | PTS 0, .1, .2, .3, .4 seconds; exact decoded RGB |
| vfr-asymmetric | Same source pixels with irregular times | PTS 0, .1, .4, .9, 1.6 seconds; exact decoded RGB |
| panorama-layout | Five 128×64 frames with declared equirectangular layout | Exact decoded RGB and 2:1 dimensions; not calibrated stitching/SfM data |

Reference pixels originate directly from the deterministic Python pattern, not
from LichtFeld output. ffprobe verifies actual timestamps and geometry; FFmpeg
independently decodes the encoded fixture for byte equality. Tests also reject
incorrect expected times/geometry and preserve existing data on preparation errors.

The manifest records tool versions, file sizes/checksums, actual timebase and
decoded-pixel checksums. Encoded container hashes need not match across FFmpeg
versions; pixel and timestamp contracts must match. The corpus is published only
after all files pass verification. Synthetic data is generated from the GPL-3.0-or-later
script; no third-party media are bundled.

## Public media later

A public-media catalog is intentionally not populated with unverified URLs. Before
adding a real sample, record its direct download URL, source page, permission or
license, size, SHA-256, format properties, test IDs and required backend. Reuse the
project's dataset-download convention once identified; keep fetching explicit and
cache verified content. Public availability alone does not permit redistribution.
Real vendor/HDR samples supplement the offline corpus instead of making basic
tests depend on network access.

## Production extraction contracts

```sh
cmake -S tests/media/extractor -B build-media-extractor -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build build-media-extractor --config Release
ctest --test-dir build-media-extractor -C Release --output-on-failure
```

Alternatively reuse an existing dynamic package prefix with `-DCMAKE_PREFIX_PATH`.
The leaf manifest pins the repository's vcpkg baseline and reuses only the OpenEXR
overlay; it does not change the app manifest or add OpenImageIO. FFmpeg development
libraries, JSON and the production image-codec dependencies are required. Windows
CTest adds the package DLL directory to PATH; override `MEDIA_RUNTIME_LIBRARY_DIR`
when runtime DLLs are in a different location.

The test adapter takes a JSON request and calls `VideoFrameExtractor::extract`.
Production selection, geometry, FFmpeg decode, PNG/JPEG codecs and schema-2
metadata are compiled directly. Only the user log sink and HDR renderer boundary
are test support: logs go to stderr, and reaching HDR throws instead of returning
fake pixels. No GUI/Python embedding, CUDA or Slang compiler is linked.

The sixteen extraction tests cover exact PNG pixels, interval/FPS selection,
trim/source naming, VFR timestamps, explicit rotations, scale/custom dimensions,
independent bilinear resize, JPEG flat-patch error (six 8-bit levels), metadata on/off,
invalid requests, bad input, cancellation, output failure and sharpness rejection.
BT.601/BT.709 limited-range YUV outputs are compared to independent FFmpeg decode.
Resize/YUV comparisons allow up to three 8-bit levels for library-version rounding;
lossless, unscaled PNG requires exact pixels.

The current interval/FPS end-boundary difference is characterized explicitly:
interval includes a frame at an exactly matching end timestamp, FPS excludes it.
VFR `source_frame` is nominal-FPS-derived rather than a decoded ordinal. Cancelled
jobs currently preserve completed images without a cancellation manifest. These
contracts describe the baseline and do not advertise future Media Ingest guarantees.

## Shared probe and preview checks

The same extractor CTest directory also compiles the production MediaProbe and
VideoPlayer, runs six additional Python probe/preview tests, and registers native
MediaProbeUnitContracts assertions. These checks use MPEG-4/MOV, PCM/WAV and lavfi
sine in addition to the original FFV1/rawvideo/NUT fixture capabilities. They run
inside the existing Release CI jobs without another workflow or job. See
../../docs/development/media-probe-contracts.md for API, behavior and verification.
## CPU frame sinks

The same CTest project also builds production FrameSurface/MemoryFrameSink and
FileFrameSink, with eight additional extraction Python methods and the native
MediaFrameSinkUnitContracts suite. They verify retained ownership, rational source
PTS and decode identity, transformations/window/tail selection, no implicit disk
output, memory limits, failure/cancellation lifecycle, PNG/JPEG equivalence,
padded rows and real writer failure. Core structured error implementation is
compiled directly; test diagnostics remain on stderr.

See [frame sink contracts](../../docs/development/media-frame-sinks.md) for callback
lifetimes, budget accounting, delivery indices and legacy compatibility limits.

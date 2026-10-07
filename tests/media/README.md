# Offline media fixtures

This baseline block creates a reproducible corpus without using LichtFeld's
encoder or a GPU. The preparation suite verifies the corpus itself; the separate
`extractor` target compiles and exercises the real VideoFrameExtractor and image
writers. The offline baseline does not qualify HDR, telemetry, lens calibration
or stitching. Native Studio backend qualification is a separate suite described
below; probe/preview orientation contracts are covered in the CPU harness.

## Run

Python 3.10+, the application root CMake/toolchain, FFmpeg and ffprobe with FFV1/rawvideo/NUT support are
required. Missing tools fail preparation rather than silently skipping it.

```sh
python tests/media/test_fixture_preparation.py
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
# Configure the app normally with BUILD_TESTS, BUILD_FORMAT_TESTS or
# BUILD_VISUALIZER_TESTS enabled, then use that root build directory.
cmake --build build --target media_contracts --config Release --parallel 2
ctest --test-dir build -C Release -L media --output-on-failure --no-tests=error
```

All production modules and dependency discovery belong to the root configuration.
Use the application's manifest, preset and package prefix. FFmpeg is linked only
by `lfs_media`; its public ABI temporarily serves existing Studio adapters.
Windows CTest adds the configured package DLL directory to PATH.

The test adapter takes a JSON request and calls production extraction/probe/preview
code. The root test runners use the real shared diagnostics, logger, errors, image
codecs and media module, plus the actual Python binding group. No logging/assert/HDR
substitutes or private core copies are used. CPU runners do not link tensor/GPU
targets; native Studio qualification uses the full production graph.

Extraction contracts cover independent pixels/PTS, trim/naming, VFR, rotation,
resize, JPEG/YUV tolerances, metadata, invalid input, cancellation, sink lifecycle,
limits and writer failures. Injected optional JPEG adapters additionally exercise
incomplete batches, failure after accepted output, CPU fallback, hardware disable
and custom sink isolation. Injection tests do not qualify a GPU.

Shared-core contracts check one error bus/dedup table across DLL consumers, writer
registration, production assertion behavior and delegated diagnostics/event state.
CLI contracts check output parity, real version/help, errors/progress and partial
results; the POSIX SIGTERM contract is explicitly skipped on Windows. Python
contracts exercise ownership, GIL and callbacks through the real binding source.
The Linux x86 CLI contracts also verify that FFmpeg's internal assembly constants
are absent from the shared module's dynamic lookup scope; public media operations
remain covered by the extraction contracts. The provider ownership check rejects public/internal FFmpeg exports, definitions
in the executable, visualizer and complete Python module, and direct Windows
FFmpeg imports. Fixture construction has explicit private test-only links; those
links do not propagate to product consumers. Session contracts also reject callback
reentrancy and ownership transfer during a write.
Static FFmpeg must link successfully before these runtime checks can run.

The current interval/FPS end-boundary difference is characterized explicitly:
interval includes a frame at an exactly matching end timestamp, FPS excludes it.
VFR `source_frame` is nominal-FPS-derived rather than a decoded ordinal. Cancelled
jobs currently preserve completed images without a cancellation manifest. These
contracts describe the baseline and do not advertise future Media Ingest guarantees.

## Shared probe and preview checks

The same root media CTest group uses production MediaProbe and a CPU
VideoPlayer reference, runs six additional Python probe/preview tests, and registers native
MediaProbeUnitContracts assertions. These checks use MPEG-4/MOV, PCM/WAV and lavfi
sine in addition to the original FFV1/rawvideo/NUT fixture capabilities. They run
inside the existing Release CI jobs without another workflow or job. See
../../docs/development/media-probe-contracts.md for API, behavior and verification.
## CPU frame sinks

The same root CTest group uses production FrameSurface/MemoryFrameSink and
FileFrameSink, with eight additional extraction Python methods and the native
MediaFrameSinkUnitContracts suite. They verify retained ownership, rational source
PTS and decode identity, transformations/window/tail selection, no implicit disk
output, memory limits, failure/cancellation lifecycle, PNG/JPEG equivalence,
padded rows and real writer failure. Core structured error implementation is
provided by the same shared library used by Studio.

See [frame sink contracts](../../docs/development/media-frame-sinks.md) for callback
lifetimes, budget accounting, delivery indices and legacy compatibility limits.

Native sink checks also cover structured layout/copy errors and JPEG zero/default
and clamped qualities, comparing exact writer bytes; PNG ignores JPEG quality.
The extractor lifecycle suite includes non-standard callback exceptions.

## Shared module, CLI and Python

The root media targets use production `lfs_media`, its shared core leaf
libraries and CLI. They also build the actual `py_media.cpp` binding group. The nested standalone SDK/package consumer has been removed; future
packaging follows the root module after contract/platform qualification.
The existing Release CI steps run the CPU suites without an additional job.
See [the module guide](../../docs/development/media-ingest-core-cli.md).

The same binding tests can also exercise the complete Studio module after building
`lfs_py`: pass `--studio --module-dir <build>/src/python --runtime-dir <build>`
to `test_media_bindings.py`, with `--dependency-dir` for required runtime DLLs on
Windows. This mode also checks that `lichtfeld.__all__` exports `media` and uses
Studio's typed Python errors. Generate and check the committed stubs using the
existing `lichtfeld_stub` and `check_python_stubs` targets.

## Native Studio backend qualification

The root `MediaStudioBackendContracts` test uses real NVDEC/nvJPEG, HDR host
adapters and CUDA profiler sampling/events. It belongs to existing test
configurations, adds no CI job, and reports CTest skip code 77 when no CUDA device
is present. See [module contracts](../../docs/development/media-ingest-core-cli.md)
for the runtime graph, Python API, shared ownership and remaining boundaries.

`MediaMcpContracts` builds the production MCP media adapter in the root test configuration. It validates probe/extraction, asynchronous state, event routing, duplicate-job rejection, cancellation and nested request errors using synthetic public-tool fixtures. It does not require a GPU device.
The macOS native runner supplies CPython's complete embedding archive because
Studio's shared libraries resolve Python from their host process. Running this
CTest contract also verifies that those dependencies load before `main`.
Its fixture is lossless RGB so exact output/rotation comparisons do not depend
on YUV conversion rounding between the fixture tool and Studio's linked FFmpeg.

RGB-to-YUV CPU regressions run inside the existing MCP contract runner and compare
with the frozen dev tensor producer (rounding boundaries, nonfinite values, strides
and storage reuse). They do not register or initialize GPU backends. Native runner
contracts separately check CUDA/Vulkan bytes, timeline ordering and plane sentinels;
the existing macOS HDR target covers Metal/Vulkan conversion. Program caches retire
through the core GPU shutdown hook.

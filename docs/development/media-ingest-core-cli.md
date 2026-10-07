# Shared Media Ingest module and optional CLI

## Production ownership

`src/media/CMakeLists.txt` defines the shared C++23 `lfs_media` target in the
application's existing build and dependency graph. It owns the probe, RGB8/float frame
sinks, ingest facade, extractor and HDR facade. `lfs_video` owns Studio's native
GPU adapters, preview and scene-video encoding. The dialog calls
`MediaIngest::extractFiles`; embedded Python exposes `lichtfeld.media` next to
`lichtfeld.io`. The legacy extractor remains a compatibility adapter to the same
implementation.

There is one shared implementation of each stateful core responsibility:

| Target | Ownership | Dependencies |
|---|---|---|
| `lfs_diagnostics` | Profiler state and CPU accounting | Standard library; optional native operations registered by Studio |
| `lfs_logger` | Existing logger and environment settings | spdlog, `lfs_diagnostics` |
| `lfs_error` | Errors, memory domains, failure reports, reporters, bus, envelopes, latches and guarded tasks | Logger, nlohmann-json; GNU stacktrace support where needed |
| `lfs_image_codecs` | Image decoding and encoding, including stream-based EXR | Shared errors; existing PNG/JPEG/ZLIB/TIFF/WebP/OpenEXRCore packages |
| `lfs_core` | Tensor/application core and native diagnostics adapter | Public shared error and codec targets; existing GPU dependencies |
| `lfs_media` | Probe, frame ownership/sinks, extraction and facade | Error, codecs, logger, FFmpeg |
| `media-ingest` | CLI arguments and JSON protocol | `lfs_media` |

Shared libraries use distinct export macros and the application's runtime/install
locations. Hidden-symbol builds must resolve the same symbols as Windows DLLs.
Portable installation includes the new libraries and optional CLI; Unix leaf
libraries resolve siblings with `$ORIGIN` or `@loader_path`, including the app's
`../lib` layout. Windows binaries receive the same product version resources.

`failure_report` owns one dedup table and invokes an optional writer hook.
`crash_handler` registers Studio's existing diagnostic writer. A headless consumer
uses the real logging/error/contract implementation without a crash-handler or
tensor dependency. `MemoryDomain` and `AllocationFailure` live in
`core/memory_domain.hpp`; allocation coordination remains in `memory_pressure.hpp`.

CUDA sampling, NVML and native event operations now live in `lfs_core`, behind an
immutable process-lifetime diagnostics backend. The CPU diagnostics DLL keeps
one profiler state and does not import CUDA. Registration rejects a different
replacement backend so live event handles remain valid. The CLI consequently
does not acquire a tensor/CUDA/Vulkan dependency through the logger.

The old nested `media/` project, duplicate manifests, SDK export/install logic,
private core copies and diagnostic shims have been removed. A separate distributable
SDK/package is deferred until the shared module's contracts and dependency
qualification are stable; there is no `LichtFeldMedia::media` package export in
this revision. Media contracts are registered by `tests/CMakeLists.txt` in the
root build and consume its production targets; no second production configuration
or standalone test project discovers dependencies.

## Capabilities and host backends

Studio registers its built HDR and CUDA-JPEG adapters before application media
work and embedded Python initialization. GUI workers also ensure registration.
The `lfs_media` leaf contains no tensor/CUDA/Vulkan implementation. Host adapters
preserve the existing NVDEC, nvImageCodec/nvJPEG, libplacebo and tensor/Metal paths.
A registered backend indicates build availability; device, codec and allocation
availability are checked when it is used. It is not a guarantee that every input
can use hardware acceleration.

`IngestRequest::allow_hardware_decode` controls optional hardware decode and JPEG
acceleration. File extraction can use the existing accelerated pipeline. Custom
frame sinks keep software RGB8 delivery and do not receive native hardware
surfaces. An unavailable optional JPEG adapter falls back to CPU; invalid batches,
writer failures and cancellation propagate failure with actual accepted-frame
counts. No failed or empty encoded batch is reported as successful extraction.

`MediaIngest::capabilities()` reports registered host HDR/CUDA support and compiled
VideoToolbox availability on Apple. The headless CLI has no Studio HDR adapter;
`--hdr-to-sdr` returns a structured Unsupported error before output is created.
Studio GUI and Python retain HDR conversion through their registered adapter.
`codecBuildInfo()` reports the actual linked FFmpeg version, license and build
configuration. Runtime fallback and supported input formats remain properties of
that FFmpeg/device build.

## Build and CLI

Use the root toolchain, manifest and existing configuration. Add
`-DLFS_BUILD_MEDIA_CLI=ON` to build the optional executable, then build the
`media-ingest` target. The option defaults to OFF. Building that target does not
compile the application/tensor targets, but root configuration still discovers
Studio dependencies; this revision does not claim a CPU-only root configuration.

```sh
cmake --build build --target media-ingest --config Release
media-ingest version
media-ingest capabilities
media-ingest probe "clip.nut" --timeout-ms 10000
media-ingest probe "clip.nut" --headers-only
media-ingest extract "clip.nut" --output "frames" --interval 2 --metadata
media-ingest extract "clip.nut" --output "frames" --fps 5 --start 1 --end 10 \
  --scale 0.5 --rotate 90 --format jpeg --quality 95 --name 'frame_%05d' --quiet
```

The CLI uses the application's generated Git version. Help includes
`--hdr-to-sdr`, selection, geometry, filename, metadata and sharpness options.
Unknown/conflicting options and invalid numeric values produce structured errors.
Output defaults to PNG; PNG/JPEG metadata is opt-in. EXR requires the bounded
linear SDR profile and always records metadata. See [linear SDR/EXR contracts](media-linear-exr.md)
for precision, input limits, independent image output and UI/API options.
Rotation remains explicit.

Except help, stdout contains one final JSON object (`schema_version: 1`, `success`).
Probe includes container/stream inventory, optional color/orientation metadata,
rationals `[numerator, denominator]` and signed-tick timestamps. Extraction reports
`frames_accepted` and `discarded`. Failures include code/domain/message, context
operations/fields and native status where present. Partial counts are retained in
error context. Progress JSON lines and diagnostic logs use stderr; `--quiet`
suppresses progress.

SIGINT and SIGTERM request cooperative cancellation. Checks occur at decode,
processing and writing boundaries; a signal does not forcibly interrupt every
blocked operation. Completed images remain available. POSIX signal tests exercise
SIGTERM after extraction starts; Windows needs separate console delivery
qualification because `terminate()` bypasses CRT signal handlers.

| Exit code | Meaning |
|---|---|
| 0 | Success/help |
| 2 | InvalidArgument |
| 3 | Unsupported, Unavailable, NotFound, PermissionDenied, DataLoss |
| 4 | Other structured failure |
| 130 | Cancelled |

## Python

```python
from pathlib import Path
import lichtfeld as lf
request = lf.media.IngestRequest()
request.input = Path("clip.nut")
request.selection.mode = lf.media.SelectionMode.Interval
request.selection.interval = 2
report, frames = lf.media.MediaIngest.extract(request, payload_budget=64 * 1024 * 1024)
files = lf.media.FileExtraction()
files.files.output_directory = Path("frames")
lf.media.MediaIngest.extract_files(request, files)
```

Probe, extraction and native processing release the GIL. Progress/cancellation
callbacks reacquire it and preserve Python exceptions. Native frame surfaces retain
ownership; `.pixels` creates a Python bytes copy only when requested. Memory
extraction defaults to a 256 MiB payload budget and 100,000-frame limit. Native
structured errors use the existing Studio exception translation. The synchronous
memory convenience call raises on failed extraction; C++ custom sinks can retain
partial frames and inspect the structured result.

## Verification

Configure the application with its existing preset/toolchain and enable
`BUILD_TESTS`, `BUILD_FORMAT_TESTS` or `BUILD_VISUALIZER_TESTS`. The root
`media_contracts` target builds the production consumers and media test runners.
Existing Release jobs build and run these targets; no CI job/workflow is added.
The CPU reference preview runner disables hardware decode for its own player
translation unit. Studio's `lfs_video` and native GPU tests keep their configured
backends. Shared leaf modules are the same targets used by the application.

```sh
cmake --build build --target media_contracts --config Release --parallel 2
ctest --test-dir build -C Release -L media --output-on-failure --no-tests=error
```

Contracts compare independent RGB/PTS references, legacy output bytes/names/
metadata, probe/preview behavior, sink ownership/budgets/lifecycle, actual writer
failures, CLI version/help, DLL-shared error state and diagnostic registration.
Injected JPEG adapters verify incomplete batches, mid-batch failure, retained
counts, disabled hardware, CPU fallback and memory sink isolation. These injected
contracts do not validate GPU processing.

The actual Python binding group is tested for nested options, native ownership,
file output, GIL/callback behavior, cancellation and exception preservation.
`MediaStudioBackendContracts`, registered in existing root test configurations,
uses a real CUDA device for NVDEC/JPEG, rotation/sharpness, host HDR and native GPU
diagnostics; a missing device gives the explicit CTest skip code 77. Hardware
qualification is distinct from the offline CPU results. Tests use generated media
and do not download a public corpus.

## Dependencies

The root package baseline/overlays and existing FFmpeg/image codec stack are reused.
OpenImageIO is not required or reintroduced. FFmpeg is discovered once in the root
configuration; product targets link FFmpeg only inside `lfs_media`, with PRIVATE
includes and links. Package `optimized`/`debug` qualifiers and transitive
dependencies are preserved. Playback, codec/muxer ownership and decoded-frame
metadata interpretation belong to media. HDR/libplacebo/tensor adapters consume
borrowed media plane/component/colour descriptors, including mastering display,
HDR10+, Dolby Vision, ICC and film-grain data; they do not interpret FFmpeg objects.

The decoded descriptor's hardware handle identifies borrowed storage only.
Hardware views expose no CPU plane pointers; VideoToolbox's pixel-buffer identity
comes from its dedicated slot. Current HDR adapters require a downloaded software
frame. Direct decoded GPU import needs a separate device, ownership and
synchronization contract; the presence of a handle does not enable that path.

The provider exports the media C++ API only. ELF/Mach-O visibility lists hide
third-party static symbols and prevent assembly constants from becoming
interposable. Windows uses the package's private import libraries without DLL
forwarders. No public `av_*`, codec or resampler functions escape media.

Root CLI contracts inspect provider exports, product consumer imports/definitions
and final link commands, including stripped Release binaries. Tests that construct
or independently inspect FFmpeg fixtures have explicit PRIVATE links and includes;
they are the only allowed exceptions to the product ownership rule.

RGB-to-YUV420p conversion belongs to the tensor library. A single Slang program
serves CUDA, Vulkan and Metal; its scalar quantization/BT.601 formulas are shared
with the CPU implementation. Producers can allocate result planes or reuse
validated storage. Byte rounding, nonfinite values, strided inputs and tensor
execution timelines retain explicit contracts.

Encoder writers borrow planes only until the callback returns. Reentrant open,
write, close and ownership transfer are rejected before session state changes;
normal serialized callers incur no extra synchronization or pixel copies.

License obligations follow the repository GPL-3.0-or-later distribution and exact
installed package notices. FFmpeg's effective license is build dependent; the
runtime reports it alongside configuration flags. There is no separate manifest
or copied SDK license inventory in this revision. A replacement codec build or
future dependency must undergo its own version/transitive/license review.

## MCP integration

`media.capabilities` exposes the registered backends and codec build. `media.probe` is a synchronous read-only tool using the same inventory JSON as the CLI. `media.extract` accepts the facade's selection, geometry, sharpness, time range and backend options and starts the `media.extract` runtime job. It returns promptly; decoding and file writing run in a guarded worker. Only one extraction can run at a time.

Use `runtime.job.describe`, `runtime.job.wait` and `runtime.job.control` (`cancel`) for state and control, or `lichtfeld://runtime/jobs/media.extract`. Media job queries/control bypass the GUI work queue. `runtime.events.tail` and subscriptions expose `media.extract.started`, `.progress`, `.completed`, `.failed` and `.cancelled`, with generation, progress and accepted counts. Progress state is updated on every callback; event delivery is throttled to 10 Hz plus the final update to keep the journal bounded. Native typed error codes are retained in the wire envelope. Application shutdown stops MCP requests, cancels and joins the worker before releasing the viewer/GPU runtime.

The `MediaMcpContracts` test uses the production tool registry and event routing, verifies file pixels against FFmpeg, overlapping-job rejection, cancellation, retained failures and nested schema validation. It is registered with existing root tests without adding a CI workflow.
On macOS, its native runner loads the complete static CPython embedding archive,
following Studio's existing test-host convention. Shared Studio libraries resolve
Python symbols from their host; an ordinary archive link can discard symbols
needed at runtime before the runner enters `main`.

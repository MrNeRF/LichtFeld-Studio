# CPU Media Ingest library and command-line interface

## Boundary and build profiles

`io/media/media_ingest.hpp` exposes `lfs::media::MediaIngest` independently of
`VideoFrameExtractor::Params`. Inputs, requests, reports, callbacks and errors
use standard C++ types and `lfs::Result`. Probe and frame-sink types remain the
same public contracts used by the extractor. No FFmpeg, tensor, renderer, GUI or
JSON types are required by an SDK consumer.

`media/CMakeLists.txt` configures a C++23-only project without configuring the
application, CUDA, shaders or GUI. It builds the static `lfs_media` archive,
exported as `LichtFeldMedia::media`, and optionally the `media-ingest` executable.
The standalone profile uses software decoding and CPU RGB8 delivery. It has no
application logger, crash/event runtime or GPU context. Minimal diagnostics send
warnings/errors to stderr. Host contract failures become standard exceptions.

Production probe, selection, scoring, timing, resizing, rotation and file writer
sources are reused. `MediaIngest` currently adapts its public request to the
existing extractor internally: the algorithms have not been copied into another
implementation. `LFS_MEDIA_CPU_ONLY` disables hardware backend selection and
uses an HDR renderer implementation which explicitly reports unavailable. An
HDR-to-SDR request returns `Unsupported` before sink callbacks or output creation.

The application's `lfs_video` target also builds the facade. Its existing
`VideoFrameExtractor::extract` entry point and hardware paths remain available.
The standalone archive and application libraries are alternative compilation
profiles of shared implementation sources; do not link both into one executable,
which would duplicate symbols. This stage does not migrate the application GUI
to the installed archive or separate every internal decoder class.

## Public request, output and failure contracts

`MediaIngest::extract(request, sink)` delivers frames synchronously on the caller
thread. `extractFiles(request, files)` uses the existing PNG/JPEG writer policy,
legacy filename formatting/deduplication and optional schema-2 metadata. The
request contains no output directory. Custom sinks receive no implicit images
or metadata files. See [frame sinks](media-frame-sinks.md) for layout validation,
ownership, memory limits, callback order and partial-result handling.

| Request field | Default and interpretation |
|---|---|
| input | Required filesystem path; empty and embedded-NUL paths are rejected |
| selection | FPS mode, one frame/second; interval mode requires a positive decoded-frame interval |
| geometry | Original dimensions; positive finite scale or positive custom width/height, followed by explicit clockwise rotation 0/90/180/270 |
| sharpness | Disabled; Combined scoring, threshold zero, window disabled, ten candidate target |
| start_seconds / end_seconds | Zero / -1 (no requested end); existing trim/timestamp semantics apply |
| convert_hdr_to_sdr | False; true returns Unsupported for this CPU API |
| progress | Optional synchronous callback, processed/estimated/discarded counters |
| cancelled | Optional synchronous predicate; accepted results survive cancellation |

`IngestReport` contains accepted frame count and discarded sharpness candidates.
Processed progress counts are the existing extractor counters, not a promise of
precise source frame inventory or elapsed-work percentage. Callback estimates can
be approximate, especially for VFR or sources without declared frame counts.

Request-only validation occurs before decoder opening/output creation; actual
source dimensions and timebase are checked by the decoder. Scaling is validated
against real dimensions, rather than rejecting a fractional scale using the
placeholder dimensions of request-only validation. Invalid selection, geometry,
sharpness method, numeric values and output format return `InvalidArgument`.
The facade follows existing parameter validation for JPEG quality 1–100.
Direct `FileFrameSink` preserves its separate legacy normalization policy (zero
means 90, other JPEG values clamp to 1–100, PNG ignores this option).

Sink `Result` failures preserve their code/domain/native status and add accepted
frame count in error context. Decoder/compatibility-adapter failures currently
return `Unavailable`, except cancelled jobs return `Cancelled`; this bridge does
not yet preserve a distinct native error for every decoder failure. Probe uses
its existing richer error taxonomy. Host exceptions become `Internal`. A failed
or cancelled extraction does not imply that already accepted images/snapshots
were removed. File writes are not atomic; metadata-writer warning behavior remains
that of the compatibility adapter.

```cpp
#include <io/media/media_ingest.hpp>

lfs::media::IngestRequest request;
request.input = "clip.nut";
request.selection.mode = lfs::media::SelectionMode::Interval;
request.selection.interval = 2;
lfs::media::MemoryFrameSink sink; // default: 256 MiB payload, 100000 surfaces
const auto result = lfs::media::MediaIngest::extract(request, sink);
if (result && !sink.frames().empty()) {
    auto retained = sink.frames().front(); // shares immutable owning pixels
    // Retained remains valid after the sink is reused or the decoder closes.
}
```

## Configure, install and consume

Use a C++23 compiler/standard library, CMake 3.24+, and the codec dependency prefix.
The application vcpkg installation can be reused without another installation:

```sh
cmake -S media -B build-media -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake \
  -DVCPKG_TARGET_TRIPLET=<triplet> \
  -DVCPKG_INSTALLED_DIR=/path/to/existing/vcpkg_installed \
  -DVCPKG_MANIFEST_INSTALL=OFF
cmake --build build-media --parallel 2
cmake --install build-media --prefix /path/to/media-sdk --config Release
```

On Windows run these commands in the Visual Studio developer environment. Native
MSVC sources use `/utf-8`; Windows CLI arguments are received by `wmain` and
converted from wide paths to UTF-8. JSON output is UTF-8. For a library-only build
set `-DLFS_MEDIA_BUILD_CLI=OFF`.

`media/vcpkg.json` is a standalone manifest with the repository's pinned baseline
and OpenEXR overlay. Without a reused prefix, configure with the vcpkg toolchain
and allow manifest installation. FFmpeg defaults are disabled and only avcodec,
avformat and swscale features are requested; the actual resulting codec/license
configuration must be checked, not inferred from the manifest. Non-vcpkg builds
find codec packages using `CMAKE_PREFIX_PATH`; FFmpeg headers/libraries and WebP
pkg-config fallback must be discoverable.

Install exports the archive, public `io/media` headers, their core error-contract
headers, CMake config/version/targets and license notices. It does not install
legacy extractor/tensor headers. The package is relocatable relative to its own
prefix; external codec dependencies still need their own discoverable prefix.
For example:

```cmake
cmake_minimum_required(VERSION 3.24)
project(MediaConsumer LANGUAGES CXX)
find_package(LichtFeldMedia 0.1 CONFIG REQUIRED)
add_executable(consumer main.cpp)
target_link_libraries(consumer PRIVATE LichtFeldMedia::media)
```

Configure the consumer with both SDK and codec prefixes/toolchain as needed. The
exported target propagates C++23 and necessary static-library definitions. SDK
installation is not a self-contained runtime release: the executable and consumer
still require dynamic codec libraries on PATH/the platform loader's search path.
Dependency DLL deployment and corresponding source bundles are release work.

## CLI commands and protocol

```sh
media-ingest version
media-ingest capabilities
media-ingest probe "clip.nut" --timeout-ms 10000
media-ingest probe "clip.nut" --headers-only
media-ingest extract "clip.nut" --output "frames" --interval 2 --metadata
media-ingest extract "clip.nut" --output "frames" --fps 5 --start 1 --end 10 \
  --scale 0.5 --rotate 90 --format jpeg --quality 95 --name 'frame_%05d' --quiet
```

Help lists selection, geometry, filename, metadata and sharpness options. Choose
FPS or interval once, and scale or custom size once. Unknown options, missing or
non-finite numeric arguments and conflicting modes produce structured failures.
File output defaults to PNG; metadata must be requested. Sharpness supports
`--sharpness`, `--window`, `--algorithm laplacian|tenengrad|combined` and
`--candidates`. Rotation is explicit, using the existing geometry behavior;
probe orientation metadata does not silently change requested output geometry.

Except help, stdout contains one final JSON object with `schema_version: 1` and
`success`. Successful probe adds `media`: container, probe-depth flag, optional
container start/duration, selected video stream and stream inventory. Rational
fields are `[numerator, denominator]`; timestamps are signed ticks with rational
timebase; absent values are null. Stream kinds and orientation origins use
strings. Successful extraction reports `frames_accepted` and `discarded`.

Failed results add `error` with code, domain, message, context operations/fields
and optional native domain/code/name. On failure, inspect `frames_accepted` in
context to account for partial outputs. `version` includes the Media Ingest
version plus the actual linked FFmpeg runtime version, license and configure
flags. `capabilities` reports software decode/RGB8/PNG/JPEG true and hardware
backend/HDR tone mapping false; input codec support depends on the linked build.

Progress JSON lines go to stderr; `--quiet` suppresses progress. Diagnostic
warnings/errors can also appear there, so consumers must not treat every stderr
line as a progress JSON object. SIGINT requests cooperative cancellation through
the public predicate. Cancellation is checked at decoder/processing boundaries,
not an immediate interruption of every blocked operation. The public predicate
and partial-result behavior are tested; OS console-signal delivery requires
separate platform qualification.

| Exit code | Meaning |
|---|---|
| 0 | Success or help |
| 2 | InvalidArgument |
| 3 | Unsupported, Unavailable, NotFound, PermissionDenied or DataLoss |
| 4 | Other structured failure, including ResourceExhausted/Internal |
| 130 | Cancelled |

## Dependencies and licensing

No library is introduced beyond the existing media/image codec stack; OpenImageIO
is not required. Dependencies remain FFmpeg, nlohmann-json, libjpeg-turbo, libpng,
zlib, TIFF, WebP and OpenEXR plus their transitive dependencies. TIFF/WebP/OpenEXR
are currently linked because the reused image-codec implementation contains
those dispatch paths. The CLI only exposes PNG/JPEG; linking OpenEXR does not
implement a new float/half EXR frame pipeline.

| Dependency | Relevant license family / source of notice |
|---|---|
| Module and reused production sources | GPL-3.0-or-later, repository LICENSE and SPDX headers |
| FFmpeg | Build-dependent LGPL-2.1-or-later or GPL; use runtime version/license/configuration and installed notice |
| nlohmann-json, stb_image | MIT; installed copyright / bundled stb MIT alternative |
| libjpeg-turbo | IJG/BSD/zlib notices from the dependency installation |
| libpng, zlib | libpng and zlib notices |
| TIFF, WebP, OpenEXR/Imath | Permissive notices from installed packages; preserve exact texts |

The SDK copies available vcpkg copyright notices for direct and known transitive
ports, plus the selected stb_image MIT text and repository LICENSE. For a
non-vcpkg dependency installation preserve notices from that installation when
preparing distribution. This is not a certification of arbitrary replacement
FFmpeg builds. The reused Windows prefix reports FFmpeg 8.0.1 with GPL version 2
or later, including libx264. [FFmpeg's licensing documentation](https://ffmpeg.org/legal.html)
explains how optional GPL components change the build's license. GPL-2.0-only
and nonfree replacement builds must not be substituted without a compatibility
review against the repository's GPL-3.0-or-later license.

## Verification and remaining boundaries

The existing media CTest project builds both the compatibility adapter and the
real standalone library/CLI using one dependency prefix. Existing Release CI jobs
run the added tests; no separate workflow/job is needed. CLI tests compare real
PNG/JPEG bytes, names and frame metadata to the legacy path across CFR/VFR,
resize, rotation, trim and sharpness windows. They also test UTF-8 paths, probe
rational metadata, quiet operation, malformed arguments, unavailable HDR, real
writer failures and retained native filesystem errors.

`MediaIngestPackageConsumer` installs and relocates the SDK, configures/builds a
separate project with no repository include paths and exercises public probe,
frame ownership, extraction to memory/files, fractional scaling, bounded memory,
structured/native sink errors and cancellation. It also verifies required license
files. This checks the archive/API rather than only a CLI test adapter.

Full application/backend builds, new EXR/HDR processing, a runtime dependency
bundle, fresh standalone dependency installation and every platform's OS signal
behavior are distinct qualification steps. Local Windows CPU results do not
certify Linux/macOS or GPU execution. Measured performance and platform status
belong in the implementation plan/validation artifact, not in the API contract.

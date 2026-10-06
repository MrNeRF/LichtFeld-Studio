# Shared media probe and video metadata contracts

Branch: `codex/media-probe-contracts`.
Base: `19eb57ecde873f161fdefee68ded1a2741fe43ea` (`upstream/dev`, includes PR #2877).
Validation and scope are recorded below; remote check results are tracked by the pull request.

## Implemented API

`src/io/include/io/media/media_probe.hpp` exposes `lfs::media::MediaProbe::inspect`
using standard C++ types. The public header requires neither FFmpeg headers nor
GUI, renderer or JSON types. Implementation is currently compiled into the
existing `lfs_video` target and the standalone CPU test runner.

| Contract | Result |
|---|---|
| Container and streams | Container name, all streams in container order, stream type and codec; optional first usable video index matching legacy selection |
| Geometry | Optional width, height, sample aspect ratio and pixel format |
| Timing | Raw signed ticks with rational timebase; separate nominal/average FPS; optional declared frame count; missing/invalid values remain absent |
| Color | Optional primaries, transfer, matrix, range and component depth derived from pixel format or codec raw sample depth |
| Orientation | Original rotate tag and nine display-matrix integers, metadata source, optional clockwise angle and reflection flag |
| Probe depth | Headers or FFmpeg stream-info probing; result records whether packet probing was requested and completed |
| Errors | Typed option/open/probe/timeout errors, native FFmpeg code and message; unsuccessful results contain no partial description |
| Lifetime | Returned strings/vectors/matrices own their data; no FFmpeg pointer escapes the API |

No images are written. StreamInfo can read packets and run FFmpeg software codec
probing; it does not initialize a hardware context. The timeout uses FFmpeg's
cooperative interrupt callback and is not a hard wall-clock guarantee. Empty or
NUL-containing paths and invalid timeout/depth options are rejected before opening.
Property values can originate from FFmpeg packet analysis; declared FPS does not
prove CFR, and a missing duration is not synthesized from frame count/FPS.

## Integration and corrected behavior

Player and extractor use a shared video-stream probe on their already-open input.
It preserves the first usable video selection and video-only discard policy.
When dimensions are absent in the header, the full fallback probe runs once; its
metadata is reused instead of calling `avformat_find_stream_info` again. A second
call after fallback caused an access violation with the local FFmpeg 8.0.1 library
on the generated MPEG-4/MOV sample. The crash was also reproduced with the original
player from dev; the shared path fixes both consumers and removes duplicate work.

The player reads stream and decoded-frame display matrices through the same
checked helper. Truncated matrices are not read and non-finite angles stay unknown.
Rotate-tag precedence is retained, but malformed/overflow tags no longer use
`atoi` or silently accept a numeric prefix. Quarter-turn compatibility is applied
separately from the raw orientation description. Reflection metadata is retained
without adding a new mirror transform to existing extraction or preview output.
The SDR player still supplies unrotated pixels and a rotation value to its caller;
explicit extractor rotation continues to transform the saved images.

## Verification

Executed locally on Windows, 6 October 2026:

- MSVC 19.44 Release build compiling actual probe, player and extractor sources.
- `MediaExtractionContracts`: 22 Python test methods, including the original 16
  extraction tests and 6 added probe/preview tests with parameterized cases.
- `MediaProbeUnitContracts`: native assertions for absent properties, signed and
  zero timestamps, invalid timebases, tag/matrix precedence, reflection,
  truncated/degenerate matrices, invalid tags/options/paths and stream selection.
- `MediaFixturePreparation`: 7 Python test methods.
- I/O discipline, backend-neutrality and Python syntax checks; `git diff --check`.

The suite compares source properties to independent ffprobe output, verifies
probe does not change source bytes, tests audio-only and audio-first/multi-video
inputs, and compares real SDR preview and explicitly rotated extraction pixels
against independent FFmpeg decoding. MOV display matrices are written directly
into a known synthetic track-header fixture and verified by ffprobe, avoiding
version-dependent rotate-tag remux behavior. Pixel comparisons allow 3/255 for
library-version rounding. Native assertions fail in Release builds as well.

The existing Release CI steps already run the expanded CTest directory: no workflow
or job is added. This branch has not run on Linux/macOS or with GPU/HDR backends;
a cooperative timeout expiry has not been qualified across blocking demuxers.
The JSON adapter is test support, not a public command-line interface.

## Commands

Reuse the installed application dependencies without another vcpkg install:

```sh
cmake -S tests/media/extractor -B build-media-extractor -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake \
  -DVCPKG_INSTALLED_DIR=/absolute/path/to/application-build/vcpkg_installed \
  -DVCPKG_TARGET_TRIPLET=x64-windows -DVCPKG_MANIFEST_INSTALL=OFF
cmake --build build-media-extractor --parallel 2
ctest --test-dir build-media-extractor --output-on-failure --no-tests=error
cmake -S tests/media -B build-media-fixtures
ctest --test-dir build-media-fixtures --output-on-failure --no-tests=error
```

Use the application's actual triplet on other platforms. Python 3.10+, FFmpeg and
ffprobe are needed; the expanded fixture tests require FFV1/rawvideo/NUT,
MPEG-4/MOV, PCM/WAV and lavfi sine support. Missing capabilities fail the tests.
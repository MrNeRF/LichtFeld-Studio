# Media Ingest baseline and extraction regression tests

Branch: `codex/media-baseline-tests`.
Starting revision: `94fa31753408ff4e8f344c3c04adacf56a6983ae` (`upstream/dev`).

## Scope

This branch prepares the baseline for Media Ingest: the current video extractor
will become one consumer of a broader session/probe/frame/sink system. It supplies
independently generated media and regression tests against the production
VideoFrameExtractor. EXR export, vendor ingest and GUI redesign remain later work.

Existing extraction naming tests use the LichtFeld encoder and CUDA buffers for
some inputs. The new corpus uses FFmpeg software encoding with deterministic
Python RGB patterns. This removes those dependencies from fixture preparation
and lets the same checks run on Windows, Linux and macOS.

## Changes

- `scripts/prepare_media_fixtures.py`: prepare CFR, irregular-PTS VFR and a
  declared panorama-layout fixture; lossless FFV1/NUT; verify exact RGB, dimensions
  and timestamps before publishing the corpus directory.
- `tests/media/test_fixture_preparation.py`: seven tests covering integrity,
  incorrect temporal/geometric expectations, existing destination, missing tools
  and a failure after a previous fixture was successfully written.
- `tests/media/CMakeLists.txt`: standalone CTest entry without a C++ compiler,
  GUI, CUDA or vcpkg dependency.
- `tests/media/extractor`: standalone SDR runner compiling production extractor
  and image codecs; test-only logging and an HDR tripwire that fails on entry.
- `tests/media/test_extraction_contracts.py`: sixteen tests of actual extraction,
  metadata and failures, with independent pixel/time/resize/YUV references.
- Production fixes: RGB resize no longer applies YUV matrix settings, and
  filesystem errors preserve Unicode paths as UTF-8.
- Existing Ubuntu, Windows and macOS build workflows run preparation and
  extraction checks inside their Release jobs, reusing installed application
  dependencies without adding workflows, jobs or a separate dependency build.
- `.gitignore`: narrow exceptions for the three new Python files.

## Verification

Executed locally on Minisforum, Windows, 6 October 2026:

```text
Python 3.11.15
FFmpeg/ffprobe 9.0.2 essentials build
cmake -S tests/media -B build-media-fixtures
ctest --test-dir build-media-fixtures --output-on-failure
cmake -S tests/media/extractor -B build-media-extractor -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=D:/repos/LichtFeld-Studio/build-windows-release/vcpkg_installed/x64-windows
cmake --build build-media-extractor
ctest --test-dir build-media-extractor --output-on-failure
```

Result: two CTest entries passed, covering 23 Python test methods and additional
parameterized cases; the SDR runner was compiled with MSVC 19.44. The corpus includes
paths with spaces and Unicode. Missing tools are errors, not successful skips.
The existing build workflows include the other platforms; Linux/macOS and Python 3.10 have not
yet been executed for this branch. No CI run, remote PR or publication is implied.

## Boundaries and next baseline work

The panorama fixture tests layout only. It has no lens calibration, stitching
reference, vendor metadata or promise of projection detection from its container.
It cannot demonstrate SfM quality. HDR/LOG, EXR export, rotation side data, audio,
telemetry and multicamera will require separate references.

File hashes identify a particular generated corpus. Reference decoded pixels
and timestamps, rather than container byte identity across FFmpeg versions,
define the invariant. Tool versions and hashes are retained in the manifest.

The extractor suite now covers interval/FPS, trim, source naming, schema-2 metadata,
PNG/JPEG, explicit rotation, resize, VFR timing, cancellation and error paths.
Automatic orientation still belongs to the player/dialog boundary. HDR/CUDA tests,
GUI checks and an installable Media Ingest library remain outside this SDR target.

Current interval/FPS semantics differ at an exact end timestamp; the suite records
that difference explicitly. VFR source numbering is nominal-FPS-derived, and
cancelled extraction preserves completed images without a partial manifest.
Future Media Ingest changes must migrate these contracts deliberately.

Public-media URLs will be added only with verified provenance, download license,
SHA-256 and properties. No network or third-party media are necessary for this
first block. See `tests/media/README.md` for the catalog contract and commands.

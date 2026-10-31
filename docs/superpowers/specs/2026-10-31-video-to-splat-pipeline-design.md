# Video to Splat Pipeline (FreeGaussian-style one-click flow)

Date: 2026-10-31
Status: approved design, awaiting implementation plan

## Goal

Turn a 360 or ordinary video into a trained Gaussian splat inside LichtFeld Studio with one action, the way
freegaussian.ai does in the cloud, but locally on the user's GPU. The user drops a video or picks it in the New
Project dialog, the app extracts frames, solves camera poses, trains, and the result is ready to export with the
existing exporters.

LichtFeld already has every stage except camera pose estimation (structure-from-motion, SfM) and the glue that
chains the stages. This project adds those two things.

## Decisions taken during design

| Question | Decision |
| --- | --- |
| What is the clone? | A one-click pipeline inside LichtFeld Studio, not a hosted service. |
| SfM engine | COLMAP 4.2.1, downloaded on first use on Windows. GLOMAP was archived in March 2026 and lives on as COLMAP's `global_mapper` command. COLMAP 4.1 added native equirectangular camera models. |
| Inputs, version 1 | Stitched equirectangular MP4/MOV (detected by 2:1 aspect or spherical metadata) and ordinary perspective video. Raw `.insv` dual-fisheye is out of scope. |
| Extras | People masking, gravity leveling + metric scale, and textured mesh export are wanted but are each a separate spec after the core ships (see Roadmap). |
| Implementation shape | Python plugin first (fast iteration while the COLMAP recipe is tuned), then port the orchestration to a native runtime job. |

## Non-goals for version 1

- Raw `.insv` or any dual-fisheye input.
- Vocabulary-tree loop detection (needs a second download).
- Headless `--video` CLI flag and a dedicated MCP runtime job. Both come with the native port.
- People masking, IMU scale/leveling, mesh export.

## Architecture

A new plugin package `src/python/lfs_plugins/video_to_splat/` plus one small C++ binding. Each module has one
job and a narrow interface.

### `toolchain.py`

Finds a usable COLMAP binary or explains what to do.

- Windows: downloads the pinned COLMAP 4.2.1 `colmap-x64-windows-cuda.zip` (about 395 MB) from the official
  GitHub release, verifies a pinned SHA-256, and extracts it into the app cache directory under
  `tools/colmap-4.2.1/`. A consent dialog runs before any download, mirroring the LPIPS weights flow in
  `src/preprocessing/preprocess.cpp` (`ensure_cached_model`). A cached copy is hash-checked on every use and
  re-downloaded if it fails.
- Linux: there is no official binary. Looks for `colmap` on `PATH` or at a user-configured path in preferences,
  runs `colmap --version`, and requires 4.1 or later (first release with spherical camera models).
- Returns the binary path or a typed error with a user-facing fix: allow the download, install COLMAP 4.1+, or set
  the path in preferences.

### `lf.video` native binding (the only new C++)

A new `py_video.cpp` in `src/python/lfs/` exposing:

- `lf.video.probe(path)` → width, height, fps, duration, rotation, and whether spherical metadata is present.
  Backed by the existing `VideoPlayer` and `video_metadata.cpp`.
- `lf.video.extract_frames(params, on_progress, cancel)` → wraps `lfs::io::VideoFrameExtractor` from
  `src/io/video_frame_extractor.hpp`. Params mirror `VideoFrameExtractor::Params` (fps or interval, trim, JPEG
  quality, max width, sharpness filter, rotation, HDR to SDR). Releases the GIL during extraction and marshals
  progress and cancel callbacks back to Python.

### `sfm.py`

Builds and runs the COLMAP command sequence for one dataset folder:

1. `feature_extractor` with `--ImageReader.single_camera 1`. Camera model `EQUIRECTANGULAR` for 360 input,
   `OPENCV` for perspective. SIFT on GPU where the binary supports it.
2. `sequential_matcher` with an overlap window from the quality preset. Loop detection off in version 1.
3. `global_mapper` writing `sparse/0/{cameras,images,points3D}.bin`.

360 detection: aspect ratio within 2 percent of 2:1, or spherical metadata from `probe`. The UI can force either
mode. Exposes `plan_commands(...)` (pure, testable) and `run(...)`.

### `process.py`

The cancellable subprocess runner: streaming output with a bounded tail, process-group creation, process-tree
kill on cancel, timeout. Lifted out of `installer.py` (`_run_cancellable_process`, `_terminate_process_tree`,
`_process_group_popen_kwargs`) into this shared module; the installer imports it from here.

### `pipeline.py`

The orchestrator. Stages in order:

`toolchain` → `extract` → `features` → `match` → `map` → `create_project` → `train`

- Runs on a worker thread; UI work is posted back to the main thread.
- One progress callback: `(stage, fraction, message, log_tail)`.
- Writes `<project>/pipeline.json` after each stage with the stage name, parameters and outcome. A rerun on the
  same project skips stages recorded as complete with identical parameters.
- Refuses to start while a scene is training.

### `panel.py` and menu integration

- The New Project dialog (`import_panels.py`) gains a `video` source kind. When the source path has a supported
  video extension (`lfs::io::video::kSupportedVideoExtensions`), the panel shows a small options block: frame
  rate, 360 mode (auto / on / off), quality preset (fast / balanced / high), and a staged progress view once
  running.
- File menu: `New Project from Video…` next to `Extract video frames`.
- Dropping a video onto the window asks `Extract frames only` or `Build splat` instead of always opening the
  extractor (`input_controller.cpp` drag-and-drop path).

## Data flow and project layout

```
input video
  → frames            <project>/dataset/images/frame_000001.jpg …   (2 fps default, sharpness filter on, JPEG 95)
  → COLMAP database   <project>/dataset/database.db
  → sparse model      <project>/dataset/sparse/0/{cameras,images,points3D}.bin
  → project           existing creation path, <project>/dataset treated as an embedded COLMAP dataset
  → training          current defaults; equirectangular cameras use the rasterizer's native path
  → export            existing PLY / SPZ / SOG / USD exporters
  → state             <project>/pipeline.json
```

Quality presets map only to frame rate, maximum image width and matcher overlap:

| Preset | fps | max width | sequential overlap |
| --- | --- | --- | --- |
| fast | 1 | 2880 | 10 |
| balanced (default) | 2 | 4096 | 15 |
| high | 3 | source | 25 |

## Error handling

- **No COLMAP.** Download refused, offline, hash mismatch, or Linux without a 4.1+ binary: stop at `toolchain`
  with a message naming the fix. Nothing else runs.
- **Bad input.** Unreadable video, zero frames after the sharpness filter, or fewer than 20 frames: stop before SfM
  with the frame count and a suggestion (raise the frame rate, lower the sharpness threshold).
- **Weak reconstruction.** After `global_mapper`, fewer than 60 percent of frames registered or the largest model
  under 20 images: keep the dataset, show the registered ratio, and ask whether to train anyway or stop. The ratio
  is always shown.
- **Cancellation.** Kills the COLMAP process tree at any stage; `pipeline.json` keeps the last completed stage so
  a rerun resumes.
- **GPU contention.** SfM finishes before training starts, so COLMAP and the trainer never share VRAM.
- **Disk.** The `extract` stage checks free space on the destination and warns under 10 GB.

## Testing

- **Python unit tests.** Toolchain resolution with a fake filesystem and fake `--version` output; 360 detection
  from probe results; `plan_commands` for both camera models and all presets; stage planning and resume from each
  `pipeline.json` state; cancel propagation in `process.py`.
- **C++ unit test.** The `lf.video` binding round-trips `Params` and reports progress on a small fixture video
  under `tests/data`.
- **Opt-in integration test.** LichtFeld can render an equirectangular video from an existing splat. The test
  renders a 360 orbit of a known scene, runs the full pipeline, and asserts COLMAP registers over 90 percent of
  frames and training reaches a PSNR floor against the source renders. Gated behind an environment variable
  because it needs COLMAP and a GPU.

## Roadmap (separate specs, in this order)

1. **Automatic people masking.** Segmentation model through the preprocessing module's ONNX Runtime, writing a
   masks folder that the existing `--mask-mode` training consumes.
2. **Textured mesh export.** COLMAP 4 bundles dense stereo, Poisson meshing and texture mapping; add those as
   pipeline stages and write GLB/USDZ through the Assimp dependency already in the build.
3. **Gravity leveling and metric scale.** Parse the CAMM or GPMF IMU track with the bundled FFmpeg, estimate
   gravity and scale, apply as a scene transform. Camera coverage is patchy, so this comes last.
4. **Native port.** Move orchestration into a `video_to_splat` runtime job with its own event stream, MCP tool and
   `--video` CLI flag. The plugin becomes the UI over that job.

## References

- COLMAP releases: https://github.com/colmap/colmap/releases (4.2.1, 2026-09-29; Windows CUDA zip 395 MB)
- COLMAP changelog: https://colmap.github.io/changelog.html (4.0.0 integrates GLOMAP as `global_mapper`;
  4.1.0 adds spherical/equirectangular camera models)
- GLOMAP archived 2026-03-09: https://github.com/colmap/glomap
- Equirectangular COLMAP model id 17 handling: `src/io/formats/colmap.cpp`
- Existing download-with-consent pattern: `src/preprocessing/preprocess.cpp` (`ensure_cached_model`)
- Existing subprocess runner: `src/python/lfs_plugins/installer.py`

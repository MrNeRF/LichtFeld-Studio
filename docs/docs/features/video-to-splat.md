---
title: Video to Splat
---

# Video to Splat

Create a trained splat from a video in one step. LichtFeld extracts frames, solves the camera poses with
COLMAP, creates the project and starts training.

## Requirements

- **Windows:** nothing to install. The first run asks to download COLMAP 4.2.1 (about 395 MB) from the official
  GitHub release. The archive is verified against a pinned SHA-256 and unpacked under the app cache directory as
  `tools/colmap-4.2.1/`.
- **Linux:** install COLMAP 4.1 or later and make sure `colmap` is on `PATH`, or set the path in the plugin
  settings (`video_to_splat` → `colmap_path`). Version 4.1 is the first with the equirectangular camera model.
- A CUDA GPU. COLMAP runs on the GPU before training starts, so the two never share memory. Creating a project
  from video is refused while another training is active. When the official COLMAP build cannot run its SIFT
  kernels on the GPU (Pascal cards, for example), the stage ends with an empty database and is rerun on the CPU
  automatically; the dialog says so.

## Workflow

1. **File → New Project from Video…**, or drop a video onto the window, or type a video path into the New
   Project source field. **File → New Project from Photos…** does the same for a folder of JPEG or PNG photos
   without camera poses; the frame extraction step is skipped and the photos are linked into the project.
2. Choose **360 video**: Auto detects a 2:1 aspect ratio or spherical metadata; override when needed.
3. Choose a **Quality** preset:

   | Preset | Frames per second | Max frame width | Matching window |
   | --- | --- | --- | --- |
   | Fast | 1 | 2880 | 10 |
   | Balanced | 2 | 4096 | 15 |
   | High | 3 | source | 25 |

4. Press **Create**. The dialog stays open and shows the current stage: locating COLMAP, extracting frames,
   detecting features, matching frames, solving cameras, leveling the scene, creating the project, starting
   training. Cancel stops COLMAP and keeps what finished.

   Leveling runs COLMAP's `model_orientation_aligner` so the floor is horizontal in the viewer: Manhattan-world
   vanishing points for ordinary video, the mean camera orientation for 360 footage (lines are curves in an
   equirectangular image). It is cosmetic; when it fails the model keeps its original orientation.
5. Training starts automatically. For 360 video the GUT rasterizer is switched on first, because the
   standard rasterizer only accepts pinhole cameras; note that most third-party viewers cannot display
   GUT-trained splats. Export with the usual exporters.

**Extract frames only** opens the frame extractor on the same video for a manual workflow.

## Inputs

Stitched equirectangular MP4 or MOV from any 360 camera (export with direction lock on), or ordinary perspective
video. Raw `.insv` dual-fisheye files are not supported; stitch them in the camera maker's software first.

A folder of photos works the same way: JPEG or PNG files directly inside the folder, at least 20, in shooting
order by name. 2:1 photos are treated as equirectangular.

Frames are extracted at the preset's rate, keeping the sharpest frame of each interval, scaled down to the
preset's maximum width and saved as JPEG. Fewer than 20 frames stops the run before COLMAP with a hint to use a
longer clip or raise the frame rate.

## When the reconstruction is weak

If COLMAP registers under 60 percent of the frames, or fewer than 20, the dialog shows the ratio and asks whether
to train anyway or stop. Usually the fix is more overlap: move slower, choose the High preset, or shoot with
better light and texture. The registered ratio is always reported when the run finishes.

## Files

A `.licht` project is a single file, so the pipeline keeps its working data in a folder of the same name next
to it. For `D:/projects/room.licht`:

```
D:/projects/room/
  dataset/images/frame_000001.jpg …   extracted frames
  dataset/extraction_metadata.json    frame timestamps and sharpness scores
  dataset/database.db                 COLMAP features and matches
  dataset/sparse/0/{cameras,images,points3D}.bin
  pipeline.json                       finished stages and their results
```

`pipeline.json` records each finished stage with the request fingerprint (video path and modification time,
360 mode, preset; for a photo folder also the list of file names). Pressing Create again on the same project name with the same video and settings resumes from
the first unfinished stage; changing the preset or the video starts over.

## Checking a build end to end

`scripts/video_to_splat_smoke.py` runs the whole chain inside the app's Python editor on a clip named by
`LFS_V2S_VIDEO` and fails when fewer than 90 percent of the frames register. It is opt-in because it needs COLMAP,
a GPU and real footage.

## Automation

The plugin registers three MCP tools on the app's endpoint, so an agent or script can run the whole chain
without the dialog. Python tools appear with a `plugin_` prefix in `tools/list`, so call
`plugin_video_to_splat_start` and friends:

| Tool | Arguments | Returns |
| --- | --- | --- |
| `video_to_splat_start` | `source` (video file or photo folder), `project` (.licht path), `mode` (auto, on, off), `preset` (fast, balanced, high), `use_gpu` | `{started, project, work_dir}` or `{started: false, error}` |
| `video_to_splat_status` | none | `{active, stage, fraction, message, done, error, cancelled, frames, registered, mode, project, work_dir}` |
| `video_to_splat_cancel` | none | `{cancelling}` |

One run at a time; a start while another run is active is refused. Training is left running when the run
finishes, so the usual `training_*` tools and `lichtfeld://runtime/jobs/training.main` take over from there.

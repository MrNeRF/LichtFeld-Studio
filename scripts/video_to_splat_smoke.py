# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""End-to-end check of the video-to-splat pipeline. Run inside LichtFeld's Python editor.

Set LFS_V2S_VIDEO to an equirectangular (or perspective) MP4 and optionally
LFS_V2S_PROJECT to a destination folder; the project is written there as
v2s-smoke.licht with its frames in v2s-smoke/. LFS_V2S_PRESET (fast, balanced,
high), LFS_V2S_MODE (auto, on, off), LFS_V2S_MASK (1/0, people masking) and
LFS_V2S_MESH (1/0, dense mesh) tune the run. Prints the registration ratio and
exits non-zero when it is under 90 percent.
"""
import os
import sys
import tempfile
import threading
from pathlib import Path

from lfs_plugins.video_to_splat import pipeline, project_bridge

MIN_RATIO = 0.9

video = Path(os.environ["LFS_V2S_VIDEO"])
destination = Path(os.environ.get("LFS_V2S_PROJECT") or tempfile.mkdtemp(prefix="v2s-smoke-"))
project = destination / "v2s-smoke.licht"
request = pipeline.Request(
    video=video,
    project=project,
    mode_override=os.environ.get("LFS_V2S_MODE", "auto"),
    preset=os.environ.get("LFS_V2S_PRESET", "balanced"),
    mask_people=os.environ.get("LFS_V2S_MASK", "1") not in ("0", "false", "no"),
    build_mesh=os.environ.get("LFS_V2S_MESH", "0") in ("1", "true", "yes"),
    publish_to_gallery=os.environ.get("LFS_V2S_PUBLISH", "0") in ("1", "true", "yes"),
)
done = threading.Event()
outcome = {}


def on_progress(stage, fraction, message, _tail):
    print(f"[{stage}] {int(fraction * 100):3d}% {message}", flush=True)


def worker():
    try:
        outcome["state"] = pipeline.run(request, project_bridge.build_hooks(), on_progress=on_progress,
                                        should_cancel=lambda: False)
    except Exception as exc:  # noqa: BLE001 - reported below
        outcome["error"] = exc
    finally:
        done.set()


threading.Thread(target=worker, daemon=True, name="VideoToSplatSmoke").start()
done.wait()
if "error" in outcome:
    raise SystemExit(f"pipeline failed: {outcome['error']}")
state = outcome["state"]
ratio = state.registered / state.frames if state.frames else 0.0
print(f"registered {state.registered}/{state.frames} ({ratio:.0%}) project={project} work_dir={request.work_dir}")
if ratio < MIN_RATIO:
    sys.exit(1)

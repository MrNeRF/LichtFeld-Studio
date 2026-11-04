# Video to Splat Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development (recommended) or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One action in LichtFeld Studio turns a 360 or perspective video into a trained Gaussian splat: extract frames, solve poses with COLMAP 4.2.1, create the project, train.

**Architecture:** A Python plugin package `src/python/lfs_plugins/video_to_splat/` orchestrates the stages on a worker thread and drives COLMAP as a subprocess. One new nanobind submodule `lf.video` exposes the existing C++ `VideoFrameExtractor` and metadata probe. The New Project dialog gains a `video` source kind. Spec: `docs/superpowers/specs/2026-10-31-video-to-splat-pipeline-design.md`.

**Tech Stack:** Python 3.12 (embedded, `import lichtfeld as lf`), nanobind, C++23, FFmpeg (libavformat), COLMAP 4.2.1 CLI, GoogleTest, pytest, RmlUi templates.

**Branch:** `video-to-splat-pipeline` (already created, spec committed). Commit after every task.

---

## Facts the engineer must know

- The native module is `lichtfeld`, always imported as `import lichtfeld as lf`. Stubs live in `src/python/stubs/lichtfeld/`; update them when adding bindings.
- Main-thread marshal from a worker thread: `lf.ui.schedule_on_ui_thread(callback)`.
- Dialogs: `lf.ui.confirm_dialog(title, message, buttons, callback, style)` calls back with the clicked **button label**. `lf.ui.message_dialog(title, message, style)`. No toast API exists.
- Settings per plugin: `lf.plugins.settings("video_to_splat")` returns an object with `get(key, default)`, `set(key, value)`. Type of `default` is enforced.
- Cache directory: environment variable `LFS_RESOLVED_CACHE_DIR` (read with `lfs_plugins.environment.value`).
- Project creation: `lf.project_create(path, discard_changes, stop_training, overwrite)`, then `lf.load_file(path=..., is_dataset=True, images_folder=..., discard_changes=True)`. `load_file` is asynchronous; poll `lf.ui.get_import_state()` (`active`, `success`, `error`) or `RuntimeState.trainer_loaded`. Training starts with `lf.start_training()`. `lf.is_training_active()` reports training.
- Built-in plugin modules are registered in `src/python/lfs_plugins/panels.py` `_build_builtin_panel_steps` (a hard-coded list of steps), **not** by `PluginManager`. New locale keys must be added to **all 10** files in `src/visualizer/gui/resources/locales/` or `tests/python/test_localization_contracts.py` fails. Use `@tr:key` in RML and `lf.ui.tr("key")` in Python; never hard-code UI strings (checked by `tools/check_ui_hardcoded.py`).
- Python tests live in `tests/python/`, run with `python -m pytest tests/python/<file> -q`. They stub `lichtfeld` as a `ModuleType` and import plugin modules fresh (see `tests/python/test_new_project_panel.py:23-115`). New fast tests must be added to `set(LFS_PYTHON_FAST_TESTS ...)` in `tests/CMakeLists.txt` (line ~1480).
- C++ tests are GoogleTest, listed in `set(TEST_FILES ...)` in `tests/CMakeLists.txt` (line ~574, video entries ~749-763). Build with `-DBUILD_TESTS=ON`, run `ctest --test-dir build/tests -L fast --output-on-failure`. Video fixtures: `tests/data/video/h264_420.mp4` (64x48, 5 frames). `PROJECT_ROOT_PATH` is defined for tests.
- Video C++ lives in static target `lfs_video` (`src/io/CMakeLists.txt:125`), **not linked by `lfs_py`** today. `VideoFrameExtractor::extract(params, error)` returns `bool`; the outcome is `lastOutcome()`. `probeVideoFileMetadata(path, VideoMetadata&, error)` in `src/io/video/video_metadata.hpp` reads rotation, fps, duration, frames, HDR; it does **not** read width, height or spherical tags yet.
- COLMAP 4.2.1 Windows CUDA asset: `https://github.com/colmap/colmap/releases/download/4.2.1/colmap-x64-windows-cuda.zip`, SHA-256 `e9c5cbd84c2ea986d2e970a2473fc2d2e6b34a2cdcf5d3df2765c319a63af881`, about 395 MB. Camera model string `EQUIRECTANGULAR` (4.1+). Commands: `feature_extractor`, `sequential_matcher`, `global_mapper`. Options confirmed from docs: `--database_path`, `--image_path`, `--output_path`, `--ImageReader.camera_model`, `--ImageReader.single_camera`, `--FeatureExtraction.use_gpu`, `--FeatureExtraction.max_image_size`, `--FeatureMatching.use_gpu`, `--log_target`. The `--SequentialMatching.overlap`, `--SequentialMatching.quadratic_overlap`, `--SequentialMatching.loop_detection` names come from COLMAP source conventions; Task 4 verifies them with `colmap sequential_matcher -h` before committing.

## File structure

| Path | Responsibility |
| --- | --- |
| `src/python/lfs_plugins/process_runner.py` | Cancellable subprocess with streaming output (shared; lifted from installer). |
| `src/python/lfs_plugins/video_to_splat/__init__.py` | `register()` / `unregister()`, hot-reload markers. |
| `src/python/lfs_plugins/video_to_splat/toolchain.py` | Find or download COLMAP. |
| `src/python/lfs_plugins/video_to_splat/probe.py` | Video facts → 360 decision, frame plan, scale. |
| `src/python/lfs_plugins/video_to_splat/sfm.py` | COLMAP command planning, model reading. |
| `src/python/lfs_plugins/video_to_splat/pipeline.py` | Stage orchestration, state file, resume, cancel. |
| `src/python/lfs_plugins/video_to_splat/project_bridge.py` | Create project, wait for import, start training (UI thread). |
| `src/python/lfs_plugins/video_to_splat/operators.py` | File-menu operator. |
| `src/python/lfs_plugins/import_panels.py` | New `video` source kind in `NewProjectPanel`. |
| `src/visualizer/gui/rmlui/resources/new_project_panel.rml` | Video options block + progress block. |
| `src/io/video/video_metadata.hpp/.cpp` | Add `width`, `height`, `spherical`. |
| `src/python/lfs/py_video.hpp/.cpp` | `lf.video.probe`, `lf.video.extract_frames`. |
| `src/python/lfs/py_ui.cpp` | `lf.ui.show_video_extractor(path)`. |
| `src/visualizer/input/input_controller.cpp` | Dropped video → New Project dialog. |
| `src/python/CMakeLists.txt` | Add `py_video.cpp`, link `lfs_video`. |
| `tests/test_video_metadata_probe.cpp` | C++ test for the probe additions. |
| `tests/python/test_process_runner.py`, `test_video_to_splat_*.py` | Python unit tests. |
| `scripts/video_to_splat_smoke.py` | Opt-in end-to-end check run inside the app. |
| `docs/docs/features/video-to-splat.md` | User documentation. |

---

### Task 1: Shared cancellable process runner

**Goal:** A reusable `run_cancellable_process` with streaming output and process-tree kill, used by both the installer and the new pipeline.

**Files:**
- Create: `src/python/lfs_plugins/process_runner.py`
- Modify: `src/python/lfs_plugins/installer.py:117-250` (make `_run_cancellable_process` a thin wrapper)
- Test: `tests/python/test_process_runner.py`
- Modify: `tests/CMakeLists.txt` (`LFS_PYTHON_FAST_TESTS` list)

**Acceptance Criteria:**
- [ ] `run_cancellable_process` streams each output line to `on_output` and returns a `CompletedProcess` with return code and last 100 lines.
- [ ] When `should_cancel()` returns true, the child is killed and `ProcessCancelled` is raised.
- [ ] `installer._run_cancellable_process` behaves exactly as before (raises `PluginLoadCancelled`).
- [ ] Existing installer tests still pass.

**Verify:** `python -m pytest tests/python/test_process_runner.py tests/python/test_plugin_installer*.py -q` → all pass.

**Steps:**

- [ ] **Step 1: Write the failing tests**

```python
# tests/python/test_process_runner.py
import sys
import time
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "python"))

from lfs_plugins.process_runner import ProcessCancelled, run_cancellable_process  # noqa: E402


def test_streams_output_lines_and_returns_code():
    lines: list[str] = []
    result = run_cancellable_process(
        [sys.executable, "-c", "print('one'); print('two'); raise SystemExit(3)"],
        on_output=lines.append,
    )
    assert lines == ["one", "two"]
    assert result.returncode == 3
    assert "two" in result.stdout


def test_cancel_kills_child_and_raises():
    started = time.monotonic()
    cancel = threading.Event()

    def on_output(_line: str) -> None:
        cancel.set()

    with pytest.raises(ProcessCancelled):
        run_cancellable_process(
            [sys.executable, "-c", "import time,sys; print('go', flush=True); time.sleep(30)"],
            on_output=on_output,
            should_cancel=cancel.is_set,
        )
    assert time.monotonic() - started < 10
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/python/test_process_runner.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'lfs_plugins.process_runner'`

- [ ] **Step 3: Create `process_runner.py`**

Move the bodies of `_wait_for_process`, `_terminate_process_tree`, `_process_group_popen_kwargs` and `_run_cancellable_process` from `installer.py` into the new module, renamed without the underscore and raising the new exception:

```python
# src/python/lfs_plugins/process_runner.py
"""Cancellable subprocess execution with streaming output.

Shared by the plugin installer and the video-to-splat pipeline.
"""
from __future__ import annotations

import os
import queue
import signal
import subprocess
import sys
import threading
from collections import deque
from typing import Callable, Optional

from _lfs_path_io import executable_kwargs

PROCESS_POLL_SECONDS = 0.05
PROCESS_TERMINATE_GRACE_SECONDS = 0.5
OUTPUT_TAIL_LINES = 100


class ProcessCancelled(Exception):
    """Raised when should_cancel() asked to stop a running child process."""


def is_windows() -> bool:
    return sys.platform.startswith("win")


def process_group_popen_kwargs() -> dict:
    if is_windows():
        return {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)}
    return {"start_new_session": True}


def wait_for_process(proc: subprocess.Popen, timeout: float) -> bool:
    try:
        proc.wait(timeout=timeout)
        return True
    except subprocess.TimeoutExpired:
        return False


def terminate_process_tree(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    if is_windows():
        try:
            subprocess.Popen(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).wait(timeout=5)
        except Exception:  # noqa: BLE001 - best effort, fall through to terminate()
            pass
        if not wait_for_process(proc, PROCESS_TERMINATE_GRACE_SECONDS):
            proc.terminate()
            if not wait_for_process(proc, PROCESS_TERMINATE_GRACE_SECONDS):
                proc.kill()
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    if not wait_for_process(proc, PROCESS_TERMINATE_GRACE_SECONDS):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        wait_for_process(proc, PROCESS_TERMINATE_GRACE_SECONDS)


def run_cancellable_process(
    cmd: list[str],
    *,
    env: Optional[dict] = None,
    cwd: Optional[str] = None,
    on_output: Optional[Callable[[str], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> subprocess.CompletedProcess:
    """Run cmd, streaming merged stdout/stderr lines to on_output.

    Does not raise on a nonzero exit code; callers check returncode.
    Raises ProcessCancelled when should_cancel() returns True.
    """
    popen_kwargs: dict = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
        "text": True,
        "bufsize": 1,
        "env": env,
        "cwd": cwd,
    }
    popen_kwargs.update(process_group_popen_kwargs())
    popen_kwargs.update(executable_kwargs(cmd))
    proc = subprocess.Popen(cmd, **popen_kwargs)

    lines: "queue.Queue[Optional[str]]" = queue.Queue()

    def read_output() -> None:
        assert proc.stdout is not None
        for raw in proc.stdout:
            lines.put(raw.rstrip("\r\n"))
        lines.put(None)

    reader = threading.Thread(target=read_output, name="process-runner-output", daemon=True)
    reader.start()

    tail: deque[str] = deque(maxlen=OUTPUT_TAIL_LINES)
    reader_done = False
    try:
        while True:
            if should_cancel is not None and should_cancel():
                terminate_process_tree(proc)
                raise ProcessCancelled(" ".join(cmd))
            try:
                item = lines.get(timeout=PROCESS_POLL_SECONDS)
            except queue.Empty:
                if proc.poll() is not None and reader_done:
                    break
                continue
            if item is None:
                reader_done = True
                if proc.poll() is not None:
                    break
                continue
            if item:
                tail.append(item)
                if on_output is not None:
                    on_output(item)
    finally:
        if proc.poll() is None and should_cancel is not None and should_cancel():
            terminate_process_tree(proc)
        reader.join(timeout=1.0)
    returncode = proc.wait()
    return subprocess.CompletedProcess(cmd, returncode, "\n".join(tail), "")
```

- [ ] **Step 4: Make the installer delegate**

In `src/python/lfs_plugins/installer.py`, replace the bodies of `_wait_for_process`, `_terminate_process_tree`, `_process_group_popen_kwargs` and `_run_cancellable_process` with:

```python
from .process_runner import (  # noqa: E402  (keep existing import block order)
    ProcessCancelled,
    process_group_popen_kwargs as _process_group_popen_kwargs,
    run_cancellable_process,
    terminate_process_tree as _terminate_process_tree,
    wait_for_process as _wait_for_process,
)


def _run_cancellable_process(
    cmd: list[str],
    *,
    env: Optional[dict] = None,
    on_output: Optional[Callable[[str], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> subprocess.CompletedProcess:
    try:
        return run_cancellable_process(cmd, env=env, on_output=on_output, should_cancel=should_cancel)
    except ProcessCancelled as exc:
        raise PluginLoadCancelled(str(exc)) from exc
```

Keep the constants `PROCESS_POLL_SECONDS` and `PROCESS_TERMINATE_GRACE_SECONDS` importable from `installer.py` if any test references them (grep `tests/python` for both names; if found, re-export them from `process_runner`).

- [ ] **Step 5: Register the test and run**

Add `test_process_runner.py` to `set(LFS_PYTHON_FAST_TESTS ...)` in `tests/CMakeLists.txt`.

Run: `python -m pytest tests/python/test_process_runner.py tests/python/ -k "installer or process_runner" -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/python/lfs_plugins/process_runner.py src/python/lfs_plugins/installer.py tests/python/test_process_runner.py tests/CMakeLists.txt
git commit -m "Extract a shared cancellable process runner from the plugin installer"
```

---

### Task 2: Video metadata probe reads width, height and spherical flag

**Goal:** `probeVideoFileMetadata` reports frame dimensions and whether the container carries spherical (360) metadata, so Python can decide 360 mode without opening a decoder.

**Files:**
- Modify: `src/io/video/video_metadata.hpp` (struct `VideoMetadata`)
- Modify: `src/io/video/video_metadata.cpp:115-160`
- Create: `tests/test_video_metadata_probe.cpp`
- Modify: `tests/CMakeLists.txt` (`TEST_FILES`)

**Acceptance Criteria:**
- [ ] `VideoMetadata` has `int width`, `int height`, `bool spherical`.
- [ ] Probing `tests/data/video/h264_420.mp4` yields width 64, height 48, spherical false.
- [ ] Existing video tests still pass.

**Verify:** `ctest --test-dir build/tests -R lichtfeld.fast --output-on-failure` (or `build/tests/lichtfeld_tests --gtest_filter=VideoMetadataProbe*`) → PASS.

**Steps:**

- [ ] **Step 1: Write the failing test**

```cpp
// tests/test_video_metadata_probe.cpp
/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "io/video/video_metadata.hpp"

#include <filesystem>
#include <gtest/gtest.h>
#include <string>

namespace {

    std::filesystem::path fixture(const char* name) {
        return std::filesystem::path(PROJECT_ROOT_PATH) / "tests" / "data" / "video" / name;
    }

    TEST(VideoMetadataProbe, ReportsDimensionsAndNoSphericalTag) {
        lfs::io::video::VideoMetadata metadata;
        std::string error;
        ASSERT_TRUE(lfs::io::video::probeVideoFileMetadata(fixture("h264_420.mp4"), metadata, error)) << error;
        EXPECT_EQ(metadata.width, 64);
        EXPECT_EQ(metadata.height, 48);
        EXPECT_FALSE(metadata.spherical);
        EXPECT_EQ(metadata.rotation, 0);
    }

    TEST(VideoMetadataProbe, MissingFileFails) {
        lfs::io::video::VideoMetadata metadata;
        std::string error;
        EXPECT_FALSE(lfs::io::video::probeVideoFileMetadata(fixture("does_not_exist.mp4"), metadata, error));
        EXPECT_FALSE(error.empty());
    }

} // namespace
```

Add `test_video_metadata_probe.cpp` next to `test_video_frame_extractor_naming.cpp` in `set(TEST_FILES ...)`.

- [ ] **Step 2: Build and run to verify failure**

Run: `cmake --build build/tests --target lichtfeld_tests && build/tests/lichtfeld_tests --gtest_filter=VideoMetadataProbe*`
Expected: compile error, `width` is not a member of `VideoMetadata`.

- [ ] **Step 3: Extend the struct and the probe**

In `video_metadata.hpp`, add to `struct VideoMetadata` after `rotation`:

```cpp
        int width = 0;          // coded frame width in pixels, 0 when unknown
        int height = 0;         // coded frame height in pixels, 0 when unknown
        bool spherical = false; // container carries spherical (360) projection metadata
```

In `video_metadata.cpp`, right after `const AVCodecParameters* const parameters = stream->codecpar;` add:

```cpp
        metadata.width = parameters->width;
        metadata.height = parameters->height;
        for (int i = 0; i < parameters->nb_coded_side_data; ++i) {
            if (parameters->coded_side_data[i].type == AV_PKT_DATA_SPHERICAL) {
                metadata.spherical = true;
                break;
            }
        }
```

`AV_PKT_DATA_SPHERICAL` comes from `<libavcodec/packet.h>`, already included transitively via the existing side-data loop. The mov demuxer fills this side data for both Spherical Video V1 (`uuid` XMP) and V2 (`sv3d`) boxes, which covers Insta360, GoPro and DJI stitched exports.

- [ ] **Step 4: Build and run**

Run: `cmake --build build/tests --target lichtfeld_tests && build/tests/lichtfeld_tests --gtest_filter=VideoMetadataProbe*:VideoFrameExtractor*`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/io/video/video_metadata.hpp src/io/video/video_metadata.cpp tests/test_video_metadata_probe.cpp tests/CMakeLists.txt
git commit -m "Report frame size and spherical metadata from the video probe"
```

---

### Task 3: `lf.video` binding (probe and extract_frames) and `lf.ui.show_video_extractor`

**Goal:** Python can probe a video and extract frames with progress and cancellation, and can open the native extractor on a path.

**Files:**
- Create: `src/python/lfs/py_video.hpp`, `src/python/lfs/py_video.cpp`
- Modify: `src/python/lfs/module.cpp` (~line 3387, submodule registration; include at ~line 21-49)
- Modify: `src/python/CMakeLists.txt:186-237` (sources) and `:257-275` (`LFS_PY_INTERNAL_LINK_LIBS`, add `lfs_video`)
- Modify: `src/python/lfs/py_ui.cpp` (add `show_video_extractor` next to `on_show_new_project_dialog`, ~line 3852)
- Create: `src/python/stubs/lichtfeld/video.pyi`
- Modify: `src/python/stubs/lichtfeld/ui/__init__.pyi` (add `show_video_extractor`)
- Test: `tests/python/test_video_binding_runtime.py` (marked `gpu`, `integration`; needs the built module)

**Acceptance Criteria:**
- [ ] `lf.video.probe("x.mp4")` returns a dict with `width`, `height`, `fps`, `duration`, `rotation`, `spherical`, `total_frames`, or raises `RuntimeError` with the probe error.
- [ ] `lf.video.extract_frames(...)` runs with the GIL released, calls `on_progress(current, total, discarded)` on the Python side, honours `should_cancel()`, and returns `{"outcome": "completed"|"cancelled"|"failed", "error": str}`.
- [ ] `lf.ui.show_video_extractor(path)` emits `cmd::ShowVideoExtractor`.
- [ ] `lfs_py` links `lfs_video`; the app builds and starts.

**Verify:** Build the app, then in the Python editor run `import lichtfeld as lf; print(lf.video.probe(r"tests/data/video/h264_420.mp4"))` → `{'width': 64, 'height': 48, ...}`. Then `python -m pytest tests/python/test_video_binding_runtime.py -q -m gpu` with `LFS_BUILD_DIR` set.

**Steps:**

- [ ] **Step 1: Write the failing runtime test**

```python
# tests/python/test_video_binding_runtime.py
"""Runtime checks for lf.video. Needs the built lichtfeld module (LFS_BUILD_DIR)."""
from pathlib import Path

import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.integration]

lf = pytest.importorskip("lichtfeld")
ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "data" / "video" / "vp9_timestamps.webm"  # 128x128, 185 frames, 30 fps


def test_probe_reports_dimensions():
    info = lf.video.probe(str(FIXTURE))
    assert info["width"] == 128
    assert info["height"] == 128
    assert info["spherical"] is False
    assert info["fps"] > 0


def test_extract_frames_writes_jpegs_and_reports_progress(tmp_path):
    calls = []
    result = lf.video.extract_frames(
        video_path=str(FIXTURE),
        output_dir=str(tmp_path),
        fps=5.0,
        jpg_quality=90,
        on_progress=lambda cur, total, discarded: calls.append((cur, total, discarded)),
    )
    assert result["outcome"] == "completed", result["error"]
    frames = sorted(tmp_path.glob("frame_*.jpg"))
    assert len(frames) >= 25
    assert calls and calls[-1][0] == calls[-1][1]


def test_extract_frames_cancels(tmp_path):
    result = lf.video.extract_frames(
        video_path=str(FIXTURE),
        output_dir=str(tmp_path),
        fps=30.0,
        should_cancel=lambda: True,
    )
    assert result["outcome"] == "cancelled"
```

- [ ] **Step 2: Add the binding**

```cpp
// src/python/lfs/py_video.hpp
/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include <nanobind/nanobind.h>
namespace nb = nanobind;
namespace lfs::python {
    void register_video(nb::module_& m);
}
```

```cpp
// src/python/lfs/py_video.cpp
/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "py_video.hpp"

#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "io/video/video_metadata.hpp"
#include "io/video_frame_extractor.hpp"

#include <nanobind/stl/string.h>
#include <string>

namespace lfs::python {

    namespace {

        struct PyProgress {
            nb::object callback;
            [[nodiscard]] bool valid() const { return callback.is_valid() && !callback.is_none(); }
            void operator()(const int current, const int total, const int discarded) const {
                if (!valid())
                    return;
                nb::gil_scoped_acquire gil;
                try {
                    callback(current, total, discarded);
                } catch (const std::exception& e) {
                    LOG_ERROR("lf.video progress callback error: {}", e.what());
                }
            }
        };

        struct PyCancel {
            nb::object callback;
            [[nodiscard]] bool valid() const { return callback.is_valid() && !callback.is_none(); }
            bool operator()() const {
                if (!valid())
                    return false;
                nb::gil_scoped_acquire gil;
                try {
                    return nb::cast<bool>(callback());
                } catch (const std::exception& e) {
                    LOG_ERROR("lf.video cancel callback error: {}", e.what());
                    return true;
                }
            }
        };

        const char* outcome_name(const lfs::io::ExtractionOutcome outcome) {
            switch (outcome) {
            case lfs::io::ExtractionOutcome::Completed: return "completed";
            case lfs::io::ExtractionOutcome::Cancelled: return "cancelled";
            case lfs::io::ExtractionOutcome::Failed: return "failed";
            }
            return "failed";
        }

    } // namespace

    void register_video(nb::module_& m) {
        m.def(
            "probe",
            [](const std::string& path) {
                lfs::io::video::VideoMetadata meta;
                std::string error;
                bool ok = false;
                {
                    nb::gil_scoped_release release;
                    ok = lfs::io::video::probeVideoFileMetadata(lfs::core::utf8_to_path(path), meta, error);
                }
                if (!ok)
                    throw std::runtime_error(error);
                nb::dict result;
                result["width"] = meta.width;
                result["height"] = meta.height;
                result["fps"] = meta.fps;
                result["duration"] = meta.duration;
                result["total_frames"] = static_cast<long long>(meta.total_frames);
                result["rotation"] = meta.rotation;
                result["spherical"] = meta.spherical;
                result["is_hdr"] = meta.is_hdr;
                return result;
            },
            nb::arg("path"),
            "Read container metadata without decoding: width, height, fps, duration, total_frames, rotation, "
            "spherical, is_hdr");

        m.def(
            "extract_frames",
            [](const std::string& video_path, const std::string& output_dir, const double fps,
               const int jpg_quality, const float scale, const bool sharpness_enabled,
               const double sharpness_threshold, const int rotation, const double start_time,
               const double end_time, const bool convert_hdr_to_sdr, const std::string& filename_pattern,
               nb::object on_progress, nb::object should_cancel) {
                lfs::io::VideoFrameExtractor::Params params;
                params.video_path = lfs::core::utf8_to_path(video_path);
                params.output_dir = lfs::core::utf8_to_path(output_dir);
                params.mode = lfs::io::ExtractionMode::FPS;
                params.fps = fps;
                params.format = lfs::io::ImageFormat::JPG;
                params.jpg_quality = jpg_quality;
                if (scale > 0.0f && scale < 1.0f) {
                    params.resolution_mode = lfs::io::ResolutionMode::Scale;
                    params.scale = scale;
                }
                params.sharpness.enabled = sharpness_enabled;
                params.sharpness.threshold = sharpness_threshold;
                params.sharpness.algorithm = lfs::io::SharpnessAlgorithm::COMBINED;
                params.rotation = rotation;
                params.start_time = start_time;
                params.end_time = end_time;
                params.convert_hdr_to_sdr = convert_hdr_to_sdr;
                params.filename_pattern = filename_pattern;
                params.generate_metadata = true;

                PyProgress progress{on_progress};
                PyCancel cancel{should_cancel};
                params.progress_callback = [progress](const int c, const int t, const int d) { progress(c, t, d); };
                params.cancel_requested = [cancel]() { return cancel(); };

                std::string error;
                lfs::io::ExtractionOutcome outcome;
                {
                    nb::gil_scoped_release release;
                    lfs::io::VideoFrameExtractor extractor;
                    extractor.extract(params, error);
                    outcome = extractor.lastOutcome();
                }
                nb::dict result;
                result["outcome"] = outcome_name(outcome);
                result["error"] = error;
                return result;
            },
            nb::arg("video_path"), nb::arg("output_dir"), nb::arg("fps") = 2.0, nb::arg("jpg_quality") = 95,
            nb::arg("scale") = 1.0f, nb::arg("sharpness_enabled") = true, nb::arg("sharpness_threshold") = 10.0,
            nb::arg("rotation") = 0, nb::arg("start_time") = 0.0, nb::arg("end_time") = -1.0,
            nb::arg("convert_hdr_to_sdr") = true, nb::arg("filename_pattern") = "frame_%06d",
            nb::arg("on_progress") = nb::none(), nb::arg("should_cancel") = nb::none(),
            "Extract JPEG frames at a fixed rate. Blocks (GIL released); returns {outcome, error}.");
    }

} // namespace lfs::python
```

In `module.cpp`, add `#include "py_video.hpp"` with the other `py_*.hpp` includes, and next to the `io` submodule registration:

```cpp
        auto video_module = m.def_submodule("video", "Video probing and frame extraction");
        lfs::python::register_video(video_module);
```

In `src/python/CMakeLists.txt`, add `lfs/py_video.cpp` to the `nanobind_add_module(lfs_py ...)` source list and add `lfs_video` to the RESCAN group in `LFS_PY_INTERNAL_LINK_LIBS` (after `lfs_io`).

- [ ] **Step 3: Add `show_video_extractor` to `py_ui.cpp`**

Next to `on_show_new_project_dialog`:

```cpp
        m.def(
            "show_video_extractor",
            [](const std::string& path) {
                lfs::core::events::cmd::ShowVideoExtractor{.video_path = lfs::core::utf8_to_path(path)}.emit();
            },
            nb::arg("path"), "Open the native video frame extractor on a video file");
```

- [ ] **Step 4: Stubs**

```python
# src/python/stubs/lichtfeld/video.pyi
from typing import Callable, Optional

def probe(path: str) -> dict:
    """Read container metadata without decoding: width, height, fps, duration, total_frames, rotation, spherical, is_hdr"""

def extract_frames(
    video_path: str,
    output_dir: str,
    fps: float = 2.0,
    jpg_quality: int = 95,
    scale: float = 1.0,
    sharpness_enabled: bool = True,
    sharpness_threshold: float = 10.0,
    rotation: int = 0,
    start_time: float = 0.0,
    end_time: float = -1.0,
    convert_hdr_to_sdr: bool = True,
    filename_pattern: str = "frame_%06d",
    on_progress: Optional[Callable[[int, int, int], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> dict:
    """Extract JPEG frames at a fixed rate. Blocks (GIL released); returns {outcome, error}."""
```

Add to `src/python/stubs/lichtfeld/ui/__init__.pyi`:

```python
def show_video_extractor(path: str) -> None:
    """Open the native video frame extractor on a video file"""
```

- [ ] **Step 5: Build, run the runtime test**

Run: `cmake --build build-windows-release --target lfs_py LichtFeld-Studio` (use the preset the repo documents in `docs/building_and_distribution.md`), then `python -m pytest tests/python/test_video_binding_runtime.py -q -m gpu`.
Expected: PASS (3 tests). If the module import fails, check `LFS_BUILD_DIR` points at the build folder.

- [ ] **Step 6: Commit**

```bash
git add src/python/lfs/py_video.hpp src/python/lfs/py_video.cpp src/python/lfs/module.cpp src/python/lfs/py_ui.cpp src/python/CMakeLists.txt src/python/stubs/lichtfeld/video.pyi src/python/stubs/lichtfeld/ui/__init__.pyi tests/python/test_video_binding_runtime.py
git commit -m "Expose video probing and frame extraction to Python as lf.video"
```

---

### Task 4: `probe.py` and `sfm.py` (pure planning and COLMAP model reading)

**Goal:** Decide 360 mode and frame plan from probe facts, build exact COLMAP command lines, and read registered-image and camera counts from a COLMAP binary model, all without running COLMAP.

**Files:**
- Create: `src/python/lfs_plugins/video_to_splat/__init__.py` (empty for now; `register()` comes in Task 8)
- Create: `src/python/lfs_plugins/video_to_splat/probe.py`
- Create: `src/python/lfs_plugins/video_to_splat/sfm.py`
- Test: `tests/python/test_video_to_splat_planning.py`
- Modify: `tests/CMakeLists.txt` (`LFS_PYTHON_FAST_TESTS`)

**Acceptance Criteria:**
- [ ] `decide_mode(probe, override)` returns `"equirectangular"` for 2:1 aspect within 2 percent or `spherical=True`, else `"perspective"`; override `"on"`/`"off"` wins.
- [ ] `PRESETS` has `fast`, `balanced`, `high` with the spec's fps / max width / overlap.
- [ ] `plan_commands(...)` returns three argv lists with the exact options listed below.
- [ ] `read_registered_image_count(images_bin)` and `read_camera_count(cameras_bin)` parse the COLMAP binary headers.
- [ ] `plan_frame_scale(width, max_width)` returns `min(1.0, max_width / width)`, or 1.0 when `max_width` is 0.

**Verify:** `python -m pytest tests/python/test_video_to_splat_planning.py -q` → PASS.

**Steps:**

- [ ] **Step 1: Write the failing tests**

```python
# tests/python/test_video_to_splat_planning.py
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "python"))

from lfs_plugins.video_to_splat import probe, sfm  # noqa: E402


def test_decide_mode_from_aspect_and_metadata():
    assert probe.decide_mode({"width": 5760, "height": 2880, "spherical": False}, "auto") == "equirectangular"
    assert probe.decide_mode({"width": 5700, "height": 2880, "spherical": False}, "auto") == "equirectangular"
    assert probe.decide_mode({"width": 3840, "height": 2160, "spherical": False}, "auto") == "perspective"
    assert probe.decide_mode({"width": 3840, "height": 2160, "spherical": True}, "auto") == "equirectangular"
    assert probe.decide_mode({"width": 5760, "height": 2880, "spherical": True}, "off") == "perspective"
    assert probe.decide_mode({"width": 1920, "height": 1080, "spherical": False}, "on") == "equirectangular"


def test_presets_and_scale():
    assert probe.PRESETS["balanced"] == probe.Preset(fps=2.0, max_width=4096, overlap=15)
    assert probe.PRESETS["fast"] == probe.Preset(fps=1.0, max_width=2880, overlap=10)
    assert probe.PRESETS["high"] == probe.Preset(fps=3.0, max_width=0, overlap=25)
    assert probe.plan_frame_scale(5760, 2880) == 0.5
    assert probe.plan_frame_scale(1920, 4096) == 1.0
    assert probe.plan_frame_scale(5760, 0) == 1.0


def test_plan_commands_equirectangular(tmp_path):
    cmds = sfm.plan_commands(
        colmap=Path("C:/tools/colmap.exe"), dataset=tmp_path, mode="equirectangular", overlap=15, use_gpu=True
    )
    feat, match, mapper = cmds
    assert feat[0] == "C:/tools/colmap.exe" or feat[0].endswith("colmap.exe")
    assert feat[1] == "feature_extractor"
    assert "--ImageReader.camera_model" in feat and feat[feat.index("--ImageReader.camera_model") + 1] == "EQUIRECTANGULAR"
    assert feat[feat.index("--ImageReader.single_camera") + 1] == "1"
    assert feat[feat.index("--FeatureExtraction.use_gpu") + 1] == "1"
    assert match[1] == "sequential_matcher"
    assert match[match.index("--SequentialMatching.overlap") + 1] == "15"
    assert match[match.index("--SequentialMatching.loop_detection") + 1] == "0"
    assert mapper[1] == "global_mapper"
    assert mapper[mapper.index("--output_path") + 1] == str(tmp_path / "sparse")
    for cmd in cmds:
        assert cmd[cmd.index("--database_path") + 1] == str(tmp_path / "database.db")
        assert "--log_target" in cmd and cmd[cmd.index("--log_target") + 1] == "stderr"


def test_plan_commands_perspective_cpu(tmp_path):
    feat, match, _ = sfm.plan_commands(colmap=Path("colmap"), dataset=tmp_path, mode="perspective", overlap=10, use_gpu=False)
    assert feat[feat.index("--ImageReader.camera_model") + 1] == "OPENCV"
    assert feat[feat.index("--FeatureExtraction.use_gpu") + 1] == "0"
    assert match[match.index("--FeatureMatching.use_gpu") + 1] == "0"


def test_read_binary_model_counts(tmp_path):
    images = tmp_path / "images.bin"
    images.write_bytes(struct.pack("<Q", 42) + b"\x00" * 16)
    cameras = tmp_path / "cameras.bin"
    cameras.write_bytes(struct.pack("<Q", 1) + struct.pack("<iiQQ", 1, 17, 5760, 2880))
    assert sfm.read_registered_image_count(images) == 42
    assert sfm.read_camera_count(cameras) == 1
    assert sfm.read_registered_image_count(tmp_path / "missing.bin") == 0


def test_find_largest_model(tmp_path):
    for idx, n in ((0, 10), (1, 80), (2, 5)):
        d = tmp_path / "sparse" / str(idx)
        d.mkdir(parents=True)
        (d / "images.bin").write_bytes(struct.pack("<Q", n))
    best = sfm.find_largest_model(tmp_path / "sparse")
    assert best == (tmp_path / "sparse" / "1", 80)
    assert sfm.find_largest_model(tmp_path / "nope") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/python/test_video_to_splat_planning.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'lfs_plugins.video_to_splat'`

- [ ] **Step 3: Implement `probe.py`**

```python
# src/python/lfs_plugins/video_to_splat/__init__.py
"""Video to splat pipeline: frames -> COLMAP poses -> project -> training."""
```

```python
# src/python/lfs_plugins/video_to_splat/probe.py
"""Decisions derived from video metadata: 360 mode, presets, frame scale."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

Mode = Literal["equirectangular", "perspective"]
ModeOverride = Literal["auto", "on", "off"]

ASPECT_TOLERANCE = 0.02


@dataclass(frozen=True)
class Preset:
    fps: float
    max_width: int  # 0 = keep source width
    overlap: int    # sequential matcher window


PRESETS: dict[str, Preset] = {
    "fast": Preset(fps=1.0, max_width=2880, overlap=10),
    "balanced": Preset(fps=2.0, max_width=4096, overlap=15),
    "high": Preset(fps=3.0, max_width=0, overlap=25),
}
DEFAULT_PRESET = "balanced"


def looks_equirectangular(width: int, height: int) -> bool:
    if width <= 0 or height <= 0:
        return False
    return abs(width / height - 2.0) <= 2.0 * ASPECT_TOLERANCE


def decide_mode(info: Mapping[str, object], override: ModeOverride = "auto") -> Mode:
    if override == "on":
        return "equirectangular"
    if override == "off":
        return "perspective"
    if bool(info.get("spherical", False)):
        return "equirectangular"
    if looks_equirectangular(int(info.get("width", 0)), int(info.get("height", 0))):
        return "equirectangular"
    return "perspective"


def plan_frame_scale(width: int, max_width: int) -> float:
    if max_width <= 0 or width <= 0 or width <= max_width:
        return 1.0
    return max_width / width
```

- [ ] **Step 4: Implement `sfm.py`**

```python
# src/python/lfs_plugins/video_to_splat/sfm.py
"""COLMAP command planning and binary model reading. No subprocess calls here."""
from __future__ import annotations

import struct
from pathlib import Path
from typing import Optional

from .probe import Mode

CAMERA_MODELS: dict[Mode, str] = {
    "equirectangular": "EQUIRECTANGULAR",  # COLMAP >= 4.1
    "perspective": "OPENCV",
}

IMAGES_DIR = "images"
DATABASE_NAME = "database.db"
SPARSE_DIR = "sparse"


def dataset_paths(dataset: Path) -> tuple[Path, Path, Path]:
    return dataset / IMAGES_DIR, dataset / DATABASE_NAME, dataset / SPARSE_DIR


def plan_commands(*, colmap: Path, dataset: Path, mode: Mode, overlap: int, use_gpu: bool,
                  max_image_size: int = 3200) -> list[list[str]]:
    """Return argv lists for feature_extractor, sequential_matcher, global_mapper in order."""
    images, database, sparse = dataset_paths(dataset)
    gpu = "1" if use_gpu else "0"
    exe = str(colmap)
    common = ["--log_target", "stderr"]
    feature = [
        exe, "feature_extractor",
        "--database_path", str(database),
        "--image_path", str(images),
        "--ImageReader.camera_model", CAMERA_MODELS[mode],
        "--ImageReader.single_camera", "1",
        "--FeatureExtraction.use_gpu", gpu,
        "--FeatureExtraction.max_image_size", str(max_image_size),
        *common,
    ]
    match = [
        exe, "sequential_matcher",
        "--database_path", str(database),
        "--SequentialMatching.overlap", str(overlap),
        "--SequentialMatching.quadratic_overlap", "1",
        "--SequentialMatching.loop_detection", "0",
        "--FeatureMatching.use_gpu", gpu,
        *common,
    ]
    mapper = [
        exe, "global_mapper",
        "--database_path", str(database),
        "--image_path", str(images),
        "--output_path", str(sparse),
        *common,
    ]
    return [feature, match, mapper]


def read_registered_image_count(images_bin: Path) -> int:
    """First uint64 of images.bin is the number of registered images."""
    try:
        with open(images_bin, "rb") as handle:
            header = handle.read(8)
    except OSError:
        return 0
    if len(header) < 8:
        return 0
    return struct.unpack("<Q", header)[0]


def read_camera_count(cameras_bin: Path) -> int:
    try:
        with open(cameras_bin, "rb") as handle:
            header = handle.read(8)
    except OSError:
        return 0
    if len(header) < 8:
        return 0
    return struct.unpack("<Q", header)[0]


def find_largest_model(sparse: Path) -> Optional[tuple[Path, int]]:
    """global_mapper writes one numbered folder per connected component; pick the biggest."""
    if not sparse.is_dir():
        return None
    best: Optional[tuple[Path, int]] = None
    for child in sorted(sparse.iterdir()):
        if not child.is_dir():
            continue
        count = read_registered_image_count(child / "images.bin")
        if count <= 0:
            continue
        if best is None or count > best[1]:
            best = (child, count)
    return best
```

- [ ] **Step 5: Verify the sequential matcher option names**

The option names `--SequentialMatching.overlap`, `--SequentialMatching.quadratic_overlap` and `--SequentialMatching.loop_detection` are not on the docs page. With a COLMAP 4.2.1 binary available (download the Windows zip once, or use Task 5's toolchain after it lands), run:

```bash
colmap sequential_matcher -h | grep SequentialMatching
```

Expected: the three names appear. If spelled differently, fix `sfm.py` and the test to match the real output.

- [ ] **Step 6: Run tests, register, commit**

Add `test_video_to_splat_planning.py` to `LFS_PYTHON_FAST_TESTS`.

Run: `python -m pytest tests/python/test_video_to_splat_planning.py -q`
Expected: PASS (6 tests).

```bash
git add src/python/lfs_plugins/video_to_splat/__init__.py src/python/lfs_plugins/video_to_splat/probe.py src/python/lfs_plugins/video_to_splat/sfm.py tests/python/test_video_to_splat_planning.py tests/CMakeLists.txt
git commit -m "Plan COLMAP commands and 360 mode for the video-to-splat pipeline"
```

---

### Task 5: `toolchain.py` (find or download COLMAP)

**Goal:** Resolve a COLMAP 4.1+ binary: configured path, `PATH`, or the cached Windows download, with a consent hook and SHA-256 verification.

**Files:**
- Create: `src/python/lfs_plugins/video_to_splat/toolchain.py`
- Test: `tests/python/test_video_to_splat_toolchain.py`
- Modify: `tests/CMakeLists.txt` (`LFS_PYTHON_FAST_TESTS`)

**Acceptance Criteria:**
- [ ] `resolve_colmap(...)` returns `Resolved(path, version)` from, in order: explicit `configured_path`, cached download dir, `PATH`.
- [ ] Versions below 4.1 are rejected with `ToolchainError` naming the found version.
- [ ] On Windows with nothing found, `resolve_colmap` calls `request_consent()`; on `False` it raises `ToolchainError` with the manual-install message; on `True` it downloads via injected `download(url, dest, on_progress)`, verifies SHA-256, extracts, and returns the binary.
- [ ] On Linux with nothing found it raises `ToolchainError` saying to install COLMAP 4.1+ or set the path.
- [ ] All I/O is injectable; tests never touch the network.

**Verify:** `python -m pytest tests/python/test_video_to_splat_toolchain.py -q` → PASS.

**Steps:**

- [ ] **Step 1: Write the failing tests**

```python
# tests/python/test_video_to_splat_toolchain.py
import hashlib
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "python"))

from lfs_plugins.video_to_splat import toolchain  # noqa: E402


def fake_version(text: str):
    def run(cmd: list[str]) -> str:
        return text
    return run


def test_parse_version():
    assert toolchain.parse_version("COLMAP 4.2.1 -- Structure-from-Motion and Multi-View Stereo") == (4, 2, 1)
    assert toolchain.parse_version("COLMAP 3.11.0") == (3, 11, 0)
    assert toolchain.parse_version("garbage") is None


def test_configured_path_wins(tmp_path):
    exe = tmp_path / "colmap.exe"
    exe.write_bytes(b"")
    resolved = toolchain.resolve_colmap(
        configured_path=str(exe), cache_dir=tmp_path / "cache", platform="win32",
        which=lambda name: None, run_version=fake_version("COLMAP 4.2.1"),
        request_consent=lambda: pytest.fail("no consent needed"), download=None,
    )
    assert resolved.path == exe
    assert resolved.version == (4, 2, 1)


def test_old_version_rejected(tmp_path):
    exe = tmp_path / "colmap"
    exe.write_bytes(b"")
    with pytest.raises(toolchain.ToolchainError) as err:
        toolchain.resolve_colmap(
            configured_path=str(exe), cache_dir=tmp_path, platform="linux",
            which=lambda name: None, run_version=fake_version("COLMAP 3.11.0"),
            request_consent=lambda: False, download=None,
        )
    assert "3.11.0" in str(err.value)


def test_linux_without_binary_explains(tmp_path):
    with pytest.raises(toolchain.ToolchainError) as err:
        toolchain.resolve_colmap(
            configured_path="", cache_dir=tmp_path, platform="linux",
            which=lambda name: None, run_version=fake_version(""),
            request_consent=lambda: pytest.fail("never on linux"), download=None,
        )
    assert "4.1" in str(err.value)


def _make_zip(path: Path, exe_name: str) -> str:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(f"colmap-x64-windows-cuda/{exe_name}", b"MZ fake")
        zf.writestr("colmap-x64-windows-cuda/readme.txt", b"hi")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_windows_download_flow(tmp_path, monkeypatch):
    archive = tmp_path / "stage.zip"
    digest = _make_zip(archive, "colmap.exe")
    monkeypatch.setattr(toolchain, "COLMAP_WINDOWS_SHA256", digest)
    consents = []

    def download(url: str, dest: Path, on_progress) -> None:
        assert url == toolchain.COLMAP_WINDOWS_URL
        dest.write_bytes(archive.read_bytes())
        on_progress("done")

    resolved = toolchain.resolve_colmap(
        configured_path="", cache_dir=tmp_path / "cache", platform="win32",
        which=lambda name: None, run_version=fake_version("COLMAP 4.2.1"),
        request_consent=lambda: consents.append(True) or True, download=download,
    )
    assert consents == [True]
    assert resolved.path == tmp_path / "cache" / "tools" / "colmap-4.2.1" / "colmap.exe"
    assert resolved.path.is_file()
    # second call finds the cached copy without asking again
    again = toolchain.resolve_colmap(
        configured_path="", cache_dir=tmp_path / "cache", platform="win32",
        which=lambda name: None, run_version=fake_version("COLMAP 4.2.1"),
        request_consent=lambda: pytest.fail("cached"), download=None,
    )
    assert again.path == resolved.path


def test_windows_consent_refused(tmp_path):
    with pytest.raises(toolchain.ToolchainError) as err:
        toolchain.resolve_colmap(
            configured_path="", cache_dir=tmp_path, platform="win32",
            which=lambda name: None, run_version=fake_version(""),
            request_consent=lambda: False, download=None,
        )
    assert "download" in str(err.value).lower()


def test_windows_hash_mismatch(tmp_path, monkeypatch):
    archive = tmp_path / "stage.zip"
    _make_zip(archive, "colmap.exe")
    monkeypatch.setattr(toolchain, "COLMAP_WINDOWS_SHA256", "00" * 32)

    def download(url, dest, on_progress):
        dest.write_bytes(archive.read_bytes())

    with pytest.raises(toolchain.ToolchainError) as err:
        toolchain.resolve_colmap(
            configured_path="", cache_dir=tmp_path / "cache", platform="win32",
            which=lambda name: None, run_version=fake_version("COLMAP 4.2.1"),
            request_consent=lambda: True, download=download,
        )
    assert "SHA-256" in str(err.value)
    assert not (tmp_path / "cache" / "tools" / "colmap-4.2.1").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/python/test_video_to_splat_toolchain.py -q`
Expected: FAIL, `cannot import name 'toolchain'`

- [ ] **Step 3: Implement `toolchain.py`**

```python
# src/python/lfs_plugins/video_to_splat/toolchain.py
"""Locate a COLMAP 4.1+ binary, downloading the official Windows build on consent."""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

COLMAP_VERSION = "4.2.1"
COLMAP_WINDOWS_URL = (
    f"https://github.com/colmap/colmap/releases/download/{COLMAP_VERSION}/colmap-x64-windows-cuda.zip"
)
COLMAP_WINDOWS_SHA256 = "e9c5cbd84c2ea986d2e970a2473fc2d2e6b34a2cdcf5d3df2765c319a63af881"
COLMAP_WINDOWS_SIZE_MB = 395
MIN_VERSION = (4, 1, 0)

Version = tuple[int, int, int]
_VERSION_RE = re.compile(r"COLMAP\s+(\d+)\.(\d+)\.(\d+)")


class ToolchainError(Exception):
    """COLMAP is unavailable. str(error) is a user-facing message."""


@dataclass(frozen=True)
class Resolved:
    path: Path
    version: Version


def parse_version(text: str) -> Optional[Version]:
    match = _VERSION_RE.search(text or "")
    if not match:
        return None
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def default_run_version(cmd: list[str]) -> str:
    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return (completed.stdout or "") + "\n" + (completed.stderr or "")


def version_string(version: Version) -> str:
    return ".".join(str(part) for part in version)


def _check_binary(path: Path, run_version: Callable[[list[str]], str]) -> Resolved:
    text = run_version([str(path), "help"])
    version = parse_version(text)
    if version is None:
        raise ToolchainError(f"'{path}' did not report a COLMAP version. Is it a COLMAP executable?")
    if version < MIN_VERSION:
        raise ToolchainError(
            f"COLMAP {version_string(version)} at '{path}' is too old. "
            f"Version {version_string(MIN_VERSION)} or later is needed for 360 cameras."
        )
    return Resolved(path=path, version=version)


def cached_binary_path(cache_dir: Path, platform: str) -> Path:
    exe = "colmap.exe" if platform.startswith("win") else "colmap"
    return cache_dir / "tools" / f"colmap-{COLMAP_VERSION}" / exe


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_flat(archive: Path, dest: Path) -> None:
    """Extract a zip, dropping its single top-level folder, rejecting path escapes."""
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
        prefix = os.path.commonpath(names) if len(names) > 1 else ""
        if prefix and not all(n.startswith(prefix + "/") for n in names):
            prefix = ""
        for name in names:
            relative = name[len(prefix) + 1:] if prefix else name
            target = (dest / relative).resolve()
            if dest.resolve() not in target.parents and target != dest.resolve():
                raise ToolchainError(f"Archive entry escapes destination: {name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(name) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)


def _download_windows_build(cache_dir: Path, download: Callable[[str, Path, Callable[[str], None]], None],
                            on_progress: Callable[[str], None]) -> Path:
    target_dir = cached_binary_path(cache_dir, "win32").parent
    staging_dir = Path(tempfile.mkdtemp(prefix="colmap-dl-", dir=str(cache_dir / "tools")))
    archive = staging_dir / "colmap.zip"
    try:
        download(COLMAP_WINDOWS_URL, archive, on_progress)
        actual = _sha256_of(archive)
        if actual.lower() != COLMAP_WINDOWS_SHA256.lower():
            raise ToolchainError(
                "The downloaded COLMAP archive failed its SHA-256 check and was discarded. "
                "Try again, or install COLMAP manually and set its path in preferences."
            )
        if target_dir.exists():
            shutil.rmtree(target_dir)
        _extract_flat(archive, target_dir)
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)
    exe = cached_binary_path(cache_dir, "win32")
    if not exe.is_file():
        shutil.rmtree(target_dir, ignore_errors=True)
        raise ToolchainError("The COLMAP archive did not contain colmap.exe.")
    return exe


def resolve_colmap(
    *,
    configured_path: str,
    cache_dir: Path,
    platform: str = sys.platform,
    which: Callable[[str], Optional[str]] = shutil.which,
    run_version: Callable[[list[str]], str] = default_run_version,
    request_consent: Callable[[], bool],
    download: Optional[Callable[[str, Path, Callable[[str], None]], None]],
    on_progress: Callable[[str], None] = lambda _msg: None,
) -> Resolved:
    """Find COLMAP 4.1+. Order: configured path, cached download, PATH, then download (Windows only)."""
    if configured_path:
        path = Path(configured_path)
        if not path.is_file():
            raise ToolchainError(f"The configured COLMAP path does not exist: {path}")
        return _check_binary(path, run_version)

    cached = cached_binary_path(cache_dir, platform)
    if cached.is_file():
        return _check_binary(cached, run_version)

    found = which("colmap")
    if found:
        return _check_binary(Path(found), run_version)

    if not platform.startswith("win"):
        raise ToolchainError(
            f"COLMAP {version_string(MIN_VERSION)} or later was not found on PATH. "
            "Install COLMAP or set its path in the Video to Splat preferences."
        )

    if not request_consent():
        raise ToolchainError(
            "COLMAP is needed to solve camera poses and the download was declined. "
            "Allow the download, or install COLMAP 4.1+ and set its path in preferences."
        )
    if download is None:
        raise ToolchainError("No downloader is available for COLMAP.")
    (cache_dir / "tools").mkdir(parents=True, exist_ok=True)
    exe = _download_windows_build(cache_dir, download, on_progress)
    return _check_binary(exe, run_version)
```

Note: `colmap help` prints the banner `COLMAP x.y.z -- Structure-from-Motion and Multi-View Stereo` on stdout for 3.x and 4.x. Confirm once against the real binary (Task 4 step 5 downloads one); if the banner is on stderr only, `default_run_version` already merges both.

- [ ] **Step 4: Run tests, register, commit**

Add `test_video_to_splat_toolchain.py` to `LFS_PYTHON_FAST_TESTS`.

Run: `python -m pytest tests/python/test_video_to_splat_toolchain.py -q`
Expected: PASS (7 tests).

```bash
git add src/python/lfs_plugins/video_to_splat/toolchain.py tests/python/test_video_to_splat_toolchain.py tests/CMakeLists.txt
git commit -m "Resolve or download COLMAP for the video-to-splat pipeline"
```

---

### Task 6: `pipeline.py` orchestrator with state file, resume and cancel

**Goal:** Run the stages in order on a worker thread, persist `pipeline.json` after each, resume on rerun, propagate cancel, and report progress through one callback. All side effects are injected so the tests run without the app or COLMAP.

**Files:**
- Create: `src/python/lfs_plugins/video_to_splat/pipeline.py`
- Test: `tests/python/test_video_to_splat_pipeline.py`
- Modify: `tests/CMakeLists.txt` (`LFS_PYTHON_FAST_TESTS`)

**Acceptance Criteria:**
- [ ] Stages run in order `toolchain, extract, features, match, map, create_project, train`.
- [ ] `pipeline.json` records each completed stage with its parameter fingerprint; a rerun with the same fingerprint skips completed stages and reruns from the first incomplete one.
- [ ] Fewer than 20 frames after extraction stops with `PipelineError` naming the count.
- [ ] After `map`, the largest model is selected; if registered ratio < 0.6 or registered < 20, the pipeline calls `confirm_weak(ratio, registered, total)` and stops when it returns `False`.
- [ ] Cancel at any stage raises `PipelineCancelled`; the state file keeps the last completed stage.
- [ ] Free-space check warns through `on_progress` when under 10 GB (injected `free_bytes`).

**Verify:** `python -m pytest tests/python/test_video_to_splat_pipeline.py -q` → PASS.

**Steps:**

- [ ] **Step 1: Write the failing tests**

```python
# tests/python/test_video_to_splat_pipeline.py
import json
import struct
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "python"))

from lfs_plugins.video_to_splat import pipeline, probe, toolchain  # noqa: E402


class Fakes:
    """Injected side effects. Records calls, lets tests script outcomes."""

    def __init__(self, project: Path, frames: int = 50, registered: int = 45, colmap_rc: int = 0):
        self.project = project
        self.frames = frames
        self.registered = registered
        self.colmap_rc = colmap_rc
        self.calls: list[str] = []
        self.progress: list[tuple] = []
        self.cancel = threading.Event()
        self.confirmed_weak = True

    def resolve_colmap(self, on_progress):
        self.calls.append("toolchain")
        return toolchain.Resolved(Path("colmap.exe"), (4, 2, 1))

    def probe(self, path):
        return {"width": 5760, "height": 2880, "fps": 30.0, "duration": 10.0, "rotation": 0, "spherical": True}

    def extract(self, params, on_progress, should_cancel):
        self.calls.append("extract")
        images = params.dataset / "images"
        images.mkdir(parents=True, exist_ok=True)
        for i in range(self.frames):
            (images / f"frame_{i:06d}.jpg").write_bytes(b"\xff\xd8")
        return {"outcome": "completed", "error": ""}

    def run(self, cmd, on_output, should_cancel):
        self.calls.append(cmd[1])
        if cmd[1] == "global_mapper":
            model = Path(cmd[cmd.index("--output_path") + 1]) / "0"
            model.mkdir(parents=True, exist_ok=True)
            (model / "images.bin").write_bytes(struct.pack("<Q", self.registered))
            (model / "cameras.bin").write_bytes(struct.pack("<Q", 1))
            (model / "points3D.bin").write_bytes(struct.pack("<Q", 0))
        on_output(f"{cmd[1]} running")
        from subprocess import CompletedProcess
        return CompletedProcess(cmd, self.colmap_rc, "", "")

    def create_project(self, project, dataset, images_folder):
        self.calls.append("create_project")

    def train(self):
        self.calls.append("train")

    def confirm_weak(self, ratio, registered, total):
        self.calls.append("confirm_weak")
        return self.confirmed_weak

    def hooks(self) -> pipeline.Hooks:
        return pipeline.Hooks(
            resolve_colmap=self.resolve_colmap, probe=self.probe, extract=self.extract, run=self.run,
            create_project=self.create_project, train=self.train, confirm_weak=self.confirm_weak,
            free_bytes=lambda path: 100 * 1024 ** 3,
        )


def make_request(project: Path, video: Path) -> pipeline.Request:
    return pipeline.Request(video=video, project=project, mode_override="auto", preset="balanced")


def test_full_run_order_and_state(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"0")
    project = tmp_path / "proj"
    fakes = Fakes(project)
    pipeline.run(make_request(project, video), fakes.hooks(), on_progress=lambda *a: fakes.progress.append(a),
                 should_cancel=fakes.cancel.is_set)
    assert fakes.calls == ["toolchain", "extract", "feature_extractor", "sequential_matcher", "global_mapper",
                           "create_project", "train"]
    state = json.loads((project / "pipeline.json").read_text())
    assert [s["name"] for s in state["completed"]] == pipeline.STAGE_NAMES
    assert state["mode"] == "equirectangular"
    assert state["registered"] == 45 and state["frames"] == 50
    assert fakes.progress[0][0] == "toolchain"


def test_rerun_skips_completed_stages(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"0")
    project = tmp_path / "proj"
    first = Fakes(project)
    pipeline.run(make_request(project, video), first.hooks(), on_progress=lambda *a: None, should_cancel=lambda: False)
    # Truncate state so only toolchain..match are complete, as if map crashed.
    state = json.loads((project / "pipeline.json").read_text())
    state["completed"] = state["completed"][:4]
    (project / "pipeline.json").write_text(json.dumps(state))
    second = Fakes(project)
    pipeline.run(make_request(project, video), second.hooks(), on_progress=lambda *a: None, should_cancel=lambda: False)
    assert second.calls == ["global_mapper", "create_project", "train"]


def test_changed_preset_invalidates_resume(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"0")
    project = tmp_path / "proj"
    first = Fakes(project)
    pipeline.run(make_request(project, video), first.hooks(), on_progress=lambda *a: None, should_cancel=lambda: False)
    second = Fakes(project)
    request = pipeline.Request(video=video, project=project, mode_override="auto", preset="high")
    pipeline.run(request, second.hooks(), on_progress=lambda *a: None, should_cancel=lambda: False)
    assert second.calls[0] == "toolchain" and "extract" in second.calls


def test_too_few_frames_stops(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"0")
    project = tmp_path / "proj"
    fakes = Fakes(project, frames=7)
    with pytest.raises(pipeline.PipelineError) as err:
        pipeline.run(make_request(project, video), fakes.hooks(), on_progress=lambda *a: None, should_cancel=lambda: False)
    assert "7" in str(err.value)
    assert "feature_extractor" not in fakes.calls


def test_weak_reconstruction_asks_and_can_stop(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"0")
    project = tmp_path / "proj"
    fakes = Fakes(project, frames=50, registered=20)  # 40 percent
    fakes.confirmed_weak = False
    with pytest.raises(pipeline.PipelineStopped):
        pipeline.run(make_request(project, video), fakes.hooks(), on_progress=lambda *a: None, should_cancel=lambda: False)
    assert "confirm_weak" in fakes.calls and "create_project" not in fakes.calls


def test_colmap_failure_raises_with_tail(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"0")
    project = tmp_path / "proj"
    fakes = Fakes(project, colmap_rc=1)
    with pytest.raises(pipeline.PipelineError) as err:
        pipeline.run(make_request(project, video), fakes.hooks(), on_progress=lambda *a: None, should_cancel=lambda: False)
    assert "feature_extractor" in str(err.value)


def test_cancel_before_map_keeps_state(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"0")
    project = tmp_path / "proj"
    fakes = Fakes(project)
    original_run = fakes.run

    def run(cmd, on_output, should_cancel):
        if cmd[1] == "global_mapper":
            fakes.cancel.set()
            raise pipeline.ProcessCancelled("killed")
        return original_run(cmd, on_output, should_cancel)

    fakes.run = run
    with pytest.raises(pipeline.PipelineCancelled):
        pipeline.run(make_request(project, video), fakes.hooks(), on_progress=lambda *a: None,
                     should_cancel=fakes.cancel.is_set)
    state = json.loads((project / "pipeline.json").read_text())
    assert [s["name"] for s in state["completed"]] == ["toolchain", "extract", "features", "match"]


def test_low_disk_space_warns(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"0")
    project = tmp_path / "proj"
    fakes = Fakes(project)
    hooks = fakes.hooks()
    hooks = pipeline.Hooks(**{**hooks.__dict__, "free_bytes": lambda p: 2 * 1024 ** 3})
    messages = []
    pipeline.run(make_request(project, video), hooks, on_progress=lambda stage, frac, msg, tail: messages.append(msg),
                 should_cancel=lambda: False)
    assert any("GB" in m for m in messages)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/python/test_video_to_splat_pipeline.py -q`
Expected: FAIL, `cannot import name 'pipeline'`

- [ ] **Step 3: Implement `pipeline.py`**

```python
# src/python/lfs_plugins/video_to_splat/pipeline.py
"""Stage orchestration for video -> frames -> COLMAP -> project -> training.

Everything that touches the app, the filesystem beyond the project folder, or a
subprocess is injected through Hooks so the orchestration is unit-testable.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from subprocess import CompletedProcess
from typing import Callable, Literal, Optional

from ..process_runner import ProcessCancelled
from . import probe as probe_mod
from . import sfm
from .toolchain import Resolved

STAGE_NAMES = ["toolchain", "extract", "features", "match", "map", "create_project", "train"]
STATE_FILE = "pipeline.json"
DATASET_DIR = "dataset"
MIN_FRAMES = 20
MIN_REGISTERED = 20
MIN_REGISTERED_RATIO = 0.6
LOW_DISK_BYTES = 10 * 1024 ** 3
STATE_VERSION = 1

ProgressFn = Callable[[str, float, str, str], None]  # stage, fraction 0..1, message, log tail


class PipelineError(Exception):
    """Unrecoverable failure; str(error) is user-facing."""


class PipelineCancelled(Exception):
    """The user cancelled."""


class PipelineStopped(Exception):
    """The user chose to stop at a decision point (weak reconstruction)."""


@dataclass(frozen=True)
class Request:
    video: Path
    project: Path
    mode_override: probe_mod.ModeOverride = "auto"
    preset: str = probe_mod.DEFAULT_PRESET
    use_gpu: bool = True

    def fingerprint(self) -> str:
        raw = json.dumps(
            {"video": str(self.video), "mtime": _mtime(self.video), "mode": self.mode_override,
             "preset": self.preset, "gpu": self.use_gpu},
            sort_keys=True,
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    @property
    def dataset(self) -> Path:
        return self.project / DATASET_DIR


@dataclass(frozen=True)
class ExtractParams:
    video: Path
    dataset: Path
    fps: float
    scale: float
    rotation: int


@dataclass
class Hooks:
    resolve_colmap: Callable[[Callable[[str], None]], Resolved]
    probe: Callable[[str], dict]
    extract: Callable[[ExtractParams, Callable[[int, int, int], None], Callable[[], bool]], dict]
    run: Callable[[list[str], Callable[[str], None], Callable[[], bool]], CompletedProcess]
    create_project: Callable[[Path, Path, str], None]
    train: Callable[[], None]
    confirm_weak: Callable[[float, int, int], bool]
    free_bytes: Callable[[Path], int] = lambda path: shutil.disk_usage(path).free


@dataclass
class State:
    fingerprint: str
    completed: list[dict] = field(default_factory=list)
    mode: str = ""
    frames: int = 0
    registered: int = 0
    model_dir: str = ""
    version: int = STATE_VERSION

    def done(self) -> set[str]:
        return {entry["name"] for entry in self.completed}

    def mark(self, name: str, **facts) -> None:
        self.completed.append({"name": name, "at": time.time(), **facts})

    def save(self, project: Path) -> None:
        project.mkdir(parents=True, exist_ok=True)
        tmp = project / (STATE_FILE + ".tmp")
        tmp.write_text(json.dumps(self.__dict__, indent=2), encoding="utf-8")
        os.replace(tmp, project / STATE_FILE)

    @classmethod
    def load(cls, project: Path, fingerprint: str) -> "State":
        path = project / STATE_FILE
        if not path.is_file():
            return cls(fingerprint=fingerprint)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return cls(fingerprint=fingerprint)
        if raw.get("fingerprint") != fingerprint or raw.get("version") != STATE_VERSION:
            return cls(fingerprint=fingerprint)
        return cls(fingerprint=fingerprint, completed=list(raw.get("completed", [])), mode=raw.get("mode", ""),
                   frames=int(raw.get("frames", 0)), registered=int(raw.get("registered", 0)),
                   model_dir=raw.get("model_dir", ""))


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _count_frames(images: Path) -> int:
    if not images.is_dir():
        return 0
    return sum(1 for p in images.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))


def run(request: Request, hooks: Hooks, *, on_progress: ProgressFn, should_cancel: Callable[[], bool]) -> State:
    """Run all stages, resuming from pipeline.json when the request fingerprint matches."""
    state = State.load(request.project, request.fingerprint())
    done = state.done()
    preset = probe_mod.PRESETS[request.preset]
    images, database, sparse = sfm.dataset_paths(request.dataset)
    tail: list[str] = []

    def log(line: str) -> None:
        tail.append(line)
        del tail[:-40]

    def report(stage: str, fraction: float, message: str) -> None:
        on_progress(stage, fraction, message, "\n".join(tail))

    def check_cancel() -> None:
        if should_cancel():
            raise PipelineCancelled()

    # toolchain
    colmap: Optional[Resolved] = None
    if "toolchain" not in done or not all(s in done for s in ("features", "match", "map")):
        report("toolchain", 0.0, "Locating COLMAP")
        colmap = hooks.resolve_colmap(lambda msg: report("toolchain", 0.0, msg))
        if "toolchain" not in done:
            state.mark("toolchain", colmap=str(colmap.path), version=list(colmap.version))
            state.save(request.project)
    check_cancel()

    info = hooks.probe(str(request.video))
    mode = probe_mod.decide_mode(info, request.mode_override)
    state.mode = mode

    # extract
    if "extract" not in done:
        free = hooks.free_bytes(request.project if request.project.exists() else request.project.parent)
        if free < LOW_DISK_BYTES:
            report("extract", 0.0, f"Only {free / 1024 ** 3:.1f} GB free on the destination; frames need several GB")
        if images.exists():
            shutil.rmtree(images)
        images.mkdir(parents=True, exist_ok=True)
        params = ExtractParams(video=request.video, dataset=request.dataset, fps=preset.fps,
                               scale=probe_mod.plan_frame_scale(int(info.get("width", 0)), preset.max_width),
                               rotation=0)
        report("extract", 0.0, "Extracting frames")

        def extract_progress(current: int, total: int, discarded: int) -> None:
            fraction = current / total if total else 0.0
            report("extract", fraction, f"Frame {current}/{total}, {discarded} discarded as blurry")

        result = hooks.extract(params, extract_progress, should_cancel)
        if result.get("outcome") == "cancelled":
            raise PipelineCancelled()
        if result.get("outcome") != "completed":
            raise PipelineError(f"Frame extraction failed: {result.get('error', 'unknown error')}")
        frames = _count_frames(images)
        if frames < MIN_FRAMES:
            raise PipelineError(
                f"Only {frames} usable frames were extracted; at least {MIN_FRAMES} are needed. "
                "Raise the frame rate or lower the sharpness threshold."
            )
        state.frames = frames
        state.mark("extract", frames=frames)
        state.save(request.project)
    else:
        state.frames = state.frames or _count_frames(images)
    check_cancel()

    # COLMAP stages
    assert colmap is not None or all(s in done for s in ("features", "match", "map"))
    commands = sfm.plan_commands(colmap=colmap.path if colmap else Path("colmap"), dataset=request.dataset,
                                 mode=mode, overlap=preset.overlap, use_gpu=request.use_gpu)
    for stage, cmd in zip(("features", "match", "map"), commands):
        if stage in done:
            continue
        if stage == "features" and database.exists():
            database.unlink()
        if stage == "map" and sparse.exists():
            shutil.rmtree(sparse)
        if stage == "map":
            sparse.mkdir(parents=True, exist_ok=True)
        report(stage, 0.0, f"Running COLMAP {cmd[1]}")
        try:
            completed = hooks.run(cmd, lambda line: (log(line), report(stage, 0.0, f"COLMAP {cmd[1]}")), should_cancel)
        except ProcessCancelled:
            raise PipelineCancelled()
        if completed.returncode != 0:
            raise PipelineError(f"COLMAP {cmd[1]} failed with exit code {completed.returncode}.\n" + "\n".join(tail[-10:]))
        state.mark(stage)
        state.save(request.project)
        check_cancel()

    # choose model, judge quality
    best = sfm.find_largest_model(sparse)
    if best is None:
        raise PipelineError("COLMAP produced no reconstruction. The footage may lack overlap or texture.")
    model_dir, registered = best
    state.registered = registered
    state.model_dir = str(model_dir)
    ratio = registered / state.frames if state.frames else 0.0
    if "create_project" not in done and (ratio < MIN_REGISTERED_RATIO or registered < MIN_REGISTERED):
        if not hooks.confirm_weak(ratio, registered, state.frames):
            state.save(request.project)
            raise PipelineStopped()

    # LichtFeld expects sparse/0; move the chosen model there when it is not already.
    canonical = sparse / "0"
    if model_dir != canonical:
        if canonical.exists():
            shutil.rmtree(canonical)
        shutil.move(str(model_dir), str(canonical))
        state.model_dir = str(canonical)

    if "create_project" not in done:
        report("create_project", 0.0, "Creating project")
        hooks.create_project(request.project, request.dataset, sfm.IMAGES_DIR)
        state.mark("create_project")
        state.save(request.project)
    check_cancel()

    if "train" not in done:
        report("train", 0.0, "Starting training")
        hooks.train()
        state.mark("train")
        state.save(request.project)
    report("train", 1.0, f"Done: {registered}/{state.frames} frames registered")
    return state
```

- [ ] **Step 4: Run tests, register, commit**

Add `test_video_to_splat_pipeline.py` to `LFS_PYTHON_FAST_TESTS`.

Run: `python -m pytest tests/python/test_video_to_splat_pipeline.py -q`
Expected: PASS (8 tests).

```bash
git add src/python/lfs_plugins/video_to_splat/pipeline.py tests/python/test_video_to_splat_pipeline.py tests/CMakeLists.txt
git commit -m "Orchestrate video-to-splat stages with resumable state"
```

---

### Task 7: `project_bridge.py` (app-side hooks: consent, extraction, project creation, training)

**Goal:** The concrete `Hooks` that talk to LichtFeld: consent dialog, `lf.video.extract_frames`, COLMAP via the process runner, project creation and import wait, training start. Marshals UI calls onto the UI thread and blocks the worker until they answer.

**Files:**
- Create: `src/python/lfs_plugins/video_to_splat/project_bridge.py`
- Test: `tests/python/test_video_to_splat_bridge.py`
- Modify: `tests/CMakeLists.txt` (`LFS_PYTHON_FAST_TESTS`)

**Acceptance Criteria:**
- [ ] `ask_on_ui_thread(fn)` runs `fn` through `lf.ui.schedule_on_ui_thread` and returns its result to the worker.
- [ ] `confirm_download()` shows `lf.ui.confirm_dialog` with the size and URL and returns True only for the accept label.
- [ ] `create_project_and_wait(project, dataset, images_folder, timeout)` calls `lf.project_create(...)`, then `lf.load_file(path=dataset, is_dataset=True, images_folder=..., discard_changes=True)`, then polls `lf.ui.get_import_state()` until `active` is False, raising `PipelineError` when `success` is False.
- [ ] `start_training()` calls `lf.start_training()` on the UI thread.
- [ ] `build_hooks(settings)` wires everything, reading `colmap_path` from `lf.plugins.settings("video_to_splat")` and the cache dir from `LFS_RESOLVED_CACHE_DIR`.

**Verify:** `python -m pytest tests/python/test_video_to_splat_bridge.py -q` → PASS.

**Steps:**

- [ ] **Step 1: Write the failing tests**

```python
# tests/python/test_video_to_splat_bridge.py
import sys
from importlib import import_module
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "python"))


@pytest.fixture
def lf_stub(monkeypatch):
    for name in list(sys.modules):
        if name == "lfs_plugins" or name.startswith("lfs_plugins."):
            sys.modules.pop(name, None)
    state = SimpleNamespace(scheduled=[], dialogs=[], created=[], loaded=[], import_states=[], started=0)

    def schedule(cb):
        state.scheduled.append(cb)
        cb()  # tests run the UI thread inline

    def confirm_dialog(title, message, buttons, callback, style="info"):
        state.dialogs.append((title, message, buttons, style))
        callback(buttons[0])

    stub = ModuleType("lichtfeld")
    stub.ui = SimpleNamespace(
        schedule_on_ui_thread=schedule,
        confirm_dialog=confirm_dialog,
        message_dialog=lambda *a, **k: None,
        tr=lambda key: key,
        get_import_state=lambda: state.import_states.pop(0),
    )
    stub.project_create = lambda path, **kw: state.created.append((path, kw)) or True
    stub.project_create_pending = lambda: False
    stub.load_file = lambda **kw: state.loaded.append(kw)
    stub.start_training = lambda: setattr(state, "started", state.started + 1)
    stub.is_training_active = lambda: False
    stub.plugins = SimpleNamespace(settings=lambda name: SimpleNamespace(get=lambda k, d="": ""))
    stub.video = SimpleNamespace(probe=lambda p: {}, extract_frames=lambda **kw: {"outcome": "completed", "error": ""})
    monkeypatch.setitem(sys.modules, "lichtfeld", stub)
    monkeypatch.setenv("LFS_RESOLVED_CACHE_DIR", "C:/fake-cache")
    module = import_module("lfs_plugins.video_to_splat.project_bridge")
    return module, state


def test_ask_on_ui_thread_returns_value(lf_stub):
    bridge, state = lf_stub
    assert bridge.ask_on_ui_thread(lambda: 42) == 42
    assert len(state.scheduled) == 1


def test_confirm_download_uses_first_button_as_accept(lf_stub):
    bridge, state = lf_stub
    assert bridge.confirm_download() is True
    title, message, buttons, style = state.dialogs[0]
    assert buttons[0] == "video_to_splat.download_accept"
    assert "395" in message


def test_create_project_and_wait_polls_import(lf_stub):
    bridge, state = lf_stub
    state.import_states.extend([{"active": True}, {"active": True}, {"active": False, "success": True}])
    bridge.create_project_and_wait(Path("P:/proj"), Path("P:/proj/dataset"), "images", timeout=5, poll_seconds=0)
    assert state.created[0][0] == "P:/proj"
    assert state.loaded[0]["is_dataset"] is True and state.loaded[0]["images_folder"] == "images"
    assert state.import_states == []


def test_create_project_and_wait_raises_on_failed_import(lf_stub):
    bridge, state = lf_stub
    state.import_states.extend([{"active": False, "success": False, "error": "bad sparse"}])
    with pytest.raises(bridge.PipelineError) as err:
        bridge.create_project_and_wait(Path("P:/proj"), Path("P:/proj/dataset"), "images", timeout=5, poll_seconds=0)
    assert "bad sparse" in str(err.value)


def test_start_training_runs_on_ui_thread(lf_stub):
    bridge, state = lf_stub
    bridge.start_training()
    assert state.started == 1


def test_build_hooks_reads_settings_and_cache(lf_stub):
    bridge, state = lf_stub
    hooks = bridge.build_hooks()
    assert hooks.probe is not None and hooks.train is not None
    assert bridge.cache_dir() == Path("C:/fake-cache")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/python/test_video_to_splat_bridge.py -q`
Expected: FAIL, `No module named 'lfs_plugins.video_to_splat.project_bridge'`

- [ ] **Step 3: Implement `project_bridge.py`**

```python
# src/python/lfs_plugins/video_to_splat/project_bridge.py
"""Concrete pipeline hooks that talk to the running LichtFeld app."""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Callable, Optional, TypeVar

import lichtfeld as lf

from ..environment import value as env_value
from ..process_runner import run_cancellable_process
from . import toolchain
from .pipeline import ExtractParams, Hooks, PipelineError
from .toolchain import Resolved

PLUGIN_NAME = "video_to_splat"
SETTING_COLMAP_PATH = "colmap_path"
T = TypeVar("T")


def tr(key: str) -> str:
    return lf.ui.tr(key)


def cache_dir() -> Path:
    resolved = env_value("LFS_RESOLVED_CACHE_DIR")
    if resolved:
        return Path(resolved)
    return Path.home() / ".lichtfeld" / "cache"


def configured_colmap_path() -> str:
    try:
        return str(lf.plugins.settings(PLUGIN_NAME).get(SETTING_COLMAP_PATH, "") or "")
    except Exception:  # noqa: BLE001 - settings unavailable in some test harnesses
        return ""


def ask_on_ui_thread(fn: Callable[[], T]) -> T:
    """Run fn on the UI thread and block the calling worker until it returns."""
    done = threading.Event()
    box: dict = {}

    def runner() -> None:
        try:
            box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 - re-raised on the worker
            box["error"] = exc
        finally:
            done.set()

    lf.ui.schedule_on_ui_thread(runner)
    done.wait()
    if "error" in box:
        raise box["error"]
    return box["value"]


def _dialog_answer(title: str, message: str, buttons: list[str], style: str = "info") -> str:
    """Show a confirm dialog from the worker thread; return the clicked label."""
    answered = threading.Event()
    chosen: dict = {}

    def show() -> None:
        def on_result(label: str) -> None:
            chosen["label"] = label
            answered.set()
        lf.ui.confirm_dialog(title, message, buttons, on_result, style)

    lf.ui.schedule_on_ui_thread(show)
    answered.wait()
    return chosen.get("label", "")


def confirm_download() -> bool:
    accept = tr("video_to_splat.download_accept")
    decline = tr("video_to_splat.download_decline")
    message = tr("video_to_splat.download_message").format(
        size_mb=toolchain.COLMAP_WINDOWS_SIZE_MB, url=toolchain.COLMAP_WINDOWS_URL, version=toolchain.COLMAP_VERSION)
    return _dialog_answer(tr("video_to_splat.download_title"), message, [accept, decline]) == accept


def confirm_weak(ratio: float, registered: int, total: int) -> bool:
    go = tr("video_to_splat.weak_train_anyway")
    stop = tr("video_to_splat.weak_stop")
    message = tr("video_to_splat.weak_message").format(percent=int(round(ratio * 100)), registered=registered, total=total)
    return _dialog_answer(tr("video_to_splat.weak_title"), message, [go, stop], "warning") == go


def download_file(url: str, dest: Path, on_progress: Callable[[str], None]) -> None:
    from ..http import urlopen  # same HTTP stack as the plugin installer

    dest.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=60) as response, open(dest, "wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        received = 0
        last_report = 0.0
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            received += len(chunk)
            now = time.monotonic()
            if now - last_report > 0.5:
                last_report = now
                if total:
                    on_progress(tr("video_to_splat.downloading_percent").format(percent=int(100 * received / total)))
                else:
                    on_progress(tr("video_to_splat.downloading_mb").format(mb=received // (1 << 20)))


def resolve_colmap(on_progress: Callable[[str], None]) -> Resolved:
    try:
        return toolchain.resolve_colmap(
            configured_path=configured_colmap_path(), cache_dir=cache_dir(),
            request_consent=confirm_download, download=download_file, on_progress=on_progress)
    except toolchain.ToolchainError as exc:
        raise PipelineError(str(exc)) from exc


def extract(params: ExtractParams, on_progress: Callable[[int, int, int], None], should_cancel: Callable[[], bool]) -> dict:
    return lf.video.extract_frames(
        video_path=str(params.video), output_dir=str(params.dataset / "images"), fps=params.fps, scale=params.scale,
        rotation=params.rotation, sharpness_enabled=True, sharpness_threshold=10.0,
        on_progress=on_progress, should_cancel=should_cancel)


def run_process(cmd: list[str], on_output: Callable[[str], None], should_cancel: Callable[[], bool]):
    return run_cancellable_process(cmd, on_output=on_output, should_cancel=should_cancel)


def create_project_and_wait(project: Path, dataset: Path, images_folder: str, *, timeout: float = 3600.0,
                            poll_seconds: float = 0.25) -> None:
    def create() -> None:
        created = lf.project_create(str(project), discard_changes=True, stop_training=False, overwrite=True)
        if created is False and not lf.project_create_pending():
            raise PipelineError(tr("video_to_splat.project_create_failed"))
        lf.load_file(path=str(dataset), is_dataset=True, images_folder=images_folder, discard_changes=True)

    ask_on_ui_thread(create)
    deadline = time.monotonic() + timeout
    while True:
        import_state = ask_on_ui_thread(lf.ui.get_import_state)
        if not import_state.get("active", False):
            if import_state.get("success", True):
                return
            raise PipelineError(tr("video_to_splat.import_failed").format(error=import_state.get("error", "")))
        if time.monotonic() > deadline:
            raise PipelineError(tr("video_to_splat.import_timeout"))
        time.sleep(poll_seconds)


def start_training() -> None:
    ask_on_ui_thread(lf.start_training)


def build_hooks() -> Hooks:
    return Hooks(
        resolve_colmap=resolve_colmap,
        probe=lf.video.probe,
        extract=extract,
        run=run_process,
        create_project=lambda project, dataset, images_folder: create_project_and_wait(project, dataset, images_folder),
        train=start_training,
        confirm_weak=confirm_weak,
    )
```

Check that `lfs_plugins/http.py` exposes `urlopen(url, timeout=...)` (installer.py imports it as `from .http import urlopen`); if its signature differs, match it.

- [ ] **Step 4: Run tests, register, commit**

Add `test_video_to_splat_bridge.py` to `LFS_PYTHON_FAST_TESTS`.

Run: `python -m pytest tests/python/test_video_to_splat_bridge.py -q`
Expected: PASS (6 tests).

```bash
git add src/python/lfs_plugins/video_to_splat/project_bridge.py tests/python/test_video_to_splat_bridge.py tests/CMakeLists.txt
git commit -m "Bridge the video-to-splat pipeline to the running app"
```

---

### Task 8: New Project dialog `video` source kind, menu operator, registration, locale keys

**Goal:** Picking or dropping a video in the New Project dialog shows the video options, "Create" runs the pipeline with a staged progress view, and an "Extract frames only" button opens the native extractor. File menu gets "New Project from Video".

**Files:**
- Modify: `src/python/lfs_plugins/import_panels.py` (`NewProjectPanel`)
- Modify: `src/visualizer/gui/rmlui/resources/new_project_panel.rml`
- Create: `src/python/lfs_plugins/video_to_splat/operators.py`
- Modify: `src/python/lfs_plugins/video_to_splat/__init__.py` (`register`/`unregister`)
- Modify: `src/python/lfs_plugins/panels.py` (`_build_builtin_panel_steps`, add a step)
- Modify: `src/python/lfs_plugins/file_menu.py` (menu item)
- Modify: all 10 files in `src/visualizer/gui/resources/locales/` (`de, en, es, fr, it, ja, ko, nl, pl, zh`)
- Test: `tests/python/test_new_project_panel_video.py`
- Modify: `tests/CMakeLists.txt` (`LFS_PYTHON_FAST_TESTS`)

**Acceptance Criteria:**
- [ ] A source path with a supported video extension sets `_source_kind == "video"`, `can_create` is True, `source_is_video` is True in the data model.
- [ ] Data model exposes `video_mode` (auto/on/off), `video_preset`, `pipeline_stage`, `pipeline_fraction`, `pipeline_message`, `pipeline_running`, and events `extract_only`, `cancel_pipeline`.
- [ ] `do_create` with a video source starts `pipeline.run` on a daemon thread with `project_bridge.build_hooks()`, and progress reaches the model through `schedule_on_ui_thread`.
- [ ] `extract_only` calls `lf.ui.show_video_extractor(source_path)` and closes the dialog.
- [ ] `PipelineError` shows `lf.ui.message_dialog(..., "error")`; `PipelineCancelled` and `PipelineStopped` just reset the view.
- [ ] `tests/python/test_localization_contracts.py` passes (all keys in all locales, no hard-coded strings).

**Verify:** `python -m pytest tests/python/test_new_project_panel_video.py tests/python/test_new_project_panel.py tests/python/test_import_dialog_panels.py tests/python/test_localization_contracts.py -q` → PASS.

**Steps:**

- [ ] **Step 1: Write the failing panel test**

Model it on `tests/python/test_new_project_panel.py` (reuse its `lf_stub` fixture construction by copying the stub-building block; add `open_video_file_dialog`, `show_video_extractor`, `message_dialog` and a `video` namespace to the stub). Then:

```python
# tests/python/test_new_project_panel_video.py
# (fixture `panel_env` built as in test_new_project_panel.py, returning (module, panel, state))
import threading


def test_video_extension_sets_video_kind(panel_env):
    module, panel, state = panel_env
    panel.show("D:/clips/room.mp4")
    assert panel._source_kind == "video"
    assert panel._can_create() is True
    assert panel._video_mode == "auto" and panel._video_preset == "balanced"


def test_extract_only_opens_native_extractor(panel_env):
    module, panel, state = panel_env
    panel.show("D:/clips/room.mp4")
    panel._on_extract_only()
    assert state.video_extractor_paths == ["D:/clips/room.mp4"]
    assert state.panel_enabled[-1] == (panel.id, False)


def test_do_create_with_video_starts_pipeline(panel_env, monkeypatch):
    module, panel, state = panel_env
    panel.show("D:/clips/room.mp4")
    panel._set_name("room")
    started = threading.Event()
    captured = {}

    def fake_run(request, hooks, *, on_progress, should_cancel):
        captured["request"] = request
        on_progress("extract", 0.5, "half", "")
        started.set()
        return None

    monkeypatch.setattr(module.v2s_pipeline, "run", fake_run)
    monkeypatch.setattr(module.v2s_bridge, "build_hooks", lambda: object())
    panel._on_do_create()
    assert started.wait(2)
    for cb in list(state.scheduled):
        cb()
    assert captured["request"].video.as_posix() == "D:/clips/room.mp4"
    assert captured["request"].project.name == "room"
    assert panel._pipeline_stage == "extract" and panel._pipeline_fraction == 0.5


def test_pipeline_error_shows_message(panel_env, monkeypatch):
    module, panel, state = panel_env
    panel.show("D:/clips/room.mp4")
    panel._set_name("room")

    def fake_run(request, hooks, *, on_progress, should_cancel):
        raise module.v2s_pipeline.PipelineError("COLMAP exploded")

    monkeypatch.setattr(module.v2s_pipeline, "run", fake_run)
    monkeypatch.setattr(module.v2s_bridge, "build_hooks", lambda: object())
    panel._on_do_create()
    panel._pipeline_thread.join(2)
    for cb in list(state.scheduled):
        cb()
    assert any("COLMAP exploded" in m for (_t, m, _s) in state.messages)
    assert panel._pipeline_running is False
```

In the stub: `state.video_extractor_paths = []`, `lf_stub.ui.show_video_extractor = state.video_extractor_paths.append`, `lf_stub.ui.message_dialog = lambda t, m, s="info", cb=None: state.messages.append((t, m, s))`, `lf_stub.ui.open_video_file_dialog = lambda: ""`, `lf_stub.ui.get_project_location = lambda: "D:/projects"`, and `schedule_on_ui_thread` appends to `state.scheduled` (do not run inline here, tests drain it).

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/python/test_new_project_panel_video.py -q`
Expected: FAIL (`_source_kind` is `"checking"`, no `_on_extract_only`).

- [ ] **Step 3: Extend `NewProjectPanel` in `import_panels.py`**

Add imports near the top:

```python
from .video_to_splat import pipeline as v2s_pipeline
from .video_to_splat import project_bridge as v2s_bridge
from .video_to_splat.probe import DEFAULT_PRESET, PRESETS

_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}  # mirrors kSupportedVideoExtensions
```

In `__init__`, add state:

```python
        self._video_mode = "auto"
        self._video_preset = DEFAULT_PRESET
        self._pipeline_running = False
        self._pipeline_stage = ""
        self._pipeline_fraction = 0.0
        self._pipeline_message = ""
        self._pipeline_cancel = threading.Event()
        self._pipeline_thread: Optional[threading.Thread] = None
```

Add a static helper and use it in `_apply_source_path` before the `checking` branch:

```python
    @staticmethod
    def _is_video_path(path: str) -> bool:
        return Path(path).suffix.lower() in _VIDEO_EXTENSIONS
```

```python
        elif self._is_video_path(next_value):
            self._source_kind = "video"
```

Add `"video"` to the accepted set in `_can_create()` and to the check at the top of `_commit`. Also refuse while `self._pipeline_running` (return False from `_can_create`).

In `on_bind_model`, add:

```python
        model.bind_func("source_is_video", lambda: self._source_kind == "video")
        model.bind("video_mode", lambda: self._video_mode, self._set_video_mode)
        model.bind("video_preset", lambda: self._video_preset, self._set_video_preset)
        model.bind_func("pipeline_running", lambda: self._pipeline_running)
        model.bind_func("pipeline_stage", lambda: lf.ui.tr(f"video_to_splat.stage.{self._pipeline_stage}") if self._pipeline_stage else "")
        model.bind_func("pipeline_fraction", lambda: self._pipeline_fraction)
        model.bind_func("pipeline_percent", lambda: int(self._pipeline_fraction * 100))
        model.bind_func("pipeline_message", lambda: self._pipeline_message)
        model.bind_event("extract_only", self._on_extract_only)
        model.bind_event("cancel_pipeline", self._on_cancel_pipeline)
```

Setters and handlers:

```python
    def _set_video_mode(self, value) -> None:
        value = str(value)
        if value in ("auto", "on", "off"):
            self._video_mode = value

    def _set_video_preset(self, value) -> None:
        value = str(value)
        if value in PRESETS:
            self._video_preset = value

    def _on_extract_only(self, _handle=None, _ev=None, _args=None):
        path = self._source_path
        lf.ui.set_panel_enabled(self.id, False)
        lf.ui.show_video_extractor(path)

    def _on_cancel_pipeline(self, _handle=None, _ev=None, _args=None):
        self._pipeline_cancel.set()
        self._pipeline_message = lf.ui.tr("video_to_splat.cancelling")
        self._dirty_model("pipeline_message")
```

In `_commit`, add a branch before the `dataset` branch:

```python
        if source_kind == "video":
            self._start_video_pipeline(Path(source_path), target)
            return
```

and the pipeline driver:

```python
    def _start_video_pipeline(self, video: Path, project: Path) -> None:
        if lf.is_training_active():
            lf.ui.message_dialog(lf.ui.tr("video_to_splat.title"), lf.ui.tr("video_to_splat.training_busy"), "warning")
            return
        request = v2s_pipeline.Request(video=video, project=project, mode_override=self._video_mode,
                                       preset=self._video_preset)
        self._pipeline_cancel.clear()
        self._pipeline_running = True
        self._pipeline_stage, self._pipeline_fraction, self._pipeline_message = "toolchain", 0.0, ""
        self._dirty_model()

        def on_progress(stage: str, fraction: float, message: str, _tail: str) -> None:
            def apply() -> None:
                self._pipeline_stage, self._pipeline_fraction, self._pipeline_message = stage, fraction, message
                self._dirty_model("pipeline_stage", "pipeline_fraction", "pipeline_percent", "pipeline_message")
            lf.ui.schedule_on_ui_thread(apply)

        def finish(error: Optional[str]) -> None:
            def apply() -> None:
                self._pipeline_running = False
                self._dirty_model()
                if error:
                    lf.ui.message_dialog(lf.ui.tr("video_to_splat.title"), error, "error")
                else:
                    lf.ui.set_panel_enabled(self.id, False)
            lf.ui.schedule_on_ui_thread(apply)

        def worker() -> None:
            try:
                v2s_pipeline.run(request, v2s_bridge.build_hooks(), on_progress=on_progress,
                                 should_cancel=self._pipeline_cancel.is_set)
            except v2s_pipeline.PipelineCancelled:
                finish(None)
            except v2s_pipeline.PipelineStopped:
                finish(None)
            except v2s_pipeline.PipelineError as exc:
                finish(str(exc))
            except Exception as exc:  # noqa: BLE001 - surface anything unexpected to the user
                finish(f"{type(exc).__name__}: {exc}")
            else:
                finish(None)

        self._pipeline_thread = threading.Thread(target=worker, daemon=True, name="VideoToSplatPipeline")
        self._pipeline_thread.start()
```

Make sure `_commit` does **not** call `lf.ui.set_panel_enabled(self.id, False)` before the video branch (the dialog stays open to show progress); move that call into the `dataset`/`splat` branches.

- [ ] **Step 4: RML**

In `new_project_panel.rml`, after the source row hint, add:

```html
      <div class="impdlg-section" data-if="source_is_video">
        <div class="impdlg-row">
          <span class="impdlg-label">@tr:video_to_splat.mode</span>
          <select class="impdlg-select" data-value="video_mode">
            <option value="auto">@tr:video_to_splat.mode_auto</option>
            <option value="on">@tr:video_to_splat.mode_on</option>
            <option value="off">@tr:video_to_splat.mode_off</option>
          </select>
        </div>
        <div class="impdlg-row">
          <span class="impdlg-label">@tr:video_to_splat.preset</span>
          <select class="impdlg-select" data-value="video_preset">
            <option value="fast">@tr:video_to_splat.preset_fast</option>
            <option value="balanced">@tr:video_to_splat.preset_balanced</option>
            <option value="high">@tr:video_to_splat.preset_high</option>
          </select>
        </div>
        <div class="impdlg-row" data-if="pipeline_running">
          <span class="impdlg-label">{{pipeline_stage}}</span>
          <progress class="impdlg-progress" data-attr-value="pipeline_fraction" max="1"></progress>
          <span class="impdlg-status">{{pipeline_percent}}% {{pipeline_message}}</span>
          <button type="button" class="btn btn--secondary impdlg-action-btn" data-event-click="cancel_pipeline">@tr:common.cancel</button>
        </div>
        <div class="impdlg-row" data-if="!pipeline_running">
          <button type="button" class="btn btn--secondary impdlg-action-btn" data-event-click="extract_only">@tr:video_to_splat.extract_only</button>
          <span class="impdlg-hint">@tr:video_to_splat.hint</span>
        </div>
      </div>
```

Check `import_dialog.rcss` for the class names actually used by the existing rows (`impdlg-row`, `impdlg-label`, `impdlg-input`, button classes) and reuse them; add a minimal `.impdlg-progress { width: 100%; height: 8dp; }` rule if `<progress>` is unstyled. The existing dataset blocks use `data-if="source_is_dataset"`; mirror that structure.

- [ ] **Step 5: Operator, registration, menu**

```python
# src/python/lfs_plugins/video_to_splat/operators.py
import lichtfeld as lf

from ..types import Operator


class NewProjectFromVideoOperator(Operator):
    label = "menu.file.new_project_from_video"
    description = "Create a project from a video: extract frames, solve cameras, train"

    def execute(self, context) -> set:
        path = lf.ui.open_video_file_dialog()
        if not path:
            return {"CANCELLED"}
        from ..import_panels import open_new_project_panel
        open_new_project_panel(path)
        return {"FINISHED"}
```

```python
# src/python/lfs_plugins/video_to_splat/__init__.py
"""Video to splat pipeline: frames -> COLMAP poses -> project -> training."""
import lichtfeld as lf

__lfs_operator_classes__ = ["NewProjectFromVideoOperator"]


def register() -> None:
    from .operators import NewProjectFromVideoOperator
    lf.register_class(NewProjectFromVideoOperator)


def unregister() -> None:
    from .operators import NewProjectFromVideoOperator
    lf.unregister_class(NewProjectFromVideoOperator)
```

Guard the module-level `import lichtfeld as lf` so the pure modules stay importable in tests without the app: move the import inside `register`/`unregister` instead, and keep `__init__.py` free of `lf` at import time.

In `panels.py` `_build_builtin_panel_steps`, add a step after `menus`:

```python
    def video_to_splat():
        from .video_to_splat import register
        register()
```

and append `("video_to_splat", video_to_splat)` to the returned list.

In `file_menu.py` `menu_items`, add right after the `New Project` entry:

```python
            menu_operator("lfs_plugins.video_to_splat.operators.NewProjectFromVideoOperator",
                          label=lf.ui.tr("menu.file.new_project_from_video")),
```

- [ ] **Step 6: Locale keys**

Add to `en.json` in the `"menu"` section: `"file.new_project_from_video": "New Project from Video..."`. Add a new top-level `"video_to_splat"` section:

```json
  "video_to_splat": {
    "title": "Video to Splat",
    "mode": "360 video",
    "mode_auto": "Auto-detect",
    "mode_on": "Yes, equirectangular",
    "mode_off": "No, normal video",
    "preset": "Quality",
    "preset_fast": "Fast (1 fps, 2880 px)",
    "preset_balanced": "Balanced (2 fps, 4096 px)",
    "preset_high": "High (3 fps, full size)",
    "extract_only": "Extract frames only",
    "hint": "Create extracts frames, solves cameras with COLMAP, and starts training.",
    "cancelling": "Cancelling...",
    "training_busy": "Stop the current training before building a splat from video.",
    "download_title": "Download COLMAP?",
    "download_message": "Solving camera poses needs COLMAP {version} ({size_mb} MB).\nIt will be downloaded from\n{url}\nand verified before use.",
    "download_accept": "Download",
    "download_decline": "Not now",
    "downloading_percent": "Downloading COLMAP: {percent}%",
    "downloading_mb": "Downloading COLMAP: {mb} MB",
    "weak_title": "Weak reconstruction",
    "weak_message": "COLMAP registered {registered} of {total} frames ({percent}%). Training on a sparse reconstruction gives poor results.",
    "weak_train_anyway": "Train anyway",
    "weak_stop": "Stop",
    "project_create_failed": "The project could not be created.",
    "import_failed": "The dataset import failed: {error}",
    "import_timeout": "The dataset import did not finish in time.",
    "stage.toolchain": "Locating COLMAP",
    "stage.extract": "Extracting frames",
    "stage.features": "Detecting features",
    "stage.match": "Matching frames",
    "stage.map": "Solving cameras",
    "stage.create_project": "Creating project",
    "stage.train": "Starting training"
  }
```

Add the same keys to the other nine locale files. Use the English text as a placeholder value in each (the contract test checks keys and `{placeholders}`, not translations) and note in the commit message that translations are pending. Keep one key per line and matching indentation; `tests/python/test_localization_contracts.py` enforces both. Confirm `common.cancel` exists in `en.json` (grep); if not, use the existing cancel key the New Project dialog's cancel button already uses.

- [ ] **Step 7: Run tests, register, commit**

Add `test_new_project_panel_video.py` to `LFS_PYTHON_FAST_TESTS`.

Run: `python -m pytest tests/python/test_new_project_panel_video.py tests/python/test_new_project_panel.py tests/python/test_import_dialog_panels.py tests/python/test_localization_contracts.py -q`
Expected: PASS. Also run `python tools/check_ui_hardcoded.py` → `No likely hardcoded UI strings found.`

```bash
git add src/python/lfs_plugins/import_panels.py src/python/lfs_plugins/panels.py src/python/lfs_plugins/file_menu.py src/python/lfs_plugins/video_to_splat/__init__.py src/python/lfs_plugins/video_to_splat/operators.py src/visualizer/gui/rmlui/resources/new_project_panel.rml src/visualizer/gui/rmlui/resources/import_dialog.rcss src/visualizer/gui/resources/locales/*.json tests/python/test_new_project_panel_video.py tests/CMakeLists.txt
git commit -m "Add a video source to the New Project dialog that runs the video-to-splat pipeline"
```

---

### Task 9: Dropped video opens the New Project dialog

**Goal:** Dragging a video onto the window opens the New Project dialog (which offers both Create and Extract frames only) instead of going straight to the extractor.

**Files:**
- Modify: `src/visualizer/input/input_controller.cpp:2824-2829`
- Modify: `tests/` existing input controller test if one covers the video drop (grep `ShowVideoExtractor` under `tests/`)

**Acceptance Criteria:**
- [ ] A single dropped video emits `cmd::ShowNewProjectDialog{.source_path = dropped_path}`.
- [ ] `.licht` and dataset drops are unchanged.
- [ ] Any existing test asserting `ShowVideoExtractor` on drop is updated to assert `ShowNewProjectDialog`.

**Verify:** Build, drop an `.mp4` onto the app → New Project dialog opens with the path filled and the video block visible. `ctest -L fast` passes.

**Steps:**

- [ ] **Step 1: Find and update any existing test**

Run: `grep -rn "ShowVideoExtractor" tests/`
If a test asserts the extractor opens on a video drop, change its expectation to `ShowNewProjectDialog` with the same path. If none exists, add this case to the nearest input-controller drop test file (grep `handleFileDrop` under `tests/`), following that file's existing pattern for capturing emitted events with `::when`.

- [ ] **Step 2: Change the drop handler**

Replace the video branch in `handleFileDrop`:

```cpp
            if (lfs::io::video::is_supported_video_extension(extension)) {
                cmd::ShowNewProjectDialog{.source_path = dropped_path}.emit();
                LOG_INFO("Opening New Project dialog for dropped video: {}",
                         lfs::core::path_to_utf8(dropped_path.filename()));
                return;
            }
```

- [ ] **Step 3: Build, run, commit**

Run: `cmake --build build/tests --target lichtfeld_tests && ctest --test-dir build/tests -L fast --output-on-failure`
Expected: PASS.

```bash
git add src/visualizer/input/input_controller.cpp tests/
git commit -m "Open the New Project dialog when a video is dropped on the window"
```

---

### Task 10: Opt-in end-to-end smoke check and documentation

**Goal:** A script that runs the whole pipeline inside the app on a user-supplied 360 clip and asserts the registered ratio, plus the user-facing feature doc.

**Files:**
- Create: `scripts/video_to_splat_smoke.py`
- Create: `docs/docs/features/video-to-splat.md`
- Modify: `docs/docs/features/_category_.json` only if it lists pages explicitly (check; Docusaurus usually auto-discovers)

**Acceptance Criteria:**
- [ ] With `LFS_V2S_VIDEO=<path to an equirectangular mp4>` set, running the script in the app's Python editor (or `editor_run` over MCP) produces a project, prints `registered X/Y (Z%)`, and exits non-zero when Z < 90.
- [ ] The doc explains requirements (Windows download or Linux COLMAP 4.1+), the three presets, 360 detection, what to do on a weak reconstruction, and where `pipeline.json` lives.

**Verify:** Run the script once on real footage; paste its final line into the PR description.

**Steps:**

- [ ] **Step 1: Script**

```python
# scripts/video_to_splat_smoke.py
"""End-to-end check of the video-to-splat pipeline. Run inside LichtFeld's Python editor.

Set LFS_V2S_VIDEO to an equirectangular MP4 and optionally LFS_V2S_PROJECT to a
destination folder. Prints the registration ratio and raises when it is under 90%.
"""
import os
import sys
import tempfile
import threading
from pathlib import Path

from lfs_plugins.video_to_splat import pipeline, project_bridge

video = Path(os.environ["LFS_V2S_VIDEO"])
project = Path(os.environ.get("LFS_V2S_PROJECT") or tempfile.mkdtemp(prefix="v2s-smoke-"))
request = pipeline.Request(video=video, project=project, mode_override="auto", preset="balanced")
done = threading.Event()
outcome = {}


def on_progress(stage, fraction, message, _tail):
    print(f"[{stage}] {int(fraction * 100):3d}% {message}")


def worker():
    try:
        outcome["state"] = pipeline.run(request, project_bridge.build_hooks(), on_progress=on_progress,
                                        should_cancel=lambda: False)
    except Exception as exc:  # noqa: BLE001
        outcome["error"] = exc
    finally:
        done.set()


threading.Thread(target=worker, daemon=True).start()
done.wait()
if "error" in outcome:
    raise SystemExit(f"pipeline failed: {outcome['error']}")
state = outcome["state"]
ratio = state.registered / state.frames if state.frames else 0.0
print(f"registered {state.registered}/{state.frames} ({ratio:.0%}) project={project}")
if ratio < 0.9:
    sys.exit(1)
```

- [ ] **Step 2: Documentation**

```markdown
---
title: Video to Splat
---

# Video to Splat

Create a trained splat from a video in one step. LichtFeld extracts frames, solves the camera poses with
COLMAP, creates the project and starts training.

## Requirements

- **Windows:** nothing to install. The first run asks to download COLMAP 4.2.1 (about 395 MB) from the official
  GitHub release. The archive is verified against a pinned SHA-256 and cached under the app cache directory.
- **Linux:** install COLMAP 4.1 or later and make sure `colmap` is on `PATH`, or set the path in the plugin
  settings (`video_to_splat` → `colmap_path`). Version 4.1 is the first with the equirectangular camera model.
- A CUDA GPU. COLMAP runs on the GPU before training starts, so the two never share memory.

## Workflow

1. **File → New Project from Video…**, or drop a video onto the window, or type a video path into the New
   Project source field.
2. Choose **360 video**: Auto detects a 2:1 aspect ratio or spherical metadata; override when needed.
3. Choose a **Quality** preset:

   | Preset | Frames per second | Max frame width | Matching window |
   | --- | --- | --- | --- |
   | Fast | 1 | 2880 | 10 |
   | Balanced | 2 | 4096 | 15 |
   | High | 3 | source | 25 |

4. Press **Create**. Progress shows the current stage. Cancel stops COLMAP and keeps what finished.
5. Training starts automatically. Export with the usual exporters.

**Extract frames only** opens the frame extractor on the same video for a manual workflow.

## Inputs

Stitched equirectangular MP4 or MOV from any 360 camera (export with direction lock on), or ordinary perspective
video. Raw `.insv` dual-fisheye files are not supported; stitch them in the camera maker's software first.

## When the reconstruction is weak

If COLMAP registers under 60 percent of the frames, or fewer than 20, the dialog shows the ratio and asks whether
to train anyway. Usually the fix is more overlap: move slower, choose the High preset, or shoot with better light
and texture.

## Files

Frames, the COLMAP database and the sparse model live in `<project>/dataset/`. `<project>/pipeline.json` records
each finished stage; rerunning Create on the same project resumes from the first unfinished stage when the
video and settings are unchanged.
```

- [ ] **Step 3: Commit**

```bash
git add scripts/video_to_splat_smoke.py docs/docs/features/video-to-splat.md
git commit -m "Document the video-to-splat workflow and add an end-to-end smoke script"
```

---

### Task 11: Run the smoke check on real footage and record results

**Goal:** Prove the chain on an actual 360 clip and tune defaults if needed before opening the PR.

**Files:**
- Possibly modify: `src/python/lfs_plugins/video_to_splat/probe.py` (preset values), `sfm.py` (matcher options)

**Acceptance Criteria:**
- [ ] The smoke script reports at least 90 percent registration on a slow handheld indoor 360 clip of 30 to 60 seconds.
- [ ] Training starts without manual intervention and the viewer shows the scene.
- [ ] The result line is recorded in the PR description.

**Verify:** `scripts/video_to_splat_smoke.py` final line, plus a screenshot of the trained scene.

**Steps:**

- [ ] **Step 1: Record a clip or use an existing stitched 360 MP4** with direction lock, slow motion, indoors.
- [ ] **Step 2: Run** the script in the app's Python editor with `LFS_V2S_VIDEO` set. If registration is under 90 percent, first try `preset="high"`; if that passes and `balanced` does not, raise `balanced` overlap from 15 to 20 and commit with the measured numbers in the message.
- [ ] **Step 3: Commit any tuning**

```bash
git add src/python/lfs_plugins/video_to_splat/probe.py src/python/lfs_plugins/video_to_splat/sfm.py
git commit -m "Tune video-to-splat presets from a real 360 capture (N/M frames registered)"
```

---

### Task 12: Accept a folder of photos as the source

**Goal:** Build a splat from JPEG/PNG photos without poses, the way the cloud service accepts photo sets.

**Files:** `video_to_splat/probe.py` (photo listing, header-only image size), `pipeline.py` (photo import in the extract stage), `import_panels.py` (`photos` source kind), `operators.py` + `file_menu.py` (`New Project from Photos…`), `new_project_panel.rml`, locales, tests.

**Acceptance Criteria:**
- [x] A folder with at least 20 photos runs COLMAP, creates the project and starts training; fewer stops with a clear message.
- [x] 2:1 photos are solved as equirectangular; the dialog shows the same 360 and preset options as for video.
- [x] The fingerprint changes when photos are added or removed.

**Verify:** `python -m pytest tests/python/test_video_to_splat_pipeline.py tests/python/test_new_project_panel_video.py -q`

---

### Task 13: Level the reconstruction

**Goal:** A horizontal floor in the viewer without manual rotation.

**Files:** `sfm.py` (`plan_level_command`), `pipeline.py` (`level` stage after `map`), locales (`stage.level`, progress keys), tests.

**Acceptance Criteria:**
- [x] `model_orientation_aligner` runs with IMAGE-ORIENTATION for equirectangular and MANHATTAN-WORLD for perspective cameras.
- [x] A failed or inconsistent alignment keeps the mapped model and the run continues.

**Verify:** `python -m pytest tests/python/test_video_to_splat_pipeline.py tests/python/test_video_to_splat_planning.py -q`

---

### Task 14: MCP tools for unattended runs

**Goal:** Start, watch and cancel a run from an agent or script without the dialog.

**Files:** `video_to_splat/runner.py`, `video_to_splat/mcp_tools.py`, `video_to_splat/__init__.py`, `tests/python/test_video_to_splat_runner.py`, docs.

**Acceptance Criteria:**
- [x] `video_to_splat_start` validates source, mode and preset, refuses while training or another run is active, and adds `.licht` when missing.
- [x] `video_to_splat_status` reports stage, fraction, message, result counts and errors; `video_to_splat_cancel` stops COLMAP.

**Verify:** `python -m pytest tests/python/test_video_to_splat_runner.py -q`, then `tools/list` on the running app shows the three tools.

---

### Task 15: Mask people automatically

**Goal:** Walkers leave holes instead of smears, as the cloud service does.

**Files:** `video_to_splat/masking.py` (RVM through onnxruntime, pinned model), `pipeline.py` (`mask` stage), `project_bridge.py` (consent, package install, `mask_mode = IGNORE`), dialog checkbox, MCP `mask_people`, locales, `tests/python/test_video_to_splat_masking.py`.

**Acceptance Criteria:**
- [x] Masks are written as `dataset/masks/<stem>.png` with 0 where a person is; training runs in ignore mode when masks exist.
- [x] Declining the setup, a failed download or a crashed model skips masking and the run continues.
- [x] The real model masks the camera operator in a Theta frame (slow test).

**Verify:** `python -m pytest tests/python/test_video_to_splat_masking.py tests/python/test_video_to_splat_bridge.py -q`

---

### Task 16: Package and mesh outputs

**Goal:** The download package of the cloud service: sparse points, camera poses, and on request a dense cloud and textured mesh.

**Files:** `sfm.py` (`read_cameras_bin`, `read_images_bin`, `write_cameras_json`, `plan_package_command`, `plan_mesh_commands`), `pipeline.py` (`package` and `mesh` stages), dialog checkbox, MCP `build_mesh`, docs.

**Acceptance Criteria:**
- [x] `package/sparse.ply` and `package/cameras.json` exist after every run.
- [x] With the mesh enabled the five dense commands run in order into `package/`; a failing tool is recorded and the splat is still built.

**Verify:** `python -m pytest tests/python/test_video_to_splat_pipeline.py tests/python/test_video_to_splat_planning.py -q`

---

### Task 17: Finish the download package when training ends

**Goal:** Every output the cloud service hands back, without waiting on a GPU feature: splat exports, dense cloud, mesh in PLY/GLB/USDZ.

**Files:** `video_to_splat/mesh_export.py` (Gaussian cloud, PLY reader, GLB and USDZ writers), `video_to_splat/finalize.py` (the package step), `project_bridge.py` (training-end hook, exporter polling), `runner.py` + `mcp_tools.py` (`video_to_splat_finalize`), tests.

**Acceptance Criteria:**
- [x] The training-end hook writes splat.ply/spz/sog/glb, dense.ply, mesh.ply, mesh.glb, mesh.usdz and package.json without manual steps.
- [x] The GLB is a valid glTF 2.0 binary the app's loader opens; the USDZ is a stored zip with 64-byte-aligned members.
- [x] A failed exporter or mesher is recorded in package.json and the other files are still written.

**Verify:** `python -m pytest tests/python/test_video_to_splat_mesh_export.py tests/python/test_video_to_splat_finalize.py -q`

---

### Task 18: Texture the mesh from the frames, ship the viewer page, announce the package

**Goal:** The last cloud deliverables: a textured mesh in glTF/USDZ, a shareable viewer, and a "ready" notice.

**Files:** `video_to_splat/texturing.py` (projection, view choice, atlas baking, textured GLB/USDZ), `finalize.py` (steps 5 and 6), `project_bridge.py` (`export_viewer`, `announce_package`), locales, `tests/python/test_video_to_splat_texturing.py`.

**Acceptance Criteria:**
- [x] Projection matches COLMAP and the rasterizer for pinhole, OpenCV and equirectangular cameras.
- [x] `mesh_textured.glb` carries TEXCOORD_0 and an embedded PNG; `mesh_textured.usdz` binds a UsdUVTexture to `atlas.png`.
- [x] `viewer.html` is written through `lf.io.export_html`; a dialog names the package folder when training ends.

**Verify:** `python -m pytest tests/python/test_video_to_splat_texturing.py tests/python/test_video_to_splat_finalize.py -q`

---

### Task 19: Publish to the gallery when the package is ready

**Goal:** The hosted, account-based sharing step of the cloud service, through the app's own gallery.

**Files:** `pipeline.py` (`publish_to_gallery` request flag passed to the train hook), `project_bridge.py` (`publish_to_gallery`, package-ready dialog buttons), dialog checkbox, MCP argument, locales, tests.

**Acceptance Criteria:**
- [x] With the option on, the gallery publish review opens right after the package is written; otherwise the ready dialog offers "Publish to gallery" and "Show folder".
- [x] The option does not change the request fingerprint, so a rerun resumes.

**Verify:** `python -m pytest tests/python/test_video_to_splat_bridge.py tests/python/test_video_to_splat_pipeline.py -q`

---

## Self-review

**Spec coverage.** Toolchain download with consent and hash (Task 5, 7); Linux PATH fallback (5); `lf.video` probe and extract (2, 3); COLMAP commands with EQUIRECTANGULAR/OPENCV and loop detection off (4); shared process runner (1); stage orchestration with `pipeline.json` resume and cancel (6); project creation and training start with import wait (7); New Project `video` kind, options, progress, Extract frames only, File menu entry (8); drop behaviour (9); error handling: no COLMAP (5, 7), bad input (6), weak reconstruction prompt (6, 7), cancel (1, 6), GPU contention via `is_training_active` (8), disk warning (6); tests: Python unit (1, 4, 5, 6, 7, 8), C++ probe test (2), binding runtime test (3), opt-in end-to-end (10, 11); docs (10). Not in scope per spec: `.insv`, vocab tree, CLI flag, MCP job, the three extras.

**Known deviations from the spec text.** The integration check is a script run inside the app rather than a pytest rendering a synthetic 360 orbit, because frame extraction needs the app's FFmpeg build and a real clip exercises COLMAP more honestly. The "Extract frames only / Build splat" choice on drop is the New Project dialog with two buttons rather than a separate prompt.

**Type consistency.** `Hooks` field names (`resolve_colmap, probe, extract, run, create_project, train, confirm_weak, free_bytes`) match between `pipeline.py`, `project_bridge.build_hooks` and the test `Fakes.hooks()`. `ExtractParams(video, dataset, fps, scale, rotation)` matches `project_bridge.extract`. `Request(video, project, mode_override, preset, use_gpu)` matches panel and smoke script. `STAGE_NAMES` matches the locale `stage.*` keys. `toolchain.Resolved(path, version)` is used by the pipeline test fakes.

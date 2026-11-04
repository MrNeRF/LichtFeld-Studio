# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the app-side hooks of the video-to-splat pipeline against a stubbed lichtfeld module."""

import sys
from importlib import import_module
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "python"))


@pytest.fixture
def lf_stub(monkeypatch, tmp_path):
    for name in list(sys.modules):
        if name == "lfs_plugins" or name.startswith("lfs_plugins."):
            sys.modules.pop(name, None)
    state = SimpleNamespace(scheduled=[], dialogs=[], created=[], loaded=[], import_states=[], started=0,
                            settings={"colmap_path": "D:/tools/colmap/bin/colmap.exe"}, extract_kwargs=None,
                            trainer_states=["starting", "running"],
                            optimization=SimpleNamespace(gut=False, mask_mode="NONE"), installed=[])

    def schedule(cb):
        state.scheduled.append(cb)
        cb()  # tests run the UI thread inline

    def confirm_dialog(title, message, buttons, callback, style="info"):
        state.dialogs.append((title, message, buttons, style))
        callback(buttons[0])

    class Settings:
        def get(self, key, default=""):
            return state.settings.get(key, default)

    def extract_frames(**kwargs):
        state.extract_kwargs = kwargs
        return {"outcome": "completed", "error": ""}

    # Messages the bridge formats; placeholder names must match the en.json entries Task 8 adds.
    templates = {
        "video_to_splat.download_message": "COLMAP {version} ({size_mb} MB) from {url}",
        "video_to_splat.weak_message": "{registered} of {total} frames ({percent}%)",
        "video_to_splat.import_failed": "import failed: {error}",
    }

    stub = ModuleType("lichtfeld")
    stub.ui = SimpleNamespace(
        schedule_on_ui_thread=schedule,
        confirm_dialog=confirm_dialog,
        message_dialog=lambda *a, **k: None,
        tr=lambda key: templates.get(key, key),
        get_import_state=lambda: state.import_states.pop(0),
    )
    stub.project_create = lambda path, **kw: state.created.append((path, kw)) or True
    stub.project_create_pending = lambda: False
    stub.load_file = lambda **kw: state.loaded.append(kw)
    stub.start_training = lambda: setattr(state, "started", state.started + 1)
    stub.trainer_state = lambda: state.trainer_states.pop(0) if len(state.trainer_states) > 1 else state.trainer_states[0]
    stub.MaskMode = SimpleNamespace(IGNORE="IGNORE", NONE="NONE")
    stub.packages = SimpleNamespace(install=lambda name: state.installed.append(name) or "ok")
    stub.optimization_params = lambda: state.optimization
    stub.is_training_active = lambda: False
    stub.plugins = SimpleNamespace(settings=lambda name: Settings())
    stub.video = SimpleNamespace(probe=lambda p: {"width": 1}, extract_frames=extract_frames)
    monkeypatch.setitem(sys.modules, "lichtfeld", stub)
    monkeypatch.setenv("LFS_RESOLVED_CACHE_DIR", str(tmp_path / "cache"))
    module = import_module("lfs_plugins.video_to_splat.project_bridge")
    return module, state


def test_ask_on_ui_thread_returns_value(lf_stub):
    bridge, state = lf_stub
    assert bridge.ask_on_ui_thread(lambda: 42) == 42
    assert len(state.scheduled) == 1


def test_ask_on_ui_thread_reraises_errors(lf_stub):
    bridge, _state = lf_stub

    def boom():
        raise ValueError("on the UI thread")

    with pytest.raises(ValueError, match="on the UI thread"):
        bridge.ask_on_ui_thread(boom)


def test_confirm_download_uses_first_button_as_accept(lf_stub):
    bridge, state = lf_stub
    assert bridge.confirm_download() is True
    title, message, buttons, style = state.dialogs[0]
    assert title == "video_to_splat.download_title"
    assert buttons == ["video_to_splat.download_accept", "video_to_splat.download_decline"]
    assert "395" in message and "4.2.1" in message and "github.com/colmap" in message


def test_confirm_download_declined(lf_stub, monkeypatch):
    bridge, state = lf_stub
    monkeypatch.setattr(bridge.lf.ui, "confirm_dialog", lambda t, m, buttons, cb, style="info": cb(buttons[1]))
    assert bridge.confirm_download() is False


def test_confirm_weak_formats_ratio(lf_stub):
    bridge, state = lf_stub
    assert bridge.confirm_weak(0.4, 20, 50) is True
    title, message, buttons, style = state.dialogs[0]
    assert buttons == ["video_to_splat.weak_train_anyway", "video_to_splat.weak_stop"]
    assert style == "warning"
    assert "40" in message and "20" in message and "50" in message


def test_create_project_and_wait_polls_import(lf_stub):
    bridge, state = lf_stub
    state.import_states.extend([{"active": True}, {"active": True}, {"active": False, "success": True}])
    bridge.create_project_and_wait(Path("P:/proj"), Path("P:/proj/dataset"), "images", timeout=5, poll_seconds=0)
    assert Path(state.created[0][0]) == Path("P:/proj")
    assert state.created[0][1]["overwrite"] is True
    assert state.loaded[0]["is_dataset"] is True and state.loaded[0]["images_folder"] == "images"
    assert Path(state.loaded[0]["path"]) == Path("P:/proj/dataset")
    assert state.import_states == []


def test_create_project_and_wait_raises_on_failed_import(lf_stub):
    bridge, state = lf_stub
    state.import_states.extend([{"active": False, "success": False, "error": "bad sparse"}])
    with pytest.raises(bridge.PipelineError) as err:
        bridge.create_project_and_wait(Path("P:/proj"), Path("P:/proj/dataset"), "images", timeout=5, poll_seconds=0)
    assert "bad sparse" in str(err.value)


def test_create_project_and_wait_raises_when_project_create_fails(lf_stub):
    bridge, state = lf_stub
    bridge.lf.project_create = lambda path, **kw: False
    with pytest.raises(bridge.PipelineError):
        bridge.create_project_and_wait(Path("P:/proj"), Path("P:/proj/dataset"), "images", timeout=5, poll_seconds=0)
    assert state.loaded == []


def test_create_project_and_wait_times_out(lf_stub):
    bridge, state = lf_stub
    bridge.lf.ui.get_import_state = lambda: {"active": True}
    with pytest.raises(bridge.PipelineError) as err:
        bridge.create_project_and_wait(Path("P:/proj"), Path("P:/proj/dataset"), "images", timeout=0, poll_seconds=0)
    assert "video_to_splat.import_timeout" in str(err.value)


def test_start_training_runs_on_ui_thread_and_waits_for_the_trainer(lf_stub):
    bridge, state = lf_stub
    bridge.start_training("perspective", poll_seconds=0)
    assert state.started == 1
    assert state.scheduled  # start and every state poll go through the UI thread
    assert state.optimization.gut is False


def test_start_training_enables_gut_for_equirectangular_cameras(lf_stub):
    # The trainer rejects non-pinhole cameras unless GUT (or undistortion, meaningless for 360) is on.
    bridge, state = lf_stub
    bridge.start_training("equirectangular", poll_seconds=0)
    assert state.started == 1
    assert state.optimization.gut is True


def test_start_training_reports_a_start_that_falls_back_to_ready(lf_stub):
    bridge, state = lf_stub
    state.trainer_states[:] = ["starting", "starting", "ready"]
    with pytest.raises(bridge.PipelineError) as err:
        bridge.start_training("perspective", poll_seconds=0)
    assert "video_to_splat.training_failed" in str(err.value)


def test_start_training_reports_a_start_that_never_dispatches(lf_stub):
    bridge, state = lf_stub
    state.trainer_states[:] = ["ready"]
    with pytest.raises(bridge.PipelineError):
        bridge.start_training("perspective", poll_seconds=0, dispatch_timeout=0.0)


def test_extract_forwards_params(lf_stub):
    bridge, state = lf_stub
    from lfs_plugins.video_to_splat.pipeline import ExtractParams

    params = ExtractParams(video=Path("D:/clips/a.mp4"), dataset=Path("P:/proj/dataset"), fps=2.0, scale=0.5, rotation=90)
    result = bridge.extract(params, lambda c, t, d: None, lambda: False)
    assert result["outcome"] == "completed"
    kwargs = state.extract_kwargs
    assert Path(kwargs["video_path"]) == Path("D:/clips/a.mp4")
    assert Path(kwargs["output_dir"]) == Path("P:/proj/dataset/images")
    assert kwargs["fps"] == 2.0 and kwargs["scale"] == 0.5 and kwargs["rotation"] == 90
    assert kwargs["sharpness_enabled"] is True
    # Keep the sharpest frame of each interval instead of a fixed edge-ratio cutoff, which empties
    # low-texture footage (a white room scored right at the old threshold of 10).
    assert kwargs["sharpness_window_mode"] is True
    assert kwargs["sharpness_window_candidates"] == -1
    assert kwargs["sharpness_threshold"] == 0.0


def test_build_hooks_reads_settings_and_cache(lf_stub, tmp_path):
    bridge, state = lf_stub
    hooks = bridge.build_hooks()
    assert hooks.probe is bridge.lf.video.probe
    assert hooks.train is bridge.start_training
    assert hooks.extract is bridge.extract
    assert hooks.confirm_weak is bridge.confirm_weak
    assert bridge.cache_dir() == tmp_path / "cache"
    assert bridge.configured_colmap_path() == "D:/tools/colmap/bin/colmap.exe"


def test_cache_dir_falls_back_to_home(lf_stub, monkeypatch):
    bridge, _state = lf_stub
    monkeypatch.delenv("LFS_RESOLVED_CACHE_DIR")
    assert bridge.cache_dir() == Path.home() / ".lichtfeld" / "cache"


def test_resolve_colmap_wraps_toolchain_errors(lf_stub, monkeypatch):
    bridge, state = lf_stub
    state.settings["colmap_path"] = "Q:/missing/colmap.exe"
    with pytest.raises(bridge.PipelineError) as err:
        bridge.resolve_colmap(lambda msg: None)
    assert "Q:/missing/colmap.exe" in str(err.value).replace("\\", "/")


def test_start_training_applies_people_masks_in_ignore_mode(lf_stub):
    # project_create resets the parameters, so the mask mode is set right before training starts.
    bridge, state = lf_stub
    bridge.start_training("equirectangular", True, poll_seconds=0)
    assert state.optimization.mask_mode == "IGNORE" and state.optimization.gut is True


def test_start_training_leaves_mask_mode_alone_without_masks(lf_stub):
    bridge, state = lf_stub
    bridge.start_training("perspective", False, poll_seconds=0)
    assert state.optimization.mask_mode == "NONE"


def test_mask_people_sets_up_packages_and_model_on_consent(lf_stub, tmp_path, monkeypatch):
    bridge, state = lf_stub
    from lfs_plugins.video_to_splat import masking

    monkeypatch.setattr(masking, "missing_packages", lambda: ["onnxruntime"])
    downloaded = []

    def download(url, dest, on_progress):
        downloaded.append(url)
        dest.write_bytes(b"model")

    monkeypatch.setattr(bridge, "download_file", download)
    import hashlib

    monkeypatch.setattr(masking, "MODEL_SHA256", hashlib.sha256(b"model").hexdigest())
    written = []

    class FakeMatter:
        def __init__(self, path):
            written.append(("model", path))

    monkeypatch.setattr(masking, "RvmMatter", FakeMatter)
    monkeypatch.setattr(masking, "write_masks", lambda images, masks_dir, matte, **kw: written.append(("masks", len(images))) or len(images))
    monkeypatch.setattr(masking, "count_masked_people", lambda masks_dir: 3)
    images = [tmp_path / "a.jpg", tmp_path / "b.jpg"]
    result = bridge.mask_people(images, tmp_path / "masks", lambda c, t: None, lambda: False, True)
    assert result == {"outcome": "completed", "masked": 3}
    assert state.installed == ["onnxruntime"]
    assert downloaded == [masking.MODEL_URL]
    assert state.dialogs[0][0] == "video_to_splat.mask_setup_title"
    assert written == [("model", masking.cached_model_path(tmp_path / "cache")), ("masks", 2)]


def test_mask_people_declined_is_skipped_not_fatal(lf_stub, tmp_path, monkeypatch):
    bridge, state = lf_stub
    from lfs_plugins.video_to_splat import masking

    monkeypatch.setattr(masking, "missing_packages", lambda: ["numpy", "pillow", "onnxruntime"])
    bridge.lf.ui.confirm_dialog = lambda title, message, buttons, callback, style="info": callback(buttons[1])
    result = bridge.mask_people([tmp_path / "a.jpg"], tmp_path / "masks", lambda c, t: None, lambda: False, True)
    assert result["outcome"] == "skipped" and state.installed == []


def test_mask_people_reports_setup_failures_as_skipped(lf_stub, tmp_path, monkeypatch):
    bridge, state = lf_stub
    from lfs_plugins.video_to_splat import masking

    monkeypatch.setattr(masking, "missing_packages", lambda: [])

    def failing_download(url, dest, on_progress):
        raise OSError("offline")

    monkeypatch.setattr(bridge, "download_file", failing_download)
    result = bridge.mask_people([tmp_path / "a.jpg"], tmp_path / "masks", lambda c, t: None, lambda: False, True)
    assert result["outcome"] == "skipped" and "offline" in result["reason"]

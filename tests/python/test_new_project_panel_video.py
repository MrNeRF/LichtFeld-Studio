# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""New Project panel behavior for a video source: options, pipeline thread, extract-only, errors."""

from importlib import import_module
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
import threading

import pytest


class _Handle:
    def __init__(self):
        self.dirty_fields = []

    def dirty(self, name):
        self.dirty_fields.append(name)

    def dirty_all(self):
        self.dirty_fields.append("*")


@pytest.fixture
def panel_env(monkeypatch, tmp_path):
    source_python = Path(__file__).parents[2] / "src" / "python"
    if str(source_python) not in sys.path:
        sys.path.insert(0, str(source_python))
    for name in list(sys.modules):
        if name == "lfs_plugins" or name.startswith("lfs_plugins."):
            sys.modules.pop(name, None)

    location = tmp_path / "projects"
    location.mkdir()
    state = SimpleNamespace(
        location=location,
        panel_enabled=[],
        prompts=[],
        calls=[],
        scheduled=[],
        training=False,
        video_extractor_paths=[],
        messages=[],
        registered_classes=[],
        registered_panels={},
    )

    def schedule_on_ui_thread(callback):
        state.scheduled.append(callback)

    def confirm(title, message, buttons, callback=None, style="info"):
        state.prompts.append((title, message, buttons, callback))

    lf_stub = ModuleType("lichtfeld")
    lf_stub.io = SimpleNamespace(is_ssog_path=lambda _path: False)
    lf_stub.ui = SimpleNamespace(
        Panel=type("Panel", (), {}),
        PanelSpace=SimpleNamespace(FLOATING="FLOATING"),
        PanelHeightMode=SimpleNamespace(CONTENT="CONTENT"),
        PanelOption=SimpleNamespace(DEFAULT_CLOSED="DEFAULT_CLOSED"),
        get_current_language=lambda: "en",
        get_project_location=lambda: str(state.location),
        get_default_project_location=lambda: str(location),
        set_panel_enabled=lambda panel_id, enabled: state.panel_enabled.append((panel_id, enabled)),
        is_panel_enabled=lambda panel_id: next((enabled for identifier, enabled in reversed(state.panel_enabled)
                                               if identifier == panel_id), False),
        tr=lambda key: key,
        confirm_dialog=confirm,
        message_dialog=lambda title, message, style="info", callback=None: state.messages.append((title, message, style)),
        open_dataset_folder_dialog=lambda: "",
        open_ply_file_dialog=lambda _start="": "",
        open_init_file_dialog=lambda _start="": "",
        open_video_file_dialog=lambda: "",
        show_video_extractor=state.video_extractor_paths.append,
        get_embed_dataset_by_default=lambda: False,
        schedule_on_ui_thread=schedule_on_ui_thread,
        get_panel_object=lambda panel_id: state.registered_panels.get(panel_id),
        get_import_state=lambda: {"active": False, "success": True},
    )

    def register_class(cls):
        state.registered_classes.append(cls)
        if hasattr(cls, "id"):
            state.registered_panels[cls.id] = cls()

    lf_stub.register_class = register_class
    lf_stub.is_dataset_path = lambda path: False
    lf_stub.detect_dataset_info = lambda _path, **kw: None
    lf_stub.optimization_params = lambda: None
    lf_stub.project_is_dirty = lambda: False
    lf_stub.project_has_path = lambda: True
    lf_stub.is_training_active = lambda: state.training
    lf_stub.project_create = lambda *args, **kwargs: state.calls.append(("create", args, kwargs)) or True
    lf_stub.project_create_pending = lambda: False
    lf_stub.load_file = lambda *args, **kwargs: state.calls.append(("load", args, kwargs))
    lf_stub.project_embed_dataset = lambda: state.calls.append(("embed", (), {}))
    lf_stub.start_training = lambda: state.calls.append(("train", (), {}))
    lf_stub.plugins = SimpleNamespace(settings=lambda name: SimpleNamespace(get=lambda k, d="": d))
    lf_stub.video = SimpleNamespace(probe=lambda p: {}, extract_frames=lambda **kw: {"outcome": "completed", "error": ""})
    monkeypatch.setitem(sys.modules, "lichtfeld", lf_stub)
    state.lf = lf_stub
    module = import_module("lfs_plugins.import_panels")
    panel = module.NewProjectPanel()
    panel._handle = _Handle()
    panel._dialog_mounted = True
    return module, panel, state


def _drain(state):
    while state.scheduled:
        callback = state.scheduled.pop(0)
        callback()


def _wait_for_thread(panel):
    thread = panel._pipeline_thread
    assert thread is not None
    thread.join(5)
    assert not thread.is_alive()


def test_video_extension_sets_video_kind(panel_env):
    module, panel, state = panel_env
    panel.show("D:/clips/room.mp4")
    assert panel._source_kind == "video"
    assert panel._can_create() is True
    assert panel._video_mode == "auto" and panel._video_preset == "balanced"
    assert panel._name == "room"


@pytest.mark.parametrize("name", ["clip.MOV", "clip.mkv", "clip.webm", "clip.avi"])
def test_other_video_extensions_are_recognised(panel_env, name):
    module, panel, state = panel_env
    panel.show(f"D:/clips/{name}")
    assert panel._source_kind == "video"


def test_non_video_path_still_probes(panel_env):
    module, panel, state = panel_env
    panel.show("D:/clips/room.txt")
    assert panel._source_kind == "checking"


def test_video_mode_and_preset_setters_validate(panel_env):
    module, panel, state = panel_env
    panel.show("D:/clips/room.mp4")
    panel._set_video_mode("on")
    panel._set_video_mode("bogus")
    assert panel._video_mode == "on"
    panel._set_video_preset("high")
    panel._set_video_preset("ultra")
    assert panel._video_preset == "high"


def test_show_resets_video_options(panel_env):
    module, panel, state = panel_env
    panel.show("D:/clips/room.mp4")
    panel._set_video_mode("off")
    panel._set_video_preset("fast")
    panel.show("D:/clips/other.mp4")
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
        captured["hooks"] = hooks
        on_progress("extract", 0.5, "half", "tail")
        started.set()
        return None

    sentinel = object()
    monkeypatch.setattr(module.v2s_pipeline, "run", fake_run)
    monkeypatch.setattr(module.v2s_bridge, "build_hooks", lambda: sentinel)
    panel._on_do_create()
    assert started.wait(5)
    _wait_for_thread(panel)
    assert captured["hooks"] is sentinel
    assert captured["request"].video == Path("D:/clips/room.mp4")
    assert captured["request"].project == panel._target_path()
    assert captured["request"].project.name == "room.licht"
    assert captured["request"].mode_override == "auto" and captured["request"].preset == "balanced"
    # the panel stayed open to show progress and no direct project_create happened here
    assert all(call[0] != "create" for call in state.calls)
    assert panel._pipeline_running is True
    _drain(state)
    assert panel._pipeline_stage == "extract" and panel._pipeline_fraction == 0.5
    assert panel._pipeline_message == "half"
    # success closes the dialog
    assert panel._pipeline_running is False
    assert state.panel_enabled[-1] == (panel.id, False)
    assert state.messages == []


def test_create_disabled_while_pipeline_runs(panel_env, monkeypatch):
    module, panel, state = panel_env
    panel.show("D:/clips/room.mp4")
    panel._set_name("room")
    release = threading.Event()

    def fake_run(request, hooks, *, on_progress, should_cancel):
        release.wait(5)

    monkeypatch.setattr(module.v2s_pipeline, "run", fake_run)
    monkeypatch.setattr(module.v2s_bridge, "build_hooks", lambda: object())
    panel._on_do_create()
    try:
        assert panel._pipeline_running is True
        assert panel._can_create() is False
        panel._on_do_create()  # second press is ignored
        assert len([c for c in state.calls if c[0] == "create"]) == 0
    finally:
        release.set()
    _wait_for_thread(panel)


def test_cancel_pipeline_sets_flag_and_resets_view(panel_env, monkeypatch):
    module, panel, state = panel_env
    panel.show("D:/clips/room.mp4")
    panel._set_name("room")
    cancelled = threading.Event()

    def fake_run(request, hooks, *, on_progress, should_cancel):
        cancelled.wait(5)
        assert should_cancel() is True
        raise module.v2s_pipeline.PipelineCancelled()

    monkeypatch.setattr(module.v2s_pipeline, "run", fake_run)
    monkeypatch.setattr(module.v2s_bridge, "build_hooks", lambda: object())
    panel._on_do_create()
    panel._on_cancel_pipeline()
    assert panel._pipeline_message == "video_to_splat.cancelling"
    cancelled.set()
    _wait_for_thread(panel)
    _drain(state)
    assert panel._pipeline_running is False
    assert state.messages == []
    # cancel keeps the dialog open so the user can retry
    assert (panel.id, False) not in state.panel_enabled[1:]


def test_pipeline_error_shows_message(panel_env, monkeypatch):
    module, panel, state = panel_env
    panel.show("D:/clips/room.mp4")
    panel._set_name("room")

    def fake_run(request, hooks, *, on_progress, should_cancel):
        raise module.v2s_pipeline.PipelineError("COLMAP exploded")

    monkeypatch.setattr(module.v2s_pipeline, "run", fake_run)
    monkeypatch.setattr(module.v2s_bridge, "build_hooks", lambda: object())
    panel._on_do_create()
    _wait_for_thread(panel)
    _drain(state)
    assert any("COLMAP exploded" in m and s == "error" for (_t, m, s) in state.messages)
    assert panel._pipeline_running is False


def test_pipeline_stopped_just_resets(panel_env, monkeypatch):
    module, panel, state = panel_env
    panel.show("D:/clips/room.mp4")
    panel._set_name("room")

    def fake_run(request, hooks, *, on_progress, should_cancel):
        raise module.v2s_pipeline.PipelineStopped()

    monkeypatch.setattr(module.v2s_pipeline, "run", fake_run)
    monkeypatch.setattr(module.v2s_bridge, "build_hooks", lambda: object())
    panel._on_do_create()
    _wait_for_thread(panel)
    _drain(state)
    assert state.messages == []
    assert panel._pipeline_running is False


def test_unexpected_exception_is_surfaced(panel_env, monkeypatch):
    module, panel, state = panel_env
    panel.show("D:/clips/room.mp4")
    panel._set_name("room")

    def fake_run(request, hooks, *, on_progress, should_cancel):
        raise RuntimeError("binding missing")

    monkeypatch.setattr(module.v2s_pipeline, "run", fake_run)
    monkeypatch.setattr(module.v2s_bridge, "build_hooks", lambda: object())
    panel._on_do_create()
    _wait_for_thread(panel)
    _drain(state)
    assert any("RuntimeError" in m and "binding missing" in m for (_t, m, _s) in state.messages)


def test_training_busy_refuses_to_start(panel_env, monkeypatch):
    module, panel, state = panel_env
    panel.show("D:/clips/room.mp4")
    panel._set_name("room")
    state.training = True
    monkeypatch.setattr(module.v2s_pipeline, "run", lambda *a, **k: pytest.fail("must not run"))
    panel._on_do_create()
    assert panel._pipeline_thread is None
    assert any(m == "video_to_splat.training_busy" for (_t, m, _s) in state.messages)
    assert panel._pipeline_running is False


def test_video_options_bound_in_model(panel_env):
    module, panel, state = panel_env
    bound = {}

    class Model:
        def bind_func(self, name, getter):
            bound[name] = getter

        def bind(self, name, getter, setter):
            bound[name] = getter

        def bind_event(self, name, handler):
            bound[name] = handler

        def get_handle(self):
            return _Handle()

    ctx = SimpleNamespace(create_data_model=lambda name: Model())
    panel.on_bind_model(ctx)
    for name in ("source_is_video", "video_mode", "video_preset", "pipeline_running", "pipeline_stage",
                 "pipeline_fraction", "pipeline_percent", "pipeline_message", "extract_only", "cancel_pipeline"):
        assert name in bound, name
    panel.show("D:/clips/room.mp4")
    assert bound["source_is_video"]() is True
    panel._pipeline_stage = "match"
    panel._pipeline_fraction = 0.25
    assert bound["pipeline_stage"]() == "video_to_splat.stage.match"
    assert bound["pipeline_percent"]() == 25


def _photo_folder(tmp_path, count=30):
    import struct
    folder = tmp_path / "shoot"
    folder.mkdir(exist_ok=True)
    ihdr = struct.pack(">IIBBBBB", 5760, 2880, 8, 2, 0, 0, 0)
    for i in range(count):
        (folder / f"IMG_{i:04d}.png").write_bytes(b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + ihdr + b"\0" * 4)
    return folder


def _probe(panel, state):
    """Run the source probe worker and apply its result on the fake UI thread."""
    import time
    panel._source_probe_due = 0.0
    panel._start_source_probe()
    deadline = time.monotonic() + 5
    while not state.scheduled and time.monotonic() < deadline:
        time.sleep(0.01)
    _drain(state)


def test_photo_folder_sets_photos_kind(panel_env, tmp_path):
    module, panel, state = panel_env
    folder = _photo_folder(tmp_path)
    panel.show(str(folder))
    assert panel._source_kind == "checking"
    _probe(panel, state)
    assert panel._source_kind == "photos"
    assert panel._name == "shoot"
    assert panel._can_create() is True
    assert panel._source_is_pipeline() is True and panel._source_kind != "video"


def test_folder_without_photos_is_invalid(panel_env, tmp_path):
    module, panel, state = panel_env
    folder = tmp_path / "empty"
    folder.mkdir()
    (folder / "notes.txt").write_text("x")
    panel.show(str(folder))
    _probe(panel, state)
    assert panel._source_kind == "invalid"
    assert panel._can_create() is False


def test_do_create_with_photos_starts_pipeline(panel_env, monkeypatch, tmp_path):
    module, panel, state = panel_env
    folder = _photo_folder(tmp_path)
    panel.show(str(folder))
    _probe(panel, state)
    panel._set_video_preset("high")
    started = threading.Event()
    captured = {}

    def fake_run(request, hooks, *, on_progress, should_cancel):
        captured["request"] = request
        started.set()
        return None

    monkeypatch.setattr(module.v2s_pipeline, "run", fake_run)
    monkeypatch.setattr(module.v2s_bridge, "build_hooks", lambda: object())
    panel._on_do_create()
    assert started.wait(5)
    _wait_for_thread(panel)
    assert captured["request"].video == folder
    assert captured["request"].is_photo_set
    assert captured["request"].preset == "high"
    assert captured["request"].project.name == "shoot.licht"
    assert all(call[0] != "create" for call in state.calls)


def test_photos_operator_opens_dialog_on_chosen_folder(panel_env, monkeypatch, tmp_path):
    module, panel, state = panel_env
    from lfs_plugins.video_to_splat.operators import NewProjectFromPhotosOperator, NewProjectFromVideoOperator

    folder = _photo_folder(tmp_path)
    opened = []
    monkeypatch.setattr(module, "open_new_project_panel", opened.append)
    monkeypatch.setattr(state.lf.ui, "open_folder_dialog", lambda title, start="": str(folder), raising=False)
    assert NewProjectFromPhotosOperator().execute(None) == {"FINISHED"}
    assert opened == [str(folder)]
    assert NewProjectFromPhotosOperator.label == "menu.file.new_project_from_photos"
    monkeypatch.setattr(state.lf.ui, "open_folder_dialog", lambda title, start="": "", raising=False)
    assert NewProjectFromPhotosOperator().execute(None) == {"CANCELLED"}
    assert NewProjectFromVideoOperator.label == "menu.file.new_project_from_video"

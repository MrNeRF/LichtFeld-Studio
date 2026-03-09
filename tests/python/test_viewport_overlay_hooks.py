# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression tests for hook-driven viewport overlay controllers."""

from importlib import import_module
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import pytest


class _DataModelHandleStub:
    def __init__(self):
        self.dirty_all_calls = 0

    def dirty_all(self):
        self.dirty_all_calls += 1


class _DataModelStub:
    def __init__(self):
        self.bound_funcs = {}
        self.bound_events = {}
        self.handle = _DataModelHandleStub()

    def bind_func(self, name, getter):
        self.bound_funcs[name] = getter

    def bind_event(self, name, callback):
        self.bound_events[name] = callback

    def get_handle(self):
        return self.handle


class _ElementStub:
    def __init__(self):
        self.attrs = {}

    def get_attribute(self, name, default_val=""):
        return self.attrs.get(name, default_val)

    def set_attribute(self, name, value):
        self.attrs[name] = value

    def remove_attribute(self, name):
        self.attrs.pop(name, None)


class _DocumentStub:
    def __init__(self):
        self.body = _ElementStub()
        self.created_models = []
        self.removed_models = []
        self.model = _DataModelStub()

    def get_element_by_id(self, element_id):
        if element_id == "overlay-body":
            return self.body
        return None

    def create_data_model(self, name):
        self.created_models.append(name)
        return self.model

    def remove_data_model(self, name):
        self.removed_models.append(name)
        return True


def _install_stub_modules(monkeypatch):
    hook_calls = []
    remove_calls = []
    dismiss_calls = []
    cancel_calls = []
    import_state = {}
    video_state = {}
    document = _DocumentStub()

    ui_stub = SimpleNamespace(
        add_hook=lambda panel, section, callback, position="append": hook_calls.append(
            (panel, section, callback, position)
        ),
        remove_hook=lambda panel, section, callback: remove_calls.append(
            (panel, section, callback)
        ),
        rml=SimpleNamespace(get_document=lambda _name: document),
        UILayout=SimpleNamespace(WindowFlags=SimpleNamespace(
            NoTitleBar=1,
            NoResize=2,
            NoMove=4,
            NoScrollbar=8,
            NoInputs=16,
            NoBackground=32,
            NoFocusOnAppearing=64,
            NoBringToFrontOnFocus=128,
        )),
        tr=lambda key: key,
        get_import_state=lambda: dict(import_state),
        get_video_export_state=lambda: dict(video_state),
        dismiss_import=lambda: dismiss_calls.append(True),
        cancel_video_export=lambda: cancel_calls.append(True),
        is_scene_empty=lambda: False,
        is_drag_hovering=lambda: False,
        is_startup_visible=lambda: False,
        is_sequencer_visible=lambda: False,
        get_sequencer_state=lambda: None,
        get_time=lambda: 0.0,
    )

    lf_stub = ModuleType("lichtfeld")
    lf_stub.ui = ui_stub
    monkeypatch.setitem(sys.modules, "lichtfeld", lf_stub)

    return hook_calls, remove_calls, dismiss_calls, cancel_calls, import_state, video_state, document


@pytest.fixture
def overlays_module(monkeypatch):
    project_root = Path(__file__).parent.parent.parent
    source_python = project_root / "src" / "python"
    if str(source_python) in sys.path:
        sys.path.remove(str(source_python))
    sys.path.insert(0, str(source_python))

    sys.modules.pop("lfs_plugins", None)
    sys.modules.pop("lfs_plugins.overlays", None)
    fixture = _install_stub_modules(monkeypatch)
    module = import_module("lfs_plugins.overlays")
    return (module, *fixture)


def test_register_uses_document_and_draw_hooks(overlays_module):
    module, hook_calls, _remove_calls, *_rest = overlays_module

    module.register()

    assert hook_calls == [
        ("viewport_overlay", "document", hook_calls[0][2], "append"),
        ("viewport_overlay", "draw", hook_calls[1][2], "append"),
    ]


def test_unregister_removes_both_hooks(overlays_module):
    module, hook_calls, remove_calls, *_rest = overlays_module

    module.register()
    module.unregister()

    assert len(hook_calls) == 2
    assert remove_calls == [
        ("viewport_overlay", "document", hook_calls[0][2]),
        ("viewport_overlay", "draw", hook_calls[1][2]),
    ]


def test_document_sync_binds_model_and_updates_actions(overlays_module):
    (
        module,
        _hook_calls,
        _remove_calls,
        dismiss_calls,
        cancel_calls,
        import_state,
        video_state,
        document,
    ) = overlays_module

    import_state.update({
        "active": True,
        "dataset_type": "dataset",
        "path": "/tmp/demo",
        "progress": 0.25,
        "stage": "Scanning",
    })
    video_state.update({
        "active": True,
        "progress": 0.5,
        "current_frame": 12,
        "total_frames": 48,
        "stage": "Encoding",
    })

    module._sync_viewport_overlay_document(document)

    assert document.created_models == ["viewport_overlay_status"]
    assert document.model.handle.dirty_all_calls == 2
    assert document.body.get_attribute("data-viewport-overlay-status-bound", "") == "1"
    assert document.model.bound_funcs["show_import_overlay"]() is True
    assert document.model.bound_funcs["show_import_backdrop"]() is True
    assert document.model.bound_funcs["import_progress_pct"]() == "25%"
    assert document.model.bound_funcs["video_frame_text"]() == "Frame 12 / 48"

    document.model.bound_events["overlay_action"](None, None, ["dismiss_import"])
    document.model.bound_events["overlay_action"](None, None, ["cancel_video_export"])

    assert dismiss_calls == [True]
    assert cancel_calls == [True]


def test_import_completion_hides_backdrop(overlays_module):
    (
        module,
        _hook_calls,
        _remove_calls,
        _dismiss_calls,
        _cancel_calls,
        import_state,
        _video_state,
        document,
    ) = overlays_module

    import_state.update({
        "active": False,
        "show_completion": True,
        "success": True,
    })

    module._sync_viewport_overlay_document(document)

    assert document.model.bound_funcs["show_import_overlay"]() is True
    assert document.model.bound_funcs["show_import_backdrop"]() is False

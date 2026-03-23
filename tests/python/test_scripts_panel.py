# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression tests for the retained scripts panel."""

from importlib import import_module
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import pytest


def _install_lf_stub(monkeypatch):
    panel_space = SimpleNamespace(
        SIDE_PANEL="SIDE_PANEL",
        FLOATING="FLOATING",
        VIEWPORT_OVERLAY="VIEWPORT_OVERLAY",
        MAIN_PANEL_TAB="MAIN_PANEL_TAB",
        SCENE_HEADER="SCENE_HEADER",
        STATUS_BAR="STATUS_BAR",
    )
    panel_height_mode = SimpleNamespace(FILL="fill", CONTENT="content")
    panel_option = SimpleNamespace(DEFAULT_CLOSED="DEFAULT_CLOSED", HIDE_HEADER="HIDE_HEADER")
    state = SimpleNamespace(
        language=["en"],
        scripts=[],
        run_result={"success": True, "error": ""},
        run_calls=[],
        clear_calls=0,
        clear_errors_calls=0,
    )

    def get_scripts():
        return [
            dict(script)
            for script in state.scripts
        ]

    def set_script_enabled(index, enabled):
        state.scripts[index]["enabled"] = enabled

    def set_script_error(index, error):
        state.scripts[index]["has_error"] = bool(error)
        state.scripts[index]["error_message"] = error

    def clear_errors():
        state.clear_errors_calls += 1
        for script in state.scripts:
            script["has_error"] = False
            script["error_message"] = ""

    def clear():
        state.clear_calls += 1
        state.scripts.clear()

    def run(paths):
        state.run_calls.append(list(paths))
        return dict(state.run_result)

    def get_enabled_paths():
        return [script["path"] for script in state.scripts if script["enabled"]]

    lf_stub = ModuleType("lichtfeld")
    lf_stub.ui = SimpleNamespace(
        PanelSpace=panel_space,
        PanelHeightMode=panel_height_mode,
        PanelOption=panel_option,
        tr=lambda key: key,
        get_current_language=lambda: state.language[0],
    )
    lf_stub.scripts = SimpleNamespace(
        get_scripts=get_scripts,
        set_script_enabled=set_script_enabled,
        set_script_error=set_script_error,
        clear_errors=clear_errors,
        clear=clear,
        run=run,
        get_enabled_paths=get_enabled_paths,
    )
    monkeypatch.setitem(sys.modules, "lichtfeld", lf_stub)
    return state


@pytest.fixture
def scripts_panel_module(monkeypatch):
    project_root = Path(__file__).parent.parent.parent
    source_python = project_root / "src" / "python"
    if str(source_python) not in sys.path:
        sys.path.insert(0, str(source_python))
    sys.modules.pop("lfs_plugins.scripts_panel", None)
    sys.modules.pop("lfs_plugins", None)
    state = _install_lf_stub(monkeypatch)
    module = import_module("lfs_plugins.scripts_panel")
    return module, state


class _HandleStub:
    def __init__(self):
        self.records = {}
        self.dirty_fields = []
        self.dirty_all_calls = 0

    def update_record_list(self, name, rows):
        self.records[name] = rows

    def dirty(self, name):
        self.dirty_fields.append(name)

    def dirty_all(self):
        self.dirty_all_calls += 1


class _ElementStub:
    def __init__(self, attrs=None):
        self.attrs = dict(attrs or {})
        self._parent = None

    def has_attribute(self, name):
        return name in self.attrs

    def get_attribute(self, name, default_val=""):
        return self.attrs.get(name, default_val)

    def parent(self):
        return self._parent


class _EventStub:
    def __init__(self, target, current_target=None):
        self._target = target
        self._current_target = current_target or target
        self.stopped = False

    def target(self):
        return self._target

    def current_target(self):
        return self._current_target

    def stop_propagation(self):
        self.stopped = True


def test_scripts_panel_builds_retained_records(scripts_panel_module):
    module, state = scripts_panel_module
    panel = module.ScriptsPanel()
    panel._handle = _HandleStub()
    state.scripts[:] = [
        {
            "path": "/tmp/alpha.py",
            "enabled": True,
            "has_error": False,
            "error_message": "",
        },
        {
            "path": "/tmp/beta.py",
            "enabled": False,
            "has_error": True,
            "error_message": "boom",
        },
    ]

    assert panel._refresh_scripts(force=True) is True
    assert panel._has_scripts is True
    assert panel._handle.records["scripts"] == [
        {
            "index": "0",
            "filename": "alpha.py",
            "path": "/tmp/alpha.py",
            "enabled": True,
            "has_error": False,
            "error_message": "",
        },
        {
            "index": "1",
            "filename": "beta.py",
            "path": "/tmp/beta.py",
            "enabled": False,
            "has_error": True,
            "error_message": "boom",
        },
    ]


def test_scripts_panel_checkbox_change_updates_script_state(scripts_panel_module):
    module, state = scripts_panel_module
    panel = module.ScriptsPanel()
    panel._handle = _HandleStub()
    state.scripts[:] = [
        {
            "path": "/tmp/alpha.py",
            "enabled": False,
            "has_error": False,
            "error_message": "",
        },
    ]

    checkbox = _ElementStub({
        "data-script-index": "0",
        "checked": "",
    })
    container = _ElementStub()

    panel._on_scripts_change(_EventStub(checkbox, container))

    assert state.scripts[0]["enabled"] is True
    assert panel._handle.records["scripts"][0]["enabled"] is True


def test_scripts_panel_reload_all_preserves_old_behavior(scripts_panel_module):
    module, state = scripts_panel_module
    panel = module.ScriptsPanel()
    panel._handle = _HandleStub()
    state.scripts[:] = [
        {
            "path": "/tmp/alpha.py",
            "enabled": True,
            "has_error": False,
            "error_message": "",
        },
        {
            "path": "/tmp/beta.py",
            "enabled": False,
            "has_error": True,
            "error_message": "old",
        },
    ]
    state.run_result = {"success": False, "error": "reload failed"}

    panel._reload_all()

    assert state.clear_errors_calls == 1
    assert state.run_calls == [["/tmp/alpha.py"]]
    assert state.scripts[0]["error_message"] == "reload failed"
    assert state.scripts[0]["has_error"] is True
    assert state.scripts[1]["error_message"] == ""
    assert state.scripts[1]["has_error"] is False

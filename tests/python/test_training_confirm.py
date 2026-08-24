# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Outcomes for the shared unsaved-work confirmation helper."""

from importlib import import_module
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_training_confirm(monkeypatch):
    for module_name in list(sys.modules):
        if module_name == "lfs_plugins" or module_name.startswith("lfs_plugins."):
            monkeypatch.delitem(sys.modules, module_name, raising=False)

    package = ModuleType("lfs_plugins")
    package.__path__ = [str(PROJECT_ROOT / "src" / "python" / "lfs_plugins")]
    monkeypatch.setitem(sys.modules, "lfs_plugins", package)

    def tr(key):
        return f"tr:{key}"

    state = SimpleNamespace(
        dirty=False,
        has_path=False,
        training_active=False,
        project_save_result=True,
        project_save_as_result=True,
        confirm_dialogs=[],
        project_save_calls=[],
        project_save_as_calls=[],
        scheduled=[],
        proceed_calls=[],
    )

    def confirm_dialog(title, message, buttons, callback=None):
        state.confirm_dialogs.append((title, message, list(buttons), callback))

    def project_save(*args, **kwargs):
        state.project_save_calls.append((args, kwargs))
        return state.project_save_result

    def project_save_as(path="", wait=False):
        state.project_save_as_calls.append((path, wait))
        return state.project_save_as_result

    lf_stub = ModuleType("lichtfeld")
    lf_stub.ui = SimpleNamespace(
        tr=tr,
        confirm_dialog=confirm_dialog,
        schedule_on_ui_thread=state.scheduled.append,
    )
    lf_stub.project_is_dirty = lambda: state.dirty
    lf_stub.project_has_path = lambda: state.has_path
    lf_stub.is_training_active = lambda: state.training_active
    lf_stub.project_save = project_save
    lf_stub.project_save_as = project_save_as
    monkeypatch.setitem(sys.modules, "lichtfeld", lf_stub)

    module = import_module("lfs_plugins.training_confirm")
    module.lf = lf_stub
    return module, state


def _proceed_recorder(state):
    def _on_proceed(stop_training=False):
        state.proceed_calls.append(stop_training)

    return _on_proceed


def test_not_dirty_proceeds_immediately(monkeypatch):
    module, state = _load_training_confirm(monkeypatch)

    module.confirm_discard_work_then("Reset", _proceed_recorder(state))

    assert state.confirm_dialogs == []
    assert state.proceed_calls == [False]


def test_dirty_save_titled_then_proceeds(monkeypatch):
    module, state = _load_training_confirm(monkeypatch)
    state.dirty = True
    state.has_path = True

    module.confirm_discard_work_then("tr:menu.file.new_project", _proceed_recorder(state))

    assert len(state.confirm_dialogs) == 1
    title, message, buttons, callback = state.confirm_dialogs[0]
    assert title == "tr:menu.file.new_project"
    assert message == "tr:exit_popup.unsaved_warning"
    assert buttons == [
        "tr:common.save",
        "tr:unsaved_work.continue_without_saving",
        "tr:common.cancel",
    ]
    assert state.proceed_calls == []

    callback("tr:common.save")
    assert state.project_save_calls == [((), {"wait": True})]
    assert state.project_save_as_calls == []
    assert state.proceed_calls == [False]


def test_dirty_save_as_untitled_cancelled_picker(monkeypatch):
    module, state = _load_training_confirm(monkeypatch)
    state.dirty = True
    state.has_path = False
    state.project_save_as_result = False

    module.confirm_discard_work_then("Open", _proceed_recorder(state))

    _title, _message, buttons, callback = state.confirm_dialogs[0]
    assert buttons[0] == "tr:menu.file.save_project_as"

    callback("tr:menu.file.save_project_as")
    assert state.project_save_as_calls == [("", True)]
    assert state.proceed_calls == []
    assert state.scheduled == []


def test_dirty_save_as_untitled_schedules_until_bound(monkeypatch):
    module, state = _load_training_confirm(monkeypatch)
    state.dirty = True
    state.has_path = False
    state.project_save_as_result = True

    module.confirm_discard_work_then("Open", _proceed_recorder(state))
    _title, _message, _buttons, callback = state.confirm_dialogs[0]
    callback("tr:menu.file.save_project_as")

    assert state.proceed_calls == []
    assert len(state.scheduled) == 1

    state.has_path = True
    state.scheduled[0]()
    assert state.proceed_calls == [False]


def test_dirty_continue_without_saving(monkeypatch):
    module, state = _load_training_confirm(monkeypatch)
    state.dirty = True
    state.has_path = True

    module.confirm_discard_work_then("Reset", _proceed_recorder(state))
    _title, _message, _buttons, callback = state.confirm_dialogs[0]
    callback("tr:unsaved_work.continue_without_saving")

    assert state.project_save_calls == []
    assert state.proceed_calls == [False]


def test_dirty_cancel_does_not_proceed(monkeypatch):
    module, state = _load_training_confirm(monkeypatch)
    state.dirty = True

    module.confirm_discard_work_then("Reset", _proceed_recorder(state))
    _title, _message, _buttons, callback = state.confirm_dialogs[0]
    callback("tr:common.cancel")

    assert state.project_save_calls == []
    assert state.proceed_calls == []


def test_training_active_stop_confirm_passes_stop_training_true(monkeypatch):
    module, state = _load_training_confirm(monkeypatch)
    state.training_active = True

    module.confirm_discard_work_then("Reset", _proceed_recorder(state))

    assert state.proceed_calls == []
    assert len(state.confirm_dialogs) == 1
    title, message, buttons, callback = state.confirm_dialogs[0]
    assert title == "tr:project_switch.stop_training_title"
    assert message == "tr:project_switch.stop_training_message"
    assert buttons == ["tr:common.yes", "tr:common.no"]

    callback("tr:common.no")
    assert state.proceed_calls == []

    callback("tr:common.yes")
    assert state.proceed_calls == [True]


def test_dirty_then_training_stop_confirm(monkeypatch):
    module, state = _load_training_confirm(monkeypatch)
    state.dirty = True
    state.has_path = True
    state.training_active = True

    module.confirm_discard_work_then("New", _proceed_recorder(state))
    _title, _message, _buttons, dirty_callback = state.confirm_dialogs[0]
    dirty_callback("tr:unsaved_work.continue_without_saving")

    assert state.proceed_calls == []
    assert len(state.confirm_dialogs) == 2
    stop_title, _stop_message, stop_buttons, stop_callback = state.confirm_dialogs[1]
    assert stop_title == "tr:project_switch.stop_training_title"
    assert stop_buttons == ["tr:common.yes", "tr:common.no"]

    stop_callback("tr:common.yes")
    assert state.proceed_calls == [True]


def test_ask_stop_training_false_skips_stop_confirm(monkeypatch):
    module, state = _load_training_confirm(monkeypatch)
    state.dirty = True
    state.has_path = True
    state.training_active = True

    module.confirm_discard_work_then(
        "Reset", _proceed_recorder(state), ask_stop_training=False
    )
    _title, _message, _buttons, callback = state.confirm_dialogs[0]
    callback("tr:unsaved_work.continue_without_saving")

    assert len(state.confirm_dialogs) == 1
    assert state.proceed_calls == [False]

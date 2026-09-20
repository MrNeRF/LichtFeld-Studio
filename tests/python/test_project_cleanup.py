# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
from importlib import import_module
from types import SimpleNamespace

import pytest

from test_file_menu_recent import _load_file_menu


@pytest.fixture
def cleanup(monkeypatch, tmp_path):
    menu = _load_file_menu(monkeypatch)
    module = import_module("lfs_plugins.project_cleanup")
    path = tmp_path / "project.licht"
    path.write_bytes(b"project")
    state = dict(path=str(path), running=False, error="", generation=3)
    forms, updates, clean_calls, saves = [], [], [], []
    plan = SimpleNamespace(input_commit_uuid="commit-3", physical_size=1024 * 1024,
        retained_checkpoints=[SimpleNamespace(scng_bound=True), SimpleNamespace(scng_bound=False)],
        drop_checkpoints=SimpleNamespace(projected_size=100))
    menu.lf.project_poll_write = lambda: state.copy()
    menu.lf.project_has_path = lambda: True
    menu.lf.project_clean = lambda *args: clean_calls.append(args)
    menu.lf.project_cancel_cleanup = lambda: clean_calls.append("cancel")
    menu.lf.ui.form_dialog = lambda *args: forms.append(args) or True
    menu.lf.ui.form_dialog_update = lambda *args: updates.append(args) or bool(forms)
    menu.lf.ui.schedule_on_ui_thread = lambda callback: callback()
    menu.lf.ui.save_project_file_dialog = lambda *args: str(tmp_path / "copy.licht")
    menu.lf.ui.tr = lambda key: key
    menu.lf.io = SimpleNamespace(plan_reduce_size=lambda path: plan,
        inspect_project_details=lambda path: SimpleNamespace(card=SimpleNamespace(commit_uuid="commit-3"), save_history=[1, 2, 3]))
    monkeypatch.setattr(module.threading, "Thread", lambda target, **kwargs: SimpleNamespace(start=target))
    pending = []
    monkeypatch.setattr(module.ProjectCleanup, "schedule", lambda self, callback: pending.append(callback))
    c = module.ProjectCleanup()
    return SimpleNamespace(menu=menu, module=module, controller=c, state=state, forms=forms,
        updates=updates, calls=clean_calls, plan=plan, pending=pending, path=path)


def test_file_menu_offers_clean_project(monkeypatch):
    menu = _load_file_menu(monkeypatch)
    assert menu.CleanProjectOperator in menu._operator_classes


def test_cleanup_previews_then_preserves_resume_point(cleanup):
    c = cleanup.controller
    c.start()
    assert cleanup.calls == []
    assert c.plan.retained_checkpoints[0].scng_bound
    buttons = cleanup.updates[-1][1]
    assert [b["label"] for b in buttons] == ["project_cleanup.clean_here", "project_cleanup.copy", "common.cancel"]
    c.choose("project_cleanup.clean_here")
    assert cleanup.calls == [("", "commit-3")]
    assert c.cleaning
    c.cancel()
    assert cleanup.calls[-1] == "cancel"
    assert cleanup.forms[-1][3] == [{"label": "common.cancel", "disabled": True}]
    assert not c.finished


@pytest.mark.parametrize("reason", ["dirty", "training", "different_project", "write"])
def test_cleanup_rejects_changes_after_preview(cleanup, reason):
    cleanup.controller.start()
    if reason == "dirty":
        cleanup.menu.lf.project_is_dirty = lambda: True
    elif reason == "training":
        cleanup.menu.lf.is_training_active = lambda: True
    elif reason == "different_project":
        cleanup.state["path"] = "other.licht"
    else:
        cleanup.state["running"] = True
    cleanup.controller.choose("project_cleanup.clean_here")
    assert cleanup.calls == []
    assert cleanup.updates[-1][1] == [{"label": "common.ok"}]


def test_cleaned_copy_uses_new_destination_and_original_commit(cleanup):
    cleanup.controller.start()
    cleanup.controller.choose("project_cleanup.copy")
    assert cleanup.calls == [(str(cleanup.path.parent / "copy.licht"), "commit-3")]
    assert cleanup.path.read_bytes() == b"project"


@pytest.mark.parametrize("cancel", [False, True])
def test_cleaned_copy_never_overwrites_existing_file(cleanup, cancel):
    cleanup.menu.lf.ui.save_project_file_dialog = lambda *args: "" if cancel else str(cleanup.path)
    cleanup.controller.start()
    cleanup.controller.choose("project_cleanup.copy")
    assert cleanup.calls == []
    assert cleanup.path.read_bytes() == b"project"


def test_cancel_preview_does_not_start_cleanup(cleanup):
    cleanup.controller.start()
    cleanup.controller.choose("common.cancel")
    assert cleanup.controller.finished
    assert cleanup.calls == []


def test_failed_save_never_starts_cleanup(cleanup):
    cleanup.menu.lf.project_is_dirty = lambda: True
    cleanup.menu.lf.project_save = lambda **kwargs: False
    cleanup.controller.start()
    assert cleanup.forms == []
    cleanup.menu.lf.confirm_dialogs[0][3]("common.save")
    assert cleanup.calls == []
    assert cleanup.forms == []
    assert cleanup.menu.lf.message_dialogs


def test_saved_edits_are_finished_before_preview(cleanup):
    cleanup.menu.lf.project_is_dirty = lambda: True
    cleanup.menu.lf.project_save = lambda **kwargs: True
    cleanup.controller.start()
    cleanup.menu.lf.confirm_dialogs[0][3]("common.save")
    cleanup.state["running"] = True
    cleanup.pending.pop(0)()
    assert cleanup.forms == []
    cleanup.state["running"] = False
    cleanup.menu.lf.project_is_dirty = lambda: False
    cleanup.pending.pop(0)()
    assert cleanup.controller.plan is cleanup.plan
    assert cleanup.calls == []


def test_inspection_error_replaces_reading_dialog(cleanup):
    cleanup.menu.lf.io.plan_reduce_size = lambda path: (_ for _ in ()).throw(RuntimeError("Unreadable project"))
    cleanup.controller.start()
    assert cleanup.controller.finished
    assert "Unreadable project" in cleanup.updates[-1][2]
    assert cleanup.updates[-1][1] == [{"label": "common.ok"}]


def test_native_error_shows_plain_user_message(cleanup):
    cleanup.controller.start()
    cleanup.state["error"] = "lfs::Error[Cancelled/IO]\n  user_message: Project cleanup was canceled.\n  detail: internal"
    cleanup.controller.poll()
    assert cleanup.updates[-1][2] == '<div class="modal-note">Project cleanup was canceled.</div>'

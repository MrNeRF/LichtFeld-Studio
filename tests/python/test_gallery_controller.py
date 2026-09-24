# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Account boundaries and domain-sensitive actions in the Gallery controller."""
from importlib import import_module
from contextlib import nullcontext
from types import SimpleNamespace
from pathlib import Path
import copy
import json
import struct

import pytest

from test_asset_manager_panel import panel_module, _gallery_fixture, _stop_gallery_controller

@pytest.fixture
def gallery(monkeypatch, panel_module):
    module = import_module("lfs_plugins.gallery_controller")
    monkeypatch.setattr(module.lf, "scene", SimpleNamespace(NodeType=SimpleNamespace(SPLAT=0)), raising=False)
    state = {
        "identity": ("https://portal.example", "one@example.com", "first", True),
        "signed_in": True, "connected": True, "email": "one@example.com",
        "display_name": "One", "scenes": [], "jobs": [], "links": {},
        "message": "", "busy": False, "version": 0,
    }
    actions = []
    service = SimpleNamespace(snapshot=lambda: state.copy(), identity=lambda: state["identity"], busy=False, pause=lambda: None,
        edit=lambda *args: actions.append(args),
        local_use=lambda job_id: nullcontext())
    monkeypatch.setattr(module, "get_gallery_sync", lambda: service)
    monkeypatch.setattr(module.lf.ui, "cancel_export", lambda: actions.append("cancel-export"), raising=False)
    monkeypatch.setattr(module.lf.ui, "dismiss_import", lambda: actions.append("dismiss-import"), raising=False)
    monkeypatch.setattr(module.lf.ui, "get_import_state", lambda: {"active": False}, raising=False)
    native_io = import_module('lichtfeld.io')
    monkeypatch.setattr(module.lf, "io", SimpleNamespace(
        inspect_project=lambda path: SimpleNamespace(project_uuid="project", commit_uuid=""),
        project_content_stamp=native_io.project_content_stamp), raising=False)
    panel = module.GalleryController()
    yield panel, state, actions
    _stop_gallery_controller(panel)

def scene(**fields):
    return dict(id="private-one", title="My scene", description="Private description",
        visibility="private", revision="original", status="ready", **fields, contentRevision="original", metadataRevision="original")


@pytest.mark.parametrize("operation", ["contents", "settings"])
@pytest.mark.parametrize("swap", ["identity", "path"])
def test_gallery_apply_rechecks_project_after_scene_changes(gallery, monkeypatch, tmp_path, operation, swap):
    panel, state, actions = gallery
    module = import_module("lfs_plugins.gallery_controller")
    path = tmp_path / "项目.licht"
    path.write_text("project", encoding="utf-8")
    alias = tmp_path / "别名.licht"
    alias.symlink_to(path)
    monkeypatch.setattr(module.lf, "project_has_path", lambda: True, raising=False)
    monkeypatch.setattr(module.lf, "project_poll_write", lambda: {"path": str(alias), "generation": 1}, raising=False)
    monkeypatch.setattr(module.lf.io, "inspect_project", lambda p: SimpleNamespace(project_uuid=Path(p).read_text()))
    monkeypatch.setattr(module.lf, "project_is_dirty", lambda: False, raising=False)
    monkeypatch.setattr(module.lf, "project_save", lambda **kw: actions.append("saved") or True, raising=False)
    monkeypatch.setattr(module.lf, "set_node_visibility", lambda *args: None, raising=False)
    project = panel._project_identity()
    stamp = module.file_stamp(path)

    def swap_project(*args, **kwargs):
        if swap == "identity":
            path.write_text("other-project")
        else:
            other = tmp_path / "other.licht"
            other.write_text("project")
            alias.unlink()
            alias.symlink_to(other)

    if operation == "contents":
        monkeypatch.setattr(import_module("lfs_plugins.gallery_sync_steps"), "restore_view", lambda *args, **kwargs: None)
        panel.service.environment_path = lambda job: None
        scene_tree = SimpleNamespace(get_node=lambda name: None, rename_node=swap_project)
        incoming = SimpleNamespace(name="incoming", uuid="incoming-id")
        with pytest.raises(ValueError, match="identity or path changed before saving"):
            panel._apply_local_update(scene_tree, incoming, {"result": {"title": "Gallery"}},
                {"old_nodes": [], "project": project, "stamp": stamp})
    else:
        monkeypatch.setattr(module, "restore_view", swap_project)
        state["jobs"] = [{"id": "settings", "localUpdate": {"id": "backup", "state": "ready"}}]
        panel._settings_pending = dict(job="settings", project=project, identity=state["identity"],
            phase="backup", backup="backup", stamp=stamp, metadata={"viewerSettings": {}})
        with pytest.raises(ValueError, match="identity or path changed before saving"):
            panel._finish_settings_apply()
    assert "saved" not in actions


@pytest.mark.parametrize("open_project", [False, True])
@pytest.mark.parametrize("swap", ["identity", "path"])
def test_download_registration_refuses_changed_project(gallery, monkeypatch, tmp_path, open_project, swap):
    panel, state, _ = gallery
    module = import_module("lfs_plugins.gallery_controller")
    path = tmp_path / "项目.licht"
    path.write_text("downloaded-project")
    alias = tmp_path / "别名.licht"
    alias.symlink_to(path)
    stage = dict(id="stage", state="ready", projectPath=str(alias), projectId="downloaded-project",
                 projectStamp=module.file_stamp(alias))
    if swap == "identity":
        path.write_text("other-project")
        # A current stat must not substitute for the planned project UUID.
        stage["projectStamp"] = module.file_stamp(alias)
    else:
        other = tmp_path / "other.licht"
        other.write_bytes(path.read_bytes())
        alias.unlink()
        alias.symlink_to(other)
    monkeypatch.setattr(module.lf.io, "inspect_project", lambda p: SimpleNamespace(project_uuid=Path(p).read_text()))
    index = SimpleNamespace(load=lambda: True,
        register_licht_asset=lambda *a, **kw: pytest.fail("Changed project was registered"))
    monkeypatch.setattr(import_module("lfs_plugins.asset_index"), "AssetIndex", lambda: index)
    job = dict(id="download", result={"title": "Gallery"}, stagedImport=stage, _accountIdentity=state["identity"])
    state["jobs"] = [job]
    if open_project:
        job["_native_project"] = dict(path=str(alias), projectId=stage["projectId"], projectStamp=stage["projectStamp"], count=1)
        monkeypatch.setattr(module.lf, "get_scene", lambda: SimpleNamespace(total_gaussian_count=1), raising=False)
        monkeypatch.setattr(module.lf, "project_poll_write", lambda: {"path": str(alias)}, raising=False)
    else:
        job["_register"] = dict(phase="staging", stage_id="stage")
    panel._download_open_steps.pending = job
    with pytest.raises(ValueError, match="downloaded project identity.*changed"):
        panel._finish_import()


def test_download_link_keeps_the_planned_project_id(gallery):
    panel, _, actions = gallery
    panel.service.link_download = lambda *args, **kwargs: actions.append("linked")
    with pytest.raises(ValueError, match="identity changed before linking"):
        panel._link_saved_download("download", "/project.licht", "previous-project")
    assert "linked" not in actions

def test_per_frame_account_check_does_not_copy_transfer_history(gallery):
    panel, state, actions = gallery
    panel.service.snapshot = lambda: (_ for _ in ()).throw(AssertionError("History copied during account check"))
    panel.service.pause = lambda: actions.append("paused")
    assert panel._check_identity() is False
    state["identity"] = ("https://portal.example", "two@example.com", "second", True)
    assert panel._check_identity() is True
    assert actions == ["paused"]


def test_force_refresh_requested_while_busy_is_preserved(gallery, monkeypatch):
    panel, _state, actions = gallery
    panel.service.refresh = lambda **kwargs: actions.append(kwargs)
    panel.service.busy = True
    panel._refresh_pending = True
    monkeypatch.setattr(panel, "_schedule_poll", lambda: None)

    panel.refresh(force=True)

    assert panel._refresh_requested is True
    assert panel._refresh_force_requested is True
    panel.service.busy = False
    panel._poll_body()
    assert actions == [{"force": True}]

@pytest.mark.parametrize("native_active,expected", [(True, 50), (False, 40)])
def test_progress_painting_uses_the_existing_model_without_copying_history(gallery, monkeypatch, native_active, expected):
    panel, state, _ = gallery
    module = import_module("lfs_plugins.gallery_controller")
    panel._download_open_steps.pending = {"id": "download"}
    panel._state["jobs"] = [{"id": "download", "stagedImport": {"completed": 4, "total": 10}}]
    monkeypatch.setattr(panel, "_finish_import", lambda: None)
    monkeypatch.setattr(module.lf.ui, "get_import_state", lambda: {"active": native_active, "progress": .5}, raising=False)
    panel.service.snapshot = lambda: (_ for _ in ()).throw(AssertionError("History copied while painting progress"))
    panel._advance_phases()
    assert panel._download_open_steps.progress == expected
    assert panel._download_open_steps.pending == {"id": "download"}

@pytest.mark.parametrize("outcome", ["success", "canceled", "account", "edited", "generation", "project", "failed"])
def test_background_save_never_continues_before_its_unchanged_generation(gallery, monkeypatch, outcome):
    panel, state, actions = gallery
    module = import_module("lfs_plugins.gallery_controller")
    project = ["project", "/project.licht"]
    poll = {"running": False, "generation": 5, "path": project[1], "error": ""}
    monkeypatch.setattr(panel, "_project_identity", lambda: tuple(project))
    monkeypatch.setattr(module.lf, "project_poll_write", lambda: dict(poll), raising=False)
    monkeypatch.setattr(module.lf, "project_is_dirty", lambda: outcome == "edited", raising=False)
    monkeypatch.setattr(panel, "_schedule_poll", lambda: None)
    def save(**kwargs):
        assert kwargs == {"wait": False, "regenerate_preview": False}
        poll["running"] = True
        return True
    monkeypatch.setattr(module.lf, "project_save", save, raising=False)
    panel._save_current_project(lambda: actions.append("continued"))
    assert panel._save_pending
    panel._finish_current_project_save()
    assert not actions and panel._save_pending
    if outcome == "canceled":
        panel._action_pause()
    elif outcome == "account":
        state["identity"] = ("https://portal.example", "two@example.com", "second", True)
    elif outcome == "project":
        project[0] = "different-project"
    poll.update(running=False, generation=7 if outcome == "generation" else 6, error="disk full" if outcome == "failed" else "")
    if outcome in ("edited", "generation", "project", "failed"):
        with pytest.raises(ValueError):
            panel._finish_current_project_save()
    else:
        panel._finish_current_project_save()
    assert actions == (["continued"] if outcome == "success" else [])
    assert panel._save_pending is None

def test_project_open_rechecks_other_native_work_after_staging(gallery, monkeypatch):
    panel, state, actions = gallery
    module = import_module("lfs_plugins.gallery_controller")
    panel._download_open_steps.pending = {"id": "download", "_accountIdentity": state["identity"],
        "_opening": {"phase": "staging", "scene": SimpleNamespace(is_valid=lambda: True)}}
    monkeypatch.setattr(module.lf, "project_is_dirty", lambda: False, raising=False)
    monkeypatch.setattr(module.lf, "is_training_active", lambda: False, raising=False)
    monkeypatch.setattr(module.lf.ui, "get_import_state", lambda: {"active": True}, raising=False)
    monkeypatch.setattr(module.lf, "new_project", lambda: actions.append("closed project"), raising=False)
    with pytest.raises(ValueError, match="Finish training or the current import"):
        panel._finish_import()
    assert not actions


@pytest.mark.parametrize("finished_generation,allowed", [(8, True), (9, False)])
def test_publish_save_accounts_for_a_thumbnail_save_before_it_starts(gallery, monkeypatch, finished_generation, allowed):
    panel, state, actions = gallery
    module = import_module("lfs_plugins.gallery_controller")
    poll = dict(running=False, generation=5, path="/project.licht", error="")
    monkeypatch.setattr(panel, "_project_identity", lambda: ("project", poll["path"]))
    monkeypatch.setattr(panel, "_schedule_poll", lambda: None)
    monkeypatch.setattr(module.lf.io, "inspect_project", lambda _: SimpleNamespace(generation=7))
    monkeypatch.setattr(module.lf, "project_poll_write", lambda: dict(poll), raising=False)
    monkeypatch.setattr(module.lf, "project_is_dirty", lambda: False, raising=False)
    monkeypatch.setattr(module.lf, "project_save", lambda **_: True, raising=False)
    panel._save_current_project(lambda: actions.append("upload"))
    poll["generation"] = finished_generation
    if allowed:
        panel._finish_current_project_save()
        assert actions == ["upload"]
    else:
        with pytest.raises(ValueError, match="changed while saving"):
            panel._finish_current_project_save()
        assert not actions

def test_update_link_failure_reports_already_saved_project(gallery, monkeypatch, tmp_path):
    panel, state, actions = gallery
    module = import_module("lfs_plugins.gallery_controller")
    source = tmp_path / "project.licht"
    source.write_bytes(b"saved local generation")
    project = ("project", str(source))
    incoming = SimpleNamespace(uuid="incoming", name="preview")
    native_scene = SimpleNamespace(get_node=lambda name: incoming, get_node_by_uuid=lambda identifier: incoming)
    update = {"project": project, "phase": "applying", "path": str(tmp_path / "preview.scene"),
        "incoming": "incoming", "generation": 3, "stamp": module.file_stamp(source), "backup_id": "backup"}
    job = {"id": "download", "_update": update}
    state["jobs"] = [{"id": "download", "localUpdate": {"id": "backup", "state": "ready"}}]
    monkeypatch.setattr(panel, "_project_identity", lambda: project)
    monkeypatch.setattr(panel, "_apply_local_update", lambda *args: actions.append("project saved"))
    monkeypatch.setattr(module.lf, "get_scene", lambda: native_scene, raising=False)
    monkeypatch.setattr(module.lf, "project_is_dirty", lambda: False, raising=False)
    monkeypatch.setattr(module.lf, "project_poll_write", lambda: {"generation": 3}, raising=False)
    panel.service.link_download = lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("Account changed"))
    panel._finish_local_update(job)
    assert update["phase"] == "save_updated"
    monkeypatch.setattr(module.lf, "project_poll_write", lambda: {"generation": 4}, raising=False)
    with pytest.raises(ValueError, match="The project was updated and its recovery copy was kept"):
        panel._finish_local_update(job)
    assert actions == ["project saved"] and update["phase"] == "linking"


@pytest.mark.parametrize("suffix", [".sog", ".ssog"])
def test_gallery_update_rejects_raw_splat_stage(gallery, monkeypatch, tmp_path, suffix):
    panel, state, _ = gallery
    module = import_module("lfs_plugins.gallery_controller")
    project = ("project", str(tmp_path / "current.licht"))
    update = {"project": project, "phase": "staging", "stage_id": "stage"}
    job = {"id": "download", "_update": update}
    state["jobs"] = [{"id": "download", "stagedImport": {
        "id": "stage", "state": "ready", "path": str(tmp_path / ("download" + suffix))}}]
    calls = []
    monkeypatch.setattr(panel, "_project_identity", lambda: project)
    monkeypatch.setattr(module.lf, "get_scene", lambda: SimpleNamespace(get_node=lambda _: None), raising=False)
    monkeypatch.setattr(module.lf, "load_file", lambda path: calls.append(path), raising=False)

    with pytest.raises(ValueError, match="downloaded project identity or path changed"):
        panel._finish_local_update(job)

    assert not calls


@pytest.mark.parametrize("outcome", ["success", "edited", "generation", "project", "account", "failed"])
def test_update_final_save_links_only_its_clean_committed_project(gallery, monkeypatch, outcome):
    panel, state, actions = gallery
    module = import_module("lfs_plugins.gallery_controller")
    project = ("project", "/project.licht")
    job = {"id": "download", "_accountIdentity": state["identity"],
        "_update": {"project": project, "phase": "save_updated", "generation": 3}}
    panel._local_update_steps.pending = job
    poll = {"running": True, "generation": 3, "error": ""}
    monkeypatch.setattr(module.lf, "project_poll_write", lambda: poll.copy(), raising=False)
    monkeypatch.setattr(module.lf, "project_is_dirty", lambda: outcome == "edited", raising=False)
    monkeypatch.setattr(panel, "_project_identity", lambda: ("other", "/other.licht") if outcome == "project" else project)
    monkeypatch.setattr(panel, "_recover_failed_update", lambda *a: (_ for _ in ()).throw(ValueError("recovery available")))
    panel.service.link_download = lambda *a, **kw: actions.append("linked") or "operation"
    panel._finish_update_save(job)
    assert not actions
    poll.update(running=False, generation=5 if outcome == "generation" else 4, error="disk full" if outcome == "failed" else "")
    if outcome == "account":
        state["identity"] = ("https://portal.example", "two@example.com", "second", True)
    if outcome in ("edited", "generation", "failed"):
        with pytest.raises(ValueError):
            panel._finish_import()
    else:
        panel._finish_import()
    assert actions == (["linked"] if outcome == "success" else [])
    if outcome == "project":
        assert "after the gallery update" in panel._message

@pytest.mark.parametrize("phase,changed,removed", [("save_before_backup", False, True), ("backup", False, True),
    ("save_before_backup", True, False), ("applying", False, False), ("save_updated", False, False)])
def test_failed_update_preparation_removes_only_its_owned_preview(gallery, monkeypatch, phase, changed, removed):
    panel, state, actions = gallery
    module = import_module("lfs_plugins.gallery_controller")
    project = ("project", "/project.licht")
    panel._local_update_steps.pending = {"id": "download", "_update": {"project": project, "phase": phase, "incoming": "owned-uuid"}}
    panel._save_pending = {"pending": True}
    monkeypatch.setattr(panel, "_finish_current_project_save", lambda: (_ for _ in ()).throw(ValueError("disk full")))
    monkeypatch.setattr(panel, "_project_identity", lambda: ("other", "/other.licht") if changed else project)
    monkeypatch.setattr(panel, "_refresh_model", lambda: None)
    def lookup(identifier):
        assert identifier == "owned-uuid"
        return SimpleNamespace(name="owned preview")
    monkeypatch.setattr(module.lf, "get_scene", lambda: SimpleNamespace(get_node_by_uuid=lookup,
        remove_node=lambda name: actions.append(name)), raising=False)
    panel._advance_phases()
    assert actions == (["owned preview"] if removed else [])
    assert panel._local_update_steps.pending is None and panel._save_pending is None
    assert "disk full" in panel._message

def test_recovery_folder_action_reveals_only_service_folder(gallery, tmp_path, monkeypatch):
    panel, state, actions = gallery
    module = import_module("lfs_plugins.gallery_controller")
    panel.service.root = tmp_path / "gallery"
    state.update(storage_issue=True, connected=False)
    monkeypatch.setattr(module.lf.ui, "reveal_in_file_manager", lambda path: actions.append(path), raising=False)
    panel._dispatch("show_recovery_folder", [])
    assert actions == [str(panel.service.root)]


def test_corrupt_journal_does_not_mark_every_project_as_failed(gallery):
    from lfs_plugins.gallery_controller import asset_sync_state
    facts = asset_sync_state(dict(id="project", exists=True), storage_issue=True, established=False)
    assert facts["state"] == "not_checked"
    assert facts["attention"] is False
    assert facts["actions"] == []

def test_native_lease_survives_detach_until_import_idle(gallery, monkeypatch):
    from contextlib import contextmanager
    panel, _, actions = gallery
    module = import_module("lfs_plugins.gallery_controller")
    native = {"active": True}

    @contextmanager
    def lease(_):
        actions.append("acquired")
        try:
            yield
        finally:
            actions.append("released")

    panel.service.local_use = lease
    monkeypatch.setattr(module.lf.ui, "get_import_state", lambda: native, raising=False)
    monkeypatch.setattr(panel, "_schedule_poll", lambda: actions.append("poll"))
    panel._acquire_native_use("job")
    panel._download_open_steps.detached = True
    panel._release_native_use()
    assert actions == ["acquired", "poll"]
    native["active"] = False
    panel._acquire_native_use("retry")
    assert actions[-2:] == ["released", "acquired"]
    panel._release_native_use()
    assert actions[-1] == "released" and panel._native_use is None

def test_account_switch_cancels_pending_scene_preparation(gallery, monkeypatch):
    panel, state, actions = gallery
    panel._publish_steps.pending = ("private.ply", {}, "project", 0)
    module = import_module("lfs_plugins.gallery_controller")
    monkeypatch.setattr(module.lf.ui, "get_export_state", lambda: {"active": True, "path": "private.ply"}, raising=False)
    state["identity"] = ("https://portal.example", "two@example.com", "second", True)
    panel._refresh_model()
    assert actions == ["cancel-export"]
    assert panel._publish_steps.cancelled

def test_failed_export_with_partial_file_is_never_uploaded(gallery, tmp_path, monkeypatch):
    panel, _, actions = gallery
    module = import_module("lfs_plugins.gallery_controller")
    export = tmp_path / "partial.ply"
    monkeypatch.setattr(module.lf.ui, "get_export_state", lambda: {"active": False, "outcome": "failed", "path": str(export)}, raising=False)
    export.write_bytes(b"unfinished export")
    panel._publish_steps.pending = (export, {}, "project", 0)
    panel._finish_export()
    assert panel._publish_steps.pending is None
    assert not export.exists()
    assert actions == []
    assert "failed" in panel._message

def test_preparation_progress_tracks_own_export(gallery, tmp_path, monkeypatch):
    panel, _, actions = gallery
    module = import_module("lfs_plugins.gallery_controller")
    export = tmp_path / "own.ply"
    panel._publish_steps.pending = (export, {}, "project", 0)
    monkeypatch.setattr(module.lf.ui, "get_export_state", lambda: {"active": True, "path": str(export), "progress": 0.42}, raising=False)
    panel._finish_export()
    assert panel._publish_steps.progress == 42
    assert "42%" in panel._message
    assert panel._publish_steps.pending is not None and not actions

def test_publish_prepares_the_saved_project_off_thread(gallery, tmp_path, monkeypatch):
    panel, state, actions = gallery
    module = import_module("lfs_plugins.gallery_controller")
    state["source_formats"] = ["ply", "licht"]
    panel.service.root = tmp_path
    monkeypatch.setattr(module.lf.io, "inspect_project", lambda _: SimpleNamespace(commit_uuid="saved"))
    monkeypatch.setattr(panel, "_project_identity", lambda: ("project", "/project.licht"))
    monkeypatch.setattr(panel, "_save_current_project", lambda proceed: proceed())
    monkeypatch.setattr(panel, "_schedule_poll", lambda: None)
    monkeypatch.setattr(module.lf, "get_scene", lambda: SimpleNamespace(get_nodes=lambda: [SimpleNamespace(id=1, name="scene", type=module.lf.scene.NodeType.SPLAT)], is_node_effectively_visible=lambda node_id: True), raising=False)
    monkeypatch.setattr(module.lf.ui, "get_export_state", lambda: {"active": False}, raising=False)
    monkeypatch.setattr(module.lf, "prepare_gallery_project", lambda *args: actions.append(args), raising=False)
    monkeypatch.setattr(module.lf, "export_scene", lambda *args, **kwargs: pytest.fail("Must preserve local multi-object geometry"), raising=False)
    panel._publish({"title": "Scene"})
    assert panel._publish_steps.pending[0].suffix == ".scene"
    assert actions == [("/project.licht", str(panel._publish_steps.pending[0]), "ply", "saved")]


def test_publish_clean_open_project_uses_saved_commit_with_live_view(gallery, monkeypatch, tmp_path):
    panel, state, actions = gallery
    module = import_module("lfs_plugins.gallery_controller")
    path = tmp_path / "project-a.licht"
    path.write_bytes(b"saved project")
    monkeypatch.setattr(panel, "_project_identity", lambda: ("project", str(path)))
    monkeypatch.setattr(panel, "_visible_splats", lambda: [SimpleNamespace(name="visible")])
    monkeypatch.setattr(panel, "_schedule_poll", lambda: None)
    monkeypatch.setattr(module.lf.io, "inspect_project", lambda _: SimpleNamespace(project_uuid="project", commit_uuid="saved"))
    monkeypatch.setattr(module.lf, "project_is_dirty", lambda: False, raising=False)
    monkeypatch.setattr(module.lf, "project_poll_write", lambda: {"path": str(path), "generation": 1, "running": False}, raising=False)
    monkeypatch.setattr(module.lf, "project_save", lambda **kwargs: pytest.fail("A clean publish saved the project"), raising=False)
    monkeypatch.setattr(panel, "_publish_saved", lambda metadata, *args, **kwargs: actions.append((metadata, args, kwargs)))
    view = {"camera": {"position": [4, 2, 4]}}
    monkeypatch.setattr(module, "capture_view", lambda _: view)
    panel._review_publish(None, {"title": "Current view", "description": "", "saveProject": False}, "sog", False)
    assert actions and actions[0][0]["viewerSettings"] == view
    assert actions[0][2]["expected_commit"] == "saved"


def test_publish_without_save_prepares_saved_commit_and_live_view(gallery, monkeypatch, tmp_path):
    panel, state, actions = gallery
    module = import_module("lfs_plugins.gallery_controller")
    path = tmp_path / "project-a.licht"
    path.write_bytes(b"saved project")
    panel.service.root = tmp_path
    state["source_formats"] = ["licht"]
    monkeypatch.setattr(panel, "_project_identity", lambda: ("project", str(path)))
    monkeypatch.setattr(panel, "_visible_splats", lambda: [SimpleNamespace(name="visible")])
    monkeypatch.setattr(panel, "_schedule_poll", lambda: None)
    monkeypatch.setattr(panel, "_patch_saved_update", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(module.lf.io, "inspect_project", lambda _: SimpleNamespace(project_uuid="project", commit_uuid="saved"))
    monkeypatch.setattr(module.lf.io, "inspect_project_details", lambda _: SimpleNamespace(references=[]), raising=False)
    monkeypatch.setattr(module.lf, "project_poll_write", lambda: {"path": str(path), "running": False}, raising=False)
    monkeypatch.setattr(module.lf, "project_save", lambda **kwargs: pytest.fail("Publishing saved the project"), raising=False)
    monkeypatch.setattr(module.lf.ui, "get_export_state", lambda: {"active": False}, raising=False)
    monkeypatch.setattr(module.lf, "prepare_gallery_project", lambda *args: actions.append(args), raising=False)
    view = {"camera": {"position": [4, 2, 4]}}

    panel._publish({"title": "Current view", "viewerSettings": view},
                   expected_project=("project", str(path)), upload_format="sog",
                   save_project=False, expected_commit="saved")

    assert actions == [(str(path), str(panel._publish_steps.pending[0]), "sog", "saved")]
    assert panel._publish_steps.pending[1]["_commitUuid"] == "saved"
    assert panel._publish_steps.pending[1]["viewerSettings"] == view


@pytest.mark.parametrize("save_project", [False, True])
def test_publish_dirty_open_project_respects_save_choice(gallery, monkeypatch, tmp_path, save_project):
    panel, _, actions = gallery
    module = import_module("lfs_plugins.gallery_controller")
    path = tmp_path / "project-a.licht"
    path.write_bytes(b"saved project")
    monkeypatch.setattr(panel, "_project_identity", lambda: ("project", str(path)))
    monkeypatch.setattr(panel, "_visible_splats", lambda: [SimpleNamespace(name="visible")])
    monkeypatch.setattr(panel, "_schedule_poll", lambda: None)
    monkeypatch.setattr(module.lf.io, "inspect_project", lambda _: SimpleNamespace(project_uuid="project", commit_uuid="saved"))
    monkeypatch.setattr(module.lf.io, "inspect_project_details", lambda _: SimpleNamespace(references=[]), raising=False)
    monkeypatch.setattr(module.lf, "project_is_dirty", lambda: True, raising=False)
    monkeypatch.setattr(module.lf, "project_poll_write", lambda: {"path": str(path), "running": False}, raising=False)
    monkeypatch.setattr(module, "capture_view", lambda _: {"camera": {"position": [4, 2, 4]}})
    monkeypatch.setattr(panel, "_save_current_project", lambda continuation: (actions.append("saved"), continuation()))
    monkeypatch.setattr(panel, "_publish_saved", lambda metadata, *args, **kwargs: actions.append((metadata, kwargs)))

    panel._review_publish(None, {"title": "Current view", "description": "", "saveProject": save_project}, "sog", False)

    assert ("saved" in actions) is save_project
    published = next(action for action in actions if isinstance(action, tuple) and isinstance(action[0], dict))
    assert published[0]["viewerSettings"]["camera"]["position"] == [4, 2, 4]
    if not save_project:
        assert published[1]["expected_commit"] == "saved"


@pytest.mark.parametrize("open_project, dirty", [(True, False), (True, True), (False, False)])
def test_publish_review_save_choice_only_for_open_project(gallery, monkeypatch, tmp_path, open_project, dirty):
    from lfs_plugins.gallery_file_panel import GalleryFilePanel
    module = import_module("lfs_plugins.gallery_file_panel")
    controller, _, _ = gallery
    path = tmp_path / "project-a.licht"
    path.write_bytes(b"saved project")
    monkeypatch.setattr(module.lf, "project_poll_write", lambda: {"path": str(path) if open_project else ""}, raising=False)
    monkeypatch.setattr(module.lf, "project_is_dirty", lambda: dirty, raising=False)
    monkeypatch.setattr(module.lf.ui, "get_panel_object", lambda _: None, raising=False)
    monkeypatch.setattr(module.lf.ui, "set_panel_enabled", lambda *_: None, raising=False)
    monkeypatch.setattr(module.lf.ui, "request_redraw", lambda: None, raising=False)
    panel = GalleryFilePanel()
    panel.show(controller=controller, asset={"id": "project", "path": str(path), "name": "Project"},
               scene=None, action="publish", fields={"title": "Project", "description": "", "upload_format": "sog"})
    assert panel._review["open_project"] is open_project
    assert ("save_project" in panel._fields) is open_project
    if open_project:
        assert panel._fields["save_project"] is dirty
        panel._set("save_project", False)
        assert (panel._review["open_project"] and not panel._fields["save_project"] and dirty) is dirty
    rml = (Path(__file__).resolve().parents[2] / "src/visualizer/gui/rmlui/resources/gallery_file_panel.rml").read_text()
    assert 'data-if="show_save_project"' in rml
    assert 'data-if="show_unsaved_hint"' in rml


def test_publish_without_save_rechecks_saved_commit(gallery, monkeypatch, tmp_path):
    panel, state, actions = gallery
    module = import_module("lfs_plugins.gallery_controller")
    path = tmp_path / "project-a.licht"
    path.write_bytes(b"saved project")
    monkeypatch.setattr(panel, "_project_identity", lambda: ("project", str(path)))
    monkeypatch.setattr(module.lf.io, "inspect_project", lambda _: SimpleNamespace(commit_uuid="changed"))
    monkeypatch.setattr(panel, "_patch_saved_update", lambda *_args, **_kwargs: pytest.fail("Changed file was published"))
    with pytest.raises(ValueError, match="error.project_changed"):
        panel._publish_saved({"title": "Current view"}, "project", str(path), state["identity"],
                             expected_commit="saved")
    assert not actions


@pytest.mark.parametrize("saved_hdr", [False, True])
def test_publish_without_save_requires_saved_hdr_background(gallery, monkeypatch, tmp_path, saved_hdr):
    panel, state, actions = gallery
    module = import_module("lfs_plugins.gallery_controller")
    path = tmp_path / "project-a.licht"
    path.write_bytes(b"saved project")
    monkeypatch.setattr(panel, "_project_identity", lambda: ("project", str(path)))
    monkeypatch.setattr(module.lf.io, "inspect_project", lambda _: SimpleNamespace(commit_uuid="saved"))
    references = [SimpleNamespace(kind="environment_map", path=tmp_path / "background.hdr")] if saved_hdr else []
    monkeypatch.setattr(module.lf.io, "inspect_project_details", lambda _: SimpleNamespace(references=references), raising=False)
    monkeypatch.setattr(panel, "_patch_saved_update", lambda *_args, **_kwargs: pytest.fail("Unsaved HDR was published"))
    view = {} if saved_hdr else {"environment": {"exposure": 1, "rotation": 0}}
    with pytest.raises(ValueError, match="error.save_hdr_first"):
        panel._publish_saved({"title": "Current view", "viewerSettings": view},
                             "project", str(path), state["identity"], environment_source=None if saved_hdr else str(tmp_path / "background.hdr"),
                             expected_commit="saved")
    assert not actions

def test_completed_native_scene_hands_off_to_background_packaging(gallery, tmp_path, monkeypatch):
    import uuid
    panel, _, actions = gallery
    module = import_module("lfs_plugins.gallery_controller")
    export = tmp_path / (str(uuid.uuid4()) + ".scene")
    export.mkdir()
    panel._publish_steps.pending = (export, {"title": "Scene"}, "project", 0)
    panel.service.queue_prepared_upload = lambda *args: actions.append(args)
    monkeypatch.setattr(module.lf.ui, "get_export_state", lambda: {"active": False, "outcome": "completed", "path": str(export)}, raising=False)
    panel._finish_export()
    assert panel._publish_steps.pending is None
    assert actions == [(export, {"title": "Scene"}, "project")]
    assert export.exists()

def test_rejected_handoff_cleans_only_the_unaccepted_native_snapshot(gallery, tmp_path, monkeypatch):
    import uuid
    from lfs_plugins.gallery_sync import GallerySync
    panel, _, _ = gallery
    module = import_module("lfs_plugins.gallery_controller")
    export = tmp_path / (str(uuid.uuid4()) + ".scene")
    export.mkdir()
    (export / "manifest.json").write_text("{}")
    (export / "0.ply").write_bytes(b"snapshot")
    kept = tmp_path / "user.ply"
    kept.write_bytes(b"keep")
    panel.service.root = tmp_path
    panel.service._unlink_temporary = GallerySync._unlink_temporary
    panel._publish_steps.pending = (export, {"title": "Scene"}, "project", 0)
    panel.service.queue_prepared_upload = lambda *args: (_ for _ in ()).throw(ValueError("Portal changed; refresh first."))
    monkeypatch.setattr(module.lf.ui, "get_export_state", lambda: {"active": False, "outcome": "completed", "path": str(export)}, raising=False)
    panel._finish_export()
    assert panel._publish_steps.pending is None and not export.exists()
    assert kept.read_bytes() == b"keep"
    assert "Portal changed" in panel._message

def test_gallery_does_not_cancel_or_consume_another_export(gallery, tmp_path, monkeypatch):
    panel, _, actions = gallery
    module = import_module("lfs_plugins.gallery_controller")
    export = tmp_path / "own.ply"
    export.write_bytes(b"unconsumed previous preparation")
    panel._publish_steps.pending = (export, {}, "project", 0)
    monkeypatch.setattr(module.lf.ui, "get_export_state", lambda: {"active": True, "path": str(tmp_path / "other.ply")}, raising=False)
    panel._action_pause()
    assert actions == []
    panel._finish_export()
    assert panel._publish_steps.pending is None
    assert not export.exists()
    assert "prepare your upload again" in panel._message

@pytest.mark.parametrize("remote", [None, scene()])
def test_publish_without_visibility_keeps_captured_details(gallery, monkeypatch, remote):
    controller, state, actions = gallery
    monkeypatch.setattr(controller, "_project_identity", lambda: ("project", "/project.licht"))
    prompts = []
    module = import_module("lfs_plugins.gallery_controller")
    monkeypatch.setattr(module.lf.ui, "confirm_dialog", lambda *args: prompts.append(args), raising=False)
    details = dict(title="My scene", description="Description")
    monkeypatch.setattr(module, "capture_view", lambda _: {})
    monkeypatch.setattr(controller, "_publish", lambda metadata, **kw: actions.append(metadata))
    controller._review_publish(remote, details, "sog", False, update=bool(remote))
    details["title"] = "Another title entered after submission"
    assert not prompts and len(actions) == 1
    assert actions[0] == dict(title="My scene", description="Description", viewerSettings={},
        **({"replaceSceneId": remote["id"], "baseRevisions": {"content": "original", "metadata": "original"}} if remote else {}))


def test_unlink_confirmation_and_cancel_never_save_the_project(gallery, panel_module, monkeypatch):
    controller, _, actions = gallery
    manager, local, _ = _gallery_fixture(panel_module)
    manager._gallery_controller = controller
    manager._select_asset_id(local['id'])
    prompts = []
    monkeypatch.setattr(panel_module.lf.ui, "confirm_dialog", lambda *args: prompts.append(args), raising=False)
    monkeypatch.setattr(panel_module.lf, "project_save", lambda **kw: actions.append("saved"), raising=False)
    controller.service.unlink = lambda project: actions.append(("unlinked", project))
    manager._gallery_command('unlink')
    assert prompts[-1][1].endswith('confirm.unlink')
    controller.service.busy = True
    prompts[-1][-1](prompts[-1][-2][-1])
    assert actions == []
    controller.service.busy = False
    manager._gallery_command('unlink')
    prompts[-1][-1](prompts[-1][-2][0])
    assert actions == []
    manager._gallery_command('unlink')
    prompts[-1][-1](prompts[-1][-2][-1])
    assert actions == [("unlinked", local['id'])]

def test_wrong_replacement_is_rejected_before_saving_or_exporting(gallery, monkeypatch):
    panel, state, actions = gallery
    state["links"] = {"project": {"sceneId": "original"}}
    monkeypatch.setattr(panel, "_project_identity", lambda: ("project", "saved.licht"))
    monkeypatch.setattr(panel, "_save_current_project", lambda proceed: actions.append("saved"))
    panel._refresh_model()
    with pytest.raises(ValueError, match="linked to a gallery item"):
        panel._publish({"replaceSceneId": "other"})
    assert actions == []

@pytest.mark.parametrize("outcome", ["ready", "failed", "different-operation", "account-changed"])
def test_import_only_reports_linked_after_its_own_link_is_persisted(gallery, outcome):
    panel, state, _ = gallery
    job = {"id": "download", "_accountIdentity": state["identity"], "_link": "link-operation"}
    panel._download_open_steps.pending = job
    panel._message = "Downloaded scene saved. Saving its gallery link…"
    panel.service.busy = True
    panel._finish_import()
    assert panel._download_open_steps.pending is job
    assert "Saving its gallery link" in panel._message

    panel.service.busy = False
    state["jobs"] = [{"id": "download", "kind": "download", "status": "completed", "project": "project",
        "completed": 100, "total": 100, "metadata": scene(), "message": "", "result": scene(),
        "linkOperation": {"id": "other" if outcome == "different-operation" else "link-operation",
            "state": "failed" if outcome == "failed" else "ready"}}]
    if outcome == "account-changed":
        state.update(identity=("https://portal.example", "two@example.com", "second", True), jobs=[])
        panel._check_identity()
    panel._finish_import()
    assert panel._download_open_steps.pending is None
    assert ("saved and linked" in panel._message) == (outcome == "ready")
    assert "saved" in panel._message

def test_account_switch_during_project_save_prevents_export(gallery, monkeypatch):
    panel, state, _ = gallery
    module = import_module("lfs_plugins.gallery_controller")
    monkeypatch.setattr(panel, "_project_identity", lambda: ("project", "/project.licht"))
    monkeypatch.setattr(module.lf, "get_scene", lambda: SimpleNamespace(get_nodes=lambda: [SimpleNamespace(id=1, name="scene", type=module.lf.scene.NodeType.SPLAT)], is_node_effectively_visible=lambda node_id: True), raising=False)
    def save(proceed):
        state["identity"] = ("https://portal.example", "two@example.com", "second", True)
        proceed()
    monkeypatch.setattr(panel, "_save_current_project", save)
    with pytest.raises(ValueError, match="account or current project changed while saving"):
        panel._publish({"title": "Private scene"})
    assert panel._publish_steps.pending is None

def test_account_switch_during_save_prevents_opening_download(gallery, monkeypatch):
    panel, state, _ = gallery
    module = import_module("lfs_plugins.gallery_controller")
    monkeypatch.setattr(module.lf, "project_has_path", lambda: True)
    def save(proceed):
        state.update(identity=("https://portal.example", "two@example.com", "second", True))
        proceed()
    monkeypatch.setattr(panel, "_save_current_project", save)
    with pytest.raises(ValueError, match="account changed while saving"):
        panel._import_download({"path": "/private.licht"})
    assert panel._download_open_steps.pending is None

@pytest.mark.parametrize("account_changed", [False, True])
def test_local_update_keeps_geometry_if_user_edits_or_changes_account(gallery, monkeypatch, account_changed):
    panel, state, actions = gallery
    module = import_module("lfs_plugins.gallery_controller")
    project = ("project", "/project.licht")
    job = {"id": "download", "_accountIdentity": state["identity"],
        "_update": {"project": project, "phase": "backup", "backup_id": "backup"}}
    state["jobs"] = [{"id": "download", "localUpdate": {"id": "backup", "state": "ready"}}]
    panel._local_update_steps.pending = job
    monkeypatch.setattr(panel, "_project_identity", lambda: project)
    monkeypatch.setattr(module.lf, "get_scene", lambda: SimpleNamespace(remove_node=lambda *a, **k: actions.append("removed")), raising=False)
    monkeypatch.setattr(module.lf, "project_is_dirty", lambda: True)
    if account_changed:
        state["identity"] = ("https://portal.example", "two@example.com", "second", True)
        panel._finish_import()
        assert panel._local_update_steps.pending is None
        assert "existing local splats remain" in panel._message
    else:
        with pytest.raises(ValueError, match="local project changed during preparation"):
            panel._finish_import()
    assert actions == []

def test_partial_local_update_reopens_the_unchanged_saved_project(gallery, monkeypatch):
    panel, state, _ = gallery
    module = import_module("lfs_plugins.gallery_controller")
    project = ("project", "/project.licht")
    update = {"project": project, "phase": "applying", "backup_id": "backup",
        "generation": 3, "stamp": [1, 2], "incoming": "incoming"}
    job = {"id": "download", "_update": update}
    state["jobs"] = [{"id": "download", "localUpdate": {"id": "backup", "state": "ready"}}]
    monkeypatch.setattr(panel, "_project_identity", lambda: project)
    monkeypatch.setattr(module.lf, "get_scene", lambda: SimpleNamespace(get_node_by_uuid=lambda _: object()), raising=False)
    monkeypatch.setattr(module.lf, "project_is_dirty", lambda: False)
    monkeypatch.setattr(module.lf, "project_poll_write", lambda: {"generation": 3}, raising=False)
    monkeypatch.setattr(import_module("lfs_plugins.gallery_sync_steps"), "file_stamp", lambda _: [1, 2])
    monkeypatch.setattr(panel, "_apply_local_update", lambda *args: (_ for _ in ()).throw(OSError("save failed")))
    with pytest.raises(ValueError, match="saved local project is being reopened"):
        panel._finish_local_update(job)
    assert module.lf._test_state.opened == [("/project.licht", True, False, True)]

def test_overlay_rows_track_processing_pause_completion_and_cleared_recovery(gallery, monkeypatch):
    panel, state, _ = gallery
    from lfs_plugins.gallery_transfer_overlay import GalleryTransferOverlay
    from test_asset_manager_panel import _Handle
    module = import_module('lfs_plugins.gallery_controller')
    monkeypatch.setattr(module.lf.ui, 'is_panel_enabled', lambda _name: False, raising=False)
    overlay = GalleryTransferOverlay()
    overlay._handle = _Handle()

    def rows():
        panel._publish_runtime_state(panel.snapshot())
        assert overlay.update()
        return overlay._handle.records['gallery_transfer_rows']

    job = dict(id='transfer', kind='upload', metadata={'title': 'Private garden'},
               status='running', serverProcessing=True, completed=40, total=100, message='Checking scene')
    state['jobs'] = [job]
    progress = rows()[0]
    assert progress['progress'] == 40 and progress['can_pause']
    assert progress['phase'].endswith('phase.processing')
    job.update(status='paused', message='Stopped waiting')
    panel._state = dict(state, jobs=[dict(job, status='running')])
    progress = rows()[0]
    assert progress['can_resume'] and not progress['can_pause']
    job.update(status='completed', serverProcessing=False)
    assert rows()[0]['progress'] == 100
    job.update(retired=True, total=0, completed=0, message='Transfer cleared. Recovery copy kept.')
    assert rows() == []
    assert state['jobs'][0]['retired']  # Recovery record remains in the journal.

@pytest.mark.parametrize('commit,remote_title,expected', [
    ('saved', 'My scene', 'equal'), ('new-save', 'My scene', 'local'),
    ('saved', 'Portal edit', 'remote'), ('new-save', 'Portal edit', 'diverged'), ('', 'My scene', 'unknown')])
def test_three_fact_freshness_uses_commit_and_shared_fields(gallery, commit, remote_title, expected):
    from lfs_plugins.gallery_controller import asset_sync_state
    from lfs_plugins.gallery_sync import shared_fields
    base = scene()
    link = dict(sceneId=base['id'], revision='original', commitUuid='saved', sharedFields=shared_fields(base), contentRevision='original', metadataRevision=base['title'])
    remote = dict(base, title=remote_title, revision='cover-edited', metadataRevision=remote_title)
    facts = asset_sync_state(dict(id='project', commit_uuid=commit, exists=True), link, remote)
    assert facts['freshness'] == expected
    assert facts['state'] == expected
    assert facts['relationship'] == 'linked'
    assert facts['icon'] == {'equal':'cloud-check','local':'cloud-up','remote':'cloud-down','diverged':'cloud-bang','unknown':'cloud-dotted'}[expected]

@pytest.mark.parametrize('status,extra,expected', [
    ('queued', {}, 'queued'), ('running', {}, 'uploading'), ('running', {'serverProcessing': True}, 'processing'),
    ('running', {'kind': 'download'}, 'downloading'), ('paused', {}, 'paused'),
    ('paused', {'interrupted': True}, 'interrupted'), ('error', {}, 'error'), ('conflict', {}, 'diverged')])
def test_single_badge_precedence_during_transfers(gallery, status, extra, expected):
    from lfs_plugins.gallery_controller import asset_sync_state
    from lfs_plugins.gallery_sync import shared_fields
    base = scene()
    link = dict(sceneId=base['id'], commitUuid='saved', contentRevision=base['contentRevision'], metadataRevision=base['metadataRevision'])
    job = dict(id='job', project='project', kind='upload', status=status, completed=7, total=10, **extra) if 'kind' not in extra else dict(id='job', project='project', status=status, completed=7, total=10, **extra)
    facts = asset_sync_state(dict(id='project', commit_uuid='saved', exists=True), link, base, [job])
    assert facts['state'] == expected
    assert facts['progress'] == 70
    assert facts['jobId'] == 'job'
    assert facts['icon'] == ('ring' if facts['active'] else {'paused':'pause','interrupted':'pause','error':'error','diverged':'cloud-bang'}[expected])
    diverged = asset_sync_state(dict(id='project', commit_uuid='new', exists=True), link, dict(base,title='remote', metadataRevision='remote-edit'), [job])
    assert diverged['state'] == (expected if diverged['active'] else 'diverged')

def test_missing_remote_and_unknown_never_use_timestamps(gallery):
    from lfs_plugins.gallery_controller import asset_sync_state
    from lfs_plugins.gallery_sync import shared_fields
    base = scene()
    project = dict(id='project', commit_uuid='same', exists=True, saved_at_unix_ns=999999999999999999)
    link = dict(sceneId=base['id'], commitUuid='same', checkedAt=0, sharedFields=shared_fields(base))
    assert asset_sync_state(project, link, None)['state'] == 'unknown'
    assert asset_sync_state(project, link, None, checked=True)['state'] == 'remote_deleted'
    assert asset_sync_state(dict(project,exists=False), link, base)['state'] == 'local_missing'
    assert asset_sync_state(None, None, base)['relationship'] == 'remote_only'
    assert asset_sync_state(dict(project,status='IDENTITY_MISMATCH'), link, base)['relationship'] == 'identity_ambiguous'


@pytest.mark.parametrize("status,reason", [
    ("UNREADABLE", "UNREADABLE"), ("UNSUPPORTED", "UNSUPPORTED"),
    ("REPAIR_ONLY", "Needs repair"), ("UNSUPPORTED_NEWER", "Saved by a newer version")])
def test_local_file_problems_need_attention_and_check(gallery, panel_module, monkeypatch, status, reason):
    from lfs_plugins.gallery_controller import asset_sync_state

    monkeypatch.setattr(panel_module.lf.ui, "tr", lambda key: {
        "projects.status.needs_repair": "Needs repair",
        "projects.status.newer_version": "Saved by a newer version",
    }.get(key, key))
    project = dict(id="project", status=status, exists=True, error="")
    facts = asset_sync_state(project, {"sceneId": "private-one"}, scene())

    assert facts["relationship"] == "local_file_problem"
    assert facts["state"] == "error"
    assert facts["icon"] == "cloud-bang"
    assert facts["action"] == ""
    assert facts["actions"] == []
    assert facts["attention"] is (status != "UNSUPPORTED_NEWER")
    assert facts["reason"] == reason


def test_local_file_problem_prefers_project_error(gallery):
    from lfs_plugins.gallery_controller import asset_sync_state

    facts = asset_sync_state(
        {"id": "project", "status": "UNREADABLE", "exists": True, "error": "permission denied"},
        {"sceneId": "private-one"},
        scene(),
    )

    assert facts["reason"] == "permission denied"

def test_upload_format_persists_and_defaults_to_sog(gallery, tmp_path):
    panel, _, _ = gallery
    panel.service.root = tmp_path
    assert panel.upload_format == 'sog'
    panel.upload_format = 'ssog'
    assert panel.upload_format == 'ssog'
    with pytest.raises(ValueError):
        panel.upload_format = 'zip'

def test_subscribers_are_coalesced_and_unsubscribe_stops_delivery(gallery, monkeypatch):
    panel, state, _ = gallery
    module = import_module('lfs_plugins.gallery_controller')
    monkeypatch.setattr(panel, '_schedule_poll', lambda: None)
    monkeypatch.setattr(panel, '_refresh_model', lambda: None)
    monkeypatch.setattr(panel, '_finish_pulls', lambda: None)
    state['signed_in'] = False
    clock = [10.0]
    monkeypatch.setattr(module.time, 'monotonic', lambda: clock[0])
    received = []
    unsubscribe = panel.subscribe(received.append)
    panel._poll()
    count = len(received)
    state['version'] += 1
    clock[0] += .05
    panel._poll()
    assert len(received) == count + 1
    clock[0] += .06
    panel._poll()
    assert len(received) == count + 1
    unsubscribe()
    state['version'] += 1
    clock[0] += 1
    panel._poll()
    assert len(received) == count + 1

def test_overlay_rows_have_all_pending_jobs_and_bounded_history(gallery):
    from lfs_plugins.gallery_transfer_ui import transfer_rows
    jobs = [dict(id=str(i), metadata={'title':str(i)}, status='completed', completed=1,total=1) for i in range(35)]
    jobs += [dict(id='upload',metadata={'title':'Upload'},status='running',completed=2,total=10),
             dict(id='paused',metadata={'title':'Paused'},status='paused',interrupted=True,completed=0,total=10)]
    rows = transfer_rows({'jobs':jobs}, 3)
    assert [r['id'] for r in rows] == ['upload','paused','34','33','32']
    assert rows[0]['can_pause'] and rows[1]['can_resume']

def test_saved_content_stamp_separates_view_and_content_evidence(tmp_path, gallery):
    from lfs_plugins.gallery_project_facts import saved_content_stamp
    from test_portable_project import FIXTURES, rewrite_chapter
    path=tmp_path/'scene.licht'
    data=(FIXTURES/'portable-sog.licht').read_bytes()
    path.write_bytes(data)
    original=saved_content_stamp(path)
    assert original
    changed=rewrite_chapter(data,b'VIEW',lambda view: view['render_settings'].update(color_exposure=2.0))
    path.write_bytes(changed)
    assert saved_content_stamp(path).split(":")[0] == original.split(":")[0]
    assert saved_content_stamp(path) != original
    changed=rewrite_chapter(data,b'SCNG',lambda graph: graph['nodes'][0].update(visible=False))
    path.write_bytes(changed)
    assert saved_content_stamp(path)!=original


@pytest.mark.parametrize("saved_stamp,expected", [
    ("same-content:saved-view", ["view"]),
    ("new-content:saved-view", ["view", "content"]),
    ("", ["view", "content"]),
])
def test_view_only_saved_commit_does_not_add_content_conflict(gallery, monkeypatch, tmp_path, saved_stamp, expected):
    from lfs_plugins.gallery_sync import shared_fields
    from test_gallery_product_regressions import _capture_review

    controller, state, _ = gallery
    module = import_module("lfs_plugins.gallery_controller")
    facts = import_module("lfs_plugins.gallery_project_facts")
    path = tmp_path / "project-a.licht"
    path.write_bytes(b"saved project")
    base = scene(viewerSettings={"exposure": 1.0})
    remote = scene(viewerSettings={"exposure": 2.0})
    remote["metadataRevision"] = "portal-view-edit"
    state.update(scenes=[remote], links={"project": {
        "sceneId": remote["id"], "commitUuid": "published-save",
        "contentRevision": base["contentRevision"], "metadataRevision": base["metadataRevision"],
        "contentStamp": "same-content:old-view", "sharedFields": shared_fields(base),
    }})
    controller._state = state
    controller._refresh_model = lambda: None
    controller._schedule_poll = lambda: None
    monkeypatch.setattr(facts, "saved_content_stamp", lambda _: saved_stamp)
    monkeypatch.setattr(module.lf, "project_poll_write", lambda: {"path": ""}, raising=False)
    reviews = _capture_review(monkeypatch)

    controller.resolve_asset({"id": "project", "path": str(path), "commit_uuid": "view-only-save"},
                             {"title": base["title"], "description": base["description"]})

    assert [row["id"] for row in reviews[0]["groups"]] == expected


@pytest.mark.parametrize("saved_stamp,expected", [
    ("same-content:saved-view", "Changed in the saved project"),
    ("same-content:old-view", '{"exposure": 1.0}'),
])
def test_closed_conflict_review_labels_changed_saved_view(gallery, monkeypatch, tmp_path, saved_stamp, expected):
    from lfs_plugins.gallery_sync import shared_fields
    from test_gallery_product_regressions import _capture_review

    controller, state, _ = gallery
    module = import_module("lfs_plugins.gallery_controller")
    facts = import_module("lfs_plugins.gallery_project_facts")
    path = tmp_path / "project-a.licht"
    path.write_bytes(b"saved project")
    base = scene(viewerSettings={"exposure": 1.0})
    remote = scene(viewerSettings={"exposure": 3.0})
    remote["metadataRevision"] = "portal-view-edit"
    state.update(scenes=[remote], links={"project": {
        "sceneId": remote["id"], "commitUuid": "published-save",
        "contentRevision": base["contentRevision"], "metadataRevision": base["metadataRevision"],
        "contentStamp": "same-content:old-view", "sharedFields": shared_fields(base),
        "localFields": shared_fields(base),
    }})
    controller._state = state
    controller._refresh_model = lambda: None
    controller._schedule_poll = lambda: None
    stamps = []
    monkeypatch.setattr(facts, "saved_content_stamp", lambda p: stamps.append(p) or saved_stamp)
    original_tr = module.tr
    monkeypatch.setattr(module, "tr", lambda key, **values: "Changed in the saved project"
                        if key == "conflict.saved_view_changed" else original_tr(key, **values))
    monkeypatch.setattr(module.lf, "project_poll_write", lambda: {"path": ""}, raising=False)
    reviews = _capture_review(monkeypatch)

    controller.resolve_asset({"id": "project", "path": str(path), "commit_uuid": "view-only-save"},
                             {"title": base["title"], "description": base["description"]})

    view_row = next(row for row in reviews[0]["groups"] if row["id"] == "view")
    assert view_row["mine_value"] == expected
    assert len(stamps) == 1

def test_saved_content_stamp_accepts_native_zstd_index(tmp_path, gallery, monkeypatch):
    from lfs_plugins.gallery_project_facts import saved_content_stamp
    from test_portable_project import FIXTURES

    io = import_module('lichtfeld.io')
    path = tmp_path / 'project-a.licht'
    path.write_bytes((FIXTURES / 'portable-sog.licht').read_bytes())
    original = saved_content_stamp(path)
    assert original == ('3bdebba4495568ec033d6adce1f076454bc351a02d8c01586478e158f1bf6eda:'
                        '26eae58de1db830a1eb2bd4b386752945321cb26e0cb8af2873fd54edee2cbe5')
    io.set_project_title(path, 'Updated')
    data = path.read_bytes()
    head = max((offset for offset in (4096, 8192) if data[offset:offset+8] == b'LFSHEAD\0'),
               key=lambda offset: struct.unpack_from('<Q', data, offset+16)[0])
    commit = struct.unpack_from('<Q', data, head + 80)[0]
    assert struct.unpack_from('<I', data, commit + 168)[0] == 1
    assert saved_content_stamp(path) == original

    panel, state, actions = gallery
    module = import_module('lfs_plugins.gallery_controller')
    monkeypatch.setattr(panel, '_project_identity', lambda: ('project', str(path)))
    monkeypatch.setattr(module.lf.io, 'inspect_project', io.inspect_project)
    monkeypatch.setattr(module.lf, 'prepare_gallery_project',
                        lambda *_: pytest.fail('Metadata update must not export the project'), raising=False)
    panel.service.edit = lambda *args, **kwargs: actions.append((args, kwargs))
    state['links'] = {'project': {'sceneId': 'scene', 'contentStamp': original}}
    panel._publish_saved({'title': 'Updated', 'replaceSceneId': 'scene',
                          'baseRevisions': {'content': 'r1', 'metadata': 'r1'}},
                         'project', str(path), state['identity'], update=True)
    assert actions == [(("scene", {'contentRevision': 'r1', 'metadataRevision': 'r1'},
                         {'title': 'Updated'}),
                        {'commit_uuid': str(io.inspect_project(path).commit_uuid),
                         'content_stamp': original, 'project_id': 'project'})]


def test_metadata_only_update_skips_native_export(gallery, monkeypatch):
    panel,state,actions=gallery
    module=import_module('lfs_plugins.gallery_controller')
    facts=import_module('lfs_plugins.gallery_project_facts')
    monkeypatch.setattr(facts,'saved_content_stamp',lambda path:'same-content:same-view')
    monkeypatch.setattr(panel,'_project_identity',lambda:('project','/project.licht'))
    monkeypatch.setattr(module.lf,'prepare_gallery_project',lambda *_: pytest.fail('Metadata PATCH must upload zero bytes'),raising=False)
    monkeypatch.setattr(module.lf,'io',SimpleNamespace(inspect_project=lambda _:SimpleNamespace(project_uuid='project',commit_uuid='new-commit')),raising=False)
    panel.service.edit=lambda *a,**kw:actions.append((a,kw))
    state['links']={'project':{'sceneId':'scene','contentStamp':'same-content:same-view'}}
    panel._publish_saved({'title':'New title','replaceSceneId':'scene','baseRevisions':{'content':'r1','metadata':'r1'}},'project','/project.licht',state['identity'], update=True)
    assert actions==[(('scene',{'contentRevision':'r1','metadataRevision':'r1'},{'title':'New title'}),{'commit_uuid':'new-commit','content_stamp':'same-content:same-view','project_id':'project'})]


@pytest.mark.parametrize('use_cover', [False, True])
def test_metadata_update_pins_requested_cover_to_saved_commit(gallery, monkeypatch, tmp_path, use_cover):
    panel, state, actions = gallery
    module = import_module('lfs_plugins.gallery_controller')
    facts = import_module('lfs_plugins.gallery_project_facts')
    project_path = tmp_path / 'project.licht'
    project_path.write_bytes(b'project')
    monkeypatch.setattr(facts, 'saved_content_stamp', lambda _: 'same-content:same-view')
    monkeypatch.setattr(panel, '_project_identity', lambda: ('project', str(project_path)))
    monkeypatch.setattr(module.lf.io, 'inspect_project', lambda _: SimpleNamespace(commit_uuid='saved-commit'))
    monkeypatch.setattr(module.lf.io, 'read_preview', lambda _: b'thumbnail', raising=False)
    monkeypatch.setattr(module.gallery_preparation, 'publication_preview', lambda image: image)
    panel.service.edit = lambda *args, **kwargs: actions.append((args, kwargs))
    state['links'] = {'project': {'sceneId': 'scene', 'contentStamp': 'same-content:same-view'}}
    metadata = dict(title='Title', replaceSceneId='scene', baseRevisions={'content': 'r1', 'metadata': 'r1'},
                    useEmbeddedPreview=use_cover)

    panel._publish_saved(metadata, 'project', str(project_path), state['identity'], update=True)

    assert len(actions) == 1
    assert actions[0][1].get('cover_png') == (b'thumbnail' if use_cover else None)


def test_replacement_pins_project_thumbnail_to_prepared_commit(gallery, monkeypatch, tmp_path):
    import base64
    panel, state, actions = gallery
    module = import_module('lfs_plugins.gallery_controller')
    project_path = tmp_path / 'project.licht'
    project_path.write_bytes(b'project')
    monkeypatch.setattr(import_module('lfs_plugins.gallery_project_facts'), 'saved_content_stamp', lambda _: '')
    monkeypatch.setattr(panel, '_project_identity', lambda: ('project', str(project_path)))
    monkeypatch.setattr(panel, '_visible_splats', lambda: [SimpleNamespace(name='visible')])
    monkeypatch.setattr(panel, '_schedule_poll', lambda: None)
    monkeypatch.setattr(module.lf.io, 'inspect_project', lambda _: SimpleNamespace(commit_uuid='saved-commit'))
    monkeypatch.setattr(module.lf.io, 'read_preview', lambda _: b'thumbnail', raising=False)
    monkeypatch.setattr(module.gallery_preparation, 'publication_preview', lambda image: image)
    monkeypatch.setattr(module.lf.ui, 'get_export_state', lambda: {'active': False}, raising=False)
    monkeypatch.setattr(module.lf, 'prepare_gallery_project', lambda *args: actions.append(args), raising=False)
    panel.service.root = tmp_path
    state.update(source_formats=['licht'], links={'project': {'sceneId': 'scene', 'contentStamp': ''}})

    panel._publish_saved(dict(title='Updated', replaceSceneId='scene', baseRevisions={'content': 'r1', 'metadata': 'r1'},
                              useEmbeddedPreview=True), 'project', str(project_path), state['identity'], update=True)

    assert actions[0][3] == 'saved-commit'
    assert base64.b64decode(panel._publish_steps.pending[1]['_previewPng']) == b'thumbnail'

def test_cancel_paused_job_does_not_pause_someone_elses_upload(gallery, monkeypatch):
    panel,state,actions=gallery
    monkeypatch.setattr(panel,'_schedule_poll',lambda:None)
    panel.service.busy=True
    panel.service.pause=lambda:actions.append('paused')
    state['jobs']=[{'id':'one','status':'running'},{'id':'two','status':'paused'}]
    panel.command('cancel','two')
    assert not actions and panel._cancel_requests=={'two'}

def test_catalog_projection_renders_before_account_snapshot_without_authorizing_write(gallery):
    from lfs_plugins.gallery_controller import asset_sync_state
    cached = {"sceneId": "published", "state": "equal", "checkedAt": 123}
    facts = asset_sync_state({"id": "local", "exists": True}, cached_projection=cached)
    assert facts["state"] == "equal"
    assert facts["freshness"] == "unknown"
    assert facts["action"] == "check"
    # Once the account-scoped snapshot arrives it supersedes the projection.
    assert asset_sync_state({"id": "local", "exists": True})["relationship"] == "unlinked"

def test_default_pull_registers_links_without_touching_open_document(gallery, monkeypatch, tmp_path):
    panel, state, actions = gallery
    module = import_module('lfs_plugins.gallery_controller')
    path = str(tmp_path / 'pulled.licht')
    Path(path).write_bytes(b"downloaded project")
    job = dict(id='pull', kind='download', status='completed', path='download.licht',
               metadata={'title': 'Portal'}, result={'title': 'Portal'})
    state['jobs'] = [job]
    panel.service.stage_download = lambda identifier: 'stage'
    panel.service.link_download = lambda *args, **kwargs: actions.append(('link', args, kwargs)) or 'link'
    monkeypatch.setattr(panel, '_schedule_poll', lambda: None)
    for name in ('project_open', 'project_save', 'new_project', 'get_scene'):
        monkeypatch.setattr(module.lf, name, lambda *a, **k: pytest.fail('Default Pull must leave the open document alone'), raising=False)
    monkeypatch.setattr(module.lf, 'project_is_dirty', lambda: True)
    monkeypatch.setattr(module.lf.io, 'inspect_project', lambda _: SimpleNamespace(project_uuid='fresh-project', commit_uuid='fresh-commit'))
    registered = []
    project = SimpleNamespace(id='fresh-project', project_uuid='fresh-project', extra={})
    index = SimpleNamespace(load=lambda: True, update_asset=lambda *a, **kw: project, get_asset=lambda _: None,
        register_licht_asset=lambda p, **kw: registered.append((p, kw)) or (project, True))
    monkeypatch.setattr(import_module('lfs_plugins.asset_index'), 'AssetIndex', lambda: index)
    panel._register_download(job, state['identity'])
    job['stagedImport'] = {'id': 'stage', 'state': 'ready', 'projectPath': path, 'projectId': 'fresh-project',
                          'projectStamp': module.file_stamp(path)}
    panel._finish_import()
    assert registered[0][0] == path
    assert actions == [('link', ('pull', 'fresh-project', 'fresh-commit'), {'project_path': path})]
    assert panel._download_open_steps.pulled_project is None
    job['linkOperation'] = {'id': 'link', 'state': 'ready'}
    panel._finish_import()
    assert panel._download_open_steps.pulled_project == {'id': 'fresh-project', 'path': path, 'jobId': 'pull'}
    assert panel._download_open_steps.pending is None

@pytest.mark.parametrize('open_after', [False, True])
def test_pull_opens_only_when_explicitly_requested(gallery, monkeypatch, open_after):
    panel, state, actions = gallery
    job = {'id': 'pull', 'status': 'completed'}
    panel._state['jobs'] = [job]
    panel._pull_requests['pull'] = ({'remote_only': True, '_pull_open': open_after}, state['identity'])
    monkeypatch.setattr(panel, '_import_download', lambda job: actions.append('open'))
    monkeypatch.setattr(panel, '_register_download', lambda job, identity: actions.append('register'))
    panel._finish_pulls()
    assert actions == ['open' if open_after else 'register']

def test_account_switch_prevents_download_registration(gallery, monkeypatch):
    panel, state, actions = gallery
    pending = {'id': 'pull', '_accountIdentity': state['identity'], '_register': {'phase': 'staging'}}
    panel._download_open_steps.pending = pending
    state['identity'] = ('other-account',)
    monkeypatch.setattr(import_module('lfs_plugins.asset_index'), 'AssetIndex', lambda: pytest.fail('Must not register across accounts'))
    panel._finish_register_download(pending)
    assert panel._download_open_steps.pending is None and panel._download_open_steps.pulled_project is None

def test_uncomparable_update_explains_reupload_and_keeps_requested_format(gallery, monkeypatch, tmp_path):
    panel, state, actions = gallery
    module = import_module('lfs_plugins.gallery_controller')
    monkeypatch.setattr(import_module('lfs_plugins.gallery_project_facts'), 'saved_content_stamp', lambda _: '')
    monkeypatch.setattr(panel, '_project_identity', lambda: ('project', '/project.licht'))
    monkeypatch.setattr(panel, '_visible_splats', lambda: [SimpleNamespace(name='visible')])
    monkeypatch.setattr(module.lf, 'prepare_gallery_project', lambda *a, **kw: actions.append((a, kw)), raising=False)
    panel.service.root = tmp_path
    monkeypatch.setattr(module.lf.ui, 'get_export_state', lambda: {'active': False}, raising=False)
    state['source_formats'] = ['licht']
    state['links'] = {'project': {'sceneId': 'scene', 'contentStamp': ''}}
    monkeypatch.setattr(panel, '_schedule_poll', lambda: None)
    panel._publish_saved({'title': 'Title', 'replaceSceneId': 'scene', 'baseRevisions': {'content':'r1','metadata':'r1'}},
                         'project', '/project.licht', state['identity'], upload_format='ssog', update=True)
    assert panel.snapshot()['reuploadReason']['message'].endswith('info.reupload_encoding')
    assert actions[0][0][2] == 'ssog'

def test_C1_both_camera_tracks_use_native_time_without_mutating_inputs(gallery):
    from lfs_plugins.gallery_controller import combine_camera_tracks
    mine = {'version': 1, 'duration': 4, 'loopMode': 'once', 'playbackSpeed': 1, 'keyframes': [{'time': .5}, {'time': 3}]}
    portal = {'version': 1, 'duration': 3, 'keyframes': [{'time': .5}, {'time': 2}]}
    originals = copy.deepcopy((mine, portal))
    result = combine_camera_tracks(mine, portal)
    assert [frame['time'] for frame in result['keyframes']] == [.5, 3, 4.5, 6]
    assert result['duration'] == 7 and (mine, portal) == originals
    assert all('t' not in frame for frame in result['keyframes'])

@pytest.mark.parametrize('missing_backup', [False, True])
@pytest.mark.parametrize('start_failure', [False, True])
def test_D1_failed_restore_keeps_undo_or_reports_missing_backup(gallery, monkeypatch, tmp_path, missing_backup, start_failure):
    panel, state, _ = gallery
    module = import_module('lfs_plugins.gallery_controller')
    backup = tmp_path/'backup.licht'
    if not missing_backup:
        backup.write_bytes(b'original')
    panel._undo_pull = {'identity': state['identity'], 'path':'/project.licht', 'backup':str(backup), 'stamp':[1]}
    def restore(*args):
        if start_failure:
            raise ValueError('The local project changed.')
        return 'restore'
    panel.service.restore_local_backup = restore
    monkeypatch.setattr(panel, '_schedule_poll', lambda: None)
    monkeypatch.setattr(module.lf, 'project_is_dirty', lambda: False)
    panel.undo_pull()
    if not start_failure:
        assert panel._undo_pull['operation'] == 'restore'
        state['undoRestore'] = {'id':'restore', 'state':'failed', 'message':'The local project changed.', 'backupMissing':missing_backup}
        panel._after_service()
    assert panel._undo_pull['attempt'] == 1
    assert panel._undo_pull['backupMissing'] == missing_backup
    assert 'operation' not in panel._undo_pull
    assert panel._message

def test_D2_removed_portal_wins_when_local_file_is_missing(gallery):
    from lfs_plugins.gallery_controller import asset_sync_state
    asset = {'id':'project', 'exists':False}
    facts = asset_sync_state(asset, {'sceneId':'removed', 'remoteDeleted':True}, checked=True)
    assert (facts['relationship'], facts['state'], facts['action']) == ('remote_deleted', 'remote_deleted', 'unlink')
    assert asset_sync_state(dict(asset, exists=True), {'sceneId':'removed', 'remoteDeleted':True})['action'] == 'publish_again'

def test_U2_portal_404_sentence_requests_refresh(gallery, monkeypatch):
    from lfs_plugins.gallery_messages import localize_message
    module = import_module('lfs_plugins.gallery_controller')
    monkeypatch.setattr(module.lf.ui, 'tr', lambda key: 'localized:' + key)
    assert localize_message('This gallery item is no longer available. Refresh the gallery.') == 'localized:projects.gallery.sidebar.refresh'

@pytest.mark.parametrize('status,expected', [('completed', '134 KB'), ('canceled', '1.0 KB'), ('running', '1.0 KB / 134 KB')])
def test_A5_finished_overlay_rows_show_one_adaptive_size(gallery, monkeypatch, status, expected):
    from lfs_plugins.gallery_transfer_ui import transfer_rows
    module = import_module('lfs_plugins.gallery_controller')
    monkeypatch.setattr(module.lf.ui, 'tr', lambda key: {
        'projects.unit.kb': 'KB', 'gallery.transfer.bytes': '{done} / {total}',
    }.get(key, key))
    job = dict(id='job', kind='upload', status=status, completed=1024, total=137114)
    row = transfer_rows({'jobs': [job]})[0]
    assert row['bytes'] == expected
    if status == 'completed':
        assert row['progress'] == 100

def test_A5_gallery_asset_borders_use_supported_longhands():
    from pathlib import Path
    import re
    resources = Path(__file__).resolve().parents[2] / 'src/visualizer/gui/rmlui/resources'
    paths = set(resources.glob('*gallery*.rcss')) | set(resources.glob('*asset*.rcss'))
    unsupported = re.compile(r'\bborder(?:-(?:top|right|bottom|left))?\s*:\s*[^;]*(?:solid|dashed|dotted|double)\s*;')
    assert paths
    assert not [(p.name, match.group()) for p in paths for match in unsupported.finditer(p.read_text())]

@pytest.mark.parametrize('upload_format', ['studio', 'sog'])
def test_closed_project_prepares_saved_file_without_opening(gallery, monkeypatch, tmp_path, upload_format):
    panel, state, actions = gallery
    module = import_module('lfs_plugins.gallery_controller')
    panel.service.root = tmp_path
    state['source_formats'] = ['licht']
    asset = {'id': 'project', 'path': str(tmp_path / 'saved.licht'), 'commit_uuid': 'reviewed-commit'}
    monkeypatch.setattr(module.lf.io, 'inspect_project', lambda _: SimpleNamespace(project_uuid='project', commit_uuid='reviewed-commit'))
    monkeypatch.setattr(module.lf, 'project_poll_write', lambda: {'path': '/another.licht'}, raising=False)
    monkeypatch.setattr(module.lf, 'prepare_gallery_project', lambda *a: actions.append(a), raising=False)
    for name in ('project_open', 'project_save', 'get_scene', 'get_camera', 'get_render_settings', 'prepare_gallery_scene'):
        monkeypatch.setattr(module.lf, name, lambda *a, **kw: pytest.fail('Closed publication touched the live document'), raising=False)
    monkeypatch.setattr(module.lf.ui, 'get_export_state', lambda: {'active': False}, raising=False)
    monkeypatch.setattr(panel, '_schedule_poll', lambda: None)
    details = {'title': 'Saved title', 'description': 'Reviewed description'}
    panel.publish_asset(asset, details, upload_format)
    export, metadata, project, _ = panel._publish_steps.pending
    assert actions == [(asset['path'], str(export), 'ply' if upload_format == 'studio' else upload_format, 'reviewed-commit')]
    assert metadata == dict(details, viewerSettings={}, _uploadFormat=upload_format, _contentStamp='')
    assert project == 'project'

def test_closed_project_update_keeps_reviewed_replacement_guard(gallery, monkeypatch, tmp_path):
    panel, state, actions = gallery
    module = import_module('lfs_plugins.gallery_controller')
    panel.service.root = tmp_path
    monkeypatch.setattr(module.lf.io, 'inspect_project', lambda _: SimpleNamespace(project_uuid='project', commit_uuid='commit'))
    state['source_formats'] = ['licht']
    state['links'] = {'project': {'sceneId': 'scene'}}
    state['scenes'] = [dict(id='scene', revision='reviewed-revision', visibility='private', contentRevision='reviewed-revision', metadataRevision='reviewed-revision')]
    monkeypatch.setattr(module, 'asset_sync_state', lambda *a: {'freshness': 'local'})
    monkeypatch.setattr(module.lf, 'project_poll_write', lambda: {'path': '/another.licht'}, raising=False)
    monkeypatch.setattr(module.lf, 'prepare_gallery_project', lambda *a: actions.append(a), raising=False)
    monkeypatch.setattr(module.lf, 'project_open', lambda *a, **kw: pytest.fail('Update opened a document'), raising=False)
    monkeypatch.setattr(module.lf.ui, 'get_export_state', lambda: {'active': False}, raising=False)
    monkeypatch.setattr(panel, '_schedule_poll', lambda: None)
    panel.publish_asset({'id': 'project', 'path': '/saved.licht', 'commit_uuid': 'commit'},
                        {'title': 'Update', 'description': '', 'visibility': 'private'}, 'sog', update=True)
    assert panel._publish_steps.pending[1]['replaceSceneId'] == 'scene'
    assert panel._publish_steps.pending[1]['baseRevisions'] == {'content': 'reviewed-revision', 'metadata': 'reviewed-revision'}
    assert actions[0][0] == '/saved.licht'

@pytest.mark.parametrize('kind', [b'SPLT', b'CKPT'])
def test_saved_content_stamp_rejects_forged_index_rows(tmp_path, gallery, kind):
    from lfs_plugins.gallery_project_facts import saved_content_stamp
    from lfs_plugins.portable_project import _crc
    from test_portable_project import FIXTURES
    original = bytearray((FIXTURES / 'portable-sog.licht').read_bytes())
    head = next(offset for offset in (4096, 8192) if original[offset:offset+8] == b'LFSHEAD\0')
    commit = struct.unpack_from('<Q', original, head + 80)[0]
    index, size = struct.unpack_from('<QQ', original, commit + 136)
    count = struct.unpack_from('<Q', original, index + 16)[0]
    row = next(index+64+i*96 for i in range(count) if original[index+64+i*96:index+68+i*96] == b'DSRC')
    original[row:row+4] = kind
    path = tmp_path / 'index-evidence.licht'

    crc = _crc(original[index:index+size])
    struct.pack_into('<II', original, commit+160, crc, crc)
    crc = _crc(original[commit:commit+252])
    struct.pack_into('<I', original, commit+252, crc)
    struct.pack_into('<I', original, head+104, crc)
    struct.pack_into('<I', original, head+4092, _crc(original[head:head+4092]))
    path.write_bytes(original)
    assert saved_content_stamp(path) == ''


def test_gallery_action_table_uses_file_activity_and_account_precedence(gallery):
    from lfs_plugins.gallery_actions import gallery_actions, gallery_eligibility
    asset = {"id": "project", "exists": True, "status": "AVAILABLE"}
    facts = dict(state="local", relationship="linked", linked=True, sceneReady=True,
                 signed_in=True, established=True, source_formats=["licht"])
    assert gallery_actions(asset, facts)[0]["id"] == "update"
    disabled = gallery_actions(asset, dict(facts, signed_in=False))[0]
    assert not disabled["enabled"] and disabled["reason"].endswith("eligibility.connect")
    assert gallery_actions(dict(asset, status="UNREADABLE"), facts) == []
    cached = dict(facts, cachedUnverified=True)
    assert gallery_actions(dict(asset, status="UNREADABLE"), cached) == []
    assert gallery_actions(dict(asset, status="MISSING"), cached)[0]["id"] == "locate"
    interrupted = dict(facts, activity="interrupted", job={"id": "j", "status": "paused"})
    assert [a["id"] for a in gallery_actions(asset, interrupted)] == ["resume", "cancel"]
    processing = dict(facts, activity="processing", active=True, job={"needsAttention": True})
    assert [a["id"] for a in gallery_actions(asset, processing)] == ["keep_waiting", "cancel"]
    assert gallery_actions(asset, dict(facts, activity="applying", active=True)) == []
    completed = dict(facts, job={"id": "done", "status": "completed"}, undoAvailable=True)
    assert [a["id"] for a in gallery_actions(asset, completed)] == ["undo"]
    assert not gallery_actions(asset, dict(completed, signed_in=False))[0]["enabled"]
    assert gallery_actions(asset, dict(facts, viewingCopy=True, state="equal"))[0]["id"] == "publish_new"
    assert gallery_actions(asset, dict(facts, viewingCopy=True, state="remote"))[0]["id"] == "publish_new"
    queued = dict(facts, activity="queued", active=True, job={"id": "j", "status": "queued"})
    assert [a["id"] for a in gallery_actions(asset, queued)] == ["resume", "cancel"]
    assert gallery_eligibility(asset, dict(facts, quotaBytes=100))["remainingBytes"] is None
    publication = dict(asset, publication={"checked": True, "preparedBytes": 30})
    quota = gallery_eligibility(publication, dict(facts, quotaBytes=100, usedBytes=60, reservedBytes=20))
    assert quota["remainingBytes"] == 20 and quota["reasons"] == ["space"]
    assert gallery_eligibility(dict(asset, embedded_dataset_complete=False), facts)["status"] == "not_checked"
    reasons = gallery_eligibility(dict(asset, publication={"visibleSplats": 0, "externalPayloads": True}), facts)
    assert reasons["reasons"] == ["no_splats", "external_payloads"]


def test_presentation_metadata_and_scene_content_have_separate_relationships(gallery):
    from lfs_plugins.gallery_controller import asset_sync_state
    asset = {"id": "project", "commit_uuid": "saved", "status": "AVAILABLE"}
    remote = dict(scene(), presentationRevision="p1")
    link = dict(sceneId=remote["id"], commitUuid="saved", contentRevision="original",
                metadataRevision="original", acknowledgedPresentationRevision="p1")
    assert asset_sync_state(asset, link, remote)["state"] == "equal"
    assert asset_sync_state(asset, link, dict(remote, presentationRevision="p2"))["state"] == "presentation"
    assert asset_sync_state(asset, link, dict(remote, metadataRevision="m2"))["state"] == "remote"
    assert asset_sync_state(asset, link, dict(remote, contentRevision="c2"))["state"] == "remote_content"
    assert asset_sync_state(asset, link, None, established=False)["state"] == "not_checked"
    assert asset_sync_state(asset, dict(link, commitUuid=""), remote)["state"] == "unknown"


def test_eligibility_uses_only_checks_for_the_saved_commit(gallery):
    from lfs_plugins.gallery_actions import gallery_actions, gallery_eligibility
    asset = dict(id="project", commit_uuid="saved", exists=True)
    failed = dict(project="project", commitUuid="saved", nativePreparation=True,
                  failureReason="gallery_project_no_splats: No visible geometry")
    facts = dict(signed_in=True, job=failed, activity="error")
    assert gallery_eligibility(asset, facts)["reasons"] == ["no_splats"]
    assert not gallery_actions(asset, facts)[0]["enabled"]
    assert gallery_eligibility(dict(asset, commit_uuid="changed"), facts)["status"] == "not_checked"
    checked = dict(project="project", commitUuid="saved", kind="upload", packaged=True,
                   status="completed", uploadFormat="sog", total=80)
    facts = dict(signed_in=True, jobs=[checked], quotaBytes=100, usedBytes=40)
    assert gallery_eligibility(asset, facts)["reasons"] == ["space"]
    assert gallery_eligibility(asset, dict(facts, replacedBytes=80))["status"] == "eligible"
    assert gallery_eligibility(dict(asset, commit_uuid="changed"), facts)["status"] == "not_checked"


@pytest.mark.parametrize("reason,resumable", [
    ("gallery_project_no_splats: No visible geometry", False),
    ("gallery_project_payload_unavailable: Missing payload", False),
    ("Could not write the prepared copy", True),
])
def test_preparation_retry_in_overlay_uses_the_failed_commit(gallery, reason, resumable):
    from lfs_plugins.gallery_transfer_ui import transfer_rows
    failure = dict(id="preparation:project", project="project", commitUuid="saved", nativePreparation=True,
                   status="error", failureReason=reason, message=reason)
    row = transfer_rows(dict(signed_in=True, preparationFailure=failure))[0]
    assert row["project"] == "project"
    assert row["can_resume"] is resumable


def test_conflict_groups_keep_both_values_and_default_content_to_mine(gallery):
    from lfs_plugins.gallery_controller import conflict_groups
    base = dict(title="Original", description="First", visibility="private", viewerSettings={})
    mine = dict(base, title="Mine", viewerSettings={"exposure": 2, "cameraPath": {"duration": 2}})
    remote = dict(base, title="Gallery", description="Edited there", visibility="public",
                  viewerSettings={"exposure": 3, "cameraPath": {"duration": 5}}, contentRevision="new")
    rows = {r["id"]: r for r in conflict_groups({"commit_uuid": "local"},
        dict(sharedFields=base, contentRevision="old", commitUuid="old"), mine, remote)}
    assert set(rows) == {"text", "view", "track", "content"}
    assert "Mine" in rows["text"]["mine_value"] and "Edited there" in rows["text"]["gallery_value"]
    assert rows["text"]["choice"] == "mine"
    assert rows["content"]["choice"] == "mine" and not rows["content"]["can_both"]
    assert rows["track"]["can_both"]

    resources = Path(__file__).resolve().parents[2] / "src/visualizer/gui/rmlui/resources"
    conflict_rml = (resources / "gallery_file_panel.rml").read_text()
    assert 'data-attr-title="part.difference"' in conflict_rml
    assert 'data-attr-title="part.values"' not in conflict_rml


def test_settings_apply_waits_for_backup_and_never_replaces_geometry(gallery, monkeypatch, tmp_path):
    panel, state, actions = gallery
    module = import_module("lfs_plugins.gallery_controller")
    path = tmp_path / "master.licht"
    path.write_bytes(b"saved project with training history")
    project = ("project", str(path))
    monkeypatch.setattr(panel, "_project_identity", lambda: project)
    monkeypatch.setattr(module.lf, "project_is_dirty", lambda: False, raising=False)
    monkeypatch.setattr(module.lf, "is_training_active", lambda: False, raising=False)
    monkeypatch.setattr(panel, "_schedule_poll", lambda: None)
    monkeypatch.setattr(panel, "_begin_local_update", lambda *_: pytest.fail("Settings entered geometry replacement"))
    monkeypatch.setattr(module, "restore_view", lambda *args, **kwargs: actions.append("view"))
    monkeypatch.setattr(panel, "_save_current_project", lambda callback, **kwargs: (actions.append("save"), callback()))
    job = dict(id="settings", status="completed", project="project",
               localUpdate={"id": "backup", "state": "preparing", "backupPath": str(tmp_path / "backup.licht")})
    state["jobs"] = [job]
    panel.service.prepare_settings_update = lambda *args: ("settings", "backup")
    def finish_settings(*args, **kwargs):
        actions.append("link")
        job["localUpdate"]["state"] = "applied"
    panel.service.finish_settings_update = finish_settings
    panel._begin_settings_apply({"id": "project"}, scene(), dict(title="Gallery title", viewerSettings={}))
    panel.service.busy = True
    panel._finish_settings_apply()
    assert actions == []
    panel.service.busy = False
    job["localUpdate"]["state"] = "ready"
    panel._finish_settings_apply()
    panel._finish_settings_apply()
    assert actions == ["view", "save", "link"]
    assert panel._undo_pull["jobId"] == "settings"
    assert path.read_bytes() == b"saved project with training history"


def test_gallery_review_survives_a_locale_document_reload(gallery, monkeypatch):
    from lfs_plugins.gallery_file_panel import GalleryFilePanel
    monkeypatch.setattr(import_module("lfs_plugins.gallery_file_panel").lf.ui, "get_panel_object", lambda _: None, raising=False)
    monkeypatch.setattr(GalleryFilePanel.__bases__[0], 'on_unmount', lambda *_: None, raising=False)
    panel = GalleryFilePanel()
    closed = []
    review = dict(mode='conflict', on_done=lambda submitted: closed.append(submitted))
    panel._review = review
    panel.on_unmount(None)
    assert panel._review is review and not closed
    panel._finish(False)
    assert panel._review is None and closed == [False]


@pytest.mark.parametrize("change", ["none", "project", "account"])
def test_file_menu_review_submits_only_for_original_project_and_account(gallery, monkeypatch, tmp_path, change):
    from lfs_plugins.gallery_file_panel import GalleryFilePanel

    controller, state, _ = gallery
    controller.service.root = tmp_path
    original = tmp_path / "original.licht"
    other = tmp_path / "other.licht"
    original.write_bytes(b"saved project")
    other.write_bytes(b"another project")
    current = [str(original)]
    monkeypatch.setattr(controller_module := import_module("lfs_plugins.gallery_controller").lf,
                        "project_has_path", lambda: True, raising=False)
    monkeypatch.setattr(controller_module, "project_poll_write", lambda: {"path": current[0]}, raising=False)
    monkeypatch.setattr(controller_module, "project_is_dirty", lambda: False, raising=False)
    monkeypatch.setattr(controller_module.ui, "get_panel_object", lambda _identifier: None, raising=False)
    monkeypatch.setattr(controller_module.ui, "set_panel_enabled", lambda *_args: None, raising=False)
    monkeypatch.setattr(controller_module.ui, "request_redraw", lambda: None, raising=False)
    submitted = []
    monkeypatch.setattr(controller, "publish_asset", lambda *args, **kwargs: submitted.append((args, kwargs)))
    panel = GalleryFilePanel()
    panel.show(
        controller=controller,
        asset={"id": "project", "path": str(original), "name": "original", "publication": {}},
        scene=None,
        action="publish",
        fields={"title": "Original", "description": "", "upload_format": "sog"},
        expected_project_path=str(original),
    )

    if change == "project":
        current[0] = str(other)
    elif change == "account":
        state["identity"] = ("https://portal.example", "two@example.com", "second", True)
    panel._submit()

    if change == "project":
        assert submitted == []
        assert panel._review is not None
        assert panel._error
    elif change == "account":
        assert submitted == []
        assert panel._review is None
    else:
        assert len(submitted) == 1
        args, kwargs = submitted[0]
        assert args[0]["path"] == str(original)
        assert args[1]["saveProject"] is False
        assert args[2] == "sog"
        assert kwargs == {"update": False, "publish_as_new": False}


def test_file_menu_update_handoff_keeps_real_conflict_review_open(gallery, monkeypatch, tmp_path):
    from lfs_plugins.gallery_file_panel import GalleryFilePanel

    controller, state, _ = gallery
    module = import_module("lfs_plugins.gallery_controller")
    panel_module = import_module("lfs_plugins.gallery_file_panel")
    project = tmp_path / "project.licht"
    project.write_bytes(b"saved project")
    scene_row = scene()
    scene_row.update(title="Gallery title", contentRevision="remote-content",
                     metadataRevision="remote-meta", contentLength=4096)
    link = dict(
        sceneId=scene_row["id"], commitUuid="base-commit",
        contentRevision="base-content", metadataRevision="base-meta",
        localFields={"title": "Local title", "description": "", "visibility": "private"},
        sharedFields={"title": "Base title", "description": "", "visibility": "private"},
    )
    state.update(links={"project": link}, scenes=[scene_row], signed_in=True, busy=False)
    controller._state = state
    controller.service.root = tmp_path
    controller._refresh_model = lambda: None
    controller._schedule_poll = lambda: None
    controller._panel_busy = lambda: False
    controller._project_identity = lambda: ("project", str(project))
    monkeypatch.setattr(module.lf, "project_has_path", lambda: True, raising=False)
    monkeypatch.setattr(module.lf, "project_poll_write", lambda: {"path": str(project)}, raising=False)
    monkeypatch.setattr(module.lf, "project_is_dirty", lambda: False, raising=False)
    monkeypatch.setattr(module, "capture_view", lambda _lf: {})
    enabled = []
    monkeypatch.setattr(panel_module.lf.ui, "get_panel_object", lambda _id: None, raising=False)
    monkeypatch.setattr(panel_module.lf.ui, "set_panel_enabled", lambda _id, value: enabled.append(value), raising=False)
    monkeypatch.setattr(panel_module.lf.ui, "request_redraw", lambda: None, raising=False)

    panel = GalleryFilePanel()
    asset = {"id": "project", "path": str(project), "name": "Project",
             "commit_uuid": "local-commit", "publication": {}}
    panel.show(
        controller=controller,
        asset=asset,
        scene=scene_row,
        action="update",
        fields={"title": "Local title", "description": "", "visibility": "private",
                "upload_format": "sog"},
        expected_project_path=str(project),
    )
    monkeypatch.setattr(panel_module.lf.ui, "get_panel_object", lambda identifier: panel if identifier == panel.id else None,
                        raising=False)

    # The panel invokes the real controller route. The stale linked state opens
    # resolve_asset(), which synchronously replaces this review with conflict choices.
    panel._submit()

    assert controller._decision_pending
    assert panel._review is not None
    assert panel._review["mode"] == "conflict"
    assert panel._review["action"] == "conflict"
    assert any(row["id"] == "content" for row in panel._review["groups"])
    assert enabled[-1] is True


def test_file_menu_review_does_not_reuse_an_asset_manager_review(gallery, monkeypatch, tmp_path):
    from lfs_plugins.gallery_file_panel import GalleryFilePanel

    controller, _, _ = gallery
    module = import_module("lfs_plugins.gallery_controller")
    monkeypatch.setattr(module.lf, "project_poll_write", lambda: {"path": ""}, raising=False)
    monkeypatch.setattr(module.lf.ui, "get_panel_object", lambda _identifier: None, raising=False)
    monkeypatch.setattr(module.lf.ui, "set_panel_enabled", lambda *_args: None, raising=False)
    monkeypatch.setattr(module.lf.ui, "request_redraw", lambda: None, raising=False)
    asset = {"id": "project", "path": str(tmp_path / "project.licht"), "name": "Project"}
    fields = {"title": "Project", "description": "", "visibility": "private"}
    finished = []
    panel = GalleryFilePanel()
    panel.show(controller=controller, asset=asset, scene=None, action="publish", fields=fields,
               on_done=lambda submitted: finished.append(submitted))
    panel.show(controller=controller, asset=asset, scene=None, action="publish", fields=fields,
               expected_project_path=asset["path"])

    assert finished == [False]
    assert panel._review["expected_project_path"] == asset["path"]


def test_publish_current_project_rechecks_identity_when_same_path_is_replaced(gallery, monkeypatch, tmp_path):
    controller, _, _ = gallery
    project = tmp_path / "project.licht"
    project.write_bytes(b"replacement project")
    module = import_module("lfs_plugins.gallery_controller")
    monkeypatch.setattr(module.lf, "project_has_path", lambda: True, raising=False)
    monkeypatch.setattr(module.lf, "project_poll_write", lambda: {"path": str(project)}, raising=False)
    monkeypatch.setattr(module.lf.io, "inspect_project", lambda _path: SimpleNamespace(
        project_uuid="replacement-project", commit_uuid="replacement-commit"))
    monkeypatch.setattr(controller, "_refresh_model", lambda: None)
    monkeypatch.setattr(controller, "_panel_busy", lambda: False)
    monkeypatch.setattr(controller, "_review_publish", lambda *_args, **_kwargs: pytest.fail("Changed project was reviewed"))

    with pytest.raises(ValueError):
        controller.publish_asset(
            {"id": "original-project", "path": str(project)},
            {"title": "Original", "description": "", "visibility": "private"},
            "sog",
        )


def test_replacement_buttons_wait_for_the_account_and_transfer(gallery):
    from lfs_plugins.gallery_file_panel import GalleryFilePanel
    panel = GalleryFilePanel()
    actions = []
    panel._review = dict(mode="replacement", on_submit=actions.append)
    for state in ({"signed_in": False}, {"signed_in": True, "busy": True}):
        panel._state = state
        panel._replacement(["replace"])
        panel._replacement(["keep"])
    assert not actions


def test_replacement_prepares_the_existing_scene_without_visibility(gallery, monkeypatch, tmp_path):
    panel, state, actions = gallery
    remote = dict(scene(), visibility="public")
    state.update(source_formats=["licht"], scenes=[remote])
    panel._state = state
    panel.service.root = tmp_path
    monkeypatch.setattr(panel, "_schedule_poll", lambda: None)
    monkeypatch.setattr(panel, "_patch_saved_update", lambda metadata, *args, **kwargs: actions.append(metadata) or True)
    monkeypatch.setattr(import_module("lfs_plugins.gallery_controller").lf.ui, "get_export_state", lambda: {"active": False}, raising=False)
    panel._publish_closed_asset(dict(id="project", path="/project.licht"), remote, "sog",
        update=False, publish_as_new=False,
        handoff=dict(sceneId=remote["id"], baseRevisions={"content": "original", "metadata": "original"}))
    assert len(actions) == 1 and actions[0]["replaceSceneId"] == remote["id"]
    assert "visibility" not in actions[0]


@pytest.mark.parametrize("health,tone", [
    ("MISSING", "warning"), ("IDENTITY_MISMATCH", "warning"),
    ("UNREADABLE", "error"), ("REPAIR_ONLY", "error"), ("UNSUPPORTED_NEWER", "error"),
])
def test_file_health_has_an_independent_glyph(gallery, health, tone):
    from lfs_plugins.gallery_controller import asset_sync_state
    facts = asset_sync_state({"id":"project", "status":health, "error":"Fixture reason"},
        {"sceneId":"one"}, jobs=[dict(id="job",project="project",kind="upload",status="running",completed=43,total=100)])
    assert facts["health_icon"] == "bang"
    assert facts["health_tone"] == tone
    assert facts["action"] == ("locate" if health == "MISSING" else "")
    assert [action["id"] for action in facts["actions"]] == (["locate"] if health == "MISSING" else [])
    assert facts["progress"] == 43


def test_text_only_conflict_ignores_server_visibility(gallery):
    from lfs_plugins.gallery_controller import conflict_groups
    local = dict(title="Mine", description="", viewerSettings={})
    remote = dict(local, title="Gallery", visibility="private", contentRevision="c")
    rows = conflict_groups({"commit_uuid": "saved"},
        dict(sharedFields=dict(remote), commitUuid="saved", contentRevision="c"), local, remote)
    assert [row["id"] for row in rows] == ["text"]


def test_saved_legacy_visibility_does_not_mark_local_changes(gallery):
    from lfs_plugins.gallery_controller import asset_sync_state
    fields = dict(title="Scene", description="", viewerSettings={})
    remote = dict(fields, id="scene", visibility="private", contentRevision="c", metadataRevision="m")
    link = dict(sceneId="scene", commitUuid="saved", contentRevision="c", metadataRevision="m",
        sharedFields=dict(fields, visibility="public"), localFields=fields)
    assert asset_sync_state({"id": "project", "commit_uuid": "saved"}, link, remote)["freshness"] == "equal"


@pytest.mark.parametrize("applied_camera", [
    pytest.param({"camera": {"position": [-5.0, 2.0, -6.0], "target": [0.0, 0.0, 0.0],
                            "up": [0.0, 1.0, 0.0], "fov": 55.0}}, id="portal-start-view"),
])
def test_applied_gallery_fields_do_not_create_a_local_change(gallery, applied_camera):
    from lfs_plugins.gallery_controller import asset_sync_state
    from lfs_plugins.gallery_sync import shared_fields

    remote = scene(viewerSettings={"camera": {"position": [-5.0, 2.0, -6.0],
        "target": [0.0, 0.0, 0.0], "up": [0.0, 1.0, 0.0], "fov": 55.0},
        "cameraPath": None})
    remote["description"] = "Changed in portal"
    fields = shared_fields(remote)
    fields["viewerSettings"].pop("cameraPath")
    fields["viewerSettings"]["camera"] = applied_camera["camera"]
    old_link = dict(sceneId=remote["id"], commitUuid="applied-save", sharedFields=shared_fields(remote),
                    localFields=fields, contentRevision=remote["contentRevision"],
                    metadataRevision=remote["metadataRevision"])
    project = {"id": "project", "commit_uuid": "applied-save", "exists": True}

    assert asset_sync_state(project, old_link, remote)["freshness"] == "equal"
    later = dict(remote, metadataRevision="later-portal-edit", description="Later edit")
    assert asset_sync_state(project, old_link, later)["freshness"] == "remote"


@pytest.mark.parametrize("has_review", [False, True])
def test_project_layout_restore_preserves_gallery_review_visibility(gallery, monkeypatch, tmp_path, has_review):
    from lfs_plugins.gallery_file_panel import GalleryFilePanel

    controller, _, _ = gallery
    module = import_module("lfs_plugins.gallery_file_panel")
    enabled = {}
    monkeypatch.setattr(module.lf.ui, "set_panel_enabled", lambda identifier, value: enabled.update({identifier: value}), raising=False)
    monkeypatch.setattr(module.lf.ui, "get_panel_object", lambda _identifier: None, raising=False)
    monkeypatch.setattr(module.lf.ui, "request_redraw", lambda: None, raising=False)
    panel = GalleryFilePanel()
    if has_review:
        panel.show(controller=controller,
                   asset={"id": "project", "path": str(tmp_path / "project.licht"), "name": "Project"},
                   scene=None, action="conflict", fields={}, mode="conflict")
    review = panel._review
    # Native project loading restores saved panel visibility before chrome.
    enabled[panel.id] = not has_review
    from lfs_plugins.panels import apply_panel_chrome
    apply_panel_chrome(panel, {})
    assert enabled[panel.id] is has_review
    assert panel._review is review


@pytest.mark.parametrize("ready", [False, True])
def test_failed_gallery_apply_has_a_way_to_review_again(gallery, ready):
    from lfs_plugins.gallery_controller import asset_sync_state

    project = {"id": "project", "path": "/project.licht", "commit_uuid": "local"}
    link = {"sceneId": "remote", "commitUuid": "base", "contentRevision": "c1", "metadataRevision": "m1"}
    remote = {"id": "remote", "contentRevision": "c1", "metadataRevision": "m2", "status": "ready" if ready else "processing"}
    job = {"id": "failed", "project": "project", "sceneId": "remote", "kind": "download", "status": "completed",
           "localUpdate": {"state": "failed", "backupPath": "/backup.licht"}}
    facts = asset_sync_state(project, link, remote, [job])
    actions = [action["id"] for action in facts["actions"]]
    assert "open_recovery" in actions
    assert ("resolve" in actions) is ready


@pytest.mark.parametrize("changed_account", [False, True])
def test_failed_settings_apply_refreshes_before_another_review(gallery, monkeypatch, changed_account):
    controller, state, actions = gallery
    controller._settings_pending = dict(job="failed", identity=state["identity"])
    if changed_account:
        state["identity"] = ("https://portal.example", "other@example.com", "second", True)
    def fail(job, reason):
        actions.append((job, reason))
        controller.service.busy = True
    controller.service.fail_local_update = fail
    monkeypatch.setattr(controller, "_schedule_poll", lambda: None)
    controller._fail_settings_apply(ValueError("Gallery changed"))
    assert controller._settings_pending is None
    assert bool(actions) is not changed_account
    assert controller._refresh_requested is not changed_account
    assert controller._refresh_force_requested is not changed_account


@pytest.mark.parametrize("panel_available", [True, False])
@pytest.mark.parametrize("local_only", [True, False])
@pytest.mark.parametrize("content", ["mine", "gallery"])
def test_conflict_review_controls_whether_changes_are_published(gallery, monkeypatch, tmp_path, local_only, content, panel_available):
    from lfs_plugins.gallery_file_panel import GalleryFilePanel
    controller, state, actions = gallery
    module = import_module("lfs_plugins.gallery_controller")
    panel_module = import_module("lfs_plugins.gallery_file_panel")
    path = tmp_path / "project.licht"
    path.write_bytes(b"saved project")
    remote = scene(viewerSettings={})
    state.update(scenes=[remote], links={"project": {"sceneId": remote["id"],
        "commitUuid": "base", "contentRevision": "base", "metadataRevision": "base",
        "localFields": {"title": "Local"}, "sharedFields": {"title": "Base"}}})
    controller._state = state
    controller._refresh_model = lambda: None
    controller._schedule_poll = lambda: None
    controller._project_identity = lambda: ("project", str(path))
    monkeypatch.setattr(module.lf, "project_poll_write", lambda: {"path": str(path)}, raising=False)
    monkeypatch.setattr(module.lf, "project_is_dirty", lambda: False, raising=False)
    monkeypatch.setattr(module, "capture_view", lambda _: {})
    panel = GalleryFilePanel()
    monkeypatch.setattr(panel_module.lf.ui, "get_panel_object", lambda key: panel if key == panel.id else None, raising=False)
    monkeypatch.setattr(panel_module.lf.ui, "set_panel_enabled", lambda *_: None, raising=False)
    monkeypatch.setattr(panel_module.lf.ui, "request_redraw", lambda: None, raising=False)
    controller._begin_settings_apply = lambda *args, **kw: actions.append((args, kw))
    controller.pull_asset = lambda *args: actions.append("pull")
    asset = {"id": "project", "path": str(path), "commit_uuid": "local"}
    details = {"title": "Local", "description": ""}
    if not panel_available:
        monkeypatch.setattr(panel_module.lf.ui, "get_panel_object", lambda _: None)
        with pytest.raises(ValueError, match="review"):
            controller.resolve_asset(asset, details)
        assert not controller._decision_pending
        return
    controller.resolve_asset(asset, details)
    for row in panel._review["groups"]:
        row["choice"] = content if row["id"] == "content" else "gallery"
    panel._submit(local_only=local_only)
    assert not panel._error
    assert panel._review is None
    assert not controller._decision_pending
    if content == "gallery":
        assert actions == ["pull"]
        assert controller._local_update_steps.overrides[3] is (not local_only)
    else:
        assert len(actions) == 1
        args, kwargs = actions[0]
        assert args[2]["title"] == remote["title"]
        assert kwargs["publish"] is (not local_only)
        assert kwargs["replace_content"] is (not local_only)


def test_reopening_hidden_review_shows_it_without_losing_choices(gallery, monkeypatch):
    from lfs_plugins.gallery_file_panel import GalleryFilePanel
    module = import_module("lfs_plugins.gallery_file_panel")
    controller, _, _ = gallery
    enabled = []
    monkeypatch.setattr(module.lf.ui, "get_panel_object", lambda _: None, raising=False)
    monkeypatch.setattr(module.lf.ui, "set_panel_enabled", lambda key, value: enabled.append(value), raising=False)
    monkeypatch.setattr(module.lf.ui, "request_redraw", lambda: None, raising=False)
    panel = GalleryFilePanel()
    review = dict(controller=controller, asset={"id": "project"}, scene=scene(), action="conflict",
                  fields={}, mode="conflict", groups=[dict(id="text", choice="mine")])
    panel.show(**review)
    panel._review["groups"][0]["choice"] = "gallery"
    enabled.clear()
    panel.show(**review)
    assert enabled == [True]
    assert panel._review["groups"][0]["choice"] == "gallery"


def test_unavailable_review_panel_reports_error(gallery, monkeypatch):
    module = import_module("lfs_plugins.gallery_file_panel")
    monkeypatch.setattr(module.lf.ui, "get_panel_object", lambda _: None, raising=False)
    with pytest.raises(ValueError, match="review"):
        module.open_gallery_file_panel()


def test_gallery_content_uses_chosen_local_environment_and_title(gallery, monkeypatch, tmp_path):
    controller, _, actions = gallery
    module = import_module("lfs_plugins.gallery_controller")
    path = tmp_path / "project.licht"
    path.write_bytes(b"saved")
    project = ("project", str(path))
    controller._project_identity = lambda: project
    controller.service.environment_path = lambda _: "gallery.hdr"
    monkeypatch.setattr(import_module("lfs_plugins.gallery_sync_steps"), "restore_view", lambda *a, **kw: actions.append(kw["environment_path"]))
    monkeypatch.setattr(module.lf, "set_node_visibility", lambda *_: None, raising=False)
    monkeypatch.setattr(module.lf, "project_save", lambda **_: True, raising=False)
    tree = SimpleNamespace(get_node=lambda _: None, rename_node=lambda *args: actions.append(args))
    job = {"result": {"title": "Gallery title", "viewerSettings": {}},
           "_local_fields": {"title": "Local title", "viewerSettings": {}}, "_local_environment_path": "local.hdr"}
    controller._apply_local_update(tree, SimpleNamespace(name="incoming", uuid="id"), job,
                                   {"project": project, "stamp": module.file_stamp(path), "old_nodes": []})
    assert actions == ["local.hdr", ("incoming", "Local title")]


def test_deleted_gallery_item_with_failed_upload_does_not_offer_conflict(gallery):
    from lfs_plugins.gallery_controller import asset_sync_state
    project = {"id": "project", "commit_uuid": "local", "exists": True}
    link = {"sceneId": "deleted", "remoteDeleted": True, "commitUuid": "old"}
    job = {"id": "failed", "project": "project", "status": "conflict", "kind": "upload",
           "metadata": {"replaceSceneId": "deleted"}}
    facts = asset_sync_state(project, link, None, [job], checked=True)
    assert "resolve" not in [a["id"] for a in facts["actions"]]
    assert facts["action"] == "cancel"
    assert asset_sync_state(project, link, None, checked=True)["action"] == "publish_again"


def test_apply_gallery_review_can_switch_to_conflict_actions(gallery, monkeypatch):
    from lfs_plugins.gallery_file_panel import GalleryFilePanel
    module = import_module("lfs_plugins.gallery_file_panel")
    controller, _, _ = gallery
    monkeypatch.setattr(module.lf.ui, "get_panel_object", lambda _: None, raising=False)
    monkeypatch.setattr(module.lf.ui, "set_panel_enabled", lambda *_: None, raising=False)
    monkeypatch.setattr(module.lf.ui, "request_redraw", lambda: None, raising=False)
    panel = GalleryFilePanel()
    review = dict(controller=controller, asset={"id": "project"}, scene=scene(), action="conflict", fields={}, mode="conflict")
    panel.show(**review, apply_only=True)
    panel.show(**review, apply_only=False)
    assert panel._review["apply_only"] is False


@pytest.mark.parametrize("action,expected", [("apply_local", {"local_only": True}), ("cancel", "canceled")])
def test_enter_on_conflict_action_does_not_publish(gallery, monkeypatch, action, expected):
    from lfs_plugins.gallery_file_panel import GalleryFilePanel
    from lfs_plugins.rml_keys import KI_RETURN
    from test_asset_manager_panel import _Document
    module = import_module("lfs_plugins.gallery_file_panel")
    monkeypatch.setattr(module.lf.ui, "get_panel_object", lambda _: None, raising=False)
    monkeypatch.setattr(module.lf.ui, "request_redraw", lambda: None, raising=False)
    panel = GalleryFilePanel()
    calls = []
    panel._submit = lambda **kw: calls.append(kw)
    panel._close = lambda _: calls.append("canceled")
    doc = _Document()
    panel.on_mount(doc)
    target = SimpleNamespace(tag_name="button", get_attribute=lambda key, default="": action)
    event = SimpleNamespace(get_parameter=lambda *args: str(KI_RETURN), target=lambda: target, stop_propagation=lambda: None)
    doc.listeners["keydown"](event)
    assert calls == [expected]


def test_enter_in_gallery_description_keeps_editing_but_title_submits(gallery, monkeypatch):
    from lfs_plugins.gallery_file_panel import GalleryFilePanel
    from lfs_plugins.rml_keys import KI_RETURN
    from test_asset_manager_panel import _Document, _Element
    module = import_module("lfs_plugins.gallery_file_panel")
    monkeypatch.setattr(module.lf.ui, "get_panel_object", lambda _: None, raising=False)
    monkeypatch.setattr(module.lf.ui, "request_redraw", lambda: None, raising=False)
    panel = GalleryFilePanel()
    calls = []
    panel._submit = lambda **kw: calls.append(("submit", kw))
    panel._close = lambda _: calls.append(("close",))
    description = _Element({"tag_name": "textarea"})
    doc = _Document({"gallery-file-description": description})
    panel.on_mount(doc)

    # RmlUi delivers Return from the focused textarea's internal widget as a div.
    description.listeners["focus"](None)
    textarea = SimpleNamespace(tag_name="div", get_attribute=lambda *_: "")
    textarea_event = SimpleNamespace(
        get_parameter=lambda *args: str(KI_RETURN), target=lambda: textarea,
        stop_propagation=lambda: calls.append(("stopped",)),
    )
    doc.listeners["keydown"](textarea_event)
    assert calls == []

    description.listeners["blur"](None)
    title = SimpleNamespace(tag_name="input", get_attribute=lambda *_: "")
    title_event = SimpleNamespace(
        get_parameter=lambda *args: str(KI_RETURN), target=lambda: title,
        stop_propagation=lambda: calls.append(("stopped",)),
    )
    doc.listeners["keydown"](title_event)
    assert calls == [("submit", {}), ("stopped",)]


def test_publish_preserves_gallery_description_text_exactly(gallery, monkeypatch):
    from lfs_plugins.gallery_file_panel import GalleryFilePanel
    module = import_module("lfs_plugins.gallery_file_panel")
    monkeypatch.setattr(module.lf.ui, "request_redraw", lambda: None, raising=False)
    description = 'first line\n"quoted" <tag> & 😀 https://example.com/a?x=1&y=2\n'
    submitted = []
    controller = SimpleNamespace(
        service=SimpleNamespace(identity=lambda: "identity"),
        publish_asset=lambda asset, fields, *args, **kwargs: submitted.append(fields),
    )
    panel = GalleryFilePanel()
    panel._review = {
        "controller": controller, "identity": "identity", "action": "publish",
        "open_project": False, "asset": {"id": "project"}, "publish_new": False,
    }
    panel._fields = {"title": "Title", "description": description, "upload_format": "sog"}
    panel._can_submit = lambda: True
    panel._close = lambda _submitted: None

    panel._submit()

    assert submitted[0]["description"] == description


def test_pull_keeps_remote_snapshot_separate_from_local_choices(gallery, monkeypatch, tmp_path):
    controller, _, _ = gallery
    module = import_module("lfs_plugins.gallery_controller")
    project = ("project", str(tmp_path / "project.licht"))
    controller._project_identity = lambda: project
    controller._visible_splats = lambda: []
    controller._acquire_native_use = lambda _: None
    controller._schedule_poll = lambda: None
    controller.service.stage_download = lambda _, **kwargs: "stage"
    monkeypatch.setattr(module.lf, "is_training_active", lambda: False, raising=False)
    monkeypatch.setattr(module.lf.ui, "get_import_state", lambda: {"active": False}, raising=False)
    remote = scene(viewerSettings={"exposure": 0})
    chosen = {"title": "Local title", "description": "", "viewerSettings": {"exposure": 2}}
    controller._local_update_steps.overrides = (remote["id"], chosen, controller._identity, False, "local.hdr")
    job = {"id": "download", "kind": "download", "status": "completed", "result": copy.deepcopy(remote)}
    controller._begin_local_update(job, project)
    assert controller._local_update_steps.pending["result"] == remote
    assert controller._local_update_steps.pending["_local_fields"] == chosen
    assert controller._local_update_steps.pending["_local_environment_path"] == "local.hdr"
    assert job["result"] == remote and "_local_fields" not in job


def _english_tr(monkeypatch):
    translations = json.loads(
        (Path(__file__).resolve().parents[2] / "src/visualizer/gui/resources/locales/en.json").read_text()
    )
    monkeypatch.setattr(import_module("lfs_plugins.gallery_controller").lf.ui, "tr",
                        lambda key: translations.get(key, key))
    return translations


def test_remote_gallery_title_change_names_the_title_field(gallery, monkeypatch):
    from lfs_plugins.gallery_controller import asset_sync_state, conflict_groups
    from lfs_plugins.gallery_sync import shared_fields
    _english_tr(monkeypatch)

    base = scene(viewerSettings={"exposure": 1})
    link = dict(sceneId=base["id"], commitUuid="saved", sharedFields=shared_fields(base),
                contentRevision=base["contentRevision"], metadataRevision=base["metadataRevision"])
    remote = dict(base, title="Portal title", metadataRevision="title-edit")
    facts = asset_sync_state(dict(id="project", commit_uuid="saved", exists=True), link, remote)
    local = dict(shared_fields(base))
    groups = conflict_groups({"id": "project", "commit_uuid": "saved"}, link, local, remote, apply_only=True)

    assert facts["state"] == "remote"
    assert facts["action"] == "apply"
    assert [row["id"] for row in groups] == ["text"]
    assert "My scene" in groups[0]["mine_value"] and "Portal title" in groups[0]["gallery_value"]
    assert facts["change_fields"] == ["Title"]
    assert "View settings" not in facts["change_fields"]
    assert "Scene content" not in facts["change_fields"]
    assert "Portal title" in facts["change_detail"]


def test_remote_gallery_description_and_view_changes_name_those_fields(gallery, monkeypatch):
    from lfs_plugins.gallery_controller import asset_sync_state
    from lfs_plugins.gallery_sync import shared_fields
    _english_tr(monkeypatch)

    base = scene(viewerSettings={"exposure": 1, "cameraPath": {"keyframes": [{"t": 0}]}})
    link = dict(sceneId=base["id"], commitUuid="saved", sharedFields=shared_fields(base),
                contentRevision=base["contentRevision"], metadataRevision=base["metadataRevision"])
    described = dict(base, description="Edited on the portal", metadataRevision="description-edit")
    viewed = dict(base, viewerSettings={"exposure": 3, "cameraPath": {"keyframes": [{"t": 0}]}},
                  metadataRevision="view-edit")
    tracked = dict(base, viewerSettings={"exposure": 1, "cameraPath": {"keyframes": [{"t": 0}, {"t": 1}]}},
                   metadataRevision="track-edit")

    description_facts = asset_sync_state(dict(id="project", commit_uuid="saved", exists=True), link, described)
    view_facts = asset_sync_state(dict(id="project", commit_uuid="saved", exists=True), link, viewed)
    track_facts = asset_sync_state(dict(id="project", commit_uuid="saved", exists=True), link, tracked)

    assert description_facts["change_fields"] == ["Description"]
    assert "Edited on the portal" in description_facts["change_detail"]
    assert view_facts["change_fields"] == ["View settings"]
    assert track_facts["change_fields"] == ["Camera track"]


def test_gallery_title_review_does_not_open_the_project(gallery, monkeypatch, tmp_path):
    from lfs_plugins.gallery_sync import shared_fields
    from test_gallery_product_regressions import _capture_review

    controller, state, _ = gallery
    module = import_module("lfs_plugins.gallery_controller")
    path = tmp_path / "published.licht"
    other = tmp_path / "other.licht"
    path.write_bytes(b"published project")
    other.write_bytes(b"another project")
    remote = scene(viewerSettings={"exposure": 1})
    remote.update(title="Portal title", metadataRevision="title-edit")
    link = dict(sceneId=remote["id"], commitUuid="saved", sharedFields=shared_fields(scene(viewerSettings={"exposure": 1})),
                contentRevision=remote["contentRevision"], metadataRevision="original")
    state.update(scenes=[remote], links={"project": link})
    controller._state = state
    controller._refresh_model = lambda: None
    controller._schedule_poll = lambda: None
    monkeypatch.setattr(module.lf, "project_poll_write", lambda: {"path": str(other)}, raising=False)
    monkeypatch.setattr(module.lf, "project_has_path", lambda: True, raising=False)
    monkeypatch.setattr(module.lf, "project_open",
                        lambda *args, **kwargs: pytest.fail("Opened a project to inspect gallery changes"),
                        raising=False)
    monkeypatch.setattr(module, "capture_view",
                        lambda _lf: pytest.fail("Captured a live view to inspect gallery changes"))
    monkeypatch.setattr(import_module("lfs_plugins.training_confirm"), "confirm_discard_work_then",
                        lambda title, callback: pytest.fail("Asked to switch projects to inspect gallery changes"))
    reviews = _capture_review(monkeypatch)
    asset = {"id": "project", "path": str(path), "commit_uuid": "saved"}

    controller.resolve_asset(asset, {"title": "My scene", "description": ""}, apply_only=True)

    assert controller._decision_pending
    assert len(reviews) == 1
    groups = {row["id"]: row for row in reviews[0]["groups"]}
    assert set(groups) == {"text"}
    assert "Portal title" in groups["text"]["gallery_value"]
    assert reviews[0]["apply_only"] is True


def test_applying_inspected_gallery_changes_opens_the_project(gallery, monkeypatch, tmp_path):
    from lfs_plugins.gallery_sync import shared_fields
    from test_gallery_product_regressions import _capture_review

    controller, state, _ = gallery
    module = import_module("lfs_plugins.gallery_controller")
    path = tmp_path / "published.licht"
    other = tmp_path / "other.licht"
    path.write_bytes(b"published project")
    other.write_bytes(b"another project")
    remote = scene(viewerSettings={"exposure": 2})
    remote.update(metadataRevision="view-edit")
    state.update(scenes=[remote], links={"project": dict(
        sceneId=remote["id"], commitUuid="saved", sharedFields=shared_fields(scene(viewerSettings={"exposure": 1})),
        contentRevision=remote["contentRevision"], metadataRevision="original")})
    controller._state = state
    controller._refresh_model = lambda: None
    controller._schedule_poll = lambda: None
    monkeypatch.setattr(module.lf, "project_poll_write", lambda: {"path": str(other)}, raising=False)
    monkeypatch.setattr(module.lf, "project_has_path", lambda: True, raising=False)
    opened = []
    monkeypatch.setattr(module.lf, "project_open",
                        lambda *args, **kwargs: opened.append(args[0]), raising=False)
    monkeypatch.setattr(import_module("lfs_plugins.training_confirm"), "confirm_discard_work_then",
                        lambda title, callback: callback(False))
    reviews = _capture_review(monkeypatch)
    asset = {"id": "project", "path": str(path), "commit_uuid": "saved"}
    controller.resolve_asset(asset, {"title": "My scene", "description": ""}, apply_only=True)
    assert opened == []

    reviews[0]["on_submit"]({row["id"]: "gallery" for row in reviews[0]["groups"]})

    assert opened == [str(path)]
    assert controller._open_continuation[0] == str(path)


@pytest.mark.parametrize("same_project", [False, True])
def test_applying_closed_description_change_keeps_current_project_open(gallery, monkeypatch, tmp_path, same_project):
    from lfs_plugins.gallery_sync import shared_fields
    from test_gallery_product_regressions import _capture_review

    controller, state, _ = gallery
    module = import_module("lfs_plugins.gallery_controller")
    path = tmp_path / "published.licht"
    other = tmp_path / "current.licht"
    path.write_bytes(b"published project")
    other.write_bytes(b"current project")
    base = scene(viewerSettings={})
    remote = dict(base, description="Changed in portal", metadataRevision="description-edit")
    state.update(scenes=[remote], links={"project": dict(
        sceneId=remote["id"], commitUuid="saved", sharedFields=shared_fields(base),
        contentRevision=remote["contentRevision"], metadataRevision=base["metadataRevision"])})
    controller._state = state
    controller._refresh_model = lambda: None
    controller._schedule_poll = lambda: None
    monkeypatch.setattr(module.lf, "project_poll_write", lambda: {"path": str(path if same_project else other)}, raising=False)
    dirty = [False]
    monkeypatch.setattr(module.lf, "project_is_dirty", lambda: dirty[0], raising=False)
    controller._project_identity = lambda: ("project", str(path.resolve()))
    monkeypatch.setattr(module.lf, "project_open", lambda *args, **kwargs: pytest.fail("Opened the project"), raising=False)
    view = [{}]
    views = []
    monkeypatch.setattr(module, "capture_view", lambda _lf: views.append(True) or copy.deepcopy(view[0]))
    monkeypatch.setattr(module, "restore_view", lambda *_args, **_kwargs: pytest.fail("Changed the live view"))
    monkeypatch.setattr(import_module("lfs_plugins.training_confirm"), "confirm_discard_work_then",
                        lambda title, callback: pytest.fail("Asked to switch projects"))
    calls = []
    controller.service.acknowledge_gallery_text = lambda *args: calls.append(args)
    reviews = _capture_review(monkeypatch)
    controller.resolve_asset({"id": "project", "path": str(path), "commit_uuid": "saved"},
                             {"title": base["title"], "description": base["description"]}, apply_only=True)
    before = path.read_bytes()
    if same_project:
        dirty[0] = True
        view[0] = {"exposure": 2}

    reviews[0]["on_submit"]({"text": "gallery"})

    assert calls == [(remote, "project", str(path), module.file_stamp(path), state["links"]["project"])]
    assert path.read_bytes() == before
    assert views == ([True] if same_project else [])


def test_applying_gallery_view_keeps_null_camera_track(gallery, monkeypatch, tmp_path):
    from lfs_plugins.gallery_sync import shared_fields
    from test_gallery_product_regressions import _capture_review

    controller, state, actions = gallery
    module = import_module("lfs_plugins.gallery_controller")
    path = tmp_path / "published.licht"
    path.write_bytes(b"published project")
    base = scene(viewerSettings={"camera": {"position": [0, 2, -7],
        "up": [0.15881019830703735, 0.9687422513961792, 0.19057223200798035],
        "fov": 55.000003814697266}, "cameraPath": None})
    remote = scene(viewerSettings={"camera": {"position": [-5, 2, -6],
        "up": [0, 1, 0], "fov": 55.0}, "cameraPath": None})
    remote["metadataRevision"] = "view-edit"
    state.update(scenes=[remote], links={"project": dict(
        sceneId=remote["id"], commitUuid="saved", sharedFields=shared_fields(base),
        contentRevision=remote["contentRevision"], metadataRevision=base["metadataRevision"])})
    controller._state = state
    controller._refresh_model = lambda: None
    controller._schedule_poll = lambda: None
    controller._resolve_pending_uploads = lambda project, scene, identity, callback: callback()
    controller._begin_settings_apply = lambda asset, scene, metadata, **kwargs: actions.append(metadata)
    monkeypatch.setattr(module.lf, "project_has_path", lambda: True, raising=False)
    monkeypatch.setattr(module.lf, "project_poll_write", lambda: {"path": str(path)}, raising=False)
    monkeypatch.setattr(module.lf, "project_is_dirty", lambda: False, raising=False)
    monkeypatch.setattr(module, "capture_view", lambda _lf: base["viewerSettings"])
    reviews = _capture_review(monkeypatch)
    controller.resolve_asset({"id": "project", "path": str(path), "commit_uuid": "saved"},
                             {"title": base["title"], "description": base["description"]}, apply_only=True)

    reviews[0]["on_submit"]({"view": "gallery"})

    assert actions[0]["viewerSettings"] == remote["viewerSettings"]


def _closed_title_review(gallery, monkeypatch, tmp_path, *, view_change=False):
    from lfs_plugins.gallery_sync import shared_fields
    from test_gallery_product_regressions import _capture_review

    controller, state, actions = gallery
    module = import_module("lfs_plugins.gallery_controller")
    path = tmp_path / "published.licht"
    other = tmp_path / "other.licht"
    path.write_bytes(b"published project")
    other.write_bytes(b"another project")
    remote = scene(viewerSettings={"exposure": 2} if view_change else {})
    remote.update(title="Portal title", metadataRevision="title-edit")
    state.update(scenes=[remote], links={"project": dict(
        sceneId=remote["id"], commitUuid="saved", sharedFields=shared_fields(
            scene(viewerSettings={"exposure": 1} if view_change else {})),
        contentRevision=remote["contentRevision"], metadataRevision="original")})
    controller._state = state
    controller._refresh_model = lambda: None
    controller._schedule_poll = lambda: None
    current = [str(other)]
    opened = []
    monkeypatch.setattr(module.lf, "project_poll_write", lambda: {"path": current[0]}, raising=False)
    monkeypatch.setattr(module.lf, "project_has_path", lambda: True, raising=False)
    monkeypatch.setattr(module.lf, "project_open",
                        lambda *args, **kwargs: opened.append(args[0]) or current.__setitem__(0, args[0]),
                        raising=False)
    monkeypatch.setattr(import_module("lfs_plugins.training_confirm"), "confirm_discard_work_then",
                        lambda title, callback: callback(False))
    reviews = _capture_review(monkeypatch)
    asset = {"id": "project", "path": str(path), "commit_uuid": "saved"}
    controller.resolve_asset(asset, {"title": "My scene", "description": ""}, apply_only=True)
    return controller, module, path, other, opened, current, reviews, actions, asset


def test_canceling_gallery_review_does_not_open_or_change_the_project(gallery, monkeypatch, tmp_path):
    controller, _, path, other, opened, current, reviews, actions, _ = _closed_title_review(
        gallery, monkeypatch, tmp_path)
    before = path.read_bytes()

    reviews[0]["on_done"](False)

    assert opened == []
    assert current == [str(other)]
    assert controller._open_continuation is None
    assert not controller._decision_pending
    assert actions == []
    assert path.read_bytes() == before
    assert other.read_bytes() == b"another project"


def test_delayed_apply_rejects_a_changed_file_stamp(gallery, monkeypatch, tmp_path):
    controller, _, path, other, opened, current, reviews, actions, _ = _closed_title_review(
        gallery, monkeypatch, tmp_path, view_change=True)
    path.write_bytes(b"changed on disk during review")

    with pytest.raises(ValueError, match="changed"):
        reviews[0]["on_submit"]({row["id"]: "gallery" for row in reviews[0]["groups"]})

    assert opened == []
    assert controller._open_continuation is None
    assert actions == []
    assert current == [str(other)]


def test_delayed_apply_rejects_account_change_before_open(gallery, monkeypatch, tmp_path):
    controller, _, _, other, opened, current, reviews, actions, _ = _closed_title_review(
        gallery, monkeypatch, tmp_path, view_change=True)
    controller._state["identity"] = ("https://portal.example", "other@example.com", "second", True)

    with pytest.raises(ValueError, match="changed"):
        reviews[0]["on_submit"]({row["id"]: "gallery" for row in reviews[0]["groups"]})

    assert opened == []
    assert controller._open_continuation is None
    assert actions == []
    assert current == [str(other)]


def test_delayed_apply_rejects_an_identity_mismatch_after_open(gallery, monkeypatch, tmp_path):
    controller, _, path, _, opened, _, reviews, actions, _ = _closed_title_review(
        gallery, monkeypatch, tmp_path, view_change=True)
    controller._project_identity = lambda: ("other-project", str(path.resolve()))

    reviews[0]["on_submit"]({row["id"]: "gallery" for row in reviews[0]["groups"]})
    with pytest.raises(ValueError, match="changed"):
        controller._open_continuation[2]()

    assert opened == [str(path)]
    assert actions == []


def test_open_project_review_rejects_a_switched_project(gallery, monkeypatch, tmp_path):
    from lfs_plugins.gallery_sync import shared_fields
    from test_gallery_product_regressions import _capture_review

    controller, state, actions = gallery
    module = import_module("lfs_plugins.gallery_controller")
    path = tmp_path / "published.licht"
    other = tmp_path / "other.licht"
    path.write_bytes(b"published project")
    other.write_bytes(b"another project")
    remote = scene(viewerSettings={"exposure": 2})
    remote.update(title="Portal title", metadataRevision="title-edit")
    state.update(scenes=[remote], links={"project": dict(
        sceneId=remote["id"], commitUuid="saved", sharedFields=shared_fields(scene(viewerSettings={"exposure": 1})),
        contentRevision=remote["contentRevision"], metadataRevision="original")})
    controller._state = state
    controller._refresh_model = lambda: None
    controller._schedule_poll = lambda: None
    current = [str(path)]
    opened = []
    controller._project_identity = lambda: (
        ("project", str(path.resolve())) if Path(current[0]).resolve() == path.resolve()
        else ("other-project", str(Path(current[0]).resolve())))
    monkeypatch.setattr(module.lf, "project_poll_write", lambda: {"path": current[0]}, raising=False)
    monkeypatch.setattr(module.lf, "project_has_path", lambda: True, raising=False)
    monkeypatch.setattr(module.lf, "project_is_dirty", lambda: False, raising=False)
    monkeypatch.setattr(module.lf, "project_open",
                        lambda *args, **kwargs: opened.append(args[0]), raising=False)
    monkeypatch.setattr(module, "capture_view", lambda _lf: {})
    monkeypatch.setattr(import_module("lfs_plugins.training_confirm"), "confirm_discard_work_then",
                        lambda title, callback: pytest.fail("Reopened the reviewed project after a switch"))
    reviews = _capture_review(monkeypatch)
    asset = {"id": "project", "path": str(path), "commit_uuid": "saved"}
    controller.resolve_asset(asset, {"title": "My scene", "description": ""}, apply_only=True)
    current[0] = str(other)

    with pytest.raises(ValueError, match="changed"):
        reviews[0]["on_submit"]({row["id"]: "gallery" for row in reviews[0]["groups"]})

    assert opened == []
    assert controller._open_continuation is None
    assert actions == []
    assert path.read_bytes() == b"published project"


def test_open_project_review_still_guards_dirty_and_view(gallery, monkeypatch, tmp_path):
    from lfs_plugins.gallery_sync import shared_fields
    from test_gallery_product_regressions import _capture_review

    controller, state, actions = gallery
    module = import_module("lfs_plugins.gallery_controller")
    path = tmp_path / "published.licht"
    path.write_bytes(b"published project")
    remote = scene(viewerSettings={"exposure": 2})
    remote.update(title="Portal title", metadataRevision="title-edit")
    state.update(scenes=[remote], links={"project": dict(
        sceneId=remote["id"], commitUuid="saved", sharedFields=shared_fields(scene(viewerSettings={"exposure": 1})),
        contentRevision=remote["contentRevision"], metadataRevision="original")})
    controller._state = state
    controller._refresh_model = lambda: None
    controller._schedule_poll = lambda: None
    controller._project_identity = lambda: ("project", str(path.resolve()))
    dirty = [False]
    view = [{}]
    opened = []
    monkeypatch.setattr(module.lf, "project_poll_write", lambda: {"path": str(path)}, raising=False)
    monkeypatch.setattr(module.lf, "project_has_path", lambda: True, raising=False)
    monkeypatch.setattr(module.lf, "project_is_dirty", lambda: dirty[0], raising=False)
    monkeypatch.setattr(module.lf, "project_open",
                        lambda *args, **kwargs: opened.append(args[0]), raising=False)
    monkeypatch.setattr(module, "capture_view", lambda _lf: copy.deepcopy(view[0]))
    reviews = _capture_review(monkeypatch)
    asset = {"id": "project", "path": str(path), "commit_uuid": "saved"}
    controller.resolve_asset(asset, {"title": "My scene", "description": ""}, apply_only=True)
    dirty[0] = True
    with pytest.raises(ValueError, match="changed"):
        reviews[0]["on_submit"]({row["id"]: "gallery" for row in reviews[0]["groups"]})
    dirty[0] = False
    view[0] = {"exposure": 3}
    with pytest.raises(ValueError, match="changed"):
        reviews[0]["on_submit"]({row["id"]: "gallery" for row in reviews[0]["groups"]})
    assert opened == []
    assert actions == []
    assert path.read_bytes() == b"published project"


@pytest.mark.parametrize("view_choice,track_choice,dirty,has_track", [
    (None, None, False, True),
    ("gallery", None, False, True),
    (None, "gallery", False, True),
    (None, "both", False, True),
    (None, "both", False, False),
    (None, None, True, True),
])
def test_closed_review_preserves_saved_local_view(
        gallery, monkeypatch, tmp_path, view_choice, track_choice, dirty, has_track):
    from lfs_plugins.gallery_sync import shared_fields
    from test_gallery_product_regressions import _capture_review

    controller, state, actions = gallery
    module = import_module("lfs_plugins.gallery_controller")
    path, other = tmp_path / "saved.licht", tmp_path / "other.licht"
    path.write_bytes(b"locally saved view with exposure=2")
    other.write_bytes(b"other project")
    old_track = {"duration": 1, "keyframes": [{"time": 0, "label": "published"}]}
    saved_track = {"duration": 2, "keyframes": [{"time": 1, "label": "saved"}]}
    remote_track = {"duration": 3, "keyframes": [{"time": 0, "label": "gallery"}]}
    published = scene(viewerSettings={"exposure": 1, "cameraPath": old_track})
    remote = scene(viewerSettings={"exposure": 3 if view_choice == "gallery" else 1, "cameraPath": remote_track})
    remote.update(title="New Gallery title", metadataRevision="new-title")
    state.update(scenes=[remote], links={"project": dict(
        sceneId=remote["id"], commitUuid="published-commit", sharedFields=shared_fields(published),
        contentRevision="original", metadataRevision="original")})
    controller._state = state
    controller._refresh_model = lambda: None
    controller._schedule_poll = lambda: None
    current = [str(other)]
    monkeypatch.setattr(module.lf, "project_poll_write", lambda: {"path": current[0]}, raising=False)
    monkeypatch.setattr(module.lf, "project_has_path", lambda: True, raising=False)
    monkeypatch.setattr(module.lf, "project_is_dirty", lambda: False, raising=False)
    monkeypatch.setattr(module.lf, "project_open",
                        lambda p, *a, **kw: current.__setitem__(0, p), raising=False)
    def capture_opened_view(_lf):
        assert current[0] == str(path), "The previous project's view must not be captured"
        return dict(exposure=2, **({"cameraPath": copy.deepcopy(saved_track)} if has_track else {}))
    monkeypatch.setattr(module, "capture_view", capture_opened_view)
    monkeypatch.setattr(import_module("lfs_plugins.training_confirm"), "confirm_discard_work_then",
                        lambda title, cb: cb(False))
    controller._resolve_pending_uploads = lambda project, scene, identity, cb: cb()
    controller._begin_settings_apply = lambda asset, scene, metadata, **kw: actions.append(metadata)
    reviews = _capture_review(monkeypatch)
    controller.resolve_asset({"id": "project", "path": str(path), "commit_uuid": "new-local-save"},
                             {"title": published["title"], "description": published["description"]}, apply_only=True)
    assert current == [str(other)]
    decisions = {"text": "gallery"}
    if view_choice:
        decisions["view"] = view_choice
    if track_choice:
        decisions["track"] = track_choice
    reviews[0]["on_submit"](decisions)
    assert actions == []
    monkeypatch.setattr(module.lf, "project_is_dirty", lambda: dirty, raising=False)
    if dirty or (track_choice == "both" and not has_track):
        with pytest.raises(ValueError, match="changed"):
            controller._open_continuation[2]()
        assert actions == []
        return
    controller._open_continuation[2]()
    assert len(actions) == 1
    assert actions[0]["title"] == "New Gallery title"
    assert actions[0]["viewerSettings"]["exposure"] == (3 if view_choice == "gallery" else 2)
    expected_track = remote_track if track_choice == "gallery" else saved_track
    if track_choice == "both":
        expected_track = {"duration": 5, "keyframes": [
            {"time": 1, "label": "saved"}, {"time": 2, "label": "gallery"}]}
    assert actions[0]["viewerSettings"]["cameraPath"] == expected_track

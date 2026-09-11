# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Account boundaries and revision-sensitive actions in the Gallery panel."""
from importlib import import_module
from contextlib import nullcontext
from types import SimpleNamespace
import copy
import math
import struct

import pytest

from test_asset_manager_panel import panel_module, _Handle


@pytest.fixture
def gallery(monkeypatch, panel_module):
    module = import_module("lfs_plugins.gallery_panel")
    monkeypatch.setattr(module.lf, "scene", SimpleNamespace(NodeType=SimpleNamespace(SPLAT=0)), raising=False)
    state = {
        "identity": ("https://portal.example", "one@example.com", "first", True),
        "signed_in": True, "connected": True, "email": "one@example.com",
        "display_name": "One", "scenes": [], "jobs": [], "links": {},
        "message": "", "busy": False, "version": 0, "trackFetch": {},
    }
    actions = []
    service = SimpleNamespace(snapshot=lambda: state.copy(), identity=lambda: state["identity"], busy=False, pause=lambda: None,
        edit=lambda *args: actions.append(args),
        send_camera_track=lambda *args: actions.append(("send_camera_track", *args)),
        fetch_camera_track=lambda scene_id: actions.append(("fetch_camera_track", scene_id)) or "fetch-op",
        local_use=lambda job_id: nullcontext())
    monkeypatch.setattr(module, "get_gallery_sync", lambda: service)
    monkeypatch.setattr(module.lf.ui, "cancel_export", lambda: actions.append("cancel-export"), raising=False)
    monkeypatch.setattr(module.lf.ui, "dismiss_import", lambda: actions.append("dismiss-import"), raising=False)
    monkeypatch.setattr(module.lf.ui, "get_import_state", lambda: {"active": False}, raising=False)
    panel = module.GalleryPanel()
    panel._handle = _Handle()
    return panel, state, actions


def scene(**fields):
    return dict(id="private-one", title="My scene", description="Private description",
        visibility="private", revision="original", status="ready", **fields)


def test_per_frame_account_check_does_not_copy_transfer_history(gallery):
    panel, state, actions = gallery
    panel.service.snapshot = lambda: (_ for _ in ()).throw(AssertionError("History copied during account check"))
    panel.service.pause = lambda: actions.append("paused")
    assert panel._check_identity() is False
    panel._title = "Previous account's draft"
    state["identity"] = ("https://portal.example", "two@example.com", "second", True)
    assert panel._check_identity() is True
    assert panel._title == "" and actions == ["paused"]


@pytest.mark.parametrize("native_active,expected", [(True, 50), (False, 40)])
def test_progress_painting_uses_the_existing_model_without_copying_history(gallery, monkeypatch, native_active, expected):
    panel, state, _ = gallery
    module = import_module("lfs_plugins.gallery_panel")
    panel._import_pending = {"id": "download"}
    panel._state["jobs"] = [{"id": "download", "stagedImport": {"completed": 4, "total": 10}}]
    monkeypatch.setattr(panel, "_finish_import", lambda: None)
    monkeypatch.setattr(module.lf.ui, "get_import_state", lambda: {"active": native_active, "progress": .5}, raising=False)
    panel.service.snapshot = lambda: (_ for _ in ()).throw(AssertionError("History copied while painting progress"))
    panel._advance_phases()
    assert panel._export_progress == expected
    assert panel._import_pending == {"id": "download"}


@pytest.mark.parametrize("outcome", ["success", "canceled", "account", "edited", "generation", "project", "failed"])
def test_background_save_never_continues_before_its_unchanged_generation(gallery, monkeypatch, outcome):
    panel, state, actions = gallery
    module = import_module("lfs_plugins.gallery_panel")
    project = ["project", "/project.licht"]
    poll = {"running": False, "generation": 5, "path": project[1], "error": ""}
    monkeypatch.setattr(panel, "_project_identity", lambda: tuple(project))
    monkeypatch.setattr(module.lf, "project_poll_write", lambda: dict(poll), raising=False)
    monkeypatch.setattr(module.lf, "project_is_dirty", lambda: outcome == "edited", raising=False)
    monkeypatch.setattr(panel, "_schedule_phase_poll", lambda: None)
    def save(**kwargs):
        assert kwargs == {"wait": False, "regenerate_preview": False}
        poll["running"] = True
        return True
    monkeypatch.setattr(module.lf, "project_save", save, raising=False)
    panel._save_current_project(lambda: actions.append("continued"))
    assert panel._save_pending and panel._can_pause()
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


@pytest.mark.parametrize("outcome", ["success", "failed", "account", "project", "generation", "edited"])
def test_new_project_registration_waits_for_successful_native_write(gallery, monkeypatch, outcome):
    import sys
    panel, state, actions = gallery
    module = import_module("lfs_plugins.gallery_panel")
    poll = {"running": True, "path": "", "generation": 0, "error": ""}
    job = {"id": "download", "_accountIdentity": state["identity"],
        "_save_new": {"path": "/download.licht", "title": "Downloaded scene", "generation": 1}}
    panel._import_pending = job
    monkeypatch.setattr(module.lf, "project_poll_write", lambda: dict(poll), raising=False)
    monkeypatch.setattr(module.lf, "project_is_dirty", lambda: outcome == "edited", raising=False)
    index = SimpleNamespace(load=lambda: True, register_licht_asset=lambda *args, **kwargs: (actions.append("registered") or object(), None))
    monkeypatch.setitem(sys.modules, "lfs_plugins.asset_index", SimpleNamespace(AssetIndex=lambda: index))
    monkeypatch.setattr(module.lf, "io", SimpleNamespace(inspect_project=lambda path: SimpleNamespace(project_uuid="saved-id")), raising=False)
    panel.service.link_download = lambda *args: actions.append("linked") or "operation"
    monkeypatch.setattr(panel, "_refresh_model", lambda: None)
    monkeypatch.setattr(panel, "_schedule_phase_poll", lambda: None)
    assert not panel._can_pause()
    panel._finish_import()
    assert not actions and panel._import_pending is job
    poll.update(running=False, path="/different.licht" if outcome == "project" else "/download.licht", generation=2 if outcome == "generation" else 1,
        error="disk full" if outcome == "failed" else "")
    if outcome == "account":
        state["identity"] = ("https://portal.example", "two@example.com", "second", True)
    if outcome in ("failed", "project", "generation", "edited"):
        with pytest.raises(ValueError):
            panel._finish_import()
        assert not actions
    else:
        panel._finish_import()
        assert actions == (["registered", "linked"] if outcome == "success" else ["registered"])
    if outcome == "success":
        assert panel._import_pending.get("_link") == "operation" and "_save_new" not in panel._import_pending
    else:
        assert panel._import_pending is None


def test_bundle_open_rechecks_other_native_work_after_staging(gallery, monkeypatch):
    panel, state, actions = gallery
    module = import_module("lfs_plugins.gallery_panel")
    panel._import_pending = {"id": "download", "_accountIdentity": state["identity"],
        "_bundle": {"phase": "staging", "scene": SimpleNamespace(is_valid=lambda: True)}}
    monkeypatch.setattr(module.lf, "project_is_dirty", lambda: False, raising=False)
    monkeypatch.setattr(module.lf, "is_training_active", lambda: False, raising=False)
    monkeypatch.setattr(module.lf.ui, "get_import_state", lambda: {"active": True}, raising=False)
    monkeypatch.setattr(module.lf, "new_project", lambda: actions.append("closed project"), raising=False)
    with pytest.raises(ValueError, match="Finish training or the current import"):
        panel._finish_import()
    assert not actions


def test_cancel_native_bundle_import_keeps_download_without_saving_or_linking(gallery, monkeypatch):
    panel, _, actions = gallery
    module = import_module("lfs_plugins.gallery_panel")
    panel._import_pending = {"id": "download", "path": "download.lfsg", "_native_path": "preview.scene",
        "_bundle": {"phase": "importing"}}
    monkeypatch.setattr(module.lf.ui, "cancel_gallery_import", lambda: actions.append("cancel requested"), raising=False)
    monkeypatch.setattr(module.lf.ui, "get_import_state", lambda: {"active": False}, raising=False)
    monkeypatch.setattr(module.lf, "get_scene", lambda: SimpleNamespace(get_node=lambda name: None), raising=False)
    monkeypatch.setattr(module.lf, "project_has_path", lambda: False, raising=False)
    monkeypatch.setattr(panel, "_refresh_model", lambda: None)
    panel._action_pause()
    panel._finish_import()
    assert panel._import_pending is None and actions == ["cancel requested"]
    assert panel._message == "Import canceled. Your download was kept."


def test_update_link_failure_reports_already_saved_project(gallery, monkeypatch, tmp_path):
    panel, state, actions = gallery
    module = import_module("lfs_plugins.gallery_panel")
    source = tmp_path / "project.licht"
    source.write_bytes(b"saved local generation")
    project = ("project", str(source))
    incoming = SimpleNamespace(uuid="incoming", name="preview")
    native_scene = SimpleNamespace(get_node=lambda name: incoming, get_node_by_uuid=lambda identifier: incoming)
    update = {"project": project, "phase": "backup", "path": str(tmp_path / "preview.scene"),
        "incoming": "incoming", "generation": 3, "stamp": module.file_stamp(source), "backup_id": "backup"}
    job = {"id": "download", "_update": update}
    state["jobs"] = [{"id": "download", "localUpdate": {"id": "backup", "state": "ready"}}]
    monkeypatch.setattr(panel, "_project_identity", lambda: project)
    monkeypatch.setattr(panel, "_apply_local_update", lambda *args: actions.append("project saved"))
    monkeypatch.setattr(module.lf, "get_scene", lambda: native_scene, raising=False)
    monkeypatch.setattr(module.lf, "project_is_dirty", lambda: False, raising=False)
    monkeypatch.setattr(module.lf, "project_poll_write", lambda: {"generation": 3}, raising=False)
    panel.service.link_download = lambda *args: (_ for _ in ()).throw(ValueError("Account changed"))
    panel._finish_local_update(job)
    assert update["phase"] == "save_updated"
    monkeypatch.setattr(module.lf, "project_poll_write", lambda: {"generation": 4}, raising=False)
    with pytest.raises(ValueError, match="The project was updated and its recovery copy was kept"):
        panel._finish_local_update(job)
    assert actions == ["project saved"] and update["phase"] == "linking"


@pytest.mark.parametrize("outcome", ["success", "edited", "generation", "project", "account", "failed"])
def test_update_final_save_links_only_its_clean_committed_project(gallery, monkeypatch, outcome):
    panel, state, actions = gallery
    module = import_module("lfs_plugins.gallery_panel")
    project = ("project", "/project.licht")
    job = {"id": "download", "_accountIdentity": state["identity"],
        "_update": {"project": project, "phase": "save_updated", "generation": 3}}
    panel._import_pending = job
    poll = {"running": True, "generation": 3, "error": ""}
    monkeypatch.setattr(module.lf, "project_poll_write", lambda: poll.copy(), raising=False)
    monkeypatch.setattr(module.lf, "project_is_dirty", lambda: outcome == "edited", raising=False)
    monkeypatch.setattr(panel, "_project_identity", lambda: ("other", "/other.licht") if outcome == "project" else project)
    monkeypatch.setattr(panel, "_recover_failed_update", lambda *a: (_ for _ in ()).throw(ValueError("recovery available")))
    panel.service.link_download = lambda *a: actions.append("linked") or "operation"
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
    module = import_module("lfs_plugins.gallery_panel")
    project = ("project", "/project.licht")
    panel._import_pending = {"id": "download", "_update": {"project": project, "phase": phase, "incoming": "owned-uuid"}}
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
    assert panel._import_pending is None and panel._save_pending is None
    assert "disk full" in panel._message


def test_account_switch_clears_form_and_cannot_accept_previous_confirmation(gallery):
    panel, state, actions = gallery
    state["scenes"] = [scene()]
    panel._refresh_model()
    panel._action_select("private-one")
    panel._confirm = ("Remove private scene?", lambda: actions.append("deleted"))
    state.update(identity=("https://portal.example", "two@example.com", "second", True),
        scenes=[], email="two@example.com", display_name="Two")
    # Connected may already be true for the new account before the next UI frame.
    panel._dispatch("confirm_action", [])
    assert actions == []
    assert panel._scene is None and panel._confirm is None
    assert panel._title == panel._description == ""
    assert "account changed" in panel._message.lower()
    assert panel._handle.records["scenes"] == []


def test_recovery_folder_action_reveals_only_service_folder(gallery, tmp_path, monkeypatch):
    panel, state, actions = gallery
    module = import_module("lfs_plugins.gallery_panel")
    panel.service.root = tmp_path / "gallery"
    state.update(storage_issue=True, connected=False)
    monkeypatch.setattr(module.lf.ui, "reveal_in_file_manager", lambda path: actions.append(path), raising=False)
    panel._dispatch("show_recovery_folder", [])
    assert actions == [str(panel.service.root)]


def test_old_paused_upload_stays_visible_and_history_is_accessible(gallery):
    panel, state, _ = gallery
    state["jobs"] = [dict(id="pending", status="paused")] + [dict(id=str(n), status="completed") for n in range(45)]
    panel._state = state.copy()
    assert panel._visible_jobs()[0]["id"] == "pending"
    assert len(panel._visible_jobs()) == 31
    panel._action_more_history()
    assert len(panel._visible_jobs()) == 46
    state["identity"] = ("https://portal.example", "two@example.com", "second", True)
    panel._check_identity()
    assert panel._history_limit == 30


def test_clear_finished_review_captures_ids_and_account_switch_cancels(gallery):
    panel, state, actions = gallery
    state["jobs"] = [dict(id="finished", status="completed"), dict(id="pending", status="paused"),
        dict(id="backup", status="completed", retired=True)]
    panel._state = state.copy()
    panel.service.clear_finished = lambda ids: actions.append(ids)
    panel._action_clear_finished()
    assert "Clear 1 finished" in panel._confirm[0]
    state["jobs"].append(dict(id="later", status="completed"))
    panel._action_confirm_action()
    assert actions == [("finished",)]
    panel._action_clear_finished()
    state["identity"] = ("https://portal.example", "two@example.com", "second", True)
    panel._check_identity()
    panel._action_confirm_action()
    assert actions == [("finished",)]


def test_native_lease_survives_detach_until_import_idle(gallery, monkeypatch):
    from contextlib import contextmanager
    panel, _, actions = gallery
    module = import_module("lfs_plugins.gallery_panel")
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
    monkeypatch.setattr(panel, "_schedule_phase_poll", lambda: actions.append("poll"))
    panel._acquire_native_use("job")
    panel._import_detached = True
    panel._release_native_use()
    assert actions == ["acquired", "poll"]
    native["active"] = False
    panel._acquire_native_use("retry")
    assert actions[-2:] == ["released", "acquired"]
    panel._release_native_use()
    assert actions[-1] == "released" and panel._native_use is None


def test_account_switch_cancels_pending_scene_preparation(gallery, monkeypatch):
    panel, state, actions = gallery
    panel._export_pending = ("private.ply", {}, "project", 0)
    module = import_module("lfs_plugins.gallery_panel")
    monkeypatch.setattr(module.lf.ui, "get_export_state", lambda: {"active": True, "path": "private.ply"}, raising=False)
    state["identity"] = ("https://portal.example", "two@example.com", "second", True)
    panel._refresh_model()
    assert actions == ["cancel-export"]
    assert panel._export_cancelled


def test_failed_export_with_partial_file_is_never_uploaded(gallery, tmp_path, monkeypatch):
    panel, _, actions = gallery
    module = import_module("lfs_plugins.gallery_panel")
    export = tmp_path / "partial.ply"
    monkeypatch.setattr(module.lf.ui, "get_export_state", lambda: {"active": False, "outcome": "failed", "path": str(export)}, raising=False)
    export.write_bytes(b"unfinished export")
    panel._export_pending = (export, {}, "project", 0)
    panel._finish_export()
    assert panel._export_pending is None
    assert not export.exists()
    assert actions == []
    assert "failed" in panel._message


def test_preparation_progress_tracks_own_export(gallery, tmp_path, monkeypatch):
    panel, _, actions = gallery
    module = import_module("lfs_plugins.gallery_panel")
    export = tmp_path / "own.ply"
    panel._export_pending = (export, {}, "project", 0)
    monkeypatch.setattr(module.lf.ui, "get_export_state", lambda: {"active": True, "path": str(export), "progress": 0.42}, raising=False)
    panel._finish_export()
    assert panel._export_progress == 42
    assert "42%" in panel._message
    assert panel._export_pending is not None and not actions


def test_publish_uses_native_scene_capture_when_portal_advertises_bundles(gallery, tmp_path, monkeypatch):
    panel, state, actions = gallery
    module = import_module("lfs_plugins.gallery_panel")
    state["source_formats"] = ["ply", "licht"]
    panel.service.root = tmp_path
    monkeypatch.setattr(panel, "_project_identity", lambda: ("project", "/project.licht"))
    monkeypatch.setattr(panel, "_save_current_project", lambda proceed: proceed())
    monkeypatch.setattr(panel, "_schedule_phase_poll", lambda: None)
    monkeypatch.setattr(module.lf, "get_scene", lambda: SimpleNamespace(get_nodes=lambda: [SimpleNamespace(id=1, name="scene", type=module.lf.scene.NodeType.SPLAT)], is_node_effectively_visible=lambda node_id: True), raising=False)
    monkeypatch.setattr(module.lf.ui, "get_export_state", lambda: {"active": False}, raising=False)
    monkeypatch.setattr(module.lf, "prepare_gallery_scene", lambda path, payload_format: actions.append((path, payload_format)), raising=False)
    monkeypatch.setattr(module.lf, "export_scene", lambda *args, **kwargs: pytest.fail("Must preserve local multi-object geometry"), raising=False)
    panel._publish({"title": "Scene"})
    assert panel._export_pending[0].suffix == ".scene"
    assert actions == [(str(panel._export_pending[0]), "ply")]


def test_completed_native_scene_hands_off_to_background_packaging(gallery, tmp_path, monkeypatch):
    import uuid
    panel, _, actions = gallery
    module = import_module("lfs_plugins.gallery_panel")
    export = tmp_path / (str(uuid.uuid4()) + ".scene")
    export.mkdir()
    panel._export_pending = (export, {"title": "Scene"}, "project", 0)
    panel.service.queue_prepared_upload = lambda *args: actions.append(args)
    monkeypatch.setattr(module.lf.ui, "get_export_state", lambda: {"active": False, "outcome": "completed", "path": str(export)}, raising=False)
    panel._finish_export()
    assert panel._export_pending is None
    assert actions == [(export, {"title": "Scene"}, "project")]
    assert export.exists()


def test_rejected_handoff_cleans_only_the_unaccepted_native_snapshot(gallery, tmp_path, monkeypatch):
    import uuid
    from lfs_plugins.gallery_sync import GallerySync
    panel, _, _ = gallery
    module = import_module("lfs_plugins.gallery_panel")
    export = tmp_path / (str(uuid.uuid4()) + ".scene")
    export.mkdir()
    (export / "manifest.json").write_text("{}")
    (export / "0.ply").write_bytes(b"snapshot")
    kept = tmp_path / "user.ply"
    kept.write_bytes(b"keep")
    panel.service.root = tmp_path
    panel.service._unlink_temporary = GallerySync._unlink_temporary
    panel._export_pending = (export, {"title": "Scene"}, "project", 0)
    panel.service.queue_prepared_upload = lambda *args: (_ for _ in ()).throw(ValueError("Portal changed; refresh first."))
    monkeypatch.setattr(module.lf.ui, "get_export_state", lambda: {"active": False, "outcome": "completed", "path": str(export)}, raising=False)
    panel._finish_export()
    assert panel._export_pending is None and not export.exists()
    assert kept.read_bytes() == b"keep"
    assert "Portal changed" in panel._message


def test_gallery_does_not_cancel_or_consume_another_export(gallery, tmp_path, monkeypatch):
    panel, _, actions = gallery
    module = import_module("lfs_plugins.gallery_panel")
    export = tmp_path / "own.ply"
    export.write_bytes(b"unconsumed previous preparation")
    panel._export_pending = (export, {}, "project", 0)
    monkeypatch.setattr(module.lf.ui, "get_export_state", lambda: {"active": True, "path": str(tmp_path / "other.ply")}, raising=False)
    panel._action_pause()
    assert actions == []
    panel._finish_export()
    assert panel._export_pending is None
    assert not export.exists()
    assert "prepare your upload again" in panel._message


def test_refresh_updates_clean_form_but_keeps_dirty_form_revision(gallery):
    panel, state, _ = gallery
    state["scenes"] = [scene()]
    panel._refresh_model()
    panel._action_select("private-one")
    state["scenes"] = [dict(scene(), title="New remote title", revision="second")]
    panel._refresh_model()
    assert panel._title == "New remote title"
    assert panel._scene["revision"] == "second"
    panel._title = "My unsaved change"
    state["scenes"] = [dict(scene(), title="Concurrent change", revision="third")]
    panel._refresh_model()
    assert panel._title == "My unsaved change"
    assert panel._scene["revision"] == "second"


def test_making_scene_public_requires_review_and_keeps_captured_details(gallery):
    panel, state, actions = gallery
    state["scenes"] = [scene()]
    panel._refresh_model()
    panel._action_select("private-one")
    panel._visibility = "public"
    panel._action_edit()
    assert actions == []
    assert "public" in panel._confirm[0]
    panel._title = "Another title entered after confirmation"
    panel._action_confirm_action()
    assert actions[0][2] == dict(title="My scene", description="Private description", visibility="public")


def test_unlink_confirmation_and_cancel_never_save_the_project(gallery, monkeypatch):
    panel, _, actions = gallery
    module = import_module("lfs_plugins.gallery_panel")
    monkeypatch.setattr(panel, "_project_identity", lambda: ("project", "saved.licht"))
    monkeypatch.setattr(module.lf, "project_save", lambda **kw: actions.append("saved"), raising=False)
    panel.service.unlink = lambda project: actions.append(("unlinked", project))
    panel._action_unlink()
    assert panel._confirm[2] == "Unlink project"
    panel.service.busy = True
    panel._dispatch("confirm_action", [])
    assert panel._confirm and actions == []
    panel._dispatch("cancel_action", [])
    assert panel._confirm is None
    assert actions == []
    panel.service.busy = False
    panel._action_unlink()
    panel._action_confirm_action()
    assert actions == [("unlinked", "project")]


def test_wrong_replacement_is_rejected_before_saving_or_exporting(gallery, monkeypatch):
    panel, state, actions = gallery
    state["links"] = {"project": {"sceneId": "original"}}
    monkeypatch.setattr(panel, "_project_identity", lambda: ("project", "saved.licht"))
    monkeypatch.setattr(panel, "_save_current_project", lambda proceed: actions.append("saved"))
    panel._refresh_model()
    with pytest.raises(ValueError, match="linked to a gallery item"):
        panel._publish({"replaceSceneId": "other"})
    assert actions == []


def test_refused_project_reset_never_appends_download_to_current_scene(gallery, monkeypatch):
    panel, _, actions = gallery
    module = import_module("lfs_plugins.gallery_panel")
    monkeypatch.setattr(module.lf, "is_training_active", lambda: False, raising=False)
    monkeypatch.setattr(module.lf, "project_is_dirty", lambda: False, raising=False)
    monkeypatch.setattr(module.lf, "project_has_path", lambda: True)
    monkeypatch.setattr(panel, "_save_current_project", lambda proceed: proceed())
    monkeypatch.setattr(module.lf, "get_scene", lambda: SimpleNamespace(get_nodes=lambda: [object()]), raising=False)
    # The native command may return without clearing while autosave is busy.
    monkeypatch.setattr(module.lf, "new_project", lambda: actions.append("new requested"), raising=False)
    monkeypatch.setattr(module.lf, "load_file", lambda path: actions.append(("loaded", path)), raising=False)
    with pytest.raises(ValueError, match="could not be closed"):
        panel._import_download({"id": "download", "path": "download.ply"})
    assert actions == ["new requested"]
    assert panel._import_pending is None


def test_account_switch_tracks_import_but_never_registers_it_for_new_account(gallery, monkeypatch):
    panel, state, actions = gallery
    module = import_module("lfs_plugins.gallery_panel")
    job = {"path": "previous-account.ply", "result": {}}
    panel._import_pending = job
    state["identity"] = ("https://portal.example", "two@example.com", "second", True)
    monkeypatch.setattr(module.lf.ui, "get_import_state", lambda: {"active": True}, raising=False)
    panel._refresh_model()
    panel._finish_import()
    assert panel._import_pending is job and panel._import_detached
    assert actions == []
    monkeypatch.setattr(module.lf.ui, "get_import_state", lambda: {"active": False})
    monkeypatch.setattr(module.lf, "get_scene", lambda: SimpleNamespace(get_node=lambda name: object()), raising=False)
    monkeypatch.setattr(module, "restore_view", lambda *args: actions.append("restored"))
    panel.service.link_download = lambda *args: actions.append("linked")
    panel._finish_import()
    assert panel._import_pending is None
    assert actions == ["dismiss-import"]
    assert "has not been linked" in panel._message


@pytest.mark.parametrize("outcome", ["ready", "failed", "different-operation", "account-changed"])
def test_import_only_reports_linked_after_its_own_link_is_persisted(gallery, outcome):
    panel, state, _ = gallery
    job = {"id": "download", "_accountIdentity": state["identity"], "_link": "link-operation"}
    panel._import_pending = job
    panel._message = "Downloaded scene saved. Saving its gallery link…"
    panel.service.busy = True
    panel._finish_import()
    assert panel._import_pending is job
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
    assert panel._import_pending is None
    assert ("saved and linked" in panel._message) == (outcome == "ready")
    assert "saved" in panel._message


def test_asset_manager_source_blocks_publish_until_that_project_opens(gallery, monkeypatch, tmp_path):
    panel, state, _ = gallery
    module = import_module("lfs_plugins.gallery_panel")
    requested = tmp_path / "selected.licht"
    current = {"path": str(tmp_path / "different.licht")}
    monkeypatch.setattr(module.lf, "project_poll_write", lambda: current, raising=False)
    monkeypatch.setattr(panel, "_project_identity", lambda: ("selected-project", current["path"]))
    panel.focus_project(requested)
    with pytest.raises(ValueError, match="Open the selected Asset Manager project"):
        panel._action_publish()
    state["links"] = {"selected-project": {"sceneId": "private-one", "metadata": scene()}}
    state["scenes"] = [scene()]
    current["path"] = str(requested)
    panel._refresh_model()
    assert panel._requested_project is None
    assert panel._scene["id"] == "private-one"
    assert panel._title == "My scene"


def test_use_current_project_clears_asset_manager_target(gallery, monkeypatch, tmp_path):
    panel, _, _ = gallery
    module = import_module("lfs_plugins.gallery_panel")
    monkeypatch.setattr(module.lf, "project_poll_write", lambda: {"path": str(tmp_path / "current.licht")}, raising=False)
    panel.focus_project(tmp_path / "different.licht")
    panel._action_use_current()
    assert panel._requested_project is None
    assert panel._title == "current"


def test_account_change_clears_pending_asset_manager_target(gallery, monkeypatch, tmp_path):
    panel, state, _ = gallery
    module = import_module("lfs_plugins.gallery_panel")
    monkeypatch.setattr(module.lf, "project_poll_write", lambda: {"path": ""}, raising=False)
    panel.focus_project(tmp_path / "different.licht")
    state["identity"] = ("https://portal.example", "two@example.com", "second", True)
    panel._refresh_model()
    assert panel._requested_project is None
    assert not panel._focus_project


def test_delayed_project_open_cannot_survive_an_account_switch(gallery, monkeypatch, tmp_path):
    panel, state, _ = gallery
    module = import_module("lfs_plugins.gallery_panel")
    prompts = []
    confirm = import_module("lfs_plugins.training_confirm")
    monkeypatch.setattr(confirm, "confirm_discard_work_then", lambda _, callback: prompts.append(callback))
    monkeypatch.setattr(module.lf, "project_poll_write", lambda: {"path": ""}, raising=False)
    panel.focus_project(tmp_path / "selected.licht")
    panel._action_open_project()
    state["identity"] = ("https://portal.example", "two@example.com", "second", True)
    prompts[0](False)
    assert module.lf._test_state.opened == []
    assert "account or selected project changed" in panel._message


def test_publish_confirmation_cannot_upload_a_different_project(gallery, monkeypatch):
    panel, _, _ = gallery
    module = import_module("lfs_plugins.gallery_panel")
    current = ["first-id", "/first.licht"]
    monkeypatch.setattr(panel, "_project_identity", lambda: tuple(current))
    monkeypatch.setattr(module, "capture_view", lambda _: {})
    panel._title, panel._visibility = "Public scene", "public"
    panel._action_publish()
    assert panel._confirm
    current[:] = ["second-id", "/second.licht"]
    with pytest.raises(ValueError, match="current project changed"):
        panel._action_confirm_action()


def test_account_switch_during_project_save_prevents_export(gallery, monkeypatch):
    panel, state, _ = gallery
    module = import_module("lfs_plugins.gallery_panel")
    monkeypatch.setattr(panel, "_project_identity", lambda: ("project", "/project.licht"))
    monkeypatch.setattr(module.lf, "get_scene", lambda: SimpleNamespace(get_nodes=lambda: [SimpleNamespace(id=1, name="scene", type=module.lf.scene.NodeType.SPLAT)], is_node_effectively_visible=lambda node_id: True), raising=False)
    def save(proceed):
        state["identity"] = ("https://portal.example", "two@example.com", "second", True)
        proceed()
    monkeypatch.setattr(panel, "_save_current_project", save)
    with pytest.raises(ValueError, match="account or current project changed while saving"):
        panel._publish({"title": "Private scene"})
    assert panel._export_pending is None


def test_account_switch_during_save_prevents_opening_download(gallery, monkeypatch):
    panel, state, _ = gallery
    module = import_module("lfs_plugins.gallery_panel")
    monkeypatch.setattr(module.lf, "project_has_path", lambda: True)
    def save(proceed):
        state.update(identity=("https://portal.example", "two@example.com", "second", True))
        proceed()
    monkeypatch.setattr(panel, "_save_current_project", save)
    with pytest.raises(ValueError, match="account changed while saving"):
        panel._import_download({"path": "/private.ply"})
    assert panel._import_pending is None


@pytest.mark.parametrize("account_changed", [False, True])
def test_local_update_keeps_geometry_if_user_edits_or_changes_account(gallery, monkeypatch, account_changed):
    panel, state, actions = gallery
    module = import_module("lfs_plugins.gallery_panel")
    project = ("project", "/project.licht")
    job = {"id": "download", "_accountIdentity": state["identity"],
        "_update": {"project": project, "phase": "backup", "backup_id": "backup"}}
    state["jobs"] = [{"id": "download", "localUpdate": {"id": "backup", "state": "ready"}}]
    panel._import_pending = job
    monkeypatch.setattr(panel, "_project_identity", lambda: project)
    monkeypatch.setattr(module.lf, "get_scene", lambda: SimpleNamespace(remove_node=lambda *a, **k: actions.append("removed")), raising=False)
    monkeypatch.setattr(module.lf, "project_is_dirty", lambda: True)
    if account_changed:
        state["identity"] = ("https://portal.example", "two@example.com", "second", True)
        panel._finish_import()
        assert panel._import_pending is None
        assert "existing local splats remain" in panel._message
    else:
        with pytest.raises(ValueError, match="local project changed during preparation"):
            panel._finish_import()
    assert actions == []


def test_partial_local_update_reopens_the_unchanged_saved_project(gallery, monkeypatch):
    panel, state, _ = gallery
    module = import_module("lfs_plugins.gallery_panel")
    project = ("project", "/project.licht")
    update = {"project": project, "phase": "backup", "backup_id": "backup",
        "generation": 3, "stamp": [1, 2], "incoming": "incoming"}
    job = {"id": "download", "_update": update}
    state["jobs"] = [{"id": "download", "localUpdate": {"id": "backup", "state": "ready"}}]
    monkeypatch.setattr(panel, "_project_identity", lambda: project)
    monkeypatch.setattr(module.lf, "get_scene", lambda: SimpleNamespace(get_node_by_uuid=lambda _: object()), raising=False)
    monkeypatch.setattr(module.lf, "project_is_dirty", lambda: False)
    monkeypatch.setattr(module.lf, "project_poll_write", lambda: {"generation": 3}, raising=False)
    monkeypatch.setattr(module, "file_stamp", lambda _: [1, 2])
    monkeypatch.setattr(panel, "_apply_local_update", lambda *args: (_ for _ in ()).throw(OSError("save failed")))
    with pytest.raises(ValueError, match="saved local project is being reopened"):
        panel._finish_local_update(job)
    assert module.lf._test_state.opened == [("/project.licht", True, False, False)]


def test_transfer_popup_tracks_processing_pause_completion_and_account_boundary(gallery):
    panel, state, _ = gallery
    from lfs_plugins.gallery_transfer_panel import transfer_state
    job = dict(id='transfer', kind='upload', metadata={'title':'Private garden'},
               status='running', serverProcessing=True, completed=40, total=100, message='Checking scene')
    state['jobs'] = [job]
    panel._state = state
    progress = transfer_state(panel)
    assert progress['progress'] == 40 and progress['can_pause']
    assert progress['message'] == 'Upload received · Checking scene'
    job.update(status='paused', message='Stopped waiting')
    # The gallery may be hidden with its last rendered running snapshot.
    panel._state = dict(state, jobs=[dict(job, status='running')])
    progress = transfer_state(panel)
    assert progress['can_resume'] and not progress['can_pause']
    job.update(status='completed', serverProcessing=False)
    assert transfer_state(panel)['progress'] == 100
    job.update(retired=True, total=0, completed=0, message='Transfer cleared. Recovery copy kept.')
    progress = transfer_state(panel)
    assert progress['message'] == job['message'] and not progress['detail']
    assert not progress['can_resume'] and not progress['can_pause']
    state['identity'] = ('another', 'account')
    progress = transfer_state(panel)
    assert 'Private garden' not in str(progress)
    assert not progress['can_resume'] and not progress['can_pause']


def test_background_metadata_confirmation_does_not_open_transfer_popup(gallery, monkeypatch):
    panel, state, _ = gallery
    module = import_module('lfs_plugins.gallery_panel')
    shown = []
    monkeypatch.setattr(module.lf.ui, 'set_panel_enabled', lambda *args: shown.append(args), raising=False)
    panel._confirm = ('Clear history?', lambda: setattr(panel.service, 'busy', True), 'Clear')
    panel._dispatch('confirm_action', [])
    assert not shown
    panel.service.busy = False
    panel._maybe_show_transfer_progress()
    assert not panel._progress_pending


def test_transfer_popup_opens_after_worker_leaves_queued_state(gallery, monkeypatch):
    panel, state, _ = gallery
    module = import_module('lfs_plugins.gallery_panel')
    shown = []
    monkeypatch.setattr(module.lf.ui, 'set_panel_enabled', lambda *args: shown.append(args), raising=False)
    job = dict(id='download', kind='download', metadata={'title':'Garden'}, status='queued',
               completed=0, total=100, message='Ready to download')

    def begin():
        state['jobs'] = [job]
        panel.service.busy = True

    monkeypatch.setattr(panel, '_action_download', begin)
    panel._dispatch('download', [])
    assert not shown and panel._progress_pending
    job['status'] = 'running'
    panel._refresh_model()
    panel._maybe_show_transfer_progress()
    assert shown == [('lfs.gallery', False), ('lfs.gallery_transfer', True)]
    panel._maybe_show_transfer_progress()
    assert len(shown) == 2


@pytest.mark.parametrize('format_name', ['studio', 'sog', 'ssog', 'spz'])
def test_upload_format_is_fixed_at_confirmation_and_all_payload_formats_keep_hdr(gallery, monkeypatch, format_name):
    panel, _, calls = gallery
    module = import_module('lfs_plugins.gallery_panel')
    monkeypatch.setattr(panel, '_project_identity', lambda: ('project', '/project.licht'))
    monkeypatch.setattr(module, 'capture_view', lambda _: {'environment':{'exposure':0,'rotation':0}, 'renderMode':'3dgut'})
    monkeypatch.setattr(module.lf, 'get_render_settings', lambda: SimpleNamespace(environment_map_path='/background.hdr'), raising=False)
    monkeypatch.setattr(panel, '_publish', lambda metadata, **kwargs: calls.append((metadata,kwargs)))
    panel._title, panel._visibility, panel._upload_format = 'Garden', 'public', format_name
    panel._action_publish()
    panel._upload_format = 'changed-after-review'
    panel._action_confirm_action()
    metadata, options = calls[-1]
    assert options['upload_format'] == format_name
    assert 'environment' in metadata['viewerSettings']
    assert metadata['viewerSettings']['renderMode'] == '3dgut'


CAMERA_TRACK = {
    "version": 1,
    "keyframes": [{"t": 0, "eye": [0, 1, 3]}],
    "duration": 2.5,
    "loopMode": "once",
    "playbackSpeed": 1.0,
}
REMOTE_PRECISE_TRACK = {
    "duration": 4.5,
    "keyframes": [{
        "easing": 0, "focal_length_mm": 25.73408317565918,
        "position": [-2.0, 2.0, -6.0],
        "rotation": [-0.15830765664577484, 0.02443433739244938, 0.9755357503890991, 0.15057116746902466],
        "time": 0.0,
    }, {
        "easing": 3, "focal_length_mm": 26.33159828186035,
        "position": [-1.1, 2.1, -5.6],
        "rotation": [-0.1110728457570076, 0.06121033430099487, 0.9799413681030273, 0.15372401475906372],
        "time": 2.0,
    }],
    "loopMode": "ping_pong",
    "playbackSpeed": 1.25,
    "version": 1,
}


def _f32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


def _native_camera_path(path):
    if path is None:
        return None

    def convert(value, key=None):
        if isinstance(value, bool) or value is None:
            return value
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return _f32(value)
        if isinstance(value, list):
            items = [convert(item) for item in value]
            if key == "rotation" and len(items) == 4 and all(isinstance(item, float) for item in items):
                length = math.sqrt(sum(item * item for item in items)) or 1.0
                return [_f32(item / length) for item in items]
            return items
        if isinstance(value, dict):
            return {name: convert(item, name) for name, item in value.items()}
        return value

    return convert(copy.deepcopy(path))
SENDING_TRACK = "Sending the gallery camera track…"
TRACK_SENT = "Gallery camera track sent."
GETTING_TRACK = "Getting the gallery camera track…"
TRACK_RECEIVED = "Gallery camera track received."
TRACK_APPLIED = "Gallery camera track applied."
TRACK_CONFLICT = "The gallery item changed. Refresh and review both versions before replacing it."


def _link_selected_scene(panel, state, monkeypatch, *, scene_id="private-one"):
    state["scenes"] = [scene()]
    state["links"] = {"project": {"sceneId": scene_id, "revision": "original", "metadata": scene()}}
    monkeypatch.setattr(panel, "_project_identity", lambda: ("project", "/project.licht"))
    panel._refresh_model()
    panel._action_select(scene_id)


def _bound_gallery_message(panel):
    from test_asset_manager_panel import _BindingContext, _BindingModel
    panel._refresh_model()
    model = _BindingModel()
    panel.on_bind_model(_BindingContext(model))
    return model.func_bindings["message"]()


def _install_camera_track_send(panel, state, actions):
    def start(*args):
        actions.append(("send_camera_track", *args))
        state["message"] = SENDING_TRACK
        state["busy"] = True
        panel.service.busy = True

    def complete(message):
        state["message"] = message
        state["busy"] = False
        panel.service.busy = False

    panel.service.send_camera_track = start
    return complete


def _install_camera_track_fetch(panel, state, actions, *, operation="fetch-op"):
    def start(scene_id):
        actions.append(("fetch_camera_track", scene_id))
        state["message"] = GETTING_TRACK
        state["busy"] = True
        panel.service.busy = True
        state["trackFetch"] = {"id": operation, "state": "running", "sceneId": scene_id}
        return operation

    def complete(path, *, message=TRACK_RECEIVED, revision="remote-track"):
        state["trackFetch"] = {"id": operation, "state": "ready", "sceneId": "private-one",
            "revision": revision, "cameraPath": path}
        state["message"] = message
        state["busy"] = False
        panel.service.busy = False

    def fail(message):
        state["trackFetch"] = {"id": operation, "state": "failed", "sceneId": "private-one"}
        state["message"] = message
        state["busy"] = False
        panel.service.busy = False

    panel.service.fetch_camera_track = start
    return complete, fail


def test_send_camera_track_captures_current_path_and_does_not_export(gallery, monkeypatch):
    panel, state, actions = gallery
    module = import_module("lfs_plugins.gallery_panel")
    _link_selected_scene(panel, state, monkeypatch)
    complete = _install_camera_track_send(panel, state, actions)
    live = dict(CAMERA_TRACK)
    monkeypatch.setattr(module.lf.ui, "get_camera_path", lambda: live, raising=False)
    monkeypatch.setattr(module.lf, "prepare_gallery_scene", lambda *a, **k: pytest.fail("camera track send must not prepare an export"), raising=False)
    monkeypatch.setattr(panel.service, "queue_prepared_upload", lambda *a, **k: pytest.fail("camera track send must not upload geometry"), raising=False)
    monkeypatch.setattr(panel.service, "edit", lambda *a, **k: pytest.fail("camera track send must not save details"))
    monkeypatch.setattr(panel, "_save_current_project", lambda proceed: proceed())
    panel._title = "Unsaved title change"
    panel._description = "Unsaved description"
    panel._action_send_camera_track()
    assert panel._confirm and panel._confirm[2] == "Send camera track"
    assert "LichtFeld Studio" in panel._confirm[0]
    panel._action_confirm_action()
    assert actions == [("send_camera_track", "private-one", "original", CAMERA_TRACK)]
    assert panel._export_pending is None
    assert panel._message == ""
    assert _bound_gallery_message(panel) == SENDING_TRACK
    complete(TRACK_SENT)
    assert _bound_gallery_message(panel) == TRACK_SENT


def test_send_camera_track_uses_snapshot_revision_when_form_is_dirty(gallery, monkeypatch):
    panel, state, actions = gallery
    module = import_module("lfs_plugins.gallery_panel")
    _link_selected_scene(panel, state, monkeypatch)
    monkeypatch.setattr(module.lf.ui, "get_camera_path", lambda: dict(CAMERA_TRACK), raising=False)
    monkeypatch.setattr(panel, "_save_current_project", lambda proceed: proceed())
    panel._title = "Unsaved title change"
    panel._action_send_camera_track()
    panel._action_confirm_action()
    assert actions == [("send_camera_track", "private-one", "original", CAMERA_TRACK)]
    state["scenes"] = [dict(scene(), revision="after-first-send")]
    state["links"]["project"] = {**state["links"]["project"], "revision": "after-first-send"}
    panel._refresh_model()
    assert panel._title == "Unsaved title change"
    assert panel._scene["revision"] == "original"
    panel._action_send_camera_track()
    panel._action_confirm_action()
    assert actions[-1] == ("send_camera_track", "private-one", "after-first-send", CAMERA_TRACK)
    panel._action_edit()
    assert actions[-1][0] == "private-one"
    assert actions[-1][1] == "original"
    assert actions[-1][2]["title"] == "Unsaved title change"


def test_send_camera_track_null_clears_remote_playback_after_review(gallery, monkeypatch):
    panel, state, actions = gallery
    module = import_module("lfs_plugins.gallery_panel")
    _link_selected_scene(panel, state, monkeypatch)
    monkeypatch.setattr(module.lf.ui, "get_camera_path", lambda: None, raising=False)
    monkeypatch.setattr(panel, "_save_current_project", lambda proceed: proceed())
    panel._action_send_camera_track()
    assert panel._confirm[2] == "Clear camera track"
    assert "playback will stop" in panel._confirm[0]
    panel._action_confirm_action()
    assert actions == [("send_camera_track", "private-one", "original", None)]


@pytest.mark.parametrize("change", ["track", "project", "account"])
def test_send_camera_track_blocks_stale_path_after_async_save(gallery, monkeypatch, change):
    panel, state, actions = gallery
    module = import_module("lfs_plugins.gallery_panel")
    _link_selected_scene(panel, state, monkeypatch)
    current = {"path": dict(CAMERA_TRACK)}
    monkeypatch.setattr(module.lf.ui, "get_camera_path", lambda: current["path"], raising=False)
    project = ["project", "/project.licht"]
    monkeypatch.setattr(panel, "_project_identity", lambda: tuple(project))

    def save(proceed):
        if change == "track":
            current["path"] = dict(CAMERA_TRACK, duration=9)
        elif change == "project":
            project[0] = "other-project"
            state["links"] = {"other-project": {"sceneId": "private-one", "revision": "original"}}
        else:
            state["identity"] = ("https://portal.example", "two@example.com", "second", True)
        proceed()

    monkeypatch.setattr(panel, "_save_current_project", save)
    panel._action_send_camera_track()
    with pytest.raises(ValueError, match="changed"):
        panel._action_confirm_action()
    assert actions == []
    assert panel._export_pending is None


def test_send_camera_track_does_not_send_when_save_never_finishes(gallery, monkeypatch):
    panel, state, actions = gallery
    module = import_module("lfs_plugins.gallery_panel")
    _link_selected_scene(panel, state, monkeypatch)
    monkeypatch.setattr(module.lf.ui, "get_camera_path", lambda: dict(CAMERA_TRACK), raising=False)
    monkeypatch.setattr(panel, "_save_current_project", lambda proceed: None)
    panel._action_send_camera_track()
    panel._action_confirm_action()
    assert actions == []


def test_edit_still_sends_details_without_camera_path_or_export(gallery, monkeypatch):
    panel, state, actions = gallery
    module = import_module("lfs_plugins.gallery_panel")
    state["scenes"] = [scene()]
    panel._refresh_model()
    panel._action_select("private-one")
    monkeypatch.setattr(module, "capture_view", lambda _: pytest.fail("details save must not capture the view"))
    monkeypatch.setattr(module, "capture_camera_path", lambda _: pytest.fail("details save must not capture the camera track"))
    monkeypatch.setattr(module.lf, "prepare_gallery_scene", lambda *a, **k: pytest.fail("details save must not export"), raising=False)
    panel._action_edit()
    assert actions == [("private-one", "original",
        {"title": "My scene", "description": "Private description", "visibility": "private"})]


def test_unlinked_or_mismatched_scene_cannot_sync_camera_track(gallery, monkeypatch):
    panel, state, _ = gallery
    module = import_module("lfs_plugins.gallery_panel")
    state["scenes"] = [scene(), dict(scene(), id="other-scene", title="Other")]
    monkeypatch.setattr(panel, "_project_identity", lambda: ("project", "/project.licht"))
    monkeypatch.setattr(module.lf.ui, "get_camera_path", lambda: dict(CAMERA_TRACK), raising=False)
    panel._refresh_model()
    panel._action_select("private-one")
    assert not panel._can_sync_camera_track()
    assert "not linked" in panel._camera_track_help()
    with pytest.raises(ValueError, match="not linked"):
        panel._action_send_camera_track()
    with pytest.raises(ValueError, match="not linked"):
        panel._action_get_camera_track()
    state["links"] = {"project": {"sceneId": "other-scene", "revision": "original"}}
    panel._refresh_model()
    assert not panel._can_sync_camera_track()
    assert "linked to this LichtFeld Studio project" in panel._camera_track_help()
    with pytest.raises(ValueError, match="linked"):
        panel._action_send_camera_track()
    with pytest.raises(ValueError, match="linked"):
        panel._action_get_camera_track()
    panel._action_select("other-scene")
    assert panel._can_sync_camera_track()
    assert panel._camera_track_help() == "Send or get playback cameras for the current LichtFeld Studio project."
    panel._requested_project = "/other.licht"
    assert not panel._can_sync_camera_track()
    with pytest.raises(ValueError, match="Open the selected Asset Manager project"):
        panel._action_send_camera_track()


def test_send_camera_track_confirmation_does_not_open_transfer_popup(gallery, monkeypatch):
    panel, state, _ = gallery
    module = import_module("lfs_plugins.gallery_panel")
    shown = []
    _link_selected_scene(panel, state, monkeypatch)
    monkeypatch.setattr(module.lf.ui, "get_camera_path", lambda: dict(CAMERA_TRACK), raising=False)
    monkeypatch.setattr(module.lf.ui, "set_panel_enabled", lambda *args: shown.append(args), raising=False)
    monkeypatch.setattr(panel, "_save_current_project", lambda proceed: setattr(panel, "_save_pending", {"continuation": proceed}) or True)
    panel._dispatch("send_camera_track", [])
    panel._dispatch("confirm_action", [])
    assert panel._save_pending and panel._can_pause()
    assert not shown
    assert not panel._progress_pending


@pytest.mark.parametrize("outcome,expected", [
    ("success", TRACK_SENT),
    ("conflict", TRACK_CONFLICT),
])
def test_send_camera_track_panel_message_follows_service_success_and_conflict(gallery, monkeypatch, outcome, expected):
    panel, state, actions = gallery
    module = import_module("lfs_plugins.gallery_panel")
    _link_selected_scene(panel, state, monkeypatch)
    complete = _install_camera_track_send(panel, state, actions)
    monkeypatch.setattr(module.lf.ui, "get_camera_path", lambda: dict(CAMERA_TRACK), raising=False)
    monkeypatch.setattr(panel, "_save_current_project", lambda proceed: proceed())
    panel._dispatch("send_camera_track", [])
    panel._dispatch("confirm_action", [])
    assert panel._message == ""
    assert _bound_gallery_message(panel) == SENDING_TRACK
    complete(expected)
    assert panel._message == ""
    assert _bound_gallery_message(panel) == expected


def test_send_camera_track_save_error_is_not_replaced_by_updating(gallery, monkeypatch):
    panel, state, actions = gallery
    module = import_module("lfs_plugins.gallery_panel")
    _link_selected_scene(panel, state, monkeypatch)
    _install_camera_track_send(panel, state, actions)
    monkeypatch.setattr(module.lf.ui, "get_camera_path", lambda: dict(CAMERA_TRACK), raising=False)

    def save(_proceed):
        raise ValueError("The project could not be saved. Your gallery operation was stopped; resolve the save error before retrying.")

    monkeypatch.setattr(panel, "_save_current_project", save)
    panel._dispatch("send_camera_track", [])
    panel._dispatch("confirm_action", [])
    assert actions == []
    status = _bound_gallery_message(panel)
    assert "could not be saved" in status
    assert SENDING_TRACK not in status


def _local_track_state(monkeypatch, module, initial=None):
    current = {"path": None if initial is None else _native_camera_path(initial)}

    def set_path(path):
        if not isinstance(path, dict):
            return False
        current["path"] = _native_camera_path(path)
        return True

    monkeypatch.setattr(module.lf.ui, "get_camera_path", lambda: None if current["path"] is None else copy.deepcopy(current["path"]), raising=False)
    monkeypatch.setattr(module.lf.ui, "set_camera_path", set_path, raising=False)
    monkeypatch.setattr(module.lf.ui, "clear_keyframes", lambda: current.update(path=None), raising=False)
    return current


def test_get_camera_track_applies_remote_poses_without_geometry_transfer(gallery, monkeypatch):
    panel, state, actions = gallery
    module = import_module("lfs_plugins.gallery_panel")
    _link_selected_scene(panel, state, monkeypatch)
    current = _local_track_state(monkeypatch, module, {"version": 1, "duration": 1})
    complete, _ = _install_camera_track_fetch(panel, state, actions)
    monkeypatch.setattr(module.lf, "prepare_gallery_scene", lambda *a, **k: pytest.fail("get camera track must not export"), raising=False)
    monkeypatch.setattr(module.lf, "load_file", lambda *a, **k: pytest.fail("get camera track must not load geometry"), raising=False)
    monkeypatch.setattr(module.lf, "load_gallery_scene", lambda *a, **k: pytest.fail("get camera track must not import geometry"), raising=False)
    monkeypatch.setattr(panel.service, "download", lambda *a, **k: pytest.fail("get camera track must not download geometry"), raising=False)
    monkeypatch.setattr(panel.service, "queue_prepared_upload", lambda *a, **k: pytest.fail("get camera track must not upload"), raising=False)
    remote = copy.deepcopy(REMOTE_PRECISE_TRACK)
    monkeypatch.setattr(panel, "_schedule_phase_poll", lambda: None)
    monkeypatch.setattr(panel, "_save_current_project", lambda proceed: proceed())
    panel._dispatch("get_camera_track", [])
    assert panel._confirm[2] == "Get camera track"
    assert "empty gallery track clears" in panel._confirm[0]
    panel._dispatch("confirm_action", [])
    assert actions == [("fetch_camera_track", "private-one")]
    assert panel._track_pull_pending and panel._can_pause()
    assert _bound_gallery_message(panel) == GETTING_TRACK
    complete(remote, revision="web-editor")
    panel._advance_phases()
    assert current["path"] == _native_camera_path(remote)
    assert current["path"] != remote
    assert _bound_gallery_message(panel) == TRACK_APPLIED
    assert state["links"]["project"]["revision"] == "original"
    assert panel._export_pending is None
    assert panel._track_pull_pending is None


def test_get_camera_track_null_clears_local_playback(gallery, monkeypatch):
    panel, state, actions = gallery
    module = import_module("lfs_plugins.gallery_panel")
    _link_selected_scene(panel, state, monkeypatch)
    current = _local_track_state(monkeypatch, module, CAMERA_TRACK)
    complete, _ = _install_camera_track_fetch(panel, state, actions)
    monkeypatch.setattr(panel, "_schedule_phase_poll", lambda: None)
    monkeypatch.setattr(panel, "_save_current_project", lambda proceed: proceed())
    panel._action_get_camera_track()
    panel._action_confirm_action()
    complete(None)
    panel._advance_phases()
    assert current["path"] is None
    assert actions == [("fetch_camera_track", "private-one")]
    assert _bound_gallery_message(panel) == TRACK_APPLIED


@pytest.mark.parametrize("change", ["track", "project", "account"])
def test_get_camera_track_blocks_stale_apply_after_fetch(gallery, monkeypatch, change):
    panel, state, actions = gallery
    module = import_module("lfs_plugins.gallery_panel")
    _link_selected_scene(panel, state, monkeypatch)
    current = _local_track_state(monkeypatch, module, CAMERA_TRACK)
    complete, _ = _install_camera_track_fetch(panel, state, actions)
    project = ["project", "/project.licht"]
    monkeypatch.setattr(panel, "_project_identity", lambda: tuple(project))
    monkeypatch.setattr(panel, "_schedule_phase_poll", lambda: None)
    monkeypatch.setattr(panel, "_save_current_project", lambda proceed: proceed())
    panel._action_get_camera_track()
    panel._action_confirm_action()
    if change == "track":
        current["path"] = dict(CAMERA_TRACK, duration=9)
    elif change == "project":
        project[0] = "other-project"
        state["links"] = {"other-project": {"sceneId": "private-one", "revision": "original"}}
    else:
        state["identity"] = ("https://portal.example", "two@example.com", "second", True)
    complete(dict(CAMERA_TRACK, duration=8))
    panel._advance_phases()
    assert panel._track_pull_pending is None
    if change == "track":
        assert current["path"] == dict(CAMERA_TRACK, duration=9)
    else:
        assert current["path"] == CAMERA_TRACK
    assert "changed" in _bound_gallery_message(panel).lower() or change == "account"


def test_get_camera_track_failed_fetch_does_not_apply_or_advance_link(gallery, monkeypatch):
    panel, state, actions = gallery
    module = import_module("lfs_plugins.gallery_panel")
    _link_selected_scene(panel, state, monkeypatch)
    current = _local_track_state(monkeypatch, module, CAMERA_TRACK)
    _, fail = _install_camera_track_fetch(panel, state, actions)
    monkeypatch.setattr(panel, "_schedule_phase_poll", lambda: None)
    monkeypatch.setattr(panel, "_save_current_project", lambda proceed: proceed())
    panel._action_get_camera_track()
    panel._action_confirm_action()
    fail(TRACK_CONFLICT)
    panel._advance_phases()
    assert current["path"] == CAMERA_TRACK
    assert state["links"]["project"]["revision"] == "original"
    assert panel._track_pull_pending is None
    assert "changed" in _bound_gallery_message(panel).lower()


def test_get_camera_track_save_error_before_fetch_does_not_call_service(gallery, monkeypatch):
    panel, state, actions = gallery
    module = import_module("lfs_plugins.gallery_panel")
    _link_selected_scene(panel, state, monkeypatch)
    _local_track_state(monkeypatch, module, CAMERA_TRACK)
    _install_camera_track_fetch(panel, state, actions)

    def save(_proceed):
        raise ValueError("The project could not be saved. Your gallery operation was stopped; resolve the save error before retrying.")

    monkeypatch.setattr(panel, "_save_current_project", save)
    panel._dispatch("get_camera_track", [])
    panel._dispatch("confirm_action", [])
    assert actions == []
    status = _bound_gallery_message(panel)
    assert "could not be saved" in status
    assert GETTING_TRACK not in status


def test_get_camera_track_save_after_compares_native_snapshot_not_wire_json(gallery, monkeypatch):
    panel, state, actions = gallery
    module = import_module("lfs_plugins.gallery_panel")
    _link_selected_scene(panel, state, monkeypatch)
    current = _local_track_state(monkeypatch, module, CAMERA_TRACK)
    complete, _ = _install_camera_track_fetch(panel, state, actions)
    proceeds = []
    monkeypatch.setattr(panel, "_schedule_phase_poll", lambda: None)
    monkeypatch.setattr(panel, "_save_current_project", lambda proceed: proceeds.append(proceed))
    panel._action_get_camera_track()
    panel._action_confirm_action()
    assert len(proceeds) == 1
    proceeds[0]()
    complete(copy.deepcopy(REMOTE_PRECISE_TRACK))
    panel._advance_phases()
    assert current["path"] == _native_camera_path(REMOTE_PRECISE_TRACK)
    assert current["path"] != REMOTE_PRECISE_TRACK
    assert len(proceeds) == 2
    current["path"] = _native_camera_path(dict(current["path"], duration=99.0))
    with pytest.raises(ValueError, match="changed while saving"):
        proceeds[1]()
    assert current["path"]["duration"] == _f32(99.0)


def test_get_camera_track_post_apply_save_failure_clears_pending(gallery, monkeypatch):
    panel, state, _ = gallery
    module = import_module("lfs_plugins.gallery_panel")
    _link_selected_scene(panel, state, monkeypatch)
    current = _local_track_state(monkeypatch, module, CAMERA_TRACK)
    applied = _native_camera_path(REMOTE_PRECISE_TRACK)
    current["path"] = applied
    pending = {"phase": "save_after", "applied": applied, "project": ("project", "/project.licht"),
        "identity": state["identity"]}
    panel._track_pull_pending = pending
    panel._save_pending = {"project": ("project", "/project.licht"), "identity": state["identity"],
        "generation": 2, "continuation": lambda: panel._camera_track_pull_applied_saved(pending)}
    monkeypatch.setattr(panel, "_project_identity", lambda: ("project", "/project.licht"))
    monkeypatch.setattr(module.lf, "project_poll_write", lambda: {"running": False, "generation": 2, "error": "disk full", "path": "/project.licht"}, raising=False)
    monkeypatch.setattr(module.lf, "project_is_dirty", lambda: False, raising=False)
    panel._advance_phases()
    assert panel._track_pull_pending is None
    assert panel._save_pending is None
    assert "could not be saved" in panel._message
    assert not panel._panel_busy()


def test_pause_clears_fetching_camera_track_pull_so_other_actions_are_not_blocked(gallery):
    panel, _, _ = gallery
    panel._track_pull_pending = {"phase": "fetching", "id": "fetch-op"}
    assert panel._panel_busy() and panel._can_pause()
    panel._action_pause()
    assert panel._track_pull_pending is None
    assert not panel._panel_busy()
    assert "canceled" in panel._message.lower()

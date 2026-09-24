# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Observable states of the gallery's native update procedure."""
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_gallery_controller import gallery
from test_asset_manager_panel import panel_module


@pytest.fixture
def update_case(gallery, monkeypatch, tmp_path):
    panel, state, actions = gallery
    module = import_module("lfs_plugins.gallery_controller")
    project_path = tmp_path / "project.licht"
    project_path.write_bytes(b"saved local project")
    preview_path = tmp_path / "preview.scene"
    preview_path.mkdir()
    backup_path = tmp_path / "recovery.licht"
    backup_path.write_bytes(project_path.read_bytes())
    project = ("project", str(project_path))
    incoming = SimpleNamespace(name="preview", uuid="incoming")
    nodes = {"preview": incoming}
    scene = SimpleNamespace(
        get_node=lambda name: nodes.get(name),
        get_node_by_uuid=lambda identifier: next((n for n in nodes.values() if n.uuid == identifier), None),
        remove_node=lambda name: actions.append("remove preview") or nodes.pop(name),
    )
    monkeypatch.setattr(panel, "_project_identity", lambda: project)
    monkeypatch.setattr(panel, "_refresh_model", lambda: None)
    monkeypatch.setattr(module.lf, "get_scene", lambda: scene, raising=False)
    monkeypatch.setattr(module.lf, "set_node_visibility", lambda *args: actions.append("hide preview"), raising=False)
    monkeypatch.setattr(module.lf.ui, "cancel_gallery_import", lambda: actions.append("cancel import"), raising=False)
    monkeypatch.setattr(module.lf, "project_is_dirty", lambda: False, raising=False)
    poll = {"running": False, "generation": 3, "error": ""}
    monkeypatch.setattr(module.lf, "project_poll_write", lambda: poll.copy(), raising=False)
    update = {"project": project, "phase": "staging", "stage_id": "stage", "path": str(preview_path),
              "incoming": "incoming", "generation": 3, "stamp": module.file_stamp(project_path),
              "backup_id": "backup", "link_operation": "link"}
    job = {"id": "download", "kind": "download", "status": "completed", "result": {"id": "scene", "title": "Gallery"},
           "_accountIdentity": state["identity"], "_update": update}
    journal = {"id": "download", "stagedImport": {"id": "stage", "state": "ready", "path": str(preview_path)},
               "localUpdate": {"id": "backup", "state": "ready", "backupPath": str(backup_path)},
               "linkOperation": {"id": "link", "state": "ready"}}
    state["jobs"] = [journal]
    panel.service.fail_local_update = lambda job_id, reason: journal["localUpdate"].update(state="failed", message=reason)
    panel._local_update_steps.pending = job
    return panel, state, actions, nodes, update, poll, project_path, preview_path, backup_path, journal


@pytest.mark.parametrize("phase,keeps_preview,keeps_pending,message", [
    ("staging", True, False, "Update canceled. Your existing local splats remain."),
    ("importing", False, False, "Update canceled. Your existing local splats remain."),
    ("save_before_backup", False, False, "Update canceled. Your existing local splats remain."),
    ("backup", False, False, "Update canceled. Your existing local splats remain."),
    ("save_updated", True, False, "The project was updated and its recovery copy was kept. Account changed; it has not been linked to this account."),
    ("linking", True, False, "The project was updated; its recovery copy was kept. Refresh the gallery to check its link."),
])
def test_update_cancel_characterization(update_case, phase, keeps_preview, keeps_pending, message):
    panel, state, actions, nodes, update, poll, project_path, preview_path, backup_path, journal = update_case
    update["phase"] = phase
    panel._action_pause()
    assert update["canceled"] is True
    if phase == "save_updated":
        poll["generation"] = 4
    panel._finish_local_update(panel._local_update_steps.pending)
    assert ("preview" in nodes) is keeps_preview
    assert (panel._local_update_steps.pending is not None) is keeps_pending
    assert panel.snapshot()["message"] == message
    assert project_path.read_bytes() == b"saved local project"
    assert preview_path.exists() and backup_path.read_bytes() == b"saved local project"
    assert journal["localUpdate"]["state"] == "ready"
    assert "linked" not in actions


def test_update_success_order_characterization(update_case, monkeypatch):
    panel, state, actions, nodes, update, poll, project_path, preview_path, backup_path, journal = update_case
    module = import_module("lfs_plugins.gallery_controller")
    job = panel._local_update_steps.pending
    nodes.clear()
    monkeypatch.setattr(panel, "_staged_nodes", lambda path: [])
    monkeypatch.setattr(module.lf, "load_gallery_scene", lambda *args, **kwargs: nodes.update(preview=SimpleNamespace(name="preview", uuid="incoming")), raising=False)
    monkeypatch.setattr(module.lf.ui, "dismiss_import", lambda: actions.append("dismiss import"))
    monkeypatch.setattr(panel, "_visible_splats", lambda: [n for n in nodes.values() if n.uuid != "incoming"])
    dirty = [True]
    monkeypatch.setattr(module.lf, "project_is_dirty", lambda: dirty[0], raising=False)
    saves = []
    def save(continuation, **kwargs):
        assert "preview" not in nodes
        saves.append(continuation)
    monkeypatch.setattr(panel, "_save_current_project", save)
    panel.service.prepare_local_update = lambda *args: actions.append("backup requested") or "backup"
    monkeypatch.setattr(panel, "_apply_local_update", lambda *args: actions.append("apply"))
    panel.service.link_download = lambda *args, **kwargs: actions.append("link requested") or "link"
    panel._finish_local_update(job)
    assert update["phase"] == "save_before_backup" and actions == [] and "preview" not in nodes
    panel._finish_local_update(job)
    assert update["phase"] == "save_before_backup" and len(saves) == 1 and "preview" not in nodes
    dirty[0] = False
    saves.pop()()
    assert update["phase"] == "backup" and actions[-1] == "backup requested"
    panel._finish_local_update(job)
    assert update["phase"] == "importing" and "preview" in nodes
    panel._finish_local_update(job)
    assert update["phase"] == "save_updated" and actions[-1] == "apply"
    poll["generation"] = 4
    panel._finish_local_update(job)
    assert update["phase"] == "linking" and actions[-1] == "link requested"
    panel._finish_local_update(job)
    assert panel._local_update_steps.pending is None
    assert panel.snapshot()["message"] == "Linked project updated. Your previous local work is kept in its recovery copy."
    assert project_path.exists() and preview_path.exists() and backup_path.exists()


@pytest.mark.parametrize("phase", ["staging", "importing", "save_before_backup", "backup",
                                   "applying", "save_updated", "linking"])
def test_update_failure_characterization(update_case, monkeypatch, phase):
    panel, state, actions, nodes, update, poll, project_path, preview_path, backup_path, journal = update_case
    module = import_module("lfs_plugins.gallery_controller")
    update["phase"] = phase
    if phase == "staging":
        journal["stagedImport"]["state"] = "failed"
        journal["stagedImport"]["message"] = "stage failed"
    elif phase == "importing":
        nodes.clear()
        monkeypatch.setattr(module.lf.ui, "get_import_state", lambda: {"active": False, "error": "import failed"})
    elif phase == "save_before_backup":
        panel._save_pending = {"project": update["project"], "identity": state["identity"],
                               "generation": 4, "continuation": lambda: None}
        poll["error"] = "save failed"
    elif phase == "backup":
        journal["localUpdate"].update(state="failed", message="backup failed")
    elif phase == "applying":
        update["phase"] = "applying"
        update.update(path=str(preview_path), incoming="incoming", old_nodes=["old"], backup_id="backup")
        monkeypatch.setattr(panel, "_apply_local_update", lambda *args: (_ for _ in ()).throw(OSError("apply failed")))
    elif phase == "save_updated":
        poll["error"] = "save failed"
    else:
        journal["linkOperation"].update(state="failed", message="link failed")
    monkeypatch.setattr(module.lf, "project_open", lambda *args, **kwargs: actions.append("reopen"), raising=False)
    panel._advance_phases()
    assert panel._local_update_steps.pending is None
    assert panel.snapshot()["actionError"]
    assert journal["localUpdate"]["state"] == "failed"
    assert project_path.read_bytes() == b"saved local project"
    assert preview_path.exists() and backup_path.read_bytes() == b"saved local project"
    assert ("preview" in nodes) is (phase not in ("importing", "save_before_backup", "backup"))
    assert "linked" not in actions


def test_update_waits_for_ready_backup_before_replacing_local_content(update_case, monkeypatch):
    panel, state, actions, nodes, update, poll, project_path, preview_path, backup_path, journal = update_case
    module = import_module("lfs_plugins.gallery_sync_steps")
    nodes["local"] = SimpleNamespace(name="local", uuid="local")
    update.update(phase="backup", path=str(preview_path))
    journal["localUpdate"]["state"] = "preparing"
    panel.service.busy = True
    monkeypatch.setattr(module, "restore_view", lambda *args, **kwargs: pytest.fail("Changed the view before backup"))
    monkeypatch.setattr(module.lf, "project_save", lambda **kwargs: pytest.fail("Saved a replacement before backup"), raising=False)
    panel._finish_local_update(panel._local_update_steps.pending)
    assert set(nodes) == {"local", "preview"} and actions == []
    assert project_path.read_bytes() == backup_path.read_bytes()
    assert journal["localUpdate"]["state"] == "preparing"


def test_update_cancel_after_import_removes_only_its_preview(update_case):
    panel, state, actions, nodes, update, poll, project_path, preview_path, backup_path, journal = update_case
    nodes["local"] = SimpleNamespace(name="local", uuid="local")
    update["phase"] = "importing"
    panel._action_pause()
    panel._finish_local_update(panel._local_update_steps.pending)
    assert set(nodes) == {"local"}
    assert actions == ["cancel import", "hide preview", "remove preview"]
    assert project_path.read_bytes() == backup_path.read_bytes()
    assert journal["localUpdate"]["state"] == "ready"


@pytest.fixture
def open_case(gallery, monkeypatch, tmp_path):
    panel, state, actions = gallery
    module = import_module("lfs_plugins.gallery_controller")
    source = tmp_path / "download.licht"
    source.write_bytes(b"kept download")
    opened = tmp_path / "opened.licht"
    opened.write_bytes(b"new project")
    native_scene = SimpleNamespace(is_valid=lambda: True, total_gaussian_count=2)
    monkeypatch.setattr(module.lf, "get_scene", lambda: native_scene, raising=False)
    monkeypatch.setattr(module.lf, "project_is_dirty", lambda: False, raising=False)
    monkeypatch.setattr(module.lf, "is_training_active", lambda: False, raising=False)
    monkeypatch.setattr(module.lf, "project_open", lambda *args, **kwargs: actions.append("opened"), raising=False)
    monkeypatch.setattr(module.lf, "project_poll_write", lambda: {"path": str(opened)}, raising=False)
    monkeypatch.setattr(module.lf.io, "inspect_project", lambda path: SimpleNamespace(project_uuid="new-project"))
    monkeypatch.setattr(import_module("lfs_plugins.gallery_sync_steps"), "restore_view", lambda *args, **kwargs: actions.append("view"))
    monkeypatch.setattr(panel, "_mark_viewing_copy", lambda *args: actions.append("viewing copy"))
    monkeypatch.setattr(panel, "_link_saved_download", lambda *args: actions.append("link requested") or "link")
    monkeypatch.setattr(panel, "_refresh_model", lambda: None)
    monkeypatch.setattr(panel, "_schedule_poll", lambda: None)
    monkeypatch.setattr(import_module("lfs_plugins.portable_project"), "ProjectFile",
                        lambda source: SimpleNamespace(manifest={"nodes": [{"count": 2}]}))
    project = SimpleNamespace(project_uuid="new-project")
    index = SimpleNamespace(load=lambda: True, get_asset=lambda identifier: None,
                            register_licht_asset=lambda *args, **kwargs: (project, True))
    monkeypatch.setattr(import_module("lfs_plugins.asset_index"), "AssetIndex", lambda: index)
    stage = {"id": "stage", "state": "ready", "projectPath": str(opened), "projectId": "new-project",
             "projectStamp": module.file_stamp(opened)}
    journal = {"id": "download", "stagedImport": stage, "linkOperation": {"id": "link", "state": "ready"}}
    state["jobs"] = [journal]
    panel.service.stage_download = lambda identifier: "stage"
    panel.service.environment_path = lambda job: None
    job = {"id": "download", "path": str(source), "result": {"title": "Gallery"}}
    panel._open_download(job, state["identity"])
    return panel, state, actions, source, opened, stage, journal


def test_download_open_order_characterization(open_case):
    panel, state, actions, source, opened, stage, journal = open_case
    assert panel._download_open_steps.pending["_opening"]["phase"] == "staging"
    assert panel.snapshot()["message"] == "Checking downloaded scene…"
    panel._finish_import()
    assert panel._download_open_steps.pending["_opening"]["phase"] == "opened"
    assert actions == ["opened"] and source.exists() and opened.exists()
    panel._finish_import()
    assert actions == ["opened", "viewing copy", "view", "link requested"]
    assert panel._download_open_steps.pending["_link"] == "link"
    assert "Saving its gallery link" in panel.snapshot()["message"]
    panel._finish_import()
    assert panel._download_open_steps.pending is None
    assert panel._download_open_steps.pulled_project == {"id": "new-project", "path": str(opened), "jobId": "download"}
    assert "saved and linked" in panel.snapshot()["message"]
    assert journal["linkOperation"]["state"] == "ready" and source.read_bytes() == b"kept download"


@pytest.mark.parametrize("phase", ["staging", "opened", "linking"])
def test_download_open_cancel_characterization(open_case, phase):
    panel, state, actions, source, opened, stage, journal = open_case
    job = panel._download_open_steps.pending
    if phase == "opened":
        panel._finish_import()
    elif phase == "linking":
        panel._finish_import()
        panel._finish_import()
    panel._action_pause()
    if phase == "staging":
        panel._finish_import()
        assert panel._download_open_steps.pending is None and not actions
        assert "download is kept" in panel.snapshot()["message"]
    elif phase == "opened":
        panel._download_open_steps.detached = True
        panel._finish_import()
        assert panel._download_open_steps.pending is None and actions == ["opened"]
        assert "kept locally" in panel.snapshot()["message"]
    else:
        panel._download_open_steps.detached = True
        panel._finish_import()
        assert panel._download_open_steps.pending is None and "refresh your gallery" in panel.snapshot()["message"]
    assert source.read_bytes() == b"kept download" and opened.read_bytes() == b"new project"
    assert journal["linkOperation"]["state"] == "ready"


def test_cancel_after_native_open_does_not_register_or_link(open_case):
    panel, state, actions, source, opened, stage, journal = open_case
    panel._finish_import()
    assert actions == ["opened"]
    panel._action_pause()
    panel._finish_import()
    assert panel._download_open_steps.pending is None
    assert actions == ["opened"]
    assert panel._message == import_module("lfs_plugins.gallery_sync_steps").tr("info.canceled")
    assert source.read_bytes() == b"kept download" and opened.read_bytes() == b"new project"
    assert journal["linkOperation"]["state"] == "ready"


@pytest.mark.parametrize("phase", ["staging", "opened", "linking"])
def test_download_open_failure_characterization(open_case, monkeypatch, phase):
    panel, state, actions, source, opened, stage, journal = open_case
    module = import_module("lfs_plugins.gallery_controller")
    if phase == "staging":
        stage.update(state="failed", message="stage failed")
    elif phase == "opened":
        panel._finish_import()
        opened.write_bytes(b"changed project")
    else:
        panel._finish_import()
        panel._finish_import()
        journal["linkOperation"].update(state="failed", message="link failed")
    if phase == "linking":
        panel._finish_import()
        assert "could not be saved" in panel.snapshot()["message"]
    else:
        with pytest.raises(ValueError):
            panel._finish_import()
    assert source.read_bytes() == b"kept download" and opened.exists()
    assert journal["linkOperation"]["state"] == ("failed" if phase == "linking" else "ready")


def test_download_keep_order_characterization(open_case):
    panel, state, actions, source, opened, stage, journal = open_case
    panel._download_open_steps.pending = {"id": "download", "result": {"title": "Gallery"},
        "_accountIdentity": state["identity"], "_register": {"phase": "staging", "stage_id": "stage"}}
    panel._finish_register_download(panel._download_open_steps.pending)
    assert panel._download_open_steps.pending["_register"]["phase"] == "linking"
    assert actions == ["viewing copy", "link requested"]
    assert source.read_bytes() == b"kept download" and opened.read_bytes() == b"new project"
    panel._finish_register_download(panel._download_open_steps.pending)
    assert panel._download_open_steps.pending is None
    assert panel._download_open_steps.pulled_project == {"id": "new-project", "path": str(opened), "jobId": "download"}
    assert journal["linkOperation"]["state"] == "ready"


@pytest.mark.parametrize("phase", ["staging", "linking"])
def test_download_keep_cancel_characterization(open_case, phase):
    panel, state, actions, source, opened, stage, journal = open_case
    panel._download_open_steps.pending = {"id": "download", "result": {"title": "Gallery"},
        "_accountIdentity": state["identity"], "_register": {"phase": "staging", "stage_id": "stage"}}
    if phase == "linking":
        panel._finish_register_download(panel._download_open_steps.pending)
    panel._action_pause()
    panel._finish_register_download(panel._download_open_steps.pending)
    assert panel._download_open_steps.pending is None
    assert (panel._download_open_steps.pulled_project is not None) is (phase == "linking")
    assert actions == (["viewing copy", "link requested"] if phase == "linking" else [])
    assert source.read_bytes() == b"kept download" and opened.read_bytes() == b"new project"
    assert journal["linkOperation"]["state"] == "ready"


@pytest.mark.parametrize("phase", ["staging", "linking"])
def test_download_keep_failure_characterization(open_case, phase):
    panel, state, actions, source, opened, stage, journal = open_case
    panel._download_open_steps.pending = {"id": "download", "result": {"title": "Gallery"},
        "_accountIdentity": state["identity"], "_register": {"phase": "staging", "stage_id": "stage"}}
    if phase == "staging":
        stage.update(state="failed", message="stage failed")
    else:
        panel._finish_register_download(panel._download_open_steps.pending)
        journal["linkOperation"].update(state="failed", message="link failed")
    with pytest.raises(ValueError):
        panel._finish_register_download(panel._download_open_steps.pending)
    assert source.read_bytes() == b"kept download" and opened.read_bytes() == b"new project"
    assert journal["linkOperation"]["state"] == ("failed" if phase == "linking" else "ready")


def test_cancel_after_download_link_submission_reports_completed_link(open_case):
    panel, state, actions, source, opened, stage, journal = open_case
    panel._download_open_steps.pending = {"id": "download", "result": {"title": "Gallery"},
        "_accountIdentity": state["identity"], "_register": {"phase": "staging", "stage_id": "stage"}}
    panel._finish_register_download(panel._download_open_steps.pending)
    assert actions == ["viewing copy", "link requested"]
    panel._action_pause()
    panel._finish_register_download(panel._download_open_steps.pending)
    assert panel._download_open_steps.pulled_project == {"id": "new-project", "path": str(opened), "jobId": "download"}
    assert panel._download_open_steps.pending is None
    assert journal["linkOperation"]["state"] == "ready"
    assert source.read_bytes() == b"kept download" and opened.read_bytes() == b"new project"


@pytest.mark.parametrize("outcome,keeps_export,queued", [
    ("active", True, False), ("canceled", False, False), ("failed", False, False),
    ("completed", True, True), ("timeout", False, False), ("foreign", False, False),
])
def test_publish_preparation_characterization(gallery, monkeypatch, tmp_path, outcome, keeps_export, queued):
    panel, state, actions = gallery
    module = import_module("lfs_plugins.gallery_controller")
    export = tmp_path / "prepared.ply"
    export.write_bytes(b"prepared scene")
    panel._publish_steps.pending = (export, {"title": "Gallery"}, "project", 0)
    panel._operation_title = "Gallery"
    panel.service.queue_prepared_upload = lambda *args: actions.append("queued")
    native = {"path": str(export), "active": outcome == "active", "progress": 0.5,
              "outcome": outcome if outcome in ("canceled", "failed", "completed") else "waiting"}
    if outcome == "foreign":
        native["path"] = str(tmp_path / "other.ply")
    monkeypatch.setattr(module.lf.ui, "get_export_state", lambda: native, raising=False)
    panel._finish_export()
    assert export.exists() is keeps_export
    assert ("queued" in actions) is queued
    assert (panel._publish_steps.pending is not None) is (outcome == "active")
    assert panel.snapshot()["phase"] == ("preparing" if outcome == "active" else "idle")
    assert panel.snapshot()["message"] == ("" if queued else panel._message)
    assert state["jobs"] == []


@pytest.mark.parametrize("outcome", ["saved", "canceled", "failed", "changed"])
def test_publish_save_step_characterization(gallery, monkeypatch, tmp_path, outcome):
    panel, state, actions = gallery
    module = import_module("lfs_plugins.gallery_controller")
    project_path = tmp_path / "project.licht"
    project_path.write_bytes(b"saved project")
    project = ("project", str(project_path))
    poll = {"path": str(project_path), "generation": 3, "running": False, "error": ""}
    monkeypatch.setattr(panel, "_project_identity", lambda: project)
    monkeypatch.setattr(panel, "_visible_splats", lambda: [SimpleNamespace(name="visible")])
    monkeypatch.setattr(panel, "_schedule_poll", lambda: None)
    monkeypatch.setattr(module.lf, "project_poll_write", lambda: poll.copy(), raising=False)
    monkeypatch.setattr(module.lf, "project_save", lambda **kwargs: True, raising=False)
    monkeypatch.setattr(module.lf, "project_is_dirty", lambda: False, raising=False)
    monkeypatch.setattr(panel, "_publish_saved", lambda *args, **kwargs: actions.append("export start"))
    panel._publish({"title": "Gallery"})
    assert panel.snapshot()["phase"] == "preparing"
    assert panel.snapshot()["message"] == "Saving your current project…"
    assert project_path.read_bytes() == b"saved project" and state["jobs"] == []
    if outcome == "canceled":
        panel._action_pause()
    elif outcome == "failed":
        poll["error"] = "disk full"
    elif outcome == "changed":
        poll["generation"] = 5
    else:
        poll["generation"] = 4
    if outcome in ("failed", "changed"):
        with pytest.raises(ValueError):
            panel._finish_current_project_save()
    else:
        panel._finish_current_project_save()
    assert actions == (["export start"] if outcome == "saved" else [])
    assert panel._save_pending is None and project_path.read_bytes() == b"saved project"
    assert state["jobs"] == []


def test_publish_never_queues_a_different_prepared_commit(gallery, monkeypatch, tmp_path):
    panel, state, actions = gallery
    module = import_module("lfs_plugins.gallery_controller")
    export = tmp_path / "prepared.ply"
    export.write_bytes(b"prepared scene")
    panel._publish_steps.pending = (export, {"title": "Gallery"}, "project", 0)
    panel._publish_steps.prepared_commit = "reviewed"
    panel.service.queue_prepared_upload = lambda *args: actions.append("queued")
    monkeypatch.setattr(module.lf.ui, "get_export_state", lambda: {
        "path": str(export), "active": False, "outcome": "completed", "commit_uuid": "different"}, raising=False)
    panel._finish_export()
    assert actions == [] and not export.exists()
    assert panel._publish_steps.pending is None and panel._publish_steps.prepared_commit is None
    assert panel.snapshot()["preparationFailure"]["status"] == "error"
    assert state["jobs"] == []

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
    panel._import_pending = job
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
    panel._finish_local_update(panel._import_pending)
    assert ("preview" in nodes) is keeps_preview
    assert (panel._import_pending is not None) is keeps_pending
    assert panel.snapshot()["message"] == message
    assert project_path.read_bytes() == b"saved local project"
    assert preview_path.exists() and backup_path.read_bytes() == b"saved local project"
    assert journal["localUpdate"]["state"] == "ready"
    assert "linked" not in actions


def test_update_success_order_characterization(update_case, monkeypatch):
    panel, state, actions, nodes, update, poll, project_path, preview_path, backup_path, journal = update_case
    module = import_module("lfs_plugins.gallery_controller")
    job = panel._import_pending
    nodes.clear()
    monkeypatch.setattr(panel, "_staged_nodes", lambda path: [])
    monkeypatch.setattr(module.lf, "load_gallery_scene", lambda *args, **kwargs: nodes.update(preview=SimpleNamespace(name="preview", uuid="incoming")), raising=False)
    monkeypatch.setattr(module.lf.ui, "dismiss_import", lambda: actions.append("dismiss import"))
    monkeypatch.setattr(panel, "_visible_splats", lambda: [SimpleNamespace(uuid="old")])
    saves = []
    monkeypatch.setattr(panel, "_save_current_project", lambda continuation, **kwargs: saves.append(continuation))
    panel.service.prepare_local_update = lambda *args: actions.append("backup requested") or "backup"
    monkeypatch.setattr(panel, "_apply_local_update", lambda *args: actions.append("apply"))
    panel.service.link_download = lambda *args, **kwargs: actions.append("link requested") or "link"
    panel._finish_local_update(job)
    assert update["phase"] == "importing" and actions == []
    panel._finish_local_update(job)
    assert update["phase"] == "save_before_backup" and actions == ["hide preview", "dismiss import"]
    assert project_path.read_bytes() == backup_path.read_bytes() and journal["localUpdate"]["state"] == "ready"
    saves.pop()()
    assert update["phase"] == "backup" and actions[-1] == "backup requested"
    panel._finish_local_update(job)
    assert update["phase"] == "save_updated" and actions[-1] == "apply"
    poll["generation"] = 4
    panel._finish_local_update(job)
    assert update["phase"] == "linking" and actions[-1] == "link requested"
    panel._finish_local_update(job)
    assert panel._import_pending is None
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
        update["phase"] = "backup"
        monkeypatch.setattr(panel, "_apply_local_update", lambda *args: (_ for _ in ()).throw(OSError("apply failed")))
    elif phase == "save_updated":
        poll["error"] = "save failed"
    else:
        journal["linkOperation"].update(state="failed", message="link failed")
    monkeypatch.setattr(module.lf, "project_open", lambda *args, **kwargs: actions.append("reopen"), raising=False)
    panel._advance_phases()
    assert panel._import_pending is None
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
    update["phase"] = "backup"
    journal["localUpdate"]["state"] = "preparing"
    panel.service.busy = True
    monkeypatch.setattr(module, "restore_view", lambda *args, **kwargs: pytest.fail("Changed the view before backup"))
    monkeypatch.setattr(module.lf, "project_save", lambda **kwargs: pytest.fail("Saved a replacement before backup"), raising=False)
    panel._finish_local_update(panel._import_pending)
    assert set(nodes) == {"local", "preview"} and actions == []
    assert project_path.read_bytes() == backup_path.read_bytes()
    assert journal["localUpdate"]["state"] == "preparing"


def test_update_cancel_after_import_removes_only_its_preview(update_case):
    panel, state, actions, nodes, update, poll, project_path, preview_path, backup_path, journal = update_case
    nodes["local"] = SimpleNamespace(name="local", uuid="local")
    update["phase"] = "importing"
    panel._action_pause()
    panel._finish_local_update(panel._import_pending)
    assert set(nodes) == {"local"}
    assert actions == ["cancel import", "hide preview", "remove preview"]
    assert project_path.read_bytes() == backup_path.read_bytes()
    assert journal["localUpdate"]["state"] == "ready"

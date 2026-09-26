from test_gallery_controller import gallery
from test_asset_manager_panel import panel_module
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_gallery_steps import update_case


def _freshness(module, path):
    commit = sha256(path.read_bytes()).hexdigest()
    link = {"sceneId": "scene", "commitUuid": commit, "contentRevision": "content", "metadataRevision": "metadata"}
    scene = {"id": "scene", "contentRevision": "content", "metadataRevision": "metadata"}
    return module.asset_sync_state({"id": "project", "commit_uuid": commit, "exists": True}, link, scene)["freshness"]


def _prepare_importing(panel, nodes, update):
    update.update(phase="importing", path=str(Path(update["project"][1]).with_name("preview.scene")))
    preview = SimpleNamespace(name="preview", uuid="incoming")
    nodes["preview"] = preview
    return preview


@pytest.mark.parametrize("phase", ["staging", "importing", "save_before_backup", "backup"])
def test_update_cancel_keeps_project_bytes_and_freshness(update_case, monkeypatch, tmp_path, phase):
    panel, state, _actions, nodes, update, _poll, project_path, _preview, _backup, journal = update_case
    module = __import__("lfs_plugins.gallery_controller", fromlist=["asset_sync_state"])
    baseline = project_path.read_bytes()
    before_hash = sha256(baseline).hexdigest()
    before_freshness = _freshness(module, project_path)
    backup_path = tmp_path / "backup.licht"
    monkeypatch.setattr(module.lf, "project_is_dirty", lambda: False, raising=False)
    monkeypatch.setattr(module.lf, "project_poll_write", lambda: {"generation": 3}, raising=False)
    if phase == "staging":
        update["phase"] = "staging"
    elif phase == "importing":
        _prepare_importing(panel, nodes, update)
    else:
        update.update(phase=phase, path=str(project_path.with_name("preview.scene")))
        if phase == "backup":
            backup_path.write_bytes(baseline)
            journal["localUpdate"].update(id="backup", state="ready", backupPath=str(backup_path))

    panel._local_update_steps.cancel()
    panel._finish_local_update(panel._local_update_steps.pending)

    assert sha256(project_path.read_bytes()).hexdigest() == before_hash
    assert _freshness(module, project_path) == before_freshness
    if phase == "backup":
        assert backup_path.read_bytes() == baseline


def test_update_backup_is_the_pre_apply_project(update_case, monkeypatch, tmp_path):
    panel, state, _actions, nodes, update, _poll, project_path, _preview, _backup, _journal = update_case
    baseline = project_path.read_bytes()
    backup_path = tmp_path / "backup.licht"
    module = __import__("lfs_plugins.gallery_controller", fromlist=["file_stamp"])
    monkeypatch.setattr(module.lf, "project_is_dirty", lambda: False, raising=False)
    monkeypatch.setattr(module.lf, "project_poll_write", lambda: {"generation": 3}, raising=False)
    def prepare_backup(*args):
        backup_path.write_bytes(project_path.read_bytes())
        return "backup"
    panel.service.prepare_local_update = prepare_backup
    update.update(phase="save_before_backup", path=str(project_path.with_name("preview.scene")))
    panel._finish_local_update(panel._local_update_steps.pending)

    assert backup_path.read_bytes() == baseline


def test_uncanceled_update_still_applies(update_case, monkeypatch):
    panel, state, actions, nodes, update, poll, project_path, preview_path, backup_path, journal = update_case
    module = __import__("lfs_plugins.gallery_controller", fromlist=["asset_sync_state"])
    job = panel._local_update_steps.pending
    panel._visible_splats = lambda: list(nodes.values())
    monkeypatch.setattr(module.lf, "project_is_dirty", lambda: False, raising=False)
    monkeypatch.setattr(module.lf, "project_poll_write", lambda: {"generation": 3}, raising=False)
    monkeypatch.setattr(panel, "_visible_splats", lambda: [SimpleNamespace(uuid="incoming"), SimpleNamespace(uuid="old")])
    monkeypatch.setattr(panel, "_staged_nodes", lambda _path: [])
    monkeypatch.setattr(panel, "_apply_local_update", lambda *args: actions.append("applied"))
    update.update(phase="backup", path=str(preview_path), generation=3,
                  stamp=module.file_stamp(project_path), backup_id="backup")
    state["jobs"][0]["localUpdate"].update(id="backup", state="ready", backupPath=str(backup_path))
    monkeypatch.setattr(module.lf, "load_gallery_scene", lambda _nodes, name, hidden: nodes.update({name: SimpleNamespace(name=name, uuid="incoming")} ), raising=False)
    panel._finish_local_update(job)
    panel._finish_local_update(job)

    assert "applied" in actions


def test_cancel_during_own_save_skips_backup(update_case, monkeypatch):
    panel, state, actions, _nodes, update, _poll, project_path, _preview, _backup, journal = update_case
    module = __import__("lfs_plugins.gallery_controller", fromlist=["file_stamp"])
    before = project_path.read_bytes()
    _nodes.clear()
    monkeypatch.setattr(module.lf, "project_is_dirty", lambda: True, raising=False)
    monkeypatch.setattr(module.lf, "project_poll_write", lambda: {"running": False, "generation": 4, "path": str(project_path)}, raising=False)
    update.update(phase="save_before_backup", path=str(project_path.with_name("preview.scene")))
    panel._save_current_project = lambda continuation, **kwargs: setattr(panel, "_save_pending", {
        "project": update["project"], "identity": state["identity"], "generation": 4,
        "continuation": lambda: actions.append("backup")})

    panel._finish_local_update(panel._local_update_steps.pending)
    assert panel._local_update_steps.save_in_progress
    panel._local_update_steps.cancel()
    panel._finish_current_project_save()
    panel._finish_local_update(panel._local_update_steps.pending)

    assert project_path.read_bytes() == before
    assert actions == []
    assert panel._local_update_steps.pending is None

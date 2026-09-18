"""Exercise closed-file project operations through the native Python bindings."""

from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path

import pytest


def _symlink_or_skip(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target)
    except OSError as error:
        if os.name == "nt" and error.winerror == 1314:
            pytest.skip("Windows symlink coverage requires Developer Mode or elevation")
        raise


def _overwrite_file(path: Path, replacement: Path) -> None:
    """Change an open project's identity without relying on rename semantics."""
    contents = replacement.read_bytes()
    with path.open("r+b") as stream:
        stream.seek(0)
        stream.write(contents)
        stream.truncate()
        stream.flush()
        os.fsync(stream.fileno())


def _overwrite_superblock(path: Path, replacement: Path) -> None:
    contents = replacement.read_bytes()[:256]
    with path.open("r+b") as stream:
        stream.seek(0)
        stream.write(contents)
        stream.flush()
        os.fsync(stream.fileno())


@pytest.fixture
def identity_project(native_io, tmp_path, monkeypatch):
    monkeypatch.setenv("LFS_HOME", str(tmp_path))
    path = tmp_path / "project.licht"
    shutil.copyfile(Path(__file__).parents[1] / "data" / "portable-sog.licht", path)
    return path, native_io.inspect_project_card(path)


@pytest.mark.parametrize("operation", [
    "restore", "rebind", "reduce", "embed", "thumbnail", "license", "clear_license",
    "compact", "rename", "dataset_reference",
])
@pytest.mark.parametrize("swap", ["identity", "path"])
def test_contents_refuses_swap_after_backup(native_io, identity_project, tmp_path, monkeypatch, operation, swap):
    from lfs_plugins.project_operations import ProjectOperations, ProjectOperationFailure

    path, card = identity_project
    replacement = tmp_path / "replacement.licht"
    native_io.restore_save(path, 1, replacement)
    selected = path
    if swap == "path":
        selected = tmp_path / "别名.licht"
        _symlink_or_skip(selected, path)
        shutil.copyfile(path, replacement)  # Even an identical project at another target is refused.
    before = replacement.read_bytes()
    store = ProjectOperations(native_io, tmp_path / "records")
    put = store._put

    def swap_after_backup(row):
        put(row)
        if row["status"] == "running":
            if swap == "identity":
                _overwrite_file(selected, replacement)
            else:
                selected.unlink()
                _symlink_or_skip(selected, replacement)

    monkeypatch.setattr(store, "_put", swap_after_backup)
    operations = {
        "restore": lambda: native_io.restore_save(selected, 1, selected),
        "rebind": lambda: native_io.rebind_checkpoint(selected, str(uuid.uuid4())),
        "reduce": lambda: native_io.reduce_size(selected, {"compact": False, "drop_thumbnail": True}),
        "embed": lambda: native_io.embed_dataset_file(selected),
        "thumbnail": lambda: native_io.set_project_preview(selected, b"\x89PNG\r\n\x1a\n"),
        "license": lambda: native_io.set_project_license(selected, "CC0-1.0", "Changed"),
        "clear_license": lambda: native_io.clear_project_license(selected),
        "compact": lambda: native_io.compact_project_file(selected),
        "rename": lambda: native_io.set_project_title(selected, "Changed"),
        "dataset_reference": lambda: native_io.set_dataset_reference(selected, tmp_path),
    }
    with pytest.raises(ProjectOperationFailure, match="(identity|path).*changed") as failure:
        store.run("guarded", {"path": str(selected), "id": str(card.project_uuid),
                  "commit_uuid": str(card.commit_uuid)}, operation, operations[operation])
    assert failure.value.record["status"] == "failed"
    assert "changed" in store.recover()["guarded"]["reason"]
    assert selected.read_bytes() == before


def test_contents_refuses_callback_for_another_existing_path(native_io, identity_project, tmp_path):
    path, card = identity_project
    other = tmp_path / "other.licht"
    shutil.copyfile(path, other)
    before = other.read_bytes()
    with pytest.raises(Exception, match="path changed"):
        native_io.run_project_operation(path, str(card.project_uuid), str(card.commit_uuid),
                                       lambda: native_io.set_project_title(other, "Wrong file"))
    assert other.read_bytes() == before


@pytest.mark.parametrize("swap", ["identity", "path"])
def test_compact_rechecks_destination_after_progress(native_io, identity_project, tmp_path, swap):
    path, _ = identity_project
    other = tmp_path / "other.licht"
    native_io.restore_save(path, 1, other)
    selected = path
    if swap == "path":
        selected = tmp_path / "alias.licht"
        _symlink_or_skip(selected, path)
    before = other.read_bytes()
    if swap == "identity":
        before = before[:256] + selected.read_bytes()[256:]
    swapped = False

    def progress(*_args):
        nonlocal swapped
        if not swapped:
            swapped = True
            if swap == "identity":
                _overwrite_superblock(path, other)
            else:
                selected.unlink()
                _symlink_or_skip(selected, other)

    with pytest.raises(Exception, match="(identity|path).*changed"):
        native_io.compact_project_file(selected, progress=progress)
    assert swapped and selected.read_bytes() == before


@pytest.mark.parametrize("swap", ["identity", "path"])
@pytest.mark.parametrize("guarded", [False, True])
def test_reduce_rechecks_planned_identity_after_progress(native_io, identity_project, tmp_path, swap, guarded):
    path, card = identity_project
    other = tmp_path / "other.licht"
    native_io.restore_save(path, 1, other)
    selected = path
    if swap == "path":
        selected = tmp_path / "alias.licht"
        _symlink_or_skip(selected, path)
        other.write_bytes(path.read_bytes())
    before = other.read_bytes()
    swapped = False

    def progress(*args):
        nonlocal swapped
        if swapped:
            return
        swapped = True
        if swap == "identity":
            _overwrite_file(path, other)
        else:
            selected.unlink()
            _symlink_or_skip(selected, other)

    with pytest.raises(Exception, match="(identity|path).*changed"):
        operation = lambda: native_io.reduce_size(selected, {"compact": False, "drop_thumbnail": True}, progress=progress)
        if guarded:
            native_io.run_project_operation(selected, str(card.project_uuid), str(card.commit_uuid), operation)
        else:
            operation()
    assert swapped and selected.read_bytes() == before


def test_repair_refuses_changed_id_without_creating_destination(native_io, identity_project, tmp_path):
    path, card = identity_project
    other = tmp_path / "other.licht"
    native_io.restore_save(path, 1, other)
    damaged = bytearray(other.read_bytes())
    damaged[4096] ^= 1
    damaged[8192] ^= 1
    path.write_bytes(damaged)
    destination = tmp_path / "修复.licht"
    with pytest.raises(Exception, match="identity changed"):
        native_io.repair_project(path, destination, str(card.project_uuid))
    assert not destination.exists() and path.read_bytes() == damaged


def test_contents_edits_and_restore_accept_unicode_alias(native_io, identity_project, tmp_path):
    path, card = identity_project
    alias = tmp_path / "别名-é.licht"
    _symlink_or_skip(alias, path)

    def edit():
        native_io.set_project_title(alias, "项目")
        native_io.set_project_license(alias, "CC0-1.0", "作者")
        native_io.compact_project_file(alias)

    native_io.run_project_operation(path, str(card.project_uuid), str(card.commit_uuid), edit)
    assert alias.is_symlink() and native_io.inspect_project_card(path).title == "项目"
    result = native_io.restore_save(alias, 1, alias)
    assert alias.is_symlink() and result.project_uuid == card.project_uuid


@pytest.mark.parametrize("operation", ["backup", "restore_backup"])
@pytest.mark.parametrize("swap", ["identity", "path"])
def test_contents_recovery_write_refuses_changed_project(native_io, identity_project, tmp_path, operation, swap):
    path, card = identity_project
    backup = native_io.backup_project_file(path)
    other = tmp_path / "other.licht"
    native_io.restore_save(path, 1, other)
    alias = tmp_path / "别名.licht"
    _symlink_or_skip(alias, path)
    if swap == "path":
        other.write_bytes(path.read_bytes())
    before = other.read_bytes()

    def write_recovery():
        if swap == "identity":
            _overwrite_file(path, other)
        else:
            alias.unlink()
            _symlink_or_skip(alias, other)
        if operation == "backup":
            native_io.backup_project_file(alias)
        else:
            native_io.restore_project_backup(alias, backup, str(card.project_uuid), str(card.commit_uuid))

    with pytest.raises(Exception, match="(identity|path).*changed"):
        native_io.run_project_operation(alias, str(card.project_uuid), str(card.commit_uuid), write_recovery)
    assert alias.read_bytes() == before


def test_contents_backup_refuses_a_different_existing_recovery(native_io, identity_project, tmp_path):
    path, _ = identity_project
    backup = native_io.backup_project_file(path)
    other = tmp_path / "other.licht"
    native_io.restore_save(path, 1, other)
    backup.write_bytes(other.read_bytes())
    before = backup.read_bytes()
    with pytest.raises(Exception, match="belongs to another file"):
        native_io.backup_project_file(path)
    assert backup.read_bytes() == before


def test_contents_recovery_accepts_unicode_relative_alias(native_io, identity_project, tmp_path, monkeypatch):
    path, card = identity_project
    original = path.read_bytes()
    alias = tmp_path / "别名-é.licht"
    _symlink_or_skip(alias, path)
    monkeypatch.chdir(tmp_path)
    relative = Path(alias.name)
    backup = native_io.backup_project_file(relative)
    changed = native_io.set_project_title(relative, "Changed")
    native_io.restore_project_backup(relative, backup, str(card.project_uuid), str(changed.commit_uuid))
    assert alias.is_symlink() and path.read_bytes() == original
    destination = Path("另一个.licht")
    restored = native_io.restore_save(relative, 1, destination)
    assert restored.project_uuid != card.project_uuid and destination.is_file()


def _fixture() -> Path:
    configured = os.environ.get("LFS_PROJECT_OPERATIONS_FIXTURE")
    if configured:
        return Path(configured)
    for parent in Path(__file__).parents:
        candidate = parent / ".codex_tmp/am_concept/testhome/projects/bonsai.licht"
        if candidate.is_file():
            return candidate
    return Path(__file__).parents[1] / "data" / "portable-sog.licht"


@pytest.fixture(scope="module")
def native_io():
    try:
        from lichtfeld import io
    except ImportError as error:
        pytest.skip(f"native lichtfeld.io is unavailable: {error}")
    return io


def test_closed_file_operations(native_io, tmp_path):
    source = _fixture()
    if not source.is_file():
        pytest.skip(f"operations fixture is unavailable: {source}")
    path = tmp_path / "copy.licht"
    shutil.copy2(source, path)

    canceled = native_io.verify_project_file(path, cancel=lambda: True)
    assert canceled.status is native_io.ProjectVerificationStatus.CANCELED
    verified = native_io.verify_project_file(path)
    assert verified.status is native_io.ProjectVerificationStatus.VERIFIED

    plan = native_io.plan_reduce_size(path)
    assert plan.physical_size > 0
    assert plan.input_commit_uuid
    reduced = native_io.reduce_size(
        path, {"drop_unbound_checkpoints": False, "drop_embedded_dataset": False}
    )
    assert reduced.card.physical_file_size > 0
    assert reduced.recovery_copy.is_file()

    restored = native_io.restore_save(path, 1, tmp_path / "restored.licht")
    assert restored.project_uuid != native_io.inspect_project_card(path).project_uuid

    png = b"\x89PNG\r\n\x1a\n"
    native_io.set_project_preview(path, png)
    native_io.set_project_license(path, "CC-BY-4.0", "Python test")
    titled = native_io.set_project_title(path, "Python operation copy")
    assert titled.title == "Python operation copy"
    assert native_io.inspect_project_details(path).license.identifier == "CC-BY-4.0"

    compacted = native_io.compact_project_file(path)
    assert "older save points" in compacted.diagnostic
    native_io.clear_project_license(path)
    assert native_io.inspect_project_details(path).license is None


def test_contents_removals_persist_until_compaction(native_io, tmp_path):
    import base64
    import json
    source = _fixture()
    if not source.is_file():
        pytest.skip(f'operations fixture is unavailable: {source}')
    path = tmp_path / 'contents.licht'
    shutil.copy2(source,path)
    native_io.set_project_title(path,'Before removal')
    native_io.set_project_license(path,'CC-BY-4.0','Credit: Studio')
    png=base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aGNcAAAAASUVORK5CYII=')
    native_io.set_project_preview(path,png)
    details=native_io.inspect_project_details(path)
    current=details.card.generation
    with pytest.raises(Exception,match='current save'):
        native_io.reduce_size(path,dict(compact=False,drop_unbound_checkpoints=False,save_generation=current))
    oldest=details.save_history[0].generation
    size=path.stat().st_size
    reduced=native_io.reduce_size(path,dict(compact=False,drop_unbound_checkpoints=False,save_generation=oldest))
    assert reduced.recovery_copy.is_file() and reduced.bytes_reclaimed==0
    assert path.stat().st_size>=size
    pending=json.loads(native_io.inspect_project_details(path).manifest['contents_removals'])['rows']
    assert pending[0]['id']==f'save:{oldest}'
    with pytest.raises(Exception,match='removed'):
        native_io.restore_save(path,oldest,tmp_path/'removed-save.licht')
    native_io.reduce_size(path,dict(compact=False,drop_unbound_checkpoints=False,drop_thumbnail=True))
    assert not native_io.inspect_project_details(path).card.has_preview
    native_io.clear_project_license(path)
    details=native_io.inspect_project_details(path)
    assert details.license is None
    assert {r['kind'] for r in json.loads(details.manifest['contents_removals'])['rows']}=={'save','thumbnail','license'}
    assert native_io.verify_project_file(path).status is native_io.ProjectVerificationStatus.VERIFIED
    before=path.stat().st_size
    native_io.compact_project_file(path)
    assert path.stat().st_size<before
    assert 'contents_removals' not in native_io.inspect_project_details(path).manifest


def test_restore_same_path_preserves_identity_and_history(native_io, tmp_path, monkeypatch):
    if not hasattr(native_io, 'undo_contents_removal'):
        pytest.skip('in-place Restore requires the rebuilt Contents native module')
    source=_fixture()
    if not source.is_file():pytest.skip(f'operations fixture is unavailable: {source}')
    home = tmp_path / 'home'
    monkeypatch.setenv('LFS_HOME', str(home))
    path=tmp_path/'restore.licht'
    shutil.copy2(source,path)
    native_io.set_project_title(path,'Old title')
    old=native_io.inspect_project_card(path)
    native_io.set_project_title(path,'New title')
    before=native_io.inspect_project_details(path)
    restored=native_io.restore_save(path,old.generation,path)
    assert restored.project_uuid==old.project_uuid
    after=native_io.inspect_project_details(path)
    assert len(after.save_history)==len(before.save_history)+1
    assert after.save_history[-1].kind==native_io.ProjectCommitKind.CONTENTS
    assert restored.title=='Old title'
    assert native_io.verify_project_file(path).status is native_io.ProjectVerificationStatus.VERIFIED
    assert len(list(tmp_path.glob('*.licht')))==1
    assert not list(tmp_path.glob('*.bak'))
    assert (home / 'data/backups/contents' / str(old.project_uuid)).is_dir()


def test_removing_bound_checkpoint_preserves_visible_model(native_io, tmp_path):
    import json

    configured = os.environ.get("LFS_PROJECT_CHECKPOINT_FIXTURE")
    source = Path(configured) if configured else _fixture().parent / "mrnf/stump/project.licht"
    if not source.is_file():
        pytest.skip(f"checkpoint fixture is unavailable: {source}")
    path = tmp_path / "checkpoint.licht"
    shutil.copy2(source, path)
    before = native_io.inspect_project_details(path)
    checkpoint = next(cp for cp in before.retained_checkpoints if cp.retained and cp.binds_scene_graph)

    result = native_io.reduce_size(path, {
        "compact": False,
        "drop_unbound_checkpoints": False,
        "checkpoint_uuid": str(checkpoint.instance_uuid),
    })
    after = native_io.inspect_project_details(path)
    assert result.checkpoints_removed == 1
    assert result.bytes_reclaimed == 0
    assert result.recovery_copy.is_file()
    assert after.scene_graph.training_node_id is None
    assert after.scene_graph.node_counts_by_type == before.scene_graph.node_counts_by_type
    assert not any(cp.retained and cp.instance_uuid == checkpoint.instance_uuid for cp in after.retained_checkpoints)
    assert any(str(part.fourcc) == "SPLT" and part.stored_bytes > 0 for part in after.chapters)
    removed = json.loads(after.manifest["contents_removals"])["rows"]
    assert any(row["id"] == f"checkpoint:{checkpoint.instance_uuid}" for row in removed)
    assert native_io.verify_project_file(path).status is native_io.ProjectVerificationStatus.VERIFIED


def test_rebinding_repeatedly_keeps_recovery_copies_out_of_catalog(native_io, tmp_path):
    configured = os.environ.get("LFS_PROJECT_CHECKPOINT_FIXTURE")
    source = Path(configured) if configured else _fixture().parent / "mrnf/stump/project.licht"
    if not source.is_file():
        pytest.skip(f"checkpoint fixture is unavailable: {source}")
    path = tmp_path / "resume.licht"
    shutil.copy2(source, path)
    details = native_io.inspect_project_details(path)
    checkpoint = next(cp for cp in details.retained_checkpoints if cp.retained and cp.binds_scene_graph)
    first = native_io.rebind_checkpoint(path, str(checkpoint.instance_uuid))
    second = native_io.rebind_checkpoint(path, str(checkpoint.instance_uuid))
    assert first.commit_uuid != second.commit_uuid
    assert first.project_uuid == second.project_uuid == details.card.project_uuid
    assert len(list(tmp_path.glob("*.licht"))) == 1
    assert not list(tmp_path.glob("*.bak"))
    backups = Path(os.environ['LFS_HOME']) / 'data/backups/contents' / str(first.project_uuid)
    assert (backups / (str(details.card.commit_uuid) + '.licht.bak')).is_file()
    assert (backups / (str(first.commit_uuid) + '.licht.bak')).is_file()
    assert native_io.verify_project_file(path).status is native_io.ProjectVerificationStatus.VERIFIED


def test_chosen_thumbnail_survives_other_contents_edits(native_io, tmp_path):
    import struct
    import zlib

    def png(rgb):
        def chunk(kind, payload):
            return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload))

        return (b"\x89PNG\r\n\x1a\n"
                + chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 1, 8, 2, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(b"\x00" + bytes(rgb) * 2))
                + chunk(b"IEND", b""))

    source = _fixture()
    if not source.is_file():
        pytest.skip(f"operations fixture is unavailable: {source}")
    path = tmp_path / "preview.licht"
    shutil.copy2(source, path)
    dataset = tmp_path / "dataset"
    for folder in ("images", "images_4"):
        images = dataset / folder
        images.mkdir(parents=True)
        (images / "first.png").write_bytes(png((255, 0, 0)))
    native_io.set_dataset_reference(path, dataset, True)
    native_io.preview_from_first_dataset_image(path)
    chosen = png((0, 0, 255))
    assert native_io.read_preview(path) != chosen

    native_io.set_project_preview(path, chosen)
    assert native_io.read_preview(path) == chosen
    native_io.set_project_license(path, "CC-BY-4.0", "Credit: Studio")
    assert native_io.read_preview(path) == chosen
    native_io.reduce_size(path, {
        "compact": False,
        "drop_unbound_checkpoints": False,
        "drop_thumbnail": True,
    })
    assert not native_io.inspect_project_card(path).has_preview
    native_io.clear_project_license(path)
    assert not native_io.inspect_project_card(path).has_preview
    assert native_io.verify_project_file(path).status is native_io.ProjectVerificationStatus.VERIFIED


def test_operation_guard_rejects_replaced_identity_and_commit(native_io, tmp_path):
    import uuid
    source = _fixture()
    if not source.is_file():
        pytest.skip('native project fixture unavailable')
    path = tmp_path / 'guarded.licht'
    shutil.copy2(source, path)
    card = native_io.inspect_project_card(path)
    called = []
    with pytest.raises(Exception, match='project changed'):
        native_io.run_project_operation(path, str(uuid.uuid4()), str(card.commit_uuid), lambda: called.append(True))
    native_io.set_project_title(path, 'Changed')
    with pytest.raises(Exception, match='project changed'):
        native_io.run_project_operation(path, str(card.project_uuid), str(card.commit_uuid), lambda: called.append(True))
    assert called == []
    current = native_io.inspect_project_card(path)
    def edit():
        native_io.set_project_title(path, 'Guarded')
        native_io.compact_project_file(path)
    native_io.run_project_operation(path, str(current.project_uuid), str(current.commit_uuid), edit)
    assert native_io.inspect_project_card(path).title == 'Guarded'
    assert native_io.verify_project_file(path).status is native_io.ProjectVerificationStatus.VERIFIED


@pytest.mark.xfail(
    os.name == "nt",
    reason="Native closed-file mutations do not yet accept CJK paths on Windows",
    strict=True,
)
def test_closed_file_mutation_accepts_unicode_path(native_io, tmp_path):
    path = tmp_path / "项目.licht"
    shutil.copy2(_fixture(), path)

    changed = native_io.set_project_title(path, "Changed")

    assert changed.title == "Changed"


@pytest.mark.parametrize('kill_point', [
    'running',
    'completed',
])
def test_contents_kill_rolls_back_on_restart(native_io, tmp_path, kill_point):
    import subprocess
    import sys
    from lfs_plugins.project_operations import ProjectOperations
    source = _fixture()
    if not source.is_file():
        pytest.skip('native project fixture unavailable')
    path = tmp_path / 'interrupted.licht'
    shutil.copy2(source, path)
    before = native_io.inspect_project_card(path)
    root = tmp_path / 'store'
    code = '''
import os, sys
from lichtfeld import io
from lfs_plugins.project_operations import ProjectOperations
path, root, point = sys.argv[1:]
class KilledOperations(ProjectOperations):
    def _put(self, row):
        if point == "completed" and row["status"] == "completed":
            os._exit(37)
        super()._put(row)
        if point == "running" and row["status"] == "running":
            os._exit(37)
card = io.inspect_project_card(path)
KilledOperations(io, root).run("project-kill", {"id": str(card.project_uuid), "path": path,
    "commit_uuid": str(card.commit_uuid)}, "Set license",
    lambda: io.set_project_license(path, "CC0-1.0", "kill marker"))
'''
    child_environment = os.environ.copy()
    child_environment["PYTHONPATH"] = os.pathsep.join(
        str(entry) for entry in sys.path if entry
    )
    child = subprocess.run([sys.executable, '-c', code, str(path), str(root), kill_point],
        capture_output=True, text=True, timeout=60, env=child_environment)
    assert child.returncode == 37, child.stderr
    rows = ProjectOperations(native_io, root).recover()
    assert rows['project-kill']['status'] == 'failed'
    assert Path(rows['project-kill']['backup_path']).is_file()
    after = native_io.inspect_project_card(path)
    assert after.project_uuid == before.project_uuid and after.commit_uuid == before.commit_uuid
    assert native_io.verify_project_file(path).status is native_io.ProjectVerificationStatus.VERIFIED
    assert ProjectOperations(native_io, root).recover() == rows


def test_contents_recovery_refuses_a_different_project(native_io, tmp_path):
    import os
    from lfs_plugins.project_operations import ProjectOperations
    source = _fixture()
    if not source.is_file():
        pytest.skip('native project fixture unavailable')
    path = tmp_path / 'changed.licht'
    shutil.copy2(source, path)
    card = native_io.inspect_project_card(path)
    backup = native_io.backup_project_file(path)
    store = ProjectOperations(native_io, tmp_path / 'store')
    store._put(dict(id='project-changed', asset_id=str(card.project_uuid), path=str(path), title='Edit',
        status='running', input_commit=str(card.commit_uuid), backup_path=str(backup)))
    replacement_path = tmp_path / 'replacement.licht'
    replacement = native_io.restore_save(path, card.generation, replacement_path)
    assert replacement.project_uuid != card.project_uuid
    os.replace(replacement_path, path)
    before = path.read_bytes()
    rows = store.recover()
    assert 'project changed' in rows['project-changed']['reason']
    assert path.read_bytes() == before
    assert native_io.inspect_project_card(path).project_uuid == replacement.project_uuid


def test_repair_checks_recovered_identity_before_creating_destination(native_io, tmp_path):
    import uuid
    source = _fixture()
    if not source.is_file():
        pytest.skip('native project fixture unavailable')
    path = tmp_path / 'damaged.licht'
    shutil.copy2(source, path)
    before = native_io.inspect_project_card(path)
    with path.open('r+b') as stream:
        stream.seek(4096)
        stream.write(bytes(8192))
    destination = tmp_path / 'repaired.licht'
    with pytest.raises(Exception, match='identity changed'):
        native_io.repair_project(path, destination, str(uuid.uuid4()))
    assert not destination.exists()
    restored = native_io.repair_project(path, destination, str(before.project_uuid))
    assert restored.card.project_uuid == before.project_uuid
    assert native_io.verify_project_file(destination).status is native_io.ProjectVerificationStatus.VERIFIED

# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Scene.add_splat input checks and scene values that exist before any dataset is loaded."""

import inspect
import time

import pytest

from test_api_camera_project_runtime import camera_runtime, camera_runtime_with_numpy, _editor, _ui_editor  # noqa: F401
from test_render_on_demand_idle import _ledger, _tool

pytestmark = [pytest.mark.gpu, pytest.mark.integration]


def wait_until(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while True:
        result = predicate()
        if result or time.monotonic() >= deadline:
            return result
        time.sleep(.05)


@pytest.fixture(autouse=True)
def editor_wait(camera_runtime):
    _editor(camera_runtime, "import time\n" + inspect.getsource(wait_until))


def test_completed_training_imports_full_model_with_result_coordinates(camera_runtime_with_numpy, tmp_path, monkeypatch):
    camera_runtime = camera_runtime_with_numpy
    import hashlib
    from lichtfeld_sessions import geometry, transport
    from test_training_sessions import arrays

    artifact = tmp_path / "saved-full.ply"
    _editor(camera_runtime, 'import lichtfeld as lf')
    _ui_editor(camera_runtime, '''
import numpy as np, time
import lfs_plugins.external_training_panel as panel
from lfs_plugins.external_training import ExternalTraining
from lfs_plugins.external_training import presentation_transform
from lichtfeld_sessions import geometry
np.testing.assert_array_equal(
    presentation_transform({"schema": 1, "source_frame": "blender-world"}, np),
    presentation_transform({"schema": 1, "source_frame": "blender-world", "data_to_source": geometry.FLIP}, np))
scene = lf.get_scene()
def tensor(value):
    return lf.Tensor.from_numpy(np.ascontiguousarray(value, dtype=np.float32))
means = np.arange(24, dtype=np.float32).reshape(8, 3) * .01
sh0 = np.full((8, 1, 3), .1234567, dtype=np.float32)
shN = np.arange(8 * 45, dtype=np.float32).reshape(8, 15, 3) * .001234567
scene.add_splat("Existing local model", tensor(means), tensor(sh0), tensor(shN),
                tensor(np.full((8, 3), -3)), tensor(np.tile([1, 0, 0, 0], (8, 1))),
                tensor(np.zeros((8, 1))), sh_degree=3)
''')
    _editor(camera_runtime, f'''
source = lf.get_scene().get_node("Existing local model").splat_data()
saved_sh0 = source.sh0_raw.cpu().numpy().copy()
saved_shN = source.shN_raw.cpu().numpy().copy()
lf.io.save_ply(source, {str(artifact)!r})
''')
    result = {"root": str(tmp_path), "path": artifact.name, "bytes": artifact.stat().st_size,
              "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(), "kind": "best", "score": {"psnr": 30},
              "iteration": 4, "count": 8, "sh_degree": 3,
              "presentation": {"schema": 1, "source_frame": "blender-world", "data_to_source": geometry.FLIP,
                               "model_to_data": [[2, 0, 0, 3], [0, 3, 0, 4], [0, 0, 4, 5], [0, 0, 0, 1]]}}
    monkeypatch.setattr(transport, "registry_root", lambda: tmp_path / "sessions")
    producer = transport.SessionPublisher({"label": "Result trainer", "state": "training"})
    try:
        producer.publish(arrays(2), 7)
        record = {**transport.read_json(producer.root / "session.json"), "directory": str(producer.root)}
        _ui_editor(camera_runtime, f'panel._current = ExternalTraining({record!r})\ncontrols = panel.ExternalTrainingPanel()')
        _editor(camera_runtime, '''
assert wait_until(lambda: panel._current.shown, 10)
''')
        producer.finish("completed", result=result)
        _editor(camera_runtime, '''
assert wait_until(panel._current.stop.is_set, 10)
assert panel._current.status["result"]["count"] == 8
''')
        _ui_editor(camera_runtime, '''
valid_status = panel._current.status
panel._current.status = {**valid_status, "result": {**valid_status["result"], "count": "invalid"}}
controls.on_update(None)
assert controls._saved_result() == lf.ui.tr("external_training.result_invalid")
assert not controls._can_import_result()
panel._current.status = valid_status
controls.on_update(None)
assert controls._can_import_result()
''')
        # Integrity failure must not modify the preview or any existing model.
        content = artifact.read_bytes()
        artifact.write_bytes(content[:-1] + bytes([content[-1] ^ 1]))
        _ui_editor(camera_runtime, 'controls._import_result()')
        _editor(camera_runtime, '''
assert wait_until(controls._result_future.done, 10)
''')
        _ui_editor(camera_runtime, '''
controls.on_update(None)
assert controls._message() == lf.ui.tr("external_training.result_invalid"), controls._message()
assert lf.get_scene().get_node("Existing local model").gaussian_count == 8
assert lf.get_scene().get_node_by_uuid(panel._current.uuid).get("visible")
''')
        artifact.write_bytes(content)
        # Scene/import failures must not be reported as a corrupt saved file.
        _ui_editor(camera_runtime, 'controls._import_result()')
        _editor(camera_runtime, 'assert wait_until(controls._result_future.done, 10)')
        _ui_editor(camera_runtime, '''
from unittest.mock import patch
with patch.object(panel.lf, "load_gallery_scene", side_effect=ValueError("Scene changed")):
    controls.on_update(None)
assert controls._message() == lf.ui.tr("progress.import_failed_title"), controls._message()
assert lf.get_scene().get_node_by_uuid(panel._current.uuid).get("visible")
''')
        _ui_editor(camera_runtime, 'controls._import_result()')
        _editor(camera_runtime, '''
assert wait_until(controls._result_future.done, 10)
''')
        _ui_editor(camera_runtime, '''
controls.on_update(None)
assert not controls._message(), controls._message()
from types import SimpleNamespace
controls.on_unmount(SimpleNamespace(remove_data_model=lambda name: None))
''')
        _editor(camera_runtime, '''
assert wait_until(lambda: lf.ui.get_import_state().get("success") or lf.ui.get_import_state().get("error"), 20)
assert lf.ui.get_import_state()["success"], lf.ui.get_import_state()
assert lf.ui.get_import_state()["path"] == "saved-full.ply"
''')
        _ui_editor(camera_runtime, '''
controls.on_update(None)
scene = lf.get_scene()
assert not scene.get_node_by_uuid(panel._current.uuid).get("visible")
local = next(node for node in scene.nodes if node.name.startswith("Result trainer") and node.type == lf.scene.NodeType.SPLAT and not node.read_only)
assert not local.read_only
assert local.gaussian_count == 8
assert local.splat_data().active_sh_degree == 3
np.testing.assert_allclose(local.splat_data().sh0_raw.cpu().numpy(), saved_sh0, rtol=1e-6)
np.testing.assert_allclose(local.splat_data().shN_raw.cpu().numpy(), saved_shN, rtol=1e-6, atol=1e-7)
np.testing.assert_allclose(local.local_transform, [[2,0,0,3],[0,0,4,5],[0,-3,0,-4],[0,0,0,1]], atol=1e-6)
assert scene.get_node("Existing local model").gaussian_count == 8
imported_name = local.name
''')
        # Reimporting one iteration must add a different group, not replace it.
        _ui_editor(camera_runtime, 'controls._import_result()')
        _editor(camera_runtime, 'assert wait_until(controls._result_future.done, 10)')
        _ui_editor(camera_runtime, 'controls.on_update(None)')
        _editor(camera_runtime, 'assert wait_until(lambda: not lf.ui.get_import_state()["active"], 20)')
        _ui_editor(camera_runtime, '''
controls.on_update(None)
copies = [node for node in scene.nodes if node.name.startswith("Result trainer") and node.type == lf.scene.NodeType.SPLAT and not node.read_only]
assert len(copies) == 2
assert len({node.parent_id for node in copies}) == 2
''')
        artifact.unlink()
        _ui_editor(camera_runtime, 'controls._import_result()')
        _editor(camera_runtime, '''
assert wait_until(controls._result_future.done, 10)
''')
        _ui_editor(camera_runtime, '''
controls.on_update(None)
assert controls._message() == lf.ui.tr("external_training.result_unreadable"), controls._message()
assert lf.get_scene().get_node(imported_name).gaussian_count == 8
controls._remove()
assert not controls._message()
assert not any(node.read_only for node in lf.get_scene().nodes)
assert lf.get_scene().get_node(imported_name).gaussian_count == 8
''')
    finally:
        producer.finish()
        producer.pool.close()


def test_external_preview_reads_session_and_late_cameras_without_edit_access(camera_runtime_with_numpy, numpy, tmp_path, monkeypatch, request):
    camera_runtime, np = camera_runtime_with_numpy, numpy
    from lichtfeld_sessions import geometry, transport
    from test_training_sessions import arrays

    monkeypatch.setattr(transport, "registry_root", lambda: tmp_path / "sessions")
    producer = transport.SessionPublisher({"label": "Independent trainer", "state": "training", "iterations": 100})
    try:
        snapshot = producer.publish(arrays(32), 7)
        record = {**transport.read_json(producer.root / "session.json"), "directory": str(producer.root)}
        output = _ui_editor(camera_runtime, '''
import lfs_plugins.external_training_panel as panel
from lfs_plugins.external_training import ExternalTraining
import lichtfeld_sessions.transport as session_transport
from _lfs_path_io import Path
session_transport.registry_root = lambda: Path(TEST_REGISTRY)
panel._current = ExternalTraining(RECORD)
assert any(tab.id == panel.ExternalTrainingPanel.id for tab in lf.ui.get_main_panel_tabs())
lf.ui.set_main_panel_active_tab(panel.ExternalTrainingPanel.id)
print("attached")
'''.replace("RECORD", repr(record)).replace("TEST_REGISTRY", repr(str(producer.root.parent))))
        assert "attached" in output
        _editor(camera_runtime, '''
assert wait_until(lambda: panel._current.shown, 20), panel._current.error
''')

        # A recovered status read must clear only the connection error, even
        # when no new snapshot arrives to trigger another installation.
        _editor(camera_runtime, '''
import lfs_plugins.external_training as observer
import time
original_status = observer.SessionClient.status
status_reads = 0
def status_with_one_failure(client):
    global status_reads
    status_reads += 1
    return {} if status_reads == 1 else original_status(client)
observer.SessionClient.status = status_with_one_failure
panel._current.error = "Unrelated import error"
try:
    assert wait_until(lambda: status_reads >= 3)
    assert not panel._current.connection_error
    assert panel._current.error == "Unrelated import error"
    assert panel._current.generation == SNAPSHOT_ID
finally:
    observer.SessionClient.status = original_status
'''.replace("SNAPSHOT_ID", repr(snapshot["id"])))

        # New calibration must install without requiring a new Gaussian generation.
        pose = np.eye(4)
        pose[:3, :3] = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
        pose[:3, 3] = [1, 2, 3]
        model_transform = np.eye(4)
        model_transform[0, 3] = 4
        value = {"schema": 1, "source_frame": "blender-world", "data_to_source": geometry.FLIP,
                 "model_to_data": model_transform.tolist(), "cameras": [{
                     "name": "camera", "camera_to_data": pose.tolist(), "fx": 600, "fy": 610,
                     "cx": 310, "cy": 235, "width": 640, "height": 480}]}
        geometry.publish(producer, value)
        revision = transport.read_json(producer.root / "session.json")["geometry"]["revision"]
        _editor(camera_runtime, '''
assert wait_until(lambda: panel._current.geometry_revision == EXPECTED_REVISION, 20), (
    panel._current.geometry_revision, panel._current.error, panel._current.stop.is_set(),
    panel._current.geometry_error)
'''.replace("EXPECTED_REVISION", repr(revision)))
        output = _ui_editor(camera_runtime, '''
import lichtfeld as lf, numpy as np
import lfs_plugins.external_training_panel as panel
from lfs_plugins.external_training import presentation_transform
viewer = panel._current
scene = lf.get_scene()
root = scene.get_node_by_uuid(viewer.uuid)
model = next(scene.get_node_by_id(i) for i in root.children if scene.get_node_by_id(i).type == lf.scene.NodeType.SPLAT)
assert viewer.generation == SNAPSHOT_ID
assert viewer.shown["iteration"] == 7
assert viewer.error == "Unrelated import error"  # Camera success cannot hide a failed splat import.
assert model.gaussian_count == 32 and model.read_only and root.read_only
assert model.splat_data() is None and scene.combined_model() is None
expected = presentation_transform(GEOMETRY, np) @ np.asarray(GEOMETRY["model_to_data"])
np.testing.assert_allclose(np.asarray(model.local_transform), expected, atol=1e-6)
camera = next(node for node in scene.nodes if node.type == lf.scene.NodeType.CAMERA)
assert camera.read_only and camera.camera_center_x == 310 and camera.camera_center_y == 235
view = np.eye(4)
view[:3, :3] = camera.camera_R.cpu().numpy()
view[:3, 3] = camera.camera_T.cpu().numpy().reshape(3)
expected_pose = presentation_transform(GEOMETRY, np) @ np.asarray(GEOMETRY["cameras"][0]["camera_to_data"])
np.testing.assert_allclose(np.asarray(camera.world_transform) @ np.linalg.inv(view), expected_pose, atol=1e-6)
returned = camera.camera_R
returned[0, 0] = 123
assert camera.camera_R.cpu().numpy()[0, 0] != 123  # No mutable alias to the live calibration.
for node in (root, model, camera):
    for prop, value in (("locked", False), ("name", "changed"), ("training_enabled", True)):
        try:
            node.set(prop, value)
        except RuntimeError:
            pass
        else:
            raise AssertionError((prop, node.name))
try:
    scene.set_node_transient(root.name, False)
except RuntimeError as error:
    assert "External preview is transient" in str(error)
else:
    raise AssertionError("A streamed preview became persistent")
print("verified")
'''.replace("SNAPSHOT_ID", repr(snapshot["id"])).replace("GEOMETRY", repr(value)))
        assert "verified" in output
        _ui_editor(camera_runtime, '''
camera_uuid, camera_uid = camera.uuid, camera.camera_uid
camera.set("visible", False)
''')
        for camera_count in (2, 1):
            geometry.publish(producer, {**value, "cameras": value["cameras"] * camera_count})
            revision = transport.read_json(producer.root / "session.json")["geometry"]["revision"]
            _editor(camera_runtime, '''
import time
assert wait_until(lambda: panel._current.geometry_revision == REVISION), panel._current.geometry_error
'''.replace("REVISION", repr(revision)))
            _ui_editor(camera_runtime, '''
scene = lf.get_scene()
camera = scene.get_node_by_uuid(camera_uuid)
assert camera.camera_uid == camera_uid and not camera.get("visible")
assert len(scene.get_nodes(lf.scene.NodeType.CAMERA)) == CAMERA_COUNT
'''.replace("CAMERA_COUNT", str(camera_count)))
        # A changing foreign point count must not move the user's local selection.
        _ui_editor(camera_runtime, '''
scene = lf.get_scene()
z = lambda shape: lf.Tensor.zeros(shape, device="cpu", dtype="float32")
rotation = z([2, 4])
rotation[0, 0] = rotation[1, 0] = 1
scene.add_splat("Local model", z([2, 3]), z([2, 1, 3]), z([2, 0, 3]), z([2, 3]), rotation, z([2, 1]), sh_degree=0)
assert scene.combined_model().num_points == 34
scene.get_node_by_uuid(panel._current.uuid).set("visible", False)
assert scene.combined_model().num_points == 2
scene.get_node_by_uuid(panel._current.uuid).set("visible", True)
assert scene.combined_model().num_points == 34
''')
        _ui_editor(camera_runtime, '''
scene = lf.get_scene()
scene.set_selection([32])
assert scene.selection_mask is not None, (scene.total_gaussian_count, scene.active_selection_group)
assert scene.selection_mask.cpu().tolist() == [0] * 32 + [1, 0]
''')
        next_snapshot = producer.publish(arrays(33), 8)
        output = _editor(camera_runtime, '''
import time
assert wait_until(lambda: panel._current.generation == NEXT_ID), panel._current.error
assert not panel._current.error
'''.replace("NEXT_ID", repr(next_snapshot["id"])))
        _ui_editor(camera_runtime, '''
scene = lf.get_scene()
assert scene.selection_mask.cpu().tolist() == [0] * 33 + [1, 0]
lf.select_node(scene.get_node_by_uuid(panel._current.uuid).name)
lf.ui.set_main_panel_active_tab(next(tab.id for tab in lf.ui.get_main_panel_tabs() if tab.id != panel.ExternalTrainingPanel.id))
''')
        # Persistence excludes the foreign subtree even when it is selected.
        project = tmp_path / "local-with-observer.licht"
        saved = _tool(camera_runtime, "project_save_as", {"path": str(project)})
        assert saved.get("success", True), saved
        opened = _tool(camera_runtime, "project_open", {"path": str(project), "discard_changes": True})
        assert opened.get("success", True), opened
        _editor(camera_runtime, '''
assert wait_until(lambda: lf.get_scene().get_node("Local model").gaussian_count == 2)
''')
        output = _ui_editor(camera_runtime, '''
import lfs_plugins.external_training_panel as panel
assert panel._current.stop.is_set()
scene = lf.get_scene()
assert scene.get_node("Local model").gaussian_count == 2
assert not any(node.read_only for node in scene.nodes)
target = scene.create_external_preview("Cancelled upload", "Preview")
scene = lf.get_scene()
cancelled = lf.scene.prepare_external_cameras(lf.Tensor.zeros([0, 22], device="cpu"), np.eye(4, dtype=np.float32))
assert scene.upload_external_preview(cancelled)
print("restored")
''')
        assert "restored" in output
        _editor(camera_runtime, '''
assert wait_until(lambda: cancelled.ready)
''')
        _ui_editor(camera_runtime, '''
scene = lf.get_scene()
scene.cancel_external_preview_upload()
assert not scene.apply_external_preview(target, cancelled, "Cameras")
scene.remove_external_preview(target)
''')
        assert transport.read_json(producer.root / "session.json")["state"] == "training"
        assert wait_until(lambda: not list((producer.root / "clients").glob("*.json")))
        # Normal application exit must drain an active reader, not require the
        # fixture's forced cleanup or stop the trainer that owns the session.
        _ui_editor(camera_runtime, '''
panel._current = ExternalTraining(RECORD)
'''.replace("RECORD", repr(record)))
        assert request.node.camera_process.poll() is None
        _editor(camera_runtime, '''
assert wait_until(lambda: panel._current.shown is not None), panel._current.error
threading.Timer(0.2, lambda: lf.ui.schedule_on_ui_thread(lf.request_exit)).start()
''')
        assert request.node.camera_process.wait(timeout=10) == 0
        assert not list((producer.root / "clients").glob("*.json"))
        assert transport.read_json(producer.root / "session.json")["state"] == "training"
    finally:
        producer.finish("stopped")
        producer.thread.join(3)


@pytest.mark.parametrize("ordinary_storage", [False, True], ids=["viewer", "ordinary-cuda"])
def test_remove_preview_with_collision_after_undo(camera_runtime_with_numpy, tmp_path, monkeypatch, numpy, request, ordinary_storage):
    from lichtfeld_sessions import transport
    from test_selection_outline_runtime import _write_splat
    from test_training_sessions import arrays

    endpoint = camera_runtime_with_numpy
    source = tmp_path / "local.ply"
    _write_splat(source, 0., (.5, .5, .15), dense=True)
    monkeypatch.setattr(transport, "registry_root", lambda: tmp_path / "sessions")
    producer = transport.SessionPublisher({"label": "Trainer", "state": "training"})
    try:
        data = arrays(100_000)
        data["means"][:] = numpy.random.default_rng(42).uniform(-.5, .5, data["means"].shape)
        data["scale"][:] = -5
        producer.publish(data, 1)
        record = {**transport.read_json(producer.root / "session.json"), "directory": str(producer.root)}
        _ui_editor(endpoint, f"""
import time, lfs_plugins.external_training_panel as panel
settings = lf.get_render_settings()
settings.raster_backend = 2
settings.point_cloud_mode = settings.lod_enabled = False
lf.set_camera((0., 0., 4.), (0., 0., 0.))
panel._current = panel.ExternalTraining({record!r})
target = panel._current.uuid
""")
        _editor(endpoint, """
assert wait_until(lambda: panel._current.shown, 15), panel._current.error
""")
        _ui_editor(endpoint, f"lf.load_file({str(source)!r})")
        _editor(endpoint, """
assert wait_until(lambda: lf.get_scene().get_node("local"), 15)
""")
        if ordinary_storage:
            _ui_editor(endpoint, """
scene = lf.get_scene()
model = scene.get_node("local").splat_data()
count = model.num_points
fields = (model.means_raw.clone(), model.sh0_raw.reshape([count, 1, 3]).clone(),
          lf.Tensor.zeros([count, 0, 3], device="cuda"), model.scaling_raw.clone(),
          model.rotation_raw.clone(), model.opacity_raw.reshape([count, 1]).clone())
scene.remove_node("local")
scene.add_splat("local", *fields, sh_degree=0)
""")
        _ui_editor(endpoint, 'lf.select_node("local")\nlf.generate_collision(.03, .1, False, False, 0.)')
        _editor(endpoint, """
assert wait_until(lambda: lf.get_scene().get_nodes(lf.scene.NodeType.COLLISION), 15)
assert lf.capture_viewport() is not None
""")
        assert _tool(endpoint, "gaussians_write", {
            "node": "local", "field": "opacities", "indices": [0], "values": [1.],
        })["success"]
        _ui_editor(endpoint, "assert lf.undo.undo()")
        _ui_editor(endpoint, """
assert lf.get_scene().get_node_by_uuid(target)
panel.ExternalTrainingPanel()._remove()
assert lf.get_scene().get_node_by_uuid(target) is None
assert lf.get_scene().get_node("local") and lf.get_scene().get_nodes(lf.scene.NodeType.COLLISION)
""")
        # Removing the preview after undo must preserve the local model,
        # collision rendering and frame presentation with either storage type.
        start = _ledger(endpoint)["frames_presented"]
        app_log = tmp_path / "app.log"
        def presented_frame():
            _ui_editor(endpoint, "lf.ui.request_redraw()")
            return _ledger(endpoint)["frames_presented"] >= start + 4
        assert wait_until(presented_frame, 15), "Viewport did not present frames after removing the preview"
        assert _tool(endpoint, "project_save_as", {"path": str(tmp_path / "local.licht")})["success"]
        _editor(endpoint, "threading.Timer(.2, lambda: lf.ui.schedule_on_ui_thread(lf.request_exit)).start()")
        assert request.node.camera_process.wait(timeout=10) == 0
        assert "VK_ERROR_DEVICE_LOST" not in app_log.read_text(encoding="utf-8", errors="replace")
    finally:
        producer.finish("stopped")
        producer.thread.join(3)


def test_add_splat_rejects_mismatched_inputs(camera_runtime):
    output = _editor(camera_runtime, '''
import lichtfeld as lf

scene = lf.get_scene()
z = lambda shape: lf.Tensor.zeros(shape, device="cpu", dtype="float32")
n = 4
valid = {"means": [n, 3], "sh0": [n, 1, 3], "shN": [n, 0, 3], "scaling": [n, 3], "rotation": [n, 4],
         "opacity": [n, 1]}
broken = {
    "means": [[n, 2], [n], [n, 3, 1]],
    "sh0": [[n - 1, 1, 3], [n, 3]],
    "shN": [[n + 1, 0, 3], [n, 3]],
    "scaling": [[n, 2]],
    "rotation": [[n, 3], [2, 4]],
    "opacity": [[n, 2], [n - 2, 1]],
}
before = scene.node_count
for field, shapes in broken.items():
    for shape in shapes:
        args = {name: z(dims) for name, dims in valid.items()}
        args[field] = z(shape)
        try:
            scene.add_splat("broken", args["means"], args["sh0"], args["shN"], args["scaling"],
                            args["rotation"], args["opacity"], sh_degree=0)
        except RuntimeError as error:
            assert field in str(error), (field, shape, str(error))
        else:
            raise AssertionError(f"add_splat accepted {field} {shape}")
assert scene.node_count == before

rotation = z([n, 4])
for row in range(n):
    rotation[row, 0] = 1.0
scene.add_splat("valid", z([n, 3]), z([n, 1, 3]), z([n, 0, 3]), z([n, 3]), rotation, z([n, 1]), sh_degree=0)
print("added", scene.get_node("valid").gaussian_count)
''')
    assert "added 4" in output
    _ui_editor(camera_runtime, '''
scene = lf.get_scene()
assert lf.project_is_dirty()  # The persistent local model still needs saving.
preview = scene.create_external_preview("Empty trainer", "Preview")
scene = lf.get_scene()
assert scene.combined_model().num_points == 4
scene.get_node_by_uuid(preview).set("visible", False)
assert lf.get_scene().combined_model().num_points == 4
lf.get_scene().remove_external_preview(preview)
assert lf.get_scene().get_node_by_uuid(preview) is None
''')


@pytest.mark.slow
def test_external_preview_bulk_removal_preserves_another_ready_upload(camera_runtime_with_numpy):
    camera_runtime = camera_runtime_with_numpy
    _editor(camera_runtime, '''
import time, lichtfeld as lf
from lichtfeld_sessions import ensure_numpy
np = ensure_numpy(lf)
rows = np.tile(np.r_[np.eye(4).reshape(-1), [500, 500, 320, 240, 640, 480]], (1000, 1)).astype(np.float32)
update = lf.scene.prepare_external_cameras(lf.Tensor.from_numpy(rows), np.eye(4, dtype=np.float32))
''')
    _ui_editor(camera_runtime, '''
target = lf.get_scene().create_external_preview("Many cameras", "Preview")
assert lf.get_scene().upload_external_preview(update)
''')
    _editor(camera_runtime, '''
assert wait_until(lambda: update.ready, 20)
replacement = lf.scene.prepare_external_cameras(lf.Tensor.zeros([0, 22], device="cpu"), np.eye(4, dtype=np.float32))
''')
    _ui_editor(camera_runtime, '''
assert lf.get_scene().apply_external_preview(target, update, "Cameras")
assert len(lf.get_scene().get_nodes(lf.scene.NodeType.CAMERA)) == 1000
other = lf.get_scene().create_external_preview("Next trainer", "Preview")
assert lf.get_scene().upload_external_preview(replacement)
''')
    _editor(camera_runtime, '''
assert wait_until(lambda: replacement.ready)
''')
    output = _ui_editor(camera_runtime, '''
started = time.perf_counter()
lf.get_scene().remove_external_preview(target)
print("camera_removal_ms", (time.perf_counter() - started) * 1000)
assert lf.get_scene().get_node_by_uuid(target) is None
assert not lf.get_scene().get_nodes(lf.scene.NodeType.CAMERA)
assert lf.get_scene().apply_external_preview(other, replacement, "Cameras")
lf.get_scene().remove_external_preview(other)
del update, replacement
''')
    assert "camera_removal_ms" in output
    print(output.strip())
    # Each retirement and following upload joins the previous worker from a
    # Python UI callback with the GIL held.
    for cycle in range(32):
        _ui_editor(camera_runtime, '''
target = lf.get_scene().create_external_preview("Repeated trainer", "Preview")
update = lf.scene.prepare_external_cameras(lf.Tensor.from_numpy(rows[:1]), np.eye(4, dtype=np.float32))
assert wait_until(lambda: lf.get_scene().upload_external_preview(update))
assert wait_until(lambda: update.ready)
assert lf.get_scene().apply_external_preview(target, update, "Cameras")
del update
lf.get_scene().remove_external_preview(target)
assert lf.get_scene().get_node_by_uuid(target) is None
''')


@pytest.mark.parametrize("query", ["list()", "is_installed('numpy')"])
def test_external_preview_exit_cancels_dependency_download(camera_runtime, request, query):
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    requested, release = threading.Event(), threading.Event()

    class StalledDownload(BaseHTTPRequestHandler):
        def do_GET(self):
            requested.set()
            release.wait(60)

        do_HEAD = do_GET

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), StalledDownload)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        _editor(camera_runtime, '''
import os, lichtfeld as lf
import lfs_plugins.external_training as observer
lf.packages.init()
os.environ.pop("UV_OFFLINE", None)
os.environ["UV_CACHE_DIR"] = os.path.join(os.environ["LFS_HOME"], "download-cache")
def stalled_dependency(lf, should_cancel):
    lf.packages.install(PACKAGE, cancel=should_cancel)
observer.ensure_numpy = stalled_dependency
'''.replace("PACKAGE", repr(f"observer-cancel-test @ http://127.0.0.1:{server.server_port}/observer_cancel_test-1.0-py3-none-any.whl")))
        _ui_editor(camera_runtime, '''
connection = observer.ExternalTraining({"id": "dependency-test", "label": "Dependency test"})
''')
        assert requested.wait(10), _editor(camera_runtime, 'print(connection.error)')
        _editor(camera_runtime, '''
query_started = threading.Event()
def query_packages():
    query_started.set()
    lf.packages.QUERY
threading.Thread(target=query_packages, daemon=True).start()
assert query_started.wait(5)
'''.replace("QUERY", query))
        _ui_editor(camera_runtime, '''
assert not lf.project_is_dirty(), "The transient preview dirtied the document"
assert connection.preparing_dependencies
''')
        _editor(camera_runtime, '''
threading.Timer(0.2, lambda: lf.ui.schedule_on_ui_thread(lf.request_exit)).start()
''')
        assert request.node.camera_process.wait(timeout=10) == 0
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        worker.join(5)



def test_external_preview_recovers_failures_and_releases_failed_final_upload(camera_runtime_with_numpy, tmp_path, monkeypatch):
    endpoint = camera_runtime_with_numpy
    from lichtfeld_sessions import transport
    from test_training_sessions import arrays

    monkeypatch.setattr(transport, "registry_root", lambda: tmp_path / "sessions")
    producer = transport.SessionPublisher({"label": "Recovery", "state": "training"})
    try:
        first = producer.publish(arrays(8), 1)
        record = {**transport.read_json(producer.root / "session.json"), "directory": str(producer.root)}
        _editor(endpoint, '''
import time
import lfs_plugins.external_training as observer
original_numpy = observer.ensure_numpy
dependency_attempts = 0
def unavailable_once(*args):
    global dependency_attempts
    dependency_attempts += 1
    if dependency_attempts == 1:
        raise RuntimeError("Dependency download unavailable")
    return original_numpy(*args)
observer.ensure_numpy = unavailable_once
''')
        _ui_editor(endpoint, "connection = observer.ExternalTraining(RECORD)".replace("RECORD", repr(record)))
        _editor(endpoint, '''
assert wait_until(lambda: connection.error)
assert connection.preparing_dependencies
assert "Dependency download unavailable" in connection.error
assert wait_until(lambda: connection.generation == FIRST, 10), connection.error
assert dependency_attempts == 2
assert not connection.preparing_dependencies
assert not connection.error
observer.ensure_numpy = original_numpy
original_scene = observer.ExternalTraining.scene
failed_installs = []
reject_upload = True
class RejectingScene:
    def __init__(self, real):
        self.real = real
    def __getattr__(self, name):
        return getattr(self.real, name)
    def apply_external_preview(self, *args):
        if reject_upload:
            failed_installs.append(time.monotonic())
            raise RuntimeError("Injected renderer allocation failure")
        return self.real.apply_external_preview(*args)
observer.ExternalTraining.scene = property(lambda self: RejectingScene(original_scene.fget(self)))
'''.replace("FIRST", repr(first["id"])))
        second = producer.publish(arrays(9), 2)
        _editor(endpoint, '''
assert wait_until(lambda: len(failed_installs) >= 2, 8), failed_installs
assert len(failed_installs) == 2, failed_installs
assert failed_installs[1] - failed_installs[0] >= 1.9, failed_installs
assert connection.generation == FIRST
assert connection.shown["count"] == 8
assert not connection.stop.is_set()
reject_upload = False
assert wait_until(lambda: connection.generation == SECOND, 8), connection.error
assert not connection.error
reject_upload = True
'''.replace("FIRST", repr(first["id"])).replace("SECOND", repr(second["id"])))
        producer.publish(arrays(10), 3)
        producer.finish("completed")
        _editor(endpoint, '''
try:
    assert wait_until(connection.stop.is_set, 8), connection.error
    assert connection.generation == SECOND
    assert connection.shown["count"] == 9
    assert "Injected renderer allocation failure" in connection.error
finally:
    observer.ExternalTraining.scene = original_scene
    observer.ensure_numpy = original_numpy
'''.replace("SECOND", repr(second["id"])))
        assert wait_until(lambda: not list((producer.root / "clients").glob("*.json")))
        _ui_editor(endpoint, "connection.close(remove=True)")
    finally:
        producer.finish("completed")


def test_scene_center_and_deleted_mask_without_dataset(camera_runtime):
    output = _editor(camera_runtime, '''
import lichtfeld as lf

scene = lf.get_scene()
assert not scene.has_training_data()
assert scene.scene_center is None, scene.scene_center

z = lambda shape: lf.Tensor.zeros(shape, device="cpu", dtype="float32")
rotation = z([3, 4])
for row in range(3):
    rotation[row, 0] = 1.0
scene.add_splat("masks", z([3, 3]), z([3, 1, 3]), z([3, 0, 3]), z([3, 3]), rotation, z([3, 1]), sh_degree=0)
splats = scene.get_node("masks").splat_data()
assert not splats.has_deleted_mask()
deleted = splats.deleted
assert list(deleted.shape) == [3], deleted
assert deleted.dtype == "bool", deleted.dtype
assert deleted.tolist() == [False, False, False], deleted.tolist()
print("checked")
''')
    assert "checked" in output

# SPDX-License-Identifier: GPL-3.0-or-later
"""Run on a disposable GUI editor worker: camera IDs, chunking and owned cleanup."""
import json
import os
from pathlib import Path
import threading
import time
import lichtfeld as lf
from lfs_plugins.training_runtime.viewer import Viewer
from lfs_plugins.training_runtime.producer import numerical_runtime
from lfs_plugins.training_runtime.geometry import viewer_model_transform

np = numerical_runtime()


def ui(action):
    ready = threading.Event()
    errors = []
    result = []
    def run():
        try:
            result.append(action())
        except Exception as error:
            errors.append(repr(error))
        finally:
            ready.set()
    lf.ui.schedule_on_ui_thread(run)
    lf.ui.request_redraw()
    assert ready.wait(20), 'UI callback timed out'
    assert not errors, errors
    return result[0] if result else None


count = 512
rows = np.zeros((count, 22), np.float32)
rows[:, :16] = np.eye(4, dtype=np.float32).reshape(-1)
rows[:, 3] = np.linspace(-5, 5, count)
rows[:, 11] = 5
rows[:, 16:] = [500, 510, 310, 235, 640, 480]
prepared = lf.scene.prepare_viewer_cameras(rows)
transform = viewer_model_transform({'source_frame':'blender-world'}).astype(np.float32)
model = lf.scene.prepare_viewer_splat(np.zeros((1,3),np.float32),np.zeros((1,1,3),np.float32),
    np.zeros((1,0,3),np.float32),np.full((1,3),-2,np.float32),np.array([[1,0,0,0]],np.float32),
    np.ones((1,1),np.float32),0,transform)
R = lf.Tensor.from_numpy(np.eye(3, dtype=np.float32))
T = lf.Tensor.from_numpy(np.zeros(3, dtype=np.float32))
name = 'Camera churn preview'
def setup():
    scene = lf.get_scene()
    scene.clear()
    scene.add_group('User sentinel')
    group = scene.add_camera_group('User cameras', -1, 2)
    scene.add_camera('Explicit high UID', group, R, T, 500, 500, 640, 480, uid=100000)
    scene = lf.get_scene()  # Earlier setup writes deliberately advanced the scene generation.
    scene.create_external_viewer(name)
    scene = lf.get_scene()
    scene.install_prepared_viewer(name, model)
    return str(scene.get_node(name).uuid)
viewer = Viewer.__new__(Viewer)
viewer.name = name
viewer.node_uuid = ui(setup)
viewer.stop = threading.Event()
viewer.camera_epoch = 0
viewer.cameras_visible = True
viewer.camera_error = ''
viewer.camera_count = 0
viewer.camera_ref = None
timings = []
schedule = lf.ui.schedule_on_ui_thread
def timed_schedule(callback):
    if callback.__name__ == 'chunk':
        def measured():
            started = time.perf_counter()
            try:
                callback()
            finally:
                timings.append((time.perf_counter()-started)*1000)
        return schedule(measured)
    return schedule(callback)
lf.ui.schedule_on_ui_thread = timed_schedule
try:
    ui(lambda: viewer._install_cameras(('test', prepared, count)))
    # A local automatic UID allocation interleaves with the streamed batch.
    def add_local():
        scene = lf.get_scene()
        return scene.add_camera('Interleaved local camera', scene.get_node('User cameras').id,
                                R, T, 500, 500, 640, 480)
    ui(add_local)
    deadline = time.monotonic()+30
    while viewer.camera_count != count and time.monotonic()<deadline:
        assert not viewer.camera_error, viewer.camera_error
        time.sleep(.02)
    assert viewer.camera_count == count, (viewer.camera_count, viewer.camera_error)
    def verify():
        scene = lf.get_scene()
        cameras = [n for n in scene.nodes if n.has_camera]
        assert len(cameras) == count+2
        assert len({n.camera_uid for n in cameras}) == count+2
        assert min(n.camera_uid for n in cameras) >= 100000
        assert len(scene.get_active_cameras()) == 2
        group = scene.get_node(name+' cameras')
        assert group is not None
        root = scene.get_node(name+' session')
        anchor = scene.get_node(name)
        assert root is not None and root.parent_id == -1
        assert anchor.parent_id == root.id and group.parent_id == root.id
        assert len(root.children) == 2
        np.testing.assert_allclose(np.asarray(root.world_transform).reshape(4,4), np.eye(4))
        np.testing.assert_allclose(np.asarray(anchor.world_transform).reshape(4,4), transform)
        assert anchor.gaussian_count == 1
        for target in (root, group):
            assert scene.add_group('Blocked local child', target.id) == -1
        external_camera = next(camera for camera in cameras if camera.parent_id == group.id)
        before_R = external_camera.camera_R.numpy()
        before_T = external_camera.camera_T.numpy()
        external_camera.camera_R.numpy(copy=False)[:] = 0
        external_camera.camera_T.numpy(copy=False)[:] = 123
        np.testing.assert_allclose(external_camera.camera_R.numpy(), before_R)
        np.testing.assert_allclose(external_camera.camera_T.numpy(), before_T)
        try:
            scene.set_camera_training_enabled(external_camera.name, True)
        except RuntimeError:
            pass
        else:
            raise AssertionError('A protected camera entered training')
        assert len(lf.get_scene().get_active_cameras()) == 2
        for camera in cameras:
            if camera.parent_id == group.id:
                assert camera.camera_center_x == 310 and camera.camera_center_y == 235
    ui(verify)
    def remove_one():
        removed, payload = lf.get_scene().remove_external_camera(viewer.node_uuid, name)
        return removed, payload
    removed = 0
    while True:
        changed, payload = ui(remove_one)
        payload.release()  # Editor worker owns retirement, never the UI callback.
        if not changed:
            break
        removed += 1
    assert removed == count+1
    def verify_cleanup():
        scene = lf.get_scene()
        assert scene.get_node('User sentinel') is not None
        assert len([n for n in scene.nodes if n.has_camera]) == 2
        assert scene.get_node(name+' cameras') is None
        return scene.remove_external_viewer(name, viewer.node_uuid)
    retired = ui(verify_cleanup)
    retired.release()
    assert ui(lambda: lf.get_scene().get_node(name+' session') is None)

    # Discard removes the splat before the incremental camera cleanup finishes.
    # The final camera-group removal must also retire the empty trainer root.
    second = lf.scene.prepare_viewer_cameras(rows[:3].copy())
    def setup_second():
        scene = lf.get_scene()
        scene.create_external_viewer(name)
        scene = lf.get_scene()
        uuid = str(scene.get_node(name).uuid)
        for index in range(3): lf.get_scene().install_external_camera(name, uuid, second, index)
        return uuid, lf.get_scene().remove_external_viewer(name, uuid)
    second_uuid, retired = ui(setup_second)
    retired.release()
    while True:
        changed, payload = ui(lambda: lf.get_scene().remove_external_camera(second_uuid, name))
        payload.release()
        if not changed: break
    second.release()
    assert ui(lambda: lf.get_scene().get_node(name+' session') is None)

    def verify_collision():
        scene = lf.get_scene()
        scene.add_group('Collision preview session')
        scene = lf.get_scene()
        uuid = str(scene.get_node('Collision preview session').uuid)
        try:
            scene.create_external_viewer('Collision preview')
        except RuntimeError:
            pass
        else:
            raise AssertionError('An unrelated local group was replaced')
        scene = lf.get_scene()
        assert str(scene.get_node('Collision preview session').uuid) == uuid
        assert scene.get_node('Collision preview') is None
    ui(verify_collision)
finally:
    lf.ui.schedule_on_ui_thread = schedule
    prepared.release()
    model.release()
proof = {'cameras':count, 'unique_uids':True, 'local_cameras_preserved':2,
         'ui_chunks':len(timings), 'max_ui_chunk_ms':max(timings, default=0), 'cleanup':True,
         'trainer_hierarchy':True, 'both_cleanup_orders':True, 'local_name_collision_safe':True}
(Path(os.environ['LFS_TEST_AREA'])/'camera-churn-proof.json').write_text(json.dumps(proof), encoding='utf-8')
print('CAMERA_CHURN_OWNERSHIP_AND_UIDS_PASS', json.dumps(proof), flush=True)

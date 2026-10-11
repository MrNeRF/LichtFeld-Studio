# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Point-cloud handles retain owned data and resolve live scene identity safely."""

import os
import time

import pytest

from test_api_camera_project_runtime import camera_runtime as _app_runtime, _editor
from test_render_on_demand_idle import _tool

pytestmark = [pytest.mark.gpu, pytest.mark.integration]


@pytest.fixture
def camera_runtime(_app_runtime, tmp_path):
    yield _app_runtime
    if os.environ.get("ASAN_OPTIONS"):
        time.sleep(.2)
        log = (tmp_path / "app.log").read_text(encoding="utf-8", errors="replace")
        if "ERROR: AddressSanitizer" in log:
            # Preserve the reporting process long enough to finish its stack trace.
            time.sleep(10)
            log = (tmp_path / "app.log").read_text(encoding="utf-8", errors="replace")
        assert "ERROR: AddressSanitizer" not in log, log


@pytest.mark.parametrize("reset", ["remove", "clear"])
def test_retained_point_cloud_rejects_removed_source(camera_runtime, reset):
    _editor(camera_runtime, f'''
import lichtfeld as lf
scene = lf.get_scene()
points = lf.Tensor.zeros([3, 3], device="cpu", dtype="float32")
colors = lf.Tensor.ones([3, 3], device="cpu", dtype="float32")
scene.add_point_cloud("retained cloud", points, colors)
node = scene.get_node("retained cloud")
cloud = node.point_cloud()
assert cloud.size == 3
saved_means = cloud.means
if {reset!r} == "remove":
    scene.remove_node("retained cloud")
else:
    scene.clear()
try:
    size = cloud.size
except RuntimeError as error:
    assert "PointCloud" in str(error)
else:
    raise AssertionError(f"stale PointCloud was accepted: {{size}}")
for attribute in ("means", "colors", "normals", "sh0", "shN", "opacity",
                  "scaling", "rotation", "attribute_names"):
    try:
        getattr(cloud, attribute)
    except RuntimeError as error:
        assert "PointCloud" in str(error)
    else:
        raise AssertionError(f"stale attribute was accepted: {{attribute}}")
for mutate in (
    lambda: cloud.is_gaussian(),
    lambda: cloud.set_means(points),
    lambda: cloud.set_colors(colors),
    lambda: cloud.set_data(points, colors),
    lambda: cloud.normalize_colors(),
    lambda: cloud.filter(lf.Tensor.ones([3], device="cpu", dtype="bool")),
    lambda: cloud.filter_indices(lf.Tensor.zeros([1], device="cpu", dtype="int32")),
):
    try:
        mutate()
    except RuntimeError as error:
        assert "PointCloud" in str(error)
    else:
        raise AssertionError("stale mutation was accepted")
assert saved_means.shape == (3, 3)
assert saved_means.cpu().tolist() == [[0., 0., 0.]] * 3
''')


def test_retained_point_cloud_follows_undo_and_live_updates(camera_runtime, tmp_path):
    _editor(camera_runtime, '''
import lichtfeld as lf
scene = lf.get_scene()
points = lf.Tensor.zeros([3, 3], device="cpu", dtype="float32")
colors = lf.Tensor.ones([3, 3], device="cpu", dtype="float32")
scene.add_point_cloud("undo cloud", points, colors)
node = scene.get_node("undo cloud")
uuid = node.uuid
cloud = node.point_cloud()
cloud.set_data(lf.Tensor.zeros([4, 3], device="cpu"), lf.Tensor.ones([4, 3], device="cpu"))
assert cloud.size == 4
assert node.gaussian_count == 4
scene.rename_node("undo cloud", "renamed cloud")
assert cloud.size == 4
scene.remove_node("renamed cloud")
assert lf.undo.undo()
restored = scene.get_node("renamed cloud")
assert restored.uuid == uuid
assert cloud.size == 4
cloud.set_means(lf.Tensor.ones([4, 3], device="cpu"))
assert restored.centroid == (1., 1., 1.)
''')
    exported = tmp_path / "restored.ply"
    _editor(camera_runtime, f"lf.io.save_point_cloud_ply(cloud, {str(exported)!r})")
    assert exported.is_file()
    _editor(camera_runtime, '''
assert lf.undo.redo()
try:
    cloud.size
except RuntimeError:
    pass
else:
    raise AssertionError("redo left a stale cloud accessible")
''')


def test_loaded_point_cloud_outlives_load_result(camera_runtime, tmp_path):
    import struct
    import zlib

    dataset = tmp_path / "dataset"
    sparse = dataset / "sparse" / "0"
    images = dataset / "images"
    sparse.mkdir(parents=True)
    images.mkdir()
    (sparse / "cameras.txt").write_text("1 PINHOLE 2 2 2 2 1 1\n")
    (sparse / "images.txt").write_text("1 1 0 0 0 0 0 0 1 image.png\n0 0 1 1 1 2\n")
    (sparse / "points3D.txt").write_text(
        "1 0 0 2 255 0 0 0 1 0\n2 1 1 3 0 255 0 0 1 1\n"
    )

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 2, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress((b"\x00" + b"\xff\x00\x00" * 2) * 2))
    png += chunk(b"IEND", b"")
    (images / "image.png").write_bytes(png)
    exported = tmp_path / "retained.ply"
    _editor(camera_runtime, f'''
import lichtfeld as lf, gc
result = lf.io.load({str(dataset)!r})
cloud = result.point_cloud
assert cloud is not None
assert cloud.size == 2
expected = cloud.means.cpu().tolist()
del result
gc.collect()
assert cloud.size == 2
assert cloud.means.cpu().tolist() == expected
lf.io.save_point_cloud_ply(cloud, {str(exported)!r})
''')
    assert exported.is_file()


def test_retained_point_cloud_rejects_different_project(camera_runtime, tmp_path):
    endpoint = camera_runtime
    _editor(endpoint, '''
import lichtfeld as lf
scene = lf.get_scene()
scene.add_point_cloud("same name", lf.Tensor.ones([2, 3], device="cpu"),
                      lf.Tensor.ones([2, 3], device="cpu"))
other_uuid = scene.get_node("same name").uuid
''')
    project = tmp_path / "other.licht"
    _tool(endpoint, "project_save_as", {"path": str(project)})
    _editor(endpoint, '''
scene.clear()
scene.add_point_cloud("same name", lf.Tensor.zeros([3, 3], device="cpu"),
                      lf.Tensor.ones([3, 3], device="cpu"))
old_cloud = scene.get_node("same name").point_cloud()
assert old_cloud.size == 3
assert scene.get_node("same name").uuid != other_uuid
''')
    _tool(endpoint, "project_open", {"path": str(project), "discard_changes": True})
    deadline = time.monotonic() + 20
    while True:
        # Opening is asynchronous; query the public Python scene on the editor thread.
        ready = _editor(endpoint, '''
node = lf.get_scene().get_node("same name")
print("PROJECT_READY" if node and node.uuid == other_uuid and node.point_cloud() is not None
      else "PROJECT_WAIT")
''')
        if "PROJECT_READY" in ready:
            break
        assert time.monotonic() < deadline, "project did not finish opening"
        time.sleep(.05)
    rejected_export = tmp_path / "stale.ply"
    _editor(endpoint, f'''
assert lf.get_scene().get_node("same name").point_cloud().size == 2
for access in (lambda: old_cloud.size,
               lambda: lf.io.save_point_cloud_ply(old_cloud, {str(rejected_export)!r})):
    try:
        access()
    except RuntimeError as error:
        assert "PointCloud" in str(error)
    else:
        raise AssertionError("old reference resolved a different project's cloud")
''')
    assert not rejected_export.exists()


def test_point_cloud_export_survives_source_removal(camera_runtime, tmp_path):
    exported = tmp_path / "concurrent.ply"
    _editor(camera_runtime, f'''
import lichtfeld as lf, threading, time, os
scene = lf.get_scene()
scene.add_point_cloud("export cloud", lf.Tensor.zeros([2000000, 3], device="cpu"),
                      lf.Tensor.ones([2000000, 3], device="cpu"))
cloud = scene.get_node("export cloud").point_cloud()
export_errors = []
def export_cloud():
    try:
        lf.io.save_point_cloud_ply(cloud, {str(exported)!r})
    except BaseException as error:
        export_errors.append(repr(error))
export_thread = threading.Thread(target=export_cloud)
export_thread.start()
deadline = time.monotonic() + 10
while not os.path.exists({str(exported)!r}) and export_thread.is_alive():
    assert time.monotonic() < deadline, "export did not open its output"
    time.sleep(.001)
assert export_thread.is_alive(), "export finished before overlap could be exercised"
scene.remove_node("export cloud")
export_thread.join(10)
assert not export_thread.is_alive(), "export did not finish"
assert not export_errors, export_errors
try:
    cloud.size
except RuntimeError:
    pass
else:
    raise AssertionError("removed export source remains accessible")
''')
    with exported.open("rb") as stream:
        header = []
        while True:
            line = stream.readline()
            assert line, "incomplete PLY header"
            header.append(line)
            if line.strip() == b"end_header":
                break
        assert b"element vertex 2000000\n" in header or b"element vertex 2000000\r\n" in header
        sizes = {b"float": 4, b"uchar": 1}
        stride = sum(sizes[line.split()[1]] for line in header if line.startswith(b"property "))
        assert stride >= 3 * 4 + 3
        assert len(stream.read()) == 2000000 * stride


def test_point_cloud_handles_allow_graceful_exit(request, monkeypatch, tmp_path):
    import subprocess

    children = []
    original_popen = subprocess.Popen

    def track_popen(*args, **kwargs):
        process = original_popen(*args, **kwargs)
        children.append(process)
        return process

    monkeypatch.setattr(subprocess, "Popen", track_popen)
    endpoint = request.getfixturevalue("camera_runtime")
    assert len(children) == 1
    _editor(endpoint, '''
import lichtfeld as lf, threading
scene = lf.get_scene()
scene.add_point_cloud("exit cloud", lf.Tensor.zeros([3, 3], device="cpu"),
                      lf.Tensor.ones([3, 3], device="cpu"))
exit_cloud = scene.get_node("exit cloud").point_cloud()
exit_tensor = exit_cloud.means
''')
    _tool(endpoint, "project_save_as", {"path": str(tmp_path / "exit.licht")})
    _editor(endpoint, "threading.Timer(.75, lf.request_exit).start()")
    assert children[0].wait(timeout=30) == 0


def test_python_history_playback_dispatch_and_callbacks(camera_runtime):
    _editor(camera_runtime, '''
import lichtfeld as lf, threading
lf.undo.clear()
caller_thread = threading.get_ident()
callback_threads = []
counter = [11]
def change(delta):
    callback_threads.append(threading.get_ident())
    counter[0] += delta
lf.undo.push("test.first", lambda: change(-1), lambda: change(1))
lf.undo.push("test.second", lambda: change(-10), lambda: change(10))
result = lf.undo.jump("undo", 2)
assert result["success"] and result["steps_performed"] == 2, result
assert counter == [0]
result = lf.undo.jump("redo", 2)
assert result["success"] and result["steps_performed"] == 2, result
assert counter == [11]
assert lf.undo.undo() and counter == [1]
assert lf.undo.redo() and counter == [11]
assert len(set(callback_threads)) == 1
assert callback_threads[0] != caller_thread
finished = threading.Event()
callback_errors = []
def undo_on_ui():
    try:
        assert lf.undo.undo()
    except BaseException as error:
        callback_errors.append(repr(error))
    finally:
        finished.set()
lf.ui.schedule_on_ui_thread(undo_on_ui)
assert finished.wait(5), "viewer-thread playback deadlocked"
assert not callback_errors, callback_errors
assert counter == [1]
''')


def test_point_cloud_access_during_viewer_clear_and_replacement(camera_runtime):
    _editor(camera_runtime, r'''
import lichtfeld as lf, threading, time
scene = lf.get_scene()
ready = threading.Event()
finished = threading.Event()
errors = []
handles = []
reads = [0]

def replace_cloud():
    try:
        scene.clear()
        scene.add_point_cloud("replace cloud", lf.Tensor.zeros([3, 3], device="cpu"),
                              lf.Tensor.ones([3, 3], device="cpu"))
        cloud = scene.get_node("replace cloud").point_cloud()
        assert cloud.size == 3  # Also exercises direct viewer-thread access.
        handles.append(cloud)
    except BaseException as error:
        errors.append(repr(error))
    finally:
        ready.set()

lf.ui.schedule_on_ui_thread(replace_cloud)
assert ready.wait(5)
assert not errors, errors
saved = handles[0].means

def read_clouds():
    try:
        while not finished.is_set():
            cloud = handles[-1]
            try:
                assert cloud.size == 3
                assert cloud.means.numel == 9
                assert cloud.colors.numel == 9
                cloud.attribute_names
                cloud.is_gaussian()
                reads[0] += 1
            except RuntimeError as error:
                assert "PointCloud" in str(error), error
    except BaseException as error:
        errors.append(repr(error))

reader = threading.Thread(target=read_clouds)
reader.start()
try:
    deadline = time.monotonic() + 5
    while reads[0] < 2:
        assert time.monotonic() < deadline, "reader did not complete an access"
        time.sleep(.001)
    for _ in range(12):
        ready.clear()
        lf.ui.schedule_on_ui_thread(replace_cloud)
        assert ready.wait(5), "clear/replacement deadlocked"
        assert not errors, errors
finally:
    finished.set()
    reader.join(5)
assert not reader.is_alive(), "PointCloud reader deadlocked"
assert not errors, errors
assert reads[0] >= 2
assert saved.numel == 9
for old_cloud in handles[:-1]:
    try:
        old_cloud.size
    except RuntimeError as error:
        assert "PointCloud" in str(error), error
    else:
        raise AssertionError("old handle resolved replacement cloud")
''')

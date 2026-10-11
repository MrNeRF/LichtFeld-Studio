# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Point-cloud handles retain owned data and resolve live scene identity safely."""

import time

import pytest

from test_api_camera_project_runtime import camera_runtime, _editor
from test_render_on_demand_idle import _tool

pytestmark = [pytest.mark.gpu, pytest.mark.integration]


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

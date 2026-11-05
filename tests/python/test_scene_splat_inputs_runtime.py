# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Scene.add_splat input checks and scene values that exist before any dataset is loaded."""

import pytest

from test_api_camera_project_runtime import camera_runtime, _editor  # noqa: F401

pytestmark = [pytest.mark.gpu, pytest.mark.integration]


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

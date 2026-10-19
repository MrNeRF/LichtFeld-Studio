# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Front splats hide a generated mesh in both independent panels."""

import math
import os
import struct
from pathlib import Path

import pytest

from runtime_test_app import isolated_app
from test_api_camera_project_runtime import _editor
from test_render_on_demand_idle import _tool
from viewer_mesh_test_support import viewport_image

pytestmark = [pytest.mark.gpu, pytest.mark.integration]


def _green(rgb):
    r, g, b = rgb
    return g > 80 and g > 2 * max(r, b)


def _front_splats(path, z=1.):
    fields = ["x", "y", "z", "nx", "ny", "nz"]
    fields += [f"f_dc_{i}" for i in range(3)] + [f"f_rest_{i}" for i in range(45)]
    fields += ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    header = "ply\nformat binary_little_endian 1.0\nelement vertex 961\n"
    header += "".join(f"property float {name}\n" for name in fields) + "end_header\n"
    with path.open("wb") as output:
        output.write(header.encode())
        for y in range(31):
            for x in range(31):
                values = [(x - 15) / 15., (y - 15) / 15., z, 0., 0., 1.]
                values += [(channel - .5) / .28209479177387814 for channel in (.95, .05, .05)]
                values += [0.] * 45 + [10.] + [math.log(.075)] * 3 + [1., 0., 0., 0.]
                output.write(struct.pack("<" + str(len(values)) + "f", *values))


MESH = """\
import lichtfeld as lf
from array import array
identity=[1.,0.,0.,0.,0.,1.,0.,0.,0.,0.,1.,0.,0.,0.,0.,1.]
for node in lf.get_scene().get_nodes():
    if node.name == 'front':
        lf.set_node_visualizer_world_transform(node.name, identity, record_history=False)
def tensor(rows, code):
    data = array(code, [value for row in rows for value in row])
    return lf.Tensor.from_numpy(memoryview(data).cast('B').cast(code, (len(rows), len(rows[0]))))
vertices=[[-1.5,-1.5,0.],[1.5,-1.5,0.],[1.5,1.5,0.],[-1.5,-1.5,0.],[1.5,1.5,0.],[-1.5,1.5,0.]]
lf.get_scene().add_mesh('QA_occlusion_mesh',tensor(vertices,'f'),tensor([[0,1,2],[3,4,5]],'i'),
                        tensor([[.05,.95,.05,1.]]*6,'f'),tensor([[0.,0.,1.]]*6,'f'))
lf.set_node_visualizer_world_transform('QA_occlusion_mesh',identity,record_history=False)
settings=lf.get_render_settings()
settings.mesh_light_intensity=0.
settings.mesh_ambient=1.
settings.mesh_backface_culling=False
settings.mesh_shadow_enabled=False
settings.mesh_wireframe=False
"""


def _red(rgb):
    r, g, b = rgb
    return r > 100 and r > 3 * max(g, b)


@pytest.mark.parametrize("backend", ["3dgs", "3dgut"])
def test_split_mesh_respects_front_splat_depth(tmp_path, backend):
    pictures = Path(os.environ.get("LFS_TEST_ARTIFACT_DIR", tmp_path)) / backend
    pictures.mkdir(parents=True, exist_ok=True)
    model = tmp_path / "front.ply"
    _front_splats(model)
    with isolated_app(tmp_path) as endpoint:
        _tool(endpoint, "scene_load_ply", {"path": str(model)})
        job = _tool(endpoint, "runtime_job_wait", {"job_id": "import.dataset", "timeout_ms": 60000})
        assert job["success"], job
        _editor(endpoint, MESH)
        _tool(endpoint, "render_settings_set", {
            "show_grid": False, "show_coord_axes": False, "show_pivot": False,
            "render_scale": 1., "raster_backend": backend,
        })
        _tool(endpoint, "camera_set_view", {
            "eye": [0., 0., 6.], "target": [0., 0., 0.], "up": [0., 1., 0.],
        })
        single = viewport_image(endpoint, pictures, "single", require_green=False)
        _tool(endpoint, "render_settings_set", {"split_view_mode": 3})
        _editor(endpoint, """\
lf.set_camera((0.,0.,6.), (0.,0.,0.), (0.,1.,0.), panel='left')
lf.set_camera((0.,0.,6.), (0.,0.,0.), (0.,1.,0.), panel='right')
""")
        split = viewport_image(endpoint, pictures, "split", require_green=False)

    assert single.size == split.size
    w, h = single.size
    y = h // 2
    assert _red(single.getpixel((w // 2, y)))
    for x in (w // 4, 3 * w // 4):
        assert _red(split.getpixel((x, y)))


@pytest.mark.parametrize("backend", ["3dgs", "3dgut"])
def test_split_mesh_shows_through_splat_sheet_behind(tmp_path, backend):
    pictures = Path(os.environ.get("LFS_TEST_ARTIFACT_DIR", tmp_path)) / backend
    pictures.mkdir(parents=True, exist_ok=True)
    model = tmp_path / "front.ply"
    _front_splats(model, -4.)
    with isolated_app(tmp_path) as endpoint:
        _tool(endpoint, "scene_load_ply", {"path": str(model)})
        job = _tool(endpoint, "runtime_job_wait", {"job_id": "import.dataset", "timeout_ms": 60000})
        assert job["success"], job
        _editor(endpoint, MESH)
        _tool(endpoint, "render_settings_set", {
            "show_grid": False, "show_coord_axes": False, "show_pivot": False,
            "render_scale": 1., "raster_backend": backend,
        })
        _tool(endpoint, "camera_set_view", {
            "eye": [0., 0., 6.], "target": [0., 0., 0.], "up": [0., 1., 0.],
        })
        single = viewport_image(endpoint, pictures, "behind-single", require_green=False)
        _tool(endpoint, "render_settings_set", {"split_view_mode": 3})
        _editor(endpoint, """\
lf.set_camera((0.,0.,6.), (0.,0.,0.), (0.,1.,0.), panel='left')
lf.set_camera((0.,0.,6.), (0.,0.,0.), (0.,1.,0.), panel='right')
""")
        split = viewport_image(endpoint, pictures, "behind-split", require_green=False)

    assert single.size == split.size
    w, h = single.size
    y = h // 2
    assert _green(single.getpixel((w // 2, y))), single.getpixel((w // 2, y))
    for x in (w // 4, 3 * w // 4):
        assert _green(split.getpixel((x, y))), split.getpixel((x, y))

# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Surface normals retain their geometry across projections and exports."""

import math
import struct

import pytest

from runtime_test_app import isolated_app
from test_api_camera_project_runtime import _editor
from test_render_on_demand_idle import _tool

pytestmark = [pytest.mark.gpu, pytest.mark.integration]


@pytest.fixture(scope="module")
def normal_endpoint(tmp_path_factory):
    directory = tmp_path_factory.mktemp("viewport-normals")
    model = directory / "planes.ply"
    fields = ["x", "y", "z"] + [f"f_dc_{i}" for i in range(3)]
    fields += ["opacity", "scale_0", "scale_1", "scale_2"] + [f"rot_{i}" for i in range(4)]
    angle = math.pi / 6
    with model.open("wb") as output:
        header = "ply\nformat binary_little_endian 1.0\nelement vertex 7442\n"
        output.write((header + "".join(f"property float {field}\n" for field in fields) + "end_header\n").encode())
        for z in (0, 6):
            for y in range(-30, 31):
                for x in range(-30, 31):
                    output.write(struct.pack(
                        "<14f", x * .06 * math.cos(angle), y * .06, z - x * .06 * math.sin(angle),
                        1, 0, 0, 10, math.log(.08), math.log(.08), math.log(.002),
                        math.cos(angle / 2), 0, math.sin(angle / 2), 0))
    with isolated_app(directory) as endpoint:
        _tool(endpoint, "scene_load_ply", {"path": str(model)})
        _tool(endpoint, "runtime_job_wait", {"job_id": "import.dataset", "timeout_ms": 30000})
        _editor(endpoint, """
import lichtfeld as lf
lf.set_node_visualizer_world_transform('planes', [1.,0.,0.,0., 0.,1.,0.,0., 0.,0.,1.,0., 0.,0.,0.,1.], record_history=False)
settings = lf.get_render_settings()
settings.show_grid = settings.show_coord_axes = settings.show_camera_frustums = False
settings.background_color = (0, 0, 0)
lf.set_camera((0,0,3), (0,0,0))
lf.set_viewport_render_mode('normal')
""")
        yield endpoint


def test_panorama_normals_preserve_a_plane_and_cover_rear_rays(normal_endpoint):
    _editor(normal_endpoint, "lf.get_render_settings().equirectangular = True")
    _editor(normal_endpoint, """
image = lf.capture_viewport().image
h, w, _ = image.shape
center = image[h//2, w//2].tolist()
assert center[2] > .8, center
for u in (.53, .55):
    sample = image[h//2, int(w*u)].tolist()
    assert max(abs(a-b) for a,b in zip(center, sample)) < .03, (u, center, sample)
for u in (.025, .975):
    sample = image[h//2, int(w*u)].tolist()
    assert sample[1] > .4, (u, sample)
""")

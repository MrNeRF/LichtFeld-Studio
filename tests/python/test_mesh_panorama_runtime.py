# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Generated mesh geometry follows the panorama camera projection."""

import os
from pathlib import Path

import pytest

from runtime_test_app import isolated_app
from test_api_camera_project_runtime import _editor
from test_render_on_demand_idle import _tool
from viewer_mesh_test_support import difference, setup_mesh, viewport_image

pytestmark = [pytest.mark.gpu, pytest.mark.integration]


SEAM_MESH = """\
import lichtfeld as lf
from array import array
# The magenta face straddles the seam; the green face keeps the mesh bound visible.
vertices=[[-0.35,-0.3,3.],[-0.35,0.4,3.],[0.35,0.05,3.],
          [-1.2,-1.,-10.],[1.2,-1.,-10.],[0.,1.2,-10.]]
indices=[[0,1,2],[3,4,5]]
colors=[[1.,0.,1.,1.]]*3+[[0.,1.,0.,1.]]*3
normals=[[0.,0.,-1.]]*6
def tensor(rows, code):
    data=array(code,[value for row in rows for value in row])
    return lf.Tensor.from_numpy(memoryview(data).cast('B').cast(code,(len(rows),len(rows[0]))))
lf.get_scene().add_mesh('QA_panorama_seam',tensor(vertices,'f'),tensor(indices,'i'),
                        tensor(colors,'f'),tensor(normals,'f'))
lf.set_node_visualizer_world_transform('QA_panorama_seam',
    [1.,0.,0.,0., 0.,1.,0.,0., 0.,0.,1.,0., 0.,0.,0.,1.], record_history=False)
settings=lf.get_render_settings()
settings.mesh_light_intensity=0.
settings.mesh_ambient=1.
settings.mesh_backface_culling=False
settings.mesh_shadow_enabled=False
settings.mesh_wireframe=False
settings.equirectangular=True
"""


def _magenta_pixels(image, bounds):
    x0, y0, x1, y1 = bounds
    return sum(
        r > 100 and b > 100 and 2 * g < min(r, b)
        for r, g, b in image.crop((x0, y0, x1, y1)).getdata()
    )


def test_mesh_equirectangular_projection_changes_geometry(tmp_path):
    pictures = Path(os.environ.get("LFS_TEST_ARTIFACT_DIR", tmp_path))
    pictures.mkdir(parents=True, exist_ok=True)
    with isolated_app(tmp_path) as endpoint:
        setup_mesh(endpoint)
        _tool(endpoint, "render_settings_set", {"raster_backend": "3dgut"})
        _tool(endpoint, "camera_set_view", {
            "eye": [4, 3, 6], "target": [0, .5, 0], "up": [0, 1, 0],
        })
        perspective = viewport_image(endpoint, pictures, "perspective")
        _editor(endpoint, "settings=lf.get_render_settings(); settings.equirectangular=True")
        assert "True" in _editor(endpoint, "print(lf.get_render_settings().equirectangular)")
        panorama = viewport_image(endpoint, pictures, "panorama", require_green=False)
        _editor(endpoint, "settings.equirectangular=False; settings.focal_length_mm=70.")
        focal_control = viewport_image(endpoint, pictures, "focal-control", require_green=False)
        _editor(endpoint, "settings.focal_length_mm=35.")
        restored = viewport_image(endpoint, pictures, "restored")

    assert perspective.size == panorama.size == focal_control.size == restored.size
    width, height = perspective.size
    roi = (60, 160, width - 35, height - 50)
    assert difference(perspective.crop(roi), focal_control.crop(roi)) > 3.
    assert difference(perspective.crop(roi), restored.crop(roi)) < .05
    assert difference(perspective.crop(roi), panorama.crop(roi)) > 3.


@pytest.mark.parametrize("backend", ["3dgs", "3dgut"])
@pytest.mark.parametrize("split", [False, True], ids=["single", "split"])
def test_mesh_equirectangular_seam_does_not_stretch_across_view(tmp_path, backend, split):
    pictures = Path(os.environ.get("LFS_TEST_ARTIFACT_DIR", tmp_path)) / f"{backend}-{'split' if split else 'single'}"
    pictures.mkdir(parents=True, exist_ok=True)
    with isolated_app(tmp_path) as endpoint:
        _editor(endpoint, SEAM_MESH)
        _tool(endpoint, "render_settings_set", {
            "show_grid": False, "show_coord_axes": False, "show_pivot": False,
            "render_scale": 1., "raster_backend": backend,
            "split_view_mode": 3 if split else 0,
        })
        if split:
            _editor(endpoint, """\
lf.set_camera((0.,0.,0.), (0.,0.,-1.), (0.,1.,0.), panel='left')
lf.set_camera((0.,0.,0.), (0.,0.,-1.), (0.,1.,0.), panel='right')
""")
        else:
            _tool(endpoint, "camera_set_view", {
                "eye": [0., 0., 0.], "target": [0., 0., -1.], "up": [0., 1., 0.],
            })
        image = viewport_image(endpoint, pictures, "seam", require_green=False)

    width, height = image.size
    if split:
        left_interior = _magenta_pixels(image, (width // 20, height // 5, width * 9 // 20, height * 4 // 5))
        right_interior = _magenta_pixels(image, (width * 11 // 20, height // 5, width * 19 // 20, height * 4 // 5))
        left_edges = _magenta_pixels(image, (0, height // 5, width // 20, height * 4 // 5))
        right_edges = _magenta_pixels(image, (width * 19 // 20, height // 5, width, height * 4 // 5))
        assert left_interior == 0 and right_interior == 0, "Seam-crossing triangle stretched through a split panel"
        assert left_edges > 0 and right_edges > 0, "Seam geometry disappeared from a split viewport edge"
    else:
        center = _magenta_pixels(image, (width // 5, height // 5, width * 4 // 5, height * 4 // 5))
        left_edge = _magenta_pixels(image, (0, height // 5, width // 20, height * 4 // 5))
        right_edge = _magenta_pixels(image, (width * 19 // 20, height // 5, width, height * 4 // 5))
        assert center == 0, "Seam-crossing triangle stretched across the panorama"
        assert left_edge > 0 and right_edge > 0, "Seam geometry disappeared from a panorama edge"

# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Generated mesh geometry follows the panorama camera projection."""

import json
import time

import pytest

from test_api_camera_project_runtime import _editor
from test_api_camera_project_runtime import camera_runtime
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


def _hide_viewport_chrome(endpoint, directory):
    """The active viewport toolbar sits on the left edge and covers the seam sliver."""
    _editor(endpoint, "import lichtfeld as lf\nlf.toggle_ui()")
    path = directory / "chrome-bounds.json"
    previous = None
    stable = 0
    started = time.monotonic()
    for _ in range(40):
        time.sleep(0.15)
        _editor(endpoint, "import json, lichtfeld as lf\nopen(%r, 'w').write(json.dumps(lf.selection.get_viewport_bounds()))" % str(path))
        bounds = json.loads(path.read_text())
        if bounds == previous and bounds[2] > 50 and time.monotonic() - started >= 1.3:
            stable += 1
            if stable >= 2:
                return
        else:
            stable = 0
        previous = bounds
    raise AssertionError("viewport chrome did not settle after hiding the UI: %s" % previous)


def _assert_seam_edges(image):
    width, height = image.size
    assert width > 200 and height > 200, image.size
    band = max(width // 20, 8)
    center = _magenta_pixels(image, (width // 5, height // 5, width * 4 // 5, height * 4 // 5))
    left_edge = _magenta_pixels(image, (0, height // 5, band, height * 4 // 5))
    right_edge = _magenta_pixels(image, (width - band, height // 5, width, height * 4 // 5))
    assert center == 0, "Seam-crossing triangle stretched across the panorama"
    assert left_edge > 0 and right_edge > 0, (
        "Seam geometry disappeared from a panorama edge (%s, %s, %s)" % (left_edge, right_edge, image.size))


def test_mesh_equirectangular_projection_changes_geometry(camera_runtime, tmp_path):
    endpoint = camera_runtime
    setup_mesh(endpoint)
    _tool(endpoint, "render_settings_set", {"raster_backend": "3dgut"})
    _tool(endpoint, "camera_set_view", {
        "eye": [4, 3, 6], "target": [0, .5, 0], "up": [0, 1, 0],
    })
    perspective = viewport_image(endpoint, tmp_path, "perspective")
    _editor(endpoint, "settings=lf.get_render_settings(); settings.equirectangular=True")
    assert "True" in _editor(endpoint, "print(lf.get_render_settings().equirectangular)")
    panorama = viewport_image(endpoint, tmp_path, "panorama", require_green=False)
    _editor(endpoint, "settings.equirectangular=False; settings.focal_length_mm=70.")
    focal_control = viewport_image(endpoint, tmp_path, "focal-control", require_green=False)
    _editor(endpoint, "settings.focal_length_mm=35.")
    restored = viewport_image(endpoint, tmp_path, "restored")

    assert perspective.size == panorama.size == focal_control.size == restored.size
    width, height = perspective.size
    roi = (60, 160, width - 35, height - 50)
    assert difference(perspective.crop(roi), focal_control.crop(roi)) > 3.
    assert difference(perspective.crop(roi), restored.crop(roi)) < .05
    assert difference(perspective.crop(roi), panorama.crop(roi)) > 3.


@pytest.mark.parametrize("backend", ["3dgs", "3dgut"])
@pytest.mark.parametrize("split", [False, True], ids=["single", "split"])
def test_mesh_equirectangular_seam_does_not_stretch_across_view(camera_runtime, tmp_path, backend, split):
    endpoint = camera_runtime
    _editor(endpoint, SEAM_MESH)
    _tool(endpoint, "render_settings_set", {
        "show_grid": False, "show_coord_axes": False, "show_pivot": False,
        "render_scale": 1., "raster_backend": backend,
    })
    if split:
        areas_path = tmp_path / "split-areas.json"
        _editor(endpoint, """\
import json, lichtfeld as lf
view = next(area for area in lf.ui.screen.areas() if area['is_view'])['id']
added = lf.ui.screen.split(view, 'vertical', .5)
assert added
views = [area['id'] for area in lf.ui.screen.areas() if area['is_view']]
assert len(views) == 2
for area in views:
    lf.ui.screen.set_view_camera(area, (0.,0.,0.), (0.,0.,-1.), (0.,1.,0.))
    lf.ui.screen.set_view_settings(area, equirectangular=True, show_grid=False,
                                   show_coord_axes=False, show_pivot=False)
open(%r, 'w').write(json.dumps(views))
""" % str(areas_path))
        _hide_viewport_chrome(endpoint, tmp_path)
        images = []
        for index, view_id in enumerate(json.loads(areas_path.read_text())):
            _editor(endpoint, "import lichtfeld as lf\nlf.ui.screen.set_active_view(%d)" % int(view_id))
            images.append(viewport_image(endpoint, tmp_path, "seam-split-%d" % index, require_green=False))
    else:
        _tool(endpoint, "camera_set_view", {
            "eye": [0., 0., 0.], "target": [0., 0., -1.], "up": [0., 1., 0.],
        })
        _hide_viewport_chrome(endpoint, tmp_path)
        images = [viewport_image(endpoint, tmp_path, "seam", require_green=False)]

    for image in images:
        _assert_seam_edges(image)

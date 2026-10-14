# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Removing a mesh leaves the splat viewport renderable."""

import pytest

from runtime_test_app import isolated_app
from test_api_camera_project_runtime import _editor
from test_render_on_demand_idle import _tool
from viewer_mesh_test_support import MESH_CODE, background_splats, viewport_image

pytestmark = [pytest.mark.gpu, pytest.mark.integration]


@pytest.mark.parametrize("duplicate", [False, True])
def test_removing_only_mesh_keeps_splats_visible(tmp_path, duplicate):
    model = tmp_path / "generated.ply"
    background_splats(model)
    with isolated_app(tmp_path) as endpoint:
        _tool(endpoint, "scene_load_ply", {"path": str(model)})
        job = _tool(endpoint, "runtime_job_wait", {"job_id": "import.dataset", "timeout_ms": 60000})
        assert job["success"], job
        _editor(endpoint, "import lichtfeld as lf\n"
                          "lf.set_node_visualizer_world_transform('generated', "
                          "[1.,0.,0.,0., 0.,1.,0.,0., 0.,0.,1.,0., 0.,0.,0.,1.], "
                          "record_history=False)")
        _editor(endpoint, MESH_CODE)
        _tool(endpoint, "render_settings_set", {
            "show_grid": False, "show_coord_axes": False, "show_pivot": False,
            "render_scale": 1., "raster_backend": "3dgs",
        })
        _tool(endpoint, "camera_set_view", {
            "eye": [0., 0., 6.], "target": [0., 0., 0.], "up": [0., 1., 0.],
        })
        if duplicate:
            _editor(endpoint, """scene = lf.get_scene()
src = next(node.splat_data() for node in scene.get_nodes() if node.name == 'generated')
added = scene.add_splat('copy', src.means_raw, src.sh0_raw, src.shN_raw,
                        src.scaling_raw, src.rotation_raw, src.opacity_raw,
                        sh_degree=src.max_sh_degree, scene_scale=src.scene_scale)
assert added >= 0""")
        before = viewport_image(endpoint, tmp_path, "before-removal", require_green=False)
        _editor(endpoint, "lf.get_scene().remove_node('QA_mesh')")
        after = viewport_image(endpoint, tmp_path, "after-removal", require_green=False)

    before_core = before.crop((80, 160, before.width - 35, before.height - 50))
    after_core = after.crop((80, 160, after.width - 35, after.height - 50))
    assert max(before_core.getextrema()[channel][1] for channel in range(3)) > 50
    assert max(after_core.getextrema()[channel][1] for channel in range(3)) > 50

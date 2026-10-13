# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Mesh headlights follow each camera and do not depend on the world origin."""

import pytest

from runtime_test_app import isolated_app
from test_api_camera_project_runtime import _editor
from test_render_on_demand_idle import _tool
from viewer_mesh_test_support import background_splats, setup_mesh, viewport_image, difference

pytestmark = [pytest.mark.gpu, pytest.mark.integration]


@pytest.mark.parametrize("shadows", [False, True])
def test_joint_camera_mesh_translation_preserves_lighting(tmp_path, shadows):
    with isolated_app(tmp_path) as endpoint:
        setup_mesh(endpoint)
        _editor(endpoint, f"settings=lf.get_render_settings(); settings.mesh_shadow_enabled={shadows}; settings.mesh_shadow_resolution=512")

        def position(offset):
            _editor(endpoint, f'''lf.set_node_visualizer_world_transform('QA_mesh',
                lf.compose_transform({offset!r}, [0.,0.,0.], [1.,1.,1.]), record_history=False)''')
            _tool(endpoint, "camera_set_view", {
                "eye": [4 + offset[0], 3 + offset[1], 6 + offset[2]],
                "target": [offset[0], .5 + offset[1], offset[2]], "up": [0, 1, 0],
            })

        for intensity, ambient in [(0., .4), (1., .15)]:
            _editor(endpoint, f"settings.mesh_light_intensity={intensity}; settings.mesh_ambient={ambient}")
            position([0., 0., 0.])
            reference = viewport_image(endpoint, tmp_path, f"reference-{intensity}")
            roi = (80, 160, reference.width - 80, reference.height - 50)
            for axis, offset in enumerate([[2.,0.,0.], [-2.,0.,0.], [0.,2.,0.], [0.,0.,2.]]):
                position(offset)
                actual = viewport_image(endpoint, tmp_path, f"translated-{intensity}-{axis}")
                assert difference(reference.crop(roi), actual.crop(roi)) < .05


@pytest.mark.parametrize("shadows", [False, True])
def test_left_camera_does_not_change_right_mesh_lighting(tmp_path, shadows):
    model = tmp_path / "background.ply"
    background_splats(model)
    with isolated_app(tmp_path) as endpoint:
        _tool(endpoint, "scene_load_ply", {"path": str(model)})
        job = _tool(endpoint, "runtime_job_wait", {"job_id": "import.dataset", "timeout_ms": 60000})
        assert job["success"], job
        setup_mesh(endpoint)
        _editor(endpoint, f"settings=lf.get_render_settings(); settings.mesh_shadow_enabled={shadows}; settings.mesh_shadow_resolution=512")
        view_text = _editor(endpoint, '''import lichtfeld as lf
areas = [area for area in lf.ui.screen.areas() if area["is_view"]]
assert len(areas) == 1, areas
left_id = areas[0]["id"]
right_id = lf.ui.screen.split(left_id, direction="vertical", factor=0.5)
assert right_id > 0
views = sorted((area for area in lf.ui.screen.areas() if area["is_view"]), key=lambda area: area["x"])
left_id, right_id = views[0]["id"], views[1]["id"]
print(f"HEADLIGHT_VIEW_IDS={left_id},{right_id}")''')
        marker = "HEADLIGHT_VIEW_IDS="
        line = next(line for line in view_text.splitlines() if line.startswith(marker))
        left_id, right_id = (int(value) for value in line[len(marker):].split(","))
        for view_id in (left_id, right_id):
            _tool(endpoint, "camera_set_view", {
                "view": view_id, "eye": [4., 3., 6.], "target": [0., .5, 0.], "up": [0., 1., 0.],
            })
        _editor(endpoint, f"assert lf.ui.screen.set_active_view({right_id})")
        try:
            for intensity, ambient in [(0., .4), (1.5, .1)]:
                _editor(endpoint, f"settings.mesh_light_intensity={intensity}; settings.mesh_ambient={ambient}")
                _editor(endpoint, f"assert lf.ui.screen.set_active_view({right_id})")
                right_before = next(
                    view for view in _tool(endpoint, "render_view_states")["views"]
                    if view["id"] == right_id
                )
                _tool(endpoint, "camera_set_view", {
                    "view": left_id, "eye": [2., 6., 4.], "target": [0., .5, 0.], "up": [0., 1., 0.],
                })
                views_after = _tool(endpoint, "render_view_states")["views"]
                right_after = next(view for view in views_after if view["id"] == right_id)
                left_after = next(view for view in views_after if view["id"] == left_id)
                assert left_after["position"] == [2., 6., 4.]
                assert right_after["position"] == right_before["position"]
                assert right_after["rotation"] == right_before["rotation"]
        finally:
            _editor(endpoint, f"lf.ui.screen.join({left_id}, {right_id}) or lf.ui.screen.close({right_id}); lf.ui.screen.set_active_view({left_id})")

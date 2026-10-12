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
        _tool(endpoint, "render_settings_set", {"split_view_mode": 3})
        _editor(endpoint, f"settings=lf.get_render_settings(); settings.mesh_shadow_enabled={shadows}; settings.mesh_shadow_resolution=512")
        for intensity, ambient in [(0., .4), (1.5, .1)]:
            _editor(endpoint, f'''settings.mesh_light_intensity={intensity}
settings.mesh_ambient={ambient}
for panel in ['left', 'right']:
    lf.set_camera((4.,3.,6.), (0.,.5,0.), (0.,1.,0.), panel=panel)
''')
            reference = viewport_image(endpoint, tmp_path, f"reference-{intensity}")
            roi = (reference.width // 2 + 70, 160, reference.width - 35, reference.height - 50)
            _editor(endpoint, "lf.set_camera((2.,6.,4.), (0.,.5,0.), (0.,1.,0.), panel='left')")
            actual = viewport_image(endpoint, tmp_path, f"left-moved-{intensity}")
            assert difference(reference.crop(roi), actual.crop(roi)) < .01

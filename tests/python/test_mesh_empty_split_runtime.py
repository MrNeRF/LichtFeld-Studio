# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent mesh-only panels draw from their own cameras."""

import os
from pathlib import Path

import pytest

from runtime_test_app import isolated_app
from test_api_camera_project_runtime import _editor
from test_render_on_demand_idle import _tool
from viewer_mesh_test_support import setup_mesh, viewport_image

pytestmark = [pytest.mark.gpu, pytest.mark.integration]


def _mesh_pixels(image, box):
    left, top, right, bottom = box

    def colored(rgb):
        return max(rgb) - min(rgb) > 35 and max(rgb) > 70

    return {(x, y) for y in range(top, bottom) for x in range(left, right)
            if colored(image.getpixel((x, y)))}


def test_independent_mesh_only_cameras_update_separately(tmp_path):
    pictures = Path(os.environ.get("LFS_TEST_ARTIFACT_DIR", tmp_path))
    pictures.mkdir(parents=True, exist_ok=True)
    with isolated_app(tmp_path) as endpoint:
        setup_mesh(endpoint)
        _tool(endpoint, "render_settings_set", {"split_view_mode": 3})
        _editor(endpoint, """\
lf.set_camera((4.,3.,6.), (0.,.5,0.), (0.,1.,0.), panel='left')
lf.set_camera((4.,3.,6.), (0.,.5,0.), (0.,1.,0.), panel='right')
""")
        reference = viewport_image(endpoint, pictures, "independent-reference", require_green=False)
        _editor(endpoint, "lf.set_camera((2.,6.,4.), (0.,.5,0.), (0.,1.,0.), panel='left')")
        moved = viewport_image(endpoint, pictures, "independent-left-moved", require_green=False)

    assert reference.size == moved.size
    width, height = reference.size
    boxes = [(70, 160, width // 2 - 20, height - 50),
             (width // 2 + 20, 160, width - 35, height - 50)]
    before = [_mesh_pixels(reference, box) for box in boxes]
    after = [_mesh_pixels(moved, box) for box in boxes]
    assert all(len(mask) > 500 for mask in before + after)
    left_change = len(before[0] ^ after[0]) / len(before[0])
    right_change = len(before[1] ^ after[1]) / len(before[1])
    assert left_change > .1
    assert right_change < .01

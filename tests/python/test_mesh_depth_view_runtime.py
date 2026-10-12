# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Generated mesh color follows the viewer's depth presentation mode."""

import os
from pathlib import Path

import pytest

from runtime_test_app import isolated_app
from test_api_camera_project_runtime import _editor
from test_render_on_demand_idle import _tool
from viewer_mesh_test_support import setup_mesh, viewport_image

pytestmark = [pytest.mark.gpu, pytest.mark.integration]


def test_mesh_depth_view_gray_palette_and_off(tmp_path):
    pictures = Path(os.environ.get("LFS_TEST_ARTIFACT_DIR", tmp_path))
    pictures.mkdir(parents=True, exist_ok=True)
    with isolated_app(tmp_path) as endpoint:
        setup_mesh(endpoint)
        _tool(endpoint, "camera_set_view", {
            "eye": [4, 3, 6], "target": [0, .5, 0], "up": [0, 1, 0],
        })
        normal = viewport_image(endpoint, pictures, "normal")
        _editor(endpoint, "lf.set_depth_view_range(1., 30.); lf.set_depth_view_mode('gray'); lf.set_depth_view(True)")
        gray = viewport_image(endpoint, pictures, "gray", require_green=False)
        _editor(endpoint, "lf.set_depth_view_range(4., 8.)")
        narrowed = viewport_image(endpoint, pictures, "gray-narrowed", require_green=False)
        _editor(endpoint, "lf.set_depth_view_mode('palette')")
        palette = viewport_image(endpoint, pictures, "palette", require_green=False)
        _editor(endpoint, "lf.set_depth_view(False)")
        restored = viewport_image(endpoint, pictures, "restored")

    assert normal.size == gray.size == narrowed.size == palette.size == restored.size
    # The colored face identifies mesh pixels without relying on a fixed window layout.
    width, height = normal.size
    normal_pixels = list(normal.getdata())

    def is_mesh_green(rgb):
        r, g, b = rgb
        return g > 1.5 * max(r, b) and g > 80

    green = [y * width + x for y in range(110, height - 50)
             for x in range(60, width - 60)
             if is_mesh_green(normal_pixels[y * width + x])]
    assert len(green) > 1000, "Generated mesh did not cover the viewport"
    gray_pixels = list(gray.getdata())
    narrowed_pixels = list(narrowed.getdata())
    palette_pixels = list(palette.getdata())
    restored_pixels = list(restored.getdata())
    assert max(max(gray_pixels[i]) - min(gray_pixels[i]) for i in green) <= 2
    assert sum(abs(gray_pixels[i][0] - narrowed_pixels[i][0]) for i in green) / len(green) > 10
    assert sum(max(palette_pixels[i]) - min(palette_pixels[i]) for i in green) / len(green) > 20
    assert max(abs(a - b) for i in green
               for a, b in zip(normal_pixels[i], restored_pixels[i])) <= 2

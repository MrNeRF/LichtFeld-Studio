# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Generated depth ties through Python, PNG export and MCP capture entrypoints."""

import json
import math
import struct

import pytest

from runtime_test_app import isolated_app
from test_api_camera_project_runtime import _editor
from test_render_capture_camera import _capture
from test_render_on_demand_idle import _tool

pytestmark = [pytest.mark.gpu, pytest.mark.integration]


def _depth_ties(path):
    properties = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2",
                  "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    count = 8193
    header = f"ply\nformat binary_little_endian 1.0\nelement vertex {count}\n"
    header += "".join(f"property float {name}\n" for name in properties) + "end_header\n"
    with path.open("wb") as output:
        output.write(header.encode())
        for index in range(count):
            colors = [(0.45 if (index // 64) % 3 == channel else -0.45) / 0.28209479177387814
                      for channel in range(3)]
            output.write(struct.pack("<17f", 0, 0, 0, 0, 0, 0, *colors,
                                     math.log(.01 / .99), *([math.log(.1)] * 3), 1, 0, 0, 0))


@pytest.mark.parametrize("split", [False, True], ids=["single", "independent"])
def test_offline_pixels_repeat_with_exact_depth_ties(tmp_path, split):
    from PIL import Image

    model = tmp_path / "depth-ties.ply"
    _depth_ties(model)
    with isolated_app(tmp_path) as endpoint:
        _tool(endpoint, "scene_load_ply", {"path": str(model)})
        job = _tool(endpoint, "runtime_job_wait", {"job_id": "import.dataset", "timeout_ms": 60000})
        assert job["success"], job
        _tool(endpoint, "render_settings_set", {
            "raster_backend": "3dgs", "render_scale": 1.,
            "split_view_mode": 3 if split else 0,
            "show_grid": False, "show_coord_axes": False, "show_pivot": False,
        })
        _editor(endpoint, """
import lichtfeld as lf
from array import array

def pixel_bytes(image):
    # Preserve float32 pixels without NumPy or 8-bit quantization.
    return array('f', image.cpu().flatten().tolist()).tobytes()

for panel in ['left', 'right']:
    lf.set_camera((0., 0., 4.), (0., 0., 0.), (0., 1., 0.), panel=panel)
""")
        reference = None
        for mode in ["alpha_blend", "stochastic", "steady_splats"]:
            _tool(endpoint, "render_settings_set", {"transparency_mode": mode})
            # Float output and the synchronous capture callback must both rebuild
            # deterministic frames, even after stochastic interactive redraws.
            _editor(endpoint, """
first = lf.render_at((0., 0., 4.), (0., 0., 0.), 64, 64)
second = lf.render_at((0., 0., 4.), (0., 0., 0.), 64, 64)
assert first is not None and second is not None
assert pixel_bytes(first) == pixel_bytes(second)
first_capture = lf.capture_viewport()
second_capture = lf.capture_viewport()
assert first_capture is not None and second_capture is not None
assert pixel_bytes(first_capture.image) == pixel_bytes(second_capture.image)
""")
            paths = [tmp_path / f"{mode}-{frame}.png" for frame in range(2)]
            # Default dimensions exercise the split-view capture export branch.
            for path in paths:
                _editor(endpoint, f"lf.export_viewport_image({json.dumps(str(path))}, include_provenance=False)")
            with Image.open(paths[0]) as first, Image.open(paths[1]) as second:
                assert first.size == second.size
                assert first.tobytes() == second.tobytes()
            for _ in range(2):
                result, image = _capture(endpoint, {})
                assert image is not None, result
                assert max(high for low, high in image.getextrema()) > 0
                pixels = (image.size, image.tobytes())
                if reference is None:
                    reference = pixels
                assert pixels == reference

# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""The Python API, reached through MCP, preserves rational rates and framing."""

import pytest

from test_api_camera_project_runtime import camera_runtime, _editor
from test_render_on_demand_idle import _tool

pytestmark = [pytest.mark.gpu, pytest.mark.integration]


@pytest.mark.parametrize("framing", ["fit", "crop", "stretch"])
def test_python_export_fractional_rate_and_independent_camera_ratio(camera_runtime, tmp_path, framing):
    endpoint = camera_runtime
    _editor(endpoint, '''
import lichtfeld as lf
vertices = lf.Tensor.zeros([3, 3], device="cpu", dtype="float32")
vertices[0, 0] = -1
vertices[1, 0] = 1
vertices[2, 1] = 1
indices = lf.Tensor.zeros([1, 3], device="cpu", dtype="int32")
indices[0, 1] = 1
indices[0, 2] = 2
lf.get_scene().add_mesh("export options triangle", vertices, indices)
assert lf.ui.set_camera_path({"version": 1, "duration": 0.1, "loopMode": "once", "playbackSpeed": 1.0,
    "keyframes": [{"time": t, "position": [0, 0, 5], "rotation": [1, 0, 0, 0],
                   "focal_length_mm": 35, "easing": 0} for t in [0.0, 0.1]]})
state = lf.ui.get_sequencer_state()
state.camera_width, state.camera_height = 9, 16
assert lf.ui.get_sequencer_state().camera_width == 9
assert lf.ui.get_sequencer_state().camera_height == 16
''')
    state = _tool(endpoint, "sequencer_get", {})
    assert (state["camera_width"], state["camera_height"]) == (9, 16)
    movie = tmp_path / "fractional.mp4"
    _editor(endpoint, f"lf.ui.export_video(320, 180, 30000, 27, {str(movie)!r}, "
            f"include_provenance=False, framerate_denominator=1001, camera_width=9, camera_height=16, framing={framing!r})")
    job = _tool(endpoint, "runtime_job_wait", {"job_id": "export.video", "until": "inactive", "timeout_ms": 60000})
    assert job["outcome"] == "completed", job
    assert job["details"]["total_frames"] == 3, job
    assert movie.stat().st_size > 0


def test_python_export_rejects_invalid_optional_settings(camera_runtime):
    _editor(camera_runtime, '''
import lichtfeld as lf
for options in [dict(framerate_denominator=0), dict(camera_width=9), dict(framing="unknown")]:
    try:
        lf.ui.export_video(320, 180, 30, 18, **options)
    except ValueError:
        pass
    else:
        raise AssertionError(options)
''')

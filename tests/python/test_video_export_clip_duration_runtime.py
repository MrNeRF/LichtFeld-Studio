# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Video export must include the hold between the last camera key and clip end."""

import json
import math

import pytest

from test_api_camera_project_runtime import camera_runtime, _editor
from test_render_on_demand_idle import _tool

pytestmark = [pytest.mark.gpu, pytest.mark.integration]


@pytest.mark.parametrize("duration,mode", [(2.0, "once"), (8.0, "once"), (8.0, "loop")])
def test_video_export_includes_clip_end(camera_runtime, tmp_path, duration, mode):
    endpoint = camera_runtime
    _editor(endpoint, '''
import lichtfeld as lf, json
vertices = lf.Tensor.zeros([3, 3], device="cpu", dtype="float32")
vertices[0, 0] = -1
vertices[1, 0] = 1
vertices[2, 1] = 1
indices = lf.Tensor.zeros([1, 3], device="cpu", dtype="int32")
indices[0, 1] = 1
indices[0, 2] = 2
lf.get_scene().add_mesh("export triangle", vertices, indices)
''')
    path = {
        "version": 1, "duration": duration, "loopMode": mode, "playbackSpeed": 1.0,
        "keyframes": [
            {"time": time, "position": position, "rotation": [1, 0, 0, 0],
             "focal_length_mm": 35, "easing": 0}
            for time, position in [(0.0, [0, 0, 5]), (2.0, [1, 0, 5])]
        ],
    }
    state = tmp_path / "path.json"
    _editor(endpoint, f'''
assert lf.ui.set_camera_path(json.loads({json.dumps(path)!r}))
open({str(state)!r}, "w").write(json.dumps(lf.ui.get_camera_path()))
''')
    actual = json.loads(state.read_text())
    assert actual["duration"] == duration
    assert actual["loopMode"] == mode
    assert actual["keyframes"][-1]["time"] == 2.0
    movie = tmp_path / "clip.mp4"
    _editor(endpoint, f"lf.ui.export_video(320, 180, 24, 27, {str(movie)!r}, include_provenance=False)")
    job = _tool(endpoint, "runtime_job_wait", {
        "job_id": "export.video", "until": "inactive", "timeout_ms": 60000,
    })
    (tmp_path / "job.json").write_text(json.dumps(job, indent=2))
    assert job["outcome"] == "completed", job
    expected = math.ceil(duration * 24) + 1
    assert job["details"]["total_frames"] == expected, job
    assert job["details"]["current_frame"] == expected, job
    assert movie.stat().st_size > 0

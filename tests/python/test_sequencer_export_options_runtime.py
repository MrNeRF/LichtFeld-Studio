# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""The Python API, reached through MCP, preserves rational rates and framing."""

import json
import shutil
import subprocess

import pytest

from test_api_camera_project_runtime import camera_runtime, _editor
from test_render_on_demand_idle import _tool

pytestmark = [pytest.mark.gpu, pytest.mark.integration]


@pytest.fixture
def video_tools():
    tools = tuple(shutil.which(name) for name in ("ffmpeg", "ffprobe"))
    if not all(tools):
        pytest.skip("requires FFmpeg and ffprobe to independently decode the export")
    return tools


@pytest.mark.parametrize("framing", ["fit", "crop", "stretch"])
def test_python_export_fractional_rate_and_independent_camera_ratio(video_tools, camera_runtime, tmp_path, framing):
    endpoint = camera_runtime
    _editor(endpoint, '''
import lichtfeld as lf, math
points = [(x * 0.1, y * 0.1) for x in range(-6, 7) for y in range(-9, 10)]
f = lambda shape: lf.Tensor.zeros(shape, device="cpu", dtype="float32")
means = f([len(points), 3])
for row, (x, y) in enumerate(points):
    means[row, 0], means[row, 1] = x, y
sh0 = f([len(points), 1, 3]); sh0[:, :, :] = 0.5 / 0.28209479177387814
scales = f([len(points), 3]); scales[:, :] = math.log(0.08)
rotations = f([len(points), 4]); rotations[:, 0] = 1.0
opacity = f([len(points), 1]); opacity[:, :] = 5.0
lf.get_scene().add_splat("white fitting rectangle", means, sh0, f([len(points), 0, 3]),
                         scales, rotations, opacity, sh_degree=0)
assert lf.ui.set_camera_path({"version": 1, "duration": 0.1, "loopMode": "once", "playbackSpeed": 1.0,
    "keyframes": [{"time": t, "position": [0, 0, 5], "rotation": [1, 0, 0, 0],
                   "focal_length_mm": 35, "easing": 0} for t in [0.0, 0.1]]})
state = lf.ui.get_sequencer_state()
state.camera_width, state.camera_height = 9, 16
assert lf.ui.get_sequencer_state().camera_width == 9
assert lf.ui.get_sequencer_state().camera_height == 16
''')
    _tool(endpoint, "render_settings_set", {"raster_backend": "3dgs", "environment_mode": 0,
          "point_cloud_mode": False, "background_color": [0, 0, 0], "show_grid": False})
    state = _tool(endpoint, "sequencer_get", {})
    assert (state["camera_width"], state["camera_height"]) == (9, 16)
    movie = tmp_path / "fractional.mp4"
    _editor(endpoint, f"lf.ui.export_video(320, 180, 30000, 27, {str(movie)!r}, "
            f"include_provenance=False, framerate_denominator=1001, camera_width=9, camera_height=16, framing={framing!r})")
    job = _tool(endpoint, "runtime_job_wait", {"job_id": "export.video", "until": "inactive", "timeout_ms": 60000})
    assert job["outcome"] == "completed", job
    assert job["details"]["total_frames"] == 3, job
    assert movie.stat().st_size > 0
    ffmpeg, ffprobe = video_tools
    metadata = json.loads(subprocess.check_output([
        ffprobe, "-v", "error", "-select_streams", "v:0", "-count_frames",
        "-show_entries", "stream=width,height,r_frame_rate,nb_read_frames", "-of", "json", str(movie),
    ], text=True))["streams"][0]
    assert (metadata["width"], metadata["height"]) == (320, 180)
    assert metadata["r_frame_rate"] == "30000/1001"
    assert int(metadata["nb_read_frames"]) == 3
    pixels = subprocess.check_output([
        ffmpeg, "-v", "error", "-i", str(movie), "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ])
    assert len(pixels) == 320 * 180 * 3
    foreground = [i for i in range(320 * 180) if sum(pixels[i * 3:i * 3 + 3]) > 384]
    assert len(foreground) > 1000, "Export is blank or the reference rectangle is missing"
    xs, ys = [i % 320 for i in foreground], [i // 320 for i in foreground]
    width, height = max(xs) - min(xs) + 1, max(ys) - min(ys) + 1
    # Independent screen-space oracle for the white 2:3 rectangle. Fit retains
    # its shape, crop fills the output height, stretch widens it substantially.
    if framing == "fit":
        assert 0.5 < width / height < 0.85
        assert width < 128 and height < 175
    elif framing == "crop":
        assert height >= 175 and width > 150
    else:
        assert 1.7 < width / height < 2.5
        assert width > 150 and height < 175


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

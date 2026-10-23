# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Exercise the real timeline job with a disposable synthetic Gaussian scene."""
import json
import struct

import pytest

from test_api_camera_project_runtime import camera_runtime, _editor
from test_render_on_demand_idle import _call, _tool

pytestmark = [pytest.mark.gpu, pytest.mark.integration]


def test_original_float_capture_contract(camera_runtime):
    # The shared renderer overload must preserve the existing analytic color,
    # alpha and tiled-capture contract as well as the new timeline path.
    from test_float_image_capture_runtime import test_real_float_color_capture
    test_real_float_color_capture(camera_runtime)


def test_timeline_image_sequence_formats_and_collision(camera_runtime, tmp_path):
    endpoint = camera_runtime
    _editor(endpoint, '''
import lichtfeld as lf, math
scene = lf.get_scene()
assert scene.node_count == 0
rotation = lf.Tensor.zeros([1, 4], device='cpu')
rotation[0, 0] = 1
sh = lf.Tensor.zeros([1, 1, 3], device='cpu')
sh[0, 0, 0] = 0.5
scene.add_splat('sequence probe', lf.Tensor.zeros([1, 3], device='cpu'), sh,
                lf.Tensor.zeros([1, 0, 3], device='cpu'),
                lf.Tensor.full([1, 3], math.log(0.3), device='cpu'), rotation,
                lf.Tensor.full([1, 1], 2.0, device='cpu'))
lf.ui.reset_video_reconstruction_selection()
''')
    path = {"version": 1, "duration": 0.125, "loopMode": "once", "playbackSpeed": 1.0,
            "keyframes": [{"time": t, "position": [x, 0, 3], "rotation": [1, 0, 0, 0],
                           "focal_length_mm": 35, "easing": 0}
                          for t, x in [(0.0, 0.0), (0.125, 0.25)]]}
    _editor(endpoint, f"assert lf.ui.set_camera_path({path!r})")
    formats = {"png": "png", "jpeg": "jpg", "exr": "exr", "tiff16": "tif", "jxl": "jxl", "avif": "avif"}
    for format, extension in formats.items():
        directory = tmp_path / format
        _editor(endpoint, f"lf.ui.export_image_sequence(63, 47, 24, {str(directory)!r}, format={format!r}, filename_pattern='timeline_%04d', include_provenance=False)")
        job = _tool(endpoint, "runtime_job_wait", {"job_id": "export.video", "until": "inactive", "timeout_ms": 60000})
        (tmp_path / f"job-{format}.json").write_text(json.dumps(job, indent=2))
        assert job["outcome"] == "completed", job
        assert job["details"]["total_frames"] == job["details"]["current_frame"] == 3, job
        assert sorted(p.name for p in directory.iterdir()) == [f"timeline_{i:04d}.{extension}" for i in range(3)]
        original = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in directory.iterdir()}
        assert all(data for data, _ in original.values())
        if format == "exr":
            for index in range(3):
                data = original[f"timeline_{index:04d}.exr"][0]
                timestamp = f"{index}/1/24".encode("ascii")
                assert b"lfsOutputTimestamp\0string\0" + struct.pack("<I", len(timestamp)) + timestamp in data
                assert b"lfsOrigin\0string\0\x08\0\0\0rendered" in data
                assert b"lfsSourceTimestamp\0" not in data
        _editor(endpoint, f"lf.ui.export_image_sequence(63, 47, 24, {str(directory)!r}, format={format!r}, filename_pattern='timeline_%04d')")
        failed = _call(endpoint, "tools/call", {"name": "runtime_job_wait", "arguments": {
            "job_id": "export.video", "until": "inactive", "timeout_ms": 60000}})
        job = failed["structuredContent"]
        assert failed.get("isError") and job["outcome"] == "failed", job
        assert original == {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in directory.iterdir()}
    for format in ("exr", "tiff16", "jxl", "avif"):
        directory = tmp_path / (format + "-alpha")
        _editor(endpoint, f"lf.ui.export_image_sequence(63, 47, 24, {str(directory)!r}, format={format!r}, transparent=True)")
        job = _tool(endpoint, "runtime_job_wait", {"job_id": "export.video", "until": "inactive", "timeout_ms": 60000})
        assert job["outcome"] == "completed" and job["details"]["current_frame"] == 3, job
        assert len(list(directory.iterdir())) == 3

    _editor(endpoint, "lf.get_render_settings().equirectangular = True")
    try:
        # The native equirectangular raster contract requires 3DGUT; do not
        # accept an empty frame or silently substitute a perspective render.
        directory = tmp_path / "unsupported-spherical"
        _editor(endpoint, f"lf.ui.export_image_sequence(64, 32, 24, {str(directory)!r})")
        failed = _call(endpoint, "tools/call", {"name": "runtime_job_wait", "arguments": {
            "job_id": "export.video", "until": "inactive", "timeout_ms": 60000}})["structuredContent"]
        assert failed["outcome"] == "failed" and not list(directory.glob("*.png")), failed
        _editor(endpoint, "lf.get_render_settings().raster_backend = 3")
        for format, extension in (("png", "png"), ("exr", "exr")):
            directory = tmp_path / (format + "-spherical")
            _editor(endpoint, f"lf.ui.export_image_sequence(64, 32, 24, {str(directory)!r}, format={format!r})")
            job = _tool(endpoint, "runtime_job_wait", {"job_id": "export.video", "until": "inactive", "timeout_ms": 60000})
            assert job["outcome"] == "completed" and job["details"]["current_frame"] == 3, job
            assert len(list(directory.glob("*." + extension))) == 3
            if format == "png":
                assert struct.unpack(">II", (directory / "frame_000000.png").read_bytes()[16:24]) == (64, 32)
    finally:
        _editor(endpoint, "lf.get_render_settings().equirectangular = False; lf.get_render_settings().raster_backend = 2")

    path["duration"] = 60.0
    _editor(endpoint, f"assert lf.ui.set_camera_path({path!r})")
    directory = tmp_path / "cancelled"
    _editor(endpoint, f"lf.ui.export_image_sequence(63, 47, 24, {str(directory)!r}); lf.ui.cancel_video_export()")
    cancelled = _call(endpoint, "tools/call", {"name": "runtime_job_wait", "arguments": {
        "job_id": "export.video", "until": "inactive", "timeout_ms": 60000}})["structuredContent"]
    assert cancelled["outcome"] == "cancelled", cancelled
    assert cancelled["details"]["current_frame"] < 1440
    path["duration"] = 0.125
    _editor(endpoint, f"assert lf.ui.set_camera_path({path!r})")
    recovered = tmp_path / "after-cancel"
    _editor(endpoint, f"lf.ui.export_image_sequence(63, 47, 24, {str(recovered)!r})")
    job = _tool(endpoint, "runtime_job_wait", {"job_id": "export.video", "until": "inactive", "timeout_ms": 60000})
    assert job["outcome"] == "completed" and job["details"]["current_frame"] == 3, job
    # The shared filename fix must keep extraction metadata and actual files in sync.
    movie = tmp_path / "source.mp4"
    _editor(endpoint, f"lf.ui.export_video(320, 180, 24, 18, {str(movie)!r}, include_provenance=False)")
    job = _tool(endpoint, "runtime_job_wait", {"job_id": "export.video", "until": "inactive", "timeout_ms": 60000})
    assert job["outcome"] == "completed", job
    extracted = tmp_path / "unicode-extraction"
    _editor(endpoint, f'''
request = lf.media.IngestRequest()
request.input = {str(movie)!r}
request.selection.mode = lf.media.SelectionMode.Interval
request.selection.interval = 1
request.allow_hardware_decode = False
files = lf.media.FileExtraction()
files.files.output_directory = {str(extracted)!r}
files.files.filename_pattern = 'rendu_\u65e5_%04d'
files.write_metadata = True
assert lf.media.MediaIngest.extract_files(request, files).frames_accepted == 3
''')
    metadata = json.loads((extracted / "extraction_metadata.json").read_text(encoding="utf-8"))
    names = [frame["file"] for frame in metadata["frames"]]
    assert names == [f"rendu_\u65e5_{i:04d}.png" for i in range(1, 4)]
    assert all((extracted / name).is_file() for name in names)

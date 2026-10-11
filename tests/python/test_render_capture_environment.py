# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Viewport captures retain alpha compositing when an environment cannot load."""

import struct
import time

import pytest

from test_render_capture_camera import _capture
from test_render_on_demand_idle import _tool
from runtime_test_app import isolated_app

pytestmark = [pytest.mark.gpu, pytest.mark.integration]


def _splat_fixture(path):
    # A translucent colored grid exposes accidental straight-alpha RGB capture.
    properties = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2",
                  "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    header = "ply\nformat binary_little_endian 1.0\nelement vertex 25\n"
    header += "".join(f"property float {name}\n" for name in properties) + "end_header\n"
    with path.open("wb") as output:
        output.write(header.encode())
        for y in range(-2, 3):
            for x in range(-2, 3):
                output.write(struct.pack("<17f", x * 0.25, y * 0.25, 0, 0, 0, 0,
                                         1.5, 0.6, -0.5, 0, -2, -2, -2, 1, 0, 0, 0))


@pytest.fixture(scope="module")
def capture_endpoint(tmp_path_factory):
    directory = tmp_path_factory.mktemp("environment-capture")
    model = directory / "translucent.ply"
    _splat_fixture(model)
    with isolated_app(directory) as endpoint:
        _tool(endpoint, "scene_load_ply", {"path": str(model)})
        job = _tool(endpoint, "runtime_job_wait", {"job_id": "import.dataset", "timeout_ms": 60000})
        assert job["success"], job
        yield endpoint


def _settings(endpoint, **settings):
    _tool(endpoint, "render_settings_set", settings)
    # A settings write schedules a frame; captures do not themselves advance it.
    time.sleep(0.2)
    _capture(endpoint, {})
    time.sleep(0.2)


def _image(endpoint):
    result, image = _capture(endpoint, {})
    assert image is not None, result
    return image


def _difference(first, second):
    from PIL import ImageChops, ImageStat

    assert first.size == second.size
    return max(ImageStat.Stat(ImageChops.difference(first, second)).mean)


def _hdr(path, pixel):
    # A constant, uncompressed Radiance fixture, large enough for the image loader.
    path.write_bytes(b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 4 +X 4\n" + bytes(pixel) * 16)
    return str(path)


@pytest.fixture(params=[False, True], ids=["plain", "appearance"])
def environment_capture(capture_endpoint, request):
    endpoint = capture_endpoint
    _tool(endpoint, "editor_run", {
        "code": "import lichtfeld as lf\nlf.get_render_settings().apply_appearance_correction = " + repr(request.param),
        "show_console": False, "wait_for_completion": True,
    })
    _settings(endpoint, environment_mode=0, background_color=[0, 0, 0])
    yield endpoint
    _settings(endpoint, environment_mode=0, background_color=[0, 0, 0])


@pytest.mark.parametrize("invalid_kind", ["missing", "empty", "corrupt", "renamed"])
def test_invalid_environment_matches_black_composite(environment_capture, tmp_path, invalid_kind):
    endpoint = environment_capture
    black = _hdr(tmp_path / "black.hdr", [0, 0, 0, 0])
    _settings(endpoint, environment_mode=1, environment_map_path=black)
    reference = _image(endpoint)
    assert max(high for low, high in reference.getextrema()) > 20, "fixture is not visible"
    path = tmp_path / ("invalid.exr" if invalid_kind == "renamed" else "invalid.hdr")
    if invalid_kind == "empty":
        path.touch()
    elif invalid_kind == "corrupt":
        path.write_bytes(bytes(range(256)) * 20)
    elif invalid_kind == "renamed":
        _hdr(path, [0, 0, 0, 0])
    # A nonblack solid setting must not leak into the failed HDRI fallback.
    _settings(endpoint, environment_map_path=str(path), background_color=[1, 0, 0])
    actual = _image(endpoint)
    assert _difference(reference, actual) < 0.01


def test_valid_environment_and_solid_restore_exactly(environment_capture, tmp_path):
    endpoint = environment_capture
    solid = _image(endpoint)
    _settings(endpoint, background_color=[1, 0, 0])
    assert _difference(solid, _image(endpoint)) > 1
    _settings(endpoint, background_color=[0, 0, 0])
    assert _difference(solid, _image(endpoint)) == 0
    path = _hdr(tmp_path / "valid.hdr", [128, 64, 32, 129])
    _settings(endpoint, environment_mode=1, environment_map_path=path)
    valid = _image(endpoint)
    assert _difference(solid, valid) > 1
    _settings(endpoint, environment_map_path=str(tmp_path / "absent.hdr"))
    _image(endpoint)
    _settings(endpoint, environment_map_path=path)
    assert _difference(valid, _image(endpoint)) == 0
    _settings(endpoint, environment_mode=0)
    assert _difference(solid, _image(endpoint)) == 0

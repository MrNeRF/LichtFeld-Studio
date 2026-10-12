# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Scene reconstruction must not change the ground-truth reference's sampling."""

import os
import time
from contextlib import nullcontext

import pytest

from runtime_test_app import isolated_app
from test_render_capture_camera import _capture
from test_render_capture_environment import _splat_fixture
from test_render_on_demand_idle import _initialize, _tool

pytestmark = [pytest.mark.gpu, pytest.mark.integration]


def _python(endpoint, code):
    result = _tool(endpoint, "editor_run", {
        "code": "import lichtfeld as lf\n" + code, "show_console": False,
        "wait_for_completion": True, "timeout_ms": 20000,
    })
    assert result.get("success", True), result


def _image(endpoint):
    # Let resize/reconstruction feedback settle, then read the presented frame.
    time.sleep(0.6)
    _capture(endpoint, {})
    time.sleep(0.6)
    result, image = _capture(endpoint, {})
    assert image is not None, result
    return image


def _settings(endpoint, **settings):
    _tool(endpoint, "render_settings_set", settings)
    return _image(endpoint)


def _reference(image):
    return image.crop((image.width // 10, image.height // 5,
                       image.width * 2 // 5, image.height * 4 // 5))


@pytest.fixture(scope="module")
def comparison(tmp_path_factory):
    from PIL import Image

    directory = tmp_path_factory.mktemp("gt-reconstruction")
    dataset = directory / "dataset"
    images = dataset / "images"
    sparse = dataset / "sparse" / "0"
    images.mkdir(parents=True)
    sparse.mkdir(parents=True)
    # An odd aspect ratio exposes rounding of the presentation rectangle at
    # reduced scene resolutions. Fine detail makes resampling visible.
    for index in range(2):
        image = Image.new("RGB", (618, 410))
        image.putdata([((x * 31 + index * 71) % 256, (y * 37) % 256,
                        ((x + y) * 19) % 256)
                       for y in range(410) for x in range(618)])
        image.save(images / f"{index}.png")
    (sparse / "cameras.txt").write_text("1 PINHOLE 618 410 580 580 309 205\n")
    (sparse / "images.txt").write_text(
        "1 1 0 0 0 0 0 4 1 0.png\n\n2 1 0 0 0 0.3 0 4 1 1.png\n\n")
    (sparse / "points3D.txt").write_text("".join(
        f"{y * 20 + x + 1} {(x - 9.5) * .1} {(y - 9.5) * .1} 0 200 100 50 0\n"
        for y in range(20) for x in range(20)))
    model = directory / "synthetic.ply"
    _splat_fixture(model)
    configured = os.environ.get("LFS_GT_RECONSTRUCTION_ENDPOINT")
    with (nullcontext(configured) if configured else isolated_app(directory)) as endpoint:
        _initialize(endpoint)
        _tool(endpoint, "scene_load_dataset", {
            "path": str(dataset), "output_path": str(directory / "output"),
            "min_track_length": 0, "max_iterations": 3,
        })
        job = _tool(endpoint, "runtime_job_wait", {
            "job_id": "import.dataset", "until": "inactive", "timeout_ms": 60000,
        })
        assert not job.get("timed_out") and job.get("status") != "failed", job
        _python(endpoint, f"s = lf.io.load({str(model)!r}).splat_data\n"
                "lf.get_scene().add_splat('synthetic', s.means_raw, s.sh0_raw, s.shN_raw, "
                "s.scaling_raw, s.rotation_raw, s.opacity_raw, s.active_sh_degree, s.scene_scale)\n"
                "lf.ui.go_to_camera_view(0)\nlf.ui.toggle_gt_comparison()\n"
                "lf.ui.set_gt_comparison_mode('rgb')\n"
                "assert lf.ui.is_gt_comparison_active()")
        _settings(endpoint, point_cloud_mode=False, environment_mode=0, show_grid=False)
        yield endpoint


@pytest.mark.parametrize("scale", [1.0, 0.5, 0.25])
@pytest.mark.parametrize("preset", ["quality", "balanced", "performance"])
def test_reconstruction_preserves_reference(comparison, tmp_path, scale, preset):
    from PIL import ImageChops

    endpoint = comparison
    native = _settings(endpoint, render_scale=scale,
                       scene_upscaler="native", scene_upscaler_preset="native")
    reconstructed = _settings(endpoint, scene_upscaler="spatial", scene_upscaler_preset=preset)
    restored = _settings(endpoint, scene_upscaler="native", scene_upscaler_preset="native")
    for name, image in [("native", native), ("reconstructed", reconstructed), ("restored", restored)]:
        image.save(tmp_path / f"{name}.png")
    reference = _reference(native)
    assert all(high - low > 30 for low, high in reference.getextrema()), "reference is missing"
    if scale == 1.0 and preset == "performance":
        right = (native.width * 3 // 5, native.height // 5,
                 native.width * 9 // 10, native.height * 4 // 5)
        assert ImageChops.difference(native.crop(right), reconstructed.crop(right)).getbbox() is not None
    # Existing native presentation and round-trip restoration must stay exact.
    assert ImageChops.difference(native, restored).getbbox() is None
    assert ImageChops.difference(reference, _reference(reconstructed)).getbbox() is None


def test_camera_navigation_and_reference_restoration(comparison):
    from PIL import ImageChops

    endpoint = comparison
    _settings(endpoint, render_scale=1.0, scene_upscaler="native", scene_upscaler_preset="native")
    _python(endpoint, "lf.ui.go_to_camera_view(0)")
    first = _image(endpoint)
    _python(endpoint, "lf.ui.go_to_camera_view(1)")
    second = _image(endpoint)
    _python(endpoint, "lf.ui.go_to_camera_view(0)")
    restored = _image(endpoint)
    assert ImageChops.difference(_reference(first), _reference(second)).getbbox() is not None
    assert ImageChops.difference(first, restored).getbbox() is None

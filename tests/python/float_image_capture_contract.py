# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Run via editor_run in an empty, disposable hardware viewer.

Independent analytic samples and the existing EXR reference reader verify the
real raster producer; the large frame also crosses the native band boundary.
"""
import importlib.util
import json
import math
import tempfile
import time
from pathlib import Path

import lichtfeld as lf

camera = lf.get_camera()
settings = lf.get_render_settings()
assert camera is not None and settings is not None
saved = {name: settings.get(name) for name in ('color_tonemapping', 'splat_render_profile')}
try:
    scene = lf.get_scene()
    assert scene is not None and scene.node_count == 0, 'Requires an empty disposable viewer'
    means = lf.Tensor.zeros([1, 3], device='cpu')
    sh = lf.Tensor.zeros([1, 1, 3], device='cpu')
    for c, value in enumerate([0.12345, 0.45678, 1.25]):
        sh[0, 0, c] = (value - 0.5) / 0.28209479177387814
    rotation = lf.Tensor.zeros([1, 4], device='cpu')
    rotation[0, 0] = 1
    node = scene.add_splat('phase5a-float-contract', means, sh, lf.Tensor.zeros([1, 0, 3], device='cpu'), lf.Tensor.full([1, 3], math.log(0.45), device='cpu'), rotation, lf.Tensor.full([1, 1], math.log(0.43 / 0.57), device='cpu'))
    lf.set_camera((0.0, 0.0, 3.0), (0.0, 0.0, 0.0))
    lf.set_camera_fov(60.0)
    settings = lf.get_render_settings()
    settings.color_tonemapping = 0
    settings.splat_render_profile = 0
    root = Path(tempfile.gettempdir())
    source = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location('float_exr_reference', source / 'tests/media/test_float_exr.py')
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)

    def eotf(v):
        return ((v + 0.055) / 1.055) ** 2.4 if v > 0.04045 else v / 12.92
    expected = [eotf(v) for v in (0.12345, 0.45678, 1.25)]
    settings = lf.get_render_settings()
    settings.color_tonemapping = 0
    settings.splat_render_profile = 0
    start = time.perf_counter()
    image = lf.render_linear_image(128, 96, True)
    elapsed = time.perf_counter() - start
    assert image.shape == (96, 128, 4), image.shape
    pixels = image.tolist()
    covered = [p for row in pixels for p in row if p[3] > 0.1]
    assert covered, 'No covered raster pixels'
    for p in covered:
        for c, display in enumerate((0.12345, 0.45678, 1.25)):
            # Bound existing FP16 source/product rounding and transmittance
            # cancellation before applying the nonlinear transfer curve.
            display_error = abs(display) * (2 * 2 ** (-11) + 2 ** (-12) / p[3])
            lower = eotf(display - display_error)
            upper = eotf(display + display_error)
            assert lower - 1e-06 <= p[c] <= upper + 1e-06, (c, p, lower, upper)
    assert max((p[2] for p in covered)) > 1.5
    assert any((abs(p[3] * 255 - round(p[3] * 255)) > 0.05 for p in covered)), 'Alpha was quantized to eight bits'
    color = lf.media.FrameColor()
    color.transfer = lf.media.ColorTransfer.Linear
    color.primaries = lf.media.ColorPrimaries.Bt709
    color.alpha = lf.media.AlphaMode.Straight
    options = lf.media.ExrOutputOptions()
    options.precision = lf.media.ExrPrecision.Float
    options.compression = lf.media.ExrCompression.Uncompressed
    with tempfile.TemporaryDirectory(prefix='lfs-render-float-', dir=root) as directory:
        path = Path(directory) / 'render.exr'
        lf.io.save_exr_image(str(path), image, color, options, False)
        w, h, channels, attrs, decoded = ref.uncompressed_exr(path)
        assert (w, h) == (128, 96)
        flat = [p for row in pixels for p in row]
        for c, name in enumerate('RGB'):
            assert max((abs(v - p[c] * p[3]) for v, p in zip(decoded[name], flat))) < 1e-06
        assert max((abs(v - p[3]) for v, p in zip(decoded['A'], flat))) < 1e-06
        lf.set_camera((0.1, 0.0, 3.0), (0.0, 0.0, 0.0))
        second = lf.render_linear_image(64, 48, False)
        assert second.shape == (48, 64, 4)
        assert all((p[3] == 1.0 for row in second.tolist() for p in row))
        assert image.tolist() == pixels, 'Later render changed earlier capture'
    print('PHASE5A_FLOAT_RENDER_PASS', json.dumps({'shape': list(image.shape), 'covered': len(covered), 'expected_linear': expected, 'sample': covered[len(covered) // 2], 'capture_seconds': elapsed, 'float_exr_roundtrip': True, 'ownership_after_next_render': True}))
    settings = lf.get_render_settings()
    lf.set_camera((0.0, 0.0, 3.0), (0.0, 0.0, 0.0))
    view = lf.get_current_view()
    results = []

    def oetf(v):
        return 12.92 * v if v <= 0.0031308 else 1.055 * v ** (1 / 2.4) - 0.055
    for profile in (0, 1):
        for tone in (0, 1, 2, 4):
            settings.splat_render_profile = profile
            settings.color_tonemapping = tone
            float_image = lf.render_linear_image(128, 96, False).tolist()
            byte_image = lf.render_view_u8(view.rotation, view.translation, 128, 96, view.fov_y).tolist()
            maximum = max((abs(min(255, max(0, round(oetf(p[c]) * 255))) - q[c]) for prow, qrow in zip(float_image, byte_image) for p, q in zip(prow, qrow) for c in range(3)))
            assert maximum <= 2, (profile, tone, maximum)
            results.append({'profile': profile, 'tone': tone, 'max_byte_difference': maximum})
    settings.splat_render_profile = 0
    settings.color_tonemapping = 0
    start = time.perf_counter()
    large = lf.render_linear_image(4096, 4608, True)
    elapsed = time.perf_counter() - start
    assert large.shape == (4608, 4096, 4)
    values = []
    for y in (500, 2303, 2304, 4095, 4096, 4107):
        p = large[y, 2048, :].tolist()
        values.append([y, p])
        assert all((math.isfinite(v) for v in p))
        assert 0 <= p[3] <= 1
    assert values[1][1][3] > 0.3 and values[2][1][3] > 0.3, values
    assert abs(values[3][1][3] - values[4][1][3]) < 0.002, values
    small = lf.render_linear_image(32, 32, True)
    assert large[2304, 2048, :].tolist() == values[2][1]
    print('PHASE5A_TONE_TILING_PASS', json.dumps({'tone_byte_parity': results, 'tiled_shape': list(large.shape), 'tiled_seconds': elapsed, 'seam_samples': values, 'retained_after_render': True}))
finally:
    lf.get_scene().remove_node('phase5a-float-contract')
    for name, value in saved.items():
        settings.set(name, value)
    lf.set_camera(camera.eye, camera.target, camera.up)
    lf.set_camera_fov(camera.fov)

# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Opt-in light-field test in an explicitly selected, empty GUI instance."""

import inspect
import os
import time

import pytest
from test_render_on_demand_idle import _call, _initialize, _tool

pytestmark = [pytest.mark.gpu, pytest.mark.integration]


def _verify_in_gui():
    import copy
    import json
    import traceback

    created = False
    name = "Light-field API test"
    try:
        import lichtfeld as lf
        import numpy as np

        scene = lf.get_scene()
        assert not lf.has_trainer(), "Refusing a session with a trainer"
        assert scene is not None and scene.node_count == 0, (
            "Requires a separate empty scene"
        )

        def settings_snapshot():
            settings = lf.get_render_settings()
            return copy.deepcopy(
                {key: settings.get(key) for key in settings.get_all_properties()}
            )

        before = settings_snapshot()

        def tensor(data):
            return lf.Tensor.from_numpy(np.ascontiguousarray(data, dtype=np.float32))

        positions, colors = [], []
        for z, color, offset in [
            (0, [1, 0.1, 0.1], -0.4),
            (1, [0.1, 1, 0.1], 0.4),
            (-1, [0.1, 0.1, 1], 0),
        ]:
            for x in np.linspace(-0.25, 0.25, 11):
                for y in np.linspace(-0.4, 0.4, 15):
                    positions.append([x + offset, y, z])
                    colors.append(color)
        count = len(positions)
        scene.add_splat(
            name,
            tensor(positions),
            tensor((np.asarray(colors) - 0.5)[:, None, :] / 0.28209479177387814),
            tensor(np.empty((count, 0, 3))),
            tensor(np.full((count, 3), np.log(0.035))),
            tensor(np.tile([1, 0, 0, 0], (count, 1))),
            tensor(np.full((count, 1), 6)),
            sh_degree=0,
        )
        created = True
        rotation, eye = lf.look_at((0, 0, 5), (0, 0, 0))

        def render(near=0, far=100, position=None, shift=0):
            result = lf.render_lightfield_view_u8(
                rotation,
                eye if position is None else position,
                192,
                256,
                fov=30,
                shift_x=shift,
                near_distance=near,
                far_distance=far,
            )
            assert result is not None
            pixels = np.asarray(result.numpy())
            assert pixels.shape == (256, 192, 3) and pixels.dtype == np.uint8
            return pixels

        all_views = render()
        assert all_views.max() > 100, "Blank RGB render"
        scene.set_selection(list(range(count // 3, 2 * count // 3)))
        try:
            assert scene.has_selection(), "Selection test did not select any splats"
            selection = scene.selection_mask.numpy().copy()
            selected = render()
            np.testing.assert_array_equal(
                selected,
                all_views,
                err_msg="Selection leaked into the light-field image",
            )
            assert scene.has_selection(), "Rendering cleared the user's selection"
            np.testing.assert_array_equal(scene.selection_mask.numpy(), selection)
        finally:
            scene.clear_selection()
        default_fov = lf.render_lightfield_view_u8(rotation, eye, 192, 256)
        explicit_fov = lf.render_lightfield_view_u8(rotation, eye, 192, 256, fov=60)
        assert default_fov is not None and explicit_fov is not None
        np.testing.assert_array_equal(default_fov.numpy(), explicit_fov.numpy())
        assert not np.array_equal(all_views, render(5.2, 100)), (
            "Near clipping has no effect"
        )
        assert not np.array_equal(all_views, render(0, 5.5)), (
            "Far clipping has no effect"
        )
        plane = render(4.9, 5.1)
        focal = 256 / (2 * np.tan(np.deg2rad(30) / 2))
        shifted = render(4.9, 5.1, tensor([0.3, 0, 5]), float(focal * 0.3 / 5))

        def centroid(pixels):
            weight = np.maximum(0, pixels[:, :, 0].astype(float) - pixels[:, :, 1])
            assert weight.sum() > 0
            return float((weight * np.arange(192)[None, :]).sum() / weight.sum())

        drift = abs(centroid(plane) - centroid(shifted))
        assert drift < 0.5, "Off-axis focus plane drifted"
        for overrides in (
            {"width": 0},
            {"height": -1},
            {"width": 2049},
            {"height": 2049},
            {"fov": float("nan")},
            {"fov": 0},
            {"fov": 121},
            {"shift_x": float("inf")},
            {"near_distance": -1},
            {"far_distance": 0},
            {"near_distance": 2, "far_distance": 1},
            {"far_distance": float("inf")},
        ):
            args = {"width": 192, "height": 256, "fov": 30}
            args.update(overrides)
            assert lf.render_lightfield_view_u8(rotation, eye, **args) is None, (
                overrides
            )
        assert before == settings_snapshot(), "Viewport render settings changed"
    except Exception:  # noqa: BLE001 - relay GUI callback failures to the remote test
        print("LIGHTFIELD_RENDER_FAIL " + traceback.format_exc())
        return
    finally:
        if created:
            scene.remove_node(name)
    print("LIGHTFIELD_RENDER_PASS " + json.dumps({"focus_drift_pixels": drift}))


def test_off_axis_rgb_clipping_and_state_isolation():
    endpoint = os.environ.get("LFS_LIGHTFIELD_TEST_ENDPOINT")
    if not endpoint:
        pytest.skip(
            "requires LFS_LIGHTFIELD_TEST_ENDPOINT pointing at a separate empty GUI"
        )
    _initialize(endpoint)
    _call(endpoint, "tools/list")
    _call(endpoint, "resources/list")
    for uri in (
        "lichtfeld://runtime/catalog",
        "lichtfeld://runtime/state",
        "lichtfeld://ui/state",
        "lichtfeld://scene/state",
        "lichtfeld://selection/current",
    ):
        _call(endpoint, "resources/read", {"uri": uri})
    script = (
        inspect.getsource(_verify_in_gui)
        + "\nimport lichtfeld as lf\nlf.ui.schedule_on_ui_thread(_verify_in_gui)\n"
    )
    _tool(
        endpoint,
        "editor_run",
        {"code": script, "show_console": False, "timeout_ms": 2000},
    )
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        output = _tool(endpoint, "editor_get_output", {"max_chars": 10000})
        text = str(output)
        assert "LIGHTFIELD_RENDER_FAIL" not in text, text
        if "LIGHTFIELD_RENDER_PASS" in text:
            return
        time.sleep(0.1)
    pytest.fail("Light-field GUI verification did not complete")

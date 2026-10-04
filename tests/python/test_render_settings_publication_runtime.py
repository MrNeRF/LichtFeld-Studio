# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Render-setting publication against an explicitly configured disposable viewer."""

import os
import time

import pytest

from test_render_on_demand_idle import _call, _initialize, _ledger, _tool

pytestmark = [pytest.mark.gpu, pytest.mark.integration]


@pytest.fixture(scope="module")
def endpoint():
    endpoint = os.environ.get("LFS_TEST_MCP_ENDPOINT")
    if not endpoint:
        pytest.skip("Requires an explicitly configured disposable MCP viewer")
    _initialize(endpoint)
    tools = {t["name"] for t in _call(endpoint, "tools/list")["tools"]}
    assert {"editor_run", "runtime_frame_ledger"} <= tools
    _call(endpoint, "resources/list")
    for uri in (
        "lichtfeld://runtime/catalog", "lichtfeld://runtime/state", "lichtfeld://ui/state",
        "lichtfeld://scene/state", "lichtfeld://selection/current",
    ):
        _call(endpoint, "resources/read", {"uri": uri})
    return endpoint


def run_python(endpoint, code):
    result = _tool(endpoint, "editor_run", {
        "code": "import lichtfeld as lf\nfrom lfs_plugins.ui.store import RuntimeState\n" + code,
        "show_console": False, "timeout_ms": 10000,
    })
    assert result["completed"] and not result["timed_out"], result
    assert "Traceback" not in result["output"]["text"], result
    return result


@pytest.mark.parametrize("field", [
    "show_grid", "render_scale", "background_color", "raster_backend",
    "scene_upscaler", "scene_upscaler_preset",
])
def test_binding_echo_does_not_publish_a_view_edit(endpoint, field):
    run_python(endpoint, f"""
settings = lf.get_render_settings()
value = settings.get({field!r})
before = (RuntimeState.scene_generation.value, RuntimeState.render_settings_generation.value)
for _ in range(100):
    settings.set({field!r}, value)
after = (RuntimeState.scene_generation.value, RuntimeState.render_settings_generation.value)
assert after == before, (before, after)
""")
    # GUI animation may present independently; the cached viewport must stay idle.
    time.sleep(1)
    before = _ledger(endpoint)["views_rendered"]
    time.sleep(.3)
    assert _ledger(endpoint)["views_rendered"] == before


def test_real_write_still_publishes_and_preserves_fresh_settings(endpoint):
    before = _ledger(endpoint)["views_rendered"]
    run_python(endpoint, """
retained = lf.get_render_settings()
grid = retained.show_grid
color = retained.background_color
try:
    lf.get_render_settings().background_color = [0.125, 0.25, 0.5]
    generation = RuntimeState.render_settings_generation.value
    retained.show_grid = not grid
    assert RuntimeState.render_settings_generation.value > generation
    current = lf.get_render_settings()
    assert current.show_grid == (not grid)
    assert list(current.background_color) == [0.125, 0.25, 0.5]
finally:
    lf.get_render_settings().show_grid = grid
    lf.get_render_settings().background_color = color
""")
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if _ledger(endpoint)["views_rendered"] > before:
            break
        time.sleep(.05)
    else:
        pytest.fail("A real settings edit left the cached viewport unrendered")


def test_invalid_write_still_fails_without_publication(endpoint):
    run_python(endpoint, """
settings = lf.get_render_settings()
before = (RuntimeState.scene_generation.value, RuntimeState.render_settings_generation.value)
try:
    settings.set("missing_render_property", 0)
except AttributeError:
    pass
else:
    raise AssertionError("Unknown property was accepted")
try:
    settings.set("render_scale", "invalid")
except (TypeError, ValueError, RuntimeError):
    pass
else:
    raise AssertionError("Invalid property value was accepted")
assert (RuntimeState.scene_generation.value, RuntimeState.render_settings_generation.value) == before
""")

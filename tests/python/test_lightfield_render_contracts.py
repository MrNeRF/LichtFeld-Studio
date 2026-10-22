# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Source-level guards for the isolated light-field rendering API."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _source(path):
    return (ROOT / path).read_text(encoding="utf-8")


def _render_body():
    source = _source("src/visualizer/rendering/rendering_manager_viewport.cpp")
    start = source.index("RenderingManager::renderLightFieldViewRgb8(")
    end = source.index("RenderingManager::renderPreviewImageRgba8(", start)
    return source[start:end]


def test_public_signature_and_viewer_thread_dispatch():
    stub = ast.parse(_source("src/python/stubs/lichtfeld/__init__.pyi"))
    function = next(
        node
        for node in stub.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "render_lightfield_view_u8"
    )
    assert [arg.arg for arg in function.args.args] == [
        "rotation",
        "translation",
        "width",
        "height",
        "fov",
        "shift_x",
        "near_distance",
        "far_distance",
    ]
    source = _source("src/python/lfs/py_rendering.cpp")
    body = source[source.index('m.def("render_lightfield_view_u8"') :]
    body = body[: body.index('m.def("compute_screen_positions"')]
    assert "renderLightFieldViewRgb8" in body
    assert "post_work_and_wait" in body and "gil_scoped_release" in body
    assert "acceptsPostedWork" in body


def test_view_size_and_projection_validation_precedes_scene_access():
    body = _render_body()
    for guard in (
        "width > 2048",
        "height > 2048",
        "std::isfinite(fov_degrees)",
        "std::isfinite(principal_shift_x)",
        "std::isfinite(near_distance)",
        "std::isfinite(far_distance)",
        "far_distance <= near_distance",
    ):
        assert guard in body
        assert body.index(guard) < body.index("acquireLiveModelRenderLock")


def test_principal_point_and_clipping_are_local_overrides():
    body = _render_body()
    assert "auto settings = getSettings()" in body
    assert "setSettings(" not in body and "updateSettings(" not in body
    assert "width * 0.5f + principal_shift_x" in body
    assert "hide_outside_depth_box = true" in body
    assert "-far_distance" in body and "-near_distance" in body
    assert "&settings" in body
    source = _source("src/visualizer/rendering/rendering_manager_viewport.cpp")
    assert "settings_override ? *settings_override : getSettings()" in source


def test_readback_is_uint8_and_releases_preview_resources():
    body = _render_body()
    assert "readOutputImageRgb8" in body
    assert "OutputSlot::Preview" in body
    error_path = body[body.index("if (!rendered)") : body.index("auto image =")]
    assert "releasePreviewImageResources()" in error_path
    readback_path = body[body.index("auto image =") :]
    assert readback_path.index("releasePreviewImageResources()") < readback_path.index(
        "if (!image)"
    )

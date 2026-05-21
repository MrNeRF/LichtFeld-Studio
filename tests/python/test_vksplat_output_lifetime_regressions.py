# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression checks for VkSplat viewport output lifetime hazards."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _read(rel_path: str) -> str:
    return (PROJECT_ROOT / rel_path).read_text(encoding="utf-8")


def test_vksplat_output_resize_waits_before_destroying_gui_sampled_images():
    source = _read("src/visualizer/rendering/vksplat_viewport_renderer.cpp")
    function_start = source.index("VksplatViewportRenderer::ensureOutputImages")
    function_end = source.index(
        "std::expected<void, std::string> VksplatViewportRenderer::ensureComposePipeline",
        function_start,
    )
    body = source[function_start:function_end]

    wait_pos = body.index("context.waitForSubmittedFrames()")
    destroy_color_pos = body.index("context.destroyExternalImage(slot.image)")
    destroy_depth_pos = body.index("context.destroyExternalImage(slot.depth_image)")

    assert "replacing_existing_output" in body
    assert wait_pos < destroy_color_pos
    assert wait_pos < destroy_depth_pos

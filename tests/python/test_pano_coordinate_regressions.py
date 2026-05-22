# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression checks for viewer-side equirectangular coordinate conventions."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _read(rel_path: str) -> str:
    return (PROJECT_ROOT / rel_path).read_text(encoding="utf-8")


def test_vksplat_viewer_rejects_equirectangular_until_native_support_exists():
    source = _read("src/visualizer/rendering/vksplat_viewport_renderer.cpp")

    assert "VkSplat forward path supports pinhole cameras, not equirectangular cameras" in source
    assert "VkSplat selection query supports pinhole cameras, not equirectangular cameras" in source


def test_viewer_equirectangular_software_projection_uses_rasterizer_mapping():
    source = _read("src/rendering/raster_rendering_engine.cpp")

    assert "const float u = 0.5f + std::atan2(dir.x, -dir.z) / (2.0f * glm::pi<float>());" in source
    assert "const float v = 0.5f + std::asin(std::clamp(dir.y, -1.0f, 1.0f)) / glm::pi<float>();" in source
    assert "const float py = v * static_cast<float>(height - 1);" in source


def test_gpu_environment_pass_uses_top_down_viewport_coordinates():
    source = _read("src/visualizer/rendering/resources/viewport/environment.frag")

    assert "vec2 viewport_uv = vec2(TexCoord.x, 1.0 - TexCoord.y);" in source
    assert "float lat = (viewport_uv.y - 0.5) * PI;" in source
    assert "vec2 pixel = viewport_uv * viewport;" in source

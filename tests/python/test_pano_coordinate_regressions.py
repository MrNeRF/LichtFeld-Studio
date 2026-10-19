# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression checks for viewer-side equirectangular coordinate conventions."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _read(rel_path: str) -> str:
    return (PROJECT_ROOT / rel_path).read_text(encoding="utf-8")


def test_vksplat_viewer_supports_equirectangular_through_gut():
    source = _read("src/visualizer/rendering/vksplat_viewport_renderer.cpp")

    assert "VkSplat forward path supports pinhole cameras, not equirectangular cameras" not in source
    assert "VkSplat selection query supports pinhole cameras, not equirectangular cameras" not in source
    assert "kVkSplatCameraModelEquirectangular = 3u" in source
    assert "request.equirectangular && !request.gut" in source
    assert "VkSplat equirectangular rendering requires the 3DGUT backend" in source
    assert "VkSplat equirectangular selection requires the 3DGUT backend" in source
    projection = _read("src/rendering/rasterizer/vulkan/shader/src/slang/equirect_projection.inc")
    assert "LFS_EQUIRECT_ATAN2(direction.x, -direction.z)" in projection
    assert "0.5 - latitude / LFS_EQUIRECT_PI" in projection


def test_vksplat_equirectangular_shaders_project_rays_and_wrap_tiles():
    utils = _read("src/rendering/rasterizer/vulkan/shader/src/slang/utils.slang")
    vertex = _read("src/rendering/rasterizer/vulkan/shader/src/slang/vertex_shader.slang")
    tile = _read("src/rendering/rasterizer/vulkan/shader/src/slang/tile_shader.slang")
    selection = _read("src/rendering/rasterizer/vulkan/shader/src/slang/selection_mask.slang")

    assert "EQUIRECTANGULAR = 3" in utils
    assert "project_point_equirect" in utils
    assert "lfs_equirect_uv(float3(dir.x, -dir.y, -dir.z))" in utils
    assert "unwrap_image_points_x(image_points, cam)" in vertex
    assert "covariance_mean2d" in vertex
    assert "get_wrapped_rectangle_tile_space" in vertex
    assert "base_camera_model(uniforms.camera_model) == uint(CameraModel::EQUIRECTANGULAR)" in tile
    assert "% grid_width" in tile
    assert "unwrap_image_points_x(image_points, cam)" in selection


def test_viewer_equirectangular_software_projection_uses_rasterizer_mapping():
    source = _read("src/rendering/raster_rendering_engine.cpp")
    mapping = _read("src/rendering/environment_math.hpp")

    assert "envmath::equirectUvForDirection(envmath::normalized(rotated))" in source
    assert "const float longitude = std::atan2(world_dir.x, -world_dir.z);" in mapping
    assert "const float latitude = std::asin(clampf(world_dir.y, -1.0f, 1.0f));" in mapping
    assert "return {longitude / (2.0f * kPi) + 0.5f, 0.5f - latitude / kPi};" in mapping
    assert "const float y = v * static_cast<float>(height - 1);" in mapping


def test_gpu_environment_pass_uses_top_down_viewport_coordinates():
    source = _read("src/visualizer/rendering/resources/viewport/environment.frag")

    assert "vec2 viewport_uv = vec2(TexCoord.x, 1.0 - TexCoord.y);" in source
    assert "return lfs_equirect_direction(vec2(viewport_uv.x, 1.0 - viewport_uv.y));" in source
    assert "return lfs_equirect_uv(dir);" in source
    assert "vec2 pixel = viewport_uv * viewport;" in source


def test_video_export_forwards_panorama_to_every_gaussian_readback():
    source = _read("src/visualizer/gui/async_task_manager.cpp")
    for method in ("renderPreviewImage", "renderPreviewImageRgb8", "renderPreviewImageRgba8"):
        call = source.index(f"rendering_manager.{method}(")
        trailing_arguments = source.index("height", call)
        end = source.index(")", trailing_arguments)
        assert "render_settings.equirectangular" in source[trailing_arguments:end], method

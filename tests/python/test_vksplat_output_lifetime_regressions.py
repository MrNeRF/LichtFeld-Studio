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


def test_point_cloud_vulkan_viewport_capture_uses_readback():
    header = _read("src/visualizer/rendering/point_cloud_vulkan_renderer.hpp")
    source = _read("src/visualizer/rendering/point_cloud_vulkan_renderer.cpp")
    manager = _read("src/visualizer/rendering/rendering_manager_vulkan.cpp")

    assert "readOutputImage(" in header
    assert "vkCmdCopyImageToBuffer" in source
    assert "VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL" in source
    assert "PointCloudVulkanRenderer::readOutputImage" in source

    lazy_capture_start = manager.index("point_cloud_vulkan_renderer_->readOutputImage")
    lazy_capture_body = manager[lazy_capture_start - 500:lazy_capture_start + 500]
    assert "Failed to capture point-cloud Vulkan viewport image" in lazy_capture_body
    assert "[]() -> std::shared_ptr<lfs::core::Tensor> { return {}; }" not in lazy_capture_body


def test_python_render_view_uses_vulkan_preview_renderer():
    source = _read("src/python/lfs/py_rendering.cpp")
    function_start = source.index("std::optional<PyTensor> render_view(")
    function_end = source.index("std::optional<PyTensor> compute_screen_positions", function_start)
    body = source[function_start:function_end]

    assert "renderViewThreadSafe" in body
    assert "rendering::flipImageVertical" not in body
    assert "detectImageLayout" not in body
    assert "image->size(2) != 3" in body
    assert "(void)rotation;" not in body
    assert "(void)translation;" not in body

    helper_start = source.index("rendering_manager->renderPreviewImage")
    helper_body = source[helper_start - 500:helper_start + 500]
    assert "lfs::rendering::vFovToFocalLength" in helper_body
    assert "scene_manager" in helper_body


def test_vulkan_preview_render_view_uses_native_device_limits_not_16k_policy():
    header = _read("src/visualizer/rendering/rendering_manager.hpp")
    source = _read("src/visualizer/rendering/rendering_manager_viewport.cpp")
    context = _read("src/visualizer/window/vulkan_context.cpp")
    constants = _read("src/rendering/include/rendering/render_constants.hpp")

    assert "MAX_VIEWPORT_SIZE" not in constants
    assert "16384" not in source

    assert "VksplatViewportRenderer::OutputSlot::Preview" in source
    assert "format_properties.imageFormatProperties.maxExtent" in context
    assert "exceeds device-supported limit" in context
    assert "kMaxNativePreviewPixelStateBytes" in source
    assert "renderPreviewImageTiledWithState" in header
    assert "renderPreviewImageTiledWithState" in source
    assert "request.frame_view.intrinsics_override" in source
    assert "copyPreviewTileToOutput" not in source
    assert "readOutputImageIntoCpuHwc" in source
    assert "lfs::core::Tensor::empty" in source


def test_python_render_view_has_uint8_export_path():
    header = _read("src/python/lfs/py_rendering.hpp")
    source = _read("src/python/lfs/py_rendering.cpp")
    stub = _read("src/python/stubs/lichtfeld/__init__.pyi")

    assert "render_view_u8" in header
    assert "render_view_u8" in source
    assert "renderPreviewImageRgb8" in source
    assert "core::DataType::UInt8" in source
    assert "releasePreviewImageResources" in source
    assert "def render_view_u8" in stub


def test_vksplat_preview_export_releases_transient_resources():
    header = _read("src/visualizer/rendering/vksplat_viewport_renderer.hpp")
    source = _read("src/visualizer/rendering/vksplat_viewport_renderer.cpp")
    manager = _read("src/visualizer/rendering/rendering_manager_viewport.cpp")

    assert "releasePreviewResources" in header
    assert "releaseOutputSlot(OutputSlot::Preview)" in source
    assert "releasePrivateScratchBuffers()" in source
    assert "releaseSharedScratchArena()" in source
    assert "logVramBreakdownIfChanged(\"preview_release\")" in source
    assert "releasePreviewImageResources" in manager


def test_gaussian_video_export_uses_vulkan_preview_renderer():
    source = _read("src/visualizer/gui/async_task_manager.cpp")
    function_start = source.index("std::expected<lfs::core::Tensor, std::string> renderVideoExportFrame(")
    function_end = source.index("AsyncTaskManager::AsyncTaskManager", function_start)
    body = source[function_start:function_end]

    assert "Gaussian video export needs a Vulkan offscreen export path" not in body
    assert "rendering_manager.renderPreviewImage" in body
    assert "makeGaussianPreviewVideoFrame" in body
    assert "materializeGpuFrame" in body

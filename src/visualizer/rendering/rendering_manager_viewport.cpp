/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/logger.hpp"
#include "model_renderability.hpp"
#include "rendering/coordinate_conventions.hpp"
#include "rendering/viewport_request_builder.hpp"
#include "rendering_manager.hpp"
#include "scene/scene_manager.hpp"
#include "scene/scene_render_state.hpp"
#include "training/trainer.hpp"
#include "training/training_manager.hpp"
#include "vksplat_viewport_renderer.hpp"
#include <algorithm>
#include <cmath>
#include <format>
#include <shared_mutex>
#include <string_view>
#include <utility>
#include <vector>

namespace lfs::vis {

    namespace {
        constexpr std::size_t kPreviewPixelStateBytesPerPixel = 4u * sizeof(float);
        constexpr std::size_t kMaxNativePreviewPixelStateBytes =
            (std::size_t{4} << 30) - (std::size_t{64} << 20);
        constexpr float kMaxValidDepth = 1e9f;
        constexpr int kMinPreviewSubdivisionHeight = 512;

        [[nodiscard]] bool isTileInstanceOverflow(const std::string_view error) {
            return error.find("tile-instance prefix sum overflowed signed 32-bit capacity") !=
                   std::string_view::npos;
        }
        [[nodiscard]] std::optional<std::shared_lock<std::shared_mutex>> acquireLiveModelRenderLock(
            const SceneManager* const scene_manager) {
            std::optional<std::shared_lock<std::shared_mutex>> lock;
            if (const auto* tm = scene_manager ? scene_manager->getTrainerManager() : nullptr) {
                if (const auto* trainer = tm->getTrainer()) {
                    lock.emplace(trainer->getRenderMutex());
                }
            }
            return lock;
        }

        [[nodiscard]] bool previewRenderNeedsTiling(const int width, const int height) {
            if (width <= 0 || height <= 0) {
                return false;
            }
            const auto pixel_count = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
            return pixel_count > kMaxNativePreviewPixelStateBytes / kPreviewPixelStateBytesPerPixel;
        }

        [[nodiscard]] int previewTileHeightForWidth(const int width) {
            if (width <= 0) {
                return 1;
            }
            const std::size_t max_pixels = kMaxNativePreviewPixelStateBytes / kPreviewPixelStateBytesPerPixel;
            return std::max(1, static_cast<int>(max_pixels / static_cast<std::size_t>(width)));
        }

        [[nodiscard]] std::optional<float> sampleDepthTensorAt(
            const lfs::core::Tensor& depth,
            const glm::ivec2& pixel) {
            if (!depth.is_valid() ||
                depth.device() != lfs::core::Device::CPU ||
                depth.dtype() != lfs::core::DataType::Float32 ||
                depth.ndim() != 2 ||
                !depth.is_contiguous()) {
                return std::nullopt;
            }
            const int width = static_cast<int>(depth.size(1));
            const int height = static_cast<int>(depth.size(0));
            if (width <= 0 || height <= 0 ||
                pixel.x < 0 || pixel.y < 0 ||
                pixel.x >= width || pixel.y >= height) {
                return std::nullopt;
            }
            const float* const values = depth.ptr<float>();
            if (!values) {
                return std::nullopt;
            }
            const float value = values[static_cast<std::size_t>(pixel.y) * static_cast<std::size_t>(width) +
                                       static_cast<std::size_t>(pixel.x)];
            if (!std::isfinite(value) || value <= 0.0f || value >= kMaxValidDepth) {
                return std::nullopt;
            }
            return value;
        }

        [[nodiscard]] lfs::rendering::CameraIntrinsics previewTileIntrinsics(
            const int full_width,
            const int full_height,
            const float focal_length_mm) {
            const auto [fx, fy] = lfs::rendering::computePixelFocalLengths(
                {full_width, full_height},
                focal_length_mm);
            return lfs::rendering::CameraIntrinsics{
                .focal_x = fx,
                .focal_y = fy,
                .center_x = static_cast<float>(full_width) * 0.5f,
                .center_y = static_cast<float>(full_height) * 0.5f,
            };
        }

    } // namespace

    RenderingManager::ContentBounds RenderingManager::getContentBounds(const glm::ivec2& viewport_size) const {
        const int viewport_width = std::max(viewport_size.x, 0);
        const int viewport_height = std::max(viewport_size.y, 0);
        ContentBounds bounds{
            0.0f,
            0.0f,
            static_cast<float>(viewport_width),
            static_cast<float>(viewport_height),
            false};

        if (split_view_service_.isGTComparisonActive(settings_)) {
            glm::ivec2 content_dims{0, 0};
            if (const auto service_dims = split_view_service_.gtContentDimensions()) {
                content_dims = *service_dims;
            } else {
                content_dims = vulkan_gt_comparison_content_size_;
            }
            if (content_dims.x <= 0 || content_dims.y <= 0 ||
                viewport_width <= 0 || viewport_height <= 0) {
                return bounds;
            }

            const float content_aspect = static_cast<float>(content_dims.x) / content_dims.y;
            const float viewport_aspect = static_cast<float>(viewport_width) / viewport_height;

            if (content_aspect > viewport_aspect) {
                const int content_height =
                    std::clamp(static_cast<int>(std::lround(static_cast<float>(viewport_width) / content_aspect)),
                               1,
                               viewport_height);
                bounds.width = static_cast<float>(viewport_width);
                bounds.height = static_cast<float>(content_height);
                bounds.x = 0.0f;
                bounds.y = static_cast<float>(std::max((viewport_height - content_height) / 2, 0));
            } else {
                const int content_width =
                    std::clamp(static_cast<int>(std::lround(static_cast<float>(viewport_height) * content_aspect)),
                               1,
                               viewport_width);
                bounds.height = static_cast<float>(viewport_height);
                bounds.width = static_cast<float>(content_width);
                bounds.x = static_cast<float>(std::max((viewport_width - content_width) / 2, 0));
                bounds.y = 0.0f;
            }
            bounds.letterboxed = true;
        }
        return bounds;
    }

    std::optional<RenderingManager::MutableViewerPanelInfo> RenderingManager::resolveViewerPanel(
        Viewport& primary_viewport,
        const glm::vec2& viewport_pos,
        const glm::vec2& viewport_size,
        const std::optional<glm::vec2> screen_point,
        const std::optional<SplitViewPanelId> panel_override) {
        const glm::ivec2 rendered_size = getRenderedSize();
        const int full_render_width =
            rendered_size.x > 0 ? rendered_size.x : std::max(static_cast<int>(viewport_size.x), 1);
        const int full_render_height =
            rendered_size.y > 0 ? rendered_size.y : std::max(static_cast<int>(viewport_size.y), 1);

        MutableViewerPanelInfo info{
            .panel = SplitViewPanelId::Left,
            .viewport = &primary_viewport,
            .x = viewport_pos.x,
            .y = viewport_pos.y,
            .width = viewport_size.x,
            .height = viewport_size.y,
            .render_width = full_render_width,
            .render_height = full_render_height,
        };

        const auto screen_layouts = split_view_service_.panelLayouts(
            settings_,
            std::max(static_cast<int>(viewport_size.x), 1));
        if (!screen_layouts || viewport_size.x <= 1.0f) {
            return info.valid() ? std::optional<MutableViewerPanelInfo>(info) : std::nullopt;
        }

        const auto render_layouts = split_view_service_.panelLayouts(settings_, full_render_width);
        if (!render_layouts) {
            return info.valid() ? std::optional<MutableViewerPanelInfo>(info) : std::nullopt;
        }

        SplitViewPanelId panel = panel_override.value_or(split_view_service_.focusedPanel());
        if (screen_point && !panel_override) {
            const float divider_x = viewport_pos.x + (*screen_layouts)[0].width;
            panel = screen_point->x >= divider_x ? SplitViewPanelId::Right : SplitViewPanelId::Left;
        }

        const size_t index = splitViewPanelIndex(panel);
        info.panel = panel;
        info.viewport = (panel == SplitViewPanelId::Right)
                            ? &split_view_service_.secondaryViewport()
                            : &primary_viewport;
        info.x = viewport_pos.x + static_cast<float>((*screen_layouts)[index].x);
        info.y = viewport_pos.y;
        info.width = static_cast<float>((*screen_layouts)[index].width);
        info.height = viewport_size.y;
        info.render_width = std::max((*render_layouts)[index].width, 1);
        info.render_height = full_render_height;
        return info.valid() ? std::optional<MutableViewerPanelInfo>(info) : std::nullopt;
    }

    std::optional<RenderingManager::ViewerPanelInfo> RenderingManager::resolveViewerPanel(
        const Viewport& primary_viewport,
        const glm::vec2& viewport_pos,
        const glm::vec2& viewport_size,
        const std::optional<glm::vec2> screen_point,
        const std::optional<SplitViewPanelId> panel_override) const {
        const glm::ivec2 rendered_size = getRenderedSize();
        const int full_render_width =
            rendered_size.x > 0 ? rendered_size.x : std::max(static_cast<int>(viewport_size.x), 1);
        const int full_render_height =
            rendered_size.y > 0 ? rendered_size.y : std::max(static_cast<int>(viewport_size.y), 1);

        ViewerPanelInfo info{
            .panel = SplitViewPanelId::Left,
            .viewport = &primary_viewport,
            .x = viewport_pos.x,
            .y = viewport_pos.y,
            .width = viewport_size.x,
            .height = viewport_size.y,
            .render_width = full_render_width,
            .render_height = full_render_height,
        };

        const auto screen_layouts = split_view_service_.panelLayouts(
            settings_,
            std::max(static_cast<int>(viewport_size.x), 1));
        if (!screen_layouts || viewport_size.x <= 1.0f) {
            return info.valid() ? std::optional<ViewerPanelInfo>(info) : std::nullopt;
        }

        const auto render_layouts = split_view_service_.panelLayouts(settings_, full_render_width);
        if (!render_layouts) {
            return info.valid() ? std::optional<ViewerPanelInfo>(info) : std::nullopt;
        }

        SplitViewPanelId panel = panel_override.value_or(split_view_service_.focusedPanel());
        if (screen_point && !panel_override) {
            const float divider_x = viewport_pos.x + (*screen_layouts)[0].width;
            panel = screen_point->x >= divider_x ? SplitViewPanelId::Right : SplitViewPanelId::Left;
        }

        const size_t index = splitViewPanelIndex(panel);
        info.panel = panel;
        info.viewport = (panel == SplitViewPanelId::Right)
                            ? &split_view_service_.secondaryViewport()
                            : &primary_viewport;
        info.x = viewport_pos.x + static_cast<float>((*screen_layouts)[index].x);
        info.y = viewport_pos.y;
        info.width = static_cast<float>((*screen_layouts)[index].width);
        info.height = viewport_size.y;
        info.render_width = std::max((*render_layouts)[index].width, 1);
        info.render_height = full_render_height;
        return info.valid() ? std::optional<ViewerPanelInfo>(info) : std::nullopt;
    }

    lfs::rendering::RenderingEngine* RenderingManager::getRenderingEngine() {
        // The Vulkan path sets initialized_ without creating engine_ — it only
        // builds the auxiliary CPU engine lazily for point-cloud renders. Other
        // callers (camera frustum picking, masked depth queries) still need it,
        // so create it on demand here too.
        if (!engine_) {
            initialize();
        }
        return engine_.get();
    }

    std::shared_ptr<lfs::core::Tensor> RenderingManager::getViewportImageIfAvailable() const {
        return viewport_artifact_service_.getCapturedImageIfCurrent();
    }

    std::shared_ptr<lfs::core::Tensor> RenderingManager::captureViewportImage() {
        if (viewport_artifact_service_.hasLazyCapture()) {
            return viewport_artifact_service_.resolveLazyCapture();
        }

        if (auto image = getViewportImageIfAvailable()) {
            return image;
        }

        if (!engine_ || !viewport_artifact_service_.hasGpuFrame()) {
            return {};
        }

        std::optional<std::shared_lock<std::shared_mutex>> render_lock;
        if (const auto* tm = viewport_interaction_context_.scene_manager
                                 ? viewport_interaction_context_.scene_manager->getTrainerManager()
                                 : nullptr) {
            if (const auto* trainer = tm->getTrainer()) {
                render_lock.emplace(trainer->getRenderMutex());
            }
        }

        auto readback_result = engine_->readbackGpuFrameColor(*viewport_artifact_service_.gpuFrame());
        if (!readback_result) {
            LOG_ERROR("Failed to capture viewport image from GPU frame: {}", readback_result.error());
            return {};
        }

        viewport_artifact_service_.storeCapturedImage(*readback_result);
        return viewport_artifact_service_.getCapturedImageIfCurrent();
    }

    int RenderingManager::pickCameraFrustum(const glm::vec2& mouse_pos) {
        const int previous_hovered_camera = camera_interaction_service_.hoveredCameraId();
        bool hover_changed = false;
        auto* const engine = getRenderingEngine();
        const int hovered_camera = camera_interaction_service_.pickCameraFrustum(
            engine,
            viewport_interaction_context_.scene_manager,
            viewport_interaction_context_,
            settings_,
            mouse_pos,
            hover_changed);

        if (hover_changed) {
            LOG_DEBUG("Camera hover changed: {} -> {}", previous_hovered_camera, hovered_camera);
            markDirty(DirtyFlag::OVERLAY);
        }

        return hovered_camera;
    }

    RenderingManager::PreviewImageReadbackConfig RenderingManager::previewImageReadbackConfig(
        const PreviewImageReadback readback,
        const bool has_background_color_override) {
        PreviewImageReadbackConfig config{};
        switch (readback) {
        case PreviewImageReadback::FloatRgb:
            config.dtype = lfs::core::DataType::Float32;
            config.channels = 3;
            break;
        case PreviewImageReadback::UInt8Rgb:
            config.dtype = lfs::core::DataType::UInt8;
            config.channels = 3;
            break;
        case PreviewImageReadback::UInt8Rgba:
            config.dtype = lfs::core::DataType::UInt8;
            config.channels = 4;
            config.transparent_background_override = true;
            break;
        }
        if (has_background_color_override && !config.transparent_background_override.has_value()) {
            config.transparent_background_override = false;
        }
        return config;
    }

    std::shared_ptr<lfs::core::Tensor> RenderingManager::renderPreviewImage(SceneManager* const scene_manager,
                                                                            const glm::mat3& rotation,
                                                                            const glm::vec3& position,
                                                                            const float focal_length_mm,
                                                                            const int width,
                                                                            const int height,
                                                                            std::optional<glm::vec3> background_color_override,
                                                                            std::optional<bool> orthographic_override,
                                                                            std::optional<float> ortho_scale_override) {
        if (width <= 0 || height <= 0) {
            return {};
        }
        auto render_lock = acquireLiveModelRenderLock(scene_manager);
        auto render_state = scene_manager ? scene_manager->buildRenderState() : SceneRenderState{};
        const auto* const model = render_state.combined_model;
        if (!hasRenderableGaussians(model)) {
            return {};
        }

        if (previewRenderNeedsTiling(width, height)) {
            return renderPreviewImageTiledWithState(
                scene_manager,
                *model,
                std::move(render_state),
                rotation,
                position,
                focal_length_mm,
                width,
                height,
                render_lock.has_value(),
                background_color_override,
                orthographic_override,
                ortho_scale_override,
                PreviewImageReadback::FloatRgb);
        }

        return renderPreviewImageWithState(
            scene_manager,
            *model,
            std::move(render_state),
            rotation,
            position,
            focal_length_mm,
            width,
            height,
            render_lock.has_value(),
            std::nullopt,
            orthographic_override,
            ortho_scale_override,
            background_color_override,
            PreviewImageReadback::FloatRgb);
    }

    std::expected<void, std::string> RenderingManager::renderDepthCaptureToPreviewSlotWithState(
        SceneManager* const scene_manager,
        const lfs::core::SplatData& model,
        SceneRenderState scene_state,
        const glm::mat3& rotation,
        const glm::vec3& position,
        const float focal_length_mm,
        const int width,
        const int height,
        const bool render_lock_held,
        const bool expected_depth,
        std::optional<glm::vec3> background_color_override,
        std::optional<bool> orthographic_override,
        std::optional<float> ortho_scale_override) {
        if (width <= 0 || height <= 0) {
            return std::unexpected("invalid preview depth render dimensions");
        }
        if (!hasRenderableGaussians(&model)) {
            return std::unexpected("no renderable Gaussian model is available");
        }
        if (previewRenderNeedsTiling(width, height)) {
            // Depth comes from the single per-frame pixel_depth scratch; a tiled
            // render would need per-tile depth assembly, which isn't wired yet.
            LOG_WARN("render_view depth unavailable for tiled preview size {}x{}", width, height);
            return std::unexpected("preview depth render would require tiled depth assembly");
        }

        // The macro-tile (HiGS) chain only yields per-macro-tile median depth;
        // force the legacy per-pixel chain for the depth-capture render so the
        // readback matches the image resolution.
        if (!vksplat_viewport_renderer_) {
            vksplat_viewport_renderer_ = std::make_unique<VksplatViewportRenderer>();
        }
        vksplat_viewport_renderer_->setDepthCaptureMode(true, expected_depth);
        struct DepthCaptureModeGuard {
            VksplatViewportRenderer* renderer;
            ~DepthCaptureModeGuard() { renderer->setDepthCaptureMode(false); }
        } depth_capture_guard{vksplat_viewport_renderer_.get()};

        auto rendered = renderPreviewImageToPreviewSlotWithState(
            scene_manager,
            model,
            scene_state,
            rotation,
            position,
            focal_length_mm,
            width,
            height,
            render_lock_held,
            std::nullopt,
            {},
            {},
            orthographic_override,
            ortho_scale_override,
            background_color_override,
            std::nullopt);
        return rendered;
    }

    RenderingManager::PreviewRgbd RenderingManager::renderPreviewImageAndDepth(
        SceneManager* const scene_manager,
        const glm::mat3& rotation,
        const glm::vec3& position,
        const float focal_length_mm,
        const int width,
        const int height,
        const bool expected_depth,
        std::optional<glm::vec3> background_color_override) {
        PreviewRgbd result{};
        if (width <= 0 || height <= 0) {
            return result;
        }
        auto render_lock = acquireLiveModelRenderLock(scene_manager);
        auto render_state = scene_manager ? scene_manager->buildRenderState() : SceneRenderState{};
        const auto* const model = render_state.combined_model;
        if (!hasRenderableGaussians(model)) {
            return result;
        }

        auto rendered = renderDepthCaptureToPreviewSlotWithState(
            scene_manager,
            *model,
            std::move(render_state),
            rotation,
            position,
            focal_length_mm,
            width,
            height,
            render_lock.has_value(),
            expected_depth,
            background_color_override,
            std::nullopt,
            std::nullopt);
        if (!rendered) {
            LOG_ERROR("Gaussian preview rgbd render failed: {}", rendered.error());
            return result;
        }

        // image and depth are read from the same render: the Preview output slot
        // and the pixel_depth scratch it just wrote (still resident — the Preview
        // path uses private scratch, which render() does not release).
        auto image = vksplat_viewport_renderer_->readOutputImage(
            *last_vulkan_context_, VksplatViewportRenderer::OutputSlot::Preview);
        if (!image) {
            LOG_ERROR("Gaussian preview rgbd image readback failed: {}", image.error());
            return result;
        }
        auto depth = vksplat_viewport_renderer_->readPreviewDepth(
            *last_vulkan_context_, VksplatViewportRenderer::OutputSlot::Preview);
        if (!depth) {
            LOG_ERROR("Gaussian preview depth readback failed: {}", depth.error());
            return result;
        }
        result.image = std::move(*image);
        result.depth = std::move(*depth);
        return result;
    }

    std::shared_ptr<lfs::core::Tensor> RenderingManager::renderPreviewImageRgb8(SceneManager* const scene_manager,
                                                                                const glm::mat3& rotation,
                                                                                const glm::vec3& position,
                                                                                const float focal_length_mm,
                                                                                const int width,
                                                                                const int height,
                                                                                std::optional<glm::vec3> background_color_override,
                                                                                std::optional<bool> orthographic_override,
                                                                                std::optional<float> ortho_scale_override) {
        if (width <= 0 || height <= 0) {
            return {};
        }
        auto render_lock = acquireLiveModelRenderLock(scene_manager);
        auto render_state = scene_manager ? scene_manager->buildRenderState() : SceneRenderState{};
        const auto* const model = render_state.combined_model;
        if (!hasRenderableGaussians(model)) {
            return {};
        }

        if (previewRenderNeedsTiling(width, height)) {
            return renderPreviewImageTiledWithState(
                scene_manager,
                *model,
                std::move(render_state),
                rotation,
                position,
                focal_length_mm,
                width,
                height,
                render_lock.has_value(),
                background_color_override,
                orthographic_override,
                ortho_scale_override,
                PreviewImageReadback::UInt8Rgb);
        }

        return renderPreviewImageWithState(
            scene_manager,
            *model,
            std::move(render_state),
            rotation,
            position,
            focal_length_mm,
            width,
            height,
            render_lock.has_value(),
            std::nullopt,
            orthographic_override,
            ortho_scale_override,
            background_color_override,
            PreviewImageReadback::UInt8Rgb);
    }

    std::shared_ptr<lfs::core::Tensor> RenderingManager::renderPreviewImageRgba8(SceneManager* const scene_manager,
                                                                                 const glm::mat3& rotation,
                                                                                 const glm::vec3& position,
                                                                                 const float focal_length_mm,
                                                                                 const int width,
                                                                                 const int height,
                                                                                 std::optional<bool> orthographic_override,
                                                                                 std::optional<float> ortho_scale_override) {
        if (width <= 0 || height <= 0) {
            return {};
        }
        auto render_lock = acquireLiveModelRenderLock(scene_manager);
        auto render_state = scene_manager ? scene_manager->buildRenderState() : SceneRenderState{};
        const auto* const model = render_state.combined_model;
        if (!hasRenderableGaussians(model)) {
            return {};
        }

        if (previewRenderNeedsTiling(width, height)) {
            return renderPreviewImageTiledWithState(
                scene_manager,
                *model,
                std::move(render_state),
                rotation,
                position,
                focal_length_mm,
                width,
                height,
                render_lock.has_value(),
                std::nullopt,
                orthographic_override,
                ortho_scale_override,
                PreviewImageReadback::UInt8Rgba);
        }

        return renderPreviewImageWithState(
            scene_manager,
            *model,
            std::move(render_state),
            rotation,
            position,
            focal_length_mm,
            width,
            height,
            render_lock.has_value(),
            std::nullopt,
            orthographic_override,
            ortho_scale_override,
            std::nullopt,
            PreviewImageReadback::UInt8Rgba);
    }

    std::shared_ptr<lfs::core::Tensor> RenderingManager::renderPreviewImage(const lfs::core::SplatData& model,
                                                                            SceneRenderState scene_state,
                                                                            const glm::mat3& rotation,
                                                                            const glm::vec3& position,
                                                                            const float focal_length_mm,
                                                                            const int width,
                                                                            const int height,
                                                                            std::optional<glm::vec3> background_color_override,
                                                                            std::optional<bool> orthographic_override,
                                                                            std::optional<float> ortho_scale_override) {
        if (width <= 0 || height <= 0) {
            return {};
        }
        if (previewRenderNeedsTiling(width, height)) {
            return renderPreviewImageTiledWithState(
                nullptr,
                model,
                std::move(scene_state),
                rotation,
                position,
                focal_length_mm,
                width,
                height,
                false,
                background_color_override,
                orthographic_override,
                ortho_scale_override,
                PreviewImageReadback::FloatRgb);
        }

        return renderPreviewImageWithState(
            nullptr,
            model,
            std::move(scene_state),
            rotation,
            position,
            focal_length_mm,
            width,
            height,
            false,
            std::nullopt,
            orthographic_override,
            ortho_scale_override,
            background_color_override,
            PreviewImageReadback::FloatRgb);
    }

    std::shared_ptr<lfs::core::Tensor> RenderingManager::renderPreviewImageRgb8(const lfs::core::SplatData& model,
                                                                                SceneRenderState scene_state,
                                                                                const glm::mat3& rotation,
                                                                                const glm::vec3& position,
                                                                                const float focal_length_mm,
                                                                                const int width,
                                                                                const int height,
                                                                                std::optional<glm::vec3> background_color_override,
                                                                                std::optional<bool> orthographic_override,
                                                                                std::optional<float> ortho_scale_override) {
        if (width <= 0 || height <= 0) {
            return {};
        }
        if (previewRenderNeedsTiling(width, height)) {
            return renderPreviewImageTiledWithState(
                nullptr,
                model,
                std::move(scene_state),
                rotation,
                position,
                focal_length_mm,
                width,
                height,
                false,
                background_color_override,
                orthographic_override,
                ortho_scale_override,
                PreviewImageReadback::UInt8Rgb);
        }

        return renderPreviewImageWithState(
            nullptr,
            model,
            std::move(scene_state),
            rotation,
            position,
            focal_length_mm,
            width,
            height,
            false,
            std::nullopt,
            orthographic_override,
            ortho_scale_override,
            background_color_override,
            PreviewImageReadback::UInt8Rgb);
    }

    std::shared_ptr<lfs::core::Tensor> RenderingManager::renderPreviewImageRgba8(const lfs::core::SplatData& model,
                                                                                 SceneRenderState scene_state,
                                                                                 const glm::mat3& rotation,
                                                                                 const glm::vec3& position,
                                                                                 const float focal_length_mm,
                                                                                 const int width,
                                                                                 const int height,
                                                                                 std::optional<bool> orthographic_override,
                                                                                 std::optional<float> ortho_scale_override) {
        if (width <= 0 || height <= 0) {
            return {};
        }
        if (previewRenderNeedsTiling(width, height)) {
            return renderPreviewImageTiledWithState(
                nullptr,
                model,
                std::move(scene_state),
                rotation,
                position,
                focal_length_mm,
                width,
                height,
                false,
                std::nullopt,
                orthographic_override,
                ortho_scale_override,
                PreviewImageReadback::UInt8Rgba);
        }

        return renderPreviewImageWithState(
            nullptr,
            model,
            std::move(scene_state),
            rotation,
            position,
            focal_length_mm,
            width,
            height,
            false,
            std::nullopt,
            orthographic_override,
            ortho_scale_override,
            std::nullopt,
            PreviewImageReadback::UInt8Rgba);
    }

    void RenderingManager::releasePreviewImageResources() {
        if (vksplat_viewport_renderer_) {
            vksplat_viewport_renderer_->releasePreviewResources();
        }
    }

    std::expected<lfs::core::Tensor, std::string> RenderingManager::renderExportImage(
        SceneManager* const scene_manager, const ExportImageRequest& request) {
        if (request.width <= 0 || request.height <= 0) {
            return std::unexpected("invalid export image dimensions");
        }

        const bool needs_alpha = request.mode != ExportPostProcessMode::Opaque;
        const auto rendered =
            needs_alpha
                ? renderPreviewImageRgba8(scene_manager,
                                          request.rotation,
                                          request.translation,
                                          request.focal_length_mm,
                                          request.width,
                                          request.height,
                                          request.orthographic_override,
                                          request.ortho_scale_override)
                : renderPreviewImageRgb8(scene_manager,
                                         request.rotation,
                                         request.translation,
                                         request.focal_length_mm,
                                         request.width,
                                         request.height,
                                         std::nullopt,
                                         request.orthographic_override,
                                         request.ortho_scale_override);
        releasePreviewImageResources();

        lfs::core::Tensor image;
        if (rendered && rendered->is_valid()) {
            image = std::move(*rendered);
        } else if (request.mode == ExportPostProcessMode::EnvironmentComposite) {
            // No renderable Gaussians: export the environment background alone
            // (zero alpha composites to pure environment), matching the video
            // export path's empty-primary-frame behavior.
            image = lfs::core::Tensor::zeros(
                {static_cast<size_t>(request.height), static_cast<size_t>(request.width), size_t{4}},
                lfs::core::Device::CPU,
                lfs::core::DataType::UInt8);
            if (!image.is_valid()) {
                return std::unexpected("export failed to allocate the environment-only image");
            }
        } else {
            return std::unexpected("export render produced no image");
        }

        const auto settings = getSettings();
        const ExportPostProcessView view{
            .rotation = request.rotation,
            .focal_length_mm = request.focal_length_mm,
            .equirectangular_view = settings.equirectangular,
            .controller_predict_size = frame_lifecycle_service_.lastViewportSize(),
        };
        return applyExportPostProcess(
            std::move(image), scene_manager, settings, getCurrentCameraId(), request.mode, view);
    }

    std::shared_ptr<lfs::core::Tensor> RenderingManager::renderPreviewImageWithState(
        SceneManager* const scene_manager,
        const lfs::core::SplatData& model,
        SceneRenderState scene_state,
        const glm::mat3& rotation,
        const glm::vec3& position,
        const float focal_length_mm,
        const int width,
        const int height,
        const bool render_lock_held,
        std::optional<lfs::rendering::CameraIntrinsics> intrinsics_override,
        std::optional<bool> orthographic_override,
        std::optional<float> ortho_scale_override,
        std::optional<glm::vec3> background_color_override,
        const PreviewImageReadback readback) {
        const auto readback_config =
            previewImageReadbackConfig(readback, background_color_override.has_value());

        auto rendered = renderPreviewImageToPreviewSlotWithState(
            scene_manager,
            model,
            scene_state,
            rotation,
            position,
            focal_length_mm,
            width,
            height,
            render_lock_held,
            intrinsics_override,
            {},
            {},
            orthographic_override,
            ortho_scale_override,
            background_color_override,
            readback_config.transparent_background_override);
        if (!rendered) {
            if (!intrinsics_override && isTileInstanceOverflow(rendered.error()) &&
                height > kMinPreviewSubdivisionHeight) {
                return renderPreviewImageTiledWithState(
                    scene_manager,
                    model,
                    std::move(scene_state),
                    rotation,
                    position,
                    focal_length_mm,
                    width,
                    height,
                    render_lock_held,
                    background_color_override,
                    orthographic_override,
                    ortho_scale_override,
                    readback);
            }
            LOG_ERROR("Gaussian preview image render failed: {}", rendered.error());
            return {};
        }

        std::expected<std::shared_ptr<lfs::core::Tensor>, std::string> image =
            std::unexpected("unsupported preview image readback format");
        if (readback_config.dtype == lfs::core::DataType::UInt8 &&
            readback_config.channels == 4) {
            image = vksplat_viewport_renderer_->readOutputImageRgba8(
                *last_vulkan_context_,
                VksplatViewportRenderer::OutputSlot::Preview);
        } else if (readback_config.dtype == lfs::core::DataType::UInt8) {
            image = vksplat_viewport_renderer_->readOutputImageRgb8(
                *last_vulkan_context_,
                VksplatViewportRenderer::OutputSlot::Preview);
        } else {
            image = vksplat_viewport_renderer_->readOutputImage(
                *last_vulkan_context_,
                VksplatViewportRenderer::OutputSlot::Preview);
        }
        if (!image) {
            LOG_ERROR("Gaussian preview image readback failed: {}", image.error());
            return {};
        }
        return std::move(*image);
    }

    std::expected<void, std::string> RenderingManager::renderPreviewImageToPreviewSlotWithState(
        SceneManager* const scene_manager,
        const lfs::core::SplatData& model,
        SceneRenderState scene_state,
        const glm::mat3& rotation,
        const glm::vec3& position,
        const float focal_length_mm,
        const int width,
        const int height,
        const bool render_lock_held,
        std::optional<lfs::rendering::CameraIntrinsics> intrinsics_override,
        const glm::ivec2 subregion_origin,
        const glm::ivec2 subregion_full_size,
        std::optional<bool> orthographic_override,
        std::optional<float> ortho_scale_override,
        std::optional<glm::vec3> background_color_override,
        std::optional<bool> transparent_background_override) {
        if (width <= 0 || height <= 0) {
            return std::unexpected("invalid preview render dimensions");
        }
        if (!last_vulkan_context_) {
            return std::unexpected("no Vulkan context is available");
        }
        if (!hasRenderableGaussians(&model)) {
            return std::unexpected("no renderable Gaussian model is available");
        }
        if (!scene_state.combined_model) {
            scene_state.combined_model = &model;
        }

        RenderSettings preview_settings = getSettings();
        preview_settings.focal_length_mm = std::clamp(
            focal_length_mm,
            lfs::rendering::MIN_FOCAL_LENGTH_MM,
            lfs::rendering::MAX_FOCAL_LENGTH_MM);
        preview_settings.split_view_mode = SplitViewMode::Disabled;
        preview_settings.equirectangular = false;
        if (background_color_override) {
            preview_settings.background_color = *background_color_override;
        }
        if (orthographic_override) {
            preview_settings.orthographic = *orthographic_override;
        }
        if (ortho_scale_override && std::isfinite(*ortho_scale_override) && *ortho_scale_override > 0.0f) {
            preview_settings.ortho_scale = *ortho_scale_override;
        }

        Viewport preview_viewport(
            static_cast<std::size_t>(width),
            static_cast<std::size_t>(height));
        preview_viewport.setViewMatrix(rotation, position);

        FrameContext frame_ctx{
            .viewport = preview_viewport,
            .render_lock_held = render_lock_held,
            .scene_manager = scene_manager,
            .model = &model,
            .scene_state = std::move(scene_state),
            .settings = preview_settings,
            .render_size = {width, height},
            .viewport_pos = {0, 0},
            .cursor_preview = {},
            .gizmo = {},
            .view_panels = {},
        };

        auto request = buildViewportRenderRequest(frame_ctx, frame_ctx.render_size);
        if (transparent_background_override) {
            request.transparent_background = *transparent_background_override;
        }
        request.frame_view.intrinsics_override = std::move(intrinsics_override);
        request.frame_view.subregion_origin = subregion_origin;
        request.frame_view.subregion_full_size = subregion_full_size;
        request.raster_backend =
            lfs::rendering::normalizeViewerRasterBackend(request.raster_backend, request.gut);
        request.gut = lfs::rendering::isGutBackend(request.raster_backend);
        if (!lfs::rendering::isVkSplatBackend(request.raster_backend)) {
            return std::unexpected(std::format(
                "unsupported raster backend '{}'",
                lfs::rendering::gaussianRasterBackendId(request.raster_backend)));
        }

        if (!vksplat_viewport_renderer_) {
            vksplat_viewport_renderer_ = std::make_unique<VksplatViewportRenderer>();
        }

        // Preview/export uses the renderer's exact two-batch count gate; one
        // render is complete for this view and can be read back immediately.
        auto render_result = vksplat_viewport_renderer_->render(
            *last_vulkan_context_,
            model,
            request,
            false,
            VksplatViewportRenderer::OutputSlot::Preview,
            false);
        if (!render_result) {
            return std::unexpected(render_result.error());
        }
        return {};
    }

    std::shared_ptr<lfs::core::Tensor> RenderingManager::renderPreviewImageTiledWithState(
        SceneManager* const scene_manager,
        const lfs::core::SplatData& model,
        SceneRenderState scene_state,
        const glm::mat3& rotation,
        const glm::vec3& position,
        const float focal_length_mm,
        const int width,
        const int height,
        const bool render_lock_held,
        std::optional<glm::vec3> background_color_override,
        std::optional<bool> orthographic_override,
        std::optional<float> ortho_scale_override,
        const PreviewImageReadback readback) {
        if (width <= 0 || height <= 0) {
            return {};
        }
        const auto readback_config =
            previewImageReadbackConfig(readback, background_color_override.has_value());

        const int tile_width = width;
        const int tile_height_limit = previewTileHeightForWidth(tile_width);
        if (tile_height_limit <= 0) {
            return {};
        }

        LOG_INFO("Gaussian preview image {}x{} uses tiled render readback: tile_width={} max_tile_height={}",
                 width,
                 height,
                 tile_width,
                 tile_height_limit);

        auto output = lfs::core::Tensor::empty(
            {static_cast<std::size_t>(height),
             static_cast<std::size_t>(width),
             static_cast<std::size_t>(readback_config.channels)},
            lfs::core::Device::CPU,
            readback_config.dtype);
        if (!output.is_valid()) {
            LOG_TRACE("Gaussian preview tiled render failed to allocate output tensor");
            return {};
        }

        int band_height_limit = tile_height_limit;
        for (int tile_y = 0; tile_y < height;) {
            int tile_height = std::min(band_height_limit, height - tile_y);
            const auto intrinsics = previewTileIntrinsics(
                width,
                height,
                focal_length_mm);
            while (true) {
                auto rendered = renderPreviewImageToPreviewSlotWithState(
                    scene_manager,
                    model,
                    scene_state,
                    rotation,
                    position,
                    focal_length_mm,
                    tile_width,
                    tile_height,
                    render_lock_held,
                    intrinsics,
                    {0, tile_y},
                    {width, height},
                    orthographic_override,
                    ortho_scale_override,
                    background_color_override,
                    readback_config.transparent_background_override);
                if (rendered) {
                    break;
                }
                if (!isTileInstanceOverflow(rendered.error()) ||
                    tile_height <= kMinPreviewSubdivisionHeight) {
                    LOG_TRACE("Gaussian preview tiled render failed at tile y={} height={}: {}",
                              tile_y,
                              tile_height,
                              rendered.error());
                    return {};
                }
                tile_height = std::max(kMinPreviewSubdivisionHeight, tile_height / 2);
                band_height_limit = tile_height;
                LOG_WARN("Gaussian preview band overflow at y={}; retrying with height={}",
                         tile_y,
                         tile_height);
            }
            auto copied = vksplat_viewport_renderer_->readOutputImageIntoCpuHwc(
                *last_vulkan_context_,
                VksplatViewportRenderer::OutputSlot::Preview,
                output,
                0,
                tile_y);
            if (!copied) {
                LOG_TRACE("Gaussian preview tiled readback failed at tile y={} height={}: {}",
                          tile_y,
                          tile_height,
                          copied.error());
                return {};
            }
            tile_y += tile_height;
        }

        return std::make_shared<lfs::core::Tensor>(std::move(output));
    }

    float RenderingManager::getDepthAtPixel(const int x, const int y,
                                            const std::optional<SplitViewPanelId> panel) const {
        const float cached_depth = viewport_artifact_service_.sampleLinearDepthAt(
            x,
            y,
            frame_lifecycle_service_.lastViewportSize(),
            panel);
        if (cached_depth > 0.0f) {
            return cached_depth;
        }

        if (!vksplat_viewport_renderer_ || !last_vulkan_context_) {
            return -1.0f;
        }

        VksplatViewportRenderer::OutputSlot output_slot = VksplatViewportRenderer::OutputSlot::Main;
        if (panel && isIndependentSplitViewActive()) {
            output_slot = *panel == SplitViewPanelId::Right
                              ? VksplatViewportRenderer::OutputSlot::SplitRight
                              : VksplatViewportRenderer::OutputSlot::SplitLeft;
        }

        glm::ivec2 source_size = frame_lifecycle_service_.lastViewportSize();
        if (source_size.x > 0 && source_size.y > 0 && panel && isIndependentSplitViewActive()) {
            if (const auto layouts = split_view_service_.panelLayouts(settings_, source_size.x)) {
                const auto& layout = (*layouts)[splitViewPanelIndex(*panel)];
                source_size.x = std::max(layout.width, 1);
            }
        }

        const auto depth = vksplat_viewport_renderer_->sampleDepthAtPixel(
            *last_vulkan_context_,
            VksplatViewportRenderer::DepthSampleRequest{
                .pixel = {x, y},
                .source_size = source_size,
                .output_slot = output_slot,
            });
        if (!depth) {
            LOG_TRACE("VkSplat depth sample failed: {}", depth.error());
            return -1.0f;
        }
        return *depth;
    }

    float RenderingManager::renderExpectedDepthAtPixel(const ExpectedDepthSampleRequest& request) {
        if (!request.scene_manager ||
            !request.viewport ||
            request.render_size.x <= 0 ||
            request.render_size.y <= 0 ||
            request.pixel.x < 0 ||
            request.pixel.y < 0 ||
            request.pixel.x >= request.render_size.x ||
            request.pixel.y >= request.render_size.y) {
            return -1.0f;
        }

        auto render_lock = acquireLiveModelRenderLock(request.scene_manager);
        auto scene_state = request.scene_manager->buildRenderState();
        const auto* const model = scene_state.combined_model;
        if (!hasRenderableGaussians(model)) {
            return -1.0f;
        }

        auto rendered = renderDepthCaptureToPreviewSlotWithState(
            request.scene_manager,
            *model,
            std::move(scene_state),
            request.viewport->camera.R,
            request.viewport->camera.t,
            request.focal_length_mm,
            request.render_size.x,
            request.render_size.y,
            render_lock.has_value(),
            true,
            std::nullopt,
            request.orthographic,
            request.ortho_scale);
        if (!rendered) {
            LOG_TRACE("Expected-depth pixel render failed: {}", rendered.error());
            return -1.0f;
        }

        auto depth = vksplat_viewport_renderer_->readPreviewDepth(
            *last_vulkan_context_,
            VksplatViewportRenderer::OutputSlot::Preview);
        if (!depth) {
            LOG_TRACE("Expected-depth pixel readback failed: {}", depth.error());
            return -1.0f;
        }

        return sampleDepthTensorAt(**depth, request.pixel).value_or(-1.0f);
    }

    float RenderingManager::renderDepthAtPixelForNodeMask(const SceneManager* const scene_manager,
                                                          const Viewport& viewport,
                                                          const glm::ivec2& render_size,
                                                          const int x,
                                                          const int y,
                                                          const std::vector<bool>& node_visibility_mask) {
        if (!scene_manager || render_size.x <= 0 || render_size.y <= 0 ||
            x < 0 || x >= render_size.x || y < 0 || y >= render_size.y ||
            node_visibility_mask.empty() ||
            !std::any_of(node_visibility_mask.begin(), node_visibility_mask.end(), [](const bool enabled) {
                return enabled;
            })) {
            return -1.0f;
        }
        auto render_lock = acquireLiveModelRenderLock(scene_manager);
        auto scene_state = scene_manager->buildRenderState();
        const auto* const model = scene_state.combined_model;
        if (!hasRenderableGaussians(model)) {
            return -1.0f;
        }

        FrameContext frame_ctx{
            .viewport = viewport,
            .scene_manager = const_cast<SceneManager*>(scene_manager),
            .model = model,
            .scene_state = std::move(scene_state),
            .settings = settings_,
            .render_size = render_size,
            .viewport_pos = {0, 0},
            .cursor_preview = {},
            .gizmo = {},
            .view_panels = {},
        };

        lfs::rendering::FrameMetadata metadata{};
        if (settings_.point_cloud_mode) {
            auto* const engine = getRenderingEngine();
            if (!engine) {
                return -1.0f;
            }
            auto request = buildPointCloudRenderRequest(
                frame_ctx,
                render_size,
                frame_ctx.scene_state.model_transforms);
            request.scene.node_visibility_mask = node_visibility_mask;
            auto result = engine->renderPointCloudImage(*model, request);
            if (!result) {
                LOG_DEBUG("Masked point-cloud depth render failed: {}", result.error());
                return -1.0f;
            }
            metadata = std::move(result->metadata);
        } else {
            LOG_TRACE("Masked Gaussian depth render skipped: no Vulkan masked-depth path is available");
            return -1.0f;
        }
        render_lock.reset();

        ViewportArtifactService artifacts;
        artifacts.updateFromImageOutput({}, metadata, render_size, true);
        return artifacts.sampleLinearDepthAt(x, y, render_size, std::nullopt);
    }

} // namespace lfs::vis

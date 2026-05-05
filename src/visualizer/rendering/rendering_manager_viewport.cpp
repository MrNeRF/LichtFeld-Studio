/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/logger.hpp"
#include "model_renderability.hpp"
#include "rendering/gl_state_guard.hpp"
#include "rendering/image_layout.hpp"
#include "rendering_manager.hpp"
#include "scene/scene_manager.hpp"
#include "training/trainer.hpp"
#include "training/training_manager.hpp"
#include <glad/glad.h>
#include <shared_mutex>

namespace lfs::vis {

    namespace {
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

        [[nodiscard]] bool uploadPreviewImageToTexture(const lfs::core::Tensor& image,
                                                       const unsigned int texture,
                                                       const glm::ivec2 expected_size) {
            if (texture == 0 || !image.is_valid() || image.ndim() != 3) {
                return false;
            }

            const auto layout = lfs::rendering::detectImageLayout(image);
            if (layout == lfs::rendering::ImageLayout::Unknown) {
                LOG_ERROR("Preview upload received unsupported tensor shape [{}, {}, {}]",
                          image.size(0), image.size(1), image.size(2));
                return false;
            }

            lfs::core::Tensor formatted = (layout == lfs::rendering::ImageLayout::HWC)
                                              ? image
                                              : image.permute({1, 2, 0}).contiguous();
            if (formatted.device() == lfs::core::Device::CUDA) {
                formatted = formatted.cpu();
            }
            formatted = formatted.contiguous();

            if (formatted.dtype() != lfs::core::DataType::UInt8) {
                formatted = (formatted.clamp(0.0f, 1.0f) * 255.0f).to(lfs::core::DataType::UInt8);
                formatted = formatted.contiguous();
            }

            const int height = static_cast<int>(formatted.size(0));
            const int width = static_cast<int>(formatted.size(1));
            const int channels = static_cast<int>(formatted.size(2));
            if (width != expected_size.x || height != expected_size.y || !formatted.ptr<unsigned char>()) {
                LOG_ERROR("Preview upload dimension mismatch: {}x{} vs {}x{}",
                          width, height, expected_size.x, expected_size.y);
                return false;
            }

            const GLenum format = (channels == 1)   ? GL_RED
                                  : (channels == 4) ? GL_RGBA
                                                    : GL_RGB;
            if (channels != 1 && channels != 3 && channels != 4) {
                LOG_ERROR("Preview upload received unsupported channel count {}", channels);
                return false;
            }

            const lfs::rendering::GLStateGuard state_guard;
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, texture);
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height,
                            format, GL_UNSIGNED_BYTE, formatted.ptr<unsigned char>());

            if (const GLenum gl_error = glGetError(); gl_error != GL_NO_ERROR) {
                LOG_ERROR("Preview texture upload failed with GL error {}", static_cast<unsigned int>(gl_error));
                return false;
            }

            return true;
        }
    } // namespace

    RenderingManager::ContentBounds RenderingManager::getContentBounds(const glm::ivec2& viewport_size) const {
        ContentBounds bounds{0.0f, 0.0f, static_cast<float>(viewport_size.x), static_cast<float>(viewport_size.y), false};

        if (split_view_service_.isGTComparisonActive(settings_)) {
            const auto content_dims = split_view_service_.gtContentDimensions();
            if (!content_dims) {
                return bounds;
            }

            const float content_aspect = static_cast<float>(content_dims->x) / content_dims->y;
            const float viewport_aspect = static_cast<float>(viewport_size.x) / viewport_size.y;

            if (content_aspect > viewport_aspect) {
                bounds.width = static_cast<float>(viewport_size.x);
                bounds.height = viewport_size.x / content_aspect;
                bounds.x = 0.0f;
                bounds.y = (viewport_size.y - bounds.height) / 2.0f;
            } else {
                bounds.height = static_cast<float>(viewport_size.y);
                bounds.width = viewport_size.y * content_aspect;
                bounds.x = (viewport_size.x - bounds.width) / 2.0f;
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
        if (!initialized_) {
            initialize();
        }
        return engine_.get();
    }

    std::shared_ptr<lfs::core::Tensor> RenderingManager::getViewportImageIfAvailable() const {
        return viewport_artifact_service_.getCapturedImageIfCurrent();
    }

    std::shared_ptr<lfs::core::Tensor> RenderingManager::captureViewportImage() {
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
        const int hovered_camera = camera_interaction_service_.pickCameraFrustum(
            engine_.get(),
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

    bool RenderingManager::renderPreviewFrame(SceneManager* const scene_manager,
                                              const glm::mat3& rotation,
                                              const glm::vec3& position,
                                              const float focal_length_mm,
                                              const unsigned int fbo,
                                              [[maybe_unused]] const unsigned int texture,
                                              const int width, const int height) {
        if (!initialized_ || !engine_) {
            return false;
        }

        auto render_lock = acquireLiveModelRenderLock(scene_manager);
        const auto render_state = scene_manager ? scene_manager->buildRenderState() : SceneRenderState{};
        const auto* const model = render_state.combined_model;
        if (!hasRenderableGaussians(model)) {
            return false;
        }

        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glViewport(0, 0, width, height);
        const auto& bg = settings_.background_color;
        glClearColor(bg.r, bg.g, bg.b, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        const lfs::rendering::FrameView frame_view{
            .rotation = rotation,
            .translation = position,
            .size = {width, height},
            .focal_length_mm = focal_length_mm,
            .intrinsics_override = std::nullopt,
            .background_color = bg};

        bool rendered = false;
        if (settings_.point_cloud_mode) {
            const lfs::rendering::PointCloudRenderRequest request{
                .frame_view = frame_view,
                .render =
                    {.scaling_modifier = settings_.scaling_modifier,
                     .voxel_size = settings_.voxel_size,
                     .equirectangular = settings_.equirectangular},
                .scene =
                    {.model_transforms = &render_state.model_transforms,
                     .transform_indices = render_state.transform_indices},
                .filters = {}};

            if (const auto result = engine_->renderPointCloudGpuFrame(*model, request)) {
                if (auto r = engine_->presentGpuFrame(*result, {0, 0}, {width, height}); !r)
                    LOG_WARN("presentGpuFrame failed: {}", r.error());
                else
                    rendered = true;
            }
        } else {
            const lfs::rendering::ViewportRenderRequest request{
                .frame_view = frame_view,
                .scaling_modifier = settings_.scaling_modifier,
                .antialiasing = false,
                .sh_degree = 0,
                .gut = settings_.gut,
                .equirectangular = settings_.equirectangular,
                .scene =
                    {.model_transforms = &render_state.model_transforms,
                     .transform_indices = render_state.transform_indices,
                     .node_visibility_mask = render_state.node_visibility_mask},
                .filters = {},
                .overlay = {}};

            if (const auto result = engine_->renderGaussiansGpuFrame(*model, request)) {
                if (auto r = engine_->presentGpuFrame(result->frame, {0, 0}, {width, height}); !r)
                    LOG_WARN("presentGpuFrame failed: {}", r.error());
                else
                    rendered = true;
            }
        }

        if (rendered) {
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            return true;
        }

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        return false;
    }

    bool RenderingManager::renderPreviewTexture(SceneManager* const scene_manager,
                                                const glm::mat3& rotation,
                                                const glm::vec3& position,
                                                const float focal_length_mm,
                                                const unsigned int texture,
                                                const int width, const int height) {
        if (!initialized_ || !engine_ || texture == 0) {
            return false;
        }

        auto render_lock = acquireLiveModelRenderLock(scene_manager);
        const auto render_state = scene_manager ? scene_manager->buildRenderState() : SceneRenderState{};
        const auto* const model = render_state.combined_model;
        if (!hasRenderableGaussians(model)) {
            return false;
        }

        const auto& bg = settings_.background_color;
        const lfs::rendering::FrameView frame_view{
            .rotation = rotation,
            .translation = position,
            .size = {width, height},
            .focal_length_mm = focal_length_mm,
            .intrinsics_override = std::nullopt,
            .background_color = bg};

        if (settings_.point_cloud_mode) {
            const lfs::rendering::PointCloudRenderRequest request{
                .frame_view = frame_view,
                .render =
                    {.scaling_modifier = settings_.scaling_modifier,
                     .voxel_size = settings_.voxel_size,
                     .equirectangular = settings_.equirectangular},
                .scene =
                    {.model_transforms = &render_state.model_transforms,
                     .transform_indices = render_state.transform_indices},
                .filters = {}};

            auto result = engine_->renderPointCloudImage(*model, request);
            render_lock.reset();
            return result && result->image &&
                   uploadPreviewImageToTexture(*result->image, texture, {width, height});
        }

        const lfs::rendering::ViewportRenderRequest request{
            .frame_view = frame_view,
            .scaling_modifier = settings_.scaling_modifier,
            .antialiasing = false,
            .sh_degree = 0,
            .gut = settings_.gut,
            .equirectangular = settings_.equirectangular,
            .scene =
                {.model_transforms = &render_state.model_transforms,
                 .transform_indices = render_state.transform_indices,
                 .node_visibility_mask = render_state.node_visibility_mask},
            .filters = {},
            .overlay = {}};

        auto result = engine_->renderGaussiansImage(*model, request);
        render_lock.reset();
        return result && result->image &&
               uploadPreviewImageToTexture(*result->image, texture, {width, height});
    }

    float RenderingManager::getDepthAtPixel(const int x, const int y,
                                            const std::optional<SplitViewPanelId> panel) const {
        return viewport_artifact_service_.sampleLinearDepthAt(
            x,
            y,
            frame_lifecycle_service_.lastViewportSize(),
            engine_.get(),
            panel);
    }

} // namespace lfs::vis

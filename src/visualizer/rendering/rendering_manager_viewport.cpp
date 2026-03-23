/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/logger.hpp"
#include "model_renderability.hpp"
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
    } // namespace

    RenderingManager::ContentBounds RenderingManager::getContentBounds(const glm::ivec2& viewport_size) const {
        ContentBounds bounds{0.0f, 0.0f, static_cast<float>(viewport_size.x), static_cast<float>(viewport_size.y), false};

        if (settings_.split_view_mode == SplitViewMode::GTComparison) {
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
        const auto* const model = scene_manager ? scene_manager->getModelForRendering() : nullptr;
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
            .background_color = bg};

        bool rendered = false;
        if (settings_.point_cloud_mode) {
            const lfs::rendering::PointCloudRenderRequest request{
                .frame_view = frame_view,
                .render =
                    {.scaling_modifier = settings_.scaling_modifier,
                     .voxel_size = settings_.voxel_size,
                     .equirectangular = settings_.equirectangular},
                .scene = {},
                .filters = {}};

            if (const auto result = engine_->renderPointCloudGpuFrame(*model, request)) {
                engine_->presentGpuFrame(*result, {0, 0}, {width, height});
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
                .scene = {},
                .filters = {},
                .overlay = {}};

            if (const auto result = engine_->renderGaussiansGpuFrame(*model, request)) {
                engine_->presentGpuFrame(result->frame, {0, 0}, {width, height});
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

    float RenderingManager::getDepthAtPixel(const int x, const int y) const {
        return viewport_artifact_service_.sampleLinearDepthAt(
            x,
            y,
            frame_lifecycle_service_.lastViewportSize(),
            engine_.get());
    }

} // namespace lfs::vis

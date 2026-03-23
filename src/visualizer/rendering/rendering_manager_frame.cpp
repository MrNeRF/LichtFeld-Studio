/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/logger.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "model_renderability.hpp"
#include "render_frame_coordinator.hpp"
#include "rendering/rasterizer/rasterization/include/rasterization_config.h"
#include "rendering_manager.hpp"
#include "scene/scene_manager.hpp"
#include "theme/theme.hpp"
#include "training/trainer.hpp"
#include "training/training_manager.hpp"
#include <glad/glad.h>
#include <optional>
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

    void RenderingManager::renderFrame(const RenderContext& context) {
        SceneManager* const scene_manager = context.scene_manager;

        if (!initialized_) {
            initialize();
        }

        if (scene_manager && (dirty_mask_.load(std::memory_order_relaxed) & DirtyFlag::SELECTION)) {
            for (const auto& group : scene_manager->getScene().getSelectionGroups()) {
                lfs::rendering::config::setSelectionGroupColor(
                    group.id, make_float3(group.color.x, group.color.y, group.color.z));
            }
        }

        glm::ivec2 current_size = context.viewport.windowSize;
        if (context.viewport_region) {
            current_size = glm::ivec2(
                static_cast<int>(context.viewport_region->width),
                static_cast<int>(context.viewport_region->height));
        }

        if (current_size.x <= 0 || current_size.y <= 0) {
            LOG_TRACE("Skipping render - invalid viewport size: {}x{}", current_size.x, current_size.y);
            const auto& shell_bg = theme().menu_background();
            glClearColor(shell_bg.x, shell_bg.y, shell_bg.z, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            return;
        }

        const auto resize_result = frame_lifecycle_service_.handleViewportResize(current_size);
        if (resize_result.dirty) {
            markDirty(resize_result.dirty);
        }
        const bool resize_completed = resize_result.completed;

        auto render_lock = acquireLiveModelRenderLock(scene_manager);
        const lfs::core::SplatData* const model = scene_manager ? scene_manager->getModelForRendering() : nullptr;
        const bool has_renderable_model = hasRenderableGaussians(model);
        const auto* const visible_point_cloud =
            (scene_manager && !has_renderable_model) ? scene_manager->getScene().getVisiblePointCloud() : nullptr;
        const bool has_visible_point_cloud = visible_point_cloud && visible_point_cloud->size() > 0;
        const size_t model_ptr = reinterpret_cast<size_t>(model);

        if (const auto model_change = frame_lifecycle_service_.handleModelChange(model_ptr, viewport_artifact_service_);
            model_change.changed) {
            LOG_DEBUG("Model ptr changed: {} -> {}, size={}",
                      model_change.previous_model_ptr, model_ptr, model ? model->size() : 0);
            markDirty(DirtyFlag::ALL);
        }

        const bool is_training = scene_manager && scene_manager->hasDataset() &&
                                 scene_manager->getTrainerManager() &&
                                 scene_manager->getTrainerManager()->isRunning();

        if (const DirtyMask training_dirty = frame_lifecycle_service_.handleTrainingRefresh(
                is_training,
                framerate_controller_.getSettings().training_frame_refresh_time_sec);
            training_dirty) {
            markDirty(training_dirty);
        }

        bool request_gt_prerender = false;
        split_view_service_.prepareGTComparisonContext(
            scene_manager,
            settings_,
            camera_interaction_service_.currentCameraId(),
            has_renderable_model || has_visible_point_cloud,
            viewport_artifact_service_.hasGpuFrame(),
            gt_texture_cache_,
            request_gt_prerender);
        if (request_gt_prerender) {
            dirty_mask_.fetch_or(DirtyFlag::SPLATS, std::memory_order_relaxed);
        }

        if (const DirtyMask required_dirty = frame_lifecycle_service_.requiredDirtyMask(
                viewport_artifact_service_.hasViewportOutput(),
                has_renderable_model || has_visible_point_cloud,
                settings_.split_view_mode);
            required_dirty) {
            dirty_mask_.fetch_or(required_dirty, std::memory_order_relaxed);
        }

        glViewport(0, 0, context.viewport.frameBufferSize.x, context.viewport.frameBufferSize.y);

        const auto& shell_bg = theme().menu_background();
        glClearColor(shell_bg.x, shell_bg.y, shell_bg.z, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        if (context.viewport_region) {
            const GLint x = static_cast<GLint>(context.viewport_region->x);
            const GLint y = context.viewport.frameBufferSize.y -
                            static_cast<GLint>(context.viewport_region->y + context.viewport_region->height);
            const GLsizei w = static_cast<GLsizei>(context.viewport_region->width);
            const GLsizei h = static_cast<GLsizei>(context.viewport_region->height);
            glViewport(x, y, w, h);
            glScissor(x, y, w, h);
            glEnable(GL_SCISSOR_TEST);
        }

        const DirtyMask frame_dirty = dirty_mask_.exchange(0);
        RenderFrameCoordinator frame_coordinator{
            {.engine = *engine_,
             .pass_graph = pass_graph_,
             .framerate_controller = framerate_controller_,
             .viewport_artifacts = viewport_artifact_service_,
             .viewport_interaction_context = viewport_interaction_context_,
             .viewport_overlay = viewport_overlay_service_,
             .split_view_service = split_view_service_,
             .render_count = render_count_}};
        const auto frame_result = frame_coordinator.execute(
            {.viewport = context.viewport,
             .viewport_region = context.viewport_region,
             .scene_manager = scene_manager,
             .model = model,
             .render_lock_held = render_lock.has_value(),
             .settings = settings_,
             .frame_dirty = frame_dirty,
             .selection_flash_intensity = getSelectionFlashIntensity(),
             .current_camera_id = camera_interaction_service_.currentCameraId(),
             .hovered_camera_id = camera_interaction_service_.hoveredCameraId()});

        if (frame_result.additional_dirty)
            markDirty(frame_result.additional_dirty);
        if (frame_result.pivot_animation_end)
            setPivotAnimationEndTime(*frame_result.pivot_animation_end);

        if (resize_completed) {
            frame_lifecycle_service_.noteResizeCompleted();
            lfs::core::Tensor::trim_memory_pool();
        }

        if (context.viewport_region) {
            glDisable(GL_SCISSOR_TEST);
        }

        viewport_interaction_context_.scene_manager = scene_manager;
        viewport_interaction_context_.updatePickContext(context.viewport_region,
                                                        viewport_interaction_context_.viewport_data);
    }

} // namespace lfs::vis

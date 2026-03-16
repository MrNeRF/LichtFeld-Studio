/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "rendering_manager_state.hpp"
#include "gt_texture_cache.hpp"
#include "scene/scene_manager.hpp"
#include "training/trainer.hpp"
#include "training/training_manager.hpp"
#include <glad/glad.h>
#include <limits>

namespace lfs::vis {

    namespace {

        [[nodiscard]] float linearizeDepthSample(const float depth_sample,
                                                 const float near_plane,
                                                 const float far_plane,
                                                 const bool orthographic,
                                                 const bool depth_is_ndc) {
            if (!depth_is_ndc) {
                return depth_sample < 1e9f ? depth_sample : -1.0f;
            }

            constexpr float DEPTH_BG_THRESHOLD = 0.9999f;
            if (depth_sample >= DEPTH_BG_THRESHOLD) {
                return -1.0f;
            }

            if (orthographic) {
                return near_plane + depth_sample * (far_plane - near_plane);
            }

            const float z_ndc = depth_sample * 2.0f - 1.0f;
            const float A = (far_plane + near_plane) / (far_plane - near_plane);
            const float B = (2.0f * far_plane * near_plane) / (far_plane - near_plane);
            return B / (A - z_ndc);
        }

    } // namespace

    GpuDepthReadbackState::~GpuDepthReadbackState() {
        if (depth_readback_fbo != 0) {
            glDeleteFramebuffers(1, &depth_readback_fbo);
        }
    }

    float GpuDepthReadbackState::readLinearDepth(const lfs::rendering::GpuFrame& frame,
                                                 const int x,
                                                 const int y,
                                                 const int viewport_height) const {
        if (!frame.valid() || !frame.depth.valid() || x < 0 || y < 0 || y >= viewport_height) {
            return -1.0f;
        }

        if (depth_readback_fbo == 0) {
            glGenFramebuffers(1, &depth_readback_fbo);
        }
        if (depth_readback_fbo == 0) {
            return -1.0f;
        }

        GLint saved_fbo = 0;
        glGetIntegerv(GL_FRAMEBUFFER_BINDING, &saved_fbo);

        glBindFramebuffer(GL_FRAMEBUFFER, depth_readback_fbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, frame.depth.id, 0);
        glDrawBuffer(GL_NONE);
        glReadBuffer(GL_NONE);

        float linear_depth = -1.0f;
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE) {
            float raw_depth = frame.depth_is_ndc ? 1.0f : std::numeric_limits<float>::infinity();
            glReadPixels(x, viewport_height - 1 - y, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &raw_depth);
            linear_depth = linearizeDepthSample(
                raw_depth,
                frame.near_plane,
                frame.far_plane,
                frame.orthographic,
                frame.depth_is_ndc);
        }

        glBindFramebuffer(GL_FRAMEBUFFER, saved_fbo);
        return linear_depth;
    }

    void SplitViewState::clear() {
        clearGTContext();
        pre_gt_equirectangular = false;
        std::lock_guard<std::mutex> lock(info_mutex);
        current_info = {};
    }

    void SplitViewState::clearGTContext() {
        gt_context.reset();
    }

    bool SplitViewState::togglePLYComparison(RenderSettings& settings) {
        const bool enabled = settings.split_view_mode != SplitViewMode::PLYComparison;
        settings.split_view_mode = enabled ? SplitViewMode::PLYComparison : SplitViewMode::Disabled;
        settings.split_view_offset = 0;
        return enabled;
    }

    SplitViewState::GTToggleResult SplitViewState::toggleGTComparison(RenderSettings& settings) {
        GTToggleResult result;
        if (settings.split_view_mode == SplitViewMode::GTComparison) {
            settings.split_view_mode = SplitViewMode::Disabled;
            settings.equirectangular = pre_gt_equirectangular;
            result.restore_equirectangular = pre_gt_equirectangular;
            clearGTContext();
            return result;
        }

        pre_gt_equirectangular = settings.equirectangular;
        settings.split_view_mode = SplitViewMode::GTComparison;
        result.enabled = true;
        return result;
    }

    void SplitViewState::handleSceneLoaded(RenderSettings& settings) {
        clearGTContext();
        {
            std::lock_guard<std::mutex> lock(info_mutex);
            current_info = {};
        }
        if (settings.split_view_mode == SplitViewMode::GTComparison) {
            settings.split_view_mode = SplitViewMode::Disabled;
        }
    }

    void SplitViewState::handleSceneCleared() {
        clear();
    }

    bool SplitViewState::handlePLYRemoved(RenderSettings& settings, SceneManager* scene_manager) {
        if (settings.split_view_mode != SplitViewMode::PLYComparison || !scene_manager) {
            return false;
        }

        const auto visible_nodes = scene_manager->getScene().getVisibleNodes();
        if (visible_nodes.size() >= 2) {
            return false;
        }

        settings.split_view_mode = SplitViewMode::Disabled;
        settings.split_view_offset = 0;
        return true;
    }

    void SplitViewState::advanceSplitOffset(RenderSettings& settings) {
        ++settings.split_view_offset;
    }

    SplitViewInfo SplitViewState::getInfo() const {
        std::lock_guard<std::mutex> lock(info_mutex);
        return current_info;
    }

    void SplitViewState::updateInfo(const FrameResources& resources) {
        std::lock_guard<std::mutex> lock(info_mutex);
        current_info = resources.split_view_executed ? resources.split_info : SplitViewInfo{};
    }

    void SplitViewState::prepareGTComparisonContext(SceneManager* scene_manager,
                                                    const RenderSettings& settings,
                                                    const int current_camera_id,
                                                    const bool has_renderable_content,
                                                    const bool has_viewport_output,
                                                    GTTextureCache& texture_cache,
                                                    bool& request_viewport_prerender) {
        request_viewport_prerender = false;

        if (settings.split_view_mode != SplitViewMode::GTComparison ||
            current_camera_id < 0 ||
            !has_renderable_content ||
            !scene_manager) {
            clearGTContext();
            return;
        }

        clearGTContext();

        auto* trainer_manager = scene_manager->getTrainerManager();
        if (!trainer_manager || !trainer_manager->hasTrainer()) {
            return;
        }

        const auto* trainer = trainer_manager->getTrainer();
        if (!trainer) {
            return;
        }

        const auto loader_owner = trainer->getActiveImageLoader();
        const auto cam = trainer_manager->getCamById(current_camera_id);
        if (!cam) {
            return;
        }

        lfs::io::LoadParams gt_load_params;
        const lfs::io::LoadParams* gt_load_params_ptr = nullptr;
        if (loader_owner) {
            const auto gt_load_config = trainer->getGTLoadConfigSnapshot();
            gt_load_params.resize_factor = gt_load_config.resize_factor;
            gt_load_params.max_width = gt_load_config.max_width;
            if (gt_load_config.undistort && cam->is_undistort_prepared()) {
                gt_load_params.undistort = &cam->undistort_params();
            }
            gt_load_params_ptr = &gt_load_params;
        }

        const auto gt_info = texture_cache.getGTTexture(
            current_camera_id,
            cam->image_path(),
            loader_owner.get(),
            gt_load_params_ptr);
        if (gt_info.texture_id == 0) {
            return;
        }

        const glm::ivec2 dims(gt_info.width, gt_info.height);
        const glm::ivec2 aligned(
            ((dims.x + GPU_ALIGNMENT - 1) / GPU_ALIGNMENT) * GPU_ALIGNMENT,
            ((dims.y + GPU_ALIGNMENT - 1) / GPU_ALIGNMENT) * GPU_ALIGNMENT);

        gt_context = GTComparisonContext{
            .gt_texture_id = gt_info.texture_id,
            .dimensions = dims,
            .gpu_aligned_dims = aligned,
            .render_texcoord_scale = glm::vec2(dims) / glm::vec2(aligned),
            .gt_texcoord_scale = gt_info.texcoord_scale,
            .gt_needs_flip = gt_info.needs_flip};

        request_viewport_prerender = hasValidGTContext() && !has_viewport_output;
    }

    ViewportFrameLifecycleState::ResizeResult
    ViewportFrameLifecycleState::handleViewportResize(const glm::ivec2& current_size) {
        constexpr int kResizeDebounceFrames = 3;

        ResizeResult result;
        const bool resize_is_active = resize_active.load(std::memory_order_relaxed);

        if (current_size != last_viewport_size) {
            last_viewport_size = current_size;
            if (resize_is_active) {
                result.dirty = DirtyFlag::OVERLAY;
                resize_debounce = kResizeDebounceFrames;
            } else {
                result.dirty = DirtyFlag::VIEWPORT | DirtyFlag::CAMERA | DirtyFlag::OVERLAY;
            }
            return result;
        }

        if (resize_debounce > 0 && !resize_is_active) {
            if (--resize_debounce == 0) {
                result.dirty = DirtyFlag::VIEWPORT | DirtyFlag::CAMERA;
                result.completed = true;
            } else {
                result.dirty = DirtyFlag::OVERLAY;
            }
        }
        return result;
    }

    ViewportFrameLifecycleState::ModelChangeResult
    ViewportFrameLifecycleState::handleModelChange(const size_t model_ptr,
                                                   ViewportArtifactState& viewport_artifacts) {
        if (model_ptr == last_model_ptr) {
            return {};
        }

        const ModelChangeResult result{
            .changed = true,
            .previous_model_ptr = last_model_ptr};
        last_model_ptr = model_ptr;
        viewport_artifacts.clearViewportOutput();
        return result;
    }

    DirtyMask ViewportFrameLifecycleState::handleTrainingRefresh(const bool is_training,
                                                                 const float refresh_interval_sec) {
        if (!is_training) {
            return 0;
        }

        const auto now = std::chrono::steady_clock::now();
        const auto interval = std::chrono::duration<float>(refresh_interval_sec);
        if (now - last_training_render <= interval) {
            return 0;
        }

        last_training_render = now;
        return DirtyFlag::SPLATS;
    }

    DirtyMask ViewportFrameLifecycleState::requiredDirtyMask(const bool has_viewport_output,
                                                             const bool has_renderable_content,
                                                             const SplitViewMode split_view_mode) const {
        DirtyMask dirty = 0;
        if (!has_viewport_output &&
            (has_renderable_content || split_view_mode != SplitViewMode::Disabled)) {
            dirty |= DirtyFlag::ALL;
        }
        if (split_view_mode != SplitViewMode::Disabled) {
            dirty |= DirtyFlag::SPLIT_VIEW;
        }
        return dirty;
    }

    DirtyMask ViewportFrameLifecycleState::setViewportResizeActive(const bool active) {
        const bool was_active = resize_active.exchange(active);
        if (!was_active || active) {
            return 0;
        }

        if (resize_debounce == 0) {
            resize_debounce = 1;
        }

        return DirtyFlag::VIEWPORT | DirtyFlag::CAMERA | DirtyFlag::OVERLAY;
    }

} // namespace lfs::vis

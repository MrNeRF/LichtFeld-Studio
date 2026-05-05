/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "viewport_frame_lifecycle_service.hpp"
#include "viewport_artifact_service.hpp"

namespace lfs::vis {

    namespace {
        constexpr int kResizeDebounceFrames = 3;
    }

    ViewportFrameLifecycleService::ResizeResult
    ViewportFrameLifecycleService::handleViewportResize(const glm::ivec2& current_size) {
        ResizeResult result;
        const bool resize_is_active = resize_active_.load(std::memory_order_relaxed);

        if (current_size != last_viewport_size_) {
            const bool had_viewport_size = last_viewport_size_.x > 0 && last_viewport_size_.y > 0;
            last_viewport_size_ = current_size;
            if (!had_viewport_size) {
                resize_debounce_ = 0;
                result.dirty = DirtyFlag::VIEWPORT | DirtyFlag::CAMERA | DirtyFlag::OVERLAY;
            } else {
                resize_debounce_ = kResizeDebounceFrames;
                result.dirty = DirtyFlag::OVERLAY;
            }
            return result;
        }

        if (resize_debounce_ > 0 && !resize_is_active) {
            if (--resize_debounce_ == 0) {
                result.dirty = DirtyFlag::VIEWPORT | DirtyFlag::CAMERA;
                result.completed = true;
            } else {
                result.dirty = DirtyFlag::OVERLAY;
            }
        }
        return result;
    }

    ViewportFrameLifecycleService::ModelChangeResult
    ViewportFrameLifecycleService::handleModelChange(const size_t model_ptr,
                                                     ViewportArtifactService& viewport_artifacts) {
        if (model_ptr == last_model_ptr_) {
            return {};
        }

        const ModelChangeResult result{
            .changed = true,
            .previous_model_ptr = last_model_ptr_};
        last_model_ptr_ = model_ptr;
        viewport_artifacts.clearViewportOutput();
        return result;
    }

    DirtyMask ViewportFrameLifecycleService::handleTrainingRefresh(const bool is_training,
                                                                   const float refresh_interval_sec) {
        if (!is_training) {
            return 0;
        }

        const auto now = std::chrono::steady_clock::now();
        const auto interval = std::chrono::duration<float>(refresh_interval_sec);
        if (now - last_training_render_ <= interval) {
            return 0;
        }

        last_training_render_ = now;
        return DirtyFlag::SPLATS;
    }

    DirtyMask ViewportFrameLifecycleService::requiredDirtyMask(const bool has_viewport_output,
                                                               const bool has_renderable_content,
                                                               const SplitViewMode split_view_mode) const {
        DirtyMask dirty = 0;
        if (!has_viewport_output &&
            (has_renderable_content || splitViewEnabled(split_view_mode))) {
            dirty |= DirtyFlag::ALL;
        }
        return dirty;
    }

    DirtyMask ViewportFrameLifecycleService::setViewportResizeActive(const bool active) {
        const bool was_active = resize_active_.exchange(active);
        if (!was_active || active) {
            return 0;
        }

        if (resize_debounce_ == 0) {
            resize_debounce_ = 1;
        }

        return DirtyFlag::VIEWPORT | DirtyFlag::CAMERA | DirtyFlag::OVERLAY;
    }

    DirtyMask ViewportFrameLifecycleService::deferViewportRefresh() {
        resize_debounce_ = kResizeDebounceFrames;
        return DirtyFlag::OVERLAY;
    }

} // namespace lfs::vis

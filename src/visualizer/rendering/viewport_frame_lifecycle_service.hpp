/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "dirty_flags.hpp"
#include "rendering_types.hpp"
#include <atomic>
#include <chrono>
#include <utility>

namespace lfs::vis {

    class ViewportArtifactService;

    class LFS_VIS_API ViewportFrameLifecycleService {
    public:
        struct ResizeResult {
            DirtyMask dirty = 0;
            bool completed = false;
        };

        struct ModelChangeResult {
            bool changed = false;
            size_t previous_model_ptr = 0;
        };

        [[nodiscard]] ResizeResult handleViewportResize(const glm::ivec2& current_size);
        [[nodiscard]] ModelChangeResult handleModelChange(size_t model_ptr, ViewportArtifactService& viewport_artifacts);
        [[nodiscard]] DirtyMask handleTrainingRefresh(bool is_training, float refresh_interval_sec);
        [[nodiscard]] DirtyMask requiredDirtyMask(bool has_viewport_output,
                                                  bool has_renderable_content,
                                                  SplitViewMode split_view_mode) const;
        [[nodiscard]] DirtyMask setViewportResizeActive(bool active);
        [[nodiscard]] bool isResizeDeferring() const {
            return resize_active_.load(std::memory_order_relaxed) || resize_debounce_ > 0;
        }
        bool consumeResizeCompleted() { return std::exchange(resize_completed_, false); }
        void noteResizeCompleted() { resize_completed_ = true; }
        void resetViewportSize() { last_viewport_size_ = glm::ivec2(0, 0); }
        void resetModelTracking() { last_model_ptr_ = 0; }
        [[nodiscard]] glm::ivec2 lastViewportSize() const { return last_viewport_size_; }

    private:
        glm::ivec2 last_viewport_size_{0, 0};
        size_t last_model_ptr_ = 0;
        std::chrono::steady_clock::time_point last_training_render_{};
        std::atomic<bool> resize_active_{false};
        int resize_debounce_ = 0;
        bool resize_completed_ = false;
    };

} // namespace lfs::vis

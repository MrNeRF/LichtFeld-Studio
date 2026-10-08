/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "dirty_flags.hpp"
#include <atomic>
#include <chrono>
#include <cstdint>

namespace lfs::vis {

    struct RenderAnimationState {
        std::atomic<bool> pivot_active{false};
        std::atomic<int64_t> pivot_end_ns{0};
        std::atomic<bool> overlay_active{false};

        static int64_t toNs(std::chrono::steady_clock::time_point tp) {
            return std::chrono::duration_cast<std::chrono::nanoseconds>(tp.time_since_epoch()).count();
        }

        static std::chrono::steady_clock::time_point fromNs(int64_t ns) {
            return std::chrono::steady_clock::time_point(std::chrono::nanoseconds(ns));
        }

        [[nodiscard]] DirtyMask pollDirtyState() {
            if (pivot_active.load() &&
                std::chrono::steady_clock::now() < fromNs(pivot_end_ns.load(std::memory_order_acquire))) {
                return DirtyFlag::CAMERA | DirtyFlag::OVERLAY;
            }
            pivot_active.store(false);

            if (overlay_active.load()) {
                return DirtyFlag::OVERLAY;
            }

            return 0;
        }

        void setPivotAnimationEndTime(std::chrono::steady_clock::time_point end_time) {
            pivot_end_ns.store(toNs(end_time), std::memory_order_release);
            pivot_active.store(true);
        }

        void setOverlayAnimationActive(bool active) { overlay_active.store(active); }
    };

} // namespace lfs::vis

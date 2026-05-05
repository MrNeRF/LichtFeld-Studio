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
        static constexpr float SELECTION_FLASH_DURATION_SEC = 0.5f;

        std::atomic<bool> pivot_active{false};
        std::atomic<int64_t> pivot_end_ns{0};
        std::atomic<bool> selection_flash_active{false};
        std::atomic<int64_t> selection_flash_start_ns{0};
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

            if (selection_flash_active.load()) {
                const auto elapsed = std::chrono::steady_clock::now() -
                                     fromNs(selection_flash_start_ns.load(std::memory_order_acquire));
                if (std::chrono::duration<float>(elapsed).count() < SELECTION_FLASH_DURATION_SEC) {
                    return DirtyFlag::MESH | DirtyFlag::OVERLAY;
                }
                selection_flash_active.store(false);
            }

            if (overlay_active.load()) {
                return DirtyFlag::OVERLAY;
            }

            return 0;
        }

        void setPivotAnimationEndTime(std::chrono::steady_clock::time_point end_time) {
            pivot_end_ns.store(toNs(end_time), std::memory_order_release);
            pivot_active.store(true);
        }

        [[nodiscard]] DirtyMask triggerSelectionFlash() {
            selection_flash_start_ns.store(toNs(std::chrono::steady_clock::now()), std::memory_order_release);
            selection_flash_active.store(true);
            return DirtyFlag::MESH | DirtyFlag::OVERLAY;
        }

        void setOverlayAnimationActive(bool active) { overlay_active.store(active); }

        [[nodiscard]] float selectionFlashIntensity() const {
            if (!selection_flash_active.load()) {
                return 0.0f;
            }
            const float t = std::chrono::duration<float>(
                                std::chrono::steady_clock::now() -
                                fromNs(selection_flash_start_ns.load(std::memory_order_acquire)))
                                .count() /
                            SELECTION_FLASH_DURATION_SEC;
            if (t >= 1.0f) {
                return 0.0f;
            }
            return 1.0f - t * t;
        }
    };

} // namespace lfs::vis

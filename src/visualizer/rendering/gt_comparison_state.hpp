/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/tensor.hpp"
#include "gt_comparison_geometry.hpp"
#include <chrono>
#include <memory>
#include <optional>
#include <string>

namespace lfs::vis {
    struct GTComparisonActualFrameSnapshot {
        detail::GTComparisonSourceKey source_key;
        uint64_t source_generation = 0;
        glm::ivec2 full_extent{0, 0};
        glm::ivec2 framebuffer_extent{0, 0};
        detail::GTComparisonCrop crop{};
    };
    struct GTComparisonActualSizeState {
        struct TileFailure {
            detail::GTComparisonTileKey key;
            std::chrono::steady_clock::time_point time{};
            std::string error;

            [[nodiscard]] bool suppresses(
                const detail::GTComparisonTileKey& candidate,
                const std::chrono::steady_clock::time_point now,
                const std::chrono::steady_clock::duration cooldown) const {
                return key == candidate &&
                       time.time_since_epoch().count() != 0 &&
                       now - time < cooldown;
            }
        };

        detail::GTComparisonSourceKey source_key;
        uint64_t source_generation = 0;
        std::shared_ptr<lfs::core::Tensor> cpu_source;
        std::shared_ptr<lfs::core::Tensor> cuda_source;
        glm::ivec2 full_extent{0, 0};
        glm::ivec2 framebuffer_extent{0, 0};
        detail::GTComparisonCrop crop{};
        std::optional<glm::dvec2> desired_crop_center;
        std::optional<detail::GTComparisonTileKey> tile_key;
        std::optional<TileFailure> tile_failure;
        // Retain the failure through automatic retries until a native frame publishes.
        std::string error;
        std::shared_ptr<lfs::core::Tensor> visible_tile;
        std::shared_ptr<lfs::core::Tensor> fit_fallback;
        std::chrono::steady_clock::time_point requested_at{};
        bool lookup_timed = false;
        bool source_cache_hit = false;

        void invalidateTile() {
            tile_key.reset();
            visible_tile.reset();
        }
        void reset() { *this = {}; }
    };
} // namespace lfs::vis

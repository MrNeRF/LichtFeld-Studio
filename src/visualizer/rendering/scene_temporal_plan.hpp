/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "rendering/scene_upscaler_registry.hpp"
#include "rendering/temporal_frame_tracker.hpp"

#include <glm/glm.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace lfs::vis {

    enum class SceneHistoryStorage : std::uint8_t {
        None = 0,
        Tensor,
        VulkanImage,
    };

    struct SceneTemporalPlan {
        SceneUpscalerRequirements requirements;
        glm::ivec2 render_extent{0, 0};
        glm::ivec2 output_extent{0, 0};

        [[nodiscard]] constexpr bool active() const {
            return requirements.any();
        }

        [[nodiscard]] constexpr bool temporal() const {
            return requirements.temporal();
        }

        [[nodiscard]] constexpr bool valid() const {
            if (!active()) {
                return render_extent == glm::ivec2(0) && output_extent == glm::ivec2(0);
            }
            return render_extent.x > 0 && render_extent.y > 0 && output_extent.x > 0 &&
                   output_extent.y > 0;
        }

        [[nodiscard]] constexpr bool zeroCost() const {
            return !active() && valid();
        }

        [[nodiscard]] constexpr bool needsHistoryColor() const {
            return requirements.history;
        }

        [[nodiscard]] constexpr bool needsHistoryDepth() const {
            return requirements.history && requirements.depth;
        }
    };

    [[nodiscard]] constexpr SceneTemporalPlan makeSceneTemporalPlan(
        const SceneUpscalerRequirements requirements,
        const glm::ivec2 render_extent,
        const glm::ivec2 output_extent) {
        if (!requirements.any()) {
            return {};
        }
        return {
            .requirements = requirements,
            .render_extent = render_extent,
            .output_extent = output_extent,
        };
    }

    struct SceneHistoryContract {
        SceneHistoryStorage color_storage = SceneHistoryStorage::None;
        SceneHistoryStorage depth_storage = SceneHistoryStorage::None;
        int width = 0;
        int height = 0;
        std::uint64_t sequence = 0;

        [[nodiscard]] constexpr bool available() const {
            return color_storage != SceneHistoryStorage::None;
        }

        [[nodiscard]] constexpr bool hasDepth() const {
            return depth_storage != SceneHistoryStorage::None;
        }

        [[nodiscard]] constexpr bool valid() const {
            if (!available()) {
                return color_storage == SceneHistoryStorage::None &&
                       depth_storage == SceneHistoryStorage::None && width == 0 && height == 0 &&
                       sequence == 0;
            }
            return width > 0 && height > 0;
        }

        [[nodiscard]] constexpr bool matchesOutputExtent(const int output_width,
                                                         const int output_height) const {
            return available() && valid() && width == output_width && height == output_height;
        }
    };

    [[nodiscard]] constexpr SceneHistoryContract makeSceneHistoryContract(
        const bool available,
        const SceneHistoryStorage color_storage,
        const SceneHistoryStorage depth_storage,
        const int width,
        const int height,
        const std::uint64_t sequence) {
        if (!available) {
            return {};
        }
        return {
            .color_storage = color_storage,
            .depth_storage = depth_storage,
            .width = width,
            .height = height,
            .sequence = sequence,
        };
    }

    class SceneHistoryTracker {
    public:
        [[nodiscard]] SceneHistoryContract prepare(
            const TemporalViewId view,
            const SceneTemporalPlan& plan,
            const TemporalFrameState& frame) const {
            if (!plan.valid() || !plan.needsHistoryColor() || !frame.history_valid) {
                return {};
            }
            const auto& history = entries_.at(index(view));
            if (!history || !history->valid() ||
                !history->matchesOutputExtent(plan.output_extent.x, plan.output_extent.y) ||
                history->sequence != frame.sequence) {
                return {};
            }
            return *history;
        }

        [[nodiscard]] bool commit(const TemporalViewId view,
                                  const SceneTemporalPlan& plan,
                                  const TemporalFrameState& frame,
                                  const SceneHistoryStorage color_storage,
                                  const SceneHistoryStorage depth_storage) {
            auto& history = entries_.at(index(view));
            if (!plan.valid() || !plan.needsHistoryColor()) {
                history.reset();
                return false;
            }
            if (color_storage == SceneHistoryStorage::None ||
                (plan.needsHistoryDepth() && depth_storage == SceneHistoryStorage::None)) {
                history.reset();
                return false;
            }
            history = makeSceneHistoryContract(
                true,
                color_storage,
                plan.needsHistoryDepth() ? depth_storage : SceneHistoryStorage::None,
                plan.output_extent.x,
                plan.output_extent.y,
                frame.sequence + 1);
            return history->valid();
        }

        void reset(const TemporalViewId view) {
            entries_.at(index(view)).reset();
        }

        void resetAll() {
            for (auto& history : entries_) {
                history.reset();
            }
        }

    private:
        [[nodiscard]] static constexpr std::size_t index(const TemporalViewId view) {
            return static_cast<std::size_t>(view);
        }

        std::array<std::optional<SceneHistoryContract>,
                   static_cast<std::size_t>(TemporalViewId::Count)>
            entries_{};
    };

} // namespace lfs::vis

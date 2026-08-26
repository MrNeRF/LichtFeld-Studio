/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "rendering/temporal_frame_tracker.hpp"

#include <glm/glm.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace lfs::vis {

    struct SceneTemporalRequirements {
        bool depth = false;
        bool motion = false;
        bool jitter = false;
        bool history_color = false;
        bool history_depth = false;

        [[nodiscard]] constexpr bool any() const noexcept {
            return depth || motion || jitter || history_color || history_depth;
        }

        [[nodiscard]] constexpr bool valid() const noexcept {
            return !history_depth || (history_color && depth);
        }

        constexpr bool operator==(const SceneTemporalRequirements&) const = default;
    };

    struct SceneTemporalPlan {
        SceneTemporalRequirements requirements;
        glm::ivec2 render_extent{0, 0};
        glm::ivec2 output_extent{0, 0};

        [[nodiscard]] constexpr bool active() const noexcept {
            return requirements.any();
        }

        [[nodiscard]] constexpr bool valid() const noexcept {
            if (!requirements.valid()) {
                return false;
            }
            if (!active()) {
                return render_extent == glm::ivec2(0) && output_extent == glm::ivec2(0);
            }
            return render_extent.x > 0 && render_extent.y > 0 && output_extent.x > 0 &&
                   output_extent.y > 0;
        }

        [[nodiscard]] constexpr bool zeroCost() const noexcept {
            return !active() && valid();
        }

        constexpr bool operator==(const SceneTemporalPlan&) const = default;
    };

    [[nodiscard]] constexpr SceneTemporalPlan makeSceneTemporalPlan(
        const SceneTemporalRequirements requirements,
        const glm::ivec2 render_extent,
        const glm::ivec2 output_extent) noexcept {
        if (!requirements.any()) {
            return {};
        }
        return {
            .requirements = requirements,
            .render_extent = render_extent,
            .output_extent = output_extent,
        };
    }

    enum class SceneHistoryStorage : std::uint8_t {
        None = 0,
        Tensor,
        VulkanImage,
    };

    struct SceneHistoryContract {
        SceneHistoryStorage color_storage = SceneHistoryStorage::None;
        SceneHistoryStorage depth_storage = SceneHistoryStorage::None;
        glm::ivec2 color_extent{0, 0};
        glm::ivec2 depth_extent{0, 0};
        std::uint64_t sequence = 0;

        [[nodiscard]] constexpr bool available() const noexcept {
            return color_storage != SceneHistoryStorage::None;
        }

        [[nodiscard]] constexpr bool hasDepth() const noexcept {
            return depth_storage != SceneHistoryStorage::None;
        }

        [[nodiscard]] constexpr bool valid() const noexcept {
            if (!available()) {
                return color_storage == SceneHistoryStorage::None &&
                       depth_storage == SceneHistoryStorage::None && color_extent == glm::ivec2(0) &&
                       depth_extent == glm::ivec2(0) && sequence == 0;
            }
            if (color_extent.x <= 0 || color_extent.y <= 0) {
                return false;
            }
            return hasDepth() ? depth_extent.x > 0 && depth_extent.y > 0
                              : depth_extent == glm::ivec2(0);
        }

        [[nodiscard]] constexpr bool matches(const SceneTemporalPlan& plan) const noexcept {
            return available() && valid() && color_extent == plan.output_extent &&
                   (!plan.requirements.history_depth ||
                    (hasDepth() && depth_extent == plan.render_extent));
        }
    };

    class LFS_VIS_API SceneHistoryTracker {
    public:
        [[nodiscard]] SceneHistoryContract prepare(TemporalViewId view,
                                                   const SceneTemporalPlan& plan,
                                                   const TemporalFrameState& frame) const;
        [[nodiscard]] bool commit(TemporalViewId view,
                                  const SceneTemporalPlan& plan,
                                  const TemporalFrameState& frame,
                                  SceneHistoryStorage color_storage,
                                  SceneHistoryStorage depth_storage);
        void reset(TemporalViewId view);
        void resetAll();

    private:
        [[nodiscard]] static constexpr std::size_t index(const TemporalViewId view) noexcept {
            return static_cast<std::size_t>(view);
        }

        std::array<std::optional<SceneHistoryContract>,
                   static_cast<std::size_t>(TemporalViewId::Count)>
            entries_{};
    };

} // namespace lfs::vis

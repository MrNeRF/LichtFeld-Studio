/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"

#include <cstdint>
#include <glm/glm.hpp>

namespace lfs::vis {

    enum class SceneHistoryRejection : std::uint32_t {
        None = 0,
        NoHistory = 1u << 0u,
        InvalidMotion = 1u << 1u,
        OutsideHistory = 1u << 2u,
        Disocclusion = 1u << 3u,
        InvalidColor = 1u << 4u,
    };

    struct SceneTemporalResolveSettings {
        float history_weight = 0.9f;
        float depth_threshold = 0.01f;
        float motion_rejection_pixels = 128.0f;
    };

    struct SceneTemporalResolveSample {
        glm::vec4 current{0.0f};
        glm::vec4 history{0.0f};
        glm::vec4 neighborhood_min{0.0f};
        glm::vec4 neighborhood_max{1.0f};
        glm::vec2 motion_pixels{0.0f};
        glm::vec2 current_pixel{0.0f};
        glm::ivec2 output_extent{0, 0};
        float current_depth = 0.0f;
        float history_depth = 0.0f;
        bool history_valid = false;
        bool depth_available = false;
    };

    struct SceneTemporalResolveResult {
        glm::vec4 color{0.0f};
        glm::vec2 previous_uv{0.0f};
        float effective_history_weight = 0.0f;
        SceneHistoryRejection rejection = SceneHistoryRejection::None;

        [[nodiscard]] constexpr bool usedHistory() const {
            return rejection == SceneHistoryRejection::None && effective_history_weight > 0.0f;
        }
    };

    [[nodiscard]] LFS_VIS_API SceneTemporalResolveResult resolveSceneTemporalSample(
        const SceneTemporalResolveSample& sample,
        const SceneTemporalResolveSettings& settings = {});

} // namespace lfs::vis

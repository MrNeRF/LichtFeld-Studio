/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"

#include <glm/glm.hpp>

#include <cstdint>

namespace lfs::vis {

    enum class SceneTemporalQuality : std::uint8_t {
        Performance = 0,
        Balanced,
        Quality,
    };

    enum class SceneHistoryRejection : std::uint8_t {
        None = 0,
        NoHistory,
        InvalidCurrent,
        InvalidHistory,
        InvalidMotion,
        OutsideHistory,
        Disocclusion,
    };

    struct SceneTemporalResolveSettings {
        float history_weight = 0.90f;
        float depth_relative_threshold = 0.01f;
        float depth_absolute_threshold = 1e-4f;
        float motion_rejection_pixels = 128.0f;
        float motion_confidence_pixels = 0.30f;
        float current_sharpness = 0.10f;
    };

    [[nodiscard]] LFS_VIS_API SceneTemporalResolveSettings sceneTemporalQualitySettings(
        SceneTemporalQuality quality) noexcept;

    // TemporalFrameInput stores projection jitter in OpenGL NDC (+Y up).
    // Resolve and motion operate in top-left image pixels (+Y down), followed
    // by the optional storage-orientation flip selected by flip_y.
    [[nodiscard]] LFS_VIS_API glm::vec2 sceneTemporalJitterPixels(
        glm::vec2 jitter_ndc, glm::ivec2 render_extent, bool flip_y) noexcept;

    [[nodiscard]] LFS_VIS_API float sceneTemporalHistoryWeight(
        float configured_weight, std::uint64_t accumulated_frames) noexcept;

    struct SceneTemporalResolveSample {
        glm::vec4 current{0.0f};
        glm::vec4 history{0.0f};
        glm::vec3 neighborhood_min{0.0f};
        glm::vec3 neighborhood_max{1.0f};
        glm::vec3 neighborhood_cross_sum{0.0f};
        glm::vec2 current_pixel_center{0.0f};
        // Jitter-free displacement from the current pixel to its previous-frame
        // position, in top-left render pixels.
        glm::vec2 current_to_previous_pixels{0.0f};
        glm::vec2 current_jitter_pixels{0.0f};
        glm::vec2 previous_jitter_pixels{0.0f};
        glm::ivec2 motion_extent{0, 0};
        glm::ivec2 output_extent{0, 0};
        float current_linear_depth = 0.0f;
        float history_linear_depth = 0.0f;
        float depth_far_plane = 0.0f;
        bool history_valid = false;
        bool depth_available = false;
    };

    struct SceneTemporalResolveResult {
        glm::vec4 color{0.0f};
        glm::vec2 current_render_uv{0.0f};
        glm::vec2 previous_render_uv{0.0f};
        glm::vec2 previous_uv{0.0f};
        float effective_history_weight = 0.0f;
        SceneHistoryRejection rejection = SceneHistoryRejection::None;

        [[nodiscard]] constexpr bool usedHistory() const noexcept {
            return rejection == SceneHistoryRejection::None &&
                   effective_history_weight > 0.0f;
        }
    };

    [[nodiscard]] LFS_VIS_API SceneTemporalResolveResult resolveSceneTemporalSample(
        const SceneTemporalResolveSample& sample,
        const SceneTemporalResolveSettings& settings = {}) noexcept;

} // namespace lfs::vis

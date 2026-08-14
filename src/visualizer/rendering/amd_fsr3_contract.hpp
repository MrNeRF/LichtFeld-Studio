/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "rendering/vulkan_scene_upscaler_adapter.hpp"

#include <array>

namespace lfs::vis {

    inline constexpr std::array<float, 3> AMD_FSR3_RECOMMENDED_INPUT_SCALES{
        0.5f,
        1.0f / 1.7f,
        1.0f / 1.5f,
    };

    [[nodiscard]] constexpr float amdFsr3RecommendedInputScale(
        const SceneTemporalQuality quality) {
        switch (quality) {
        case SceneTemporalQuality::Performance:
            return AMD_FSR3_RECOMMENDED_INPUT_SCALES[0];
        case SceneTemporalQuality::Balanced:
            return AMD_FSR3_RECOMMENDED_INPUT_SCALES[1];
        case SceneTemporalQuality::Quality:
            return AMD_FSR3_RECOMMENDED_INPUT_SCALES[2];
        }
        return AMD_FSR3_RECOMMENDED_INPUT_SCALES[1];
    }

    // FidelityFX expects the actual pixel offset applied to the rendered camera,
    // not the negated Halton sample sometimes used by SDK examples before their
    // projection-convention conversion. LFS shifts cx/cy by these exact values.
    [[nodiscard]] constexpr glm::vec2 amdFsr3DispatchJitterOffset(
        const glm::vec2 applied_camera_jitter_pixels) {
        return applied_camera_jitter_pixels;
    }

    [[nodiscard]] inline bool validAmdFsr3Dispatch(
        const VulkanSceneUpscalerDispatch& dispatch) {
        constexpr SceneUpscalerRequirements requirements{
            .depth = true,
            .motion_vectors = true,
            .jitter = true,
            .history = true,
        };
        return dispatch.valid(requirements) && dispatch.validCamera() &&
               dispatch.frame_time_seconds > 0.0f &&
               dispatch.color.format == VK_FORMAT_R8G8B8A8_UNORM &&
               dispatch.depth.format == VK_FORMAT_R32_SFLOAT &&
               dispatch.motion.format == VK_FORMAT_R16G16_SFLOAT &&
               dispatch.color.valid_extent == dispatch.depth.valid_extent &&
               dispatch.color.valid_extent == dispatch.motion.valid_extent &&
               dispatch.output_extent.x >= dispatch.color.valid_extent.x &&
               dispatch.output_extent.y >= dispatch.color.valid_extent.y;
    }

} // namespace lfs::vis

/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <glm/glm.hpp>

#include <cmath>
#include <optional>

namespace lfs::vis {

    struct SceneMotionReprojectionParams {
        glm::mat4 inverse_current_view_projection{1.0f};
        glm::mat4 previous_view_projection{1.0f};
        glm::ivec2 render_extent{0, 0};
        bool flip_y = false;

        [[nodiscard]] bool valid() const {
            if (render_extent.x <= 0 || render_extent.y <= 0) {
                return false;
            }
            for (int column = 0; column < 4; ++column) {
                for (int row = 0; row < 4; ++row) {
                    if (!std::isfinite(inverse_current_view_projection[column][row]) ||
                        !std::isfinite(previous_view_projection[column][row])) {
                        return false;
                    }
                }
            }
            return true;
        }
    };

    [[nodiscard]] inline std::optional<glm::vec2> reprojectSceneMotionPixels(
        const SceneMotionReprojectionParams& params,
        const glm::vec2 pixel_center,
        const float ndc_depth) {
        constexpr float MIN_ABS_W = 1e-7f;
        if (!params.valid() || !std::isfinite(pixel_center.x) ||
            !std::isfinite(pixel_center.y) || pixel_center.x < 0.0f || pixel_center.y < 0.0f ||
            pixel_center.x >= static_cast<float>(params.render_extent.x) ||
            pixel_center.y >= static_cast<float>(params.render_extent.y) ||
            !std::isfinite(ndc_depth) || ndc_depth < 0.0f || ndc_depth >= 1.0f) {
            return std::nullopt;
        }

        const glm::vec2 extent(params.render_extent);
        const glm::vec2 current_ndc{
            2.0f * pixel_center.x / extent.x - 1.0f,
            1.0f - 2.0f * pixel_center.y / extent.y};
        glm::vec4 world = params.inverse_current_view_projection *
                          glm::vec4(current_ndc, ndc_depth, 1.0f);
        if (!std::isfinite(world.w) || std::abs(world.w) <= MIN_ABS_W) {
            return std::nullopt;
        }
        world /= world.w;

        const glm::vec4 previous_clip = params.previous_view_projection * world;
        if (!std::isfinite(previous_clip.x) || !std::isfinite(previous_clip.y) ||
            !std::isfinite(previous_clip.w) || std::abs(previous_clip.w) <= MIN_ABS_W) {
            return std::nullopt;
        }
        const glm::vec2 previous_ndc = glm::vec2(previous_clip) / previous_clip.w;
        glm::vec2 motion_pixels{
            (previous_ndc.x - current_ndc.x) * 0.5f * extent.x,
            (current_ndc.y - previous_ndc.y) * 0.5f * extent.y};
        if (params.flip_y) {
            motion_pixels.y = -motion_pixels.y;
        }
        return motion_pixels;
    }

} // namespace lfs::vis

/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cmath>
#include <cstdint>

namespace lfs::vis {

    enum class SceneDepthEncoding : std::uint8_t {
        Unavailable = 0,
        LinearView,
        VulkanNdc,
    };

    enum class SceneDepthStorage : std::uint8_t {
        None = 0,
        Tensor,
        VulkanImage,
    };

    struct SceneDepthContract {
        SceneDepthEncoding encoding = SceneDepthEncoding::Unavailable;
        SceneDepthStorage storage = SceneDepthStorage::None;
        int width = 0;
        int height = 0;
        float near_plane = 0.0f;
        float far_plane = 0.0f;
        bool orthographic = false;
        bool flip_y = false;

        [[nodiscard]] constexpr bool available() const {
            return encoding != SceneDepthEncoding::Unavailable &&
                   storage != SceneDepthStorage::None;
        }

        [[nodiscard]] bool valid() const {
            if (!available()) {
                return encoding == SceneDepthEncoding::Unavailable &&
                       storage == SceneDepthStorage::None && width == 0 && height == 0;
            }
            return width > 0 && height > 0 && std::isfinite(near_plane) &&
                   std::isfinite(far_plane) && near_plane > 0.0f &&
                   far_plane > near_plane;
        }

        [[nodiscard]] bool matchesRenderExtent(const int render_width,
                                               const int render_height) const {
            return valid() && width == render_width && height == render_height;
        }

        [[nodiscard]] constexpr bool requiresLinearization() const {
            return encoding == SceneDepthEncoding::VulkanNdc;
        }
    };

    [[nodiscard]] inline SceneDepthContract makeSceneDepthContract(
        const bool available,
        const SceneDepthStorage storage,
        const bool depth_is_ndc,
        const int width,
        const int height,
        const float near_plane,
        const float far_plane,
        const bool orthographic,
        const bool flip_y) {
        if (!available) {
            return {};
        }
        return {
            .encoding = depth_is_ndc ? SceneDepthEncoding::VulkanNdc
                                     : SceneDepthEncoding::LinearView,
            .storage = storage,
            .width = width,
            .height = height,
            .near_plane = near_plane,
            .far_plane = far_plane,
            .orthographic = orthographic,
            .flip_y = flip_y,
        };
    }

} // namespace lfs::vis

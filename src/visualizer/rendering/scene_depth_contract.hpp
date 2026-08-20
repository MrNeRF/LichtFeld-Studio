/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <glm/glm.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>

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

        [[nodiscard]] constexpr bool available() const noexcept {
            return encoding != SceneDepthEncoding::Unavailable &&
                   storage != SceneDepthStorage::None;
        }

        [[nodiscard]] bool valid() const noexcept {
            if (!available()) {
                return encoding == SceneDepthEncoding::Unavailable &&
                       storage == SceneDepthStorage::None && width == 0 && height == 0 &&
                       near_plane == 0.0f && far_plane == 0.0f && !orthographic && !flip_y;
            }
            return width > 0 && height > 0 && std::isfinite(near_plane) &&
                   std::isfinite(far_plane) && near_plane > 0.0f && far_plane > near_plane;
        }

        [[nodiscard]] bool matchesRenderExtent(const glm::ivec2 extent) const noexcept {
            return available() && valid() && width == extent.x && height == extent.y;
        }

        [[nodiscard]] constexpr bool requiresLinearization() const noexcept {
            return encoding == SceneDepthEncoding::VulkanNdc;
        }
    };

    [[nodiscard]] inline SceneDepthContract makeSceneDepthContract(
        const bool available,
        const SceneDepthStorage storage,
        const SceneDepthEncoding encoding,
        const glm::ivec2 extent,
        const float near_plane,
        const float far_plane,
        const bool orthographic,
        const bool flip_y) noexcept {
        if (!available) {
            return {};
        }
        return {
            .encoding = encoding,
            .storage = storage,
            .width = extent.x,
            .height = extent.y,
            .near_plane = near_plane,
            .far_plane = far_plane,
            .orthographic = orthographic,
            .flip_y = flip_y,
        };
    }

    [[nodiscard]] inline std::optional<float> sceneDepthToVulkanNdc(
        const float depth, const SceneDepthContract& contract) noexcept {
        if (!contract.available() || !contract.valid() || !std::isfinite(depth)) {
            return std::nullopt;
        }
        if (contract.encoding == SceneDepthEncoding::VulkanNdc) {
            return depth >= 0.0f && depth < 1.0f ? std::optional<float>(depth) : std::nullopt;
        }
        if (depth <= 0.0f) {
            return std::nullopt;
        }
        const float ndc = contract.orthographic
                              ? (depth - contract.near_plane) /
                                    (contract.far_plane - contract.near_plane)
                              : contract.far_plane /
                                    (contract.far_plane - contract.near_plane) *
                                    (1.0f - contract.near_plane /
                                                std::max(depth, contract.near_plane));
        return std::isfinite(ndc) && ndc >= 0.0f && ndc < 1.0f
                   ? std::optional<float>(ndc)
                   : std::nullopt;
    }

} // namespace lfs::vis

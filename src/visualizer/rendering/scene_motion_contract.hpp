/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstdint>

namespace lfs::vis {

    enum class SceneMotionEncoding : std::uint8_t {
        Unavailable = 0,
        PixelDisplacement,
        NormalizedUvDisplacement,
        NdcDisplacement,
    };

    enum class SceneMotionStorage : std::uint8_t {
        None = 0,
        Tensor,
        VulkanImage,
    };

    enum class SceneMotionDirection : std::uint8_t {
        CurrentToPrevious = 0,
        PreviousToCurrent,
    };

    struct SceneMotionContract {
        SceneMotionEncoding encoding = SceneMotionEncoding::Unavailable;
        SceneMotionStorage storage = SceneMotionStorage::None;
        SceneMotionDirection direction = SceneMotionDirection::CurrentToPrevious;
        int width = 0;
        int height = 0;
        bool includes_jitter = false;
        bool flip_y = false;

        [[nodiscard]] constexpr bool available() const {
            return encoding != SceneMotionEncoding::Unavailable &&
                   storage != SceneMotionStorage::None;
        }

        [[nodiscard]] constexpr bool valid() const {
            if (!available()) {
                return encoding == SceneMotionEncoding::Unavailable &&
                       storage == SceneMotionStorage::None && width == 0 && height == 0 &&
                       !includes_jitter && !flip_y;
            }
            return width > 0 && height > 0;
        }

        [[nodiscard]] constexpr bool matchesRenderExtent(const int render_width,
                                                         const int render_height) const {
            return available() && valid() && width == render_width && height == render_height;
        }

        [[nodiscard]] constexpr bool requiresPixelConversion() const {
            return encoding == SceneMotionEncoding::NormalizedUvDisplacement ||
                   encoding == SceneMotionEncoding::NdcDisplacement;
        }
    };

    [[nodiscard]] constexpr SceneMotionContract makeSceneMotionContract(
        const bool available,
        const SceneMotionStorage storage,
        const SceneMotionEncoding encoding,
        const SceneMotionDirection direction,
        const int width,
        const int height,
        const bool includes_jitter,
        const bool flip_y) {
        if (!available) {
            return {};
        }
        return {
            .encoding = encoding,
            .storage = storage,
            .direction = direction,
            .width = width,
            .height = height,
            .includes_jitter = includes_jitter,
            .flip_y = flip_y,
        };
    }

} // namespace lfs::vis

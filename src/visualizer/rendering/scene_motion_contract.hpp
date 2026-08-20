/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <glm/glm.hpp>

#include <cmath>
#include <cstdint>
#include <optional>

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

    [[nodiscard]] inline std::optional<glm::vec2> canonicalSceneMotionPixels(
        const glm::vec2 encoded_motion, const SceneMotionContract& contract) {
        if (!contract.available() || !contract.valid() ||
            !std::isfinite(encoded_motion.x) || !std::isfinite(encoded_motion.y)) {
            return std::nullopt;
        }

        glm::vec2 pixels = encoded_motion;
        if (contract.encoding == SceneMotionEncoding::NormalizedUvDisplacement) {
            pixels *= glm::vec2(contract.width, contract.height);
        } else if (contract.encoding == SceneMotionEncoding::NdcDisplacement) {
            pixels *= 0.5f * glm::vec2(contract.width, contract.height);
        }
        if (contract.flip_y) {
            pixels.y = -pixels.y;
        }
        if (contract.direction == SceneMotionDirection::PreviousToCurrent) {
            pixels = -pixels;
        }
        return pixels;
    }

    [[nodiscard]] inline std::optional<glm::vec2> currentToPreviousSceneMotionNdc(
        const glm::vec4 current_clip, const glm::vec4 previous_clip) {
        constexpr float MIN_ABS_W = 1e-7f;
        const auto finite_clip = [](const glm::vec4 value) {
            return std::isfinite(value.x) && std::isfinite(value.y) &&
                   std::isfinite(value.z) && std::isfinite(value.w);
        };
        if (!finite_clip(current_clip) || !finite_clip(previous_clip) ||
            std::abs(current_clip.w) <= MIN_ABS_W ||
            std::abs(previous_clip.w) <= MIN_ABS_W) {
            return std::nullopt;
        }
        const glm::vec2 current_ndc = glm::vec2(current_clip) / current_clip.w;
        const glm::vec2 previous_ndc = glm::vec2(previous_clip) / previous_clip.w;
        return previous_ndc - current_ndc;
    }

} // namespace lfs::vis

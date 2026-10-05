/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace lfs::io::reframe {
    // Reframe SH0 describes linear light. Ordinary viewer splats store sRGB
    // samples in SH0, so decode, apply the transfer function, and encode again.
    inline float viewerSh0(float coefficient) {
        constexpr float c0 = 0.28209479177387814f;
        if (!std::isfinite(coefficient))
            throw std::runtime_error("Reframe returned a non-finite color");
        const float linear = std::clamp(coefficient * c0 + 0.5f, 0.0f, 1.0f);
        const float srgb = linear <= 0.0031308f ? 12.92f * linear : 1.055f * std::pow(linear, 1.0f / 2.4f) - 0.055f;
        return (srgb - 0.5f) / c0;
    }

    inline std::array<float, 3> supportRadii(const std::array<float, 3>& log_scales,
                                             const std::array<float, 4>& q) {
        const float w = q[0], x = q[1], y = q[2], z = q[3];
        const std::array<std::array<float, 3>, 3> rows{{{1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)},
                                                        {2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)},
                                                        {2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)}}};
        std::array<float, 3> radii{};
        for (size_t axis = 0; axis < 3; ++axis) {
            float variance = 0;
            for (size_t component = 0; component < 3; ++component) {
                const float projected = rows[axis][component] * std::exp(log_scales[component]);
                variance += projected * projected;
            }
            radii[axis] = 3.0f * std::sqrt(variance);
        }
        return radii;
    }

    struct Placement {
        float scale = 1.0f;
        std::array<float, 3> translation{};
    };

    inline Placement groundedPlacement(float aspect, const std::array<float, 3>& lower,
                                       const std::array<float, 3>& upper) {
        if (!std::isfinite(aspect) || aspect <= 0)
            throw std::runtime_error("Invalid reconstruction aspect ratio");
        for (size_t axis = 0; axis < 3; ++axis) {
            if (!std::isfinite(lower[axis]) || !std::isfinite(upper[axis]) || upper[axis] < lower[axis])
                throw std::runtime_error("Invalid reconstruction bounds");
        }
        // A four-unit envelope fits the standard scene camera. Never enlarge
        // small reconstructions or depend on the user's current camera/FOV.
        const float extent = std::max({aspect * (upper[0] - lower[0]), upper[1] - lower[1], upper[2] - lower[2]});
        Placement result;
        result.scale = extent > 4.0f ? 4.0f / extent : 1.0f;
        // Data +Y points down; the viewer flips Y/Z. The bottom edge is
        // therefore data max-Y, and must become world Y=0.
        result.translation = {-result.scale * aspect * (lower[0] + upper[0]) * 0.5f,
                              -result.scale * upper[1],
                              -result.scale * (lower[2] + upper[2]) * 0.5f};
        return result;
    }
    // Reframe already returns degree-zero SH coefficients and linear activated
    // scales/opacity. SplatData stores log scales and opacity logits.
    inline float logScale(float scale) {
        if (!std::isfinite(scale) || scale <= 0)
            throw std::runtime_error("Reframe returned an invalid Gaussian scale");
        return std::log(scale);
    }
    inline float opacityLogit(float alpha) {
        if (!std::isfinite(alpha) || alpha < 0 || alpha > 1)
            throw std::runtime_error("Reframe returned an invalid Gaussian opacity");
        alpha = std::clamp(alpha, 1e-6f, 1.0f - 1e-6f);
        return std::log(alpha) - std::log1p(-alpha);
    }
    inline std::array<float, 4> normalizedRotation(std::array<float, 4> rotation) {
        float squared = 0;
        for (float value : rotation) {
            if (!std::isfinite(value))
                throw std::runtime_error("Reframe returned a non-finite Gaussian rotation");
            squared += value * value;
        }
        if (!std::isfinite(squared) || squared <= 1e-12f)
            throw std::runtime_error("Reframe returned a zero Gaussian rotation");
        for (float& value : rotation)
            value /= std::sqrt(squared);
        return rotation;
    }
} // namespace lfs::io::reframe

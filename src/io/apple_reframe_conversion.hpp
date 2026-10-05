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

    inline float fitVerticalFov(float source_aspect, float viewport_aspect) {
        if (!std::isfinite(source_aspect) || !std::isfinite(viewport_aspect) || source_aspect <= 0 || viewport_aspect <= 0)
            throw std::runtime_error("Invalid reconstruction aspect ratio");
        // Apple's canonical reconstruction spans [-1,1] on both image axes.
        return 2.0f * std::atan(std::max(1.0f, source_aspect / viewport_aspect)) * 180.0f / 3.14159265358979323846f;
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

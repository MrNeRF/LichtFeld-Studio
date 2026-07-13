/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <vector>

#include "core/image_io.hpp"

namespace {

    float srgb_from_linear(const float v) {
        if (v <= 0.0031308f) {
            return v * 12.92f;
        }
        return 1.055f * std::pow(v, 1.0f / 2.4f) - 0.055f;
    }

} // namespace

TEST(NormalPriorDecodeTest, SrgbEncodingToLinearRoundTrips) {
    for (const float v : {0.0f, 0.01f, 0.25f, 0.5f, 0.7354f, 0.9f, 1.0f}) {
        EXPECT_NEAR(lfs::core::srgb_encoding_to_linear(srgb_from_linear(v)), v, 1e-5f);
    }
    // The value that misled the linear decode: sRGB(0.5) encodes a zero
    // normal component but decodes linearly to 0.459.
    EXPECT_NEAR(lfs::core::srgb_encoding_to_linear(0.7354f), 0.5f, 1e-3f);
}

TEST(NormalPriorDecodeTest, SrgbNormalPriorToLinearChwFixesUnitNorm) {
    // Axis-aligned unit normals written through the sRGB display transform,
    // decoded with the (wrong) linear assumption first.
    const std::array<std::array<float, 3>, 3> normals = {{{0.f, 0.f, 1.f},
                                                          {0.f, 1.f, 0.f},
                                                          {-1.f, 0.f, 0.f}}};
    const size_t pixel_count = normals.size();
    std::vector<float> chw(pixel_count * 3);
    for (size_t i = 0; i < pixel_count; ++i) {
        for (size_t c = 0; c < 3; ++c) {
            const float encoded = srgb_from_linear(normals[i][c] * 0.5f + 0.5f);
            chw[c * pixel_count + i] = encoded * 2.0f - 1.0f; // linear decode of sRGB data
        }
    }

    for (size_t i = 0; i < pixel_count; ++i) {
        const float x = chw[i];
        const float y = chw[pixel_count + i];
        const float z = chw[2 * pixel_count + i];
        EXPECT_GT(std::abs(std::sqrt(x * x + y * y + z * z) - 1.0f), 0.15f)
            << "linear decode of sRGB data should be visibly non-unit";
    }

    lfs::core::srgb_normal_prior_to_linear_chw(chw.data(), chw.size());

    for (size_t i = 0; i < pixel_count; ++i) {
        for (size_t c = 0; c < 3; ++c) {
            EXPECT_NEAR(chw[c * pixel_count + i], normals[i][c], 2e-3f);
        }
    }
}

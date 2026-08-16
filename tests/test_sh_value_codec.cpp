/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * SH-rest 16-bit value codec unit tests.
 */

#include "lfs/training/sh_value_codec.hpp"

#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <random>
#include <vector>

using namespace lfs::training::sh_value;

TEST(ShValueCodecTest, EndpointExactRoundtrip) {
    float vals[2] = {-0.75f, 1.25f};
    float bounds[2];
    Codec16::reduce_bounds(vals, 2, bounds);
    EXPECT_FLOAT_EQ(bounds[0], -0.75f);
    EXPECT_FLOAT_EQ(bounds[1], 1.25f);

    const auto q0 = Codec16::encode(vals[0], bounds[0], bounds[1]);
    const auto q1 = Codec16::encode(vals[1], bounds[0], bounds[1]);
    EXPECT_NEAR(Codec16::decode(q0, bounds[0], bounds[1]), vals[0], 1e-6f);
    EXPECT_NEAR(Codec16::decode(q1, bounds[0], bounds[1]), vals[1], 1e-6f);
}

TEST(ShValueCodecTest, ZeroRangeDecodesToConstant) {
    // lo==hi: any code maps to lo (range clamped by encode/decode math).
    const float lo = 0.3f, hi = 0.3f;
    // encode uses range eps so codes still land; decode with lo==hi → lo
    EXPECT_FLOAT_EQ(Codec16::decode(0, lo, hi), lo);
    EXPECT_FLOAT_EQ(Codec16::decode(65535, lo, hi), lo);
}

TEST(ShValueCodecTest, RoundtripErrorBounded) {
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.0f, 0.2f);
    std::vector<float> vals(256);
    for (auto& v : vals)
        v = nd(rng);
    float bounds[2];
    Codec16::reduce_bounds(vals.data(), vals.size(), bounds);

    double mse = 0.0;
    for (float v : vals) {
        const auto q = Codec16::encode(v, bounds[0], bounds[1]);
        const float d = Codec16::decode(q, bounds[0], bounds[1]);
        const double e = static_cast<double>(d) - static_cast<double>(v);
        mse += e * e;
    }
    mse /= static_cast<double>(vals.size());
    // 16-bit over block range is tight for SH coeffs (typically |c| < few units).
    EXPECT_LT(mse, 1e-7);
}

TEST(ShValueCodecTest, BoundsReductionMatchesManual) {
    const float v[] = {0.1f, -0.5f, 0.0f, 2.0f};
    float bounds[2];
    Codec16::reduce_bounds(v, 4, bounds);
    EXPECT_FLOAT_EQ(bounds[0], -0.5f);
    EXPECT_FLOAT_EQ(bounds[1], 2.0f);
}

TEST(ShValueCodecTest, FootprintConstants) {
    EXPECT_EQ(kShNBpsFp32Sh3, 192u);
    EXPECT_EQ(kShNBpsQ16Sh3, 90u);
    EXPECT_EQ(kParamsBpsFp32Sh3, 248u);
    EXPECT_EQ(kParamsBpsQ16Sh3, 146u); // 56 + 90
    EXPECT_LT(kShNBpsQ16Sh3, kShNBpsFp32Sh3);
}

TEST(ShValueCodecTest, TestingOverrideWorks) {
    set_sh_value_quant_enabled_for_testing(true);
    EXPECT_TRUE(sh_value_quant_enabled());
    set_sh_value_quant_enabled_for_testing(false);
    EXPECT_FALSE(sh_value_quant_enabled());
    set_sh_value_quant_enabled_for_testing(std::nullopt);
}

/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "components/ppisp.hpp"
#include "core/tensor.hpp"

#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <utility>
#include <vector>

namespace {

    using lfs::core::Device;
    using lfs::training::PPISP;
    using lfs::training::PPISPConfig;

    std::vector<float> exposure_host(const PPISP& ppisp) {
        return ppisp.exposure_params().cpu().contiguous().to_vector();
    }

} // namespace

TEST(PPISPExposureSeedTest, CentersAndScalesKnownFrames) {
    PPISP ppisp(100);
    ppisp.register_frame(10, 0);
    ppisp.register_frame(20, 0);
    ppisp.register_frame(30, 0);
    ppisp.register_frame(40, 0);
    ppisp.finalize();

    const std::vector<std::pair<int, float>> uid_ev{{10, 0.0f}, {20, 2.0f}, {40, 4.0f}};
    ppisp.seed_exposure(uid_ev);

    const auto values = exposure_host(ppisp);
    ASSERT_EQ(values.size(), 4u);
    const float mean = (0.0f + 2.0f + 4.0f) / 3.0f;
    EXPECT_NEAR(values[0], 0.5f * (0.0f - mean), 1e-6f);
    EXPECT_NEAR(values[1], 0.5f * (2.0f - mean), 1e-6f);
    EXPECT_NEAR(values[2], 0.0f, 1e-6f);
    EXPECT_NEAR(values[3], 0.5f * (4.0f - mean), 1e-6f);
}

TEST(PPISPExposureSeedTest, CopyInferenceWeightsOverwritesSeed) {
    PPISPConfig config;
    config.warmup_steps = 0;

    PPISP seeded(100, config);
    seeded.register_frame(1, 0);
    seeded.register_frame(2, 0);
    seeded.register_frame(3, 0);
    seeded.register_frame(4, 0);
    seeded.finalize();
    seeded.seed_exposure({{1, -2.0f}, {2, 0.0f}, {3, 2.0f}});

    PPISP blank(100, config);
    blank.register_frame(1, 0);
    blank.register_frame(2, 0);
    blank.register_frame(3, 0);
    blank.register_frame(4, 0);
    blank.finalize();

    const auto import = seeded.copy_inference_weights_from(blank, {0, 1, 2, 3}, {0});
    ASSERT_TRUE(import) << import.error();

    const auto values = exposure_host(seeded);
    ASSERT_EQ(values.size(), 4u);
    for (float value : values) {
        EXPECT_NEAR(value, 0.0f, 1e-6f);
    }
}

TEST(PPISPApplyWithExposureTest, MatchesApplyForRegisteredFrame) {
    PPISP ppisp(100);
    ppisp.register_frame(10, 7);
    ppisp.register_frame(20, 7);
    ppisp.register_frame(30, 7);
    ppisp.finalize();
    ppisp.seed_exposure({{10, 0.0f}, {20, 2.0f}, {30, 4.0f}});

    const auto values = exposure_host(ppisp);
    ASSERT_EQ(values.size(), 3u);
    const float e = values[0];
    ASSERT_NE(e, 0.0f);

    std::vector<float> pixels(3 * 4 * 4);
    for (size_t i = 0; i < pixels.size(); ++i) {
        pixels[i] = 0.2f + 0.01f * static_cast<float>(i);
    }
    const auto rgb = lfs::core::Tensor::from_vector(pixels, {3, 4, 4}, Device::CUDA);

    const auto from_apply = ppisp.apply(rgb, 7, 10).cpu().contiguous().to_vector();
    const auto from_explicit = ppisp.apply_with_exposure(rgb, 7, e).cpu().contiguous().to_vector();
    ASSERT_EQ(from_apply.size(), from_explicit.size());
    EXPECT_EQ(std::memcmp(from_apply.data(), from_explicit.data(), from_apply.size() * sizeof(float)), 0);

    const auto other = ppisp.apply_with_exposure(rgb, 7, e + 1.0f).cpu().contiguous().to_vector();
    EXPECT_NE(std::memcmp(from_apply.data(), other.data(), from_apply.size() * sizeof(float)), 0);
}

TEST(PPISPExposureSeedTest, ClampsToExposureRange) {
    PPISP ppisp(100);
    ppisp.register_frame(1, 0);
    ppisp.register_frame(2, 0);
    ppisp.finalize();
    ppisp.seed_exposure({{1, 0.0f}, {2, 100.0f}});

    const auto values = exposure_host(ppisp);
    ASSERT_EQ(values.size(), 2u);
    EXPECT_GE(values[0], -16.0f);
    EXPECT_LE(values[0], 16.0f);
    EXPECT_GE(values[1], -16.0f);
    EXPECT_LE(values[1], 16.0f);
    EXPECT_FLOAT_EQ(values[1], 16.0f);
}

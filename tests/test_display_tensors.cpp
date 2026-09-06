/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// The depth and normal display images are tensor programs that end on the
// CPU; each is checked against the per-pixel arithmetic it replaced.

#include "core/tensor.hpp"
#include "rendering/display_tensors.hpp"
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace {
    using namespace lfs::core;
    using lfs::rendering::DepthVisualizationMode;

    constexpr size_t kHeight = 61;
    constexpr size_t kWidth = 83;
    constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
    constexpr float kInf = std::numeric_limits<float>::infinity();

    std::vector<float> depth_pattern() {
        std::vector<float> depth(kHeight * kWidth);
        for (size_t i = 0; i < depth.size(); ++i)
            depth[i] = 0.5f + static_cast<float>((i * 131) % 977) * 0.02f;
        depth[5] = kNaN;
        depth[17] = -kInf;
        depth[29] = 0.0f;
        depth[41] = -3.0f;
        depth[53] = 2.0e9f;
        depth[kHeight * kWidth - 1] = kInf;
        return depth;
    }

    bool valid_depth(const float d) { return std::isfinite(d) && d > 0.0f && d < 1.0e9f; }

    std::vector<float> reference_depth(const std::vector<float>& depth, const DepthVisualizationMode mode,
                                       const glm::vec3& background) {
        std::vector<float> valid;
        for (const float d : depth)
            if (valid_depth(d))
                valid.push_back(d);
        float lo = 0.0f, hi = 0.0f;
        if (valid.size() >= 2) {
            const auto quantile = [&](const float q) {
                const auto n = static_cast<size_t>(q * static_cast<float>(valid.size() - 1));
                std::nth_element(valid.begin(), valid.begin() + n, valid.end());
                return valid[n];
            };
            lo = quantile(0.02f);
            hi = quantile(0.98f);
        }
        const float span = hi - lo;
        const size_t pixels = depth.size();
        std::vector<float> out(3 * pixels);
        for (size_t i = 0; i < pixels; ++i) {
            glm::vec3 color = background;
            if (valid_depth(depth[i]) && span > 1.0e-6f) {
                const float t = std::clamp((depth[i] - lo) / span, 0.0f, 1.0f);
                const float near_t = 1.0f - t;
                color = mode == DepthVisualizationMode::Grayscale ? glm::vec3(near_t) : lfs::vis::depthPaletteForDisplay(near_t);
            }
            out[i] = color.r;
            out[pixels + i] = color.g;
            out[2 * pixels + i] = color.b;
        }
        return out;
    }

    void expect_close(const std::vector<float>& got, const std::vector<float>& expected, const float tolerance) {
        ASSERT_EQ(got.size(), expected.size());
        for (size_t i = 0; i < got.size(); ++i)
            EXPECT_NEAR(got[i], expected[i], tolerance) << "index=" << i;
    }
} // namespace

TEST(DisplayTensors, DepthDisplayMatchesThePerPixelReferenceInBothModesFromEitherDevice) {
    const auto depth = depth_pattern();
    const glm::vec3 background(0.1f, 0.2f, 0.3f);
    const Tensor cpu = Tensor::from_vector(depth, TensorShape{kHeight, kWidth}, Device::CPU);
    for (const Tensor& source : {cpu, cpu.to(Device::GPU)}) {
        for (const DepthVisualizationMode mode : {DepthVisualizationMode::Palette, DepthVisualizationMode::Grayscale}) {
            SCOPED_TRACE(static_cast<int>(mode));
            const auto image = lfs::vis::makeDepthDisplayTensor(source, mode, background);
            ASSERT_TRUE(image && image->is_valid());
            EXPECT_EQ(image->device(), Device::CPU);
            ASSERT_EQ(image->shape(), TensorShape({size_t{3}, kHeight, kWidth}));
            expect_close(image->to_vector(), reference_depth(depth, mode, background), 2.0e-6f);
        }
    }
}

TEST(DisplayTensors, DepthWithoutValidPixelsIsAllBackground) {
    std::vector<float> depth(kHeight * kWidth, kNaN);
    depth[3] = 7.0f;
    const glm::vec3 background(0.25f, 0.5f, 0.75f);
    const auto image = lfs::vis::makeDepthDisplayTensor(
        Tensor::from_vector(depth, TensorShape{kHeight, kWidth}, Device::CPU), DepthVisualizationMode::Palette, background);
    ASSERT_TRUE(image && image->is_valid());
    expect_close(image->to_vector(), reference_depth(depth, DepthVisualizationMode::Palette, background), 0.0f);
    EXPECT_FALSE(lfs::vis::makeDepthDisplayTensor(Tensor{}, DepthVisualizationMode::Palette, background));
}

TEST(DisplayTensors, NormalDisplayNormalizesAndGraysDegenerateNormalsInBothLayouts) {
    std::vector<float> chw(3 * kHeight * kWidth);
    const size_t pixels = kHeight * kWidth;
    for (size_t i = 0; i < pixels; ++i) {
        chw[i] = static_cast<float>(static_cast<int>((i * 7) % 41) - 20) * 0.1f;
        chw[pixels + i] = static_cast<float>(static_cast<int>((i * 11) % 37) - 18) * 0.1f;
        chw[2 * pixels + i] = static_cast<float>(static_cast<int>((i * 13) % 29) - 14) * 0.1f;
    }
    for (const size_t degenerate : {size_t{9}, size_t{200}}) {
        chw[degenerate] = 0.0f;
        chw[pixels + degenerate] = 0.0f;
        chw[2 * pixels + degenerate] = 0.0f;
    }
    chw[pixels + 300] = kNaN;
    std::vector<float> expected(3 * pixels, 0.5f);
    for (size_t i = 0; i < pixels; ++i) {
        const glm::vec3 n(chw[i], chw[pixels + i], chw[2 * pixels + i]);
        const float len = glm::length(n);
        if (std::isfinite(len) && len > 1.0e-6f) {
            const glm::vec3 c = glm::clamp(n / len * 0.5f + glm::vec3(0.5f), glm::vec3(0.0f), glm::vec3(1.0f));
            expected[i] = c.r;
            expected[pixels + i] = c.g;
            expected[2 * pixels + i] = c.b;
        }
    }
    std::vector<float> hwc(3 * pixels);
    for (size_t i = 0; i < pixels; ++i)
        for (size_t c = 0; c < 3; ++c)
            hwc[i * 3 + c] = chw[c * pixels + i];
    const Tensor chw_tensor = Tensor::from_vector(chw, TensorShape{size_t{3}, kHeight, kWidth}, Device::CPU);
    const Tensor hwc_tensor = Tensor::from_vector(hwc, TensorShape{kHeight, kWidth, size_t{3}}, Device::CPU);
    for (const Tensor& source : {chw_tensor, chw_tensor.to(Device::GPU), hwc_tensor, hwc_tensor.to(Device::GPU)}) {
        const auto image = lfs::vis::makeNormalDisplayTensor(source);
        ASSERT_TRUE(image && image->is_valid());
        EXPECT_EQ(image->device(), Device::CPU);
        ASSERT_EQ(image->shape(), TensorShape({size_t{3}, kHeight, kWidth}));
        expect_close(image->to_vector(), expected, 2.0e-6f);
    }
    EXPECT_FALSE(lfs::vis::makeNormalDisplayTensor(Tensor{}));
}

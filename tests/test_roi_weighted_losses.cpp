/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include "training/kernels/depth_loss.hpp"
#include "training/kernels/normal_consistency_loss.hpp"
#include "training/kernels/normal_loss.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

namespace {

    using lfs::core::Device;
    using lfs::core::Tensor;

    struct DepthResult {
        Tensor loss;
        Tensor grad_depth;
        Tensor grad_alpha;
    };

    DepthResult run_depth_loss(
        const Tensor& depth,
        const Tensor& alpha,
        const Tensor& target,
        const lfs::training::kernels::DepthAnchor& anchor,
        const Tensor& pixel_weight = {}) {
        const int height = static_cast<int>(depth.shape()[0]);
        const int width = static_cast<int>(depth.shape()[1]);
        const cudaStream_t stream = depth.stream();
        DepthResult result{
            .loss = Tensor::empty({size_t{1}}, Device::CUDA),
            .grad_depth = Tensor::empty(depth.shape(), Device::CUDA),
            .grad_alpha = Tensor::empty(alpha.shape(), Device::CUDA)};
        auto partials = Tensor::empty(
            {lfs::training::kernels::depth_loss_partial_count(depth.numel())},
            Device::CUDA);
        result.loss.set_stream(stream);
        result.grad_depth.set_stream(stream);
        result.grad_alpha.set_stream(stream);
        partials.set_stream(stream);
        if (pixel_weight.is_valid()) {
            pixel_weight.sync_to_stream(stream);
        }

        lfs::training::kernels::launch_depth_loss(
            depth.ptr<float>(),
            alpha.ptr<float>(),
            target.ptr<float>(),
            result.grad_depth.ptr<float>(),
            result.grad_alpha.ptr<float>(),
            result.loss.ptr<float>(),
            partials.ptr<float>(),
            width,
            height,
            0.7f,
            0.2f,
            0.0f,
            &anchor,
            stream,
            pixel_weight.is_valid() ? pixel_weight.ptr<float>() : nullptr);
        return result;
    }

    struct NormalResult {
        Tensor loss;
        Tensor grad_normal;
    };

    NormalResult run_normal_loss(
        const Tensor& rendered,
        const Tensor& alpha,
        const Tensor& target,
        const Tensor& pixel_weight = {}) {
        const int height = static_cast<int>(alpha.shape()[0]);
        const int width = static_cast<int>(alpha.shape()[1]);
        const cudaStream_t stream = rendered.stream();
        NormalResult result{
            .loss = Tensor::empty({size_t{1}}, Device::CUDA),
            .grad_normal = Tensor::empty(rendered.shape(), Device::CUDA)};
        auto partials = Tensor::empty(
            {lfs::training::kernels::normal_loss_partial_count(alpha.numel())},
            Device::CUDA);
        result.loss.set_stream(stream);
        result.grad_normal.set_stream(stream);
        partials.set_stream(stream);
        if (pixel_weight.is_valid()) {
            pixel_weight.sync_to_stream(stream);
        }

        lfs::training::kernels::launch_normal_loss(
            rendered.ptr<float>(),
            alpha.ptr<float>(),
            target.ptr<float>(),
            result.grad_normal.ptr<float>(),
            result.loss.ptr<float>(),
            partials.ptr<float>(),
            width,
            height,
            0.4f,
            stream,
            pixel_weight.is_valid() ? pixel_weight.ptr<float>() : nullptr);
        return result;
    }

    struct ConsistencyResult {
        Tensor loss;
        Tensor grad_normal;
        Tensor grad_depth;
        Tensor grad_alpha;
    };

    ConsistencyResult run_consistency_loss(
        const Tensor& rendered_normal,
        const Tensor& depth,
        const Tensor& alpha,
        const Tensor& pixel_weight = {}) {
        const int height = static_cast<int>(depth.shape()[0]);
        const int width = static_cast<int>(depth.shape()[1]);
        const cudaStream_t stream = depth.stream();
        ConsistencyResult result{
            .loss = Tensor::empty({size_t{1}}, Device::CUDA),
            .grad_normal = Tensor::zeros(rendered_normal.shape(), Device::CUDA),
            .grad_depth = Tensor::zeros(depth.shape(), Device::CUDA),
            .grad_alpha = Tensor::zeros(alpha.shape(), Device::CUDA)};
        auto partials = Tensor::empty(
            {lfs::training::kernels::normal_consistency_partial_count(depth.numel())},
            Device::CUDA);
        result.loss.set_stream(stream);
        result.grad_normal.set_stream(stream);
        result.grad_depth.set_stream(stream);
        result.grad_alpha.set_stream(stream);
        partials.set_stream(stream);
        if (pixel_weight.is_valid()) {
            pixel_weight.sync_to_stream(stream);
        }

        lfs::training::kernels::launch_normal_consistency_loss(
            rendered_normal.ptr<float>(),
            depth.ptr<float>(),
            alpha.ptr<float>(),
            result.grad_normal.ptr<float>(),
            result.grad_depth.ptr<float>(),
            result.grad_alpha.ptr<float>(),
            result.loss.ptr<float>(),
            partials.ptr<float>(),
            width,
            height,
            20.0f,
            20.0f,
            static_cast<float>(width) * 0.5f,
            static_cast<float>(height) * 0.5f,
            0.3f,
            stream,
            pixel_weight.is_valid() ? pixel_weight.ptr<float>() : nullptr);
        return result;
    }

    ConsistencyResult run_prior_depth_loss(
        const Tensor& prior_normal,
        const Tensor& depth,
        const Tensor& alpha,
        const Tensor& pixel_weight = {}) {
        const int height = static_cast<int>(depth.shape()[0]);
        const int width = static_cast<int>(depth.shape()[1]);
        const cudaStream_t stream = depth.stream();
        ConsistencyResult result{
            .loss = Tensor::empty({size_t{1}}, Device::CUDA),
            .grad_normal = {},
            .grad_depth = Tensor::zeros(depth.shape(), Device::CUDA),
            .grad_alpha = Tensor::zeros(alpha.shape(), Device::CUDA)};
        auto partials = Tensor::empty(
            {lfs::training::kernels::normal_consistency_partial_count(depth.numel())},
            Device::CUDA);
        result.loss.set_stream(stream);
        result.grad_depth.set_stream(stream);
        result.grad_alpha.set_stream(stream);
        partials.set_stream(stream);
        if (pixel_weight.is_valid()) {
            pixel_weight.sync_to_stream(stream);
        }

        lfs::training::kernels::launch_normal_prior_depth_loss(
            prior_normal.ptr<float>(),
            depth.ptr<float>(),
            alpha.ptr<float>(),
            result.grad_depth.ptr<float>(),
            result.grad_alpha.ptr<float>(),
            result.loss.ptr<float>(),
            partials.ptr<float>(),
            width,
            height,
            20.0f,
            20.0f,
            static_cast<float>(width) * 0.5f,
            static_cast<float>(height) * 0.5f,
            0.3f,
            stream,
            pixel_weight.is_valid() ? pixel_weight.ptr<float>() : nullptr);
        return result;
    }

    Tensor make_half_weight(const int height, const int width) {
        std::vector<float> values(static_cast<size_t>(height) * width, 1.0f);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width / 2; ++x) {
                values[static_cast<size_t>(y) * width + x] = 0.0f;
            }
        }
        return Tensor::from_vector(
            values,
            {static_cast<size_t>(height), static_cast<size_t>(width)},
            Device::CUDA);
    }

    void expect_tensors_near(
        const Tensor& actual,
        const Tensor& expected,
        const float tolerance) {
        ASSERT_EQ(actual.shape(), expected.shape());
        EXPECT_LE((actual - expected).abs().max().item<float>(), tolerance);
    }

    class RoiWeightedLossTest : public ::testing::Test {
    protected:
        void SetUp() override {
            int device_count = 0;
            ASSERT_EQ(cudaGetDeviceCount(&device_count), cudaSuccess);
            if (device_count == 0) {
                GTEST_SKIP() << "No CUDA device available";
            }
        }
    };

} // namespace

TEST_F(RoiWeightedLossTest, DepthZeroWeightSuppressesOutsideGradientAndOnesMatchBaseline) {
    constexpr int height = 16;
    constexpr int width = 16;
    std::vector<float> depth_values(height * width);
    std::vector<float> target_values(height * width);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const size_t idx = static_cast<size_t>(y) * width + x;
            depth_values[idx] = 1.5f + 0.02f * x + 0.01f * y;
            target_values[idx] = 0.35f + 0.004f * x - 0.002f * y;
        }
    }
    const auto depth = Tensor::from_vector(
        depth_values, {size_t{height}, size_t{width}}, Device::CUDA);
    const auto alpha = Tensor::ones({size_t{height}, size_t{width}}, Device::CUDA);
    const auto target = Tensor::from_vector(
        target_values, {size_t{height}, size_t{width}}, Device::CUDA);
    const auto ones = Tensor::ones({size_t{height}, size_t{width}}, Device::CUDA);
    const auto half_weight = make_half_weight(height, width);
    lfs::training::kernels::DepthAnchor anchor{
        .valid = true,
        .model = 0,
        .scale = 1.0f,
        .shift = 0.0f,
        .floor = 0.05f};

    const auto baseline = run_depth_loss(depth, alpha, target, anchor);
    const auto all_ones = run_depth_loss(depth, alpha, target, anchor, ones);
    const auto weighted = run_depth_loss(depth, alpha, target, anchor, half_weight);

    EXPECT_GT(baseline.loss.item<float>(), 0.0f);
    expect_tensors_near(all_ones.loss, baseline.loss, 1.0e-6f);
    expect_tensors_near(all_ones.grad_depth, baseline.grad_depth, 1.0e-6f);
    expect_tensors_near(all_ones.grad_alpha, baseline.grad_alpha, 1.0e-6f);
    EXPECT_EQ(
        weighted.grad_depth.slice(1, 0, width / 2).abs().max().item<float>(),
        0.0f);
    EXPECT_EQ(
        weighted.grad_alpha.slice(1, 0, width / 2).abs().max().item<float>(),
        0.0f);
}

TEST_F(RoiWeightedLossTest, NormalZeroWeightSuppressesOutsideGradientAndOnesMatchBaseline) {
    constexpr int height = 16;
    constexpr int width = 16;
    const size_t pixels = static_cast<size_t>(height) * width;
    std::vector<float> rendered_values(3 * pixels, 0.0f);
    std::vector<float> target_values(3 * pixels, 0.0f);
    std::fill(rendered_values.begin(), rendered_values.begin() + pixels, 1.0f);
    std::fill(target_values.begin() + 2 * pixels, target_values.end(), 1.0f);

    const auto rendered = Tensor::from_vector(
        rendered_values, {size_t{3}, size_t{height}, size_t{width}}, Device::CUDA);
    const auto target = Tensor::from_vector(
        target_values, {size_t{3}, size_t{height}, size_t{width}}, Device::CUDA);
    const auto alpha = Tensor::ones({size_t{height}, size_t{width}}, Device::CUDA);
    const auto ones = Tensor::ones({size_t{height}, size_t{width}}, Device::CUDA);
    const auto half_weight = make_half_weight(height, width);

    const auto baseline = run_normal_loss(rendered, alpha, target);
    const auto all_ones = run_normal_loss(rendered, alpha, target, ones);
    const auto weighted = run_normal_loss(rendered, alpha, target, half_weight);

    EXPECT_GT(baseline.loss.item<float>(), 0.0f);
    expect_tensors_near(all_ones.loss, baseline.loss, 1.0e-6f);
    expect_tensors_near(all_ones.grad_normal, baseline.grad_normal, 1.0e-6f);
    EXPECT_EQ(
        weighted.grad_normal.slice(2, 0, width / 2).abs().max().item<float>(),
        0.0f);
    EXPECT_GT(
        weighted.grad_normal.slice(2, width / 2, width).abs().max().item<float>(),
        0.0f);
}

TEST_F(RoiWeightedLossTest, NormalConsistencyUsesWeightedCentersAndOnesMatchBaseline) {
    constexpr int height = 16;
    constexpr int width = 16;
    const size_t pixels = static_cast<size_t>(height) * width;
    std::vector<float> depth_values(pixels);
    std::vector<float> normal_values(3 * pixels, 0.0f);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            depth_values[static_cast<size_t>(y) * width + x] =
                2.0f + 0.005f * x + 0.003f * y;
        }
    }
    std::fill(normal_values.begin(), normal_values.begin() + pixels, 0.2f);
    std::fill(normal_values.begin() + 2 * pixels, normal_values.end(), -0.98f);

    const auto rendered_normal = Tensor::from_vector(
        normal_values, {size_t{3}, size_t{height}, size_t{width}}, Device::CUDA);
    const auto depth = Tensor::from_vector(
        depth_values, {size_t{height}, size_t{width}}, Device::CUDA);
    const auto alpha = Tensor::ones({size_t{height}, size_t{width}}, Device::CUDA);
    const auto ones = Tensor::ones({size_t{height}, size_t{width}}, Device::CUDA);
    const auto half_weight = make_half_weight(height, width);

    const auto baseline = run_consistency_loss(rendered_normal, depth, alpha);
    const auto all_ones = run_consistency_loss(rendered_normal, depth, alpha, ones);
    const auto weighted =
        run_consistency_loss(rendered_normal, depth, alpha, half_weight);

    EXPECT_GT(baseline.loss.item<float>(), 0.0f);
    expect_tensors_near(all_ones.loss, baseline.loss, 1.0e-6f);
    expect_tensors_near(all_ones.grad_normal, baseline.grad_normal, 1.0e-6f);
    expect_tensors_near(all_ones.grad_depth, baseline.grad_depth, 1.0e-6f);
    expect_tensors_near(all_ones.grad_alpha, baseline.grad_alpha, 1.0e-6f);
    EXPECT_EQ(
        weighted.grad_normal.slice(2, 0, width / 2).abs().max().item<float>(),
        0.0f);
    EXPECT_GT(
        weighted.grad_normal.slice(2, width / 2, width).abs().max().item<float>(),
        0.0f);

    std::vector<float> prior_values(3 * pixels, 0.0f);
    std::fill(prior_values.begin(), prior_values.begin() + pixels, 0.3f);
    std::fill(prior_values.begin() + 2 * pixels, prior_values.end(), -0.95f);
    const auto prior_normal = Tensor::from_vector(
        prior_values, {size_t{3}, size_t{height}, size_t{width}}, Device::CUDA);
    const auto prior_baseline = run_prior_depth_loss(prior_normal, depth, alpha);
    const auto prior_all_ones =
        run_prior_depth_loss(prior_normal, depth, alpha, ones);
    EXPECT_GT(prior_baseline.loss.item<float>(), 0.0f);
    expect_tensors_near(prior_all_ones.loss, prior_baseline.loss, 1.0e-6f);
    expect_tensors_near(
        prior_all_ones.grad_depth, prior_baseline.grad_depth, 1.0e-6f);
    expect_tensors_near(
        prior_all_ones.grad_alpha, prior_baseline.grad_alpha, 1.0e-6f);
}

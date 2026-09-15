/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include "training/kernels/depth_loss.hpp"
#include "training/kernels/normal_consistency_loss.hpp"
#include "training/kernels/normal_loss.hpp"
#include "training/losses/mask_loss.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <utility>
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
        Tensor diagnostics;
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
        result.diagnostics = partials;
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

TEST_F(RoiWeightedLossTest, PriorDepthRequiresMinimumCountAndWeight) {
    using namespace lfs::training::kernels;
    namespace slots = normal_consistency_slots;
    constexpr int height = 16;
    constexpr int width = 16;
    constexpr size_t pixels = height * width;
    const auto depth = Tensor::full({size_t{height}, size_t{width}}, 2.0f, Device::CUDA);
    const auto alpha = Tensor::ones({size_t{height}, size_t{width}}, Device::CUDA);

    // Only prior validity limits the count; every selected centre has a valid
    // four-neighbour depth stencil. Include both unweighted and weighted gates.
    for (const int count : {1, 63, 64}) {
        for (const float pixel_weight : {1.0f, 0.25f, 0.24f}) {
            SCOPED_TRACE(::testing::Message() << "count=" << count << " weight=" << pixel_weight);
            std::vector<float> prior_values(3 * pixels, 0.0f);
            int remaining = count;
            for (int y = 1; y < height - 1 && remaining > 0; ++y) {
                for (int x = 1; x < width - 1 && remaining > 0; ++x, --remaining) {
                    const size_t idx = static_cast<size_t>(y) * width + x;
                    prior_values[idx] = 0.6f;
                    prior_values[2 * pixels + idx] = -0.8f;
                }
            }
            const auto prior = Tensor::from_vector(
                prior_values, {size_t{3}, size_t{height}, size_t{width}}, Device::CUDA);
            const auto weight = pixel_weight == 1.0f
                                    ? Tensor{}
                                    : Tensor::full({size_t{height}, size_t{width}}, pixel_weight, Device::CUDA);
            const auto result = run_prior_depth_loss(prior, depth, alpha, weight);
            const bool valid = count >= kNormalConsistencyMinValidCount &&
                               count * pixel_weight >= kNormalConsistencyMinValidWeight;
            const auto slot = [&](int index) {
                return result.diagnostics.slice(0, index, index + 1).item<float>();
            };
            EXPECT_FLOAT_EQ(slot(slots::kCount), static_cast<float>(count));
            EXPECT_FLOAT_EQ(slot(slots::kSumAlpha), count * pixel_weight);
            EXPECT_FLOAT_EQ(slot(slots::kValid), valid ? 1.0f : 0.0f);
            if (valid) {
                EXPECT_GT(slot(slots::kInvNorm), 0.0f);
                EXPECT_GT(result.loss.item<float>(), 0.0f);
                EXPECT_GT(result.grad_depth.abs().max().item<float>(), 0.0f);
                EXPECT_GT(result.grad_alpha.abs().max().item<float>(), 0.0f);
            } else {
                EXPECT_EQ(slot(slots::kInvNorm), 0.0f);
                EXPECT_EQ(result.loss.item<float>(), 0.0f);
                EXPECT_EQ(result.grad_depth.abs().max().item<float>(), 0.0f);
                EXPECT_EQ(result.grad_alpha.abs().max().item<float>(), 0.0f);
            }
        }
    }
}

TEST_F(RoiWeightedLossTest, ComposedUserMaskSuppressesAllNormalLossCenters) {
    constexpr int height = 16;
    constexpr int width = 16;
    constexpr size_t pixels = height * width;
    std::vector<float> rendered_values(3 * pixels, 0.0f);
    std::vector<float> prior_values(3 * pixels, 0.0f);
    std::fill(rendered_values.begin(), rendered_values.begin() + pixels, 0.6f);
    std::fill(rendered_values.begin() + 2 * pixels, rendered_values.end(), -0.8f);
    std::fill(prior_values.begin(), prior_values.begin() + pixels, -0.6f);
    std::fill(prior_values.begin() + 2 * pixels, prior_values.end(), -0.8f);
    const auto rendered = Tensor::from_vector(
        rendered_values, {size_t{3}, size_t{height}, size_t{width}}, Device::CUDA);
    const auto prior = Tensor::from_vector(
        prior_values, {size_t{3}, size_t{height}, size_t{width}}, Device::CUDA);
    const auto depth = Tensor::full({size_t{height}, size_t{width}}, 2.0f, Device::CUDA);
    const auto alpha = Tensor::ones({size_t{height}, size_t{width}}, Device::CUDA);
    // Independent oracle: invalidate normal loss centres instead of masking them.
    const auto half_weight = make_half_weight(height, width);
    const auto active_rendered = rendered * half_weight;
    const auto active_prior = prior * half_weight;

    for (const bool segment_and_ignore : {false, true}) {
        for (const bool with_roi : {false, true}) {
            SCOPED_TRACE(::testing::Message() << "segment_and_ignore=" << segment_and_ignore
                                              << " roi=" << with_roi);
            std::vector<uint8_t> mask_values(pixels, 255);
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width / 2; ++x) {
                    // Segment/Ignore use binary inclusion; SegmentAndIgnore's
                    // nonzero ignore band must become exactly zero.
                    mask_values[static_cast<size_t>(y) * width + x] = segment_and_ignore ? 127 : 0;
                }
            }
            const auto mask = Tensor::from_blob(
                                  mask_values.data(), {size_t{height}, size_t{width}},
                                  Device::CPU, lfs::core::DataType::UInt8)
                                  .to(Device::CUDA);
            ASSERT_EQ(mask.dtype(), lfs::core::DataType::UInt8);
            ASSERT_EQ(mask.device(), Device::CUDA);
            const auto roi = with_roi
                                 ? Tensor::full({size_t{height}, size_t{width}}, 0.5f, Device::CUDA)
                                 : Tensor{};
            lfs::training::losses::MaskPreprocessWorkspace workspace;
            const auto weight = lfs::training::losses::fuse_photometric_mask_weight(
                workspace, mask, roi, segment_and_ignore, true);
            ASSERT_EQ(weight.dtype(), lfs::core::DataType::Float32);
            expect_tensors_near(weight, half_weight * (with_roi ? 0.5f : 1.0f), 0.0f);

            const auto normal = run_normal_loss(rendered, alpha, prior, weight);
            const auto normal_reference = run_normal_loss(rendered, alpha, active_prior, roi);
            const float normal_loss = normal.loss.item<float>();
            const float normal_grad_max = normal.grad_normal.abs().max().item<float>();
            const float ignored_normal_grad_max =
                normal.grad_normal.slice(2, 0, width / 2).abs().max().item<float>();
            EXPECT_GT(normal_loss, 0.0f);
            EXPECT_GT(normal_grad_max, 0.0f);
            EXPECT_EQ(ignored_normal_grad_max, 0.0f);
            expect_tensors_near(normal.loss, normal_reference.loss, 1.0e-6f);
            expect_tensors_near(normal.grad_normal, normal_reference.grad_normal, 1.0e-6f);

            const auto consistency = run_consistency_loss(rendered, depth, alpha, weight);
            const auto consistency_reference = run_consistency_loss(active_rendered, depth, alpha, roi);
            EXPECT_EQ(consistency.grad_normal.slice(2, 0, width / 2).abs().max().item<float>(), 0.0f);
            expect_tensors_near(consistency.grad_normal, consistency_reference.grad_normal, 1.0e-6f);
            const auto prior_depth = run_prior_depth_loss(prior, depth, alpha, weight);
            const auto prior_depth_reference = run_prior_depth_loss(active_prior, depth, alpha, roi);
            for (const auto& pair : {std::pair{consistency, consistency_reference},
                                     std::pair{prior_depth, prior_depth_reference}}) {
                const auto& actual = pair.first;
                const auto& reference = pair.second;
                EXPECT_GT(actual.loss.item<float>(), 0.0f);
                EXPECT_GT(actual.grad_depth.abs().max().item<float>(), 0.0f);
                EXPECT_GT(actual.grad_alpha.abs().max().item<float>(), 0.0f);
                expect_tensors_near(actual.loss, reference.loss, 1.0e-6f);
                expect_tensors_near(actual.grad_depth, reference.grad_depth, 1.0e-6f);
                expect_tensors_near(actual.grad_alpha, reference.grad_alpha, 1.0e-6f);
                // The stencil writes one pixel beyond an active centre. Only
                // that boundary column can receive neighbouring contributions.
                EXPECT_EQ(actual.grad_depth.slice(1, 0, width / 2 - 1).abs().max().item<float>(), 0.0f);
                EXPECT_EQ(actual.grad_alpha.slice(1, 0, width / 2 - 1).abs().max().item<float>(), 0.0f);
            }
        }
    }
}

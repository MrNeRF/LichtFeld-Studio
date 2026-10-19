/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <cmath>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>
#include <vector>

#include "core/cuda_error.hpp"
#include "core/tensor.hpp"
#include "kernels/mcmc_kernels.hpp"
#include "lfs/cuda_scratch.hpp"

using namespace lfs::core;

namespace {

    constexpr size_t N = 1024;
    constexpr size_t N_SAMPLES = 200000;

    struct SampleOutputs {
        std::vector<int64_t> indices;
        std::vector<float> opacities;
        std::vector<float> scales;
    };

    SampleOutputs run_sample_and_gather(const std::vector<float>& weights,
                                        const std::vector<int>& alive,
                                        const std::vector<float>& opacities,
                                        const std::vector<float>& scaling_raw) {
        auto w = Tensor::from_vector(weights, TensorShape({weights.size()}), Device::CUDA);
        auto a = Tensor::from_vector(alive, TensorShape({alive.size()}), Device::CUDA)
                     .to(DataType::Int64);
        auto o = Tensor::from_vector(opacities, TensorShape({opacities.size()}), Device::CUDA);
        auto s = Tensor::from_vector(scaling_raw, TensorShape({scaling_raw.size()}), Device::CUDA);

        auto out_idx = Tensor::zeros({N_SAMPLES}, Device::CUDA, DataType::Int64);
        auto out_opa = Tensor::zeros({N_SAMPLES}, Device::CUDA, DataType::Float32);
        auto out_scale = Tensor::zeros({N_SAMPLES * 3}, Device::CUDA, DataType::Float32);

        lfs::training::mcmc::launch_multinomial_sample_and_gather(
            w.ptr<float>(), o.ptr<float>(), s.ptr<float>(),
            a.ptr<int64_t>(), alive.size(), N_SAMPLES, 1234ull,
            out_idx.ptr<int64_t>(), out_opa.ptr<float>(), out_scale.ptr<float>(),
            weights.size(), nullptr);
        EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

        SampleOutputs result;
        result.indices = out_idx.to_vector_int64();
        result.opacities = out_opa.to_vector();
        result.scales = out_scale.to_vector();
        return result;
    }

} // namespace

class McmcMultinomialTest : public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    }
};

TEST_F(McmcMultinomialTest, ScratchAllocationFailureAbortsAndRecovers) {
    auto weights = Tensor::ones({32}, Device::CUDA);
    auto opacities = Tensor::ones({32}, Device::CUDA);
    auto scales = Tensor::zeros({32, 3}, Device::CUDA);
    auto out_indices = Tensor::zeros({8}, Device::CUDA, DataType::Int64);
    auto out_opacities = Tensor::zeros({8}, Device::CUDA);
    auto out_scales = Tensor::zeros({8, 3}, Device::CUDA);

    EXPECT_THROW(
        (void)lfs::training::cuda_scratch::DeviceBuffer(
            std::numeric_limits<size_t>::max(), nullptr, "test.impossible_allocation"),
        std::runtime_error);

    // The impossible allocation deliberately leaves a recoverable
    // cudaErrorMemoryAllocation in the CUDA status. Per the handled-failure
    // contract (docs/docs/development/assertions.md), a caller that recovers must
    // consume it through the central status adapter, otherwise the next checked
    // call correctly reports it as a pre-existing error.
    ensure_cuda_success(cudaGetLastError(),
                        "recover from injected scratch allocation failure", {},
                        LFS_SOURCE_SITE_CURRENT(),
                        CudaFailureDisposition::LogOnly);

    EXPECT_NO_THROW(lfs::training::mcmc::launch_multinomial_sample_all(
        weights.ptr<float>(), opacities.ptr<float>(), scales.ptr<float>(),
        32, 8, 1234,
        out_indices.ptr<int64_t>(), out_opacities.ptr<float>(), out_scales.ptr<float>(),
        nullptr));
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
}

TEST_F(McmcMultinomialTest, SamplesFollowWeightsAndGatherExactValues) {
    std::vector<float> weights(N, 0.0f);
    std::vector<float> opacities(N);
    std::vector<float> scaling_raw(N * 3);
    std::vector<int> alive;

    // Half the gaussians alive with weights 1..N/2; the rest dead (weight 0).
    for (size_t i = 0; i < N; ++i) {
        opacities[i] = 0.001f * static_cast<float>(i);
        scaling_raw[i * 3 + 0] = -2.0f + 0.001f * static_cast<float>(i);
        scaling_raw[i * 3 + 1] = -1.0f;
        scaling_raw[i * 3 + 2] = 0.5f;
        if (i % 2 == 0) {
            weights[i] = static_cast<float>(i / 2 + 1);
            alive.push_back(static_cast<int>(i));
        }
    }

    const auto out = run_sample_and_gather(weights, alive, opacities, scaling_raw);

    double weight_total = 0.0;
    std::vector<double> expected_prob(N, 0.0);
    for (const int i : alive) {
        weight_total += weights[i];
    }
    for (const int i : alive) {
        expected_prob[i] = weights[i] / weight_total;
    }

    std::vector<size_t> histogram(N, 0);
    for (size_t k = 0; k < N_SAMPLES; ++k) {
        const int64_t idx = out.indices[k];
        ASSERT_GE(idx, 0);
        ASSERT_LT(idx, static_cast<int64_t>(N));
        ASSERT_GT(weights[idx], 0.0f) << "sampled a zero-weight gaussian";
        ++histogram[idx];

        ASSERT_FLOAT_EQ(out.opacities[k], opacities[idx]);
        ASSERT_FLOAT_EQ(out.scales[k * 3 + 0], std::exp(scaling_raw[idx * 3 + 0]));
        ASSERT_FLOAT_EQ(out.scales[k * 3 + 1], std::exp(scaling_raw[idx * 3 + 1]));
        ASSERT_FLOAT_EQ(out.scales[k * 3 + 2], std::exp(scaling_raw[idx * 3 + 2]));
    }

    // Empirical frequency must match the weight distribution (loose bound,
    // ~40x the binomial stddev at these counts for the heaviest bins).
    for (const int i : alive) {
        if (expected_prob[i] < 1e-3) {
            continue;
        }
        const double observed = static_cast<double>(histogram[i]) / N_SAMPLES;
        EXPECT_NEAR(observed, expected_prob[i], expected_prob[i] * 0.25 + 1e-4)
            << "index " << i;
    }
}

TEST_F(McmcMultinomialTest, AllZeroWeightsProduceZeroOutputs) {
    std::vector<float> weights(N, 0.0f);
    std::vector<float> opacities(N, 0.7f);
    std::vector<float> scaling_raw(N * 3, -1.0f);
    std::vector<int> alive;
    for (size_t i = 0; i < N; ++i) {
        alive.push_back(static_cast<int>(i));
    }

    const auto out = run_sample_and_gather(weights, alive, opacities, scaling_raw);

    for (size_t k = 0; k < N_SAMPLES; k += 977) {
        EXPECT_EQ(out.indices[k], 0);
        EXPECT_EQ(out.opacities[k], 0.0f);
        EXPECT_EQ(out.scales[k * 3 + 0], 0.0f);
        EXPECT_EQ(out.scales[k * 3 + 1], 0.0f);
        EXPECT_EQ(out.scales[k * 3 + 2], 0.0f);
    }
}

TEST_F(McmcMultinomialTest, SparseWeightsRemainScaleInvariant) {
    constexpr size_t count = 32768;
    const std::vector<float> opacity_values{0.1f, 0.2f, 0.3f, 0.4f, 0.5f};
    auto opacity = Tensor::from_vector(opacity_values, {5}, Device::CUDA);
    auto scales = Tensor::zeros({5, 3}, Device::CUDA);
    auto alive = Tensor::from_vector(std::vector<int>{4, 1, 2, 3, 0}, {5}, Device::CUDA).to(DataType::Int64);
    for (const bool gather : {false, true}) {
        for (const int exponent : {0, -126}) {
            SCOPED_TRACE(::testing::Message() << "gather=" << gather << " exponent=" << exponent);
            const float unit = std::ldexp(1.0f, exponent);
            auto weights = Tensor::from_vector({0.0f, unit, 0.0f, 3.0f * unit, 0.0f}, {5}, Device::CUDA);
            auto indices = Tensor::empty({count}, Device::CUDA, DataType::Int64);
            auto sampled_opacity = Tensor::empty({count}, Device::CUDA);
            auto sampled_scales = Tensor::empty({count, 3}, Device::CUDA);
            if (gather) {
                lfs::training::mcmc::launch_multinomial_sample_and_gather(
                    weights.ptr<float>(), opacity.ptr<float>(), scales.ptr<float>(), alive.ptr<int64_t>(),
                    5, count, 1234, indices.ptr<int64_t>(), sampled_opacity.ptr<float>(),
                    sampled_scales.ptr<float>(), 5);
            } else {
                lfs::training::mcmc::launch_multinomial_sample_all(
                    weights.ptr<float>(), opacity.ptr<float>(), scales.ptr<float>(), 5, count, 1234,
                    indices.ptr<int64_t>(), sampled_opacity.ptr<float>(), sampled_scales.ptr<float>());
            }
            const auto selected = indices.to_vector_int64();
            const auto gathered = sampled_opacity.to_vector();
            const auto gathered_scales = sampled_scales.to_vector();
            size_t first = 0, zero_weight = 0;
            for (size_t i = 0; i < count; ++i) {
                const auto index = selected[i];
                ASSERT_GE(index, 0);
                ASSERT_LT(index, 5);
                zero_weight += index != 1 && index != 3;
                first += index == 1;
                ASSERT_FLOAT_EQ(gathered[i], opacity_values[index]);
                for (size_t axis = 0; axis < 3; ++axis)
                    ASSERT_FLOAT_EQ(gathered_scales[3 * i + axis], 1.0f);
            }
            EXPECT_EQ(zero_weight, 0);
            EXPECT_NEAR(static_cast<double>(first) / count, 0.25, 0.015);
        }
    }
}

TEST_F(McmcMultinomialTest, ExactCdfStepsSkipZeroWeightPlateaus) {
    auto opacity = Tensor::ones({5}, Device::CUDA);
    auto scales = Tensor::zeros({5, 3}, Device::CUDA);
    auto alive = Tensor::from_vector(std::vector<int>{4, 1, 2, 3, 0}, {5}, Device::CUDA).to(DataType::Int64);
    // Philox(seed, subsequence=0, offset=0) gives exactly 1/4 and 1.
    const std::vector<uint64_t> seeds{29734499, 16390493};
    for (bool gather : {false, true}) {
        for (int exponent : {0, -126}) {
            const float unit = std::ldexp(1.0f, exponent);
            auto weights = Tensor::from_vector({0.0f, unit, 0.0f, 3.0f * unit, 0.0f}, {5}, Device::CUDA);
            for (size_t draw = 0; draw < seeds.size(); ++draw) {
                auto selected = Tensor::empty({1}, Device::CUDA, DataType::Int64);
                auto sampled_opacity = Tensor::empty({1}, Device::CUDA);
                auto sampled_scales = Tensor::empty({1, 3}, Device::CUDA);
                if (gather) {
                    lfs::training::mcmc::launch_multinomial_sample_and_gather(
                        weights.ptr<float>(), opacity.ptr<float>(), scales.ptr<float>(), alive.ptr<int64_t>(),
                        5, 1, seeds[draw], selected.ptr<int64_t>(), sampled_opacity.ptr<float>(),
                        sampled_scales.ptr<float>(), 5);
                } else {
                    lfs::training::mcmc::launch_multinomial_sample_all(
                        weights.ptr<float>(), opacity.ptr<float>(), scales.ptr<float>(), 5, 1, seeds[draw],
                        selected.ptr<int64_t>(), sampled_opacity.ptr<float>(), sampled_scales.ptr<float>());
                }
                EXPECT_EQ(selected.to_vector_int64(), (std::vector<int64_t>{draw == 0 ? 1 : 3}));
            }
        }
    }
}

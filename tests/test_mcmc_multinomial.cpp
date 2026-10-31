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
    constexpr float kAsGiven = -std::numeric_limits<float>::infinity();

    std::vector<int64_t> run_sample(const std::vector<float>& weights, const std::vector<int>& alive) {
        auto w = Tensor::from_vector(weights, TensorShape({weights.size()}), Device::CUDA);
        std::vector<float> dead(weights.size(), 1.0f);
        for (const int i : alive) {
            dead[i] = 0.0f;
        }
        const auto excluded = Tensor::from_vector(dead, TensorShape({dead.size()}), Device::CUDA) > 0.5f;
        auto out_idx = Tensor::zeros({N_SAMPLES}, Device::CUDA, DataType::Int64);
        EXPECT_TRUE(lfs::training::mcmc::launch_multinomial_sample_all(
            w.ptr<float>(), kAsGiven, excluded.ptr<uint8_t>(), weights.size(), N_SAMPLES, 1234ull,
            out_idx.ptr<int64_t>()));
        EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        return out_idx.to_vector_int64();
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
    auto out_indices = Tensor::zeros({8}, Device::CUDA, DataType::Int64);

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

    EXPECT_NO_THROW((void)lfs::training::mcmc::launch_multinomial_sample_all(
        weights.ptr<float>(), kAsGiven, nullptr, 32, 8, 1234, out_indices.ptr<int64_t>(), nullptr));
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
}

TEST_F(McmcMultinomialTest, SamplesFollowWeightsAndSkipExcludedRows) {
    std::vector<float> weights(N, 0.0f);
    std::vector<int> alive;

    // Half the gaussians alive with weights 1..N/2; the rest dead. Dead rows
    // carry the heaviest weights, so only the exclusion keeps them out.
    for (size_t i = 0; i < N; ++i) {
        if (i % 2 == 0) {
            weights[i] = static_cast<float>(i / 2 + 1);
            alive.push_back(static_cast<int>(i));
        } else {
            weights[i] = 1000.0f;
        }
    }

    const auto out = run_sample(weights, alive);

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
        const int64_t idx = out[k];
        ASSERT_GE(idx, 0);
        ASSERT_LT(idx, static_cast<int64_t>(N));
        ASSERT_EQ(idx % 2, 0) << "sampled an excluded gaussian";
        ++histogram[idx];
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

TEST_F(McmcMultinomialTest, NothingSampleableSamplesNothing) {
    const auto zeros = Tensor::zeros({N}, Device::CUDA);
    const auto ones = Tensor::ones({N}, Device::CUDA);
    const auto all_excluded = Tensor::ones_bool({N}, Device::CUDA);
    auto out = Tensor::full({8}, -7.0f, Device::CUDA).to(DataType::Int64);
    EXPECT_FALSE(lfs::training::mcmc::launch_multinomial_sample_all(
        zeros.ptr<float>(), kAsGiven, nullptr, N, 8, 1234, out.ptr<int64_t>()));
    EXPECT_FALSE(lfs::training::mcmc::launch_multinomial_sample_all(
        ones.ptr<float>(), 1e-12f, all_excluded.ptr<uint8_t>(), N, 8, 1234, out.ptr<int64_t>()));
    EXPECT_EQ(out.to_vector_int64(), std::vector<int64_t>(8, -7));
    // A floor makes zero weights sampleable, as Tensor::clamp_min would.
    EXPECT_TRUE(lfs::training::mcmc::launch_multinomial_sample_all(
        zeros.ptr<float>(), 1e-12f, nullptr, N, 8, 1234, out.ptr<int64_t>()));
}

// Flooring the weights as the sampler reads them must draw exactly what the
// clamp_min copy it replaces draws. CUB's float scan adds tiles in a timing
// dependent order, so the weights are multiples of 1/4 with a total below
// 2^22: every partial sum is exact and the CDF is the same in every run.
TEST_F(McmcMultinomialTest, FloorAsReadMatchesClampedCopy) {
    constexpr size_t rows = size_t{1} << 21;
    constexpr size_t draws = size_t{1} << 20;
    constexpr float floor = 0.25f;
    std::vector<float> scores(rows);
    std::vector<float> dead(rows, 0.0f);
    uint32_t state = 777u;
    for (size_t i = 0; i < rows; ++i) {
        state = state * 1664525u + 1013904223u;
        const uint32_t r = state >> 8;
        // 0 and 0.125 fall below the floor.
        scores[i] = static_cast<float>(r % 9) * 0.125f;
        dead[i] = (r % 6 == 1) ? 1.0f : 0.0f;
    }
    const auto raw = Tensor::from_vector(scores, TensorShape({rows}), Device::CUDA);
    const auto clamped = raw.clamp_min(floor);
    const auto excluded = Tensor::from_vector(dead, TensorShape({rows}), Device::CUDA) > 0.5f;
    for (const bool masked : {false, true}) {
        SCOPED_TRACE(::testing::Message() << "masked=" << masked);
        const uint8_t* mask = masked ? excluded.ptr<uint8_t>() : nullptr;
        auto from_copy = Tensor::empty({draws}, Device::CUDA, DataType::Int64);
        auto as_read = Tensor::empty({draws}, Device::CUDA, DataType::Int64);
        ASSERT_TRUE(lfs::training::mcmc::launch_multinomial_sample_all(
            clamped.ptr<float>(), kAsGiven, mask, rows, draws, 4242, from_copy.ptr<int64_t>()));
        ASSERT_TRUE(lfs::training::mcmc::launch_multinomial_sample_all(
            raw.ptr<float>(), floor, mask, rows, draws, 4242, as_read.ptr<int64_t>()));
        EXPECT_EQ(from_copy.to_vector_int64(), as_read.to_vector_int64());
    }
}

TEST_F(McmcMultinomialTest, SparseWeightsRemainScaleInvariant) {
    constexpr size_t count = 32768;
    const auto none_excluded = Tensor::zeros_bool({5}, Device::CUDA);
    for (const bool masked : {false, true}) {
        for (const int exponent : {0, -126}) {
            SCOPED_TRACE(::testing::Message() << "masked=" << masked << " exponent=" << exponent);
            const float unit = std::ldexp(1.0f, exponent);
            auto weights = Tensor::from_vector({0.0f, unit, 0.0f, 3.0f * unit, 0.0f}, {5}, Device::CUDA);
            auto indices = Tensor::empty({count}, Device::CUDA, DataType::Int64);
            ASSERT_TRUE(lfs::training::mcmc::launch_multinomial_sample_all(
                weights.ptr<float>(), kAsGiven, masked ? none_excluded.ptr<uint8_t>() : nullptr, 5, count, 1234,
                indices.ptr<int64_t>()));
            const auto selected = indices.to_vector_int64();
            size_t first = 0, zero_weight = 0;
            for (size_t i = 0; i < count; ++i) {
                const auto index = selected[i];
                ASSERT_GE(index, 0);
                ASSERT_LT(index, 5);
                zero_weight += index != 1 && index != 3;
                first += index == 1;
            }
            EXPECT_EQ(zero_weight, 0);
            EXPECT_NEAR(static_cast<double>(first) / count, 0.25, 0.015);
        }
    }
}

TEST_F(McmcMultinomialTest, ExactCdfStepsSkipZeroWeightPlateaus) {
    const auto none_excluded = Tensor::zeros_bool({5}, Device::CUDA);
    // Philox(seed, subsequence=0, offset=0) gives exactly 1/4 and 1.
    const std::vector<uint64_t> seeds{29734499, 16390493};
    for (bool masked : {false, true}) {
        for (int exponent : {0, -126}) {
            const float unit = std::ldexp(1.0f, exponent);
            auto weights = Tensor::from_vector({0.0f, unit, 0.0f, 3.0f * unit, 0.0f}, {5}, Device::CUDA);
            for (size_t draw = 0; draw < seeds.size(); ++draw) {
                auto selected = Tensor::empty({1}, Device::CUDA, DataType::Int64);
                ASSERT_TRUE(lfs::training::mcmc::launch_multinomial_sample_all(
                    weights.ptr<float>(), kAsGiven, masked ? none_excluded.ptr<uint8_t>() : nullptr, 5, 1,
                    seeds[draw], selected.ptr<int64_t>()));
                EXPECT_EQ(selected.to_vector_int64(), (std::vector<int64_t>{draw == 0 ? 1 : 3}));
            }
        }
    }
}

// Millions of rows put zero-weight and excluded rows on scan tile boundaries,
// where the float CDF can step up by an ulp. None of them may be drawn.
TEST_F(McmcMultinomialTest, LargeScanNeverDrawsExcludedOrZeroWeightRows) {
    constexpr size_t rows = size_t{1} << 22;
    constexpr size_t draws = size_t{1} << 21;
    std::vector<float> weights(rows);
    std::vector<float> dead(rows, 0.0f);
    uint32_t state = 12345u;
    for (size_t i = 0; i < rows; ++i) {
        state = state * 1664525u + 1013904223u;
        const uint32_t r = state >> 8;
        weights[i] = (r % 7 == 0) ? 0.0f : 1e-3f + static_cast<float>(r % 1000) * 1e-2f;
        dead[i] = (r % 4 == 1) ? 1.0f : 0.0f;
    }
    const auto w = Tensor::from_vector(weights, TensorShape({rows}), Device::CUDA);
    const auto excluded = Tensor::from_vector(dead, TensorShape({rows}), Device::CUDA) > 0.5f;
    auto indices = Tensor::empty({draws}, Device::CUDA, DataType::Int64);

    ASSERT_TRUE(lfs::training::mcmc::launch_multinomial_sample_all(
        w.ptr<float>(), kAsGiven, excluded.ptr<uint8_t>(), rows, draws, 99, indices.ptr<int64_t>()));

    size_t bad = 0;
    for (const int64_t index : indices.to_vector_int64()) {
        bad += !(weights[index] > 0.0f) || dead[index] != 0.0f;
    }
    EXPECT_EQ(bad, 0u);
}

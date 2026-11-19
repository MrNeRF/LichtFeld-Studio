/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Counts and ranks that must stay exact past 2^24, where a float stops holding every integer, on the
// session's tensor backend in builds without CUDA. test_mrnf_strategy.cpp and
// test_tensor_kernels_6c_elem.cpp hold the CUDA versions; the names are the same, so the strategies'
// friend declarations apply to the free-slot count here too.

#include "core/sh_layout.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "core/tensor_backend.hpp"
#include "lfs/training/ops/registry.hpp"
#include "training/strategies/improved_gs_plus.hpp"
#include "training/strategies/mrnf.hpp"

#include <gtest/gtest.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

using namespace lfs::core;
using namespace lfs::training;

class MRNFStrategyTest : public ::testing::Test {};

namespace {
    SplatData create_mrnf_test_splat_data(const int n_gaussians = 10, const int sh_degree = 3) {
        const size_t n = static_cast<size_t>(n_gaussians);
        std::vector<float> means_data(n_gaussians * 3, 0.0f);
        for (int i = 0; i < n_gaussians; ++i) {
            means_data[i * 3 + 0] = static_cast<float>(i);
        }

        std::vector<float> sh0_data(n_gaussians * 3, 0.5f);
        std::vector<float> scaling_data(n_gaussians * 3, 0.0f);
        std::vector<float> rotation_data(n_gaussians * 4, 0.0f);
        std::vector<float> opacity_data(n_gaussians, 0.0f);
        const size_t sh_rest = sh_rest_coefficients_for_degree(sh_degree);

        for (int i = 0; i < n_gaussians; ++i) {
            rotation_data[i * 4 + 0] = 1.0f; // identity quaternion
        }

        auto means = Tensor::from_vector(means_data, TensorShape({n, 3}), Device::GPU);
        auto sh0 = Tensor::from_vector(sh0_data, TensorShape({n, 1, 3}), Device::GPU);
        auto shN = Tensor::zeros(TensorShape({n, sh_rest, 3}), Device::GPU);
        auto scaling = Tensor::from_vector(scaling_data, TensorShape({n, 3}), Device::GPU);
        auto rotation = Tensor::from_vector(rotation_data, TensorShape({n, 4}), Device::GPU);
        auto opacity = Tensor::from_vector(opacity_data, TensorShape({n, 1}), Device::GPU);

        return SplatData(sh_degree, means, sh0, shN, scaling, rotation, opacity, 1.0f);
    }
} // namespace

TEST_F(MRNFStrategyTest, FreeSlotCountIsExactPastFloat32Precision) {
    constexpr size_t b = size_t{1} << 24;
    const auto all_free = Tensor::ones_bool({b + 3}, Device::GPU);
    auto splats = create_mrnf_test_splat_data(1);
    const auto one_mean = splats.means();
    MRNF mrnf(splats);
    ImprovedGSPlus igs(splats);
    mrnf._free_mask = all_free;
    igs._free_mask = all_free;
    for (const size_t rows : {b + 1, b + 3}) {
        SCOPED_TRACE(rows);
        // A broadcast view supplies the logical row count without allocating a large model.
        splats.means() = one_mean.expand(TensorShape({rows, 3}));
        EXPECT_EQ(mrnf.free_count(), rows);
        EXPECT_EQ(mrnf.active_count(), 0u);
        EXPECT_EQ(igs.free_count(), rows);
        EXPECT_EQ(igs.active_count(), 0u);
    }
}

// Built through a float32 arange, the four rows appended at B = 2^24 came out as B, B, B+2, B+4 and one row as none,
// so ray_depth_inherit wrote an existing splat's state and read child rows past the end of the list.
TEST_F(MRNFStrategyTest, SplitChildRowsAreExactPastFloat32Precision) {
    constexpr int64_t b = int64_t{1} << 24;
    const auto reused = Tensor::from_vector(std::vector<int>{7}, TensorShape({1}), Device::GPU).to(DataType::Int64);
    const auto expect_rows = [](const Tensor& rows, const std::vector<int64_t>& expected) {
        ASSERT_EQ(rows.dtype(), DataType::Int64);
        ASSERT_EQ(rows.device(), Device::GPU);
        ASSERT_EQ(rows.shape(), TensorShape({expected.size()}));
        EXPECT_EQ(rows.cpu().to_vector_int64(), expected);
    };
    expect_rows(split_child_rows(reused, static_cast<size_t>(b), 4), {7, b, b + 1, b + 2, b + 3});
    expect_rows(split_child_rows(Tensor(), static_cast<size_t>(b), 1), {b});
    expect_rows(split_child_rows(reused, static_cast<size_t>(b), 0), {7});
    expect_rows(split_child_rows(Tensor(), static_cast<size_t>(b), 0), {});
}

// CUDA checks these ranks in mrnf_strategy::percentile_bounds_ranks. The other backends compute them inside
// their percentile_bounds, so the same cases go through it here: every sorted value is the float whose bit
// pattern is 1.0f's plus its rank, and the bounds must be made of the values at CUDA's ranks.
TEST_F(MRNFStrategyTest, PercentileRankStaysInsideTheSortedValues) {
    const auto& table = training_ops(default_gpu_backend());
    ASSERT_NE(table.mrnf, nullptr);
    using Ranks = std::pair<size_t, size_t>;
    const auto value = [](const size_t rank) {
        return std::bit_cast<float>(std::bit_cast<uint32_t>(1.0f) + static_cast<uint32_t>(rank));
    };
    const auto rank_of = [](const float v) {
        return static_cast<int64_t>(std::bit_cast<uint32_t>(v)) - static_cast<int64_t>(std::bit_cast<uint32_t>(1.0f));
    };
    const auto expect_ranks = [&](const size_t n, const float percentile, const Ranks ranks) {
        SCOPED_TRACE(::testing::Message() << "n " << n << ", percentile " << percentile);
        std::vector<float> means(3 * n);
        for (size_t i = 0; i < n; ++i)
            means[3 * i] = means[3 * i + 1] = means[3 * i + 2] = value(n - 1 - i);
        const auto bounds = table.mrnf->percentile_bounds(Tensor::from_vector(means, TensorShape({n, 3}), Device::GPU), percentile);
        const float low = value(ranks.first), high = value(ranks.second);
        for (int axis = 0; axis < 3; ++axis) {
            EXPECT_EQ(bounds.center[axis], (low + high) * 0.5f) << "axis " << axis << ": the ranks used are about "
                                                                << rank_of(bounds.center[axis] - bounds.extent[axis]) << " and "
                                                                << rank_of(bounds.center[axis] + bounds.extent[axis]);
            EXPECT_EQ(bounds.extent[axis], (high - low) * 0.5f) << "axis " << axis;
        }
    };
    constexpr size_t past_float32 = (size_t{1} << 24) + 4;
    expect_ranks(past_float32, 1.0f, {0, past_float32 - 1});
    expect_ranks(20'000'000, 1.0f, {0, 19'999'999});
    expect_ranks(20'000'000, std::nextafter(1.0f, 0.0f), {0, 19'999'998});
    expect_ranks(20'000'000, 0.0f, {9'999'999, 9'999'999});
    expect_ranks(1, 1.0f, {0, 0});
    expect_ranks(11, 0.5f, {2, 7});
}

// Catches a float count, exact only up to 2^24: callers size buffers from it, and the MRNF crop box
// split compares it with an exact compaction count (#3806).
TEST(TensorElementwiseKernels, CountNonzeroIsExactAbove2To24) {
    for (const auto& [n, seed] : {std::pair<size_t, uint32_t>{20'000'003, 1u},
                                  std::pair<size_t, uint32_t>{33'554'467, 2u},
                                  std::pair<size_t, uint32_t>{40'000'001, 3u}}) {
        SCOPED_TRACE(n);
        std::vector<float> vals(n, 0.f);
        size_t expected = 0;
        uint32_t state = seed;
        for (size_t i = 0; i < n; ++i) {
            state = state * 1664525u + 1013904223u;
            if ((state >> 29) != 0) { // about 7 in 8 nonzero
                vals[i] = 1.f;
                ++expected;
            }
        }
        ASSERT_GT(expected, size_t{1} << 24);
        auto t = Tensor::from_vector(vals, TensorShape({n}), Device::GPU);
        EXPECT_EQ(t.count_nonzero(), expected);
        EXPECT_EQ((t > 0.0f).count_nonzero(), expected);
    }
}

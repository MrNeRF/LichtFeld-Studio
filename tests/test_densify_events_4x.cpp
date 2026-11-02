/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Densify-event kernels: free-slot fuse, packed counts,
 * positive-median normalize, workspace growth helpers.
 */

#include "core/tensor.hpp"
#include "training/kernels/densification_kernels.hpp"
#include "training/kernels/mrnf_kernels.hpp"
#include "training/strategies/strategy_utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include <functional>
#include <gtest/gtest.h>
#include <limits>
#include <vector>

using namespace lfs::core;
using namespace lfs::training;

namespace {

    std::vector<float> to_host(const Tensor& t) {
        return t.cpu().to_vector();
    }

} // namespace

TEST(DensifyEvents4x, FillFreeSlotsFusedWritesAttrsAndZerosAdam) {
    constexpr size_t N = 8;
    constexpr size_t K = 3;

    auto means = Tensor::zeros({N, 3}, Device::CUDA);
    auto rots = Tensor::zeros({N, 4}, Device::CUDA);
    auto scales = Tensor::zeros({N, 3}, Device::CUDA);
    auto sh0 = Tensor::zeros({N, 1, 3}, Device::CUDA);
    auto opac = Tensor::zeros({N}, Device::CUDA);
    auto free_mask = Tensor::zeros_bool({N}, Device::CUDA);
    // Mark slots 1, 3, 5 free.
    {
        auto host = free_mask.cpu();
        auto* p = host.ptr<bool>();
        p[1] = p[3] = p[5] = true;
        free_mask = host.cuda();
    }

    auto src_means = Tensor::from_vector(
        std::vector<float>{10, 11, 12, 20, 21, 22, 30, 31, 32},
        TensorShape({K, 3}), Device::CUDA);
    auto src_rots = Tensor::from_vector(
        std::vector<float>{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0},
        TensorShape({K, 4}), Device::CUDA);
    auto src_scales = Tensor::from_vector(
        std::vector<float>{0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f},
        TensorShape({K, 3}), Device::CUDA);
    auto src_sh0 = Tensor::from_vector(
        std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9},
        TensorShape({K, 1, 3}), Device::CUDA);
    auto src_opac = Tensor::from_vector(
        std::vector<float>{0.5f, 0.6f, 0.7f}, TensorShape({K}), Device::CUDA);

    auto targets = Tensor::zeros({K}, Device::CUDA, DataType::Int64);
    {
        const int64_t host_t[3] = {1, 3, 5};
        ASSERT_EQ(cudaMemcpy(targets.ptr<int64_t>(), host_t, K * sizeof(int64_t),
                             cudaMemcpyHostToDevice),
                  cudaSuccess);
    }

    auto adam0 = Tensor::ones({N}, Device::CUDA);
    auto adam1 = Tensor::ones({N}, Device::CUDA);
    float* adam_ptrs[2] = {adam0.ptr<float>(), adam1.ptr<float>()};

    kernels::launch_fill_free_slots_fused(
        targets.ptr<int64_t>(), K,
        src_means.ptr<float>(), src_rots.ptr<float>(), src_scales.ptr<float>(),
        src_sh0.ptr<float>(), src_opac.ptr<float>(),
        means.ptr<float>(), rots.ptr<float>(), scales.ptr<float>(),
        sh0.ptr<float>(), opac.ptr<float>(),
        /*opacity_dim=*/0, adam_ptrs, 2, free_mask.ptr<bool>(), N);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    auto mh = to_host(means);
    EXPECT_FLOAT_EQ(mh[1 * 3 + 0], 10.f);
    EXPECT_FLOAT_EQ(mh[3 * 3 + 1], 21.f);
    EXPECT_FLOAT_EQ(mh[5 * 3 + 2], 32.f);
    EXPECT_FLOAT_EQ(mh[0], 0.f); // untouched

    auto ah0 = to_host(adam0);
    auto ah1 = to_host(adam1);
    EXPECT_FLOAT_EQ(ah0[1], 0.f);
    EXPECT_FLOAT_EQ(ah0[3], 0.f);
    EXPECT_FLOAT_EQ(ah0[0], 1.f);
    EXPECT_FLOAT_EQ(ah1[5], 0.f);

    auto fm = free_mask.cpu();
    EXPECT_FALSE(fm.ptr<bool>()[1]);
    EXPECT_FALSE(fm.ptr<bool>()[3]);
    EXPECT_FALSE(fm.ptr<bool>()[5]);
    EXPECT_FALSE(fm.ptr<bool>()[0]);
}

TEST(DensifyEvents4x, PackedRefineCountsMatchHost) {
    constexpr size_t N = 16;
    std::vector<uint8_t> b0(N, 0), b1(N, 0);
    std::vector<float> f0(N, 0.f), f1(N, 0.f);
    b0[0] = b0[2] = b0[4] = 1; // 3
    b1[1] = b1[3] = 1;         // 2
    f0[0] = 1.f;
    f0[1] = -1.f;
    f0[5] = 0.5f; // 2 positives
    f1[7] = 3.f;  // 1 positive

    auto bool0 = Tensor::zeros_bool({N}, Device::CUDA);
    auto bool1 = Tensor::zeros_bool({N}, Device::CUDA);
    {
        auto h0 = bool0.cpu();
        auto h1 = bool1.cpu();
        for (size_t i = 0; i < N; ++i) {
            h0.ptr<bool>()[i] = b0[i] != 0;
            h1.ptr<bool>()[i] = b1[i] != 0;
        }
        bool0 = h0.cuda();
        bool1 = h1.cuda();
    }
    auto tf0 = Tensor::from_vector(f0, TensorShape({N}), Device::CUDA);
    auto tf1 = Tensor::from_vector(f1, TensorShape({N}), Device::CUDA);
    auto out = Tensor::zeros({4}, Device::CUDA, DataType::Int64);

    kernels::launch_packed_refine_counts(
        bool0.ptr<bool>(), N, bool1.ptr<bool>(), N,
        tf0.ptr<float>(), N, tf1.ptr<float>(), N,
        out.ptr<int64_t>());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    int64_t counts[4] = {};
    ASSERT_EQ(cudaMemcpy(counts, out.ptr<int64_t>(), 4 * sizeof(int64_t),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    EXPECT_EQ(counts[0], 3);
    EXPECT_EQ(counts[1], 2);
    EXPECT_EQ(counts[2], 2);
    EXPECT_EQ(counts[3], 1);
}

TEST(DensifyEvents4x, PositiveMedianNormalizeMatchesSortReference) {
    // Values: positives 1,2,3,4,5 → median 3; zeros stay 0; result /3
    std::vector<float> data = {0.f, 1.f, 0.f, 5.f, 2.f, 4.f, 3.f, 0.f, -1.f};
    auto t = Tensor::from_vector(data, TensorShape({data.size()}), Device::CUDA);

    // Reference: positives only sorted mid
    std::vector<float> pos;
    for (float v : data)
        if (v > 0.f)
            pos.push_back(v);
    std::sort(pos.begin(), pos.end());
    const float median = pos[pos.size() / 2];
    std::vector<float> ref = data;
    for (float& v : ref)
        v /= std::max(median, 1e-9f);

    kernels::launch_normalize_by_positive_median(t.ptr<float>(), t.numel());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    auto got = to_host(t);
    ASSERT_EQ(got.size(), ref.size());
    for (size_t i = 0; i < ref.size(); ++i) {
        EXPECT_NEAR(got[i], ref[i], 1e-5f) << "i=" << i;
    }
}

namespace {
    // Host reference in cub::DeviceRadixSort float order (NaN and -0.0 included).
    uint32_t radix_order_key(const float v) {
        uint32_t bits = 0;
        std::memcpy(&bits, &v, sizeof(bits));
        return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
    }

    std::vector<float> reference_positive_median_normalize(std::vector<float> values) {
        std::vector<float> positives;
        for (auto& v : values) {
            if (std::isnan(v))
                v = 0.0f;
            if (v > 0.0f)
                positives.push_back(v);
        }
        if (positives.empty())
            return std::vector<float>(values.size(), 0.0f);
        std::ranges::sort(positives, {}, radix_order_key);
        const float median = std::max(positives[positives.size() / 2], 1e-9f);
        for (auto& v : values)
            v /= median;
        return values;
    }

    // Value sets that stress the three radix digits: runs of equal keys that
    // straddle digit boundaries, adjacent ulps, extremes and non-finite input.
    std::vector<std::vector<float>> median_cases() {
        std::vector<std::vector<float>> cases;
        cases.push_back({0.f, 1.f, 0.f, 5.f, 2.f, 4.f, 3.f, 0.f, -1.f});
        cases.push_back({3.f});
        cases.push_back({7.f, 7.f, 7.f, 7.f});
        cases.push_back({1e-30f, 1e30f, std::numeric_limits<float>::infinity(), -0.0f, 0.0f,
                         std::numeric_limits<float>::quiet_NaN(), 2.5f, 2.5f,
                         std::nextafter(2.5f, 3.0f), std::numeric_limits<float>::denorm_min()});
        std::vector<float> ramp;
        for (int i = 0; i < 100003; ++i)
            ramp.push_back(static_cast<float>((i * 7919) % 100003) * 1.25e-3f - 3.0f);
        cases.push_back(std::move(ramp));
        std::vector<float> ulps;
        for (int i = 0; i < 40000; ++i)
            ulps.push_back(std::nextafter(1.0f + static_cast<float>(i % 17) * 1e-7f, 2.0f));
        cases.push_back(std::move(ulps));
        return cases;
    }
} // namespace

TEST(DensifyEvents4x, PositiveMedianNormalizeMatchesSortedReference) {
    for (const auto& values : median_cases()) {
        auto t = Tensor::from_vector(values, TensorShape({values.size()}), Device::CUDA);
        kernels::launch_normalize_by_positive_median(t.ptr<float>(), t.numel());
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        const auto got = to_host(t);
        const auto expected = reference_positive_median_normalize(values);
        ASSERT_EQ(got.size(), expected.size());
        for (size_t i = 0; i < got.size(); ++i) {
            // The median is exact (see SelectMedianMatchesRadixSortOrder); the GPU
            // division itself may round one ulp differently from the host.
            const auto a = radix_order_key(got[i]);
            const auto b = radix_order_key(expected[i]);
            ASSERT_LE(a > b ? a - b : b - a, 2u)
                << "n=" << values.size() << " i=" << i << " got=" << got[i] << " expected=" << expected[i];
        }
    }
}

TEST(DensifyEvents4x, SelectMedianMatchesRadixSortOrder) {
    for (const auto& values : median_cases()) {
        auto t = Tensor::from_vector(values, TensorShape({values.size()}), Device::CUDA);
        auto sorted = values;
        std::ranges::sort(sorted, {}, radix_order_key);
        const float got = kernels::launch_select_median(t.ptr<float>(), t.numel(), nullptr);
        EXPECT_EQ(radix_order_key(got), radix_order_key(sorted[sorted.size() / 2])) << "n=" << values.size();
    }
}

TEST(DensifyEvents4x, MedianSelectionMatchesReferenceAcrossDigitBoundaries) {
    // Place the selected bin on either side of warp, block and radix boundaries.
    for (const uint32_t shift : {0u, 10u, 21u}) {
        for (const uint32_t digit : {0u, 1u, 31u, 32u, 255u, 256u, 1023u, 1024u, 2047u}) {
            const uint32_t key = digit << shift;
            std::vector<float> values;
            for (int delta = -19; delta <= 19; ++delta) {
                const uint32_t ordered = key + static_cast<uint32_t>(delta);
                const uint32_t bits = (ordered & 0x80000000u) ? (ordered & 0x7fffffffu) : ~ordered;
                float value;
                std::memcpy(&value, &bits, sizeof(value));
                values.insert(values.end(), 17, value);
            }
            auto tensor = Tensor::from_vector(values, TensorShape({values.size()}), Device::CUDA);
            std::ranges::sort(values, {}, radix_order_key);
            EXPECT_EQ(radix_order_key(kernels::launch_select_median(tensor.ptr<float>(), tensor.numel(), nullptr)),
                      radix_order_key(values[values.size() / 2]))
                << "shift=" << shift << " digit=" << digit;
        }
    }
}

TEST(DensifyEvents4x, PositiveMedianNormalizeZerosWhenNoPositives) {
    std::vector<float> data = {0.f, -1.f, 0.f, -2.f};
    auto t = Tensor::from_vector(data, TensorShape({data.size()}), Device::CUDA);
    kernels::launch_normalize_by_positive_median(t.ptr<float>(), t.numel());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    for (float v : to_host(t)) {
        EXPECT_FLOAT_EQ(v, 0.f);
    }
}

TEST(DensifyEvents4x, GumbelIsDeterministicPerSeed) {
    constexpr size_t n = 4096;
    constexpr size_t k = 300;
    std::vector<float> weights(n);
    for (size_t i = 0; i < n; ++i)
        weights[i] = i % 3 == 0 ? 0.0f : 0.5f + static_cast<float>(i % 17);
    const auto w = Tensor::from_vector(weights, TensorShape({n}), Device::CUDA);
    auto first = Tensor::empty({k}, Device::CUDA, DataType::Int64);
    auto second = Tensor::empty({k}, Device::CUDA, DataType::Int64);
    mrnf_strategy::launch_gumbel_topk(w.ptr<float>(), n, k, 0xC0FFEEULL, first.ptr<int64_t>());
    mrnf_strategy::launch_gumbel_topk(w.ptr<float>(), n, k, 0xC0FFEEULL, second.ptr<int64_t>());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    EXPECT_EQ(first.cpu().to_vector_int64(), second.cpu().to_vector_int64());
}

// Output is in descending key order: rows whose weight no Gumbel noise can
// overcome come first, in their weight order.
TEST(DensifyEvents4x, GumbelOrdersPicksByKey) {
    constexpr size_t n = 1000;
    std::vector<float> weights(n, 1.0f);
    weights[700] = 1.0e30f;
    weights[3] = 1.0e20f;
    weights[512] = 1.0e10f;
    const auto w = Tensor::from_vector(weights, TensorShape({n}), Device::CUDA);
    auto picks = Tensor::empty({5}, Device::CUDA, DataType::Int64);
    mrnf_strategy::launch_gumbel_topk(w.ptr<float>(), n, 5, 7, picks.ptr<int64_t>());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    const auto values = picks.cpu().to_vector_int64();
    EXPECT_EQ(values[0], 700);
    EXPECT_EQ(values[1], 3);
    EXPECT_EQ(values[2], 512);
}

TEST(DensifyEvents4x, GumbelUInt32PayloadsAreBoundedAndUnique) {
    constexpr size_t n = 257;
    constexpr size_t k = 31;
    std::vector<float> weights(n, 1.0f);
    weights.back() = 1.0e30f;
    auto w = Tensor::from_vector(weights, TensorShape({n}), Device::CUDA);
    auto selected = Tensor::empty({k}, Device::CUDA, DataType::Int64);

    mrnf_strategy::launch_gumbel_topk(w.ptr<float>(), n, k, 0x12345678ULL, selected.ptr<int64_t>());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    const auto values = selected.cpu().to_vector_int64();
    ASSERT_EQ(values.size(), k);
    EXPECT_NE(std::find(values.begin(), values.end(), static_cast<int64_t>(n - 1)), values.end());
    for (size_t i = 0; i < values.size(); ++i) {
        ASSERT_GE(values[i], 0);
        ASSERT_LT(values[i], static_cast<int64_t>(n));
        for (size_t j = 0; j < i; ++j) {
            EXPECT_NE(values[i], values[j]) << "duplicate at " << i;
        }
    }
}

// Catches a NaN weight getting a NaN key, which a descending order puts before every real candidate.
TEST(DensifyEvents4x, GumbelNeverPicksNanOrNonPositiveWeights) {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const std::vector<float> weights{nan, 1.0f, 0.0f, -2.0f, 3.0f, nan, 4.0f, 0.5f};
    const std::vector<int64_t> positive{1, 4, 6, 7};
    const auto w = Tensor::from_vector(weights, TensorShape({weights.size()}), Device::CUDA);
    auto picks = Tensor::empty({positive.size()}, Device::CUDA, DataType::Int64);
    mrnf_strategy::launch_gumbel_topk(w.ptr<float>(), weights.size(), positive.size(), 0x5eedULL,
                                      picks.ptr<int64_t>());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    auto values = picks.cpu().to_vector_int64();
    std::sort(values.begin(), values.end());
    EXPECT_EQ(values, positive);
}

// More picks than positive weights: the rest tie at the lowest key and go
// to the lowest rows, after every positive row.
TEST(DensifyEvents4x, GumbelTiesAtTheThresholdTakeTheLowestRows) {
    const std::vector<float> weights{0.0f, 2.0f, 0.0f, 0.0f, 5.0f, 0.0f, 1.0f, 0.0f};
    const auto w = Tensor::from_vector(weights, TensorShape({weights.size()}), Device::CUDA);
    auto picks = Tensor::empty({5}, Device::CUDA, DataType::Int64);
    mrnf_strategy::launch_gumbel_topk(w.ptr<float>(), weights.size(), 5, 11, picks.ptr<int64_t>());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    const auto values = picks.cpu().to_vector_int64();
    std::vector<int64_t> positive(values.begin(), values.begin() + 3);
    std::sort(positive.begin(), positive.end());
    EXPECT_EQ(positive, (std::vector<int64_t>{1, 4, 6}));
    EXPECT_EQ(values[3], 0);
    EXPECT_EQ(values[4], 2);
}

TEST(DensifyEvents4x, GumbelLargeSelectionIsUniqueAndPositive) {
    constexpr size_t n = size_t{3} << 20;
    constexpr size_t k = size_t{1} << 20;
    std::vector<float> weights(n);
    for (size_t i = 0; i < n; ++i)
        weights[i] = i % 3 == 1 ? 0.0f : 1.0f + static_cast<float>((i * 2654435761u) % 997);
    const auto w = Tensor::from_vector(weights, TensorShape({n}), Device::CUDA);
    auto picks = Tensor::empty({k}, Device::CUDA, DataType::Int64);
    mrnf_strategy::launch_gumbel_topk(w.ptr<float>(), n, k, 99, picks.ptr<int64_t>());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    auto values = picks.cpu().to_vector_int64();
    for (const int64_t v : values)
        ASSERT_GT(weights[static_cast<size_t>(v)], 0.0f);
    std::sort(values.begin(), values.end());
    EXPECT_EQ(std::adjacent_find(values.begin(), values.end()), values.end());
}

TEST(DensifyEvents4x, KthLargestMatchesASortOverIncludedRows) {
    constexpr size_t n = size_t{1} << 20;
    std::vector<float> values(n);
    std::vector<float> included_values;
    std::vector<float> include_flags(n);
    for (size_t i = 0; i < n; ++i) {
        // Few distinct values, so the threshold is shared by many rows.
        values[i] = static_cast<float>(static_cast<int>((i * 2654435761u) % 4001) - 2000) * 0.25f;
        include_flags[i] = (i % 5 != 0) ? 1.0f : 0.0f;
        if (include_flags[i] != 0.0f)
            included_values.push_back(values[i]);
    }
    std::sort(included_values.begin(), included_values.end(), std::greater<float>());
    const auto v = Tensor::from_vector(values, TensorShape({n}), Device::CUDA);
    const auto include = Tensor::from_vector(include_flags, TensorShape({n}), Device::CUDA) > 0.5f;
    for (const size_t k : {size_t{1}, size_t{777}, included_values.size() / 2, included_values.size()}) {
        SCOPED_TRACE(k);
        EXPECT_EQ(mrnf_strategy::launch_kth_largest(v.ptr<float>(), include.ptr<bool>(), n, k),
                  included_values[k - 1]);
    }
    std::vector<float> all = values;
    std::sort(all.begin(), all.end(), std::greater<float>());
    EXPECT_EQ(mrnf_strategy::launch_kth_largest(v.ptr<float>(), nullptr, n, 12345), all[12344]);
}

TEST(DensifyEvents4x, DensifyChildWorkspaceGrowsOnly) {
    DensifyChildWorkspace ws;
    ws.ensure(100, 0, false, false, Device::CUDA);
    const size_t cap1 = ws.capacity;
    EXPECT_GE(cap1, 100u);
    auto m1 = ws.means.ptr<float>();

    ws.ensure(50, 0, false, false, Device::CUDA); // smaller — no realloc
    EXPECT_EQ(ws.capacity, cap1);
    EXPECT_EQ(ws.means.ptr<float>(), m1);

    ws.ensure(cap1 + 10, 0, false, false, Device::CUDA); // grow
    EXPECT_GT(ws.capacity, cap1);
}

// The first ensure is sized exactly; the growth headroom applies only when a
// reused workspace grows.
TEST(DensifyEvents4x, DensifyChildWorkspaceStartsAtExactCapacity) {
    DensifyChildWorkspace ws;
    ws.ensure(100, 0, false, false, Device::CUDA);
    EXPECT_EQ(ws.capacity, 100u);
}

TEST(DensifyEvents4x, ScoreBufferAppendZerosInPlace) {
    Tensor scores = Tensor::zeros_direct(TensorShape({4}), /*capacity=*/16, Device::CUDA);
    std::vector<float> init = {1.f, 2.f, 3.f, 4.f};
    ASSERT_EQ(cudaMemcpy(scores.ptr<float>(), init.data(), 4 * sizeof(float),
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    ensure_score_buffer_inplace(scores, 10, Device::CUDA, 16);
    EXPECT_EQ(scores.numel(), 10u);
    EXPECT_GE(scores.capacity(), 16u);
    auto v = to_host(scores);
    EXPECT_FLOAT_EQ(v[0], 1.f);
    EXPECT_FLOAT_EQ(v[3], 4.f);
    EXPECT_FLOAT_EQ(v[4], 0.f);
    EXPECT_FLOAT_EQ(v[9], 0.f);
}

TEST(DensifyEvents4x, DensificationInfoReusesMatchingShape) {
    Tensor info = Tensor::zeros({2, 32}, Device::CUDA);
    float* p0 = info.ptr<float>();
    ensure_densification_info_shape_inplace(info, 32, Device::CUDA);
    EXPECT_EQ(info.ptr<float>(), p0); // same storage
    ensure_densification_info_shape_inplace(info, 40, Device::CUDA);
    EXPECT_EQ(info.shape()[1], 40u);
    EXPECT_NE(info.ptr<float>(), p0); // reallocated for new N
}

TEST(DensifyEvents4x, DensificationInfoLayoutChangeKeepsTheErrorRow) {
    constexpr size_t n = 6;
    const std::vector<float> weight = {1, 2, 3, 4, 5, 6};
    const std::vector<float> error = {0.5f, 0.f, 7.f, 0.25f, 9.f, 1.f};
    std::vector<float> rows = weight;
    rows.insert(rows.end(), error.begin(), error.end());
    Tensor info = Tensor::from_vector(rows, {2, n}, Device::CUDA);

    ensure_densification_info_shape_inplace(info, n, Device::CUDA, 1);
    ASSERT_EQ(info.ndim(), 1u);
    ASSERT_EQ(info.shape()[0], n);
    EXPECT_EQ(to_host(info), error);
    float* const single = info.ptr<float>();
    ensure_densification_info_shape_inplace(info, n, Device::CUDA, 1);
    EXPECT_EQ(info.ptr<float>(), single);

    ensure_densification_info_shape_inplace(info, n, Device::CUDA);
    ASSERT_EQ(info.ndim(), 2u);
    ASSERT_EQ(info.shape()[0], 2u);
    const auto both = to_host(info);
    EXPECT_EQ(std::vector<float>(both.begin(), both.begin() + n), std::vector<float>(n, 0.f));
    EXPECT_EQ(std::vector<float>(both.begin() + n, both.end()), error);

    ensure_densification_info_shape_inplace(info, n + 2, Device::CUDA, 1);
    EXPECT_EQ(to_host(info), std::vector<float>(n + 2, 0.f));
}

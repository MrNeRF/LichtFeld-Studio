/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor/internal/warp_reduce.cuh"
#include <algorithm>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <vector>

namespace {
    namespace warp_ops = lfs::core::warp_ops;

    // 64 threads span two warps, so the second block stage pads 30 empty lanes.
    // The vectorized and segment reductions read four values per thread, so most
    // threads see no input at all. Both rely on the reduction's start value.
    template <typename T>
    __global__ void reduce(const T* input, const size_t n, const int reduction, T* output) {
        const T value = input[threadIdx.x % n];
        T result{};
        switch (reduction) {
        case 0: result = warp_ops::block_reduce_max(value); break;
        case 1: result = warp_ops::vectorized_block_reduce_max(input, n); break;
        case 2: result = warp_ops::vectorized_segment_reduce_max(input, n); break;
        case 3: result = warp_ops::block_reduce_min(value); break;
        case 4: result = warp_ops::vectorized_block_reduce_min(input, n); break;
        case 5: result = warp_ops::vectorized_segment_reduce_min(input, n); break;
        }
        if (threadIdx.x == 0)
            *output = result;
    }

    template <typename T>
    void expect_extrema(const std::vector<T>& values) {
        T* d_input = nullptr;
        T* d_output = nullptr;
        ASSERT_EQ(cudaMalloc(&d_input, values.size() * sizeof(T)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_output, sizeof(T)), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(d_input, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice), cudaSuccess);
        for (int reduction = 0; reduction < 6; ++reduction) {
            reduce<<<1, 64>>>(d_input, values.size(), reduction, d_output);
            ASSERT_EQ(cudaGetLastError(), cudaSuccess);
            T result{};
            ASSERT_EQ(cudaMemcpy(&result, d_output, sizeof(T), cudaMemcpyDeviceToHost), cudaSuccess);
            const auto expected = reduction < 3 ? std::max_element(values.begin(), values.end())
                                                : std::min_element(values.begin(), values.end());
            EXPECT_EQ(result, *expected) << "reduction " << reduction;
        }
        EXPECT_EQ(cudaFree(d_input), cudaSuccess);
        EXPECT_EQ(cudaFree(d_output), cudaSuccess);
    }

    TEST(WarpReduceIdentity, ExtremaIgnoreEmptyLanes) {
        constexpr float inf = std::numeric_limits<float>::infinity();
        expect_extrema<int32_t>({-9, -4, -7, -12, -5, -30, -8, -6});
        expect_extrema<int32_t>({9, 4, 7, 12, 5, 30, 8, 6});
        expect_extrema<uint32_t>({9, 4, 7, 4'000'000'000u, 5, 30, 8, 6});
        expect_extrema<float>(std::vector<float>(8, -inf));
        expect_extrema<float>(std::vector<float>(8, inf));
    }
} // namespace

/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "warp_gradient_reduce.cuh"
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <vector>

namespace {
    namespace reduction = fast_lfs::rasterization::kernels;

    template <int N, int Offset>
    __device__ void halve_down_to(float* values, const unsigned lane) {
        if constexpr (Offset > 0) {
            reduction::halving_step<N, Offset>(values, lane);
            halve_down_to<(N + 1) / 2, Offset / 2>(values, lane);
        }
    }

    // Full reduce-scatter: every field total ends in exactly one lane.
    template <int N>
    __global__ void scatter_all(const float* input, float* output) {
        const unsigned lane = threadIdx.x;
        float values[N];
#pragma unroll
        for (int field = 0; field < N; ++field)
            values[field] = input[lane * N + field];
        halve_down_to<N, 16>(values, lane);
        const int field = reduction::halving_index<N, 16u, 5>(lane);
        if (field >= 0)
            output[field] = values[0];
    }

    // Column phase only: lanes differing in bits 16 and 8 are summed, and each
    // lane keeps several column totals.
    template <int N>
    __global__ void scatter_columns(const float* input, float* output) {
        const unsigned lane = threadIdx.x;
        float values[N];
#pragma unroll
        for (int field = 0; field < N; ++field)
            values[field] = input[lane * N + field];
        reduction::halving_step<N, 16>(values, lane);
        reduction::halving_step<(N + 1) / 2, 8>(values, lane);
        constexpr int kHeld = (N + 3) / 4;
#pragma unroll
        for (int e = 0; e < kHeld; ++e) {
            const int field = reduction::halving_index<N, 16u, 2>(lane, e);
            if (field >= 0)
                output[(lane & 7u) * N + field] = values[e];
        }
    }

    float field_value(const int lane, const int field) {
        return (lane < 16 ? 0.25f : -0.5f) * static_cast<float>(lane + 1) + static_cast<float>(field * field);
    }

    template <int N>
    void check_scatter_all() {
        SCOPED_TRACE(::testing::Message() << "fields=" << N);
        std::vector<float> input(32 * N);
        for (int lane = 0; lane < 32; ++lane)
            for (int field = 0; field < N; ++field)
                input[lane * N + field] = field_value(lane, field);
        std::vector<float> output(N, std::numeric_limits<float>::quiet_NaN());
        float *d_input = nullptr, *d_output = nullptr;
        ASSERT_EQ(cudaMalloc(&d_input, input.size() * sizeof(float)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_output, output.size() * sizeof(float)), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(d_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(d_output, output.data(), output.size() * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
        scatter_all<N><<<1, 32>>>(d_input, d_output);
        ASSERT_EQ(cudaGetLastError(), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(output.data(), d_output, output.size() * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
        EXPECT_EQ(cudaFree(d_input), cudaSuccess);
        EXPECT_EQ(cudaFree(d_output), cudaSuccess);
        for (int field = 0; field < N; ++field) {
            double total = 0;
            for (int lane = 0; lane < 32; ++lane)
                total += input[lane * N + field];
            EXPECT_NEAR(output[field], total, 1e-3) << "field " << field;
        }
    }

    template <int N>
    void check_scatter_columns() {
        SCOPED_TRACE(::testing::Message() << "fields=" << N);
        std::vector<float> input(32 * N);
        for (int lane = 0; lane < 32; ++lane)
            for (int field = 0; field < N; ++field)
                input[lane * N + field] = field_value(lane, field);
        std::vector<float> output(8 * N, std::numeric_limits<float>::quiet_NaN());
        float *d_input = nullptr, *d_output = nullptr;
        ASSERT_EQ(cudaMalloc(&d_input, input.size() * sizeof(float)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_output, output.size() * sizeof(float)), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(d_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(d_output, output.data(), output.size() * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
        scatter_columns<N><<<1, 32>>>(d_input, d_output);
        ASSERT_EQ(cudaGetLastError(), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(output.data(), d_output, output.size() * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
        EXPECT_EQ(cudaFree(d_input), cudaSuccess);
        EXPECT_EQ(cudaFree(d_output), cudaSuccess);
        for (int column = 0; column < 8; ++column) {
            for (int field = 0; field < N; ++field) {
                double total = 0;
                for (int row = 0; row < 4; ++row)
                    total += input[(row * 8 + column) * N + field];
                EXPECT_NEAR(output[column * N + field], total, 1e-4) << "column " << column << " field " << field;
            }
        }
    }

    TEST(RasterGradientReduction, ReduceScatterPlacesEveryTotalInOneLane) {
        check_scatter_all<1>();
        check_scatter_all<5>();
        check_scatter_all<8>();
        check_scatter_all<9>();
        check_scatter_all<14>();
        check_scatter_all<32>();
    }

    TEST(RasterGradientReduction, ColumnPhaseKeepsEveryColumnTotal) {
        check_scatter_columns<1>();
        check_scatter_columns<4>();
        check_scatter_columns<9>();
        check_scatter_columns<12>();
    }
} // namespace

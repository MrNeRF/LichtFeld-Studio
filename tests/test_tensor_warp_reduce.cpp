/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "cuda_backend_test.hpp"

#include <cmath>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

void launch_consecutive_tensor_warp_reductions(float* output, int blocks);

class TensorWarpReduceTest : public lfs::test::CudaBackendTest {};

TEST_F(TensorWarpReduceTest, ConsecutiveBlockReductionsKeepIndependentResults) {
    constexpr int kBlocks = 128;
    constexpr int kValuesPerBlock = 8;
    float* device_output = nullptr;
    ASSERT_EQ(cudaMalloc(&device_output, kBlocks * kValuesPerBlock * sizeof(float)), cudaSuccess);

    launch_consecutive_tensor_warp_reductions(device_output, kBlocks);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<float> results(kBlocks * kValuesPerBlock);
    ASSERT_EQ(cudaMemcpy(results.data(), device_output, results.size() * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaFree(device_output), cudaSuccess);

    for (int block = 0; block < kBlocks; ++block) {
        const auto* values = results.data() + block * kValuesPerBlock;
        EXPECT_FLOAT_EQ(values[0], 32896.0f);
        EXPECT_FLOAT_EQ(values[1], -65792.0f);
        EXPECT_FLOAT_EQ(values[2], 256.0f);
        EXPECT_FLOAT_EQ(values[3], -1.0f);
        EXPECT_FLOAT_EQ(values[4], 1.0f);
        EXPECT_FLOAT_EQ(values[5], -256.0f);
        EXPECT_FLOAT_EQ(values[6], 1.0f);
        EXPECT_NEAR(values[7], std::pow(1.0001f, 256.0f), 1e-5f);
    }
}

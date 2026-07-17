/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/cuda/selection_ops.hpp"
#include "core/tensor.hpp"

#include <algorithm>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

using lfs::core::DataType;
using lfs::core::Device;
using lfs::core::Tensor;

namespace {

    Tensor make_uint8_mask(const std::vector<uint8_t>& values) {
        auto tensor = Tensor::empty({values.size()}, Device::CPU, DataType::UInt8);
        std::copy(values.begin(), values.end(), tensor.ptr<uint8_t>());
        return tensor.cuda();
    }

    Tensor make_means(const std::vector<float>& xyz) {
        // xyz is a flat [N*3] list
        return Tensor::from_vector(xyz, {xyz.size() / 3, 3}, Device::CUDA);
    }

} // namespace

// Success-path regression for selection_ops build_grid AWAIT + launch checks
// (Phase 6B-2 P1 §6.3). Exercises build_grid via selection_grow / selection_shrink.
class SelectionOpsCudaTest : public ::testing::Test {
protected:
    void SetUp() override {
        int device = -1;
        if (cudaGetDevice(&device) != cudaSuccess) {
            (void)cudaGetLastError();
            GTEST_SKIP() << "a live CUDA device is required";
        }
    }
};

TEST_F(SelectionOpsCudaTest, GrowAndShrinkSuccessPathDoesNotThrow) {
    // Three points: seed at origin (selected), neighbor within radius, far point.
    const auto means = make_means({
        0.0f, 0.0f, 0.0f, // 0: seed
        0.5f, 0.0f, 0.0f, // 1: within radius 1.0
        5.0f, 0.0f, 0.0f, // 2: far
    });
    const auto mask = make_uint8_mask({1, 0, 0});

    Tensor grown;
    EXPECT_NO_THROW(grown = lfs::core::cuda::selection_grow(mask, means, 1.0f, /*group_id=*/1));
    ASSERT_EQ(grown.numel(), 3u);
    ASSERT_EQ(grown.device(), Device::CUDA);

    const auto grown_cpu = grown.cpu().to_vector_uint8();
    EXPECT_EQ(grown_cpu[0], 1);
    EXPECT_EQ(grown_cpu[1], 1); // neighbor absorbed
    EXPECT_EQ(grown_cpu[2], 0); // far point stays unselected

    Tensor shrunk;
    EXPECT_NO_THROW(shrunk = lfs::core::cuda::selection_shrink(grown, means, 1.0f));
    ASSERT_EQ(shrunk.numel(), 3u);
    const auto shrunk_cpu = shrunk.cpu().to_vector_uint8();
    // Erosion by radius 1: seed has an unselected neighbor within radius (point 2 is far;
    // point 1 is selected). Point 1's neighborhood includes selected seed — shrink keeps
    // interior points and drops boundary. At minimum, success path must not throw and
    // return a well-formed mask of the same size.
    EXPECT_EQ(shrunk_cpu.size(), 3u);
    for (const auto v : shrunk_cpu) {
        EXPECT_TRUE(v == 0 || v == 1);
    }
}

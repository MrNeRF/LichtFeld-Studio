/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/selection_ops.hpp"
#include "core/tensor.hpp"
#include "core/tensor_backend.hpp"
#include <limits>

#include <algorithm>
#include <gtest/gtest.h>
#include <vector>

using lfs::core::DataType;
using lfs::core::Device;
using lfs::core::Tensor;

namespace {

    Tensor make_uint8_mask(const std::vector<uint8_t>& values) {
        auto tensor = Tensor::empty({values.size()}, Device::CPU, DataType::UInt8);
        std::copy(values.begin(), values.end(), tensor.ptr<uint8_t>());
        return tensor.gpu();
    }

    Tensor make_means(const std::vector<float>& xyz) {
        // xyz is a flat [N*3] list
        return Tensor::from_vector(xyz, {xyz.size() / 3, 3}, Device::GPU);
    }

} // namespace

// Success-path regression for selection_ops build_grid AWAIT + launch checks
// (Phase 6B-2 P1 §6.3). Exercises build_grid via selection_grow / selection_shrink.
class SelectionOpsCudaTest : public ::testing::Test {};

TEST_F(SelectionOpsCudaTest, GrowAndShrinkSuccessPathDoesNotThrow) {
    // Three points: seed at origin (selected), neighbor within radius, far point.
    const auto means = make_means({
        0.0f,
        0.0f,
        0.0f, // 0: seed
        0.5f,
        0.0f,
        0.0f, // 1: within radius 1.0
        5.0f,
        0.0f,
        0.0f, // 2: far
    });
    const auto mask = make_uint8_mask({1, 0, 0});

    Tensor grown;
    EXPECT_NO_THROW(grown = lfs::core::selection_grow(mask, means, 1.0f, /*group_id=*/1));
    ASSERT_EQ(grown.numel(), 3u);
    ASSERT_EQ(grown.device(), Device::GPU);

    const auto grown_cpu = grown.cpu().to_vector_uint8();
    EXPECT_EQ(grown_cpu[0], 1);
    EXPECT_EQ(grown_cpu[1], 1); // neighbor absorbed
    EXPECT_EQ(grown_cpu[2], 0); // far point stays unselected

    Tensor shrunk;
    EXPECT_NO_THROW(shrunk = lfs::core::selection_shrink(grown, means, 1.0f));
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

TEST(SelectionOpsBackends, GrowShrinkPreserveGroupsAndIgnoreNonfinitePoints) {
    using namespace lfs::core;
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    const auto cpu_points = Tensor::from_vector(std::vector<float>{
                                                    -3, 0, 0, -2, 0, 0, -1, 0, 0, .5f, 0, 0, 5000, 1, 0, 5000.5f, 1, 0,
                                                    10000, 2, 0, nan, 0, 0, inf, 0, 0, -inf, 0, 0},
                                                {10, 3}, Device::CPU);
    const std::vector<uint8_t> values{3, 0, 9, 0, 5, 0, 9, 9, 0, 0};
    auto cpu_mask = Tensor::empty({10}, Device::CPU, DataType::UInt8);
    std::copy(values.begin(), values.end(), cpu_mask.ptr<uint8_t>());
    for (int storage = 0; storage < 3; ++storage) {
        const auto backend = storage == 2 ? GpuBackend::Vulkan : GpuBackend::CUDA;
        if (storage && !gpu_backend_available(backend))
            continue;
        const GpuBackendScope scope(backend);
        const auto device = storage ? Device::GPU : Device::CPU;
        const auto points = cpu_points.to(device);
        const auto mask = cpu_mask.to(device);
        EXPECT_EQ(selection_grow(mask, points, 1.f, 7).to_vector_uint8(), (std::vector<uint8_t>{3, 7, 9, 0, 5, 7, 9, 9, 0, 0}));
        EXPECT_EQ(selection_shrink(mask, points, 1.f).to_vector_uint8(), (std::vector<uint8_t>{0, 0, 0, 0, 0, 0, 9, 9, 0, 0}));
        EXPECT_EQ(mask.to_vector_uint8(), values);
        const auto empty_points = Tensor::empty({0, 3}, device);
        const auto empty_mask = Tensor::empty({0}, device, DataType::UInt8);
        const auto small_points = Tensor::from_vector(std::vector<float>{0, 0, 0, .25f, 0, 0, 2, 0, 0}, {3, 3}, device);
        const auto none = Tensor::zeros({3}, device, DataType::UInt8);
        const auto all = Tensor::from_vector(std::vector<int>{3, 7, 9}, {3}, device).to(DataType::UInt8);
        const GpuBackendScope opposite(backend == GpuBackend::CUDA ? GpuBackend::Vulkan : GpuBackend::CUDA);
        EXPECT_EQ(selection_grow(empty_mask, empty_points, 1.f, 1).numel(), 0u);
        EXPECT_EQ(selection_shrink(empty_mask, empty_points, 1.f).numel(), 0u);
        EXPECT_EQ(selection_grow(none, small_points, 1.f, 5).to_vector_uint8(), (std::vector<uint8_t>{0, 0, 0}));
        EXPECT_EQ(selection_shrink(all, small_points, 1.f).to_vector_uint8(), (std::vector<uint8_t>{3, 7, 9}));
        EXPECT_EQ(gpu_backend_of(select_by_opacity(Tensor::empty_like(empty_points), 0.f, 1.f, 1)), gpu_backend_of(empty_points));
    }
}

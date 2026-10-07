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

    Tensor make_transform_indices(const std::vector<int>& indices) {
        return Tensor::from_vector(indices, {indices.size()}, Device::GPU);
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
    EXPECT_EQ(grown_cpu, (std::vector<uint8_t>{1, 1, 0})); // legacy untransformed result

    Tensor shrunk;
    EXPECT_NO_THROW(shrunk = lfs::core::selection_shrink(grown, means, 1.0f));
    ASSERT_EQ(shrunk.numel(), 3u);
    const auto shrunk_cpu = shrunk.cpu().to_vector_uint8();
    EXPECT_EQ(shrunk_cpu, (std::vector<uint8_t>{1, 1, 0})); // legacy untransformed result
}

TEST_F(SelectionOpsCudaTest, GrowUsesWorldPositionsAcrossTransformedNodes) {
    const auto means = make_means({
        0.0f,
        0.0f,
        0.0f, // selected seed
        0.4f,
        0.0f,
        0.0f, // within radius in node-local space
        0.8f,
        0.0f,
        0.0f, // outside radius in node-local space
        0.0f,
        0.0f,
        0.0f, // same local position, translated and scaled away
        0.1f,
        0.0f,
        0.0f,
        0.2f,
        0.0f,
        0.0f,
    });
    const auto mask = make_uint8_mask({1, 0, 0, 0, 0, 0});
    const auto transform_indices = make_transform_indices({0, 0, 0, 1, 1, 1});
    glm::mat4 distant_node(1.0f);
    distant_node[0][0] = 4.0f;
    distant_node[1][1] = 4.0f;
    distant_node[2][2] = 4.0f;
    distant_node[3][0] = 3.0f;
    const std::vector<glm::mat4> transforms{glm::mat4(1.0f), distant_node};

    const auto grown = lfs::core::selection_grow(
                           mask, means, 0.41f, /*group_id=*/1, &transform_indices, &transforms)
                           .cpu()
                           .to_vector_uint8();
    EXPECT_EQ(grown, (std::vector<uint8_t>{1, 1, 0, 0, 0, 0}));
}

TEST_F(SelectionOpsCudaTest, ShrinkUsesWorldSpacingAfterNodeScale) {
    const auto means = make_means({
        -0.4f,
        -0.4f,
        0.0f,
        0.0f,
        -0.4f,
        0.0f,
        0.4f,
        -0.4f,
        0.0f,
        -0.4f,
        0.0f,
        0.0f,
        0.0f,
        0.0f,
        0.0f,
        0.4f,
        0.0f,
        0.0f,
        -0.4f,
        0.4f,
        0.0f,
        0.0f,
        0.4f,
        0.0f,
        0.4f,
        0.4f,
        0.0f,
    });
    const auto mask = make_uint8_mask({0, 1, 0, 1, 1, 1, 0, 1, 0});
    const auto transform_indices = make_transform_indices(std::vector<int>(9, 0));
    glm::mat4 scaled_node(1.0f);
    scaled_node[0][0] = 4.0f;
    scaled_node[1][1] = 4.0f;
    scaled_node[2][2] = 4.0f;
    const std::vector<glm::mat4> transforms{scaled_node};

    const auto shrunk = lfs::core::selection_shrink(
                            mask, means, 0.41f, &transform_indices, &transforms)
                            .cpu()
                            .to_vector_uint8();
    EXPECT_EQ(shrunk, (std::vector<uint8_t>{0, 1, 0, 1, 1, 1, 0, 1, 0}));
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

TEST(SelectionOpsBackends, WorldTransformsMatchExplicitWorldPositions) {
    using namespace lfs::core;
    const std::vector<float> local{0, 0, 0, .25f, 0, 0, 0, 0, 0, .5f, .25f, 0};
    glm::mat4 transformed(1.f);
    transformed[0][0] = 2.f;
    transformed[1][0] = 1.f;
    transformed[1][1] = .5f;
    transformed[3][0] = .125f;
    const std::vector<glm::mat4> matrices{glm::mat4(1.f), transformed};
    const std::vector<int> indices{0, 0, 1, 1};
    std::vector<float> world;
    for (size_t i = 0; i < indices.size(); ++i) {
        const auto point = matrices[indices[i]] * glm::vec4(local[3 * i], local[3 * i + 1], local[3 * i + 2], 1.f);
        world.insert(world.end(), {point.x, point.y, point.z});
    }
    const auto cpu_world = Tensor::from_vector(world, {4, 3}, Device::CPU);
    const auto cpu_mask = Tensor::from_vector(std::vector<int>{3, 0, 7, 0}, {4}, Device::CPU).to(DataType::UInt8);
    const auto expected_grow = selection_grow(cpu_mask, cpu_world, .2f, 9).to_vector_uint8();
    const auto expected_shrink = selection_shrink(cpu_mask, cpu_world, .2f).to_vector_uint8();
    for (const auto backend : {GpuBackend::CUDA, GpuBackend::Vulkan, GpuBackend::Metal}) {
        if (!gpu_backend_available(backend))
            continue;
        SCOPED_TRACE(static_cast<int>(backend));
        const GpuBackendScope scope(backend);
        const auto means = Tensor::from_vector(local, {4, 3}, Device::GPU);
        const auto mask = cpu_mask.to(Device::GPU);
        const auto node_indices = Tensor::from_vector(indices, {4}, Device::CPU);
        const GpuBackendScope opposite(backend == GpuBackend::CUDA ? GpuBackend::Vulkan : GpuBackend::CUDA);
        const auto grown = selection_grow(mask, means, .2f, 9, &node_indices, &matrices);
        const auto shrunk = selection_shrink(mask, means, .2f, &node_indices, &matrices);
        EXPECT_EQ(grown.to_vector_uint8(), expected_grow);
        EXPECT_EQ(shrunk.to_vector_uint8(), expected_shrink);
        EXPECT_EQ(gpu_backend_of(grown), backend);
        EXPECT_EQ(gpu_backend_of(shrunk), backend);
        EXPECT_EQ(mask.to_vector_uint8(), cpu_mask.to_vector_uint8());
        const std::vector<glm::mat4> identity(2, glm::mat4(1.f));
        EXPECT_EQ(selection_grow(mask, means, .2f, 9, &node_indices, &identity).to_vector_uint8(),
                  selection_grow(mask, means, .2f, 9).to_vector_uint8());
        EXPECT_EQ(selection_shrink(mask, means, .2f, &node_indices, &identity).to_vector_uint8(),
                  selection_shrink(mask, means, .2f).to_vector_uint8());
    }
}

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

    Tensor make_transform_indices(const std::vector<int>& indices) {
        return Tensor::from_vector(indices, {indices.size()}, Device::CUDA);
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
    EXPECT_NO_THROW(grown = lfs::core::cuda::selection_grow(mask, means, 1.0f, /*group_id=*/1));
    ASSERT_EQ(grown.numel(), 3u);
    ASSERT_EQ(grown.device(), Device::CUDA);

    const auto grown_cpu = grown.cpu().to_vector_uint8();
    EXPECT_EQ(grown_cpu, (std::vector<uint8_t>{1, 1, 0})); // legacy untransformed result

    Tensor shrunk;
    EXPECT_NO_THROW(shrunk = lfs::core::cuda::selection_shrink(grown, means, 1.0f));
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

    const auto grown = lfs::core::cuda::selection_grow(
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

    const auto shrunk = lfs::core::cuda::selection_shrink(
                            mask, means, 0.41f, &transform_indices, &transforms)
                            .cpu()
                            .to_vector_uint8();
    EXPECT_EQ(shrunk, (std::vector<uint8_t>{0, 1, 0, 1, 1, 1, 0, 1, 0}));
}

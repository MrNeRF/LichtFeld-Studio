/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// The merged point cloud transforms points and converts colors on the device;
// both match the per-point CPU arithmetic.

#include "core/tensor.hpp"
#include "scene/point_cloud_merge.hpp"
#include <gtest/gtest.h>

#include <glm/gtc/matrix_transform.hpp>
#include <vector>

namespace {
    using namespace lfs::core;

    constexpr size_t kPoints = 4099;

    std::vector<float> points() {
        std::vector<float> values(kPoints * 3);
        for (size_t i = 0; i < values.size(); ++i)
            values[i] = static_cast<float>(static_cast<int>((i * 31) % 2001) - 1000) * 0.037f;
        return values;
    }
} // namespace

TEST(PointCloudMerge, TransformMatchesGlmForCpuAndGpuPoints) {
    const auto values = points();
    glm::mat4 world = glm::translate(glm::mat4(1.0f), glm::vec3(1.5f, -2.25f, 4.0f));
    world = glm::rotate(world, 0.7f, glm::normalize(glm::vec3(0.3f, 1.0f, -0.2f)));
    world = glm::scale(world, glm::vec3(1.25f, 0.8f, 1.1f));
    const Tensor cpu = Tensor::from_vector(values, TensorShape{kPoints, 3}, Device::CPU);
    for (const Tensor& means : {cpu, cpu.to(Device::GPU)}) {
        const Tensor transformed = lfs::vis::transformPointsToWorld(means, world);
        ASSERT_EQ(transformed.device(), Device::CPU);
        ASSERT_EQ(transformed.shape(), TensorShape({kPoints, size_t{3}}));
        const auto got = transformed.to_vector();
        for (size_t i = 0; i < kPoints; ++i) {
            const glm::vec4 expected = world * glm::vec4(values[i * 3], values[i * 3 + 1], values[i * 3 + 2], 1.0f);
            EXPECT_NEAR(got[i * 3], expected.x, 1.0e-4f) << "point=" << i;
            EXPECT_NEAR(got[i * 3 + 1], expected.y, 1.0e-4f) << "point=" << i;
            EXPECT_NEAR(got[i * 3 + 2], expected.z, 1.0e-4f) << "point=" << i;
        }
    }
    EXPECT_EQ(lfs::vis::transformPointsToWorld(Tensor{}, world).numel(), 0u);
}

TEST(PointCloudMerge, ColorsScaleBytesAndKeepFloats) {
    std::vector<int> bytes(kPoints * 3);
    for (size_t i = 0; i < bytes.size(); ++i)
        bytes[i] = static_cast<int>((i * 7) % 256);
    const Tensor byte_colors = Tensor::from_vector(bytes, TensorShape{kPoints, 3}, Device::CPU).to(DataType::UInt8);
    const auto scaled = lfs::vis::pointColorsAsFloat(byte_colors).to_vector();
    ASSERT_EQ(scaled.size(), bytes.size());
    for (size_t i = 0; i < bytes.size(); ++i)
        EXPECT_EQ(scaled[i], static_cast<float>(bytes[i]) * (1.0f / 255.0f)) << "index=" << i;

    const auto floats = points();
    const Tensor float_colors = Tensor::from_vector(floats, TensorShape{kPoints, 3}, Device::CPU).to(Device::GPU);
    EXPECT_EQ(lfs::vis::pointColorsAsFloat(float_colors).to_vector(), floats);
    EXPECT_EQ(lfs::vis::pointColorsAsFloat(Tensor{}).numel(), 0u);
}

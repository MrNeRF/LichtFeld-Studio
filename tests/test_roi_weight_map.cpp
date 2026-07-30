/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include "training/kernels/roi_weight_map.hpp"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace {

    bool ray_intersects_box(
        const glm::vec3& world_origin,
        const glm::vec3& world_direction,
        const glm::mat4& world_to_cropbox,
        const glm::vec3& crop_min,
        const glm::vec3& crop_max) {
        const glm::vec3 origin =
            glm::vec3(world_to_cropbox * glm::vec4(world_origin, 1.0f));
        const glm::vec3 direction =
            glm::vec3(world_to_cropbox * glm::vec4(world_direction, 0.0f));

        float t_enter = -std::numeric_limits<float>::infinity();
        float t_exit = std::numeric_limits<float>::infinity();
        for (int axis = 0; axis < 3; ++axis) {
            if (std::abs(direction[axis]) < 1.0e-8f) {
                if (origin[axis] < crop_min[axis] || origin[axis] > crop_max[axis]) {
                    return false;
                }
                continue;
            }
            float t0 = (crop_min[axis] - origin[axis]) / direction[axis];
            float t1 = (crop_max[axis] - origin[axis]) / direction[axis];
            if (t0 > t1) {
                std::swap(t0, t1);
            }
            t_enter = std::max(t_enter, t0);
            t_exit = std::min(t_exit, t1);
            if (t_exit < t_enter) {
                return false;
            }
        }
        return t_exit >= 0.0f;
    }

    void expect_gpu_matches_reference(
        const glm::mat4& world_to_cropbox,
        const glm::vec3& crop_min,
        const glm::vec3& crop_max,
        const bool inverse,
        const glm::mat4& world_to_camera = glm::mat4(1.0f),
        const glm::vec3& camera_position = glm::vec3(0.0f)) {
        constexpr int width = 9;
        constexpr int height = 7;
        constexpr float fx = 6.0f;
        constexpr float fy = 6.0f;
        constexpr float cx = 4.5f;
        constexpr float cy = 3.5f;
        constexpr float outside_weight = 0.2f;

        std::vector<float> world_to_camera_values(16);
        for (size_t row = 0; row < 4; ++row) {
            for (size_t column = 0; column < 4; ++column) {
                world_to_camera_values[row * 4 + column] =
                    world_to_camera[column][row];
            }
        }
        const std::vector<float> camera_position_values{
            camera_position.x, camera_position.y, camera_position.z};
        const auto w2c = lfs::core::Tensor::from_vector(
            world_to_camera_values, {size_t{4}, size_t{4}}, lfs::core::Device::CUDA);
        const auto camera_pos = lfs::core::Tensor::from_vector(
            camera_position_values, {size_t{3}}, lfs::core::Device::CUDA);
        auto output = lfs::core::Tensor::empty(
            {size_t{height}, size_t{width}},
            lfs::core::Device::CUDA,
            lfs::core::DataType::Float32);

        lfs::training::kernels::launch_roi_weight_map(
            w2c.ptr<float>(),
            camera_pos.ptr<float>(),
            fx,
            fy,
            cx,
            cy,
            width,
            height,
            world_to_cropbox,
            crop_min,
            crop_max,
            inverse,
            outside_weight,
            output.ptr<float>(),
            output.stream());

        std::vector<float> expected;
        expected.reserve(static_cast<size_t>(width) * height);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                const glm::vec3 camera_ray{
                    (static_cast<float>(x) + 0.5f - cx) / fx,
                    (static_cast<float>(y) + 0.5f - cy) / fy,
                    1.0f};
                const glm::vec3 world_ray =
                    glm::transpose(glm::mat3(world_to_camera)) * camera_ray;
                const bool hit = ray_intersects_box(
                    camera_position, world_ray, world_to_cropbox, crop_min, crop_max);
                const bool full_weight = inverse ? !hit : hit;
                expected.push_back(full_weight ? 1.0f : outside_weight);
            }
        }

        const auto actual = output.to(lfs::core::Device::CPU).to_vector();
        ASSERT_EQ(actual.size(), expected.size());
        EXPECT_GT(std::count(expected.begin(), expected.end(), 1.0f), 0);
        EXPECT_GT(std::count(expected.begin(), expected.end(), outside_weight), 0);
        for (size_t i = 0; i < actual.size(); ++i) {
            EXPECT_FLOAT_EQ(actual[i], expected[i]) << "pixel " << i;
        }
    }

} // namespace

TEST(RoiWeightMapTest, AxisAlignedBoxMatchesCpuSlabReference) {
    expect_gpu_matches_reference(
        glm::mat4(1.0f),
        {-0.6f, -0.4f, 2.0f},
        {0.6f, 0.4f, 3.0f},
        false);
}

TEST(RoiWeightMapTest, RotatedObbMatchesCpuSlabReference) {
    const glm::mat4 cropbox_to_world =
        glm::rotate(
            glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, 3.0f)),
            glm::radians(45.0f),
            glm::vec3(0.0f, 0.0f, 1.0f));
    expect_gpu_matches_reference(
        glm::inverse(cropbox_to_world),
        {-0.8f, -0.15f, -0.4f},
        {0.8f, 0.15f, 0.4f},
        false);
}

TEST(RoiWeightMapTest, InverseBoxFlipsHitAndMissWeights) {
    expect_gpu_matches_reference(
        glm::mat4(1.0f),
        {-0.6f, -0.4f, 2.0f},
        {0.6f, 0.4f, 3.0f},
        true);
}

TEST(RoiWeightMapTest, RotatedTranslatedCameraUsesWorldSpaceRays) {
    const glm::vec3 camera_position{-1.0f, 0.0f, 0.0f};
    const glm::mat4 camera_to_world =
        glm::translate(glm::mat4(1.0f), camera_position) *
        glm::rotate(
            glm::mat4(1.0f),
            glm::radians(90.0f),
            glm::vec3(0.0f, 1.0f, 0.0f));
    expect_gpu_matches_reference(
        glm::mat4(1.0f),
        {1.0f, -0.5f, -0.4f},
        {2.0f, 0.5f, 0.4f},
        false,
        glm::inverse(camera_to_world),
        camera_position);
}

/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/point_cloud.hpp"
#include "core/scene.hpp"
#include "core/scene_partitioner.hpp"
#include <cmath>
#include <glm/glm.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/euler_angles.hpp>
#include <gtest/gtest.h>

namespace {
    lfs::core::PointCloud cloud(const std::vector<float>& coordinates) {
        const size_t count = coordinates.size() / 3;
        return {lfs::core::Tensor::from_vector(coordinates, {count, 3}, lfs::core::Device::CPU),
                lfs::core::Tensor::zeros({count, 3}, lfs::core::Device::CPU)};
    }

    TEST(ScenePartitioner, IgnoresSparseOutliers) {
        std::vector<float> points;
        for (int i = 0; i < 100; ++i) {
            points.insert(points.end(), {float(i % 10), float(i / 10), float(i % 3)});
        }
        points.insert(points.end(), {100000.0f, 100000.0f, 100000.0f});
        const auto bounds = lfs::core::compute_robust_scene_bounds(cloud(points), {});
        EXPECT_LT(bounds.max.x, 20.0f);
        EXPECT_LT(bounds.max.y, 20.0f);
        EXPECT_LT(bounds.max.z, 20.0f);
    }

    TEST(ScenePartitioner, FindsDominantAngle) {
        std::vector<float> points;
        constexpr float angle = 0.5235987756f;
        for (int i = -30; i <= 30; ++i) {
            points.insert(points.end(), {i * std::cos(angle), i * std::sin(angle), 0.0f});
        }
        points.insert(points.end(), {100000.0f, -100000.0f, 0.0f});
        EXPECT_NEAR(lfs::core::compute_dominant_orientation_pca(cloud(points)), 30.0f, 0.1f);
    }

    TEST(ScenePartitioner, UsesCameraCentersWhenSparseCloudIsEmpty) {
        const auto bounds = lfs::core::compute_robust_scene_bounds(
            {}, {{0, 0, 0}, {10, 5, 2}});
        EXPECT_NEAR(bounds.min.x, 0.0f, 1e-4f);
        EXPECT_NEAR(bounds.max.x, 10.0f, 1e-4f);
    }

    TEST(ScenePartitioner, UsesCameraCentersWhenSparseCloudHasOnePoint) {
        const auto bounds = lfs::core::compute_robust_scene_bounds(
            cloud({100.0f, 100.0f, 100.0f}), {{0, 0, 0}, {10, 5, 2}});
        EXPECT_NEAR(bounds.min.x, 0.0f, 1e-4f);
        EXPECT_NEAR(bounds.max.x, 10.0f, 1e-4f);
    }

    TEST(ScenePartitioner, OverlapAndRotation) {
        const lfs::core::SceneBounds bounds{{0, 0, -1}, {20, 10, 1}, 30};
        const lfs::core::TileGridConfig config{2, 1, 0, 0.2f, true};
        const auto tiles = lfs::core::generate_tile_grid(bounds, config);
        ASSERT_EQ(tiles.size(), 2);
        EXPECT_EQ(tiles[0].name, "tile_0_0");
        EXPECT_EQ(tiles[1].name, "tile_1_0");
        EXPECT_NEAR(tiles[0].max.x - tiles[0].min.x, 11.0f, 1e-4f);
        EXPECT_NEAR(tiles[1].max.x - tiles[1].min.x, 11.0f, 1e-4f);
        EXPECT_NEAR(tiles[1].rotation_degrees.z, 30.0f, 1e-4f);
        const glm::vec3 center0 = (tiles[0].min + tiles[0].max) * 0.5f;
        const glm::vec3 center1 = (tiles[1].min + tiles[1].max) * 0.5f;
        EXPECT_NEAR(glm::length(center1 - center0), 10.0f, 1e-4f);
    }

    TEST(ScenePartitioner, DerivesGridFromTargetSize) {
        const auto tiles = lfs::core::generate_tile_grid(
            {{0, 0, 0}, {21, 11, 2}, 0}, {0, 0, 10.0f, 0.0f, false});
        ASSERT_EQ(tiles.size(), 6);
        EXPECT_EQ(tiles.back().name, "tile_2_1");
    }

    TEST(ScenePartitioner, MergePreservesBothTileModels) {
        const auto model = [](float x) {
            return lfs::core::SplatData(
                0, lfs::core::Tensor::from_vector(std::vector<float>{x, 0, 0}, {1, 3}, lfs::core::Device::CPU),
                lfs::core::Tensor::zeros({1, 1, 3}, lfs::core::Device::CPU),
                lfs::core::Tensor::zeros({1, 0, 3}, lfs::core::Device::CPU),
                lfs::core::Tensor::zeros({1, 3}, lfs::core::Device::CPU),
                lfs::core::Tensor::from_vector(std::vector<float>{1, 0, 0, 0}, {1, 4}, lfs::core::Device::CPU),
                lfs::core::Tensor::zeros({1, 1}, lfs::core::Device::CPU), 1.0f);
        };
        const auto left = model(0.0f), right = model(0.5f);
        auto merged = lfs::core::Scene::mergeSplatsWithTransforms({{&left, glm::mat4{1.0f}},
                                                                    {&right, glm::mat4{1.0f}}});
        ASSERT_NE(merged, nullptr);
        EXPECT_EQ(merged->size(), 2);
        const auto means = merged->means().cpu().contiguous();
        const auto* positions = means.ptr<float>();
        ASSERT_NE(positions, nullptr);
        EXPECT_NEAR(positions[0], 0.0f, 1e-4f);
        EXPECT_NEAR(positions[3], 0.5f, 1e-4f);
    }

    TEST(ScenePartitioner, PreviewCreatesIndependentInactiveBoxes) {
        lfs::core::Scene scene;
        const auto tiles = lfs::core::generate_tile_grid(
            {{0, 0, 0}, {10, 10, 2}, 0}, {2, 1, 0, 0.15f, false});
        const auto group = scene.createTileCropBoxes(tiles);
        ASSERT_NE(group, lfs::core::NULL_NODE);
        const auto* node = scene.getNodeById(group);
        ASSERT_NE(node, nullptr);
        ASSERT_EQ(node->children.size(), 2);
        for (const auto child : node->children) {
            const auto* box = scene.getNodeById(child);
            ASSERT_NE(box, nullptr);
            ASSERT_NE(box->cropbox, nullptr);
            EXPECT_FALSE(box->cropbox->enabled);
        }
    }

    TEST(ScenePartitioner, Robust3dObbOrientation) {
        // Create a tilted planar point cloud with large spread along local X,
        // medium spread along local Y, and thin spread along local Z.
        std::vector<float> points;
        const float rx = glm::radians(30.0f);
        const float rz = glm::radians(45.0f);
        const glm::mat4 R = glm::eulerAngleXYZ(rx, 0.0f, rz);
        for (int i = -10; i <= 10; ++i) {
            for (int j = -5; j <= 5; ++j) {
                const glm::vec4 local_p(static_cast<float>(i), static_cast<float>(j), 0.0f, 1.0f);
                const glm::vec4 world_p = R * local_p;
                points.push_back(world_p.x);
                points.push_back(world_p.y);
                points.push_back(world_p.z);
            }
        }
        // Add random sparse outlier
        points.push_back(9999.0f); points.push_back(-9999.0f); points.push_back(9999.0f);

        const auto pc = cloud(points);
        const auto bounds_obb = lfs::core::compute_robust_scene_bounds_obb(
            pc, {}, 0.02f, lfs::core::OrientationMode::Obb3D, lfs::core::BoundsSource::Both);
        const auto bounds_aabb = lfs::core::compute_robust_scene_bounds_obb(
            pc, {}, 0.02f, lfs::core::OrientationMode::AxisAligned, lfs::core::BoundsSource::Both);

        const glm::vec3 extent_obb = bounds_obb.max - bounds_obb.min;
        const glm::vec3 extent_aabb = bounds_aabb.max - bounds_aabb.min;

        // OBB thickness (local Z) should be very small (~1 or less due to clamping),
        // whereas AABB extent in Z must span the tilted plane (~10*sin(30) + 5*sin(...)).
        EXPECT_LT(extent_obb.z, extent_aabb.z);
        EXPECT_LT(extent_obb.z, 2.0f);
        EXPECT_GT(extent_aabb.z, 8.0f);

        // OBB local X and Y extents should closely match the ground truth (-10 to 10 and -5 to 5)
        EXPECT_NEAR(extent_obb.x, 20.0f, 1.5f);
        EXPECT_NEAR(extent_obb.y, 10.0f, 1.5f);
    }

    TEST(ScenePartitioner, PlanarPitchAndRollFollowTheirPlanes) {
        std::vector<float> pitch_points;
        std::vector<float> roll_points;
        constexpr float angle = 30.0f;
        const float radians = glm::radians(angle);
        for (int i = -20; i <= 20; ++i) {
            for (int j = -2; j <= 2; ++j) {
                pitch_points.insert(pitch_points.end(),
                                    {i * std::cos(radians), float(j), i * std::sin(radians)});
                roll_points.insert(roll_points.end(),
                                   {float(j), i * std::cos(radians), i * std::sin(radians)});
            }
        }
        const auto pitch = lfs::core::compute_robust_scene_bounds_obb(
            cloud(pitch_points), {}, 0.0f, lfs::core::OrientationMode::PlanarPitch);
        const auto roll = lfs::core::compute_robust_scene_bounds_obb(
            cloud(roll_points), {}, 0.0f, lfs::core::OrientationMode::PlanarRoll);
        EXPECT_NEAR(pitch.rotation_degrees.y, -angle, 0.1f);
        EXPECT_NEAR(pitch.rotation_degrees.x, 0.0f, 1e-4f);
        EXPECT_NEAR(roll.rotation_degrees.x, angle, 0.1f);
        EXPECT_NEAR(roll.rotation_degrees.y, 0.0f, 1e-4f);
        EXPECT_NEAR((pitch.max - pitch.min).z, 1.0f, 0.01f);
        EXPECT_NEAR((roll.max - roll.min).z, 1.0f, 0.01f);
    }

    TEST(ScenePartitioner, ThreeDimensionalPcaFallsBackWhenTrimLeavesTooFewPoints) {
        const std::vector<glm::vec3> points{{-10, -5, 0}, {0, 0, 0}, {10, 5, 0}};
        const auto rotation = lfs::core::compute_dominant_orientation_3d(points, 0.49f);
        EXPECT_GT(std::abs(rotation.z), 10.0f);
    }

    TEST(ScenePartitioner, ZeroOverlapAndQuantile) {
        std::vector<float> points;
        for (int i = 0; i <= 10; ++i) {
            for (int j = 0; j <= 10; ++j) {
                points.push_back(static_cast<float>(i));
                points.push_back(static_cast<float>(j));
                points.push_back(0.0f);
            }
        }
        const auto pc = cloud(points);
        const auto bounds = lfs::core::compute_robust_scene_bounds_obb(
            pc, {}, 0.0f, lfs::core::OrientationMode::AxisAligned, lfs::core::BoundsSource::Points);
        EXPECT_NEAR(bounds.min.x, 0.0f, 1e-4f);
        EXPECT_NEAR(bounds.max.x, 10.0f, 1e-4f);
        EXPECT_NEAR(bounds.min.y, 0.0f, 1e-4f);
        EXPECT_NEAR(bounds.max.y, 10.0f, 1e-4f);

        const auto tiles = lfs::core::generate_tile_grid(bounds, {2, 2, 0.0f, 0.0f, false, lfs::core::OrientationMode::AxisAligned, 0.0f});
        ASSERT_EQ(tiles.size(), 4);
        EXPECT_NEAR(tiles[0].max.x, tiles[1].min.x, 1e-4f);
        EXPECT_NEAR(tiles[0].max.y, tiles[2].min.y, 1e-4f);
    }
} // namespace

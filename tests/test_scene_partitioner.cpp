/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/point_cloud.hpp"
#include "core/scene.hpp"
#include "core/scene_partitioner.hpp"
#include <cmath>
#include <glm/glm.hpp>
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
} // namespace

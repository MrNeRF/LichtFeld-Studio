/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/point_cloud.hpp"
#include "core/scene.hpp"
#include "core/scene_partitioner.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
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

    TEST(ScenePartitioner, PointBoundsFollowGeometryAndFallBackToCameras) {
        const auto points = cloud({0, 0, 0, 10, 0, 0, 0, 10, 0, 10, 10, 0});
        const std::vector<glm::vec3> cameras{{100, 0, 0}, {110, 0, 0}};
        const auto point_bounds = lfs::core::compute_robust_scene_bounds_obb(
            points, cameras, 0.0f, lfs::core::OrientationMode::AxisAligned,
            lfs::core::BoundsSource::Points);
        EXPECT_NEAR(point_bounds.min.x, 0.0f, 1e-4f);
        EXPECT_NEAR(point_bounds.max.x, 10.0f, 1e-4f);
        const auto fallback_bounds = lfs::core::compute_robust_scene_bounds_obb(
            {}, cameras, 0.0f, lfs::core::OrientationMode::AxisAligned,
            lfs::core::BoundsSource::Points);
        EXPECT_NEAR(fallback_bounds.min.x, 100.0f, 1e-4f);
        EXPECT_NEAR(fallback_bounds.max.x, 110.0f, 1e-4f);
    }

    TEST(ScenePartitioner, AdaptiveTilesFollowUnevenSurfaceDensity) {
        std::vector<float> coordinates;
        for (int i = 0; i < 80; ++i)
            coordinates.insert(coordinates.end(), {1.0f + 0.01f * i, 1.0f, 1.0f});
        for (int i = 0; i < 20; ++i)
            coordinates.insert(coordinates.end(), {8.0f + 0.01f * i, 1.0f, 1.0f});
        const lfs::core::SceneBounds bounds{{0, 0, 0}, {10, 4, 2}};
        const auto tiles = lfs::core::generate_adaptive_tiles(
            bounds, 2, 0.2f, bounds, cloud(coordinates), {});
        ASSERT_EQ(tiles.size(), 2);
        EXPECT_LT(tiles[0].max.x, 6.0f);
        EXPECT_GE(tiles[0].max.x, tiles[1].min.x);
        EXPECT_LT(tiles[0].min.x, 0.0f);
        EXPECT_GT(tiles[1].max.x, 10.0f);
        EXPECT_EQ(tiles[0].name, "tile_0_0");
        EXPECT_EQ(tiles[1].name, "tile_1_0");
        const auto more_tiles = lfs::core::generate_adaptive_tiles(
            bounds, 4, 0.2f, bounds, cloud(coordinates), {});
        ASSERT_EQ(more_tiles.size(), 4);
        const auto dense_region_tiles = std::count_if(more_tiles.begin(), more_tiles.end(),
            [](const auto& tile) { return (tile.min.x + tile.max.x) * 0.5f < 5.0f; });
        EXPECT_GE(dense_region_tiles, 3);
    }

    TEST(ScenePartitioner, AdaptiveTilesAvoidFullSceneSlivers) {
        std::vector<float> coordinates;
        for (int y = 0; y < 16; ++y)
            for (int x = 0; x < 16; ++x)
                coordinates.insert(coordinates.end(),
                                   {0.3125f + 0.625f * x, 0.3125f + 0.625f * y, 1.0f});
        const lfs::core::SceneBounds bounds{{0, 0, 0}, {10, 10, 2}};
        const auto tiles = lfs::core::generate_adaptive_tiles(
            bounds, 8, 0.0f, bounds, cloud(coordinates), {}, nullptr,
            {.all_cameras_for_training = true});
        ASSERT_EQ(tiles.size(), 8);
        for (const auto& tile : tiles) {
            const float width = tile.max.x - tile.min.x;
            const float height = tile.max.y - tile.min.y;
            EXPECT_LT(std::max(width / height, height / width), 3.5f);
        }
    }

    TEST(ScenePartitioner, AdaptiveTilesBalanceViewAwareWorkAndKeepSharedViews) {
        std::vector<float> coordinates;
        for (int i = 0; i < 100; ++i)
            coordinates.insert(coordinates.end(), {0.05f + 0.1f * i, 1.0f, 1.0f});
        std::vector<lfs::core::TileCameraEvidence> cameras(16);
        for (int i = 0; i < 16; ++i) {
            auto& view = cameras[i];
            view.uid = i;
            view.center = {i < 8 ? 2.0f : 8.0f, 1.0f, -100.0f};
            view.world_to_camera = {1, 0, 0, 0, 0, 1, 0, 0,
                                    0, 0, 1, 0, 0, 0, 0, 1};
            view.width = view.height = i < 8 ? 4000 : 1000;
            view.fx = view.fy = 100.0f;
            view.cx = view.cy = view.width * 0.5f;
            view.pinhole = true;
            for (int j = i < 8 ? 0 : 50; j < (i < 8 ? 50 : 100); ++j)
                view.observations.emplace_back(0.05f + 0.1f * j, 1.0f, 1.0f);
        }
        const lfs::core::SceneBounds bounds{{0, 0, 0}, {10, 2, 2}};
        std::vector<lfs::core::TilePlanDiagnostics> diagnostics;
        const auto tiles = lfs::core::generate_adaptive_tiles(
            bounds, 2, 0.0f, bounds, cloud(coordinates), cameras, &diagnostics);
        ASSERT_EQ(tiles.size(), 2);
        ASSERT_EQ(diagnostics.size(), 2);
        // Equal point counts would cut near x=5. Camera coverage and image
        // resolution make a later cut cheaper while preserving shared views.
        EXPECT_GT(tiles[0].max.x, 6.0f);
        EXPECT_NEAR(tiles[0].max.x, tiles[1].min.x, 1e-4f);
        EXPECT_GE(diagnostics[0].supporting_cameras, 8);
        EXPECT_GE(diagnostics[1].supporting_cameras, 8);
        EXPECT_EQ(diagnostics[0].covered_cells, diagnostics[0].occupied_cells);
        EXPECT_EQ(diagnostics[1].covered_cells, diagnostics[1].occupied_cells);
        EXPECT_GE(diagnostics[0].sampled_crop_points, diagnostics[0].sampled_points);
        EXPECT_GE(diagnostics[1].sampled_crop_points, diagnostics[1].sampled_points);
        EXPECT_GT(diagnostics[0].estimated_work_units, 0.0);
        EXPECT_GT(diagnostics[1].estimated_work_units, 0.0);
        EXPECT_LT(std::max(diagnostics[0].estimated_work_units,
                           diagnostics[1].estimated_work_units), 800.0);

        std::vector<lfs::core::TilePlanDiagnostics> all_camera_diagnostics;
        const auto all_camera_tiles = lfs::core::generate_adaptive_tiles(
            bounds, 2, 0.0f, bounds, cloud(coordinates), cameras,
            &all_camera_diagnostics, {.all_cameras_for_training = true});
        ASSERT_EQ(all_camera_tiles.size(), 2);
        ASSERT_EQ(all_camera_diagnostics.size(), 2);
        EXPECT_NEAR(all_camera_tiles[0].max.x, 5.0f, 0.25f);
        EXPECT_EQ(all_camera_diagnostics[0].estimated_selected_cameras, 16);
        EXPECT_EQ(all_camera_diagnostics[1].estimated_selected_cameras, 16);
        EXPECT_NEAR(all_camera_diagnostics[0].estimated_work_units,
                    all_camera_diagnostics[1].estimated_work_units, 1e-4);

        std::vector<lfs::core::TilePlanDiagnostics> resized_diagnostics;
        const auto resized_tiles = lfs::core::generate_adaptive_tiles(
            bounds, 2, 0.0f, bounds, cloud(coordinates), cameras,
            &resized_diagnostics, {.max_width = 1000});
        ASSERT_EQ(resized_tiles.size(), 2);
        ASSERT_EQ(resized_diagnostics.size(), 2);
        EXPECT_NEAR(resized_diagnostics[0].mean_image_megapixels, 1.0f, 1e-4f);
        EXPECT_NEAR(resized_diagnostics[1].mean_image_megapixels, 1.0f, 1e-4f);
    }

    TEST(ScenePartitioner, PlanningAccountsForMinimumCameraAllowlist) {
        std::vector<float> coordinates;
        for (int i = 0; i < 40; ++i)
            coordinates.insert(coordinates.end(), {0.25f + 0.25f * i, 1.0f, 1.0f});
        std::vector<lfs::core::TileCameraEvidence> cameras(12);
        for (int i = 0; i < 12; ++i) {
            cameras[i].uid = i;
            cameras[i].center = {static_cast<float>(i), 1.0f, 100.0f};
            cameras[i].width = cameras[i].height = 1000;
            cameras[i].pinhole = true;
        }
        const lfs::core::SceneBounds bounds{{0, 0, 0}, {10, 2, 2}};
        std::vector<lfs::core::TilePlanDiagnostics> diagnostics;
        const auto tiles = lfs::core::generate_adaptive_tiles(
            bounds, 2, 0.0f, bounds, cloud(coordinates), cameras, &diagnostics);
        ASSERT_EQ(tiles.size(), 2);
        ASSERT_EQ(diagnostics.size(), 2);
        EXPECT_EQ(diagnostics[0].supporting_cameras, 0);
        EXPECT_EQ(diagnostics[1].supporting_cameras, 0);
        EXPECT_EQ(diagnostics[0].estimated_selected_cameras, 8);
        EXPECT_EQ(diagnostics[1].estimated_selected_cameras, 8);
    }

    TEST(ScenePartitioner, CameraSelectionUsesProjectedAirspaceAndTrackSupport) {
        lfs::core::TileSpec tile{{-1, -1, 9}, {1, 1, 11}, {}, 0, 0, "tile_0_0"};
        std::vector<lfs::core::TileCameraEvidence> cameras(12);
        for (int i = 0; i < 12; ++i) {
            auto& view = cameras[i];
            view.uid = i;
            view.center = {100.0f + i, 0, 0};
            view.world_to_camera = {1, 0, 0, 0, 0, 1, 0, 0,
                                    0, 0, 1, 0, 0, 0, 0, 1};
            view.fx = 80; view.fy = 80; view.cx = 50; view.cy = 50;
            view.width = 100; view.height = 100; view.pinhole = true;
        }
        cameras[0].center = {0, 0, 0}; // Sees the box, even without SfM tracks.
        cameras[11].observations = {{0, 0, 10}, {0.1f, 0, 10}, {0, 0.1f, 10}};
        const auto selected = lfs::core::select_tile_camera_uids(tile, cameras);
        EXPECT_NE(std::find(selected.begin(), selected.end(), 0), selected.end());
        EXPECT_NE(std::find(selected.begin(), selected.end(), 11), selected.end());
        EXPECT_LT(selected.size(), cameras.size());
        cameras[11].observations.resize(1);
        const auto without_track_support = lfs::core::select_tile_camera_uids(tile, cameras);
        EXPECT_EQ(std::find(without_track_support.begin(), without_track_support.end(), 11),
                  without_track_support.end());
    }

    TEST(ScenePartitioner, ProjectedBoxSelectsThinVisibleRegionBetweenRaySamples) {
        const lfs::core::TileSpec tile{{-1.25f, -6.0f, 9.0f},
                                       {-0.9f, 6.0f, 9.01f}, {}, 0, 0, "tile_0_0"};
        std::vector<lfs::core::TileCameraEvidence> cameras(12);
        for (int i = 0; i < 12; ++i) {
            auto& view = cameras[i];
            view.uid = i;
            view.center = {-1.075f, 0.0f, 8.0f};
            view.world_to_camera = {1, 0, 0, 0, 0, -1, 0, 0,
                                    0, 0, -1, 0, 0, 0, 0, 1};
            view.fx = view.fy = 80.0f;
            view.cx = view.cy = 50.0f;
            view.width = view.height = 100;
            view.pinhole = true;
        }
        cameras[0].center = {0.0f, 0.0f, 0.0f};
        cameras[0].world_to_camera = {1, 0, 0, 0, 0, 1, 0, 0,
                                      0, 0, 1, 0, 0, 0, 0, 1};
        const auto selected = lfs::core::select_tile_camera_uids(tile, cameras);
        EXPECT_NE(std::find(selected.begin(), selected.end(), 0), selected.end());
    }

    TEST(ScenePartitioner, AdaptiveTilesRejectUnsupportedCount) {
        const lfs::core::SceneBounds bounds{{0, 0, 0}, {10, 10, 2}};
        const auto identical = cloud({1, 1, 1, 1, 1, 1, 1, 1, 1});
        EXPECT_THROW(lfs::core::generate_adaptive_tiles(
                         bounds, 2, 0.15f, bounds, identical, {}),
                     std::runtime_error);
    }

    TEST(ScenePartitioner, AdaptiveTilesFallBackToCameraPositions) {
        const lfs::core::SceneBounds bounds{{0, 0, 0}, {10, 10, 2}};
        std::vector<lfs::core::TileCameraEvidence> cameras(2);
        cameras[0].center = {1, 1, 1};
        cameras[1].center = {9, 1, 1};
        const auto tiles = lfs::core::generate_adaptive_tiles(
            bounds, 2, 0.15f, bounds, cloud({1, 1, 1}), cameras);
        EXPECT_EQ(tiles.size(), 2);
    }

    TEST(ScenePartitioner, TileInitialPointsIncludeLocalAndCameraObservedGeometry) {
        std::vector<float> coordinates{0.0f, 0.0f, 0.0f};
        for (int i = 0; i < 300; ++i)
            coordinates.insert(coordinates.end(), {100.0f + i, 0.0f, 0.0f});
        const auto points = cloud(coordinates);
        lfs::core::TileSpec tile{{-1, -1, -1}, {1, 1, 1}, {}, 0, 0, "tile_0_0"};
        lfs::core::TileCameraEvidence view;
        view.observations = {{399.0f, 0.0f, 0.0f}};
        const auto selected = lfs::core::select_tile_initial_point_indices(points, tile, {view});
        EXPECT_NE(std::find(selected.begin(), selected.end(), 0), selected.end());
        EXPECT_NE(std::find(selected.begin(), selected.end(), 300), selected.end());
        EXPECT_EQ(std::find(selected.begin(), selected.end(), 299), selected.end());
        EXPECT_LT(selected.size(), static_cast<size_t>(points.size()));
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

    TEST(ScenePartitioner, ExtendsOnlyOuterFacesAndKeepsLocalZBounds) {
        const lfs::core::SceneBounds partition{{0, 0, -2}, {40, 30, 8}, 0};
        const lfs::core::SceneBounds outer{{-3, -100, -50}, {100, 36, 100}, 0};
        const auto tiles = lfs::core::generate_tile_grid(
            partition, {4, 3, 0.0f, 0.2f, false}, outer);
        ASSERT_EQ(tiles.size(), 12);
        // An outer percentile can still contain noise; expansion is capped at half a cell.
        EXPECT_NEAR(tiles[0].min.x, -3.0f, 1e-4f);
        EXPECT_NEAR(tiles[0].min.y, -5.0f, 1e-4f);
        EXPECT_NEAR(tiles[3].max.x, 45.0f, 1e-4f);
        EXPECT_NEAR(tiles[11].max.y, 35.0f, 1e-4f);
        EXPECT_NEAR(tiles[5].min.x, 9.0f, 1e-4f);
        EXPECT_NEAR(tiles[5].max.x, 21.0f, 1e-4f);
        EXPECT_NEAR(tiles[5].min.y, 9.0f, 1e-4f);
        EXPECT_NEAR(tiles[5].max.y, 21.0f, 1e-4f);
        for (const auto& tile : tiles) {
            EXPECT_NEAR(tile.min.z, -2.0f, 1e-4f);
            EXPECT_NEAR(tile.max.z, 8.0f, 1e-4f);
        }
    }

    TEST(ScenePartitioner, OuterBoundsIgnoreExtremePointAndKeepOrientation) {
        std::vector<float> points;
        for (int i = 0; i < 100; ++i)
            points.insert(points.end(), {float(i), float(i % 5), float(i % 3)});
        points.insert(points.end(), {100000.0f, 100000.0f, 100000.0f});
        const auto pc = cloud(points);
        const auto partition = lfs::core::compute_robust_scene_bounds_obb(
            pc, {}, 0.02f, lfs::core::OrientationMode::AxisAligned,
            lfs::core::BoundsSource::Points);
        const auto outer = lfs::core::compute_outer_scene_bounds(
            pc, {}, partition, 0.02f, lfs::core::BoundsSource::Points);
        EXPECT_LT(outer.min.x, partition.min.x);
        EXPECT_GT(outer.max.x, partition.max.x);
        EXPECT_LT(outer.max.x, 100.0f);
        EXPECT_NEAR(outer.rotation_degrees.z, partition.rotation_degrees.z, 1e-4f);
    }

    TEST(ScenePartitioner, BoundaryExtensionHasBaselineWithNoQuantileGap) {
        const lfs::core::SceneBounds bounds{{0, 0, 0}, {20, 20, 2}, 0};
        const auto tiles = lfs::core::generate_tile_grid(
            bounds, {2, 2, 0.0f, 0.0f, false}, bounds);
        ASSERT_EQ(tiles.size(), 4);
        EXPECT_NEAR(tiles[0].min.x, -1.0f, 1e-4f);
        EXPECT_NEAR(tiles[0].min.y, -1.0f, 1e-4f);
        EXPECT_NEAR(tiles[3].max.x, 21.0f, 1e-4f);
        EXPECT_NEAR(tiles[3].max.y, 21.0f, 1e-4f);
        EXPECT_NEAR(tiles[0].max.x, tiles[1].min.x, 1e-4f);
    }

    TEST(ScenePartitioner, BoundaryExtensionFollowsRotatedGridAxes) {
        const lfs::core::SceneBounds partition{{0, 0, -1}, {20, 10, 1}, 30};
        const lfs::core::SceneBounds outer{{-4, 0, -1}, {20, 10, 1}, 30};
        const auto tiles = lfs::core::generate_tile_grid(
            partition, {2, 1, 0.0f, 0.0f, true}, outer);
        ASSERT_EQ(tiles.size(), 2);
        const auto center = (tiles[0].min + tiles[0].max) * 0.5f;
        const glm::mat4 inverse_rotation = glm::eulerAngleXYZ(0.0f, 0.0f, glm::radians(-30.0f));
        const auto local_center = glm::vec3(inverse_rotation * glm::vec4(center, 1.0f));
        EXPECT_NEAR(local_center.x, 3.0f, 1e-4f);
        EXPECT_NEAR(local_center.y, 5.0f, 1e-4f);
        EXPECT_NEAR(tiles[0].max.x - tiles[0].min.x, 14.0f, 1e-4f);
        EXPECT_NEAR(tiles[0].rotation_degrees.z, 30.0f, 1e-4f);
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

/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/camera.hpp"
#include "io/formats/colmap.hpp"
#include "rendering/coordinate_conventions.hpp"
#include "rendering/rendering.hpp"
#include <algorithm>
#include <cuda_runtime.h>
#include <filesystem>
#include <glm/gtc/matrix_transform.hpp>
#include <gtest/gtest.h>
#include <tuple>
#include <vector>

namespace {
    using lfs::core::Camera;
    using lfs::rendering::CameraFrustumPickRequest;
    using lfs::rendering::RenderingEngine;

    class CameraFrustumPickTest : public ::testing::Test {
    protected:
        void SetUp() override {
            int devices = 0;
            if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
                GTEST_SKIP() << "requires CUDA";
            const auto model = std::filesystem::path(PROJECT_ROOT_PATH) / "data" / "bicycle" / "sparse" / "0";
            if (!std::filesystem::exists(model))
                GTEST_SKIP() << "missing data/bicycle/sparse/0";
            auto loaded = lfs::io::read_colmap_cameras_only(model);
            ASSERT_TRUE(loaded.has_value()) << loaded.error().format();
            owned_cameras = std::move(std::get<0>(*loaded));
            ASSERT_GT(owned_cameras.size(), 100u);
            cameras.assign(owned_cameras.begin(), owned_cameras.end());

            // The first camera becomes the origin, three units ahead of the viewer, so the rig fills the view.
            auto rotation = owned_cameras.front()->R().cpu();
            auto translation = owned_cameras.front()->T().cpu();
            auto r = rotation.accessor<float, 2>();
            auto t = translation.accessor<float, 1>();
            glm::mat4 world_to_camera(1.0f);
            for (int row = 0; row < 3; ++row) {
                for (int col = 0; col < 3; ++col)
                    world_to_camera[col][row] = r(row, col);
                world_to_camera[3][row] = t(row);
            }
            request.scene_transform =
                glm::inverse(glm::inverse(world_to_camera) * lfs::rendering::DATA_TO_VISUALIZER_CAMERA_AXES_4);
            request.viewport_size = {1600.0f, 1000.0f};
            request.viewport.size = {1600, 1000};
            request.viewport.rotation = glm::mat3(1.0f);
            request.viewport.translation = {0.0f, 0.0f, 3.0f};
            request.scale = 0.4f;
        }

        std::vector<int> pickGrid(RenderingEngine& engine) {
            std::vector<int> picks;
            for (float y = 0.0f; y < request.viewport_size.y; y += 16.0f) {
                for (float x = 0.0f; x < request.viewport_size.x; x += 16.0f) {
                    request.mouse_pos = {x, y};
                    const auto picked = engine.pickCameraFrustum(cameras, request);
                    EXPECT_TRUE(picked.has_value());
                    picks.push_back(picked.value_or(-2));
                }
            }
            return picks;
        }

        std::vector<int> uncachedPickGrid() {
            const auto engine = RenderingEngine::create();
            return pickGrid(*engine);
        }

        std::vector<std::shared_ptr<Camera>> owned_cameras;
        std::vector<std::shared_ptr<const Camera>> cameras;
        CameraFrustumPickRequest request;
    };

    // A frustum cache that misses any of these inputs returns the picks of the previous state.
    TEST_F(CameraFrustumPickTest, CachedFrustumsFollowEveryInputChange) {
        const auto engine = RenderingEngine::create();
        const auto initial = pickGrid(*engine);
        EXPECT_GT(std::count_if(initial.begin(), initial.end(), [](const int uid) { return uid >= 0; }), 100);
        EXPECT_EQ(pickGrid(*engine), initial);

        const auto expect_changed_and_uncached = [&] {
            const auto picks = pickGrid(*engine);
            EXPECT_NE(picks, initial);
            EXPECT_EQ(picks, uncachedPickGrid());
        };

        request.scale = 0.6f;
        expect_changed_and_uncached();
        request.scale = 0.4f;

        std::vector<glm::mat4> moved(cameras.size(), request.scene_transform);
        for (size_t i = 0; i < moved.size(); i += 2)
            moved[i] = glm::translate(glm::mat4(1.0f), glm::vec3(0.3f, 0.0f, 0.0f)) * moved[i];
        request.scene_transforms = moved;
        expect_changed_and_uncached();
        request.scene_transforms = {};

        cameras.resize(cameras.size() / 2);
        expect_changed_and_uncached();
        cameras.assign(owned_cameras.begin(), owned_cameras.end());

        // Undistortion rewrites the calibration of the scene's own camera objects.
        for (const auto& camera : owned_cameras) {
            lfs::core::UndistortParams params{};
            params.dst_fx = camera->focal_x();
            params.dst_fy = camera->focal_y();
            params.dst_cx = camera->center_x() * 0.5f;
            params.dst_cy = camera->center_y() * 0.5f;
            params.dst_width = camera->camera_width() / 2;
            params.dst_height = camera->camera_height() / 2;
            camera->adopt_undistortion(params);
            camera->prepare_undistortion();
        }
        expect_changed_and_uncached();
    }

} // namespace

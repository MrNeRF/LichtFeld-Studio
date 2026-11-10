/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "gui/camera_selection_algorithms.hpp"
#include <gtest/gtest.h>
#include <random>

namespace selection = lfs::vis::gui::camera_selection;
namespace {
    struct Camera {
        bool pose_valid = true;
        glm::vec3 center{0.0F};
        glm::vec3 forward{0.0F, 0.0F, 1.0F};
        float fov = 0.8F;
        glm::mat4 world_to_camera{1.0F};
        float focal_x = 50.0F, focal_y = 50.0F;
        float center_x = 50.0F, center_y = 50.0F;
        float width = 100.0F, height = 100.0F;
    };

    std::vector<glm::vec3> corners(const glm::vec3 min, const glm::vec3 max) {
        std::vector<glm::vec3> points;
        for (float x : {min.x, max.x})
            for (float y : {min.y, max.y})
                for (float z : {min.z, max.z})
                    points.emplace_back(x, y, z);
        return points;
    }
} // namespace

TEST(CameraSelectionAlgorithms, EveryNIncludesFirstAndClampsStep) {
    const auto active = [] { return false; };
    EXPECT_EQ(selection::every_n(8, 3, active), (std::vector<size_t>{0, 3, 6}));
    EXPECT_EQ(selection::every_n(3, 0, active), (std::vector<size_t>{0, 1, 2}));
    EXPECT_TRUE(selection::every_n(0, 2, active).empty());
}

TEST(CameraSelectionAlgorithms, RandomRoundsAndSelectsUniqueIndices) {
    std::mt19937 rng(42);
    auto indices = selection::random(10, 35.0F, rng);
    ASSERT_EQ(indices.size(), 4u);
    std::sort(indices.begin(), indices.end());
    EXPECT_EQ(std::unique(indices.begin(), indices.end()), indices.end());
    EXPECT_LT(indices.back(), 10u);
    EXPECT_TRUE(selection::random(10, -1.0F, rng).empty());
    EXPECT_EQ(selection::random(10, 101.0F, rng).size(), 10u);
}

TEST(CameraSelectionAlgorithms, DistinctSuppressesDuplicateViewsAndInvalidPoses) {
    std::vector<Camera> cameras(4);
    cameras[2].center.x = 10.0F;
    cameras[3].pose_valid = false;
    const auto active = [] { return false; };
    const auto report = [](float) {};
    EXPECT_EQ(selection::distinct(cameras, 100.0F, active, report), (std::vector<size_t>{0, 2}));
    EXPECT_EQ(selection::distinct(cameras, 0.0F, active, report), (std::vector<size_t>{0}));
}

TEST(CameraSelectionAlgorithms, CropCoverageHandlesContainingAndClippedBoxes) {
    const Camera camera;
    const glm::mat4 identity(1.0F);
    const glm::vec3 contain_min(-1.0F), contain_max(1.0F);
    EXPECT_FLOAT_EQ(selection::crop_coverage(camera, corners(contain_min, contain_max), identity, contain_min, contain_max), 100.0F);
    const glm::vec3 front_min(-0.5F, -0.5F, 2.0F), front_max(0.5F, 0.5F, 3.0F);
    EXPECT_NEAR(selection::crop_coverage(camera, corners(front_min, front_max), identity, front_min, front_max), 6.25F, 0.01F);
    const glm::vec3 clip_min(0.25F, -0.25F, -1.0F), clip_max(1.0F, 0.25F, 2.0F);
    EXPECT_NEAR(selection::crop_coverage(camera, corners(clip_min, clip_max), identity, clip_min, clip_max), 43.75F, 0.01F);
    const glm::vec3 behind_min(-1.0F, -1.0F, -3.0F), behind_max(1.0F, 1.0F, -1.0F);
    EXPECT_FLOAT_EQ(selection::crop_coverage(camera, corners(behind_min, behind_max), identity, behind_min, behind_max), 0.0F);
}

TEST(CameraSelectionAlgorithms, CancellationDoesNotPublishPartialSelection) {
    std::vector<size_t> installed{1, 3};
    int checks = 0;
    auto partial = selection::every_n(10, 1, [&] { return ++checks > 3; });
    ASSERT_EQ(partial.size(), 3u);
    EXPECT_FALSE(selection::apply_if_ready(true, false, [&] { installed = partial; }));
    EXPECT_EQ(installed, (std::vector<size_t>{1, 3}));
    EXPECT_TRUE(selection::apply_if_ready(false, false, [&] { installed = partial; }));
    EXPECT_EQ(installed, partial);
}

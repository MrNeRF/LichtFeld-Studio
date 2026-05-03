/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "app/include/app/view_info_json.hpp"
#include "rendering/coordinate_conventions.hpp"
#include "visualizer/ipc/view_context.hpp"

#include <glm/glm.hpp>
#include <gtest/gtest.h>

#include <cmath>

namespace {

    using lfs::app::ortho_view_extent_world;
    using lfs::app::view_info_json;
    using lfs::vis::ViewInfo;

    ViewInfo make_identity_view(const int width = 1280, const int height = 720, const float fov = 60.0f) {
        return ViewInfo{
            .rotation = {1.0f, 0.0f, 0.0f,
                         0.0f, 1.0f, 0.0f,
                         0.0f, 0.0f, 1.0f},
            .translation = {0.0f, 0.0f, 5.0f},
            .pivot = {0.0f, 0.0f, 0.0f},
            .width = width,
            .height = height,
            .fov = fov,
        };
    }

    TEST(ViewInfoJson, DefaultsToPerspectiveAndZeroExtent) {
        const auto info = make_identity_view();
        const auto js = view_info_json(info);

        ASSERT_TRUE(js.contains("camera"));
        const auto& cam = js["camera"];
        EXPECT_FALSE(cam["orthographic"].get<bool>());
        EXPECT_FLOAT_EQ(cam["ortho_scale"].get<float>(), 100.0f);
        EXPECT_FLOAT_EQ(cam["ortho_view_extent_world"].get<float>(), 0.0f)
            << "Perspective views must report zero ortho extent so consumers don't misuse the value";
    }

    TEST(ViewInfoJson, OrthographicExposesScaleAndExtent) {
        auto info = make_identity_view(/*width=*/800, /*height=*/600);
        info.orthographic = true;
        info.ortho_scale = 200.0f;

        const auto js = view_info_json(info);
        const auto& cam = js["camera"];
        EXPECT_TRUE(cam["orthographic"].get<bool>());
        EXPECT_FLOAT_EQ(cam["ortho_scale"].get<float>(), 200.0f);
        EXPECT_FLOAT_EQ(cam["ortho_view_extent_world"].get<float>(), 600.0f / 200.0f);
    }

    TEST(ViewInfoJson, OrthoExtentInvertsScaleSemanticForBlenderCompat) {
        auto zoomed_in = make_identity_view(/*width=*/1000, /*height=*/1000);
        zoomed_in.orthographic = true;
        zoomed_in.ortho_scale = 500.0f;

        auto zoomed_out = zoomed_in;
        zoomed_out.ortho_scale = 50.0f;

        EXPECT_GT(zoomed_in.ortho_scale, zoomed_out.ortho_scale)
            << "Internal ortho_scale grows when zooming in (pixels per world unit)";
        EXPECT_LT(ortho_view_extent_world(zoomed_in), ortho_view_extent_world(zoomed_out))
            << "Blender-style ortho_view_extent_world must shrink when zooming in";
    }

    TEST(ViewInfoJson, OrthoExtentMatchesProjectionGeometry) {
        const int width = 1280;
        const int height = 720;
        ViewInfo info = make_identity_view(width, height);
        info.orthographic = true;
        info.ortho_scale = 50.0f;

        const auto rotation_array = info.rotation;
        const glm::mat3 R(
            glm::vec3(rotation_array[0], rotation_array[3], rotation_array[6]),
            glm::vec3(rotation_array[1], rotation_array[4], rotation_array[7]),
            glm::vec3(rotation_array[2], rotation_array[5], rotation_array[8]));
        const glm::vec3 t(info.translation[0], info.translation[1], info.translation[2]);

        const float extent = ortho_view_extent_world(info);
        ASSERT_GT(extent, 0.0f);
        const float half_height_world = extent * 0.5f;

        const glm::vec3 top_world = t + R * glm::vec3(0.0f, half_height_world, -1.0f);
        const auto projected = lfs::rendering::projectWorldPoint(
            R, t, glm::ivec2(width, height), top_world,
            /*focal_length_mm=*/35.0f, /*orthographic=*/true, /*ortho_scale=*/info.ortho_scale);

        ASSERT_TRUE(projected.has_value());
        const float cy = static_cast<float>(height) * 0.5f;
        EXPECT_NEAR(projected->y, cy - static_cast<float>(height) * 0.5f, 1e-2f)
            << "World point at half_height must project to the top edge of the viewport";
    }

    TEST(ViewInfoJson, ZeroOrthoScaleProducesZeroExtent) {
        ViewInfo info = make_identity_view();
        info.orthographic = true;
        info.ortho_scale = 0.0f;

        const auto js = view_info_json(info);
        EXPECT_FLOAT_EQ(js["camera"]["ortho_view_extent_world"].get<float>(), 0.0f);
    }

    TEST(ViewInfoJson, ContainsAllExpectedCameraFields) {
        const auto js = view_info_json(make_identity_view());
        const auto& cam = js["camera"];

        for (const char* key : {"eye", "target", "pivot", "up", "forward",
                                "rotation_matrix", "width", "height",
                                "fov_degrees", "orthographic", "ortho_scale",
                                "ortho_view_extent_world"}) {
            EXPECT_TRUE(cam.contains(key)) << "missing camera field: " << key;
        }
    }

} // namespace

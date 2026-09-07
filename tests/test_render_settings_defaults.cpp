/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "visualizer/ipc/render_settings_convert.hpp"
#include "visualizer/ipc/view_context.hpp"
#include "visualizer/project/session_state.hpp"
#include "visualizer/rendering/rendering_manager.hpp"
#include "visualizer/rendering/rendering_types.hpp"

#include <glm/gtc/quaternion.hpp>
#include <gtest/gtest.h>

TEST(RenderSettingsDefaults, CameraFrustumsAreDisabledByDefault) {
    const lfs::vis::RenderSettings render_settings;
    const lfs::vis::RenderSettingsProxy proxy_settings;

    EXPECT_FALSE(render_settings.show_camera_frustums);
    EXPECT_FALSE(proxy_settings.show_camera_frustums);
    EXPECT_FLOAT_EQ(render_settings.camera_frustum_scale, 0.25f);
    EXPECT_FLOAT_EQ(proxy_settings.camera_frustum_scale, 0.25f);
}

TEST(RenderSettingsDefaults, GTComparisonUsesFitByDefault) {
    const lfs::vis::RenderSettings render_settings;
    const lfs::vis::RenderSettingsProxy proxy_settings;

    EXPECT_FALSE(render_settings.gt_comparison_actual_size);
    EXPECT_FALSE(proxy_settings.gt_comparison_actual_size);
}

TEST(RenderSettingsProxy, GTComparisonActualSizeRoundTrips) {
    lfs::vis::RenderSettings settings;
    settings.gt_comparison_actual_size = true;

    const auto proxy = lfs::vis::to_proxy(settings);
    EXPECT_TRUE(proxy.gt_comparison_actual_size);

    lfs::vis::RenderSettings roundtrip;
    lfs::vis::apply_proxy(roundtrip, proxy);
    EXPECT_TRUE(roundtrip.gt_comparison_actual_size);
}

TEST(RenderSettingsSanitization, NonRgbGTComparisonClearsActualSize) {
    lfs::vis::RenderSettings settings;
    settings.gt_comparison_mode = lfs::vis::GTComparisonMode::Normal;
    settings.gt_comparison_actual_size = true;

    lfs::vis::sanitizeGTComparisonSettings(settings);

    EXPECT_FALSE(settings.gt_comparison_actual_size);
}

TEST(RenderSettingsSession, GTComparisonActualSizeIsNotSaved) {
    lfs::vis::RenderSettings settings;
    settings.gt_comparison_actual_size = true;

    const auto json = lfs::vis::project::renderSettingsToProjectJson(settings);
    EXPECT_FALSE(json.contains("gt_comparison_actual_size"));
    EXPECT_TRUE(settings.gt_comparison_actual_size);

    const lfs::io::project::ViewSessionChapter view;
    EXPECT_FALSE(view.dom().get_json("render_settings.gt_comparison_actual_size"));
}

TEST(RenderSettingsSession, GTComparisonAlwaysLoadsInFit) {
    lfs::vis::RenderSettings active_settings;
    active_settings.gt_comparison_actual_size = true;

    for (const auto mode : {lfs::vis::GTComparisonMode::RGB,
                            lfs::vis::GTComparisonMode::Depth,
                            lfs::vis::GTComparisonMode::Normal}) {
        active_settings.gt_comparison_mode = mode;
        const auto json = lfs::vis::project::renderSettingsToProjectJson(active_settings);
        const auto restored =
            lfs::vis::project::renderSettingsFromProjectJson(json, active_settings);
        ASSERT_TRUE(restored);
        EXPECT_FALSE(restored->gt_comparison_actual_size);
        EXPECT_EQ(restored->gt_comparison_mode, mode);
    }
}

TEST(RenderSettingsDefaults, SceneReconstructionIsDisabledByDefault) {
    const lfs::vis::RenderSettings render_settings;
    const lfs::vis::RenderSettingsProxy proxy_settings;

    EXPECT_EQ(render_settings.scene_upscaler, "native");
    EXPECT_EQ(render_settings.scene_upscaler_preset, "native");
    EXPECT_FLOAT_EQ(render_settings.scene_upscaler_scale, 1.0f);
    EXPECT_EQ(proxy_settings.scene_upscaler, "native");
    EXPECT_EQ(proxy_settings.scene_upscaler_preset, "native");
    EXPECT_FLOAT_EQ(proxy_settings.scene_upscaler_scale, 1.0f);
}

TEST(RenderSettingsDefaults, SceneReconstructionScaleComposesWithBaseScale) {
    EXPECT_FLOAT_EQ(lfs::vis::effectiveSceneRenderScale(0.8f, 0.5f, true), 0.4f);
    EXPECT_FLOAT_EQ(lfs::vis::effectiveSceneRenderScale(0.8f, 0.5f, false), 0.8f);
    EXPECT_FLOAT_EQ(lfs::vis::effectiveSceneRenderScale(1.0f, 0.1f, true), 0.25f);
}

TEST(RenderSettingsProxy, DepthFilterTransformRoundTrips) {
    lfs::vis::RenderSettings render_settings;
    const glm::quat rotation = glm::normalize(glm::quat(0.9f, 0.1f, -0.2f, 0.3f));
    const glm::vec3 translation{1.25f, -2.5f, 3.75f};

    render_settings.depth_filter_enabled = true;
    render_settings.depth_filter_transform = lfs::geometry::EuclideanTransform(rotation, translation);
    render_settings.depth_filter_min = {-4.0f, -5.0f, -6.0f};
    render_settings.depth_filter_max = {4.0f, 5.0f, 6.0f};

    const auto proxy = lfs::vis::to_proxy(render_settings);

    lfs::vis::RenderSettings roundtrip;
    lfs::vis::apply_proxy(roundtrip, proxy);

    const glm::quat roundtrip_rotation = roundtrip.depth_filter_transform.getRotation();
    const glm::vec3 roundtrip_translation = roundtrip.depth_filter_transform.getTranslation();

    EXPECT_TRUE(roundtrip.depth_filter_enabled);
    EXPECT_EQ(roundtrip.depth_filter_min, render_settings.depth_filter_min);
    EXPECT_EQ(roundtrip.depth_filter_max, render_settings.depth_filter_max);
    EXPECT_FLOAT_EQ(roundtrip_rotation.w, rotation.w);
    EXPECT_FLOAT_EQ(roundtrip_rotation.x, rotation.x);
    EXPECT_FLOAT_EQ(roundtrip_rotation.y, rotation.y);
    EXPECT_FLOAT_EQ(roundtrip_rotation.z, rotation.z);
    EXPECT_FLOAT_EQ(roundtrip_translation.x, translation.x);
    EXPECT_FLOAT_EQ(roundtrip_translation.y, translation.y);
    EXPECT_FLOAT_EQ(roundtrip_translation.z, translation.z);
}

TEST(RenderSettingsProxy, SceneReconstructionFieldsRoundTrip) {
    lfs::vis::RenderSettings settings;
    settings.scene_upscaler = "spatial";
    settings.scene_upscaler_preset = "performance";
    settings.scene_upscaler_scale = 0.5f;

    const auto proxy = lfs::vis::to_proxy(settings);
    EXPECT_EQ(proxy.scene_upscaler, "spatial");
    EXPECT_EQ(proxy.scene_upscaler_preset, "performance");
    EXPECT_FLOAT_EQ(proxy.scene_upscaler_scale, 0.5f);

    lfs::vis::RenderSettings roundtrip;
    lfs::vis::apply_proxy(roundtrip, proxy);
    EXPECT_EQ(roundtrip.scene_upscaler, "spatial");
    EXPECT_EQ(roundtrip.scene_upscaler_preset, "performance");
    EXPECT_FLOAT_EQ(roundtrip.scene_upscaler_scale, 0.5f);
}

TEST(RenderSettingsProxy, DepthViewSplitOffsetAndLodFieldsRoundTrip) {
    lfs::vis::RenderSettings settings;
    settings.depth_view = true;
    settings.split_view_offset = 3;
    settings.lod_auto_enable_rad = true;
    settings.lod_behind_camera_penalty = 0.55f;

    const auto proxy = lfs::vis::to_proxy(settings);
    EXPECT_TRUE(proxy.depth_view);
    EXPECT_EQ(proxy.split_view_offset, 3u);
    EXPECT_TRUE(proxy.lod_auto_enable_rad);
    EXPECT_FLOAT_EQ(proxy.lod_behind_camera_penalty, 0.55f);

    lfs::vis::RenderSettings roundtrip;
    lfs::vis::apply_proxy(roundtrip, proxy);
    EXPECT_TRUE(roundtrip.depth_view);
    EXPECT_EQ(roundtrip.split_view_offset, 3u);
    EXPECT_TRUE(roundtrip.lod_auto_enable_rad);
    EXPECT_FLOAT_EQ(roundtrip.lod_behind_camera_penalty, 0.55f);
}

TEST(RenderSettingsBackendNormalization, Explicit3dgsBackendBeatsStaleGutMirror) {
    using Backend = lfs::rendering::GaussianRasterBackend;

    EXPECT_EQ(lfs::rendering::normalizeViewerRasterBackend(Backend::ThreeDgs, true),
              Backend::ThreeDgs);
    EXPECT_EQ(lfs::rendering::normalizeViewerRasterBackend(Backend::ThreeDgut, false),
              Backend::ThreeDgut);
    EXPECT_EQ(lfs::rendering::gaussianRasterBackendId(Backend::ThreeDgs), "3dgs");
    EXPECT_EQ(lfs::rendering::gaussianRasterBackendId(Backend::ThreeDgut), "3dgut");
    EXPECT_EQ(lfs::rendering::gaussianRasterBackendFromId("3dgs"), Backend::ThreeDgs);
    EXPECT_EQ(lfs::rendering::gaussianRasterBackendFromId("3dgut"), Backend::ThreeDgut);
}

TEST(RenderSettingsProxy, GutMirrorStillSwitchesViewerBackend) {
    using Backend = lfs::rendering::GaussianRasterBackend;

    lfs::vis::RenderSettings settings;
    settings.raster_backend = Backend::ThreeDgs;
    settings.gut = false;

    auto proxy = lfs::vis::to_proxy(settings);
    proxy.gut = true;
    lfs::vis::apply_proxy(settings, proxy);

    EXPECT_EQ(settings.raster_backend, Backend::ThreeDgut);
    EXPECT_TRUE(settings.gut);

    proxy = lfs::vis::to_proxy(settings);
    proxy.gut = false;
    lfs::vis::apply_proxy(settings, proxy);

    EXPECT_EQ(settings.raster_backend, Backend::ThreeDgs);
    EXPECT_FALSE(settings.gut);
}

TEST(RenderSettingsProxy, EquirectangularForcesGutBackend) {
    using Backend = lfs::rendering::GaussianRasterBackend;

    lfs::vis::RenderSettings settings;
    settings.raster_backend = Backend::ThreeDgs;
    settings.gut = false;
    settings.equirectangular = false;

    auto proxy = lfs::vis::to_proxy(settings);
    proxy.equirectangular = true;
    lfs::vis::apply_proxy(settings, proxy);

    EXPECT_TRUE(settings.equirectangular);
    EXPECT_EQ(settings.raster_backend, Backend::ThreeDgut);
    EXPECT_TRUE(settings.gut);
}

TEST(RenderSettingsBackendNormalization, RenderingManagerCanSwitchBackFromGutTo3dgs) {
    using Backend = lfs::rendering::GaussianRasterBackend;

    lfs::vis::RenderingManager manager;
    auto settings = manager.getSettings();
    settings.raster_backend = Backend::ThreeDgut;
    settings.gut = true;
    manager.updateSettings(settings);

    settings = manager.getSettings();
    ASSERT_EQ(settings.raster_backend, Backend::ThreeDgut);
    ASSERT_TRUE(settings.gut);

    settings.raster_backend = Backend::ThreeDgs;
    settings.gut = true;
    manager.updateSettings(settings);

    settings = manager.getSettings();
    EXPECT_EQ(settings.raster_backend, Backend::ThreeDgs);
    EXPECT_FALSE(settings.gut);
}

TEST(RenderSettingsBackendNormalization, RenderingManagerEquirectangularUpdateForcesGutBackend) {
    using Backend = lfs::rendering::GaussianRasterBackend;

    lfs::vis::RenderingManager manager;
    auto settings = manager.getSettings();
    settings.raster_backend = Backend::ThreeDgs;
    settings.gut = false;
    settings.equirectangular = true;
    manager.updateSettings(settings);

    settings = manager.getSettings();
    EXPECT_TRUE(settings.equirectangular);
    EXPECT_EQ(settings.raster_backend, Backend::ThreeDgut);
    EXPECT_TRUE(settings.gut);
}

TEST(RenderSettingsBackendNormalization, RenderingManagerKeepsGutToggleWorking) {
    using Backend = lfs::rendering::GaussianRasterBackend;

    lfs::vis::RenderingManager manager;
    auto settings = manager.getSettings();
    ASSERT_EQ(settings.raster_backend, Backend::ThreeDgs);
    ASSERT_FALSE(settings.gut);

    settings.gut = true;
    manager.updateSettings(settings);

    settings = manager.getSettings();
    EXPECT_EQ(settings.raster_backend, Backend::ThreeDgut);
    EXPECT_TRUE(settings.gut);

    settings.gut = false;
    manager.updateSettings(settings);

    settings = manager.getSettings();
    EXPECT_EQ(settings.raster_backend, Backend::ThreeDgs);
    EXPECT_FALSE(settings.gut);
}

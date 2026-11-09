/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/event_bridge/event_bridge.hpp"
#include "ipc/render_settings_convert.hpp"
#include "project/session_state.hpp"
#include "rendering/rendering_manager.hpp"
#include "rendering/viewport_request_builder.hpp"
#include "scene/scene_manager.hpp"
#include <gtest/gtest.h>

namespace lfs::vis {
    TEST(ViewportRenderModes, PersistsModeButNotSelectorVisibilityAndLoadsOlderProjects) {
        RenderSettings settings;
        settings.normal_view = true;
        settings.render_mode_controls_open = true;
        RenderSettings copy;
        apply_proxy(copy, to_proxy(settings));
        EXPECT_TRUE(copy.normal_view);
        auto json = project::renderSettingsToProjectJson(settings);
        EXPECT_FALSE(json.contains("render_mode_controls_open"));
        auto restored = project::renderSettingsFromProjectJson(json);
        ASSERT_TRUE(restored);
        EXPECT_TRUE(restored->normal_view);
        EXPECT_FALSE(restored->render_mode_controls_open);
        json.erase("normal_view");
        restored = project::renderSettingsFromProjectJson(json);
        ASSERT_TRUE(restored);
        EXPECT_FALSE(restored->normal_view);
    }

    TEST(ViewportRenderModes, NormalsRequireWholeFrameExactDepthAndSkipColorCompositing) {
        struct EventGuard {
            ~EventGuard() { lfs::event::EventBridge::instance().clear_all(); }
        } guard;
        Viewport viewport(640, 480);
        SceneManager manager;
        const ViewportRegion region{0, 0, 640, 480};
        FrameContext frame{.viewport = viewport, .viewport_region = &region, .scene_manager = &manager, .settings = {}, .render_size = {640, 480}};
        frame.settings.normal_view = true;
        frame.settings.show_camera_frustums = true;
        frame.settings.environment_mode = EnvironmentBackgroundMode::Equirectangular;
        frame.settings.environment_map_path = "test.hdr";
        const auto request = buildViewportRenderRequest(frame, {640, 480});
        EXPECT_TRUE(request.normal_view);
        EXPECT_TRUE(request.require_exact_depth);
        EXPECT_TRUE(request.exact_depth_sample_mask.empty());
        EXPECT_FALSE(request.transparent_background);
        EXPECT_FALSE(RenderingManager::vrRenderSettings(frame.settings).normal_view);
        frame.settings.depth_view = true;
        EXPECT_FALSE(normalViewEnabled(frame.settings));
        frame.settings.depth_view = false;
        frame.settings.split_view_mode = SplitViewMode::GTComparison;
        EXPECT_FALSE(normalViewEnabled(frame.settings));
    }
} // namespace lfs::vis

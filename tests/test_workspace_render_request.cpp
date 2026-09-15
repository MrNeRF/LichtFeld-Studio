/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "rendering/workspace_render_request.hpp"
#include "workspace/pane_interaction.hpp"

#include <cmath>
#include <gtest/gtest.h>

namespace lfs::vis {
    namespace {

        TEST(WorkspaceRenderRequest, IndependentCameraOverridesGlobalAndComparisonState) {
            Viewport legacy(800, 600);
            FrameContext shared{.viewport = legacy};
            shared.settings.orthographic = false;
            shared.settings.focal_length_mm = 20.0f;
            shared.settings.split_view_mode = SplitViewMode::GTComparison;
            shared.scene_state.model_transforms = {glm::mat4(1.0f)};
            shared.scene_state.node_active_sh_degrees = {2};
            shared.scene_state.node_visibility_mask = {true};
            PaneSnapshot a{.id = 11, .rect = {20, 40, 300, 200}};
            a.translation = {1.0f, 2.0f, 3.0f};
            a.projection.focal_length_mm = 70.0f;
            a.projection.near_plane = 0.4f;
            a.projection.far_plane = 220.0f;
            PaneSnapshot b = a;
            b.id = 12;
            b.translation = {-20.0f, 4.0f, 6.0f};
            b.projection.orthographic = true;
            b.projection.ortho_scale = 17.0f;

            const auto request_a = buildWorkspaceRenderRequest(shared, a, {600, 400});
            const auto request_b = buildWorkspaceRenderRequest(shared, b, {360, 240});
            EXPECT_EQ(request_a.frame_view.translation, a.translation);
            EXPECT_EQ(request_b.frame_view.translation, b.translation);
            EXPECT_FALSE(request_a.frame_view.orthographic);
            EXPECT_TRUE(request_b.frame_view.orthographic);
            EXPECT_FLOAT_EQ(request_a.frame_view.focal_length_mm, 70.0f);
            EXPECT_FLOAT_EQ(request_b.frame_view.ortho_scale, 17.0f);
            EXPECT_FLOAT_EQ(request_a.frame_view.near_plane, 0.4f);
            EXPECT_FLOAT_EQ(request_a.frame_view.far_plane, 220.0f);
            EXPECT_EQ(request_a.frame_view.size, glm::ivec2(600, 400));
            EXPECT_EQ(request_b.frame_view.size, glm::ivec2(360, 240));
            // Borrow the shared frame's storage, not a destroyed temporary's vector.
            EXPECT_EQ(request_a.scene.model_transforms, &shared.scene_state.model_transforms);
            EXPECT_EQ(request_b.scene.model_transforms, request_a.scene.model_transforms);
            EXPECT_EQ(request_b.scene.node_active_sh_degrees, std::vector<int>{2});
            EXPECT_EQ(request_b.scene.node_visibility_mask, std::vector<bool>{true});
            EXPECT_EQ(shared.settings.split_view_mode, SplitViewMode::GTComparison);
            EXPECT_FALSE(shared.settings.orthographic);
            EXPECT_FLOAT_EQ(shared.settings.focal_length_mm, 20.0f);
        }

        TEST(WorkspaceRenderRequest, OrthographicResizeResolutionPreservesProjectedPosition) {
            Viewport legacy(800, 600);
            FrameContext shared{.viewport = legacy};
            PaneSnapshot pane{.id = 1};
            pane.window_size = {800, 600};
            pane.framebuffer_size = {800, 600};
            pane.projection.orthographic = true;
            pane.projection.ortho_scale = 130.0f;
            const glm::vec4 point(0.5f, 0.25f, -2.0f, 1.0f);
            const auto full = makeWorkspaceFrameView(pane, {800, 600}, {});
            const auto reference = full.getProjectionMatrix() * point;
            for (const auto extent : {glm::ivec2(264, 198), glm::ivec2(400, 300),
                                      glm::ivec2(800, 600), glm::ivec2(1600, 1200)}) {
                const auto request = buildWorkspaceRenderRequest(shared, pane, extent);
                const auto projected = request.frame_view.getProjectionMatrix() * point;
                EXPECT_NEAR(projected.x, reference.x, 1.0e-6f);
                EXPECT_NEAR(projected.y, reference.y, 1.0e-6f);
                const auto intrinsics = request.frame_view.getCameraIntrinsics();
                EXPECT_NEAR(intrinsics.focal_x / extent.x, 130.0f / 800.0f, 1.0e-6f);
                EXPECT_NEAR(intrinsics.focal_y / extent.y, 130.0f / 600.0f, 1.0e-6f);
            }
            // Resizing the logical area still exposes more world space at the
            // same zoom; only changing raster resolution preserves the bounds.
            pane.window_size = {1000, 750};
            const auto resized = makeWorkspaceFrameView(pane, {1000, 750}, {});
            EXPECT_FLOAT_EQ(resized.ortho_scale, 130.0f);
            EXPECT_LT((resized.getProjectionMatrix() * point).x, reference.x);
        }

        TEST(WorkspaceRenderRequest, LodPixelScaleLimitUsesWorldUnitsPerPixel) {
            lfs::rendering::FrameView perspective;
            perspective.size = {800, 400};
            perspective.focal_length_mm = 35.0f;
            const float perspective_scale = lodPixelScaleLimit(perspective);
            EXPECT_GT(perspective_scale, 0.0f);
            perspective.size.y *= 2;
            EXPECT_FLOAT_EQ(lodPixelScaleLimit(perspective), perspective_scale * 0.5f);

            lfs::rendering::FrameView orthographic = perspective;
            orthographic.orthographic = true;
            orthographic.ortho_scale = 80.0f;
            const float ortho_pixel = lodPixelScaleLimit(orthographic);
            EXPECT_FLOAT_EQ(ortho_pixel, 0.0125f);
            // Resizing exposes more world area but does not change detail at a
            // fixed pixels-per-world-unit zoom. Zooming in resolves finer nodes.
            orthographic.size.y *= 2;
            EXPECT_FLOAT_EQ(lodPixelScaleLimit(orthographic), ortho_pixel);
            orthographic.ortho_scale *= 2;
            EXPECT_FLOAT_EQ(lodPixelScaleLimit(orthographic), ortho_pixel * 0.5f);
        }

        TEST(WorkspaceRenderRequest, DepthWindowsAndTransientSelectionBelongToView) {
            Viewport legacy(800, 600);
            FrameContext shared{.viewport = legacy};
            shared.settings.depth_filter_enabled = true;
            shared.settings.split_view_mode = SplitViewMode::GTComparison;
            shared.cursor_preview.active = true;
            shared.cursor_preview.x = 15.0f;
            shared.cursor_preview.y = 25.0f;
            shared.cursor_preview.radius = 8.0f;
            shared.cursor_preview.panel = SplitViewPanelId::Right;
            shared.depth_window_drag_preview = true;
            shared.scene_state.selection_mask = std::make_shared<lfs::core::Tensor>();
            shared.scene_state.has_selection = true;
            PaneSnapshot a{.id = 1};
            a.depth = {.near_plane = 2.0f, .far_plane = 8.0f, .scale_x = 0.2f, .scale_y = 0.3f, .offset_x = 0.1f};
            PaneSnapshot b{.id = 2};
            b.depth = {.near_plane = 10.0f, .far_plane = 30.0f, .scale_x = 0.6f, .scale_y = 0.7f, .offset_y = -0.2f};
            const auto request_a = buildWorkspaceRenderRequest(shared, a, {300, 200}, {}, a.id);
            const auto request_b = buildWorkspaceRenderRequest(shared, b, {300, 200}, {}, a.id);
            ASSERT_TRUE(request_a.filters.view_volume);
            ASSERT_TRUE(request_b.filters.view_volume);
            EXPECT_FLOAT_EQ(request_a.filters.view_volume->min.z, -8.0f);
            EXPECT_FLOAT_EQ(request_a.filters.view_volume->max.z, -2.0f);
            EXPECT_FLOAT_EQ(request_b.filters.view_volume->min.z, -30.0f);
            EXPECT_FLOAT_EQ(request_b.filters.view_volume->max.z, -10.0f);
            ASSERT_TRUE(request_a.filters.screen_window);
            ASSERT_TRUE(request_b.filters.screen_window);
            EXPECT_FLOAT_EQ(request_a.filters.screen_window->scale_x, 0.2f);
            EXPECT_FLOAT_EQ(request_b.filters.screen_window->scale_x, 0.6f);
            EXPECT_TRUE(request_a.filters.screen_window->drag_preview);
            EXPECT_FALSE(request_b.filters.screen_window->drag_preview);
            EXPECT_TRUE(request_a.overlay.cursor.enabled);
            EXPECT_FALSE(request_b.overlay.cursor.enabled);
            EXPECT_EQ(request_a.overlay.emphasis.mask, shared.scene_state.selection_mask);
            EXPECT_EQ(request_b.overlay.emphasis.mask, shared.scene_state.selection_mask);
            EXPECT_TRUE(request_a.overlay.has_selection);
            EXPECT_TRUE(request_b.overlay.has_selection);
        }

        TEST(WorkspaceRenderRequest, HitAndRenderUseSameImmutableCameraAfterWorkspaceChanges) {
            ViewportWorkspace workspace({800, 600});
            ASSERT_TRUE(workspace.setPreset(LayoutPreset::Quad));
            const auto snapshot = workspace.snapshot({30, 40, 800, 600}, 64, 4, {2.0f, 2.0f});
            ASSERT_EQ(snapshot.panes.size(), 4u);
            const auto& pane = snapshot.panes[2];
            const auto hit = PaneInteraction::hitTest(
                snapshot, {pane.rect.x + 10.0f, pane.rect.y + 10.0f});
            ASSERT_TRUE(hit);
            ASSERT_TRUE(workspace.close(pane.id));
            const auto view = makeWorkspaceFrameView(pane, pane.framebuffer_size, {});
            EXPECT_EQ(hit->view, pane.id);
            EXPECT_EQ(view.rotation, hit->camera.rotation);
            EXPECT_EQ(view.translation, hit->camera.translation);
            EXPECT_EQ(view.orthographic, hit->camera.projection.orthographic);
            EXPECT_EQ(view.size, hit->camera.framebuffer_size);
            EXPECT_EQ(view.size, glm::ivec2(pane.rect.width * 2, pane.rect.height * 2));
        }

        TEST(WorkspaceRenderRequest, TrainingSuppressesSelectionInEveryPane) {
            Viewport legacy(800, 600);
            FrameContext shared{.viewport = legacy};
            shared.training_active = true;
            shared.cursor_preview.active = true;
            shared.scene_state.selection_mask = std::make_shared<lfs::core::Tensor>();
            shared.scene_state.has_selection = true;
            PaneSnapshot pane{.id = 1};
            const auto request = buildWorkspaceRenderRequest(shared, pane, {300, 200}, {}, pane.id);
            EXPECT_FALSE(request.overlay.cursor.enabled);
            EXPECT_FALSE(request.overlay.emphasis.mask);
            EXPECT_FALSE(request.overlay.has_selection);
        }

        TEST(WorkspaceRenderRequest, RejectsInvalidIdentityAndEmptyExtent) {
            PaneSnapshot pane{.id = 1};
            EXPECT_THROW((void)makeWorkspaceFrameView(pane, {0, 200}, {}), std::invalid_argument);
            pane.id = kInvalidViewId;
            EXPECT_THROW((void)makeWorkspaceFrameView(pane, {300, 200}, {}), std::invalid_argument);
        }

    } // namespace
} // namespace lfs::vis

/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "workspace/workspace_serialization.hpp"

#include <gtest/gtest.h>

#include <limits>

namespace lfs::vis {
    namespace {

        TEST(WorkspaceSerialization, RoundTripsLayoutViewsAndDurableCameraState) {
            ViewportWorkspace source({800, 600});
            ASSERT_TRUE(source.setPreset(LayoutPreset::Quad));
            const auto ids = source.layout().leafIds();
            ASSERT_EQ(ids.size(), 4u);
            ASSERT_TRUE(source.focus(ids[2]));
            ASSERT_TRUE(source.maximize(ids[2]));

            auto* record = source.findView(ids[2]);
            ASSERT_NE(record, nullptr);
            record->camera.camera.t = {1.0f, 2.0f, 3.0f};
            record->camera.camera.pivot = {4.0f, 5.0f, 6.0f};
            record->projection.orthographic = true;
            record->projection.ortho_scale = 12.0f;
            record->depth.near_plane = 2.0f;
            record->depth.far_plane = 60.0f;
            record->grid_plane = 2;

            const auto json = workspaceStateToJson(source.exportState());
            ASSERT_TRUE(json);
            const auto restored = workspaceStateFromJson(*json);
            ASSERT_TRUE(restored);

            ViewportWorkspace target({320, 240});
            ASSERT_TRUE(target.importState(*restored));
            EXPECT_EQ(target.layout().leafIds(), ids);
            EXPECT_EQ(target.layout().focused(), ids[2]);
            EXPECT_EQ(target.activeViewport(), ids[2]);
            EXPECT_EQ(target.layout().maximized(), ids[2]);
            const auto* restored_record = target.findView(ids[2]);
            ASSERT_NE(restored_record, nullptr);
            EXPECT_EQ(restored_record->camera.camera.t, glm::vec3(1.0f, 2.0f, 3.0f));
            EXPECT_EQ(restored_record->camera.camera.pivot, glm::vec3(4.0f, 5.0f, 6.0f));
            EXPECT_TRUE(restored_record->projection.orthographic);
            EXPECT_FLOAT_EQ(restored_record->projection.ortho_scale, 12.0f);
            EXPECT_FLOAT_EQ(restored_record->depth.near_plane, 2.0f);
            EXPECT_FLOAT_EQ(restored_record->depth.far_plane, 60.0f);
            EXPECT_EQ(restored_record->grid_plane, 2);
        }

        TEST(WorkspaceSerialization, MissingActiveViewportUsesFocusedViewportFallback) {
            ViewportWorkspace source({800, 600});
            const ViewId primary = source.primaryView();
            const auto second_result = source.split(primary, SplitAxis::Horizontal);
            ASSERT_TRUE(second_result);
            const ViewId second = *second_result;
            ASSERT_TRUE(source.focus(second));

            auto json = workspaceStateToJson(source.exportState());
            ASSERT_TRUE(json);
            json->erase("active_viewport");
            const auto decoded = workspaceStateFromJson(*json);
            ASSERT_TRUE(decoded);
            EXPECT_FALSE(decoded->active_viewport);

            ViewportWorkspace restored;
            ASSERT_TRUE(restored.importState(*decoded));
            EXPECT_EQ(restored.layout().focused(), second);
            EXPECT_EQ(restored.activeViewport(), second);
        }

        TEST(WorkspaceSerialization, RejectsMalformedStateWithoutPartialImport) {
            ViewportWorkspace source({800, 600});
            auto json = workspaceStateToJson(source.exportState());
            ASSERT_TRUE(json);

            (*json)["version"] = 2;
            const auto unsupported = workspaceStateFromJson(*json);
            ASSERT_FALSE(unsupported);
            EXPECT_EQ(unsupported.error().code(), lfs::ErrorCode::Unsupported);

            json = workspaceStateToJson(source.exportState());
            ASSERT_TRUE(json);
            (*json)["layout"]["root"] = 999u;
            const auto malformed = workspaceStateFromJson(*json);
            ASSERT_FALSE(malformed);

            ViewportWorkspace target({320, 240});
            ASSERT_TRUE(target.setPreset(LayoutPreset::DualVertical));
            const auto target_primary = target.primaryView();
            ASSERT_NE(target.findView(target_primary), nullptr);
            target.findView(target_primary)->camera.camera.t = {9.0f, 8.0f, 7.0f};
            ASSERT_TRUE(target.focus(target_primary));
            const auto before = target.exportState();
            EXPECT_FALSE(target.importState(malformed ? *malformed : WorkspacePersistentState{}));
            const auto after = target.exportState();
            const auto before_json = workspaceStateToJson(before);
            const auto after_json = workspaceStateToJson(after);
            ASSERT_TRUE(before_json);
            ASSERT_TRUE(after_json);
            EXPECT_EQ(*after_json, *before_json);
        }

        TEST(WorkspaceSerialization, LegacyPrimaryInitializationSeedsCoherentSingleLayout) {
            ViewportWorkspace workspace({800, 600});
            ASSERT_TRUE(workspace.setPreset(LayoutPreset::Quad));
            const ViewId primary = workspace.primaryView();
            Viewport legacy(640, 480);
            ASSERT_EQ(legacy.frameBufferSize, glm::ivec2(640, 480));
            legacy.camera.t = {7.0f, 8.0f, 9.0f};
            legacy.camera.pivot = {1.0f, 2.0f, 3.0f};
            legacy.camera.zoomSpeed = 23.0f;
            legacy.camera.maxZoomSpeed = 80.0f;
            legacy.ortho_scale_override = 14.0f;
            RenderSettings settings;
            settings.orthographic = true;
            settings.focal_length_mm = 52.0f;
            settings.grid_plane = 2;
            settings.depth_filter_enabled = true;
            settings.depth_filter_min.z = -70.0f;
            settings.depth_filter_max.z = -3.0f;
            settings.depth_filter_scale_x = 0.45f;
            settings.depth_filter_scale_y = 0.55f;
            settings.depth_filter_offset_x = -0.2f;
            settings.depth_filter_offset_y = 0.3f;

            const auto migrated = initializeWorkspacePrimaryFromLegacy(workspace, legacy, settings);
            ASSERT_TRUE(migrated) << migrated.error().detail();
            const auto* record = workspace.findView(primary);
            ASSERT_NE(record, nullptr);
            EXPECT_EQ(record->camera.camera.t, legacy.camera.t);
            EXPECT_EQ(record->camera.camera.pivot, legacy.camera.pivot);
            EXPECT_TRUE(record->projection.orthographic);
            EXPECT_FLOAT_EQ(record->projection.ortho_scale, 14.0f);
            EXPECT_FLOAT_EQ(record->camera.camera.zoomSpeed, 23.0f);
            EXPECT_FLOAT_EQ(record->camera.camera.maxZoomSpeed, 80.0f);
            EXPECT_FLOAT_EQ(record->depth.near_plane, 3.0f);
            EXPECT_FLOAT_EQ(record->depth.far_plane, 70.0f);
            EXPECT_FLOAT_EQ(record->depth.scale_x, 0.45f);
            EXPECT_FLOAT_EQ(record->depth.scale_y, 0.55f);
            EXPECT_FLOAT_EQ(record->depth.offset_x, -0.2f);
            EXPECT_FLOAT_EQ(record->depth.offset_y, 0.3f);
            EXPECT_EQ(record->grid_plane, 2);
            EXPECT_EQ(workspace.layout().leafCount(), 1u);
            EXPECT_EQ(workspace.layout().focused(), primary);
            EXPECT_FALSE(workspace.layout().maximized());
        }

    } // namespace
} // namespace lfs::vis

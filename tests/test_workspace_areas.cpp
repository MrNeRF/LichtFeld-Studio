/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "workspace/workspace_serialization.hpp"

#include <algorithm>
#include <gtest/gtest.h>

namespace {
    using lfs::vis::SplitAxis;
    using lfs::vis::ViewId;
    using lfs::vis::ViewportWorkspace;

    TEST(WorkspaceAreas, MixedEditorsExposeFullAreasAndViewportContentOnly) {
        ViewportWorkspace workspace({800, 600});
        const ViewId primary = workspace.primaryView();
        const auto second = workspace.split(primary, SplitAxis::Horizontal);
        ASSERT_TRUE(second);
        ASSERT_TRUE(workspace.setAreaEditor(primary, "timeline"));
        auto* source = workspace.findView(primary);
        ASSERT_NE(source, nullptr);
        source->chrome["scroll"] = "42";
        ASSERT_TRUE(workspace.setAreaEditor(*second, "viewport"));

        const auto snapshot = workspace.snapshot({0, 0, 800, 600}, 64, 4, {1.0f, 1.0f}, 24);
        ASSERT_EQ(snapshot.areas.size(), 2u);
        ASSERT_EQ(snapshot.panes.size(), 1u);
        EXPECT_EQ(snapshot.areas[0].id, primary);
        EXPECT_EQ(snapshot.areas[0].editor_id, "timeline");
        EXPECT_EQ(snapshot.areas[0].content_rect.y, snapshot.areas[0].rect.y + 24);
        EXPECT_EQ(snapshot.panes[0].id, *second);
        EXPECT_EQ(snapshot.panes[0].rect.y, snapshot.areas[1].rect.y + 24);
        EXPECT_EQ(snapshot.live_viewport_ids, std::vector<ViewId>{*second});
    }

    TEST(WorkspaceAreas, LiveViewportIdsRetainMaximizedOutViewportAreas) {
        ViewportWorkspace workspace({800, 600});
        const ViewId first = workspace.primaryView();
        const auto second = workspace.split(first, SplitAxis::Horizontal);
        ASSERT_TRUE(second);
        ASSERT_TRUE(workspace.maximize(first));

        const auto snapshot = workspace.snapshot({0, 0, 800, 600});
        ASSERT_EQ(snapshot.areas.size(), 1u);
        ASSERT_EQ(snapshot.panes.size(), 1u);
        EXPECT_EQ(snapshot.live_viewport_ids, (std::vector<ViewId>{first, *second}));
        EXPECT_EQ(snapshot.live_ids, (std::vector<ViewId>{first, *second}));
    }

    TEST(WorkspaceAreas, SwitchingEditorPreservesCameraAndResetsTransientNavigation) {
        ViewportWorkspace workspace;
        const ViewId id = workspace.primaryView();
        auto* record = workspace.findView(id);
        ASSERT_NE(record, nullptr);
        record->camera.camera.t = {9.0f, 8.0f, 7.0f};
        record->camera.camera.prePos = {3.0f, 4.0f};
        record->camera.camera.isOrbiting = true;
        const auto generation = record->state_generation;

        ASSERT_TRUE(workspace.setAreaEditor(id, "outliner"));
        EXPECT_EQ(record->camera.camera.t, glm::vec3(9.0f, 8.0f, 7.0f));
        EXPECT_EQ(record->camera.camera.prePos, glm::vec2(0.0f));
        EXPECT_FALSE(record->camera.camera.isOrbiting);
        EXPECT_GT(record->state_generation, generation);
        ASSERT_TRUE(workspace.setAreaEditor(id, "viewport"));
        EXPECT_EQ(record->camera.camera.t, glm::vec3(9.0f, 8.0f, 7.0f));
    }

    TEST(WorkspaceAreas, ActiveViewportIsIndependentFromEditorFocus) {
        ViewportWorkspace workspace({800, 600});
        const ViewId first = workspace.primaryView();
        const auto second_result = workspace.split(first, SplitAxis::Horizontal);
        ASSERT_TRUE(second_result);
        const ViewId second = *second_result;

        EXPECT_EQ(workspace.activeViewport(), first);
        ASSERT_TRUE(workspace.focus(second));
        EXPECT_EQ(workspace.activeViewport(), second);

        ASSERT_TRUE(workspace.setAreaEditor(second, "timeline"));
        EXPECT_EQ(workspace.activeViewport(), first);
        ASSERT_TRUE(workspace.focus(second));
        EXPECT_EQ(workspace.activeViewport(), first);

        ASSERT_TRUE(workspace.setAreaEditor(first, "outliner"));
        EXPECT_FALSE(workspace.activeViewport());
        ASSERT_TRUE(workspace.setAreaEditor(first, "viewport"));
        EXPECT_EQ(workspace.activeViewport(), first);

        const auto third_result = workspace.split(first, SplitAxis::Vertical);
        ASSERT_TRUE(third_result);
        const ViewId third = *third_result;
        ASSERT_TRUE(workspace.focus(third));
        EXPECT_EQ(workspace.activeViewport(), third);
        ASSERT_TRUE(workspace.close(third));
        EXPECT_EQ(workspace.activeViewport(), first);

        const auto snapshot = workspace.snapshot({0, 0, 800, 600});
        EXPECT_EQ(snapshot.active_viewport, first);
    }

    TEST(WorkspaceAreas, RepeatedEditorAssignmentDoesNotStealActiveViewport) {
        ViewportWorkspace workspace({800, 600});
        const ViewId first_viewport = workspace.primaryView();
        const auto second_result = workspace.split(first_viewport, SplitAxis::Horizontal);
        ASSERT_TRUE(second_result);
        const ViewId second_viewport = *second_result;
        const auto panel_result = workspace.split(second_viewport, SplitAxis::Vertical);
        ASSERT_TRUE(panel_result);
        const ViewId panel = *panel_result;

        ASSERT_TRUE(workspace.focus(second_viewport));
        EXPECT_EQ(workspace.activeViewport(), second_viewport);
        // Header synchronization may assign the already-selected editor on
        // every frame. Idempotent assignments must not change the 3D target.
        ASSERT_TRUE(workspace.setAreaEditor(first_viewport, "viewport"));
        ASSERT_TRUE(workspace.setAreaEditor(first_viewport, "viewport"));
        EXPECT_EQ(workspace.activeViewport(), second_viewport);

        ASSERT_TRUE(workspace.setAreaEditor(panel, "lfs.scene"));
        ASSERT_TRUE(workspace.focus(panel));
        EXPECT_EQ(workspace.layout().focused(), panel);
        EXPECT_EQ(workspace.activeViewport(), second_viewport);
        ASSERT_TRUE(workspace.setAreaEditor(panel, "lfs.scene"));
        ASSERT_TRUE(workspace.setAreaEditor(panel, "lfs.scene"));
        EXPECT_EQ(workspace.layout().focused(), panel);
        EXPECT_EQ(workspace.activeViewport(), second_viewport);
    }

    TEST(WorkspaceAreas, SingleTreeSupportsMoreThanFourMixedAreas) {
        ViewportWorkspace workspace({1600, 1000});
        ViewId source = workspace.primaryView();
        for (int i = 0; i < 6; ++i) {
            const auto made = workspace.split(source, i % 2 ? SplitAxis::Vertical
                                                            : SplitAxis::Horizontal);
            ASSERT_TRUE(made);
            if (i % 2 == 0)
                ASSERT_TRUE(workspace.setAreaEditor(*made, "timeline"));
            source = *made;
        }
        const auto snapshot = workspace.snapshot({0, 0, 1600, 1000});
        EXPECT_EQ(snapshot.areas.size(), 7u);
        EXPECT_LT(snapshot.panes.size(), snapshot.areas.size());
        EXPECT_TRUE(workspace.layout().validate());
    }

    TEST(WorkspaceAreas, ResetToDefaultThreeAreasRetainsActiveCameraAndMonotonicIds) {
        ViewportWorkspace workspace({1600, 1000});
        const ViewId first = workspace.primaryView();
        const auto second_result = workspace.split(first, SplitAxis::Horizontal);
        ASSERT_TRUE(second_result);
        const ViewId second = *second_result;
        ASSERT_TRUE(workspace.setAreaEditor(first, "lfs.scene"));
        ASSERT_TRUE(workspace.focus(second));
        auto* active = workspace.findView(second);
        ASSERT_NE(active, nullptr);
        active->camera.camera.t = {12.0f, -4.0f, 8.0f};
        const auto next_before = workspace.views().nextId();

        const auto third_result = workspace.split(second, SplitAxis::Vertical);
        ASSERT_TRUE(third_result);
        ASSERT_TRUE(workspace.setAreaEditor(*third_result, "lfs.rendering"));
        ASSERT_TRUE(workspace.resetToDefaultThreeAreas());

        const auto snapshot = workspace.snapshot({0, 0, 1600, 1000});
        ASSERT_EQ(snapshot.areas.size(), 3u);
        EXPECT_EQ(snapshot.active_viewport, second);
        EXPECT_EQ(snapshot.focused, second);
        ASSERT_NE(workspace.findView(second), nullptr);
        EXPECT_EQ(workspace.findView(second)->camera.camera.t,
                  glm::vec3(12.0f, -4.0f, 8.0f));
        EXPECT_GT(workspace.views().nextId(), next_before);
        EXPECT_TRUE(workspace.layout().validate());
        EXPECT_EQ(workspace.findView(second)->editor_id, "viewport");
        EXPECT_EQ(std::count_if(snapshot.areas.begin(), snapshot.areas.end(),
                                [](const auto& area) { return area.editor_id == "lfs.scene"; }),
                  1);
        EXPECT_EQ(std::count_if(snapshot.areas.begin(), snapshot.areas.end(),
                                [](const auto& area) { return area.editor_id == "lfs.rendering"; }),
                  1);
    }

    TEST(WorkspaceAreas, MixedEditorMetadataSurvivesSerializationAndLegacyDefaults) {
        ViewportWorkspace workspace;
        const auto second = workspace.split(workspace.primaryView(), SplitAxis::Horizontal);
        ASSERT_TRUE(second);
        const ViewId first = workspace.primaryView();
        ASSERT_TRUE(workspace.setAreaEditor(first, "timeline"));
        ASSERT_TRUE(workspace.setAreaEditor(*second, "outliner"));
        workspace.findView(first)->chrome = {{"tab", "A"}, {"zoom", "1.25"}};

        const auto encoded = lfs::vis::workspaceStateToJson(workspace.exportState());
        ASSERT_TRUE(encoded);
        const auto decoded = lfs::vis::workspaceStateFromJson(*encoded);
        ASSERT_TRUE(decoded);
        ViewportWorkspace restored;
        ASSERT_TRUE(restored.importState(*decoded));
        EXPECT_EQ(restored.findView(first)->editor_id, "timeline");
        EXPECT_EQ(restored.findView(first)->chrome.at("tab"), "A");
        EXPECT_EQ(restored.findView(*second)->editor_id, "outliner");

        auto legacy = *encoded;
        for (auto& view : legacy["views"]) {
            view.erase("editor_id");
            view.erase("chrome");
        }
        const auto legacy_state = lfs::vis::workspaceStateFromJson(legacy);
        ASSERT_TRUE(legacy_state);
        for (const auto& view : legacy_state->views) {
            EXPECT_EQ(view.editor_id, "viewport");
            EXPECT_TRUE(view.chrome.empty());
        }
    }
} // namespace

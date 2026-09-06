/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Panel-addressed camera actions. The per-viewport gizmo buttons stamp
// their own panel into the event, so Home and Eye act on the panel whose toolbar
// was clicked -- not on whichever panel happens to hold focus, and without
// moving focus. These tests drive the native entry points the Python bindings
// call (lf.reset_camera(panel=...) / lf.focus_selection(panel=...)) against a
// real RenderingManager and real Viewports.

#include "core/event_bridge/event_bridge.hpp"
#include "core/event_bus.hpp"
#include "core/events.hpp"
#include "core/point_cloud.hpp"
#include "core/tensor.hpp"
#include "gui/gui_focus_state.hpp"
#include "input/input_controller.hpp"
#include "internal/viewport.hpp"
#include "rendering/depth_window_state.hpp"
#include "rendering/rendering_manager.hpp"
#include "rendering/rendering_types.hpp"
#include "scene/scene_manager.hpp"
#include "tools/selection_tool.hpp"
#include "tools/tool_base.hpp"

#include <gtest/gtest.h>
#include <memory>
#include <vector>

namespace lfs::vis {

    namespace {
        class InputControllerPanelCameraTest : public ::testing::Test {
        protected:
            void SetUp() override {
                lfs::event::EventBridge::instance().clear_all();
                lfs::core::event::bus().clear_all();
                services().clear();
                gui::guiFocusState().reset();
            }

            void TearDown() override {
                gui::guiFocusState().reset();
                services().clear();
                lfs::core::event::bus().clear_all();
                lfs::event::EventBridge::instance().clear_all();
            }
        };

        std::shared_ptr<core::PointCloud> makePointCloud() {
            const std::vector<float> positions{
                -1.0f, -1.0f, -1.0f,
                1.0f, 1.0f, 1.0f};
            auto means = core::Tensor::from_vector(
                positions, {positions.size() / 3, size_t{3}}, core::Device::CPU);
            auto colors = core::Tensor::from_vector(
                std::vector<float>(positions.size(), 1.0f),
                {positions.size() / 3, size_t{3}},
                core::Device::CPU);
            return std::make_shared<core::PointCloud>(std::move(means), std::move(colors));
        }

        // Distinct, recognisable poses so a test can name WHICH camera moved.
        void seedCamera(Viewport& viewport, const glm::vec3& home, const glm::vec3& current) {
            viewport.camera.home_R = glm::mat3(1.0f);
            viewport.camera.home_t = home;
            viewport.camera.home_pivot = home;
            viewport.camera.R = glm::mat3(1.0f);
            viewport.camera.t = current;
            viewport.camera.pivot = current;
        }

        void enableIndependentDual(Viewport& primary_viewport) {
            core::events::cmd::ToggleIndependentSplitView{
                .viewport = &primary_viewport,
            }
                .emit();
        }
        // --- shared depth-anchor rig -----------------------------------------
        //
        // depth_filter_transform and depth_filter_min/max.x/y are ONE global
        // camera anchor: DepthWindowState holds only near/far/scale/offset. So a
        // panel-addressed Home/Eye fired at the panel the user is NOT looking
        // through must move that panel's camera WITHOUT re-anchoring the shared
        // box under the focused panel. That is implemented by skipping
        // SelectionTool::syncDepthFilterToCamera on that one path only.
        //
        // The fixture installs a REAL, enabled SelectionTool with depth filtering
        // on, wired exactly as production does (visualizer_impl.cpp:414 ->
        // setSelectionTool). A fixture with no tool would be blind here: with
        // selection_tool_ null, publishCameraMove's depth branch never runs and
        // an anchor regression is invisible to every assertion.
        struct DepthAnchorRig {
            Viewport primary_viewport{200, 200};
            RenderingManager rendering_manager;
            SceneManager scene_manager;
            InputController controller{nullptr, primary_viewport};
            ToolContext tool_context{&rendering_manager, &scene_manager, &primary_viewport, nullptr};
            std::shared_ptr<tools::SelectionTool> selection_tool =
                std::make_shared<tools::SelectionTool>();
            Viewport* secondary_viewport = nullptr;

            // independent_dual=false exercises the compatibility mode where
            // resolvePanelViewport degrades to the one primary camera.
            void build(const SplitViewPanelId focused, const bool independent_dual = true) {
                services().set(&rendering_manager);
                scene_manager.getScene().addPointCloud("points", makePointCloud());
                controller.setToolContext(&tool_context);

                if (independent_dual) {
                    enableIndependentDual(primary_viewport);
                    secondary_viewport =
                        &rendering_manager.resolvePanelViewport(primary_viewport, SplitViewPanelId::Right);
                    // Distinct slots, so "both panel slots unchanged" is a real
                    // assertion and not two copies of one default.
                    rendering_manager.setDepthWindowForPanel(
                        SplitViewPanelId::Left,
                        DepthWindowState{.near_plane = 0.5f, .far_plane = 20.0f, .scale_x = 0.4f, .scale_y = 0.4f});
                    rendering_manager.setDepthWindowForPanel(
                        SplitViewPanelId::Right,
                        DepthWindowState{.near_plane = 2.0f, .far_plane = 60.0f, .scale_x = 0.7f, .scale_y = 0.7f});
                }

                seedCamera(primary_viewport, glm::vec3(1.0f, 2.0f, 3.0f), glm::vec3(-7.0f, -7.0f, -7.0f));
                if (secondary_viewport)
                    seedCamera(*secondary_viewport, glm::vec3(9.0f, 8.0f, 7.0f), glm::vec3(-4.0f, -4.0f, -4.0f));

                rendering_manager.setFocusedSplitPanel(focused);

                ASSERT_TRUE(selection_tool->initialize(tool_context));
                selection_tool->setEnabled(true);
                selection_tool->setDepthFilterEnabled(true);
                ASSERT_TRUE(selection_tool->isEnabled());
                ASSERT_TRUE(rendering_manager.getSettings().depth_filter_enabled);
                controller.setSelectionTool(selection_tool);
            }

            Viewport& viewportFor(const SplitViewPanelId panel) {
                return rendering_manager.resolvePanelViewport(primary_viewport, panel);
            }
        };

        // Everything the shared anchor is made of, plus both per-panel slots.
        struct AnchorSnapshot {
            glm::quat transform_rotation{};
            glm::vec3 transform_translation{};
            glm::vec3 filter_min{};
            glm::vec3 filter_max{};
            float scale_x = 0.0f;
            float scale_y = 0.0f;
            float offset_x = 0.0f;
            float offset_y = 0.0f;
            DepthWindowState left{};
            DepthWindowState right{};
        };

        AnchorSnapshot snapshotAnchor(RenderingManager& rendering_manager) {
            const auto settings = rendering_manager.getSettings();
            return AnchorSnapshot{
                .transform_rotation = settings.depth_filter_transform.getRotation(),
                .transform_translation = settings.depth_filter_transform.getTranslation(),
                .filter_min = settings.depth_filter_min,
                .filter_max = settings.depth_filter_max,
                .scale_x = settings.depth_filter_scale_x,
                .scale_y = settings.depth_filter_scale_y,
                .offset_x = settings.depth_filter_offset_x,
                .offset_y = settings.depth_filter_offset_y,
                .left = rendering_manager.getDepthWindowForPanel(SplitViewPanelId::Left),
                .right = rendering_manager.getDepthWindowForPanel(SplitViewPanelId::Right)};
        }

        void expectAnchorUnchanged(const AnchorSnapshot& before, const AnchorSnapshot& after) {
            EXPECT_EQ(before.transform_rotation, after.transform_rotation);
            EXPECT_EQ(before.transform_translation, after.transform_translation);
            EXPECT_EQ(before.filter_min, after.filter_min);
            EXPECT_EQ(before.filter_max, after.filter_max);
            EXPECT_EQ(before.scale_x, after.scale_x);
            EXPECT_EQ(before.scale_y, after.scale_y);
            EXPECT_EQ(before.offset_x, after.offset_x);
            EXPECT_EQ(before.offset_y, after.offset_y);
            EXPECT_EQ(before.left, after.left);
            EXPECT_EQ(before.right, after.right);
        }
    } // namespace

    // The defect panel addressing fixes: Home on the RIGHT panel's toolbar used
    // to reset the
    // primary camera, because the cmd::ResetCamera handler reads viewport_
    // directly. Addressed at 'right' it must reset the secondary camera and leave
    // the primary one exactly where it was.
    TEST_F(InputControllerPanelCameraTest, HomeResetsTheAddressedPanelInIndependentDual) {
        Viewport primary_viewport(200, 200);
        InputController controller(nullptr, primary_viewport);
        RenderingManager rendering_manager;
        services().set(&rendering_manager);

        enableIndependentDual(primary_viewport);
        ASSERT_EQ(rendering_manager.getSettings().split_view_mode, SplitViewMode::IndependentDual);

        auto& secondary_viewport =
            rendering_manager.resolvePanelViewport(primary_viewport, SplitViewPanelId::Right);
        ASSERT_NE(&secondary_viewport, &primary_viewport);

        seedCamera(primary_viewport, glm::vec3(1.0f, 2.0f, 3.0f), glm::vec3(-7.0f, -7.0f, -7.0f));
        seedCamera(secondary_viewport, glm::vec3(9.0f, 8.0f, 7.0f), glm::vec3(-4.0f, -4.0f, -4.0f));

        controller.resetCameraForPanel(SplitViewPanelId::Right);

        EXPECT_EQ(secondary_viewport.camera.t, glm::vec3(9.0f, 8.0f, 7.0f));
        EXPECT_EQ(secondary_viewport.camera.pivot, glm::vec3(9.0f, 8.0f, 7.0f));
        // The primary camera is untouched -- this is the assertion that fails if
        // the panel identity is dropped anywhere along RML -> Python -> native.
        EXPECT_EQ(primary_viewport.camera.t, glm::vec3(-7.0f, -7.0f, -7.0f));
        EXPECT_EQ(primary_viewport.camera.pivot, glm::vec3(-7.0f, -7.0f, -7.0f));
    }

    TEST_F(InputControllerPanelCameraTest, HomeAddressedLeftResetsThePrimaryPanelInIndependentDual) {
        Viewport primary_viewport(200, 200);
        InputController controller(nullptr, primary_viewport);
        RenderingManager rendering_manager;
        services().set(&rendering_manager);

        enableIndependentDual(primary_viewport);
        auto& secondary_viewport =
            rendering_manager.resolvePanelViewport(primary_viewport, SplitViewPanelId::Right);

        seedCamera(primary_viewport, glm::vec3(1.0f, 2.0f, 3.0f), glm::vec3(-7.0f, -7.0f, -7.0f));
        seedCamera(secondary_viewport, glm::vec3(9.0f, 8.0f, 7.0f), glm::vec3(-4.0f, -4.0f, -4.0f));

        // Focus is on the RIGHT panel, and the LEFT group's button must still act
        // on the LEFT panel: the addressing does not consult focus at all.
        rendering_manager.setFocusedSplitPanel(SplitViewPanelId::Right);
        controller.resetCameraForPanel(SplitViewPanelId::Left);

        EXPECT_EQ(primary_viewport.camera.t, glm::vec3(1.0f, 2.0f, 3.0f));
        EXPECT_EQ(secondary_viewport.camera.t, glm::vec3(-4.0f, -4.0f, -4.0f));
    }

    // Outside independent-dual split there is one camera, so both groups act on
    // it. resolvePanelViewport degrades to the primary viewport; this is today's
    // behavior and panel addressing keeps it.
    TEST_F(InputControllerPanelCameraTest, HomeOutsideIndependentDualAlwaysReachesThePrimaryCamera) {
        Viewport primary_viewport(200, 200);
        InputController controller(nullptr, primary_viewport);
        RenderingManager rendering_manager;
        services().set(&rendering_manager);

        ASSERT_EQ(rendering_manager.getSettings().split_view_mode, SplitViewMode::Disabled);
        seedCamera(primary_viewport, glm::vec3(1.0f, 2.0f, 3.0f), glm::vec3(-7.0f, -7.0f, -7.0f));

        controller.resetCameraForPanel(SplitViewPanelId::Right);
        EXPECT_EQ(primary_viewport.camera.t, glm::vec3(1.0f, 2.0f, 3.0f));

        seedCamera(primary_viewport, glm::vec3(4.0f, 5.0f, 6.0f), glm::vec3(-8.0f, -8.0f, -8.0f));
        controller.resetCameraForPanel(SplitViewPanelId::Left);
        EXPECT_EQ(primary_viewport.camera.t, glm::vec3(4.0f, 5.0f, 6.0f));
    }

    // The panel-less path is preserved byte-for-byte in behavior: an emitter that
    // carries no identity (lf.reset_camera() with no panel, the MCP tool) still
    // resets the primary viewport, even in independent-dual split with the right
    // panel focused.
    TEST_F(InputControllerPanelCameraTest, PanelLessResetCameraEventStillTargetsThePrimaryViewport) {
        Viewport primary_viewport(200, 200);
        InputController controller(nullptr, primary_viewport);
        RenderingManager rendering_manager;
        services().set(&rendering_manager);

        enableIndependentDual(primary_viewport);
        auto& secondary_viewport =
            rendering_manager.resolvePanelViewport(primary_viewport, SplitViewPanelId::Right);

        seedCamera(primary_viewport, glm::vec3(1.0f, 2.0f, 3.0f), glm::vec3(-7.0f, -7.0f, -7.0f));
        seedCamera(secondary_viewport, glm::vec3(9.0f, 8.0f, 7.0f), glm::vec3(-4.0f, -4.0f, -4.0f));
        rendering_manager.setFocusedSplitPanel(SplitViewPanelId::Right);

        core::events::cmd::ResetCamera{}.emit();

        EXPECT_EQ(primary_viewport.camera.t, glm::vec3(1.0f, 2.0f, 3.0f));
        EXPECT_EQ(secondary_viewport.camera.t, glm::vec3(-4.0f, -4.0f, -4.0f));
    }

    // Eye, addressed at the panel whose toolbar was clicked, while the OTHER
    // panel holds focus. Before panel addressing this followed focus.
    TEST_F(InputControllerPanelCameraTest, FocusSelectionAddressesItsOwnPanelNotTheFocusedOne) {
        Viewport primary_viewport(200, 200);
        InputController controller(nullptr, primary_viewport);
        RenderingManager rendering_manager;
        services().set(&rendering_manager);
        SceneManager scene_manager;
        scene_manager.getScene().addPointCloud("points", makePointCloud());
        ToolContext tool_context(nullptr, &scene_manager, &primary_viewport, nullptr);
        controller.setToolContext(&tool_context);

        enableIndependentDual(primary_viewport);
        auto& secondary_viewport =
            rendering_manager.resolvePanelViewport(primary_viewport, SplitViewPanelId::Right);

        seedCamera(primary_viewport, glm::vec3(1.0f, 2.0f, 3.0f), glm::vec3(-7.0f, -7.0f, -7.0f));
        seedCamera(secondary_viewport, glm::vec3(9.0f, 8.0f, 7.0f), glm::vec3(-4.0f, -4.0f, -4.0f));
        rendering_manager.setFocusedSplitPanel(SplitViewPanelId::Left);

        ASSERT_TRUE(controller.focusSelectionForPanel(SplitViewPanelId::Right));

        // The secondary camera framed the scene; the focused (primary) one did not.
        EXPECT_NE(secondary_viewport.camera.t, glm::vec3(-4.0f, -4.0f, -4.0f));
        EXPECT_EQ(primary_viewport.camera.t, glm::vec3(-7.0f, -7.0f, -7.0f));
    }

    // The chrome focus veto is untouched: addressing a panel is not focusing it.
    // Neither action may move the focused panel, in either direction.
    TEST_F(InputControllerPanelCameraTest, PanelAddressedCameraActionsNeverMoveTheFocusedPanel) {
        Viewport primary_viewport(200, 200);
        InputController controller(nullptr, primary_viewport);
        RenderingManager rendering_manager;
        services().set(&rendering_manager);
        SceneManager scene_manager;
        scene_manager.getScene().addPointCloud("points", makePointCloud());
        ToolContext tool_context(nullptr, &scene_manager, &primary_viewport, nullptr);
        controller.setToolContext(&tool_context);

        enableIndependentDual(primary_viewport);
        auto& secondary_viewport =
            rendering_manager.resolvePanelViewport(primary_viewport, SplitViewPanelId::Right);
        seedCamera(primary_viewport, glm::vec3(1.0f, 2.0f, 3.0f), glm::vec3(-7.0f, -7.0f, -7.0f));
        seedCamera(secondary_viewport, glm::vec3(9.0f, 8.0f, 7.0f), glm::vec3(-4.0f, -4.0f, -4.0f));

        rendering_manager.setFocusedSplitPanel(SplitViewPanelId::Left);
        controller.resetCameraForPanel(SplitViewPanelId::Right);
        EXPECT_EQ(rendering_manager.getFocusedSplitPanel(), SplitViewPanelId::Left);
        controller.focusSelectionForPanel(SplitViewPanelId::Right);
        EXPECT_EQ(rendering_manager.getFocusedSplitPanel(), SplitViewPanelId::Left);

        rendering_manager.setFocusedSplitPanel(SplitViewPanelId::Right);
        controller.resetCameraForPanel(SplitViewPanelId::Left);
        EXPECT_EQ(rendering_manager.getFocusedSplitPanel(), SplitViewPanelId::Right);
        controller.focusSelectionForPanel(SplitViewPanelId::Left);
        EXPECT_EQ(rendering_manager.getFocusedSplitPanel(), SplitViewPanelId::Right);
    }

    // The 'main' token is an EXPLICIT focused-panel request. module.cpp
    // resolves it to getFocusedSplitPanel() and then takes the very same
    // panel-addressed path 'left'/'right' take; this pins that resolution's
    // downstream half -- with Right focused, the resolved panel acts on Right,
    // and the unfocused panel does not move. The token->panel half needs the
    // Python binding and is covered by a Python test instead.
    TEST_F(InputControllerPanelCameraTest, MainResolvesToTheFocusedPanelForBothActions) {
        Viewport primary_viewport(200, 200);
        InputController controller(nullptr, primary_viewport);
        RenderingManager rendering_manager;
        services().set(&rendering_manager);
        SceneManager scene_manager;
        scene_manager.getScene().addPointCloud("points", makePointCloud());
        ToolContext tool_context(nullptr, &scene_manager, &primary_viewport, nullptr);
        controller.setToolContext(&tool_context);

        enableIndependentDual(primary_viewport);
        auto& secondary_viewport =
            rendering_manager.resolvePanelViewport(primary_viewport, SplitViewPanelId::Right);

        seedCamera(primary_viewport, glm::vec3(1.0f, 2.0f, 3.0f), glm::vec3(-7.0f, -7.0f, -7.0f));
        seedCamera(secondary_viewport, glm::vec3(9.0f, 8.0f, 7.0f), glm::vec3(-4.0f, -4.0f, -4.0f));
        rendering_manager.setFocusedSplitPanel(SplitViewPanelId::Right);

        // Home, panel='main' -> the RIGHT (focused) panel.
        controller.resetCameraForPanel(rendering_manager.getFocusedSplitPanel());
        EXPECT_EQ(secondary_viewport.camera.t, glm::vec3(9.0f, 8.0f, 7.0f));
        EXPECT_EQ(primary_viewport.camera.t, glm::vec3(-7.0f, -7.0f, -7.0f));

        // Eye, panel='main' -> the RIGHT (focused) panel, and focus still does
        // not move.
        seedCamera(secondary_viewport, glm::vec3(9.0f, 8.0f, 7.0f), glm::vec3(-4.0f, -4.0f, -4.0f));
        ASSERT_TRUE(controller.focusSelectionForPanel(rendering_manager.getFocusedSplitPanel()));
        EXPECT_NE(secondary_viewport.camera.t, glm::vec3(-4.0f, -4.0f, -4.0f));
        EXPECT_EQ(primary_viewport.camera.t, glm::vec3(-7.0f, -7.0f, -7.0f));
        EXPECT_EQ(rendering_manager.getFocusedSplitPanel(), SplitViewPanelId::Right);
    }

    // --- shared depth anchor ------------------------------------------------
    // An off-focus Home moves the addressed panel's camera and leaves the shared
    // depth anchor exactly where the focused panel put it. ACCEPTED
    // CONSEQUENCE: the acted panel's box does not follow its camera until
    // that panel is focused and publishes a camera move of its own.
    TEST_F(InputControllerPanelCameraTest, OffFocusHomeLeftFocusedRightActedLeavesTheSharedDepthAnchor) {
        DepthAnchorRig rig;
        rig.build(SplitViewPanelId::Left);
        const auto before = snapshotAnchor(rig.rendering_manager);

        rig.controller.resetCameraForPanel(SplitViewPanelId::Right);

        // The addressed camera moved...
        EXPECT_EQ(rig.secondary_viewport->camera.t, glm::vec3(9.0f, 8.0f, 7.0f));
        // ...the focused one did not, and focus did not move.
        EXPECT_EQ(rig.primary_viewport.camera.t, glm::vec3(-7.0f, -7.0f, -7.0f));
        EXPECT_EQ(rig.rendering_manager.getFocusedSplitPanel(), SplitViewPanelId::Left);
        expectAnchorUnchanged(before, snapshotAnchor(rig.rendering_manager));
    }

    TEST_F(InputControllerPanelCameraTest, OffFocusHomeRightFocusedLeftActedLeavesTheSharedDepthAnchor) {
        DepthAnchorRig rig;
        rig.build(SplitViewPanelId::Right);
        const auto before = snapshotAnchor(rig.rendering_manager);

        rig.controller.resetCameraForPanel(SplitViewPanelId::Left);

        EXPECT_EQ(rig.primary_viewport.camera.t, glm::vec3(1.0f, 2.0f, 3.0f));
        EXPECT_EQ(rig.secondary_viewport->camera.t, glm::vec3(-4.0f, -4.0f, -4.0f));
        EXPECT_EQ(rig.rendering_manager.getFocusedSplitPanel(), SplitViewPanelId::Right);
        expectAnchorUnchanged(before, snapshotAnchor(rig.rendering_manager));
    }

    TEST_F(InputControllerPanelCameraTest, OffFocusEyeLeftFocusedRightActedLeavesTheSharedDepthAnchor) {
        DepthAnchorRig rig;
        rig.build(SplitViewPanelId::Left);
        const auto before = snapshotAnchor(rig.rendering_manager);

        ASSERT_TRUE(rig.controller.focusSelectionForPanel(SplitViewPanelId::Right));

        EXPECT_NE(rig.secondary_viewport->camera.t, glm::vec3(-4.0f, -4.0f, -4.0f));
        EXPECT_EQ(rig.primary_viewport.camera.t, glm::vec3(-7.0f, -7.0f, -7.0f));
        EXPECT_EQ(rig.rendering_manager.getFocusedSplitPanel(), SplitViewPanelId::Left);
        expectAnchorUnchanged(before, snapshotAnchor(rig.rendering_manager));
    }

    TEST_F(InputControllerPanelCameraTest, OffFocusEyeRightFocusedLeftActedLeavesTheSharedDepthAnchor) {
        DepthAnchorRig rig;
        rig.build(SplitViewPanelId::Right);
        const auto before = snapshotAnchor(rig.rendering_manager);

        ASSERT_TRUE(rig.controller.focusSelectionForPanel(SplitViewPanelId::Left));

        EXPECT_NE(rig.primary_viewport.camera.t, glm::vec3(-7.0f, -7.0f, -7.0f));
        EXPECT_EQ(rig.secondary_viewport->camera.t, glm::vec3(-4.0f, -4.0f, -4.0f));
        EXPECT_EQ(rig.rendering_manager.getFocusedSplitPanel(), SplitViewPanelId::Right);
        expectAnchorUnchanged(before, snapshotAnchor(rig.rendering_manager));
    }

    // The control: acting on the FOCUSED panel still re-anchors, exactly as
    // before. Without this, "the anchor never moves" would pass a build that
    // simply deleted the sync call.
    TEST_F(InputControllerPanelCameraTest, FocusedPanelHomeStillReanchorsTheSharedDepthBox) {
        DepthAnchorRig rig;
        rig.build(SplitViewPanelId::Right);
        const auto before = snapshotAnchor(rig.rendering_manager);
        ASSERT_EQ(before.transform_translation, glm::vec3(-4.0f, -4.0f, -4.0f));

        rig.controller.resetCameraForPanel(SplitViewPanelId::Right);

        EXPECT_EQ(rig.secondary_viewport->camera.t, glm::vec3(9.0f, 8.0f, 7.0f));
        const auto after = snapshotAnchor(rig.rendering_manager);
        EXPECT_EQ(after.transform_translation, glm::vec3(9.0f, 8.0f, 7.0f));
        EXPECT_EQ(rig.rendering_manager.getFocusedSplitPanel(), SplitViewPanelId::Right);
    }

    TEST_F(InputControllerPanelCameraTest, FocusedPanelEyeStillReanchorsTheSharedDepthBox) {
        DepthAnchorRig rig;
        rig.build(SplitViewPanelId::Left);
        const auto before = snapshotAnchor(rig.rendering_manager);
        ASSERT_EQ(before.transform_translation, glm::vec3(-7.0f, -7.0f, -7.0f));

        ASSERT_TRUE(rig.controller.focusSelectionForPanel(SplitViewPanelId::Left));

        const auto after = snapshotAnchor(rig.rendering_manager);
        EXPECT_EQ(after.transform_translation, rig.primary_viewport.camera.t);
        EXPECT_NE(after.transform_translation, before.transform_translation);
    }

    // Compatibility 1: the panel-less path carries no identity, so it is never
    // guarded -- cmd::ResetCamera still re-anchors, byte-equivalently.
    TEST_F(InputControllerPanelCameraTest, PanelLessResetCameraStillReanchorsWhileTheRightPanelIsFocused) {
        DepthAnchorRig rig;
        rig.build(SplitViewPanelId::Right);
        const auto before = snapshotAnchor(rig.rendering_manager);

        core::events::cmd::ResetCamera{}.emit();

        EXPECT_EQ(rig.primary_viewport.camera.t, glm::vec3(1.0f, 2.0f, 3.0f));
        const auto after = snapshotAnchor(rig.rendering_manager);
        // The panel-less path anchors to the primary viewport it moved, as before.
        EXPECT_EQ(after.transform_translation, glm::vec3(1.0f, 2.0f, 3.0f));
        EXPECT_NE(after.transform_translation, before.transform_translation);
    }

    // Compatibility 2: outside independent-dual there is one camera and one
    // panel-agnostic anchor, so a panel-addressed action must still sync even
    // when the addressed panel is not the focused one.
    TEST_F(InputControllerPanelCameraTest, OutsideIndependentDualPanelAddressedHomeStillReanchors) {
        DepthAnchorRig rig;
        rig.build(SplitViewPanelId::Left, /*independent_dual=*/false);
        ASSERT_EQ(rig.rendering_manager.getSettings().split_view_mode, SplitViewMode::Disabled);
        const auto before = snapshotAnchor(rig.rendering_manager);

        rig.controller.resetCameraForPanel(SplitViewPanelId::Right);

        EXPECT_EQ(rig.primary_viewport.camera.t, glm::vec3(1.0f, 2.0f, 3.0f));
        const auto after = snapshotAnchor(rig.rendering_manager);
        EXPECT_EQ(after.transform_translation, glm::vec3(1.0f, 2.0f, 3.0f));
        EXPECT_NE(after.transform_translation, before.transform_translation);
    }

} // namespace lfs::vis

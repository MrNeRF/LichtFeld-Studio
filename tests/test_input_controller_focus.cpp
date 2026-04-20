/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/event_bridge/scoped_handler.hpp"
#include "core/events.hpp"
#include "core/services.hpp"
#include "gui/gui_focus_state.hpp"
#include "input/input_controller.hpp"
#include "input/input_router.hpp"
#include "input/key_codes.hpp"
#include "internal/viewport.hpp"
#include "rendering/coordinate_conventions.hpp"

#include <glm/gtc/constants.hpp>
#include <gtest/gtest.h>
#include <variant>
#include <imgui.h>

namespace lfs::vis {

    namespace {
        class InputControllerFocusTest : public ::testing::Test {
        protected:
            void SetUp() override {
                services().clear();
                gui::guiFocusState().reset();

                IMGUI_CHECKVERSION();
                ImGui::CreateContext();
            }

            void TearDown() override {
                ImGui::DestroyContext();

                gui::guiFocusState().reset();
                services().clear();
            }
        };
    } // namespace

    TEST_F(InputControllerFocusTest, CameraViewHotkeysDoNotBypassGuiKeyboardCapture) {
        Viewport viewport(200, 200);
        InputController controller(nullptr, viewport);
        input::InputRouter router;
        router.setInputController(&controller);
        controller.setInputRouter(&router);

        lfs::event::ScopedHandler handlers;
        int goto_cam_view_count = 0;
        handlers.subscribe<core::events::cmd::GoToCamView>(
            [&](const auto&) { ++goto_cam_view_count; });

        auto& focus = gui::guiFocusState();
        focus.want_capture_keyboard = true;

        controller.handleKey(input::KEY_RIGHT, input::ACTION_PRESS, input::KEYMOD_NONE);

        EXPECT_EQ(goto_cam_view_count, 0);
    }

    TEST_F(InputControllerFocusTest, ViewportViewHotkeysDoNotBypassGuiKeyboardFocus) {
        Viewport viewport(200, 200);
        InputController controller(nullptr, viewport);
        input::InputRouter router;
        router.setInputController(&controller);
        controller.setInputRouter(&router);

        lfs::event::ScopedHandler handlers;
        int toggle_gt_count = 0;
        int toggle_split_count = 0;
        handlers.subscribe<core::events::cmd::ToggleGTComparison>(
            [&](const auto&) { ++toggle_gt_count; });
        handlers.subscribe<core::events::cmd::ToggleSplitView>(
            [&](const auto&) { ++toggle_split_count; });

        auto& focus = gui::guiFocusState();
        focus.want_capture_keyboard = true;
        focus.any_item_active = true;

        controller.handleKey(input::KEY_G, input::ACTION_PRESS, input::KEYMOD_NONE);
        controller.handleKey(input::KEY_V, input::ACTION_PRESS, input::KEYMOD_NONE);

        EXPECT_EQ(toggle_gt_count, 0);
        EXPECT_EQ(toggle_split_count, 0);
    }

    TEST_F(InputControllerFocusTest, ViewportViewHotkeysWorkAfterViewportFocus) {
        Viewport viewport(200, 200);
        InputController controller(nullptr, viewport);
        input::InputRouter router;
        router.setInputController(&controller);
        controller.setInputRouter(&router);

        lfs::event::ScopedHandler handlers;
        int toggle_gt_count = 0;
        int toggle_split_count = 0;
        handlers.subscribe<core::events::cmd::ToggleGTComparison>(
            [&](const auto&) { ++toggle_gt_count; });
        handlers.subscribe<core::events::cmd::ToggleSplitView>(
            [&](const auto&) { ++toggle_split_count; });

        router.beginMouseButton(input::ACTION_PRESS, 40.0, 50.0);
        router.endMouseButton(input::ACTION_RELEASE);

        controller.handleKey(input::KEY_G, input::ACTION_PRESS, input::KEYMOD_NONE);
        controller.handleKey(input::KEY_V, input::ACTION_PRESS, input::KEYMOD_NONE);

        EXPECT_EQ(toggle_gt_count, 1);
        EXPECT_EQ(toggle_split_count, 1);
    }

    TEST_F(InputControllerFocusTest, ProgrammaticViewportFocusAllowsViewportHotkeys) {
        Viewport viewport(200, 200);
        InputController controller(nullptr, viewport);
        input::InputRouter router;
        router.setInputController(&controller);
        controller.setInputRouter(&router);

        lfs::event::ScopedHandler handlers;
        int toggle_gt_count = 0;
        handlers.subscribe<core::events::cmd::ToggleGTComparison>(
            [&](const auto&) { ++toggle_gt_count; });

        auto& focus = gui::guiFocusState();
        focus.want_capture_keyboard = true;
        focus.any_item_active = true;

        router.focusViewportKeyboard();
        controller.handleKey(input::KEY_G, input::ACTION_PRESS, input::KEYMOD_NONE);

        EXPECT_EQ(toggle_gt_count, 1);
    }

    TEST_F(InputControllerFocusTest, ViewportViewHotkeysStayBlockedDuringTextEntry) {
        Viewport viewport(200, 200);
        InputController controller(nullptr, viewport);
        input::InputRouter router;
        router.setInputController(&controller);
        controller.setInputRouter(&router);

        lfs::event::ScopedHandler handlers;
        int toggle_gt_count = 0;
        int toggle_split_count = 0;
        handlers.subscribe<core::events::cmd::ToggleGTComparison>(
            [&](const auto&) { ++toggle_gt_count; });
        handlers.subscribe<core::events::cmd::ToggleSplitView>(
            [&](const auto&) { ++toggle_split_count; });

        auto& focus = gui::guiFocusState();
        focus.want_capture_keyboard = true;
        focus.want_text_input = true;
        focus.any_item_active = true;

        router.beginMouseButton(input::ACTION_PRESS, 40.0, 50.0);
        router.endMouseButton(input::ACTION_RELEASE);

        controller.handleKey(input::KEY_G, input::ACTION_PRESS, input::KEYMOD_NONE);
        controller.handleKey(input::KEY_V, input::ACTION_PRESS, input::KEYMOD_NONE);

        EXPECT_EQ(toggle_gt_count, 0);
        EXPECT_EQ(toggle_split_count, 0);
    }

    TEST_F(InputControllerFocusTest, GlobalShortcutsUseLogicalKeyWhileMovementUsesPhysicalKey) {
        Viewport viewport(200, 200);
        InputController controller(nullptr, viewport);
        controller.initialize();
        input::InputRouter router;
        router.setInputController(&controller);
        controller.setInputRouter(&router);

        lfs::event::ScopedHandler handlers;
        int undo_count = 0;
        handlers.subscribe<core::events::cmd::Undo>(
            [&](const auto&) { ++undo_count; });

        router.beginMouseButton(input::ACTION_PRESS, 40.0, 50.0);
        router.endMouseButton(input::ACTION_RELEASE);

        controller.handleKey(input::KEY_W, input::KEY_Z, 0, input::ACTION_PRESS, input::KEYMOD_CTRL);

        EXPECT_EQ(undo_count, 1);
        EXPECT_FALSE(controller.isContinuousInputActive());

        controller.handleKey(input::KEY_W, input::KEY_Z, 0, input::ACTION_PRESS, input::KEYMOD_NONE);

        EXPECT_TRUE(controller.isContinuousInputActive());
    }

    TEST_F(InputControllerFocusTest, RedoAliasUsesLogicalKeyAndDoesNotTriggerMovement) {
        Viewport viewport(200, 200);
        InputController controller(nullptr, viewport);
        controller.initialize();
        input::InputRouter router;
        router.setInputController(&controller);
        controller.setInputRouter(&router);

        lfs::event::ScopedHandler handlers;
        int redo_count = 0;
        handlers.subscribe<core::events::cmd::Redo>(
            [&](const auto&) { ++redo_count; });

        router.beginMouseButton(input::ACTION_PRESS, 40.0, 50.0);
        router.endMouseButton(input::ACTION_RELEASE);

        controller.handleKey(input::KEY_W, input::KEY_Z, 0, input::ACTION_PRESS,
                             input::KEYMOD_CTRL | input::KEYMOD_SHIFT);

        EXPECT_EQ(redo_count, 1);
        EXPECT_FALSE(controller.isContinuousInputActive());
    }

    TEST_F(InputControllerFocusTest, CameraDragBindingsIgnoreExtraShiftModifier) {
        Viewport viewport(200, 200);
        InputController controller(nullptr, viewport);

        EXPECT_EQ(controller.getBindings().getActionForDrag(
                      input::ToolMode::GLOBAL, input::MouseButton::MIDDLE, input::KEYMOD_SHIFT),
                  input::Action::CAMERA_ORBIT);
        EXPECT_EQ(controller.getBindings().getActionForDrag(
                      input::ToolMode::GLOBAL, input::MouseButton::RIGHT, input::KEYMOD_SHIFT),
                  input::Action::CAMERA_PAN);
    }

    TEST_F(InputControllerFocusTest, StaleMouseCaptureDoesNotRequireSecondViewportClick) {
        Viewport viewport(200, 200);
        InputController controller(nullptr, viewport);
        input::InputRouter router;
        router.setInputController(&controller);
        controller.setInputRouter(&router);

        gui::guiFocusState().want_capture_mouse = true;

        router.beginMouseButton(input::ACTION_PRESS, 40.0, 50.0);
        controller.handleMouseButton(static_cast<int>(input::AppMouseButton::MIDDLE),
                                     input::ACTION_PRESS, 40.0, 50.0);
        router.endMouseButton(input::ACTION_PRESS);

        EXPECT_TRUE(controller.hasViewportKeyboardFocus());
        EXPECT_TRUE(controller.isContinuousInputActive());
    }

    TEST_F(InputControllerFocusTest, MissedMouseReleaseClearsPointerCapture) {
        Viewport viewport(200, 200);
        InputController controller(nullptr, viewport);
        input::InputRouter router;
        router.setInputController(&controller);
        controller.setInputRouter(&router);

        router.beginMouseButton(input::ACTION_PRESS, 40.0, 50.0);

        EXPECT_EQ(router.state().pointer_capture, input::InputTarget::Viewport);
        EXPECT_EQ(router.pointerTarget(2500.0, 2500.0), input::InputTarget::Viewport);

        router.syncPressedMouseButtons(false);

        EXPECT_EQ(router.state().pointer_capture, input::InputTarget::None);
        EXPECT_EQ(router.pointerTarget(2500.0, 2500.0), input::InputTarget::None);
    }

    TEST_F(InputControllerFocusTest, HoverTargetIgnoresPointerCapture) {
        Viewport viewport(200, 200);
        InputController controller(nullptr, viewport);
        input::InputRouter router;
        router.setInputController(&controller);
        controller.setInputRouter(&router);

        router.beginMouseButton(input::ACTION_PRESS, 40.0, 50.0);

        EXPECT_EQ(router.pointerTarget(2500.0, 2500.0), input::InputTarget::Viewport);
        EXPECT_EQ(router.hoverTarget(2500.0, 2500.0), input::InputTarget::None);
    }

    TEST_F(InputControllerFocusTest, SplitToggleClearsActiveCameraDrag) {
        Viewport viewport(200, 200);
        InputController controller(nullptr, viewport);
        input::InputRouter router;
        router.setInputController(&controller);
        controller.setInputRouter(&router);

        router.beginMouseButton(input::ACTION_PRESS, 40.0, 50.0);
        controller.handleMouseButton(static_cast<int>(input::AppMouseButton::MIDDLE),
                                     input::ACTION_PRESS, 40.0, 50.0);

        ASSERT_TRUE(controller.isContinuousInputActive());

        core::events::cmd::ToggleSplitView{}.emit();

        EXPECT_FALSE(controller.isContinuousInputActive());
    }

    TEST_F(InputControllerFocusTest, FpvModeUsesInPlaceLookForPrimaryCameraDrag) {
        Viewport viewport(200, 200);
        InputController controller(nullptr, viewport);
        viewport.camera.t = glm::vec3(0.0f, 0.0f, 5.0f);
        viewport.camera.setPivot(glm::vec3(0.0f));
        viewport.camera.R = glm::mat3(1.0f);
        controller.setCameraNavigationMode(InputController::CameraNavigationMode::FPV);

        controller.handleMouseButton(static_cast<int>(input::AppMouseButton::MIDDLE),
                                     input::ACTION_PRESS, 40.0, 50.0);
        controller.handleMouseMove(40.0, 0.0);
        controller.handleMouseButton(static_cast<int>(input::AppMouseButton::MIDDLE),
                                     input::ACTION_RELEASE, 40.0, 0.0);

        const glm::vec3 forward = lfs::rendering::cameraForward(viewport.camera.R);
        const glm::vec3 to_pivot = glm::normalize(viewport.camera.getPivot() - viewport.camera.t);
        EXPECT_NEAR(viewport.camera.t.x, 0.0f, 1e-5f);
        EXPECT_NEAR(viewport.camera.t.y, 0.0f, 1e-5f);
        EXPECT_NEAR(viewport.camera.t.z, 5.0f, 1e-5f);
        EXPECT_GT(forward.y, 0.0f);
        EXPECT_NEAR(glm::dot(forward, to_pivot), 1.0f, 1e-4f);
    }

    TEST_F(InputControllerFocusTest, TrackballModeAllowsPerfectTopView) {
        Viewport viewport(200, 200);
        InputController controller(nullptr, viewport);
        viewport.camera.t = glm::vec3(0.0f, 0.0f, 5.0f);
        viewport.camera.setPivot(glm::vec3(0.0f));
        viewport.camera.R = lfs::rendering::makeVisualizerLookAtRotation(
            viewport.camera.t, viewport.camera.getPivot());
        controller.setCameraNavigationMode(InputController::CameraNavigationMode::Trackball);

        controller.handleMouseButton(static_cast<int>(input::AppMouseButton::MIDDLE),
                                     input::ACTION_PRESS, 40.0, 50.0);
        controller.handleMouseMove(40.0, 50.0 - glm::half_pi<float>() / 0.002f);
        controller.handleMouseButton(static_cast<int>(input::AppMouseButton::MIDDLE),
                                     input::ACTION_RELEASE, 40.0, 50.0 - glm::half_pi<float>() / 0.002f);

        const glm::vec3 forward = lfs::rendering::cameraForward(viewport.camera.R);
        EXPECT_NEAR(glm::length(viewport.camera.getPivot() - viewport.camera.t), 5.0f, 1e-3f);
        EXPECT_NEAR(std::abs(forward.y), 1.0f, 1e-3f);
    }

    TEST_F(InputControllerFocusTest, TrackballSnapAlignsToNearestAxisView) {
        Viewport viewport(200, 200);
        InputController controller(nullptr, viewport);
        viewport.camera.t = glm::vec3(0.0f, 0.0f, 5.0f);
        viewport.camera.setPivot(glm::vec3(0.0f));
        viewport.camera.R = lfs::rendering::makeVisualizerLookAtRotation(
            viewport.camera.t, viewport.camera.getPivot());
        controller.setCameraNavigationMode(InputController::CameraNavigationMode::Trackball);
        controller.setCameraViewSnapEnabled(true);

        const double snap_target_y = 50.0 - glm::radians(85.0f) / 0.002f;
        controller.handleMouseButton(static_cast<int>(input::AppMouseButton::MIDDLE),
                                     input::ACTION_PRESS, 40.0, 50.0);
        controller.handleMouseMove(40.0, snap_target_y);
        controller.handleMouseButton(static_cast<int>(input::AppMouseButton::MIDDLE),
                                     input::ACTION_RELEASE, 40.0, snap_target_y);

        const glm::vec3 forward = lfs::rendering::cameraForward(viewport.camera.R);
        EXPECT_NEAR(std::abs(forward.y), 1.0f, 1e-4f);
        EXPECT_NEAR(std::abs(forward.x), 0.0f, 1e-4f);
        EXPECT_NEAR(std::abs(forward.z), 0.0f, 1e-4f);
    }

    TEST_F(InputControllerFocusTest, AxisAlignedViewPreservesPivotAndDistance) {
        Viewport viewport(200, 200);
        const glm::vec3 pivot(12.0f, 1.5f, -7.0f);
        viewport.camera.t = glm::vec3(9.0f, 4.0f, -2.0f);
        viewport.camera.setPivot(pivot);
        viewport.camera.R = lfs::rendering::makeVisualizerLookAtRotation(
            viewport.camera.t, viewport.camera.getPivot());

        const float initial_distance = glm::length(viewport.camera.getPivot() - viewport.camera.t);
        viewport.camera.setAxisAlignedView(1, false);

        const glm::vec3 forward = lfs::rendering::cameraForward(viewport.camera.R);
        EXPECT_NEAR(glm::length(viewport.camera.getPivot() - viewport.camera.t), initial_distance, 1e-4f);
        EXPECT_NEAR(glm::distance(viewport.camera.getPivot(), pivot), 0.0f, 1e-6f);
        EXPECT_NEAR(std::abs(forward.y), 1.0f, 1e-4f);
        EXPECT_NEAR(std::abs(forward.x), 0.0f, 1e-4f);
        EXPECT_NEAR(std::abs(forward.z), 0.0f, 1e-4f);
    }

    TEST_F(InputControllerFocusTest, PointerTargetsExposeHoverAndCapturedTargets) {
        Viewport viewport(200, 200);
        InputController controller(nullptr, viewport);
        input::InputRouter router;
        router.setInputController(&controller);
        controller.setInputRouter(&router);

        router.beginMouseButton(input::ACTION_PRESS, 40.0, 50.0);

        const auto targets = router.pointerTargets(2500.0, 2500.0);
        EXPECT_EQ(targets.pointer_target, input::InputTarget::Viewport);
        EXPECT_EQ(targets.hover_target, input::InputTarget::None);
    }

    TEST_F(InputControllerFocusTest, NavigationMouseCaptureFinalizesToSingleClickBinding) {
        input::InputBindings bindings;
        bindings.startCapture(input::ToolMode::GLOBAL, input::Action::CAMERA_ORBIT);
        bindings.captureMouseButton(static_cast<int>(input::MouseButton::RIGHT), input::MODIFIER_NONE);

        auto& capture_state = const_cast<input::CaptureState&>(bindings.getCaptureState());
        capture_state.first_click_time -= std::chrono::milliseconds(500);
        bindings.updateCapture();

        const auto captured = bindings.getAndClearCaptured();
        ASSERT_TRUE(captured.has_value());

        const auto* mouse_trigger = std::get_if<input::MouseButtonTrigger>(&*captured);
        ASSERT_NE(mouse_trigger, nullptr);
        EXPECT_EQ(mouse_trigger->button, input::MouseButton::RIGHT);
        EXPECT_EQ(mouse_trigger->modifiers, input::MODIFIER_NONE);
        EXPECT_FALSE(mouse_trigger->double_click);
        EXPECT_FALSE(bindings.isCapturing());
    }

    TEST_F(InputControllerFocusTest, SelectionMouseCaptureStillFinalizesToDragBinding) {
        input::InputBindings bindings;
        bindings.startCapture(input::ToolMode::SELECTION, input::Action::SELECTION_REPLACE);
        bindings.captureMouseButton(static_cast<int>(input::MouseButton::LEFT), input::MODIFIER_NONE);

        auto& capture_state = const_cast<input::CaptureState&>(bindings.getCaptureState());
        capture_state.first_click_time -= std::chrono::milliseconds(500);
        bindings.updateCapture();

        const auto captured = bindings.getAndClearCaptured();
        ASSERT_TRUE(captured.has_value());

        const auto* drag_trigger = std::get_if<input::MouseDragTrigger>(&*captured);
        ASSERT_NE(drag_trigger, nullptr);
        EXPECT_EQ(drag_trigger->button, input::MouseButton::LEFT);
        EXPECT_EQ(drag_trigger->modifiers, input::MODIFIER_NONE);
        EXPECT_FALSE(bindings.isCapturing());
    }

} // namespace lfs::vis

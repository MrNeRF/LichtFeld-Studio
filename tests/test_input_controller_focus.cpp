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

#include <gtest/gtest.h>
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

} // namespace lfs::vis

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

    TEST_F(InputControllerFocusTest, GtComparisonHotkeyRemainsAvailableOutsideTextEntry) {
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

        controller.handleKey(input::KEY_G, input::ACTION_PRESS, input::KEYMOD_NONE);

        EXPECT_EQ(toggle_gt_count, 1);
    }

    TEST_F(InputControllerFocusTest, GtComparisonHotkeyStaysBlockedDuringTextEntry) {
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
        focus.want_text_input = true;
        focus.any_item_active = true;

        controller.handleKey(input::KEY_G, input::ACTION_PRESS, input::KEYMOD_NONE);

        EXPECT_EQ(toggle_gt_count, 0);
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

} // namespace lfs::vis

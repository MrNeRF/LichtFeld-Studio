/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/gui_focus_state.hpp"
#include "input/input_controller.hpp"
#include "input/sdl_coordinate_utils.hpp"
#include "workspace/viewport_workspace.hpp"
#include <SDL3/SDL_keyboard.h>
#include <cmath>
#include <gtest/gtest.h>

namespace lfs::vis {
    TEST(WorkspaceNavigation, WheelAndPinchUpdateOnlyThePointedOrthographicView) {
        struct RestoreInput {
            gui::GuiFocusState focus = gui::guiFocusState();
            SDL_Keymod modifiers = SDL_GetModState();
            ~RestoreInput() {
                gui::guiFocusState() = focus;
                SDL_SetModState(modifiers);
            }
        } restore_input;
        gui::guiFocusState().reset();
        SDL_SetModState(SDL_KMOD_NONE);
        const auto pointer = input::wheelPointerInPixels(nullptr);
        const ViewRect rect{static_cast<int>(std::floor(pointer.x)) - 10,
                            static_cast<int>(std::floor(pointer.y)) - 10, 400, 200};
        Viewport legacy(400, 200);
        ViewportWorkspace workspace({400, 200});
        const auto first = workspace.primaryView();
        const auto second = workspace.split(first, SplitAxis::Vertical);
        ASSERT_TRUE(second);
        auto projection = workspace.findView(first)->projection;
        projection.orthographic = true;
        projection.ortho_scale = 100.0f;
        ASSERT_TRUE(workspace.setViewProjection(first, projection));
        projection.ortho_scale = 200.0f;
        ASSERT_TRUE(workspace.setViewProjection(*second, projection));
        const auto first_position = workspace.findCamera(first)->camera.t;
        const auto second_position = workspace.findCamera(*second)->camera.t;

        InputController controller(nullptr, legacy);
        controller.setTrackpadPreferences({.device = NavigationDevice::Mouse});
        controller.bindWorkspace(&workspace);
        controller.setWorkspaceFrameSnapshot(workspace.snapshot(rect));
        controller.handleScroll(0.0, 1.0);
        const float after_wheel = workspace.findView(first)->projection.ortho_scale;
        EXPECT_NE(after_wheel, 100.0f);
        EXPECT_FLOAT_EQ(workspace.findView(*second)->projection.ortho_scale, 200.0f);

        controller.handlePinch(1.25f);
        EXPECT_NE(workspace.findView(first)->projection.ortho_scale, after_wheel);
        EXPECT_FLOAT_EQ(workspace.findView(*second)->projection.ortho_scale, 200.0f);
        EXPECT_EQ(workspace.findCamera(first)->camera.t, first_position);
        EXPECT_EQ(workspace.findCamera(*second)->camera.t, second_position);
    }
} // namespace lfs::vis

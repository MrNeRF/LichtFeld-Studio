/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "input/input_bindings.hpp"
#include "input/sdl_key_mapping.hpp"
#include <array>
#include <gtest/gtest.h>
#include <set>

namespace lfs::vis::input {
    namespace {
        TEST(KeymapSideMouseTest, SideMouseButtonsHaveDifferentReadableNames) {
            const auto back = static_cast<MouseButton>(3);
            const auto forward = static_cast<MouseButton>(4);
            EXPECT_NE(getMouseButtonName(back), getMouseButtonName(forward));
            EXPECT_NE(getMouseButtonName(back), "Mouse?");
            EXPECT_NE(getMouseButtonName(forward), "Mouse?");
            struct RestorePersistence {
                bool enabled = InputBindings::isPersistenceEnabled();
                ~RestorePersistence() { InputBindings::setPersistenceEnabled(enabled); }
            } restore;
            InputBindings::setPersistenceEnabled(false);
            InputBindings bindings;
            bindings.setBinding(ToolMode::GLOBAL, Action::CAMERA_SET_PIVOT,
                                MouseButtonTrigger{back});
            const auto back_label = bindings.getTriggerDescription(
                Action::CAMERA_SET_PIVOT, ToolMode::GLOBAL);
            bindings.setBinding(ToolMode::GLOBAL, Action::CAMERA_SET_PIVOT,
                                MouseButtonTrigger{forward});
            const auto forward_label = bindings.getTriggerDescription(
                Action::CAMERA_SET_PIVOT, ToolMode::GLOBAL);
            EXPECT_NE(back_label, forward_label);
        }
    } // namespace
} // namespace lfs::vis::input

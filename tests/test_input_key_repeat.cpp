/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/gui_focus_state.hpp"
#include "input/input_controller.hpp"
#include "input/input_router.hpp"
#include "input/key_codes.hpp"
#include "internal/viewport.hpp"
#include "licht_test_support.hpp"
#include <gtest/gtest.h>

namespace lfs::vis {
    namespace {
        class InputKeyRepeatTest : public ::testing::Test {
        protected:
            void SetUp() override {
                input::InputBindings::setPersistenceEnabled(false);
                gui::guiFocusState().reset();
            }

            void TearDown() override {
                gui::guiFocusState().reset();
                input::InputBindings::setPersistenceEnabled(true);
            }

            lfs::test::licht::TemporaryDirectory home_{"lfs-input-key-repeat"};
            lfs::test::licht::ScopedEnvironmentVariable lfs_home_{"LFS_HOME", home_.path.string()};
        };
    } // namespace

    TEST_F(InputKeyRepeatTest, HeldKeyRepeatsOnlyWhenBindingOrActionAsks) {
        Viewport viewport(200, 200);
        InputController controller(nullptr, viewport);
        input::InputRouter router;
        router.setInputController(&controller);
        controller.setInputRouter(&router);
        router.focusViewportKeyboard();

        // Speed controls repeat on a held key by default.
        const float speed0 = viewport.camera.getWasdSpeed();
        controller.handleKey(input::KEY_EQUAL, input::ACTION_PRESS, input::KEYMOD_CTRL);
        const float speed1 = viewport.camera.getWasdSpeed();
        controller.handleKey(input::KEY_EQUAL, input::ACTION_REPEAT, input::KEYMOD_CTRL);
        controller.handleKey(input::KEY_EQUAL, input::ACTION_RELEASE, input::KEYMOD_CTRL);
        EXPECT_GT(speed1, speed0);
        EXPECT_GT(viewport.camera.getWasdSpeed(), speed1);

        // Go Home ignores a held key until its binding asks to repeat.
        const glm::vec3 away(3.0f, 2.0f, 1.0f);
        controller.handleKey(input::KEY_H, input::ACTION_PRESS, 0);
        const glm::vec3 home = viewport.camera.t;
        viewport.camera.t = away;
        controller.handleKey(input::KEY_H, input::ACTION_REPEAT, 0);
        controller.handleKey(input::KEY_H, input::ACTION_RELEASE, 0);
        EXPECT_EQ(viewport.camera.t, away);

        controller.getBindings().setBinding(input::ToolMode::GLOBAL, input::Action::CAMERA_RESET_HOME,
                                            input::KeyTrigger{input::KEY_H, input::MODIFIER_NONE, true});
        controller.handleKey(input::KEY_H, input::ACTION_PRESS, 0);
        viewport.camera.t = away;
        controller.handleKey(input::KEY_H, input::ACTION_REPEAT, 0);
        controller.handleKey(input::KEY_H, input::ACTION_RELEASE, 0);
        EXPECT_EQ(viewport.camera.t, home);
    }
} // namespace lfs::vis

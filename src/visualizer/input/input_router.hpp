/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include <cstdint>

namespace lfs::vis {
    class InputController;
}

namespace lfs::vis::input {

    enum class InputTarget : uint8_t {
        None,
        Gui,
        Viewport,
    };

    struct PointerTargets {
        InputTarget hover_target = InputTarget::None;
        InputTarget pointer_target = InputTarget::None;
    };

    struct InputState {
        InputTarget keyboard_focus = InputTarget::None;
        InputTarget pointer_capture = InputTarget::None;
    };

    class LFS_VIS_API InputRouter {
    public:
        void setInputController(InputController* controller) { input_controller_ = controller; }

        void reset();
        void onWindowFocusLost();
        void beginMouseButton(int action, double x, double y);
        void endMouseButton(int action);
        void syncPressedMouseButtons(bool any_buttons_pressed);
        void focusViewportKeyboard();

        [[nodiscard]] InputTarget hoverTarget(double x, double y) const;
        [[nodiscard]] InputTarget pointerTarget(double x, double y) const;
        [[nodiscard]] PointerTargets pointerTargets(double x, double y) const;
        [[nodiscard]] InputTarget keyboardFocus() const;
        [[nodiscard]] bool isViewportKeyboardFocused() const;
        [[nodiscard]] bool isGuiKeyboardFocused() const;
        [[nodiscard]] bool isTextInputActive() const;
        [[nodiscard]] bool isModalOpen() const;
        [[nodiscard]] const InputState& state() const { return state_; }

    private:
        [[nodiscard]] InputTarget hitTestHoverTarget(double x, double y) const;
        [[nodiscard]] bool isViewportPoint(double x, double y) const;

        InputController* input_controller_ = nullptr;
        InputState state_{};
        int pressed_mouse_buttons_ = 0;
    };

} // namespace lfs::vis::input

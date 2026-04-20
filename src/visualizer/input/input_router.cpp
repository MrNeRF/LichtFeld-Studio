/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "input/input_router.hpp"
#include "core/services.hpp"
#include "gui/gui_focus_state.hpp"
#include "gui/gui_manager.hpp"
#include "input/input_controller.hpp"
#include "input/key_codes.hpp"
#include <algorithm>

namespace lfs::vis::input {

    namespace {
        gui::GuiInputState queryGuiInputState() {
            if (auto* gui = services().guiOrNull()) {
                return gui->inputState();
            }

            const auto& focus = gui::guiFocusState();
            return {
                .has_keyboard_focus = focus.any_item_active || focus.want_capture_keyboard,
                .text_input_active = focus.want_text_input,
                .modal_open = false,
            };
        }
    } // namespace

    void InputRouter::reset() {
        state_ = {};
        pressed_mouse_buttons_ = 0;
    }

    void InputRouter::onWindowFocusLost() {
        reset();
    }

    void InputRouter::beginMouseButton(const int action, const double x, const double y) {
        if (action != ACTION_PRESS) {
            return;
        }

        ++pressed_mouse_buttons_;

        if (state_.pointer_capture == InputTarget::None) {
            state_.pointer_capture = hoverTarget(x, y);
        }

        switch (state_.pointer_capture) {
        case InputTarget::Gui:
            state_.keyboard_focus = InputTarget::Gui;
            break;
        case InputTarget::Viewport:
            state_.keyboard_focus = InputTarget::Viewport;
            break;
        case InputTarget::None:
            if (!isViewportPoint(x, y)) {
                state_.keyboard_focus = InputTarget::None;
            }
            break;
        }
    }

    void InputRouter::endMouseButton(const int action) {
        if (action != ACTION_RELEASE) {
            return;
        }

        pressed_mouse_buttons_ = std::max(pressed_mouse_buttons_ - 1, 0);
        if (pressed_mouse_buttons_ == 0) {
            state_.pointer_capture = InputTarget::None;
        }
    }

    void InputRouter::syncPressedMouseButtons(const bool any_buttons_pressed) {
        if (any_buttons_pressed || pressed_mouse_buttons_ == 0) {
            return;
        }

        pressed_mouse_buttons_ = 0;
        state_.pointer_capture = InputTarget::None;
    }

    void InputRouter::focusViewportKeyboard() {
        state_.keyboard_focus = InputTarget::Viewport;
    }

    InputTarget InputRouter::hitTestHoverTarget(const double x, const double y) const {
        if (auto* gui = services().guiOrNull()) {
            const auto hit = gui->hitTestPointer(x, y);
            if (hit.blocks_pointer || !gui->isPositionInViewport(x, y)) {
                return InputTarget::Gui;
            }
            return InputTarget::Viewport;
        }

        return isViewportPoint(x, y) ? InputTarget::Viewport : InputTarget::None;
    }

    InputTarget InputRouter::hoverTarget(const double x, const double y) const {
        return hitTestHoverTarget(x, y);
    }

    InputTarget InputRouter::pointerTarget(const double x, const double y) const {
        if (state_.pointer_capture != InputTarget::None) {
            return state_.pointer_capture;
        }

        return hitTestHoverTarget(x, y);
    }

    PointerTargets InputRouter::pointerTargets(const double x, const double y) const {
        const auto hover_target = hitTestHoverTarget(x, y);
        return {
            .hover_target = hover_target,
            .pointer_target = state_.pointer_capture != InputTarget::None
                                  ? state_.pointer_capture
                                  : hover_target,
        };
    }

    InputTarget InputRouter::keyboardFocus() const {
        const auto gui_state = queryGuiInputState();
        if (gui_state.modal_open || gui_state.text_input_active) {
            return InputTarget::Gui;
        }
        if (state_.keyboard_focus != InputTarget::None) {
            return state_.keyboard_focus;
        }
        return gui_state.has_keyboard_focus ? InputTarget::Gui : InputTarget::None;
    }

    bool InputRouter::isViewportKeyboardFocused() const {
        return keyboardFocus() == InputTarget::Viewport;
    }

    bool InputRouter::isGuiKeyboardFocused() const {
        return keyboardFocus() == InputTarget::Gui;
    }

    bool InputRouter::isTextInputActive() const {
        return queryGuiInputState().text_input_active;
    }

    bool InputRouter::isModalOpen() const {
        return queryGuiInputState().modal_open;
    }

    bool InputRouter::isViewportPoint(const double x, const double y) const {
        if (auto* gui = services().guiOrNull()) {
            return gui->isPositionInViewport(x, y);
        }
        return input_controller_ && input_controller_->isViewportPoint(x, y);
    }

} // namespace lfs::vis::input

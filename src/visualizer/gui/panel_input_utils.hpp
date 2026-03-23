/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "gui/panel_layout.hpp"
#include "input/frame_input_buffer.hpp"
#include "sequencer/rml_sequencer_panel.hpp"

#include <algorithm>

namespace lfs::vis::gui {

    inline PanelInputState buildPanelInputFromSDL(const FrameInputBuffer& buf) {
        PanelInputState input;
        input.mouse_x = buf.mouse_x;
        input.mouse_y = buf.mouse_y;
        for (int i = 0; i < 3; ++i) {
            input.mouse_down[i] = buf.mouse_down[i];
            input.mouse_clicked[i] = buf.mouse_clicked[i];
            input.mouse_released[i] = buf.mouse_released[i];
        }
        input.mouse_wheel = buf.mouse_wheel;
        input.screen_w = buf.window_w;
        input.screen_h = buf.window_h;
        input.key_ctrl = (buf.key_mods & SDL_KMOD_CTRL) != 0;
        input.key_shift = (buf.key_mods & SDL_KMOD_SHIFT) != 0;
        input.key_alt = (buf.key_mods & SDL_KMOD_ALT) != 0;
        input.key_super = (buf.key_mods & SDL_KMOD_GUI) != 0;
        input.keys_pressed.reserve(buf.keys_pressed.size());
        for (auto sc : buf.keys_pressed)
            input.keys_pressed.push_back(static_cast<int>(sc));
        input.keys_released.reserve(buf.keys_released.size());
        for (auto sc : buf.keys_released)
            input.keys_released.push_back(static_cast<int>(sc));
        input.text_codepoints = buf.text_codepoints;
        input.text_inputs = buf.text_inputs;
        input.text_editing = buf.text_editing;
        input.text_editing_start = buf.text_editing_start;
        input.text_editing_length = buf.text_editing_length;
        input.has_text_editing = buf.has_text_editing;
        return input;
    }

    inline lfs::vis::PanelInputState buildSequencerPanelInputFromSDL(const FrameInputBuffer& buf) {
        lfs::vis::PanelInputState input;
        input.mouse_x = buf.mouse_x;
        input.mouse_y = buf.mouse_y;
        for (int i = 0; i < 3; ++i) {
            input.mouse_down[i] = buf.mouse_down[i];
            input.mouse_clicked[i] = buf.mouse_clicked[i];
            input.mouse_released[i] = buf.mouse_released[i];
        }
        input.mouse_wheel = buf.mouse_wheel;
        input.key_shift = (buf.key_mods & SDL_KMOD_SHIFT) != 0;
        input.key_ctrl = (buf.key_mods & SDL_KMOD_CTRL) != 0;
        input.key_alt = (buf.key_mods & SDL_KMOD_ALT) != 0;
        input.key_super = (buf.key_mods & SDL_KMOD_GUI) != 0;
        input.key_delete_pressed =
            std::find(buf.keys_pressed.begin(), buf.keys_pressed.end(), SDL_SCANCODE_DELETE) !=
            buf.keys_pressed.end();
        input.keys_pressed.reserve(buf.keys_pressed.size());
        for (auto sc : buf.keys_pressed)
            input.keys_pressed.push_back(static_cast<int>(sc));
        input.keys_released.reserve(buf.keys_released.size());
        for (auto sc : buf.keys_released)
            input.keys_released.push_back(static_cast<int>(sc));
        input.text_codepoints = buf.text_codepoints;
        input.text_inputs = buf.text_inputs;
        input.text_editing = buf.text_editing;
        input.text_editing_start = buf.text_editing_start;
        input.text_editing_length = buf.text_editing_length;
        input.has_text_editing = buf.has_text_editing;
        input.screen_w = buf.window_w;
        input.screen_h = buf.window_h;
        return input;
    }

} // namespace lfs::vis::gui

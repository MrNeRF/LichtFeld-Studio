/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "gui/gui_input.hpp"
#include "input/frame_input_buffer.hpp"
#include "sequencer/rml_sequencer_panel.hpp"

#include <algorithm>

namespace lfs::vis::gui {

    inline void buildPanelInputFromSDL(const FrameInputBuffer& buf, PanelInputState& input) {
        input.mouse_x = buf.mouse_x;
        input.mouse_y = buf.mouse_y;
        for (int i = 0; i < 3; ++i) {
            input.mouse_down[i] = buf.mouse_down[i];
            input.mouse_clicked[i] = buf.mouse_clicked[i];
            input.mouse_released[i] = buf.mouse_released[i];
        }
        input.mouse_wheel = buf.mouse_wheel;
        input.mouse_wheel_x = buf.mouse_wheel_x;
        // Copy complete canonical events in order, keeping coordinates, DOWN/UP identity,
        // timestamp, click count and event-time ownership together.
        input.mouse_button_events = buf.mouse_button_events;
        input.screen_w = buf.window_w;
        input.screen_h = buf.window_h;
        input.key_ctrl = (buf.key_mods & SDL_KMOD_CTRL) != 0;
        input.key_shift = (buf.key_mods & SDL_KMOD_SHIFT) != 0;
        input.key_alt = (buf.key_mods & SDL_KMOD_ALT) != 0;
        input.key_super = (buf.key_mods & SDL_KMOD_GUI) != 0;
        input.input_events = buf.input_events;
        input.keys_pressed.clear();
        input.keys_pressed.reserve(buf.keys_pressed.size());
        for (auto sc : buf.keys_pressed)
            input.keys_pressed.push_back(static_cast<int>(sc));
    }

    inline PanelInputState buildPanelInputFromSDL(const FrameInputBuffer& buf) {
        PanelInputState input;
        buildPanelInputFromSDL(buf, input);
        return input;
    }

    inline PanelInputState fromSequencerPanelInput(const lfs::vis::PanelInputState& source) {
        PanelInputState input;
        input.mouse_x = source.mouse_x;
        input.mouse_y = source.mouse_y;
        input.screen_x = source.screen_x;
        input.screen_y = source.screen_y;
        input.screen_w = source.screen_w;
        input.screen_h = source.screen_h;
        for (int i = 0; i < 3; ++i) {
            input.mouse_down[i] = source.mouse_down[i];
            input.mouse_clicked[i] = source.mouse_clicked[i];
            input.mouse_released[i] = source.mouse_released[i];
        }
        input.mouse_wheel = source.mouse_wheel;
        input.key_ctrl = source.key_ctrl;
        input.key_shift = source.key_shift;
        input.key_alt = source.key_alt;
        input.key_super = source.key_super;
        input.input_events = source.input_events;
        input.keys_pressed = source.keys_pressed;
        return input;
    }

    inline lfs::vis::PanelInputState toSequencerPanelInput(const PanelInputState& panel_input) {
        lfs::vis::PanelInputState input;
        input.mouse_x = panel_input.mouse_x;
        input.mouse_y = panel_input.mouse_y;
        input.screen_x = panel_input.screen_x;
        input.screen_y = panel_input.screen_y;
        for (int i = 0; i < 3; ++i) {
            input.mouse_down[i] = panel_input.mouse_down[i];
            input.mouse_clicked[i] = panel_input.mouse_clicked[i];
            input.mouse_released[i] = panel_input.mouse_released[i];
        }
        input.mouse_wheel = panel_input.mouse_wheel;
        input.key_shift = panel_input.key_shift;
        input.key_ctrl = panel_input.key_ctrl;
        input.key_alt = panel_input.key_alt;
        input.key_super = panel_input.key_super;
        input.key_delete_pressed =
            std::find(panel_input.keys_pressed.begin(), panel_input.keys_pressed.end(),
                      SDL_SCANCODE_DELETE) != panel_input.keys_pressed.end();
        input.input_events = panel_input.input_events;
        input.keys_pressed = panel_input.keys_pressed;

        input.screen_w = panel_input.screen_w;
        input.screen_h = panel_input.screen_h;
        return input;
    }

} // namespace lfs::vis::gui

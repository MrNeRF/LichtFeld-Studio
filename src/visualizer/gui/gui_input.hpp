/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once
#include "rendering/view_source.hpp"

#include "input/frame_input_buffer.hpp"
#include <cstdint>
#include <glm/glm.hpp>
#include <string>
#include <vector>

namespace lfs::vis::gui {

    inline constexpr float kStatusBarHeight = 22.0f;

    struct ViewportLayout {
        ViewId view = kNoView;
        glm::vec2 pos{0, 0};
        glm::vec2 size{0, 0};
        bool has_focus = false;
    };

    struct PanelInputState {
        float mouse_x = 0;
        float mouse_y = 0;
        float screen_x = 0;
        float screen_y = 0;
        bool mouse_down[3] = {};
        bool mouse_clicked[3] = {};
        bool mouse_released[3] = {};
        int screen_w = 0;
        int screen_h = 0;
        float mouse_wheel = 0;
        float mouse_wheel_x = 0;
        std::vector<FrameMouseButtonEvent> mouse_button_events;
        bool key_ctrl = false;
        bool key_shift = false;
        bool key_alt = false;
        bool key_super = false;
        bool viewport_keyboard_focus = false;
        std::vector<int> keys_pressed;
        std::vector<FrameInputEvent> input_events;
        void* bg_draw_list = nullptr;
        void* fg_draw_list = nullptr;

        [[nodiscard]] const FrameMouseButtonEvent* lastPress(const int button) const {
            if (button < 0 || button > 2)
                return nullptr;
            for (auto it = mouse_button_events.rbegin(); it != mouse_button_events.rend(); ++it) {
                if (it->down && it->button == static_cast<uint8_t>(button))
                    return &*it;
            }
            return nullptr;
        }
    };

    struct ScreenState {
        glm::vec2 work_pos{0, 0};
        glm::vec2 work_size{0, 0};
        bool any_item_active = false;
    };

} // namespace lfs::vis::gui

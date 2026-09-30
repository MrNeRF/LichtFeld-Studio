/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once
#include "rendering/view_source.hpp"

#include "core/export.hpp"
#include "gui/panel_registry.hpp"
#include "gui/ui_context.hpp"
#include "input/frame_input_buffer.hpp"
#include <cstdint>
#include <glm/glm.hpp>
#include <string>
#include <vector>

namespace lfs::vis::gui {

    struct ViewportLayout {
        ViewId view = kNoView;
        glm::vec2 pos{0, 0};
        glm::vec2 size{0, 0};
        bool has_focus = false;
    };

    enum class CursorRequest : uint8_t { None,
                                         ResizeEW,
                                         ResizeNS };

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
        std::vector<int> keys_repeated;
        std::vector<int> keys_released;
        std::vector<uint32_t> text_codepoints;
        std::vector<std::string> text_inputs;
        std::string text_editing;
        int text_editing_start = -1;
        int text_editing_length = -1;
        bool has_text_editing = false;
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

    struct PanelLayoutProjectState {
        bool show_sequencer = false;
        std::string active_tab_id;
    };

    class LFS_VIS_API PanelLayoutManager {
    public:
        PanelLayoutManager();

        void loadState();
        [[nodiscard]] PanelLayoutProjectState captureProjectState() const;
        void applyProjectState(const PanelLayoutProjectState& state);

        ViewportLayout computeViewportLayout(bool show_main_panel, bool ui_hidden,
                                             bool python_console_visible,
                                             const ScreenState& screen) const;

        bool isResizingPanel() const { return false; }
        bool isResizeInteractionActive() const { return false; }
        CursorRequest getCursorRequest() const { return CursorRequest::None; }

        bool isShowSequencer() const { return show_sequencer_; }
        void setShowSequencer(bool v);

        static constexpr float STATUS_BAR_HEIGHT = 22.0f;
        static constexpr float PANEL_GAP = 2.0f;

    private:
        bool show_sequencer_ = false;
    };

} // namespace lfs::vis::gui

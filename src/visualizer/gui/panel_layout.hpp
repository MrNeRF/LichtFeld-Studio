/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "gui/layout_state.hpp"
#include "gui/panel_registry.hpp"
#include "gui/ui_context.hpp"
#include <cstdint>
#include <glm/glm.hpp>
#include <string>
#include <unordered_map>
#include <vector>

namespace lfs::vis::gui {

    struct ViewportLayout {
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
    };

    struct ScreenState {
        glm::vec2 work_pos{0, 0};
        glm::vec2 work_size{0, 0};
        bool any_item_active = false;
    };

    class PanelLayoutManager {
    public:
        PanelLayoutManager();

        void loadState();
        void saveState() const;

        void renderRightPanel(const UIContext& ctx, const PanelDrawContext& draw_ctx,
                              bool show_main_panel, bool ui_hidden,
                              std::unordered_map<std::string, bool>& window_states,
                              std::string& focus_panel_name,
                              const PanelInputState& input,
                              const ScreenState& screen);

        void renderBottomDock(const PanelDrawContext& draw_ctx, bool show_main_panel,
                              bool ui_hidden, const PanelInputState& input,
                              const ScreenState& screen);

        ViewportLayout computeViewportLayout(bool show_main_panel, bool ui_hidden,
                                             bool python_console_visible,
                                             const ScreenState& screen) const;

        bool isResizingPanel() const {
            return python_console_resizing_ || python_console_hovering_edge_ ||
                   bottom_dock_resizing_ || bottom_dock_hovering_edge_;
        }

        CursorRequest getCursorRequest() const { return cursor_request_; }

        void applyResizeDelta(float dx, const ScreenState& screen);

        float getRightPanelWidth() const { return right_panel_width_; }
        float getScenePanelRatio() const { return scene_panel_ratio_; }
        void setScenePanelRatio(float r) { scene_panel_ratio_ = std::clamp(r, 0.15f, 0.85f); }
        void adjustScenePanelRatio(float delta_y, const ScreenState& screen);
        float getPythonConsoleWidth() const { return python_console_width_; }
        float getBottomDockHeight() const { return bottom_dock_height_; }
        bool isBottomDockVisible() const { return bottom_dock_visible_; }
        float bottomDockTopY() const { return bottom_dock_top_y_; }
        bool isShowSequencer() const { return show_sequencer_; }
        void setShowSequencer(bool v) { show_sequencer_ = v; }

        const std::string& getActiveTab() const { return active_tab_id_; }
        void setActiveTab(const std::string& id) { active_tab_id_ = id; }
        bool syncActiveTab(const std::vector<PanelSummary>& main_tabs,
                           std::string& focus_panel_name);

        static constexpr float SPLITTER_H = 6.0f;
        static constexpr float TAB_BAR_H = 28.0f;
        static constexpr float STATUS_BAR_HEIGHT = 22.0f;
        static constexpr float PANEL_GAP = 2.0f;

    private:
        void renderDockedPythonConsole(const UIContext& ctx, float panel_x, float panel_h,
                                       const PanelInputState& input, const ScreenState& screen);
        float computeViewportWidth(bool show_main_panel, bool ui_hidden,
                                   bool python_console_visible,
                                   const ScreenState& screen) const;
        float computeBottomDockWidth(bool show_main_panel, bool ui_hidden,
                                     const ScreenState& screen) const;
        float computeBottomDockReservedHeight(bool show_main_panel, bool ui_hidden,
                                              const ScreenState& screen) const;

        float right_panel_width_ = 340.0f;
        float scene_panel_ratio_ = 0.4f;

        float python_console_width_ = -1.0f;
        bool python_console_resizing_ = false;
        bool python_console_hovering_edge_ = false;
        float bottom_dock_height_ = 320.0f;
        bool bottom_dock_resizing_ = false;
        bool bottom_dock_hovering_edge_ = false;
        bool bottom_dock_visible_ = false;
        float bottom_dock_top_y_ = -1.0f;

        bool show_sequencer_ = false;
        std::string active_tab_id_;

        float tab_scroll_offset_ = 0.0f;
        float tab_content_total_h_ = 0.0f;
        size_t background_preload_index_ = 0;

        CursorRequest cursor_request_ = CursorRequest::None;
        float prev_mouse_x_ = 0;
        float prev_mouse_y_ = 0;

        static constexpr float RIGHT_PANEL_MIN_RATIO = 0.01f;
        static constexpr float RIGHT_PANEL_MAX_RATIO = 0.99f;
        static constexpr float PYTHON_CONSOLE_MIN_WIDTH = 200.0f;
        static constexpr float PYTHON_CONSOLE_MAX_RATIO = 0.5f;
        static constexpr float BOTTOM_DOCK_MIN_HEIGHT = 180.0f;
        static constexpr float BOTTOM_DOCK_DEFAULT_HEIGHT = 440.0f;
        static constexpr float BOTTOM_DOCK_MAX_RATIO = 0.65f;
        static constexpr float MIN_VIEWPORT_HEIGHT = 140.0f;
    };

} // namespace lfs::vis::gui

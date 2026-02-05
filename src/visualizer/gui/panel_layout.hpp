/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "gui/layout_state.hpp"
#include "gui/ui_context.hpp"
#include <string>
#include <unordered_map>
#include <imgui.h>

namespace lfs::vis::gui {

    struct ViewportLayout {
        ImVec2 pos{0, 0};
        ImVec2 size{0, 0};
        bool has_focus = false;
    };

    class PanelLayoutManager {
    public:
        PanelLayoutManager();

        void loadState();
        void saveState() const;

        void renderRightPanel(const UIContext& ctx, bool show_main_panel, bool ui_hidden,
                              std::unordered_map<std::string, bool>& window_states,
                              std::string& focus_panel_name);

        ViewportLayout computeViewportLayout(bool show_main_panel, bool ui_hidden,
                                             bool python_console_visible) const;

        bool isResizingPanel() const {
            return resizing_panel_ || hovering_panel_edge_ ||
                   python_console_resizing_ || python_console_hovering_edge_;
        }

        float getRightPanelWidth() const { return right_panel_width_; }
        float getScenePanelRatio() const { return scene_panel_ratio_; }
        float getPythonConsoleWidth() const { return python_console_width_; }
        bool isShowSequencer() const { return show_sequencer_; }
        void setShowSequencer(bool v) { show_sequencer_ = v; }

    private:
        void renderDockedPythonConsole(const UIContext& ctx, float panel_x, float panel_h);

        float right_panel_width_ = 300.0f;
        bool resizing_panel_ = false;
        bool hovering_panel_edge_ = false;
        float scene_panel_ratio_ = 0.4f;

        float python_console_width_ = -1.0f;
        bool python_console_resizing_ = false;
        bool python_console_hovering_edge_ = false;

        bool show_sequencer_ = false;

        static constexpr float RIGHT_PANEL_MIN_RATIO = 0.01f;
        static constexpr float RIGHT_PANEL_MAX_RATIO = 0.99f;
        static constexpr float PYTHON_CONSOLE_MIN_WIDTH = 200.0f;
        static constexpr float PYTHON_CONSOLE_MAX_RATIO = 0.5f;
        static constexpr float STATUS_BAR_HEIGHT = 22.0f;
        static constexpr float PANEL_GAP = 2.0f;
    };

} // namespace lfs::vis::gui

/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/panel_layout.hpp"
#include "gui/panels/python_console_panel.hpp"
#include "gui/rmlui/rml_fbo.hpp"
#include "python/python_runtime.hpp"
#include "theme/theme.hpp"
#include "visualizer_impl.hpp"
#include <algorithm>

namespace lfs::vis::gui {

    PanelLayoutManager::PanelLayoutManager() = default;

    void PanelLayoutManager::loadState() {
        LayoutState state;
        state.load();
        // right_panel_width_ intentionally not loaded — always start at default
        scene_panel_ratio_ = state.scene_panel_ratio;
        python_console_width_ = state.python_console_width;
        show_sequencer_ = state.show_sequencer;
    }

    void PanelLayoutManager::saveState() const {
        LayoutState state;
        // right_panel_width not saved — always start at default
        state.scene_panel_ratio = scene_panel_ratio_;
        state.python_console_width = python_console_width_;
        state.show_sequencer = show_sequencer_;
        state.save();
    }

    void PanelLayoutManager::renderRightPanel(const UIContext& ctx, const PanelDrawContext& draw_ctx,
                                              bool show_main_panel, bool ui_hidden,
                                              std::unordered_map<std::string, bool>& window_states,
                                              std::string& focus_panel_name,
                                              const PanelInputState& input,
                                              const ScreenState& screen) {
        cursor_request_ = CursorRequest::None;

        if (!show_main_panel || ui_hidden || screen.work_size.x <= 0 || screen.work_size.y <= 0) {
            python_console_hovering_edge_ = false;
            python_console_resizing_ = false;
            return;
        }

        const float dpi = lfs::python::get_shared_dpi_scale();
        const float panel_h = screen.work_size.y - STATUS_BAR_HEIGHT * dpi;
        const float min_w = screen.work_size.x * RIGHT_PANEL_MIN_RATIO;
        const float max_w = screen.work_size.x * RIGHT_PANEL_MAX_RATIO;

        right_panel_width_ = std::clamp(right_panel_width_, min_w, max_w);

        const bool python_console_visible = window_states["python_console"];
        const float available_for_split = screen.work_size.x - right_panel_width_ - PANEL_GAP;

        if (python_console_visible && python_console_width_ < 0.0f) {
            python_console_width_ = (available_for_split - PANEL_GAP) / 2.0f;
        }

        if (python_console_visible) {
            const float max_console_w = available_for_split - PYTHON_CONSOLE_MIN_WIDTH;
            python_console_width_ = std::clamp(python_console_width_, PYTHON_CONSOLE_MIN_WIDTH, max_console_w);
        }

        const float right_panel_x = screen.work_pos.x + screen.work_size.x - right_panel_width_;
        const float console_x = right_panel_x - (python_console_visible ? python_console_width_ + PANEL_GAP : 0.0f);

        if (python_console_visible) {
            renderDockedPythonConsole(ctx, console_x, panel_h, input, screen);
        } else {
            python_console_hovering_edge_ = false;
            python_console_resizing_ = false;
        }

        const float panel_x = right_panel_x;
        constexpr float PAD = 8.0f;
        const float content_x = panel_x + PAD;
        const float content_w = right_panel_width_ - 2.0f * PAD;
        const float content_top = screen.work_pos.y + PAD;

        const float splitter_h = SPLITTER_H * dpi;
        const float tab_bar_h = TAB_BAR_H * dpi;
        constexpr float MIN_H = 80.0f;
        const float min_h = MIN_H * dpi;
        const float avail_h = panel_h - 2.0f * PAD;

        const float scene_h = std::max(min_h, avail_h * scene_panel_ratio_ - splitter_h * 0.5f);

        auto& reg = PanelRegistry::instance();
        reg.preload_panels_direct(PanelSpace::SceneHeader, content_w, scene_h, draw_ctx,
                                  -1.0f, -1.0f, &input);
        reg.draw_panels_direct(PanelSpace::SceneHeader, content_x, content_top,
                               content_w, scene_h, draw_ctx, &input);

        const auto main_tabs = reg.get_panels_for_space(PanelSpace::MainPanelTab);

        const std::string prev_tab = active_tab_idname_;

        if (!focus_panel_name.empty()) {
            for (const auto& tab : main_tabs) {
                if (focus_panel_name == tab.label || focus_panel_name == tab.idname) {
                    active_tab_idname_ = tab.idname;
                    focus_panel_name.clear();
                    break;
                }
            }
        }

        if (active_tab_idname_.empty() && !main_tabs.empty())
            active_tab_idname_ = main_tabs[0].idname;

        if (active_tab_idname_ != prev_tab)
            tab_scroll_offset_ = 0.0f;

        const float tab_content_y = content_top + scene_h + splitter_h + tab_bar_h;
        const float tab_content_h = std::max(0.0f, content_top + avail_h - tab_content_y);

        const float clip_y_min = tab_content_y;
        const float clip_y_max = tab_content_y + tab_content_h;
        constexpr float kPreloadMaxHeight = 100000.0f;

        const float preloaded_main_h =
            reg.preload_single_panel_direct(active_tab_idname_, content_w, kPreloadMaxHeight, draw_ctx,
                                            clip_y_min, clip_y_max, &input);
        const float preloaded_child_h =
            reg.preload_child_panels_direct(active_tab_idname_, content_w, kPreloadMaxHeight, draw_ctx,
                                            clip_y_min, clip_y_max, &input);
        const float preloaded_total_h = preloaded_main_h + preloaded_child_h;
        const float preloaded_max_scroll =
            std::max(0.0f, preloaded_total_h - tab_content_h);
        tab_scroll_offset_ = std::clamp(tab_scroll_offset_, 0.0f, preloaded_max_scroll);

        const float y_cursor = tab_content_y - tab_scroll_offset_;
        const float main_h = reg.draw_single_panel_direct(active_tab_idname_,
                                                          content_x, y_cursor, content_w, kPreloadMaxHeight, draw_ctx,
                                                          clip_y_min, clip_y_max, &input);
        const float child_h = reg.draw_child_panels_direct(active_tab_idname_,
                                                           content_x, y_cursor + main_h, content_w, kPreloadMaxHeight, draw_ctx,
                                                           clip_y_min, clip_y_max, &input);

        for (size_t attempt = 0; attempt < main_tabs.size(); ++attempt) {
            const size_t idx = (background_preload_index_ + attempt) % main_tabs.size();
            const auto& tab = main_tabs[idx];
            if (tab.idname == active_tab_idname_)
                continue;

            reg.preload_single_panel_direct(tab.idname, content_w, tab_content_h, draw_ctx,
                                            clip_y_min, clip_y_max, &input);
            background_preload_index_ = idx + 1;
            break;
        }

        tab_content_total_h_ = main_h + child_h;

        const float max_scroll = std::max(0.0f, tab_content_total_h_ - tab_content_h);
        tab_scroll_offset_ = std::clamp(tab_scroll_offset_, 0.0f, max_scroll);

        if (input.mouse_x >= content_x && input.mouse_x < content_x + content_w &&
            input.mouse_y >= tab_content_y && input.mouse_y < tab_content_y + tab_content_h) {
            if (input.mouse_wheel != 0.0f) {
                tab_scroll_offset_ -= input.mouse_wheel * 30.0f;
                tab_scroll_offset_ = std::clamp(tab_scroll_offset_, 0.0f, max_scroll);
            }
        }

        if (max_scroll > 0.0f && tab_content_h > 0.0f) {
            auto* dl = static_cast<ImDrawList*>(input.bg_draw_list);
            const auto& t = lfs::vis::theme();
            constexpr float SCROLLBAR_W = 4.0f;
            constexpr float SCROLLBAR_PAD = 2.0f;

            const float track_x = content_x + content_w - SCROLLBAR_W - SCROLLBAR_PAD;
            const float track_y = tab_content_y;
            const float track_h = tab_content_h;

            const float ratio = tab_content_h / tab_content_total_h_;
            const float thumb_h = std::max(20.0f, track_h * ratio);
            const float scroll_frac = tab_scroll_offset_ / max_scroll;
            const float thumb_y = track_y + scroll_frac * (track_h - thumb_h);

            const auto& style = ImGui::GetStyle();
            const ImU32 col = ImGui::ColorConvertFloat4ToU32(style.Colors[ImGuiCol_ScrollbarGrab]);
            const float rounding = t.sizes.scrollbar_rounding;

            dl->AddRectFilled(ImVec2(track_x, thumb_y),
                              ImVec2(track_x + SCROLLBAR_W, thumb_y + thumb_h),
                              col, rounding);
        }
    }

    void PanelLayoutManager::adjustScenePanelRatio(float delta_y, const ScreenState& screen) {
        const float panel_h = screen.work_size.y - STATUS_BAR_HEIGHT * lfs::python::get_shared_dpi_scale();
        const float padding = 16.0f;
        const float avail_h = panel_h - padding;
        if (avail_h > 0)
            scene_panel_ratio_ = std::clamp(scene_panel_ratio_ + delta_y / avail_h, 0.15f, 0.85f);
    }

    void PanelLayoutManager::applyResizeDelta(float dx, const ScreenState& screen) {
        const float min_w = screen.work_size.x * RIGHT_PANEL_MIN_RATIO;
        const float max_w = screen.work_size.x * RIGHT_PANEL_MAX_RATIO;
        right_panel_width_ = std::clamp(right_panel_width_ - dx, min_w, max_w);
    }

    ViewportLayout PanelLayoutManager::computeViewportLayout(bool show_main_panel, bool ui_hidden,
                                                             bool python_console_visible,
                                                             const ScreenState& screen) const {
        float console_w = 0.0f;
        if (python_console_visible && show_main_panel && !ui_hidden) {
            if (python_console_width_ < 0.0f) {
                const float available = screen.work_size.x - right_panel_width_ - PANEL_GAP;
                console_w = (available - PANEL_GAP) / 2.0f + PANEL_GAP;
            } else {
                console_w = python_console_width_ + PANEL_GAP;
            }
        }

        const float w = (show_main_panel && !ui_hidden)
                            ? screen.work_size.x - right_panel_width_ - console_w - PANEL_GAP
                            : screen.work_size.x;
        const float h = ui_hidden ? screen.work_size.y
                                  : screen.work_size.y - STATUS_BAR_HEIGHT * lfs::python::get_shared_dpi_scale();

        ViewportLayout layout;
        layout.pos = {screen.work_pos.x, screen.work_pos.y};
        layout.size = {w, h};
        layout.has_focus = !screen.any_item_active;
        return layout;
    }

    void PanelLayoutManager::renderDockedPythonConsole(const UIContext& ctx, float panel_x, float panel_h,
                                                       const PanelInputState& input, const ScreenState& screen) {
        constexpr float EDGE_GRAB_W = 8.0f;

        const float delta_x = input.mouse_x - prev_mouse_x_;
        prev_mouse_x_ = input.mouse_x;

        python_console_hovering_edge_ = input.mouse_x >= panel_x - EDGE_GRAB_W &&
                                        input.mouse_x <= panel_x + EDGE_GRAB_W &&
                                        input.mouse_y >= screen.work_pos.y &&
                                        input.mouse_y <= screen.work_pos.y + panel_h;

        if (python_console_resizing_ && !input.mouse_down[0])
            python_console_resizing_ = false;

        if (python_console_resizing_) {
            const float max_console_w = screen.work_size.x * PYTHON_CONSOLE_MAX_RATIO;
            python_console_width_ = std::clamp(python_console_width_ - delta_x,
                                               PYTHON_CONSOLE_MIN_WIDTH, max_console_w);
        } else if (python_console_hovering_edge_ && input.mouse_clicked[0]) {
            python_console_resizing_ = true;
        }

        if (python_console_hovering_edge_ || python_console_resizing_)
            cursor_request_ = CursorRequest::ResizeEW;

        panels::DrawDockedPythonConsole(ctx, panel_x, screen.work_pos.y, python_console_width_, panel_h);
    }

} // namespace lfs::vis::gui

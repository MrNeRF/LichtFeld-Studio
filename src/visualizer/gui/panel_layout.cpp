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
#include <imgui.h>

namespace lfs::vis::gui {

    PanelLayoutManager::PanelLayoutManager() = default;

    void PanelLayoutManager::loadState() {
        LayoutState state;
        state.load();
        // right_panel_width_ intentionally not loaded — always start at default
        scene_panel_ratio_ = state.scene_panel_ratio;
        python_console_width_ = state.python_console_width;
        bottom_dock_height_ = state.bottom_dock_height;
        show_sequencer_ = false;
    }

    void PanelLayoutManager::saveState() const {
        LayoutState state;
        // right_panel_width not saved — always start at default
        state.scene_panel_ratio = scene_panel_ratio_;
        state.python_console_width = python_console_width_;
        state.bottom_dock_height = bottom_dock_height_;
        state.show_sequencer = show_sequencer_;
        state.save();
    }

    bool PanelLayoutManager::syncActiveTab(const std::vector<PanelSummary>& main_tabs,
                                           std::string& focus_panel_name) {
        const std::string prev_tab = active_tab_id_;

        if (!focus_panel_name.empty()) {
            const auto focused_tab = std::find_if(
                main_tabs.begin(), main_tabs.end(), [&](const PanelSummary& tab) {
                    return focus_panel_name == tab.label || focus_panel_name == tab.id;
                });
            if (focused_tab != main_tabs.end()) {
                active_tab_id_ = focused_tab->id;
                focus_panel_name.clear();
            }
        }

        const bool active_valid = std::any_of(
            main_tabs.begin(), main_tabs.end(), [&](const PanelSummary& tab) {
                return tab.id == active_tab_id_;
            });
        if (!active_valid)
            active_tab_id_ = main_tabs.empty() ? std::string{} : main_tabs.front().id;

        if (active_tab_id_ == prev_tab)
            return false;

        tab_scroll_offset_ = 0.0f;
        tab_scrollbar_dragging_ = false;
        background_preload_index_ = 0;
        return true;
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
        const float bottom_dock_h = computeBottomDockReservedHeight(show_main_panel, ui_hidden, screen);
        const float min_w = screen.work_size.x * RIGHT_PANEL_MIN_RATIO;
        const float max_w = screen.work_size.x * RIGHT_PANEL_MAX_RATIO;

        right_panel_width_ = std::clamp(right_panel_width_, min_w, max_w);

        auto& reg = PanelRegistry::instance();
        const bool float_blocks_right_panel =
            reg.isPositionOverFloatingPanel(input.mouse_x, input.mouse_y);
        const auto mask_mouse_input = [&](const PanelInputState& src) {
            PanelInputState masked = src;
            masked.mouse_x = -1.0e9f;
            masked.mouse_y = -1.0e9f;
            for (auto& v : masked.mouse_clicked)
                v = false;
            for (auto& v : masked.mouse_released)
                v = false;
            for (auto& v : masked.mouse_down)
                v = false;
            masked.mouse_wheel = 0.0f;
            return masked;
        };
        const PanelInputState masked_panel_input =
            float_blocks_right_panel ? mask_mouse_input(input) : input;

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
            renderDockedPythonConsole(ctx, console_x, std::max(0.0f, panel_h - bottom_dock_h),
                                      masked_panel_input, screen);
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

        reg.preload_panels_direct(PanelSpace::SceneHeader, content_w, scene_h, draw_ctx,
                                  -1.0f, -1.0f, &masked_panel_input);
        reg.draw_panels_direct(PanelSpace::SceneHeader, content_x, content_top,
                               content_w, scene_h, draw_ctx, &masked_panel_input);

        const auto main_tabs = reg.get_panels_for_space(PanelSpace::MainPanelTab);
        syncActiveTab(main_tabs, focus_panel_name);

        const float tab_content_y = content_top + scene_h + splitter_h + tab_bar_h;
        const float tab_content_h = std::max(0.0f, content_top + avail_h - tab_content_y);

        if (active_tab_id_.empty()) {
            tab_content_total_h_ = 0.0f;
            tab_scroll_offset_ = 0.0f;
            tab_scrollbar_dragging_ = false;
            return;
        }

        const float clip_y_min = tab_content_y;
        const float clip_y_max = tab_content_y + tab_content_h;
        constexpr float kPreloadMaxHeight = 100000.0f;

        const float preloaded_main_h =
            reg.preload_single_panel_direct(active_tab_id_, content_w, kPreloadMaxHeight, draw_ctx,
                                            clip_y_min, clip_y_max, &masked_panel_input);
        const float preloaded_child_h =
            reg.preload_child_panels_direct(active_tab_id_, content_w, kPreloadMaxHeight, draw_ctx,
                                            clip_y_min, clip_y_max, &masked_panel_input);
        const float preloaded_total_h = preloaded_main_h + preloaded_child_h;
        const float preloaded_max_scroll =
            std::max(0.0f, preloaded_total_h - tab_content_h);
        tab_scroll_offset_ = std::clamp(tab_scroll_offset_, 0.0f, preloaded_max_scroll);

        const auto& t = lfs::vis::theme();
        const float scrollbar_w = t.sizes.scrollbar_size;
        const float scrollbar_pad = 2.0f;
        const float scrollbar_gutter =
            (preloaded_max_scroll > 0.0f && tab_content_h > 0.0f)
                ? (scrollbar_w + scrollbar_pad * 2.0f)
                : 0.0f;
        const float content_draw_w = std::max(0.0f, content_w - scrollbar_gutter);
        const bool over_tab_content =
            masked_panel_input.mouse_x >= content_x &&
            masked_panel_input.mouse_x < content_x + content_w &&
            masked_panel_input.mouse_y >= tab_content_y &&
            masked_panel_input.mouse_y < tab_content_y + tab_content_h;

        bool suppress_content_input = false;
        if (!masked_panel_input.mouse_down[0])
            tab_scrollbar_dragging_ = false;

        if (preloaded_max_scroll > 0.0f && tab_content_h > 0.0f) {
            const float track_y = tab_content_y;
            const float track_h = tab_content_h;
            const float ratio = tab_content_h / preloaded_total_h;
            const float thumb_h = std::max(t.sizes.grab_min_size, track_h * ratio);
            const float thumb_range = std::max(0.0f, track_h - thumb_h);
            const float scroll_frac = preloaded_max_scroll > 0.0f
                                          ? (tab_scroll_offset_ / preloaded_max_scroll)
                                          : 0.0f;
            const float thumb_y = track_y + scroll_frac * thumb_range;
            const bool over_scrollbar =
                masked_panel_input.mouse_x >= content_x + content_draw_w &&
                masked_panel_input.mouse_x < content_x + content_w &&
                masked_panel_input.mouse_y >= track_y &&
                masked_panel_input.mouse_y < track_y + track_h;

            if (tab_scrollbar_dragging_) {
                suppress_content_input = true;
                if (thumb_range > 0.0f) {
                    const float clamped_thumb_y = std::clamp(
                        masked_panel_input.mouse_y - tab_scrollbar_drag_offset_,
                        track_y,
                        track_y + thumb_range);
                    const float next_frac = (clamped_thumb_y - track_y) / thumb_range;
                    tab_scroll_offset_ = next_frac * preloaded_max_scroll;
                }
            } else if (over_scrollbar && masked_panel_input.mouse_clicked[0]) {
                suppress_content_input = true;
                if (masked_panel_input.mouse_y >= thumb_y &&
                    masked_panel_input.mouse_y <= thumb_y + thumb_h) {
                    tab_scrollbar_dragging_ = true;
                    tab_scrollbar_drag_offset_ = masked_panel_input.mouse_y - thumb_y;
                } else if (thumb_range > 0.0f) {
                    const float target_thumb_y = std::clamp(
                        masked_panel_input.mouse_y - thumb_h * 0.5f,
                        track_y,
                        track_y + thumb_range);
                    const float next_frac = (target_thumb_y - track_y) / thumb_range;
                    tab_scroll_offset_ = next_frac * preloaded_max_scroll;
                    tab_scrollbar_dragging_ = true;
                    tab_scrollbar_drag_offset_ = masked_panel_input.mouse_y - target_thumb_y;
                }
            }
        } else {
            tab_scrollbar_dragging_ = false;
        }

        if (over_tab_content && masked_panel_input.mouse_wheel != 0.0f) {
            tab_scroll_offset_ -= masked_panel_input.mouse_wheel * 30.0f;
            tab_scroll_offset_ = std::clamp(tab_scroll_offset_, 0.0f, preloaded_max_scroll);
        }

        PanelInputState content_input = masked_panel_input;
        if (tab_scrollbar_dragging_ || suppress_content_input) {
            content_input.mouse_down[0] = false;
            content_input.mouse_clicked[0] = false;
            content_input.mouse_released[0] = false;
            content_input.mouse_wheel = 0.0f;
        }

        const float y_cursor = tab_content_y - tab_scroll_offset_;
        const float main_h = reg.draw_single_panel_direct(active_tab_id_,
                                                          content_x, y_cursor, content_draw_w, kPreloadMaxHeight, draw_ctx,
                                                          clip_y_min, clip_y_max, &content_input);
        const float child_h = reg.draw_child_panels_direct(active_tab_id_,
                                                           content_x, y_cursor + main_h, content_draw_w, kPreloadMaxHeight, draw_ctx,
                                                           clip_y_min, clip_y_max, &content_input);

        for (size_t attempt = 0; attempt < main_tabs.size(); ++attempt) {
            const size_t idx = (background_preload_index_ + attempt) % main_tabs.size();
            const auto& tab = main_tabs[idx];
            if (tab.id == active_tab_id_)
                continue;

            reg.preload_single_panel_direct(tab.id, content_draw_w, tab_content_h, draw_ctx,
                                            clip_y_min, clip_y_max, &masked_panel_input);
            background_preload_index_ = idx + 1;
            break;
        }

        tab_content_total_h_ = main_h + child_h;

        const float max_scroll = std::max(0.0f, tab_content_total_h_ - tab_content_h);
        tab_scroll_offset_ = std::clamp(tab_scroll_offset_, 0.0f, max_scroll);

        if (max_scroll > 0.0f && tab_content_h > 0.0f) {
            auto* dl = static_cast<ImDrawList*>(input.bg_draw_list);
            const float track_x = content_x + content_draw_w + scrollbar_pad;
            const float track_y = tab_content_y;
            const float track_h = tab_content_h;

            const float ratio = tab_content_h / tab_content_total_h_;
            const float thumb_h = std::max(t.sizes.grab_min_size, track_h * ratio);
            const float scroll_frac = tab_scroll_offset_ / max_scroll;
            const float thumb_y = track_y + scroll_frac * (track_h - thumb_h);

            const float rounding = std::max(t.sizes.scrollbar_rounding, scrollbar_w * 0.5f);
            const bool over_scrollbar =
                masked_panel_input.mouse_x >= content_x + content_draw_w &&
                masked_panel_input.mouse_x < content_x + content_w &&
                masked_panel_input.mouse_y >= track_y &&
                masked_panel_input.mouse_y < track_y + track_h;

            const ImU32 track_col = ImGui::ColorConvertFloat4ToU32(
                withAlpha(t.palette.background, 0.5f));
            const ImU32 thumb_col = ImGui::ColorConvertFloat4ToU32(
                withAlpha(tab_scrollbar_dragging_
                              ? t.palette.primary
                              : (over_scrollbar ? t.palette.primary_dim : t.palette.text_dim),
                          tab_scrollbar_dragging_ ? 0.9f : (over_scrollbar ? 0.78f : 0.63f)));

            dl->AddRectFilled(ImVec2(track_x, track_y),
                              ImVec2(track_x + scrollbar_w, track_y + track_h),
                              track_col, rounding);
            dl->AddRectFilled(ImVec2(track_x, thumb_y),
                              ImVec2(track_x + scrollbar_w, thumb_y + thumb_h),
                              thumb_col, rounding);
        }
    }

    void PanelLayoutManager::renderBottomDock(const PanelDrawContext& draw_ctx,
                                              const bool show_main_panel,
                                              const bool ui_hidden,
                                              const PanelInputState& input,
                                              const ScreenState& screen) {
        auto& reg = PanelRegistry::instance();
        if (!show_main_panel || ui_hidden || screen.work_size.x <= 0 || screen.work_size.y <= 0 ||
            !reg.has_panels(PanelSpace::BottomDock)) {
            bottom_dock_hovering_edge_ = false;
            bottom_dock_resizing_ = false;
            bottom_dock_visible_ = false;
            bottom_dock_top_y_ = -1.0f;
            prev_mouse_y_ = input.mouse_y;
            return;
        }

        const float dpi = lfs::python::get_shared_dpi_scale();
        const float status_bar_h = STATUS_BAR_HEIGHT * dpi;
        const float panel_w = computeBottomDockWidth(show_main_panel, ui_hidden, screen);
        const float max_panel_h = std::min(
            (screen.work_size.y - status_bar_h) * BOTTOM_DOCK_MAX_RATIO,
            screen.work_size.y - status_bar_h - MIN_VIEWPORT_HEIGHT * dpi);

        if (panel_w <= 0.0f || max_panel_h <= 0.0f) {
            bottom_dock_hovering_edge_ = false;
            bottom_dock_resizing_ = false;
            bottom_dock_visible_ = false;
            bottom_dock_top_y_ = -1.0f;
            prev_mouse_y_ = input.mouse_y;
            return;
        }

        const float min_panel_h = std::min(BOTTOM_DOCK_MIN_HEIGHT * dpi, max_panel_h);
        const float default_panel_h = BOTTOM_DOCK_DEFAULT_HEIGHT * dpi;
        bottom_dock_height_ = std::clamp(
            bottom_dock_height_ > 0.0f ? bottom_dock_height_ : default_panel_h,
            min_panel_h,
            max_panel_h);

        const auto mask_mouse_input = [&](const PanelInputState& src) {
            PanelInputState masked = src;
            masked.mouse_x = -1.0e9f;
            masked.mouse_y = -1.0e9f;
            for (auto& v : masked.mouse_clicked)
                v = false;
            for (auto& v : masked.mouse_released)
                v = false;
            for (auto& v : masked.mouse_down)
                v = false;
            masked.mouse_wheel = 0.0f;
            return masked;
        };

        const bool float_blocks_bottom_dock =
            reg.isPositionOverFloatingPanel(input.mouse_x, input.mouse_y);
        const PanelInputState dock_input =
            float_blocks_bottom_dock ? mask_mouse_input(input) : input;

        const float delta_y = input.mouse_y - prev_mouse_y_;
        prev_mouse_y_ = input.mouse_y;

        if (bottom_dock_resizing_ && !dock_input.mouse_down[0])
            bottom_dock_resizing_ = false;

        const float edge_grab_h = std::max(SPLITTER_H * dpi, 8.0f * dpi);
        float panel_h = bottom_dock_height_;
        float panel_y = screen.work_pos.y + screen.work_size.y - status_bar_h - panel_h;

        bottom_dock_hovering_edge_ =
            !float_blocks_bottom_dock &&
            dock_input.mouse_x >= screen.work_pos.x &&
            dock_input.mouse_x <= screen.work_pos.x + panel_w &&
            dock_input.mouse_y >= panel_y - edge_grab_h &&
            dock_input.mouse_y <= panel_y + edge_grab_h;

        if (bottom_dock_resizing_) {
            bottom_dock_height_ = std::clamp(bottom_dock_height_ - delta_y, min_panel_h, max_panel_h);
        } else if (bottom_dock_hovering_edge_ && dock_input.mouse_clicked[0]) {
            bottom_dock_resizing_ = true;
        }

        if (bottom_dock_hovering_edge_ || bottom_dock_resizing_)
            cursor_request_ = CursorRequest::ResizeNS;

        panel_h = bottom_dock_height_;
        panel_y = screen.work_pos.y + screen.work_size.y - status_bar_h - panel_h;

        const float preloaded_h =
            reg.preload_panels_direct(PanelSpace::BottomDock, panel_w, panel_h, draw_ctx,
                                      panel_y, panel_y + panel_h, &dock_input);
        bottom_dock_visible_ = preloaded_h > 0.0f;
        bottom_dock_top_y_ = bottom_dock_visible_ ? panel_y : -1.0f;
        if (!bottom_dock_visible_)
            return;

        if (auto* dl = static_cast<ImDrawList*>(input.bg_draw_list)) {
            const auto& t = lfs::vis::theme();
            const ImU32 line_col = ImGui::ColorConvertFloat4ToU32(withAlpha(t.palette.border, 0.7f));
            const float grip_y = panel_y;
            dl->AddLine(ImVec2(screen.work_pos.x, grip_y),
                        ImVec2(screen.work_pos.x + panel_w, grip_y),
                        line_col,
                        1.0f);
        }

        reg.draw_panels_direct(PanelSpace::BottomDock,
                               screen.work_pos.x,
                               panel_y,
                               panel_w,
                               panel_h,
                               draw_ctx,
                               &dock_input);
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

    float PanelLayoutManager::computeViewportWidth(const bool show_main_panel,
                                                   const bool ui_hidden,
                                                   const bool python_console_visible,
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

        if (!(show_main_panel && !ui_hidden))
            return screen.work_size.x;

        const float viewport_gap = python_console_visible ? PANEL_GAP : 0.0f;
        return std::max(0.0f, screen.work_size.x - right_panel_width_ - console_w - viewport_gap);
    }

    float PanelLayoutManager::computeBottomDockWidth(const bool show_main_panel,
                                                     const bool ui_hidden,
                                                     const ScreenState& screen) const {
        if (!(show_main_panel && !ui_hidden))
            return screen.work_size.x;
        return std::max(0.0f, screen.work_size.x - right_panel_width_);
    }

    float PanelLayoutManager::computeBottomDockReservedHeight(const bool show_main_panel,
                                                              const bool ui_hidden,
                                                              const ScreenState& screen) const {
        if (!show_main_panel || ui_hidden || !bottom_dock_visible_)
            return 0.0f;

        const float dpi = lfs::python::get_shared_dpi_scale();
        const float status_bar_h = STATUS_BAR_HEIGHT * dpi;
        const float max_panel_h = std::min(
            (screen.work_size.y - status_bar_h) * BOTTOM_DOCK_MAX_RATIO,
            screen.work_size.y - status_bar_h - MIN_VIEWPORT_HEIGHT * dpi);
        if (max_panel_h <= 0.0f)
            return 0.0f;

        const float default_panel_h = BOTTOM_DOCK_DEFAULT_HEIGHT * dpi;
        const float current_h = bottom_dock_height_ > 0.0f ? bottom_dock_height_ : default_panel_h;
        return std::clamp(current_h, 0.0f, max_panel_h);
    }

    ViewportLayout PanelLayoutManager::computeViewportLayout(bool show_main_panel, bool ui_hidden,
                                                             bool python_console_visible,
                                                             const ScreenState& screen) const {
        const float w = computeViewportWidth(show_main_panel, ui_hidden,
                                             python_console_visible, screen);
        const float h = ui_hidden
                            ? screen.work_size.y
                            : screen.work_size.y -
                                  STATUS_BAR_HEIGHT * lfs::python::get_shared_dpi_scale() -
                                  computeBottomDockReservedHeight(show_main_panel, ui_hidden, screen);

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

/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/panel_layout.hpp"
#include "core/logger.hpp"
#include "gui/panels/python_console_panel.hpp"
#include "gui/rml_viewport_overlay.hpp"
#include "python/python_runtime.hpp"
#include "visualizer_impl.hpp"
#include <algorithm>

namespace lfs::vis::gui {
    namespace {
        void drawLeftDockResizeIndicator(const PanelDrawContext& draw_ctx,
                                         const float dpi,
                                         const bool visible,
                                         const bool active) {
            if (!draw_ctx.ui || !draw_ctx.ui->viewport_overlay)
                return;

            const float thickness =
                std::max(active ? 3.0f : 2.0f,
                         (active ? 3.0f : 2.0f) * dpi);
            draw_ctx.ui->viewport_overlay->setLeftDockResizeIndicator(
                visible, active, thickness);
        }
    } // namespace

    PanelLayoutManager::PanelLayoutManager() = default;

    void PanelLayoutManager::loadState() {
        LayoutState state;
        state.load();
        // right_panel_width_ intentionally not loaded — always start at default
        scene_panel_ratio_ = state.scene_panel_ratio;
        python_console_width_ = state.python_console_width;
        bottom_dock_height_ = state.bottom_dock_height;
        left_dock_width_ = state.left_dock_width;
        show_sequencer_ = false;
    }

    void PanelLayoutManager::saveState() const {
        LayoutState state;
        state.load();
        // right_panel_width not saved — always start at default
        state.scene_panel_ratio = scene_panel_ratio_;
        state.python_console_width = python_console_width_;
        state.bottom_dock_height = bottom_dock_height_;
        state.left_dock_width = left_dock_width_;
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
        return true;
    }

    void PanelLayoutManager::renderRightPanel(const UIContext& ctx, const PanelDrawContext& draw_ctx,
                                              bool show_main_panel, bool ui_hidden,
                                              std::unordered_map<std::string, bool>& window_states,
                                              std::string& focus_panel_name,
                                              const PanelInputState& input,
                                              const ScreenState& screen,
                                              const RightPanelRenderDemand demand) {
        LOG_TIMER("gui_render.panel_layout.renderRightPanel");
        cursor_request_ = CursorRequest::None;

        if (!show_main_panel || ui_hidden || screen.work_size.x <= 0 || screen.work_size.y <= 0) {
            python_console_hovering_edge_ = false;
            python_console_resizing_ = false;
            return;
        }

        const float dpi = lfs::python::get_shared_dpi_scale();
        const float panel_h = screen.work_size.y;
        const float bottom_dock_h = computeBottomDockReservedHeight(show_main_panel, ui_hidden, screen);
        const float max_w = maxRightPanelWidth(show_main_panel, ui_hidden, screen);
        const float min_w = std::min(RIGHT_PANEL_MIN_VISIBLE_WIDTH * dpi, max_w);

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
            LOG_TIMER_THRESHOLD("gui_render.panel_layout.right_panel.python_console", 0.25);
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

        if (demand.scene_header_live) {
            {
                LOG_TIMER_THRESHOLD("gui_render.panel_layout.scene_header.preload", 0.25);
                reg.preload_panels_direct(PanelSpace::SceneHeader, content_w, scene_h, draw_ctx,
                                          -1.0f, -1.0f, &masked_panel_input);
            }
            {
                LOG_TIMER_THRESHOLD("gui_render.panel_layout.scene_header.draw", 0.25);
                reg.draw_panels_direct(PanelSpace::SceneHeader, content_x, content_top,
                                       content_w, scene_h, draw_ctx, &masked_panel_input);
            }
        } else {
            LOG_TIMER_THRESHOLD("gui_render.panel_layout.scene_header.draw_cached", 0.25);
            reg.draw_panels_direct_cached(PanelSpace::SceneHeader, content_x, content_top,
                                          content_w, scene_h, draw_ctx, &masked_panel_input);
        }

        std::vector<PanelSummary> main_tabs;
        {
            LOG_TIMER_THRESHOLD("gui_render.panel_layout.main_tabs.lookup", 0.25);
            main_tabs = reg.get_panels_for_space(PanelSpace::MainPanelTab);
        }
        syncActiveTab(main_tabs, focus_panel_name);

        const float tab_content_y = content_top + scene_h + splitter_h + tab_bar_h;
        const float tab_content_h = std::max(0.0f, content_top + avail_h - tab_content_y);

        if (active_tab_id_.empty()) {
            tab_content_total_h_ = 0.0f;
            tab_scroll_offset_ = 0.0f;
            return;
        }

        const float clip_y_min = tab_content_y;
        const float clip_y_max = tab_content_y + tab_content_h;
        constexpr float kPreloadMaxHeight = 100000.0f;

        float scroll_limit = 0.0f;
        if (demand.active_tab_live) {
            float preloaded_main_h = 0.0f;
            float preloaded_child_h = 0.0f;
            {
                LOG_TIMER_THRESHOLD("gui_render.panel_layout.active_tab.preload", 0.25);
                preloaded_main_h =
                    reg.preload_single_panel_direct(active_tab_id_, content_w, kPreloadMaxHeight, draw_ctx,
                                                    clip_y_min, clip_y_max, &masked_panel_input);
            }
            {
                LOG_TIMER_THRESHOLD("gui_render.panel_layout.active_children.preload", 0.25);
                preloaded_child_h =
                    reg.preload_child_panels_direct(active_tab_id_, content_w, kPreloadMaxHeight, draw_ctx,
                                                    clip_y_min, clip_y_max, &masked_panel_input);
            }
            const float preloaded_total_h = preloaded_main_h + preloaded_child_h;
            scroll_limit = std::max(0.0f, preloaded_total_h - tab_content_h);
            tab_scroll_offset_ = std::clamp(tab_scroll_offset_, 0.0f, scroll_limit);
        } else {
            scroll_limit = std::max(0.0f, tab_content_total_h_ - tab_content_h);
            tab_scroll_offset_ = std::clamp(tab_scroll_offset_, 0.0f, scroll_limit);
        }

        const bool over_tab_content =
            masked_panel_input.mouse_x >= content_x &&
            masked_panel_input.mouse_x < content_x + content_w &&
            masked_panel_input.mouse_y >= tab_content_y &&
            masked_panel_input.mouse_y < tab_content_y + tab_content_h;

        if (over_tab_content && masked_panel_input.mouse_wheel != 0.0f) {
            tab_scroll_offset_ -= masked_panel_input.mouse_wheel * 30.0f;
            tab_scroll_offset_ = std::clamp(tab_scroll_offset_, 0.0f, scroll_limit);
        }

        const float y_cursor = tab_content_y - tab_scroll_offset_;
        float main_h = 0.0f;
        float child_h = 0.0f;
        if (demand.active_tab_live) {
            {
                LOG_TIMER_THRESHOLD("gui_render.panel_layout.active_tab.draw", 0.25);
                main_h = reg.draw_single_panel_direct(active_tab_id_,
                                                      content_x, y_cursor, content_w, kPreloadMaxHeight, draw_ctx,
                                                      clip_y_min, clip_y_max, &masked_panel_input);
            }
            {
                LOG_TIMER_THRESHOLD("gui_render.panel_layout.active_children.draw", 0.25);
                child_h = reg.draw_child_panels_direct(active_tab_id_,
                                                       content_x, y_cursor + main_h, content_w, kPreloadMaxHeight, draw_ctx,
                                                       clip_y_min, clip_y_max, &masked_panel_input);
            }
        } else {
            {
                LOG_TIMER_THRESHOLD("gui_render.panel_layout.active_tab.draw_cached", 0.25);
                main_h = reg.draw_single_panel_direct_cached(active_tab_id_,
                                                             content_x, y_cursor, content_w,
                                                             kPreloadMaxHeight, draw_ctx,
                                                             clip_y_min, clip_y_max, &masked_panel_input);
            }
            {
                LOG_TIMER_THRESHOLD("gui_render.panel_layout.active_children.draw_cached", 0.25);
                child_h = reg.draw_child_panels_direct_cached(active_tab_id_,
                                                              content_x, y_cursor + main_h,
                                                              content_w, kPreloadMaxHeight, draw_ctx,
                                                              clip_y_min, clip_y_max, &masked_panel_input);
            }
        }

        tab_content_total_h_ = main_h + child_h;

        const float max_scroll = std::max(0.0f, tab_content_total_h_ - tab_content_h);
        tab_scroll_offset_ = std::clamp(tab_scroll_offset_, 0.0f, max_scroll);
    }

    void PanelLayoutManager::renderRightPanelCached(const UIContext& ctx,
                                                    const PanelDrawContext& draw_ctx,
                                                    bool show_main_panel, bool ui_hidden,
                                                    std::unordered_map<std::string, bool>& window_states,
                                                    std::string& focus_panel_name,
                                                    const PanelInputState& input,
                                                    const ScreenState& screen) {
        LOG_TIMER("gui_render.panel_layout.renderRightPanel.cached");
        cursor_request_ = CursorRequest::None;

        if (!show_main_panel || ui_hidden || screen.work_size.x <= 0 || screen.work_size.y <= 0) {
            python_console_hovering_edge_ = false;
            python_console_resizing_ = false;
            return;
        }

        const float dpi = lfs::python::get_shared_dpi_scale();
        const float panel_h = screen.work_size.y;
        const float bottom_dock_h = computeBottomDockReservedHeight(show_main_panel, ui_hidden, screen);
        const float max_w = maxRightPanelWidth(show_main_panel, ui_hidden, screen);
        const float min_w = std::min(RIGHT_PANEL_MIN_VISIBLE_WIDTH * dpi, max_w);

        right_panel_width_ = std::clamp(right_panel_width_, min_w, max_w);

        const bool python_console_visible = window_states["python_console"];
        const float available_for_split = screen.work_size.x - right_panel_width_ - PANEL_GAP;

        if (python_console_visible && python_console_width_ < 0.0f)
            python_console_width_ = (available_for_split - PANEL_GAP) / 2.0f;

        if (python_console_visible) {
            const float max_console_w = available_for_split - PYTHON_CONSOLE_MIN_WIDTH;
            python_console_width_ = std::clamp(python_console_width_, PYTHON_CONSOLE_MIN_WIDTH, max_console_w);
        }

        const float right_panel_x = screen.work_pos.x + screen.work_size.x - right_panel_width_;
        const float console_x = right_panel_x - (python_console_visible ? python_console_width_ + PANEL_GAP : 0.0f);

        if (python_console_visible) {
            LOG_TIMER_THRESHOLD("gui_render.panel_layout.right_panel.python_console", 0.25);
            renderDockedPythonConsole(ctx, console_x, std::max(0.0f, panel_h - bottom_dock_h),
                                      input, screen);
        } else {
            python_console_hovering_edge_ = false;
            python_console_resizing_ = false;
        }

        constexpr float PAD = 8.0f;
        const float content_x = right_panel_x + PAD;
        const float content_w = right_panel_width_ - 2.0f * PAD;
        const float content_top = screen.work_pos.y + PAD;

        const float splitter_h = SPLITTER_H * dpi;
        const float tab_bar_h = TAB_BAR_H * dpi;
        constexpr float MIN_H = 80.0f;
        const float min_h = MIN_H * dpi;
        const float avail_h = panel_h - 2.0f * PAD;
        const float scene_h = std::max(min_h, avail_h * scene_panel_ratio_ - splitter_h * 0.5f);

        auto& reg = PanelRegistry::instance();
        {
            LOG_TIMER_THRESHOLD("gui_render.panel_layout.scene_header.draw_cached", 0.25);
            reg.draw_panels_direct_cached(PanelSpace::SceneHeader, content_x, content_top,
                                          content_w, scene_h, draw_ctx, &input);
        }

        std::vector<PanelSummary> main_tabs;
        {
            LOG_TIMER_THRESHOLD("gui_render.panel_layout.main_tabs.lookup", 0.25);
            main_tabs = reg.get_panels_for_space(PanelSpace::MainPanelTab);
        }
        syncActiveTab(main_tabs, focus_panel_name);

        const float tab_content_y = content_top + scene_h + splitter_h + tab_bar_h;
        const float tab_content_h = std::max(0.0f, content_top + avail_h - tab_content_y);

        if (active_tab_id_.empty()) {
            tab_content_total_h_ = 0.0f;
            tab_scroll_offset_ = 0.0f;
            return;
        }

        const float clip_y_min = tab_content_y;
        const float clip_y_max = tab_content_y + tab_content_h;
        constexpr float kPreloadMaxHeight = 100000.0f;
        const float max_scroll = std::max(0.0f, tab_content_total_h_ - tab_content_h);
        tab_scroll_offset_ = std::clamp(tab_scroll_offset_, 0.0f, max_scroll);

        const float y_cursor = tab_content_y - tab_scroll_offset_;
        float main_h = 0.0f;
        float child_h = 0.0f;
        {
            LOG_TIMER_THRESHOLD("gui_render.panel_layout.active_tab.draw_cached", 0.25);
            main_h = reg.draw_single_panel_direct_cached(active_tab_id_,
                                                         content_x, y_cursor, content_w,
                                                         kPreloadMaxHeight, draw_ctx,
                                                         clip_y_min, clip_y_max, &input);
        }
        {
            LOG_TIMER_THRESHOLD("gui_render.panel_layout.active_children.draw_cached", 0.25);
            child_h = reg.draw_child_panels_direct_cached(active_tab_id_,
                                                          content_x, y_cursor + main_h,
                                                          content_w, kPreloadMaxHeight, draw_ctx,
                                                          clip_y_min, clip_y_max, &input);
        }

        tab_content_total_h_ = main_h + child_h;
        const float next_max_scroll = std::max(0.0f, tab_content_total_h_ - tab_content_h);
        tab_scroll_offset_ = std::clamp(tab_scroll_offset_, 0.0f, next_max_scroll);
    }

    void PanelLayoutManager::renderBottomDock(const PanelDrawContext& draw_ctx,
                                              const bool show_main_panel,
                                              const bool ui_hidden,
                                              const PanelInputState& input,
                                              const ScreenState& screen) {
        LOG_TIMER("gui_render.panel_layout.renderBottomDock");
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
        const float panel_w = computeBottomDockWidth(show_main_panel, ui_hidden, screen);
        const float max_panel_h = std::min(
            screen.work_size.y * BOTTOM_DOCK_MAX_RATIO,
            screen.work_size.y - MIN_VIEWPORT_HEIGHT * dpi);

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
        float panel_y = screen.work_pos.y + screen.work_size.y - panel_h;

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
        panel_y = screen.work_pos.y + screen.work_size.y - panel_h;

        float preloaded_h = 0.0f;
        {
            LOG_TIMER_THRESHOLD("gui_render.panel_layout.bottom_dock.preload", 0.25);
            preloaded_h =
                reg.preload_panels_direct(PanelSpace::BottomDock, panel_w, panel_h, draw_ctx,
                                          panel_y, panel_y + panel_h, &dock_input);
        }
        bottom_dock_visible_ = preloaded_h > 0.0f;
        bottom_dock_top_y_ = bottom_dock_visible_ ? panel_y : -1.0f;
        if (!bottom_dock_visible_)
            return;

        {
            LOG_TIMER_THRESHOLD("gui_render.panel_layout.bottom_dock.draw", 0.25);
            reg.draw_panels_direct(PanelSpace::BottomDock,
                                   screen.work_pos.x,
                                   panel_y,
                                   panel_w,
                                   panel_h,
                                   draw_ctx,
                                   &dock_input);
        }
    }

    void PanelLayoutManager::renderBottomDockCached(const PanelDrawContext& draw_ctx,
                                                    const bool show_main_panel,
                                                    const bool ui_hidden,
                                                    const PanelInputState& input,
                                                    const ScreenState& screen) {
        LOG_TIMER("gui_render.panel_layout.renderBottomDock.cached");
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
        const float panel_w = computeBottomDockWidth(show_main_panel, ui_hidden, screen);
        const float max_panel_h = std::min(
            screen.work_size.y * BOTTOM_DOCK_MAX_RATIO,
            screen.work_size.y - MIN_VIEWPORT_HEIGHT * dpi);

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

        const float panel_h = bottom_dock_height_;
        const float panel_y = screen.work_pos.y + screen.work_size.y - panel_h;
        const float drawn_h = reg.draw_panels_direct_cached(PanelSpace::BottomDock,
                                                            screen.work_pos.x,
                                                            panel_y,
                                                            panel_w,
                                                            panel_h,
                                                            draw_ctx,
                                                            &input);
        bottom_dock_visible_ = drawn_h > 0.0f;
        bottom_dock_top_y_ = bottom_dock_visible_ ? panel_y : -1.0f;
        prev_mouse_y_ = input.mouse_y;
    }

    void PanelLayoutManager::renderLeftDock(const PanelDrawContext& draw_ctx,
                                            const bool show_main_panel,
                                            const bool ui_hidden,
                                            const PanelInputState& input,
                                            const ScreenState& screen) {
        LOG_TIMER("gui_render.panel_layout.renderLeftDock");
        auto& reg = PanelRegistry::instance();
        if (!show_main_panel || ui_hidden || screen.work_size.x <= 0 || screen.work_size.y <= 0 ||
            !reg.has_panels(PanelSpace::LeftDock)) {
            drawLeftDockResizeIndicator(draw_ctx, 1.0f, false, false);
            left_dock_hovering_edge_ = false;
            left_dock_resizing_ = false;
            left_dock_visible_ = false;
            prev_mouse_x_ = input.mouse_x;
            return;
        }

        const float dpi = lfs::python::get_shared_dpi_scale();
        const float icon_bar_w = ICON_BAR_WIDTH * dpi;
        const float panel_h = screen.work_size.y;
        const float max_panel_w = maxLeftDockPanelWidth(show_main_panel, ui_hidden, screen);

        if (max_panel_w <= 0.0f) {
            drawLeftDockResizeIndicator(draw_ctx, dpi, false, false);
            left_dock_hovering_edge_ = false;
            left_dock_resizing_ = false;
            left_dock_visible_ = false;
            prev_mouse_x_ = input.mouse_x;
            return;
        }

        const float min_panel_w = std::min(LEFT_DOCK_MIN_WIDTH * dpi, max_panel_w);
        const float default_panel_w = LEFT_DOCK_DEFAULT_WIDTH * dpi;
        left_dock_width_ = std::clamp(
            left_dock_width_ > 0.0f ? left_dock_width_ : default_panel_w,
            min_panel_w,
            max_panel_w);

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

        const bool float_blocks_left_dock =
            reg.isPositionOverFloatingPanel(input.mouse_x, input.mouse_y);
        const PanelInputState dock_input =
            float_blocks_left_dock ? mask_mouse_input(input) : input;

        const float delta_x = input.mouse_x - prev_mouse_x_;
        prev_mouse_x_ = input.mouse_x;

        if (left_dock_resizing_ && !dock_input.mouse_down[0])
            left_dock_resizing_ = false;

        const float edge_grab_w = std::max(SPLITTER_H * dpi, 8.0f * dpi);
        float panel_w = left_dock_width_;
        float panel_x = screen.work_pos.x + icon_bar_w;
        float panel_right_x = panel_x + panel_w;

        left_dock_hovering_edge_ =
            !float_blocks_left_dock &&
            dock_input.mouse_x >= panel_right_x - edge_grab_w &&
            dock_input.mouse_x <= panel_right_x + edge_grab_w &&
            dock_input.mouse_y >= screen.work_pos.y &&
            dock_input.mouse_y <= screen.work_pos.y + panel_h;

        if (left_dock_resizing_) {
            left_dock_width_ = std::clamp(left_dock_width_ + delta_x, min_panel_w, max_panel_w);
        } else if (left_dock_hovering_edge_ && dock_input.mouse_clicked[0]) {
            left_dock_resizing_ = true;
        }

        if (left_dock_hovering_edge_ || left_dock_resizing_)
            cursor_request_ = CursorRequest::ResizeEW;

        panel_w = left_dock_width_;
        panel_right_x = panel_x + panel_w;

        float preloaded_h = 0.0f;
        {
            LOG_TIMER_THRESHOLD("gui_render.panel_layout.left_dock.preload", 0.25);
            preloaded_h =
                reg.preload_panels_direct(PanelSpace::LeftDock, panel_w, panel_h, draw_ctx,
                                          screen.work_pos.y, screen.work_pos.y + panel_h, &dock_input);
        }
        left_dock_visible_ = preloaded_h > 0.0f;
        if (!left_dock_visible_) {
            drawLeftDockResizeIndicator(draw_ctx, dpi, false, false);
            return;
        }

        {
            LOG_TIMER_THRESHOLD("gui_render.panel_layout.left_dock.draw", 0.25);
            reg.draw_panels_direct(PanelSpace::LeftDock,
                                   panel_x,
                                   screen.work_pos.y,
                                   panel_w,
                                   panel_h,
                                   draw_ctx,
                                   &dock_input);
        }

        drawLeftDockResizeIndicator(
            draw_ctx, dpi,
            left_dock_hovering_edge_ || left_dock_resizing_,
            left_dock_resizing_);
    }

    void PanelLayoutManager::renderLeftDockCached(const PanelDrawContext& draw_ctx,
                                                  const bool show_main_panel,
                                                  const bool ui_hidden,
                                                  const PanelInputState& input,
                                                  const ScreenState& screen) {
        LOG_TIMER("gui_render.panel_layout.renderLeftDock.cached");
        drawLeftDockResizeIndicator(draw_ctx, 1.0f, false, false);
        auto& reg = PanelRegistry::instance();
        if (!show_main_panel || ui_hidden || screen.work_size.x <= 0 || screen.work_size.y <= 0 ||
            !reg.has_panels(PanelSpace::LeftDock)) {
            left_dock_hovering_edge_ = false;
            left_dock_resizing_ = false;
            left_dock_visible_ = false;
            prev_mouse_x_ = input.mouse_x;
            return;
        }

        const float dpi = lfs::python::get_shared_dpi_scale();
        const float icon_bar_w = ICON_BAR_WIDTH * dpi;
        const float panel_h = screen.work_size.y;
        const float max_panel_w = maxLeftDockPanelWidth(show_main_panel, ui_hidden, screen);

        if (max_panel_w <= 0.0f) {
            left_dock_hovering_edge_ = false;
            left_dock_resizing_ = false;
            left_dock_visible_ = false;
            prev_mouse_x_ = input.mouse_x;
            return;
        }

        const float min_panel_w = std::min(LEFT_DOCK_MIN_WIDTH * dpi, max_panel_w);
        const float default_panel_w = LEFT_DOCK_DEFAULT_WIDTH * dpi;
        left_dock_width_ = std::clamp(
            left_dock_width_ > 0.0f ? left_dock_width_ : default_panel_w,
            min_panel_w,
            max_panel_w);

        const float panel_w = left_dock_width_;
        const float panel_x = screen.work_pos.x + icon_bar_w;
        const float drawn_h = reg.draw_panels_direct_cached(PanelSpace::LeftDock,
                                                            panel_x,
                                                            screen.work_pos.y,
                                                            panel_w,
                                                            panel_h,
                                                            draw_ctx,
                                                            &input);
        left_dock_visible_ = drawn_h > 0.0f;
        prev_mouse_x_ = input.mouse_x;
    }

    void PanelLayoutManager::adjustScenePanelRatio(float delta_y, const ScreenState& screen) {
        const float panel_h = screen.work_size.y;
        const float padding = 16.0f;
        const float avail_h = panel_h - padding;
        if (avail_h > 0)
            scene_panel_ratio_ = std::clamp(scene_panel_ratio_ + delta_y / avail_h, 0.15f, 0.85f);
    }

    void PanelLayoutManager::applyResizeDelta(float dx, const ScreenState& screen) {
        const float max_w = maxRightPanelWidth(true, false, screen);
        const float min_w = std::min(RIGHT_PANEL_MIN_VISIBLE_WIDTH * lfs::python::get_shared_dpi_scale(), max_w);
        right_panel_width_ = std::clamp(right_panel_width_ - dx, min_w, max_w);
    }

    float PanelLayoutManager::maxRightPanelWidth(const bool show_main_panel,
                                                 const bool ui_hidden,
                                                 const ScreenState& screen) const {
        if (!(show_main_panel && !ui_hidden))
            return screen.work_size.x;

        const float dpi = lfs::python::get_shared_dpi_scale();
        const float icon_bar_w = ICON_BAR_WIDTH * dpi;
        const float viewport_min_w = MIN_VIEWPORT_WIDTH * dpi;
        const float right_min_w = RIGHT_PANEL_MIN_VISIBLE_WIDTH * dpi;
        const float panel_budget = std::max(0.0f, screen.work_size.x - icon_bar_w - viewport_min_w - PANEL_GAP);
        const float left_w = shouldReserveLeftDockWidth()
                                 ? std::min(std::max(0.0f, left_dock_width_),
                                            std::max(0.0f, panel_budget - right_min_w))
                                 : 0.0f;
        const float reserved_w = icon_bar_w + left_w + viewport_min_w + PANEL_GAP;
        const float effective_min_w = std::min(right_min_w, panel_budget);
        return std::max(effective_min_w,
                        std::min(screen.work_size.x * RIGHT_PANEL_MAX_RATIO,
                                 screen.work_size.x - reserved_w));
    }

    float PanelLayoutManager::maxLeftDockPanelWidth(const bool show_main_panel,
                                                    const bool ui_hidden,
                                                    const ScreenState& screen) const {
        if (!(show_main_panel && !ui_hidden))
            return 0.0f;

        const float dpi = lfs::python::get_shared_dpi_scale();
        const float icon_bar_w = ICON_BAR_WIDTH * dpi;
        const float viewport_min_w = MIN_VIEWPORT_WIDTH * dpi;
        const float panel_budget = std::max(0.0f, screen.work_size.x - icon_bar_w - viewport_min_w - PANEL_GAP);
        const float right_min_w = std::min(RIGHT_PANEL_MIN_VISIBLE_WIDTH * dpi, panel_budget);
        const float right_w = std::clamp(right_panel_width_,
                                         right_min_w,
                                         std::max(right_min_w, panel_budget));
        return std::max(0.0f, panel_budget - right_w);
    }

    void PanelLayoutManager::enforceWidthConstraints(const bool show_main_panel,
                                                     const bool ui_hidden,
                                                     const ScreenState& screen) {
        if (!(show_main_panel && !ui_hidden) || screen.work_size.x <= 0.0f)
            return;

        const float dpi = lfs::python::get_shared_dpi_scale();
        const float icon_bar_w = ICON_BAR_WIDTH * dpi;
        const float viewport_min_w = MIN_VIEWPORT_WIDTH * dpi;
        const float panel_budget = std::max(0.0f, screen.work_size.x - icon_bar_w - viewport_min_w - PANEL_GAP);
        const float right_min_w = std::min(RIGHT_PANEL_MIN_VISIBLE_WIDTH * dpi, panel_budget);
        const float right_pref_w = std::max(right_panel_width_, right_min_w);

        if (shouldReserveLeftDockWidth()) {
            const float left_pref_w = std::max(0.0f, left_dock_width_);
            const float left_soft_min_w = std::min(LEFT_DOCK_MIN_VISIBLE_WIDTH * dpi,
                                                   std::max(0.0f, panel_budget - right_min_w));
            float left_max_w = std::max(0.0f, panel_budget - right_min_w);
            if (right_pref_w + left_pref_w <= panel_budget) {
                left_dock_width_ = left_pref_w;
                right_panel_width_ = right_pref_w;
            } else {
                left_dock_width_ = std::clamp(left_pref_w, 0.0f, left_max_w);
                if (left_dock_width_ < left_soft_min_w &&
                    panel_budget - left_soft_min_w >= right_min_w) {
                    left_dock_width_ = left_soft_min_w;
                }
                right_panel_width_ = std::clamp(right_pref_w,
                                                right_min_w,
                                                std::max(right_min_w, panel_budget - left_dock_width_));
                left_max_w = std::max(0.0f, panel_budget - right_panel_width_);
                left_dock_width_ = std::min(left_dock_width_, left_max_w);
            }
        } else {
            right_panel_width_ = std::clamp(right_pref_w,
                                            right_min_w,
                                            std::max(right_min_w, panel_budget));
        }
    }

    bool PanelLayoutManager::shouldReserveLeftDockWidth() const {
        return PanelRegistry::instance().has_panels(PanelSpace::LeftDock);
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
        const float max_panel_h = std::min(
            screen.work_size.y * BOTTOM_DOCK_MAX_RATIO,
            screen.work_size.y - MIN_VIEWPORT_HEIGHT * dpi);
        if (max_panel_h <= 0.0f)
            return 0.0f;

        const float default_panel_h = BOTTOM_DOCK_DEFAULT_HEIGHT * dpi;
        const float current_h = bottom_dock_height_ > 0.0f ? bottom_dock_height_ : default_panel_h;
        return std::clamp(current_h, 0.0f, max_panel_h);
    }

    float PanelLayoutManager::computeLeftDockReservedWidth(const bool show_main_panel,
                                                           const bool ui_hidden,
                                                           const ScreenState& screen) const {
        const float dpi = lfs::python::get_shared_dpi_scale();
        const float icon_bar_w = ICON_BAR_WIDTH * dpi;

        if (!show_main_panel || ui_hidden)
            return 0.0f;

        if (!left_dock_visible_)
            return 0.0f;

        const float max_panel_w = maxLeftDockPanelWidth(show_main_panel, ui_hidden, screen);
        if (max_panel_w <= 0.0f)
            return icon_bar_w;

        const float default_panel_w = LEFT_DOCK_DEFAULT_WIDTH * dpi;
        const float current_w = left_dock_width_ > 0.0f ? left_dock_width_ : default_panel_w;
        return std::clamp(current_w, 0.0f, max_panel_w) + icon_bar_w;
    }

    ViewportLayout PanelLayoutManager::computeViewportLayout(bool show_main_panel, bool ui_hidden,
                                                             bool python_console_visible,
                                                             const ScreenState& screen) const {
        const float w = computeViewportWidth(show_main_panel, ui_hidden,
                                             python_console_visible, screen);
        const float h = ui_hidden
                            ? screen.work_size.y
                            : screen.work_size.y -
                                  computeBottomDockReservedHeight(show_main_panel, ui_hidden, screen);

        const float left_w = computeLeftDockReservedWidth(show_main_panel, ui_hidden, screen);

        ViewportLayout layout;
        layout.pos = {screen.work_pos.x + left_w, screen.work_pos.y};
        layout.size = {std::max(0.0f, w - left_w), h};
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

        panels::DrawDockedPythonConsole(ctx, panel_x, screen.work_pos.y, python_console_width_, panel_h, &input);
    }

} // namespace lfs::vis::gui

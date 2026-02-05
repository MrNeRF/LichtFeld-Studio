/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/panel_layout.hpp"
#include "gui/panels/python_console_panel.hpp"
#include "python/python_runtime.hpp"
#include "scene/scene_manager.hpp"
#include "theme/theme.hpp"
#include "visualizer_impl.hpp"
#include <imgui.h>

namespace lfs::vis::gui {

    PanelLayoutManager::PanelLayoutManager() = default;

    void PanelLayoutManager::loadState() {
        LayoutState state;
        state.load();
        right_panel_width_ = state.right_panel_width;
        scene_panel_ratio_ = state.scene_panel_ratio;
        python_console_width_ = state.python_console_width;
        show_sequencer_ = state.show_sequencer;
    }

    void PanelLayoutManager::saveState() const {
        LayoutState state;
        state.right_panel_width = right_panel_width_;
        state.scene_panel_ratio = scene_panel_ratio_;
        state.python_console_width = python_console_width_;
        state.show_sequencer = show_sequencer_;
        state.save();
    }

    void PanelLayoutManager::renderRightPanel(const UIContext& ctx, bool show_main_panel, bool ui_hidden,
                                              std::unordered_map<std::string, bool>& window_states,
                                              std::string& focus_panel_name) {
        if (!show_main_panel || ui_hidden) {
            hovering_panel_edge_ = false;
            resizing_panel_ = false;
            python_console_hovering_edge_ = false;
            python_console_resizing_ = false;
            return;
        }

        const auto* const vp = ImGui::GetMainViewport();
        const float panel_h = vp->WorkSize.y - STATUS_BAR_HEIGHT;
        const float min_w = vp->WorkSize.x * RIGHT_PANEL_MIN_RATIO;
        const float max_w = vp->WorkSize.x * RIGHT_PANEL_MAX_RATIO;

        right_panel_width_ = std::clamp(right_panel_width_, min_w, max_w);

        const bool python_console_visible = window_states["python_console"];
        const float available_for_split = vp->WorkSize.x - right_panel_width_ - PANEL_GAP;

        if (python_console_visible && python_console_width_ < 0.0f) {
            python_console_width_ = (available_for_split - PANEL_GAP) / 2.0f;
        }

        if (python_console_visible) {
            const float max_console_w = available_for_split - PYTHON_CONSOLE_MIN_WIDTH;
            python_console_width_ = std::clamp(python_console_width_, PYTHON_CONSOLE_MIN_WIDTH, max_console_w);
        }

        const float right_panel_x = vp->WorkPos.x + vp->WorkSize.x - right_panel_width_;
        const float console_x = right_panel_x - (python_console_visible ? python_console_width_ + PANEL_GAP : 0.0f);

        if (python_console_visible) {
            renderDockedPythonConsole(ctx, console_x, panel_h);
        } else {
            python_console_hovering_edge_ = false;
            python_console_resizing_ = false;
        }

        const float panel_x = right_panel_x;
        ImGui::SetNextWindowPos({panel_x, vp->WorkPos.y}, ImGuiCond_Always);
        ImGui::SetNextWindowSize({right_panel_width_, panel_h}, ImGuiCond_Always);

        constexpr ImGuiWindowFlags PANEL_FLAGS =
            ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse |
            ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoDocking |
            ImGuiWindowFlags_NoTitleBar;

        const auto& t = theme();
        ImGui::PushStyleColor(ImGuiCol_WindowBg, withAlpha(t.palette.surface, 0.95f));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {8.0f, 8.0f});

        constexpr float EDGE_GRAB_W = 8.0f;
        const auto& io = ImGui::GetIO();
        hovering_panel_edge_ = io.MousePos.x >= panel_x - EDGE_GRAB_W &&
                               io.MousePos.x <= panel_x + EDGE_GRAB_W &&
                               io.MousePos.y >= vp->WorkPos.y &&
                               io.MousePos.y <= vp->WorkPos.y + panel_h;

        if (hovering_panel_edge_ && ImGui::IsMouseClicked(ImGuiMouseButton_Left))
            resizing_panel_ = true;
        if (resizing_panel_ && !ImGui::IsMouseDown(ImGuiMouseButton_Left))
            resizing_panel_ = false;
        if (resizing_panel_) {
            right_panel_width_ = std::clamp(right_panel_width_ - io.MouseDelta.x, min_w, max_w);
        }
        if (hovering_panel_edge_ || resizing_panel_)
            ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);

        if (ImGui::Begin("##RightPanel", nullptr, PANEL_FLAGS)) {
            ImGui::PushStyleColor(ImGuiCol_ChildBg, {0, 0, 0, 0});
            auto* const sm = ctx.viewer->getSceneManager();
            lfs::vis::Scene* scene = sm ? &sm->getScene() : nullptr;

            const float avail_h = ImGui::GetContentRegionAvail().y;
            const float dpi = lfs::python::get_shared_dpi_scale();
            constexpr float SPLITTER_H = 6.0f;
            constexpr float MIN_H = 80.0f;
            const float splitter_h = SPLITTER_H * dpi;
            const float min_h = MIN_H * dpi;

            const float scene_h = std::max(min_h, avail_h * scene_panel_ratio_ - splitter_h * 0.5f);
            if (ImGui::BeginChild("##ScenePanel", {0, scene_h}, ImGuiChildFlags_None, ImGuiWindowFlags_NoBackground)) {
                python::draw_python_panels(python::PanelSpace::SceneHeader, scene);
            }
            ImGui::EndChild();

            ImGui::PushStyleColor(ImGuiCol_Button, withAlpha(t.palette.border, 0.4f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, withAlpha(t.palette.info, 0.6f));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, withAlpha(t.palette.info, 0.8f));
            ImGui::Button("##SceneSplitter", {-1, splitter_h});
            if (ImGui::IsItemActive()) {
                scene_panel_ratio_ = std::clamp(scene_panel_ratio_ + ImGui::GetIO().MouseDelta.y / avail_h, 0.15f, 0.85f);
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNS);
            }
            ImGui::PopStyleColor(3);

            const auto main_tabs = python::get_main_panel_tabs();
            if (ImGui::BeginTabBar("##MainPanelTabs")) {
                for (const auto& tab : main_tabs) {
                    ImGuiTabItemFlags flags = ImGuiTabItemFlags_None;
                    if (focus_panel_name == tab.label || focus_panel_name == tab.idname) {
                        flags = ImGuiTabItemFlags_SetSelected;
                        focus_panel_name.clear();
                    }
                    const std::string tab_label = tab.label + "##" + tab.idname;
                    if (ImGui::BeginTabItem(tab_label.c_str(), nullptr, flags)) {
                        const std::string child_id = "##" + tab.idname + "Panel";
                        if (ImGui::BeginChild(child_id.c_str(), {0, 0}, ImGuiChildFlags_None, ImGuiWindowFlags_NoBackground)) {
                            python::draw_main_panel_tab(tab.idname, scene);
                        }
                        ImGui::EndChild();
                        ImGui::EndTabItem();
                    }
                }
                ImGui::EndTabBar();
            }
            ImGui::PopStyleColor();
        }
        ImGui::End();
        ImGui::PopStyleVar();
        ImGui::PopStyleColor();
    }

    ViewportLayout PanelLayoutManager::computeViewportLayout(bool show_main_panel, bool ui_hidden,
                                                             bool python_console_visible) const {
        const auto* const vp = ImGui::GetMainViewport();

        float console_w = 0.0f;
        if (python_console_visible && show_main_panel && !ui_hidden) {
            if (python_console_width_ < 0.0f) {
                const float available = vp->WorkSize.x - right_panel_width_ - PANEL_GAP;
                console_w = (available - PANEL_GAP) / 2.0f + PANEL_GAP;
            } else {
                console_w = python_console_width_ + PANEL_GAP;
            }
        }

        const float w = (show_main_panel && !ui_hidden)
                            ? vp->WorkSize.x - right_panel_width_ - console_w - PANEL_GAP
                            : vp->WorkSize.x;
        const float h = ui_hidden ? vp->WorkSize.y : vp->WorkSize.y - STATUS_BAR_HEIGHT;

        ViewportLayout layout;
        layout.pos = {vp->WorkPos.x, vp->WorkPos.y};
        layout.size = {w, h};
        layout.has_focus = !ImGui::IsAnyItemActive();
        return layout;
    }

    void PanelLayoutManager::renderDockedPythonConsole(const UIContext& ctx, float panel_x, float panel_h) {
        const auto* const vp = ImGui::GetMainViewport();
        const auto& io = ImGui::GetIO();
        constexpr float EDGE_GRAB_W = 8.0f;

        python_console_hovering_edge_ = io.MousePos.x >= panel_x - EDGE_GRAB_W &&
                                        io.MousePos.x <= panel_x + EDGE_GRAB_W &&
                                        io.MousePos.y >= vp->WorkPos.y &&
                                        io.MousePos.y <= vp->WorkPos.y + panel_h;

        if (python_console_hovering_edge_ && ImGui::IsMouseClicked(ImGuiMouseButton_Left))
            python_console_resizing_ = true;
        if (python_console_resizing_ && !ImGui::IsMouseDown(ImGuiMouseButton_Left))
            python_console_resizing_ = false;
        if (python_console_resizing_) {
            const float max_console_w = vp->WorkSize.x * PYTHON_CONSOLE_MAX_RATIO;
            python_console_width_ = std::clamp(python_console_width_ - io.MouseDelta.x,
                                               PYTHON_CONSOLE_MIN_WIDTH, max_console_w);
        }
        if (python_console_hovering_edge_ || python_console_resizing_)
            ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);

        const ImVec2 pos{panel_x, vp->WorkPos.y};
        const ImVec2 size{python_console_width_, panel_h};
        panels::DrawDockedPythonConsole(ctx, pos, size);
    }

} // namespace lfs::vis::gui

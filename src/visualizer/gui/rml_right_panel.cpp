/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// clang-format off
#include <glad/glad.h>
// clang-format on

#include "gui/rml_right_panel.hpp"
#include "core/logger.hpp"
#include "gui/panel_layout.hpp"
#include "gui/rmlui/rml_theme.hpp"
#include "gui/rmlui/rmlui_manager.hpp"
#include "gui/rmlui/rmlui_render_interface.hpp"
#include "internal/resource_paths.hpp"
#include "theme/theme.hpp"

#include <RmlUi/Core.h>
#include <RmlUi/Core/Element.h>
#include <RmlUi/Core/ElementUtilities.h>
#include <cassert>
#include <format>
#include <imgui.h>

namespace lfs::vis::gui {

    void RmlRightPanel::init(RmlUIManager* mgr) {
        assert(mgr);
        rml_manager_ = mgr;

        rml_context_ = rml_manager_->createContext("right_panel", 400, 600);
        if (!rml_context_) {
            LOG_ERROR("RmlRightPanel: failed to create RML context");
            return;
        }

        rml_context_->EnableMouseCursor(false);

        try {
            const auto rml_path = lfs::vis::getAssetPath("rmlui/right_panel.rml");
            document_ = rml_context_->LoadDocument(rml_path.string());
            if (!document_) {
                LOG_ERROR("RmlRightPanel: failed to load right_panel.rml");
                return;
            }
            document_->Show();
        } catch (const std::exception& e) {
            LOG_ERROR("RmlRightPanel: resource not found: {}", e.what());
            return;
        }

        resize_handle_el_ = document_->GetElementById("resize-handle");
        left_border_el_ = document_->GetElementById("left-border");
        splitter_el_ = document_->GetElementById("splitter");
        tab_bar_el_ = document_->GetElementById("tab-bar");
        tab_separator_el_ = document_->GetElementById("tab-separator");

        updateTheme();
    }

    void RmlRightPanel::shutdown() {
        fbo_.destroy();
        if (rml_context_ && rml_manager_)
            rml_manager_->destroyContext("right_panel");
        rml_context_ = nullptr;
        document_ = nullptr;
        resize_handle_el_ = nullptr;
        left_border_el_ = nullptr;
        splitter_el_ = nullptr;
        tab_bar_el_ = nullptr;
        tab_separator_el_ = nullptr;
    }

    std::string RmlRightPanel::generateThemeRCSS() const {
        using rml_theme::colorToRml;
        using rml_theme::colorToRmlAlpha;
        const auto& t = lfs::vis::theme();
        const auto& p = t.palette;

        const auto tab_hover = colorToRmlAlpha(p.surface_bright, 0.5f);
        const auto tab_active_bg = colorToRmlAlpha(p.surface_bright, 0.4f);
        const auto tab_accent = colorToRml(p.primary);
        const auto tab_text = colorToRml(p.text);
        const auto tab_text_dim = colorToRml(p.text_dim);
        const auto splitter_bg = colorToRmlAlpha(p.border, 0.4f);
        const auto splitter_hover = colorToRmlAlpha(p.info, 0.6f);
        const auto splitter_active = colorToRmlAlpha(p.info, 0.8f);
        const auto border_color = colorToRmlAlpha(p.border, 0.6f);
        const auto separator_color = colorToRmlAlpha(p.border, 0.4f);
        const auto resize_hover = colorToRmlAlpha(p.info, 0.3f);
        const auto resize_active = colorToRmlAlpha(p.info, 0.5f);

        return std::format(
            "#splitter {{ background-color: {}; }}\n"
            "#splitter:hover {{ background-color: {}; }}\n"
            "#splitter.dragging {{ background-color: {}; }}\n"
            ".tab {{ background-color: transparent; color: {}; }}\n"
            ".tab:hover {{ background-color: {}; }}\n"
            ".tab.active {{ background-color: {}; color: {}; "
            "border-bottom-width: 2dp; border-bottom-color: {}; }}\n"
            "#left-border {{ background-color: {}; }}\n"
            "#tab-separator {{ background-color: {}; }}\n"
            "#resize-handle:hover {{ background-color: {}; }}\n"
            "#resize-handle.dragging {{ background-color: {}; }}\n",
            splitter_bg, splitter_hover, splitter_active,
            tab_text_dim,
            tab_hover,
            tab_active_bg, tab_text, tab_accent,
            border_color,
            separator_color,
            resize_hover,
            resize_active);
    }

    void RmlRightPanel::updateTheme() {
        if (!document_)
            return;

        const auto& t = lfs::vis::theme();
        if (t.name == last_theme_)
            return;
        last_theme_ = t.name;

        if (base_rcss_.empty())
            base_rcss_ = rml_theme::loadBaseRCSS("rmlui/right_panel.rcss");

        rml_theme::applyTheme(document_, base_rcss_, generateThemeRCSS());
    }

    void RmlRightPanel::rebuildTabs(const std::vector<TabSnapshot>& tabs,
                                    const std::string& active_tab) {
        if (!tab_bar_el_)
            return;

        bool tabs_changed = tabs.size() != last_tabs_.size();
        if (!tabs_changed) {
            for (size_t i = 0; i < tabs.size(); ++i) {
                if (tabs[i].idname != last_tabs_[i].idname ||
                    tabs[i].label != last_tabs_[i].label) {
                    tabs_changed = true;
                    break;
                }
            }
        }

        if (tabs_changed) {
            tab_bar_el_->SetInnerRML("");
            for (const auto& tab : tabs) {
                auto el = document_->CreateElement("div");
                el->SetAttribute("class", tab.idname == active_tab ? "tab active" : "tab");
                el->SetAttribute("data-idname", tab.idname);
                el->SetInnerRML(Rml::String(tab.label));
                tab_bar_el_->AppendChild(std::move(el));
            }
            last_tabs_ = tabs;
            last_active_tab_ = active_tab;
        } else if (active_tab != last_active_tab_) {
            for (int i = 0; i < tab_bar_el_->GetNumChildren(); ++i) {
                auto* child = tab_bar_el_->GetChild(i);
                const auto idname = child->GetAttribute<Rml::String>("data-idname", "");
                if (idname == active_tab)
                    child->SetAttribute("class", "tab active");
                else
                    child->SetAttribute("class", "tab");
            }
            last_active_tab_ = active_tab;
        }
    }

    static Rml::Element* findAncestorWithAttribute(Rml::Element* el, const Rml::String& attr) {
        while (el) {
            if (el->HasAttribute(attr))
                return el;
            el = el->GetParentNode();
        }
        return nullptr;
    }

    static bool isOrHasAncestor(Rml::Element* el, const Rml::String& id) {
        while (el) {
            if (el->GetId() == id)
                return true;
            el = el->GetParentNode();
        }
        return false;
    }

    CursorRequest RmlRightPanel::getCursorRequest() const {
        return cursor_request_;
    }

    void RmlRightPanel::processInput(const RightPanelLayout& layout, const PanelInputState& input) {
        wants_input_ = false;
        cursor_request_ = CursorRequest::None;

        const float delta_x = input.mouse_x - prev_mouse_x_;
        const float delta_y = input.mouse_y - prev_mouse_y_;
        prev_mouse_x_ = input.mouse_x;
        prev_mouse_y_ = input.mouse_y;

        if (!rml_context_ || !document_)
            return;
        if (layout.size.x <= 0 || layout.size.y <= 0)
            return;

        const float dp_ratio = rml_manager_->getDpRatio();

        const float mx = (input.mouse_x - layout.pos.x) * dp_ratio;
        const float my = (input.mouse_y - layout.pos.y) * dp_ratio;

        rml_context_->ProcessMouseMove(static_cast<int>(mx), static_cast<int>(my), 0);

        auto* hover = rml_context_->GetHoverElement();
        const bool over_interactive = hover && hover->GetTagName() != "body" &&
                                      hover->GetId() != "rp-body" &&
                                      hover->GetId() != "left-border" &&
                                      hover->GetId() != "tab-separator";

        if (resize_dragging_) {
            wants_input_ = true;

            if (input.mouse_down[0]) {
                if (on_resize_delta && delta_x != 0.0f)
                    on_resize_delta(delta_x);
                cursor_request_ = CursorRequest::ResizeEW;
            } else {
                resize_dragging_ = false;
                if (resize_handle_el_)
                    resize_handle_el_->SetAttribute("class", "");
            }
            return;
        }

        if (splitter_dragging_) {
            wants_input_ = true;

            if (input.mouse_down[0]) {
                if (on_splitter_delta && delta_y != 0.0f)
                    on_splitter_delta(delta_y);
            } else {
                splitter_dragging_ = false;
                if (splitter_el_)
                    splitter_el_->SetAttribute("class", "");
            }
            return;
        }

        if (over_interactive) {
            wants_input_ = true;

            if (isOrHasAncestor(hover, "resize-handle")) {
                cursor_request_ = CursorRequest::ResizeEW;
                if (input.mouse_clicked[0]) {
                    resize_dragging_ = true;
                    if (resize_handle_el_)
                        resize_handle_el_->SetAttribute("class", "dragging");
                }
            } else if (isOrHasAncestor(hover, "splitter")) {
                cursor_request_ = CursorRequest::ResizeNS;
                if (input.mouse_clicked[0]) {
                    splitter_dragging_ = true;
                    drag_start_y_ = input.mouse_y;
                    if (splitter_el_)
                        splitter_el_->SetAttribute("class", "dragging");
                }
            } else {
                if (input.mouse_clicked[0]) {
                    auto* tab_el = findAncestorWithAttribute(hover, "data-idname");
                    if (tab_el) {
                        auto idname = tab_el->GetAttribute<Rml::String>("data-idname", "");
                        if (!idname.empty() && on_tab_changed)
                            on_tab_changed(std::string(idname));
                    }
                }
            }
        }
    }

    void RmlRightPanel::render(const RightPanelLayout& layout,
                               const std::vector<TabSnapshot>& tabs,
                               const std::string& active_tab) {
        if (!rml_context_ || !document_)
            return;
        if (layout.size.x <= 0 || layout.size.y <= 0)
            return;

        updateTheme();

        const float dp_ratio = rml_manager_->getDpRatio();
        const int w = static_cast<int>(layout.size.x * dp_ratio);
        const int h = static_cast<int>(layout.size.y * dp_ratio);

        if (w <= 0 || h <= 0)
            return;

        const float tab_bar_h = 28.0f;

        if (resize_handle_el_) {
            resize_handle_el_->SetProperty("top", "0dp");
            resize_handle_el_->SetProperty("height", std::format("{:.0f}dp", layout.size.y));
        }
        if (left_border_el_) {
            left_border_el_->SetProperty("top", "0dp");
            left_border_el_->SetProperty("height", std::format("{:.0f}dp", layout.size.y));
        }
        if (splitter_el_) {
            splitter_el_->SetProperty("top", std::format("{:.0f}dp", layout.scene_h));
            splitter_el_->SetProperty("height", std::format("{:.0f}dp", layout.splitter_h));
        }
        if (tab_bar_el_) {
            const float tab_top = layout.scene_h + layout.splitter_h;
            tab_bar_el_->SetProperty("top", std::format("{:.0f}dp", tab_top));
            tab_bar_el_->SetProperty("height", std::format("{:.0f}dp", tab_bar_h));
        }
        if (tab_separator_el_) {
            const float sep_top = layout.scene_h + layout.splitter_h + tab_bar_h;
            tab_separator_el_->SetProperty("top", std::format("{:.0f}dp", sep_top));
        }

        rebuildTabs(tabs, active_tab);

        rml_context_->SetDimensions(Rml::Vector2i(w, h));
        rml_context_->Update();

        fbo_.ensure(w, h);
        if (!fbo_.valid())
            return;

        auto* render = rml_manager_->getRenderInterface();
        assert(render);
        render->SetViewport(w, h);

        GLint prev_fbo = 0;
        fbo_.bind(&prev_fbo);

        render->BeginFrame();
        rml_context_->Render();
        render->EndFrame();

        fbo_.unbind(prev_fbo);

        auto* vp = ImGui::GetMainViewport();
        fbo_.blitToDrawList(ImGui::GetForegroundDrawList(vp),
                            {layout.pos.x, layout.pos.y},
                            {layout.size.x, layout.size.y});
    }

} // namespace lfs::vis::gui

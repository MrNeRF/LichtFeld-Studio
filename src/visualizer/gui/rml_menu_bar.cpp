/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// clang-format off
#include <glad/glad.h>
// clang-format on

#include "gui/rml_menu_bar.hpp"
#include "core/logger.hpp"
#include "gui/rmlui/rml_theme.hpp"
#include "gui/rmlui/rmlui_manager.hpp"
#include "gui/rmlui/rmlui_render_interface.hpp"
#include "internal/resource_paths.hpp"
#include "operator/operator_registry.hpp"
#include "python/python_runtime.hpp"
#include "theme/theme.hpp"

#include <RmlUi/Core.h>
#include <RmlUi/Core/Element.h>
#include <cassert>
#include <format>

namespace lfs::vis::gui {

    namespace {
        std::string escapeRml(const std::string& s) {
            std::string out;
            out.reserve(s.size());
            for (char c : s) {
                switch (c) {
                case '<': out += "&lt;"; break;
                case '>': out += "&gt;"; break;
                case '&': out += "&amp;"; break;
                case '"': out += "&quot;"; break;
                default: out += c; break;
                }
            }
            return out;
        }
    } // namespace

    void RmlMenuBar::init(RmlUIManager* mgr) {
        assert(mgr);
        rml_manager_ = mgr;

        rml_context_ = rml_manager_->createContext("menu_bar", 800, 30);
        if (!rml_context_) {
            LOG_ERROR("RmlMenuBar: failed to create RML context");
            return;
        }

        try {
            const auto rml_path = lfs::vis::getAssetPath("rmlui/menubar.rml");
            document_ = rml_context_->LoadDocument(rml_path.string());
            if (!document_) {
                LOG_ERROR("RmlMenuBar: failed to load menubar.rml");
                return;
            }
            document_->Show();
        } catch (const std::exception& e) {
            LOG_ERROR("RmlMenuBar: resource not found: {}", e.what());
            return;
        }

        menu_items_ = document_->GetElementById("menu-items");
        bottom_border_ = document_->GetElementById("bottom-border");
        dropdown_overlay_ = document_->GetElementById("dropdown-overlay");
        dropdown_container_ = document_->GetElementById("dropdown-container");

        updateTheme();
    }

    void RmlMenuBar::shutdown() {
        fbo_.destroy();
        if (rml_context_ && rml_manager_)
            rml_manager_->destroyContext("menu_bar");
        rml_context_ = nullptr;
        document_ = nullptr;
        menu_items_ = nullptr;
        bottom_border_ = nullptr;
        dropdown_container_ = nullptr;
        dropdown_overlay_ = nullptr;
    }

    void RmlMenuBar::updateLabels(const std::vector<std::string>& labels,
                                  const std::vector<std::string>& idnames) {
        assert(labels.size() == idnames.size());
        if (labels == current_labels_)
            return;
        current_labels_ = labels;
        current_idnames_ = idnames;
        rebuildLabels();
    }

    void RmlMenuBar::rebuildLabels() {
        if (!menu_items_)
            return;

        std::string rml;
        for (size_t i = 0; i < current_labels_.size(); ++i) {
            rml += std::format("<span class=\"menu-label\" data-index=\"{}\">{}</span>",
                               i, escapeRml(current_labels_[i]));
        }
        menu_items_->SetInnerRML(rml);
    }

    void RmlMenuBar::processInput(const PanelInputState& input) {
        if (!menu_items_ || !document_)
            return;

        wants_input_ = false;

        const float dp_ratio = rml_manager_ ? rml_manager_->getDpRatio() : 1.0f;
        const float mx = input.mouse_x * dp_ratio;
        const float my = input.mouse_y * dp_ratio;

        rml_context_->ProcessMouseMove(static_cast<int>(mx), static_cast<int>(my), 0);

        const bool is_open = open_menu_index_ >= 0;

        const int count = menu_items_->GetNumChildren();
        int hovered_label = -1;
        for (int i = 0; i < count; ++i) {
            auto* child = menu_items_->GetChild(i);
            const auto box = child->GetAbsoluteOffset(Rml::BoxArea::Border);
            const auto size = child->GetBox().GetSize(Rml::BoxArea::Border);
            if (mx >= box.x && mx < box.x + size.x && my >= box.y && my < box.y + size.y) {
                hovered_label = i;
                break;
            }
        }

        if (is_open) {
            wants_input_ = true;

            if (hovered_label >= 0 && hovered_label != open_menu_index_) {
                openDropdown(hovered_label);
                return;
            }

            if (input.mouse_clicked[0]) {
                if (hovered_label >= 0 && hovered_label == open_menu_index_) {
                    closeDropdown();
                    return;
                }

                if (hovered_label < 0 && dropdown_container_) {
                    auto* hit = dropdown_container_;
                    {
                        Rml::Element* clicked = rml_context_->GetElementAtPoint(Rml::Vector2f(mx, my));
                        if (clicked) {
                            while (clicked && clicked != hit) {
                                if (clicked->HasAttribute("data-action")) {
                                    const std::string action = clicked->GetAttribute<Rml::String>("data-action", "");
                                    if (action == "operator") {
                                        const std::string op_id = clicked->GetAttribute<Rml::String>("data-operator-id", "");
                                        if (!op_id.empty() && clicked->GetAttribute<Rml::String>("data-disabled", "") != "true") {
                                            op::operators().invoke(op_id);
                                        }
                                        closeDropdown();
                                        return;
                                    } else if (action == "callback") {
                                        const int cb_idx = clicked->GetAttribute<int>("data-callback-index", -1);
                                        if (cb_idx >= 0 && clicked->GetAttribute<Rml::String>("data-disabled", "") != "true") {
                                            python::execute_menu_callback(open_content_.menu_idname, cb_idx);
                                        }
                                        closeDropdown();
                                        return;
                                    }
                                }
                                clicked = clicked->GetParentNode();
                            }
                        }
                    }

                    closeDropdown();
                    return;
                }
            }
        } else {
            if (hovered_label >= 0) {
                wants_input_ = true;
                if (input.mouse_clicked[0]) {
                    openDropdown(hovered_label);
                    return;
                }
            }
        }

        // Update active label highlighting
        int display_active = is_open ? open_menu_index_ : -1;
        if (display_active != active_index_) {
            if (active_index_ >= 0 && active_index_ < count)
                menu_items_->GetChild(active_index_)->SetClass("active", false);
            active_index_ = display_active;
            if (active_index_ >= 0 && active_index_ < count)
                menu_items_->GetChild(active_index_)->SetClass("active", true);
        }
    }

    void RmlMenuBar::openDropdown(int index) {
        assert(index >= 0 && index < static_cast<int>(current_idnames_.size()));

        open_menu_index_ = index;

        MenuDropdownContent content;
        content.menu_idname = current_idnames_[index];

        python::collect_menu_content(
            current_idnames_[index],
            [](const python::MenuItemInfo* info, void* ctx) {
                auto* c = static_cast<MenuDropdownContent*>(ctx);
                MenuItemDesc item;
                item.type = static_cast<MenuItemDesc::Type>(info->type);
                item.label = info->label ? info->label : "";
                item.operator_id = info->operator_id ? info->operator_id : "";
                item.shortcut = info->shortcut ? info->shortcut : "";
                item.enabled = info->enabled;
                item.selected = info->selected;
                item.callback_index = info->callback_index;
                c->items.push_back(std::move(item));
            },
            &content);

        open_content_ = std::move(content);
        rebuildDropdownDOM();

        // Update active highlighting
        const int count = menu_items_->GetNumChildren();
        if (active_index_ >= 0 && active_index_ < count)
            menu_items_->GetChild(active_index_)->SetClass("active", false);
        active_index_ = open_menu_index_;
        if (active_index_ >= 0 && active_index_ < count)
            menu_items_->GetChild(active_index_)->SetClass("active", true);
    }

    void RmlMenuBar::closeDropdown() {
        open_menu_index_ = -1;
        open_content_.items.clear();

        if (dropdown_container_) {
            dropdown_container_->SetInnerRML("");
            dropdown_container_->SetClass("visible", false);
        }
        if (dropdown_overlay_)
            dropdown_overlay_->SetClass("visible", false);

        const int count = menu_items_ ? menu_items_->GetNumChildren() : 0;
        if (active_index_ >= 0 && active_index_ < count)
            menu_items_->GetChild(active_index_)->SetClass("active", false);
        active_index_ = -1;

        wants_input_ = false;
    }

    namespace {
        std::string buildMenuItemsRml(const std::vector<MenuItemDesc>& items, size_t& pos) {
            std::string rml;
            while (pos < items.size()) {
                const auto& item = items[pos];
                switch (item.type) {
                case MenuItemDesc::Type::Operator: {
                    std::string cls = "menu-item";
                    if (!item.enabled)
                        cls += " disabled";
                    rml += std::format(
                        "<div class=\"{}\" data-action=\"operator\" data-operator-id=\"{}\"{}>"
                        "<span class=\"checkmark\"></span>"
                        "<span class=\"label\">{}</span>"
                        "</div>",
                        cls, escapeRml(item.operator_id),
                        item.enabled ? "" : " data-disabled=\"true\"",
                        escapeRml(item.label));
                    ++pos;
                    break;
                }
                case MenuItemDesc::Type::Separator:
                    rml += "<div class=\"menu-separator\"></div>";
                    ++pos;
                    break;
                case MenuItemDesc::Type::SubMenuBegin: {
                    rml += std::format(
                        "<div class=\"submenu-container\">"
                        "<div class=\"menu-item\">"
                        "<span class=\"checkmark\"></span>"
                        "<span class=\"label\">{}</span>"
                        "<span class=\"submenu-arrow\">&gt;</span>"
                        "</div>"
                        "<div class=\"submenu-popup dropdown-popup\">",
                        escapeRml(item.label));
                    ++pos;
                    rml += buildMenuItemsRml(items, pos);
                    rml += "</div></div>";
                    break;
                }
                case MenuItemDesc::Type::SubMenuEnd:
                    ++pos;
                    return rml;
                case MenuItemDesc::Type::Toggle: {
                    std::string cls = "menu-item";
                    const std::string check = item.selected ? "&#x2713;" : "";
                    rml += std::format(
                        "<div class=\"{}\" data-action=\"callback\" data-callback-index=\"{}\">"
                        "<span class=\"checkmark\">{}</span>"
                        "<span class=\"label\">{}</span>"
                        "<span class=\"shortcut\">{}</span>"
                        "</div>",
                        cls, item.callback_index, check,
                        escapeRml(item.label), escapeRml(item.shortcut));
                    ++pos;
                    break;
                }
                case MenuItemDesc::Type::ShortcutItem: {
                    std::string cls = "menu-item";
                    if (!item.enabled)
                        cls += " disabled";
                    rml += std::format(
                        "<div class=\"{}\" data-action=\"callback\" data-callback-index=\"{}\"{}>"
                        "<span class=\"checkmark\"></span>"
                        "<span class=\"label\">{}</span>"
                        "<span class=\"shortcut\">{}</span>"
                        "</div>",
                        cls, item.callback_index,
                        item.enabled ? "" : " data-disabled=\"true\"",
                        escapeRml(item.label), escapeRml(item.shortcut));
                    ++pos;
                    break;
                }
                case MenuItemDesc::Type::Item: {
                    std::string cls = "menu-item";
                    if (!item.enabled)
                        cls += " disabled";
                    rml += std::format(
                        "<div class=\"{}\" data-action=\"callback\" data-callback-index=\"{}\"{}>"
                        "<span class=\"checkmark\"></span>"
                        "<span class=\"label\">{}</span>"
                        "</div>",
                        cls, item.callback_index,
                        item.enabled ? "" : " data-disabled=\"true\"",
                        escapeRml(item.label));
                    ++pos;
                    break;
                }
                }
            }
            return rml;
        }
    } // namespace

    void RmlMenuBar::rebuildDropdownDOM() {
        if (!dropdown_container_ || !dropdown_overlay_ || !menu_items_)
            return;

        const int count = menu_items_->GetNumChildren();
        if (open_menu_index_ < 0 || open_menu_index_ >= count)
            return;

        auto* label_el = menu_items_->GetChild(open_menu_index_);
        const auto label_offset = label_el->GetAbsoluteOffset(Rml::BoxArea::Border);
        const auto label_size = label_el->GetBox().GetSize(Rml::BoxArea::Border);

        size_t pos = 0;
        std::string items_rml = buildMenuItemsRml(open_content_.items, pos);
        std::string popup_rml = std::format("<div class=\"dropdown-popup\">{}</div>", items_rml);

        dropdown_container_->SetInnerRML(popup_rml);
        dropdown_container_->SetProperty("left", std::format("{}px", label_offset.x));
        dropdown_container_->SetProperty("top", std::format("{}px", label_offset.y + label_size.y));
        dropdown_container_->SetClass("visible", true);
        dropdown_overlay_->SetClass("visible", true);
    }

    std::string RmlMenuBar::generateThemeRCSS() const {
        using rml_theme::colorToRml;
        const auto& t = lfs::vis::theme();

        const auto bg = colorToRml(t.menu_background());
        const auto text = colorToRml(t.palette.text);
        const auto text_dim = colorToRml(t.palette.text_dim);
        const auto hover = colorToRml(t.menu_hover());
        const auto active = colorToRml(t.menu_active());
        const auto border = rml_theme::darkenColorToRml(t.palette.surface, t.menu.bottom_border_darken);
        const auto popup_bg = colorToRml(t.menu_popup_background());
        const auto popup_border = colorToRml(t.menu_border());

        return std::format(
            "body {{ color: {}; }}\n"
            "#menu-items {{ background-color: {}; }}\n"
            ".menu-label:hover {{ background-color: {}; }}\n"
            ".menu-label.active {{ background-color: {}; }}\n"
            "#bottom-border {{ background-color: {}; }}\n"
            ".dropdown-popup {{ background-color: {}; border-color: {}; }}\n"
            ".menu-item:hover {{ background-color: {}; }}\n"
            ".menu-item.disabled {{ color: {}; }}\n"
            ".menu-item.disabled:hover {{ background-color: transparent; }}\n"
            ".menu-item .shortcut {{ color: {}; }}\n"
            ".menu-separator {{ background-color: {}; }}\n",
            text, bg, hover, active, border,
            popup_bg, popup_border,
            hover, text_dim, text_dim, popup_border);
    }

    void RmlMenuBar::updateTheme() {
        if (!document_)
            return;

        const auto& t = lfs::vis::theme();
        if (t.name == last_theme_)
            return;
        last_theme_ = t.name;

        if (base_rcss_.empty())
            base_rcss_ = rml_theme::loadBaseRCSS("rmlui/menubar.rcss");

        rml_theme::applyTheme(document_, base_rcss_, generateThemeRCSS());
    }

    void RmlMenuBar::draw(int screen_w, int screen_h) {
        if (!rml_context_ || !document_)
            return;

        updateTheme();

        const float dp_ratio = rml_manager_->getDpRatio();
        const int bar_h = static_cast<int>(bar_height_ * dp_ratio);

        int ctx_w = static_cast<int>(screen_w * dp_ratio);
        int ctx_h;

        if (open_menu_index_ >= 0) {
            ctx_h = static_cast<int>(screen_h * dp_ratio);
        } else {
            ctx_h = bar_h;
        }

        rml_context_->SetDimensions(Rml::Vector2i(ctx_w, ctx_h));
        document_->SetProperty("height", std::format("{}px", ctx_h));
        rml_context_->Update();

        fbo_.ensure(ctx_w, ctx_h);
        if (!fbo_.valid())
            return;

        auto* render = rml_manager_->getRenderInterface();
        assert(render);
        render->SetViewport(ctx_w, ctx_h);

        GLint prev_fbo = 0;
        fbo_.bind(&prev_fbo);

        render->BeginFrame();
        rml_context_->Render();
        render->EndFrame();

        fbo_.unbind(prev_fbo);
    }

} // namespace lfs::vis::gui

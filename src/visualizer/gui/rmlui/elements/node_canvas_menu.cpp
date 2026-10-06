/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "gui/gui_manager.hpp"
#include "node_canvas_element.hpp"
#include "visualizer/core/services.hpp"

#include "core/event_bridge/localization_manager.hpp"
#include "gui/rmlui/elements/node_canvas_dom.hpp"
#include "gui/rmlui/elements/node_canvas_widgets.hpp"
#include "gui/utils/native_file_dialog.hpp"
#include "visualizer/nodes/modifier_manager.hpp"

#include <RmlUi/Core/ElementDocument.h>
#include <RmlUi/Core/Elements/ElementFormControlInput.h>
#include <RmlUi/Core/Event.h>
#include <RmlUi/Core/Input.h>
#include <algorithm>
#include <cctype>
#include <format>
#include <ranges>

namespace lfs::vis::gui {
    void NodeCanvasElement::openTemplateBrowser() {
        if (const auto host = activeHost(); host && services().guiOrNull())
            services().gui().openTemplateBrowser(*host);
    }

    void NodeCanvasElement::openTemplateSaveDialog(std::string tree_uuid) {
        if (const auto host = activeHost(); host && services().guiOrNull())
            services().gui().openTemplateBrowser(*host, std::move(tree_uuid));
    }

    void NodeCanvasElement::closeAddMenu() {
        if (add_menu_)
            RemoveChild(add_menu_);
        add_menu_ = nullptr;
        first_add_type_.clear();
    }

    void NodeCanvasElement::openAddMenu(const float screen_x, const float screen_y) {
        if (!manager_ || !activeTree() || !GetOwnerDocument())
            return;
        closeAddMenu();
        const auto size = GetBox().GetSize(Rml::BoxArea::Content);
        const auto offset = GetAbsoluteOffset(Rml::BoxArea::Content);
        const float available = std::max(1.0f, size.x - (sidebar_visible_ ? 280.0f * dp_ratio_ : 0.0f));
        const float width = std::min(210.0f * dp_ratio_, available - 16.0f * dp_ratio_);
        const float height = size.y * 0.60f;
        const CanvasPoint pointer{screen_x - offset.x, screen_y - offset.y};
        add_position_ = interaction_.screenToGraph({std::clamp(pointer.x, 0.0f, available),
                                                    std::clamp(pointer.y, 20.0f * dp_ratio_, size.y)});
        auto menu = GetOwnerDocument()->CreateElement("div");
        menu->SetId("node-add-menu");
        menu->SetProperty("width", std::format("{}px", width));
        const float left = std::clamp(pointer.x, 8.0f * dp_ratio_, std::max(8.0f * dp_ratio_, available - width - 8.0f * dp_ratio_));
        menu->SetProperty("left", std::format("{}px", left));
        menu->SetClass("cascade-left", left + 2.0f * width + 260.0f * dp_ratio_ > available && left > available * 0.5f);
        std::string html = "<input id=\"node-add-search\" type=\"text\" placeholder=\"" +
                           node_widgets::escape(LOC("node_editor.search_nodes")) +
                           "\"/><div class=\"node-add-categories\">";
        auto types = manager_->registry().list_localized();
        std::ranges::sort(types, [](const auto& a, const auto& b) {
            return std::tie(a->category, a->label) < std::tie(b->category, b->label);
        });
        std::string category;
        std::string submenus;
        std::size_t category_count = 0;
        bool column = false;
        for (const auto& type : types) {
            if (!std::ranges::contains(type->tree_types, activeTree()->tree_type) ||
                type->id == "lfs.group_input" || type->id == "lfs.group_output")
                continue;
            if (!column || type->category != category) {
                if (column)
                    submenus += "</div>";
                category = type->category;
                column = true;
                ++category_count;
                html += "<button class=\"node-add-heading\" data-category=\"" + node_widgets::escape(category) +
                        "\">" + node_widgets::categoryIcon(category) + node_widgets::escape(category) + "<span class=\"settings-arrow\"></span></button>";
                submenus += "<div class=\"node-add-category\" data-category=\"" + node_widgets::escape(category) + "\">";
            }
            submenus += std::format(R"(<button class="node-add-item" data-type="{}" data-search="{}">{}</button>)",
                                    node_widgets::escape(type->id), node_widgets::escape(type->label),
                                    node_widgets::categoryIcon(type->category) + node_widgets::escape(type->label));
        }
        if (column)
            submenus += "</div>";
        html += "</div><div class=\"node-add-results\">" + submenus + "</div><div id=\"node-add-empty\">" + node_widgets::escape(LOC("node_editor.no_results")) + "</div><div id=\"node-add-preview\"></div>";
        const float natural_height = (40.0f + 12.0f * category_count) * dp_ratio_;
        const float top = std::clamp(pointer.y, 8.0f * dp_ratio_, std::max(8.0f * dp_ratio_, size.y - std::min(height, natural_height) - 8.0f * dp_ratio_));
        const float content_height = std::max(12.0f * dp_ratio_, std::min(height, size.y - top - 8.0f * dp_ratio_) - 40.0f * dp_ratio_);
        menu->SetProperty("top", std::format("{}px", top));
        menu->SetInnerRML(html);
        menu->GetElementById("node-add-preview")->SetProperty("max-height", std::format("{}px", size.y - top - 8.0f * dp_ratio_));
        Rml::ElementList scroll_regions;
        menu->QuerySelectorAll(scroll_regions, ".node-add-categories, .node-add-category, .node-add-results");
        for (auto* region : scroll_regions)
            region->SetProperty("max-height", std::format("{}px", content_height));
        add_menu_ = AppendChild(std::move(menu));
        filterAddMenu({});
        add_menu_->GetElementById("node-add-search")->Focus();
    }

    void NodeCanvasElement::filterAddMenu(const std::string_view search) {
        if (!add_menu_)
            return;
        const auto lower = [](const std::string_view value) {
            std::string text(value);
            std::ranges::transform(text, text.begin(), [](const unsigned char c) { return static_cast<char>(std::tolower(c)); });
            return text;
        };
        const auto needle = lower(search);
        add_menu_->SetClass("searching", !needle.empty());
        first_add_type_.clear();
        Rml::ElementList columns;
        add_menu_->QuerySelectorAll(columns, ".node-add-category");
        for (auto* column : columns) {
            bool visible = false;
            Rml::ElementList buttons;
            column->QuerySelectorAll(buttons, ".node-add-item");
            for (auto* button : buttons) {
                const bool match = lower(button->GetAttribute<Rml::String>("data-search", "")).find(needle) != std::string::npos;
                button->SetClass("filtered", !match);
                button->SetClass("first-hit", !needle.empty() && match && first_add_type_.empty());
                if (match && first_add_type_.empty())
                    first_add_type_ = button->GetAttribute<Rml::String>("data-type", "");
                visible |= match;
            }
            column->SetClass("filtered", !visible);
        }
        add_menu_->GetElementById("node-add-empty")->SetClass("filtered", !first_add_type_.empty());
        highlightAddType(needle.empty() ? std::string{} : first_add_type_);
    }

    void NodeCanvasElement::highlightAddType(const std::string_view id) {
        if (!add_menu_)
            return;
        const auto type = manager_->registry().find_localized(id);
        add_menu_->SetClass("has-preview", type != nullptr);
        if (!type)
            return;
        first_add_type_ = id;
        Rml::ElementList buttons;
        add_menu_->QuerySelectorAll(buttons, ".node-add-item");
        for (auto* button : buttons)
            button->SetClass("first-hit", button->GetAttribute<Rml::String>("data-type", "") == id);
        auto* preview = add_menu_->GetElementById("node-add-preview");
        const auto html = "<div class=\"node-add-preview-title\">" + node_widgets::categoryIcon(type->category) +
                          node_widgets::escape(type->label) + "</div><div class=\"node-add-preview-description\">" +
                          node_widgets::escape(type->description) + "</div><div class=\"node-add-preview-help\">" +
                          node_widgets::escape(type->help) + "</div>";
        node_widgets::patchMarkup(*preview, html);
    }

    void NodeCanvasElement::moveAddHighlight(const int direction) {
        if (!add_menu_)
            return;
        Rml::ElementList buttons;
        add_menu_->QuerySelectorAll(buttons, ".node-add-item");
        std::erase_if(buttons, [&](const auto* button) {
            return button->IsClassSet("filtered") ||
                   (!add_menu_->IsClassSet("searching") && !button->GetParentNode()->IsClassSet("open"));
        });
        if (buttons.empty())
            return;
        const auto selected = std::ranges::find_if(buttons, [&](const Rml::Element* button) {
            return button->GetAttribute<Rml::String>("data-type", "") == first_add_type_;
        });
        const int count = static_cast<int>(buttons.size());
        const int index = selected == buttons.end() ? (direction > 0 ? 0 : count - 1)
                                                    : (static_cast<int>(selected - buttons.begin()) + direction + count) % count;
        auto* button = buttons[index];
        highlightAddType(button->GetAttribute<Rml::String>("data-type", ""));
        button->ScrollIntoView();
    }

    bool NodeCanvasElement::processAddMenuEvent(Rml::Event& event) {
        if (!add_menu_)
            return false;
        auto* target = event.GetTargetElement();
        bool inside = false;
        Rml::Element* button = nullptr;
        Rml::Element* category = nullptr;
        for (auto* element = target; element && element != this; element = element->GetParentNode()) {
            inside |= element == add_menu_;
            if (element->HasAttribute("data-type"))
                button = element;
            if (element->IsClassSet("node-add-heading"))
                category = element;
        }
        const auto& type = event.GetType();
        if ((type == "mouseover" || type == "click") && category) {
            const auto name = category->GetAttribute<Rml::String>("data-category", "");
            Rml::ElementList panels;
            add_menu_->QuerySelectorAll(panels, ".node-add-category, .node-add-heading");
            for (auto* panel : panels)
                panel->SetClass("open", panel->GetAttribute<Rml::String>("data-category", "") == name);
            highlightAddType({});
        } else if (type == "mouseover" && button) {
            highlightAddType(button->GetAttribute<Rml::String>("data-type", ""));
        } else if (type == "change" && target->GetId() == "node-add-search") {
            if (const auto* input = dynamic_cast<Rml::ElementFormControlInput*>(target))
                filterAddMenu(input->GetValue());
        } else if (type == "click" && button) {
            const auto id = button->GetAttribute<Rml::String>("data-type", "");
            closeAddMenu();
            addNode(id, add_position_);
        } else if (type == "keydown") {
            const int key = event.GetParameter("key_identifier", 0);
            if (key == Rml::Input::KI_UP || key == Rml::Input::KI_DOWN) {
                moveAddHighlight(key == Rml::Input::KI_UP ? -1 : 1);
            } else if (key == Rml::Input::KI_ESCAPE) {
                closeAddMenu();
                focusCanvas();
            } else if (key == Rml::Input::KI_RETURN || key == Rml::Input::KI_NUMPADENTER) {
                const auto id = first_add_type_;
                closeAddMenu();
                if (!id.empty())
                    addNode(id, add_position_);
            } else {
                return false;
            }
        } else if (type == "mousedown" && !inside) {
            closeAddMenu();
        } else if (inside && type == "mousescroll") {
            return true;
        } else if (!inside) {
            return false;
        }
        event.StopPropagation();
        return true;
    }
} // namespace lfs::vis::gui

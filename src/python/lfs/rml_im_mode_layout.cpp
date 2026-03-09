/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "rml_im_mode_layout.hpp"
#include "core/event_bridge/localization_manager.hpp"
#include "core/logger.hpp"
#include "py_ui.hpp"
#include "python/python_runtime.hpp"
#include "visualizer/operator/operator_registry.hpp"

#include <RmlUi/Core/Element.h>
#include <RmlUi/Core/ElementDocument.h>
#include <RmlUi/Core/Factory.h>

#include <cassert>
#include <cmath>
#include <format>
#include <limits>

namespace {
    std::string strip_imgui_id(const std::string& label) {
        auto pos = label.find("##");
        if (pos == std::string::npos)
            return label;
        return label.substr(0, pos);
    }

    std::string hidden_imgui_id(const std::string& label) {
        const auto pos = label.find("##");
        if (pos == std::string::npos || pos + 2 >= label.size())
            return {};
        return label.substr(pos + 2);
    }
} // namespace

namespace lfs::python {

    void SlotEventListener::ProcessEvent(Rml::Event& event) {
        assert(state_);
        const auto type = event.GetId();
        if (type == Rml::EventId::Click) {
            state_->clicked = true;
        } else if (type == Rml::EventId::Change) {
            state_->changed = true;
            auto* el = event.GetCurrentElement();
            if (!el)
                return;
            const auto tag = el->GetTagName();
            if (tag == "input") {
                const auto input_type = el->GetAttribute<Rml::String>("type", "");
                if (input_type == "checkbox") {
                    state_->bool_value = el->HasAttribute("checked");
                } else if (input_type == "range") {
                    state_->float_value = el->GetAttribute<float>("value", 0.0f);
                } else if (input_type == "text") {
                    state_->string_value = el->GetAttribute<Rml::String>("value", "");
                }
            } else if (tag == "select") {
                state_->int_value = static_cast<int>(el->GetAttribute<float>("value", 0.0f));
            }
        }
    }

    void RmlImModeLayout::begin_frame(Rml::ElementDocument* doc, const MouseState& mouse) {
        assert(doc);
        doc_ = doc;
        mouse_ = mouse;
        root_ = doc->GetElementById("im-root");
        if (!root_) {
            root_ = doc->GetElementById("content");
            if (!root_)
                root_ = doc_->GetFirstChild();
        }
        assert(root_);

        removed_elements_.clear();

        auto line_height_prop = root_->GetProperty("line-height");
        if (line_height_prop)
            cached_line_height_ = line_height_prop->Get<float>();

        if (containers_.empty() || containers_[0].parent != root_) {
            containers_.clear();
            containers_.push_back({root_, {}, 0});
        } else {
            containers_.resize(1);
            containers_[0].cursor = 0;
        }
        current_line_ = nullptr;
        next_same_line_ = false;
        indent_level_ = 0;
        disabled_ = false;
        disabled_depth_ = 0;
        id_stack_.clear();
        force_next_open_ = false;
        table_.reset();
        item_width_stack_.clear();
        last_element_ = nullptr;
        last_clicked_ = false;
        tooltip_el_ = doc->GetElementById("im-tooltip");
        tooltip_shown_ = false;
        popup_backdrop_ = doc->GetElementById("im-popup-backdrop");
        popup_dialog_ = doc->GetElementById("im-popup-dialog");
        active_popup_id_.clear();
    }

    void RmlImModeLayout::end_frame() {
        finish_current_line();

        if (!tooltip_shown_ && tooltip_el_)
            tooltip_el_->SetClass("visible", false);

        for (auto& level : containers_)
            prune_excess_slots(level);

        containers_.resize(1);
        current_line_ = nullptr;
        doc_ = nullptr;
        root_ = nullptr;
    }

    void RmlImModeLayout::prune_excess_slots(ContainerLevel& level) {
        while (static_cast<int>(level.slots.size()) > level.cursor) {
            auto& slot = level.slots.back();
            if (slot.element && slot.element->GetParentNode())
                removed_elements_.push_back(slot.element->GetParentNode()->RemoveChild(slot.element));
            level.slots.pop_back();
        }
    }

    std::string RmlImModeLayout::build_id(const std::string& key) const {
        std::string result;
        for (const auto& id : id_stack_) {
            result += id;
            result += '/';
        }
        result += key;
        return result;
    }

    std::string RmlImModeLayout::stable_label_token(const std::string& label) {
        return hidden_imgui_id(label);
    }

    std::string RmlImModeLayout::build_slot_id(const char* prefix,
                                               const std::string* label) const {
        std::string key(prefix);
        if (label) {
            const auto token = stable_label_token(*label);
            if (!token.empty()) {
                key += ":" + token;
                return build_id(key);
            }
        }
        key += "#" + std::to_string(containers_.empty() ? 0 : containers_.back().cursor);
        return build_id(key);
    }

    std::string RmlImModeLayout::color_to_css(nb::object color) const {
        if (color.is_none())
            return {};
        try {
            auto tup = nb::cast<nb::tuple>(color);
            const auto len = nb::len(tup);
            if (len >= 3) {
                float r = nb::cast<float>(tup[0]);
                float g = nb::cast<float>(tup[1]);
                float b = nb::cast<float>(tup[2]);
                float a = len >= 4 ? nb::cast<float>(tup[3]) : 1.0f;
                if (r <= 1.0f && g <= 1.0f && b <= 1.0f) {
                    r *= 255.0f;
                    g *= 255.0f;
                    b *= 255.0f;
                    a *= 255.0f;
                }
                return std::format("rgba({},{},{},{})",
                                   static_cast<int>(r), static_cast<int>(g),
                                   static_cast<int>(b), static_cast<int>(a));
            }
        } catch (...) {}
        return {};
    }

    Rml::Element* RmlImModeLayout::ensure_line_container() {
        if (table_ && table_->current_cell)
            return table_->current_cell;

        if (next_same_line_ && current_line_) {
            next_same_line_ = false;
            return current_line_;
        }

        finish_current_line();

        assert(!containers_.empty());
        auto& slot = ensure_slot(SlotType::Line, build_slot_id("line"));

        if (!slot.element) {
            auto line = doc_->CreateElement("div");
            line->SetClass("im-line", true);
            if (indent_level_ > 0)
                line->SetProperty("margin-left", Rml::String(std::to_string(indent_level_ * 20) + "dp"));
            if (disabled_)
                line->SetClass("disabled-overlay", true);
            slot.element = containers_.back().parent->AppendChild(std::move(line));
        }

        current_line_ = slot.element;
        return current_line_;
    }

    void RmlImModeLayout::finish_current_line() {
        current_line_ = nullptr;
        next_same_line_ = false;
    }

    Slot& RmlImModeLayout::ensure_slot(SlotType type, const std::string& key) {
        assert(!containers_.empty());
        auto& level = containers_.back();
        const auto idx = level.cursor;

        if (idx < static_cast<int>(level.slots.size())) {
            auto& slot = level.slots[idx];
            if (slot.type == type && slot.key == key) {
                level.cursor++;
                return slot;
            }
            if (slot.element && slot.element->GetParentNode())
                removed_elements_.push_back(slot.element->GetParentNode()->RemoveChild(slot.element));
            slot = Slot{type, key, nullptr, {}};
        } else {
            level.slots.push_back(Slot{type, key, nullptr, {}});
        }

        level.cursor++;
        return level.slots[idx];
    }

    Rml::Element* RmlImModeLayout::create_element(SlotType type, const std::string& key) {
        (void)key;
        switch (type) {
        case SlotType::Label:
        case SlotType::TextColored:
        case SlotType::TextDisabled:
        case SlotType::TextWrapped:
        case SlotType::BulletText:
            return doc_->CreateElement("p").get();
        case SlotType::Heading:
            return doc_->CreateElement("h3").get();
        case SlotType::Button:
        case SlotType::ButtonStyled:
        case SlotType::SmallButton:
            return doc_->CreateElement("button").get();
        case SlotType::Separator:
            return doc_->CreateElement("div").get();
        case SlotType::Spacing:
            return doc_->CreateElement("div").get();
        case SlotType::ProgressBar:
            return doc_->CreateElement("progress").get();
        default:
            return doc_->CreateElement("div").get();
        }
    }

    void RmlImModeLayout::warn_unsupported(const char* method) {
        if (warned_methods_.insert(method).second)
            LOG_INFO("RmlImModeLayout: '{}' not yet supported, using no-op", method);
    }

    // ── Text ────────────────────────────────────────────────

    void RmlImModeLayout::label(const std::string& text) {
        if (!doc_)
            return;
        auto* line = ensure_line_container();
        auto& slot = ensure_slot(SlotType::Label, build_slot_id("label", &text));
        const auto display = strip_imgui_id(text);

        if (!slot.element) {
            auto el = doc_->CreateElement("p");
            el->SetClass("im-label", true);
            el->SetInnerRML(Rml::String(display));
            slot.element = line->AppendChild(std::move(el));
        } else {
            if (slot.element->GetParentNode() != line)
                line->AppendChild(slot.element->GetParentNode()->RemoveChild(slot.element));
            slot.element->SetInnerRML(Rml::String(display));
        }
        last_element_ = slot.element;
        last_clicked_ = false;
    }

    void RmlImModeLayout::label_centered(const std::string& text) {
        if (!doc_)
            return;
        auto* line = ensure_line_container();
        auto& slot = ensure_slot(SlotType::Label, build_slot_id("label_center", &text));
        const auto display = strip_imgui_id(text);

        if (!slot.element) {
            auto el = doc_->CreateElement("p");
            el->SetClass("im-label", true);
            el->SetProperty("text-align", "center");
            el->SetProperty("width", "100%");
            el->SetInnerRML(Rml::String(display));
            slot.element = line->AppendChild(std::move(el));
        } else {
            if (slot.element->GetParentNode() != line)
                line->AppendChild(slot.element->GetParentNode()->RemoveChild(slot.element));
            slot.element->SetInnerRML(Rml::String(display));
        }
        last_element_ = slot.element;
        last_clicked_ = false;
    }

    void RmlImModeLayout::heading(const std::string& text) {
        if (!doc_)
            return;
        auto* line = ensure_line_container();
        auto& slot = ensure_slot(SlotType::Heading, build_slot_id("heading", &text));
        const auto display = strip_imgui_id(text);

        if (!slot.element) {
            auto el = doc_->CreateElement("div");
            el->SetClass("panel-title", true);
            el->SetInnerRML(Rml::String(display));
            slot.element = line->AppendChild(std::move(el));
        } else {
            if (slot.element->GetParentNode() != line)
                line->AppendChild(slot.element->GetParentNode()->RemoveChild(slot.element));
            slot.element->SetInnerRML(Rml::String(display));
        }
        last_element_ = slot.element;
        last_clicked_ = false;
    }

    void RmlImModeLayout::text_colored(const std::string& text, nb::object color) {
        if (!doc_)
            return;
        auto* line = ensure_line_container();
        auto& slot = ensure_slot(SlotType::TextColored, build_slot_id("text_colored", &text));
        auto css_color = color_to_css(color);
        const auto display = strip_imgui_id(text);

        if (!slot.element) {
            auto el = doc_->CreateElement("p");
            el->SetClass("im-label", true);
            if (!css_color.empty())
                el->SetProperty("color", Rml::String(css_color));
            el->SetInnerRML(Rml::String(display));
            slot.element = line->AppendChild(std::move(el));
        } else {
            if (slot.element->GetParentNode() != line)
                line->AppendChild(slot.element->GetParentNode()->RemoveChild(slot.element));
            if (!css_color.empty())
                slot.element->SetProperty("color", Rml::String(css_color));
            slot.element->SetInnerRML(Rml::String(display));
        }
        last_element_ = slot.element;
        last_clicked_ = false;
    }

    void RmlImModeLayout::text_colored_centered(const std::string& text, nb::object color) {
        if (!doc_)
            return;
        auto* line = ensure_line_container();
        auto& slot = ensure_slot(SlotType::TextColored, build_slot_id("text_colored_center", &text));
        auto css_color = color_to_css(color);
        const auto display = strip_imgui_id(text);

        if (!slot.element) {
            auto el = doc_->CreateElement("p");
            el->SetClass("im-label", true);
            el->SetProperty("text-align", "center");
            el->SetProperty("width", "100%");
            if (!css_color.empty())
                el->SetProperty("color", Rml::String(css_color));
            el->SetInnerRML(Rml::String(display));
            slot.element = line->AppendChild(std::move(el));
        } else {
            if (slot.element->GetParentNode() != line)
                line->AppendChild(slot.element->GetParentNode()->RemoveChild(slot.element));
            if (!css_color.empty())
                slot.element->SetProperty("color", Rml::String(css_color));
            slot.element->SetInnerRML(Rml::String(display));
        }
        last_element_ = slot.element;
        last_clicked_ = false;
    }

    void RmlImModeLayout::text_selectable(const std::string& text, float /*height*/) {
        label(text);
    }

    void RmlImModeLayout::text_wrapped(const std::string& text) {
        if (!doc_)
            return;
        auto* line = ensure_line_container();
        auto& slot = ensure_slot(SlotType::TextWrapped, build_slot_id("text_wrapped", &text));
        const auto display = strip_imgui_id(text);

        if (!slot.element) {
            auto el = doc_->CreateElement("p");
            el->SetClass("im-text-wrapped", true);
            el->SetInnerRML(Rml::String(display));
            slot.element = line->AppendChild(std::move(el));
        } else {
            if (slot.element->GetParentNode() != line)
                line->AppendChild(slot.element->GetParentNode()->RemoveChild(slot.element));
            slot.element->SetInnerRML(Rml::String(display));
        }
        last_element_ = slot.element;
        last_clicked_ = false;
    }

    void RmlImModeLayout::text_disabled(const std::string& text) {
        if (!doc_)
            return;
        auto* line = ensure_line_container();
        auto& slot = ensure_slot(SlotType::TextDisabled, build_slot_id("text_disabled", &text));
        const auto display = strip_imgui_id(text);

        if (!slot.element) {
            auto el = doc_->CreateElement("p");
            el->SetClass("text-disabled", true);
            el->SetInnerRML(Rml::String(display));
            slot.element = line->AppendChild(std::move(el));
        } else {
            if (slot.element->GetParentNode() != line)
                line->AppendChild(slot.element->GetParentNode()->RemoveChild(slot.element));
            slot.element->SetInnerRML(Rml::String(display));
        }
        last_element_ = slot.element;
        last_clicked_ = false;
    }

    void RmlImModeLayout::bullet_text(const std::string& text) {
        if (!doc_)
            return;
        auto* line = ensure_line_container();
        auto& slot = ensure_slot(SlotType::BulletText, build_slot_id("bullet_text", &text));
        const auto display = strip_imgui_id(text);

        if (!slot.element) {
            auto el = doc_->CreateElement("p");
            el->SetClass("im-bullet", true);
            el->SetInnerRML(Rml::String(std::format("\xe2\x80\xa2 {}", display)));
            slot.element = line->AppendChild(std::move(el));
        } else {
            if (slot.element->GetParentNode() != line)
                line->AppendChild(slot.element->GetParentNode()->RemoveChild(slot.element));
            slot.element->SetInnerRML(Rml::String(std::format("\xe2\x80\xa2 {}", display)));
        }
        last_element_ = slot.element;
        last_clicked_ = false;
    }

    // ── Buttons ─────────────────────────────────────────────

    bool RmlImModeLayout::button(const std::string& label, std::tuple<float, float> size) {
        if (!doc_)
            return false;
        auto* line = ensure_line_container();
        auto& slot = ensure_slot(SlotType::Button, build_slot_id("button", &label));
        const auto display = strip_imgui_id(label);

        if (!slot.element) {
            auto el = doc_->CreateElement("button");
            el->SetClass("btn", true);
            el->SetInnerRML(Rml::String(display));
            auto [w, h] = size;
            if (w < 0)
                el->SetClass("btn--full", true);
            else if (w > 0)
                el->SetProperty("width", Rml::String(std::to_string(static_cast<int>(w)) + "dp"));

            el->AddEventListener(Rml::EventId::Click, new SlotEventListener(&slot.events));

            slot.element = line->AppendChild(std::move(el));
        } else {
            if (slot.element->GetParentNode() != line)
                line->AppendChild(slot.element->GetParentNode()->RemoveChild(slot.element));
            slot.element->SetInnerRML(Rml::String(display));
        }

        bool clicked = slot.events.clicked;
        slot.events.clicked = false;
        last_element_ = slot.element;
        last_clicked_ = clicked;
        return clicked;
    }

    bool RmlImModeLayout::button_callback(const std::string& label, nb::object callback,
                                          std::tuple<float, float> size) {
        bool clicked = button(label, size);
        if (clicked && !callback.is_none()) {
            try {
                callback();
            } catch (const std::exception& e) {
                LOG_ERROR("button_callback error: {}", e.what());
            }
        }
        return clicked;
    }

    bool RmlImModeLayout::small_button(const std::string& label) {
        if (!doc_)
            return false;
        auto* line = ensure_line_container();
        auto& slot = ensure_slot(SlotType::SmallButton, build_slot_id("small_button", &label));
        const auto display = strip_imgui_id(label);

        if (!slot.element) {
            auto el = doc_->CreateElement("button");
            el->SetClass("btn", true);
            el->SetInnerRML(Rml::String(display));

            el->AddEventListener(Rml::EventId::Click, new SlotEventListener(&slot.events));

            slot.element = line->AppendChild(std::move(el));
        } else {
            if (slot.element->GetParentNode() != line)
                line->AppendChild(slot.element->GetParentNode()->RemoveChild(slot.element));
            slot.element->SetInnerRML(Rml::String(display));
        }

        bool clicked = slot.events.clicked;
        slot.events.clicked = false;
        last_element_ = slot.element;
        last_clicked_ = clicked;
        return clicked;
    }

    bool RmlImModeLayout::button_styled(const std::string& label, const std::string& style,
                                        std::tuple<float, float> size) {
        if (!doc_)
            return false;
        auto* line = ensure_line_container();
        auto& slot = ensure_slot(SlotType::ButtonStyled, build_slot_id("button_styled", &label));
        const auto display = strip_imgui_id(label);

        if (!slot.element) {
            auto el = doc_->CreateElement("button");
            el->SetClass("btn", true);
            el->SetClass("btn--" + style, true);
            el->SetInnerRML(Rml::String(display));
            auto [w, h] = size;
            if (w < 0)
                el->SetClass("btn--full", true);
            else if (w > 0)
                el->SetProperty("width", Rml::String(std::to_string(static_cast<int>(w)) + "dp"));

            el->AddEventListener(Rml::EventId::Click, new SlotEventListener(&slot.events));

            slot.element = line->AppendChild(std::move(el));
        } else {
            if (slot.element->GetParentNode() != line)
                line->AppendChild(slot.element->GetParentNode()->RemoveChild(slot.element));
            slot.element->SetInnerRML(Rml::String(display));
        }

        bool clicked = slot.events.clicked;
        slot.events.clicked = false;
        last_element_ = slot.element;
        last_clicked_ = clicked;
        return clicked;
    }

    std::tuple<bool, bool> RmlImModeLayout::checkbox(const std::string& label, bool value) {
        if (!doc_)
            return {false, value};
        auto* line = ensure_line_container();
        auto& slot = ensure_slot(SlotType::Checkbox, build_slot_id("checkbox", &label));

        if (!slot.element) {
            auto wrapper = doc_->CreateElement("div");
            wrapper->SetClass("setting-row", true);

            auto input = doc_->CreateElement("input");
            input->SetAttribute("type", "checkbox");
            if (value)
                input->SetAttribute("checked", "");

            auto lbl = doc_->CreateElement("span");
            lbl->SetClass("setting-label", true);
            lbl->SetInnerRML(Rml::String(strip_imgui_id(label)));

            slot.events.bool_value = value;
            input->AddEventListener(Rml::EventId::Change, new SlotEventListener(&slot.events));

            wrapper->AppendChild(std::move(input));
            wrapper->AppendChild(std::move(lbl));
            slot.element = line->AppendChild(std::move(wrapper));
        } else {
            if (slot.element->GetParentNode() != line)
                line->AppendChild(slot.element->GetParentNode()->RemoveChild(slot.element));
            auto* input = slot.element->GetChild(0);
            if (input) {
                if (value && !input->HasAttribute("checked"))
                    input->SetAttribute("checked", "");
                else if (!value && input->HasAttribute("checked"))
                    input->RemoveAttribute("checked");
            }
        }

        last_element_ = slot.element;
        last_clicked_ = false;
        if (slot.events.changed) {
            slot.events.changed = false;
            return {true, slot.events.bool_value};
        }
        return {false, value};
    }

    std::tuple<bool, int> RmlImModeLayout::radio_button(const std::string& label, int current, int value) {
        if (!doc_)
            return {false, current};
        auto* line = ensure_line_container();
        auto& slot = ensure_slot(SlotType::RadioButton, build_slot_id("radio_button", &label));
        const bool selected = (current == value);

        if (!slot.element) {
            auto wrapper = doc_->CreateElement("div");
            wrapper->SetClass("im-radio", true);
            if (selected)
                wrapper->SetClass("selected", true);

            auto dot = doc_->CreateElement("span");
            dot->SetClass("im-radio-dot", true);
            dot->SetInnerRML(selected ? "\xe2\x97\x89" : "\xe2\x97\x8b");

            auto lbl = doc_->CreateElement("span");
            lbl->SetClass("im-radio-label", true);
            lbl->SetInnerRML(Rml::String(strip_imgui_id(label)));

            wrapper->AddEventListener(Rml::EventId::Click, new SlotEventListener(&slot.events));

            wrapper->AppendChild(std::move(dot));
            wrapper->AppendChild(std::move(lbl));
            slot.element = line->AppendChild(std::move(wrapper));
        } else {
            if (slot.element->GetParentNode() != line)
                line->AppendChild(slot.element->GetParentNode()->RemoveChild(slot.element));
            slot.element->SetClass("selected", selected);
            auto* dot = slot.element->GetChild(0);
            if (dot)
                dot->SetInnerRML(selected ? "\xe2\x97\x89" : "\xe2\x97\x8b");
        }

        bool clicked = slot.events.clicked;
        last_element_ = slot.element;
        last_clicked_ = clicked;
        if (clicked) {
            slot.events.clicked = false;
            return {true, value};
        }
        return {false, current};
    }

    // ── Sliders ─────────────────────────────────────────────

    std::tuple<bool, float> RmlImModeLayout::slider_float(const std::string& label,
                                                          float value, float min, float max) {
        if (!doc_)
            return {false, value};
        auto* line = ensure_line_container();
        auto& slot = ensure_slot(SlotType::SliderFloat, build_slot_id("slider_float", &label));

        if (!slot.element) {
            auto wrapper = doc_->CreateElement("div");
            wrapper->SetClass("setting-row", true);

            auto lbl = doc_->CreateElement("span");
            lbl->SetClass("prop-label", true);
            lbl->SetInnerRML(Rml::String(strip_imgui_id(label)));

            auto input = doc_->CreateElement("input");
            input->SetAttribute("type", "range");
            input->SetAttribute("min", Rml::String(std::to_string(min)));
            input->SetAttribute("max", Rml::String(std::to_string(max)));
            input->SetAttribute("step", "any");
            input->SetAttribute("value", Rml::String(std::to_string(value)));
            input->SetClass("setting-slider", true);

            slot.events.float_value = value;
            input->AddEventListener(Rml::EventId::Change, new SlotEventListener(&slot.events));

            auto val_text = doc_->CreateElement("span");
            val_text->SetClass("slider-value", true);
            val_text->SetInnerRML(Rml::String(std::format("{:.2f}", value)));

            wrapper->AppendChild(std::move(lbl));
            wrapper->AppendChild(std::move(input));
            wrapper->AppendChild(std::move(val_text));
            slot.element = line->AppendChild(std::move(wrapper));
            apply_item_width(slot.element);
        } else {
            if (slot.element->GetParentNode() != line)
                line->AppendChild(slot.element->GetParentNode()->RemoveChild(slot.element));
            auto* input = slot.element->GetChild(1);
            if (input && !slot.events.changed)
                input->SetAttribute("value", Rml::String(std::to_string(value)));
            auto* val_text = slot.element->GetChild(2);
            if (val_text) {
                float display_val = slot.events.changed ? slot.events.float_value : value;
                val_text->SetInnerRML(Rml::String(std::format("{:.2f}", display_val)));
            }
        }

        last_element_ = slot.element;
        last_clicked_ = false;
        if (slot.events.changed) {
            slot.events.changed = false;
            return {true, slot.events.float_value};
        }
        return {false, value};
    }

    std::tuple<bool, int> RmlImModeLayout::slider_int(const std::string& label,
                                                      int value, int min, int max) {
        if (!doc_)
            return {false, value};
        auto* line = ensure_line_container();
        auto& slot = ensure_slot(SlotType::SliderInt, build_slot_id("slider_int", &label));

        if (!slot.element) {
            auto wrapper = doc_->CreateElement("div");
            wrapper->SetClass("setting-row", true);

            auto lbl = doc_->CreateElement("span");
            lbl->SetClass("prop-label", true);
            lbl->SetInnerRML(Rml::String(strip_imgui_id(label)));

            auto input = doc_->CreateElement("input");
            input->SetAttribute("type", "range");
            input->SetAttribute("min", Rml::String(std::to_string(min)));
            input->SetAttribute("max", Rml::String(std::to_string(max)));
            input->SetAttribute("step", "1");
            input->SetAttribute("value", Rml::String(std::to_string(value)));
            input->SetClass("setting-slider", true);

            slot.events.float_value = static_cast<float>(value);
            input->AddEventListener(Rml::EventId::Change, new SlotEventListener(&slot.events));

            auto val_text = doc_->CreateElement("span");
            val_text->SetClass("slider-value", true);
            val_text->SetInnerRML(Rml::String(std::to_string(value)));

            wrapper->AppendChild(std::move(lbl));
            wrapper->AppendChild(std::move(input));
            wrapper->AppendChild(std::move(val_text));
            slot.element = line->AppendChild(std::move(wrapper));
            apply_item_width(slot.element);
        } else {
            if (slot.element->GetParentNode() != line)
                line->AppendChild(slot.element->GetParentNode()->RemoveChild(slot.element));
            auto* input = slot.element->GetChild(1);
            if (input && !slot.events.changed)
                input->SetAttribute("value", Rml::String(std::to_string(value)));
            auto* val_text = slot.element->GetChild(2);
            if (val_text) {
                int display_val = slot.events.changed ? static_cast<int>(slot.events.float_value) : value;
                val_text->SetInnerRML(Rml::String(std::to_string(display_val)));
            }
        }

        last_element_ = slot.element;
        last_clicked_ = false;
        if (slot.events.changed) {
            slot.events.changed = false;
            return {true, static_cast<int>(std::round(slot.events.float_value))};
        }
        return {false, value};
    }

    std::tuple<bool, std::tuple<float, float>> RmlImModeLayout::slider_float2(
        const std::string& label, std::tuple<float, float> value, float min, float max) {
        auto [v0, v1] = value;
        auto [c0, nv0] = slider_float(label + " X", v0, min, max);
        same_line();
        auto [c1, nv1] = slider_float(label + " Y", v1, min, max);
        return {c0 || c1, {nv0, nv1}};
    }

    std::tuple<bool, std::tuple<float, float, float>> RmlImModeLayout::slider_float3(
        const std::string& label, std::tuple<float, float, float> value, float min, float max) {
        auto [v0, v1, v2] = value;
        auto [c0, nv0] = slider_float(label + " X", v0, min, max);
        same_line();
        auto [c1, nv1] = slider_float(label + " Y", v1, min, max);
        same_line();
        auto [c2, nv2] = slider_float(label + " Z", v2, min, max);
        return {c0 || c1 || c2, {nv0, nv1, nv2}};
    }

    // ── Drags ───────────────────────────────────────────────

    std::tuple<bool, float> RmlImModeLayout::drag_float(const std::string& label, float value,
                                                        float /*speed*/, float min, float max) {
        return slider_float(label, value, min, max);
    }

    std::tuple<bool, int> RmlImModeLayout::drag_int(const std::string& label, int value,
                                                    float /*speed*/, int min, int max) {
        return slider_int(label, value, min, max);
    }

    // ── Input ───────────────────────────────────────────────

    std::tuple<bool, std::string> RmlImModeLayout::input_text(const std::string& label,
                                                              const std::string& value) {
        if (!doc_)
            return {false, value};
        auto* line = ensure_line_container();
        auto& slot = ensure_slot(SlotType::InputText, build_slot_id("input_text", &label));

        if (!slot.element) {
            auto wrapper = doc_->CreateElement("div");
            wrapper->SetClass("setting-row", true);

            auto lbl = doc_->CreateElement("span");
            lbl->SetClass("prop-label", true);
            lbl->SetInnerRML(Rml::String(strip_imgui_id(label)));

            auto input = doc_->CreateElement("input");
            input->SetAttribute("type", "text");
            input->SetAttribute("value", Rml::String(value));
            input->SetProperty("flex", "1");

            slot.events.string_value = value;
            input->AddEventListener(Rml::EventId::Change, new SlotEventListener(&slot.events));

            wrapper->AppendChild(std::move(lbl));
            wrapper->AppendChild(std::move(input));
            slot.element = line->AppendChild(std::move(wrapper));
            apply_item_width(slot.element);
        } else {
            if (slot.element->GetParentNode() != line)
                line->AppendChild(slot.element->GetParentNode()->RemoveChild(slot.element));
        }

        last_element_ = slot.element;
        last_clicked_ = false;
        if (slot.events.changed) {
            slot.events.changed = false;
            return {true, slot.events.string_value};
        }
        return {false, value};
    }

    std::tuple<bool, std::string> RmlImModeLayout::input_text_with_hint(const std::string& label,
                                                                        const std::string& /*hint*/,
                                                                        const std::string& value) {
        return input_text(label, value);
    }

    std::tuple<bool, std::string> RmlImModeLayout::input_text_enter(const std::string& label,
                                                                    const std::string& value) {
        return input_text(label, value);
    }

    std::tuple<bool, float> RmlImModeLayout::input_float(const std::string& label, float value,
                                                         float /*step*/, float /*step_fast*/,
                                                         const std::string& /*format*/) {
        if (!doc_)
            return {false, value};
        auto* line = ensure_line_container();
        auto& slot = ensure_slot(SlotType::InputFloat, build_slot_id("input_float", &label));

        if (!slot.element) {
            auto wrapper = doc_->CreateElement("div");
            wrapper->SetClass("setting-row", true);

            auto display = strip_imgui_id(label);
            if (!display.empty()) {
                auto lbl = doc_->CreateElement("span");
                lbl->SetClass("prop-label", true);
                lbl->SetInnerRML(Rml::String(display));
                wrapper->AppendChild(std::move(lbl));
            }

            auto input = doc_->CreateElement("input");
            input->SetAttribute("type", "text");
            input->SetAttribute("value", Rml::String(std::format("{:.3f}", value)));
            input->SetClass("number-input", true);
            input->SetProperty("flex", "1");

            slot.events.float_value = value;
            slot.events.string_value = std::format("{:.3f}", value);
            input->AddEventListener(Rml::EventId::Change, new SlotEventListener(&slot.events));

            wrapper->AppendChild(std::move(input));
            slot.element = line->AppendChild(std::move(wrapper));
            apply_item_width(slot.element);
        } else {
            if (slot.element->GetParentNode() != line)
                line->AppendChild(slot.element->GetParentNode()->RemoveChild(slot.element));
        }

        last_element_ = slot.element;
        last_clicked_ = false;
        if (slot.events.changed) {
            slot.events.changed = false;
            try {
                float parsed = std::stof(slot.events.string_value);
                return {true, parsed};
            } catch (const std::exception&) {
                return {false, value};
            }
        }
        return {false, value};
    }

    std::tuple<bool, int> RmlImModeLayout::input_int(const std::string& label, int value,
                                                     int /*step*/, int /*step_fast*/) {
        if (!doc_)
            return {false, value};
        auto* line = ensure_line_container();
        auto& slot = ensure_slot(SlotType::InputInt, build_slot_id("input_int", &label));

        if (!slot.element) {
            auto wrapper = doc_->CreateElement("div");
            wrapper->SetClass("setting-row", true);

            auto display = strip_imgui_id(label);
            if (!display.empty()) {
                auto lbl = doc_->CreateElement("span");
                lbl->SetClass("prop-label", true);
                lbl->SetInnerRML(Rml::String(display));
                wrapper->AppendChild(std::move(lbl));
            }

            auto input = doc_->CreateElement("input");
            input->SetAttribute("type", "text");
            input->SetAttribute("value", Rml::String(std::to_string(value)));
            input->SetClass("number-input", true);
            input->SetProperty("flex", "1");

            slot.events.string_value = std::to_string(value);
            input->AddEventListener(Rml::EventId::Change, new SlotEventListener(&slot.events));

            wrapper->AppendChild(std::move(input));
            slot.element = line->AppendChild(std::move(wrapper));
            apply_item_width(slot.element);
        } else {
            if (slot.element->GetParentNode() != line)
                line->AppendChild(slot.element->GetParentNode()->RemoveChild(slot.element));
        }

        last_element_ = slot.element;
        last_clicked_ = false;
        if (slot.events.changed) {
            slot.events.changed = false;
            try {
                int parsed = std::stoi(slot.events.string_value);
                return {true, parsed};
            } catch (const std::exception&) {
                return {false, value};
            }
        }
        return {false, value};
    }

    std::tuple<bool, int> RmlImModeLayout::input_int_formatted(const std::string& label, int value,
                                                               int step, int step_fast) {
        return input_int(label, value, step, step_fast);
    }

    std::tuple<bool, float> RmlImModeLayout::stepper_float(const std::string& label, float value,
                                                           const std::vector<float>& /*steps*/) {
        return input_float(label, value);
    }

    std::tuple<bool, std::string> RmlImModeLayout::path_input(const std::string& label,
                                                              const std::string& value,
                                                              bool /*folder_mode*/,
                                                              const std::string& /*dialog_title*/) {
        return input_text(label, value);
    }

    // ── Color ───────────────────────────────────────────────

    std::tuple<bool, std::tuple<float, float, float>> RmlImModeLayout::color_edit3(
        const std::string& label, std::tuple<float, float, float> color) {
        if (!doc_)
            return {false, color};

        auto [r, g, b] = color;
        auto* line = ensure_line_container();
        auto& slot = ensure_slot(SlotType::Label, build_slot_id("color_edit3", &label));

        auto css = std::format("rgba({},{},{},255)",
                               static_cast<int>(r * 255), static_cast<int>(g * 255), static_cast<int>(b * 255));
        if (!slot.element) {
            auto wrapper = doc_->CreateElement("div");
            wrapper->SetClass("setting-row", true);

            auto lbl = doc_->CreateElement("span");
            lbl->SetClass("prop-label", true);
            lbl->SetInnerRML(Rml::String(strip_imgui_id(label)));

            auto swatch = doc_->CreateElement("div");
            swatch->SetClass("im-color-swatch", true);
            swatch->SetProperty("background-color", Rml::String(css));

            wrapper->AppendChild(std::move(lbl));
            wrapper->AppendChild(std::move(swatch));
            slot.element = line->AppendChild(std::move(wrapper));
        } else {
            if (slot.element->GetParentNode() != line)
                line->AppendChild(slot.element->GetParentNode()->RemoveChild(slot.element));
            auto* swatch = slot.element->GetChild(1);
            if (swatch)
                swatch->SetProperty("background-color", Rml::String(css));
        }

        auto [cr, nr] = slider_float(label + " R", r, 0.0f, 1.0f);
        same_line();
        auto [cg, ng] = slider_float(label + " G", g, 0.0f, 1.0f);
        same_line();
        auto [cb, nb_val] = slider_float(label + " B", b, 0.0f, 1.0f);
        bool changed = cr || cg || cb;
        return {changed, {nr, ng, nb_val}};
    }

    std::tuple<bool, std::tuple<float, float, float, float>> RmlImModeLayout::color_edit4(
        const std::string& label, std::tuple<float, float, float, float> color) {
        auto [r, g, b, a] = color;
        auto [c3, rgb] = color_edit3(label, {r, g, b});
        auto [ca, na] = slider_float(label + " A", a, 0.0f, 1.0f);
        auto [nr, ng, nb_val] = rgb;
        return {c3 || ca, {nr, ng, nb_val, na}};
    }

    std::tuple<bool, std::tuple<float, float, float>> RmlImModeLayout::color_picker3(
        const std::string& label, std::tuple<float, float, float> color) {
        return color_edit3(label, color);
    }

    bool RmlImModeLayout::color_button(const std::string& label, nb::object color,
                                       std::tuple<float, float> size) {
        if (!doc_)
            return false;
        auto css = color_to_css(color);
        auto* line = ensure_line_container();
        auto& slot = ensure_slot(SlotType::Button, build_slot_id("color_button", &label));

        if (!slot.element) {
            auto el = doc_->CreateElement("button");
            el->SetClass("btn", true);
            el->SetClass("im-color-btn", true);
            if (!css.empty())
                el->SetProperty("background-color", Rml::String(css));
            auto [w, h] = size;
            if (w > 0)
                el->SetProperty("width", Rml::String(std::to_string(static_cast<int>(w)) + "dp"));
            if (h > 0)
                el->SetProperty("height", Rml::String(std::to_string(static_cast<int>(h)) + "dp"));

            el->AddEventListener(Rml::EventId::Click, new SlotEventListener(&slot.events));

            slot.element = line->AppendChild(std::move(el));
        } else {
            if (slot.element->GetParentNode() != line)
                line->AppendChild(slot.element->GetParentNode()->RemoveChild(slot.element));
            if (!css.empty())
                slot.element->SetProperty("background-color", Rml::String(css));
        }

        bool clicked = slot.events.clicked;
        slot.events.clicked = false;
        last_element_ = slot.element;
        last_clicked_ = clicked;
        return clicked;
    }

    // ── Selection ───────────────────────────────────────────

    std::tuple<bool, int> RmlImModeLayout::combo(const std::string& label, int current_idx,
                                                 const std::vector<std::string>& items) {
        if (!doc_)
            return {false, current_idx};
        auto* line = ensure_line_container();
        auto& slot = ensure_slot(SlotType::Combo, build_slot_id("combo", &label));

        if (!slot.element) {
            auto wrapper = doc_->CreateElement("div");
            wrapper->SetClass("setting-row", true);

            auto lbl = doc_->CreateElement("span");
            lbl->SetClass("prop-label", true);
            lbl->SetInnerRML(Rml::String(strip_imgui_id(label)));

            auto select = doc_->CreateElement("select");
            select->SetProperty("flex", "1");
            for (int i = 0; i < static_cast<int>(items.size()); ++i) {
                auto option = doc_->CreateElement("option");
                option->SetAttribute("value", Rml::String(std::to_string(i)));
                option->SetInnerRML(Rml::String(items[i]));
                if (i == current_idx)
                    option->SetAttribute("selected", "");
                select->AppendChild(std::move(option));
            }

            slot.events.int_value = current_idx;
            select->AddEventListener(Rml::EventId::Change, new SlotEventListener(&slot.events));

            wrapper->AppendChild(std::move(lbl));
            wrapper->AppendChild(std::move(select));
            slot.element = line->AppendChild(std::move(wrapper));
            apply_item_width(slot.element);
        } else {
            if (slot.element->GetParentNode() != line)
                line->AppendChild(slot.element->GetParentNode()->RemoveChild(slot.element));
            auto* select = slot.element->GetChild(1);
            if (select && !slot.events.changed)
                select->SetAttribute("value", Rml::String(std::to_string(current_idx)));
        }

        last_element_ = slot.element;
        last_clicked_ = false;
        if (slot.events.changed) {
            slot.events.changed = false;
            return {true, slot.events.int_value};
        }
        return {false, current_idx};
    }

    std::tuple<bool, int> RmlImModeLayout::listbox(const std::string& label, int current_idx,
                                                   const std::vector<std::string>& items,
                                                   int /*height_items*/) {
        return combo(label, current_idx, items);
    }

    bool RmlImModeLayout::selectable(const std::string& label, bool selected, float /*height*/) {
        if (!doc_)
            return false;
        auto* line = ensure_line_container();
        auto& slot = ensure_slot(SlotType::Selectable, build_slot_id("selectable", &label));
        const auto display = strip_imgui_id(label);

        if (!slot.element) {
            auto el = doc_->CreateElement("div");
            el->SetClass("context-menu-item", true);
            if (selected)
                el->SetClass("active", true);
            el->SetInnerRML(Rml::String(display));

            el->AddEventListener(Rml::EventId::Click, new SlotEventListener(&slot.events));

            slot.element = line->AppendChild(std::move(el));
        } else {
            if (slot.element->GetParentNode() != line)
                line->AppendChild(slot.element->GetParentNode()->RemoveChild(slot.element));
            slot.element->SetClass("active", selected);
            slot.element->SetInnerRML(Rml::String(display));
        }

        bool clicked = slot.events.clicked;
        slot.events.clicked = false;
        last_element_ = slot.element;
        last_clicked_ = clicked;
        return clicked;
    }

    // ── Layout ──────────────────────────────────────────────

    void RmlImModeLayout::separator() {
        if (!doc_)
            return;
        finish_current_line();
        assert(!containers_.empty());
        auto& level = containers_.back();
        auto& slot = ensure_slot(SlotType::Separator, build_slot_id("separator"));

        if (!slot.element) {
            auto el = doc_->CreateElement("div");
            el->SetClass("separator", true);
            slot.element = level.parent->AppendChild(std::move(el));
        } else if (slot.element->GetParentNode() != level.parent) {
            level.parent->AppendChild(slot.element->GetParentNode()->RemoveChild(slot.element));
        }
    }

    void RmlImModeLayout::spacing() {
        if (!doc_)
            return;
        finish_current_line();
        assert(!containers_.empty());
        auto& level = containers_.back();
        auto& slot = ensure_slot(SlotType::Spacing, build_slot_id("spacing"));

        if (!slot.element) {
            auto el = doc_->CreateElement("div");
            el->SetProperty("height", "8dp");
            slot.element = level.parent->AppendChild(std::move(el));
        } else if (slot.element->GetParentNode() != level.parent) {
            level.parent->AppendChild(slot.element->GetParentNode()->RemoveChild(slot.element));
        }
    }

    void RmlImModeLayout::same_line(float /*offset*/, float /*spacing*/) {
        next_same_line_ = true;
    }

    void RmlImModeLayout::new_line() {
        finish_current_line();
    }

    void RmlImModeLayout::indent(float width) {
        indent_level_ += (width > 0) ? static_cast<int>(width / 20.0f) : 1;
    }

    void RmlImModeLayout::unindent(float width) {
        indent_level_ -= (width > 0) ? static_cast<int>(width / 20.0f) : 1;
        if (indent_level_ < 0)
            indent_level_ = 0;
    }

    void RmlImModeLayout::set_next_item_width(float /*width*/) {}
    void RmlImModeLayout::begin_group() {}
    void RmlImModeLayout::end_group() {}

    // ── Grouping ────────────────────────────────────────────

    bool RmlImModeLayout::collapsing_header(const std::string& label, bool default_open) {
        if (!doc_)
            return false;
        finish_current_line();
        auto& slot = ensure_slot(SlotType::CollapsHeader, build_slot_id("collapsing_header", &label));

        if (!slot.element) {
            slot.events.open = force_next_open_ || default_open;
            force_next_open_ = false;

            auto header = doc_->CreateElement("div");
            header->SetClass("section-header", true);

            auto arrow = doc_->CreateElement("span");
            arrow->SetClass("section-arrow", true);
            arrow->SetInnerRML(slot.events.open ? "\xe2\x96\xbc" : "\xe2\x96\xb6");

            auto text = doc_->CreateElement("span");
            text->SetInnerRML(Rml::String(strip_imgui_id(label)));

            header->AddEventListener(Rml::EventId::Click, new SlotEventListener(&slot.events));

            header->AppendChild(std::move(arrow));
            header->AppendChild(std::move(text));

            assert(!containers_.empty());
            slot.element = containers_.back().parent->AppendChild(std::move(header));
        } else {
            if (force_next_open_) {
                slot.events.open = true;
                force_next_open_ = false;
            }
            if (slot.events.clicked) {
                slot.events.clicked = false;
                slot.events.open = !slot.events.open;
            }
            auto* arrow = slot.element->GetChild(0);
            if (arrow)
                arrow->SetInnerRML(slot.events.open ? "\xe2\x96\xbc" : "\xe2\x96\xb6");
        }

        last_element_ = slot.element;
        last_clicked_ = false;
        return slot.events.open;
    }

    bool RmlImModeLayout::tree_node(const std::string& label) {
        return collapsing_header(label, false);
    }

    bool RmlImModeLayout::tree_node_ex(const std::string& label, const std::string& /*flags*/) {
        return collapsing_header(label, false);
    }

    void RmlImModeLayout::set_next_item_open(bool is_open) {
        force_next_open_ = is_open;
    }

    void RmlImModeLayout::tree_pop() {}

    // ── Tables ──────────────────────────────────────────────

    bool RmlImModeLayout::begin_table(const std::string& id, int columns) {
        if (!doc_)
            return false;
        assert(columns > 0);
        finish_current_line();

        assert(!containers_.empty());
        auto& slot = ensure_slot(SlotType::Line, build_id("table:" + id));

        if (!slot.element) {
            auto el = doc_->CreateElement("div");
            el->SetClass("im-table", true);
            slot.element = containers_.back().parent->AppendChild(std::move(el));
        }

        while (slot.element->HasChildNodes())
            removed_elements_.push_back(slot.element->RemoveChild(slot.element->GetFirstChild()));

        table_ = TableState{};
        table_->num_columns = columns;
        table_->column_widths.resize(columns, 0.0f);
        table_->table_element = slot.element;
        table_->current_row = nullptr;
        table_->current_cell = nullptr;
        table_->current_column = -1;
        return true;
    }

    void RmlImModeLayout::table_setup_column(const std::string& /*label*/, float width) {
        if (!table_)
            return;
        int col = 0;
        for (auto& w : table_->column_widths) {
            if (w == 0.0f) {
                w = width;
                break;
            }
            col++;
        }
    }

    void RmlImModeLayout::end_table() {
        table_.reset();
    }

    void RmlImModeLayout::table_next_row() {
        if (!table_ || !table_->table_element || !doc_)
            return;
        auto row = doc_->CreateElement("div");
        row->SetClass("im-table-row", true);
        table_->current_row = table_->table_element->AppendChild(std::move(row));
        table_->current_cell = nullptr;
        table_->current_column = -1;
    }

    void RmlImModeLayout::table_next_column() {
        if (!table_ || !table_->current_row || !doc_)
            return;
        table_->current_column++;
        auto cell = doc_->CreateElement("div");
        cell->SetClass("im-table-cell", true);

        int col = table_->current_column;
        if (col < static_cast<int>(table_->column_widths.size())) {
            float w = table_->column_widths[col];
            if (w > 0.0f)
                cell->SetProperty("width", Rml::String(std::to_string(static_cast<int>(w)) + "dp"));
            else
                cell->SetProperty("flex", "1");
        }

        table_->current_cell = table_->current_row->AppendChild(std::move(cell));
    }

    bool RmlImModeLayout::table_set_column_index(int column) {
        if (!table_ || !table_->current_row || !doc_)
            return false;

        while (table_->current_column < column - 1)
            table_next_column();
        table_next_column();
        return true;
    }

    void RmlImModeLayout::table_headers_row() {
        table_next_row();
    }

    void RmlImModeLayout::table_set_bg_color(int /*target*/, nb::object color) {
        if (!table_ || !table_->current_row)
            return;
        auto css = color_to_css(color);
        if (!css.empty())
            table_->current_row->SetProperty("background-color", Rml::String(css));
    }

    // ── Disabled state ──────────────────────────────────────

    void RmlImModeLayout::begin_disabled(bool disabled) {
        if (disabled) {
            disabled_ = true;
            disabled_depth_++;
        }
    }

    void RmlImModeLayout::end_disabled() {
        if (disabled_depth_ > 0) {
            disabled_depth_--;
            if (disabled_depth_ == 0)
                disabled_ = false;
        }
    }

    // ── Progress ────────────────────────────────────────────

    void RmlImModeLayout::progress_bar(float fraction, const std::string& overlay,
                                       float /*width*/, float /*height*/) {
        if (!doc_)
            return;
        auto* line = ensure_line_container();
        auto& slot = ensure_slot(SlotType::ProgressBar, build_slot_id("progress"));

        if (!slot.element) {
            auto wrapper = doc_->CreateElement("div");
            wrapper->SetProperty("position", "relative");
            wrapper->SetProperty("width", "100%");

            auto prog = doc_->CreateElement("progress");
            prog->SetAttribute("value", Rml::String(std::to_string(fraction)));
            prog->SetAttribute("max", "1");

            auto text = doc_->CreateElement("span");
            text->SetClass("progress__text", true);
            if (!overlay.empty())
                text->SetInnerRML(Rml::String(overlay));
            else
                text->SetInnerRML(Rml::String(std::format("{}%", static_cast<int>(fraction * 100))));

            wrapper->AppendChild(std::move(prog));
            wrapper->AppendChild(std::move(text));
            slot.element = line->AppendChild(std::move(wrapper));
        } else {
            if (slot.element->GetParentNode() != line)
                line->AppendChild(slot.element->GetParentNode()->RemoveChild(slot.element));
            auto* prog = slot.element->GetChild(0);
            if (prog)
                prog->SetAttribute("value", Rml::String(std::to_string(fraction)));
            auto* text = slot.element->GetChild(1);
            if (text) {
                if (!overlay.empty())
                    text->SetInnerRML(Rml::String(overlay));
                else
                    text->SetInnerRML(Rml::String(std::format("{}%", static_cast<int>(fraction * 100))));
            }
        }
        last_element_ = slot.element;
        last_clicked_ = false;
    }

    // ── ID stack ────────────────────────────────────────────

    void RmlImModeLayout::push_id(const std::string& id) { id_stack_.push_back(id); }
    void RmlImModeLayout::push_id_int(int id) { id_stack_.push_back(std::to_string(id)); }
    void RmlImModeLayout::pop_id() {
        if (!id_stack_.empty())
            id_stack_.pop_back();
    }

    // ── Style (no-op for now) ───────────────────────────────

    void RmlImModeLayout::push_style_var_float(const std::string& /*var*/, float /*value*/) {}
    void RmlImModeLayout::push_style_var_vec2(const std::string& /*var*/, std::tuple<float, float> /*value*/) {}
    void RmlImModeLayout::pop_style_var(int /*count*/) {}
    void RmlImModeLayout::push_style_color(const std::string& /*col*/, nb::object /*color*/) {}
    void RmlImModeLayout::pop_style_color(int /*count*/) {}
    void RmlImModeLayout::push_item_width(float width) {
        item_width_stack_.push_back(width);
    }

    void RmlImModeLayout::pop_item_width() {
        if (!item_width_stack_.empty())
            item_width_stack_.pop_back();
    }

    void RmlImModeLayout::apply_item_width(Rml::Element* el) {
        if (item_width_stack_.empty() || !el)
            return;
        float w = item_width_stack_.back();
        if (w > 0) {
            el->SetProperty("width", Rml::String(std::format("{:.0f}dp", w)));
        } else if (w < 0) {
            el->SetProperty("flex", "1");
            if (w != -1.0f)
                el->SetProperty("margin-right", Rml::String(std::format("{:.0f}dp", -w)));
        }
    }

    // ── Tooltip ─────────────────────────────────────────────

    void RmlImModeLayout::set_tooltip(const std::string& text) {
        if (!tooltip_el_ || !last_element_ || !last_element_->IsPseudoClassSet("hover"))
            return;

        auto body_offset = doc_->GetAbsoluteOffset(Rml::BoxArea::Content);
        float local_x = mouse_.pos_x - body_offset.x;
        float local_y = mouse_.pos_y - body_offset.y;

        tooltip_el_->SetInnerRML(Rml::String(text));
        tooltip_el_->SetClass("visible", true);
        tooltip_el_->SetProperty("left", Rml::String(std::format("{:.0f}dp", local_x + 12.0f)));
        tooltip_el_->SetProperty("top", Rml::String(std::format("{:.0f}dp", local_y + 12.0f)));
        tooltip_shown_ = true;
    }

    // ── Item state ──────────────────────────────────────────

    bool RmlImModeLayout::is_item_hovered() {
        return last_element_ && last_element_->IsPseudoClassSet("hover");
    }
    bool RmlImModeLayout::is_item_clicked(int /*button*/) { return last_clicked_; }
    bool RmlImModeLayout::is_item_active() {
        return last_element_ && last_element_->IsPseudoClassSet("active");
    }

    // ── Mouse ───────────────────────────────────────────────

    bool RmlImModeLayout::is_mouse_double_clicked(int /*button*/) { return mouse_.double_clicked; }
    bool RmlImModeLayout::is_mouse_dragging(int /*button*/) { return mouse_.dragging; }
    float RmlImModeLayout::get_mouse_wheel() { return mouse_.wheel; }
    std::tuple<float, float> RmlImModeLayout::get_mouse_delta() {
        return {mouse_.delta_x, mouse_.delta_y};
    }
    std::tuple<float, float> RmlImModeLayout::get_mouse_pos() const {
        return {mouse_.pos_x, mouse_.pos_y};
    }
    void RmlImModeLayout::set_mouse_cursor_hand() {
        if (root_)
            root_->SetProperty("cursor", "pointer");
    }

    // ── Window queries ──────────────────────────────────────

    std::tuple<float, float> RmlImModeLayout::get_content_region_avail() {
        if (!root_)
            return {0.0f, 0.0f};
        auto sz = root_->GetBox().GetSize(Rml::BoxArea::Content);
        return {sz.x, sz.y};
    }
    float RmlImModeLayout::get_window_width() const {
        return root_ ? root_->GetBox().GetSize().x : 0.0f;
    }
    float RmlImModeLayout::get_text_line_height() const { return cached_line_height_; }
    std::tuple<float, float> RmlImModeLayout::get_cursor_pos() { return {0.0f, 0.0f}; }
    std::tuple<float, float> RmlImModeLayout::get_cursor_screen_pos() const {
        if (!root_)
            return {0.0f, 0.0f};
        auto off = root_->GetAbsoluteOffset(Rml::BoxArea::Content);
        return {off.x, off.y};
    }
    void RmlImModeLayout::set_cursor_pos(std::tuple<float, float> /*pos*/) {}
    void RmlImModeLayout::set_cursor_pos_x(float /*x*/) {}
    std::tuple<float, float> RmlImModeLayout::calc_text_size(const std::string& text) {
        float char_width = cached_line_height_ * 0.6f;
        float w = static_cast<float>(text.size()) * char_width;
        return {w, cached_line_height_};
    }
    std::tuple<float, float> RmlImModeLayout::get_window_pos() const {
        if (!root_)
            return {0.0f, 0.0f};
        auto off = root_->GetAbsoluteOffset(Rml::BoxArea::Border);
        return {off.x, off.y};
    }

    // ── Viewport ────────────────────────────────────────────

    std::tuple<float, float> RmlImModeLayout::get_viewport_pos() {
        float x, y, w, h;
        lfs::python::get_viewport_bounds(x, y, w, h);
        return {x, y};
    }
    std::tuple<float, float> RmlImModeLayout::get_viewport_size() {
        float x, y, w, h;
        lfs::python::get_viewport_bounds(x, y, w, h);
        return {w, h};
    }
    float RmlImModeLayout::get_dpi_scale() { return lfs::python::get_shared_dpi_scale(); }

    // ── Window management (no-op) ───────────────────────────

    bool RmlImModeLayout::begin_window(const std::string& /*title*/, int /*flags*/) { return true; }
    std::tuple<bool, bool> RmlImModeLayout::begin_window_closable(const std::string& /*title*/, int /*flags*/) { return {true, true}; }
    void RmlImModeLayout::end_window() {}
    void RmlImModeLayout::push_window_style() {}
    void RmlImModeLayout::pop_window_style() {}
    void RmlImModeLayout::set_next_window_pos(std::tuple<float, float> /*pos*/, bool /*first_use*/) {}
    void RmlImModeLayout::set_next_window_size(std::tuple<float, float> /*size*/, bool /*first_use*/) {}
    void RmlImModeLayout::set_next_window_pos_centered(bool /*first_use*/) {}
    void RmlImModeLayout::set_next_window_bg_alpha(float /*alpha*/) {}
    void RmlImModeLayout::set_next_window_pos_center() {}
    void RmlImModeLayout::set_next_window_pos_viewport_center(bool /*always*/) {}
    void RmlImModeLayout::set_next_window_focus() {}

    // ── Focus ───────────────────────────────────────────────

    void RmlImModeLayout::set_keyboard_focus_here() {}
    bool RmlImModeLayout::is_window_focused() const { return false; }
    bool RmlImModeLayout::is_window_hovered() const { return false; }
    void RmlImModeLayout::capture_keyboard_from_app(bool /*capture*/) {}
    void RmlImModeLayout::capture_mouse_from_app(bool /*capture*/) {}

    // ── Scrolling ───────────────────────────────────────────

    void RmlImModeLayout::set_scroll_here_y(float /*center_y_ratio*/) {}

    bool RmlImModeLayout::begin_child(const std::string& id, std::tuple<float, float> size, bool border) {
        if (!doc_)
            return true;
        finish_current_line();
        assert(!containers_.empty());

        auto& slot = ensure_slot(SlotType::Line, build_id("child:" + id));

        if (!slot.element) {
            auto el = doc_->CreateElement("div");
            el->SetClass("im-child", true);
            auto [w, h] = size;
            if (w > 0)
                el->SetProperty("width", Rml::String(std::to_string(static_cast<int>(w)) + "dp"));
            if (h > 0)
                el->SetProperty("height", Rml::String(std::to_string(static_cast<int>(h)) + "dp"));
            if (border)
                el->SetClass("im-child-bordered", true);
            slot.element = containers_.back().parent->AppendChild(std::move(el));
        }

        while (slot.element->HasChildNodes())
            removed_elements_.push_back(slot.element->RemoveChild(slot.element->GetFirstChild()));

        containers_.push_back({slot.element, {}, 0});
        return true;
    }

    void RmlImModeLayout::end_child() {
        if (containers_.size() <= 1)
            return;
        finish_current_line();
        prune_excess_slots(containers_.back());
        containers_.pop_back();
    }

    // ── Menu bar (no-op) ────────────────────────────────────

    bool RmlImModeLayout::begin_menu_bar() {
        warn_unsupported("begin_menu_bar");
        return false;
    }
    void RmlImModeLayout::end_menu_bar() {}
    bool RmlImModeLayout::begin_menu(const std::string& /*label*/) { return false; }
    void RmlImModeLayout::end_menu() {}
    bool RmlImModeLayout::menu_item(const std::string& /*label*/, bool /*enabled*/, bool /*selected*/) { return false; }
    std::tuple<bool, bool> RmlImModeLayout::menu_item_toggle(const std::string& /*label*/, const std::string& /*shortcut*/, bool selected) { return {false, selected}; }
    bool RmlImModeLayout::menu_item_shortcut(const std::string& /*label*/, const std::string& /*shortcut*/, bool /*enabled*/) { return false; }

    // ── Popups (no-op) ──────────────────────────────────────

    bool RmlImModeLayout::begin_popup(const std::string& id) {
        return begin_popup_modal(id);
    }

    void RmlImModeLayout::open_popup(const std::string& id) {
        popup_open_[id] = true;
    }

    void RmlImModeLayout::end_popup() {
        end_popup_modal();
    }

    bool RmlImModeLayout::begin_context_menu(const std::string& /*id*/) { return false; }
    void RmlImModeLayout::end_context_menu() {}

    bool RmlImModeLayout::begin_popup_modal(const std::string& title) {
        auto it = popup_open_.find(title);
        if (it == popup_open_.end() || !it->second)
            return false;
        if (!popup_backdrop_ || !popup_dialog_)
            return false;

        popup_backdrop_->SetClass("visible", true);
        popup_dialog_->SetClass("visible", true);
        popup_dialog_->SetInnerRML("");

        auto title_el = doc_->CreateElement("div");
        title_el->SetClass("im-popup-title", true);
        title_el->SetInnerRML(Rml::String(strip_imgui_id(title)));
        popup_dialog_->AppendChild(std::move(title_el));

        active_popup_id_ = title;
        finish_current_line();
        containers_.push_back({popup_dialog_, {}, 0});
        return true;
    }

    void RmlImModeLayout::end_popup_modal() {
        if (containers_.size() <= 1)
            return;
        if (active_popup_id_.empty())
            return;
        finish_current_line();
        prune_excess_slots(containers_.back());
        containers_.pop_back();
    }

    void RmlImModeLayout::close_current_popup() {
        if (!active_popup_id_.empty()) {
            popup_open_[active_popup_id_] = false;
            active_popup_id_.clear();
        }
        if (popup_backdrop_)
            popup_backdrop_->SetClass("visible", false);
        if (popup_dialog_)
            popup_dialog_->SetClass("visible", false);
    }

    void RmlImModeLayout::push_modal_style() {}
    void RmlImModeLayout::pop_modal_style() {}

    // ── Images (no-op) ──────────────────────────────────────

    void RmlImModeLayout::image(uint64_t /*texture_id*/, std::tuple<float, float> /*size*/, nb::object /*tint*/) { warn_unsupported("image"); }
    void RmlImModeLayout::image_uv(uint64_t /*texture_id*/, std::tuple<float, float> /*size*/, std::tuple<float, float> /*uv0*/, std::tuple<float, float> /*uv1*/, nb::object /*tint*/) { warn_unsupported("image_uv"); }
    bool RmlImModeLayout::image_button(const std::string& /*id*/, uint64_t /*texture_id*/, std::tuple<float, float> /*size*/, nb::object /*tint*/) {
        warn_unsupported("image_button");
        return false;
    }
    bool RmlImModeLayout::toolbar_button(const std::string& /*id*/, uint64_t /*texture_id*/, std::tuple<float, float> /*size*/, bool /*selected*/, bool /*disabled*/, const std::string& /*tooltip*/) {
        warn_unsupported("toolbar_button");
        return false;
    }
    bool RmlImModeLayout::invisible_button(const std::string& /*id*/, std::tuple<float, float> /*size*/) { return false; }

    // ── Drag-drop (no-op) ───────────────────────────────────

    bool RmlImModeLayout::begin_drag_drop_source() { return false; }
    void RmlImModeLayout::set_drag_drop_payload(const std::string& /*type*/, const std::string& /*data*/) {}
    void RmlImModeLayout::end_drag_drop_source() {}
    bool RmlImModeLayout::begin_drag_drop_target() { return false; }
    std::optional<std::string> RmlImModeLayout::accept_drag_drop_payload(const std::string& /*type*/) { return std::nullopt; }
    void RmlImModeLayout::end_drag_drop_target() {}

    // ── Drawing primitives (no-op) ──────────────────────────

    void RmlImModeLayout::draw_circle(float, float, float, nb::object, int, float) {}
    void RmlImModeLayout::draw_circle_filled(float, float, float, nb::object, int) {}
    void RmlImModeLayout::draw_rect(float, float, float, float, nb::object, float) {}
    void RmlImModeLayout::draw_rect_filled(float, float, float, float, nb::object, bool) {}
    void RmlImModeLayout::draw_rect_rounded(float, float, float, float, nb::object, float, float, bool) {}
    void RmlImModeLayout::draw_rect_rounded_filled(float, float, float, float, nb::object, float, bool) {}
    void RmlImModeLayout::draw_triangle_filled(float, float, float, float, float, float, nb::object, bool) {}
    void RmlImModeLayout::draw_line(float, float, float, float, nb::object, float) {}
    void RmlImModeLayout::draw_polyline(nb::object, nb::object, bool, float) {}
    void RmlImModeLayout::draw_poly_filled(nb::object, nb::object) {}
    void RmlImModeLayout::draw_text(float, float, const std::string&, nb::object, bool) {}
    void RmlImModeLayout::draw_window_rect_filled(float, float, float, float, nb::object) {}
    void RmlImModeLayout::draw_window_rect(float, float, float, float, nb::object, float) {}
    void RmlImModeLayout::draw_window_rect_rounded(float, float, float, float, nb::object, float, float) {}
    void RmlImModeLayout::draw_window_rect_rounded_filled(float, float, float, float, nb::object, float) {}
    void RmlImModeLayout::draw_window_line(float, float, float, float, nb::object, float) {}
    void RmlImModeLayout::draw_window_text(float, float, const std::string&, nb::object) {}
    void RmlImModeLayout::draw_window_triangle_filled(float, float, float, float, float, float, nb::object) {}

    // ── Specialized (no-op) ─────────────────────────────────

    void RmlImModeLayout::crf_curve_preview(const std::string& /*label*/, float, float, float, float, float, float) {}
    std::tuple<bool, std::vector<float>> RmlImModeLayout::chromaticity_diagram(const std::string& /*label*/,
                                                                               float, float, float, float,
                                                                               float, float, float, float, float) {
        return {false, {}};
    }

    // ── Plots (no-op) ───────────────────────────────────────

    void RmlImModeLayout::plot_lines(const std::string& /*label*/, nb::object /*values*/,
                                     float /*scale_min*/, float /*scale_max*/,
                                     std::tuple<float, float> /*size*/) {
        warn_unsupported("plot_lines");
    }

    // ── Sub-layouts (return self as no-op context manager) ──

    nb::object RmlImModeLayout::row() {
        return nb::cast(RmlSubLayout(this, RmlLayoutDirection::Row));
    }
    nb::object RmlImModeLayout::column() {
        return nb::cast(RmlSubLayout(this, RmlLayoutDirection::Column));
    }
    nb::object RmlImModeLayout::split(float /*factor*/) {
        return nb::cast(RmlSubLayout(this, RmlLayoutDirection::Row));
    }
    nb::object RmlImModeLayout::box() {
        return nb::cast(RmlSubLayout(this, RmlLayoutDirection::Column));
    }
    nb::object RmlImModeLayout::grid_flow(int /*columns*/, bool /*even_columns*/, bool /*even_rows*/) {
        return nb::cast(RmlSubLayout(this, RmlLayoutDirection::Row));
    }

    // ── Property binding ────────────────────────────────────

    std::tuple<bool, nb::object> RmlImModeLayout::prop(nb::object data, const std::string& prop_id,
                                                       std::optional<std::string> text) {
        if (!nb::hasattr(data, "get_all_properties"))
            return {false, nb::none()};

        nb::dict all_props = nb::cast<nb::dict>(data.attr("get_all_properties")());
        nb::str prop_key(prop_id.c_str());
        if (!all_props.contains(prop_key))
            return {false, nb::none()};

        nb::object prop_desc = all_props[prop_key];

        const bool has_get = nb::hasattr(data, "get") && PyCallable_Check(data.attr("get").ptr());
        nb::object current_value = has_get
                                       ? data.attr("get")(nb::cast(prop_id))
                                       : data.attr(prop_id.c_str());
        const std::string prop_type = nb::cast<std::string>(
            nb::object(prop_desc.attr("__class__").attr("__name__")));

        std::string prop_name_str;
        if (nb::hasattr(prop_desc, "name"))
            prop_name_str = nb::cast<std::string>(prop_desc.attr("name"));
        std::string display_name = text.value_or(
            !prop_name_str.empty() ? prop_name_str : prop_id);

        std::string description;
        if (nb::hasattr(prop_desc, "description"))
            description = nb::cast<std::string>(prop_desc.attr("description"));

        bool changed = false;
        nb::object new_value = current_value;

        if (prop_type == "FloatProperty") {
            float v = nb::cast<float>(current_value);
            float min_v = nb::cast<float>(prop_desc.attr("min"));
            float max_v = nb::cast<float>(prop_desc.attr("max"));
            float step = nb::cast<float>(prop_desc.attr("step"));

            bool use_slider = (min_v != -std::numeric_limits<float>::infinity() &&
                               max_v != std::numeric_limits<float>::infinity());
            if (use_slider) {
                auto [c, nv] = slider_float(display_name, v, min_v, max_v);
                changed = c;
                new_value = nb::cast(nv);
            } else {
                auto [c, nv] = drag_float(display_name, v, step, min_v, max_v);
                changed = c;
                new_value = nb::cast(nv);
            }
        } else if (prop_type == "IntProperty") {
            int v = nb::cast<int>(current_value);
            int min_v = nb::cast<int>(prop_desc.attr("min"));
            int max_v = nb::cast<int>(prop_desc.attr("max"));

            bool use_slider = (min_v != -(1 << 30) && max_v != (1 << 30));
            if (use_slider) {
                auto [c, nv] = slider_int(display_name, v, min_v, max_v);
                changed = c;
                new_value = nb::cast(nv);
            } else {
                auto [c, nv] = drag_int(display_name, v, 1.0f, min_v, max_v);
                changed = c;
                new_value = nb::cast(nv);
            }
        } else if (prop_type == "BoolProperty") {
            bool v = nb::cast<bool>(current_value);
            auto [c, nv] = checkbox(display_name, v);
            changed = c;
            new_value = nb::cast(nv);
        } else if (prop_type == "StringProperty") {
            std::string v = nb::cast<std::string>(current_value);
            auto [c, nv] = input_text(display_name, v);
            changed = c;
            new_value = nb::cast(nv);
        } else if (prop_type == "EnumProperty") {
            nb::object items_obj = prop_desc.attr("items");
            std::vector<std::string> items;
            int current_idx = 0;
            std::string current_id = nb::cast<std::string>(current_value);

            size_t idx = 0;
            for (auto item : items_obj) {
                nb::tuple t = nb::cast<nb::tuple>(item);
                std::string identifier = nb::cast<std::string>(t[0]);
                std::string lbl = nb::cast<std::string>(t[1]);
                items.push_back(lbl);
                if (identifier == current_id)
                    current_idx = static_cast<int>(idx);
                idx++;
            }

            auto [c, new_idx] = combo(display_name, current_idx, items);
            changed = c;
            if (changed && new_idx >= 0 && new_idx < static_cast<int>(items.size())) {
                nb::tuple t = nb::cast<nb::tuple>(items_obj[new_idx]);
                new_value = t[0];
            } else {
                new_value = nb::cast(current_id);
            }
        } else if (prop_type == "FloatVectorProperty") {
            nb::tuple t = nb::cast<nb::tuple>(current_value);
            int size = nb::cast<int>(prop_desc.attr("size"));
            std::string subtype;
            if (nb::hasattr(prop_desc, "subtype"))
                subtype = nb::cast<std::string>(prop_desc.attr("subtype"));

            if (size == 3) {
                float v0 = nb::cast<float>(t[0]);
                float v1 = nb::cast<float>(t[1]);
                float v2 = nb::cast<float>(t[2]);
                if (subtype == "COLOR" || subtype == "COLOR_GAMMA") {
                    auto [c, rgb] = color_edit3(display_name, {v0, v1, v2});
                    changed = c;
                    auto [r, g, b] = rgb;
                    new_value = nb::make_tuple(r, g, b);
                } else {
                    auto [c, val] = slider_float3(display_name, {v0, v1, v2}, -10.0f, 10.0f);
                    changed = c;
                    auto [r, g, b] = val;
                    new_value = nb::make_tuple(r, g, b);
                }
            } else if (size == 4) {
                float v0 = nb::cast<float>(t[0]);
                float v1 = nb::cast<float>(t[1]);
                float v2 = nb::cast<float>(t[2]);
                float v3 = nb::cast<float>(t[3]);
                if (subtype == "COLOR" || subtype == "COLOR_GAMMA") {
                    auto [c, rgba] = color_edit4(display_name, {v0, v1, v2, v3});
                    changed = c;
                    auto [r, g, b, a] = rgba;
                    new_value = nb::make_tuple(r, g, b, a);
                } else {
                    auto [c0, nv0] = slider_float(display_name + " X", v0, -10.0f, 10.0f);
                    same_line();
                    auto [c1, nv1] = slider_float(display_name + " Y", v1, -10.0f, 10.0f);
                    same_line();
                    auto [c2, nv2] = slider_float(display_name + " Z", v2, -10.0f, 10.0f);
                    same_line();
                    auto [c3, nv3] = slider_float(display_name + " W", v3, -10.0f, 10.0f);
                    changed = c0 || c1 || c2 || c3;
                    new_value = nb::make_tuple(nv0, nv1, nv2, nv3);
                }
            } else if (size == 2) {
                float v0 = nb::cast<float>(t[0]);
                float v1 = nb::cast<float>(t[1]);
                auto [c, val] = slider_float2(display_name, {v0, v1}, -10.0f, 10.0f);
                changed = c;
                auto [r0, r1] = val;
                new_value = nb::make_tuple(r0, r1);
            }
        } else if (prop_type == "IntVectorProperty") {
            nb::tuple t = nb::cast<nb::tuple>(current_value);
            int size = nb::cast<int>(prop_desc.attr("size"));

            if (size >= 2 && size <= 4) {
                bool any_changed = false;
                std::vector<int> vals(size);
                for (int i = 0; i < size; ++i)
                    vals[i] = nb::cast<int>(t[i]);
                for (int i = 0; i < size; ++i) {
                    std::string suffix = (size <= 3)
                                             ? std::string(1, "XYZ"[i])
                                             : std::string(1, "XYZW"[i]);
                    auto [c, nv] = slider_int(display_name + " " + suffix, vals[i], -100, 100);
                    if (c) {
                        any_changed = true;
                        vals[i] = nv;
                    }
                    if (i < size - 1)
                        same_line();
                }
                changed = any_changed;
                if (size == 2)
                    new_value = nb::make_tuple(vals[0], vals[1]);
                else if (size == 3)
                    new_value = nb::make_tuple(vals[0], vals[1], vals[2]);
                else
                    new_value = nb::make_tuple(vals[0], vals[1], vals[2], vals[3]);
            }
        } else if (prop_type == "TensorProperty") {
            std::string info_str = "None";
            if (!current_value.is_none()) {
                try {
                    nb::object shape = current_value.attr("shape");
                    nb::object dtype = current_value.attr("dtype");
                    nb::object device = current_value.attr("device");

                    std::string shape_str = "[";
                    size_t si = 0;
                    for (auto dim : shape) {
                        if (si > 0)
                            shape_str += ", ";
                        shape_str += std::to_string(nb::cast<int64_t>(dim));
                        ++si;
                    }
                    shape_str += "]";

                    std::string dtype_str = nb::cast<std::string>(nb::str(dtype));
                    info_str = "Tensor(" + shape_str + ", " + dtype_str + ", " +
                               nb::cast<std::string>(nb::str(device)) + ")";
                } catch (...) {
                    info_str = "Tensor(...)";
                }
            }
            text_disabled(display_name + ": " + info_str);
        }

        if (!description.empty() && is_item_hovered())
            set_tooltip(description);

        if (changed) {
            const bool has_set = nb::hasattr(data, "set") && PyCallable_Check(data.attr("set").ptr());
            if (has_set)
                data.attr("set")(nb::cast(prop_id), new_value);
            else
                nb::setattr(data, prop_id.c_str(), new_value);

            if (nb::hasattr(prop_desc, "update")) {
                nb::object update_cb = prop_desc.attr("update");
                if (!update_cb.is_none() && PyCallable_Check(update_cb.ptr())) {
                    try {
                        update_cb(data, nb::none());
                    } catch (const nb::python_error& e) {
                        LOG_ERROR("Property update callback error: {}", e.what());
                    }
                }
            }
        }

        return {changed, new_value};
    }

    bool RmlImModeLayout::prop_enum(nb::object data, const std::string& prop_id,
                                    const std::string& value, const std::string& text) {
        const std::string current = nb::cast<std::string>(data.attr(prop_id.c_str()));
        const bool selected = (current == value);
        const std::string& display = text.empty() ? value : text;

        bool clicked;
        if (selected)
            clicked = button_styled(display, "primary");
        else
            clicked = button(display);

        if (clicked && !selected) {
            data.attr(prop_id.c_str()) = nb::cast(value);
            return true;
        }
        return false;
    }

    nb::object RmlImModeLayout::operator_(const std::string& operator_id, const std::string& text,
                                          const std::string& /*icon*/) {
        const auto* desc = vis::op::operators().getDescriptor(operator_id);
        std::string label_str = text.empty() ? (desc ? desc->label : operator_id) : text;
        std::string btn_text = LOC(label_str.c_str());

        const bool can_execute = vis::op::operators().poll(operator_id);

        if (!can_execute)
            begin_disabled();

        bool clicked = button(btn_text);

        if (!can_execute)
            end_disabled();

        if (clicked)
            vis::op::operators().invoke(operator_id);

        return nb::cast(PyOperatorProperties(operator_id));
    }

    std::tuple<bool, int> RmlImModeLayout::prop_search(nb::object /*data*/, const std::string& prop_id,
                                                       nb::object search_data, const std::string& search_prop,
                                                       const std::string& text) {
        std::vector<std::string> items;
        int current_idx = 0;

        try {
            if (nb::hasattr(search_data, search_prop.c_str())) {
                nb::object collection = search_data.attr(search_prop.c_str());
                if (nb::hasattr(collection, "__iter__")) {
                    size_t idx = 0;
                    for (auto item : collection) {
                        if (nb::hasattr(item, "name"))
                            items.push_back(nb::cast<std::string>(item.attr("name")));
                        else
                            items.push_back("Item " + std::to_string(idx));
                        ++idx;
                    }
                }
            }
        } catch (...) {
            LOG_WARN("prop_search: failed to enumerate items for '{}'", prop_id);
        }

        std::string label = text.empty() ? prop_id : text;
        return combo(label, current_idx, items);
    }

    std::tuple<int, int> RmlImModeLayout::template_list(const std::string& /*list_type_id*/,
                                                        const std::string& list_id,
                                                        nb::object data, const std::string& prop_id,
                                                        nb::object active_data, const std::string& active_prop,
                                                        int rows) {
        std::vector<std::string> items;
        int active_idx = 0;

        try {
            if (nb::hasattr(data, prop_id.c_str())) {
                nb::object collection = data.attr(prop_id.c_str());
                if (nb::hasattr(collection, "__iter__")) {
                    size_t idx = 0;
                    for (auto item : collection) {
                        if (nb::hasattr(item, "name"))
                            items.push_back(nb::cast<std::string>(item.attr("name")));
                        else
                            items.push_back("Item " + std::to_string(idx));
                        ++idx;
                    }
                }
            }
            if (nb::hasattr(active_data, active_prop.c_str()))
                active_idx = nb::cast<int>(active_data.attr(active_prop.c_str()));
        } catch (...) {
            LOG_WARN("template_list: failed to get active index for '{}'", active_prop);
        }

        auto [changed, new_idx] = listbox(list_id, active_idx, items, rows);
        if (changed) {
            try {
                nb::setattr(active_data, active_prop.c_str(), nb::cast(new_idx));
            } catch (...) {
                LOG_WARN("template_list: failed to write selection for '{}'", active_prop);
            }
        }

        return {new_idx, static_cast<int>(items.size())};
    }

    void RmlImModeLayout::menu(const std::string& /*menu_id*/, const std::string& /*text*/,
                               const std::string& /*icon*/) {
        warn_unsupported("menu");
    }
    void RmlImModeLayout::popover(const std::string& /*panel_id*/, const std::string& /*text*/,
                                  const std::string& /*icon*/) {
        warn_unsupported("popover");
    }

    // ── RmlSubLayout ─────────────────────────────────────────

    RmlSubLayout::RmlSubLayout(RmlImModeLayout* parent, RmlLayoutDirection dir)
        : parent_(parent),
          direction_(dir) {
        assert(parent_);
    }

    RmlSubLayout& RmlSubLayout::enter() {
        assert(!entered_);
        entered_ = true;
        parent_->finish_current_line();
        assert(!parent_->containers_.empty());

        const char* cls = direction_ == RmlLayoutDirection::Row ? "im-row" : "im-column";
        auto& slot = parent_->ensure_slot(
            SlotType::Line,
            parent_->build_slot_id(direction_ == RmlLayoutDirection::Row ? "row" : "col"));

        if (!slot.element) {
            auto el = parent_->doc_->CreateElement("div");
            el->SetClass(cls, true);
            slot.element = parent_->containers_.back().parent->AppendChild(std::move(el));
        }

        while (slot.element->HasChildNodes())
            parent_->removed_elements_.push_back(slot.element->RemoveChild(slot.element->GetFirstChild()));

        parent_->containers_.push_back({slot.element, {}, 0});
        return *this;
    }

    void RmlSubLayout::exit() {
        if (!entered_)
            return;
        entered_ = false;
        parent_->finish_current_line();
        if (parent_->containers_.size() > 1) {
            parent_->prune_excess_slots(parent_->containers_.back());
            parent_->containers_.pop_back();
        }
    }

    RmlSubLayout RmlSubLayout::row() {
        return RmlSubLayout(parent_, RmlLayoutDirection::Row);
    }

    RmlSubLayout RmlSubLayout::column() {
        return RmlSubLayout(parent_, RmlLayoutDirection::Column);
    }

    RmlSubLayout RmlSubLayout::split(float /*factor*/) {
        return RmlSubLayout(parent_, RmlLayoutDirection::Row);
    }

    RmlSubLayout RmlSubLayout::box() {
        return RmlSubLayout(parent_, RmlLayoutDirection::Column);
    }

    RmlSubLayout RmlSubLayout::grid_flow(int /*columns*/, bool /*even_columns*/, bool /*even_rows*/) {
        return RmlSubLayout(parent_, RmlLayoutDirection::Row);
    }

} // namespace lfs::python

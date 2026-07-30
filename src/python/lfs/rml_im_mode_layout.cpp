/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "rml_im_mode_layout.hpp"
#include "core/event_bridge/localization_manager.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "py_error.hpp"
#include "py_ui.hpp"
#include "python/python_runtime.hpp"
#include "visualizer/gui/rmlui/elements/loss_graph_element.hpp"
#include "visualizer/gui/rmlui/rml_tooltip.hpp"
#include "visualizer/gui/utils/native_file_dialog.hpp"
#include "visualizer/operator/operator_registry.hpp"

#include <RmlUi/Core/Context.h>
#include <RmlUi/Core/Element.h>
#include <RmlUi/Core/ElementDocument.h>
#include <RmlUi/Core/ElementText.h>
#include <RmlUi/Core/Factory.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <deque>
#include <format>
#include <functional>
#include <limits>
#include <mutex>
#include <string_view>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {
    std::string strip_legacy_id(const std::string& label) {
        auto pos = label.find("##");
        if (pos == std::string::npos)
            return label;
        return label.substr(0, pos);
    }

    std::string hidden_legacy_id(const std::string& label) {
        const auto pos = label.find("##");
        if (pos == std::string::npos || pos + 2 >= label.size())
            return {};
        return label.substr(pos + 2);
    }

    bool is_checkbox_or_radio(const Rml::Element* element) {
        if (!element || element->GetTagName() != "input")
            return false;
        const auto input_type = element->GetAttribute<Rml::String>("type", "");
        return input_type == "checkbox" || input_type == "radio";
    }

    bool get_checked_state(const Rml::Element* element) {
        if (!element || !is_checkbox_or_radio(element))
            return false;
        // RmlUi tracks live form state via pseudo-classes; the "checked" attribute
        // can be stale after user interaction.
        return element->IsPseudoClassSet("checked");
    }

    float compute_slider_step(float min, float max) {
        const float range = std::fabs(max - min);
        if (!std::isfinite(range) || range <= 0.0f)
            return 0.01f;
        // Use a small explicit step; "any" is effectively quantized on some RmlUi builds.
        return std::max(range / 1000.0f, 0.0001f);
    }

    Rml::String step_to_string(float step) {
        return Rml::String(std::format("{:.6g}", step));
    }

    Rml::String float_attribute_string(const float value) {
        return Rml::String(std::to_string(value));
    }

    Rml::String int_attribute_string(const int value) {
        return Rml::String(std::to_string(value));
    }

    void update_values_deque(nb::object values, std::deque<float>& out) {
        out.clear();
        try {
            for (auto item : values) {
                const float value = nb::cast<float>(item);
                if (std::isfinite(value))
                    out.push_back(value);
            }
        } catch (const std::exception& e) {
            LOG_WARN("RmlImModeLayout: plot_lines could not read values: {}", e.what());
        }
    }

    struct ComboItemsFingerprint {
        size_t primary = 0;
        size_t secondary = 0;
    };

    ComboItemsFingerprint hash_combo_items(const std::vector<std::string>& items) {
        ComboItemsFingerprint result{
            .primary = items.size(),
            .secondary = items.size() ^ 0xd6e8feb86659fd93ULL,
        };
        const std::hash<std::string_view> hash_string;
        for (const auto& item : items) {
            const size_t item_hash = hash_string(item);
            result.primary ^=
                item_hash + 0x9e3779b9 + (result.primary << 6) + (result.primary >> 2);
            result.secondary ^= item.size();
            result.secondary *= 1099511628211ULL;
            for (const unsigned char byte : std::string_view(item)) {
                result.secondary ^= byte;
                result.secondary *= 1099511628211ULL;
            }
            result.secondary ^= 0xff;
            result.secondary *= 1099511628211ULL;
        }
        return result;
    }

    using PathDialogCallback = std::function<std::filesystem::path(
        bool, const std::filesystem::path&, const std::optional<std::string>&)>;

    PathDialogCallback default_path_dialog_callback() {
        return [](const bool folder_mode, const std::filesystem::path& default_path,
                  const std::optional<std::string>& title) {
            if (folder_mode)
                return title ? lfs::vis::gui::PickFolderDialog(default_path, *title)
                             : lfs::vis::gui::PickFolderDialog(default_path);
            return title ? lfs::vis::gui::OpenFileDialog(default_path, *title)
                         : lfs::vis::gui::OpenFileDialog(default_path);
        };
    }

    PathDialogCallback& path_dialog_callback() {
        static PathDialogCallback callback = default_path_dialog_callback();
        return callback;
    }

    bool set_attribute_if_changed(Rml::Element* element, const char* name,
                                  const Rml::String& value) {
        assert(element);
        if (element->HasAttribute(name) &&
            element->GetAttribute<Rml::String>(name, "") == value)
            return false;
        element->SetAttribute(name, value);
        return true;
    }

    template <typename T>
    bool numeric_attribute_equals(const Rml::Element* element, const char* name,
                                  const T value) {
        assert(element);
        if (!element->HasAttribute(name))
            return false;

        if constexpr (std::is_integral_v<T>) {
            return element->GetAttribute<int>(name, 0) == value;
        } else {
            const float current = element->GetAttribute<float>(
                name, std::numeric_limits<float>::quiet_NaN());
            const float expected = static_cast<float>(value);
            if (current == expected)
                return true;
            if (!std::isfinite(current) || !std::isfinite(expected))
                return false;
            const float scale =
                std::max({1.0f, std::fabs(current), std::fabs(expected)});
            return std::fabs(current - expected) <=
                   std::numeric_limits<float>::epsilon() * 8.0f * scale;
        }
    }

    template <typename T, typename Formatter>
    bool set_numeric_attribute_if_changed(Rml::Element* element, const char* name,
                                          const T value, Formatter&& formatter) {
        if (numeric_attribute_equals(element, name, value))
            return false;
        element->SetAttribute(
            name, std::invoke(std::forward<Formatter>(formatter), value));
        return true;
    }

    template <typename T, typename Formatter>
    bool set_slot_numeric_attribute_if_changed(
        lfs::python::Slot& slot, const size_t index, Rml::Element* element,
        const char* name, const T value, Formatter&& formatter) {
        assert(index < slot.numeric_content.size());
        const double numeric_value = static_cast<double>(value);
        if (slot.numeric_content[index] &&
            *slot.numeric_content[index] == numeric_value)
            return false;
        const bool changed = set_numeric_attribute_if_changed(
            element, name, value, std::forward<Formatter>(formatter));
        slot.numeric_content[index] = numeric_value;
        return changed;
    }

    template <typename T, typename Formatter>
    bool set_slot_numeric_property_if_changed(
        lfs::python::Slot& slot, const size_t index, Rml::Element* element,
        const char* name, const T value, Formatter&& formatter) {
        assert(element);
        assert(index < slot.numeric_content.size());
        const double numeric_value = static_cast<double>(value);
        if (slot.numeric_content[index] &&
            *slot.numeric_content[index] == numeric_value)
            return false;
        element->SetProperty(
            name, std::invoke(std::forward<Formatter>(formatter), value));
        slot.numeric_content[index] = numeric_value;
        return true;
    }

    bool set_slot_property_if_changed(lfs::python::Slot& slot,
                                      const size_t index,
                                      Rml::Element* element,
                                      const char* name,
                                      const Rml::String& value) {
        assert(element);
        assert(index < slot.property_content.size());
        if (slot.property_content[index] &&
            *slot.property_content[index] == value)
            return false;
        element->SetProperty(name, value);
        slot.property_content[index] = value;
        return true;
    }

    bool set_slot_class_if_changed(lfs::python::Slot& slot,
                                   const size_t index,
                                   Rml::Element* element,
                                   const char* name,
                                   const bool enabled) {
        assert(element);
        assert(index < slot.class_content.size());
        if (slot.class_content[index] &&
            *slot.class_content[index] == enabled)
            return false;
        element->SetClass(name, enabled);
        slot.class_content[index] = enabled;
        return true;
    }

    bool is_focused(const Rml::Element* element) {
        return element && element->GetContext() &&
               element->GetContext()->GetFocusElement() == element;
    }

    Rml::Element* find_direct_input(Rml::Element* parent) {
        if (!parent)
            return nullptr;
        for (int i = 0; i < parent->GetNumChildren(); ++i) {
            auto* child = parent->GetChild(i);
            if (child && child->GetTagName() == "input")
                return child;
        }
        return nullptr;
    }

    bool is_descendant_of(const Rml::Element* element, const Rml::Element* ancestor) {
        for (auto* current = element; current; current = current->GetParentNode()) {
            if (current == ancestor)
                return true;
        }
        return false;
    }

    void warn_retained_custom_element_once(const char* method, const char* element) {
        static std::mutex mutex;
        static std::unordered_set<std::string> warned_methods;
        std::lock_guard lock(mutex);
        if (warned_methods.emplace(method).second) {
            LOG_WARN("RmlImModeLayout::{} is unavailable in layout APIs; use the retained RmlUi {} element",
                     method, element);
        }
    }
} // namespace

namespace lfs::python {

    void SlotEventListener::ProcessEvent(Rml::Event& event) {
        const auto state = state_.lock();
        if (!state || !state->active)
            return;
        const auto type = event.GetId();
        if (type == Rml::EventId::Click) {
            auto* el = event.GetCurrentElement();
            if (el && el->GetTagName() == "input") {
                const auto input_type = el->GetAttribute<Rml::String>("type", "");
                if (input_type == "checkbox") {
                    state->changed = true;
                    state->bool_value = !state->bool_value;
                    return;
                }
            }
            state->clicked = true;
        } else if (type == Rml::EventId::Change) {
            state->changed = true;
            auto* el = event.GetCurrentElement();
            if (!el)
                return;
            const auto tag = el->GetTagName();
            if (tag == "input") {
                const auto input_type = el->GetAttribute<Rml::String>("type", "");
                if (input_type == "checkbox") {
                    state->bool_value = get_checked_state(el);
                } else if (input_type == "range") {
                    // RmlUi 6.2 dispatches range changes before updating the value attribute.
                    const float fallback = el->GetAttribute<float>("value", 0.0f);
                    state->float_value = event.GetParameter<float>("value", fallback);
                } else if (input_type == "text") {
                    state->string_value = el->GetAttribute<Rml::String>("value", "");
                }
            } else if (tag == "select") {
                state->int_value = static_cast<int>(el->GetAttribute<float>("value", 0.0f));
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
            child_slots_.clear();
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
        child_key_stack_.clear();
        force_next_open_ = false;
        table_.reset();
        item_width_stack_.clear();
        last_element_ = nullptr;
        last_clicked_ = false;
        auto* tooltip = doc->GetElementById("im-tooltip");
        if (tooltip != tooltip_el_)
            rendered_tooltip_text_.clear();
        tooltip_el_ = tooltip;
        tooltip_shown_ = false;
        tooltip_candidate_seen_ = false;
        popup_backdrop_ = doc->GetElementById("im-popup-backdrop");
        popup_dialog_ = doc->GetElementById("im-popup-dialog");
        active_popup_id_.clear();
    }

    void RmlImModeLayout::end_frame() {
        if (table_)
            end_table();
        finish_current_line();

        if (!tooltip_shown_ && tooltip_el_)
            tooltip_el_->SetClass("visible", false);
        if (!tooltip_candidate_seen_) {
            tooltip_hover_el_ = nullptr;
            tooltip_text_.clear();
            tooltip_hover_started_at_ = {};
        }

        for (auto& level : containers_)
            prune_excess_slots(level);

        std::erase_if(child_slots_, [this](const auto& pair) {
            return !pair.second.container ||
                   !is_descendant_of(pair.second.container, root_);
        });

        containers_.resize(1);
        current_line_ = nullptr;
        doc_ = nullptr;
        root_ = nullptr;
    }

    void RmlImModeLayout::release_elements() {
        const auto neutralize = [](auto& slots) {
            for (auto& slot : slots) {
                if (slot.events)
                    slot.events->active = false;
            }
        };
        for (auto& level : containers_)
            neutralize(level.slots);
        for (auto& [key, cache] : child_slots_) {
            (void)key;
            neutralize(cache.slots);
        }

        removed_elements_.clear();
        child_slots_.clear();
        child_key_stack_.clear();
        containers_.clear();
        current_line_ = nullptr;
        doc_ = nullptr;
        root_ = nullptr;
        tooltip_el_ = nullptr;
        rendered_tooltip_text_.clear();
    }

    void RmlImModeLayout::prune_excess_slots(ContainerLevel& level) {
        while (static_cast<int>(level.slots.size()) > level.cursor) {
            auto& slot = level.slots.back();
            if (slot.element && slot.element->GetParentNode())
                removed_elements_.push_back(slot.element->GetParentNode()->RemoveChild(slot.element));
            level.slots.pop_back();
        }
    }

    void RmlImModeLayout::push_persistent_container(const std::string& key, Rml::Element* container) {
        auto it = child_slots_.find(key);
        if (it != child_slots_.end()) {
            containers_.push_back({container, std::move(it->second.slots), 0});
            child_slots_.erase(it);
        } else {
            containers_.push_back({container, {}, 0});
        }
        child_key_stack_.push_back(key);
    }

    void RmlImModeLayout::pop_persistent_container() {
        if (containers_.size() <= 1)
            return;

        finish_current_line();
        prune_excess_slots(containers_.back());

        if (!child_key_stack_.empty()) {
            auto* container = containers_.back().parent;
            child_slots_[child_key_stack_.back()] = {container, std::move(containers_.back().slots)};
            child_key_stack_.pop_back();
        }
        containers_.pop_back();
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
        return hidden_legacy_id(label);
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
        } catch (...) {
            // LFS-CENSUS-OK(empty-catch): optional color parse for styling; a malformed
            // tuple yields no color override rather than aborting the draw.
        }
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
            slot = Slot{type, key};
        } else {
            level.slots.push_back(Slot{type, key});
        }

        level.cursor++;
        return level.slots[idx];
    }

    void RmlImModeLayout::set_slot_text(Slot& slot, size_t index, Rml::Element* element,
                                        const std::string& content) {
        assert(index < slot.content.size());
        assert(element);
        if (slot.content[index] && *slot.content[index] == content) {
            slot.numeric_content[index].reset();
            return;
        }

        if (element->GetNumChildren() == 0) {
            assert(doc_);
            element->AppendChild(doc_->CreateTextNode(Rml::String(content)));
        } else if (element->GetNumChildren() == 1) {
            if (auto* text = dynamic_cast<Rml::ElementText*>(element->GetFirstChild()))
                text->SetText(Rml::String(content));
            else
                element->SetInnerRML(Rml::String(content));
        } else {
            element->SetInnerRML(Rml::String(content));
        }
        slot.content[index] = content;
        slot.numeric_content[index].reset();
    }

    void RmlImModeLayout::set_slot_float_text(Slot& slot, const size_t index,
                                              Rml::Element* element, const float value) {
        assert(index < slot.numeric_content.size());
        if (slot.numeric_content[index] && *slot.numeric_content[index] == value)
            return;
        set_slot_text(slot, index, element, std::format("{:.2f}", value));
        slot.numeric_content[index] = value;
    }

    void RmlImModeLayout::set_slot_int_text(Slot& slot, const size_t index,
                                            Rml::Element* element, const int value) {
        assert(index < slot.numeric_content.size());
        if (slot.numeric_content[index] && *slot.numeric_content[index] == value)
            return;
        set_slot_text(slot, index, element, std::to_string(value));
        slot.numeric_content[index] = value;
    }

    void RmlImModeLayout::set_path_dialog_callback_for_testing(
        PathDialogCallback callback) {
        assert(callback);
        path_dialog_callback() = std::move(callback);
    }

    void RmlImModeLayout::reset_path_dialog_callback_for_testing() {
        path_dialog_callback() = default_path_dialog_callback();
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
        const auto display = strip_legacy_id(text);

        if (!slot.element) {
            auto el = doc_->CreateElement("p");
            el->SetClass("im-label", true);
            set_slot_text(slot, 0, el.get(), display);
            slot.element = line->AppendChild(std::move(el));
        } else {
            if (slot.element->GetParentNode() != line)
                line->AppendChild(slot.element->GetParentNode()->RemoveChild(slot.element));
            set_slot_text(slot, 0, slot.element, display);
        }
        last_element_ = slot.element;
        last_clicked_ = false;
    }

    void RmlImModeLayout::label_centered(const std::string& text) {
        if (!doc_)
            return;
        auto* line = ensure_line_container();
        auto& slot = ensure_slot(SlotType::Label, build_slot_id("label_center", &text));
        const auto display = strip_legacy_id(text);

        if (!slot.element) {
            auto el = doc_->CreateElement("p");
            el->SetClass("im-label", true);
            el->SetClass("im-label--centered", true);
            set_slot_text(slot, 0, el.get(), display);
            slot.element = line->AppendChild(std::move(el));
        } else {
            if (slot.element->GetParentNode() != line)
                line->AppendChild(slot.element->GetParentNode()->RemoveChild(slot.element));
            set_slot_text(slot, 0, slot.element, display);
        }
        last_element_ = slot.element;
        last_clicked_ = false;
    }

    void RmlImModeLayout::heading(const std::string& text) {
        if (!doc_)
            return;
        auto* line = ensure_line_container();
        auto& slot = ensure_slot(SlotType::Heading, build_slot_id("heading", &text));
        const auto display = strip_legacy_id(text);

        if (!slot.element) {
            auto el = doc_->CreateElement("div");
            el->SetClass("panel-title", true);
            set_slot_text(slot, 0, el.get(), display);
            slot.element = line->AppendChild(std::move(el));
        } else {
            if (slot.element->GetParentNode() != line)
                line->AppendChild(slot.element->GetParentNode()->RemoveChild(slot.element));
            set_slot_text(slot, 0, slot.element, display);
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
        const auto display = strip_legacy_id(text);

        if (!slot.element) {
            auto el = doc_->CreateElement("p");
            el->SetClass("im-label", true);
            if (!css_color.empty())
                set_slot_property_if_changed(
                    slot, 0, el.get(), "color", Rml::String(css_color));
            set_slot_text(slot, 0, el.get(), display);
            slot.element = line->AppendChild(std::move(el));
        } else {
            if (slot.element->GetParentNode() != line)
                line->AppendChild(slot.element->GetParentNode()->RemoveChild(slot.element));
            if (!css_color.empty())
                set_slot_property_if_changed(
                    slot, 0, slot.element, "color", Rml::String(css_color));
            set_slot_text(slot, 0, slot.element, display);
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
        const auto display = strip_legacy_id(text);

        if (!slot.element) {
            auto el = doc_->CreateElement("p");
            el->SetClass("im-label", true);
            el->SetClass("im-label--centered", true);
            if (!css_color.empty())
                set_slot_property_if_changed(
                    slot, 0, el.get(), "color", Rml::String(css_color));
            set_slot_text(slot, 0, el.get(), display);
            slot.element = line->AppendChild(std::move(el));
        } else {
            if (slot.element->GetParentNode() != line)
                line->AppendChild(slot.element->GetParentNode()->RemoveChild(slot.element));
            if (!css_color.empty())
                set_slot_property_if_changed(
                    slot, 0, slot.element, "color", Rml::String(css_color));
            set_slot_text(slot, 0, slot.element, display);
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
        const auto display = strip_legacy_id(text);

        if (!slot.element) {
            auto el = doc_->CreateElement("p");
            el->SetClass("im-text-wrapped", true);
            set_slot_text(slot, 0, el.get(), display);
            slot.element = line->AppendChild(std::move(el));
        } else {
            if (slot.element->GetParentNode() != line)
                line->AppendChild(slot.element->GetParentNode()->RemoveChild(slot.element));
            set_slot_text(slot, 0, slot.element, display);
        }
        last_element_ = slot.element;
        last_clicked_ = false;
    }

    void RmlImModeLayout::text_disabled(const std::string& text) {
        if (!doc_)
            return;
        auto* line = ensure_line_container();
        auto& slot = ensure_slot(SlotType::TextDisabled, build_slot_id("text_disabled", &text));
        const auto display = strip_legacy_id(text);

        if (!slot.element) {
            auto el = doc_->CreateElement("p");
            el->SetClass("text-disabled", true);
            set_slot_text(slot, 0, el.get(), display);
            slot.element = line->AppendChild(std::move(el));
        } else {
            if (slot.element->GetParentNode() != line)
                line->AppendChild(slot.element->GetParentNode()->RemoveChild(slot.element));
            set_slot_text(slot, 0, slot.element, display);
        }
        last_element_ = slot.element;
        last_clicked_ = false;
    }

    void RmlImModeLayout::bullet_text(const std::string& text) {
        if (!doc_)
            return;
        auto* line = ensure_line_container();
        auto& slot = ensure_slot(SlotType::BulletText, build_slot_id("bullet_text", &text));
        const auto display = strip_legacy_id(text);
        const auto bullet = std::format("\xe2\x80\xa2 {}", display);

        if (!slot.element) {
            auto el = doc_->CreateElement("p");
            el->SetClass("im-bullet", true);
            set_slot_text(slot, 0, el.get(), bullet);
            slot.element = line->AppendChild(std::move(el));
        } else {
            if (slot.element->GetParentNode() != line)
                line->AppendChild(slot.element->GetParentNode()->RemoveChild(slot.element));
            set_slot_text(slot, 0, slot.element, bullet);
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
        const auto display = strip_legacy_id(label);

        if (!slot.element) {
            auto el = doc_->CreateElement("button");
            el->SetClass("btn", true);
            set_slot_text(slot, 0, el.get(), display);
            auto [w, h] = size;
            if (w < 0)
                el->SetClass("btn--full", true);
            else if (w > 0)
                el->SetProperty("width", Rml::String(std::to_string(static_cast<int>(w)) + "dp"));

            el->AddEventListener(Rml::EventId::Click, new SlotEventListener(slot.events));

            slot.element = line->AppendChild(std::move(el));
        } else {
            if (slot.element->GetParentNode() != line)
                line->AppendChild(slot.element->GetParentNode()->RemoveChild(slot.element));
            set_slot_text(slot, 0, slot.element, display);
        }

        bool clicked = slot.events->clicked;
        slot.events->clicked = false;
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
        const auto display = strip_legacy_id(label);

        if (!slot.element) {
            auto el = doc_->CreateElement("button");
            el->SetClass("btn", true);
            set_slot_text(slot, 0, el.get(), display);

            el->AddEventListener(Rml::EventId::Click, new SlotEventListener(slot.events));

            slot.element = line->AppendChild(std::move(el));
        } else {
            if (slot.element->GetParentNode() != line)
                line->AppendChild(slot.element->GetParentNode()->RemoveChild(slot.element));
            set_slot_text(slot, 0, slot.element, display);
        }

        bool clicked = slot.events->clicked;
        slot.events->clicked = false;
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
        const auto display = strip_legacy_id(label);

        if (!slot.element) {
            auto el = doc_->CreateElement("button");
            el->SetClass("btn", true);
            el->SetClass("btn--" + style, true);
            set_slot_text(slot, 0, el.get(), display);
            auto [w, h] = size;
            if (w < 0)
                el->SetClass("btn--full", true);
            else if (w > 0)
                el->SetProperty("width", Rml::String(std::to_string(static_cast<int>(w)) + "dp"));

            el->AddEventListener(Rml::EventId::Click, new SlotEventListener(slot.events));

            slot.element = line->AppendChild(std::move(el));
        } else {
            if (slot.element->GetParentNode() != line)
                line->AppendChild(slot.element->GetParentNode()->RemoveChild(slot.element));
            set_slot_text(slot, 0, slot.element, display);
        }

        bool clicked = slot.events->clicked;
        slot.events->clicked = false;
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

            auto text_span = doc_->CreateElement("span");
            text_span->SetClass("prop-label", true);
            text_span->SetInnerRML(Rml::String(strip_legacy_id(label)));

            auto input = doc_->CreateElement("input");
            input->SetAttribute("type", "checkbox");
            if (value)
                input->SetAttribute("checked", "");

            slot.events->bool_value = value;
            input->AddEventListener(Rml::EventId::Click, new SlotEventListener(slot.events));

            wrapper->AppendChild(std::move(text_span));
            wrapper->AppendChild(std::move(input));
            slot.element = line->AppendChild(std::move(wrapper));
        } else {
            if (slot.element->GetParentNode() != line)
                line->AppendChild(slot.element->GetParentNode()->RemoveChild(slot.element));
            auto* input = slot.element->GetChild(1);
            if (!slot.events->changed) {
                slot.events->bool_value = value;
                if (input) {
                    if (value && !input->HasAttribute("checked"))
                        input->SetAttribute("checked", "");
                    else if (!value && input->HasAttribute("checked"))
                        input->RemoveAttribute("checked");
                }
            }
        }

        last_element_ = slot.element;
        last_clicked_ = false;
        if (slot.events->changed) {
            slot.events->changed = false;
            bool new_value = slot.events->bool_value;
            slot.events->bool_value = new_value;
            return {true, new_value};
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
            set_slot_class_if_changed(
                slot, 0, wrapper.get(), "selected", selected);

            auto dot = doc_->CreateElement("span");
            dot->SetClass("im-radio-dot", true);
            set_slot_text(slot, 0, dot.get(), selected ? "\xe2\x97\x89" : "\xe2\x97\x8b");

            auto lbl = doc_->CreateElement("span");
            lbl->SetClass("im-radio-label", true);
            lbl->SetInnerRML(Rml::String(strip_legacy_id(label)));

            wrapper->AddEventListener(Rml::EventId::Click, new SlotEventListener(slot.events));

            wrapper->AppendChild(std::move(dot));
            wrapper->AppendChild(std::move(lbl));
            slot.element = line->AppendChild(std::move(wrapper));
        } else {
            if (slot.element->GetParentNode() != line)
                line->AppendChild(slot.element->GetParentNode()->RemoveChild(slot.element));
            set_slot_class_if_changed(
                slot, 0, slot.element, "selected", selected);
            auto* dot = slot.element->GetChild(0);
            if (dot)
                set_slot_text(slot, 0, dot, selected ? "\xe2\x97\x89" : "\xe2\x97\x8b");
        }

        bool clicked = slot.events->clicked;
        last_element_ = slot.element;
        last_clicked_ = clicked;
        if (clicked) {
            slot.events->clicked = false;
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
        const auto min_text = Rml::String(std::to_string(min));
        const auto max_text = Rml::String(std::to_string(max));
        const auto step_text = step_to_string(compute_slider_step(min, max));
        const auto pushed_value = Rml::String(std::to_string(value));

        if (!slot.element) {
            auto wrapper = doc_->CreateElement("div");
            wrapper->SetClass("setting-row", true);

            auto lbl = doc_->CreateElement("span");
            lbl->SetClass("prop-label", true);
            lbl->SetInnerRML(Rml::String(strip_legacy_id(label)));

            auto input = doc_->CreateElement("input");
            input->SetAttribute("type", "range");
            input->SetAttribute("min", min_text);
            input->SetAttribute("max", max_text);
            input->SetAttribute("step", step_text);
            input->SetAttribute("value", pushed_value);
            input->SetClass("setting-slider", true);

            slot.events->float_value = value;
            input->AddEventListener(Rml::EventId::Change, new SlotEventListener(slot.events));

            auto val_text = doc_->CreateElement("span");
            val_text->SetClass("slider-value", true);
            set_slot_float_text(slot, 0, val_text.get(), value);

            wrapper->AppendChild(std::move(lbl));
            wrapper->AppendChild(std::move(input));
            wrapper->AppendChild(std::move(val_text));
            slot.element = line->AppendChild(std::move(wrapper));
            apply_item_width(slot.element);
        } else {
            if (slot.element->GetParentNode() != line)
                line->AppendChild(slot.element->GetParentNode()->RemoveChild(slot.element));
            auto* input = slot.element->GetChild(1);
            if (input) {
                set_attribute_if_changed(input, "min", min_text);
                set_attribute_if_changed(input, "max", max_text);
                set_attribute_if_changed(input, "step", step_text);
            }
            if (input && !slot.events->changed && !input->IsPseudoClassSet("active"))
                set_attribute_if_changed(input, "value", pushed_value);
            auto* val_text = slot.element->GetChild(2);
            if (val_text) {
                const float display_val =
                    slot.events->changed ? slot.events->float_value : value;
                set_slot_float_text(slot, 0, val_text, display_val);
            }
        }

        last_element_ = slot.element;
        last_clicked_ = false;
        if (slot.events->changed) {
            slot.events->changed = false;
            return {true, slot.events->float_value};
        }
        return {false, value};
    }

    std::tuple<bool, int> RmlImModeLayout::slider_int(const std::string& label,
                                                      int value, int min, int max) {
        if (!doc_)
            return {false, value};
        auto* line = ensure_line_container();
        auto& slot = ensure_slot(SlotType::SliderInt, build_slot_id("slider_int", &label));
        const auto min_text = Rml::String(std::to_string(min));
        const auto max_text = Rml::String(std::to_string(max));
        const auto value_text = Rml::String(std::to_string(value));

        if (!slot.element) {
            auto wrapper = doc_->CreateElement("div");
            wrapper->SetClass("setting-row", true);

            auto lbl = doc_->CreateElement("span");
            lbl->SetClass("prop-label", true);
            lbl->SetInnerRML(Rml::String(strip_legacy_id(label)));

            auto input = doc_->CreateElement("input");
            input->SetAttribute("type", "range");
            input->SetAttribute("min", min_text);
            input->SetAttribute("max", max_text);
            input->SetAttribute("step", "1");
            input->SetAttribute("value", value_text);
            input->SetClass("setting-slider", true);

            slot.events->float_value = static_cast<float>(value);
            input->AddEventListener(Rml::EventId::Change, new SlotEventListener(slot.events));

            auto val_text = doc_->CreateElement("span");
            val_text->SetClass("slider-value", true);
            set_slot_int_text(slot, 0, val_text.get(), value);

            wrapper->AppendChild(std::move(lbl));
            wrapper->AppendChild(std::move(input));
            wrapper->AppendChild(std::move(val_text));
            slot.element = line->AppendChild(std::move(wrapper));
            apply_item_width(slot.element);
        } else {
            if (slot.element->GetParentNode() != line)
                line->AppendChild(slot.element->GetParentNode()->RemoveChild(slot.element));
            auto* input = slot.element->GetChild(1);
            if (input) {
                set_attribute_if_changed(input, "min", min_text);
                set_attribute_if_changed(input, "max", max_text);
                set_attribute_if_changed(input, "step", "1");
            }
            if (input && !slot.events->changed && !input->IsPseudoClassSet("active"))
                set_attribute_if_changed(input, "value", value_text);
            auto* val_text = slot.element->GetChild(2);
            if (val_text) {
                const int display_val =
                    slot.events->changed ? static_cast<int>(slot.events->float_value) : value;
                set_slot_int_text(slot, 0, val_text, display_val);
            }
        }

        last_element_ = slot.element;
        last_clicked_ = false;
        if (slot.events->changed) {
            slot.events->changed = false;
            return {true, static_cast<int>(std::round(slot.events->float_value))};
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
                                                        float speed, float min, float max) {
        if (!doc_)
            return {false, value};
        auto* line = ensure_line_container();
        auto& slot = ensure_slot(SlotType::DragFloat, build_slot_id("drag_float", &label));
        const auto min_text = Rml::String(std::to_string(min));
        const auto max_text = Rml::String(std::to_string(max));
        const float step = (speed > 0.0f) ? std::fabs(speed) : compute_slider_step(min, max);
        const auto step_text = step_to_string(step);
        const auto pushed_value = Rml::String(std::to_string(value));

        if (!slot.element) {
            auto wrapper = doc_->CreateElement("div");
            wrapper->SetClass("setting-row", true);

            auto lbl = doc_->CreateElement("span");
            lbl->SetClass("prop-label", true);
            lbl->SetInnerRML(Rml::String(strip_legacy_id(label)));

            auto input = doc_->CreateElement("input");
            input->SetAttribute("type", "range");
            input->SetAttribute("min", min_text);
            input->SetAttribute("max", max_text);
            input->SetAttribute("step", step_text);
            input->SetAttribute("value", pushed_value);
            input->SetClass("setting-slider", true);

            slot.events->float_value = value;
            input->AddEventListener(Rml::EventId::Change, new SlotEventListener(slot.events));

            auto val_text = doc_->CreateElement("span");
            val_text->SetClass("slider-value", true);
            set_slot_float_text(slot, 0, val_text.get(), value);

            wrapper->AppendChild(std::move(lbl));
            wrapper->AppendChild(std::move(input));
            wrapper->AppendChild(std::move(val_text));
            slot.element = line->AppendChild(std::move(wrapper));
            apply_item_width(slot.element);
        } else {
            if (slot.element->GetParentNode() != line)
                line->AppendChild(slot.element->GetParentNode()->RemoveChild(slot.element));
            auto* input = slot.element->GetChild(1);
            if (input) {
                set_attribute_if_changed(input, "min", min_text);
                set_attribute_if_changed(input, "max", max_text);
                set_attribute_if_changed(input, "step", step_text);
            }
            if (input && !slot.events->changed && !input->IsPseudoClassSet("active"))
                set_attribute_if_changed(input, "value", pushed_value);
            auto* val_text = slot.element->GetChild(2);
            if (val_text) {
                const float display_val =
                    slot.events->changed ? slot.events->float_value : value;
                set_slot_float_text(slot, 0, val_text, display_val);
            }
        }

        last_element_ = slot.element;
        last_clicked_ = false;
        if (slot.events->changed) {
            slot.events->changed = false;
            return {true, slot.events->float_value};
        }
        return {false, value};
    }

    std::tuple<bool, int> RmlImModeLayout::drag_int(const std::string& label, int value,
                                                    float speed, int min, int max) {
        if (!doc_)
            return {false, value};
        auto* line = ensure_line_container();
        auto& slot = ensure_slot(SlotType::DragInt, build_slot_id("drag_int", &label));
        const auto min_text = Rml::String(std::to_string(min));
        const auto max_text = Rml::String(std::to_string(max));
        const int step = std::max(1, static_cast<int>(std::lround(std::fabs(speed))));
        const auto step_text = Rml::String(std::to_string(step));
        const auto pushed_value = Rml::String(std::to_string(value));

        if (!slot.element) {
            auto wrapper = doc_->CreateElement("div");
            wrapper->SetClass("setting-row", true);

            auto lbl = doc_->CreateElement("span");
            lbl->SetClass("prop-label", true);
            lbl->SetInnerRML(Rml::String(strip_legacy_id(label)));

            auto input = doc_->CreateElement("input");
            input->SetAttribute("type", "range");
            input->SetAttribute("min", min_text);
            input->SetAttribute("max", max_text);
            input->SetAttribute("step", step_text);
            input->SetAttribute("value", pushed_value);
            input->SetClass("setting-slider", true);

            slot.events->float_value = static_cast<float>(value);
            input->AddEventListener(Rml::EventId::Change, new SlotEventListener(slot.events));

            auto val_text = doc_->CreateElement("span");
            val_text->SetClass("slider-value", true);
            set_slot_int_text(slot, 0, val_text.get(), value);

            wrapper->AppendChild(std::move(lbl));
            wrapper->AppendChild(std::move(input));
            wrapper->AppendChild(std::move(val_text));
            slot.element = line->AppendChild(std::move(wrapper));
            apply_item_width(slot.element);
        } else {
            if (slot.element->GetParentNode() != line)
                line->AppendChild(slot.element->GetParentNode()->RemoveChild(slot.element));
            auto* input = slot.element->GetChild(1);
            if (input) {
                set_attribute_if_changed(input, "min", min_text);
                set_attribute_if_changed(input, "max", max_text);
                set_attribute_if_changed(input, "step", step_text);
            }
            if (input && !slot.events->changed && !input->IsPseudoClassSet("active"))
                set_attribute_if_changed(input, "value", pushed_value);
            auto* val_text = slot.element->GetChild(2);
            if (val_text) {
                const int display_val =
                    slot.events->changed ? static_cast<int>(slot.events->float_value) : value;
                set_slot_int_text(slot, 0, val_text, display_val);
            }
        }

        last_element_ = slot.element;
        last_clicked_ = false;
        if (slot.events->changed) {
            slot.events->changed = false;
            return {true, static_cast<int>(std::round(slot.events->float_value))};
        }
        return {false, value};
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
            lbl->SetInnerRML(Rml::String(strip_legacy_id(label)));

            auto input = doc_->CreateElement("input");
            input->SetAttribute("type", "text");
            input->SetAttribute("value", Rml::String(value));
            input->SetClass("im-control--fill", true);

            slot.events->string_value = value;
            input->AddEventListener(Rml::EventId::Change, new SlotEventListener(slot.events));

            wrapper->AppendChild(std::move(lbl));
            wrapper->AppendChild(std::move(input));
            slot.element = line->AppendChild(std::move(wrapper));
            apply_item_width(slot.element);
        } else {
            if (slot.element->GetParentNode() != line)
                line->AppendChild(slot.element->GetParentNode()->RemoveChild(slot.element));
            if (auto* input = find_direct_input(slot.element);
                input && !slot.events->changed && !is_focused(input) &&
                set_attribute_if_changed(input, "value", Rml::String(value))) {
                slot.events->string_value = value;
            }
        }

        last_element_ = slot.element;
        last_clicked_ = false;
        if (slot.events->changed) {
            slot.events->changed = false;
            return {true, slot.events->string_value};
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
        const auto value_text = std::format("{:.3f}", value);

        if (!slot.element) {
            auto wrapper = doc_->CreateElement("div");
            wrapper->SetClass("setting-row", true);

            auto display = strip_legacy_id(label);
            if (!display.empty()) {
                auto lbl = doc_->CreateElement("span");
                lbl->SetClass("prop-label", true);
                lbl->SetInnerRML(Rml::String(display));
                wrapper->AppendChild(std::move(lbl));
            }

            auto input = doc_->CreateElement("input");
            input->SetAttribute("type", "text");
            input->SetAttribute("value", Rml::String(value_text));
            input->SetClass("number-input", true);
            input->SetClass("im-control--fill", true);

            slot.events->float_value = value;
            slot.events->string_value = value_text;
            input->AddEventListener(Rml::EventId::Change, new SlotEventListener(slot.events));

            wrapper->AppendChild(std::move(input));
            slot.element = line->AppendChild(std::move(wrapper));
            apply_item_width(slot.element);
        } else {
            if (slot.element->GetParentNode() != line)
                line->AppendChild(slot.element->GetParentNode()->RemoveChild(slot.element));
            if (auto* input = find_direct_input(slot.element);
                input && !slot.events->changed && !is_focused(input) &&
                set_attribute_if_changed(input, "value", Rml::String(value_text))) {
                slot.events->string_value = value_text;
                slot.events->float_value = value;
            }
        }

        last_element_ = slot.element;
        last_clicked_ = false;
        if (slot.events->changed) {
            slot.events->changed = false;
            try {
                float parsed = std::stof(slot.events->string_value);
                return {true, parsed};
            } catch (const std::exception&) {
                // LFS-CENSUS-OK(empty-catch): pure C++ std::stof parse of widget text (no
                // Python boundary); an unparseable entry keeps the prior value.
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
        const auto value_text = std::to_string(value);

        if (!slot.element) {
            auto wrapper = doc_->CreateElement("div");
            wrapper->SetClass("setting-row", true);

            auto display = strip_legacy_id(label);
            if (!display.empty()) {
                auto lbl = doc_->CreateElement("span");
                lbl->SetClass("prop-label", true);
                lbl->SetInnerRML(Rml::String(display));
                wrapper->AppendChild(std::move(lbl));
            }

            auto input = doc_->CreateElement("input");
            input->SetAttribute("type", "text");
            input->SetAttribute("value", Rml::String(value_text));
            input->SetClass("number-input", true);
            input->SetClass("im-control--fill", true);

            slot.events->string_value = value_text;
            input->AddEventListener(Rml::EventId::Change, new SlotEventListener(slot.events));

            wrapper->AppendChild(std::move(input));
            slot.element = line->AppendChild(std::move(wrapper));
            apply_item_width(slot.element);
        } else {
            if (slot.element->GetParentNode() != line)
                line->AppendChild(slot.element->GetParentNode()->RemoveChild(slot.element));
            if (auto* input = find_direct_input(slot.element);
                input && !slot.events->changed && !is_focused(input) &&
                set_attribute_if_changed(input, "value", Rml::String(value_text))) {
                slot.events->string_value = value_text;
            }
        }

        last_element_ = slot.element;
        last_clicked_ = false;
        if (slot.events->changed) {
            slot.events->changed = false;
            try {
                int parsed = std::stoi(slot.events->string_value);
                return {true, parsed};
            } catch (const std::exception&) {
                // LFS-CENSUS-OK(empty-catch): pure C++ std::stoi parse of widget text (no
                // Python boundary); an unparseable entry keeps the prior value.
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
                                                              const bool folder_mode,
                                                              const std::string& dialog_title) {
        push_id("path_input:" + label);
        auto [changed, edited_value] = input_text(label, value);
        assert(!containers_.empty() && containers_.back().cursor > 0);
        Slot* const input_slot =
            &containers_.back().slots[containers_.back().cursor - 1];
        assert(input_slot->type == SlotType::InputText);
        same_line();
        const bool browse = small_button("Browse##path_browse");
        pop_id();

        if (!browse)
            return {changed, std::move(edited_value)};

        const auto default_path = lfs::core::utf8_to_path(edited_value);
        const std::optional<std::string> title =
            dialog_title.empty() ? std::nullopt
                                 : std::optional<std::string>(dialog_title);
        std::filesystem::path selected;
        {
            nb::gil_scoped_release release;
            selected = path_dialog_callback()(folder_mode, default_path, title);
        }
        if (selected.empty())
            return {changed, std::move(edited_value)};

        auto selected_value = lfs::core::path_to_utf8(selected);
        auto* const input = find_direct_input(input_slot->element);
        assert(input);
        set_attribute_if_changed(input, "value", Rml::String(selected_value));
        input_slot->events->string_value = selected_value;
        input_slot->events->changed = false;
        return {true, std::move(selected_value)};
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
            lbl->SetInnerRML(Rml::String(strip_legacy_id(label)));

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

            el->AddEventListener(Rml::EventId::Click, new SlotEventListener(slot.events));

            slot.element = line->AppendChild(std::move(el));
        } else {
            if (slot.element->GetParentNode() != line)
                line->AppendChild(slot.element->GetParentNode()->RemoveChild(slot.element));
            if (!css.empty())
                slot.element->SetProperty("background-color", Rml::String(css));
        }

        bool clicked = slot.events->clicked;
        slot.events->clicked = false;
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

        const ComboItemsFingerprint items_fingerprint = hash_combo_items(items);
        const bool items_dirty =
            !slot.events->items_initialized ||
            slot.events->items_count != items.size() ||
            slot.events->items_hash != items_fingerprint.primary ||
            slot.events->items_hash_secondary != items_fingerprint.secondary;

        if (!slot.element) {
            auto wrapper = doc_->CreateElement("div");
            wrapper->SetClass("setting-row", true);

            auto lbl = doc_->CreateElement("span");
            lbl->SetClass("prop-label", true);
            lbl->SetInnerRML(Rml::String(strip_legacy_id(label)));

            auto select = doc_->CreateElement("select");
            select->SetClass("im-control--fill", true);
            for (int i = 0; i < static_cast<int>(items.size()); ++i) {
                auto option = doc_->CreateElement("option");
                option->SetAttribute("value", Rml::String(std::to_string(i)));
                option->SetInnerRML(Rml::String(items[i]));
                if (i == current_idx)
                    option->SetAttribute("selected", "");
                select->AppendChild(std::move(option));
            }

            slot.events->int_value = current_idx;
            slot.events->items_hash = items_fingerprint.primary;
            slot.events->items_hash_secondary = items_fingerprint.secondary;
            slot.events->items_count = items.size();
            slot.events->items_initialized = true;
            select->AddEventListener(Rml::EventId::Change, new SlotEventListener(slot.events));

            wrapper->AppendChild(std::move(lbl));
            wrapper->AppendChild(std::move(select));
            slot.element = line->AppendChild(std::move(wrapper));
            apply_item_width(slot.element);
        } else {
            if (slot.element->GetParentNode() != line)
                line->AppendChild(slot.element->GetParentNode()->RemoveChild(slot.element));

            auto* select = slot.element->GetChild(1);
            if (select) {
                if (items_dirty) {
                    // Replace the entire <select> element to avoid RmlUi
                    // retaining stale internal form state from child removal.
                    auto new_select = doc_->CreateElement("select");
                    new_select->SetClass("im-control--fill", true);
                    for (int i = 0; i < static_cast<int>(items.size()); ++i) {
                        auto option = doc_->CreateElement("option");
                        option->SetAttribute("value", Rml::String(std::to_string(i)));
                        option->SetInnerRML(Rml::String(items[i]));
                        if (i == current_idx)
                            option->SetAttribute("selected", "");
                        new_select->AppendChild(std::move(option));
                    }
                    new_select->AddEventListener(Rml::EventId::Change, new SlotEventListener(slot.events));
                    slot.element->RemoveChild(select);
                    slot.element->AppendChild(std::move(new_select));

                    slot.events->items_hash = items_fingerprint.primary;
                    slot.events->items_hash_secondary = items_fingerprint.secondary;
                    slot.events->items_count = items.size();
                    slot.events->items_initialized = true;
                    slot.events->int_value = current_idx;
                    slot.events->changed = false;
                } else if (!slot.events->changed) {
                    set_attribute_if_changed(
                        select, "value", Rml::String(std::to_string(current_idx)));
                }
            }
        }

        last_element_ = slot.element;
        last_clicked_ = false;
        if (slot.events->changed) {
            slot.events->changed = false;
            return {true, slot.events->int_value};
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
        const auto display = strip_legacy_id(label);

        if (!slot.element) {
            auto el = doc_->CreateElement("div");
            el->SetClass("context-menu-item", true);
            if (selected)
                el->SetClass("active", true);
            set_slot_text(slot, 0, el.get(), display);

            el->AddEventListener(Rml::EventId::Click, new SlotEventListener(slot.events));

            slot.element = line->AppendChild(std::move(el));
        } else {
            if (slot.element->GetParentNode() != line)
                line->AppendChild(slot.element->GetParentNode()->RemoveChild(slot.element));
            slot.element->SetClass("active", selected);
            set_slot_text(slot, 0, slot.element, display);
        }

        bool clicked = slot.events->clicked;
        slot.events->clicked = false;
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
            el->SetClass("im-spacing", true);
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
            slot.events->open = force_next_open_ || default_open;
            force_next_open_ = false;

            auto header = doc_->CreateElement("div");
            header->SetClass("section-header", true);

            auto arrow = doc_->CreateElement("span");
            arrow->SetClass("section-arrow", true);
            set_slot_text(
                slot, 0, arrow.get(), slot.events->open ? "\xe2\x96\xbc" : "\xe2\x96\xb6");

            auto text = doc_->CreateElement("span");
            text->SetInnerRML(Rml::String(strip_legacy_id(label)));

            header->AddEventListener(Rml::EventId::Click, new SlotEventListener(slot.events));

            header->AppendChild(std::move(arrow));
            header->AppendChild(std::move(text));

            assert(!containers_.empty());
            slot.element = containers_.back().parent->AppendChild(std::move(header));
        } else {
            if (force_next_open_) {
                slot.events->open = true;
                force_next_open_ = false;
            }
            if (slot.events->clicked) {
                slot.events->clicked = false;
                slot.events->open = !slot.events->open;
            }
            auto* arrow = slot.element->GetChild(0);
            if (arrow)
                set_slot_text(
                    slot, 0, arrow, slot.events->open ? "\xe2\x96\xbc" : "\xe2\x96\xb6");
        }

        last_element_ = slot.element;
        last_clicked_ = false;
        return slot.events->open;
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
        assert(!table_);
        finish_current_line();

        assert(!containers_.empty());
        auto& slot = ensure_slot(SlotType::Line, build_id("table:" + id));

        if (!slot.element) {
            auto el = doc_->CreateElement("div");
            el->SetClass("im-table", true);
            slot.element = containers_.back().parent->AppendChild(std::move(el));
        }

        table_ = TableState{};
        table_->num_columns = columns;
        table_->column_widths.resize(columns, 0.0f);
        table_->key = slot.key;
        table_->id_stack_depth = id_stack_.size();
        table_->table_element = slot.element;
        return true;
    }

    void RmlImModeLayout::table_setup_column(const std::string& /*label*/, float width) {
        if (!table_ || table_->setup_column >= table_->num_columns)
            return;
        table_->column_widths[table_->setup_column++] = width;
    }

    void RmlImModeLayout::finish_table_cell() {
        if (!table_ || !table_->cell_container_open)
            return;
        pop_persistent_container();
        table_->cell_container_open = false;
        table_->current_cell = nullptr;
    }

    void RmlImModeLayout::finish_table_row() {
        if (!table_ || !table_->current_row)
            return;

        finish_table_cell();
        const int keep_cells = table_->current_column + 1;
        while (table_->current_row->GetNumChildren() > keep_cells) {
            const int cell_index = table_->current_row->GetNumChildren() - 1;
            auto* const cell = table_->current_row->GetChild(cell_index);
            const auto cell_key = cell->GetAttribute<Rml::String>("data-im-cell", "");
            if (!cell_key.empty())
                child_slots_.erase(cell_key);
            removed_elements_.push_back(
                table_->current_row->RemoveChild(cell));
        }
    }

    void RmlImModeLayout::end_table() {
        if (!table_)
            return;

        finish_table_row();
        const int keep_rows = table_->current_row_index + 1;
        while (table_->table_element->GetNumChildren() > keep_rows) {
            const int row_index = table_->table_element->GetNumChildren() - 1;
            auto* row = table_->table_element->GetChild(row_index);
            for (int cell_index = 0; cell_index < row->GetNumChildren(); ++cell_index) {
                const auto cell_key = row->GetChild(cell_index)
                                          ->GetAttribute<Rml::String>("data-im-cell", "");
                if (!cell_key.empty())
                    child_slots_.erase(cell_key);
            }
            removed_elements_.push_back(table_->table_element->RemoveChild(row));
        }
        table_.reset();
    }

    void RmlImModeLayout::table_next_row() {
        if (!table_ || !table_->table_element || !doc_)
            return;

        finish_table_row();
        table_->current_row_index++;

        if (id_stack_.size() > table_->id_stack_depth) {
            table_->current_row_key = table_->key + "/row-id";
            for (size_t i = table_->id_stack_depth; i < id_stack_.size(); ++i) {
                auto token = stable_label_token(id_stack_[i]);
                if (token.empty())
                    token = id_stack_[i];
                table_->current_row_key +=
                    std::format(":{}:{}", token.size(), token);
            }
        } else {
            table_->current_row_key =
                std::format("{}/row:{}", table_->key, table_->current_row_index);
        }

        Rml::Element* matching_row = nullptr;
        for (int i = table_->current_row_index;
             i < table_->table_element->GetNumChildren(); ++i) {
            auto* const candidate = table_->table_element->GetChild(i);
            if (candidate->GetAttribute<Rml::String>("data-im-row", "") ==
                table_->current_row_key) {
                matching_row = candidate;
                break;
            }
        }

        if (!matching_row) {
            auto row = doc_->CreateElement("div");
            row->SetClass("im-table-row", true);
            row->SetAttribute("data-im-row", Rml::String(table_->current_row_key));
            auto* const insertion_point =
                table_->current_row_index < table_->table_element->GetNumChildren()
                    ? table_->table_element->GetChild(table_->current_row_index)
                    : nullptr;
            table_->current_row =
                table_->table_element->InsertBefore(std::move(row), insertion_point);
        } else if (matching_row !=
                   table_->table_element->GetChild(table_->current_row_index)) {
            Rml::Element* restore_focus = nullptr;
            if (auto* const context = matching_row->GetContext()) {
                auto* const focused = context->GetFocusElement();
                if (is_descendant_of(focused, matching_row))
                    restore_focus = focused;
            }
            auto moved_row = table_->table_element->RemoveChild(matching_row);
            table_->current_row = table_->table_element->InsertBefore(
                std::move(moved_row),
                table_->table_element->GetChild(table_->current_row_index));
            if (restore_focus)
                restore_focus->Focus();
        } else {
            table_->current_row = matching_row;
        }
        table_->current_cell = nullptr;
        table_->current_column = -1;
    }

    void RmlImModeLayout::table_next_column() {
        if (!table_ || !table_->current_row || !doc_)
            return;

        finish_table_cell();
        table_->current_column++;
        const auto cell_key = std::format(
            "{}/cell:{}", table_->current_row_key, table_->current_column);

        if (table_->current_column < table_->current_row->GetNumChildren()) {
            table_->current_cell = table_->current_row->GetChild(table_->current_column);
        } else {
            auto cell = doc_->CreateElement("div");
            cell->SetClass("im-table-cell", true);
            cell->SetAttribute("data-im-cell", Rml::String(cell_key));
            table_->current_cell = table_->current_row->AppendChild(std::move(cell));
        }

        int col = table_->current_column;
        if (col < static_cast<int>(table_->column_widths.size())) {
            const float width = table_->column_widths[col];
            const Rml::String width_key =
                width > 0.0f
                    ? Rml::String(std::to_string(static_cast<int>(width)) + "dp")
                    : Rml::String("fill");
            if (set_attribute_if_changed(
                    table_->current_cell, "data-im-cell-width", width_key)) {
                table_->current_cell->SetClass(
                    "im-table-cell--fill", width <= 0.0f);
                if (width > 0.0f)
                    table_->current_cell->SetProperty("width", width_key);
                else
                    table_->current_cell->RemoveProperty("width");
            }
        }

        push_persistent_container(cell_key, table_->current_cell);
        table_->cell_container_open = true;
    }

    bool RmlImModeLayout::table_set_column_index(int column) {
        if (!table_ || !table_->current_row || !doc_)
            return false;

        if (column < 0 || column >= table_->num_columns)
            return false;
        while (table_->current_column < column)
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
        const auto value = Rml::String(std::to_string(fraction));
        const auto text_content =
            overlay.empty()
                ? std::format("{}%", static_cast<int>(fraction * 100))
                : overlay;

        if (!slot.element) {
            auto wrapper = doc_->CreateElement("div");
            wrapper->SetClass("im-progress", true);

            auto prog = doc_->CreateElement("progress");
            prog->SetAttribute("value", value);
            prog->SetAttribute("max", "1");

            auto text = doc_->CreateElement("span");
            text->SetClass("progress__text", true);
            set_slot_text(slot, 0, text.get(), text_content);

            wrapper->AppendChild(std::move(prog));
            wrapper->AppendChild(std::move(text));
            slot.element = line->AppendChild(std::move(wrapper));
        } else {
            if (slot.element->GetParentNode() != line)
                line->AppendChild(slot.element->GetParentNode()->RemoveChild(slot.element));
            auto* prog = slot.element->GetChild(0);
            if (prog)
                set_attribute_if_changed(prog, "value", value);
            auto* text = slot.element->GetChild(1);
            if (text)
                set_slot_text(slot, 0, text, text_content);
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

        tooltip_candidate_seen_ = true;
        const auto now = std::chrono::steady_clock::now();
        if (tooltip_hover_el_ != last_element_ || tooltip_text_ != text) {
            tooltip_hover_el_ = last_element_;
            tooltip_text_ = text;
            tooltip_hover_started_at_ = now;
        }

        if (tooltip_hover_started_at_ == std::chrono::steady_clock::time_point{} ||
            now - tooltip_hover_started_at_ < lfs::vis::gui::kRmlTooltipShowDelay)
            return;

        auto body_offset = doc_->GetAbsoluteOffset(Rml::BoxArea::Content);
        float local_x = mouse_.pos_x - body_offset.x;
        float local_y = mouse_.pos_y - body_offset.y;

        if (rendered_tooltip_text_ != text) {
            if (tooltip_el_->GetNumChildren() == 1) {
                if (auto* text_node =
                        dynamic_cast<Rml::ElementText*>(tooltip_el_->GetFirstChild())) {
                    text_node->SetText(Rml::String(text));
                } else {
                    tooltip_el_->SetInnerRML(Rml::String(text));
                }
            } else {
                tooltip_el_->SetInnerRML(Rml::String(text));
            }
            rendered_tooltip_text_ = text;
        }
        tooltip_el_->SetClass("visible", true);
        tooltip_el_->SetProperty("left", Rml::String(std::format("{:.0f}dp", local_x + 12.0f)));
        tooltip_el_->SetProperty("top", Rml::String(std::format("{:.0f}dp", local_y + 12.0f)));
        tooltip_shown_ = true;
    }

    // ── Item state ──────────────────────────────────────────

    bool RmlImModeLayout::is_item_hovered() {
        return last_element_ && last_element_->IsPseudoClassSet("hover");
    }
    bool RmlImModeLayout::is_item_clicked(int button) {
        if (button == 1)
            return last_element_ && last_element_->IsPseudoClassSet("hover") && mouse_.right_clicked;
        return last_clicked_;
    }
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

        auto key = build_id("child:" + id);
        auto& slot = ensure_slot(SlotType::Line, key);

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

        push_persistent_container(key, slot.element);
        return true;
    }

    void RmlImModeLayout::end_child() {
        pop_persistent_container();
    }

    // ── Menu bar ────────────────────────────────────────────

    bool RmlImModeLayout::begin_menu_bar() {
        if (!doc_)
            return false;

        finish_current_line();
        const auto key = build_slot_id("menu_bar");
        auto& slot = ensure_slot(SlotType::MenuBar, key);

        if (!slot.element) {
            auto el = doc_->CreateElement("div");
            el->SetClass("im-menu-bar", true);
            slot.element = containers_.back().parent->AppendChild(std::move(el));
        } else if (slot.element->GetParentNode() != containers_.back().parent) {
            containers_.back().parent->AppendChild(slot.element->GetParentNode()->RemoveChild(slot.element));
        }

        push_persistent_container(key, slot.element);
        last_element_ = slot.element;
        last_clicked_ = false;
        return true;
    }

    void RmlImModeLayout::end_menu_bar() {
        pop_persistent_container();
    }

    bool RmlImModeLayout::begin_menu(const std::string& label) {
        if (!doc_)
            return false;

        finish_current_line();
        const auto key = build_slot_id("menu", &label);
        auto& slot = ensure_slot(SlotType::Menu, key);
        const auto display = strip_legacy_id(label);

        if (!slot.element) {
            slot.events->open = force_next_open_;
            force_next_open_ = false;

            auto wrapper = doc_->CreateElement("div");
            wrapper->SetClass("im-menu", true);

            auto trigger = doc_->CreateElement("button");
            trigger->SetClass("btn", true);
            trigger->SetClass("im-menu-trigger", true);
            trigger->AddEventListener(Rml::EventId::Click, new SlotEventListener(slot.events));

            auto panel = doc_->CreateElement("div");
            panel->SetClass("im-menu-panel", true);

            wrapper->AppendChild(std::move(trigger));
            wrapper->AppendChild(std::move(panel));
            slot.element = containers_.back().parent->AppendChild(std::move(wrapper));
        } else {
            if (slot.element->GetParentNode() != containers_.back().parent)
                containers_.back().parent->AppendChild(slot.element->GetParentNode()->RemoveChild(slot.element));
            if (force_next_open_) {
                slot.events->open = true;
                force_next_open_ = false;
            }
        }

        if (slot.events->clicked) {
            slot.events->clicked = false;
            slot.events->open = !slot.events->open;
        }

        if (auto* trigger = slot.element->GetChild(0)) {
            set_slot_text(
                slot, 0, trigger,
                std::format("{} {}", slot.events->open ? "v" : ">", display));
        }

        auto* panel = slot.element->GetChild(1);
        if (panel)
            panel->SetClass("open", slot.events->open);

        last_element_ = slot.element;
        last_clicked_ = false;

        if (!slot.events->open || !panel)
            return false;

        push_persistent_container(key, panel);
        return true;
    }

    void RmlImModeLayout::end_menu() {
        pop_persistent_container();
    }

    bool RmlImModeLayout::menu_item_impl(const std::string& label, const std::string& shortcut,
                                         bool enabled, bool selected) {
        if (!doc_)
            return false;

        finish_current_line();
        auto& slot = ensure_slot(SlotType::MenuItem, build_slot_id("menu_item", &label));
        const auto display = strip_legacy_id(label);

        const auto ensure_children = [&]() {
            if (!slot.element)
                return;
            if (slot.element->GetNumChildren() >= 2)
                return;
            slot.element->SetInnerRML("");
            slot.content = {};
            auto label_el = doc_->CreateElement("span");
            label_el->SetClass("im-menu-label", true);
            auto shortcut_el = doc_->CreateElement("span");
            shortcut_el->SetClass("im-menu-shortcut", true);
            slot.element->AppendChild(std::move(label_el));
            slot.element->AppendChild(std::move(shortcut_el));
        };

        if (!slot.element) {
            auto item = doc_->CreateElement("button");
            item->SetClass("im-menu-item", true);
            item->AddEventListener(Rml::EventId::Click, new SlotEventListener(slot.events));
            slot.element = containers_.back().parent->AppendChild(std::move(item));
            ensure_children();
        } else if (slot.element->GetParentNode() != containers_.back().parent) {
            containers_.back().parent->AppendChild(slot.element->GetParentNode()->RemoveChild(slot.element));
        }

        if (auto* label_el = slot.element->GetChild(0))
            set_slot_text(slot, 0, label_el, display);
        if (auto* shortcut_el = slot.element->GetChild(1))
            set_slot_text(slot, 1, shortcut_el, shortcut);

        const bool item_enabled = enabled && !disabled_;
        slot.element->SetClass("selected", selected);
        slot.element->SetClass("disabled-overlay", !item_enabled);
        if (item_enabled) {
            if (slot.element->HasAttribute("disabled"))
                slot.element->RemoveAttribute("disabled");
        } else {
            set_attribute_if_changed(slot.element, "disabled", "");
        }

        const bool clicked = item_enabled && slot.events->clicked;
        slot.events->clicked = false;
        last_element_ = slot.element;
        last_clicked_ = clicked;
        return clicked;
    }

    bool RmlImModeLayout::menu_item(const std::string& label, bool enabled, bool selected) {
        return menu_item_impl(label, "", enabled, selected);
    }

    std::tuple<bool, bool> RmlImModeLayout::menu_item_toggle(const std::string& label,
                                                             const std::string& shortcut,
                                                             bool selected) {
        const bool clicked = menu_item_impl(label, shortcut, true, selected);
        return {clicked, clicked ? !selected : selected};
    }

    bool RmlImModeLayout::menu_item_shortcut(const std::string& label,
                                             const std::string& shortcut,
                                             bool enabled) {
        return menu_item_impl(label, shortcut, enabled, false);
    }

    // ── Popups ──────────────────────────────────────────────

    bool RmlImModeLayout::begin_popup(const std::string& id) {
        return begin_popup_modal(id);
    }

    void RmlImModeLayout::open_popup(const std::string& id) {
        popup_open_[id] = true;
    }

    void RmlImModeLayout::end_popup() {
        end_popup_modal();
    }

    bool RmlImModeLayout::begin_context_menu(const std::string& id) {
        const std::string popup_id = id.empty() ? "__context_menu" : id;
        if (last_element_ && last_element_->IsPseudoClassSet("hover") && mouse_.right_clicked)
            popup_open_[popup_id] = true;
        return begin_popup(popup_id);
    }

    void RmlImModeLayout::end_context_menu() {
        end_popup();
    }

    bool RmlImModeLayout::begin_popup_modal(const std::string& title) {
        auto it = popup_open_.find(title);
        if (it == popup_open_.end() || !it->second)
            return false;
        if (!popup_backdrop_ || !popup_dialog_)
            return false;

        popup_backdrop_->SetClass("visible", true);
        popup_dialog_->SetClass("visible", true);

        const auto doc_size = doc_->GetBox().GetSize(Rml::BoxArea::Content);
        const auto body_offset = doc_->GetAbsoluteOffset(Rml::BoxArea::Content);
        float popup_x = mouse_.pos_x - body_offset.x + 12.0f;
        float popup_y = mouse_.pos_y - body_offset.y + 12.0f;
        const float max_x = std::max(8.0f, doc_size.x - 320.0f);
        const float max_y = std::max(8.0f, doc_size.y - 160.0f);
        popup_x = std::min(std::max(popup_x, 8.0f), max_x);
        popup_y = std::min(std::max(popup_y, 8.0f), max_y);
        popup_dialog_->SetProperty("left", Rml::String(std::format("{:.0f}dp", popup_x)));
        popup_dialog_->SetProperty("top", Rml::String(std::format("{:.0f}dp", popup_y)));

        const auto dialog_id = popup_dialog_->GetAttribute<Rml::String>("data-popup-id", "");
        if (dialog_id != title || popup_dialog_->GetNumChildren() < 2) {
            popup_dialog_->SetInnerRML("");
            popup_dialog_->SetAttribute("data-popup-id", Rml::String(title));

            auto title_el = doc_->CreateElement("div");
            title_el->SetClass("im-popup-title", true);

            auto body_el = doc_->CreateElement("div");
            body_el->SetClass("im-popup-body", true);

            popup_dialog_->AppendChild(std::move(title_el));
            popup_dialog_->AppendChild(std::move(body_el));
            popup_dialog_->GetChild(0)->SetInnerRML(Rml::String(strip_legacy_id(title)));
        }

        active_popup_id_ = title;
        finish_current_line();
        auto* body = popup_dialog_->GetChild(1);
        if (!body)
            return false;
        push_persistent_container(build_id("popup:" + title), body);
        return true;
    }

    void RmlImModeLayout::end_popup_modal() {
        if (active_popup_id_.empty())
            return;
        pop_persistent_container();
        active_popup_id_.clear();
    }

    void RmlImModeLayout::close_current_popup() {
        if (!active_popup_id_.empty())
            popup_open_[active_popup_id_] = false;
        if (popup_backdrop_)
            popup_backdrop_->SetClass("visible", false);
        if (popup_dialog_)
            popup_dialog_->SetClass("visible", false);
    }

    void RmlImModeLayout::push_modal_style() {}
    void RmlImModeLayout::pop_modal_style() {}

    // ── Images ──────────────────────────────────────────────

    void RmlImModeLayout::image(uint64_t texture_id, std::tuple<float, float> size, nb::object /*tint*/) {
        if (!doc_)
            return;
        auto* line = ensure_line_container();
        auto& slot = ensure_slot(SlotType::Image, build_slot_id("image"));
        const auto [w, h] = size;
        const std::string src = rml_src_for_dynamic_texture(
            texture_id, static_cast<int>(w), static_cast<int>(h));

        if (!slot.element) {
            auto wrapper = doc_->CreateElement("div");
            wrapper->SetClass("im-image-frame", true);

            auto img = doc_->CreateElement("img");
            img->SetClass("im-image", true);

            auto placeholder = doc_->CreateElement("span");
            placeholder->SetClass("im-image-placeholder", true);
            placeholder->SetInnerRML("texture");

            wrapper->AppendChild(std::move(img));
            wrapper->AppendChild(std::move(placeholder));
            slot.element = line->AppendChild(std::move(wrapper));
        } else if (slot.element->GetParentNode() != line) {
            line->AppendChild(slot.element->GetParentNode()->RemoveChild(slot.element));
        }

        slot.element->SetClass("im-image-missing", src.empty());
        if (w > 0)
            slot.element->SetProperty("width", Rml::String(std::format("{:.0f}dp", w)));
        if (h > 0)
            slot.element->SetProperty("height", Rml::String(std::format("{:.0f}dp", h)));
        if (auto* img = slot.element->GetChild(0))
            set_attribute_if_changed(img, "src", Rml::String(src));

        last_element_ = slot.element;
        last_clicked_ = false;
    }

    void RmlImModeLayout::image_uv(uint64_t texture_id, std::tuple<float, float> size,
                                   std::tuple<float, float> /*uv0*/,
                                   std::tuple<float, float> /*uv1*/,
                                   nb::object tint) {
        image(texture_id, size, tint);
    }

    bool RmlImModeLayout::image_button(const std::string& id, uint64_t texture_id,
                                       std::tuple<float, float> size, nb::object /*tint*/) {
        if (!doc_)
            return false;
        auto* line = ensure_line_container();
        auto& slot = ensure_slot(SlotType::ImageButton, build_slot_id("image_button", &id));
        const auto [w, h] = size;
        const std::string src = rml_src_for_dynamic_texture(
            texture_id, static_cast<int>(w), static_cast<int>(h));
        const auto display = strip_legacy_id(id);

        if (!slot.element) {
            auto button = doc_->CreateElement("button");
            button->SetClass("btn", true);
            button->SetClass("im-image-button", true);

            auto img = doc_->CreateElement("img");
            img->SetClass("im-image", true);

            auto placeholder = doc_->CreateElement("span");
            placeholder->SetClass("im-image-placeholder", true);
            set_slot_text(
                slot, 0, placeholder.get(), display.empty() ? "texture" : display);

            button->AppendChild(std::move(img));
            button->AppendChild(std::move(placeholder));
            button->AddEventListener(Rml::EventId::Click, new SlotEventListener(slot.events));
            slot.element = line->AppendChild(std::move(button));
        } else if (slot.element->GetParentNode() != line) {
            line->AppendChild(slot.element->GetParentNode()->RemoveChild(slot.element));
        }

        slot.element->SetClass("im-image-missing", src.empty());
        if (w > 0)
            slot.element->SetProperty("width", Rml::String(std::format("{:.0f}dp", w)));
        if (h > 0)
            slot.element->SetProperty("height", Rml::String(std::format("{:.0f}dp", h)));
        if (auto* img = slot.element->GetChild(0))
            set_attribute_if_changed(img, "src", Rml::String(src));
        if (auto* placeholder = slot.element->GetChild(1))
            set_slot_text(
                slot, 0, placeholder, display.empty() ? "texture" : display);

        const bool clicked = slot.events->clicked;
        slot.events->clicked = false;
        last_element_ = slot.element;
        last_clicked_ = clicked;
        return clicked;
    }

    bool RmlImModeLayout::toolbar_button(const std::string& id, uint64_t texture_id,
                                         std::tuple<float, float> size, bool selected,
                                         bool disabled, const std::string& tooltip) {
        const bool clicked = image_button(id, texture_id, size);
        if (last_element_) {
            last_element_->SetClass("im-toolbar-button", true);
            last_element_->SetClass("selected", selected);
            last_element_->SetClass("disabled-overlay", disabled);
        }
        if (!tooltip.empty())
            set_tooltip(tooltip);
        return disabled ? false : clicked;
    }
    bool RmlImModeLayout::invisible_button(const std::string& id, std::tuple<float, float> size) {
        if (!doc_)
            return false;

        auto* line = ensure_line_container();
        auto& slot = ensure_slot(SlotType::Button, build_slot_id("invisible_button", &id));
        const auto [w, h] = size;

        if (!slot.element) {
            auto button = doc_->CreateElement("button");
            button->SetClass("im-invisible-button", true);
            button->AddEventListener(Rml::EventId::Click, new SlotEventListener(slot.events));
            slot.element = line->AppendChild(std::move(button));
        } else if (slot.element->GetParentNode() != line) {
            line->AppendChild(slot.element->GetParentNode()->RemoveChild(slot.element));
        }

        slot.element->SetProperty("width", Rml::String(std::format("{:.0f}dp", w > 0.0f ? w : 1.0f)));
        slot.element->SetProperty("height", Rml::String(std::format("{:.0f}dp", h > 0.0f ? h : cached_line_height_)));

        const bool clicked = slot.events->clicked;
        slot.events->clicked = false;
        last_element_ = slot.element;
        last_clicked_ = clicked;
        return clicked;
    }

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

    // ── Retained-only specialized widgets ───────────────────

    void RmlImModeLayout::crf_curve_preview(const std::string& /*label*/, float, float, float, float, float, float) {
        warn_retained_custom_element_once("crf_curve_preview", "<crf-curve>");
    }
    std::tuple<bool, std::vector<float>> RmlImModeLayout::chromaticity_diagram(const std::string& /*label*/,
                                                                               const float red_x, const float red_y,
                                                                               const float green_x, const float green_y,
                                                                               const float blue_x, const float blue_y,
                                                                               const float neutral_x, const float neutral_y,
                                                                               float) {
        warn_retained_custom_element_once(
            "chromaticity_diagram", "<chromaticity-diagram>");
        return {false, {red_x, red_y, green_x, green_y, blue_x, blue_y, neutral_x, neutral_y}};
    }

    // ── Plots ───────────────────────────────────────────────

    void RmlImModeLayout::plot_lines(const std::string& label, nb::object values,
                                     float /*scale_min*/, float /*scale_max*/,
                                     std::tuple<float, float> size) {
        if (!doc_)
            return;

        finish_current_line();
        assert(!containers_.empty());

        auto& slot = ensure_slot(SlotType::PlotLines, build_slot_id("plot_lines", &label));
        auto* parent = containers_.back().parent;
        const auto display = strip_legacy_id(label);
        update_values_deque(values, slot.plot_scratch);
        const bool data_dirty =
            !slot.plot_initialized || slot.plot_values != slot.plot_scratch;
        if (data_dirty) {
            slot.plot_values.swap(slot.plot_scratch);
            slot.plot_initialized = true;
        }

        if (!slot.element) {
            auto wrapper = doc_->CreateElement("div");
            wrapper->SetClass("im-plot", true);

            auto title = doc_->CreateElement("p");
            title->SetClass("im-plot-label", true);
            set_slot_text(slot, 0, title.get(), display);

            auto graph = doc_->CreateElement("loss-graph");
            graph->SetClass("im-plot-graph", true);

            wrapper->AppendChild(std::move(title));
            wrapper->AppendChild(std::move(graph));
            slot.element = parent->AppendChild(std::move(wrapper));
        } else if (slot.element->GetParentNode() != parent) {
            parent->AppendChild(slot.element->GetParentNode()->RemoveChild(slot.element));
        }

        if (auto* title = slot.element->GetChild(0))
            set_slot_text(slot, 0, title, display);

        const auto [width, height] = size;
        const float effective_width = width > 0.0f ? width : 0.0f;
        if (!slot.plot_width || *slot.plot_width != effective_width) {
            const Rml::String width_value =
                width > 0.0f ? Rml::String(std::format("{:.0f}dp", width))
                             : Rml::String("100%");
            if (set_attribute_if_changed(
                    slot.element, "data-im-plot-width", width_value))
                slot.element->SetProperty("width", width_value);
            slot.plot_width = effective_width;
        }

        auto* graph_el = slot.element->GetChild(1);
        if (graph_el) {
            const float effective_height = height > 0.0f ? height : 72.0f;
            if (!slot.plot_height || *slot.plot_height != effective_height) {
                const Rml::String height_value =
                    Rml::String(std::format("{:.0f}dp", effective_height));
                if (set_attribute_if_changed(
                        graph_el, "data-im-plot-height", height_value))
                    graph_el->SetProperty("height", height_value);
                slot.plot_height = effective_height;
            }
            if (data_dirty) {
                if (auto* graph =
                        dynamic_cast<lfs::vis::gui::LossGraphElement*>(graph_el))
                    graph->setData(slot.plot_values);
                else
                    set_slot_text(
                        slot, 1, graph_el,
                        slot.plot_values.empty() ? "" : "plot");
            }
        }

        last_element_ = slot.element;
        last_clicked_ = false;
    }

    // ── Persistent sub-layout containers ────────────────────

    nb::object RmlImModeLayout::row() {
        return nb::cast(RmlSubLayout(this, RmlLayoutType::Row));
    }
    nb::object RmlImModeLayout::column() {
        return nb::cast(RmlSubLayout(this, RmlLayoutType::Column));
    }
    nb::object RmlImModeLayout::split(const float factor) {
        return nb::cast(RmlSubLayout(this, RmlLayoutType::Split, factor));
    }
    nb::object RmlImModeLayout::box() {
        return nb::cast(RmlSubLayout(this, RmlLayoutType::Box));
    }
    nb::object RmlImModeLayout::grid_flow(const int columns,
                                          const bool even_columns,
                                          const bool even_rows) {
        return nb::cast(RmlSubLayout(
            this, RmlLayoutType::GridFlow, 0.5f, columns, even_columns, even_rows));
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
                    // LFS-CENSUS-OK(empty-catch): read-only tensor introspection for a
                    // display label; any failure falls back to a placeholder string.
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
                    } catch (nb::python_error& e) {
                        (void)contain_python_callback(e, PyCallbackPolicy::WarnAndContinue);
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

    RmlSubLayout::RmlSubLayout(RmlImModeLayout* parent, const RmlLayoutType type,
                               const float factor, const int columns,
                               const bool even_columns, const bool even_rows)
        : parent_(parent),
          type_(type),
          factor_(factor),
          columns_(columns),
          even_columns_(even_columns),
          even_rows_(even_rows) {
        assert(parent_);
        assert(std::isfinite(factor_));
        assert(columns_ >= 0);
    }

    RmlSubLayout& RmlSubLayout::enter() {
        assert(!entered_);
        assert(parent_->doc_);
        entered_ = true;
        parent_->finish_current_line();
        assert(!parent_->containers_.empty());

        const char* token = nullptr;
        const char* cls = nullptr;
        switch (type_) {
        case RmlLayoutType::Row:
            token = "row";
            cls = "im-row";
            break;
        case RmlLayoutType::Column:
            token = "column";
            cls = "im-column";
            break;
        case RmlLayoutType::Split:
            token = "split";
            cls = "im-split";
            break;
        case RmlLayoutType::Box:
            token = "box";
            cls = "im-box";
            break;
        case RmlLayoutType::GridFlow:
            token = "grid_flow";
            cls = "im-grid-flow";
            break;
        }
        assert(token && cls);

        const int slot_index = parent_->containers_.back().cursor;
        const std::string local_key =
            parent_->build_id(std::format("sublayout:{}:{}", token, slot_index));
        container_key_ =
            parent_->child_key_stack_.empty()
                ? local_key
                : parent_->child_key_stack_.back() + "/" + local_key;
        auto& slot = parent_->ensure_slot(SlotType::Line, container_key_);

        if (!slot.element) {
            auto el = parent_->doc_->CreateElement("div");
            el->SetClass(cls, true);
            slot.element = parent_->containers_.back().parent->AppendChild(std::move(el));
        } else if (slot.element->GetParentNode() != parent_->containers_.back().parent) {
            parent_->containers_.back().parent->AppendChild(
                slot.element->GetParentNode()->RemoveChild(slot.element));
        }

        container_element_ = slot.element;
        parent_->push_persistent_container(container_key_, container_element_);
        return *this;
    }

    void RmlSubLayout::exit() {
        if (!entered_)
            return;
        entered_ = false;

        parent_->pop_persistent_container();
        assert(container_element_);
        const int child_count = container_element_->GetNumChildren();

        const auto apply_basis = [](Rml::Element* child, const Rml::String& basis) {
            assert(child);
            if (set_attribute_if_changed(child, "data-im-flex-basis", basis))
                child->SetProperty("flex-basis", basis);
        };
        const auto apply_property = [](Rml::Element* child, const char* marker,
                                       const char* property, const Rml::String& value) {
            assert(child);
            if (set_attribute_if_changed(child, marker, value))
                child->SetProperty(property, value);
        };

        if (type_ == RmlLayoutType::Split) {
            if (child_count > 2 && !parent_->warned_split_overflow_) {
                parent_->warned_split_overflow_ = true;
                LOG_WARN("RmlImModeLayout: split supports two children; hiding {} excess child(ren)",
                         child_count - 2);
            }
            for (int i = 0; i < child_count; ++i) {
                auto* const child = container_element_->GetChild(i);
                if (i < 2) {
                    if (child->HasAttribute("data-im-split-overflow")) {
                        child->RemoveAttribute("data-im-split-overflow");
                        child->RemoveProperty("display");
                    }
                } else {
                    apply_property(
                        child, "data-im-split-overflow", "display", "none");
                }
            }

            const float first =
                std::isfinite(factor_) ? std::clamp(factor_, 0.0f, 1.0f) : 0.5f;
            if (child_count > 0)
                apply_basis(container_element_->GetChild(0),
                            Rml::String(std::format("{:.4g}%", first * 100.0f)));
            if (child_count > 1)
                apply_basis(container_element_->GetChild(1),
                            Rml::String(std::format("{:.4g}%", (1.0f - first) * 100.0f)));
        } else if (type_ == RmlLayoutType::GridFlow) {
            const Rml::String basis =
                even_columns_ && columns_ > 0
                    ? Rml::String(std::format("{:.4g}%", 100.0f / static_cast<float>(columns_)))
                    : Rml::String(even_columns_ ? "100dp" : "auto");
            for (int i = 0; i < child_count; ++i) {
                auto* const child = container_element_->GetChild(i);
                apply_basis(child, basis);
                apply_property(
                    child, "data-im-flex-grow", "flex-grow",
                    even_rows_ ? Rml::String("1") : Rml::String("0"));
                apply_property(
                    child, "data-im-align-self", "align-self",
                    even_rows_ ? Rml::String("stretch")
                               : Rml::String("flex-start"));
            }
        }
        container_element_ = nullptr;
        container_key_.clear();
    }

    RmlSubLayout RmlSubLayout::row() {
        return RmlSubLayout(parent_, RmlLayoutType::Row);
    }

    RmlSubLayout RmlSubLayout::column() {
        return RmlSubLayout(parent_, RmlLayoutType::Column);
    }

    RmlSubLayout RmlSubLayout::split(const float factor) {
        return RmlSubLayout(parent_, RmlLayoutType::Split, factor);
    }

    RmlSubLayout RmlSubLayout::box() {
        return RmlSubLayout(parent_, RmlLayoutType::Box);
    }

    RmlSubLayout RmlSubLayout::grid_flow(const int columns,
                                         const bool even_columns,
                                         const bool even_rows) {
        return RmlSubLayout(
            parent_, RmlLayoutType::GridFlow, 0.5f, columns, even_columns, even_rows);
    }

} // namespace lfs::python

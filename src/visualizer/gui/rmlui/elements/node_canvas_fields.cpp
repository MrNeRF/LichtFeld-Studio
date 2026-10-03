/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/rmlui/elements/node_canvas_element.hpp"

#include "core/event_bridge/localization_manager.hpp"
#include "core/nodes/tree.hpp"
#include "gui/global_context_menu.hpp"
#include "gui/rmlui/elements/node_canvas_widgets.hpp"
#include "visualizer/nodes/modifier_manager.hpp"

#include <RmlUi/Core/Context.h>
#include <RmlUi/Core/ElementDocument.h>
#include <RmlUi/Core/ElementText.h>
#include <RmlUi/Core/Elements/ElementFormControlInput.h>
#include <RmlUi/Core/Elements/ElementFormControlSelect.h>
#include <RmlUi/Core/Elements/ElementFormControlTextArea.h>
#include <RmlUi/Core/Event.h>
#include <RmlUi/Core/Input.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <format>
#include <ranges>

namespace lfs::vis::gui {
    namespace {
        void updateField(Rml::Element& field, double value) {
            if (field.HasAttribute("data-min"))
                value = std::max(value, field.GetAttribute<double>("data-min", value));
            if (field.HasAttribute("data-max"))
                value = std::min(value, field.GetAttribute<double>("data-max", value));
            const bool integer = field.GetAttribute<int>("data-integer", 0) != 0;
            if (integer)
                value = std::round(value);
            field.SetAttribute("data-value", value);
            const std::string text = integer ? std::format("{:.0f}", value) : std::format("{:.3g}", value);
            if (auto* display = field.QuerySelector(".scrub-field-display"))
                if (auto* value_text = dynamic_cast<Rml::ElementText*>(display->GetChild(0)); value_text && value_text->GetText() != text)
                    value_text->SetText(text);
            if (auto* input = dynamic_cast<Rml::ElementFormControlInput*>(field.QuerySelector("input")))
                input->SetValue(std::format("{}", value));
            const double minimum = field.GetAttribute<double>("data-fill-min", 0.0);
            const double maximum = field.GetAttribute<double>("data-fill-max", 0.0);
            if (auto* fill = field.QuerySelector(".scrub-field-fill"); fill && maximum > minimum)
                fill->SetProperty("width", std::format("{:.1f}%", std::clamp((value - minimum) / (maximum - minimum), 0.0, 1.0) * 100.0));
        }
    } // namespace

    void NodeCanvasElement::finishFieldEdit(const bool cancel) {
        auto* field = active_field_;
        if (!field)
            return;
        active_field_ = nullptr;
        field_step_direction_ = 0;
        bool discard = cancel;
        if (!discard && field->IsClassSet("is-editing")) {
            if (const auto* input = dynamic_cast<Rml::ElementFormControlInput*>(field->QuerySelector("input"))) {
                const std::string text = input->GetValue();
                char* end = nullptr;
                const double value = std::strtod(text.c_str(), &end);
                discard = end == text.c_str() || *end != '\0' || !std::isfinite(value);
                if (!discard)
                    updateField(*field, value);
            }
        }
        if (discard)
            updateField(*field, field_start_value_);
        field->SetClass("is-dragging", false);
        field->SetClass("is-editing", false);
        if (!discard) {
            if (auto* input = field->QuerySelector("input"))
                commitControl(input, nullptr, false);
        }
        if (field_before_) {
            if (auto* tree = activeTree()) {
                if (discard && tree->to_json() != *field_before_) {
                    *tree = lfs::nodes::NodeTree::from_json(*field_before_, manager_->registry());
                    manager_->markDirty();
                } else if (!discard) {
                    manager_->recordTreeEdit(tree->uuid, std::move(*field_before_), {}, false);
                }
                dom_dirty_ = true;
            }
        }
        field_before_.reset();
        field_dragged_ = false;
    }

    void NodeCanvasElement::stepField() {
        if (!active_field_)
            return;
        const double step = active_field_->GetAttribute<double>("data-step", 0.01);
        const double value = active_field_->GetAttribute<double>("data-value", 0.0);
        updateField(*active_field_, value + field_step_direction_ * field_step_multiplier_ * step);
        if (auto* input = active_field_->QuerySelector("input"))
            commitControl(input, nullptr, false);
    }

    void NodeCanvasElement::repeatFieldStep() {
        if (!active_field_ || field_step_direction_ == 0)
            return;
        const auto now = std::chrono::steady_clock::now();
        if (now >= field_repeat_at_) {
            stepField();
            field_repeat_at_ = now + std::chrono::milliseconds(70);
        }
        if (auto* context = GetContext())
            context->RequestNextUpdate(0.02);
    }

    bool NodeCanvasElement::processFieldEvent(Rml::Event& event) {
        if (!editableMode())
            return true;
        const auto& type = event.GetType();
        auto* target = event.GetTargetElement();
        Rml::Element* field = nullptr;
        Rml::Element* swatch = nullptr;
        bool control = false;
        bool sidebar = false;
        bool popup = false;
        Rml::Element* colour = nullptr;
        for (auto* element = target; element && element != this; element = element->GetParentNode()) {
            if (element->IsClassSet("node-scrub"))
                field = element;
            if (element->IsClassSet("node-swatch"))
                swatch = element;
            if (element->GetTagName() == "colour-offset" || element->GetTagName() == "color-picker")
                colour = element;
            control |= element->GetTagName() == "input" || element->GetTagName() == "select" ||
                       element->GetTagName() == "button" || colour;
            sidebar |= element == sidebar_element_;
            popup |= element->IsClassSet("node-colour-popup");
        }
        if (pointer_down_ && (type == "mousemove" || type == "mouseup" || type == "drag" || type == "dragend"))
            return false;
        if (type == "mousedown" && !sidebar && !popup &&
            (event.GetParameter("button", 0) == 2 || isPanPress(event)))
            return false;
        if (type == "mousedown" && !popup && !swatch)
            if (auto* existing = GetElementById("node-colour-popup"))
                RemoveChild(existing);
        if (type == "click" && swatch) {
            if (auto* existing = GetElementById("node-colour-popup"))
                RemoveChild(existing);
            auto element = GetOwnerDocument()->CreateElement("div");
            element->SetId("node-colour-popup");
            element->SetClass("node-colour-popup", true);
            const auto size = GetBox().GetSize(Rml::BoxArea::Content);
            const auto pointer = localPointer(event);
            element->SetProperty("left", std::format("{}px", std::clamp(pointer.x, 0.0f, std::max(0.0f, size.x - 232.0f * dp_ratio_))));
            element->SetProperty("top", std::format("{}px", std::clamp(pointer.y, 0.0f, std::max(0.0f, size.y - 177.0f * dp_ratio_))));
            auto picker = GetOwnerDocument()->CreateElement("color-picker");
            for (const char* name : {"data-node", "data-input", "data-offset"})
                picker->SetAttribute(name, swatch->GetAttribute<Rml::String>(name, ""));
            for (const char* channel : {"red", "green", "blue"}) {
                const float value = swatch->GetAttribute<float>(std::string("data-") + channel, 0.0f);
                picker->SetAttribute(channel, swatch->GetAttribute<int>("data-offset", 0) ? (value + 1.0f) * 0.5f : value);
            }
            element->AppendChild(std::move(picker));
            AppendChild(std::move(element));
            event.StopPropagation();
            return true;
        }
        const int key = event.GetParameter("key_identifier", 0);
        const bool editing_colour = active_field_ &&
                                    (active_field_->GetTagName() == "colour-offset" || active_field_->GetTagName() == "color-picker");
        if (editing_colour && (type == "mouseup" || type == "dragend"))
            finishFieldEdit(false);
        if (editing_colour && type == "keydown" && key == Rml::Input::KI_ESCAPE)
            finishFieldEdit(true);
        // Skip canvas interaction, but let motion reach the colour control.
        if (editing_colour && (type == "mousedown" || type == "mousemove" || type == "mouseup" ||
                               type == "dragstart" || type == "drag" || type == "dragend"))
            return true;
        const bool enter = key == Rml::Input::KI_RETURN || key == Rml::Input::KI_NUMPADENTER;
        if ((type == "dblclick" || (type == "keydown" && enter && !active_field_)) && field && !target->IsClassSet("node-step")) {
            active_field_ = field;
            if (const auto* tree = activeTree())
                field_before_ = tree->to_json();
            field_start_value_ = field->GetAttribute<double>("data-value", 0.0);
            field->SetClass("is-editing", true);
            if (auto* input = dynamic_cast<Rml::ElementFormControlInput*>(field->QuerySelector("input"))) {
                input->Focus();
                input->Select();
            }
            event.StopPropagation();
            return true;
        }
        if (type == "mousedown" && field && !field->IsClassSet("is-editing") && event.GetParameter("button", 0) == 0) {
            active_field_ = field;
            if (const auto* tree = activeTree())
                field_before_ = tree->to_json();
            field_start_x_ = localPointer(event).x;
            field_start_value_ = field->GetAttribute<double>("data-value", 0.0);
            field_dragged_ = false;
            field->Focus();
            field_step_direction_ = target->GetAttribute<int>("data-step-direction", 0);
            field_step_multiplier_ = event.GetParameter<int>("shift_key", 0) != 0 ? 10.0 : event.GetParameter<int>("alt_key", 0) != 0 ? 0.1
                                                                                                                                      : 1.0;
            if (field_step_direction_ != 0) {
                stepField();
                field_repeat_at_ = std::chrono::steady_clock::now() + std::chrono::milliseconds(400);
                repeatFieldStep();
            }
            event.StopPropagation();
            return true;
        }
        if (active_field_ && !editing_colour && !active_field_->IsClassSet("is-editing")) {
            if (type == "mousemove" || type == "drag") {
                if (field_step_direction_ != 0) {
                    event.StopPropagation();
                    return true;
                }
                const float delta = (localPointer(event).x - field_start_x_) / dp_ratio_;
                if (std::abs(delta) > 2.0f)
                    field_dragged_ = true;
                if (field_dragged_) {
                    const double step = active_field_->GetAttribute<double>("data-step", 0.01);
                    const double precision = event.GetParameter<int>("shift_key", 0) != 0 ? 0.1 : 1.0;
                    updateField(*active_field_, field_start_value_ + delta * step * precision);
                    active_field_->SetClass("is-dragging", true);
                    if (auto* input = active_field_->QuerySelector("input"))
                        commitControl(input, nullptr, false);
                }
                event.StopPropagation();
                return true;
            }
            if (type == "mouseup" || type == "dragend") {
                finishFieldEdit(!field_dragged_ && field_step_direction_ == 0);
                event.StopPropagation();
                return true;
            }
        }
        if (type == "keydown" && active_field_) {
            const int key = event.GetParameter("key_identifier", 0);
            if (key == Rml::Input::KI_ESCAPE || key == Rml::Input::KI_RETURN || key == Rml::Input::KI_NUMPADENTER) {
                finishFieldEdit(key == Rml::Input::KI_ESCAPE);
                event.StopPropagation();
                return true;
            }
        }
        if (type == "blur" && active_field_ && field == active_field_ && active_field_->IsClassSet("is-editing"))
            finishFieldEdit(false);
        if (type == "change" && field) {
            event.StopPropagation();
            return true;
        }
        if ((field || control || sidebar || popup) &&
            (type == "mousedown" || type == "mousemove" || type == "mouseup" ||
             type == "drag" || type == "dragstart" || type == "dragend" || type == "keydown")) {
            event.StopPropagation();
            return true;
        }
        return (sidebar || popup) && type == "mousescroll";
    }

    void NodeCanvasElement::openModifierLibrary(const float x, const float y) {
        if (!context_menu_ || !manager_)
            return;
        std::vector<ContextMenuItem> items{{.label = LOC("node_editor.new_graph"), .action = "new"}};
        for (const auto* tree : manager_->trees())
            items.push_back({.label = tree->name, .action = tree->uuid, .separator_before = items.size() == 1});
        context_menu_->request(std::move(items), panel_screen_offset_.x + x, panel_screen_offset_.y + y, [this](const std::string_view action) {
            if (action == "new") {
                addModifier();
            } else if (const auto host = activeHost(); host && manager_->tree(action)) {
                auto& modifier = manager_->addModifier(*host, std::string(action));
                active_modifier_uuid_ = modifier.uuid;
                active_tree_uuid_ = modifier.tree_uuid;
                dom_dirty_ = true;
            }
        });
    }

    void NodeCanvasElement::openModifierMenu(const std::string_view uuid, const float x, const float y) {
        if (!context_menu_)
            return;
        const auto host = activeHost();
        if (!host)
            return;
        std::vector<ContextMenuItem> items{
            {.label = LOC("node_editor.move_up"), .action = "up"},
            {.label = LOC("node_editor.move_down"), .action = "down"},
            {.label = LOC("node_editor.apply"), .action = "apply", .separator_before = true},
            {.label = LOC("node_editor.remove"), .action = "remove"}};
        context_menu_->request(std::move(items), panel_screen_offset_.x + x, panel_screen_offset_.y + y, [this, host = *host, uuid = std::string(uuid)](const std::string_view action) {
            auto* stack = manager_->stack(host);
            if (!stack)
                return;
            const auto found = std::ranges::find(stack->modifiers, uuid, &Modifier::uuid);
            if (found == stack->modifiers.end())
                return;
            const std::string name = found->name;
            const size_t index = static_cast<size_t>(std::distance(stack->modifiers.begin(), found));
            if (action == "remove")
                manager_->removeModifier(host, name);
            else if (action == "up" && index > 0)
                manager_->moveModifier(host, name, index - 1);
            else if (action == "down" && index + 1 < stack->modifiers.size())
                manager_->moveModifier(host, name, index + 1);
            else if (action == "apply")
                (void)manager_->applyModifier(host, name);
            dom_dirty_ = true;
        });
    }
    void NodeCanvasElement::commitControl(Rml::Element* target, const Rml::Event* event,
                                          const bool record_undo) {
        const std::string node_name = target->GetAttribute<Rml::String>("data-node", "");
        const std::string input_id = target->GetAttribute<Rml::String>("data-input", "");
        const std::string property_id =
            target->GetAttribute<Rml::String>("data-property", "");
        auto* tree = activeTree();
        auto* node = tree ? tree->find_node(node_name) : nullptr;
        const auto descriptor = node ? manager_->registry().find(node->type_id) : nullptr;
        if (tree && node && descriptor && (!input_id.empty() || !property_id.empty())) {
            const auto before = tree->to_json();
            const auto* input = dynamic_cast<Rml::ElementFormControlInput*>(target);
            const auto* select = dynamic_cast<Rml::ElementFormControlSelect*>(target);
            const auto* textarea = dynamic_cast<Rml::ElementFormControlTextArea*>(target);
            const std::string text = input ? input->GetValue()
                                     : select ? select->GetValue()
                                     : textarea ? textarea->GetValue()
                                                : "";
            char* parse_end = nullptr;
            if (!input_id.empty()) {
                const lfs::nodes::TreeResolver resolver = [this](const std::string_view uuid) {
                    return manager_->tree(uuid);
                };
                const auto inputs = lfs::nodes::effective_inputs(*tree, *node, resolver);
                const auto socket = std::ranges::find(inputs, input_id,
                                                      &lfs::nodes::SocketDecl::identifier);
                if (socket != inputs.end()) {
                    if ((target->GetTagName() == "color-picker" || target->GetTagName() == "colour-offset") && event) {
                        const bool signed_picker = target->GetTagName() == "color-picker" && target->GetAttribute<int>("data-offset", 0);
                        const auto channel = [&](const char* name) {
                            const float value = event->GetParameter(name, 0.0f);
                            return signed_picker ? value * 2.0f - 1.0f : value;
                        };
                        node->input_values[input_id] = glm::vec4(
                            channel("red"), channel("green"), channel("blue"), 1.0f);
                    } else if (target->HasAttribute("data-component")) {
                        const size_t component = static_cast<size_t>(
                            target->GetAttribute<int>("data-component", 0));
                        const float parsed = std::strtof(text.c_str(), &parse_end);
                        const auto value = node_widgets::valuePayload(*node, input_id, socket->default_value);
                        glm::vec4 vector(0.0f, 0.0f, 0.0f, 1.0f);
                        for (size_t index = 0; index < std::min<size_t>(4, value.size()); ++index)
                            vector[index] = value[index].get<float>();
                        if (parse_end != text.c_str() && component < 4)
                            vector[component] = parsed;
                        node->input_values[input_id] = socket->type == lfs::nodes::COLOUR_SOCKET
                                                           ? lfs::nodes::Value(vector)
                                                           : lfs::nodes::Value(glm::vec3(vector));
                    } else if (socket->type == lfs::nodes::BOOL_SOCKET) {
                        node->input_values[input_id] = target->HasAttribute("checked");
                    } else if (socket->type == lfs::nodes::INT_SOCKET) {
                        const long long parsed = std::strtoll(text.c_str(), &parse_end, 10);
                        if (parse_end != text.c_str())
                            node->input_values[input_id] = static_cast<std::int64_t>(parsed);
                    } else if (socket->type == lfs::nodes::FLOAT_SOCKET) {
                        float parsed = std::strtof(text.c_str(), &parse_end);
                        if (socket->min)
                            parsed = std::max(parsed, static_cast<float>(*socket->min));
                        if (socket->max)
                            parsed = std::min(parsed, static_cast<float>(*socket->max));
                        if (parse_end != text.c_str())
                            node->input_values[input_id] = parsed;
                    } else {
                        node->input_values[input_id] = text;
                    }
                }
            } else {
                const auto property = std::ranges::find(descriptor->properties, property_id,
                                                        &lfs::nodes::PropertyDecl::identifier);
                if (property != descriptor->properties.end()) {
                    if (property->kind == lfs::nodes::PropertyKind::Bool)
                        node->properties[property_id] = target->HasAttribute("checked");
                    else if (property->kind == lfs::nodes::PropertyKind::Int) {
                        const long long parsed = std::strtoll(text.c_str(), &parse_end, 10);
                        if (parse_end != text.c_str())
                            node->properties[property_id] = parsed;
                    } else if (property->kind == lfs::nodes::PropertyKind::Float) {
                        const double parsed = std::strtod(text.c_str(), &parse_end);
                        if (parse_end != text.c_str())
                            node->properties[property_id] = parsed;
                    } else {
                        node->properties[property_id] = text;
                    }
                }
            }
            if (tree->to_json() != before) {
                if (record_undo)
                    manager_->recordTreeEdit(tree->uuid, before);
                else
                    manager_->markDirty();
                dom_dirty_ = true;
            }
        }
    }
} // namespace lfs::vis::gui

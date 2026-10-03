/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/rmlui/elements/node_canvas_widgets.hpp"
#include "core/event_bridge/localization_manager.hpp"
#include "gui/rmlui/rml_theme.hpp"
#include "internal/resource_paths.hpp"

#include <RmlUi/Core/Element.h>

#include <algorithm>
#include <format>
#include <ranges>

namespace lfs::vis::gui::node_widgets {
    std::string categoryIcon(const std::string_view category) {
        const auto icon = category == "Input"        ? "scene/group"
                          : category == "Selection"  ? "selection"
                          : category == "Colour"     ? "color-picker"
                          : category == "Splat"      ? "scene/splat"
                          : category == "Geometry"   ? "scene/mesh"
                          : category == "Clean-up"   ? "brush"
                          : category == "Conversion" ? "puzzle"
                          : category == "Output"     ? "viewport-export"
                                                     : "settings";
        const auto path = rml_theme::pathToRmlImageSource(getAssetPath("icon/" + std::string(icon) + ".png"));
        return "<img class=\"node-category-icon\" src=\"" + escape(path) + "\"/>";
    }

    void layoutCard(Rml::Element& card, const float zoom, const float dp_ratio) {
        const auto px = [](const float value) { return std::format("{}px", value); };
        const float scale = zoom * dp_ratio;
        const auto apply = [&](const char* selector, const char* property, const float dp) {
            Rml::ElementList elements;
            card.QuerySelectorAll(elements, selector);
            for (auto* element : elements)
                element->SetProperty(property, px(dp * scale));
        };
        // Layout at the displayed size so text and native card decorations are
        // rasterized at their actual density, without scaling cached textures.
        const float minimum_font = 11.0f * std::min(zoom / 0.75f, 1.0f);
        card.SetProperty("font-size", px(std::max(12.0f * zoom, minimum_font) * dp_ratio));
        card.SetProperty("border-radius", px(6.0f * scale));
        card.SetProperty("border-width", px(std::max(zoom, 0.75f) * dp_ratio));
        card.SetClass("lod-values", zoom < 0.75f);
        // Only value editors disappear with LOD. Below editing zoom, body text
        // scales to fit existing rows instead of hiding or enlarging the cards.
        if (auto* title = card.QuerySelector(".node-title")) {
            title->SetProperty("font-size", px(std::max(13.0f * zoom, 11.0f) * dp_ratio));
            const auto height = px(std::max(26.0f * zoom, 18.0f) * dp_ratio);
            title->SetProperty("height", height);
            title->SetProperty("min-height", height);
            title->SetProperty("padding", "0 " + px(10.0f * scale));
        }
        if (auto* rows = card.QuerySelector(".socket-rows"))
            rows->SetProperty("padding", px(6.0f * scale) + " " + px(10.0f * scale));
        apply(".node-footer", "height", 20.0f);
        apply(".node-category-icon", "width", 14.0f);
        apply(".node-category-icon", "height", 14.0f);
        apply(".node-footer", "line-height", 20.0f);
        apply(".node-footer", "padding-left", 10.0f);
        Rml::ElementList captions;
        card.QuerySelectorAll(captions, ".node-footer, .node-settings");
        for (auto* caption : captions)
            caption->SetProperty("font-size", px(std::max(11.0f * zoom, minimum_font) * dp_ratio));
        apply(".socket-row, .socket-input, .node-bool, .node-colour", "height", 22.0f);
        apply(".socket-row, .socket-input", "min-height", 22.0f);
        apply(".node-settings", "height", 22.0f);
        apply(".node-settings", "min-height", 22.0f);
        apply(".node-settings", "line-height", 22.0f);
        apply(".socket-input > .socket-label", "line-height", 22.0f);
        Rml::ElementList inputs;
        card.QuerySelectorAll(inputs, ".socket-input");
        for (auto* input : inputs)
            input->SetProperty("height", px(input->GetAttribute<int>("data-rows", 1) * 22.0f * scale));
        apply(".node-scrub, select, .node-string", "height", 20.0f);
        apply(".node-scrub, select", "min-height", 20.0f);
        apply(".node-scrub", "margin-top", 1.0f);
        apply(".node-scrub", "margin-bottom", 1.0f);
        apply(".node-scrub", "padding-left", 6.0f);
        apply(".node-scrub", "padding-right", 6.0f);
        apply(".node-scrub, .node-swatch", "border-radius", 4.0f);
        apply(".node-swatch", "width", 62.0f);
        apply(".node-swatch", "height", 16.0f);
        apply(".node-swatch", "min-height", 16.0f);
        apply(".node-step", "width", 14.0f);
        apply(".node-step", "min-width", 14.0f);
        apply(".node-step", "height", 16.0f);
        apply(".node-step", "line-height", 16.0f);
        apply("input[type=checkbox]", "width", 14.0f);
        apply("input[type=checkbox]", "height", 14.0f);
    }

    std::string escape(const std::string_view text) {
        std::string result;
        result.reserve(text.size());
        for (const char ch : text) {
            switch (ch) {
            case '&': result += "&amp;"; break;
            case '<': result += "&lt;"; break;
            case '>': result += "&gt;"; break;
            case '"': result += "&quot;"; break;
            default: result += ch; break;
            }
        }
        return result;
    }

    std::string nonBreakingStatus(const std::string_view text) {
        std::string result;
        std::size_t start = 0;
        while (start < text.size()) {
            const auto separator = text.find(" · ", start);
            const auto end = separator == std::string_view::npos ? text.size() : separator;
            for (const char ch : text.substr(start, end - start)) {
                if (ch == ' ')
                    result += "\xc2\xa0";
                else
                    result += ch;
            }
            if (separator == std::string_view::npos)
                break;
            result += " · ";
            start = separator + std::string_view(" · ").size();
        }
        return result;
    }

    nlohmann::json valuePayload(const lfs::nodes::Node& node, const std::string_view identifier,
                                const lfs::nodes::Value& fallback) {
        const auto found = node.input_values.find(std::string(identifier));
        const nlohmann::json encoded = found == node.input_values.end()
                                           ? nlohmann::json(fallback)
                                           : nlohmann::json(found->second);
        return encoded.value("value", nlohmann::json{});
    }

    bool linked(const lfs::nodes::NodeTree& tree, const lfs::nodes::Node& node,
                const lfs::nodes::SocketDecl& socket) {
        return std::ranges::any_of(tree.links, [&](const auto& link) {
            return link.to_node == node.name && link.to_socket == socket.identifier;
        });
    }

    bool settingsExpanded(const lfs::nodes::Node& node) {
        return node.ui.value("settings_expanded", false);
    }

    bool singleValue(const lfs::nodes::SocketDecl& socket) {
        return !socket.field && socket.type != lfs::nodes::GEOMETRY_SOCKET &&
               socket.identifier != "Selection";
    }

    bool inputVisible(const lfs::nodes::NodeTree& tree, const lfs::nodes::Node& node,
                      const lfs::nodes::SocketDecl& socket) {
        return !singleValue(socket) || settingsExpanded(node) || linked(tree, node, socket);
    }

    int inputRows(const lfs::nodes::NodeTree& tree, const lfs::nodes::Node& node,
                  const lfs::nodes::SocketDecl& socket) {
        return socket.type == lfs::nodes::VECTOR_SOCKET && !socket.hide_value &&
                       !linked(tree, node, socket)
                   ? 3
                   : 1;
    }

    namespace {
        std::string number(const double value, const bool integer) {
            return integer ? std::format("{:.0f}", value) : std::format("{:.3g}", value);
        }

        std::string numericField(const std::string_view label, const double value,
                                 const bool integer, const std::optional<double> minimum,
                                 const std::optional<double> maximum, const double step,
                                 const std::string& attributes,
                                 const std::optional<double> soft_min = {},
                                 const std::optional<double> soft_max = {},
                                 const bool optional_selection = false) {
            const auto lower = minimum ? minimum : soft_min;
            const auto upper = maximum ? maximum : soft_max;
            const double fill = lower && upper && *upper > *lower
                                    ? std::clamp((value - *lower) / (*upper - *lower), 0.0, 1.0) * 100.0
                                    : 0.0;
            std::string limits;
            if (minimum)
                limits += std::format(" data-min=\"{}\"", *minimum);
            if (maximum)
                limits += std::format(" data-max=\"{}\"", *maximum);
            if (lower && upper)
                limits += std::format(" data-fill-min=\"{}\" data-fill-max=\"{}\"", *lower, *upper);
            return "<div class=\"scrub-field node-scrub\" drag=\"drag\" data-value=\"" +
                   std::format("{}", value) + "\" data-step=\"" + std::format("{}", step) +
                   "\" data-integer=\"" + (integer ? "1" : "0") + "\"" + limits +
                   "><div class=\"scrub-field-fill\" style=\"width:" + std::format("{:.1f}%", fill) +
                   ";\"></div><button class=\"node-step\" data-step-direction=\"-1\">−</button><span class=\"scrub-field-label\">" + escape(label) +
                   (optional_selection ? "<span class=\"socket-optional\"> ?</span>" : "") +
                   "</span><span class=\"scrub-field-display\">" + number(value, integer) +
                   "</span><input type=\"text\" class=\"scrub-field-input\" value=\"" +
                   number(value, integer) + "\"" + attributes + "/><button class=\"node-step\" data-step-direction=\"1\">+</button></div>";
        }
    } // namespace

    std::string input(const lfs::nodes::Node& node, const lfs::nodes::SocketDecl& socket,
                      const bool inline_value) {
        const auto value = valuePayload(node, socket.identifier, socket.default_value);
        const std::string attributes = " data-node=\"" + escape(node.name) + "\" data-input=\"" +
                                       escape(socket.identifier) + "\"";
        const std::string label = inline_value ? socket.label : "";
        if (socket.type == lfs::nodes::FLOAT_SOCKET || socket.type == lfs::nodes::INT_SOCKET) {
            const bool integer = socket.type == lfs::nodes::INT_SOCKET;
            return numericField(label, value.is_number() ? value.get<double>() : 0.0, integer,
                                socket.min, socket.max, socket.step.value_or(integer ? 1.0 : 0.01),
                                attributes, socket.soft_min, socket.soft_max,
                                inline_value && socket.identifier == "Selection");
        }
        if (socket.type == lfs::nodes::BOOL_SOCKET)
            return "<label class=\"node-bool\"><input type=\"checkbox\"" + attributes +
                   (value.is_boolean() && value.get<bool>() ? " checked" : "") +
                   "/><span>" + escape(label) + "</span></label>";
        if (socket.type == lfs::nodes::VECTOR_SOCKET) {
            std::string html = "<div class=\"node-vector\">";
            constexpr std::string_view axes[] = {"X", "Y", "Z"};
            for (size_t index = 0; index < 3; ++index) {
                const double channel = value.is_array() && value.size() > index
                                           ? value[index].get<double>()
                                           : 0.0;
                const std::string axis_label = (index == 0 && inline_value ? label + " · " : "") +
                                               std::string(axes[index]);
                html += numericField(axis_label, channel, false, socket.min, socket.max,
                                     socket.step.value_or(0.01), attributes + std::format(" data-component=\"{}\"", index),
                                     socket.soft_min, socket.soft_max);
            }
            return html + "</div>";
        }
        if (socket.type == lfs::nodes::COLOUR_SOCKET) {
            const auto channel = [&](const size_t index) {
                return value.is_array() && value.size() > index ? value[index].get<float>() : 0.0f;
            };
            const bool offset = socket.min == -1.0 && socket.max == 1.0;
            const auto byte = [offset](const float v) { return static_cast<int>(std::clamp(offset ? (v + 1.0f) * 0.5f : v, 0.0f, 1.0f) * 255.0f); };
            std::string html = "<div class=\"node-colour\"><span>" + escape(label) +
                               "</span><button class=\"color-swatch node-swatch\" data-action=\"colour-popup\"" +
                               attributes + std::format(" data-offset=\"{}\" data-red=\"{}\" data-green=\"{}\" data-blue=\"{}\" style=\"background-color:rgb({},{},{});\"></button></div>", offset ? 1 : 0, channel(0), channel(1), channel(2), byte(channel(0)), byte(channel(1)), byte(channel(2)));
            if (!inline_value && offset)
                html += std::format("<colour-offset{} red=\"{}\" green=\"{}\" blue=\"{}\" title=\"{}\"/>",
                                    attributes, channel(0), channel(1), channel(2), escape(LOC("node_editor.colour_offset_hint")));
            return html;
        }
        const auto text = value.is_string() ? value.get<std::string>() : std::string{};
        return "<input type=\"text\" class=\"node-string\" value=\"" + escape(text) + "\"" +
               attributes + "/>";
    }

    std::string property(const lfs::nodes::Node& node, const lfs::nodes::PropertyDecl& property) {
        const auto value = node.properties.value(property.identifier, property.default_value);
        const std::string attributes = " data-node=\"" + escape(node.name) + "\" data-property=\"" +
                                       escape(property.identifier) + "\"";
        using lfs::nodes::PropertyKind;
        if (property.kind == PropertyKind::Enum) {
            std::string html = "<select" + attributes + ">";
            for (const auto& item : property.items)
                html += "<option value=\"" + escape(item) + "\"" +
                        (value == item ? " selected" : "") + ">" + escape(item) + "</option>";
            return html + "</select>";
        }
        if (property.kind == PropertyKind::Bool)
            return "<input type=\"checkbox\"" + attributes +
                   (value.is_boolean() && value.get<bool>() ? " checked" : "") + "/>";
        if (property.kind == PropertyKind::Int || property.kind == PropertyKind::Float)
            return numericField("", value.is_number() ? value.get<double>() : 0.0,
                                property.kind == PropertyKind::Int, property.min, property.max,
                                property.kind == PropertyKind::Int ? 1.0 : 0.01, attributes);
        const auto text = value.is_string() ? value.get<std::string>() : std::string{};
        return "<input type=\"text\" value=\"" + escape(text) + "\"" + attributes + "/>";
    }
} // namespace lfs::vis::gui::node_widgets

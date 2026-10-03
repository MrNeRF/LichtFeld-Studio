/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "builtin_text.hpp"
#include "core/nodes/builtin.hpp"

namespace lfs::nodes {
    void set_builtin_node_text(NodeTypeInfo& info) {
        static const auto catalogue = nlohmann::json::parse(kBuiltinNodeText);
        const auto id = info.id.substr(4);
        const auto& text = catalogue.at(id);
        info.localization_key = "nodes." + id;
        info.label = text.at("label");
        info.description = text.at("description");
        info.help = text.at("help");
        for (auto& input : info.inputs) {
            input.label = text.at("input_labels").at(input.identifier);
            input.description = text.at("inputs").at(input.identifier);
        }
        for (auto& output : info.outputs) {
            output.label = text.at("output_labels").at(output.identifier);
            output.description = text.at("outputs").at(output.identifier);
        }
        for (auto& property : info.properties) {
            property.label = text.at("property_labels").at(property.identifier);
            property.description = text.at("properties").at(property.identifier);
        }
    }
} // namespace lfs::nodes

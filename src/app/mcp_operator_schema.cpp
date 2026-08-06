/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "app/mcp_operator_schema.hpp"

#include "core/property_registry.hpp"
#include "visualizer/operator/operator_properties.hpp"

#include <glm/glm.hpp>
#include <optional>

namespace lfs::app {

    namespace {

        std::optional<int> fixed_vector_size(const core::prop::PropertyMeta& property) {
            using core::prop::PropType;
            switch (property.type) {
            case PropType::Vec2:
                return 2;
            case PropType::Vec3:
            case PropType::Color3:
                return 3;
            case PropType::Vec4:
            case PropType::Quat:
            case PropType::Color4:
                return 4;
            case PropType::Mat4:
                return 16;
            case PropType::FloatVector:
            case PropType::IntVector:
                return property.vector_size;
            default:
                return std::nullopt;
            }
        }

    } // namespace

    json property_meta_to_json(const core::prop::PropertyMeta& property) {
        using core::prop::PropType;

        json schema;
        switch (property.type) {
        case PropType::Bool:
            schema["type"] = "boolean";
            break;
        case PropType::Int:
        case PropType::SizeT:
            schema["type"] = "integer";
            break;
        case PropType::Float:
            schema["type"] = "number";
            break;
        case PropType::String:
        case PropType::Enum:
            schema["type"] = "string";
            break;
        case PropType::Vec2:
        case PropType::Vec3:
        case PropType::Vec4:
        case PropType::Quat:
        case PropType::Mat4:
        case PropType::Color3:
        case PropType::Color4:
        case PropType::FloatVector:
            schema["type"] = "array";
            schema["items"] = json{{"type", "number"}};
            if (const auto size = fixed_vector_size(property)) {
                schema["minItems"] = *size;
                schema["maxItems"] = *size;
            }
            break;
        case PropType::IntVector:
            schema["type"] = "array";
            schema["items"] = json{{"type", "integer"}};
            if (const auto size = fixed_vector_size(property)) {
                schema["minItems"] = *size;
                schema["maxItems"] = *size;
            }
            break;
        case PropType::Tensor:
            schema["type"] = "array";
            break;
        }

        if (!property.description.empty()) {
            schema["description"] = property.description;
        }
        if (property.min_value) {
            schema["minimum"] = *property.min_value;
        }
        if (property.max_value) {
            schema["maximum"] = *property.max_value;
        }
        if (!property.enum_items.empty()) {
            json values = json::array();
            for (const auto& item : property.enum_items) {
                values.push_back(item.identifier);
            }
            schema["enum"] = std::move(values);
        }

        return schema;
    }

    mcp::McpToolInputSchema build_operator_input_schema(
        const std::string& operator_key,
        const std::vector<std::string>& required) {
        mcp::McpToolInputSchema schema;
        schema.type = "object";
        schema.properties = json::object();
        schema.required = required;

        const auto group = core::prop::PropertyRegistry::instance().get_group_snapshot(
            "operator." + operator_key);
        if (group) {
            for (const auto& property : group->properties) {
                if (property.has_flag(core::prop::PROP_OPERATOR_ARG)) {
                    schema.properties[property.id] = property_meta_to_json(property);
                }
            }
        }
        return schema;
    }

    std::expected<void, std::string> assign_operator_property_from_json(
        const json& args,
        const core::prop::PropertyMeta& property,
        vis::op::OperatorProperties& props) {
        using core::prop::PropType;

        if (!args.contains(property.id) || args[property.id].is_null()) {
            return {};
        }

        const auto& value = args[property.id];
        switch (property.type) {
        case PropType::Bool:
            if (!value.is_boolean()) {
                return std::unexpected("Field '" + property.id + "' must be a boolean");
            }
            props.set(property.id, value.get<bool>());
            return {};
        case PropType::Int:
        case PropType::SizeT:
            if (!value.is_number_integer()) {
                return std::unexpected("Field '" + property.id + "' must be an integer");
            }
            props.set(property.id, value.get<int>());
            return {};
        case PropType::Float:
            if (!value.is_number()) {
                return std::unexpected("Field '" + property.id + "' must be a number");
            }
            props.set(property.id, value.get<float>());
            return {};
        case PropType::String:
        case PropType::Enum:
            if (!value.is_string()) {
                return std::unexpected("Field '" + property.id + "' must be a string");
            }
            props.set(property.id, value.get<std::string>());
            return {};
        case PropType::Vec2:
        case PropType::Vec3:
        case PropType::Vec4:
        case PropType::Quat:
        case PropType::Mat4:
        case PropType::Color3:
        case PropType::Color4:
        case PropType::FloatVector: {
            if (!value.is_array()) {
                return std::unexpected("Field '" + property.id + "' must be an array");
            }
            if (const auto size = fixed_vector_size(property);
                size && value.size() != static_cast<size_t>(*size)) {
                return std::unexpected(
                    "Field '" + property.id + "' must have exactly " + std::to_string(*size) +
                    " entries");
            }

            std::vector<float> values;
            values.reserve(value.size());
            for (const auto& item : value) {
                if (!item.is_number()) {
                    return std::unexpected("Field '" + property.id + "' must contain only numbers");
                }
                values.push_back(item.get<float>());
            }

            if ((property.type == PropType::FloatVector && property.vector_size == 3) ||
                property.type == PropType::Vec3 || property.type == PropType::Color3) {
                props.set(property.id, glm::vec3(values[0], values[1], values[2]));
            } else {
                props.set(property.id, std::move(values));
            }
            return {};
        }
        case PropType::IntVector: {
            if (!value.is_array()) {
                return std::unexpected("Field '" + property.id + "' must be an array");
            }
            if (const auto size = fixed_vector_size(property);
                size && value.size() != static_cast<size_t>(*size)) {
                return std::unexpected(
                    "Field '" + property.id + "' must have exactly " + std::to_string(*size) +
                    " entries");
            }

            std::vector<int> values;
            values.reserve(value.size());
            for (const auto& item : value) {
                if (!item.is_number_integer()) {
                    return std::unexpected("Field '" + property.id + "' must contain only integers");
                }
                values.push_back(item.get<int>());
            }
            props.set(property.id, std::move(values));
            return {};
        }
        case PropType::Tensor:
            return std::unexpected("Field '" + property.id + "' is not supported through MCP yet");
        }

        return std::unexpected("Field '" + property.id + "' has an unsupported schema type");
    }

    std::expected<void, std::string> populate_operator_props(
        const json& args,
        const std::string& operator_key,
        const std::vector<std::string>& required,
        vis::op::OperatorProperties& props) {
        if (!args.is_object()) {
            return std::unexpected("Tool arguments must be a JSON object");
        }

        for (const auto& field : required) {
            if (!args.contains(field) || args[field].is_null()) {
                return std::unexpected("Field '" + field + "' must be provided");
            }
        }

        const auto group = core::prop::PropertyRegistry::instance().get_group_snapshot(
            "operator." + operator_key);
        if (group) {
            for (const auto& property : group->properties) {
                if (!property.has_flag(core::prop::PROP_OPERATOR_ARG)) {
                    continue;
                }
                if (auto result = assign_operator_property_from_json(args, property, props); !result) {
                    return result;
                }
            }
        }
        return {};
    }

} // namespace lfs::app

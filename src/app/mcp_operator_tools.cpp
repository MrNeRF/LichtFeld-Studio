/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "app/mcp_operator_tools.hpp"

#include "core/logger.hpp"
#include "visualizer/operator/operator_flags.hpp"
#include "visualizer/operator/operator_properties.hpp"
#include "visualizer/operator/operator_registry.hpp"
#include "visualizer/operator/operator_result.hpp"
#include "visualizer/operator/property_schema.hpp"
#include "visualizer/visualizer.hpp"

#include <algorithm>
#include <atomic>
#include <future>
#include <glm/glm.hpp>
#include <type_traits>

namespace lfs::app {

    namespace {

        template <typename T>
        struct dependent_false : std::false_type {};

        template <typename R>
        R make_post_failure(const std::string& error) {
            if constexpr (std::is_same_v<R, json>) {
                return json{{"error", error}};
            } else {
                static_assert(dependent_false<R>::value, "Unsupported post_and_wait return type");
            }
        }

        template <typename F>
        auto post_and_wait(vis::Visualizer* viewer, F&& fn) {
            using R = std::invoke_result_t<F>;
            constexpr const char* shutdown_error = "Viewer is shutting down";

            auto task = std::make_shared<std::decay_t<F>>(std::forward<F>(fn));
            auto promise = std::make_shared<std::promise<R>>();
            auto completed = std::make_shared<std::atomic_bool>(false);
            auto future = promise->get_future();

            auto finish_with_value = [promise, completed](auto&& value) mutable {
                if (!completed->exchange(true)) {
                    promise->set_value(std::forward<decltype(value)>(value));
                }
            };
            auto finish_with_exception = [promise, completed](std::exception_ptr error) {
                if (!completed->exchange(true)) {
                    promise->set_exception(std::move(error));
                }
            };

            const bool posted = viewer->postWork(vis::Visualizer::WorkItem{
                .run =
                    [task, finish_with_value, finish_with_exception]() mutable {
                        try {
                            finish_with_value(std::invoke(*task));
                        } catch (...) {
                            finish_with_exception(std::current_exception());
                        }
                    },
                .cancel =
                    [finish_with_value]() mutable {
                        finish_with_value(make_post_failure<R>(shutdown_error));
                    }});

            if (!posted) {
                return make_post_failure<R>(shutdown_error);
            }

            return future.get();
        }

        json property_schema_to_json(const vis::op::PropertySchema& property) {
            json schema;

            switch (property.type) {
            case vis::op::PropertyType::BOOL:
                schema["type"] = "boolean";
                break;
            case vis::op::PropertyType::INT:
                schema["type"] = "integer";
                break;
            case vis::op::PropertyType::FLOAT:
                schema["type"] = "number";
                break;
            case vis::op::PropertyType::STRING:
            case vis::op::PropertyType::ENUM:
                schema["type"] = "string";
                break;
            case vis::op::PropertyType::FLOAT_VECTOR:
                schema["type"] = "array";
                schema["items"] = json{{"type", "number"}};
                if (property.size) {
                    schema["minItems"] = *property.size;
                    schema["maxItems"] = *property.size;
                }
                break;
            case vis::op::PropertyType::INT_VECTOR:
                schema["type"] = "array";
                schema["items"] = json{{"type", "integer"}};
                if (property.size) {
                    schema["minItems"] = *property.size;
                    schema["maxItems"] = *property.size;
                }
                break;
            case vis::op::PropertyType::TENSOR:
                schema["type"] = "array";
                break;
            }

            if (!property.description.empty()) {
                schema["description"] = property.description;
            }
            if (property.min) {
                schema["minimum"] = *property.min;
            }
            if (property.max) {
                schema["maximum"] = *property.max;
            }
            if (property.enum_items.size() > 0) {
                json values = json::array();
                for (const auto& [value, label, description] : property.enum_items) {
                    values.push_back(value);
                }
                schema["enum"] = std::move(values);
            }

            return schema;
        }

        mcp::McpToolInputSchema build_input_schema(const std::string& operator_key,
                                                   const std::vector<std::string>& required) {
            mcp::McpToolInputSchema schema;
            schema.type = "object";
            schema.properties = json::object();
            schema.required = required;

            if (const auto* properties = vis::op::propertySchemas().getSchema(operator_key)) {
                for (const auto& property : *properties) {
                    schema.properties[property.name] = property_schema_to_json(property);
                }
            }

            return schema;
        }

        std::expected<void, std::string> assign_property_from_json(
            const json& args,
            const vis::op::PropertySchema& schema,
            vis::op::OperatorProperties& props) {
            if (!args.contains(schema.name) || args[schema.name].is_null()) {
                return {};
            }

            const auto& value = args[schema.name];
            switch (schema.type) {
            case vis::op::PropertyType::BOOL:
                if (!value.is_boolean()) {
                    return std::unexpected("Field '" + schema.name + "' must be a boolean");
                }
                props.set(schema.name, value.get<bool>());
                return {};
            case vis::op::PropertyType::INT:
                if (!value.is_number_integer()) {
                    return std::unexpected("Field '" + schema.name + "' must be an integer");
                }
                props.set(schema.name, value.get<int>());
                return {};
            case vis::op::PropertyType::FLOAT:
                if (!value.is_number()) {
                    return std::unexpected("Field '" + schema.name + "' must be a number");
                }
                props.set(schema.name, value.get<float>());
                return {};
            case vis::op::PropertyType::STRING:
            case vis::op::PropertyType::ENUM:
                if (!value.is_string()) {
                    return std::unexpected("Field '" + schema.name + "' must be a string");
                }
                props.set(schema.name, value.get<std::string>());
                return {};
            case vis::op::PropertyType::FLOAT_VECTOR: {
                if (!value.is_array()) {
                    return std::unexpected("Field '" + schema.name + "' must be an array");
                }
                if (schema.size && value.size() != static_cast<size_t>(*schema.size)) {
                    return std::unexpected(
                        "Field '" + schema.name + "' must have exactly " + std::to_string(*schema.size) +
                        " entries");
                }

                std::vector<float> values;
                values.reserve(value.size());
                for (const auto& item : value) {
                    if (!item.is_number()) {
                        return std::unexpected("Field '" + schema.name + "' must contain only numbers");
                    }
                    values.push_back(item.get<float>());
                }

                if (schema.size && *schema.size == 3) {
                    props.set(schema.name, glm::vec3(values[0], values[1], values[2]));
                } else {
                    props.set(schema.name, std::move(values));
                }
                return {};
            }
            case vis::op::PropertyType::INT_VECTOR: {
                if (!value.is_array()) {
                    return std::unexpected("Field '" + schema.name + "' must be an array");
                }
                if (schema.size && value.size() != static_cast<size_t>(*schema.size)) {
                    return std::unexpected(
                        "Field '" + schema.name + "' must have exactly " + std::to_string(*schema.size) +
                        " entries");
                }

                std::vector<int> values;
                values.reserve(value.size());
                for (const auto& item : value) {
                    if (!item.is_number_integer()) {
                        return std::unexpected("Field '" + schema.name + "' must contain only integers");
                    }
                    values.push_back(item.get<int>());
                }
                props.set(schema.name, std::move(values));
                return {};
            }
            case vis::op::PropertyType::TENSOR:
                return std::unexpected("Field '" + schema.name + "' is not supported through MCP yet");
            }

            return std::unexpected("Field '" + schema.name + "' has an unsupported schema type");
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

            if (const auto* properties = vis::op::propertySchemas().getSchema(operator_key)) {
                for (const auto& property : *properties) {
                    if (auto result = assign_property_from_json(args, property, props); !result) {
                        return result;
                    }
                }
            }

            return {};
        }

        mcp::McpToolMetadata build_metadata(const GuiOperatorToolBinding& binding,
                                            const vis::op::OperatorDescriptor& descriptor) {
            return mcp::McpToolMetadata{
                .category = binding.category,
                .kind = "command",
                .runtime = "gui",
                .thread_affinity = "gui_thread",
                .destructive = binding.destructive,
                .long_running = hasFlag(descriptor.flags, vis::op::OperatorFlags::BLOCKING) ||
                                hasFlag(descriptor.flags, vis::op::OperatorFlags::MODAL),
                .user_visible = !hasFlag(descriptor.flags, vis::op::OperatorFlags::INTERNAL),
            };
        }

        std::string operator_cancel_message(const vis::op::OperatorDescriptor& descriptor) {
            if (!descriptor.label.empty()) {
                return descriptor.label + " could not be performed";
            }
            return "Operator was cancelled";
        }

    } // namespace

    void register_gui_operator_tool(mcp::ToolRegistry& registry,
                                    vis::Visualizer* viewer,
                                    GuiOperatorToolBinding binding) {
        const auto* descriptor = vis::op::operators().getDescriptor(binding.operator_id);
        if (!descriptor) {
            LOG_WARN("Cannot register GUI operator tool '{}' because operator '{}' is missing",
                     binding.tool_name, vis::op::to_string(binding.operator_id));
            return;
        }

        registry.register_tool(
            mcp::McpTool{
                .name = binding.tool_name,
                .description = binding.description.empty() ? descriptor->description : binding.description,
                .input_schema = build_input_schema(descriptor->id(), binding.required),
                .metadata = build_metadata(binding, *descriptor),
            },
            [viewer, binding = std::move(binding)](const json& args) -> json {
                return post_and_wait(viewer, [viewer, binding, args]() -> json {
                    auto* scene_manager = viewer->getSceneManager();
                    if (!scene_manager) {
                        return json{{"error", "Scene manager not initialized"}};
                    }

                    const auto* descriptor = vis::op::operators().getDescriptor(binding.operator_id);
                    if (!descriptor) {
                        return json{{"error", "Operator is not registered"}};
                    }

                    vis::op::OperatorProperties props;
                    if (auto result = populate_operator_props(args, descriptor->id(), binding.required, props);
                        !result) {
                        return json{{"error", result.error()}};
                    }

                    if (binding.prepare) {
                        if (auto result = binding.prepare(*viewer, args, props); !result) {
                            return json{{"error", result.error()}};
                        }
                    }

                    const auto result = vis::op::operators().invoke(binding.operator_id, &props);
                    if (!result.is_finished()) {
                        if (result.is_running_modal()) {
                            return json{{"success", true}, {"status", "running_modal"}};
                        }
                        return json{{"error", operator_cancel_message(*descriptor)}};
                    }

                    if (binding.on_success) {
                        return binding.on_success(*viewer, args, props, result);
                    }

                    return json{{"success", true}};
                });
            });
    }

} // namespace lfs::app

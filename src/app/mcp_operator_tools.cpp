/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "app/mcp_operator_tools.hpp"
#include "app/mcp_app_utils.hpp"
#include "app/mcp_operator_schema.hpp"

#include "core/event_bridge/localization_manager.hpp"
#include "core/logger.hpp"
#include "visualizer/operator/operator_flags.hpp"
#include "visualizer/operator/operator_properties.hpp"
#include "visualizer/operator/operator_registry.hpp"
#include "visualizer/operator/operator_result.hpp"
#include "visualizer/visualizer.hpp"

#include <algorithm>
#include <atomic>
#include <future>
#include <optional>
#include <type_traits>

namespace lfs::app {

    namespace {

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

        std::string operator_label(const vis::op::OperatorDescriptor& descriptor) {
            auto& localization = lfs::event::LocalizationManager::getInstance();
            return localization.hasKey(descriptor.label) ? localization.get(descriptor.label)
                                                         : descriptor.label;
        }

        std::string operator_cancel_message(const vis::op::OperatorDescriptor& descriptor) {
            return descriptor.label.empty()
                       ? LOC("runtime.operator_cancelled")
                       : lfs::event::formatLocalized("runtime.operator_could_not_be_performed",
                                                     operator_label(descriptor));
        }

        const char* operator_source_to_string(const vis::op::OperatorSource source) {
            switch (source) {
            case vis::op::OperatorSource::CPP:
                return "cpp";
            case vis::op::OperatorSource::PYTHON:
                return "python";
            }
            return "unknown";
        }

        const char* modal_state_to_string(const vis::op::ModalState state) {
            switch (state) {
            case vis::op::ModalState::IDLE:
                return "idle";
            case vis::op::ModalState::ACTIVE_CPP:
                return "active_cpp";
            case vis::op::ModalState::ACTIVE_PYTHON:
                return "active_python";
            }
            return "unknown";
        }

        json poll_dependencies_to_json(const vis::op::PollDependency deps) {
            json values = json::array();
            if ((deps & vis::op::PollDependency::SCENE) != vis::op::PollDependency::NONE) {
                values.push_back("scene");
            }
            if ((deps & vis::op::PollDependency::SELECTION) != vis::op::PollDependency::NONE) {
                values.push_back("selection");
            }
            if ((deps & vis::op::PollDependency::TRAINING) != vis::op::PollDependency::NONE) {
                values.push_back("training");
            }
            return values;
        }

        json operator_flags_to_json(const vis::op::OperatorFlags flags) {
            return json{
                {"register", hasFlag(flags, vis::op::OperatorFlags::REGISTER)},
                {"undo", hasFlag(flags, vis::op::OperatorFlags::UNDO)},
                {"undo_grouped", hasFlag(flags, vis::op::OperatorFlags::UNDO_GROUPED)},
                {"internal", hasFlag(flags, vis::op::OperatorFlags::INTERNAL)},
                {"modal", hasFlag(flags, vis::op::OperatorFlags::MODAL)},
                {"blocking", hasFlag(flags, vis::op::OperatorFlags::BLOCKING)},
            };
        }

        json input_schema_to_json(const mcp::McpToolInputSchema& schema) {
            json result{
                {"type", schema.type},
                {"properties", schema.properties},
            };
            if (!schema.required.empty()) {
                result["required"] = schema.required;
            }
            return result;
        }

        json operator_descriptor_json(const vis::op::OperatorDescriptor& descriptor,
                                      const bool include_schema,
                                      const std::optional<bool> poll_without_args) {
            json result{
                {"id", descriptor.id()},
                {"label", operator_label(descriptor)},
                {"description", descriptor.description},
                {"source", operator_source_to_string(descriptor.source)},
                {"builtin", descriptor.builtin_id.has_value()},
                {"flags", operator_flags_to_json(descriptor.flags)},
                {"poll_dependencies", poll_dependencies_to_json(descriptor.poll_deps)},
            };

            if (poll_without_args.has_value()) {
                result["poll_without_args"] = *poll_without_args;
            }
            if (include_schema) {
                result["input_schema"] = input_schema_to_json(build_operator_input_schema(descriptor.id(), {}));
            }

            return result;
        }

        std::vector<const vis::op::OperatorDescriptor*> sorted_operator_descriptors(const bool include_internal) {
            auto descriptors = vis::op::operators().getAllOperators();
            std::erase_if(descriptors, [include_internal](const auto* descriptor) {
                return descriptor == nullptr ||
                       (!include_internal &&
                        hasFlag(descriptor->flags, vis::op::OperatorFlags::INTERNAL));
            });

            std::sort(descriptors.begin(), descriptors.end(), [](const auto* lhs, const auto* rhs) {
                return lhs->id() < rhs->id();
            });
            return descriptors;
        }

        json modal_state_json() {
            auto& operators = vis::op::operators();
            json payload{
                {"has_modal_operator", operators.hasModalOperator()},
                {"state", modal_state_to_string(operators.modalState())},
            };
            const std::string active_modal_id = operators.activeModalId();
            if (!active_modal_id.empty()) {
                payload["operator_id"] = active_modal_id;
            }
            return payload;
        }

        const char* operator_result_to_string(const vis::op::OperatorResult result) {
            switch (result) {
            case vis::op::OperatorResult::FINISHED:
                return "finished";
            case vis::op::OperatorResult::CANCELLED:
                return "cancelled";
            case vis::op::OperatorResult::RUNNING_MODAL:
                return "running_modal";
            case vis::op::OperatorResult::PASS_THROUGH:
                return "pass_through";
            }
            return "unknown";
        }

        std::expected<vis::op::ModalEvent, std::string> modal_event_from_json(const json& args) {
            if (!args.contains("type") || !args["type"].is_string()) {
                return std::unexpected("Field 'type' must be provided");
            }

            const std::string type = args["type"].get<std::string>();
            vis::op::ModalEvent event;

            if (type == "mouse_button") {
                if (!args.contains("button") || !args["button"].is_number_integer()) {
                    return std::unexpected("Field 'button' must be provided for mouse_button events");
                }
                if (!args.contains("action") || !args["action"].is_number_integer()) {
                    return std::unexpected("Field 'action' must be provided for mouse_button events");
                }

                event.type = vis::op::ModalEvent::Type::MOUSE_BUTTON;
                event.data = vis::MouseButtonEvent{
                    .button = args["button"].get<int>(),
                    .action = args["action"].get<int>(),
                    .mods = args.value("mods", 0),
                    .position = {
                        args.value("x", 0.0),
                        args.value("y", 0.0),
                    },
                };
                return event;
            }

            if (type == "mouse_move") {
                event.type = vis::op::ModalEvent::Type::MOUSE_MOVE;
                event.data = vis::MouseMoveEvent{
                    .position = {
                        args.value("x", 0.0),
                        args.value("y", 0.0),
                    },
                    .delta = {
                        args.value("dx", 0.0),
                        args.value("dy", 0.0),
                    },
                };
                return event;
            }

            if (type == "mouse_scroll") {
                event.type = vis::op::ModalEvent::Type::MOUSE_SCROLL;
                event.data = vis::MouseScrollEvent{
                    .xoffset = args.value("scroll_x", 0.0),
                    .yoffset = args.value("scroll_y", 0.0),
                };
                return event;
            }

            if (type == "key") {
                if (!args.contains("key") || !args["key"].is_number_integer()) {
                    return std::unexpected("Field 'key' must be provided for key events");
                }
                if (!args.contains("action") || !args["action"].is_number_integer()) {
                    return std::unexpected("Field 'action' must be provided for key events");
                }

                event.type = vis::op::ModalEvent::Type::KEY;
                event.data = vis::KeyEvent{
                    .key = args["key"].get<int>(),
                    .scancode = args.value("scancode", 0),
                    .action = args["action"].get<int>(),
                    .mods = args.value("mods", 0),
                };
                return event;
            }

            return std::unexpected(
                "Unsupported modal event type: " + type +
                " (expected one of mouse_button, mouse_move, mouse_scroll, key)");
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

        auto input_schema = build_operator_input_schema(descriptor->id(), binding.required);
        auto metadata = build_metadata(binding, *descriptor);
        std::string tool_name = binding.tool_name;
        std::string description =
            binding.description.empty() ? descriptor->description : binding.description;

        registry.register_tool(
            mcp::McpTool{
                .name = std::move(tool_name),
                .description = std::move(description),
                .input_schema = std::move(input_schema),
                .metadata = std::move(metadata),
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
                        if (const auto error = props.get<std::string>("error"); error && !error->empty()) {
                            return json{{"error", *error}};
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

    void register_generic_gui_operator_tools(mcp::ToolRegistry& registry,
                                             vis::Visualizer* viewer) {
        registry.register_tool(
            mcp::McpTool{
                .name = "operator.list",
                .description = "List registered GUI operators, including Python operators, schemas, and modal/poll metadata",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"include_internal", json{{"type", "boolean"}, {"description", "Include internal operators (default: false)"}}},
                        {"include_schema", json{{"type", "boolean"}, {"description", "Include the operator input schema derived from registered properties (default: true)"}}},
                        {"include_poll", json{{"type", "boolean"}, {"description", "Include poll results without explicit arguments (default: true)"}}}},
                    .required = {}},
                .metadata = mcp::McpToolMetadata{
                    .category = "operator",
                    .kind = "query",
                    .runtime = "gui",
                    .thread_affinity = "gui_thread",
                }},
            [viewer](const json& args) -> json {
                const bool include_internal = args.value("include_internal", false);
                const bool include_schema = args.value("include_schema", true);
                const bool include_poll = args.value("include_poll", true);

                return post_and_wait(viewer, [include_internal, include_schema, include_poll]() -> json {
                    json operators_json = json::array();
                    for (const auto* descriptor : sorted_operator_descriptors(include_internal)) {
                        const std::optional<bool> poll_without_args =
                            include_poll ? std::optional<bool>(vis::op::operators().poll(descriptor->id()))
                                         : std::nullopt;
                        operators_json.push_back(
                            operator_descriptor_json(*descriptor, include_schema, poll_without_args));
                    }

                    return json{
                        {"success", true},
                        {"count", static_cast<int64_t>(operators_json.size())},
                        {"operators", std::move(operators_json)},
                    };
                });
            });

        registry.register_tool(
            mcp::McpTool{
                .name = "operator.describe",
                .description = "Describe a registered GUI operator, including schema, flags, and current poll state",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"operator_id", json{{"type", "string"}, {"description", "Registered operator id, for example 'transform.translate' or a Python operator id"}}},
                        {"include_schema", json{{"type", "boolean"}, {"description", "Include the operator input schema (default: true)"}}},
                        {"include_poll", json{{"type", "boolean"}, {"description", "Include poll result without explicit arguments (default: true)"}}}},
                    .required = {"operator_id"}},
                .metadata = mcp::McpToolMetadata{
                    .category = "operator",
                    .kind = "query",
                    .runtime = "gui",
                    .thread_affinity = "gui_thread",
                }},
            [viewer](const json& args) -> json {
                const std::string operator_id = args["operator_id"].get<std::string>();
                const bool include_schema = args.value("include_schema", true);
                const bool include_poll = args.value("include_poll", true);

                return post_and_wait(viewer, [operator_id, include_schema, include_poll]() -> json {
                    const auto* descriptor = vis::op::operators().getDescriptor(operator_id);
                    if (!descriptor) {
                        return json{{"error", "Operator is not registered: " + operator_id}};
                    }

                    const std::optional<bool> poll_without_args =
                        include_poll ? std::optional<bool>(vis::op::operators().poll(operator_id))
                                     : std::nullopt;

                    return json{
                        {"success", true},
                        {"operator", operator_descriptor_json(*descriptor, include_schema, poll_without_args)},
                    };
                });
            });

        registry.register_tool(
            mcp::McpTool{
                .name = "operator.invoke",
                .description = "Invoke any registered GUI operator by id using its registered property schema",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"operator_id", json{{"type", "string"}, {"description", "Registered operator id, for example 'transform.translate' or a Python operator id"}}},
                        {"arguments", json{{"type", "object"}, {"description", "Operator properties keyed by schema field name"}}}},
                    .required = {"operator_id"}},
                .metadata = mcp::McpToolMetadata{
                    .category = "operator",
                    .kind = "command",
                    .runtime = "gui",
                    .thread_affinity = "gui_thread",
                    .destructive = true,
                    .long_running = true,
                }},
            [viewer](const json& args) -> json {
                const std::string operator_id = args["operator_id"].get<std::string>();
                const json operator_args = args.value("arguments", json::object());

                return post_and_wait(viewer, [viewer, operator_id, operator_args]() -> json {
                    auto* scene_manager = viewer->getSceneManager();
                    if (!scene_manager) {
                        return json{{"error", "Scene manager not initialized"}};
                    }

                    const auto* descriptor = vis::op::operators().getDescriptor(operator_id);
                    if (!descriptor) {
                        return json{{"error", "Operator is not registered: " + operator_id}};
                    }

                    vis::op::OperatorProperties props;
                    if (auto result = populate_operator_props(operator_args, descriptor->id(), {}, props);
                        !result) {
                        return json{{"error", result.error()}};
                    }

                    const auto invocation = vis::op::operators().invoke(operator_id, &props);
                    if (invocation.is_running_modal()) {
                        return json{
                            {"success", true},
                            {"operator_id", descriptor->id()},
                            {"status", "running_modal"},
                            {"modal", modal_state_json()},
                        };
                    }
                    if (!invocation.is_finished()) {
                        return json{
                            {"error", operator_cancel_message(*descriptor)},
                            {"operator_id", descriptor->id()},
                            {"modal", modal_state_json()},
                        };
                    }

                    return json{
                        {"success", true},
                        {"operator_id", descriptor->id()},
                        {"status", "finished"},
                        {"modal", modal_state_json()},
                    };
                });
            });

        registry.register_tool(
            mcp::McpTool{
                .name = "operator.modal_state",
                .description = "Inspect whether a modal operator is currently active",
                .input_schema = {.type = "object", .properties = json::object(), .required = {}},
                .metadata = mcp::McpToolMetadata{
                    .category = "operator",
                    .kind = "query",
                    .runtime = "gui",
                    .thread_affinity = "gui_thread",
                }},
            [viewer](const json&) -> json {
                return post_and_wait(viewer, []() -> json {
                    auto payload = modal_state_json();
                    payload["success"] = true;
                    return payload;
                });
            });

        registry.register_tool(
            mcp::McpTool{
                .name = "operator.modal_event",
                .description = "Dispatch a modal mouse or key event to the currently active modal operator",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"type", json{{"type", "string"}, {"enum", json::array({"mouse_button", "mouse_move", "mouse_scroll", "key"})}, {"description", "Modal event type"}}},
                        {"x", json{{"type", "number"}, {"description", "Mouse X position for mouse events"}}},
                        {"y", json{{"type", "number"}, {"description", "Mouse Y position for mouse events"}}},
                        {"dx", json{{"type", "number"}, {"description", "Mouse X delta for mouse_move events"}}},
                        {"dy", json{{"type", "number"}, {"description", "Mouse Y delta for mouse_move events"}}},
                        {"button", json{{"type", "integer"}, {"description", "Mouse button for mouse_button events (0=left, 1=right, 2=middle)"}}},
                        {"key", json{{"type", "integer"}, {"description", "Key code for key events (for example 256=escape, 257=enter)"}}},
                        {"action", json{{"type", "integer"}, {"description", "Action code (0=release, 1=press, 2=repeat)"}}},
                        {"mods", json{{"type", "integer"}, {"description", "Modifier bitmask (1=shift, 2=ctrl, 4=alt, 8=super)"}}},
                        {"scancode", json{{"type", "integer"}, {"description", "Optional scan code for key events"}}},
                        {"scroll_x", json{{"type", "number"}, {"description", "Horizontal scroll delta for mouse_scroll events"}}},
                        {"scroll_y", json{{"type", "number"}, {"description", "Vertical scroll delta for mouse_scroll events"}}}},
                    .required = {"type"}},
                .metadata = mcp::McpToolMetadata{
                    .category = "operator",
                    .kind = "command",
                    .runtime = "gui",
                    .thread_affinity = "gui_thread",
                    .long_running = true,
                }},
            [viewer](const json& args) -> json {
                auto modal_event = modal_event_from_json(args);
                if (!modal_event) {
                    return json{{"error", modal_event.error()}};
                }

                return post_and_wait(viewer, [event = *modal_event]() -> json {
                    const auto result = vis::op::operators().dispatchModalEvent(event);
                    return json{
                        {"success", true},
                        {"status", operator_result_to_string(result)},
                        {"modal", modal_state_json()},
                    };
                });
            });

        registry.register_tool(
            mcp::McpTool{
                .name = "operator.cancel_modal",
                .description = "Cancel the currently active modal operator, if any",
                .input_schema = {.type = "object", .properties = json::object(), .required = {}},
                .metadata = mcp::McpToolMetadata{
                    .category = "operator",
                    .kind = "command",
                    .runtime = "gui",
                    .thread_affinity = "gui_thread",
                    .destructive = false,
                }},
            [viewer](const json&) -> json {
                return post_and_wait(viewer, []() -> json {
                    const auto before = modal_state_json();
                    vis::op::operators().cancelModalOperator();
                    return json{
                        {"success", true},
                        {"before", before},
                        {"after", modal_state_json()},
                    };
                });
            });
    }

    void register_generic_gui_operator_resources(mcp::ResourceRegistry& registry,
                                                 vis::Visualizer* viewer) {
        registry.register_resource(
            mcp::McpResource{
                .uri = "lichtfeld://operators/registry",
                .name = "Operator Registry",
                .description = "Registered GUI operators with flags, schemas, and poll status",
                .mime_type = "application/json"},
            [viewer](const std::string& uri) -> std::expected<std::vector<mcp::McpResourceContent>, std::string> {
                return post_and_wait(viewer, [uri]() -> std::expected<std::vector<mcp::McpResourceContent>, std::string> {
                    json operators_json = json::array();
                    for (const auto* descriptor : sorted_operator_descriptors(false)) {
                        operators_json.push_back(operator_descriptor_json(
                            *descriptor,
                            true,
                            std::optional<bool>(vis::op::operators().poll(descriptor->id()))));
                    }

                    return single_json_resource(
                        uri,
                        json{
                            {"count", static_cast<int64_t>(operators_json.size())},
                            {"operators", std::move(operators_json)},
                        });
                });
            });

        registry.register_resource(
            mcp::McpResource{
                .uri = "lichtfeld://operators/modal_state",
                .name = "Operator Modal State",
                .description = "Current modal operator state",
                .mime_type = "application/json"},
            [viewer](const std::string& uri) -> std::expected<std::vector<mcp::McpResourceContent>, std::string> {
                return post_and_wait(viewer, [uri]() -> std::expected<std::vector<mcp::McpResourceContent>, std::string> {
                    return single_json_resource(uri, modal_state_json());
                });
            });
    }

} // namespace lfs::app

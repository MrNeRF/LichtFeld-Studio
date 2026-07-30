/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "mcp_tools.hpp"
#include "mcp_training_context.hpp"

#include "core/error.hpp"
#include "core/error_envelope.hpp"
#include "core/error_reporter.hpp"
#include "core/event_bridge/command_center_bridge.hpp"
#include "core/guarded_task.hpp"
#include "core/logger.hpp"

#include <algorithm>
#include <cassert>

namespace lfs::mcp {

    namespace {

        bool is_valid_tool_name(const std::string& name) {
            if (name.empty() || name.size() > 64) {
                return false;
            }
            return std::all_of(name.begin(), name.end(), [](const unsigned char ch) {
                return (ch >= 'a' && ch <= 'z') ||
                       (ch >= 'A' && ch <= 'Z') ||
                       (ch >= '0' && ch <= '9') ||
                       ch == '_' || ch == '-';
            });
        }

        std::string target_to_string(training::CommandTarget target) {
            switch (target) {
            case training::CommandTarget::Model:
                return "model";
            case training::CommandTarget::Optimizer:
                return "optimizer";
            case training::CommandTarget::Session:
                return "session";
            }
            return "unknown";
        }

        training::CommandTarget string_to_target(const std::string& s) {
            if (s == "model")
                return training::CommandTarget::Model;
            if (s == "optimizer")
                return training::CommandTarget::Optimizer;
            if (s == "session")
                return training::CommandTarget::Session;
            return training::CommandTarget::Session;
        }

        json parameter_error_envelope(const lfs::ErrorCode code, const std::string& message,
                                      const std::string& parameter, const lfs::OperationId operation_id) {
            lfs::Error error = lfs::make_error(lfs::ErrorInit{
                .code = code,
                .domain = lfs::ErrorDomain::MCP,
                .operation_id = operation_id,
                .user_message = message,
                .detection = LFS_SOURCE_SITE_CURRENT(),
                .fields = lfs::SmallFields{}.add("parameter", parameter),
            });
            return json{{"error", lfs::core::to_wire_envelope(error)}, {"error_message", message}};
        }

        json invoke_handler_guarded(const std::string& name, const ToolRegistry::ToolHandler& handler,
                                    const json& arguments, const lfs::OperationId operation_id) {
            try {
                return handler(arguments);
            } catch (...) {
                const lfs::Error error = lfs::core::detail::normalize_current_exception(lfs::core::TaskContext{
                    .name = "mcp.tool:" + name,
                    .domain = lfs::ErrorDomain::MCP,
                    .operation_id = operation_id,
                    .site = LFS_SOURCE_SITE_CURRENT(),
                });
                lfs::core::ErrorReporter::get().report(error, lfs::core::ReportChannel::OwnerLog);
                json envelope = lfs::core::to_wire_envelope(error);
                std::string message = envelope.value("message", std::string{});
                return json{{"error", std::move(envelope)}, {"error_message", std::move(message)}};
            }
        }

        bool is_wire_envelope(const json& error) {
            return error.is_object() &&
                   error.contains("code") && error.at("code").is_string() &&
                   error.contains("domain") && error.at("domain").is_string();
        }

        json bridge_tool_result(json result, const std::string& name, const lfs::OperationId operation_id) {
            if (!result.is_object() || !result.contains("error")) {
                return result;
            }
            const json& error = result.at("error");
            if (error.is_string()) {
                std::string message = error.get<std::string>();
                if (message.empty()) {
                    return result;
                }
                // BF-10: a handful of successful job-status payloads
                // (mcp_runtime_tools.cpp) carry an informational top-level
                // "error" string; those get rewrapped as a FailedPrecondition
                // envelope here. error_message preserves the text so no data is
                // lost; Phase 11 refines the semantics per site.
                lfs::Error typed = lfs::make_legacy_error(message, lfs::LegacyErrorContext{
                                                                       .code = lfs::ErrorCode::FailedPrecondition,
                                                                       .domain = lfs::ErrorDomain::MCP,
                                                                       .operation = "mcp.tool:" + name,
                                                                       .source = LFS_SOURCE_SITE_CURRENT(),
                                                                       .operation_id = operation_id,
                                                                   });
                result["error"] = lfs::core::to_wire_envelope(typed);
                result["error_message"] = result.at("error").value("message", std::string{});
                return result;
            }
            if (is_wire_envelope(error) && !result.contains("error_message") &&
                error.contains("message") && error.at("message").is_string()) {
                std::string mirror = error.at("message").get<std::string>();
                result["error_message"] = std::move(mirror);
            }
            return result;
        }

    } // namespace

    ToolRegistry& ToolRegistry::instance() {
        static ToolRegistry inst;
        return inst;
    }

    ResourceRegistry& ResourceRegistry::instance() {
        static ResourceRegistry inst;
        return inst;
    }

    void ToolRegistry::register_tool(McpTool tool, ToolHandler handler) {
        const std::string normalized_name = normalize_tool_name(tool.name);
        if (!is_valid_tool_name(normalized_name)) {
            LOG_ERROR(
                "Cannot register MCP tool '{}': normalized name '{}' does not match "
                "^[a-zA-Z0-9_-]{{1,64}}$",
                tool.name,
                normalized_name);
            return;
        }

        std::lock_guard lock(mutex_);
        const auto existing = tools_.find(normalized_name);
        if (existing != tools_.end() && existing->second.tool.name != tool.name) {
            LOG_ERROR(
                "Cannot register MCP tool '{}': existing tool '{}' already uses normalized name '{}'",
                tool.name,
                existing->second.tool.name,
                normalized_name);
            return;
        }

        tools_[normalized_name] = RegisteredTool{std::move(tool), std::move(handler)};
    }

    void ToolRegistry::unregister_tool(const std::string& name) {
        std::lock_guard lock(mutex_);
        tools_.erase(normalize_tool_name(name));
    }

    std::vector<McpTool> ToolRegistry::list_tools() const {
        std::lock_guard lock(mutex_);
        std::vector<McpTool> result;
        result.reserve(tools_.size());
        for (const auto& [name, reg] : tools_) {
            result.push_back(reg.tool);
        }
        std::sort(result.begin(), result.end(), [](const McpTool& a, const McpTool& b) {
            return a.name < b.name;
        });
        return result;
    }

    json ToolRegistry::call_tool(const std::string& name, const json& arguments,
                                 lfs::OperationId operation_id) {
        ToolHandler handler;
        std::vector<std::string> required;
        {
            std::lock_guard lock(mutex_);
            const std::string normalized_name = normalize_tool_name(name);
            auto it = tools_.find(normalized_name);
            if (it == tools_.end())
                return parameter_error_envelope(lfs::ErrorCode::NotFound, "Tool not found: " + name,
                                                name, operation_id);
            handler = it->second.handler;
            required = it->second.tool.input_schema.required;
        }

        for (const auto& field : required) {
            if (!arguments.contains(field))
                return parameter_error_envelope(lfs::ErrorCode::InvalidArgument,
                                                "Missing required parameter: " + field, field,
                                                operation_id);
        }

        return bridge_tool_result(invoke_handler_guarded(name, handler, arguments, operation_id),
                                  name, operation_id);
    }

    void ResourceRegistry::register_resource(McpResource resource, ResourceHandler handler) {
        std::lock_guard lock(mutex_);
        const std::string uri = resource.uri;
        resources_[uri] = RegisteredResource{std::move(resource), handler};
    }

    void ResourceRegistry::unregister_resource(const std::string& uri) {
        std::lock_guard lock(mutex_);
        resources_.erase(uri);
    }

    void ResourceRegistry::register_resource_prefix(std::string uri_prefix, ResourceHandler handler) {
        std::lock_guard lock(mutex_);
        const std::string prefix = uri_prefix;
        prefix_handlers_[prefix] = handler;
    }

    void ResourceRegistry::unregister_resource_prefix(const std::string& uri_prefix) {
        std::lock_guard lock(mutex_);
        prefix_handlers_.erase(uri_prefix);
    }

    std::vector<McpResource> ResourceRegistry::list_resources() const {
        std::lock_guard lock(mutex_);
        std::vector<McpResource> result;
        result.reserve(resources_.size());
        for (const auto& [uri, reg] : resources_) {
            result.push_back(reg.resource);
        }
        std::sort(result.begin(), result.end(), [](const McpResource& a, const McpResource& b) {
            return a.uri < b.uri;
        });
        return result;
    }

    std::expected<std::vector<McpResourceContent>, std::string> ResourceRegistry::read_resource(const std::string& uri) const {
        ResourceHandler handler;
        {
            std::lock_guard lock(mutex_);
            if (const auto it = resources_.find(uri); it != resources_.end()) {
                handler = it->second.handler;
            }
            if (!handler) {
                for (const auto& [key, reg] : resources_) {
                    if (reg.resource.uri == uri && reg.handler) {
                        handler = reg.handler;
                        break;
                    }
                }
            }
            if (!handler) {
                size_t best_prefix_len = 0;
                for (const auto& [prefix, prefix_handler] : prefix_handlers_) {
                    if (!uri.starts_with(prefix) || prefix.size() < best_prefix_len)
                        continue;
                    best_prefix_len = prefix.size();
                    handler = prefix_handler;
                }
            }
        }

        if (!handler)
            return std::unexpected("Unknown resource URI: " + uri);

        return handler(uri);
    }

    json ToolRegistry::arg_type_to_json_schema(training::ArgType type) const {
        json schema;
        switch (type) {
        case training::ArgType::Int:
            schema["type"] = "integer";
            break;
        case training::ArgType::Float:
            schema["type"] = "number";
            break;
        case training::ArgType::Bool:
            schema["type"] = "boolean";
            break;
        case training::ArgType::String:
            schema["type"] = "string";
            break;
        case training::ArgType::IntList:
            schema["type"] = "array";
            schema["items"] = json{{"type", "integer"}};
            break;
        case training::ArgType::FloatList:
            schema["type"] = "array";
            schema["items"] = json{{"type", "number"}};
            break;
        }
        return schema;
    }

    McpTool ToolRegistry::operation_to_tool(const training::OperationInfo& op) const {
        McpTool tool;

        std::string target_str = target_to_string(op.target);
        tool.name = target_str + "." + op.name;
        tool.description = op.description;
        tool.metadata.category = target_str;
        tool.metadata.kind = "command";

        json properties = json::object();
        std::vector<std::string> required;

        for (const auto& arg : op.args) {
            json prop = arg_type_to_json_schema(arg.type);
            if (arg.description) {
                prop["description"] = *arg.description;
            }
            properties[arg.name] = prop;

            if (arg.required) {
                required.push_back(arg.name);
            }
        }

        bool has_selection = !op.selectors.empty() &&
                             op.selectors.size() > 1; // More than just "All"
        if (has_selection) {
            json selection_prop;
            selection_prop["type"] = "object";
            selection_prop["description"] = "Selection of gaussians to operate on";
            selection_prop["properties"] = json{
                {"kind", json{{"type", "string"}, {"enum", json::array({"all", "range", "indices"})}}},
                {"start", json{{"type", "integer"}, {"description", "Start index for range selection"}}},
                {"end", json{{"type", "integer"}, {"description", "End index (exclusive) for range selection"}}},
                {"indices", json{{"type", "array"}, {"items", json{{"type", "integer"}}}, {"description", "Specific indices to select"}}}};
            properties["selection"] = selection_prop;
        }

        tool.input_schema.properties = properties;
        tool.input_schema.required = required;

        return tool;
    }

    training::Command ToolRegistry::json_to_command(
        const training::OperationInfo& op,
        const json& arguments) const {

        training::Command cmd;
        cmd.target = op.target;
        cmd.op = op.name;

        if (arguments.contains("selection")) {
            const auto& sel = arguments["selection"];
            std::string kind = sel.value("kind", "all");

            if (kind == "range") {
                cmd.selection.kind = training::SelectionKind::Range;
                cmd.selection.start = sel.value("start", int64_t(0));
                cmd.selection.end = sel.value("end", int64_t(0));
            } else if (kind == "indices") {
                cmd.selection.kind = training::SelectionKind::Indices;
                if (sel.contains("indices")) {
                    cmd.selection.indices = sel["indices"].get<std::vector<int64_t>>();
                }
            } else {
                cmd.selection.kind = training::SelectionKind::All;
            }
        }

        for (const auto& arg_spec : op.args) {
            if (!arguments.contains(arg_spec.name)) {
                continue;
            }

            const auto& val = arguments[arg_spec.name];
            training::ArgValue arg_val;

            switch (arg_spec.type) {
            case training::ArgType::Int:
                arg_val = val.get<int64_t>();
                break;
            case training::ArgType::Float:
                arg_val = val.get<double>();
                break;
            case training::ArgType::Bool:
                arg_val = val.get<bool>();
                break;
            case training::ArgType::String:
                arg_val = val.get<std::string>();
                break;
            case training::ArgType::IntList:
                arg_val = val.get<std::vector<int64_t>>();
                break;
            case training::ArgType::FloatList:
                arg_val = val.get<std::vector<double>>();
                break;
            }

            cmd.args[arg_spec.name] = arg_val;
        }

        return cmd;
    }

    ToolRegistry::ToolHandler ToolRegistry::create_command_handler(
        const training::OperationInfo& op) const {

        return [this, op](const json& arguments) -> json {
            auto* cc = event::command_center();
            if (!cc) {
                return json{{"error", "Training system not initialized"}};
            }

            training::Command cmd = json_to_command(op, arguments);

            auto result = cc->execute(cmd);
            if (!result) {
                return json{{"error", result.error()}};
            }

            json response;
            response["success"] = true;
            response["operation"] = target_to_string(op.target) + "." + op.name;

            auto snapshot = cc->snapshot();
            response["state"] = json{
                {"iteration", snapshot.iteration},
                {"num_gaussians", snapshot.num_gaussians},
                {"loss", snapshot.loss},
                {"is_running", snapshot.is_running},
                {"is_paused", snapshot.is_paused}};

            return response;
        };
    }

    void ToolRegistry::generate_from_command_center() {
        auto* cc = event::command_center();
        if (!cc) {
            LOG_WARN("Cannot generate MCP tools: CommandCenter not available");
            return;
        }

        auto ops = cc->operations();

        for (const auto& op : ops) {
            McpTool tool = operation_to_tool(op);
            ToolHandler handler = create_command_handler(op);
            register_tool(std::move(tool), std::move(handler));
        }

        LOG_INFO("Generated {} MCP tools from CommandCenter", ops.size());
    }

    void register_core_tools() {
        auto& registry = ToolRegistry::instance();

        registry.register_tool(
            McpTool{
                .name = "training.get_state",
                .description = "Get current training state snapshot",
                .input_schema = {.type = "object", .properties = json::object(), .required = {}},
                .metadata = McpToolMetadata{.category = "training", .kind = "query"}},
            [](const json&) -> json {
                auto* cc = event::command_center();
                if (!cc) {
                    return json{{"error", "Training system not initialized"}};
                }

                auto snapshot = cc->snapshot();
                return json{
                    {"iteration", snapshot.iteration},
                    {"max_iterations", snapshot.max_iterations},
                    {"num_gaussians", snapshot.num_gaussians},
                    {"loss", snapshot.loss},
                    {"is_running", snapshot.is_running},
                    {"is_paused", snapshot.is_paused},
                    {"is_refining", snapshot.is_refining}};
            });

        registry.register_tool(
            McpTool{
                .name = "training.list_operations",
                .description = "List all available CommandCenter operations",
                .input_schema = {.type = "object", .properties = json::object(), .required = {}},
                .metadata = McpToolMetadata{.category = "training", .kind = "query"}},
            [](const json&) -> json {
                auto* cc = event::command_center();
                if (!cc) {
                    return json{{"error", "Training system not initialized"}};
                }

                auto ops = cc->operations();
                json result = json::array();

                for (const auto& op : ops) {
                    json op_json;
                    op_json["name"] = target_to_string(op.target) + "." + op.name;
                    op_json["description"] = op.description;

                    json args = json::array();
                    for (const auto& arg : op.args) {
                        json arg_json;
                        arg_json["name"] = arg.name;
                        arg_json["required"] = arg.required;
                        if (arg.description) {
                            arg_json["description"] = *arg.description;
                        }
                        args.push_back(arg_json);
                    }
                    op_json["args"] = args;
                    result.push_back(op_json);
                }

                return json{{"operations", result}};
            });

        registry.register_tool(
            McpTool{
                .name = "training.get_loss_history",
                .description = "Get training loss history",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"last_n", json{{"type", "integer"}, {"description", "Return only last N points (default: all)"}}}},
                    .required = {}}},
            [](const json& args) -> json {
                auto* cc = event::command_center();
                if (!cc) {
                    return json{{"error", "Training system not initialized"}};
                }

                auto history = cc->loss_history();
                int last_n = args.contains("last_n") ? args["last_n"].get<int>() : 0;

                json points = json::array();
                size_t start = 0;
                if (last_n > 0 && static_cast<size_t>(last_n) < history.size()) {
                    start = history.size() - last_n;
                }

                for (size_t i = start; i < history.size(); ++i) {
                    points.push_back(json{
                        {"iteration", history[i].iteration},
                        {"loss", history[i].loss}});
                }

                return json{{"count", points.size()}, {"points", points}};
            });

        registry.generate_from_command_center();
    }

    void register_core_resources() {
        auto& registry = ResourceRegistry::instance();

        registry.register_resource(
            McpResource{
                .uri = "lichtfeld://scene/state",
                .name = "Training State",
                .description = "Current training state snapshot (iteration, loss, gaussians)",
                .mime_type = "application/json"},
            [](const std::string& uri) -> std::expected<std::vector<McpResourceContent>, std::string> {
                json content;
                if (auto* cc = event::command_center()) {
                    auto snapshot = cc->snapshot();
                    content["iteration"] = snapshot.iteration;
                    content["max_iterations"] = snapshot.max_iterations;
                    content["num_gaussians"] = snapshot.num_gaussians;
                    content["loss"] = snapshot.loss;
                    content["is_running"] = snapshot.is_running;
                    content["is_paused"] = snapshot.is_paused;
                    content["is_refining"] = snapshot.is_refining;
                } else {
                    content["error"] = "Training system not initialized";
                }

                return std::vector<McpResourceContent>{
                    McpResourceContent{
                        .uri = uri,
                        .mime_type = "application/json",
                        .content = content.dump(2)}};
            });

        registry.register_resource(
            McpResource{
                .uri = "lichtfeld://training/loss_curve",
                .name = "Loss Curve",
                .description = "Training loss history",
                .mime_type = "application/json"},
            [](const std::string& uri) -> std::expected<std::vector<McpResourceContent>, std::string> {
                json content;
                if (auto* cc = event::command_center()) {
                    auto history = cc->loss_history();
                    json points = json::array();
                    for (const auto& p : history) {
                        points.push_back(json{{"iteration", p.iteration}, {"loss", p.loss}});
                    }
                    content["points"] = std::move(points);
                    content["count"] = history.size();
                } else {
                    content["error"] = "Training system not initialized";
                }

                return std::vector<McpResourceContent>{
                    McpResourceContent{
                        .uri = uri,
                        .mime_type = "application/json",
                        .content = content.dump(2)}};
            });

        registry.register_resource(
            McpResource{
                .uri = "lichtfeld://gaussians/stats",
                .name = "Gaussian Statistics",
                .description = "Statistics about the Gaussian model",
                .mime_type = "application/json"},
            [](const std::string& uri) -> std::expected<std::vector<McpResourceContent>, std::string> {
                json content;
                if (auto* cc = event::command_center()) {
                    auto snapshot = cc->snapshot();
                    content["count"] = snapshot.num_gaussians;
                    content["is_refining"] = snapshot.is_refining;
                } else {
                    content["error"] = "Training system not initialized";
                }

                return std::vector<McpResourceContent>{
                    McpResourceContent{
                        .uri = uri,
                        .mime_type = "application/json",
                        .content = content.dump(2)}};
            });
    }

    void register_builtin_tools() {
        register_core_tools();
        register_core_resources();
        register_scene_tools();
    }

} // namespace lfs::mcp

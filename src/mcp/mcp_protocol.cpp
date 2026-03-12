/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "mcp_protocol.hpp"

#include <cassert>

namespace lfs::mcp {

    JsonRpcRequest parse_request(const std::string& input) {
        JsonRpcRequest req;

        auto j = json::parse(input);

        req.jsonrpc = j.value("jsonrpc", "2.0");

        if (j.contains("id")) {
            if (j["id"].is_number_integer()) {
                req.id = j["id"].get<int64_t>();
            } else if (j["id"].is_string()) {
                req.id = j["id"].get<std::string>();
            }
        }

        req.method = j.value("method", "");

        if (j.contains("params")) {
            req.params = j["params"];
        }

        return req;
    }

    std::string serialize_response(const JsonRpcResponse& response) {
        json j;
        j["jsonrpc"] = response.jsonrpc;

        std::visit([&j](const auto& id) { j["id"] = id; }, response.id);

        if (response.result) {
            j["result"] = *response.result;
        }

        if (response.error) {
            json err;
            err["code"] = response.error->code;
            err["message"] = response.error->message;
            if (response.error->data) {
                err["data"] = *response.error->data;
            }
            j["error"] = err;
        }

        return j.dump();
    }

    std::string serialize_notification(const std::string& method, const json& params) {
        json j;
        j["jsonrpc"] = "2.0";
        j["method"] = method;
        j["params"] = params;
        return j.dump();
    }

    json tool_to_json(const McpTool& tool) {
        json j;
        j["name"] = tool.name;
        j["description"] = tool.description;

        json schema;
        schema["type"] = tool.input_schema.type;
        schema["properties"] = tool.input_schema.properties;
        if (!tool.input_schema.required.empty()) {
            schema["required"] = tool.input_schema.required;
        }
        j["inputSchema"] = schema;

        json annotations{
            {"readOnlyHint", tool.metadata.kind == "query"},
            {"destructiveHint", tool.metadata.destructive},
            {"idempotentHint", tool.metadata.kind == "query"},
            {"x-lfs-category", tool.metadata.category},
            {"x-lfs-kind", tool.metadata.kind},
            {"x-lfs-runtime", tool.metadata.runtime},
            {"x-lfs-thread-affinity", tool.metadata.thread_affinity},
            {"x-lfs-user-visible", tool.metadata.user_visible},
        };
        if (tool.metadata.long_running) {
            annotations["x-lfs-long-running"] = true;
        }
        j["annotations"] = std::move(annotations);

        return j;
    }

    json resource_to_json(const McpResource& resource) {
        json j;
        j["uri"] = resource.uri;
        j["name"] = resource.name;
        j["description"] = resource.description;
        if (resource.mime_type) {
            j["mimeType"] = *resource.mime_type;
        }
        return j;
    }

    json capabilities_to_json(const McpCapabilities& caps) {
        json j;
        if (caps.tools) {
            j["tools"] = json::object();
        }
        if (caps.resources) {
            j["resources"] = json::object();
        }
        if (caps.prompts) {
            j["prompts"] = json::object();
        }
        if (caps.logging) {
            j["logging"] = json::object();
        }
        return j;
    }

    json initialize_result_to_json(const McpInitializeResult& result) {
        json j;
        j["protocolVersion"] = result.protocol_version;
        j["capabilities"] = capabilities_to_json(result.capabilities);

        json server;
        server["name"] = result.server_info.name;
        server["version"] = result.server_info.version;
        j["serverInfo"] = server;

        return j;
    }

    JsonRpcResponse make_error_response(
        const std::variant<int64_t, std::string>& id,
        int code,
        const std::string& message,
        const std::optional<json>& data) {

        JsonRpcResponse resp;
        resp.id = id;
        resp.error = JsonRpcError{code, message, data};
        return resp;
    }

    JsonRpcResponse make_success_response(
        const std::variant<int64_t, std::string>& id,
        const json& result) {

        JsonRpcResponse resp;
        resp.id = id;
        resp.result = result;
        return resp;
    }

} // namespace lfs::mcp

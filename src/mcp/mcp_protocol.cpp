/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "mcp_protocol.hpp"

#include <algorithm>
#include <cassert>
#include <type_traits>

namespace lfs::mcp {

    std::string normalize_tool_name(std::string name) {
        std::replace(name.begin(), name.end(), '.', '_');
        return name;
    }

    RequestId RequestId::from_json(const json& request_object) {
        if (!request_object.contains("id")) {
            return RequestId{};
        }

        const auto& id = request_object["id"];
        if (id.is_number_integer()) {
            return RequestId{id.get<std::int64_t>()};
        }
        if (id.is_string()) {
            return RequestId{id.get<std::string>()};
        }
        return RequestId{nullptr};
    }

    std::optional<json> RequestId::to_json() const {
        return std::visit(
            []<typename T>(const T& value) -> std::optional<json> {
                if constexpr (std::is_same_v<T, std::monostate>) {
                    return std::nullopt;
                } else {
                    return json(value);
                }
            },
            value_);
    }

    JsonRpcRequest parse_request(const std::string& input) {
        JsonRpcRequest req;

        auto j = json::parse(input);

        req.jsonrpc = j.value("jsonrpc", "2.0");
        req.id = RequestId::from_json(j);
        req.method = j.value("method", "");

        if (j.contains("params")) {
            req.params = j["params"];
        }

        return req;
    }

    std::string serialize_response(const JsonRpcResponse& response) {
        json j;
        j["jsonrpc"] = response.jsonrpc;

        if (const auto id = response.id.to_json()) {
            j["id"] = *id;
        }

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

        return j.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);
    }

    std::string serialize_notification(const std::string& method, const json& params) {
        json j;
        j["jsonrpc"] = "2.0";
        j["method"] = method;
        j["params"] = params;
        return j.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);
    }

    json tool_to_json(const McpTool& tool) {
        json j;
        j["name"] = normalize_tool_name(tool.name);
        j["description"] = tool.description;

        json schema;
        schema["type"] = tool.input_schema.type;
        schema["properties"] = tool.input_schema.properties.is_object()
                                   ? tool.input_schema.properties
                                   : json::object();
        if (!tool.input_schema.required.empty()) {
            schema["required"] = tool.input_schema.required;
        }
        j["inputSchema"] = schema;

        json annotations{
            {"readOnlyHint", tool.metadata.kind == "query"},
            {"destructiveHint", tool.metadata.destructive},
            {"idempotentHint", tool.metadata.kind == "query"},
        };
        j["annotations"] = std::move(annotations);

        json meta{
            {"app.lichtfeld/category", tool.metadata.category},
            {"app.lichtfeld/kind", tool.metadata.kind},
            {"app.lichtfeld/runtime", tool.metadata.runtime},
            {"app.lichtfeld/thread_affinity", tool.metadata.thread_affinity},
            {"app.lichtfeld/user_visible", tool.metadata.user_visible},
        };
        if (tool.metadata.long_running) {
            meta["app.lichtfeld/long_running"] = true;
        }
        j["_meta"] = std::move(meta);

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
        const RequestId& id,
        int code,
        const std::string& message,
        const std::optional<json>& data) {

        JsonRpcResponse resp;
        resp.id = id;
        resp.error = JsonRpcError{code, message, data};
        return resp;
    }

    JsonRpcResponse make_success_response(
        const RequestId& id,
        const json& result) {

        JsonRpcResponse resp;
        resp.id = id;
        resp.result = result;
        return resp;
    }

} // namespace lfs::mcp

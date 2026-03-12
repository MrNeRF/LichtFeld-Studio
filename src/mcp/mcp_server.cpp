/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "mcp_server.hpp"
#include "core/base64.hpp"
#include "core/event_bridge/command_center_bridge.hpp"

#include <cassert>

namespace lfs::mcp {

    McpServer::McpServer(const McpServerOptions& options) {
        capabilities_.tools = options.enable_tools;
        capabilities_.resources = options.enable_resources;
        capabilities_.prompts = false;
        capabilities_.logging = options.enable_logging;
    }

    McpServer::~McpServer() {
    }

    JsonRpcResponse McpServer::handle_request(const JsonRpcRequest& req) {
        if (req.method == "initialize") {
            return handle_initialize(req);
        }
        if (req.method == "notifications/initialized" || req.method == "initialized") {
            return handle_initialized(req);
        }
        if (req.method == "ping") {
            return handle_ping(req);
        }

        if (!initialized_ && req.method != "initialize") {
            return make_error_response(
                req.id,
                JsonRpcError::INVALID_REQUEST,
                "Server not initialized. Call 'initialize' first.");
        }

        if (capabilities_.tools && req.method == "tools/list") {
            return handle_tools_list(req);
        }
        if (capabilities_.tools && req.method == "tools/call") {
            return handle_tools_call(req);
        }
        if (capabilities_.resources && req.method == "resources/list") {
            return handle_resources_list(req);
        }
        if (capabilities_.resources && req.method == "resources/read") {
            return handle_resources_read(req);
        }

        return make_error_response(
            req.id,
            JsonRpcError::METHOD_NOT_FOUND,
            "Method not found: " + req.method);
    }

    JsonRpcResponse McpServer::handle_initialize(const JsonRpcRequest& req) {
        McpInitializeResult result;
        result.capabilities = capabilities_;
        result.server_info.name = "lichtfeld-mcp";
        result.server_info.version = "1.0.0";

        initialized_ = true;

        return make_success_response(req.id, initialize_result_to_json(result));
    }

    JsonRpcResponse McpServer::handle_initialized(const JsonRpcRequest& req) {
        return make_success_response(req.id, json::object());
    }

    JsonRpcResponse McpServer::handle_ping(const JsonRpcRequest& req) {
        return make_success_response(req.id, json::object());
    }

    JsonRpcResponse McpServer::handle_tools_list(const JsonRpcRequest& req) {
        auto tools = ToolRegistry::instance().list_tools();

        json tools_array = json::array();
        for (const auto& tool : tools) {
            tools_array.push_back(tool_to_json(tool));
        }

        return make_success_response(req.id, json{{"tools", tools_array}});
    }

    JsonRpcResponse McpServer::handle_tools_call(const JsonRpcRequest& req) {
        if (!req.params) {
            return make_error_response(
                req.id,
                JsonRpcError::INVALID_PARAMS,
                "Missing params for tools/call");
        }

        const auto& params = *req.params;

        if (!params.contains("name")) {
            return make_error_response(
                req.id,
                JsonRpcError::INVALID_PARAMS,
                "Missing 'name' parameter");
        }

        std::string tool_name = params["name"].get<std::string>();
        json arguments = params.value("arguments", json::object());

        json result = ToolRegistry::instance().call_tool(tool_name, arguments);
        const bool is_error = result.is_object() && result.contains("error");

        json content = json::array();
        content.push_back(json{
            {"type", "text"},
            {"text", result.dump(2)}});

        return make_success_response(req.id, json{
                                                 {"content", content},
                                                 {"structuredContent", result},
                                                 {"isError", is_error},
                                             });
    }

    JsonRpcResponse McpServer::handle_resources_list(const JsonRpcRequest& req) {
        auto resources = ResourceRegistry::instance().list_resources();
        json resources_json = json::array();
        for (const auto& resource : resources) {
            resources_json.push_back(resource_to_json(resource));
        }

        return make_success_response(req.id, json{{"resources", resources_json}});
    }

    JsonRpcResponse McpServer::handle_resources_read(const JsonRpcRequest& req) {
        if (!req.params || !req.params->contains("uri")) {
            return make_error_response(
                req.id,
                JsonRpcError::INVALID_PARAMS,
                "Missing 'uri' parameter");
        }

        std::string uri = (*req.params)["uri"].get<std::string>();
        auto contents = ResourceRegistry::instance().read_resource(uri);
        if (!contents) {
            return make_error_response(
                req.id,
                JsonRpcError::INVALID_PARAMS,
                contents.error());
        }

        json result;
        result["contents"] = json::array();
        for (const auto& content : *contents) {
            json item{{"uri", content.uri}};
            if (content.mime_type)
                item["mimeType"] = *content.mime_type;

            if (std::holds_alternative<std::string>(content.content)) {
                const auto& string_content = std::get<std::string>(content.content);
                const bool is_blob = content.mime_type && content.mime_type->starts_with("image/");
                if (is_blob) {
                    item["blob"] = string_content;
                } else {
                    item["text"] = string_content;
                }
            } else {
                item["blob"] = core::base64_encode(std::get<std::vector<uint8_t>>(content.content));
            }
            result["contents"].push_back(std::move(item));
        }

        return make_success_response(req.id, result);
    }
} // namespace lfs::mcp

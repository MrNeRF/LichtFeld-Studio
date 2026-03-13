/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"

#include <nlohmann/json.hpp>

#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace lfs::mcp {

    using json = nlohmann::json;

    struct JsonRpcError {
        int code;
        std::string message;
        std::optional<json> data;

        static constexpr int PARSE_ERROR = -32700;
        static constexpr int INVALID_REQUEST = -32600;
        static constexpr int METHOD_NOT_FOUND = -32601;
        static constexpr int INVALID_PARAMS = -32602;
        static constexpr int INTERNAL_ERROR = -32603;
    };

    struct JsonRpcRequest {
        std::string jsonrpc = "2.0";
        std::variant<int64_t, std::string> id;
        std::string method;
        std::optional<json> params;
    };

    struct JsonRpcResponse {
        std::string jsonrpc = "2.0";
        std::variant<int64_t, std::string> id;
        std::optional<json> result;
        std::optional<JsonRpcError> error;
    };

    struct McpCapabilities {
        bool tools = true;
        bool resources = true;
        bool prompts = false;
        bool logging = true;
    };

    struct McpServerInfo {
        std::string name = "lichtfeld-mcp";
        std::string version = "1.0.0";
    };

    struct McpInitializeResult {
        std::string protocol_version = "2024-11-05";
        McpCapabilities capabilities;
        McpServerInfo server_info;
    };

    struct McpToolInputSchema {
        std::string type = "object";
        json properties;
        std::vector<std::string> required;
    };

    struct McpToolMetadata {
        std::string category;
        std::string kind = "command";
        std::string runtime = "shared";
        std::string thread_affinity = "any";
        bool destructive = false;
        bool long_running = false;
        bool user_visible = true;
    };

    struct McpTool {
        std::string name;
        std::string description;
        McpToolInputSchema input_schema;
        McpToolMetadata metadata;
    };

    struct McpResource {
        std::string uri;
        std::string name;
        std::string description;
        std::optional<std::string> mime_type;
    };

    struct McpResourceContent {
        std::string uri;
        std::optional<std::string> mime_type;
        std::variant<std::string, std::vector<uint8_t>> content;
    };

    LFS_MCP_API JsonRpcRequest parse_request(const std::string& input);
    LFS_MCP_API std::string serialize_response(const JsonRpcResponse& response);
    LFS_MCP_API std::string serialize_notification(const std::string& method, const json& params);

    LFS_MCP_API json tool_to_json(const McpTool& tool);
    LFS_MCP_API json resource_to_json(const McpResource& resource);
    LFS_MCP_API json capabilities_to_json(const McpCapabilities& caps);
    LFS_MCP_API json initialize_result_to_json(const McpInitializeResult& result);

    LFS_MCP_API JsonRpcResponse make_error_response(
        const std::variant<int64_t, std::string>& id,
        int code,
        const std::string& message,
        const std::optional<json>& data = std::nullopt);

    LFS_MCP_API JsonRpcResponse make_success_response(
        const std::variant<int64_t, std::string>& id,
        const json& result);

} // namespace lfs::mcp

/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "mcp_protocol.hpp"
#include "mcp_tools.hpp"

#include <atomic>
#include <string>

namespace lfs::mcp {

    struct McpServerOptions {
        bool enable_tools = true;
        bool enable_resources = true;
        bool enable_logging = true;
    };

    class LFS_MCP_API McpServer {
    public:
        explicit McpServer(const McpServerOptions& options = {});
        ~McpServer();

        McpServer(const McpServer&) = delete;
        McpServer& operator=(const McpServer&) = delete;

        JsonRpcResponse handle_request(const JsonRpcRequest& req);

    private:
        JsonRpcResponse handle_initialize(const JsonRpcRequest& req);
        JsonRpcResponse handle_initialized(const JsonRpcRequest& req);
        JsonRpcResponse handle_tools_list(const JsonRpcRequest& req);
        JsonRpcResponse handle_tools_call(const JsonRpcRequest& req);
        JsonRpcResponse handle_resources_list(const JsonRpcRequest& req);
        JsonRpcResponse handle_resources_read(const JsonRpcRequest& req);
        JsonRpcResponse handle_ping(const JsonRpcRequest& req);

        std::atomic<bool> initialized_{false};
        McpCapabilities capabilities_;
    };

} // namespace lfs::mcp

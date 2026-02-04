/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "mcp_protocol.hpp"
#include "mcp_tools.hpp"

#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace lfs::mcp {

    class LFS_MCP_API McpServer {
    public:
        McpServer();
        ~McpServer();

        McpServer(const McpServer&) = delete;
        McpServer& operator=(const McpServer&) = delete;

        void run_stdio();
        void stop();

        bool is_running() const { return running_.load(); }

    private:
        JsonRpcResponse handle_request(const JsonRpcRequest& req);

        JsonRpcResponse handle_initialize(const JsonRpcRequest& req);
        JsonRpcResponse handle_initialized(const JsonRpcRequest& req);
        JsonRpcResponse handle_tools_list(const JsonRpcRequest& req);
        JsonRpcResponse handle_tools_call(const JsonRpcRequest& req);
        JsonRpcResponse handle_resources_list(const JsonRpcRequest& req);
        JsonRpcResponse handle_resources_read(const JsonRpcRequest& req);
        JsonRpcResponse handle_ping(const JsonRpcRequest& req);

        void write_response(const std::string& response);
        std::string read_line();

        std::atomic<bool> running_{false};
        bool initialized_{false};
        McpCapabilities capabilities_;
        std::mutex io_mutex_;
    };

    LFS_MCP_API int run_mcp_server_main(int argc, char* argv[]);

} // namespace lfs::mcp

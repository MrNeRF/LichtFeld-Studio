/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "mcp_server.hpp"

#include <memory>
#include <thread>

namespace httplib {
    class Server;
}

namespace lfs::mcp {

    class LFS_MCP_API McpHttpServer {
    public:
        explicit McpHttpServer(const McpServerOptions& server_options = {});
        ~McpHttpServer();

        McpHttpServer(const McpHttpServer&) = delete;
        McpHttpServer& operator=(const McpHttpServer&) = delete;

        bool start(int port = 45677);
        void stop();

    private:
        std::unique_ptr<McpServer> mcp_server_;
        std::unique_ptr<httplib::Server> http_server_;
        std::jthread listener_thread_;
    };

} // namespace lfs::mcp

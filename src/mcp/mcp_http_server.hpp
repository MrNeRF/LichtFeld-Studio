/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "mcp_server.hpp"

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace httplib {
    class Server;
}

namespace lfs::mcp {

    struct McpHttpConfig {
        bool enabled = true;
        bool expose_network = false;
        int port = 45677;
    };

    struct McpHttpStatus {
        bool enabled = true;
        bool running = false;
        bool expose_network = false;
        int port = 45677;
        std::uint64_t request_count = 0;
        std::string error;
    };

    class LFS_MCP_API McpHttpServer {
    public:
        explicit McpHttpServer(const McpServerOptions& server_options = {});
        ~McpHttpServer();

        McpHttpServer(const McpHttpServer&) = delete;
        McpHttpServer& operator=(const McpHttpServer&) = delete;

        bool start(const McpHttpConfig& config = {});
        bool start(int port) { return start(McpHttpConfig{.port = port}); }
        void stop();
        bool applyConfig(const McpHttpConfig& config);
        [[nodiscard]] McpHttpStatus status() const;

    private:
        std::unique_ptr<McpServer> mcp_server_;
        std::unique_ptr<httplib::Server> http_server_;
        std::jthread listener_thread_;
        mutable std::mutex lifecycle_mutex_;
        McpHttpStatus status_;
        std::atomic<std::uint64_t> request_count_{0};
        std::atomic<bool> running_{false};
    };

    LFS_MCP_API void setActiveMcpHttpServer(McpHttpServer* server);
    [[nodiscard]] LFS_MCP_API McpHttpStatus activeMcpHttpStatus();
    LFS_MCP_API bool applyActiveMcpHttpConfig(const McpHttpConfig& config);

} // namespace lfs::mcp

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
#include <unordered_set>
#include <vector>

namespace httplib {
    class Server;
}

namespace lfs::mcp {

    struct McpHttpConfig {
        bool enabled = true;
        bool expose_network = false;
        int port = 45677;
        bool request_logging = false;
    };

    struct McpHttpStatus {
        bool enabled = true;
        bool running = false;
        bool expose_network = false;
        int port = 45677;
        std::uint64_t request_count = 0;
        std::uint64_t success_count = 0;
        std::uint64_t error_count = 0;
        std::vector<std::string> endpoints;
        bool request_logging = false;
        std::string log_file;
        std::string error;
    };

    LFS_MCP_API bool applyActiveMcpHttpConfig(const McpHttpConfig& config);

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
        friend bool applyActiveMcpHttpConfig(const McpHttpConfig& config);

        std::unique_ptr<McpServer> mcp_server_;
        std::unique_ptr<httplib::Server> http_server_;
        std::jthread listener_thread_;
        mutable std::mutex lifecycle_mutex_;
        McpHttpStatus status_;
        std::atomic<std::uint64_t> request_count_{0};
        std::atomic<std::uint64_t> success_count_{0};
        std::atomic<std::uint64_t> error_count_{0};
        std::atomic<bool> running_{false};
        std::atomic<bool> request_logging_{false};
        mutable std::mutex log_mutex_;
        std::string log_session_timestamp_;
        std::string log_filename_;
        std::string log_contents_;
        bool log_failure_reported_ = false;
        std::unordered_set<std::string> announced_listener_urls_;

        void appendSessionLog(const nlohmann::json& event);
        void stageConfig(const McpHttpConfig& config);
    };

    LFS_MCP_API void setActiveMcpHttpServer(McpHttpServer* server);
    [[nodiscard]] LFS_MCP_API McpHttpStatus activeMcpHttpStatus();

} // namespace lfs::mcp

/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "mcp_http_server.hpp"
#include "mcp_server.hpp"

#include "core/logger.hpp"

#include <httplib/httplib.h>

namespace lfs::mcp {

    McpHttpServer::McpHttpServer(const McpServerOptions& server_options)
        : mcp_server_(std::make_unique<McpServer>(server_options)),
          http_server_(std::make_unique<httplib::Server>()) {}

    McpHttpServer::~McpHttpServer() {
        stop();
    }

    bool McpHttpServer::start(int port) {
        http_server_->Post("/mcp", [this](const httplib::Request& req, httplib::Response& res) {
            try {
                JsonRpcRequest rpc_req = parse_request(req.body);
                JsonRpcResponse rpc_resp = mcp_server_->handle_request(rpc_req);
                res.set_content(serialize_response(rpc_resp), "application/json");
            } catch (const json::parse_error& e) {
                auto resp = make_error_response(
                    int64_t(0),
                    JsonRpcError::PARSE_ERROR,
                    std::string("Parse error: ") + e.what());
                res.set_content(serialize_response(resp), "application/json");
            } catch (const std::exception& e) {
                auto resp = make_error_response(
                    int64_t(0),
                    JsonRpcError::INTERNAL_ERROR,
                    std::string("Internal error: ") + e.what());
                res.set_content(serialize_response(resp), "application/json");
            }
        });

        if (!http_server_->bind_to_port("127.0.0.1", port)) {
            LOG_WARN("MCP HTTP server failed to bind to port {}", port);
            return false;
        }

        listener_thread_ = std::jthread([this, port](std::stop_token /*st*/) {
            LOG_INFO("MCP HTTP server listening on http://127.0.0.1:{}/mcp", port);
            http_server_->listen_after_bind();
        });

        return true;
    }

    void McpHttpServer::stop() {
        if (http_server_) {
            http_server_->stop();
        }
        if (listener_thread_.joinable()) {
            listener_thread_.join();
        }
    }

} // namespace lfs::mcp

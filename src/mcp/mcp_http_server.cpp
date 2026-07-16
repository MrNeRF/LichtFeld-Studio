/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "mcp_http_server.hpp"
#include "mcp_server.hpp"

#include "core/logger.hpp"

#include <httplib/httplib.h>

#include <optional>
#include <type_traits>

namespace lfs::mcp {

    namespace {
        constexpr size_t MAX_MCP_HTTP_BODY_BYTES = 4 * 1024 * 1024;

        // Runs fn, logging (never surfacing) any exception it throws so a
        // single misbehaving request or handler can't take the server down.
        template <typename Fn>
            requires std::is_void_v<std::invoke_result_t<Fn>>
        void try_or_log(const char* log_context, Fn&& fn) {
            try {
                fn();
            } catch (const std::exception& e) {
                LOG_ERROR("{}: {}", log_context, e.what());
            } catch (...) {
                LOG_ERROR("{}: unknown exception", log_context);
            }
        }

        template <typename Fn>
            requires(!std::is_void_v<std::invoke_result_t<Fn>>)
        std::optional<std::invoke_result_t<Fn>> try_or_log(const char* log_context, Fn&& fn) {
            try {
                return fn();
            } catch (const std::exception& e) {
                LOG_ERROR("{}: {}", log_context, e.what());
            } catch (...) {
                LOG_ERROR("{}: unknown exception", log_context);
            }
            return std::nullopt;
        }
    } // namespace

    McpHttpServer::McpHttpServer(const McpServerOptions& server_options)
        : mcp_server_(std::make_unique<McpServer>(server_options)),
          http_server_(std::make_unique<httplib::Server>()) {}

    McpHttpServer::~McpHttpServer() {
        stop();
    }

    bool McpHttpServer::start(int port) {
        http_server_->set_payload_max_length(MAX_MCP_HTTP_BODY_BYTES);
        http_server_->Post("/mcp", [this](const httplib::Request& req, httplib::Response& res) {
            auto rpc_req = try_or_log("MCP request parse failed", [&] { return parse_request(req.body); });
            if (!rpc_req) {
                res.set_content(
                    serialize_response(make_error_response(nullptr, JsonRpcError::PARSE_ERROR, "Parse error")),
                    "application/json");
                return;
            }

            auto rpc_resp = try_or_log("MCP request handler failed", [&] { return mcp_server_->handle_request(*rpc_req); });
            if (!rpc_resp) {
                rpc_resp = make_error_response(rpc_req->id, JsonRpcError::INTERNAL_ERROR, "internal error");
            }
            res.set_content(serialize_response(*rpc_resp), "application/json");
        });

        if (!http_server_->bind_to_port("127.0.0.1", port)) {
            LOG_WARN("MCP HTTP server failed to bind to port {}", port);
            return false;
        }

        listener_thread_ = std::jthread([this, port](std::stop_token /*st*/) {
            LOG_INFO("MCP HTTP server listening on http://127.0.0.1:{}/mcp", port);
            try_or_log("MCP HTTP server listener thread failed", [this] { http_server_->listen_after_bind(); });
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

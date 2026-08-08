/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "mcp_http_server.hpp"
#include "mcp_server.hpp"

#include "core/error.hpp"
#include "core/error_envelope.hpp"
#include "core/error_reporter.hpp"
#include "core/guarded_task.hpp"
#include "core/logger.hpp"

#include <httplib/httplib.h>

#include <format>
#include <mutex>
#include <optional>
#include <type_traits>

namespace lfs::mcp {

    namespace {
        constexpr size_t MAX_MCP_HTTP_BODY_BYTES = 4 * 1024 * 1024;
        std::mutex g_active_server_mutex;
        McpHttpServer* g_active_server = nullptr;
        std::jthread g_config_thread;

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
          http_server_(std::make_unique<httplib::Server>()) {
        http_server_->set_payload_max_length(MAX_MCP_HTTP_BODY_BYTES);
        http_server_->Post("/mcp", [this](const httplib::Request& req, httplib::Response& res) {
            request_count_.fetch_add(1, std::memory_order_relaxed);
            auto rpc_req = try_or_log("MCP request parse failed", [&] { return parse_request(req.body); });
            if (!rpc_req) {
                res.set_content(
                    serialize_response(make_error_response(nullptr, JsonRpcError::PARSE_ERROR, "Parse error")),
                    "application/json");
                return;
            }

            const lfs::OperationId operation_id = lfs::OperationId::generate();
            JsonRpcResponse rpc_resp;
            lfs::core::run_guarded<JsonRpcResponse>(
                lfs::core::TaskContext{
                    .name = "mcp.request",
                    .domain = lfs::ErrorDomain::MCP,
                    .operation_id = operation_id,
                    .site = LFS_SOURCE_SITE_CURRENT(),
                },
                [this, &rpc_req, operation_id]() -> lfs::Result<JsonRpcResponse> {
                    return mcp_server_->handle_request(*rpc_req, operation_id);
                },
                [&rpc_resp, &rpc_req](lfs::Result<JsonRpcResponse>&& result) {
                    if (result) {
                        rpc_resp = std::move(result).value();
                    } else {
                        rpc_resp = make_error_response(rpc_req->id, JsonRpcError::INTERNAL_ERROR,
                                                       "internal error",
                                                       lfs::core::to_wire_envelope(result.error()));
                    }
                });
            res.set_content(serialize_response(rpc_resp), "application/json");
        });
    }

    McpHttpServer::~McpHttpServer() {
        stop();
    }

    bool McpHttpServer::start(const McpHttpConfig& config) {
        std::lock_guard lock(lifecycle_mutex_);
        if (listener_thread_.joinable()) {
            http_server_->stop();
            listener_thread_.join();
        }
        status_ = {
            .enabled = config.enabled,
            .running = false,
            .expose_network = config.expose_network,
            .port = config.port,
        };
        if (!config.enabled)
            return true;
        if (config.port < 1 || config.port > 65535) {
            status_.error = "Port must be between 1 and 65535";
            return false;
        }

        const char* const bind_address = config.expose_network ? "0.0.0.0" : "127.0.0.1";
        if (!http_server_->bind_to_port(bind_address, config.port)) {
            status_.error = std::format("Unable to bind {}:{}", bind_address, config.port);
            LOG_WARN("MCP HTTP server failed to bind to {}:{}", bind_address, config.port);
            return false;
        }

        running_.store(true, std::memory_order_release);
        listener_thread_ = std::jthread([this, address = std::string(bind_address), port = config.port](std::stop_token /*st*/) {
            LOG_INFO("MCP HTTP server listening on http://{}:{}/mcp", address, port);
            lfs::core::run_guarded<void>(
                lfs::core::TaskContext{
                    .name = "mcp.http-listener",
                    .domain = lfs::ErrorDomain::MCP,
                    .operation_id = lfs::OperationId::generate(),
                    .site = LFS_SOURCE_SITE_CURRENT(),
                },
                [this]() -> lfs::Result<void> {
                    http_server_->listen_after_bind();
                    return {};
                },
                [](lfs::Result<void>&& result) {
                    if (!result) {
                        lfs::core::ErrorReporter::get().report(result.error(),
                                                               lfs::core::ReportChannel::OwnerLog);
                    }
                });
            running_.store(false, std::memory_order_release);
        });

        return true;
    }

    void McpHttpServer::stop() {
        std::lock_guard lock(lifecycle_mutex_);
        if (http_server_)
            http_server_->stop();
        if (listener_thread_.joinable())
            listener_thread_.join();
        running_.store(false, std::memory_order_release);
    }

    bool McpHttpServer::applyConfig(const McpHttpConfig& config) {
        stop();
        return start(config);
    }

    McpHttpStatus McpHttpServer::status() const {
        std::lock_guard lock(lifecycle_mutex_);
        auto result = status_;
        result.running = running_.load(std::memory_order_acquire);
        result.request_count = request_count_.load(std::memory_order_relaxed);
        return result;
    }

    void setActiveMcpHttpServer(McpHttpServer* const server) {
        std::lock_guard lock(g_active_server_mutex);
        if (!server && g_config_thread.joinable())
            g_config_thread.join();
        g_active_server = server;
    }

    McpHttpStatus activeMcpHttpStatus() {
        std::lock_guard lock(g_active_server_mutex);
        return g_active_server ? g_active_server->status() : McpHttpStatus{.enabled = false};
    }

    bool applyActiveMcpHttpConfig(const McpHttpConfig& config) {
        std::lock_guard lock(g_active_server_mutex);
        if (!g_active_server)
            return false;
        if (g_config_thread.joinable())
            g_config_thread.join();
        auto* const server = g_active_server;
        g_config_thread = std::jthread([server, config](std::stop_token /*stop_token*/) {
            server->applyConfig(config);
        });
        return true;
    }

} // namespace lfs::mcp

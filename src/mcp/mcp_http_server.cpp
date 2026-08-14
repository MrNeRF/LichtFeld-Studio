/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "mcp_http_server.hpp"
#include "mcp_server.hpp"

#include "core/environment.hpp"
#include "core/error.hpp"
#include "core/error_envelope.hpp"
#include "core/error_reporter.hpp"
#include "core/guarded_task.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "core/user_paths.hpp"

#include <httplib/httplib.h>

#include <algorithm>
#include <chrono>
#include <ctime>
#include <format>
#include <iomanip>
#include <mutex>
#include <optional>
#include <sstream>
#include <type_traits>
#include <vector>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>

#include <iphlpapi.h>
#else
#include <arpa/inet.h>
#include <ifaddrs.h>
#include <net/if.h>
#endif

namespace lfs::mcp {

    namespace {
        constexpr size_t MAX_MCP_HTTP_BODY_BYTES = 4 * 1024 * 1024;
        std::mutex g_active_server_mutex;
        McpHttpServer* g_active_server = nullptr;
        std::jthread g_config_thread;

        std::string sessionTimestamp() {
            const auto now = std::chrono::system_clock::now();
            const std::time_t value = std::chrono::system_clock::to_time_t(now);
            std::tm local{};
#ifdef _WIN32
            localtime_s(&local, &value);
#else
            localtime_r(&value, &local);
#endif
            std::ostringstream stream;
            stream << std::put_time(&local, "%Y%m%d-%H%M%S");
            const auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(
                                          now.time_since_epoch()) %
                                      std::chrono::seconds(1);
            stream << '-' << std::setfill('0') << std::setw(3) << milliseconds.count();
            return stream.str();
        }

        std::vector<std::string> networkEndpoints(const bool exposed, const int port) {
            if (!exposed)
                return {
                    std::format("http://127.0.0.1:{}/mcp", port),
                    std::format("http://localhost:{}/mcp", port),
                };

            std::vector<std::string> addresses;
#ifdef _WIN32
            ULONG size = 0;
            if (GetAdaptersAddresses(AF_INET, GAA_FLAG_SKIP_ANYCAST | GAA_FLAG_SKIP_MULTICAST | GAA_FLAG_SKIP_DNS_SERVER,
                                     nullptr, nullptr, &size) == ERROR_BUFFER_OVERFLOW) {
                std::vector<unsigned char> storage(size);
                auto* adapters = reinterpret_cast<IP_ADAPTER_ADDRESSES*>(storage.data());
                if (GetAdaptersAddresses(AF_INET, GAA_FLAG_SKIP_ANYCAST | GAA_FLAG_SKIP_MULTICAST | GAA_FLAG_SKIP_DNS_SERVER,
                                         nullptr, adapters, &size) == NO_ERROR) {
                    for (auto* adapter = adapters; adapter; adapter = adapter->Next) {
                        if (adapter->OperStatus != IfOperStatusUp || adapter->IfType == IF_TYPE_SOFTWARE_LOOPBACK)
                            continue;
                        for (auto* unicast = adapter->FirstUnicastAddress; unicast; unicast = unicast->Next) {
                            const auto* address = reinterpret_cast<const sockaddr_in*>(unicast->Address.lpSockaddr);
                            char text[INET_ADDRSTRLEN]{};
                            if (address && InetNtopA(AF_INET, &address->sin_addr, text, sizeof(text)))
                                addresses.emplace_back(text);
                        }
                    }
                }
            }
#else
            ifaddrs* interfaces = nullptr;
            if (getifaddrs(&interfaces) == 0) {
                for (auto* interface = interfaces; interface; interface = interface->ifa_next) {
                    if (!interface->ifa_addr || interface->ifa_addr->sa_family != AF_INET ||
                        (interface->ifa_flags & IFF_UP) == 0 ||
                        (interface->ifa_flags & IFF_LOOPBACK) != 0)
                        continue;
                    const auto* address = reinterpret_cast<const sockaddr_in*>(interface->ifa_addr);
                    char text[INET_ADDRSTRLEN]{};
                    if (inet_ntop(AF_INET, &address->sin_addr, text, sizeof(text)))
                        addresses.emplace_back(text);
                }
                freeifaddrs(interfaces);
            }
#endif
            std::erase_if(addresses, [](const std::string& address) {
                return address.starts_with("127.") || address.starts_with("169.254.") || address == "0.0.0.0";
            });
            std::ranges::sort(addresses);
            addresses.erase(std::ranges::unique(addresses).begin(), addresses.end());

            std::vector<std::string> endpoints{
                std::format("http://127.0.0.1:{}/mcp", port),
                std::format("http://localhost:{}/mcp", port),
            };
            endpoints.reserve(addresses.size() + 2);
            for (const auto& address : addresses)
                endpoints.push_back(std::format("http://{}:{}/mcp", address, port));
            return endpoints;
        }

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
        log_session_timestamp_ = sessionTimestamp();
        http_server_->set_payload_max_length(MAX_MCP_HTTP_BODY_BYTES);
        http_server_->Post("/mcp", [this](const httplib::Request& req, httplib::Response& res) {
            const auto request_started = std::chrono::steady_clock::now();
            request_count_.fetch_add(1, std::memory_order_relaxed);
            auto rpc_req = try_or_log("MCP request parse failed", [&] { return parse_request(req.body); });
            if (!rpc_req) {
                error_count_.fetch_add(1, std::memory_order_relaxed);
                appendSessionLog({
                    {"event", "request"},
                    {"outcome", "error"},
                    {"error_code", JsonRpcError::PARSE_ERROR},
                    {"duration_ms", std::chrono::duration<double, std::milli>(
                                        std::chrono::steady_clock::now() - request_started)
                                        .count()},
                });
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
            if (rpc_resp.error)
                error_count_.fetch_add(1, std::memory_order_relaxed);
            else
                success_count_.fetch_add(1, std::memory_order_relaxed);
            json log_event = {
                {"event", "request"},
                {"method", rpc_req->method},
                {"outcome", rpc_resp.error ? "error" : "success"},
                {"duration_ms", std::chrono::duration<double, std::milli>(
                                    std::chrono::steady_clock::now() - request_started)
                                    .count()},
            };
            if (const auto id = rpc_req->id.to_json())
                log_event["request_id"] = *id;
            if (rpc_resp.error)
                log_event["error_code"] = rpc_resp.error->code;
            appendSessionLog(log_event);
            res.set_content(serialize_response(rpc_resp), "application/json");
        });
    }

    void McpHttpServer::appendSessionLog(const json& event) {
        if (!request_logging_.load(std::memory_order_acquire))
            return;
        std::lock_guard lock(log_mutex_);
        auto paths = core::UserPaths::resolve();
        if (!paths) {
            if (!log_failure_reported_) {
                LOG_WARN("Unable to resolve MCP log directory: {}", paths.error());
                log_failure_reported_ = true;
            }
            return;
        }
        if (log_filename_.empty())
            log_filename_ = std::format("{}-mcp.jsonl", log_session_timestamp_);
        auto record = event;
        record["timestamp_ms"] = std::chrono::duration_cast<std::chrono::milliseconds>(
                                     std::chrono::system_clock::now().time_since_epoch())
                                     .count();
        log_contents_ += record.dump();
        log_contents_ += '\n';
        if (const auto result = paths->writeMcpLogAtomically(log_filename_, log_contents_); !result) {
            if (!log_failure_reported_) {
                LOG_WARN("Unable to write MCP session log: {}", result.error());
                log_failure_reported_ = true;
            }
        }
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
        const bool safe_mode = core::environment::flag("LFS_SAFE_MODE", false);
        const bool enabled = config.enabled && !safe_mode;
        const bool request_logging = config.request_logging && !safe_mode;
        status_ = {
            .enabled = enabled,
            .running = false,
            .expose_network = config.expose_network,
            .port = config.port,
            .endpoints = networkEndpoints(config.expose_network, config.port),
            .request_logging = request_logging,
        };
        request_logging_.store(request_logging, std::memory_order_release);
        if (!enabled) {
            last_announced_listener_url_.clear();
            appendSessionLog({{"event", "state"}, {"state", "disabled"}, {"expose_network", config.expose_network}, {"port", config.port}});
            return true;
        }
        if (config.port < 1 || config.port > 65535) {
            last_announced_listener_url_.clear();
            status_.error = "Port must be between 1 and 65535";
            appendSessionLog({{"event", "configuration_error"}, {"reason", "invalid_port"}, {"port", config.port}});
            return false;
        }

        const char* const bind_address = config.expose_network ? "0.0.0.0" : "127.0.0.1";
        if (!http_server_->bind_to_port(bind_address, config.port)) {
            last_announced_listener_url_.clear();
            status_.error = std::format("Unable to bind {}:{}", bind_address, config.port);
            LOG_WARN("MCP HTTP server failed to bind to {}:{}", bind_address, config.port);
            appendSessionLog({{"event", "configuration_error"}, {"reason", "bind_failed"}, {"address", bind_address}, {"port", config.port}});
            return false;
        }

        running_.store(true, std::memory_order_release);
        appendSessionLog({{"event", "state"}, {"state", "started"}, {"address", bind_address}, {"port", config.port}});
        const auto listener_url = std::format("http://{}:{}/mcp", bind_address, config.port);
        const bool announce_listener = listener_url != last_announced_listener_url_;
        if (announce_listener)
            last_announced_listener_url_ = listener_url;
        listener_thread_ = std::jthread([this, listener_url, announce_listener](std::stop_token /*st*/) {
            if (announce_listener)
                LOG_INFO("MCP HTTP server listening on {}", listener_url);
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
        if (running_.load(std::memory_order_acquire)) {
            appendSessionLog({{"event", "state"}, {"state", "stopped"}});
            LOG_INFO("MCP HTTP server stopped");
        }
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

    void McpHttpServer::stageConfig(const McpHttpConfig& config) {
        const bool safe_mode = core::environment::flag("LFS_SAFE_MODE", false);
        std::lock_guard lock(lifecycle_mutex_);
        status_.enabled = config.enabled && !safe_mode;
        status_.expose_network = config.expose_network;
        status_.port = config.port;
        status_.endpoints = networkEndpoints(config.expose_network, config.port);
        status_.request_logging = config.request_logging && !safe_mode;
        status_.error.clear();
        request_logging_.store(status_.request_logging, std::memory_order_release);
    }

    McpHttpStatus McpHttpServer::status() const {
        std::lock_guard lock(lifecycle_mutex_);
        auto result = status_;
        result.running = running_.load(std::memory_order_acquire);
        result.request_count = request_count_.load(std::memory_order_relaxed);
        result.success_count = success_count_.load(std::memory_order_relaxed);
        result.error_count = error_count_.load(std::memory_order_relaxed);
        result.request_logging = request_logging_.load(std::memory_order_acquire);
        std::lock_guard log_lock(log_mutex_);
        if (!log_filename_.empty()) {
            if (const auto paths = core::UserPaths::resolve())
                result.log_file = core::path_to_utf8(paths->mcpLogDir() / log_filename_);
        }
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
        server->stageConfig(config);
        g_config_thread = std::jthread([server, config](std::stop_token /*stop_token*/) {
            server->applyConfig(config);
        });
        return true;
    }

} // namespace lfs::mcp

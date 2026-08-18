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
#include <cerrno>
#include <chrono>
#include <ctime>
#include <exception>
#include <format>
#include <iomanip>
#include <mutex>
#include <optional>
#include <sstream>
#include <type_traits>
#include <utility>
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
        std::condition_variable_any g_config_cv;
        McpHttpServer* g_active_server = nullptr;
        std::optional<McpHttpConfig> g_pending_config;
        std::jthread g_config_worker;

        void configureSingleOwnerListenerSocket(const socket_t socket) {
#ifdef _WIN32
            constexpr int enabled = 1;
            if (::setsockopt(socket,
                             SOL_SOCKET,
                             SO_EXCLUSIVEADDRUSE,
                             reinterpret_cast<const char*>(&enabled),
                             sizeof(enabled)) != 0) {
                LOG_WARN("MCP HTTP listener could not enable exclusive port ownership: WSA {}",
                         WSAGetLastError());
            }
#else
            // Keep fast rebinding after a clean restart without enabling
            // SO_REUSEPORT, which would allow multiple MCP listeners to share
            // the same address and port.
            constexpr int enabled = 1;
            if (::setsockopt(socket,
                             SOL_SOCKET,
                             SO_REUSEADDR,
                             &enabled,
                             sizeof(enabled)) != 0) {
                LOG_WARN("MCP HTTP listener could not enable address reuse: errno {}", errno);
            }
#endif
        }

        class LifecycleTransition final {
        public:
            LifecycleTransition(std::mutex& mutex,
                                std::condition_variable& cv,
                                bool& active)
                : mutex_(mutex), cv_(cv), active_(active) {
                std::unique_lock lock(mutex_);
                cv_.wait(lock, [this] { return !active_; });
                active_ = true;
            }

            ~LifecycleTransition() {
                {
                    std::lock_guard lock(mutex_);
                    active_ = false;
                }
                cv_.notify_all();
            }

            LifecycleTransition(const LifecycleTransition&) = delete;
            LifecycleTransition& operator=(const LifecycleTransition&) = delete;

        private:
            std::mutex& mutex_;
            std::condition_variable& cv_;
            bool& active_;
        };

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

        std::vector<std::string> loopbackEndpoints(const int port) {
            return {
                std::format("http://127.0.0.1:{}/mcp", port),
                std::format("http://localhost:{}/mcp", port),
            };
        }

        std::vector<std::string> networkEndpoints(const bool exposed, const int port) {
            if (!exposed)
                return loopbackEndpoints(port);

            std::vector<std::string> addresses;
#ifdef _WIN32
            ULONG size = 0;
            if (GetAdaptersAddresses(AF_INET,
                                     GAA_FLAG_SKIP_ANYCAST | GAA_FLAG_SKIP_MULTICAST |
                                         GAA_FLAG_SKIP_DNS_SERVER,
                                     nullptr, nullptr, &size) == ERROR_BUFFER_OVERFLOW) {
                std::vector<unsigned char> storage(size);
                auto* adapters = reinterpret_cast<IP_ADAPTER_ADDRESSES*>(storage.data());
                if (GetAdaptersAddresses(AF_INET,
                                         GAA_FLAG_SKIP_ANYCAST | GAA_FLAG_SKIP_MULTICAST |
                                             GAA_FLAG_SKIP_DNS_SERVER,
                                         nullptr, adapters, &size) == NO_ERROR) {
                    for (auto* adapter = adapters; adapter; adapter = adapter->Next) {
                        if (adapter->OperStatus != IfOperStatusUp ||
                            adapter->IfType == IF_TYPE_SOFTWARE_LOOPBACK)
                            continue;
                        for (auto* unicast = adapter->FirstUnicastAddress; unicast;
                             unicast = unicast->Next) {
                            const auto* address = reinterpret_cast<const sockaddr_in*>(
                                unicast->Address.lpSockaddr);
                            char text[INET_ADDRSTRLEN]{};
                            if (address &&
                                InetNtopA(AF_INET, &address->sin_addr, text, sizeof(text)))
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
                    const auto* address =
                        reinterpret_cast<const sockaddr_in*>(interface->ifa_addr);
                    char text[INET_ADDRSTRLEN]{};
                    if (inet_ntop(AF_INET, &address->sin_addr, text, sizeof(text)))
                        addresses.emplace_back(text);
                }
                freeifaddrs(interfaces);
            }
#endif
            std::erase_if(addresses, [](const std::string& address) {
                return address.starts_with("127.") || address.starts_with("169.254.") ||
                       address == "0.0.0.0";
            });
            std::ranges::sort(addresses);
            addresses.erase(std::ranges::unique(addresses).begin(), addresses.end());

            auto endpoints = loopbackEndpoints(port);
            endpoints.reserve(addresses.size() + 2);
            for (const auto& address : addresses)
                endpoints.push_back(std::format("http://{}:{}/mcp", address, port));
            return endpoints;
        }

        bool responseCountsAsError(const JsonRpcRequest& request,
                                   const JsonRpcResponse& response) {
            if (response.error)
                return true;

            // MCP tool execution failures intentionally use a successful
            // JSON-RPC envelope and report their outcome in CallToolResult.
            if (request.method != "tools/call" || !response.result ||
                !response.result->is_object())
                return false;

            const auto is_error = response.result->find("isError");
            return is_error != response.result->end() && is_error->is_boolean() &&
                   is_error->get<bool>();
        }

        const char* jsonRpcErrorReason(const int code) {
            switch (code) {
            case JsonRpcError::PARSE_ERROR:
                return "parse_error";
            case JsonRpcError::INVALID_REQUEST:
                return "invalid_request";
            case JsonRpcError::METHOD_NOT_FOUND:
                return "method_not_found";
            case JsonRpcError::INVALID_PARAMS:
                return "invalid_params";
            case JsonRpcError::INTERNAL_ERROR:
                return "internal_error";
            default:
                return "json_rpc_error";
            }
        }

        const char* jsonRpcErrorStage(const int code) {
            switch (code) {
            case JsonRpcError::PARSE_ERROR:
                return "parse";
            case JsonRpcError::INVALID_REQUEST:
            case JsonRpcError::INVALID_PARAMS:
                return "validation";
            case JsonRpcError::METHOD_NOT_FOUND:
                return "dispatch";
            case JsonRpcError::INTERNAL_ERROR:
                return "execution";
            default:
                return "protocol";
            }
        }

        void appendWireErrorMetadata(nlohmann::json& event,
                                     const nlohmann::json& envelope) {
            if (!envelope.is_object())
                return;

            if (const auto code = envelope.find("code");
                code != envelope.end() && code->is_string())
                event["application_error_code"] = *code;
            if (const auto domain = envelope.find("domain");
                domain != envelope.end() && domain->is_string())
                event["application_error_domain"] = *domain;
            if (const auto retryable = envelope.find("retryable");
                retryable != envelope.end() && retryable->is_boolean())
                event["retryable"] = *retryable;
            if (const auto operation_id = envelope.find("operation_id");
                operation_id != envelope.end() && operation_id->is_string())
                event["operation_id"] = *operation_id;
        }

        void appendTransportMetadata(nlohmann::json& event,
                                     const httplib::Request& request) {
            if (!request.remote_addr.empty())
                event["source_ip"] = request.remote_addr;
            if (request.remote_port >= 0)
                event["source_port"] = request.remote_port;
            if (!request.local_addr.empty())
                event["destination_ip"] = request.local_addr;
            if (request.local_port >= 0)
                event["destination_port"] = request.local_port;
        }

        void appendResponseErrorMetadata(nlohmann::json& event,
                                         const JsonRpcRequest& request,
                                         const JsonRpcResponse& response) {
            if (response.error) {
                event["error_type"] = "json_rpc";
                event["error_stage"] = jsonRpcErrorStage(response.error->code);
                event["error_reason"] = jsonRpcErrorReason(response.error->code);
                event["jsonrpc_error_code"] = response.error->code;
                // Retain the original field for compatibility with existing logs.
                event["error_code"] = response.error->code;
                if (response.error->data)
                    appendWireErrorMetadata(event, *response.error->data);
                return;
            }

            if (request.method != "tools/call" || !response.result ||
                !response.result->is_object())
                return;

            event["error_type"] = "tool_execution";
            event["error_stage"] = "tool_call";
            event["error_reason"] = "tool_reported_error";

            const auto structured = response.result->find("structuredContent");
            if (structured == response.result->end() || !structured->is_object())
                return;
            const auto error = structured->find("error");
            if (error == structured->end())
                return;
            appendWireErrorMetadata(event, *error);
            if (error->is_object()) {
                const auto code = error->find("code");
                if (code != error->end() && code->is_string())
                    event["error_reason"] = *code;
            }
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
        http_server_->set_socket_options(configureSingleOwnerListenerSocket);
        http_server_->set_payload_max_length(MAX_MCP_HTTP_BODY_BYTES);
        http_server_->Post("/mcp", [this](const httplib::Request& req, httplib::Response& res) {
            const auto request_started = std::chrono::steady_clock::now();
            request_count_.fetch_add(1, std::memory_order_relaxed);
            auto rpc_req = try_or_log("MCP request parse failed", [&] {
                return parse_request(req.body);
            });
            if (!rpc_req) {
                error_count_.fetch_add(1, std::memory_order_relaxed);
                nlohmann::json log_event = {
                    {"event", "request"},
                    {"outcome", "error"},
                    {"error_type", "json_rpc"},
                    {"error_stage", "parse"},
                    {"error_reason", "parse_error"},
                    {"jsonrpc_error_code", JsonRpcError::PARSE_ERROR},
                    {"error_code", JsonRpcError::PARSE_ERROR},
                    {"duration_ms", std::chrono::duration<double, std::milli>(
                                        std::chrono::steady_clock::now() - request_started)
                                        .count()},
                };
                appendTransportMetadata(log_event, req);
                appendSessionLog(log_event);
                res.set_content(
                    serialize_response(make_error_response(
                        nullptr, JsonRpcError::PARSE_ERROR, "Parse error")),
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
                        rpc_resp = make_error_response(
                            rpc_req->id, JsonRpcError::INTERNAL_ERROR, "internal error",
                            lfs::core::to_wire_envelope(result.error()));
                    }
                });
            const bool request_failed = responseCountsAsError(*rpc_req, rpc_resp);
            if (request_failed)
                error_count_.fetch_add(1, std::memory_order_relaxed);
            else
                success_count_.fetch_add(1, std::memory_order_relaxed);

            nlohmann::json log_event = {
                {"event", "request"},
                {"method", rpc_req->method},
                {"outcome", request_failed ? "error" : "success"},
                {"duration_ms", std::chrono::duration<double, std::milli>(
                                    std::chrono::steady_clock::now() - request_started)
                                    .count()},
            };
            if (const auto id = rpc_req->id.to_json())
                log_event["request_id"] = *id;
            appendTransportMetadata(log_event, req);
            if (request_failed)
                appendResponseErrorMetadata(log_event, *rpc_req, rpc_resp);
            appendSessionLog(log_event);
            res.set_content(serialize_response(rpc_resp), "application/json");
        });
    }

    void McpHttpServer::appendSessionLog(const nlohmann::json& event) {
        if (!request_logging_.load(std::memory_order_acquire))
            return;
        auto paths = core::UserPaths::resolve();
        if (!paths) {
            bool report_failure = false;
            {
                std::lock_guard state_lock(log_state_mutex_);
                report_failure = !log_failure_reported_;
                log_failure_reported_ = true;
            }
            if (report_failure) {
                LOG_WARN("Unable to resolve MCP log directory: {}",
                         lfs::format_for_developer(paths.error()));
            }
            return;
        }
        std::string log_filename;
        {
            std::lock_guard state_lock(log_state_mutex_);
            if (log_filename_.empty()) {
                log_filename_ = std::format("{}-mcp.jsonl", log_session_timestamp_);
                log_file_path_ = core::path_to_utf8(paths->mcpLogDir() / log_filename_);
            }
            log_filename = log_filename_;
        }
        auto record = event;
        record["timestamp_ms"] = std::chrono::duration_cast<std::chrono::milliseconds>(
                                     std::chrono::system_clock::now().time_since_epoch())
                                     .count();
        const auto result = [&] {
            std::lock_guard write_lock(log_write_mutex_);
            return paths->appendMcpLogLine(log_filename, record.dump());
        }();
        if (!result) {
            bool report_failure = false;
            {
                std::lock_guard state_lock(log_state_mutex_);
                report_failure = !log_failure_reported_;
                log_failure_reported_ = true;
            }
            if (!report_failure)
                return;
            LOG_WARN("Unable to write MCP session log: {}",
                     lfs::format_for_developer(result.error()));
        }
    }

    McpHttpServer::~McpHttpServer() {
        stop();
    }

    void McpHttpServer::stopListenerAndJoin() {
        if (listener_thread_.joinable()) {
            // bind_to_port() and listen_after_bind() are separate in cpp-httplib.
            // Wait until the listener has entered its accept loop (or failed)
            // before stop(): cpp-httplib ignores stop() while startup is pending.
            http_server_->wait_until_ready();
            // stop() invalidates-then-closes the listener socket exactly once.
            // After join, a second stop() only resets the decommissioned flag so
            // the same endpoint can be bound again.
            http_server_->stop();
            listener_thread_.join();
            http_server_->stop();
        } else if (http_server_) {
            // stop() also clears cpp-httplib's decommissioned flag after a bind
            // failure, which is required before retrying the same endpoint.
            http_server_->stop();
        }
    }

    bool McpHttpServer::start(const McpHttpConfig& config) {
        LifecycleTransition transition(lifecycle_mutex_, lifecycle_cv_,
                                       lifecycle_transition_active_);
        const auto listener_generation =
            listener_generation_.fetch_add(1, std::memory_order_acq_rel) + 1;
        stopListenerAndJoin();

        const bool safe_mode = core::environment::flag("LFS_SAFE_MODE", false);
        const bool enabled = config.enabled && !safe_mode;
        const bool request_logging = config.request_logging && !safe_mode;
        auto endpoints = networkEndpoints(config.expose_network, config.port);
        {
            std::lock_guard status_lock(status_mutex_);
            status_ = {
                .enabled = enabled,
                .running = false,
                .phase = enabled ? McpHttpPhase::Starting : McpHttpPhase::Disabled,
                .expose_network = config.expose_network,
                .port = config.port,
                .endpoints = std::move(endpoints),
                .request_logging = request_logging,
            };
        }
        has_applied_config_ = false;
        request_logging_.store(request_logging, std::memory_order_release);

        if (!enabled) {
            applied_config_ = {
                .enabled = false,
                .expose_network = config.expose_network,
                .port = config.port,
                .request_logging = false,
            };
            has_applied_config_ = true;
            last_announced_listener_url_.clear();
            appendSessionLog({
                {"event", "state"},
                {"state", "disabled"},
                {"expose_network", config.expose_network},
                {"port", config.port},
            });
            return true;
        }
        if (config.port < 1 || config.port > 65535) {
            last_announced_listener_url_.clear();
            {
                std::lock_guard status_lock(status_mutex_);
                status_.phase = McpHttpPhase::Failed;
                status_.error = "Port must be between 1 and 65535";
                status_.error_kind = McpHttpErrorKind::InvalidPort;
                status_.error_port = config.port;
            }
            appendSessionLog({
                {"event", "configuration_error"},
                {"reason", "invalid_port"},
                {"port", config.port},
            });
            return false;
        }

        const char* const bind_address = config.expose_network ? "0.0.0.0" : "127.0.0.1";
        if (!http_server_->bind_to_port(bind_address, config.port)) {
            // A failed bind decommissions cpp-httplib until stop() resets it.
            http_server_->stop();
            last_announced_listener_url_.clear();
            {
                std::lock_guard status_lock(status_mutex_);
                status_.phase = McpHttpPhase::Failed;
                status_.error = std::format("Unable to bind {}:{}", bind_address, config.port);
                status_.error_kind = McpHttpErrorKind::BindFailed;
                status_.error_address = bind_address;
                status_.error_port = config.port;
            }
            LOG_WARN("MCP HTTP server failed to bind to {}:{}", bind_address, config.port);
            appendSessionLog({
                {"event", "configuration_error"},
                {"reason", "bind_failed"},
                {"address", bind_address},
                {"port", config.port},
            });
            return false;
        }

        const auto listener_url = std::format("http://{}:{}/mcp", bind_address, config.port);
        const bool announce_listener = listener_url != last_announced_listener_url_;
        listener_thread_ = std::jthread([this, listener_url, announce_listener,
                                         listener_generation](
                                            std::stop_token /*st*/) {
            if (announce_listener)
                LOG_INFO("MCP HTTP server listening on {}", listener_url);
            bool listener_completed_normally = false;
            lfs::core::run_guarded<void>(
                lfs::core::TaskContext{
                    .name = "mcp.http-listener",
                    .domain = lfs::ErrorDomain::MCP,
                    .operation_id = lfs::OperationId::generate(),
                    .site = LFS_SOURCE_SITE_CURRENT(),
                },
                [this, &listener_completed_normally]() -> lfs::Result<void> {
                    listener_completed_normally = http_server_->listen_after_bind();
                    return {};
                },
                [](lfs::Result<void>&& result) {
                    if (!result) {
                        lfs::core::ErrorReporter::get().report(result.error(),
                                                               lfs::core::ReportChannel::OwnerLog);
                    }
                });
            // cpp-httplib's readiness wait observes only is_running_ and its
            // decommissioned flag. Ensure an exception before normal teardown
            // cannot leave wait_until_ready() spinning forever.
            if (!http_server_->is_running())
                http_server_->decommission();
            std::lock_guard status_lock(status_mutex_);
            if (listener_generation ==
                listener_generation_.load(std::memory_order_acquire)) {
                status_.running = false;
            }
            if (listener_generation ==
                    listener_generation_.load(std::memory_order_acquire) &&
                status_.phase != McpHttpPhase::Stopping &&
                status_.phase != McpHttpPhase::Disabled) {
                status_.phase = McpHttpPhase::Failed;
                status_.error_kind = McpHttpErrorKind::ListenerFailed;
                status_.error = listener_completed_normally
                                    ? "MCP HTTP listener stopped unexpectedly"
                                    : "MCP HTTP listener failed";
            }
        });

        // Do not publish Running until listen_after_bind() has entered its
        // accept loop. This closes the enable-then-disable race in cpp-httplib.
        http_server_->wait_until_ready();
        bool listener_running = false;
        {
            std::lock_guard status_lock(status_mutex_);
            listener_running =
                listener_generation == listener_generation_.load(std::memory_order_acquire) &&
                http_server_->is_running() && status_.phase != McpHttpPhase::Failed;
            if (listener_running) {
                status_.running = true;
                status_.phase = McpHttpPhase::Running;
                status_.error.clear();
                status_.error_kind = McpHttpErrorKind::None;
                status_.error_address.clear();
                status_.error_port = 0;
            } else {
                status_.running = false;
                status_.phase = McpHttpPhase::Failed;
                status_.error_kind = McpHttpErrorKind::ListenerFailed;
                if (status_.error.empty())
                    status_.error = "MCP HTTP listener failed to start";
            }
        }
        if (!listener_running) {
            if (listener_thread_.joinable())
                listener_thread_.join();
            http_server_->stop();
            return false;
        }

        last_announced_listener_url_ = listener_url;
        applied_config_ = {
            .enabled = true,
            .expose_network = config.expose_network,
            .port = config.port,
            .request_logging = request_logging,
        };
        has_applied_config_ = true;
        appendSessionLog({
            {"event", "state"},
            {"state", "started"},
            {"address", bind_address},
            {"port", config.port},
        });

        return true;
    }

    void McpHttpServer::stop() {
        LifecycleTransition transition(lifecycle_mutex_, lifecycle_cv_,
                                       lifecycle_transition_active_);
        listener_generation_.fetch_add(1, std::memory_order_acq_rel);
        bool was_running = false;
        {
            std::lock_guard status_lock(status_mutex_);
            was_running = status_.running;
            if (was_running)
                status_.phase = McpHttpPhase::Stopping;
        }
        if (was_running) {
            appendSessionLog({{"event", "state"}, {"state", "stopped"}});
            LOG_INFO("MCP HTTP server stopped");
        }
        stopListenerAndJoin();
        has_applied_config_ = false;
        {
            std::lock_guard status_lock(status_mutex_);
            status_.enabled = false;
            status_.running = false;
            status_.phase = McpHttpPhase::Disabled;
            status_.error.clear();
            status_.error_kind = McpHttpErrorKind::None;
            status_.error_address.clear();
            status_.error_port = 0;
        }
    }

    bool McpHttpServer::applyConfig(const McpHttpConfig& config) {
        const bool safe_mode = core::environment::flag("LFS_SAFE_MODE", false);
        const bool effective_enabled = config.enabled && !safe_mode;
        const bool effective_logging = config.request_logging && !safe_mode;
        {
            std::unique_lock lock(lifecycle_mutex_);
            lifecycle_cv_.wait(lock, [this] { return !lifecycle_transition_active_; });
            const bool listener_healthy =
                effective_enabled ? http_server_->is_running() : true;
            if (has_applied_config_ && applied_config_.enabled == effective_enabled &&
                applied_config_.expose_network == config.expose_network &&
                applied_config_.port == config.port && listener_healthy) {
                auto endpoints = networkEndpoints(config.expose_network, config.port);
                {
                    std::lock_guard status_lock(status_mutex_);
                    status_.enabled = effective_enabled;
                    status_.running = effective_enabled;
                    status_.phase = effective_enabled ? McpHttpPhase::Running
                                                      : McpHttpPhase::Disabled;
                    status_.request_logging = effective_logging;
                    status_.endpoints = std::move(endpoints);
                }
                applied_config_.request_logging = effective_logging;
                request_logging_.store(effective_logging, std::memory_order_release);
                return true;
            }
        }
        return start(config);
    }

    void McpHttpServer::stageConfig(const McpHttpConfig& config) {
        const bool safe_mode = core::environment::flag("LFS_SAFE_MODE", false);
        // Staging may run on any caller thread. Keep it allocation-only (no
        // adapter walk / no I/O); the worker publishes the complete endpoint
        // list when it applies the configuration.
        auto endpoints = loopbackEndpoints(config.port);
        std::lock_guard status_lock(status_mutex_);
        const bool was_running = status_.running;
        const bool enabled = config.enabled && !safe_mode;
        const bool runtime_unchanged =
            status_.enabled == enabled &&
            status_.expose_network == config.expose_network &&
            status_.port == config.port &&
            ((enabled && status_.phase == McpHttpPhase::Running) ||
             (!enabled && status_.phase == McpHttpPhase::Disabled));
        status_.enabled = enabled;
        if (!runtime_unchanged) {
            status_.running = enabled ? false : was_running;
            status_.phase = enabled
                                ? McpHttpPhase::Starting
                                : (was_running ? McpHttpPhase::Stopping
                                               : McpHttpPhase::Disabled);
        }
        status_.expose_network = config.expose_network;
        status_.port = config.port;
        status_.endpoints = std::move(endpoints);
        status_.request_logging = config.request_logging && !safe_mode;
        status_.error.clear();
        status_.error_kind = McpHttpErrorKind::None;
        status_.error_address.clear();
        status_.error_port = 0;
        request_logging_.store(status_.request_logging, std::memory_order_release);
    }

    McpHttpStatus McpHttpServer::status() const {
        McpHttpStatus result;
        {
            std::lock_guard status_lock(status_mutex_);
            result = status_;
        }
        result.request_count = request_count_.load(std::memory_order_relaxed);
        result.success_count = success_count_.load(std::memory_order_relaxed);
        result.error_count = error_count_.load(std::memory_order_relaxed);
        result.request_logging = request_logging_.load(std::memory_order_acquire);
        {
            std::lock_guard state_lock(log_state_mutex_);
            result.log_file = log_file_path_;
        }
        return result;
    }

    void McpHttpServer::reportConfigWorkerFailure() {
        std::lock_guard status_lock(status_mutex_);
        status_.running = false;
        status_.phase = McpHttpPhase::Failed;
        status_.error = "MCP HTTP configuration failed";
        status_.error_kind = McpHttpErrorKind::ListenerFailed;
        status_.error_address.clear();
        status_.error_port = status_.port;
    }

    void setActiveMcpHttpServer(McpHttpServer* const server) {
        std::jthread worker_to_join;
        McpHttpServer* detached_server = nullptr;
        {
            std::lock_guard lock(g_active_server_mutex);
            if (!server) {
                detached_server = std::exchange(g_active_server, nullptr);
                g_pending_config.reset();
                if (g_config_worker.joinable()) {
                    g_config_worker.request_stop();
                    worker_to_join = std::move(g_config_worker);
                }
            } else {
                g_active_server = server;
                g_pending_config.reset();
            }
            if (server && !g_config_worker.joinable()) {
                g_config_worker = std::jthread([](const std::stop_token stop_token) {
                    while (!stop_token.stop_requested()) {
                        McpHttpServer* active_server = nullptr;
                        McpHttpConfig config;
                        {
                            std::unique_lock lock(g_active_server_mutex);
                            if (!g_config_cv.wait(lock, stop_token, [] {
                                    return g_pending_config.has_value();
                                }))
                                break;
                            active_server = g_active_server;
                            config = *g_pending_config;
                            g_pending_config.reset();
                        }
                        if (!active_server)
                            continue;
                        try {
                            active_server->applyConfig(config);
                        } catch (const std::exception& e) {
                            LOG_ERROR("MCP configuration worker recovered from an exception: {}",
                                      e.what());
                            try {
                                active_server->stop();
                            } catch (...) {
                                // LFS-CENSUS-OK(empty-catch): best-effort listener stop during failure recovery; the outer handler already logged and reportConfigWorkerFailure records the state.
                            }
                            active_server->reportConfigWorkerFailure();
                        } catch (...) {
                            LOG_ERROR("MCP configuration worker recovered from an unknown exception");
                            try {
                                active_server->stop();
                            } catch (...) {
                                // LFS-CENSUS-OK(empty-catch): best-effort listener stop during failure recovery; the outer handler already logged and reportConfigWorkerFailure records the state.
                            }
                            active_server->reportConfigWorkerFailure();
                        }
                    }
                });
            }
        }
        g_config_cv.notify_all();
        if (worker_to_join.joinable())
            worker_to_join.join();
        // Detach is an explicit cancellation boundary: discard queued work,
        // wait for any in-flight transition, and leave the outgoing listener
        // stopped. This makes shutdown deterministic even if an enable was
        // already being applied when the final disable was queued.
        if (detached_server) {
            try {
                detached_server->stop();
            } catch (const std::exception& e) {
                LOG_ERROR("MCP detach failed while stopping the listener: {}", e.what());
                detached_server->reportConfigWorkerFailure();
            } catch (...) {
                LOG_ERROR("MCP detach failed while stopping the listener");
                detached_server->reportConfigWorkerFailure();
            }
        }
    }

    McpHttpStatus activeMcpHttpStatus() {
        McpHttpServer* active_server = nullptr;
        {
            std::lock_guard lock(g_active_server_mutex);
            active_server = g_active_server;
        }
        return active_server ? active_server->status()
                             : McpHttpStatus{
                                   .enabled = false,
                                   .phase = McpHttpPhase::Disabled,
                               };
    }

    bool applyActiveMcpHttpConfig(const McpHttpConfig& config) {
        McpHttpServer* active_server = nullptr;
        {
            std::lock_guard lock(g_active_server_mutex);
            if (!g_active_server || !g_config_worker.joinable())
                return false;
            active_server = g_active_server;
        }
        active_server->stageConfig(config);
        {
            std::lock_guard lock(g_active_server_mutex);
            if (g_active_server != active_server || !g_config_worker.joinable())
                return false;
            // Coalesce rapid UI changes; the worker applies only the newest
            // configuration still pending when it becomes available.
            g_pending_config = config;
        }
        g_config_cv.notify_one();
        return true;
    }

} // namespace lfs::mcp

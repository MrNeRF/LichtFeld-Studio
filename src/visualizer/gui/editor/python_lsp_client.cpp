/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "python_lsp_client.hpp"

#include "stdio_process.hpp"

#include <core/logger.hpp>
#include <core/path_utils.hpp>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <numeric>
#include <optional>
#include <stop_token>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#ifndef _WIN32
#include <unistd.h>
#else
#include <processthreadsapi.h>
#include <windows.h>
#endif

namespace lfs::vis::editor {

    namespace {

        using json = nlohmann::json;
        namespace fs = std::filesystem;

        constexpr auto WORKER_IDLE_WAIT = std::chrono::milliseconds(16);
        constexpr auto WORKER_ACTIVE_WAIT = std::chrono::milliseconds(2);
        constexpr auto RESTART_DELAY = std::chrono::milliseconds(400);
        constexpr auto INIT_TIMEOUT = std::chrono::seconds(5);

        struct ServerCommand {
            std::string program;
            std::vector<std::string> args;
            std::string label;
        };

        struct PendingCompletionRequest {
            int document_version = 0;
            int line = 0;
            int character = 0;
            bool manual = false;
            std::string trigger_character;
        };

        struct PendingSemanticTokensRequest {
            int document_version = 0;
        };

        json semantic_token_types_capability() {
            return json::array({
                "namespace",
                "type",
                "class",
                "enum",
                "interface",
                "struct",
                "typeParameter",
                "parameter",
                "variable",
                "property",
                "enumMember",
                "event",
                "function",
                "method",
                "macro",
                "keyword",
                "modifier",
                "comment",
                "string",
                "number",
                "regexp",
                "operator",
                "decorator",
            });
        }

        json semantic_token_modifiers_capability() {
            return json::array({
                "declaration",
                "definition",
                "readonly",
                "static",
                "deprecated",
                "abstract",
                "async",
                "modification",
                "documentation",
                "defaultLibrary",
            });
        }

        bool is_unreserved(const unsigned char ch) {
            return (ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z') ||
                   (ch >= '0' && ch <= '9') || ch == '-' || ch == '_' || ch == '.' ||
                   ch == '~' || ch == '/' || ch == ':';
        }

        std::string percent_encode(std::string_view text) {
            static constexpr std::array<char, 16> HEX = {
                '0',
                '1',
                '2',
                '3',
                '4',
                '5',
                '6',
                '7',
                '8',
                '9',
                'A',
                'B',
                'C',
                'D',
                'E',
                'F',
            };

            std::string encoded;
            encoded.reserve(text.size());
            for (const unsigned char ch : text) {
                if (is_unreserved(ch)) {
                    encoded.push_back(static_cast<char>(ch));
                    continue;
                }

                encoded.push_back('%');
                encoded.push_back(HEX[(ch >> 4) & 0xF]);
                encoded.push_back(HEX[ch & 0xF]);
            }
            return encoded;
        }

        std::string file_uri_from_path(const fs::path& path) {
            fs::path absolute = path;
            std::error_code ec;
            if (absolute.is_relative()) {
                absolute = fs::absolute(absolute, ec);
            }
            if (!ec) {
                absolute = absolute.lexically_normal();
            }

            std::string utf8 = lfs::core::path_to_utf8(absolute);
#ifdef _WIN32
            return "file:///" + percent_encode(fs::path(utf8).generic_string());
#else
            return "file://" + percent_encode(fs::path(utf8).generic_string());
#endif
        }

        void append_candidate(std::vector<ServerCommand>& commands,
                              std::unordered_set<std::string>& seen,
                              ServerCommand command) {
            const std::string key = command.program + '\n' + std::accumulate(command.args.begin(), command.args.end(), std::string(), [](std::string acc, const std::string& arg) {
                                        acc += arg;
                                        acc.push_back('\n');
                                        return acc;
                                    });
            if (seen.insert(key).second) {
                commands.push_back(std::move(command));
            }
        }

        bool is_executable_file(const fs::path& path) {
            std::error_code ec;
            if (!fs::exists(path, ec) || ec) {
                return false;
            }
#ifdef _WIN32
            return fs::is_regular_file(path, ec);
#else
            return ::access(path.c_str(), X_OK) == 0;
#endif
        }

        std::optional<std::string> resolve_program_path(const std::string_view program) {
            if (program.empty()) {
                return std::nullopt;
            }

            const fs::path candidate(program);
            if (candidate.has_parent_path() || candidate.is_absolute()) {
                if (!is_executable_file(candidate)) {
                    return std::nullopt;
                }
                return lfs::core::path_to_utf8(candidate);
            }

#ifdef _WIN32
            const DWORD buffer_size = SearchPathA(nullptr, std::string(program).c_str(), nullptr, 0, nullptr, nullptr);
            if (buffer_size == 0) {
                return std::nullopt;
            }

            std::string buffer(static_cast<size_t>(buffer_size), '\0');
            if (SearchPathA(nullptr, std::string(program).c_str(), nullptr, buffer_size, buffer.data(), nullptr) == 0) {
                return std::nullopt;
            }
            if (!buffer.empty() && buffer.back() == '\0') {
                buffer.pop_back();
            }
            return buffer;
#else
            const char* const path_env = std::getenv("PATH");
            if (!path_env || !*path_env) {
                return std::nullopt;
            }

            std::string_view remaining(path_env);
            while (true) {
                const size_t separator = remaining.find(':');
                const std::string_view entry = remaining.substr(0, separator);
                const fs::path dir = entry.empty() ? fs::current_path() : fs::path(entry);
                const fs::path resolved = dir / candidate;
                if (is_executable_file(resolved)) {
                    return lfs::core::path_to_utf8(resolved);
                }

                if (separator == std::string_view::npos) {
                    break;
                }
                remaining.remove_prefix(separator + 1);
            }

            return std::nullopt;
#endif
        }

        std::vector<ServerCommand> discover_server_commands() {
            std::vector<ServerCommand> commands;
            std::unordered_set<std::string> seen;

            if (const char* override_program = std::getenv("LFS_PYTHON_LSP");
                override_program && *override_program) {
                append_candidate(commands, seen,
                                 {.program = override_program,
                                  .args = {},
                                  .label = "custom Python language server"});
                return commands;
            }

            const fs::path project_root(PROJECT_ROOT_PATH);
#ifdef _WIN32
            constexpr std::array<std::string_view, 4> DIRECT_EXECUTABLES = {
                "py/.venv/Scripts/basedpyright-langserver.exe",
                ".venv/Scripts/basedpyright-langserver.exe",
                "py/.venv/Scripts/pyright-langserver.exe",
                ".venv/Scripts/pyright-langserver.exe",
            };
#else
            constexpr std::array<std::string_view, 4> DIRECT_EXECUTABLES = {
                "py/.venv/bin/basedpyright-langserver",
                ".venv/bin/basedpyright-langserver",
                "py/.venv/bin/pyright-langserver",
                ".venv/bin/pyright-langserver",
            };
#endif

            for (const auto relative : DIRECT_EXECUTABLES) {
                const fs::path candidate = project_root / relative;
                if (!is_executable_file(candidate)) {
                    continue;
                }

                const bool is_basedpyright =
                    candidate.filename().string().find("basedpyright") != std::string::npos;
                append_candidate(commands, seen,
                                 {.program = lfs::core::path_to_utf8(candidate),
                                  .args = {"--stdio"},
                                  .label = is_basedpyright ? "basedpyright" : "pyright"});
            }

            if (const auto resolved = resolve_program_path("basedpyright-langserver")) {
                append_candidate(commands, seen,
                                 {.program = *resolved,
                                  .args = {"--stdio"},
                                  .label = "basedpyright"});
            }
            if (const auto resolved = resolve_program_path("pyright-langserver")) {
                append_candidate(commands, seen,
                                 {.program = *resolved,
                                  .args = {"--stdio"},
                                  .label = "pyright"});
            }
            if (const auto resolved = resolve_program_path("uv")) {
                append_candidate(commands, seen,
                                 {.program = *resolved,
                                  .args = {"tool", "run", "--from", "basedpyright",
                                           "basedpyright-langserver", "--stdio"},
                                  .label = "basedpyright via uv"});
            }
            return commands;
        }

        uint64_t current_process_id() {
#ifdef _WIN32
            return static_cast<uint64_t>(GetCurrentProcessId());
#else
            return static_cast<uint64_t>(getpid());
#endif
        }

        std::string json_rpc_frame(const json& payload) {
            const std::string body = payload.dump();
            return "Content-Length: " + std::to_string(body.size()) + "\r\n\r\n" + body;
        }

        json python_language_server_settings() {
            return {
                {"analysis",
                 {{"diagnosticMode", "openFilesOnly"},
                  {"autoImportCompletions", false},
                  {"useLibraryCodeForTypes", false}}},
            };
        }

        fs::path get_lichtfeld_dir() {
            if (const char* override_workspace = std::getenv("LFS_PYTHON_LSP_WORKSPACE");
                override_workspace && *override_workspace) {
                return fs::path(override_workspace);
            }
#ifdef _WIN32
            const char* const home = std::getenv("USERPROFILE");
            return fs::path(home ? home : "C:\\") / ".lichtfeld";
#else
            const char* const home = std::getenv("HOME");
            return fs::path(home ? home : "/tmp") / ".lichtfeld";
#endif
        }

        json workspace_configuration_result_for_section(const std::string& section) {
            const json settings = python_language_server_settings();
            if (section.empty()) {
                return {
                    {"basedpyright", settings},
                    {"python", settings},
                };
            }
            if (section == "basedpyright") {
                return settings;
            }
            if (section == "basedpyright.analysis") {
                return settings.value("analysis", json::object());
            }
            if (section == "python") {
                return settings;
            }
            if (section == "python.analysis") {
                return settings.value("analysis", json::object());
            }
            return nullptr;
        }

        bool write_python_lsp_config(const fs::path& workspace_root, const fs::path& project_root) {
            std::error_code ec;
            fs::create_directories(workspace_root, ec);
            if (ec) {
                LOG_WARN("PythonLspClient: failed to create LSP workspace '{}': {}",
                         lfs::core::path_to_utf8(workspace_root), ec.message());
                return false;
            }

            const fs::path config_path = workspace_root / "pyrightconfig.json";
            const fs::path build_python = project_root / "build" / "src" / "python";
            const fs::path build_typings = build_python / "typings";
            const fs::path source_stubs = project_root / "src" / "python" / "stubs";
            const fs::path active_stub_path = fs::exists(build_typings) ? build_typings : source_stubs;
            const fs::path active_python_path = fs::exists(build_python) ? build_python : source_stubs;
            const auto relative_to_workspace = [&](const fs::path& path) {
                std::error_code relative_ec;
                fs::path relative = fs::relative(path, workspace_root, relative_ec);
                return lfs::core::path_to_utf8(relative_ec ? path : relative);
            };

            json config = {
                {"include", json::array({"script_editor.py"})},
                {"exclude", json::array({"**/__pycache__", "**/.mypy_cache"})},
                {"extraPaths",
                 json::array({
                     relative_to_workspace(active_python_path),
                     relative_to_workspace(active_stub_path),
                     relative_to_workspace(source_stubs),
                 })},
                {"stubPath", relative_to_workspace(active_stub_path)},
                {"pythonVersion", "3.12"},
                {"pythonPlatform", "Linux"},
                {"typeCheckingMode", "basic"},
                {"reportMissingImports", "warning"},
                {"reportMissingTypeStubs", "none"},
            };

            std::string rendered = config.dump(4);
            rendered.push_back('\n');

            std::ofstream output(config_path, std::ios::binary | std::ios::trunc);
            if (!output) {
                LOG_WARN("PythonLspClient: failed to open '{}'",
                         lfs::core::path_to_utf8(config_path));
                return false;
            }

            output << rendered;
            return output.good();
        }

        std::optional<PythonLspClient::TextEdit> parse_text_edit(const json& node) {
            if (!node.is_object()) {
                return std::nullopt;
            }

            const json* range = nullptr;
            if (node.contains("range") && node["range"].is_object()) {
                range = &node["range"];
            } else if (node.contains("replace") && node["replace"].is_object()) {
                range = &node["replace"];
            } else if (node.contains("insert") && node["insert"].is_object()) {
                range = &node["insert"];
            }

            if (range == nullptr || !range->contains("start") || !range->contains("end")) {
                return std::nullopt;
            }

            const auto& start = (*range)["start"];
            const auto& end = (*range)["end"];
            if (!start.is_object() || !end.is_object()) {
                return std::nullopt;
            }

            return PythonLspClient::TextEdit{
                .start_line = start.value("line", 0),
                .start_character = start.value("character", 0),
                .end_line = end.value("line", 0),
                .end_character = end.value("character", 0),
                .new_text = node.value("newText", std::string()),
            };
        }

        std::optional<json> pop_json_rpc_message(std::string& buffer) {
            while (true) {
                const size_t header_end = buffer.find("\r\n\r\n");
                if (header_end == std::string::npos) {
                    return std::nullopt;
                }

                std::string_view headers(buffer.data(), header_end);
                size_t content_length = 0;

                size_t line_start = 0;
                while (line_start < headers.size()) {
                    const size_t line_end = headers.find("\r\n", line_start);
                    const std::string_view line = headers.substr(
                        line_start,
                        line_end == std::string::npos ? headers.size() - line_start
                                                      : line_end - line_start);
                    if (line.rfind("Content-Length:", 0) == 0) {
                        const std::string_view value = line.substr(std::string_view("Content-Length:").size());
                        content_length = static_cast<size_t>(std::stoull(std::string(value)));
                        break;
                    }
                    if (line_end == std::string::npos) {
                        break;
                    }
                    line_start = line_end + 2;
                }

                if (content_length == 0) {
                    buffer.erase(0, header_end + 4);
                    continue;
                }

                const size_t body_offset = header_end + 4;
                if (buffer.size() < body_offset + content_length) {
                    return std::nullopt;
                }

                const std::string body = buffer.substr(body_offset, content_length);
                buffer.erase(0, body_offset + content_length);

                try {
                    return json::parse(body);
                } catch (const json::parse_error& error) {
                    LOG_WARN("PythonLspClient: failed to parse LSP JSON: {}", error.what());
                }
            }
        }

    } // namespace

    struct PythonLspClient::Impl {
        enum class State {
            Starting,
            Ready,
            Failed,
        };

        Impl()
            : commands(discover_server_commands()),
              project_root(PROJECT_ROOT_PATH),
              workspace_root(get_lichtfeld_dir() / "python-lsp"),
              document_path(workspace_root / "script_editor.py"),
              root_uri(file_uri_from_path(workspace_root)),
              document_uri(file_uri_from_path(document_path)),
              next_retry_at(std::chrono::steady_clock::now()) {
            if (commands.empty()) {
                state = State::Failed;
                status = "No Python language server configured";
                return;
            }

            status = "Starting Python completions...";
            write_python_lsp_config(workspace_root, project_root);
            worker = std::jthread([this](std::stop_token stop_token) { run(stop_token); });
        }

        ~Impl() {
            cv.notify_all();
        }

        int updateDocument(const std::string& text) {
            std::scoped_lock lock(mutex);
            if (document_text == text) {
                return document_version;
            }

            document_text = text;
            ++document_version;
            document_dirty = true;
            latest_completion.reset();
            latest_semantic_tokens.reset();
            cv.notify_all();
            return document_version;
        }

        void requestCompletion(const PendingCompletionRequest& request) {
            std::scoped_lock lock(mutex);
            queued_completion = request;
            cv.notify_all();
        }

        void requestSemanticTokens(const PendingSemanticTokensRequest& request) {
            std::scoped_lock lock(mutex);
            queued_semantic_tokens = request;
            cv.notify_all();
        }

        std::optional<CompletionList> takeLatestCompletion() {
            std::scoped_lock lock(mutex);
            auto result = std::move(latest_completion);
            latest_completion.reset();
            return result;
        }

        std::optional<SemanticTokenList> takeLatestSemanticTokens() {
            std::scoped_lock lock(mutex);
            auto result = std::move(latest_semantic_tokens);
            latest_semantic_tokens.reset();
            return result;
        }

        bool isReady() const {
            std::scoped_lock lock(mutex);
            return state == State::Ready;
        }

        bool isAvailable() const {
            std::scoped_lock lock(mutex);
            return state != State::Failed;
        }

        std::string statusText() const {
            std::scoped_lock lock(mutex);
            return status;
        }

        std::chrono::milliseconds workerWaitDurationLocked() const {
            const bool waiting_on_server =
                !initialized || document_dirty || queued_completion.has_value() ||
                queued_semantic_tokens.has_value() || !inflight_completions.empty() ||
                !inflight_semantic_tokens.empty();
            return waiting_on_server ? WORKER_ACTIVE_WAIT : WORKER_IDLE_WAIT;
        }

        void cancelRequests(const std::vector<uint64_t>& request_ids) {
            for (const auto request_id : request_ids) {
                const json payload = {
                    {"jsonrpc", "2.0"},
                    {"method", "$/cancelRequest"},
                    {"params", {{"id", request_id}}},
                };
                process.writeAll(json_rpc_frame(payload));
            }
        }

        void run(const std::stop_token stop_token) {
            while (!stop_token.stop_requested()) {
                bool did_work = false;

                if (active_candidate_index.has_value() && !process.isRunning()) {
                    handle_process_exit();
                    did_work = true;
                }

                if (!active_candidate_index.has_value() &&
                    attempt_candidate_index < commands.size() &&
                    std::chrono::steady_clock::now() >= next_retry_at) {
                    did_work = try_start_candidate(attempt_candidate_index) || did_work;
                }

                if (active_candidate_index.has_value() && !initialized &&
                    std::chrono::steady_clock::now() - candidate_started_at > INIT_TIMEOUT) {
                    LOG_WARN("PythonLspClient: {} did not respond within {}s, killing",
                             commands[active_candidate_index.value()].label,
                             std::chrono::duration_cast<std::chrono::seconds>(INIT_TIMEOUT).count());
                    process.kill();
                    handle_process_exit();
                    did_work = true;
                }

                if (active_candidate_index.has_value()) {
                    did_work = drain_stdout() || did_work;
                    did_work = drain_stderr(false) || did_work;
                    did_work = flush_document_sync() || did_work;
                    did_work = flush_completion_request() || did_work;
                    did_work = flush_semantic_tokens_request() || did_work;
                }

                if (!did_work) {
                    std::unique_lock lock(mutex);
                    cv.wait_for(lock, workerWaitDurationLocked());
                }
            }

            process.kill();
            drain_stderr(true);
        }

        bool try_start_candidate(size_t index) {
            if (index >= commands.size()) {
                return false;
            }

            const auto& command = commands[index];
            if (!process.start(command.program, command.args)) {
                LOG_WARN("PythonLspClient: failed to start {}", command.label);
                attempt_candidate_index = index + 1;
                next_retry_at = std::chrono::steady_clock::now() + RESTART_DELAY;
                if (attempt_candidate_index >= commands.size()) {
                    std::scoped_lock lock(mutex);
                    state = State::Failed;
                    status = "Python completions unavailable";
                }
                return false;
            }

            LOG_INFO("PythonLspClient: starting {}", command.label);

            {
                std::scoped_lock lock(mutex);
                state = State::Starting;
                status = "Starting " + command.label + "...";
                document_dirty = true;
            }

            active_candidate_index = index;
            initialized = false;
            ready_once = false;
            candidate_started_at = std::chrono::steady_clock::now();
            document_opened = false;
            stdout_buffer.clear();
            stderr_buffer.clear();
            inflight_completions.clear();
            inflight_semantic_tokens.clear();
            last_completion_request.reset();
            last_semantic_tokens_request.reset();
            semantic_tokens_supported = false;
            semantic_token_types.clear();
            semantic_token_modifiers.clear();

            initialize_request_id = next_request_id++;
            json workspace_folders = json::array();
            workspace_folders.push_back({
                {"uri", root_uri},
                {"name", workspace_root.filename().empty()
                             ? "LichtFeld Studio"
                             : lfs::core::path_to_utf8(workspace_root.filename())},
            });

            json initialize_capabilities = {
                {"general", {{"positionEncodings", json::array({"utf-16"})}}},
                {"textDocument",
                 {{"synchronization",
                   {{"willSave", false},
                    {"willSaveWaitUntil", false},
                    {"didSave", false}}},
                  {"completion",
                   {{"contextSupport", true},
                    {"completionItem",
                     {{"snippetSupport", false},
                      {"documentationFormat", json::array({"markdown", "plaintext"})},
                      {"labelDetailsSupport", true},
                      {"deprecatedSupport", true}}}}},
                  {"semanticTokens",
                   {{"requests", {{"full", true}}},
                    {"formats", json::array({"relative"})},
                    {"tokenTypes", semantic_token_types_capability()},
                    {"tokenModifiers", semantic_token_modifiers_capability()},
                    {"overlappingTokenSupport", false},
                    {"multilineTokenSupport", false}}}}},
            };

            json initialize = {
                {"jsonrpc", "2.0"},
                {"id", initialize_request_id},
                {"method", "initialize"},
                {"params",
                 {
                     {"processId", current_process_id()},
                     {"rootUri", root_uri},
                     {"rootPath", lfs::core::path_to_utf8(workspace_root)},
                     {"clientInfo", {{"name", "LichtFeld Studio"}}},
                     {"workspaceFolders", workspace_folders},
                     {"capabilities", initialize_capabilities},
                 }},
            };

            if (!process.writeAll(json_rpc_frame(initialize))) {
                process.kill();
                return false;
            }

            return true;
        }

        void handle_process_exit() {
            drain_stderr(true);

            const size_t failed_candidate = active_candidate_index.value_or(0);
            const int exit_code = process.exitCode();
            const bool restart_same_candidate = ready_once;

            if (restart_same_candidate) {
                LOG_WARN("PythonLspClient: {} exited with code {}",
                         commands[failed_candidate].label, exit_code);
            } else {
                LOG_INFO("PythonLspClient: {} not available (exit code {})",
                         commands[failed_candidate].label, exit_code);
            }

            process.kill();
            active_candidate_index.reset();
            initialized = false;
            document_opened = false;
            inflight_completions.clear();
            inflight_semantic_tokens.clear();
            last_completion_request.reset();
            last_semantic_tokens_request.reset();

            {
                std::scoped_lock lock(mutex);
                state = State::Starting;
                status = restart_same_candidate ? "Restarting Python completions..."
                                                : "Trying fallback Python completions...";
                document_dirty = true;
            }

            if (restart_same_candidate) {
                attempt_candidate_index = failed_candidate;
            } else {
                attempt_candidate_index = failed_candidate + 1;
                if (attempt_candidate_index >= commands.size()) {
                    std::scoped_lock lock(mutex);
                    state = State::Failed;
                    status = "Python completions unavailable";
                }
            }

            next_retry_at = std::chrono::steady_clock::now() + RESTART_DELAY;
            ready_once = false;
        }

        bool drain_stdout() {
            bool did_work = false;
            std::array<char, 4096> buffer = {};

            while (true) {
                const ssize_t read = process.readStdout(buffer.data(), buffer.size());
                if (read <= 0) {
                    break;
                }

                stdout_buffer.append(buffer.data(), static_cast<size_t>(read));
                did_work = true;

                while (auto message = pop_json_rpc_message(stdout_buffer)) {
                    handle_message(*message);
                }
            }

            return did_work;
        }

        bool drain_stderr(const bool flush_all) {
            bool did_work = false;
            std::array<char, 2048> buffer = {};

            while (true) {
                const ssize_t read = process.readStderr(buffer.data(), buffer.size());
                if (read <= 0) {
                    break;
                }
                stderr_buffer.append(buffer.data(), static_cast<size_t>(read));
                did_work = true;
            }

            size_t line_end = 0;
            while ((line_end = stderr_buffer.find('\n')) != std::string::npos) {
                std::string line = stderr_buffer.substr(0, line_end);
                if (!line.empty() && line.back() == '\r') {
                    line.pop_back();
                }
                if (!line.empty()) {
                    LOG_DEBUG("PythonLspClient: {}", line);
                }
                stderr_buffer.erase(0, line_end + 1);
                did_work = true;
            }

            if (flush_all && !stderr_buffer.empty()) {
                LOG_DEBUG("PythonLspClient: {}", stderr_buffer);
                stderr_buffer.clear();
                did_work = true;
            }

            return did_work;
        }

        bool flush_document_sync() {
            std::string text;
            int version = 0;
            bool open_document = false;

            {
                std::scoped_lock lock(mutex);
                if (!initialized || !document_dirty) {
                    return false;
                }

                text = document_text;
                version = document_version;
                open_document = !document_opened;
                document_dirty = false;
            }

            json payload = {
                {"jsonrpc", "2.0"},
                {"method", open_document ? "textDocument/didOpen" : "textDocument/didChange"},
                {"params",
                 open_document
                     ? json{{"textDocument",
                             {{"uri", document_uri},
                              {"languageId", "python"},
                              {"version", version},
                              {"text", text}}}}
                     : json{{"textDocument",
                             {{"uri", document_uri},
                              {"version", version}}},
                            {"contentChanges", json::array({{{"text", text}}})}}}};

            if (!process.writeAll(json_rpc_frame(payload))) {
                std::scoped_lock lock(mutex);
                document_dirty = true;
                return false;
            }

            if (open_document) {
                document_opened = true;
            }
            return true;
        }

        bool flush_completion_request() {
            PendingCompletionRequest request;
            uint64_t request_id = 0;
            std::vector<uint64_t> request_ids_to_cancel;

            {
                std::scoped_lock lock(mutex);
                if (!initialized || !document_opened || document_dirty || !queued_completion.has_value()) {
                    return false;
                }

                if (last_completion_request.has_value() &&
                    last_completion_request->document_version == queued_completion->document_version &&
                    last_completion_request->line == queued_completion->line &&
                    last_completion_request->character == queued_completion->character &&
                    last_completion_request->manual == queued_completion->manual) {
                    queued_completion.reset();
                    return false;
                }

                request = *queued_completion;
                queued_completion.reset();
                request_ids_to_cancel.reserve(inflight_completions.size());
                for (const auto& [id, _] : inflight_completions) {
                    request_ids_to_cancel.push_back(id);
                }
                inflight_completions.clear();
                request_id = next_request_id++;
                inflight_completions.emplace(request_id, request);
                last_completion_request = request;
            }

            cancelRequests(request_ids_to_cancel);

            json context = {
                {"triggerKind", request.manual || request.trigger_character.empty() ? 1 : 2},
            };
            if (!request.manual && !request.trigger_character.empty()) {
                context["triggerCharacter"] = request.trigger_character;
            }

            json payload = {
                {"jsonrpc", "2.0"},
                {"id", request_id},
                {"method", "textDocument/completion"},
                {"params",
                 {{"textDocument", {{"uri", document_uri}}},
                  {"position", {{"line", request.line}, {"character", request.character}}},
                  {"context", context}}}};

            if (!process.writeAll(json_rpc_frame(payload))) {
                std::scoped_lock lock(mutex);
                queued_completion = request;
                inflight_completions.erase(request_id);
                last_completion_request.reset();
                return false;
            }

            return true;
        }

        bool flush_semantic_tokens_request() {
            PendingSemanticTokensRequest request;
            uint64_t request_id = 0;
            std::vector<uint64_t> request_ids_to_cancel;

            {
                std::scoped_lock lock(mutex);
                if (!initialized || !document_opened || document_dirty || !semantic_tokens_supported ||
                    !queued_semantic_tokens.has_value()) {
                    return false;
                }

                if (queued_completion.has_value() || !inflight_completions.empty()) {
                    return false;
                }

                if (last_semantic_tokens_request.has_value() &&
                    last_semantic_tokens_request->document_version ==
                        queued_semantic_tokens->document_version) {
                    queued_semantic_tokens.reset();
                    return false;
                }

                request = *queued_semantic_tokens;
                queued_semantic_tokens.reset();
                request_ids_to_cancel.reserve(inflight_semantic_tokens.size());
                for (const auto& [id, _] : inflight_semantic_tokens) {
                    request_ids_to_cancel.push_back(id);
                }
                inflight_semantic_tokens.clear();
                request_id = next_request_id++;
                inflight_semantic_tokens.emplace(request_id, request);
                last_semantic_tokens_request = request;
            }

            cancelRequests(request_ids_to_cancel);

            json payload = {
                {"jsonrpc", "2.0"},
                {"id", request_id},
                {"method", "textDocument/semanticTokens/full"},
                {"params", {{"textDocument", {{"uri", document_uri}}}}},
            };

            if (!process.writeAll(json_rpc_frame(payload))) {
                std::scoped_lock lock(mutex);
                queued_semantic_tokens = request;
                inflight_semantic_tokens.erase(request_id);
                last_semantic_tokens_request.reset();
                return false;
            }

            return true;
        }

        void handle_message(const json& message) {
            if (message.contains("method")) {
                const std::string method = message.value("method", std::string());
                if (method == "window/logMessage" || method == "window/showMessage") {
                    const auto& params = message.value("params", json::object());
                    const std::string text = params.value("message", std::string());
                    if (!text.empty()) {
                        LOG_DEBUG("PythonLspClient: {}", text);
                    }
                } else if (method == "workspace/configuration" && message.contains("id")) {
                    json result = json::array();
                    const auto& params = message.value("params", json::object());
                    const auto items = params.value("items", json::array());
                    if (items.is_array()) {
                        for (const auto& item : items) {
                            const std::string section =
                                item.is_object() ? item.value("section", std::string()) : std::string();
                            result.push_back(workspace_configuration_result_for_section(section));
                        }
                    }

                    const json response = {
                        {"jsonrpc", "2.0"},
                        {"id", message["id"]},
                        {"result", result},
                    };
                    process.writeAll(json_rpc_frame(response));
                } else if (method == "client/registerCapability" && message.contains("id")) {
                    const json response = {
                        {"jsonrpc", "2.0"},
                        {"id", message["id"]},
                        {"result", nullptr},
                    };
                    process.writeAll(json_rpc_frame(response));
                }
                return;
            }

            if (!message.contains("id")) {
                return;
            }

            const uint64_t id = message["id"].get<uint64_t>();
            if (id == initialize_request_id) {
                if (message.contains("result")) {
                    const auto& result = message["result"];
                    const auto& capabilities = result.value("capabilities", json::object());
                    if (capabilities.contains("semanticTokensProvider") &&
                        capabilities["semanticTokensProvider"].is_object()) {
                        const auto& provider = capabilities["semanticTokensProvider"];
                        const auto& legend = provider.value("legend", json::object());
                        const auto token_types = legend.value("tokenTypes", json::array());
                        const auto token_modifiers =
                            legend.value("tokenModifiers", json::array());

                        semantic_token_types.clear();
                        semantic_token_modifiers.clear();

                        if (token_types.is_array()) {
                            semantic_token_types.reserve(token_types.size());
                            for (const auto& entry : token_types) {
                                if (entry.is_string()) {
                                    semantic_token_types.push_back(entry.get<std::string>());
                                }
                            }
                        }

                        if (token_modifiers.is_array()) {
                            semantic_token_modifiers.reserve(token_modifiers.size());
                            for (const auto& entry : token_modifiers) {
                                if (entry.is_string()) {
                                    semantic_token_modifiers.push_back(entry.get<std::string>());
                                }
                            }
                        }

                        semantic_tokens_supported = !semantic_token_types.empty();
                    }

                    initialized = true;
                    ready_once = true;
                    document_opened = false;

                    {
                        std::scoped_lock lock(mutex);
                        state = State::Ready;
                        status = "Python completions ready";
                        document_dirty = true;
                    }

                    json initialized_payload = {
                        {"jsonrpc", "2.0"},
                        {"method", "initialized"},
                        {"params", json::object()},
                    };
                    process.writeAll(json_rpc_frame(initialized_payload));

                    json configuration_payload = {
                        {"jsonrpc", "2.0"},
                        {"method", "workspace/didChangeConfiguration"},
                        {"params",
                         {{"settings",
                           {{"basedpyright", python_language_server_settings()},
                            {"python", python_language_server_settings()}}}}},
                    };
                    process.writeAll(json_rpc_frame(configuration_payload));
                } else {
                    LOG_WARN("PythonLspClient: initialize failed");
                    process.kill();
                }
                return;
            }

            auto semantic_request_it = inflight_semantic_tokens.find(id);
            if (semantic_request_it != inflight_semantic_tokens.end()) {
                SemanticTokenList semantic_tokens = {
                    .document_version = semantic_request_it->second.document_version,
                    .tokens = {},
                };
                inflight_semantic_tokens.erase(semantic_request_it);

                if (message.contains("result")) {
                    const json& result = message["result"];
                    const json data =
                        result.is_object() ? result.value("data", json::array()) : json::array();
                    if (data.is_array()) {
                        semantic_tokens.tokens.reserve(data.size() / 5);

                        int line = 0;
                        int start_character = 0;
                        for (size_t index = 0; index + 4 < data.size(); index += 5) {
                            if (!data[index].is_number_integer() ||
                                !data[index + 1].is_number_integer() ||
                                !data[index + 2].is_number_integer() ||
                                !data[index + 3].is_number_integer() ||
                                !data[index + 4].is_number_integer()) {
                                continue;
                            }

                            const int delta_line = data[index].get<int>();
                            const int delta_start = data[index + 1].get<int>();
                            const int length = data[index + 2].get<int>();
                            const int token_type_index = data[index + 3].get<int>();
                            const uint32_t token_modifiers = data[index + 4].get<uint32_t>();

                            line += delta_line;
                            start_character =
                                delta_line == 0 ? start_character + delta_start : delta_start;

                            if (token_type_index < 0 ||
                                token_type_index >= static_cast<int>(semantic_token_types.size()) ||
                                length <= 0) {
                                continue;
                            }

                            semantic_tokens.tokens.push_back({
                                .line = line,
                                .start_character = start_character,
                                .length = length,
                                .type = semantic_token_types[static_cast<size_t>(token_type_index)],
                                .modifiers = token_modifiers,
                            });
                        }
                    }
                } else if (message.contains("error")) {
                    LOG_DEBUG("PythonLspClient: semantic tokens request failed");
                }

                std::scoped_lock lock(mutex);
                latest_semantic_tokens = std::move(semantic_tokens);
                return;
            }

            auto request_it = inflight_completions.find(id);
            if (request_it == inflight_completions.end()) {
                return;
            }

            CompletionList completion = {
                .document_version = request_it->second.document_version,
                .line = request_it->second.line,
                .character = request_it->second.character,
                .is_incomplete = false,
                .items = {},
            };

            inflight_completions.erase(request_it);

            const json& result = message.contains("result") ? message["result"] : json();
            json items = json::array();
            if (result.is_array()) {
                items = result;
            } else if (result.is_object()) {
                completion.is_incomplete = result.value("isIncomplete", false);
                items = result.value("items", json::array());
            }

            if (items.is_array()) {
                completion.items.reserve(items.size());
                for (const auto& item : items) {
                    if (!item.is_object()) {
                        continue;
                    }

                    CompletionItem parsed = {
                        .label = item.value("label", std::string()),
                        .detail = item.value("detail", std::string()),
                        .description = item.value("labelDetails", json::object()).value("description", std::string()),
                        .sort_text = item.value("sortText", item.value("label", std::string())),
                        .filter_text = item.value("filterText", item.value("label", std::string())),
                        .insert_text = item.value("insertText", item.value("label", std::string())),
                        .kind = item.value("kind", 0),
                        .deprecated = item.value("deprecated", false),
                        .text_edit = std::nullopt,
                        .additional_text_edits = {},
                    };

                    if (item.contains("tags") && item["tags"].is_array()) {
                        parsed.deprecated = parsed.deprecated ||
                                            std::find(item["tags"].begin(), item["tags"].end(), 1) !=
                                                item["tags"].end();
                    }

                    if (item.contains("textEdit")) {
                        parsed.text_edit = parse_text_edit(item["textEdit"]);
                    }

                    if (item.contains("additionalTextEdits") && item["additionalTextEdits"].is_array()) {
                        for (const auto& edit : item["additionalTextEdits"]) {
                            if (auto parsed_edit = parse_text_edit(edit)) {
                                parsed.additional_text_edits.push_back(*parsed_edit);
                            }
                        }
                    }

                    if (!parsed.label.empty()) {
                        completion.items.push_back(std::move(parsed));
                    }
                }
            }

            std::ranges::sort(completion.items, [](const CompletionItem& lhs, const CompletionItem& rhs) {
                if (lhs.sort_text != rhs.sort_text) {
                    return lhs.sort_text < rhs.sort_text;
                }
                return lhs.label < rhs.label;
            });

            std::scoped_lock lock(mutex);
            latest_completion = std::move(completion);
        }

        std::vector<ServerCommand> commands;
        fs::path project_root;
        fs::path workspace_root;
        fs::path document_path;
        std::string root_uri;
        std::string document_uri;

        mutable std::mutex mutex;
        std::condition_variable_any cv;
        StdioProcess process;

        State state = State::Starting;
        std::string status;

        std::optional<size_t> active_candidate_index;
        size_t attempt_candidate_index = 0;
        std::chrono::steady_clock::time_point next_retry_at;

        bool initialized = false;
        bool ready_once = false;
        std::chrono::steady_clock::time_point candidate_started_at;
        bool document_opened = false;
        std::string document_text;
        int document_version = 1;
        bool document_dirty = true;

        uint64_t next_request_id = 1;
        uint64_t initialize_request_id = 0;
        std::unordered_map<uint64_t, PendingCompletionRequest> inflight_completions;
        std::unordered_map<uint64_t, PendingSemanticTokensRequest> inflight_semantic_tokens;
        std::optional<PendingCompletionRequest> queued_completion;
        std::optional<PendingSemanticTokensRequest> queued_semantic_tokens;
        std::optional<PendingCompletionRequest> last_completion_request;
        std::optional<PendingSemanticTokensRequest> last_semantic_tokens_request;
        std::optional<CompletionList> latest_completion;
        std::optional<SemanticTokenList> latest_semantic_tokens;
        bool semantic_tokens_supported = false;
        std::vector<std::string> semantic_token_types;
        std::vector<std::string> semantic_token_modifiers;

        std::string stdout_buffer;
        std::string stderr_buffer;
        std::jthread worker;
    };

    PythonLspClient::PythonLspClient()
        : impl_(std::make_unique<Impl>()) {
    }

    PythonLspClient::~PythonLspClient() = default;

    int PythonLspClient::updateDocument(const std::string& text) {
        return impl_->updateDocument(text);
    }

    void PythonLspClient::requestCompletion(const int document_version,
                                            const int line,
                                            const int character,
                                            const bool manual,
                                            std::string trigger_character) {
        impl_->requestCompletion({
            .document_version = document_version,
            .line = line,
            .character = character,
            .manual = manual,
            .trigger_character = std::move(trigger_character),
        });
    }

    void PythonLspClient::requestSemanticTokens(const int document_version) {
        impl_->requestSemanticTokens({
            .document_version = document_version,
        });
    }

    std::optional<PythonLspClient::CompletionList> PythonLspClient::takeLatestCompletion() {
        return impl_->takeLatestCompletion();
    }

    std::optional<PythonLspClient::SemanticTokenList> PythonLspClient::takeLatestSemanticTokens() {
        return impl_->takeLatestSemanticTokens();
    }

    bool PythonLspClient::isReady() const {
        return impl_->isReady();
    }

    bool PythonLspClient::isAvailable() const {
        return impl_->isAvailable();
    }

    std::string PythonLspClient::statusText() const {
        return impl_->statusText();
    }

} // namespace lfs::vis::editor

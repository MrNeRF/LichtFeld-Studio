/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "app/mcp_event_handlers.hpp"
#include "app/mcp_media_tools.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include <chrono>
#include <condition_variable>
#include <fstream>
#include <iostream>
#include <mutex>
#include <thread>

nlohmann::json runCpuColorContracts();

int main(int argc, char** argv) {
    using namespace lfs;
    using nlohmann::json;
    try {
        if (argc != 2)
            throw std::runtime_error("Expected JSON request path");
        core::Logger::get().init(core::LogLevel::Warn, "", "", true,
                                 core::path_to_utf8(core::utf8_to_path(argv[1]).parent_path()));
        std::ifstream stream(core::utf8_to_path(argv[1]));
        const auto input = json::parse(stream);
        if (input.value("operation", "") == "cpu-color") {
            std::cout << runCpuColorContracts().dump() << std::endl;
            return 0;
        }
        mcp::ToolRegistry registry;
        app::register_media_tools(registry);
        json output;
        output["capabilities"] = registry.call_tool("media.capabilities", json::object());
        output["probe"] = registry.call_tool("media.probe", {{"input", input.at("request").at("input")}});
        std::mutex mutex;
        std::condition_variable gate;
        bool started = false, release = false;
        json events = json::array();
        event::ScopedHandler handlers;
        app::register_mcp_event_handlers(handlers, app::McpEventStreamKind::RuntimeJournal,
                                         [&](const std::string& type, json payload) {
                                             {
                                                 std::unique_lock lock(mutex);
                                                 events.push_back({{"type", type}, {"data", std::move(payload)}});
                                                 if (type == "media.extract.started") {
                                                     started = true;
                                                     gate.notify_all();
                                                     if (!gate.wait_for(lock, std::chrono::seconds(10), [&] { return release; }))
                                                         throw std::runtime_error("Test worker gate timed out");
                                                 }
                                             }
                                             if (type == "media.extract.progress" && input.value("cancel", false))
                                                 static_cast<void>(app::cancel_media_extract_job());
                                         });
        struct JoinWorker {
            std::mutex& mutex;
            std::condition_variable& gate;
            bool& release;
            ~JoinWorker() {
                {
                    std::lock_guard lock(mutex);
                    release = true;
                }
                gate.notify_all();
                app::shutdown_media_extract_job();
            }
        } join{mutex, gate, release};
        output["start"] = registry.call_tool("media.extract", input.at("request"));
        if (output.at("start").value("success", false)) {
            {
                std::unique_lock lock(mutex);
                if (!gate.wait_for(lock, std::chrono::seconds(10), [&] { return started; }))
                    throw std::runtime_error("Media worker did not start");
            }
            output["duplicate"] = registry.call_tool("media.extract", input.at("request"));
            {
                std::lock_guard lock(mutex);
                release = true;
            }
            gate.notify_all();
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
            while (app::media_extract_job_snapshot().at("active").get<bool>()) {
                if (std::chrono::steady_clock::now() >= deadline)
                    throw std::runtime_error("Media job timed out");
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
        }
        app::shutdown_media_extract_job();
        output["job"] = app::media_extract_job_snapshot();
        output["events"] = events;
        std::cout << output.dump() << std::endl;
        return 0;
    } catch (const std::exception& error) {
        app::shutdown_media_extract_job();
        std::cerr << error.what() << '\n';
        return 1;
    }
}

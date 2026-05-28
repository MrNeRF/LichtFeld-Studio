/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "tcp_server.hpp"
#include <atomic>
#include <core/logger.hpp>
#include <functional>
#include <mutex>
#include <optional>
#include <vector>

namespace lfs::tcp {
    class PublisherServer : public TCPServer {
    public:
        PublisherServer(int port, std::shared_ptr<lfs::vis::TrainerManager> trainer_manager, core::LogLevel level = core::LogLevel::Info, bool warm_up = true);
        ~PublisherServer() override;
        void start() override;
        void stop() override;
        void join() override {}

    private:
        static nlohmann::json makeEventMessage(const nlohmann::json& data, const std::string& event_type);

    private:
        std::mutex send_mutex_;     // Avoids multiple calls to send by different event threads
        std::atomic<bool> stopped_; // Avoids calls to send when the stop is taking place
        std::vector<std::function<void()>> subscriptions_;
        core::LogLevel level_;
        std::optional<core::LogHandlerToken> log_handler_token_;
    };
} // namespace lfs::tcp

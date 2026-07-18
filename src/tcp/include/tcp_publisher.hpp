/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "tcp_server.hpp"
#include <atomic>
#include <condition_variable>
#include <core/logger.hpp>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace lfs::core::events::state {
    struct TrainingCompleted;
}

namespace lfs::tcp {
    class PublisherServer : public TCPServer {
    public:
        PublisherServer(int port, std::shared_ptr<lfs::vis::TrainerManager> trainer_manager, core::LogLevel level = core::LogLevel::Info, bool warm_up = true);
        ~PublisherServer() override;
        void start() override;
        void stop() override;
        void join() override;

        // The exact wire message the publisher emits for one event. The
        // ENABLE_TO_JSON serializers are TU-local to tcp_publisher.cpp, so tests
        // must serialize through this seam instead of redefining to_json (IFNDR).
        static nlohmann::json wireMessageFor(const core::events::state::TrainingCompleted& event);

    private:
        struct QueuedEvent {
            nlohmann::json message;
            std::string event_type;
            size_t estimated_bytes = 0;
            bool lossy = false;
        };

        struct QueueState {
            std::mutex mutex;
            std::condition_variable cv;
            std::deque<QueuedEvent> queue;
            std::atomic<bool> accepting{false};
            std::atomic<size_t> dropped{0};
            size_t queued_bytes = 0;
        };

        static nlohmann::json makeEventMessage(const nlohmann::json& data, const std::string& event_type);
        static void enqueueEvent(const std::shared_ptr<QueueState>& state,
                                 nlohmann::json data,
                                 std::string event_type) noexcept;
        void runPublisher(std::shared_ptr<QueueState> state) noexcept;

    private:
        std::shared_ptr<QueueState> queue_state_;
        std::thread publisher_thread_;
        std::vector<std::function<void()>> subscriptions_;
        core::LogLevel level_;
        std::optional<core::LogHandlerToken> log_handler_token_;
    };
} // namespace lfs::tcp

/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "tcp_server.hpp"
#include <atomic>
#include <thread>

namespace lfs::tcp {
    class ResponderServer : public TCPServer {
    public:
        explicit ResponderServer(int port, std::shared_ptr<lfs::vis::TrainerManager> trainer_manager);
        ~ResponderServer() override;
        void start() override;
        void stop() override;
        void join() override;

    private:
        void run();
        [[nodiscard]] nlohmann::json generateResponse(const nlohmann::json& request);
        [[nodiscard]] nlohmann::json getValue(std::string_view parameter, bool& success);

    private:
        std::thread response_thread_;
        std::atomic<bool> running_;
    };
} // namespace lfs::tcp
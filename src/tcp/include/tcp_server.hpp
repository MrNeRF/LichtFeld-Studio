/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <visualizer/training/training_manager.hpp>
#include <zmq.hpp>

namespace lfs::tcp {
    class TCPServer {
        static constexpr int kNumberOfThreads = 2; // To handle async network I/O in ZMQ

    public:
        TCPServer(int port, std::shared_ptr<lfs::vis::TrainerManager> trainer_manager, zmq::socket_type type);
        virtual ~TCPServer() = default;
        virtual void start() = 0;
        virtual void stop() = 0;
        virtual void join() = 0;
        [[nodiscard]] std::string getEndpoint() const;

    protected:
        void send(const nlohmann::json& data);
        [[nodiscard]] bool receive(nlohmann::json& data);

    private:
        [[nodiscard]] static zmq::message_t toZMQ(const nlohmann::json& data);
        [[nodiscard]] static nlohmann::json fromZMQ(const zmq::message_t& msg_zmq, unsigned long long size);

    protected:
        int port_;
        std::string endpoint_;
        std::shared_ptr<lfs::vis::TrainerManager> trainer_manager_;
        zmq::context_t context_;
        zmq::socket_t socket_;
    };
} // namespace lfs::tcp

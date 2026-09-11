/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/error.hpp"

#include <cstdint>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <visualizer/training/training_manager.hpp>
#include <zmq.hpp>

namespace lfs::tcp {

    // Outcome of one TCPServer::receive() call. Timeout is the common,
    // non-error wakeup (the socket has a short rcvtimeo); MalformedJson and
    // Transport carry a typed error through the out-parameter so no raw
    // parser text or zmq errno string ever reaches the wire.
    enum class TcpReceiveStatus : std::uint8_t {
        Message,
        Timeout,
        MalformedJson,
        Transport,
    };

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
        [[nodiscard]] lfs::Status send(const nlohmann::json& data);
        [[nodiscard]] TcpReceiveStatus receive(nlohmann::json& data, lfs::Error* out_error = nullptr);

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

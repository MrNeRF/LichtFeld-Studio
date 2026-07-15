/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "tcp_server.hpp"

#include <algorithm>
#include <cstring>

namespace lfs::tcp {

    TCPServer::TCPServer(int port, std::shared_ptr<lfs::vis::TrainerManager> trainer_manager, zmq::socket_type type)
        : port_(port),
          trainer_manager_(std::move(trainer_manager)),
          context_(kNumberOfThreads),
          socket_(context_, type) {
        port_ = std::max(port_, 0); // Port == 0 sets automatic port
        socket_.bind("tcp://*:" + std::to_string(port_));
        endpoint_ = socket_.get(zmq::sockopt::last_endpoint);
        auto str_port = endpoint_.substr(endpoint_.find_last_of(':') + 1);
        port_ = std::stoi(str_port);
    }

    std::string TCPServer::getEndpoint() const {
        // The socket belongs to the server's I/O thread after start(). Cache the
        // endpoint at bind time so callers never query it cross-thread.
        return endpoint_;
    }

    void TCPServer::send(const nlohmann::json& data) {
        socket_.send(toZMQ(data), zmq::send_flags::none);
    }

    bool TCPServer::receive(nlohmann::json& data) {
        zmq::message_t zqm_request;
        auto result = socket_.recv(zqm_request, zmq::recv_flags::none);

        if (!result.has_value()) {
            return false; // Returns false on time-outs
        }

        auto res_size = result.value();
        if (res_size == 0) {
            data = nlohmann::json{};
            return true;
        }

        data = fromZMQ(zqm_request, res_size);
        return true;
    }

    zmq::message_t TCPServer::toZMQ(const nlohmann::json& data) {
        auto msg_str = data.dump();
        zmq::message_t req(msg_str.length());
        memcpy(req.data(), msg_str.data(), msg_str.length());
        return req;
    }

    nlohmann::json TCPServer::fromZMQ(const zmq::message_t& msg_zmq, unsigned long long size) {
        return nlohmann::json::parse(std::string(static_cast<const char*>(msg_zmq.data()), size));
    }
} // namespace lfs::tcp

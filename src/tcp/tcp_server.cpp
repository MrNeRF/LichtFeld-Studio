/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "tcp_server.hpp"

#include "core/error.hpp"
#include "core/source_site.hpp"

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

    lfs::Status TCPServer::send(const nlohmann::json& data) {
        try {
            if (socket_.send(toZMQ(data), zmq::send_flags::none).has_value()) {
                return {};
            }
            return lfs::Status::failure(lfs::make_error(lfs::ErrorInit{
                .code = lfs::ErrorCode::Unavailable,
                .domain = lfs::ErrorDomain::TCP,
                .retryability = lfs::Retryability::Retryable,
                .user_message = "TCP send timed out",
                .detection = LFS_SOURCE_SITE_CURRENT(),
            }));
        } catch (const zmq::error_t& e) {
            return lfs::Status::failure(lfs::make_error(lfs::ErrorInit{
                .code = lfs::ErrorCode::Unavailable,
                .domain = lfs::ErrorDomain::TCP,
                .user_message = "TCP send failed",
                .detection = LFS_SOURCE_SITE_CURRENT(),
                .native = lfs::NativeError{lfs::ErrorDomain::TCP, e.num(), e.what()},
            }));
        } catch (const std::exception& e) {
            return lfs::Status::failure(lfs::make_error(lfs::ErrorInit{
                .code = lfs::ErrorCode::Internal,
                .domain = lfs::ErrorDomain::TCP,
                .user_message = "TCP send failed",
                .detail = e.what(),
                .detection = LFS_SOURCE_SITE_CURRENT(),
            }));
        }
    }

    TcpReceiveStatus TCPServer::receive(nlohmann::json& data, lfs::Error* out_error) {
        zmq::message_t zqm_request;
        std::size_t res_size = 0;
        try {
            const auto result = socket_.recv(zqm_request, zmq::recv_flags::none);
            if (!result.has_value()) {
                return TcpReceiveStatus::Timeout;
            }
            res_size = result.value();
        } catch (const zmq::error_t& e) {
            if (out_error) {
                *out_error = lfs::make_error(lfs::ErrorInit{
                    .code = lfs::ErrorCode::Unavailable,
                    .domain = lfs::ErrorDomain::TCP,
                    .user_message = "TCP receive failed",
                    .detection = LFS_SOURCE_SITE_CURRENT(),
                    .native = lfs::NativeError{lfs::ErrorDomain::TCP, e.num(), e.what()},
                });
            }
            return TcpReceiveStatus::Transport;
        }

        if (res_size == 0) {
            data = nlohmann::json{};
            return TcpReceiveStatus::Message;
        }

        try {
            data = fromZMQ(zqm_request, res_size);
        } catch (const std::exception& e) {
            if (out_error) {
                *out_error = lfs::make_error(lfs::ErrorInit{
                    .code = lfs::ErrorCode::InvalidArgument,
                    .domain = lfs::ErrorDomain::TCP,
                    .user_message = "Malformed JSON request",
                    .detail = e.what(),
                    .detection = LFS_SOURCE_SITE_CURRENT(),
                });
            }
            return TcpReceiveStatus::MalformedJson;
        }
        return TcpReceiveStatus::Message;
    }

    zmq::message_t TCPServer::toZMQ(const nlohmann::json& data) {
        auto msg_str = data.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);
        zmq::message_t req(msg_str.length());
        memcpy(req.data(), msg_str.data(), msg_str.length());
        return req;
    }

    nlohmann::json TCPServer::fromZMQ(const zmq::message_t& msg_zmq, unsigned long long size) {
        return nlohmann::json::parse(std::string(static_cast<const char*>(msg_zmq.data()), size));
    }
} // namespace lfs::tcp

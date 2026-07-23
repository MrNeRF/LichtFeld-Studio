/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/error.hpp"
#include "core/guarded_task.hpp"
#include "tcp_responder.hpp"
#include "tcp_server.hpp"

#include <nlohmann/json.hpp>
#include <zmq.hpp>

#include <gtest/gtest.h>

#include <memory>
#include <string>

TEST(TcpResponderTest, UnknownThreadEntryExceptionSettlesInsteadOfEscaping) {
    bool completed = false;
    lfs::core::run_guarded<void>(
        lfs::core::TaskContext{
            .name = "tcp.responder-thread",
            .domain = lfs::ErrorDomain::TCP,
            .operation_id = lfs::OperationId::generate(),
            .site = LFS_SOURCE_SITE_CURRENT(),
        },
        []() -> lfs::Result<void> { throw 7; },
        [&completed](lfs::Result<void>&& result) {
            completed = true;
            ASSERT_FALSE(result);
            EXPECT_EQ(result.error().code(), lfs::ErrorCode::Internal);
            EXPECT_EQ(result.error().domain(), lfs::ErrorDomain::TCP);
            EXPECT_EQ(result.error().detail(), "unknown exception");
        });
    EXPECT_TRUE(completed);
}

namespace {

    using json = nlohmann::json;

    // Exposes the protected transport primitives so the receive() status
    // machine can be driven directly (Section 5.3.5).
    class DirectTcpServer : public lfs::tcp::TCPServer {
    public:
        DirectTcpServer()
            : TCPServer(0, nullptr, zmq::socket_type::rep) {
            socket_.set(zmq::sockopt::rcvtimeo, 100);
        }
        void start() override {}
        void stop() override {}
        void join() override {}
        using TCPServer::receive;
    };

    class TcpResponderRoundtripTest : public ::testing::Test {
    protected:
        void SetUp() override {
            responder_ = std::make_unique<lfs::tcp::ResponderServer>(0, nullptr);
            responder_->start();
            endpoint_ = responder_->getEndpoint();
        }

        void TearDown() override {
            responder_->stop();
        }

        json roundtrip(const json& request) {
            zmq::context_t context;
            zmq::socket_t client(context, zmq::socket_type::req);
            client.set(zmq::sockopt::rcvtimeo, 2000);
            client.connect(endpoint_);

            const std::string payload = request.dump();
            zmq::message_t out(payload.data(), payload.size());
            client.send(out, zmq::send_flags::none);

            zmq::message_t reply;
            const auto received = client.recv(reply, zmq::recv_flags::none);
            EXPECT_TRUE(received.has_value());
            return json::parse(reply.to_string());
        }

        std::string send_raw(const std::string& payload) {
            zmq::context_t context;
            zmq::socket_t client(context, zmq::socket_type::req);
            client.set(zmq::sockopt::rcvtimeo, 2000);
            client.connect(endpoint_);

            zmq::message_t out(payload.data(), payload.size());
            client.send(out, zmq::send_flags::none);

            zmq::message_t reply;
            const auto received = client.recv(reply, zmq::recv_flags::none);
            EXPECT_TRUE(received.has_value());
            return reply.to_string();
        }

        std::unique_ptr<lfs::tcp::ResponderServer> responder_;
        std::string endpoint_;
    };

} // namespace

TEST_F(TcpResponderRoundtripTest, UnknownCommandReturnsNotFoundEnvelope) {
    const auto reply = roundtrip(json{{"command", "bogus"}});
    EXPECT_FALSE(reply["success"].get<bool>());
    EXPECT_EQ(reply["command"], "bogus");
    EXPECT_EQ(reply["error"]["code"], "NotFound");
    EXPECT_EQ(reply["error"]["domain"], "TCP");
    EXPECT_EQ(reply["error"]["details"]["command"], "bogus");
    EXPECT_EQ(reply["error_message"], "Unknown command: bogus");
}

TEST_F(TcpResponderRoundtripTest, MalformedJsonReturnsInvalidArgumentEnvelopeWithoutParserText) {
    const std::string body = send_raw("{not json");
    EXPECT_EQ(body.find("parse error"), std::string::npos);

    const auto reply = json::parse(body);
    EXPECT_FALSE(reply["success"].get<bool>());
    EXPECT_EQ(reply["error"]["code"], "InvalidArgument");
    EXPECT_EQ(reply["error"]["domain"], "TCP");
}

TEST_F(TcpResponderRoundtripTest, UnknownGetParameterReturnsEnvelopeWithValueCompat) {
    const auto reply = roundtrip(json{{"command", "get"}, {"parameter", "bogus_param"}});
    EXPECT_FALSE(reply["success"].get<bool>());
    EXPECT_EQ(reply["parameter"], "bogus_param");
    EXPECT_EQ(reply["error"]["code"], "NotFound");
    EXPECT_EQ(reply["error"]["details"]["parameter"], "bogus_param");
    EXPECT_EQ(reply["value"], "");
}

TEST(TcpReceiveStatusTest, TimeoutReturnsTimeoutWithoutError) {
    DirectTcpServer server;
    json data;
    EXPECT_EQ(server.receive(data), lfs::tcp::TcpReceiveStatus::Timeout);
}

TEST(TcpReceiveStatusTest, MalformedFrameReturnsMalformedJsonWithTypedError) {
    DirectTcpServer server;
    zmq::context_t context;
    zmq::socket_t client(context, zmq::socket_type::req);
    client.connect(server.getEndpoint());
    const std::string payload = "{not json";
    zmq::message_t out(payload.data(), payload.size());
    client.send(out, zmq::send_flags::none);

    lfs::Error out_error = lfs::make_error(lfs::ErrorInit{
        .code = lfs::ErrorCode::Internal,
        .domain = lfs::ErrorDomain::TCP,
        .detection = LFS_SOURCE_SITE_CURRENT(),
    });

    json data;
    auto status = lfs::tcp::TcpReceiveStatus::Timeout;
    for (int attempt = 0; attempt < 50 && status == lfs::tcp::TcpReceiveStatus::Timeout; ++attempt) {
        status = server.receive(data, &out_error);
    }

    EXPECT_EQ(status, lfs::tcp::TcpReceiveStatus::MalformedJson);
    EXPECT_EQ(out_error.code(), lfs::ErrorCode::InvalidArgument);
    EXPECT_EQ(out_error.domain(), lfs::ErrorDomain::TCP);
}

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

    std::string loopback_endpoint(const lfs::tcp::TCPServer& server) {
        const auto bound = server.getEndpoint();
        // A wildcard bind address is not a portable client destination.
        return "tcp://127.0.0.1" + bound.substr(bound.find_last_of(':'));
    }

    void configure_test_client(zmq::socket_t& client) {
        // Do not wait forever for undelivered messages when a test fails.
        client.set(zmq::sockopt::linger, 0);
        client.set(zmq::sockopt::sndtimeo, 2000);
        client.set(zmq::sockopt::rcvtimeo, 2000);
    }

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
            endpoint_ = loopback_endpoint(*responder_);
        }

        void TearDown() override {
            responder_->stop();
        }

        void roundtrip(const json& request, json& response) {
            std::string body;
            ASSERT_NO_FATAL_FAILURE(send_raw(request.dump(), body));
            response = json::parse(body);
        }

        void send_raw(const std::string& payload, std::string& response) {
            zmq::context_t context;
            zmq::socket_t client(context, zmq::socket_type::req);
            configure_test_client(client);
            client.connect(endpoint_);

            zmq::message_t out(payload.data(), payload.size());
            ASSERT_TRUE(client.send(out, zmq::send_flags::none).has_value()) << endpoint_;

            zmq::message_t reply;
            const auto received = client.recv(reply, zmq::recv_flags::none);
            ASSERT_TRUE(received.has_value()) << endpoint_;
            response = reply.to_string();
        }

        std::unique_ptr<lfs::tcp::ResponderServer> responder_;
        std::string endpoint_;
    };

} // namespace

TEST_F(TcpResponderRoundtripTest, UnknownCommandReturnsNotFoundEnvelope) {
    json reply;
    ASSERT_NO_FATAL_FAILURE(roundtrip(json{{"command", "bogus"}}, reply));
    EXPECT_FALSE(reply["success"].get<bool>());
    EXPECT_EQ(reply["command"], "bogus");
    EXPECT_EQ(reply["error"]["code"], "NotFound");
    EXPECT_EQ(reply["error"]["domain"], "TCP");
    EXPECT_EQ(reply["error"]["details"]["command"], "bogus");
    EXPECT_EQ(reply["error_message"], "Unknown command: bogus");
}

TEST_F(TcpResponderRoundtripTest, MalformedJsonReturnsInvalidArgumentEnvelopeWithoutParserText) {
    std::string body;
    ASSERT_NO_FATAL_FAILURE(send_raw("{not json", body));
    EXPECT_EQ(body.find("parse error"), std::string::npos);

    const auto reply = json::parse(body);
    EXPECT_FALSE(reply["success"].get<bool>());
    EXPECT_EQ(reply["error"]["code"], "InvalidArgument");
    EXPECT_EQ(reply["error"]["domain"], "TCP");
}

TEST_F(TcpResponderRoundtripTest, UnknownGetParameterReturnsEnvelopeWithValueCompat) {
    json reply;
    ASSERT_NO_FATAL_FAILURE(roundtrip(json{{"command", "get"}, {"parameter", "bogus_param"}}, reply));
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
    configure_test_client(client);
    client.connect(loopback_endpoint(server));
    const std::string payload = "{not json";
    zmq::message_t out(payload.data(), payload.size());
    ASSERT_TRUE(client.send(out, zmq::send_flags::none).has_value());

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

/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "httplib/httplib.h"

#define LLM_RETRY_IMPLEMENTATION
#include "llm_retry/llm_retry.hpp"

#include <atomic>
#include <gtest/gtest.h>
#include <string>
#include <thread>

namespace {

    // Same shape as LLMClient::Impl::do_request: non-200 becomes LLMError with status and Retry-After.
    httplib::Response post_with_retry(httplib::Client& client) {
        return llm::with_retry<httplib::Response>([&] {
                   auto res = client.Post("/v1/messages", "{}", "application/json");
                   if (!res) {
                       throw llm::LLMError{0, "HTTP request failed: " + httplib::to_string(res.error()), true};
                   }
                   if (res->status != 200) {
                       llm::LLMError err{res->status, "HTTP " + std::to_string(res->status) + ": " + res->body, true};
                       try {
                           err.retry_after_seconds = std::stod(res->get_header_value("Retry-After"));
                       } catch (...) {
                       }
                       throw err;
                   }
                   return *res;
               })
            .value;
    }

    struct StubServer {
        httplib::Server server;
        std::thread thread;
        int port = 0;
        std::atomic<int> calls{0};

        explicit StubServer(int fail_status, int failures) {
            server.Post("/v1/messages", [this, fail_status, failures](const httplib::Request&, httplib::Response& res) {
                if (calls.fetch_add(1) < failures) {
                    res.status = fail_status;
                    res.set_header("Retry-After", "0");
                    res.set_content(R"({"error":{"message":"busy"}})", "application/json");
                    return;
                }
                res.set_content(R"({"content":[{"type":"text","text":"ok"}]})", "application/json");
            });
            port = server.bind_to_any_port("127.0.0.1");
            thread = std::thread([this] { server.listen_after_bind(); });
            server.wait_until_ready();
        }

        ~StubServer() {
            server.stop();
            thread.join();
        }
    };

} // namespace

TEST(McpLlmRetry, RateLimitIsRetried) {
    for (int status : {429, 529, 503}) {
        StubServer stub(status, 1);
        httplib::Client client("127.0.0.1", stub.port);
        const auto res = post_with_retry(client);
        EXPECT_EQ(res.status, 200) << status;
        EXPECT_EQ(stub.calls.load(), 2) << status;
    }
}

TEST(McpLlmRetry, ClientErrorIsNotRetried) {
    StubServer stub(400, 10);
    httplib::Client client("127.0.0.1", stub.port);
    try {
        post_with_retry(client);
        FAIL() << "expected LLMError";
    } catch (const llm::LLMError& err) {
        EXPECT_EQ(err.http_code, 400);
    }
    EXPECT_EQ(stub.calls.load(), 1);
}

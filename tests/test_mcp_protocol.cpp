/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <gtest/gtest.h>

#include "config.h"
#include "mcp/mcp_http_server.hpp"
#include "mcp/mcp_protocol.hpp"
#include "mcp/mcp_server.hpp"
#include "mcp/mcp_tools.hpp"

#include <httplib/httplib.h>

#include <stdexcept>

namespace lfs::mcp {

    namespace {

        class ScopedToolRegistration {
        public:
            explicit ScopedToolRegistration(std::string name) : name_(std::move(name)) {}
            ~ScopedToolRegistration() {
                ToolRegistry::instance().unregister_tool(name_);
            }

        private:
            std::string name_;
        };

        class ScopedResourcePrefixRegistration {
        public:
            explicit ScopedResourcePrefixRegistration(std::string prefix) : prefix_(std::move(prefix)) {}
            ~ScopedResourcePrefixRegistration() {
                ResourceRegistry::instance().unregister_resource_prefix(prefix_);
            }

        private:
            std::string prefix_;
        };

    } // namespace

    TEST(McpProtocolTest, ToolJsonIncludesCapabilityAnnotations) {
        const auto payload = tool_to_json(McpTool{
            .name = "test.describe",
            .description = "Describe metadata",
            .input_schema = {.type = "object", .properties = json::object(), .required = {}},
            .metadata = McpToolMetadata{
                .category = "editor",
                .kind = "query",
                .runtime = "gui",
                .thread_affinity = "gui_thread",
                .destructive = false,
                .long_running = true,
            }});

        ASSERT_TRUE(payload.contains("annotations"));
        const auto& annotations = payload["annotations"];
        EXPECT_EQ(annotations["x-lfs-category"], "editor");
        EXPECT_EQ(annotations["x-lfs-kind"], "query");
        EXPECT_EQ(annotations["x-lfs-runtime"], "gui");
        EXPECT_EQ(annotations["x-lfs-thread-affinity"], "gui_thread");
        EXPECT_TRUE(annotations["readOnlyHint"].get<bool>());
        EXPECT_TRUE(annotations["idempotentHint"].get<bool>());
        EXPECT_TRUE(annotations["x-lfs-long-running"].get<bool>());
    }

    TEST(McpProtocolTest, InitializeReportsBuildVersion) {
        McpServer server;
        const auto response = server.handle_request(JsonRpcRequest{
            .id = int64_t{1},
            .method = "initialize",
            .params = json::object()});

        ASSERT_TRUE(response.result.has_value());
        const auto& result = *response.result;
        ASSERT_TRUE(result.contains("serverInfo"));
        EXPECT_EQ(result["serverInfo"]["name"], "lichtfeld-mcp");
        EXPECT_EQ(result["serverInfo"]["version"], GIT_TAGGED_VERSION);
        EXPECT_NE(result["serverInfo"]["version"], "1.0.0");
    }

    TEST(McpProtocolTest, ToolCallReturnsStructuredContent) {
        static constexpr const char* tool_name = "test.structured_response";
        ScopedToolRegistration cleanup(tool_name);

        ToolRegistry::instance().register_tool(
            McpTool{
                .name = tool_name,
                .description = "Structured response test",
                .input_schema = {.type = "object", .properties = json::object(), .required = {}},
                .metadata = McpToolMetadata{.category = "test", .kind = "query"}},
            [](const json& args) -> json {
                return json{
                    {"success", true},
                    {"echo", args.value("value", 0)},
                };
            });

        McpServer server;
        const auto init_response = server.handle_request(JsonRpcRequest{
            .id = int64_t{1},
            .method = "initialize",
            .params = json::object()});
        ASSERT_TRUE(init_response.result.has_value());

        const auto response = server.handle_request(JsonRpcRequest{
            .id = int64_t{2},
            .method = "tools/call",
            .params = json{
                {"name", tool_name},
                {"arguments", json{{"value", 42}}},
            }});

        ASSERT_TRUE(response.result.has_value());
        const auto& result = *response.result;
        ASSERT_TRUE(result.contains("structuredContent"));
        EXPECT_EQ(result["structuredContent"]["echo"], 42);
        EXPECT_FALSE(result["isError"].get<bool>());
        ASSERT_TRUE(result.contains("content"));
        ASSERT_TRUE(result["content"].is_array());
        ASSERT_FALSE(result["content"].empty());
        EXPECT_NE(result["content"][0]["text"].get<std::string>().find("\"echo\": 42"), std::string::npos);
    }

    TEST(McpProtocolTest, ToolCallIgnoresEmptyErrorStringForTransportErrors) {
        static constexpr const char* tool_name = "test.empty_error_string";
        ScopedToolRegistration cleanup(tool_name);

        ToolRegistry::instance().register_tool(
            McpTool{
                .name = tool_name,
                .description = "Empty error string should not mark transport failure",
                .input_schema = {.type = "object", .properties = json::object(), .required = {}},
                .metadata = McpToolMetadata{.category = "test", .kind = "query"}},
            [](const json&) -> json {
                return json{
                    {"success", true},
                    {"error", ""},
                };
            });

        McpServer server;
        const auto init_response = server.handle_request(JsonRpcRequest{
            .id = int64_t{1},
            .method = "initialize",
            .params = json::object()});
        ASSERT_TRUE(init_response.result.has_value());

        const auto response = server.handle_request(JsonRpcRequest{
            .id = int64_t{2},
            .method = "tools/call",
            .params = json{
                {"name", tool_name},
                {"arguments", json::object()},
            }});

        ASSERT_TRUE(response.result.has_value());
        const auto& result = *response.result;
        EXPECT_FALSE(result["isError"].get<bool>());
        EXPECT_EQ(result["structuredContent"]["error"], "");
    }

    TEST(McpProtocolTest, ResourceReadUsesMostSpecificPrefixHandler) {
        static constexpr std::string_view broad_prefix = "lichtfeld://test/";
        static constexpr std::string_view narrow_prefix = "lichtfeld://test/items/";
        ScopedResourcePrefixRegistration cleanup_broad{std::string(broad_prefix)};
        ScopedResourcePrefixRegistration cleanup_narrow{std::string(narrow_prefix)};

        ResourceRegistry::instance().register_resource_prefix(
            std::string(broad_prefix),
            [](const std::string& uri) -> std::expected<std::vector<McpResourceContent>, std::string> {
                return std::vector<McpResourceContent>{
                    McpResourceContent{
                        .uri = uri,
                        .mime_type = "application/json",
                        .content = json{{"handler", "broad"}}.dump()}};
            });

        ResourceRegistry::instance().register_resource_prefix(
            std::string(narrow_prefix),
            [](const std::string& uri) -> std::expected<std::vector<McpResourceContent>, std::string> {
                return std::vector<McpResourceContent>{
                    McpResourceContent{
                        .uri = uri,
                        .mime_type = "application/json",
                        .content = json{
                            {"handler", "narrow"},
                            {"id", uri.substr(narrow_prefix.size())}}
                                       .dump()}};
            });

        McpServer server;
        const auto init_response = server.handle_request(JsonRpcRequest{
            .id = int64_t{1},
            .method = "initialize",
            .params = json::object()});
        ASSERT_TRUE(init_response.result.has_value());

        const auto response = server.handle_request(JsonRpcRequest{
            .id = int64_t{2},
            .method = "resources/read",
            .params = json{{"uri", "lichtfeld://test/items/example"}}});

        ASSERT_TRUE(response.result.has_value());
        const auto& result = *response.result;
        ASSERT_TRUE(result.contains("contents"));
        ASSERT_TRUE(result["contents"].is_array());
        ASSERT_EQ(result["contents"].size(), 1);

        const auto parsed = json::parse(result["contents"][0]["text"].get<std::string>());
        EXPECT_EQ(parsed["handler"], "narrow");
        EXPECT_EQ(parsed["id"], "example");
    }

    TEST(McpProtocolTest, ParseRequestExtractsIdForEachIdKind) {
        const auto int_req = parse_request(R"({"jsonrpc":"2.0","id":42,"method":"ping"})");
        EXPECT_EQ(int_req.id, RequestId(int64_t{42}));

        const auto string_req = parse_request(R"({"jsonrpc":"2.0","id":"abc-123","method":"ping"})");
        EXPECT_EQ(string_req.id, RequestId(std::string("abc-123")));

        const auto null_req = parse_request(R"({"jsonrpc":"2.0","id":null,"method":"ping"})");
        EXPECT_EQ(null_req.id, RequestId(nullptr));

        const auto notification_req = parse_request(R"({"jsonrpc":"2.0","method":"notifications/initialized"})");
        EXPECT_EQ(notification_req.id, RequestId());

        EXPECT_THROW(parse_request("{not json"), json::parse_error);
    }

    TEST(McpProtocolTest, RequestIdIntegerEchoedOnSuccessAndErrorPaths) {
        McpServer server;
        const auto success = server.handle_request(JsonRpcRequest{.id = int64_t{7}, .method = "ping"});
        EXPECT_EQ(success.id, RequestId(int64_t{7}));
        EXPECT_EQ(json::parse(serialize_response(success))["id"], 7);

        const auto failure = server.handle_request(JsonRpcRequest{.id = int64_t{7}, .method = "unknown/method"});
        EXPECT_EQ(failure.id, RequestId(int64_t{7}));
        EXPECT_EQ(json::parse(serialize_response(failure))["id"], 7);
    }

    TEST(McpProtocolTest, RequestIdStringEchoedOnSuccessAndErrorPaths) {
        McpServer server;
        const auto success = server.handle_request(JsonRpcRequest{.id = std::string("req-a"), .method = "ping"});
        EXPECT_EQ(success.id, RequestId(std::string("req-a")));
        EXPECT_EQ(json::parse(serialize_response(success))["id"], "req-a");

        const auto failure = server.handle_request(JsonRpcRequest{.id = std::string("req-a"), .method = "unknown/method"});
        EXPECT_EQ(failure.id, RequestId(std::string("req-a")));
        EXPECT_EQ(json::parse(serialize_response(failure))["id"], "req-a");
    }

    TEST(McpProtocolTest, RequestIdNullSerializesAsJsonNull) {
        McpServer server;
        const auto response = server.handle_request(JsonRpcRequest{.id = nullptr, .method = "unknown/method"});
        EXPECT_EQ(response.id, RequestId(nullptr));

        const auto body = json::parse(serialize_response(response));
        ASSERT_TRUE(body.contains("id"));
        EXPECT_TRUE(body["id"].is_null());
    }

    TEST(McpProtocolTest, RequestIdAbsentOmitsIdFieldOnSerialization) {
        McpServer server;
        JsonRpcRequest req;
        req.method = "ping";
        EXPECT_EQ(req.id, RequestId());

        const auto response = server.handle_request(req);
        EXPECT_EQ(response.id, RequestId());

        const auto body = json::parse(serialize_response(response));
        EXPECT_FALSE(body.contains("id"));
    }

    TEST(McpProtocolTest, ToolsCallMissingNameReturnsInvalidParams) {
        McpServer server;
        ASSERT_TRUE(server.handle_request(JsonRpcRequest{.id = int64_t{1}, .method = "initialize"}).result.has_value());

        const auto response = server.handle_request(JsonRpcRequest{
            .id = int64_t{2},
            .method = "tools/call",
            .params = json{{"arguments", json::object()}}});

        ASSERT_TRUE(response.error.has_value());
        EXPECT_EQ(response.error->code, JsonRpcError::INVALID_PARAMS);
        EXPECT_EQ(response.id, RequestId(int64_t{2}));
    }

    TEST(McpProtocolTest, ToolsCallNonStringNameReturnsInvalidParamsNotInternalError) {
        McpServer server;
        ASSERT_TRUE(server.handle_request(JsonRpcRequest{.id = int64_t{1}, .method = "initialize"}).result.has_value());

        const auto response = server.handle_request(JsonRpcRequest{
            .id = int64_t{2},
            .method = "tools/call",
            .params = json{{"name", 123}, {"arguments", json::object()}}});

        ASSERT_TRUE(response.error.has_value());
        EXPECT_EQ(response.error->code, JsonRpcError::INVALID_PARAMS);
        EXPECT_EQ(response.id, RequestId(int64_t{2}));
    }

    TEST(McpHttpServerTest, MalformedJsonBodyRespondsWithNullIdAndParseError) {
        McpHttpServer server;
        ASSERT_TRUE(server.start(47691));

        httplib::Client client("127.0.0.1", 47691);
        const auto res = client.Post("/mcp", "{not json", "application/json");
        ASSERT_TRUE(res);

        const auto body = json::parse(res->body);
        ASSERT_TRUE(body.contains("id"));
        EXPECT_TRUE(body["id"].is_null());
        ASSERT_TRUE(body.contains("error"));
        EXPECT_EQ(body["error"]["code"], JsonRpcError::PARSE_ERROR);

        server.stop();
    }

    TEST(McpHttpServerTest, ToolHandlerThrowRespondsWithInternalErrorAndEchoedIdNoLeak) {
        static constexpr const char* tool_name = "test.throwing_tool";
        static constexpr const char* leaked_detail = "sensitive internal detail";
        ScopedToolRegistration cleanup(tool_name);

        ToolRegistry::instance().register_tool(
            McpTool{
                .name = tool_name,
                .description = "Throws for firewall testing",
                .input_schema = {.type = "object", .properties = json::object(), .required = {}},
                .metadata = McpToolMetadata{.category = "test", .kind = "command"}},
            [](const json&) -> json {
                throw std::runtime_error(leaked_detail);
            });

        McpHttpServer server;
        ASSERT_TRUE(server.start(47692));

        httplib::Client client("127.0.0.1", 47692);

        const json init_req{
            {"jsonrpc", "2.0"},
            {"id", 1},
            {"method", "initialize"},
            {"params", json::object()}};
        ASSERT_TRUE(client.Post("/mcp", init_req.dump(), "application/json"));

        const json call_req{
            {"jsonrpc", "2.0"},
            {"id", "req-42"},
            {"method", "tools/call"},
            {"params", json{{"name", tool_name}, {"arguments", json::object()}}}};
        const auto res = client.Post("/mcp", call_req.dump(), "application/json");
        ASSERT_TRUE(res);

        EXPECT_EQ(res->body.find(leaked_detail), std::string::npos);

        const auto body = json::parse(res->body);
        ASSERT_TRUE(body.contains("id"));
        EXPECT_EQ(body["id"], "req-42");
        ASSERT_TRUE(body.contains("error"));
        EXPECT_EQ(body["error"]["code"], JsonRpcError::INTERNAL_ERROR);
        EXPECT_EQ(body["error"]["message"].get<std::string>().find(leaked_detail), std::string::npos);

        server.stop();
    }

} // namespace lfs::mcp

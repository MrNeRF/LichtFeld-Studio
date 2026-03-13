/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <gtest/gtest.h>

#include "mcp/mcp_protocol.hpp"
#include "mcp/mcp_server.hpp"
#include "mcp/mcp_tools.hpp"

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

} // namespace lfs::mcp

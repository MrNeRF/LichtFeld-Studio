/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <gtest/gtest.h>

#include <algorithm>
#include <cctype>

#include "config.h"
#include "core/event_bridge/command_center_bridge.hpp"
#include "git_version.h"
#include "mcp/mcp_http_server.hpp"
#include "mcp/mcp_protocol.hpp"
#include "mcp/mcp_server.hpp"
#include "mcp/mcp_tools.hpp"

#include <httplib/httplib.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <thread>

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

        class ScopedEnvironmentVariable {
        public:
            ScopedEnvironmentVariable(const char* name,
                                      const std::optional<std::string>& value)
                : name_(name) {
                if (const char* previous = std::getenv(name))
                    previous_ = previous;
                set(value);
            }

            ~ScopedEnvironmentVariable() { set(previous_); }

        private:
            void set(const std::optional<std::string>& value) const {
#ifdef _WIN32
                (void)_putenv_s(name_.c_str(), value ? value->c_str() : "");
#else
                if (value)
                    (void)setenv(name_.c_str(), value->c_str(), 1);
                else
                    (void)unsetenv(name_.c_str());
#endif
            }

            std::string name_;
            std::optional<std::string> previous_;
        };

        int availableLoopbackPort() {
            httplib::Server picker;
            const int port = picker.bind_to_any_port("127.0.0.1");
            if (port <= 0)
                return port;
            std::jthread listener([&] { picker.listen_after_bind(); });
            picker.wait_until_ready();
            picker.stop();
            listener.join();
            return port;
        }

        bool is_claude_compatible_tool_name(const std::string& name) {
            if (name.empty() || name.size() > 64)
                return false;

            for (const unsigned char ch : name) {
                if (!std::isalnum(ch) && ch != '_' && ch != '-')
                    return false;
            }
            return true;
        }

        std::string command_tool_name(const training::OperationInfo& operation) {
            std::string target;
            switch (operation.target) {
            case training::CommandTarget::Model:
                target = "model";
                break;
            case training::CommandTarget::Optimizer:
                target = "optimizer";
                break;
            case training::CommandTarget::Session:
                target = "session";
                break;
            }
            return target + "." + operation.name;
        }

        class ScopedCommandCenterToolGeneration {
        public:
            ScopedCommandCenterToolGeneration()
                : previous_(event::CommandCenterBridge::instance().get()) {
                auto& command_center = training::CommandCenter::instance();
                for (const auto& operation : command_center.operations()) {
                    tool_names_.push_back(command_tool_name(operation));
                }
                event::CommandCenterBridge::instance().set(&command_center);
                ToolRegistry::instance().generate_from_command_center();
            }

            ~ScopedCommandCenterToolGeneration() {
                for (const auto& name : tool_names_) {
                    ToolRegistry::instance().unregister_tool(name);
                }
                event::CommandCenterBridge::instance().set(previous_);
            }

        private:
            training::CommandCenter* previous_;
            std::vector<std::string> tool_names_;
        };

    } // namespace

    TEST(McpProtocolTest, ToolJsonKeepsStandardAnnotationsAndMovesLichtFeldMetadataToMeta) {
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

        EXPECT_EQ(payload["name"], "test_describe");

        ASSERT_TRUE(payload.contains("annotations"));
        const auto& annotations = payload["annotations"];
        ASSERT_TRUE(annotations.is_object());
        EXPECT_EQ(annotations.size(), 3);
        EXPECT_TRUE(annotations.contains("readOnlyHint"));
        EXPECT_TRUE(annotations.contains("destructiveHint"));
        EXPECT_TRUE(annotations.contains("idempotentHint"));
        EXPECT_TRUE(annotations["readOnlyHint"].get<bool>());
        EXPECT_TRUE(annotations["idempotentHint"].get<bool>());
        EXPECT_FALSE(annotations["destructiveHint"].get<bool>());

        ASSERT_TRUE(payload.contains("_meta"));
        const auto& meta = payload["_meta"];
        EXPECT_EQ(meta["app.lichtfeld/category"], "editor");
        EXPECT_EQ(meta["app.lichtfeld/kind"], "query");
        EXPECT_EQ(meta["app.lichtfeld/runtime"], "gui");
        EXPECT_EQ(meta["app.lichtfeld/thread_affinity"], "gui_thread");
        EXPECT_TRUE(meta["app.lichtfeld/user_visible"].get<bool>());
        EXPECT_TRUE(meta["app.lichtfeld/long_running"].get<bool>());
    }

    TEST(McpProtocolTest, ToolJsonSerializesNullSchemaPropertiesAsObject) {
        const auto payload = tool_to_json(McpTool{
            .name = "test.empty_schema",
            .description = "Empty schema properties should serialize as an object",
            .input_schema = {.type = "object", .properties = json(), .required = {}}});

        ASSERT_TRUE(payload.contains("inputSchema"));
        const auto& schema = payload["inputSchema"];
        ASSERT_TRUE(schema.contains("properties"));
        EXPECT_TRUE(schema["properties"].is_object());
        EXPECT_TRUE(schema["properties"].empty());
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

    TEST(McpProtocolTest, ToolsListSerializesNullSchemaPropertiesAsObjectAndCallContractIsUnchanged) {
        static constexpr const char* tool_name = "test.null_schema_properties";
        ScopedToolRegistration cleanup(tool_name);

        ToolRegistry::instance().register_tool(
            McpTool{
                .name = tool_name,
                .description = "Null schema properties test",
                .input_schema = {.type = "object", .properties = json(), .required = {}}},
            [](const json&) -> json {
                return json{
                    {"success", true},
                    {"state", "ok"},
                };
            });

        McpServer server;
        const auto init_response = server.handle_request(JsonRpcRequest{
            .id = int64_t{1},
            .method = "initialize",
            .params = json::object()});
        ASSERT_TRUE(init_response.result.has_value());

        const auto list_response = server.handle_request(JsonRpcRequest{
            .id = int64_t{2},
            .method = "tools/list",
            .params = json::object()});
        ASSERT_TRUE(list_response.result.has_value());

        const auto& tools = (*list_response.result)["tools"];
        const auto found = std::find_if(tools.begin(), tools.end(), [](const json& tool) {
            return tool.value("name", "") == "test_null_schema_properties";
        });
        ASSERT_NE(found, tools.end());
        ASSERT_TRUE((*found)["inputSchema"].contains("properties"));
        EXPECT_TRUE((*found)["inputSchema"]["properties"].is_object());
        EXPECT_TRUE((*found)["inputSchema"]["properties"].empty());

        const auto call_response = server.handle_request(JsonRpcRequest{
            .id = int64_t{3},
            .method = "tools/call",
            .params = json{
                {"name", tool_name},
                {"arguments", json::object()},
            }});
        ASSERT_TRUE(call_response.result.has_value());
        const auto& result = *call_response.result;
        ASSERT_TRUE(result.contains("content"));
        ASSERT_TRUE(result.contains("structuredContent"));
        ASSERT_TRUE(result.contains("isError"));
        EXPECT_FALSE(result["isError"].get<bool>());
        EXPECT_EQ(result["structuredContent"]["state"], "ok");
    }

    TEST(McpProtocolTest, ToolNamesAreNormalizedForListAndCallsAcceptBothForms) {
        static constexpr const char* dotted_tool_name = "test.normalized_tool";
        static constexpr const char* normalized_tool_name = "test_normalized_tool";
        ScopedToolRegistration cleanup(dotted_tool_name);

        ToolRegistry::instance().register_tool(
            McpTool{
                .name = dotted_tool_name,
                .description = "Normalized tool name test",
                .input_schema = {.type = "object", .properties = json::object(), .required = {}}},
            [](const json& args) -> json {
                return json{
                    {"success", true},
                    {"marker", "normalized"},
                    {"value", args.value("value", 0)},
                };
            });

        const auto registered_tools = ToolRegistry::instance().list_tools();
        const auto internal_tool = std::find_if(
            registered_tools.begin(),
            registered_tools.end(),
            [](const McpTool& tool) { return tool.name == dotted_tool_name; });
        ASSERT_NE(internal_tool, registered_tools.end());

        McpServer server;
        const auto init_response = server.handle_request(JsonRpcRequest{
            .id = int64_t{1},
            .method = "initialize",
            .params = json::object()});
        ASSERT_TRUE(init_response.result.has_value());

        const auto list_response = server.handle_request(JsonRpcRequest{
            .id = int64_t{2},
            .method = "tools/list",
            .params = json::object()});
        ASSERT_TRUE(list_response.result.has_value());

        bool found_normalized_name = false;
        for (const auto& tool : (*list_response.result)["tools"]) {
            const auto name = tool["name"].get<std::string>();
            EXPECT_TRUE(is_claude_compatible_tool_name(name)) << name;
            EXPECT_NE(name, dotted_tool_name);
            if (name == normalized_tool_name)
                found_normalized_name = true;
        }
        EXPECT_TRUE(found_normalized_name);

        const auto dotted_call_response = server.handle_request(JsonRpcRequest{
            .id = int64_t{3},
            .method = "tools/call",
            .params = json{
                {"name", dotted_tool_name},
                {"arguments", json{{"value", 7}}},
            }});
        ASSERT_TRUE(dotted_call_response.result.has_value());
        EXPECT_FALSE((*dotted_call_response.result)["isError"].get<bool>());
        EXPECT_EQ((*dotted_call_response.result)["structuredContent"]["marker"], "normalized");
        EXPECT_EQ((*dotted_call_response.result)["structuredContent"]["value"], 7);

        const auto normalized_call_response = server.handle_request(JsonRpcRequest{
            .id = int64_t{4},
            .method = "tools/call",
            .params = json{
                {"name", normalized_tool_name},
                {"arguments", json{{"value", 11}}},
            }});
        ASSERT_TRUE(normalized_call_response.result.has_value());
        EXPECT_FALSE((*normalized_call_response.result)["isError"].get<bool>());
        EXPECT_EQ((*normalized_call_response.result)["structuredContent"]["marker"], "normalized");
        EXPECT_EQ((*normalized_call_response.result)["structuredContent"]["value"], 11);
        ASSERT_TRUE((*normalized_call_response.result).contains("content"));
        ASSERT_TRUE((*normalized_call_response.result).contains("structuredContent"));
    }

    TEST(McpProtocolTest, ToolRegistryUnregisterAcceptsDottedAndNormalizedNames) {
        static constexpr const char* dotted_tool_name = "test.unregister_forms";
        static constexpr const char* normalized_tool_name = "test_unregister_forms";
        ScopedToolRegistration cleanup(dotted_tool_name);

        const auto register_test_tool = [] {
            ToolRegistry::instance().register_tool(
                McpTool{
                    .name = dotted_tool_name,
                    .description = "Unregister name forms test",
                    .input_schema = {.type = "object", .properties = json::object(), .required = {}}},
                [](const json&) -> json { return json{{"marker", "registered"}}; });
        };
        const auto is_registered = [] {
            const auto tools = ToolRegistry::instance().list_tools();
            return std::any_of(tools.begin(), tools.end(), [](const McpTool& tool) {
                return normalize_tool_name(tool.name) == normalized_tool_name;
            });
        };

        register_test_tool();
        ASSERT_TRUE(is_registered());
        ToolRegistry::instance().unregister_tool(dotted_tool_name);
        EXPECT_FALSE(is_registered());

        register_test_tool();
        ASSERT_TRUE(is_registered());
        ToolRegistry::instance().unregister_tool(normalized_tool_name);
        EXPECT_FALSE(is_registered());
    }

    TEST(McpProtocolTest, ToolRegistryRejectsNormalizedNameCollisions) {
        static constexpr const char* dotted_tool_name = "foo.bar";
        static constexpr const char* normalized_tool_name = "foo_bar";
        ScopedToolRegistration cleanup(dotted_tool_name);

        ToolRegistry::instance().register_tool(
            McpTool{
                .name = dotted_tool_name,
                .description = "First collision candidate",
                .input_schema = {.type = "object", .properties = json::object(), .required = {}}},
            [](const json&) -> json { return json{{"marker", "first"}}; });
        ToolRegistry::instance().register_tool(
            McpTool{
                .name = normalized_tool_name,
                .description = "Second collision candidate",
                .input_schema = {.type = "object", .properties = json::object(), .required = {}}},
            [](const json&) -> json { return json{{"marker", "second"}}; });

        const auto tools = ToolRegistry::instance().list_tools();
        const auto matching_tool = std::find_if(tools.begin(), tools.end(), [](const McpTool& tool) {
            return normalize_tool_name(tool.name) == normalized_tool_name;
        });
        ASSERT_NE(matching_tool, tools.end());
        EXPECT_EQ(matching_tool->name, dotted_tool_name);
        EXPECT_EQ(
            std::count_if(tools.begin(), tools.end(), [](const McpTool& tool) {
                return normalize_tool_name(tool.name) == normalized_tool_name;
            }),
            1);

        EXPECT_EQ(ToolRegistry::instance().call_tool(dotted_tool_name, json::object())["marker"], "first");
        EXPECT_EQ(ToolRegistry::instance().call_tool(normalized_tool_name, json::object())["marker"], "first");
    }

    TEST(McpProtocolTest, ToolRegistryListNameRoundTripsThroughBothUnregisterForms) {
        static constexpr const char* dotted_tool_name = "test.registry_roundtrip";
        static constexpr const char* normalized_tool_name = "test_registry_roundtrip";
        ScopedToolRegistration cleanup(dotted_tool_name);

        McpServer server;
        ASSERT_TRUE(server.handle_request(JsonRpcRequest{
                                              .id = int64_t{1},
                                              .method = "initialize",
                                              .params = json::object()})
                        .result.has_value());

        int64_t request_id = 2;
        for (const std::string unregister_name : {dotted_tool_name, normalized_tool_name}) {
            ToolRegistry::instance().register_tool(
                McpTool{
                    .name = dotted_tool_name,
                    .description = "Registry round-trip test",
                    .input_schema = {.type = "object", .properties = json::object(), .required = {}}},
                [](const json&) -> json { return json{{"success", true}}; });

            const auto list_response = server.handle_request(JsonRpcRequest{
                .id = request_id++,
                .method = "tools/list",
                .params = json::object()});
            ASSERT_TRUE(list_response.result.has_value());
            const auto& listed_tools = (*list_response.result)["tools"];
            EXPECT_NE(
                std::find_if(listed_tools.begin(), listed_tools.end(), [](const json& tool) {
                    return tool.value("name", "") == normalized_tool_name;
                }),
                listed_tools.end());

            ToolRegistry::instance().unregister_tool(unregister_name);

            const auto after_unregister = server.handle_request(JsonRpcRequest{
                .id = request_id++,
                .method = "tools/list",
                .params = json::object()});
            ASSERT_TRUE(after_unregister.result.has_value());
            const auto& remaining_tools = (*after_unregister.result)["tools"];
            EXPECT_EQ(
                std::find_if(remaining_tools.begin(), remaining_tools.end(), [](const json& tool) {
                    return tool.value("name", "") == normalized_tool_name;
                }),
                remaining_tools.end());
        }
    }

    TEST(McpProtocolTest, ToolRegistryRejectsInvalidNormalizedNames) {
        const std::string invalid_character_name = "test/invalid";
        const std::string overlong_name(65, 'x');
        ScopedToolRegistration cleanup_invalid_character(invalid_character_name);
        ScopedToolRegistration cleanup_overlong(overlong_name);

        ToolRegistry::instance().register_tool(
            McpTool{
                .name = invalid_character_name,
                .description = "Invalid character test",
                .input_schema = {.type = "object", .properties = json::object(), .required = {}}},
            [](const json&) -> json { return json{{"success", true}}; });
        ToolRegistry::instance().register_tool(
            McpTool{
                .name = overlong_name,
                .description = "Overlong name test",
                .input_schema = {.type = "object", .properties = json::object(), .required = {}}},
            [](const json&) -> json { return json{{"success", true}}; });

        McpServer server;
        ASSERT_TRUE(server.handle_request(JsonRpcRequest{
                                              .id = int64_t{1},
                                              .method = "initialize",
                                              .params = json::object()})
                        .result.has_value());
        const auto list_response = server.handle_request(JsonRpcRequest{
            .id = int64_t{2},
            .method = "tools/list",
            .params = json::object()});
        ASSERT_TRUE(list_response.result.has_value());
        const auto& listed_tools = (*list_response.result)["tools"];
        EXPECT_EQ(
            std::find_if(listed_tools.begin(), listed_tools.end(), [&](const json& tool) {
                return tool.value("name", "") == normalize_tool_name(invalid_character_name);
            }),
            listed_tools.end());
        EXPECT_EQ(
            std::find_if(listed_tools.begin(), listed_tools.end(), [&](const json& tool) {
                return tool.value("name", "") == normalize_tool_name(overlong_name);
            }),
            listed_tools.end());
    }

    TEST(McpProtocolTest, CommandCenterZeroArgumentToolHasObjectPropertiesBeforeSerialization) {
        ScopedCommandCenterToolGeneration generated_tools;

        const auto tools = ToolRegistry::instance().list_tools();
        const auto pause_tool = std::find_if(tools.begin(), tools.end(), [](const McpTool& tool) {
            return tool.name == "session.pause";
        });
        ASSERT_NE(pause_tool, tools.end());
        EXPECT_TRUE(pause_tool->input_schema.properties.is_object());
        EXPECT_TRUE(pause_tool->input_schema.properties.empty());
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

    TEST(McpProtocolTest, WireSerializationIsTotalOverIllFormedUtf8) {
        JsonRpcResponse response;
        response.id = 7;
        response.result = json{{"error", std::string("bad \xC3 byte")}};
        std::string serialized;
        EXPECT_NO_THROW(serialized = serialize_response(response));
        EXPECT_NE(serialized.find("\xEF\xBF\xBD"), std::string::npos);

        EXPECT_NO_THROW(
            serialized = serialize_notification("event", json{{"error", std::string("\xFF")}}));
        EXPECT_NE(serialized.find("\xEF\xBF\xBD"), std::string::npos);
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

        const auto status = server.status();
        EXPECT_EQ(status.request_count, 1u);
        EXPECT_EQ(status.success_count, 0u);
        EXPECT_EQ(status.error_count, 1u);
        EXPECT_EQ(status.endpoints,
                  (std::vector<std::string>{
                      "http://127.0.0.1:47691/mcp",
                      "http://localhost:47691/mcp",
                  }));

        server.stop();
    }

    TEST(McpHttpServerTest, DisabledAndInvalidConfigurationsReportTruthfulStatus) {
        McpHttpServer server;
        EXPECT_TRUE(server.start(McpHttpConfig{
            .enabled = false,
            .expose_network = true,
            .port = 50001,
            .request_logging = true,
        }));
        auto status = server.status();
        EXPECT_FALSE(status.enabled);
        EXPECT_FALSE(status.running);
        EXPECT_EQ(status.phase, McpHttpPhase::Disabled);
        EXPECT_TRUE(status.expose_network);
        EXPECT_EQ(status.port, 50001);
        EXPECT_TRUE(status.request_logging);
        EXPECT_TRUE(status.error.empty());

        EXPECT_FALSE(server.applyConfig(McpHttpConfig{.enabled = true, .port = 0}));
        status = server.status();
        EXPECT_TRUE(status.enabled);
        EXPECT_FALSE(status.running);
        EXPECT_EQ(status.phase, McpHttpPhase::Failed);
        EXPECT_EQ(status.error_kind, McpHttpErrorKind::InvalidPort);
        EXPECT_FALSE(status.error.empty());

        EXPECT_TRUE(server.applyConfig(McpHttpConfig{.enabled = false, .port = 50002}));
        status = server.status();
        EXPECT_FALSE(status.enabled);
        EXPECT_EQ(status.phase, McpHttpPhase::Disabled);
        EXPECT_EQ(status.port, 50002);
        EXPECT_TRUE(status.error.empty());
    }

    TEST(McpHttpServerTest, FailedBindCanRetryTheSameConfiguration) {
        httplib::Server blocker;
        bool exclusive_socket_configured = false;
        blocker.set_socket_options([&exclusive_socket_configured](const socket_t socket) {
#ifdef _WIN32
            constexpr int enabled = 1;
            exclusive_socket_configured =
                ::setsockopt(socket,
                             SOL_SOCKET,
                             SO_EXCLUSIVEADDRUSE,
                             reinterpret_cast<const char*>(&enabled),
                             sizeof(enabled)) == 0;
#else
            // Supplying a no-op callback prevents cpp-httplib from enabling
            // SO_REUSEPORT/SO_REUSEADDR for this test-only port holder.
            (void)socket;
            exclusive_socket_configured = true;
#endif
        });
        const int port = blocker.bind_to_any_port("127.0.0.1");
        ASSERT_GT(port, 0);
        ASSERT_TRUE(exclusive_socket_configured);
        std::jthread blocker_thread([&] { blocker.listen_after_bind(); });
        blocker.wait_until_ready();

        McpHttpServer server;
        const McpHttpConfig config{
            .enabled = true,
            .expose_network = false,
            .port = port,
            .request_logging = false,
        };
        EXPECT_FALSE(server.start(config));
        auto status = server.status();
        EXPECT_EQ(status.phase, McpHttpPhase::Failed);
        EXPECT_EQ(status.error_kind, McpHttpErrorKind::BindFailed);
        EXPECT_EQ(status.error_address, "127.0.0.1");
        EXPECT_EQ(status.error_port, port);

        blocker.stop();
        blocker_thread.join();

        EXPECT_TRUE(server.applyConfig(config));
        status = server.status();
        EXPECT_EQ(status.phase, McpHttpPhase::Running);
        EXPECT_TRUE(status.running);
        EXPECT_EQ(status.error_kind, McpHttpErrorKind::None);
        server.stop();
    }

    TEST(McpHttpServerTest, ListenerRejectsConcurrentOwnershipOfTheSamePort) {
        const int port = availableLoopbackPort();
        ASSERT_GT(port, 0);
        const McpHttpConfig config{
            .enabled = true,
            .expose_network = false,
            .port = port,
            .request_logging = false,
        };

        McpHttpServer first;
        ASSERT_TRUE(first.start(config));

        McpHttpServer second;
        EXPECT_FALSE(second.start(config));
        const auto status = second.status();
        EXPECT_EQ(status.phase, McpHttpPhase::Failed);
        EXPECT_EQ(status.error_kind, McpHttpErrorKind::BindFailed);
        EXPECT_EQ(status.error_address, "127.0.0.1");
        EXPECT_EQ(status.error_port, port);

        first.stop();
    }

    TEST(McpHttpServerTest, FastPathApplyConfigPreservesStartedEndpoints) {
        const int port = availableLoopbackPort();
        ASSERT_GT(port, 0);
        McpHttpServer server;
        ASSERT_TRUE(server.start(McpHttpConfig{
            .enabled = true,
            .expose_network = false,
            .port = port,
            .request_logging = false,
        }));
        const auto started_endpoints = server.status().endpoints;
        ASSERT_FALSE(started_endpoints.empty());

        ASSERT_TRUE(server.applyConfig(McpHttpConfig{
            .enabled = true,
            .expose_network = false,
            .port = port,
            .request_logging = true,
        }));
        EXPECT_EQ(server.status().endpoints, started_endpoints);
        EXPECT_FALSE(server.status().endpoints.empty());
        EXPECT_TRUE(server.status().request_logging);
        server.stop();
    }

    TEST(McpHttpServerTest, HttplibStopClosesBoundButNotListeningSocket) {
        httplib::Server bound;
        const int port = bound.bind_to_any_port("127.0.0.1");
        ASSERT_GT(port, 0);
        bound.stop();

        httplib::Server rebound;
        EXPECT_TRUE(rebound.bind_to_port("127.0.0.1", port));
        rebound.stop();
    }

    TEST(McpHttpServerTest, RapidQueuedDisableCannotLeaveAListenerRunning) {
        const int port = availableLoopbackPort();
        ASSERT_GT(port, 0);
        McpHttpServer server;
        ASSERT_TRUE(server.start(McpHttpConfig{.enabled = false, .port = port}));
        setActiveMcpHttpServer(&server);

        ASSERT_TRUE(applyActiveMcpHttpConfig({.enabled = true, .port = port}));
        ASSERT_TRUE(applyActiveMcpHttpConfig({.enabled = false, .port = port}));

        const auto deadline =
            std::chrono::steady_clock::now() + std::chrono::seconds(5);
        bool reached_disabled = false;
        while (std::chrono::steady_clock::now() < deadline) {
            const auto polled = server.status();
            if (polled.phase == McpHttpPhase::Disabled && !polled.running) {
                reached_disabled = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        ASSERT_TRUE(reached_disabled);

        setActiveMcpHttpServer(nullptr);

        const auto status = server.status();
        EXPECT_FALSE(status.enabled);
        EXPECT_FALSE(status.running);
        EXPECT_EQ(status.phase, McpHttpPhase::Disabled);
    }

    TEST(McpHttpServerTest, DetachCancelsQueuedEnableAndStopsListener) {
        const int port = availableLoopbackPort();
        ASSERT_GT(port, 0);
        McpHttpServer server;
        ASSERT_TRUE(server.start(McpHttpConfig{.enabled = false, .port = port}));
        setActiveMcpHttpServer(&server);

        ASSERT_TRUE(applyActiveMcpHttpConfig({.enabled = true, .port = port}));
        setActiveMcpHttpServer(nullptr);

        const auto status = server.status();
        EXPECT_FALSE(status.enabled);
        EXPECT_FALSE(status.running);
        EXPECT_EQ(status.phase, McpHttpPhase::Disabled);
    }

    TEST(McpHttpServerTest, ActiveConfigPublishesStagedStatusBeforeRestartCompletes) {
        McpHttpServer server;
        ASSERT_TRUE(server.start(McpHttpConfig{.enabled = false, .port = 47694}));
        setActiveMcpHttpServer(&server);

        EXPECT_TRUE(applyActiveMcpHttpConfig({
            .enabled = false,
            .expose_network = true,
            .port = 47695,
            .request_logging = false,
        }));
        const auto status = activeMcpHttpStatus();
        setActiveMcpHttpServer(nullptr);

        EXPECT_FALSE(status.enabled);
        EXPECT_TRUE(status.expose_network);
        EXPECT_EQ(status.port, 47695);
        ASSERT_GE(status.endpoints.size(), 2u);
        EXPECT_EQ(status.endpoints[0], "http://127.0.0.1:47695/mcp");
        EXPECT_EQ(status.endpoints[1], "http://localhost:47695/mcp");
    }

    TEST(McpHttpServerTest, RequestLogClassifiesErrorsWithoutParametersOrPayloads) {
        const auto root = std::filesystem::temp_directory_path() /
                          "lfs_mcp_http_request_logging";
        std::error_code filesystem_error;
        std::filesystem::remove_all(root, filesystem_error);
        const ScopedEnvironmentVariable home("LFS_HOME", root.string());
        const ScopedEnvironmentVariable safe_mode("LFS_SAFE_MODE", std::nullopt);

        McpHttpServer server;
        ASSERT_TRUE(server.start(McpHttpConfig{
            .enabled = true,
            .expose_network = false,
            .port = 47696,
            .request_logging = true,
        }));

        constexpr std::string_view secret = "secret-value-that-must-not-be-logged";
        const json request{
            {"jsonrpc", "2.0"},
            {"id", "log-contract"},
            {"method", "initialize"},
            {"params", json{{"secret", secret}}},
        };
        httplib::Client client("127.0.0.1", 47696);
        ASSERT_TRUE(client.Post("/mcp", request.dump(), "application/json"));

        const json unknown_method{
            {"jsonrpc", "2.0"},
            {"id", "unknown-method"},
            {"method", "method/that/does/not/exist"},
            {"params", json::object()},
        };
        ASSERT_TRUE(client.Post("/mcp", unknown_method.dump(), "application/json"));

        const json missing_tool{
            {"jsonrpc", "2.0"},
            {"id", "missing-tool"},
            {"method", "tools/call"},
            {"params", json{{"name", secret}, {"arguments", json::object()}}},
        };
        ASSERT_TRUE(client.Post("/mcp", missing_tool.dump(), "application/json"));
        ASSERT_TRUE(client.Post("/mcp", "{malformed", "application/json"));
        server.stop();

        const auto status = server.status();
        ASSERT_FALSE(status.log_file.empty());
        std::ifstream input(std::filesystem::path(status.log_file), std::ios::binary);
        ASSERT_TRUE(input.is_open());
        const std::string contents((std::istreambuf_iterator<char>(input)), {});
        EXPECT_NE(contents.find("initialize"), std::string::npos);
        EXPECT_NE(contents.find("log-contract"), std::string::npos);
        EXPECT_NE(contents.find("success"), std::string::npos);
        EXPECT_NE(contents.find("\"error_type\":\"json_rpc\""), std::string::npos);
        EXPECT_NE(contents.find("\"error_stage\":\"parse\""), std::string::npos);
        EXPECT_NE(contents.find("\"error_reason\":\"parse_error\""),
                  std::string::npos);
        EXPECT_NE(contents.find("\"jsonrpc_error_code\":-32700"), std::string::npos);
        EXPECT_NE(contents.find("\"error_stage\":\"dispatch\""), std::string::npos);
        EXPECT_NE(contents.find("\"error_reason\":\"method_not_found\""),
                  std::string::npos);
        EXPECT_NE(contents.find("\"jsonrpc_error_code\":-32601"), std::string::npos);
        EXPECT_NE(contents.find("\"error_type\":\"tool_execution\""),
                  std::string::npos);
        EXPECT_NE(contents.find("\"error_stage\":\"tool_call\""), std::string::npos);
        EXPECT_NE(contents.find("\"error_reason\":\"NotFound\""), std::string::npos);
        EXPECT_NE(contents.find("\"application_error_code\":\"NotFound\""),
                  std::string::npos);
        EXPECT_NE(contents.find("\"application_error_domain\":\"MCP\""),
                  std::string::npos);
        EXPECT_NE(contents.find("\"retryable\":false"), std::string::npos);
        EXPECT_EQ(contents.find(secret), std::string::npos);
        EXPECT_EQ(contents.find("\"params\""), std::string::npos);

        std::istringstream records(contents);
        std::string line;
        size_t request_records = 0;
        while (std::getline(records, line)) {
            const auto record = json::parse(line);
            if (record.value("event", std::string{}) != "request")
                continue;
            ++request_records;
            EXPECT_EQ(record.value("source_ip", std::string{}), "127.0.0.1");
            EXPECT_GT(record.value("source_port", -1), 0);
            EXPECT_EQ(record.value("destination_ip", std::string{}), "127.0.0.1");
            EXPECT_EQ(record.value("destination_port", -1), 47696);
        }
        EXPECT_EQ(request_records, 4u);

        EXPECT_EQ(status.request_count, 4u);
        EXPECT_EQ(status.success_count, 1u);
        EXPECT_EQ(status.error_count, 3u);

        input.close();
        std::filesystem::remove_all(root, filesystem_error);
    }

    TEST(McpHttpServerTest, ToolErrorsUseResultEnvelopeWithoutLeakingDetailsAndCountAsErrors) {
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

        // A thrown tool handler is now a tool-execution error, not a protocol
        // error: the registry catches it and returns a successful JSON-RPC
        // response whose tool result carries a stable envelope. The two
        // load-bearing invariants are preserved: the id is echoed and the
        // thrown detail appears nowhere in the body.
        EXPECT_EQ(res->body.find(leaked_detail), std::string::npos);

        const auto body = json::parse(res->body);
        ASSERT_TRUE(body.contains("id"));
        EXPECT_EQ(body["id"], "req-42");
        ASSERT_TRUE(body.contains("result"));
        const auto& result = body["result"];
        EXPECT_TRUE(result["isError"].get<bool>());
        ASSERT_TRUE(result["structuredContent"].contains("error"));
        EXPECT_EQ(result["structuredContent"]["error"]["code"], "Internal");

        const json missing_tool_req{
            {"jsonrpc", "2.0"},
            {"id", "req-missing"},
            {"method", "tools/call"},
            {"params", json{{"name", "tool_that_does_not_exist"},
                            {"arguments", json::object()}}}};
        const auto missing_tool_res =
            client.Post("/mcp", missing_tool_req.dump(), "application/json");
        ASSERT_TRUE(missing_tool_res);
        const auto missing_tool_body = json::parse(missing_tool_res->body);
        ASSERT_TRUE(missing_tool_body.contains("result"));
        EXPECT_TRUE(missing_tool_body["result"]["isError"].get<bool>());

        const auto status = server.status();
        EXPECT_EQ(status.request_count, 3u);
        EXPECT_EQ(status.success_count, 1u);
        EXPECT_EQ(status.error_count, 2u);

        server.stop();
    }

    TEST(McpHttpServerTest, HandlerThrowOutsideToolBecomesInternalErrorWithEnvelopeData) {
        static constexpr const char* prefix = "lichtfeld://throwtest/";
        static constexpr const char* leaked_detail = "resource sensitive detail";
        ScopedResourcePrefixRegistration cleanup(prefix);

        ResourceRegistry::instance().register_resource_prefix(
            prefix,
            [](const std::string&) -> std::expected<std::vector<McpResourceContent>, std::string> {
                throw std::runtime_error(leaked_detail);
            });

        McpHttpServer server;
        ASSERT_TRUE(server.start(47693));

        httplib::Client client("127.0.0.1", 47693);

        const json init_req{
            {"jsonrpc", "2.0"},
            {"id", 1},
            {"method", "initialize"},
            {"params", json::object()}};
        ASSERT_TRUE(client.Post("/mcp", init_req.dump(), "application/json"));

        const json read_req{
            {"jsonrpc", "2.0"},
            {"id", "res-7"},
            {"method", "resources/read"},
            {"params", json{{"uri", "lichtfeld://throwtest/item"}}}};
        const auto res = client.Post("/mcp", read_req.dump(), "application/json");
        ASSERT_TRUE(res);

        EXPECT_EQ(res->body.find(leaked_detail), std::string::npos);

        const auto body = json::parse(res->body);
        EXPECT_EQ(body["id"], "res-7");
        ASSERT_TRUE(body.contains("error"));
        EXPECT_EQ(body["error"]["code"], JsonRpcError::INTERNAL_ERROR);
        ASSERT_TRUE(body["error"].contains("data"));
        EXPECT_TRUE(body["error"]["data"].contains("code"));
        EXPECT_EQ(body["error"]["data"]["domain"], "MCP");

        server.stop();
    }

    TEST(McpProtocolTest, LegacyStringErrorBecomesEnvelopeWithCompatMirror) {
        static constexpr const char* tool_name = "test.legacy_string_error";
        ScopedToolRegistration cleanup(tool_name);

        ToolRegistry::instance().register_tool(
            McpTool{
                .name = tool_name,
                .description = "Legacy string error handler",
                .input_schema = {.type = "object", .properties = json::object(), .required = {}},
                .metadata = McpToolMetadata{.category = "test", .kind = "query"}},
            [](const json&) -> json {
                return json{{"error", "No scene loaded"}, {"detail_field", 7}};
            });

        McpServer server;
        ASSERT_TRUE(server.handle_request(JsonRpcRequest{
                                              .id = int64_t{1},
                                              .method = "initialize",
                                              .params = json::object()})
                        .result.has_value());

        const auto response = server.handle_request(JsonRpcRequest{
            .id = int64_t{2},
            .method = "tools/call",
            .params = json{{"name", tool_name}, {"arguments", json::object()}}});

        ASSERT_TRUE(response.result.has_value());
        const auto& structured = (*response.result)["structuredContent"];
        EXPECT_TRUE((*response.result)["isError"].get<bool>());
        EXPECT_EQ(structured["error"]["code"], "FailedPrecondition");
        EXPECT_EQ(structured["error"]["domain"], "MCP");
        EXPECT_EQ(structured["error"]["message"], "No scene loaded");
        EXPECT_FALSE(structured["error"]["retryable"].get<bool>());
        EXPECT_EQ(structured["error_message"], "No scene loaded");
        EXPECT_EQ(structured["detail_field"], 7);
    }

    TEST(McpProtocolTest, ToolNotFoundAndMissingParameterYieldTypedEnvelopes) {
        const auto not_found =
            ToolRegistry::instance().call_tool("does.not.exist", json::object());
        ASSERT_TRUE(not_found.contains("error"));
        EXPECT_EQ(not_found["error"]["code"], "NotFound");
        EXPECT_EQ(not_found["error"]["details"]["parameter"], "does.not.exist");
        EXPECT_EQ(not_found["error_message"], "Tool not found: does.not.exist");

        static constexpr const char* tool_name = "test.requires_param";
        ScopedToolRegistration cleanup(tool_name);
        ToolRegistry::instance().register_tool(
            McpTool{
                .name = tool_name,
                .description = "Requires a parameter",
                .input_schema = {.type = "object", .properties = json::object(), .required = {"value"}},
                .metadata = McpToolMetadata{.category = "test", .kind = "query"}},
            [](const json&) -> json { return json{{"success", true}}; });

        const auto missing = ToolRegistry::instance().call_tool(tool_name, json::object());
        ASSERT_TRUE(missing.contains("error"));
        EXPECT_EQ(missing["error"]["code"], "InvalidArgument");
        EXPECT_EQ(missing["error"]["details"]["parameter"], "value");
        EXPECT_EQ(missing["error_message"], "Missing required parameter: value");
    }

    TEST(McpProtocolTest, TypedEnvelopeHandlerResultIsPassedThroughWithMirror) {
        static constexpr const char* tool_name = "test.typed_envelope";
        ScopedToolRegistration cleanup(tool_name);
        ToolRegistry::instance().register_tool(
            McpTool{
                .name = tool_name,
                .description = "Emits a typed envelope directly",
                .input_schema = {.type = "object", .properties = json::object(), .required = {}},
                .metadata = McpToolMetadata{.category = "test", .kind = "query"}},
            [](const json&) -> json {
                return json{{"error", json{
                                          {"code", "NotFound"},
                                          {"domain", "IO"},
                                          {"message", "Dataset was not found"},
                                          {"retryable", false},
                                          {"operation_id", 0}}}};
            });

        const auto result = ToolRegistry::instance().call_tool(tool_name, json::object());
        ASSERT_TRUE(result["error"].is_object());
        EXPECT_EQ(result["error"]["code"], "NotFound");
        EXPECT_EQ(result["error"]["domain"], "IO");
        EXPECT_EQ(result["error_message"], "Dataset was not found");
    }

    TEST(McpProtocolTest, ResourceUnknownUriYieldsNotFoundEnvelopeData) {
        McpServer server;
        ASSERT_TRUE(server.handle_request(JsonRpcRequest{
                                              .id = int64_t{1},
                                              .method = "initialize",
                                              .params = json::object()})
                        .result.has_value());

        const auto response = server.handle_request(JsonRpcRequest{
            .id = int64_t{2},
            .method = "resources/read",
            .params = json{{"uri", "lichtfeld://nonexistent/thing"}}});

        ASSERT_TRUE(response.error.has_value());
        EXPECT_EQ(response.error->code, JsonRpcError::INVALID_PARAMS);
        ASSERT_TRUE(response.error->data.has_value());
        EXPECT_EQ((*response.error->data)["code"], "NotFound");
        EXPECT_EQ((*response.error->data)["domain"], "MCP");
    }

    TEST(McpProtocolTest, RegistryCaughtHandlerFailureCarriesOperationId) {
        static constexpr const char* tool_name = "test.correlated_throw";
        ScopedToolRegistration cleanup(tool_name);
        ToolRegistry::instance().register_tool(
            McpTool{
                .name = tool_name,
                .description = "Throws for correlation testing",
                .input_schema = {.type = "object", .properties = json::object(), .required = {}},
                .metadata = McpToolMetadata{.category = "test", .kind = "command"}},
            [](const json&) -> json { throw std::runtime_error("boom"); });

        const lfs::OperationId operation_id = lfs::OperationId::generate();
        const auto result =
            ToolRegistry::instance().call_tool(tool_name, json::object(), operation_id);

        ASSERT_TRUE(result.contains("error"));
        EXPECT_NE(operation_id.value(), 0u);
        EXPECT_EQ(result["error"]["operation_id"].get<std::uint64_t>(), operation_id.value());
        EXPECT_EQ(result["error"]["code"], "Internal");
    }

} // namespace lfs::mcp

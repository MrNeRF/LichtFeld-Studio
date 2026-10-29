/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "app/mcp_editor_tools.hpp"
#include "app/mcp_shader_tools.hpp"
#include "core/scene.hpp"
#include "mcp/mcp_tools.hpp"
#include "operation/undo_history.hpp"
#include "shaders/shader_api.hpp"
#include <gtest/gtest.h>
#include <limits>

namespace {
    using json = nlohmann::json;
    namespace shaders = lfs::vis::shaders;

    TEST(McpShaderTools, DiscoveryUsesNormalizedNamesAndTruthfulMetadata) {
        lfs::mcp::ToolRegistry registry;
        lfs::app::register_shader_tools(registry, nullptr,
                                        [] { return static_cast<shaders::ShaderService*>(nullptr); });
        const auto tools = registry.list_tools();
        ASSERT_EQ(tools.size(), 7u);
        for (const auto& tool : tools) {
            const auto wire = lfs::mcp::tool_to_json(tool);
            const auto name = wire["name"].get<std::string>();
            EXPECT_TRUE(name.starts_with("shader_"));
            EXPECT_EQ(wire["_meta"]["app.lichtfeld/category"], "shader");
            EXPECT_EQ(wire["_meta"]["app.lichtfeld/runtime"], "gui");
            EXPECT_EQ(wire["_meta"]["app.lichtfeld/thread_affinity"], "gui_thread");
            EXPECT_EQ(wire["annotations"]["readOnlyHint"], name == "shader_get" || name == "shader_list");
            EXPECT_EQ(wire["annotations"]["destructiveHint"], name == "shader_remove");
            EXPECT_EQ(wire["_meta"].value("app.lichtfeld/long_running", false), name == "shader_bind");
        }
    }

    TEST(McpEditorTools, DiscoveryAndInvalidArgumentsDoNotNeedViewerAccess) {
        lfs::mcp::ToolRegistry registry;
        lfs::app::register_editor_tools(registry, nullptr);
        const auto tools = registry.list_tools();
        ASSERT_EQ(tools.size(), 12u);
        for (const auto& tool : tools) {
            const auto wire = lfs::mcp::tool_to_json(tool);
            const auto name = wire["name"].get<std::string>();
            EXPECT_TRUE(name.starts_with("editor_"));
            EXPECT_EQ(wire["_meta"]["app.lichtfeld/category"], "editor");
            EXPECT_EQ(wire["_meta"]["app.lichtfeld/thread_affinity"], "gui_thread");
            EXPECT_EQ(wire["_meta"].value("app.lichtfeld/long_running", false),
                      name == "editor_run" || name == "editor_wait");
            EXPECT_EQ(wire["annotations"]["readOnlyHint"],
                      name == "editor_get_code" || name == "editor_get_output" || name == "editor_is_running" ||
                          name == "editor_list_documents" || name == "editor_wait");
            EXPECT_FALSE(wire["annotations"]["destructiveHint"].get<bool>());
            const auto& properties = wire["inputSchema"]["properties"];
            EXPECT_FALSE(properties.contains("include_ui"));
            EXPECT_EQ(properties.contains("document_id"),
                      name != "editor_list_documents" && name != "editor_new_document");
        }
        for (const auto& [name, args] : std::vector<std::pair<std::string, json>>{
                 {"editor_set_code", json::object()},
                 {"editor_activate_document", json::object()},
                 {"editor_set_mode", {{"mode", "other"}}},
                 {"editor_get_code", {{"include_ui", true}}},
                 {"editor_run", {{"document_id", "invalid"}}},
                 {"editor_run", {{"code", json::array()}}},
                 {"editor_run", {{"timeout_ms", true}}},
                 {"editor_run", {{"timeout_ms", 1.5}}},
                 {"editor_run", {{"timeout_ms", std::numeric_limits<std::uint64_t>::max()}}},
                 {"editor_wait", {{"asset_revision", -1}}},
                 {"editor_wait", {{"asset_revision", false}}},
                 {"editor_get_output", {{"max_chars", std::numeric_limits<std::int64_t>::min()}}}}) {
            const auto result = registry.call_tool(name, args);
            EXPECT_TRUE(result.contains("error")) << name << ": " << result.dump();
        }
    }

    TEST(McpShaderTools, InvalidWireArgumentsCannotMutateAssetsOrBindings) {
        lfs::core::Scene scene;
        shaders::ShaderService service(scene, {}, [] {});
        auto& history = lfs::vis::op::undoHistory();
        history.clear();
        const auto created =
            shaders::shaderOperation(service, "set_source", {{"source", "source"}, {"name", "Effect"}});
        ASSERT_TRUE(created);
        const auto id = (*created)["shader_id"];
        const auto before = shaders::shaderState(service.inspect());
        for (const auto& args : std::vector<json>{{{"shader_id", id}, {"source", "new"}, {"expected_revision", true}},
                                                  {{"shader_id", id}, {"source", "new"}, {"expected_revision", -1}},
                                                  {{"shader_id", id}, {"source", "new"}, {"expected_revision", 0}},
                                                  {{"shader_id", id}, {"source", "new"}, {"name", ""}},
                                                  {{"shader_id", id}, {"source", "new"}, {"extra", false}},
                                                  {{"shader_id", 4}, {"source", "new"}},
                                                  {{"source", "new"}, {"expected_revision", 1}}}) {
            const auto result = shaders::shaderOperation(service, "set_source", args);
            EXPECT_FALSE(result) << args.dump();
            EXPECT_EQ(shaders::shaderState(service.inspect()), before);
        }
        for (const auto& target : std::vector<json>{{{"kind", "node"}},
                                                    {{"kind", "global"}, {"node_uuid", id}},
                                                    {{"kind", "reveal"}},
                                                    {{"kind", "global"}, {"extra", 1}},
                                                    {{"kind", "node"}, {"node_uuid", "invalid"}}}) {
            EXPECT_FALSE(shaders::shaderOperation(service, "bind", {{"shader_id", id}, {"target", target}}));
            EXPECT_EQ(shaders::shaderState(service.inspect()), before);
        }
        EXPECT_FALSE(
            shaders::shaderOperation(service, "set_param", {{"shader_id", id}, {"name", "gain"}, {"value", true}}));
        EXPECT_FALSE(shaders::shaderOperation(service, "get", {{"shader_id", id}, {"include_source", 1}}));
        const auto read = shaders::shaderOperation(service, "get", {{"shader_id", id}, {"include_source", false}});
        ASSERT_TRUE(read);
        EXPECT_FALSE(read->contains("source"));
        EXPECT_EQ((*read)["status"]["state"], "unbound");
        EXPECT_FALSE(shaders::shaderJob(service.inspect())["active"].get<bool>());
        ASSERT_TRUE(history.undo().success);
        EXPECT_TRUE(service.assets().empty());
        history.clear();
    }
} // namespace

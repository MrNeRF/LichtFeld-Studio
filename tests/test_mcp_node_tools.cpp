/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "app/mcp_node_tools.hpp"
#include "core/event_bridge/event_bridge.hpp"
#include "mcp/mcp_tools.hpp"
#include "visualizer/gui/gui_manager.hpp"
#include "visualizer/gui/rmlui/elements/node_canvas_element.hpp"
#include "visualizer/gui/screen_host.hpp"
#include "visualizer/nodes/modifier_manager.hpp"
#include "visualizer/operation/undo_history.hpp"
#include "visualizer/scene/scene_manager.hpp"
#include "visualizer/visualizer_impl.hpp"

#include <RmlUi/Core.h>
#include <SDL3/SDL.h>
#include <gtest/gtest.h>

namespace lfs::vis {
    namespace {
        using json = nlohmann::json;
        class NodeToolRenderer final : public Rml::RenderInterface {
        public:
            Rml::CompiledGeometryHandle CompileGeometry(Rml::Span<const Rml::Vertex>, Rml::Span<const int>) override { return 1; }
            void RenderGeometry(Rml::CompiledGeometryHandle, Rml::Vector2f, Rml::TextureHandle) override {}
            void ReleaseGeometry(Rml::CompiledGeometryHandle) override {}
            Rml::TextureHandle LoadTexture(Rml::Vector2i&, const Rml::String&) override { return 0; }
            Rml::TextureHandle GenerateTexture(Rml::Span<const Rml::byte>, Rml::Vector2i) override { return 0; }
            void ReleaseTexture(Rml::TextureHandle) override {}
            void EnableScissorRegion(bool) override {}
            void SetScissorRegion(Rml::Rectanglei) override {}
        };
    } // namespace

    class McpNodeToolsTest : public ::testing::Test {
    protected:
        void SetUp() override {
            op::undoHistory().clear();
            lfs::event::EventBridge::instance().clear_all();
            ASSERT_TRUE(SDL_Init(SDL_INIT_VIDEO));
            window_ = SDL_CreateWindow("Node MCP tests", 1200, 800, SDL_WINDOW_HIDDEN);
            ASSERT_NE(window_, nullptr);
            ViewerOptions options;
            options.show_startup_overlay = false;
            options.safe_mode = true;
            viewer_ = std::make_unique<VisualizerImpl>(options);
            auto& scene = *viewer_->getSceneManager();
            scene.changeContentType(SceneManager::ContentType::SplatFiles);
            using core::Device;
            using core::Tensor;
            const auto id = scene.getScene().addSplat("Host", std::make_unique<core::SplatData>(0,
                                                                                                Tensor::zeros({1, 3}, Device::GPU), Tensor::zeros({1, 1, 3}, Device::GPU), Tensor::zeros({1, 0, 3}, Device::GPU),
                                                                                                Tensor::zeros({1, 3}, Device::GPU), Tensor::from_vector({1.f, 0.f, 0.f, 0.f}, {1, 4}, Device::CPU).to(Device::GPU),
                                                                                                Tensor::zeros({1, 1}, Device::GPU), 1.f));
            target_ = scene.getScene().getNodeUuid(id).to_string();
            scene.selectNode(id);
            ASSERT_TRUE(rml_.initWithRenderInterface(window_, 2.f, std::make_unique<NodeToolRenderer>(), nullptr));
            viewer_->getGuiManager()->screenHost().init({.screens = &viewer_->screens(), .rml = &rml_, .scene_manager = &scene});
            app::register_gui_node_tools(mcp::ToolRegistry::instance(), viewer_.get());
            app::register_gui_node_resources(mcp::ResourceRegistry::instance(), viewer_.get());
        }
        void TearDown() override {
            for (const auto& tool : mcp::ToolRegistry::instance().list_tools())
                if (tool.metadata.category == "nodes")
                    mcp::ToolRegistry::instance().unregister_tool(tool.name);
            mcp::ResourceRegistry::instance().unregister_resource_prefix("lichtfeld://nodes/");
            op::undoHistory().clear();
            viewer_->getGuiManager()->screenHost().shutdown();
            rml_.shutdown();
            viewer_.reset();
            lfs::event::EventBridge::instance().clear_all();
            SDL_DestroyWindow(window_);
            SDL_QuitSubSystem(SDL_INIT_VIDEO);
        }
        json call(const std::string& name, const json& args = json::object()) {
            auto result = mcp::ToolRegistry::instance().call_tool("nodes." + name, args);
            EXPECT_FALSE(result.contains("error")) << name << ": " << result.dump();
            return result;
        }
        json resource(const std::string& name) {
            const auto result = mcp::ResourceRegistry::instance().read_resource("lichtfeld://nodes/" + name);
            EXPECT_TRUE(result.has_value());
            if (!result || result->empty())
                return {};
            return json::parse(std::get<std::string>(result->front().content));
        }
        std::string graph() {
            return call("tree_create", {{"name", "Test Graph"}})["tree"]["uuid"];
        }
        std::unique_ptr<VisualizerImpl> viewer_;
        gui::RmlUIManager rml_;
        SDL_Window* window_ = nullptr;
        std::string target_;
    };

    TEST_F(McpNodeToolsTest, DescriptorsResourcesAndMetadata) {
        EXPECT_GT(resource("types").size(), 20u);
        EXPECT_TRUE(resource("trees").empty());
        EXPECT_TRUE(resource("stacks").empty());
        EXPECT_FALSE(resource("editor")["open"].get<bool>());
        EXPECT_EQ(resource("stacks/" + target_)["target"], target_);
        std::size_t count = 0;
        for (const auto& tool : mcp::ToolRegistry::instance().list_tools()) {
            if (tool.metadata.category != "nodes")
                continue;
            ++count;
            EXPECT_EQ(tool.metadata.runtime, "gui");
            EXPECT_EQ(tool.metadata.thread_affinity, "gui_thread");
            const json descriptor = mcp::tool_to_json(tool);
            EXPECT_TRUE(descriptor.contains("_meta"));
            EXPECT_TRUE(descriptor["annotations"].contains("readOnlyHint"));
            EXPECT_TRUE(descriptor["annotations"].contains("destructiveHint"));
            EXPECT_TRUE(descriptor["annotations"].contains("idempotentHint"));
        }
        EXPECT_EQ(count, 28u);
    }

    TEST_F(McpNodeToolsTest, TreeAndNodeCommandsUndoAndValidate) {
        const auto id = graph();
        call("tree_rename", {{"tree", id}, {"name", "Renamed"}});
        EXPECT_EQ(resource("trees/" + id)["name"], "Renamed");
        const auto exported = call("tree_export_json", {{"tree", id}})["tree"];
        const auto imported = call("tree_import_json", {{"json", exported}})["tree"]["uuid"].get<std::string>();
        EXPECT_NE(imported, id);
        call("tree_delete", {{"tree", imported}});
        const auto added = call("node_add", {{"tree", id}, {"type_id", "lfs.colour_correct"}, {"name", "Correct"}, {"location", {10, 20}}});
        EXPECT_EQ(added["node"], "Correct");
        call("node_set_input", {{"tree", id}, {"node", "Correct"}, {"input", "Exposure"}, {"value", 1.0}});
        call("node_set_property", {{"tree", id}, {"node", "Correct"}, {"property", "auto_range"}, {"value", false}});
        call("node_mute", {{"tree", id}, {"node", "Correct"}, {"muted", true}});
        call("node_move", {{"tree", id}, {"node", "Correct"}, {"location", {120, 90}}});
        EXPECT_EQ(viewer_->getSceneManager()->modifierManager().tree(id)->find_node("Correct")->location[0], 120);
        call("node_remove", {{"tree", id}, {"node", "Correct"}});
        ASSERT_TRUE(op::undoHistory().undo().success);
        EXPECT_NE(viewer_->getSceneManager()->modifierManager().tree(id)->find_node("Correct"), nullptr);
        const auto unknown = mcp::ToolRegistry::instance().call_tool("nodes.node_add", {{"tree", id}, {"type_id", "lfs.colour_correc"}});
        EXPECT_TRUE(unknown.contains("error"));
        EXPECT_TRUE(unknown.contains("available_types"));
    }

    TEST_F(McpNodeToolsTest, UniqueNamesResolveToExactIdentitiesAndRenameIsUndoable) {
        const auto first = call("tree_create", {{"name", "Autumn Lawn"}})["tree"];
        const auto second = call("tree_create", {{"name", "Autumn Lawn"}})["tree"];
        EXPECT_EQ(second["name"], "Autumn Lawn 2");
        EXPECT_EQ(call("tree_export_json", {{"tree", "Autumn Lawn"}})["tree"]["uuid"], first["uuid"]);
        auto renamed = call("tree_rename", {{"tree", "Autumn Lawn 2"}, {"name", "Autumn Lawn"}});
        EXPECT_EQ(renamed["tree"]["name"], "Autumn Lawn 2");
        const auto imported = call("tree_import_json", {{"json", first}})["tree"];
        EXPECT_EQ(imported["name"], "Autumn Lawn 3");
        call("node_add", {{"tree", "Autumn Lawn"}, {"type_id", "lfs.value"}, {"name", "Value"}});
        const auto stack = call("modifier_add", {{"target", "Host"}, {"tree", "Autumn Lawn"}});
        EXPECT_EQ(stack["target"], target_);
        EXPECT_EQ(stack["name"], "Host");
        EXPECT_EQ(stack["modifiers"][0]["tree_name"], "Autumn Lawn");
        call("tree_rename", {{"tree", "Autumn Lawn 2"}, {"name", "Summer"}});
        ASSERT_TRUE(op::undoHistory().undo().success);
        EXPECT_EQ(resource("trees/" + second["uuid"].get<std::string>())["name"], "Autumn Lawn 2");
    }

    TEST_F(McpNodeToolsTest, LinksModifiersEvaluateCaptureApplyAndUndo) {
        const auto id = graph();
        call("node_add", {{"tree", id}, {"type_id", "lfs.colour_correct"}, {"name", "Correct"}});
        const json direct = {{"tree", id}, {"from_node", "Group Input"}, {"from_socket", "Geometry"}, {"to_node", "Group Output"}, {"to_socket", "Geometry"}};
        call("unlink", direct);
        auto link = direct;
        link["to_node"] = "Correct";
        call("link", link);
        link = direct;
        link["from_node"] = "Correct";
        call("link", link);
        link["to_node"] = "Correct";
        link["to_socket"] = "Exposure";
        const auto invalid = mcp::ToolRegistry::instance().call_tool("nodes.link", link);
        EXPECT_TRUE(invalid.contains("error"));
        EXPECT_EQ(invalid["from_type"], "lfs.geometry");
        EXPECT_EQ(invalid["to_type"], "lfs.float");
        const auto first = call("modifier_add", {{"target", target_}, {"tree", id}})["modifier"];
        const auto second = call("modifier_add", {{"target", target_}, {"tree", id}})["modifier"];
        call("modifier_move", {{"target", target_}, {"modifier", second}, {"index", 0}});
        call("modifier_set", {{"target", target_}, {"modifier", second}, {"enabled", false}, {"show_viewport", false}, {"name", "Disabled"}, {"input_overrides", json::object()}});
        call("modifier_remove", {{"target", target_}, {"modifier", second}});
        call("node_add", {{"tree", id}, {"type_id", "lfs.stored_selection"}, {"name", "Stored"}});
        call("modifier_capture_selection", {{"target", target_}, {"modifier", first}, {"node", "Stored"}});
        const auto evaluated = call("evaluate", {{"target", target_}, {"wait", true}});
        ASSERT_TRUE(evaluated["evaluation"]["ok"].get<bool>()) << evaluated.dump();
        EXPECT_FALSE(evaluated["progress"]["busy"].get<bool>());
        EXPECT_EQ(resource("stacks").size(), 1u);
        EXPECT_TRUE(app::node_evaluation_job(*viewer_).contains("results"));
        call("modifier_apply", {{"target", target_}, {"modifier", first}});
        EXPECT_TRUE(resource("stacks/" + target_)["modifiers"].empty());
        ASSERT_TRUE(op::undoHistory().undo().success);
        EXPECT_EQ(resource("stacks/" + target_)["modifiers"].size(), 1u);
    }

    TEST_F(McpNodeToolsTest, EditorCommandsAndReopenRetainGraph) {
        const auto id = graph();
        call("node_add", {{"tree", id}, {"type_id", "lfs.colour_correct"}, {"name", "Correct"}});
        const auto modifier = call("modifier_add", {{"target", target_}, {"tree", id}})["modifier"];
        EXPECT_TRUE(call("editor_open")["open"].get<bool>());
        auto* canvas = viewer_->getGuiManager()->screenHost().nodeCanvas();
        ASSERT_NE(canvas, nullptr);
        canvas->GetContext()->SetDimensions({2200, 1400});
        canvas->GetContext()->Update();
        EXPECT_EQ(call("editor_show", {{"target", target_}, {"modifier", modifier}})["tree"], id);
        call("editor_select", {{"nodes", {"Correct"}}});
        EXPECT_EQ(resource("editor")["selected_nodes"], json::array({"Correct"}));
        op::undoHistory().clear();
        call("editor_arrange");
        EXPECT_EQ(op::undoHistory().undoCount(), 0u); // A single selected card stays anchored.
        call("editor_arrange", {{"nodes", {"Group Input", "Correct"}}});
        EXPECT_EQ(op::undoHistory().undoCount(), 1u);
        EXPECT_TRUE(mcp::ToolRegistry::instance().call_tool("nodes.editor_arrange", {{"nodes", {"Missing"}}}).contains("error"));
        call("editor_arrange", {{"selection_only", false}});
        call("editor_frame");
        call("editor_view", {{"pan", {123, 234}}, {"zoom", 0.75}});
        EXPECT_EQ(resource("editor")["pan"], json::array({123, 234}));
        call("editor_preview_selection", {{"enabled", false}});
        EXPECT_FALSE(resource("editor")["preview_selection"].get<bool>());
        EXPECT_FALSE(call("editor_close")["open"].get<bool>());
        EXPECT_TRUE(call("editor_open")["open"].get<bool>());
        EXPECT_EQ(resource("editor")["nodes"].size(), 3u);
    }
} // namespace lfs::vis

/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "app/mcp_node_tools.hpp"
#include "core/event_bridge/event_bridge.hpp"
#include "mcp/mcp_tools.hpp"
#include "visualizer/gui/gui_manager.hpp"
#include "visualizer/gui/rmlui/elements/node_canvas_element.hpp"
#include "visualizer/gui/screen_host.hpp"
#include "visualizer/nodes/modifier_manager.hpp"
#include "visualizer/nodes/node_animation.hpp"
#include "visualizer/operation/undo_history.hpp"
#include "visualizer/scene/scene_manager.hpp"
#include "visualizer/selection/selection_service.hpp"
#include "visualizer/sequencer/sequencer_controller.hpp"
#include "visualizer/visualizer_impl.hpp"

#include <RmlUi/Core.h>
#include <SDL3/SDL.h>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <sstream>

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
        for (const auto& type : resource("types")) {
            SCOPED_TRACE(type["id"].get<std::string>());
            EXPECT_FALSE(type["description"].get<std::string>().empty());
            EXPECT_FALSE(type["help"].get<std::string>().empty());
            for (const auto* collection : {"inputs", "outputs", "properties"})
                for (const auto& declaration : type[collection])
                    EXPECT_FALSE(declaration["description"].get<std::string>().empty());
        }
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
        EXPECT_EQ(count, 60u);
    }

    TEST_F(McpNodeToolsTest, TemplatesAndPreviewRoundTrip) {
        const auto listed = call("template_list");
        ASSERT_GE(listed["templates"].size(), 15u);
        const auto applied = call("template_apply", {{"target", target_}, {"template", "colour_grade"}, {"name", "Template MCP"}});
        ASSERT_TRUE(applied["success"]);
        const auto preview = call("preview_set", {{"target", target_}, {"node", "Colour Correct"}, {"socket", "Geometry"}});
        EXPECT_EQ(preview["socket"], "Geometry");
        EXPECT_EQ(resource("editor")["preview"]["node"], "Colour Correct");
        ASSERT_TRUE(call("preview_clear")["success"]);
        EXPECT_TRUE(resource("editor")["preview"].is_null());
        const auto saved = call("template_save", {{"tree", applied["tree"]}, {"name", "MCP template"}, {"description", "Saved by the MCP test"}, {"category", "Test"}});
        ASSERT_TRUE(saved["success"]);
        ASSERT_TRUE(call("template_rename", {{"template", saved["template"]["id"]}, {"name", "Renamed MCP"}})["success"]);
        ASSERT_TRUE(call("template_delete", {{"template", saved["template"]["id"]}})["success"]);
    }

    TEST_F(McpNodeToolsTest, KeyframesDefaultsRenameJsonCopyAndRemove) {
        auto& manager = viewer_->getSceneManager()->modifierManager();
        auto& sequence = viewer_->getGuiManager()->sequencer();
        manager.setSequencer(&sequence);
        const auto uuid = graph();
        ASSERT_TRUE(call("node_add", {{"tree", uuid}, {"type_id", "lfs.value"}, {"name", "Radius"}})["success"]);
        ASSERT_TRUE(call("node_set_input", {{"tree", uuid}, {"node", "Radius"}, {"input", "Value"}, {"value", 2}})["success"]);
        sequence.seek(0.5f);
        ASSERT_TRUE(call("keyframe_set", {{"tree", uuid}, {"node", "Radius"}, {"input", "Value"}})["success"]);
        ASSERT_TRUE(call("keyframe_set", {{"tree", uuid}, {"node", "Radius"}, {"input", "Value"}, {"time", 2}, {"value", 8}, {"easing", 3}})["success"]);
        ASSERT_TRUE(call("node_rename", {{"tree", uuid}, {"node", "Radius"}, {"name", "Reveal radius"}})["success"]);
        const auto* clip = sequence.timeline().animationClip();
        ASSERT_NE(clip, nullptr);
        EXPECT_EQ(clip->getTrackByPath(nodeInputTrackPath(uuid, "Radius", "Value")), nullptr);
        const auto path = nodeInputTrackPath(uuid, "Reveal radius", "Value");
        ASSERT_NE(clip->getTrackByPath(path), nullptr);
        EXPECT_FLOAT_EQ(std::get<float>(*clip->getTrackByPath(path)->evaluate(0.5f)), 2);
        const auto exported = call("tree_export_json", {{"tree", uuid}})["tree"];
        const auto imported = call("tree_import_json", {{"json", exported}})["tree"]["uuid"].get<std::string>();
        EXPECT_NE(imported, uuid);
        const auto* copied = clip->getTrackByPath(nodeInputTrackPath(imported, "Reveal radius", "Value"));
        ASSERT_NE(copied, nullptr);
        EXPECT_EQ(copied->keyframeCount(), 2);
        EXPECT_FLOAT_EQ(std::get<float>(*copied->evaluate(1.25f)), 5);
        ASSERT_TRUE(call("keyframe_remove", {{"tree", uuid}, {"node", "Reveal radius"}, {"input", "Value"}})["success"]);
        EXPECT_EQ(clip->getTrackByPath(path)->keyframeCount(), 1);
        op::undoHistory().undo();
        EXPECT_EQ(sequence.timeline().animationClip()->getTrackByPath(path)->keyframeCount(), 2);
    }

    TEST_F(McpNodeToolsTest, HostOnlyReferenceMatchesDescriptorText) {
        const auto types = resource("types");
        const auto found = std::ranges::find_if(types, [](const json& type) {
            return type["id"] == "lfs.object_info";
        });
        ASSERT_NE(found, types.end());
        std::ifstream input(std::filesystem::path(PROJECT_ROOT_PATH) /
                            "docs/docs/development/node-graph/nodes/lfs.object_info.md");
        ASSERT_TRUE(input.is_open());
        std::ostringstream contents;
        contents << input.rdbuf();
        const auto page = contents.str();
        for (const auto* key : {"label", "description", "category"})
            EXPECT_NE(page.find(found->at(key).get<std::string>()), std::string::npos) << key;
        std::istringstream help(found->at("help").get<std::string>());
        std::string line;
        while (std::getline(help, line))
            EXPECT_NE(page.find(line), std::string::npos);
        for (const auto* collection : {"inputs", "outputs", "properties"})
            for (const auto& declaration : found->at(collection))
                for (const auto* key : {"identifier", "label", "description"})
                    EXPECT_NE(page.find(declaration.at(key).get<std::string>()), std::string::npos) << key;
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

    TEST_F(McpNodeToolsTest, GraphEditingToolsArePortableAndUndoable) {
        const auto id = graph();
        call("node_add", {{"tree", id}, {"type_id", "lfs.colour_correct"}, {"name", "Correct"}});
        call("unlink", {{"tree", id}, {"from_node", "Group Input"}, {"from_socket", "Geometry"}, {"to_node", "Group Output"}, {"to_socket", "Geometry"}});
        call("link", {{"tree", id}, {"from_node", "Group Input"}, {"from_socket", "Geometry"}, {"to_node", "Correct"}, {"to_socket", "Geometry"}});
        call("link", {{"tree", id}, {"from_node", "Correct"}, {"from_socket", "Geometry"}, {"to_node", "Group Output"}, {"to_socket", "Geometry"}});

        const auto copied = call("copy", {{"tree", id}, {"nodes", {"Correct"}}})["clipboard"];
        op::undoHistory().clear();
        const auto pasted = call("paste", {{"tree", id}, {"clipboard", copied}, {"location", {400, 200}}});
        ASSERT_EQ(pasted["nodes"].size(), 1u);
        EXPECT_EQ(op::undoHistory().undoCount(), 1u);
        ASSERT_TRUE(op::undoHistory().undo().success);
        EXPECT_EQ(resource("trees/" + id)["nodes"].size(), 3u);
        ASSERT_TRUE(op::undoHistory().redo().success);

        const auto grouped = call("group_make", {{"tree", id}, {"nodes", {"Correct"}}, {"name", "Grade"}});
        const auto group_node = grouped["group_node"];
        const auto group_graph = grouped["graph"];
        EXPECT_FALSE(resource("trees/" + group_graph.get<std::string>())["interface"].empty());
        call("interface_add", {{"tree", group_graph}, {"side", "input"}, {"type", "float"}, {"label", "Strength"}});
        auto group_state = resource("trees/" + group_graph.get<std::string>());
        const auto identifier = group_state["interface"]["inputs"].back()["identifier"];
        call("interface_update", {{"tree", group_graph}, {"side", "input"}, {"identifier", identifier}, {"label", "Amount"}});
        call("interface_move", {{"tree", group_graph}, {"side", "input"}, {"identifier", identifier}, {"index", 0}});
        call("node_set_input", {{"tree", id}, {"node", group_node}, {"input", identifier}, {"value", 0.25}});
        const auto owner_state = resource("trees/" + id);
        const auto group_state_node = std::ranges::find_if(owner_state["nodes"], [&](const auto& node) {
            return node["name"] == group_node;
        });
        ASSERT_NE(group_state_node, owner_state["nodes"].end());
        EXPECT_FLOAT_EQ((*group_state_node)["input_values"][identifier]["value"].get<float>(), 0.25f);

        const auto modifier = call("modifier_add", {{"target", target_}, {"tree", id}})["modifier"];
        call("editor_open");
        call("editor_show", {{"target", target_}, {"modifier", modifier}});
        const auto entered = call("editor_enter_group", {{"node", group_node}});
        EXPECT_EQ(entered["tree"], group_graph);
        EXPECT_EQ(entered["breadcrumb"].size(), 3u);
        EXPECT_EQ(call("editor_exit_group")["tree"], id);

        call("frame_wrap", {{"tree", id}, {"nodes", pasted["nodes"]}, {"label", "Copies"}});
        const auto tree_state = resource("trees/" + id);
        const auto link = *std::ranges::find_if(tree_state["links"], [](const auto& item) {
            return item["from_node"] == "Group Input";
        });
        call("reroute_insert", {{"tree", id}, {"link", link}, {"location", {100, 50}}});
        EXPECT_TRUE(std::ranges::any_of(resource("trees/" + id)["nodes"], [](const auto& node) {
            return node["type_id"] == "lfs.reroute";
        }));
        call("group_ungroup", {{"tree", id}, {"node", group_node}});
        call("interface_remove", {{"tree", group_graph}, {"side", "input"}, {"identifier", identifier}});
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

    TEST_F(McpNodeToolsTest, ViewportGizmoPaintAndColourPickToolsAreUndoable) {
        const auto id = graph();
        call("node_add", {{"tree", id}, {"type_id", "lfs.box_selection"}, {"name", "Box"}});
        call("node_add", {{"tree", id}, {"type_id", "lfs.paint_selection"}, {"name", "Paint"}});
        call("node_add", {{"tree", id}, {"type_id", "lfs.colour_key"}, {"name", "Key"}});
        const auto modifier = call("modifier_add", {{"target", target_}, {"tree", id}})["modifier"];
        call("editor_open");
        call("editor_show", {{"target", target_}, {"modifier", modifier}});

        call("editor_select", {{"nodes", {"Box"}}});
        const auto gizmo = call("gizmo_get");
        EXPECT_EQ(gizmo["kind"], "box");
        EXPECT_TRUE(gizmo["editable"].get<bool>());
        EXPECT_EQ(gizmo["local"]["translation"], json::array({0.0f, 0.0f, 0.0f}));

        call("editor_select", {{"nodes", {"Paint"}}});
        const auto painting = call("paint_mode", {{"node", "Paint"}, {"enabled", true}, {"radius", 40.0f}});
        EXPECT_TRUE(painting["paint_mode"].get<bool>());
        EXPECT_FLOAT_EQ(painting["radius"].get<float>(), 40.0f);
        EXPECT_FALSE(call("paint_mode", {{"node", "Paint"}, {"enabled", false}})["paint_mode"].get<bool>());
        op::undoHistory().clear();
        call("paint_stroke", {{"node", "Paint"},
                              {"space", "world"},
                              {"samples", {{0.0f, 0.0f, 0.0f, 0.25f}, {0.1f, 0.0f, 0.0f, 0.25f}}}});
        // Undo replaces the graph object, so look it up again after every command.
        const auto node = [&](const char* name) -> lfs::nodes::Node* {
            auto* const tree = viewer_->getSceneManager()->modifierManager().tree(id);
            return tree ? tree->find_node(name) : nullptr;
        };
        ASSERT_NE(node("Paint"), nullptr);
        EXPECT_EQ(node("Paint")->properties["data"].size(), 1u);
        EXPECT_EQ(op::undoHistory().undoCount(), 1u);
        ASSERT_TRUE(op::undoHistory().undo().success);
        ASSERT_NE(node("Paint"), nullptr);
        EXPECT_TRUE(node("Paint")->properties["data"].empty());

        auto& scene_manager = *viewer_->getSceneManager();
        scene_manager.initSelectionService();
        auto* const selection = scene_manager.getSelectionService();
        ASSERT_NE(selection, nullptr);
        selection->setTestingViewport({.x = 0.0f,
                                       .y = 0.0f,
                                       .width = 100.0f,
                                       .height = 100.0f,
                                       .render_width = 100,
                                       .render_height = 100});
        selection->setTestingHoveredGaussianId(0);
        call("editor_select", {{"nodes", {"Key"}}});
        op::undoHistory().clear();
        const auto picked = call("pick_colour", {{"node", "Key"},
                                                 {"input", "Colour"},
                                                 {"screen_x", 50.0f},
                                                 {"screen_y", 50.0f}});
        EXPECT_TRUE(picked["stored_payload"].get<bool>());
        ASSERT_NE(node("Key"), nullptr);
        const auto* const colour = node("Key")->input_values["Colour"].get_if<glm::vec4>();
        ASSERT_NE(colour, nullptr);
        EXPECT_EQ(*colour, glm::vec4(0.5f, 0.5f, 0.5f, 1.0f));
        EXPECT_EQ(op::undoHistory().undoCount(), 1u);
    }
} // namespace lfs::vis

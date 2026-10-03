/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/event_bridge/event_bridge.hpp"
#include "core/nodes/tree.hpp"
#include "gui/rmlui/elements/node_canvas_dom.hpp"
#include "gui/rmlui/elements/node_canvas_element.hpp"
#include "gui/rmlui/elements/node_canvas_widgets.hpp"
#include "gui/rmlui/rmlui_manager.hpp"
#include "scene/scene_manager.hpp"
#include "visualizer/nodes/modifier_manager.hpp"
#include "visualizer/operation/undo_history.hpp"

#include <RmlUi/Core.h>
#include <RmlUi/Core/ElementInstancer.h>
#include <RmlUi/Core/Elements/ElementFormControlInput.h>
#include <SDL3/SDL_scancode.h>
#include <filesystem>
#include <gtest/gtest.h>
#include <thread>

namespace {
    class WidgetRenderer final : public Rml::RenderInterface {
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

    class NodeCanvasWidgets : public ::testing::Test {
    protected:
        void SetUp() override {
            lfs::vis::op::undoHistory().clear();
            ASSERT_TRUE(Rml::Initialise());
            const auto resources = std::filesystem::path(PROJECT_ROOT_PATH) / "src/visualizer/gui";
            ASSERT_TRUE(Rml::LoadFontFace((resources / "assets/fonts/Inter-Regular.ttf").string()));
            Rml::Factory::RegisterElementInstancer("node-canvas", &instancer_);
            context_ = Rml::CreateContext("node_widgets", {1000, 700}, &renderer_);
            ASSERT_NE(context_, nullptr);
            context_->SetDensityIndependentPixelRatio(2.0f);
            const auto styles = resources / "rmlui/resources";
            document_ = context_->LoadDocumentFromMemory(
                "<rml><head><link type='text/rcss' href='components.rcss'/>"
                "<link type='text/rcss' href='node_editor.rcss'/></head>"
                "<body><node-canvas id='node-editor-canvas'/></body></rml>",
                (styles / "node_editor.rml").string());
            ASSERT_NE(document_, nullptr);
            auto* canvas = dynamic_cast<lfs::vis::gui::NodeCanvasElement*>(document_->GetElementById("node-editor-canvas"));
            ASSERT_NE(canvas, nullptr);
            scene_.changeContentType(lfs::vis::SceneManager::ContentType::SplatFiles);
            canvas->setContext(&scene_, nullptr);
            document_->Show();
            context_->Update();
            auto container = document_->CreateElement("div");
            container->SetProperty("position", "absolute");
            container->SetProperty("left", "20dp");
            container->SetProperty("top", "20dp");
            container->SetProperty("width", "200dp");
            const lfs::nodes::Node node{.name = "Test"};
            const lfs::nodes::SocketDecl socket{
                .identifier = "Value",
                .label = "Value",
                .type = std::string(lfs::nodes::FLOAT_SOCKET),
                .default_value = 0.25f,
                .min = 0.0,
                .max = 1.0,
                .step = 0.01};
            container->SetInnerRML(lfs::vis::gui::node_widgets::input(node, socket, true));
            field_ = container->QuerySelector(".node-scrub");
            input_ = dynamic_cast<Rml::ElementFormControlInput*>(container->QuerySelector("input"));
            canvas->AppendChild(std::move(container));
            context_->Update();
            ASSERT_NE(field_, nullptr);
            ASSERT_NE(input_, nullptr);
        }

        void TearDown() override {
            lfs::vis::op::undoHistory().clear();
            // SceneManager's process-lifetime handlers must not outlive this fixture.
            lfs::event::EventBridge::instance().clear_all();
            Rml::RemoveContext("node_widgets");
            Rml::Shutdown();
        }

        void attachGraph() {
            using lfs::core::Device;
            using lfs::core::Tensor;
            context_->SetDimensions({2200, 1600});
            const auto id = scene_.getScene().addPointCloud("Host", std::make_shared<lfs::core::PointCloud>(
                                                                        Tensor::zeros({1, 3}, Device::CPU), Tensor::ones({1, 3}, Device::CPU)));
            host_ = scene_.getScene().getNodeUuid(id);
            scene_.selectNode(id);
            auto& manager = scene_.modifierManager();
            auto& tree = manager.newTree("Interaction");
            tree_ = tree.uuid;
            const auto input = tree.input_node().name;
            const auto output = tree.output_node().name;
            tree.find_node(input)->location = {20, 20};
            tree.find_node(output)->location = {530, 20};
            tree.add_node("lfs.colour_correct", "Correct").location = {270, 20};
            tree.add_node("lfs.value", "Value").location = {20, 280};
            tree.remove_link({input, "Geometry", output, "Geometry"});
            ASSERT_TRUE(tree.add_link({input, "Geometry", "Correct", "Geometry"}));
            ASSERT_TRUE(tree.add_link({"Correct", "Geometry", output, "Geometry"}));
            manager.addModifier(host_, tree.uuid);
            ASSERT_TRUE(manager.evaluate(host_).ok);
            context_->Update();
            auto* canvas = dynamic_cast<lfs::vis::gui::NodeCanvasElement*>(document_->GetElementById("node-editor-canvas"));
            canvas->setView({}, 1.0f);
            context_->Update();
            (void)manager.performance(true);
        }

        void pointer(const char* event, const float x, const float y, const int button = 0) {
            auto* canvas = document_->GetElementById("node-editor-canvas");
            Rml::Dictionary parameters;
            parameters["mouse_x"] = x;
            parameters["mouse_y"] = y;
            parameters["button"] = button;
            canvas->DispatchEvent(event, parameters);
            context_->Update();
            scene_.modifierManager().tick();
        }

        void expectNoEvaluation() {
            const auto metrics = scene_.modifierManager().performance();
            EXPECT_EQ(metrics["requests"], 0);
            EXPECT_EQ(metrics["evaluations"], 0);
        }

        WidgetRenderer renderer_;
        Rml::ElementInstancerGeneric<lfs::vis::gui::NodeCanvasElement> instancer_;
        lfs::vis::SceneManager scene_;
        Rml::Context* context_ = nullptr;
        Rml::ElementDocument* document_ = nullptr;
        Rml::Element* field_ = nullptr;
        Rml::ElementFormControlInput* input_ = nullptr;
        lfs::core::Uuid host_;
        std::string tree_;
    };

    TEST_F(NodeCanvasWidgets, DoubleClickOpensEditableNumberAndEnterClampsValue) {
        field_->DispatchEvent("dblclick", {});
        context_->Update();
        EXPECT_TRUE(field_->IsClassSet("is-editing"));
        EXPECT_EQ(context_->GetFocusElement(), input_);
        input_->SetValue("2.5");
        context_->ProcessKeyDown(Rml::Input::KI_RETURN, 0);
        EXPECT_FALSE(field_->IsClassSet("is-editing"));
        EXPECT_DOUBLE_EQ(field_->GetAttribute<double>("data-value", 0.0), 1.0);
    }

    TEST_F(NodeCanvasWidgets, EscapeRestoresTypedValue) {
        field_->DispatchEvent("dblclick", {});
        context_->Update();
        input_->SetValue("0.75");
        context_->ProcessKeyDown(Rml::Input::KI_ESCAPE, 0);
        EXPECT_FALSE(field_->IsClassSet("is-editing"));
        EXPECT_DOUBLE_EQ(field_->GetAttribute<double>("data-value", 0.0), 0.25);
    }

    TEST_F(NodeCanvasWidgets, DetachedFieldCannotAcquireKeyboardFocus) {
        lfs::vis::gui::RmlUIManager manager;
        auto detached = document_->CreateElement("input");
        detached->SetAttribute("type", "text");
        detached->AddEventListener("focus", &manager);
        EXPECT_EQ(detached->GetContext(), nullptr);
        EXPECT_NO_THROW(detached->DispatchEvent("focus", {}));
        detached->RemoveEventListener("focus", &manager);
    }

    TEST_F(NodeCanvasWidgets, PanZoomSelectBoxSelectAndMoveNeverEvaluateOrReplaceCards) {
        attachGraph();
        auto* canvas = document_->GetElementById("node-editor-canvas");
        auto* card = canvas->QuerySelector(".node-box[data-node=Correct]");
        ASSERT_NE(card, nullptr);
        pointer("mousedown", 600, 65);
        pointer("mouseup", 600, 65);
        EXPECT_TRUE(card->IsClassSet("selected"));
        expectNoEvaluation();
        pointer("mousedown", 600, 65);
        for (int step = 1; step <= 20; ++step)
            pointer("mousemove", 600.0f + step, 65.0f + step);
        pointer("mouseup", 620, 85);
        expectNoEvaluation();
        EXPECT_EQ(canvas->QuerySelector(".node-box[data-node=Correct]"), card);
        const auto* moved = scene_.modifierManager().tree(tree_)->find_node("Correct");
        ASSERT_NE(moved, nullptr);
        EXPECT_EQ(moved->location[0], 280.0f);
        pointer("mousedown", 1500, 1300);
        pointer("mousemove", 20, 20);
        pointer("mouseup", 20, 20);
        expectNoEvaluation();
        pointer("mousedown", 1000, 1000, 2);
        pointer("mousemove", 1060, 1030, 2);
        pointer("mouseup", 1060, 1030, 2);
        expectNoEvaluation();
        Rml::Dictionary wheel;
        wheel["mouse_x"] = 800.0f;
        wheel["mouse_y"] = 700.0f;
        wheel["wheel_delta_y"] = 1.0f;
        wheel["ctrl_key"] = 1;
        canvas->DispatchEvent("mousescroll", wheel);
        context_->Update();
        scene_.modifierManager().tick();
        expectNoEvaluation();
        EXPECT_EQ(canvas->QuerySelector(".node-box[data-node=Correct]"), card);
    }

    TEST_F(NodeCanvasWidgets, AddPopupFiltersAndEnterAddsWithoutReplacingExistingCards) {
        attachGraph();
        auto* canvas = dynamic_cast<lfs::vis::gui::NodeCanvasElement*>(document_->GetElementById("node-editor-canvas"));
        auto* card = canvas->QuerySelector(".node-box[data-node=Correct]");
        canvas->headerAction("add", 600, 300);
        context_->Update();
        auto* menu = canvas->GetElementById("node-add-menu");
        ASSERT_NE(menu, nullptr);
        EXPECT_LT(menu->GetBox().GetSize().y, context_->GetDimensions().y * 0.7f);
        auto* search = dynamic_cast<Rml::ElementFormControlInput*>(menu->GetElementById("node-add-search"));
        ASSERT_NE(search, nullptr);
        search->SetValue("HSV");
        search->DispatchEvent("change", {});
        context_->Update();
        Rml::ElementList items;
        menu->QuerySelectorAll(items, ".node-add-item");
        EXPECT_EQ(std::ranges::count_if(items, [](const auto* item) { return !item->IsClassSet("filtered"); }), 1);
        expectNoEvaluation();
        ASSERT_TRUE(canvas->handleKey(SDL_SCANCODE_RETURN, false, false, false));
        context_->Update();
        EXPECT_EQ(canvas->GetElementById("node-add-menu"), nullptr);
        const auto* tree = scene_.modifierManager().tree(tree_);
        EXPECT_EQ(std::ranges::count(tree->nodes, "lfs.hsv_range", &lfs::nodes::Node::type_id), 1);
        EXPECT_EQ(canvas->QuerySelector(".node-box[data-node=Correct]"), card);
    }

    TEST_F(NodeCanvasWidgets, InspectorSwitchUsesChangeEventAndRequestsOnce) {
        attachGraph();
        pointer("mousedown", 600, 65);
        pointer("mouseup", 600, 65);
        auto* canvas = document_->GetElementById("node-editor-canvas");
        auto* control = canvas->QuerySelector("input[data-action=node-on]");
        ASSERT_NE(control, nullptr);
        control->RemoveAttribute("checked");
        control->DispatchEvent("change", {});
        context_->Update();
        auto& manager = scene_.modifierManager();
        EXPECT_TRUE(manager.tree(tree_)->find_node("Correct")->muted);
        manager.tick();
        ASSERT_TRUE(manager.evaluate(host_).ok);
        const auto metrics = manager.performance();
        EXPECT_EQ(metrics["requests"], 1);
        EXPECT_EQ(metrics["evaluations"], 1);
    }

    TEST_F(NodeCanvasWidgets, WireDragDoesNotEvaluateUntilDropAndDropRequestsOnce) {
        attachGraph();
        auto* canvas = document_->GetElementById("node-editor-canvas");
        auto* field = canvas->QuerySelector(".node-box[data-node=Correct] input[data-input=Exposure]");
        ASSERT_NE(field, nullptr);
        canvas->QuerySelector(".node-box[data-node=Correct] .node-settings")->DispatchEvent("click", {});
        context_->Update();
        pointer("mousedown", 488, 646);
        for (int step = 1; step <= 20; ++step)
            pointer("mousemove", 488.0f + step * 2.6f, 646.0f - step * 15.0f);
        expectNoEvaluation();
        pointer("mouseup", 540, 346);
        SCOPED_TRACE(dynamic_cast<lfs::vis::gui::NodeCanvasElement*>(canvas)->viewState().dump());
        EXPECT_EQ(canvas->QuerySelector(".node-box[data-node=Correct] input[data-input=Exposure]"), field);
        EXPECT_TRUE(field->GetParentNode()->GetParentNode()->GetParentNode()->IsClassSet("linked"));
        auto& manager = scene_.modifierManager();
        const auto* tree = manager.tree(tree_);
        ASSERT_TRUE(std::ranges::any_of(tree->links, [](const auto& link) {
            return link.from_node == "Value" && link.to_node == "Correct" && link.to_socket == "Exposure";
        }));
        ASSERT_TRUE(manager.evaluate(host_).ok);
        const auto metrics = manager.performance();
        EXPECT_EQ(metrics["requests"], 1);
        EXPECT_EQ(metrics["evaluations"], 1);
    }

    TEST_F(NodeCanvasWidgets, CompactSettingsPersistWithoutEvaluationAndLinkedInputsStayVisible) {
        attachGraph();
        auto* canvas = dynamic_cast<lfs::vis::gui::NodeCanvasElement*>(document_->GetElementById("node-editor-canvas"));
        auto* card = canvas->QuerySelector(".node-box[data-node=Correct]");
        auto* exposure = card->QuerySelector("input[data-input=Exposure]");
        auto* row = exposure->GetParentNode()->GetParentNode()->GetParentNode();
        EXPECT_EQ(row->GetComputedValues().display(), Rml::Style::Display::None);
        const float compact_height = card->GetOffsetHeight();
        card->QuerySelector(".node-settings")->DispatchEvent("click", {});
        context_->Update();
        EXPECT_GT(card->GetOffsetHeight(), compact_height + 300.0f);
        auto& manager = scene_.modifierManager();
        auto* tree = manager.tree(tree_);
        const auto restored = lfs::nodes::NodeTree::from_json(tree->to_json(), manager.registry());
        EXPECT_TRUE(restored.find_node("Correct")->ui.at("settings_expanded").get<bool>());
        card->QuerySelector(".node-settings")->DispatchEvent("click", {});
        context_->Update();
        manager.tick();
        expectNoEvaluation();
        const auto before = tree->to_json();
        ASSERT_TRUE(tree->add_link({"Value", "Value", "Correct", "Exposure"}));
        manager.recordTreeEdit(tree_, before);
        context_->Update();
        EXPECT_NE(row->GetComputedValues().display(), Rml::Style::Display::None);
        EXPECT_EQ(card->QuerySelector("input[data-input=Exposure]"), exposure);
    }

    TEST_F(NodeCanvasWidgets, CardsSurviveSubmitBusyInstallAndMoveBeforeRelease) {
        attachGraph();
        auto& manager = scene_.modifierManager();
        lfs::nodes::NodeTypeInfo slow;
        slow.id = "test.slow";
        slow.label = "Slow";
        slow.inputs = {{"Geometry", "Geometry", std::string(lfs::nodes::GEOMETRY_SOCKET)}};
        slow.outputs = slow.inputs;
        slow.evaluate = [](lfs::nodes::NodeContext& context) {
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
            context.set_output("Geometry", context.input("Geometry"));
        };
        ASSERT_TRUE(manager.registry().register_type(std::move(slow)));
        auto* tree = manager.tree(tree_);
        const auto before = tree->to_json();
        tree->find_node("Correct")->type_id = "test.slow";
        manager.recordTreeEdit(tree_, before);
        context_->Update();
        auto* canvas = dynamic_cast<lfs::vis::gui::NodeCanvasElement*>(document_->GetElementById("node-editor-canvas"));
        auto* card = canvas->QuerySelector(".node-box[data-node=Correct]");
        ASSERT_NE(card, nullptr);
        const auto node_count = canvas->viewState()["nodes"].size();
        const auto link_count = canvas->viewState()["links"];
        manager.tick();
        context_->Update();
        EXPECT_FALSE(card->IsClassSet("evaluating"));
        for (int attempt = 0; attempt < 100 && manager.progress().node != "Correct"; ++attempt)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        std::this_thread::sleep_for(std::chrono::milliseconds(90));
        context_->Update();
        context_->Render();
        EXPECT_TRUE(card->IsClassSet("evaluating"));
        EXPECT_EQ(canvas->viewState()["nodes"].size(), node_count);
        EXPECT_EQ(canvas->viewState()["links"], link_count);
        pointer("mousedown", 600, 65);
        pointer("mousemove", 670, 105);
        SCOPED_TRACE(canvas->viewState().dump());
        EXPECT_TRUE(card->IsClassSet("selected"));
        EXPECT_NEAR(card->GetAbsoluteOffset().x, 610.0f, 2.0f);
        EXPECT_EQ(tree->find_node("Correct")->location[0], 270.0f);
        pointer("mouseup", 670, 105);
        ASSERT_TRUE(manager.evaluate(host_).ok);
        context_->Update();
        context_->Render();
        EXPECT_EQ(canvas->QuerySelector(".node-box[data-node=Correct]"), card);
        EXPECT_EQ(canvas->viewState()["nodes"].size(), node_count);
        EXPECT_EQ(canvas->viewState()["links"], link_count);
        EXPECT_FALSE(card->IsClassSet("evaluating"));
    }

    TEST_F(NodeCanvasWidgets, InspectorKeepsFullWidthAcrossPatches) {
        attachGraph();
        context_->SetDimensions({2488, 1246});
        auto* canvas = dynamic_cast<lfs::vis::gui::NodeCanvasElement*>(document_->GetElementById("node-editor-canvas"));
        for (int i = 0; i < 10; ++i) {
            ASSERT_TRUE(canvas->selectNodes({i % 2 == 0 ? "Correct" : "Value"}, std::nullopt));
            context_->Update();
            context_->Render();
            Rml::ElementList sections;
            canvas->QuerySelectorAll(sections, ".sidebar-section");
            ASSERT_EQ(sections.size(), 3u);
            for (auto* section : sections) {
                EXPECT_GE(section->GetBox().GetSize(Rml::BoxArea::Border).x, 520.0f);
            }
        }
    }

    TEST_F(NodeCanvasWidgets, DomPatchNeverReplacesPrivateScrollbarChildren) {
        auto* parent = document_->AppendChild(document_->CreateElement("div"));
        parent->SetInnerRML("<div id='first'>First</div>");
        auto private_child = document_->CreateElement("scrollbarvertical");
        auto* scrollbar = parent->AppendChild(std::move(private_child), false);
        ASSERT_EQ(parent->GetNumChildren(), 1);
        ASSERT_EQ(parent->GetNumChildren(true), 2);
        lfs::vis::gui::node_widgets::patchMarkup(*parent, "<div id='first'>First</div><div id='second'>Second</div>");
        EXPECT_EQ(parent->GetNumChildren(), 2);
        EXPECT_EQ(parent->GetNumChildren(true), 3);
        EXPECT_EQ(parent->GetChild(2), scrollbar);
        lfs::vis::gui::node_widgets::patchMarkup(*parent, "<div id='first'>Changed</div>");
        EXPECT_EQ(parent->GetNumChildren(), 1);
        EXPECT_EQ(parent->GetChild(1), scrollbar);
    }

    TEST_F(NodeCanvasWidgets, AddPopupFocusesSearchAndEnterAddsExactTopHit) {
        attachGraph();
        auto* canvas = dynamic_cast<lfs::vis::gui::NodeCanvasElement*>(document_->GetElementById("node-editor-canvas"));
        canvas->headerAction("add", 400, 400);
        context_->Update();
        auto* search = document_->GetElementById("node-add-search");
        ASSERT_NE(search, nullptr);
        EXPECT_EQ(context_->GetFocusElement(), search);
        context_->ProcessTextInput("HSV");
        context_->Update();
        auto* popup = document_->GetElementById("node-add-menu");
        ASSERT_NE(popup, nullptr);
        EXPECT_TRUE(popup->IsClassSet("searching"));
        auto* top = popup->QuerySelector(".first-hit");
        ASSERT_NE(top, nullptr);
        EXPECT_EQ(top->GetAttribute<Rml::String>("data-type", ""), "lfs.hsv_range");
        const auto count = scene_.modifierManager().tree(tree_)->nodes.size();
        context_->ProcessKeyDown(Rml::Input::KI_RETURN, 0);
        context_->Update();
        EXPECT_EQ(document_->GetElementById("node-add-menu"), nullptr);
        EXPECT_EQ(scene_.modifierManager().tree(tree_)->nodes.size(), count + 1);
        EXPECT_EQ(scene_.modifierManager().tree(tree_)->nodes.back().type_id, "lfs.hsv_range");
    }

    TEST_F(NodeCanvasWidgets, RenderNeverPatchesDomAndReplacingGraphRebindsAndFrames) {
        attachGraph();
        auto& manager = scene_.modifierManager();
        auto* canvas = dynamic_cast<lfs::vis::gui::NodeCanvasElement*>(document_->GetElementById("node-editor-canvas"));
        auto* tree = manager.tree(tree_);
        const auto before = tree->to_json();
        tree->add_node("lfs.value", "Added");
        manager.recordTreeEdit(tree_, before);
        context_->Render();
        EXPECT_EQ(canvas->QuerySelector(".node-box[data-node=Added]"), nullptr);
        context_->Update();
        EXPECT_NE(canvas->QuerySelector(".node-box[data-node=Added]"), nullptr);
        ASSERT_TRUE(manager.removeTree(tree_));
        auto& replacement = manager.newTree("Replacement");
        replacement.add_node("lfs.colour_correct", "Fresh");
        manager.addModifier(host_, replacement.uuid);
        canvas->invalidateView();
        context_->Update();
        context_->Render();
        EXPECT_EQ(canvas->viewState()["tree"], replacement.uuid);
        EXPECT_NE(canvas->QuerySelector(".node-box[data-node=Fresh]"), nullptr);
        EXPECT_EQ(canvas->QuerySelector(".node-box[data-node=Correct]"), nullptr);
        EXPECT_EQ(canvas->viewState()["nodes"].size(), 3u);
        EXPECT_GT(canvas->viewState()["zoom"].get<float>(), 0.3f);
    }
} // namespace

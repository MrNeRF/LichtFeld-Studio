/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// The dataset point cloud carries two crop volumes before training: the ROI that bounds
// training and the Crop that removes points. These tests pin their independence.

#include "core/crop_polygon.hpp"
#include "core/editor_context.hpp"
#include "core/event_bridge/event_bridge.hpp"
#include "core/events.hpp"
#include "core/point_cloud.hpp"
#include "core/services.hpp"
#include "core/splat_data.hpp"
#include "gui/gizmo_manager.hpp"
#include "gui/gui_manager.hpp"
#include "gui/panel_registry.hpp"
#include "gui/rmlui/elements/scene_graph_element.hpp"
#include "input/input_controller.hpp"
#include "input/key_codes.hpp"
#include "io/scene_chapter_adapter.hpp"
#include "operation/undo_history.hpp"
#include "scene/scene_manager.hpp"
#include "tools/unified_tool_registry.hpp"
#include "visualizer/gui_capabilities.hpp"
#include "visualizer/visualizer_impl.hpp"
#include <RmlUi/Core.h>
#include <RmlUi/Core/ElementInstancer.h>
#include <RmlUi/Core/Factory.h>
#include <array>
#include <filesystem>
#include <gtest/gtest.h>
#include <string>
#include <vector>

namespace {
    using namespace lfs;
    using namespace lfs::vis;
    namespace gui = lfs::vis::gui;

    constexpr std::array kAllShapes{core::NodeType::CROPBOX, core::NodeType::ELLIPSOID, core::NodeType::CROP_POLYGON};

    std::shared_ptr<core::PointCloud> threePoints() {
        return std::make_shared<core::PointCloud>(
            core::Tensor::from_vector({-2.f, 0.f, 0.f, 0.f, 0.f, 0.f, 2.f, 0.f, 0.f}, {3, 3}, core::Device::CPU),
            core::Tensor::ones({3, 3}, core::Device::CPU));
    }

    std::unique_ptr<core::SplatData> fourSplats() {
        using core::Device;
        using core::Tensor;
        return std::make_unique<core::SplatData>(0,
                                                 Tensor::from_vector({-2.f, 0.f, 2.f, 0.f, 0.f, 2.f, .5f, 0.f, 2.f, 2.f, 0.f, 2.f}, {size_t{4}, size_t{3}}, Device::CPU),
                                                 Tensor::ones({size_t{4}, size_t{1}, size_t{3}}, Device::CPU),
                                                 Tensor::zeros({size_t{4}, size_t{0}, size_t{3}}, Device::CPU),
                                                 Tensor::zeros({size_t{4}, size_t{3}}, Device::CPU),
                                                 Tensor::from_vector({1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f}, {size_t{4}, size_t{4}}, Device::CPU),
                                                 Tensor::ones({size_t{4}, size_t{1}}, Device::CPU), 1.f);
    }

    core::CropPolygonData triangle(const core::CropRole role) {
        core::CropPolygonData data;
        data.vertices = {{-1.f, -1.f}, {1.f, -1.f}, {0.f, 1.f}};
        data.half_depth = 1.f;
        data.role = role;
        return data;
    }

    core::NodeId shapeFor(SceneManager& manager, RenderingManager* rendering, const core::NodeId owner,
                          const core::CropRole role, const core::NodeType shape) {
        if (shape == core::NodeType::CROPBOX)
            return cap::ensureCropBox(manager, rendering, owner, role).value_or(core::NULL_NODE);
        if (shape == core::NodeType::ELLIPSOID)
            return cap::ensureEllipsoid(manager, rendering, owner, role).value_or(core::NULL_NODE);
        const auto polygon = cap::createCropPolygon(manager, rendering, owner, triangle(role), glm::mat4(1.0f));
        return polygon ? *polygon : core::NULL_NODE;
    }

    core::NodeId roleVolume(const core::Scene& scene, const core::NodeId owner, const core::CropRole role) {
        for (const auto type : kAllShapes)
            if (const auto id = scene.getCropForRole(owner, role, type); id != core::NULL_NODE)
                return id;
        return core::NULL_NODE;
    }

    class CropRoleSceneTest : public ::testing::Test {
    protected:
        std::unique_ptr<SceneManager> sm;
        core::NodeId cloud = core::NULL_NODE;

        void SetUp() override {
            event::EventBridge::instance().clear_all();
            services().clear();
            op::undoHistory().clear();
            sm = std::make_unique<SceneManager>();
            services().set(sm.get());
            auto& scene = sm->getScene();
            cloud = scene.addPointCloud("Cloud", threePoints(), scene.addDataset("Dataset"));
            ASSERT_NE(cloud, core::NULL_NODE);
            ASSERT_EQ(scene.getDatasetPointCloudNode(), scene.getNodeById(cloud));
        }
        void TearDown() override {
            op::undoHistory().clear();
            services().clear();
            sm.reset();
            event::EventBridge::instance().clear_all();
        }
    };

    TEST_F(CropRoleSceneTest, RolesStayIndependentInEitherCreationOrder) {
        for (const auto first_role : {core::CropRole::Primary, core::CropRole::Secondary}) {
            SCOPED_TRACE(static_cast<int>(first_role));
            SetUp();
            auto& scene = sm->getScene();
            const auto other_role = first_role == core::CropRole::Primary ? core::CropRole::Secondary : core::CropRole::Primary;
            const auto first = cap::ensureCropBox(*sm, nullptr, cloud, first_role);
            const auto second = cap::ensureCropBox(*sm, nullptr, cloud, other_role);
            ASSERT_TRUE(first);
            ASSERT_TRUE(second);
            EXPECT_NE(*first, *second);
            EXPECT_EQ(scene.getCropForRole(cloud, first_role, core::NodeType::CROPBOX), *first);
            EXPECT_EQ(scene.getCropForRole(cloud, other_role, core::NodeType::CROPBOX), *second);
            EXPECT_EQ(cap::ensureCropBox(*sm, nullptr, cloud, first_role), first);
            const auto roi = first_role == core::CropRole::Primary ? *first : *second;
            const auto crop = first_role == core::CropRole::Primary ? *second : *first;
            EXPECT_EQ(scene.getNodeById(roi)->name, "Cloud_ROI");
            EXPECT_EQ(scene.getNodeById(crop)->name, "Cloud_Crop");
            EXPECT_TRUE(isPointCloudRoi(scene, *scene.getNodeById(roi)));
            EXPECT_FALSE(isPointCloudRoi(scene, *scene.getNodeById(crop)));
            // The legacy single-volume getters keep answering with the ROI.
            EXPECT_EQ(scene.getCropBoxForSplat(cloud), roi);
            TearDown();
        }
    }

    TEST_F(CropRoleSceneTest, ShapeChangesKeepTheRoleAndLeaveTheSiblingAlone) {
        auto& scene = sm->getScene();
        const auto roi = cap::ensureCropBox(*sm, nullptr, cloud, core::CropRole::Primary);
        const auto crop = cap::ensureCropBox(*sm, nullptr, cloud, core::CropRole::Secondary);
        ASSERT_TRUE(roi);
        ASSERT_TRUE(crop);
        const auto roi_uuid = scene.getNodeById(*roi)->uuid;
        const auto crop_uuid = scene.getNodeById(*crop)->uuid;
        for (const auto shape : {core::NodeType::ELLIPSOID, core::NodeType::CROP_POLYGON, core::NodeType::CROPBOX}) {
            SCOPED_TRACE(static_cast<int>(shape));
            const auto id = shapeFor(*sm, nullptr, cloud, core::CropRole::Secondary, shape);
            ASSERT_NE(id, core::NULL_NODE);
            EXPECT_EQ(scene.getNodeById(id)->type, shape);
            EXPECT_EQ(scene.getNodeById(id)->cropRole(), core::CropRole::Secondary);
            // A role volume stays one node, so its identity survives every shape.
            EXPECT_EQ(scene.getNodeById(id)->uuid, crop_uuid);
            ASSERT_NE(scene.getNodeByUuid(roi_uuid), nullptr);
            EXPECT_EQ(scene.getNodeByUuid(roi_uuid)->type, core::NodeType::CROPBOX);
            EXPECT_EQ(scene.getNodeByUuid(roi_uuid)->cropRole(), core::CropRole::Primary);
        }

        const auto polygon_roi = shapeFor(*sm, nullptr, cloud, core::CropRole::Primary, core::NodeType::CROP_POLYGON);
        ASSERT_NE(polygon_roi, core::NULL_NODE);
        EXPECT_EQ(scene.getNodeById(polygon_roi)->cropRole(), core::CropRole::Primary);
        EXPECT_EQ(scene.getNodeById(polygon_roi)->uuid, roi_uuid);
        EXPECT_EQ(scene.getNodeById(roleVolume(scene, cloud, core::CropRole::Secondary))->type, core::NodeType::CROPBOX);

        // Undo walks back one role at a time.
        ASSERT_TRUE(op::undoHistory().undo().success);
        EXPECT_EQ(scene.getNodeById(roleVolume(scene, cloud, core::CropRole::Primary))->type, core::NodeType::CROPBOX);
        EXPECT_EQ(scene.getNodeById(roleVolume(scene, cloud, core::CropRole::Secondary))->type, core::NodeType::CROPBOX);
        ASSERT_TRUE(op::undoHistory().undo().success);
        EXPECT_EQ(scene.getNodeById(roleVolume(scene, cloud, core::CropRole::Secondary))->type, core::NodeType::CROP_POLYGON);
        EXPECT_EQ(scene.getNodeById(roleVolume(scene, cloud, core::CropRole::Primary))->type, core::NodeType::CROPBOX);
        EXPECT_EQ(scene.getNodeById(roleVolume(scene, cloud, core::CropRole::Primary))->uuid, roi_uuid);
        EXPECT_EQ(scene.getNodeById(roleVolume(scene, cloud, core::CropRole::Secondary))->uuid, crop_uuid);
    }

    TEST_F(CropRoleSceneTest, NewRoiIsABoxOrPolygonWhileAnOlderEllipsoidRoiStillConverts) {
        auto& scene = sm->getScene();
        EXPECT_FALSE(cap::ensureEllipsoid(*sm, nullptr, cloud, core::CropRole::Primary));
        const auto box = cap::ensureCropBox(*sm, nullptr, cloud, core::CropRole::Primary);
        ASSERT_TRUE(box);
        EXPECT_FALSE(cap::ensureEllipsoid(*sm, nullptr, cloud, core::CropRole::Primary));
        EXPECT_EQ(scene.getNodeById(*box)->type, core::NodeType::CROPBOX);
        EXPECT_TRUE(cap::ensureEllipsoid(*sm, nullptr, cloud, core::CropRole::Secondary));
        EXPECT_EQ(scene.getNodeById(*box)->type, core::NodeType::CROPBOX);

        sm->removeNode(*box);
        const auto older = scene.addEllipsoid("Older ROI", cloud);
        ASSERT_NE(older, core::NULL_NODE);
        EXPECT_EQ(cap::ensureEllipsoid(*sm, nullptr, cloud, core::CropRole::Primary), older);
        const auto converted = cap::ensureCropBox(*sm, nullptr, cloud, core::CropRole::Primary);
        ASSERT_TRUE(converted);
        EXPECT_EQ(scene.getNodeById(*converted)->type, core::NodeType::CROPBOX);
        EXPECT_EQ(scene.getCropForRole(cloud, core::CropRole::Primary, core::NodeType::ELLIPSOID), core::NULL_NODE);
        EXPECT_NE(scene.getCropForRole(cloud, core::CropRole::Secondary, core::NodeType::ELLIPSOID), core::NULL_NODE);
    }

    TEST_F(CropRoleSceneTest, DirectPolygonCreationUsesTheRequestedRoleInEitherOrder) {
        for (const auto first_role : {core::CropRole::Primary, core::CropRole::Secondary}) {
            SCOPED_TRACE(static_cast<int>(first_role));
            auto& scene = sm->getScene();
            const auto owner = scene.addPointCloud(first_role == core::CropRole::Primary ? "ROI first" : "Crop first", threePoints());
            ASSERT_NE(owner, core::NULL_NODE);
            auto data = triangle(first_role);
            const auto first = scene.addCropPolygon("First polygon", owner, data);
            ASSERT_NE(first, core::NULL_NODE);
            data.role = first_role == core::CropRole::Primary ? core::CropRole::Secondary : core::CropRole::Primary;
            const auto second = scene.addCropPolygon("Second polygon", owner, data);
            ASSERT_NE(second, core::NULL_NODE);
            ASSERT_NE(first, second);
            EXPECT_EQ(scene.getNodeById(first)->cropRole(), first_role);
            EXPECT_EQ(scene.getNodeById(second)->cropRole(), data.role);
            EXPECT_EQ(scene.addCropPolygon("Same role", owner, data), second);
        }
    }

    TEST_F(CropRoleSceneTest, OnlyTheDatasetPointCloudAndTheTrainingModelHaveAnRoi) {
        auto& scene = sm->getScene();
        const auto standalone = scene.addPointCloud("Standalone", threePoints());
        const auto splat = scene.addSplat("Splat", fourSplats());
        ASSERT_NE(standalone, core::NULL_NODE);
        ASSERT_NE(splat, core::NULL_NODE);
        for (const auto owner : {standalone, splat}) {
            const auto box = cap::ensureCropBox(*sm, nullptr, owner, core::CropRole::Primary);
            ASSERT_TRUE(box);
            const auto& node = *scene.getNodeById(*box);
            EXPECT_FALSE(isPointCloudRoi(scene, node));
            EXPECT_FALSE(isTrainingRoi(scene, node));
            EXPECT_EQ(node.name, scene.getNodeById(owner)->name + "_cropbox");
        }
        scene.setTrainingModelNode(splat);
        EXPECT_TRUE(isTrainingRoi(scene, *scene.getNodeById(scene.getCropBoxForSplat(splat))));
    }

    TEST(CropRolePersistenceTest, RolesRoundTripAndOldProjectsLoadTheRoi) {
        namespace pj = lfs::io::project;
        core::Scene scene;
        const auto owner = scene.addSplat("Role persistence", fourSplats());
        ASSERT_NE(owner, core::NULL_NODE);
        const auto owner_uuid = scene.getNodeById(owner)->uuid;
        const auto primary = scene.addCropBox("Primary", owner);
        const auto secondary = scene.addEllipsoid("Secondary", owner, core::CropRole::Secondary);
        ASSERT_NE(primary, core::NULL_NODE);
        ASSERT_NE(secondary, core::NULL_NODE);
        ASSERT_NE(primary, secondary);
        // The four splats lie on a line, so the ellipsoid sized from their bounds has zero radii, which a project rejects.
        auto ellipsoid = *scene.getNodeById(secondary)->ellipsoid;
        ellipsoid.radii = glm::vec3(2.0f, 1.0f, 1.0f);
        scene.setEllipsoidData(secondary, ellipsoid);
        const auto primary_uuid = scene.getNodeById(primary)->uuid;
        const auto secondary_uuid = scene.getNodeById(secondary)->uuid;
        const pj::ScenePayloadBindings bindings{{owner_uuid, pj::PayloadBinding{.fourcc = "DSRC", .instance_uuid = owner_uuid, .reference_uuid = std::nullopt, .source_kind = "ply"}}};
        const auto chapter = pj::capture_scene_graph(scene, bindings);
        ASSERT_TRUE(chapter) << lfs::format_for_developer(chapter.error());
        const auto bytes = chapter->to_bytes();
        const std::string text(reinterpret_cast<const char*>(bytes.data()), bytes.size());
        // Only the secondary volume stores its role, so a project without the key loads every volume as the ROI.
        const auto key = text.find("\"secondary_crop\"");
        ASSERT_NE(key, std::string::npos);
        EXPECT_EQ(text.find("\"secondary_crop\"", key + 1), std::string::npos);
        const auto decoded = pj::SceneGraphChapter::from_bytes(bytes);
        ASSERT_TRUE(decoded) << lfs::format_for_developer(decoded.error());
        core::Scene restored;
        pj::ScenePayloadResolver resolver{.splat = [](const pj::PayloadBinding&) -> lfs::Result<std::unique_ptr<core::SplatData>> { return fourSplats(); }};
        ASSERT_TRUE(pj::hydrate_scene_graph(*decoded, restored, resolver));
        const auto* restored_owner = restored.getNodeByUuid(owner_uuid);
        ASSERT_NE(restored_owner, nullptr);
        ASSERT_NE(restored.getNodeByUuid(primary_uuid), nullptr);
        ASSERT_NE(restored.getNodeByUuid(secondary_uuid), nullptr);
        EXPECT_EQ(restored.getNodeByUuid(primary_uuid)->cropRole(), core::CropRole::Primary);
        EXPECT_EQ(restored.getNodeByUuid(secondary_uuid)->cropRole(), core::CropRole::Secondary);
        EXPECT_EQ(restored.getCropForRole(restored_owner->id, core::CropRole::Primary, core::NodeType::CROPBOX),
                  restored.getNodeByUuid(primary_uuid)->id);
        EXPECT_EQ(restored.getCropForRole(restored_owner->id, core::CropRole::Secondary, core::NodeType::ELLIPSOID),
                  restored.getNodeByUuid(secondary_uuid)->id);
    }

    class TreeRenderer final : public Rml::RenderInterface {
        Rml::CompiledGeometryHandle CompileGeometry(Rml::Span<const Rml::Vertex>, Rml::Span<const int>) override { return 1; }
        void RenderGeometry(Rml::CompiledGeometryHandle, Rml::Vector2f, Rml::TextureHandle) override {}
        void ReleaseGeometry(Rml::CompiledGeometryHandle) override {}
        Rml::TextureHandle LoadTexture(Rml::Vector2i&, const Rml::String&) override { return 0; }
        Rml::TextureHandle GenerateTexture(Rml::Span<const Rml::byte>, Rml::Vector2i) override { return 1; }
        void ReleaseTexture(Rml::TextureHandle) override {}
        void EnableScissorRegion(bool) override {}
        void SetScissorRegion(Rml::Rectanglei) override {}
    };

    class TreeSystemInterface final : public Rml::SystemInterface {
        bool LogMessage(Rml::Log::Type type, const Rml::String& message) override {
            if (type == Rml::Log::LT_ASSERT || message.find("Leak detected:") != Rml::String::npos)
                ADD_FAILURE() << message;
            return true;
        }
    };

    class RmlDefaultActionAccess : public Rml::Element {
    public:
        using Rml::Element::ProcessDefaultAction;
    };

    // A viewer before training: the dataset point cloud with its ROI box and its Crop box.
    class PreTrainingCropTest : public ::testing::Test {
    protected:
        TreeRenderer renderer;
        TreeSystemInterface system;
        Rml::ElementInstancerGeneric<gui::SceneGraphElement> tree_instancer;
        std::unique_ptr<VisualizerImpl> viewer;
        Rml::Context* context = nullptr;
        gui::SceneGraphElement* tree = nullptr;
        core::NodeId cloud = core::NULL_NODE;
        core::NodeId roi = core::NULL_NODE;
        core::NodeId crop = core::NULL_NODE;
        uint64_t generation = 1;

        void SetUp() override {
            event::EventBridge::instance().clear_all();
            services().clear();
            op::undoHistory().clear();
            ViewerOptions options;
            options.show_startup_overlay = false;
            viewer = std::make_unique<VisualizerImpl>(options);
            auto* manager = viewer->getSceneManager();
            auto& scene = manager->getScene();
            cloud = scene.addPointCloud("Cloud", threePoints(), scene.addDataset("Dataset"));
            ASSERT_NE(cloud, core::NULL_NODE);
            scene.getMutableNode("Cloud")->gaussian_count.store(3);
            manager->changeContentType(SceneManager::ContentType::Dataset);
            viewer->getEditorContext().update(manager, viewer->getTrainerManager());
            ASSERT_TRUE(viewer->getEditorContext().forcePointCloudMode());
            const auto first = cap::ensureCropBox(*manager, viewer->getRenderingManager(), cloud, core::CropRole::Primary);
            const auto second = cap::ensureCropBox(*manager, viewer->getRenderingManager(), cloud, core::CropRole::Secondary);
            ASSERT_TRUE(first);
            ASSERT_TRUE(second);
            roi = *first;
            crop = *second;
            op::undoHistory().clear();
            Rml::SetSystemInterface(&system);
            ASSERT_TRUE(Rml::Initialise());
            Rml::Factory::RegisterElementInstancer("scene-graph", &tree_instancer);
            ASSERT_TRUE(Rml::LoadFontFace((std::filesystem::path(PROJECT_ROOT_PATH) / "src/visualizer/gui/assets/fonts/Inter-Regular.ttf").generic_string()));
            context = Rml::CreateContext("crop-role-test", {500, 700}, &renderer);
            ASSERT_NE(context, nullptr);
            auto* doc = context->LoadDocumentFromMemory("<rml><head><style>body{font-family:Inter;font-size:12px;} scene-graph{display:block;position:relative;overflow:auto;width:480px;height:600px;} .scene-graph-content{position:relative;width:100%;} .tree-row{display:block;position:absolute;left:0;right:0;height:20px;}</style></head><body/></rml>");
            ASSERT_NE(doc, nullptr);
            auto element = doc->CreateElement("scene-graph");
            ASSERT_NE(element, nullptr);
            tree = static_cast<gui::SceneGraphElement*>(element.get());
            doc->AppendChild(std::move(element));
            doc->Show();
            sync();
        }
        void TearDown() override {
            if (context)
                Rml::RemoveContext("crop-role-test");
            Rml::Shutdown();
            viewer.reset();
            op::undoHistory().clear();
            services().clear();
            event::EventBridge::instance().clear_all();
            UnifiedToolRegistry::instance().setActiveTool("");
        }
        void sync() {
            gui::PanelDrawContext ctx;
            ctx.scene = &viewer->getScene();
            ctx.scene_generation = generation++;
            (void)tree->syncFromScene(ctx);
            context->Update();
        }
        Rml::Element* row(const core::NodeId id) {
            return tree->QuerySelector(".tree-row[data-node-id='" + std::to_string(id) + "']");
        }
        void click(const core::NodeId id) {
            auto* r = row(id);
            ASSERT_NE(r, nullptr);
            auto* name = r->QuerySelector(".node-name");
            ASSERT_NE(name, nullptr);
            Rml::Event event(name, Rml::EventId::Click, "click", {}, true);
            const auto action = &RmlDefaultActionAccess::ProcessDefaultAction;
            (tree->*action)(event);
            sync();
        }
        void pressEnter() {
            InputController controller(nullptr, viewer->getViewport());
            controller.getBindings().setBinding(input::ToolMode::CROP_BOX, input::Action::APPLY_CROP_BOX,
                                                input::KeyTrigger{input::KEY_ENTER, input::MODIFIER_NONE});
            controller.handleKey(input::KEY_ENTER, input::ACTION_PRESS, input::MODIFIER_NONE);
        }
        gui::GizmoManager& gizmo() { return viewer->getGuiManager()->gizmo(); }
    };

    TEST_F(PreTrainingCropTest, RoiRowIsPurpleAndCropRowKeepsTheDefaultIconColour) {
        auto* roi_row = row(roi);
        auto* crop_row = row(crop);
        ASSERT_NE(roi_row, nullptr);
        ASSERT_NE(crop_row, nullptr);
        auto* roi_icon = roi_row->QuerySelector(".type-icon");
        auto* crop_icon = crop_row->QuerySelector(".type-icon");
        ASSERT_NE(roi_icon, nullptr);
        ASSERT_NE(crop_icon, nullptr);
        const auto* color = roi_icon->GetLocalProperty("image-color");
        ASSERT_NE(color, nullptr);
        EXPECT_EQ(color->Get<Rml::Colourb>(), Rml::Colourb(179, 77, 255, 255));
        EXPECT_EQ(crop_icon->GetLocalProperty("image-color"), nullptr);
    }

    TEST_F(PreTrainingCropTest, SelectingEachRoleOpensItsOwnTool) {
        auto& registry = UnifiedToolRegistry::instance();
        auto& scene = viewer->getScene();
        for (const auto& [id, tool] : {std::pair{roi, "builtin.roi"}, std::pair{crop, "builtin.crop"}}) {
            SCOPED_TRACE(tool);
            click(id);
            EXPECT_EQ(viewer->getSceneManager()->getSelectedNodeIds(), (std::vector<core::NodeId>{id}));
            EXPECT_EQ(registry.getActiveTool(), tool);
            EXPECT_EQ(viewer->getEditorContext().getActiveOperator(), tool);
            EXPECT_EQ(gizmo().cropToolVolumeNodeId(), id);
            // Selecting one volume never changes the other's visibility.
            EXPECT_TRUE(scene.getNodeById(roi)->visible);
            EXPECT_TRUE(scene.getNodeById(crop)->visible);
        }
    }

    TEST_F(PreTrainingCropTest, ToolbarToolsSelectOrCreateTheirOwnVolume) {
        auto& scene = viewer->getScene();
        auto* manager = viewer->getSceneManager();
        const auto crop_uuid = scene.getNodeById(crop)->uuid;
        manager->selectNode(cloud);
        ASSERT_TRUE(gizmo().activateCropTool("builtin.roi"));
        EXPECT_EQ(manager->getSelectedNodeIds(), (std::vector<core::NodeId>{roi}));
        EXPECT_EQ(UnifiedToolRegistry::instance().getActiveTool(), "builtin.roi");

        // Without a Crop, the Crop tool starts with a box and leaves the ROI alone.
        manager->removeNode(crop);
        ASSERT_EQ(scene.getNodeByUuid(crop_uuid), nullptr);
        manager->selectNode(cloud);
        ASSERT_TRUE(gizmo().activateCropTool("builtin.crop"));
        const auto created = roleVolume(scene, cloud, core::CropRole::Secondary);
        ASSERT_NE(created, core::NULL_NODE);
        EXPECT_EQ(scene.getNodeById(created)->type, core::NodeType::CROPBOX);
        EXPECT_EQ(manager->getSelectedNodeIds(), (std::vector<core::NodeId>{created}));
        EXPECT_EQ(UnifiedToolRegistry::instance().getActiveTool(), "builtin.crop");
        EXPECT_EQ(roleVolume(scene, cloud, core::CropRole::Primary), roi);
    }

    TEST_F(PreTrainingCropTest, RoiRejectsTheEllipsoidWhileTheCropTakesIt) {
        auto& scene = viewer->getScene();
        // The toolbar's ellipsoid button: set the tool shape, then add an ellipsoid to the selection.
        const auto pick_ellipsoid = [&] {
            gizmo().setCropToolShape("ellipsoid");
            core::events::cmd::AddCropEllipsoid{.node_name = viewer->getSceneManager()->getSelectedNodeName()}.emit();
            sync();
        };
        click(roi);
        pick_ellipsoid();
        EXPECT_EQ(gizmo().cropToolShape(), "box");
        EXPECT_EQ(scene.getNodeById(roi)->type, core::NodeType::CROPBOX);
        EXPECT_EQ(scene.getEllipsoidForSplat(cloud), core::NULL_NODE);
        click(crop);
        pick_ellipsoid();
        const auto ellipsoid = scene.getCropForRole(cloud, core::CropRole::Secondary, core::NodeType::ELLIPSOID);
        EXPECT_NE(ellipsoid, core::NULL_NODE);
        EXPECT_EQ(scene.getNodeById(roi)->type, core::NodeType::CROPBOX);
        EXPECT_EQ(roleVolume(scene, cloud, core::CropRole::Primary), roi);
    }

    TEST_F(PreTrainingCropTest, EnterAppliesTheCropAndUndoRestoresItsPoints) {
        auto& scene = viewer->getScene();
        auto* volume = scene.getMutableNode(scene.getNodeById(crop)->name);
        volume->cropbox->min = {-1.f, -1.f, -1.f};
        volume->cropbox->max = {1.f, 1.f, 1.f};
        const auto crop_uuid = volume->uuid;
        const auto roi_uuid = scene.getNodeById(roi)->uuid;
        click(crop);
        ASSERT_EQ(UnifiedToolRegistry::instance().getActiveTool(), "builtin.crop");
        pressEnter();
        ASSERT_NE(scene.getNodeById(cloud)->point_cloud, nullptr);
        EXPECT_EQ(scene.getNodeById(cloud)->point_cloud->size(), 1);
        EXPECT_EQ(scene.getNodeByUuid(crop_uuid), nullptr);
        EXPECT_NE(scene.getNodeByUuid(roi_uuid), nullptr);
        ASSERT_TRUE(op::undoHistory().undo().success);
        EXPECT_EQ(scene.getNodeById(cloud)->point_cloud->size(), 3);
        EXPECT_NE(scene.getNodeByUuid(crop_uuid), nullptr);
        EXPECT_NE(scene.getNodeByUuid(roi_uuid), nullptr);
    }

    TEST_F(PreTrainingCropTest, EnterOnTheRoiCommitsWithoutRemovingPointsOrVolumes) {
        auto& scene = viewer->getScene();
        auto* volume = scene.getMutableNode(scene.getNodeById(roi)->name);
        volume->cropbox->min = {-1.f, -1.f, -1.f};
        volume->cropbox->max = {1.f, 1.f, 1.f};
        const auto roi_uuid = scene.getNodeById(roi)->uuid;
        const auto crop_uuid = scene.getNodeById(crop)->uuid;
        click(roi);
        ASSERT_EQ(UnifiedToolRegistry::instance().getActiveTool(), "builtin.roi");
        pressEnter();
        EXPECT_TRUE(UnifiedToolRegistry::instance().getActiveTool().empty());
        EXPECT_EQ(scene.getNodeById(cloud)->point_cloud->size(), 3);
        EXPECT_NE(scene.getNodeByUuid(roi_uuid), nullptr);
        EXPECT_NE(scene.getNodeByUuid(crop_uuid), nullptr);
    }
} // namespace

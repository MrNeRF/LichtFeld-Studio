/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/collision_volume.hpp"
#include "core/event_bridge/event_bridge.hpp"
#include "core/services.hpp"
#include "core/splat_data.hpp"
#include "core/splat_operations.hpp"
#include "gui/gizmo_manager.hpp"
#include "gui/panel_registry.hpp"
#include "gui/rmlui/elements/scene_graph_element.hpp"
#include "gui/video_export_utils.hpp"
#include "internal/viewport.hpp"
#include "io/scene_chapter_adapter.hpp"
#include "operation/undo_history.hpp"
#include "operator/operator_registry.hpp"
#include "operator/ops/crop_polygon_ops.hpp"
#include "operator/ops/scene_ops.hpp"
#include "rendering/viewport_request_builder.hpp"
#include "rendering/vksplat_overlay_params.hpp"
#include "scene/scene_manager.hpp"
#include "selection/selection_service.hpp"
#include "tools/unified_tool_registry.hpp"
#include "visualizer/gui_capabilities.hpp"
#include <RmlUi/Core.h>
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <glm/gtc/matrix_transform.hpp>
#include <gtest/gtest.h>
#include <limits>
#include <regex>

namespace {
    using namespace lfs;
    using namespace lfs::vis;
    std::unique_ptr<core::SplatData> groupTestSplat(core::Device device = core::Device::CPU) {
        using core::Device;
        using core::Tensor;
        return std::make_unique<core::SplatData>(0,
                                                 Tensor::from_vector({-2.f, 0.f, 2.f, 0.f, 0.f, 2.f, .5f, 0.f, 2.f, 2.f, 0.f, 2.f}, {size_t{4}, size_t{3}}, device),
                                                 Tensor::ones({size_t{4}, size_t{1}, size_t{3}}, device),
                                                 Tensor::zeros({size_t{4}, size_t{0}, size_t{3}}, device),
                                                 Tensor::zeros({size_t{4}, size_t{3}}, device),
                                                 Tensor::from_vector({1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f}, {size_t{4}, size_t{4}}, device),
                                                 Tensor::ones({size_t{4}, size_t{1}}, device), 1.f);
    }

    class SplatCropWorkflowTest : public ::testing::Test {
    protected:
        std::unique_ptr<SceneManager> sm;
        core::NodeId owner, other_owner, box, other_box;
        void SetUp() override {
            event::EventBridge::instance().clear_all();
            services().clear();
            op::operators().clear();
            op::undoHistory().clear();
            sm = std::make_unique<SceneManager>();
            services().set(sm.get());
            op::operators().setSceneManager(sm.get());
            op::registerSceneOperators();
            op::registerCropPolygonOperators();
            auto& scene = sm->getScene();
            owner = scene.addSplatPlaceholder("first");
            other_owner = scene.addSplatPlaceholder("second");
            box = scene.addCropBox("volume", owner);
            other_box = scene.addCropBox("volume", other_owner);
            ASSERT_NE(box, core::NULL_NODE);
            ASSERT_NE(other_box, core::NULL_NODE);
        }
        void TearDown() override {
            op::undoHistory().clear();
            op::unregisterCropPolygonOperators();
            op::unregisterSceneOperators();
            op::operators().clear();
            services().clear();
            sm.reset();
            event::EventBridge::instance().clear_all();
        }
        void select(core::NodeId id) { sm->selectNode(id); }
    };

    TEST_F(SplatCropWorkflowTest, EveryReferencedOperatorIsRegistered) {
        for (const auto& descriptor : core::splatOperationDescriptors())
            for (const auto id : {descriptor.set_operator, descriptor.duplicate_operator})
                if (!id.empty())
                    EXPECT_NE(op::operators().getDescriptor(std::string(id)), nullptr) << id;
    }

    TEST_F(SplatCropWorkflowTest, ExplicitUuidIgnoresOtherSelectionAndNeverFallsBackWhenInvalid) {
        auto& scene = sm->getScene();
        select(other_box);
        op::OperatorProperties props;
        props.set("node_uuid", scene.getNodeById(box)->uuid.to_string());
        props.set("enabled", false);
        ASSERT_TRUE(op::operators().invoke("crop_box.set", &props).is_finished());
        EXPECT_FALSE(scene.getNodeById(box)->cropbox->enabled);
        const bool other = scene.getNodeById(other_box)->cropbox->enabled;
        props.set("node_uuid", std::string("invalid"));
        props.set("node", scene.getNodeById(other_box)->name);
        EXPECT_TRUE(op::operators().invoke("crop_box.set", &props).is_cancelled());
        EXPECT_EQ(scene.getNodeById(other_box)->cropbox->enabled, other);
    }

    TEST_F(SplatCropWorkflowTest, EllipsoidAndPolygonUseTheirRegisteredSettersWithExactTargets) {
        auto& scene = sm->getScene();
        const auto ellipsoid = scene.addEllipsoid("ellipsoid", owner);
        core::CropPolygonData data;
        data.vertices = {{-1, -1}, {1, -1}, {1, 1}, {-1, 1}};
        const auto polygon = scene.addCropPolygon("polygon", other_owner, data);
        ASSERT_NE(polygon, core::NULL_NODE);
        select(ellipsoid);
        op::OperatorProperties props;
        props.set("node_uuid", scene.getNodeById(polygon)->uuid.to_string());
        props.set("half_depth", 2.5f);
        ASSERT_TRUE(op::operators().invoke("crop_polygon.set", &props).is_finished());
        EXPECT_FLOAT_EQ(scene.getNodeById(polygon)->crop_polygon->half_depth, 2.5f);
        props.clear();
        props.set("node_uuid", scene.getNodeById(ellipsoid)->uuid.to_string());
        props.set("radii", glm::vec3(2, 3, 4));
        ASSERT_TRUE(op::operators().invoke("ellipsoid.set", &props).is_finished());
        EXPECT_EQ(scene.getNodeById(ellipsoid)->ellipsoid->radii, glm::vec3(2, 3, 4));
        for (const auto* id : {"crop_polygon.set", "ellipsoid.set"}) {
            props.set("node_uuid", std::string("invalid"));
            EXPECT_TRUE(op::operators().invoke(id, &props).is_cancelled());
        }
        select(polygon);
    }

    TEST_F(SplatCropWorkflowTest, NewCropsPreserveExistingInstancesAndHaveIndependentUndo) {
        auto& scene = sm->getScene();
        const auto uuid = scene.getNodeById(owner)->uuid;
        const auto source_uuid = scene.getNodeById(box)->uuid;
        const auto first = cap::createSplatCrop(*sm, nullptr, uuid, core::NodeType::CROPBOX);
        const auto second = cap::createSplatCrop(*sm, nullptr, uuid, core::NodeType::ELLIPSOID);
        ASSERT_TRUE(first);
        ASSERT_TRUE(second);
        const auto new_uuid = scene.getNodeById(*first)->uuid;
        EXPECT_NE(new_uuid, source_uuid);
        EXPECT_NE(scene.getNodeByUuid(source_uuid), nullptr);
        op::undoHistory().undo();
        EXPECT_NE(scene.getNodeByUuid(source_uuid), nullptr);
        EXPECT_NE(scene.getNodeByUuid(new_uuid), nullptr);
        op::undoHistory().undo();
        EXPECT_EQ(scene.getNodeByUuid(new_uuid), nullptr);
        EXPECT_NE(scene.getNodeByUuid(source_uuid), nullptr);
    }

    TEST_F(SplatCropWorkflowTest, NewPolygonNeverReplacesExistingCropVolumes) {
        auto& scene = sm->getScene();
        core::CropPolygonData data;
        data.vertices = {{-1, -1}, {1, -1}, {1, 1}, {-1, 1}};
        const auto first = cap::createCropPolygon(*sm, nullptr, owner, data, glm::mat4(1), true);
        const auto second = cap::createCropPolygon(*sm, nullptr, owner, data, glm::mat4(1), true);
        ASSERT_TRUE(first);
        ASSERT_TRUE(second);
        EXPECT_NE(*first, *second);
        EXPECT_NE(scene.getNodeById(box), nullptr);
        EXPECT_NE(scene.getNodeById(*first), nullptr);
        EXPECT_NE(scene.getNodeById(*second), nullptr);
    }

    TEST_F(SplatCropWorkflowTest, DuplicatingACropCreatesIndependentIdentityAndUndoableState) {
        auto& scene = sm->getScene();
        select(box);
        const auto original_uuid = scene.getNodeById(box)->uuid;
        const auto original_data = *scene.getNodeById(box)->cropbox;
        op::OperatorProperties duplicate;
        duplicate.set("node_uuid", original_uuid.to_string());
        ASSERT_TRUE(op::operators().invoke("crop_volume.duplicate", &duplicate).is_finished());
        const auto selected = sm->getSelectedNodeIds();
        ASSERT_EQ(selected.size(), 1);
        const auto copy_id = selected.front();
        ASSERT_NE(copy_id, box);
        const auto* copy = scene.getNodeById(copy_id);
        ASSERT_NE(copy, nullptr);
        const auto copy_uuid = copy->uuid;
        EXPECT_NE(copy_uuid, original_uuid);
        EXPECT_EQ(copy->parent_id, owner);
        EXPECT_EQ(copy->cropbox->min, original_data.min);
        EXPECT_EQ(copy->cropbox->enabled, original_data.enabled);
        EXPECT_NE(copy->cropbox.get(), scene.getNodeById(box)->cropbox.get());
        ASSERT_TRUE(op::undoHistory().undo().success);
        EXPECT_EQ(scene.getNodeByUuid(copy_uuid), nullptr);
        EXPECT_NE(scene.getNodeByUuid(original_uuid), nullptr);
        ASSERT_TRUE(op::undoHistory().redo().success);
        ASSERT_NE(scene.getNodeByUuid(copy_uuid), nullptr);
        op::OperatorProperties props;
        props.set("node_uuid", copy_uuid.to_string());
        props.set("inverse", true);
        ASSERT_TRUE(op::operators().invoke("crop_box.set", &props).is_finished());
        EXPECT_FALSE(scene.getNodeById(box)->cropbox->inverse);
        EXPECT_TRUE(scene.getNodeByUuid(copy_uuid)->cropbox->inverse);
    }

    TEST_F(SplatCropWorkflowTest, AllThreeCropShapesCanBeRepeatedWithoutReplacingSiblings) {
        auto& scene = sm->getScene();
        const auto ellipsoid = scene.addEllipsoid("ellipsoid", owner);
        core::CropPolygonData data;
        data.vertices = {{-1, -1}, {1, -1}, {1, 1}, {-1, 1}};
        const auto polygon = scene.addCropPolygon("polygon", owner, data);
        ASSERT_NE(polygon, core::NULL_NODE);
        for (const auto id : {box, ellipsoid, polygon}) {
            const auto uuid = scene.getNodeById(id)->uuid;
            const auto copy = cap::duplicateSplatCrop(*sm, nullptr, uuid);
            ASSERT_TRUE(copy);
            EXPECT_NE(*copy, id);
            EXPECT_EQ(scene.getNodeById(*copy)->type, scene.getNodeById(id)->type);
        }
        EXPECT_EQ(scene.getNodeById(owner)->children.size(), 6);
    }

    TEST_F(SplatCropWorkflowTest, RepeatedCropsBeyondSixteenRemainIndependentAndUndoable) {
        auto& scene = sm->getScene();
        const auto uuid = scene.getNodeById(box)->uuid;
        for (int i = 0; i < 40; ++i)
            ASSERT_TRUE(cap::duplicateSplatCrop(*sm, nullptr, uuid));
        EXPECT_EQ(scene.getNodeById(owner)->children.size(), 41);
        ASSERT_TRUE(op::undoHistory().undo().success);
        EXPECT_EQ(scene.getNodeById(owner)->children.size(), 40);
        ASSERT_TRUE(op::undoHistory().redo().success);
        EXPECT_EQ(scene.getNodeById(owner)->children.size(), 41);
        EXPECT_NE(scene.getNodeByUuid(uuid), nullptr);
    }

    TEST_F(SplatCropWorkflowTest, ShapeSwitchingAndPolygonCreationPreserveOtherInstances) {
        auto& scene = sm->getScene();
        const auto copy = cap::duplicateSplatCrop(*sm, nullptr, scene.getNodeById(box)->uuid);
        ASSERT_TRUE(copy);
        select(*copy);
        const auto converted = cap::ensureEllipsoid(*sm, nullptr, owner);
        ASSERT_TRUE(converted);
        EXPECT_EQ(*converted, *copy);
        EXPECT_NE(scene.getNodeById(box)->cropbox, nullptr);
        core::CropPolygonData data;
        data.vertices = {{-1, -1}, {1, -1}, {1, 1}, {-1, 1}};
        const auto polygon = cap::createCropPolygon(*sm, nullptr, owner, data, glm::mat4(1));
        ASSERT_TRUE(polygon);
        EXPECT_NE(scene.getNodeById(box), nullptr);
        EXPECT_NE(scene.getNodeById(*copy)->ellipsoid, nullptr);
        EXPECT_EQ(scene.getNodeById(owner)->children.size(), 3);
    }

    TEST_F(SplatCropWorkflowTest, LivePreviewReplacesOnlyTheSelectedCropInstance) {
        auto& scene = sm->getScene();
        scene.getNodeById(box)->cropbox->enabled = true;
        const auto copy = cap::duplicateSplatCrop(*sm, nullptr, scene.getNodeById(box)->uuid);
        ASSERT_TRUE(copy);
        select(*copy);
        Viewport viewport(640, 480);
        RenderSettings settings;
        FrameContext ctx{.viewport = viewport, .scene_manager = sm.get(), .settings = settings, .render_size = {640, 480}};
        ctx.scene_state.cropboxes = scene.getRenderableCropBoxes();
        std::erase_if(ctx.scene_state.cropboxes, [&](const auto& r) { return r.parent_splat_id != owner; });
        for (auto& r : ctx.scene_state.cropboxes)
            r.parent_node_index = 0;
        ctx.gizmo.cropbox_active = true;
        ctx.gizmo.cropbox_affects_render = true;
        ctx.gizmo.cropbox_parent_node_index = 0;
        ctx.gizmo.cropbox_min = {-2, -2, -2};
        ctx.gizmo.cropbox_max = {2, 2, 2};
        ctx.gizmo.cropbox_transform = glm::translate(glm::mat4(1), glm::vec3(4, 0, 0));
        const auto request = buildViewportRenderRequest(ctx, {640, 480});
        ASSERT_EQ(request.filters.crop_regions.size(), 2);
        const auto& filters = request.filters.crop_regions;
        const auto source = std::find_if(filters.begin(), filters.end(), [&](const auto& r) { return r.source_node_id == box; });
        const auto preview = std::find_if(filters.begin(), filters.end(), [&](const auto& r) { return r.source_node_id == *copy; });
        ASSERT_NE(source, filters.end());
        ASSERT_NE(preview, filters.end());
        EXPECT_EQ(source->bounds.min, scene.getNodeById(box)->cropbox->min);
        EXPECT_EQ(preview->bounds.min, ctx.gizmo.cropbox_min);
        EXPECT_EQ(preview->bounds.transform, glm::inverse(ctx.gizmo.cropbox_transform));
        scene.getNodeById(*copy)->cropbox->enabled = false;
        EXPECT_EQ(buildViewportRenderRequest(ctx, {640, 480}).filters.crop_regions.size(), 1);
        ASSERT_TRUE(sm->setSplatCropGroupEnabled(scene.getNodeById(owner)->uuid, false));
        ctx.scene_state.cropboxes = scene.getRenderableCropBoxes();
        std::erase_if(ctx.scene_state.cropboxes, [&](const auto& r) { return r.parent_splat_id != owner; });
        for (auto& r : ctx.scene_state.cropboxes)
            r.parent_node_index = 0;
        // A stale live-preview flag must not recreate a suspended persistent filter.
        const auto suspended = buildViewportRenderRequest(ctx, {640, 480});
        EXPECT_TRUE(suspended.filters.crop_regions.empty());
        EXPECT_FALSE(suspended.filters.crop_region.has_value());
        EXPECT_TRUE(scene.getNodeById(box)->cropbox->enabled);
    }

    TEST_F(SplatCropWorkflowTest, VideoStateKeepsAllEllipsoidsAndAcceptsLegacySnapshots) {
        gui::VideoExportSceneSnapshot snapshot;
        snapshot.ellipsoids = {{.node_id = 4, .parent_node_index = 0}, {.node_id = 5, .parent_node_index = 0}};
        const auto state = gui::makeVideoExportGaussianSceneState(snapshot);
        ASSERT_EQ(state.ellipsoids.size(), 2);
        EXPECT_EQ(state.ellipsoids[0].node_id, 4);
        EXPECT_EQ(state.ellipsoids[1].node_id, 5);
        snapshot.active_ellipsoid = snapshot.ellipsoids[0];
        snapshot.ellipsoids.clear();
        EXPECT_EQ(gui::makeVideoExportGaussianSceneState(snapshot).ellipsoids.size(), 1);
    }

    TEST_F(SplatCropWorkflowTest, SelectedCropCannotRenderOrHitTestHandlesWithoutExplicitEditing) {
        auto& scene = sm->getScene();
        const auto ellipsoid = scene.addEllipsoid("ellipsoid", owner);
        core::CropPolygonData polygon_data;
        polygon_data.vertices = {{-1, -1}, {1, -1}, {1, 1}, {-1, 1}};
        const auto polygon = scene.addCropPolygon("polygon", owner, polygon_data);
        gui::GizmoManager gizmo(nullptr);
        gui::UIContext no_render_context;
        gui::ViewportLayout viewport;
        struct ResetTool {
            ~ResetTool() { UnifiedToolRegistry::instance().setActiveTool(""); }
        } reset;
        for (const auto id : {box, ellipsoid, polygon}) {
            select(id);
            const auto before = scene.getNodeById(id)->local_transform.get();
            for (const auto* tool : {"", "builtin.select", "builtin.translate", "builtin.rotate", "builtin.scale"}) {
                UnifiedToolRegistry::instance().setActiveTool(tool);
                // No viewport, GPU or viewer is needed: an inactive crop editor
                // must stop before reaching any interactive render/hit-test path.
                gizmo.renderCropBoxGizmo(no_render_context, viewport);
                gizmo.renderEllipsoidGizmo(no_render_context, viewport);
                EXPECT_EQ(scene.getNodeById(id)->local_transform.get(), before);
            }
        }
    }

    TEST_F(SplatCropWorkflowTest, ShapeConversionsKeepExactIdentityBoundsFlagsAndSiblingsAcrossEveryPair) {
        auto& scene = sm->getScene();
        const auto owner_uuid = scene.getNodeById(owner)->uuid;
        const auto sibling_uuid = scene.getNodeById(box)->uuid;
        scene.setNodeTransform(scene.getNodeById(owner)->name,
                               glm::scale(glm::rotate(glm::translate(glm::mat4(1), glm::vec3(10, -3, 7)), .3f, glm::vec3(0, 1, 0)), glm::vec3(2, 3, 4)));
        for (const auto from : {core::NodeType::CROPBOX, core::NodeType::ELLIPSOID, core::NodeType::CROP_POLYGON}) {
            for (const auto to : {core::NodeType::CROPBOX, core::NodeType::ELLIPSOID, core::NodeType::CROP_POLYGON}) {
                const auto created = cap::createSplatCrop(*sm, nullptr, owner_uuid, core::NodeType::CROPBOX);
                ASSERT_TRUE(created);
                const auto uuid = scene.getNodeById(*created)->uuid;
                ASSERT_TRUE(cap::convertSplatCrop(*sm, nullptr, uuid, from));
                const auto name = scene.getNodeByUuid(uuid)->name;
                const auto* descriptor = core::splatOperationDescriptor(from);
                op::OperatorProperties flags;
                flags.set("node_uuid", uuid.to_string());
                flags.set("enabled", false);
                flags.set("inverse", true);
                ASSERT_TRUE(op::operators().invoke(std::string(descriptor->set_operator), &flags).is_finished());
                const glm::mat4 local = glm::rotate(glm::translate(glm::mat4(1), glm::vec3(3, -4, 5)), .4f, glm::vec3(0, 0, 1));
                scene.setNodeTransform(name, local);
                const auto world = scene.getWorldTransform(*created);
                const auto node_count = scene.getNodeCount();
                select(other_box); // Conversion must never follow an unrelated selection.
                op::undoHistory().clear();
                ASSERT_TRUE(cap::convertSplatCrop(*sm, nullptr, uuid, to));
                const auto* converted = scene.getNodeByUuid(uuid);
                ASSERT_NE(converted, nullptr);
                EXPECT_EQ(converted->id, *created);
                EXPECT_EQ(converted->parent_id, owner);
                EXPECT_EQ(converted->name, name);
                EXPECT_EQ(converted->type, to);
                EXPECT_EQ(scene.getNodeCount(), node_count);
                EXPECT_EQ(scene.getWorldTransform(converted->id), world);
                EXPECT_EQ(scene.getNodeByUuid(sibling_uuid)->type, core::NodeType::CROPBOX);
                const auto snapshot = core::readSplatOperation(*scene.getNodeById(owner), *converted);
                ASSERT_TRUE(snapshot);
                EXPECT_EQ(snapshot->enabled, false);
                EXPECT_TRUE(std::visit([](const auto& p) { if constexpr (requires { p.inverse; }) return p.inverse; else return false; }, snapshot->parameters));
                if (from != to) {
                    ASSERT_TRUE(op::undoHistory().undo().success);
                    ASSERT_NE(scene.getNodeByUuid(uuid), nullptr);
                    EXPECT_EQ(scene.getNodeByUuid(uuid)->type, from);
                    ASSERT_TRUE(op::undoHistory().redo().success);
                    EXPECT_EQ(scene.getNodeByUuid(uuid)->type, to);
                } else {
                    EXPECT_EQ(op::undoHistory().undoCount(), 0);
                }
                scene.removeNode(name, false);
                op::undoHistory().clear();
            }
        }
    }

    TEST_F(SplatCropWorkflowTest, ShapeConversionRejectsLockedOrMissingVolumesWithoutMutation) {
        auto& scene = sm->getScene();
        const auto uuid = scene.getNodeById(box)->uuid;
        scene.setNodeLocked(scene.getNodeById(owner)->name, true);
        op::undoHistory().clear();
        EXPECT_FALSE(cap::convertSplatCrop(*sm, nullptr, uuid, core::NodeType::CROP_POLYGON));
        EXPECT_EQ(scene.getNodeById(box)->type, core::NodeType::CROPBOX);
        EXPECT_EQ(op::undoHistory().undoCount(), 0);
        EXPECT_FALSE(cap::convertSplatCrop(*sm, nullptr, core::generate_uuid_v4(), core::NodeType::ELLIPSOID));
    }

    TEST_F(SplatCropWorkflowTest, RemovingACropKeepsItsOwnerAndSiblingsAndCanBeUndone) {
        auto& scene = sm->getScene();
        const auto owner_uuid = scene.getNodeById(owner)->uuid;
        const auto sibling_uuid = scene.getNodeById(box)->uuid;
        for (const auto type : {core::NodeType::CROPBOX, core::NodeType::ELLIPSOID, core::NodeType::CROP_POLYGON}) {
            const auto created = cap::createSplatCrop(*sm, nullptr, owner_uuid, core::NodeType::CROPBOX);
            ASSERT_TRUE(created);
            const auto uuid = scene.getNodeById(*created)->uuid;
            ASSERT_TRUE(cap::convertSplatCrop(*sm, nullptr, uuid, type));
            select(*created);
            op::undoHistory().clear();
            ASSERT_TRUE(sm->removeNodesByIdsWithResult({*created}, false));
            EXPECT_EQ(scene.getNodeByUuid(uuid), nullptr);
            EXPECT_NE(scene.getNodeByUuid(owner_uuid), nullptr);
            EXPECT_NE(scene.getNodeByUuid(sibling_uuid), nullptr);
            ASSERT_TRUE(op::undoHistory().undo().success);
            ASSERT_NE(scene.getNodeByUuid(uuid), nullptr);
            EXPECT_EQ(scene.getNodeByUuid(uuid)->type, type);
            ASSERT_TRUE(op::undoHistory().redo().success);
            EXPECT_EQ(scene.getNodeByUuid(uuid), nullptr);
        }
    }

    TEST_F(SplatCropWorkflowTest, OffCenterBoxConversionPreservesWorldPlacementAndUndoRestoresBounds) {
        auto& scene = sm->getScene();
        const auto owner_uuid = scene.getNodeById(owner)->uuid;
        for (const auto type : {core::NodeType::ELLIPSOID, core::NodeType::CROP_POLYGON}) {
            const auto created = cap::createSplatCrop(*sm, nullptr, owner_uuid, core::NodeType::CROPBOX);
            ASSERT_TRUE(created);
            const auto uuid = scene.getNodeById(*created)->uuid;
            auto data = *scene.getNodeByUuid(uuid)->cropbox;
            data.min = {1, 2, 3};
            data.max = {5, 8, 11};
            scene.setCropBoxData(*created, data);
            const glm::mat4 original = glm::rotate(glm::translate(glm::mat4(1), glm::vec3(4, 5, 6)), .5f, glm::vec3(0, 1, 0));
            scene.setNodeTransform(scene.getNodeByUuid(uuid)->name, original);
            const auto expected = original * glm::translate(glm::mat4(1), glm::vec3(3, 5, 7));
            op::undoHistory().clear();
            ASSERT_TRUE(cap::convertSplatCrop(*sm, nullptr, uuid, type));
            EXPECT_EQ(scene.getNodeByUuid(uuid)->local_transform.get(), expected);
            if (type == core::NodeType::ELLIPSOID)
                EXPECT_EQ(scene.getNodeByUuid(uuid)->ellipsoid->radii, glm::vec3(2, 3, 4));
            else
                EXPECT_FLOAT_EQ(scene.getNodeByUuid(uuid)->crop_polygon->half_depth, 4);
            ASSERT_TRUE(op::undoHistory().undo().success);
            EXPECT_EQ(scene.getNodeByUuid(uuid)->local_transform.get(), original);
            EXPECT_EQ(scene.getNodeByUuid(uuid)->cropbox->min, data.min);
            EXPECT_EQ(scene.getNodeByUuid(uuid)->cropbox->max, data.max);
            scene.removeNode(scene.getNodeByUuid(uuid)->name, false);
        }
    }

    TEST_F(SplatCropWorkflowTest, ConversionBeyondSixteenPreservesIdentityAndUndo) {
        auto& scene = sm->getScene();
        for (int i = 0; i < 40; ++i)
            ASSERT_TRUE(cap::createSplatCrop(*sm, nullptr, scene.getNodeById(owner)->uuid, core::NodeType::ELLIPSOID));
        op::undoHistory().clear();
        const auto uuid = scene.getNodeById(box)->uuid;
        ASSERT_TRUE(cap::convertSplatCrop(*sm, nullptr, uuid, core::NodeType::ELLIPSOID));
        EXPECT_EQ(scene.getNodeByUuid(uuid)->type, core::NodeType::ELLIPSOID);
        ASSERT_TRUE(op::undoHistory().undo().success);
        EXPECT_EQ(scene.getNodeByUuid(uuid)->type, core::NodeType::CROPBOX);
        ASSERT_TRUE(op::undoHistory().redo().success);
        EXPECT_EQ(scene.getNodeByUuid(uuid)->type, core::NodeType::ELLIPSOID);
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

    class SplatCropTreeTest : public SplatCropWorkflowTest {
    protected:
        TreeRenderer renderer;
        Rml::Context* context = nullptr;
        gui::SceneGraphElement* tree = nullptr;
        uint64_t generation = 1;
        void SetUp() override {
            SplatCropWorkflowTest::SetUp();
            ASSERT_TRUE(Rml::Initialise());
            ASSERT_TRUE(Rml::LoadFontFace((std::filesystem::path(PROJECT_ROOT_PATH) / "src/visualizer/gui/assets/fonts/Inter-Regular.ttf").generic_string()));
            context = Rml::CreateContext("crop-tree-test", {500, 700}, &renderer);
            ASSERT_NE(context, nullptr);
            auto* doc = context->LoadDocumentFromMemory("<rml><head><style>body{font-family:Inter;font-size:12px;} scene-graph{display:block;width:480px;height:600px;}</style></head><body/></rml>");
            ASSERT_NE(doc, nullptr);
            Rml::ElementPtr element(new gui::SceneGraphElement("scene-graph"));
            tree = static_cast<gui::SceneGraphElement*>(element.get());
            doc->AppendChild(std::move(element));
            doc->Show();
            sync();
            for (const auto id : {owner, other_owner}) {
                auto* group = tree->QuerySelector(".crop-group[data-crop-owner='" + std::to_string(id) + "']");
                ASSERT_NE(group, nullptr);
                group->QuerySelector(".expand-toggle")->DispatchEvent("click", {});
                sync();
            }
        }
        void sync() {
            gui::PanelDrawContext ctx;
            ctx.scene = &sm->getScene();
            ctx.scene_generation = generation++;
            (void)tree->syncFromScene(ctx);
            context->Update();
        }
        Rml::Element* toggle(core::NodeId id) {
            return tree->QuerySelector(".crop-effect-toggle[data-node-id='" + std::to_string(id) + "']");
        }
        void TearDown() override {
            if (context)
                Rml::RemoveContext("crop-tree-test");
            Rml::Shutdown();
            SplatCropWorkflowTest::TearDown();
        }
    };

    TEST_F(SplatCropTreeTest, RowSwitchTogglesEffectWithoutChangingHelperVisibilityAndSupportsUndo) {
        auto& scene = sm->getScene();
        auto* control = toggle(box);
        ASSERT_NE(control, nullptr);
        const bool enabled = scene.getNodeById(box)->cropbox->enabled;
        const bool visible = scene.getNodeById(box)->visible;
        auto* dot = control->QuerySelector(".crop-effect-dot");
        ASSERT_NE(dot, nullptr);
        dot->DispatchEvent("click", {});
        EXPECT_EQ(scene.getNodeById(box)->cropbox->enabled, !enabled);
        EXPECT_EQ(static_cast<bool>(scene.getNodeById(box)->visible), visible);
        ASSERT_TRUE(op::undoHistory().undo().success);
        EXPECT_EQ(scene.getNodeById(box)->cropbox->enabled, enabled);
    }

    TEST_F(SplatCropTreeTest, LockedCropSwitchDoesNotMutateState) {
        auto& scene = sm->getScene();
        scene.setNodeLocked(scene.getNodeById(owner)->name, true);
        sync();
        const bool enabled = scene.getNodeById(box)->cropbox->enabled;
        auto* control = toggle(box);
        ASSERT_NE(control, nullptr);
        control->DispatchEvent("click", {});
        EXPECT_EQ(scene.getNodeById(box)->cropbox->enabled, enabled);
    }

    TEST_F(SplatCropTreeTest, AllCropShapesExposeDeleteInTheGraphAndRespectInheritedLocks) {
        auto& scene = sm->getScene();
        const auto uuid = scene.getNodeById(box)->uuid;
        for (const auto type : {core::NodeType::CROPBOX, core::NodeType::ELLIPSOID, core::NodeType::CROP_POLYGON}) {
            ASSERT_TRUE(cap::convertSplatCrop(*sm, nullptr, uuid, type));
            sync();
            auto* trash = tree->QuerySelector(".trash-icon[data-node-id='" + std::to_string(box) + "']");
            ASSERT_NE(trash, nullptr);
            EXPECT_NE(trash->GetComputedValues().display(), Rml::Style::Display::None);
            EXPECT_FALSE(trash->IsClassSet("trash-disabled"));
            scene.setNodeLocked(scene.getNodeById(owner)->name, true);
            sync();
            EXPECT_TRUE(trash->IsClassSet("trash-disabled"));
            scene.setNodeLocked(scene.getNodeById(owner)->name, false);
        }
    }

    TEST_F(SplatCropTreeTest, PointCloudCropEyeRetainsLegacyVisibilityWithoutTogglingEffect) {
        auto& scene = sm->getScene();
        core::Scene::RestoreNodeDesc desc;
        desc.uuid = core::generate_uuid_v4();
        desc.name = "Cloud";
        desc.type = core::NodeType::POINTCLOUD;
        const auto cloud = scene.restoreNodeWithUuid(std::move(desc));
        core::CropPolygonData data;
        data.vertices = {{-1, -1}, {1, -1}, {1, 1}, {-1, 1}};
        for (const auto id : {scene.addCropBox("Cloud box", cloud), scene.addEllipsoid("Cloud ellipsoid", cloud),
                              scene.addCropPolygon("Cloud polygon", cloud, data)}) {
            ASSERT_NE(id, core::NULL_NODE);
            sync();
            auto* eye = tree->QuerySelector(".row-icon[data-action='toggle-vis'][data-node-id='" + std::to_string(id) + "']");
            ASSERT_NE(eye, nullptr);
            EXPECT_NE(eye->GetComputedValues().display(), Rml::Style::Display::None);
            const auto enabled = [&] {
                const auto* node = scene.getNodeById(id);
                return node->cropbox ? node->cropbox->enabled : node->ellipsoid ? node->ellipsoid->enabled
                                                                                : node->crop_polygon->enabled;
            };
            const bool visible = scene.getNodeById(id)->visible;
            const bool effect = enabled();
            eye->DispatchEvent("click", {});
            EXPECT_EQ(static_cast<bool>(scene.getNodeById(id)->visible), !visible);
            EXPECT_EQ(enabled(), effect);
        }
    }

    TEST(DynamicCropParamsTest, CustomShaderTailKeepsItsFixedAddressAndRelocatesCropDirectory) {
        namespace ov = vksplat_overlay;
        lfs::rendering::ViewportRenderRequest request{};
        request.filters.crop_regions.push_back({.parent_node_index = 0});
        const std::vector<float> tail{11, 12, 13, 14, 21, 22, 23, 24};
        const auto params = ov::buildParams(request, false, false, false, 0, false, tail);
        ASSERT_FALSE(params.empty());
        EXPECT_TRUE(std::equal(tail.begin(), tail.end(), params.begin() + ov::ParamCount * 4));
        const size_t directory = ov::ParamCount + tail.size() / 4;
        EXPECT_EQ(params[ov::CropDirectory * 4], directory);
        EXPECT_EQ(params[ov::CropDirectory * 4 + 1], 1);
        EXPECT_EQ(params[directory * 4], 0);
        EXPECT_EQ(params[directory * 4 + 1], directory + ov::CropDirectoryStride);
        EXPECT_EQ(params.size() / 4, directory + ov::CropDirectoryStride + ov::CropParamStride);
        EXPECT_TRUE(ov::buildParams(request, false, false, false, 0, false, std::span<const float>(tail.data(), 3)).empty());
    }

    TEST(DynamicCropParamsTest, SparseOwnersAndAllShapesPackWithoutTruncationOrCrossOwnerRanges) {
        namespace ov = vksplat_overlay;
        lfs::rendering::ViewportRenderRequest request{};
        // Sparse owner indices do not allocate empty owner slots.
        for (const int parent : {-1, 0, 7, 1000000}) {
            for (int i = 0; i < 40; ++i) {
                request.filters.crop_regions.push_back({.bounds = {.min = glm::vec3(-1), .max = glm::vec3(1), .transform = glm::mat4(1)}, .inverse = i == 39, .parent_node_index = parent});
                request.filters.ellipsoid_regions.push_back({.bounds = {.radii = glm::vec3(1), .transform = glm::mat4(1)}, .desaturate = i == 39, .parent_node_index = parent});
                request.filters.crop_polygon_regions.push_back({.bounds = {.vertices = {{-1, -1}, {1, -1}, {1, 1}, {-1, 1}}, .half_depth = 0.5f, .transform = glm::mat4(1)}, .parent_node_index = parent});
            }
        }
        auto params = ov::buildParams(request, false, false, true, 0, false);
        ASSERT_EQ(params.size(), (ov::ParamCount + 4 * ov::CropDirectoryStride + 160 * (ov::CropParamStride + ov::EllipsoidParamStride + ov::CropPolygonParamStride)) * 4u);
        EXPECT_EQ(params[ov::CropDirectory * 4 + 1], 4.0f);
        const auto slot = [&](size_t index) { return glm::vec4(params[index * 4], params[index * 4 + 1], params[index * 4 + 2], params[index * 4 + 3]); };
        size_t directory = ov::ParamCount;
        for (const int parent : {-1, 0, 7, 1000000}) {
            const auto first = slot(directory), second = slot(directory + 1);
            EXPECT_EQ(first.x, static_cast<float>(parent));
            EXPECT_EQ(first.z, 40.0f);
            EXPECT_EQ(second.x, 40.0f);
            EXPECT_EQ(second.z, 40.0f);
            const auto boxes = static_cast<size_t>(first.y), ellipsoids = static_cast<size_t>(first.w), polygons = static_cast<size_t>(second.y);
            EXPECT_EQ(slot(boxes + 39 * ov::CropParamStride), glm::vec4(1, 1, 0, parent));
            EXPECT_EQ(slot(ellipsoids + 39 * ov::EllipsoidParamStride), glm::vec4(1, 0, 1, parent));
            EXPECT_EQ(slot(polygons + 39 * ov::CropPolygonParamStride), glm::vec4(1, 0, 0, parent));
            EXPECT_LE((polygons + 40 * ov::CropPolygonParamStride) * 4, params.size());
            directory += ov::CropDirectoryStride;
        }
        request.filters = {};
        params = ov::buildParams(request, false, false, false, 0, false);
        ASSERT_EQ(params.size(), ov::ParamCount * 4u);
        EXPECT_EQ(params[ov::CropDirectory * 4 + 1], 0.0f);
        request.filters.crop_region = lfs::rendering::GaussianScopedBoxFilter{.bounds = {.min = glm::vec3(-2), .max = glm::vec3(2), .transform = glm::mat4(1)}, .inverse = true, .parent_node_index = -1};
        params = ov::buildParams(request, false, false, false, 0, false);
        EXPECT_EQ(slot(ov::CropFlags), glm::vec4(1, 1, 0, -1));
        EXPECT_EQ(params.size(), ov::ParamCount * 4u);
    }

    TEST(DynamicCropParamsTest, PolygonPayloadPreservesOddAndMaximumOutlinesAndTransforms) {
        namespace ov = vksplat_overlay;
        lfs::rendering::ViewportRenderRequest request{};
        const glm::mat4 transform = glm::scale(glm::translate(glm::mat4(1), glm::vec3(1, -2, 3)), glm::vec3(2, 3, 4));
        std::vector<glm::vec2> full(core::kMaxCropPolygonVertices);
        for (size_t i = 0; i < full.size(); ++i)
            full[i] = {std::cos(0.1f * static_cast<float>(i)), std::sin(0.1f * static_cast<float>(i))};
        request.filters.crop_polygon_regions = {
            {.bounds = {.vertices = {{0, 0}, {1, 0}, {1, 1}, {0.5f, 2}, {0, 1}}, .half_depth = 0.5f, .transform = transform}, .inverse = true, .parent_node_index = 3},
            {.bounds = {.vertices = {{0, 0}, {1, 0}}, .half_depth = 1, .transform = glm::mat4(1)}, .parent_node_index = 1},
            {.bounds = {.vertices = full, .half_depth = 2, .transform = transform}, .parent_node_index = 2}};
        const auto params = ov::buildParams(request, false, false, false, 0, false);
        ASSERT_EQ(params.size(), (ov::ParamCount + 2 * ov::CropDirectoryStride + 2 * ov::CropPolygonParamStride) * 4u);
        const auto slot = [&](size_t index) { return glm::vec4(params[index * 4], params[index * 4 + 1], params[index * 4 + 2], params[index * 4 + 3]); };
        const auto max_outline = static_cast<size_t>(slot(ov::ParamCount + 1).y);
        const auto odd_outline = static_cast<size_t>(slot(ov::ParamCount + ov::CropDirectoryStride + 1).y);
        EXPECT_EQ(slot(max_outline + ov::CropPolygonParams).y, static_cast<float>(full.size()));
        EXPECT_EQ(slot(max_outline + ov::CropPolygonParamStride - 1), glm::vec4(full[62], full[63]));
        EXPECT_EQ(slot(odd_outline), glm::vec4(1, 1, 0, 3));
        EXPECT_EQ(slot(odd_outline + ov::CropPolygonVertices + 2), glm::vec4(0, 1, 0, 0));
        for (int row = 0; row < 4; ++row)
            EXPECT_EQ(slot(odd_outline + ov::CropPolygonTransform + row), glm::vec4(transform[0][row], transform[1][row], transform[2][row], transform[3][row]));
    }

    TEST(DynamicCropParamsTest, UnrepresentableOwnerIndexFailsInsteadOfTargetingAnotherObject) {
        lfs::rendering::ViewportRenderRequest request{};
        request.filters.crop_regions.push_back({.parent_node_index = std::numeric_limits<int>::max()});
        EXPECT_TRUE(vksplat_overlay::buildParams(request, false, false, false, 0, false).empty());
    }

    // Clipboard tensor hydration uses CUDA. Keep the focused default suite CPU-only;
    // explicitly opt in on a GPU validation host.
    TEST_F(SplatCropWorkflowTest, ClipboardGpuPreservesEveryRepeatedChildAndSupportsUndo) {
        if (!std::getenv("LFS_TEST_CROP_CLIPBOARD_GPU"))
            GTEST_SKIP() << "Set LFS_TEST_CROP_CLIPBOARD_GPU=1 for the GPU clipboard regression";
        using core::Device;
        using core::Tensor;
        auto data = std::make_unique<core::SplatData>(
            0, Tensor::from_vector({0.0f, 0.0f, 2.0f}, {size_t{1}, size_t{3}}, Device::CPU),
            Tensor::from_vector({1.0f, 1.0f, 1.0f}, {size_t{1}, size_t{1}, size_t{3}}, Device::CPU),
            Tensor::zeros({size_t{1}, size_t{0}, size_t{3}}, Device::CPU, core::DataType::Float32),
            Tensor::from_vector({0.0f, 0.0f, 0.0f}, {size_t{1}, size_t{3}}, Device::CPU),
            Tensor::from_vector({1.0f, 0.0f, 0.0f, 0.0f}, {size_t{1}, size_t{4}}, Device::CPU),
            Tensor::from_vector({8.0f}, {size_t{1}, size_t{1}}, Device::CPU), 1.0f);
        auto& scene = sm->getScene();
        const auto source = scene.addSplat("Clipboard source", std::move(data));
        const auto uuid = scene.getNodeById(source)->uuid;
        for (int i = 0; i < 40; ++i) {
            ASSERT_TRUE(cap::createSplatCrop(*sm, nullptr, uuid, core::NodeType::CROPBOX));
            ASSERT_TRUE(cap::createSplatCrop(*sm, nullptr, uuid, core::NodeType::ELLIPSOID));
            core::CropPolygonData polygon;
            polygon.vertices = {{-1, -1}, {1, -1}, {1, 1}, {-1, 1}};
            ASSERT_TRUE(cap::createCropPolygon(*sm, nullptr, source, polygon, glm::mat4(1), true));
        }
        ASSERT_TRUE(sm->setSplatCropGroupEnabled(uuid, false));
        const auto duplicated = sm->duplicateNodeTree(source);
        ASSERT_FALSE(duplicated.empty());
        EXPECT_FALSE(scene.getNode(duplicated)->crop_effects_enabled);
        EXPECT_EQ(scene.getNode(duplicated)->children.size(), 120);
        select(source);
        ASSERT_TRUE(sm->copySelectedNodes());
        const auto pasted = sm->pasteNodes();
        ASSERT_EQ(pasted.size(), 1);
        const auto* copy = scene.getNode(pasted[0]);
        ASSERT_NE(copy, nullptr);
        EXPECT_NE(copy->uuid, uuid);
        EXPECT_FALSE(copy->crop_effects_enabled);
        ASSERT_EQ(copy->children.size(), 120);
        size_t boxes = 0, ellipsoids = 0, polygons = 0;
        for (const auto id : copy->children) {
            const auto* child = scene.getNodeById(id);
            ASSERT_NE(child, nullptr);
            EXPECT_EQ(child->parent_id, copy->id);
            EXPECT_NE(child->uuid, core::Uuid{});
            boxes += child->type == core::NodeType::CROPBOX;
            ellipsoids += child->type == core::NodeType::ELLIPSOID;
            polygons += child->type == core::NodeType::CROP_POLYGON;
        }
        EXPECT_EQ(boxes, 40);
        EXPECT_EQ(ellipsoids, 40);
        EXPECT_EQ(polygons, 40);
        const auto copy_uuid = copy->uuid;
        ASSERT_TRUE(op::undoHistory().undo().success);
        EXPECT_EQ(scene.getNodeByUuid(copy_uuid), nullptr);
        ASSERT_TRUE(op::undoHistory().redo().success);
        ASSERT_NE(scene.getNodeByUuid(copy_uuid), nullptr);
        EXPECT_EQ(scene.getNodeByUuid(copy_uuid)->children.size(), 120);
    }

    TEST_F(SplatCropWorkflowTest, SelectionGpuUsesEveryCropOnlyOnItsOwnerAndHonorsCollectiveGate) {
        if (!std::getenv("LFS_TEST_CROP_CLIPBOARD_GPU"))
            GTEST_SKIP() << "Set LFS_TEST_CROP_CLIPBOARD_GPU=1 for CUDA selection checks";
        auto& scene = sm->getScene();
        const auto first = scene.addSplat("Selection first", groupTestSplat(core::Device::CUDA));
        const auto second = scene.addSplat("Selection second", groupTestSplat(core::Device::CUDA));
        const auto first_uuid = scene.getNodeById(first)->uuid;
        const auto crop = scene.addCropBox("Selection bounds", first);
        scene.setNodeTransform("Selection bounds", glm::mat4(1));
        scene.getCropBoxData(crop)->min = {-1, -1, 1};
        scene.getCropBoxData(crop)->max = {1, 1, 3};
        scene.getCropBoxData(crop)->enabled = true;
        const auto duplicate = cap::duplicateSplatCrop(*sm, nullptr, scene.getNodeById(crop)->uuid);
        ASSERT_TRUE(duplicate);
        scene.getCropBoxData(*duplicate)->inverse = true;
        SelectionService selection(sm.get(), nullptr);
        const auto count = [&] {
            selection.beginStroke();
            auto* mask = selection.getStrokeSelection();
            if (!mask)
                return int64_t{-1};
            mask->fill_(true);
            selection.applyCropFilterToStroke();
            const auto result = static_cast<int64_t>(mask->sum_scalar());
            selection.cancelStroke();
            return result;
        };
        for (const auto selected : {first, second, core::NULL_NODE}) {
            if (selected == core::NULL_NODE)
                sm->clearSelection();
            else
                select(selected);
            EXPECT_EQ(count(), 4); // First owner's intersection is empty; the second is untouched.
        }
        scene.getCropBoxData(*duplicate)->enabled = false;
        EXPECT_EQ(count(), 6);
        ASSERT_TRUE(sm->setSplatCropGroupEnabled(first_uuid, false));
        EXPECT_EQ(count(), 8);
        ASSERT_TRUE(sm->setSplatCropGroupEnabled(first_uuid, true));
        EXPECT_EQ(count(), 6);
        scene.getCropBoxData(crop)->enabled = false;
        const auto ellipse = scene.addEllipsoid("Selection ellipse", first);
        scene.getEllipsoidData(ellipse)->radii = glm::vec3(10000);
        scene.getEllipsoidData(ellipse)->inverse = true;
        scene.getEllipsoidData(ellipse)->enabled = true;
        EXPECT_EQ(count(), 4);
        scene.getEllipsoidData(ellipse)->enabled = false;
        core::CropPolygonData polygon;
        polygon.vertices = {{-100, -100}, {100, -100}, {100, 100}, {-100, 100}};
        polygon.half_depth = 100;
        polygon.inverse = true;
        polygon.enabled = true;
        scene.addCropPolygon("Selection polygon", first, polygon);
        EXPECT_EQ(count(), 4);
        ASSERT_TRUE(sm->setSplatCropGroupEnabled(first_uuid, false));
        EXPECT_EQ(count(), 8);
    }

    TEST_F(SplatCropWorkflowTest, PointPreviewDoesNotReintroduceSuspendedOrIndividuallyDisabledCrop) {
        auto& scene = sm->getScene();
        select(box);
        Viewport viewport(640, 480);
        RenderSettings settings;
        FrameContext ctx{.viewport = viewport, .scene_manager = sm.get(), .settings = settings, .render_size = {640, 480}};
        ctx.gizmo.cropbox_active = true;
        ctx.gizmo.cropbox_affects_render = true;
        ctx.gizmo.cropbox_parent_node_index = 0;
        const auto update = [&] {
            ctx.scene_state.cropboxes = scene.getRenderableCropBoxes();
            for (auto& volume : ctx.scene_state.cropboxes)
                volume.parent_node_index = volume.parent_splat_id == owner ? 0 : 1;
        };
        scene.getCropBoxData(box)->enabled = true;
        update();
        EXPECT_TRUE(buildPointCloudRenderRequest(ctx, {640, 480}, {}).filters.crop_box.has_value());
        ASSERT_TRUE(sm->setSplatCropGroupEnabled(scene.getNodeById(owner)->uuid, false));
        update();
        EXPECT_FALSE(buildPointCloudRenderRequest(ctx, {640, 480}, {}).filters.crop_box.has_value());
        ASSERT_TRUE(sm->setSplatCropGroupEnabled(scene.getNodeById(owner)->uuid, true));
        scene.getCropBoxData(box)->enabled = false;
        update();
        EXPECT_FALSE(buildPointCloudRenderRequest(ctx, {640, 480}, {}).filters.crop_box.has_value());
    }

    TEST_F(SplatCropWorkflowTest, CollectiveGatePreservesMixedFlagsAndUndoRestoresRendering) {
        auto& scene = sm->getScene();
        const auto uuid = scene.getNodeById(owner)->uuid;
        scene.getNodeById(box)->cropbox->enabled = true;
        const auto ellipse = scene.addEllipsoid("Disabled ellipse", owner);
        scene.getNodeById(ellipse)->ellipsoid->enabled = false;
        ASSERT_TRUE(sm->setSplatCropGroupEnabled(uuid, false));
        EXPECT_FALSE(scene.getNodeByUuid(uuid)->crop_effects_enabled);
        EXPECT_TRUE(scene.getNodeById(box)->cropbox->enabled);
        EXPECT_FALSE(scene.getNodeById(ellipse)->ellipsoid->enabled);
        for (const auto& entry : scene.getRenderableCropBoxes())
            if (entry.parent_splat_id == owner)
                EXPECT_FALSE(entry.crop_effects_enabled);
        ASSERT_TRUE(op::undoHistory().undo().success);
        EXPECT_TRUE(scene.getNodeByUuid(uuid)->crop_effects_enabled);
        ASSERT_TRUE(op::undoHistory().redo().success);
        EXPECT_FALSE(scene.getNodeByUuid(uuid)->crop_effects_enabled);
        ASSERT_TRUE(sm->setSplatCropGroupEnabled(uuid, true));
        EXPECT_TRUE(scene.getNodeById(box)->cropbox->enabled);
        EXPECT_FALSE(scene.getNodeById(ellipse)->ellipsoid->enabled);
    }

    TEST_F(SplatCropWorkflowTest, DeleteGroupRemovesOnlyCropsAndUndoesGateAndChildrenTogether) {
        auto& scene = sm->getScene();
        const auto uuid = scene.getNodeById(owner)->uuid;
        const auto crop_uuid = scene.getNodeById(box)->uuid;
        const auto unrelated = scene.addCollision("Unrelated", std::make_shared<core::CollisionVolume>(), owner);
        ASSERT_TRUE(sm->setSplatCropGroupEnabled(uuid, false));
        op::undoHistory().clear();
        ASSERT_TRUE(sm->deleteSplatCropGroup(uuid));
        EXPECT_EQ(scene.getNodeByUuid(crop_uuid), nullptr);
        EXPECT_NE(scene.getNodeById(unrelated), nullptr);
        EXPECT_TRUE(scene.getNodeByUuid(uuid)->crop_effects_enabled);
        EXPECT_EQ(op::undoHistory().undoCount(), 1);
        ASSERT_TRUE(op::undoHistory().undo().success);
        EXPECT_NE(scene.getNodeByUuid(crop_uuid), nullptr);
        EXPECT_FALSE(scene.getNodeByUuid(uuid)->crop_effects_enabled);
        ASSERT_TRUE(op::undoHistory().redo().success);
        EXPECT_EQ(scene.getNodeByUuid(crop_uuid), nullptr);
        EXPECT_NE(scene.getNodeByUuid(uuid), nullptr);
    }

    TEST_F(SplatCropTreeTest, VirtualContainerDoesNotReparentOrEnterEditingAndItsSwitchUsesOneGate) {
        auto& scene = sm->getScene();
        const auto count = scene.getNodeCount();
        auto* group = tree->QuerySelector(".crop-group[data-crop-owner='" + std::to_string(owner) + "']");
        ASSERT_NE(group, nullptr);
        EXPECT_EQ(tree->nodeCount(), count);
        EXPECT_EQ(scene.getNodeById(box)->parent_id, owner);
        const auto original_tool = UnifiedToolRegistry::instance().getActiveTool();
        group->QuerySelector(".node-name")->DispatchEvent("click", {});
        EXPECT_TRUE(sm->getSelectedNodeIds().empty());
        group->QuerySelector(".node-name")->DispatchEvent("dblclick", {});
        EXPECT_EQ(UnifiedToolRegistry::instance().getActiveTool(), original_tool);
        EXPECT_EQ(scene.getNodeById(box)->parent_id, owner);
        group->QuerySelector(".crop-effect-toggle")->DispatchEvent("click", {});
        EXPECT_FALSE(scene.getNodeById(owner)->crop_effects_enabled);
        EXPECT_EQ(scene.getNodeCount(), count);
        UnifiedToolRegistry::instance().setActiveTool("");
    }

    TEST_F(SplatCropWorkflowTest, GroupMutationsRejectMissingOwnerAndLocks) {
        auto& scene = sm->getScene();
        const auto uuid = scene.getNodeById(owner)->uuid;
        scene.setNodeLocked(scene.getNodeById(owner)->name, true);
        EXPECT_FALSE(sm->setSplatCropGroupEnabled(uuid, false));
        EXPECT_FALSE(sm->deleteSplatCropGroup(uuid));
        EXPECT_FALSE(sm->bakeSplatCropGroup(uuid));
        EXPECT_TRUE(scene.getNodeById(owner)->crop_effects_enabled);
        EXPECT_NE(scene.getNodeById(box), nullptr);
        EXPECT_EQ(op::undoHistory().undoCount(), 0);
        EXPECT_FALSE(sm->setSplatCropGroupEnabled(core::generate_uuid_v4(), false));
    }

    TEST_F(SplatCropWorkflowTest, BakeCollectiveIntersectionKeepsDisabledCropsAndIsOneUndo) {
        auto& scene = sm->getScene();
        const auto source = scene.addSplat("Bake source", groupTestSplat());
        const auto uuid = scene.getNodeById(source)->uuid;
        const auto a = scene.addCropBox("Active box", source);
        auto* box_data = scene.getCropBoxData(a);
        box_data->min = {-1, -1, 1};
        box_data->max = {1, 1, 3};
        box_data->enabled = true;
        scene.setNodeTransform("Active box", glm::mat4(1));
        const auto b = scene.addEllipsoid("Active ellipsoid", source);
        auto* ellipse = scene.getEllipsoidData(b);
        ellipse->radii = {.25f, 1, 1};
        ellipse->enabled = true;
        scene.setNodeTransform("Active ellipsoid", glm::translate(glm::mat4(1), glm::vec3(0, 0, 2)));
        core::CropPolygonData polygon;
        polygon.vertices = {{-.75f, -1}, {.75f, -1}, {.75f, 1}, {-.75f, 1}};
        polygon.half_depth = 1;
        const auto c = scene.addCropPolygon("Active polygon", source, polygon, glm::translate(glm::mat4(1), glm::vec3(0, 0, 2)));
        const auto disabled = cap::duplicateSplatCrop(*sm, nullptr, scene.getNodeById(a)->uuid);
        ASSERT_TRUE(disabled);
        scene.getNodeById(*disabled)->cropbox->enabled = false;
        scene.getNodeById(*disabled)->cropbox->inverse = true;
        const auto disabled_uuid = scene.getNodeById(*disabled)->uuid;
        op::undoHistory().clear();
        ASSERT_TRUE(sm->setSplatCropGroupEnabled(uuid, false));
        op::undoHistory().clear();
        EXPECT_FALSE(sm->bakeSplatCropGroup(uuid));
        EXPECT_EQ(scene.getNodeByUuid(uuid)->model->visible_count(), 4);
        EXPECT_EQ(op::undoHistory().undoCount(), 0);
        ASSERT_TRUE(sm->setSplatCropGroupEnabled(uuid, true));
        op::undoHistory().clear();
        scene.getCropPolygonData(c)->vertices = {{0, 0}, {1, 0}};
        EXPECT_FALSE(sm->bakeSplatCropGroup(uuid));
        EXPECT_EQ(scene.getNodeByUuid(uuid)->model->visible_count(), 4);
        EXPECT_EQ(scene.getNodeByUuid(uuid)->children.size(), 4);
        EXPECT_EQ(op::undoHistory().undoCount(), 0);
        scene.getCropPolygonData(c)->vertices = polygon.vertices;
        const auto result = sm->bakeSplatCropGroup(uuid);
        ASSERT_TRUE(result) << (result ? "" : std::string(result.error().detail()));
        EXPECT_EQ(scene.getNodeByUuid(uuid)->model->visible_count(), 1);
        EXPECT_EQ(scene.getNodeByUuid(uuid)->children.size(), 1);
        EXPECT_NE(scene.getNodeByUuid(disabled_uuid), nullptr);
        EXPECT_FALSE(scene.getNodeByUuid(disabled_uuid)->cropbox->enabled);
        EXPECT_EQ(op::undoHistory().undoCount(), 1);
        ASSERT_TRUE(op::undoHistory().undo().success);
        EXPECT_EQ(scene.getNodeByUuid(uuid)->model->visible_count(), 4);
        EXPECT_EQ(scene.getNodeByUuid(uuid)->children.size(), 4);
        ASSERT_TRUE(op::undoHistory().redo().success);
        EXPECT_EQ(scene.getNodeByUuid(uuid)->model->visible_count(), 1);
        EXPECT_EQ(scene.getNodeByUuid(uuid)->children.size(), 1);
        EXPECT_FALSE(scene.getNodeByUuid(disabled_uuid)->cropbox->enabled);
    }

    TEST(CropGroupPersistenceTest, OptionalGateRoundTripsAndOldProjectsDefaultEnabled) {
        namespace pj = lfs::io::project;
        core::Scene scene;
        const auto id = scene.addSplat("Persist group", groupTestSplat());
        const auto uuid = scene.getNodeById(id)->uuid;
        scene.getNodeById(id)->crop_effects_enabled = false;
        const pj::ScenePayloadBindings bindings{{uuid, pj::PayloadBinding{.fourcc = "DSRC", .instance_uuid = uuid, .reference_uuid = std::nullopt, .source_kind = "ply"}}};
        const auto chapter = pj::capture_scene_graph(scene, bindings);
        ASSERT_TRUE(chapter);
        const auto bytes = chapter->to_bytes();
        const auto decoded = pj::SceneGraphChapter::from_bytes(bytes);
        ASSERT_TRUE(decoded);
        core::Scene restored;
        pj::ScenePayloadResolver resolver{.splat = [](const pj::PayloadBinding&) -> lfs::Result<std::unique_ptr<core::SplatData>> { return groupTestSplat(); }};
        ASSERT_TRUE(pj::hydrate_scene_graph(*decoded, restored, resolver));
        ASSERT_NE(restored.getNodeByUuid(uuid), nullptr);
        EXPECT_FALSE(restored.getNodeByUuid(uuid)->crop_effects_enabled);
        std::string text(reinterpret_cast<const char*>(bytes.data()), bytes.size());
        const std::regex field(R"("crop_effects_enabled"\s*:\s*false\s*,)");
        std::smatch match;
        ASSERT_TRUE(std::regex_search(text, match, field));
        const auto pos = match.position();
        text.erase(pos, match.length());
        const auto legacy = pj::SceneGraphChapter::parse(text);
        ASSERT_TRUE(legacy);
        const auto nodes = legacy->nodes();
        ASSERT_TRUE(nodes);
        ASSERT_EQ(nodes->size(), 1);
        EXPECT_TRUE(nodes->front().crop_effects_enabled);
        text.insert(pos, "\"crop_effects_enabled\":\"bad\",");
        const auto invalid = pj::SceneGraphChapter::parse(text);
        EXPECT_FALSE(invalid);
    }

} // namespace

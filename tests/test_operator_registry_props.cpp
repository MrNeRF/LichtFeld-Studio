/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/event_bridge/event_bridge.hpp"
#include "core/event_bus.hpp"
#include "core/services.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "operation/ops/transform_ops.hpp"
#include "operation/undo_history.hpp"
#include "operator/operator_properties.hpp"
#include "operator/operator_registry.hpp"
#include "operator/ops/edit_ops.hpp"
#include "operator/ops/transform_ops.hpp"
#include "operator/property_schema.hpp"
#include "rendering/coordinate_conventions.hpp"
#include "rendering/rendering_manager.hpp"
#include "scene/scene_manager.hpp"
#include "visualizer/core/editor_context.hpp"
#include "visualizer/gui_capabilities.hpp"

#include <glm/gtc/matrix_transform.hpp>
#include <gtest/gtest.h>
#include <memory>
#include <vector>

using lfs::core::DataType;
using lfs::core::Device;
using lfs::core::Tensor;

namespace {

    std::unique_ptr<lfs::core::SplatData> make_test_splat(const std::vector<float>& xyz) {
        const size_t count = xyz.size() / 3;
        auto means = Tensor::from_vector(xyz, {count, size_t{3}}, Device::CUDA).to(DataType::Float32);
        auto sh0 = Tensor::zeros({count, size_t{1}, size_t{3}}, Device::CUDA, DataType::Float32);
        auto shN = Tensor::zeros({count, size_t{3}, size_t{3}}, Device::CUDA, DataType::Float32);
        auto scaling = Tensor::zeros({count, size_t{3}}, Device::CUDA, DataType::Float32);

        std::vector<float> rotation_data(count * 4, 0.0f);
        for (size_t i = 0; i < count; ++i) {
            rotation_data[i * 4] = 1.0f;
        }
        auto rotation = Tensor::from_vector(rotation_data, {count, size_t{4}}, Device::CUDA).to(DataType::Float32);
        auto opacity = Tensor::zeros({count, size_t{1}}, Device::CUDA, DataType::Float32);

        return std::make_unique<lfs::core::SplatData>(
            1,
            std::move(means),
            std::move(sh0),
            std::move(shN),
            std::move(scaling),
            std::move(rotation),
            std::move(opacity),
            1.0f);
    }

} // namespace

class OperatorRegistryPropsTest : public ::testing::Test {
protected:
    void SetUp() override {
        lfs::event::EventBridge::instance().clear_all();
        lfs::core::event::bus().clear_all();
        lfs::vis::services().clear();
        lfs::vis::op::undoHistory().clear();
        lfs::vis::op::operators().clear();

        rendering_manager_ = std::make_unique<lfs::vis::RenderingManager>();
        scene_manager_ = std::make_unique<lfs::vis::SceneManager>();
        lfs::vis::services().set(rendering_manager_.get());
        lfs::vis::services().set(scene_manager_.get());
        lfs::vis::op::operators().setSceneManager(scene_manager_.get());

        lfs::vis::op::registerEditOperators();
        lfs::vis::op::registerTransformOperators();
    }

    void TearDown() override {
        lfs::vis::op::unregisterTransformOperators();
        lfs::vis::op::unregisterEditOperators();
        lfs::vis::op::operators().clear();
        lfs::vis::op::operators().setSceneManager(nullptr);
        lfs::vis::op::undoHistory().clear();
        lfs::vis::services().clear();
        scene_manager_.reset();
        rendering_manager_.reset();
        lfs::event::EventBridge::instance().clear_all();
        lfs::core::event::bus().clear_all();
    }

    void add_node(const std::string& name,
                  const std::vector<float>& xyz = {
                      0.0f,
                      0.0f,
                      0.0f,
                      1.0f,
                      0.0f,
                      0.0f,
                  }) {
        scene_manager_->getScene().addNode(
            name,
            make_test_splat(xyz));
    }

    std::unique_ptr<lfs::vis::RenderingManager> rendering_manager_;
    std::unique_ptr<lfs::vis::SceneManager> scene_manager_;
};

TEST_F(OperatorRegistryPropsTest, DeleteOperatorCanDeleteNamedNodeWithoutSelection) {
    add_node("delete_me");
    EXPECT_FALSE(scene_manager_->hasSelectedNode());

    lfs::vis::op::OperatorProperties props;
    props.set("name", std::string("delete_me"));
    props.set("keep_children", false);

    const auto result = lfs::vis::op::operators().invoke(lfs::vis::op::BuiltinOp::Delete, &props);
    ASSERT_TRUE(result.is_finished());
    EXPECT_EQ(scene_manager_->getScene().getNode("delete_me"), nullptr);

    const auto resolved = props.get<std::vector<std::string>>("resolved_node_names");
    ASSERT_TRUE(resolved.has_value());
    ASSERT_EQ(resolved->size(), 1u);
    EXPECT_EQ(resolved->front(), "delete_me");
}

TEST_F(OperatorRegistryPropsTest, TransformTranslateOperatorUsesVisualizerWorldCoordinates) {
    add_node("move_me");
    EXPECT_FALSE(scene_manager_->hasSelectedNode());

    lfs::vis::op::OperatorProperties props;
    props.set("node", std::string("move_me"));
    props.set("value", glm::vec3(1.0f, 2.0f, 3.0f));

    const auto result = lfs::vis::op::operators().invoke(lfs::vis::op::BuiltinOp::TransformTranslate, &props);
    ASSERT_TRUE(result.is_finished());

    const auto components = lfs::vis::cap::decomposeTransform(
        scene_manager_->getScene().getNodeTransform("move_me"));
    EXPECT_FLOAT_EQ(components.translation.x, 1.0f);
    EXPECT_FLOAT_EQ(components.translation.y, -2.0f);
    EXPECT_FLOAT_EQ(components.translation.z, -3.0f);

    const auto* const node = scene_manager_->getScene().getNode("move_me");
    ASSERT_NE(node, nullptr);
    const glm::mat4 world_transform =
        lfs::rendering::dataWorldTransformToVisualizerWorld(scene_manager_->getScene().getWorldTransform(node->id));
    EXPECT_FLOAT_EQ(world_transform[3].x, 1.0f);
    EXPECT_FLOAT_EQ(world_transform[3].y, 2.0f);
    EXPECT_FLOAT_EQ(world_transform[3].z, 3.0f);

    const auto resolved = props.get<std::vector<std::string>>("resolved_node_names");
    ASSERT_TRUE(resolved.has_value());
    ASSERT_EQ(resolved->size(), 1u);
    EXPECT_EQ(resolved->front(), "move_me");
}

TEST_F(OperatorRegistryPropsTest, EditorContextDisablesTransformToolsForMixedLockedSelection) {
    add_node("editable");
    add_node("locked");
    scene_manager_->getScene().setNodeLocked("locked", true);
    scene_manager_->selectNodes({"editable", "locked"});

    lfs::vis::EditorContext editor;
    editor.update(scene_manager_.get(), nullptr);

    EXPECT_FALSE(editor.canTransformSelectedNode());
    EXPECT_FALSE(editor.isToolAvailable(lfs::vis::ToolType::Translate));
    EXPECT_STREQ(editor.getToolUnavailableReason(lfs::vis::ToolType::Translate),
                 "selection contains locked nodes");
}

TEST_F(OperatorRegistryPropsTest, EditorContextDisablesTransformToolsForMixedUnsupportedSelection) {
    add_node("editable");
    ASSERT_NE(scene_manager_->getScene().addGroup("group"), lfs::core::NULL_NODE);
    scene_manager_->selectNodes({"editable", "group"});

    lfs::vis::EditorContext editor;
    editor.update(scene_manager_.get(), nullptr);

    EXPECT_FALSE(editor.canTransformSelectedNode());
    EXPECT_FALSE(editor.isToolAvailable(lfs::vis::ToolType::Translate));
    EXPECT_STREQ(editor.getToolUnavailableReason(lfs::vis::ToolType::Translate),
                 "selection contains unsupported nodes");
}

TEST_F(OperatorRegistryPropsTest, LegacyTransformRotateUsesEditableTargetPivotOnly) {
    add_node("editable", {
                             0.0f,
                             0.0f,
                             0.0f,
                             1.0f,
                             0.0f,
                             0.0f,
                         });
    add_node("locked", {
                           10.0f,
                           0.0f,
                           0.0f,
                           11.0f,
                           0.0f,
                           0.0f,
                       });
    scene_manager_->getScene().setNodeLocked("locked", true);
    scene_manager_->selectNodes({"editable", "locked"});

    lfs::vis::op::TransformRotate op;
    lfs::vis::op::OperatorProperties props;
    props.set("axis", glm::vec3(0.0f, 0.0f, 1.0f));
    props.set("angle", 180.0f);

    const auto result = op.execute(*scene_manager_, props, {});
    ASSERT_TRUE(result.ok());

    const auto components = lfs::vis::cap::decomposeTransform(
        scene_manager_->getScene().getNodeTransform("editable"));
    EXPECT_FLOAT_EQ(components.translation.x, 1.0f);
    EXPECT_NEAR(components.translation.y, 0.0f, 1e-5f);
    EXPECT_NEAR(components.translation.z, 0.0f, 1e-5f);

    const auto locked_components = lfs::vis::cap::decomposeTransform(
        scene_manager_->getScene().getNodeTransform("locked"));
    EXPECT_FLOAT_EQ(locked_components.translation.x, 0.0f);
    EXPECT_FLOAT_EQ(locked_components.translation.y, 0.0f);
    EXPECT_FLOAT_EQ(locked_components.translation.z, 0.0f);
}

TEST_F(OperatorRegistryPropsTest, VisualizerFacingTransformSelectionUsesVisualizerWorldCenter) {
    add_node("target", {
                           0.0f,
                           0.0f,
                           0.0f,
                           2.0f,
                           4.0f,
                           6.0f,
                       });
    scene_manager_->setNodeTransform(
        "target",
        glm::translate(glm::mat4(1.0f), glm::vec3(10.0f, 20.0f, 30.0f)));
    scene_manager_->selectNode("target");

    const auto resolved = lfs::vis::cap::resolveEditableTransformSelection(*scene_manager_, std::nullopt);
    ASSERT_TRUE(resolved.has_value());
    ASSERT_EQ(resolved->node_names.size(), 1u);
    EXPECT_EQ(resolved->node_names.front(), "target");

    const glm::vec3 expected_center =
        lfs::rendering::visualizerWorldPointFromDataWorld(glm::vec3(11.0f, 22.0f, 33.0f));
    EXPECT_NEAR(resolved->world_center.x, expected_center.x, 1e-5f);
    EXPECT_NEAR(resolved->world_center.y, expected_center.y, 1e-5f);
    EXPECT_NEAR(resolved->world_center.z, expected_center.z, 1e-5f);

    const glm::vec3 selection_center = scene_manager_->getSelectionVisualizerWorldCenter();
    EXPECT_NEAR(selection_center.x, expected_center.x, 1e-5f);
    EXPECT_NEAR(selection_center.y, expected_center.y, 1e-5f);
    EXPECT_NEAR(selection_center.z, expected_center.z, 1e-5f);

    const glm::mat4 selected_world = scene_manager_->getSelectedNodeVisualizerWorldTransform();
    EXPECT_NEAR(selected_world[3].x, 10.0f, 1e-5f);
    EXPECT_NEAR(selected_world[3].y, -20.0f, 1e-5f);
    EXPECT_NEAR(selected_world[3].z, -30.0f, 1e-5f);
}

TEST_F(OperatorRegistryPropsTest, LegacySelectionWorldCenterRemainsDataWorld) {
    add_node("target", {
                           0.0f,
                           0.0f,
                           0.0f,
                           2.0f,
                           4.0f,
                           6.0f,
                       });
    scene_manager_->setNodeTransform(
        "target",
        glm::translate(glm::mat4(1.0f), glm::vec3(10.0f, 20.0f, 30.0f)));
    scene_manager_->selectNode("target");

    const glm::vec3 data_center = scene_manager_->getSelectionWorldCenter();
    EXPECT_NEAR(data_center.x, 11.0f, 1e-5f);
    EXPECT_NEAR(data_center.y, 22.0f, 1e-5f);
    EXPECT_NEAR(data_center.z, 33.0f, 1e-5f);

    const glm::vec3 expected_visualizer_center =
        lfs::rendering::visualizerWorldPointFromDataWorld(data_center);
    const glm::vec3 visualizer_center = scene_manager_->getSelectionVisualizerWorldCenter();
    EXPECT_NEAR(visualizer_center.x, expected_visualizer_center.x, 1e-5f);
    EXPECT_NEAR(visualizer_center.y, expected_visualizer_center.y, 1e-5f);
    EXPECT_NEAR(visualizer_center.z, expected_visualizer_center.z, 1e-5f);

    const auto* const node = scene_manager_->getScene().getNode("target");
    ASSERT_NE(node, nullptr);
    const glm::mat4 data_world = scene_manager_->getScene().getWorldTransform(node->id);
    EXPECT_NEAR(data_world[3].x, 10.0f, 1e-5f);
    EXPECT_NEAR(data_world[3].y, 20.0f, 1e-5f);
    EXPECT_NEAR(data_world[3].z, 30.0f, 1e-5f);

    const glm::mat4 visualizer_world = scene_manager_->getSelectedNodeVisualizerWorldTransform();
    EXPECT_NEAR(visualizer_world[3].x, 10.0f, 1e-5f);
    EXPECT_NEAR(visualizer_world[3].y, -20.0f, 1e-5f);
    EXPECT_NEAR(visualizer_world[3].z, -30.0f, 1e-5f);
}

TEST_F(OperatorRegistryPropsTest, ResolveCropBoxIdFindsAttachedChildForParentNodeAndSelection) {
    auto& scene = scene_manager_->getScene();
    const auto parent_id = scene.addGroup("crop_parent");
    ASSERT_NE(parent_id, lfs::core::NULL_NODE);

    const auto cropbox_id = scene.addCropBox("crop_parent_cropbox", parent_id);
    ASSERT_NE(cropbox_id, lfs::core::NULL_NODE);

    auto explicit_result = lfs::vis::cap::resolveCropBoxId(*scene_manager_, std::string("crop_parent"));
    ASSERT_TRUE(explicit_result.has_value());
    EXPECT_EQ(*explicit_result, cropbox_id);

    scene_manager_->selectNode("crop_parent");
    auto selected_result = lfs::vis::cap::resolveCropBoxId(*scene_manager_, std::nullopt);
    ASSERT_TRUE(selected_result.has_value());
    EXPECT_EQ(*selected_result, cropbox_id);
}

TEST_F(OperatorRegistryPropsTest, SetTransformMatrixCreatesUndoableEntry) {
    add_node("matrix_target");

    const glm::mat4 transform = glm::translate(glm::mat4(1.0f), glm::vec3(3.0f, -2.0f, 1.5f));
    auto result = lfs::vis::cap::setTransformMatrix(*scene_manager_, {"matrix_target"}, transform,
                                                    "python.set_node_transform");
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(lfs::vis::op::undoHistory().undoCount(), 1u);

    auto components = lfs::vis::cap::decomposeTransform(
        scene_manager_->getScene().getNodeTransform("matrix_target"));
    EXPECT_FLOAT_EQ(components.translation.x, 3.0f);
    EXPECT_FLOAT_EQ(components.translation.y, -2.0f);
    EXPECT_FLOAT_EQ(components.translation.z, 1.5f);

    lfs::vis::op::undoHistory().undo();
    components = lfs::vis::cap::decomposeTransform(
        scene_manager_->getScene().getNodeTransform("matrix_target"));
    EXPECT_FLOAT_EQ(components.translation.x, 0.0f);
    EXPECT_FLOAT_EQ(components.translation.y, 0.0f);
    EXPECT_FLOAT_EQ(components.translation.z, 0.0f);
}

TEST_F(OperatorRegistryPropsTest, TransformSetOperatorUsesVisualizerWorldCoordinates) {
    add_node("target");

    lfs::vis::op::OperatorProperties props;
    props.set("node", std::string("target"));
    props.set("translation", glm::vec3(4.0f, 5.0f, 6.0f));

    const auto result = lfs::vis::op::operators().invoke(lfs::vis::op::BuiltinOp::TransformSet, &props);
    ASSERT_TRUE(result.is_finished());

    const auto components = lfs::vis::cap::decomposeTransform(
        scene_manager_->getScene().getNodeTransform("target"));
    EXPECT_FLOAT_EQ(components.translation.x, 4.0f);
    EXPECT_FLOAT_EQ(components.translation.y, -5.0f);
    EXPECT_FLOAT_EQ(components.translation.z, -6.0f);

    const auto* const node = scene_manager_->getScene().getNode("target");
    ASSERT_NE(node, nullptr);
    const glm::mat4 world_transform =
        lfs::rendering::dataWorldTransformToVisualizerWorld(scene_manager_->getScene().getWorldTransform(node->id));
    EXPECT_FLOAT_EQ(world_transform[3].x, 4.0f);
    EXPECT_FLOAT_EQ(world_transform[3].y, 5.0f);
    EXPECT_FLOAT_EQ(world_transform[3].z, 6.0f);
}

TEST_F(OperatorRegistryPropsTest, BuiltinOperatorSchemasAreRegistered) {
    const auto* delete_schema = lfs::vis::op::propertySchemas().getSchema("ed.delete");
    ASSERT_NE(delete_schema, nullptr);
    EXPECT_EQ(delete_schema->size(), 2u);
    EXPECT_EQ(delete_schema->at(0).name, "name");
    EXPECT_EQ(delete_schema->at(1).name, "keep_children");

    const auto* translate_schema = lfs::vis::op::propertySchemas().getSchema("transform.translate");
    ASSERT_NE(translate_schema, nullptr);
    ASSERT_EQ(translate_schema->size(), 2u);
    EXPECT_EQ(translate_schema->at(0).name, "node");
    EXPECT_EQ(translate_schema->at(1).name, "value");
}

TEST_F(OperatorRegistryPropsTest, CallbackInvokeReleasesRegistryMutexDuringInvoke) {
    bool callback_called = false;
    bool mutex_was_unlocked = false;

    lfs::vis::op::operators().registerCallbackOperator(
        lfs::vis::op::OperatorDescriptor{
            .python_class_id = "test.callback.unlock",
            .label = "Test Callback Unlock",
            .description = "Regression test callback operator",
        },
        lfs::vis::op::CallbackOperator{
            .poll = [] { return true; },
            .invoke =
                [&](lfs::vis::op::OperatorProperties&) {
                    callback_called = true;
                    mutex_was_unlocked = lfs::vis::op::operators().canLockMutexForTest();
                    return lfs::vis::op::OperatorResult::FINISHED;
                },
        });

    const auto result = lfs::vis::op::operators().invoke("test.callback.unlock");
    ASSERT_TRUE(result.is_finished());
    EXPECT_TRUE(callback_called);
    EXPECT_TRUE(mutex_was_unlocked);
}

TEST_F(OperatorRegistryPropsTest, CallbackInvokeCanSelfUnregisterWhileEnteringModal) {
    constexpr const char* kOperatorId = "test.callback.self_unregister.invoke";

    lfs::vis::op::operators().registerCallbackOperator(
        lfs::vis::op::OperatorDescriptor{
            .python_class_id = kOperatorId,
            .label = "Self Unregister Invoke",
            .description = "Regression test for callback invoke self-unregister",
        },
        lfs::vis::op::CallbackOperator{
            .invoke =
                [](lfs::vis::op::OperatorProperties&) {
                    lfs::vis::op::operators().unregisterOperator("test.callback.self_unregister.invoke");
                    return lfs::vis::op::OperatorResult::RUNNING_MODAL;
                },
            .modal =
                [](const lfs::vis::op::ModalEvent&, lfs::vis::op::OperatorProperties&) {
                    return lfs::vis::op::OperatorResult::CANCELLED;
                },
        });

    const auto result = lfs::vis::op::operators().invoke(kOperatorId);
    EXPECT_EQ(result.status, lfs::vis::op::OperatorResult::RUNNING_MODAL);
    EXPECT_EQ(lfs::vis::op::operators().dispatchModalEvent({}), lfs::vis::op::OperatorResult::CANCELLED);
}

TEST_F(OperatorRegistryPropsTest, CallbackModalCanSelfUnregisterWithoutDoubleCancel) {
    constexpr const char* kOperatorId = "test.callback.self_unregister.modal";
    int cancel_count = 0;

    lfs::vis::op::operators().registerCallbackOperator(
        lfs::vis::op::OperatorDescriptor{
            .python_class_id = kOperatorId,
            .label = "Self Unregister Modal",
            .description = "Regression test for callback modal self-unregister",
        },
        lfs::vis::op::CallbackOperator{
            .invoke =
                [](lfs::vis::op::OperatorProperties&) {
                    return lfs::vis::op::OperatorResult::RUNNING_MODAL;
                },
            .modal =
                [](const lfs::vis::op::ModalEvent&, lfs::vis::op::OperatorProperties&) {
                    lfs::vis::op::operators().unregisterOperator("test.callback.self_unregister.modal");
                    return lfs::vis::op::OperatorResult::CANCELLED;
                },
            .cancel =
                [&cancel_count] {
                    ++cancel_count;
                },
        });

    const auto invoke_result = lfs::vis::op::operators().invoke(kOperatorId);
    ASSERT_EQ(invoke_result.status, lfs::vis::op::OperatorResult::RUNNING_MODAL);

    EXPECT_EQ(lfs::vis::op::operators().dispatchModalEvent({}), lfs::vis::op::OperatorResult::CANCELLED);
    EXPECT_EQ(cancel_count, 1);
}

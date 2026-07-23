/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/camera.hpp"
#include "core/event_bridge/event_bridge.hpp"
#include "core/event_bus.hpp"
#include "core/services.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "gui/gizmo_manager.hpp"
#include "gui/gizmo_transform.hpp"
#include "operation/ops/select_ops.hpp"
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
#include "training/trainer.hpp"
#include "training/training_manager.hpp"
#include "visualizer/core/editor_context.hpp"
#include "visualizer/gui_capabilities.hpp"
#include "visualizer/scene_coordinate_utils.hpp"

#include <algorithm>
#include <cmath>
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

    std::vector<uint8_t> selection_mask_values(const lfs::core::Scene& scene) {
        const auto mask = scene.getSelectionMask();
        if (!mask || !mask->is_valid()) {
            return {};
        }
        return mask->cpu().to_vector_uint8();
    }

    Tensor make_uint8_mask(const std::vector<uint8_t>& values) {
        auto tensor = Tensor::empty({values.size()}, Device::CPU, DataType::UInt8);
        std::copy(values.begin(), values.end(), tensor.ptr<uint8_t>());
        return tensor.cuda();
    }

    void expect_matrix_near(const glm::mat4& actual, const glm::mat4& expected, const float epsilon = 1e-4f) {
        for (int column = 0; column < 4; ++column) {
            for (int row = 0; row < 4; ++row) {
                EXPECT_NEAR(actual[column][row], expected[column][row], epsilon)
                    << "matrix mismatch at column " << column << ", row " << row;
            }
        }
    }

    float matrix_max_abs_diff(const glm::mat4& a, const glm::mat4& b) {
        float max_diff = 0.0f;
        for (int column = 0; column < 4; ++column) {
            for (int row = 0; row < 4; ++row) {
                max_diff = std::max(max_diff, std::abs(a[column][row] - b[column][row]));
            }
        }
        return max_diff;
    }

    void expect_matrix_finite(const glm::mat4& value) {
        for (int column = 0; column < 4; ++column) {
            for (int row = 0; row < 4; ++row) {
                EXPECT_TRUE(std::isfinite(value[column][row]))
                    << "matrix value is not finite at column " << column << ", row " << row;
            }
        }
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
        scene_manager_->getScene().addSplat(
            name,
            make_test_splat(xyz));
    }

    std::vector<glm::mat4> capture_visualizer_world_transforms(const std::vector<std::string>& names) const {
        std::vector<glm::mat4> transforms;
        transforms.reserve(names.size());
        for (const auto& name : names) {
            transforms.push_back(
                lfs::vis::scene_coords::nodeVisualizerWorldTransform(scene_manager_->getScene(), name).value());
        }
        return transforms;
    }

    void apply_group_visualizer_delta(const std::vector<std::string>& names,
                                      const std::vector<glm::mat4>& original_visualizer_world,
                                      const glm::mat4& delta) {
        for (const auto& transform : lfs::vis::gui::gizmo_ops::computeNodeSharedSelectionLocalTransforms(
                 scene_manager_->getScene(), names, original_visualizer_world, delta)) {
            scene_manager_->setNodeTransform(transform.name, transform.local_transform);
        }
    }

    void apply_individual_visualizer_delta(const std::vector<std::string>& names,
                                           const std::vector<glm::mat4>& original_visualizer_world,
                                           const glm::mat4& delta) {
        for (const auto& transform : lfs::vis::gui::gizmo_ops::computeNodeIndividualLocalTransforms(
                 scene_manager_->getScene(), names, original_visualizer_world, delta)) {
            scene_manager_->setNodeTransform(transform.name, transform.local_transform);
        }
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

TEST_F(OperatorRegistryPropsTest, DeleteOperatorDeletesMultipleSelectedNodes) {
    add_node("delete_a");
    add_node("delete_b");
    scene_manager_->selectNodes({"delete_a", "delete_b"});

    lfs::vis::op::OperatorProperties props;
    props.set("keep_children", false);

    const auto result = lfs::vis::op::operators().invoke(lfs::vis::op::BuiltinOp::Delete, &props);
    ASSERT_TRUE(result.is_finished());
    EXPECT_EQ(scene_manager_->getScene().getNode("delete_a"), nullptr);
    EXPECT_EQ(scene_manager_->getScene().getNode("delete_b"), nullptr);

    const auto resolved = props.get<std::vector<std::string>>("resolved_node_names");
    ASSERT_TRUE(resolved.has_value());
    ASSERT_EQ(resolved->size(), 2u);
    std::vector<std::string> sorted = *resolved;
    std::sort(sorted.begin(), sorted.end());
    EXPECT_EQ(sorted, (std::vector<std::string>{"delete_a", "delete_b"}));
}

TEST_F(OperatorRegistryPropsTest, DeleteOperatorRejectsMixedTrainingBatchWithoutPartialRemoval) {
    auto trainer_manager = std::make_unique<lfs::vis::TrainerManager>();
    lfs::vis::services().set(trainer_manager.get());

    auto& scene = scene_manager_->getScene();
    const auto model_id = scene.addSplat("Model", make_test_splat({0.0f, 0.0f, 0.0f}));
    const auto unrelated_id = scene.addSplat("unrelated", make_test_splat({1.0f, 0.0f, 0.0f}));
    const auto cameras_id = scene.addGroup("Cameras");
    const auto train_group_id = scene.addCameraGroup("Training", cameras_id, 1);
    const auto val_group_id = scene.addCameraGroup("Validation", cameras_id, 1);
    scene.addCamera("train.png", train_group_id, std::make_shared<lfs::core::Camera>());
    const auto val_camera_id = scene.addCamera("val.png", val_group_id, std::make_shared<lfs::core::Camera>());
    scene.setCameraTrainingEnabled(val_camera_id, false);
    scene.setTrainingModelNode("Model");
    scene_manager_->changeContentType(lfs::vis::SceneManager::ContentType::Dataset);

    trainer_manager->setScene(&scene);
    trainer_manager->setTrainerFromCheckpoint(std::make_unique<lfs::training::Trainer>(scene), 0);
    ASSERT_FALSE(trainer_manager->canPerform(lfs::vis::TrainingAction::DeleteTrainingNode));

    scene_manager_->selectNodes({"unrelated", "Model"});
    lfs::vis::op::OperatorProperties props;
    props.set("keep_children", false);

    const auto result = lfs::vis::op::operators().invoke(lfs::vis::op::BuiltinOp::Delete, &props);
    ASSERT_TRUE(result.is_cancelled());
    EXPECT_NE(scene.getNodeById(model_id), nullptr);
    EXPECT_NE(scene.getNodeById(unrelated_id), nullptr);

    const auto error = props.get<std::string>("error");
    ASSERT_TRUE(error.has_value());
    EXPECT_NE(error->find("Cannot delete 'Model'"), std::string::npos);
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

TEST_F(OperatorRegistryPropsTest, MultiNodeGizmoTranslationMovesSelectedTargetsTogether) {
    add_node("left");
    add_node("right");
    scene_manager_->setNodeTransform("left", glm::translate(glm::mat4(1.0f), glm::vec3(-2.0f, 0.0f, 0.0f)));
    scene_manager_->setNodeTransform("right", glm::translate(glm::mat4(1.0f), glm::vec3(2.0f, 1.0f, 0.0f)));

    const std::vector<std::string> names{"left", "right"};
    const auto originals = capture_visualizer_world_transforms(names);
    const glm::mat4 delta = glm::translate(glm::mat4(1.0f), glm::vec3(3.0f, -4.0f, 2.0f));

    apply_group_visualizer_delta(names, originals, delta);

    for (size_t i = 0; i < names.size(); ++i) {
        const glm::mat4 actual =
            lfs::vis::scene_coords::nodeVisualizerWorldTransform(scene_manager_->getScene(), names[i]).value();
        expect_matrix_near(actual, delta * originals[i]);
    }
}

TEST_F(OperatorRegistryPropsTest, MultiNodeGizmoRotationUsesSharedPivot) {
    add_node("left");
    add_node("right");
    scene_manager_->setNodeTransform("left", glm::translate(glm::mat4(1.0f), glm::vec3(-2.0f, 0.0f, 0.0f)));
    scene_manager_->setNodeTransform("right", glm::translate(glm::mat4(1.0f), glm::vec3(2.0f, 0.0f, 0.0f)));

    const std::vector<std::string> names{"left", "right"};
    const auto originals = capture_visualizer_world_transforms(names);
    const glm::vec3 pivot(0.0f, 0.0f, 0.0f);
    const glm::mat4 delta = glm::translate(glm::mat4(1.0f), pivot) *
                            glm::rotate(glm::mat4(1.0f), 1.57079632679f, glm::vec3(0.0f, 0.0f, 1.0f)) *
                            glm::translate(glm::mat4(1.0f), -pivot);

    apply_group_visualizer_delta(names, originals, delta);

    for (size_t i = 0; i < names.size(); ++i) {
        const glm::mat4 actual =
            lfs::vis::scene_coords::nodeVisualizerWorldTransform(scene_manager_->getScene(), names[i]).value();
        expect_matrix_near(actual, delta * originals[i]);
    }
}

TEST_F(OperatorRegistryPropsTest, MultiNodeGizmoScaleUsesSharedPivot) {
    add_node("near");
    add_node("far");
    scene_manager_->setNodeTransform("near", glm::translate(glm::mat4(1.0f), glm::vec3(1.0f, 0.0f, 0.0f)));
    scene_manager_->setNodeTransform("far", glm::translate(glm::mat4(1.0f), glm::vec3(3.0f, 0.0f, 0.0f)));

    const std::vector<std::string> names{"near", "far"};
    const auto originals = capture_visualizer_world_transforms(names);
    const glm::vec3 pivot(1.0f, 0.0f, 0.0f);
    const glm::mat4 delta = glm::translate(glm::mat4(1.0f), pivot) *
                            glm::scale(glm::mat4(1.0f), glm::vec3(2.0f, 3.0f, 0.5f)) *
                            glm::translate(glm::mat4(1.0f), -pivot);

    apply_group_visualizer_delta(names, originals, delta);

    for (size_t i = 0; i < names.size(); ++i) {
        const glm::mat4 actual =
            lfs::vis::scene_coords::nodeVisualizerWorldTransform(scene_manager_->getScene(), names[i]).value();
        expect_matrix_near(actual, delta * originals[i]);
    }
}

TEST_F(OperatorRegistryPropsTest, MultiNodeGizmoSelectionScaleUsesCombinedVisualizerBounds) {
    add_node("left");
    add_node("right");
    scene_manager_->setNodeTransform("left", glm::translate(glm::mat4(1.0f), glm::vec3(-2.0f, 0.0f, 0.0f)));
    scene_manager_->setNodeTransform("right", glm::translate(glm::mat4(1.0f), glm::vec3(3.0f, 1.0f, 0.0f)));

    const std::vector<std::string> names{"left", "right"};
    glm::vec3 bounds_min, bounds_max;
    ASSERT_TRUE(lfs::vis::gui::gizmo_ops::computeCombinedVisualizerWorldBounds(
        scene_manager_->getScene(), names, bounds_min, bounds_max));

    const auto originals = capture_visualizer_world_transforms(names);
    const glm::vec3 center = (bounds_min + bounds_max) * 0.5f;
    const glm::vec3 scale(2.0f, 1.5f, 0.5f);
    const glm::mat4 delta = glm::translate(glm::mat4(1.0f), center) *
                            glm::scale(glm::mat4(1.0f), scale) *
                            glm::translate(glm::mat4(1.0f), -center);

    apply_group_visualizer_delta(names, originals, delta);

    glm::vec3 scaled_min, scaled_max;
    ASSERT_TRUE(lfs::vis::gui::gizmo_ops::computeCombinedVisualizerWorldBounds(
        scene_manager_->getScene(), names, scaled_min, scaled_max));
    expect_matrix_near(
        glm::translate(glm::mat4(1.0f), scaled_min),
        glm::translate(glm::mat4(1.0f), center + (bounds_min - center) * scale));
    expect_matrix_near(
        glm::translate(glm::mat4(1.0f), scaled_max),
        glm::translate(glm::mat4(1.0f), center + (bounds_max - center) * scale));
}

TEST_F(OperatorRegistryPropsTest, MultiNodeGizmoCombinedBoundsFallbackAndZeroSizeScaleAreStable) {
    auto& scene = scene_manager_->getScene();
    const auto group_id = scene.addGroup("empty_group");
    ASSERT_NE(group_id, lfs::core::NULL_NODE);
    const glm::vec3 group_origin(4.0f, -2.0f, 1.5f);
    scene_manager_->setNodeTransform("empty_group", glm::translate(glm::mat4(1.0f), group_origin));
    const glm::vec3 group_visualizer_origin = glm::vec3(
        lfs::vis::scene_coords::nodeVisualizerWorldTransform(scene, "empty_group").value()[3]);

    glm::vec3 group_min, group_max;
    ASSERT_TRUE(lfs::vis::gui::gizmo_ops::computeCombinedVisualizerWorldBounds(
        scene, {"empty_group"}, group_min, group_max));
    EXPECT_NEAR(group_min.x, group_visualizer_origin.x, 1e-4f);
    EXPECT_NEAR(group_min.y, group_visualizer_origin.y, 1e-4f);
    EXPECT_NEAR(group_min.z, group_visualizer_origin.z, 1e-4f);
    EXPECT_NEAR(group_max.x, group_visualizer_origin.x, 1e-4f);
    EXPECT_NEAR(group_max.y, group_visualizer_origin.y, 1e-4f);
    EXPECT_NEAR(group_max.z, group_visualizer_origin.z, 1e-4f);

    add_node("coincident_a", {0.0f, 0.0f, 0.0f});
    add_node("coincident_b", {0.0f, 0.0f, 0.0f});
    const glm::vec3 coincident_origin(1.0f, 2.0f, -3.0f);
    scene_manager_->setNodeTransform("coincident_a", glm::translate(glm::mat4(1.0f), coincident_origin));
    scene_manager_->setNodeTransform("coincident_b", glm::translate(glm::mat4(1.0f), coincident_origin));
    const glm::vec3 coincident_visualizer_origin = glm::vec3(
        lfs::vis::scene_coords::nodeVisualizerWorldTransform(scene, "coincident_a").value()[3]);

    const std::vector<std::string> names{"coincident_a", "coincident_b"};
    glm::vec3 bounds_min, bounds_max;
    ASSERT_TRUE(lfs::vis::gui::gizmo_ops::computeCombinedVisualizerWorldBounds(
        scene, names, bounds_min, bounds_max));
    expect_matrix_near(glm::translate(glm::mat4(1.0f), bounds_min),
                       glm::translate(glm::mat4(1.0f), coincident_visualizer_origin));
    expect_matrix_near(glm::translate(glm::mat4(1.0f), bounds_max),
                       glm::translate(glm::mat4(1.0f), coincident_visualizer_origin));

    const glm::vec3 safe_bounds = glm::max(bounds_max - bounds_min, glm::vec3(1e-6f));
    const glm::vec3 requested_size(0.25f, 0.5f, 0.75f);
    const glm::vec3 scale_ratio = requested_size / safe_bounds;
    EXPECT_TRUE(std::isfinite(scale_ratio.x));
    EXPECT_TRUE(std::isfinite(scale_ratio.y));
    EXPECT_TRUE(std::isfinite(scale_ratio.z));

    const auto originals = capture_visualizer_world_transforms(names);
    const glm::vec3 center = (bounds_min + bounds_max) * 0.5f;
    const glm::mat4 delta = glm::translate(glm::mat4(1.0f), center) *
                            glm::scale(glm::mat4(1.0f), scale_ratio) *
                            glm::translate(glm::mat4(1.0f), -center);

    apply_group_visualizer_delta(names, originals, delta);

    for (const auto& name : names) {
        const glm::mat4 actual =
            lfs::vis::scene_coords::nodeVisualizerWorldTransform(scene, name).value();
        expect_matrix_finite(actual);
        EXPECT_NEAR(actual[3].x, coincident_visualizer_origin.x, 1e-4f);
        EXPECT_NEAR(actual[3].y, coincident_visualizer_origin.y, 1e-4f);
        EXPECT_NEAR(actual[3].z, coincident_visualizer_origin.z, 1e-4f);
    }
}

TEST_F(OperatorRegistryPropsTest, MultiNodeGizmoIndividualRotateUsesEachNodeOrigin) {
    add_node("left");
    add_node("right");
    scene_manager_->setNodeTransform("left", glm::translate(glm::mat4(1.0f), glm::vec3(-2.0f, 0.0f, 0.0f)));
    scene_manager_->setNodeTransform("right", glm::translate(glm::mat4(1.0f), glm::vec3(2.0f, 1.0f, 0.0f)));

    const std::vector<std::string> names{"left", "right"};
    const auto originals = capture_visualizer_world_transforms(names);
    const glm::mat4 local_delta =
        glm::rotate(glm::mat4(1.0f), 1.57079632679f, glm::vec3(0.0f, 0.0f, 1.0f));

    apply_individual_visualizer_delta(names, originals, local_delta);

    for (size_t i = 0; i < names.size(); ++i) {
        const glm::mat4 actual =
            lfs::vis::scene_coords::nodeVisualizerWorldTransform(scene_manager_->getScene(), names[i]).value();
        expect_matrix_near(actual, originals[i] * local_delta);
        EXPECT_NEAR(actual[3].x, originals[i][3].x, 1e-4f);
        EXPECT_NEAR(actual[3].y, originals[i][3].y, 1e-4f);
        EXPECT_NEAR(actual[3].z, originals[i][3].z, 1e-4f);
    }
}

TEST_F(OperatorRegistryPropsTest, MultiNodeGizmoIndividualRotatePreservesDragCompositionOrder) {
    add_node("tilted");
    add_node("offset");
    scene_manager_->setNodeTransform(
        "tilted",
        glm::translate(glm::mat4(1.0f), glm::vec3(-1.0f, 0.5f, 2.0f)) *
            glm::rotate(glm::mat4(1.0f), 0.35f, glm::vec3(0.0f, 0.0f, 1.0f)));
    scene_manager_->setNodeTransform(
        "offset",
        glm::translate(glm::mat4(1.0f), glm::vec3(2.0f, -1.0f, 0.25f)) *
            glm::rotate(glm::mat4(1.0f), -0.45f, glm::vec3(0.0f, 1.0f, 0.0f)));

    const std::vector<std::string> names{"tilted", "offset"};
    const auto originals = capture_visualizer_world_transforms(names);
    const glm::mat4 delta_x =
        glm::rotate(glm::mat4(1.0f), 0.55f, glm::vec3(1.0f, 0.0f, 0.0f));
    const glm::mat4 delta_y =
        glm::rotate(glm::mat4(1.0f), -0.40f, glm::vec3(0.0f, 1.0f, 0.0f));
    const glm::mat4 cumulative_drag_delta = delta_y * delta_x;
    const glm::mat4 opposite_order_delta = delta_x * delta_y;

    apply_individual_visualizer_delta(names, originals, cumulative_drag_delta);

    for (size_t i = 0; i < names.size(); ++i) {
        const glm::mat4 actual =
            lfs::vis::scene_coords::nodeVisualizerWorldTransform(scene_manager_->getScene(), names[i]).value();
        expect_matrix_near(actual, originals[i] * cumulative_drag_delta);
        EXPECT_GT(matrix_max_abs_diff(actual, originals[i] * opposite_order_delta), 1e-3f);
        EXPECT_NEAR(actual[3].x, originals[i][3].x, 1e-4f);
        EXPECT_NEAR(actual[3].y, originals[i][3].y, 1e-4f);
        EXPECT_NEAR(actual[3].z, originals[i][3].z, 1e-4f);
    }
}

TEST_F(OperatorRegistryPropsTest, MultiNodeGizmoIndividualScaleUsesEachNodeOrigin) {
    add_node("near");
    add_node("far");
    scene_manager_->setNodeTransform("near", glm::translate(glm::mat4(1.0f), glm::vec3(1.0f, 0.0f, 0.0f)));
    scene_manager_->setNodeTransform("far", glm::translate(glm::mat4(1.0f), glm::vec3(3.0f, 2.0f, 0.0f)));

    const std::vector<std::string> names{"near", "far"};
    const auto originals = capture_visualizer_world_transforms(names);
    const glm::mat4 local_delta = glm::scale(glm::mat4(1.0f), glm::vec3(2.0f, 3.0f, 0.5f));

    apply_individual_visualizer_delta(names, originals, local_delta);

    for (size_t i = 0; i < names.size(); ++i) {
        const glm::mat4 actual =
            lfs::vis::scene_coords::nodeVisualizerWorldTransform(scene_manager_->getScene(), names[i]).value();
        expect_matrix_near(actual, originals[i] * local_delta);
        EXPECT_NEAR(actual[3].x, originals[i][3].x, 1e-4f);
        EXPECT_NEAR(actual[3].y, originals[i][3].y, 1e-4f);
        EXPECT_NEAR(actual[3].z, originals[i][3].z, 1e-4f);
    }
}

TEST_F(OperatorRegistryPropsTest, MultiNodeGizmoFiltersSelectedDescendants) {
    auto& scene = scene_manager_->getScene();
    const auto parent_id = scene.addGroup("group");
    ASSERT_NE(parent_id, lfs::core::NULL_NODE);
    const auto child_id = scene.addSplat(
        "child",
        make_test_splat({0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f}),
        parent_id);
    ASSERT_NE(child_id, lfs::core::NULL_NODE);

    scene_manager_->setNodeTransform("group", glm::translate(glm::mat4(1.0f), glm::vec3(2.0f, 0.0f, 0.0f)));
    scene_manager_->setNodeTransform("child", glm::translate(glm::mat4(1.0f), glm::vec3(1.0f, 0.0f, 0.0f)));

    const std::vector<std::string> selected_names{"group", "child"};
    const auto top_level =
        lfs::vis::gui::gizmo_ops::topLevelTransformTargets(scene_manager_->getScene(), selected_names);
    ASSERT_EQ(top_level, (std::vector<std::string>{"group"}));

    const auto group_originals = capture_visualizer_world_transforms(top_level);
    const glm::mat4 child_original =
        lfs::vis::scene_coords::nodeVisualizerWorldTransform(scene_manager_->getScene(), "child").value();
    const glm::mat4 delta = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 5.0f, 0.0f));

    apply_group_visualizer_delta(top_level, group_originals, delta);

    const glm::mat4 child_actual =
        lfs::vis::scene_coords::nodeVisualizerWorldTransform(scene_manager_->getScene(), "child").value();
    expect_matrix_near(child_actual, delta * child_original);
}

TEST_F(OperatorRegistryPropsTest, MultiNodeGizmoRequiresAllTargetsEditable) {
    add_node("editable");
    add_node("locked");
    scene_manager_->getScene().setNodeLocked("locked", true);
    scene_manager_->selectNodes({"editable", "locked"});

    const auto result = lfs::vis::cap::resolveEditableTransformSelection(
        *scene_manager_, std::nullopt, lfs::vis::cap::TransformTargetPolicy::RequireAllEditable);

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), "selection contains locked nodes");
}

TEST_F(OperatorRegistryPropsTest, MultiNodeGizmoInvalidModeFallsBackToSelection) {
    EXPECT_EQ(
        lfs::vis::gui::normalizeMultiTransformMode(lfs::vis::gui::MultiTransformMode::Individual),
        lfs::vis::gui::MultiTransformMode::Individual);
    EXPECT_EQ(
        lfs::vis::gui::normalizeMultiTransformMode(static_cast<lfs::vis::gui::MultiTransformMode>(99)),
        lfs::vis::gui::MultiTransformMode::Selection);
}

TEST_F(OperatorRegistryPropsTest, MultiNodeGizmoTransformHelpersRejectMismatchedSnapshots) {
    add_node("first");
    add_node("second");
    scene_manager_->setNodeTransform("first", glm::translate(glm::mat4(1.0f), glm::vec3(1.0f, 0.0f, 0.0f)));
    scene_manager_->setNodeTransform("second", glm::translate(glm::mat4(1.0f), glm::vec3(2.0f, 0.0f, 0.0f)));

    const std::vector<std::string> names{"first", "second"};
    const auto originals = capture_visualizer_world_transforms({"first"});
    const glm::mat4 delta = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 3.0f, 0.0f));

    const auto selection_results = lfs::vis::gui::gizmo_ops::computeNodeSharedSelectionLocalTransforms(
        scene_manager_->getScene(), names, originals, delta);
    const auto individual_results = lfs::vis::gui::gizmo_ops::computeNodeIndividualLocalTransforms(
        scene_manager_->getScene(), names, originals, delta);

    EXPECT_TRUE(selection_results.empty());
    EXPECT_TRUE(individual_results.empty());
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
    ASSERT_NE(scene_manager_->getScene().addPlySequence("sequence"), lfs::core::NULL_NODE);
    scene_manager_->selectNodes({"editable", "sequence"});

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

TEST_F(OperatorRegistryPropsTest, LegacySelectInvertUsesVisibleMaskWithHiddenSibling) {
    add_node("original");
    add_node("copy");

    const auto original_id = scene_manager_->getScene().getNodeIdByName("original");
    ASSERT_NE(original_id, lfs::core::NULL_NODE);
    scene_manager_->setNodeVisibility(original_id, false);
    scene_manager_->getScene().setSelectionMask(
        std::make_shared<Tensor>(make_uint8_mask({0, 0, 1, 0})));

    lfs::vis::op::SelectInvert op;
    lfs::vis::op::OperatorProperties props;

    const auto result = op.execute(*scene_manager_, props, {});

    ASSERT_TRUE(result.ok());
    EXPECT_EQ(selection_mask_values(scene_manager_->getScene()), (std::vector<uint8_t>{0, 0, 0, 1}));
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

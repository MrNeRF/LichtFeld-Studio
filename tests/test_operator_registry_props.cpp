/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/event_bridge/event_bridge.hpp"
#include "core/event_bus.hpp"
#include "core/services.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "operator/operator_properties.hpp"
#include "operator/operator_registry.hpp"
#include "operator/property_schema.hpp"
#include "operation/undo_history.hpp"
#include "operator/ops/edit_ops.hpp"
#include "operator/ops/transform_ops.hpp"
#include "rendering/rendering_manager.hpp"
#include "scene/scene_manager.hpp"
#include "visualizer/gui_capabilities.hpp"

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

    void add_node(const std::string& name) {
        scene_manager_->getScene().addNode(
            name,
            make_test_splat({
                0.0f,
                0.0f,
                0.0f,
                1.0f,
                0.0f,
                0.0f,
            }));
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

TEST_F(OperatorRegistryPropsTest, TransformTranslateOperatorUsesNamedNodeWithoutSelection) {
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
    EXPECT_FLOAT_EQ(components.translation.y, 2.0f);
    EXPECT_FLOAT_EQ(components.translation.z, 3.0f);

    const auto resolved = props.get<std::vector<std::string>>("resolved_node_names");
    ASSERT_TRUE(resolved.has_value());
    ASSERT_EQ(resolved->size(), 1u);
    EXPECT_EQ(resolved->front(), "move_me");
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

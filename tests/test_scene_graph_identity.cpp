/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/event_bridge/event_bridge.hpp"
#include "core/event_bus.hpp"
#include "core/events.hpp"
#include "core/services.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "operation/undo_history.hpp"
#include "rendering/rendering_manager.hpp"
#include "scene/scene_manager.hpp"

#include <algorithm>
#include <atomic>
#include <gtest/gtest.h>
#include <memory>
#include <string>
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

class SceneGraphIdentityTest : public ::testing::Test {
protected:
    void SetUp() override {
        lfs::event::EventBridge::instance().clear_all();
        lfs::core::event::bus().clear_all();
        lfs::vis::services().clear();
        lfs::vis::op::undoHistory().clear();

        scene_manager_ = std::make_unique<lfs::vis::SceneManager>();
        rendering_manager_ = std::make_unique<lfs::vis::RenderingManager>();
        lfs::vis::services().set(scene_manager_.get());
        lfs::vis::services().set(rendering_manager_.get());
    }

    void TearDown() override {
        lfs::vis::op::undoHistory().clear();
        lfs::vis::services().clear();
        lfs::core::event::bus().clear_all();
        lfs::event::EventBridge::instance().clear_all();
        rendering_manager_.reset();
        scene_manager_.reset();
    }

    std::unique_ptr<lfs::vis::SceneManager> scene_manager_;
    std::unique_ptr<lfs::vis::RenderingManager> rendering_manager_;
};

// Catches PLYRemoved reading its name from the node-owned string that removeNode just destroyed.
TEST_F(SceneGraphIdentityTest, DeleteGroupByIdEmitsRemovedEventsWithCorrectNames) {
    auto& scene = scene_manager_->getScene();
    const auto group_id = scene.addGroup("group");
    const auto child_a_id = scene.addSplat("child_a", make_test_splat({0.0f, 0.0f, 0.0f}), group_id);
    const auto child_b_id = scene.addSplat("child_b", make_test_splat({1.0f, 0.0f, 0.0f}), group_id);
    ASSERT_NE(group_id, lfs::core::NULL_NODE);
    ASSERT_NE(child_a_id, lfs::core::NULL_NODE);
    ASSERT_NE(child_b_id, lfs::core::NULL_NODE);
    scene_manager_->changeContentType(lfs::vis::SceneManager::ContentType::SplatFiles);

    std::vector<std::string> removed_names;
    lfs::core::events::state::PLYRemoved::when(
        [&removed_names](const auto& event) { removed_names.push_back(event.name); });

    scene_manager_->removeNode(group_id, false);

    EXPECT_EQ(removed_names, (std::vector<std::string>{"group"}));
    EXPECT_TRUE(std::none_of(
        removed_names.begin(), removed_names.end(), [](const std::string& name) { return name.empty(); }));
    EXPECT_EQ(scene.getNodeById(group_id), nullptr);
    EXPECT_EQ(scene.getNodeById(child_a_id), nullptr);
    EXPECT_EQ(scene.getNodeById(child_b_id), nullptr);
}

// Catches PLYRemoved reading its name from the node-owned string that removeNode just destroyed.
TEST_F(SceneGraphIdentityTest, DeleteGroupByIdKeepChildrenEmitsCorrectName) {
    auto& scene = scene_manager_->getScene();
    const auto parent_id = scene.addGroup("parent");
    const auto group_id = scene.addGroup("group", parent_id);
    const auto child_a_id = scene.addSplat("child_a", make_test_splat({0.0f, 0.0f, 0.0f}), group_id);
    const auto child_b_id = scene.addSplat("child_b", make_test_splat({1.0f, 0.0f, 0.0f}), group_id);
    ASSERT_NE(parent_id, lfs::core::NULL_NODE);
    ASSERT_NE(group_id, lfs::core::NULL_NODE);
    ASSERT_NE(child_a_id, lfs::core::NULL_NODE);
    ASSERT_NE(child_b_id, lfs::core::NULL_NODE);
    scene_manager_->changeContentType(lfs::vis::SceneManager::ContentType::SplatFiles);

    std::vector<std::string> removed_names;
    lfs::core::events::state::PLYRemoved::when(
        [&removed_names](const auto& event) { removed_names.push_back(event.name); });

    scene_manager_->removeNode(group_id, true);

    EXPECT_EQ(removed_names, (std::vector<std::string>{"group"}));
    EXPECT_EQ(scene.getNodeById(group_id), nullptr);
    const auto* child_a = scene.getNodeById(child_a_id);
    const auto* child_b = scene.getNodeById(child_b_id);
    ASSERT_NE(child_a, nullptr);
    ASSERT_NE(child_b, nullptr);
    EXPECT_EQ(child_a->parent_id, parent_id);
    EXPECT_EQ(child_b->parent_id, parent_id);
}

// Catches MergeGroupById passing a node-owned name into code that destroys its node (issue #1458).
TEST_F(SceneGraphIdentityTest, MergeGroupByIdCommandCreatesNamedSplat) {
    auto& scene = scene_manager_->getScene();
    const auto group_id = scene.addGroup("group");
    const auto child_a_id = scene.addSplat("child_a", make_test_splat({0.0f, 0.0f, 0.0f}), group_id);
    const auto child_b_id = scene.addSplat(
        "child_b",
        make_test_splat({
            1.0f,
            0.0f,
            0.0f,
            2.0f,
            0.0f,
            0.0f,
        }),
        group_id);
    ASSERT_NE(group_id, lfs::core::NULL_NODE);
    ASSERT_NE(child_a_id, lfs::core::NULL_NODE);
    ASSERT_NE(child_b_id, lfs::core::NULL_NODE);
    scene_manager_->changeContentType(lfs::vis::SceneManager::ContentType::SplatFiles);

    lfs::core::events::cmd::MergeGroupById{.node_id = group_id}.emit();

    const auto* merged = scene.getNode("group");
    ASSERT_NE(merged, nullptr);
    EXPECT_EQ(merged->type, lfs::core::NodeType::SPLAT);
    EXPECT_EQ(merged->gaussian_count.load(std::memory_order_acquire), 3u);
    EXPECT_EQ(scene.getTotalGaussianCount(), 3u);
    EXPECT_EQ(scene.getNodeById(child_a_id), nullptr);
    EXPECT_EQ(scene.getNodeById(child_b_id), nullptr);
}

// Catches nested merge retaining the outer group's node-owned name across its destruction.
TEST_F(SceneGraphIdentityTest, MergeNestedGroupsKeepsOuterName) {
    auto& scene = scene_manager_->getScene();
    const auto outer_id = scene.addGroup("outer");
    const auto inner_id = scene.addGroup("inner", outer_id);
    const auto child_a_id = scene.addSplat("child_a", make_test_splat({0.0f, 0.0f, 0.0f}), inner_id);
    const auto child_b_id = scene.addSplat(
        "child_b",
        make_test_splat({
            1.0f,
            0.0f,
            0.0f,
            2.0f,
            0.0f,
            0.0f,
        }),
        inner_id);
    ASSERT_NE(outer_id, lfs::core::NULL_NODE);
    ASSERT_NE(inner_id, lfs::core::NULL_NODE);
    ASSERT_NE(child_a_id, lfs::core::NULL_NODE);
    ASSERT_NE(child_b_id, lfs::core::NULL_NODE);
    scene_manager_->changeContentType(lfs::vis::SceneManager::ContentType::SplatFiles);

    const std::string merged_name = scene_manager_->mergeGroupNode(outer_id);

    EXPECT_EQ(merged_name, "outer");
    const auto* merged = scene.getNode("outer");
    ASSERT_NE(merged, nullptr);
    EXPECT_EQ(merged->type, lfs::core::NodeType::SPLAT);
    EXPECT_EQ(merged->gaussian_count.load(std::memory_order_acquire), 3u);
    EXPECT_EQ(scene.getTotalGaussianCount(), 3u);
    EXPECT_EQ(scene.getNode("inner"), nullptr);
    EXPECT_EQ(scene.getNodeById(inner_id), nullptr);
    EXPECT_EQ(scene.getNodeById(child_a_id), nullptr);
    EXPECT_EQ(scene.getNodeById(child_b_id), nullptr);
    EXPECT_EQ(scene.getNodeCount(), 1u);
}

// Catches public NodeId entry points asserting on NULL_NODE instead of no-opping.
TEST_F(SceneGraphIdentityTest, NullNodeIdOperationsAreNoOps) {
    auto& scene = scene_manager_->getScene();
    const auto group_id = scene.addGroup("group");
    const auto child_a_id = scene.addSplat("child_a", make_test_splat({0.0f, 0.0f, 0.0f}), group_id);
    const auto child_b_id = scene.addSplat("child_b", make_test_splat({1.0f, 0.0f, 0.0f}), group_id);
    ASSERT_NE(group_id, lfs::core::NULL_NODE);
    ASSERT_NE(child_a_id, lfs::core::NULL_NODE);
    ASSERT_NE(child_b_id, lfs::core::NULL_NODE);
    const size_t node_count = scene.getNodeCount();

    EXPECT_TRUE(scene_manager_->mergeGroupNode(lfs::core::NULL_NODE).empty());
    EXPECT_TRUE(scene_manager_->duplicateNodeTree(lfs::core::NULL_NODE).empty());
    EXPECT_FALSE(scene_manager_->renameNode(lfs::core::NULL_NODE, "x"));
    scene_manager_->removeNode(lfs::core::NULL_NODE, false);

    EXPECT_EQ(scene.getNodeCount(), node_count);
    const auto* group = scene.getNodeById(group_id);
    const auto* child_a = scene.getNodeById(child_a_id);
    const auto* child_b = scene.getNodeById(child_b_id);
    ASSERT_NE(group, nullptr);
    ASSERT_NE(child_a, nullptr);
    ASSERT_NE(child_b, nullptr);
    EXPECT_EQ(group->children, (std::vector<lfs::core::NodeId>{child_a_id, child_b_id}));
    EXPECT_EQ(child_a->parent_id, group_id);
    EXPECT_EQ(child_b->parent_id, group_id);
    EXPECT_EQ(scene.getNode("x"), nullptr);
}

// Catches public NodeId entry points asserting on or dereferencing a stale ID instead of no-opping.
TEST_F(SceneGraphIdentityTest, StaleNodeIdOperationsAreNoOps) {
    auto& scene = scene_manager_->getScene();
    const auto group_id = scene.addGroup("group");
    const auto stale_id = scene.addSplat("child_a", make_test_splat({0.0f, 0.0f, 0.0f}), group_id);
    const auto child_b_id = scene.addSplat("child_b", make_test_splat({1.0f, 0.0f, 0.0f}), group_id);
    ASSERT_NE(group_id, lfs::core::NULL_NODE);
    ASSERT_NE(stale_id, lfs::core::NULL_NODE);
    ASSERT_NE(child_b_id, lfs::core::NULL_NODE);
    scene_manager_->changeContentType(lfs::vis::SceneManager::ContentType::SplatFiles);

    scene_manager_->removeNode(stale_id, false);
    ASSERT_EQ(scene.getNodeById(stale_id), nullptr);
    const size_t node_count = scene.getNodeCount();

    EXPECT_TRUE(scene_manager_->mergeGroupNode(stale_id).empty());
    EXPECT_TRUE(scene_manager_->duplicateNodeTree(stale_id).empty());
    scene_manager_->removeNode(stale_id, false);

    EXPECT_EQ(scene.getNodeCount(), node_count);
    EXPECT_EQ(scene.getNodeById(stale_id), nullptr);
    const auto* group = scene.getNodeById(group_id);
    const auto* child_b = scene.getNodeById(child_b_id);
    ASSERT_NE(group, nullptr);
    ASSERT_NE(child_b, nullptr);
    EXPECT_EQ(group->children, (std::vector<lfs::core::NodeId>{child_b_id}));
    EXPECT_EQ(child_b->parent_id, group_id);
}

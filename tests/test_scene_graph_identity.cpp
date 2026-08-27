/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/camera.hpp"
#include "core/event_bridge/event_bridge.hpp"
#include "core/event_bus.hpp"
#include "core/events.hpp"
#include "core/path_utils.hpp"
#include "core/scene.hpp"
#include "core/services.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "operation/undo_history.hpp"
#include "rendering/rendering_manager.hpp"
#include "scene/scene_manager.hpp"

#include <algorithm>
#include <atomic>
#include <cuda_runtime.h>
#include <filesystem>
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <vector>

using lfs::core::DataType;
using lfs::core::Device;
using lfs::core::Tensor;

namespace {

    class ScopedPlyRemovedSubscription {
    public:
        explicit ScopedPlyRemovedSubscription(const lfs::event::HandlerId id) : id_(id) {}
        ~ScopedPlyRemovedSubscription() { reset(); }

        ScopedPlyRemovedSubscription(const ScopedPlyRemovedSubscription&) = delete;
        ScopedPlyRemovedSubscription& operator=(const ScopedPlyRemovedSubscription&) = delete;

        void reset() {
            if (id_ == 0) {
                return;
            }
            lfs::event::EventBridge::instance().unsubscribe(
                typeid(lfs::core::events::state::PLYRemoved), id_);
            id_ = 0;
        }

    private:
        lfs::event::HandlerId id_;
    };

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

// Locks the removed node's real event name and recursive subtree deletion; the underlying
// use-after-free requires a lifetime sanitizer to detect directly.
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
    ScopedPlyRemovedSubscription subscription{
        lfs::core::events::state::PLYRemoved::when(
            [&removed_names](const auto& event) { removed_names.push_back(event.name); })};

    scene_manager_->removeNode(group_id, false);
    subscription.reset();

    EXPECT_EQ(removed_names, (std::vector<std::string>{"group"}));
    EXPECT_TRUE(std::none_of(
        removed_names.begin(), removed_names.end(), [](const std::string& name) { return name.empty(); }));
    EXPECT_EQ(scene.getNodeById(group_id), nullptr);
    EXPECT_EQ(scene.getNodeById(child_a_id), nullptr);
    EXPECT_EQ(scene.getNodeById(child_b_id), nullptr);
}

// Locks the removed node's real event name and child promotion; the underlying use-after-free
// requires a lifetime sanitizer to detect directly.
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
    ScopedPlyRemovedSubscription subscription{
        lfs::core::events::state::PLYRemoved::when(
            [&removed_names](const auto& event) { removed_names.push_back(event.name); })};

    scene_manager_->removeNode(group_id, true);
    subscription.reset();

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

// Catches Scene::clear resetting the allocator so stale IDs alias newly created nodes.
TEST_F(SceneGraphIdentityTest, NodeIdsAreNotRecycledAcrossSceneClear) {
    auto& scene = scene_manager_->getScene();
    const std::vector<lfs::core::NodeId> old_ids{
        scene.addGroup("old_a"),
        scene.addGroup("old_b"),
    };
    ASSERT_NE(old_ids[0], lfs::core::NULL_NODE);
    ASSERT_NE(old_ids[1], lfs::core::NULL_NODE);

    scene.clear();

    const std::vector<lfs::core::NodeId> new_ids{
        scene.addGroup("new_a"),
        scene.addGroup("new_b"),
    };
    ASSERT_NE(new_ids[0], lfs::core::NULL_NODE);
    ASSERT_NE(new_ids[1], lfs::core::NULL_NODE);
    for (const auto new_id : new_ids) {
        EXPECT_EQ(std::find(old_ids.begin(), old_ids.end(), new_id), old_ids.end());
    }
}

// Catches source-path selection following lexicographic node names instead of node creation order.
TEST_F(SceneGraphIdentityTest, SceneInfoSourcePathIsMostRecentlyAddedNode) {
    auto& scene = scene_manager_->getScene();
    const auto older_id = scene.addSplat("z_older", make_test_splat({0.0f, 0.0f, 0.0f}));
    const auto newer_id = scene.addSplat("a_newer", make_test_splat({1.0f, 0.0f, 0.0f}));
    ASSERT_NE(older_id, lfs::core::NULL_NODE);
    ASSERT_NE(newer_id, lfs::core::NULL_NODE);
    const std::filesystem::path older_path{"/tmp/older.ply"};
    const std::filesystem::path newer_path{"/tmp/newer.sog"};
    scene_manager_->setPlyPath(older_id, older_path);
    scene_manager_->setPlyPath(newer_id, newer_path);
    scene_manager_->changeContentType(lfs::vis::SceneManager::ContentType::SplatFiles);

    const auto info = scene_manager_->getSceneInfo();

    EXPECT_EQ(info.source_path, newer_path);
    EXPECT_EQ(info.source_type, "SOG");
}

// Catches group merge leaving source paths keyed by IDs from the destroyed subtree.
TEST_F(SceneGraphIdentityTest, MergeClearsDescendantPlyPaths) {
    auto& scene = scene_manager_->getScene();
    const auto group_id = scene.addGroup("group");
    const auto child_a_id = scene.addSplat("child_a", make_test_splat({0.0f, 0.0f, 0.0f}), group_id);
    const auto child_b_id = scene.addSplat("child_b", make_test_splat({1.0f, 0.0f, 0.0f}), group_id);
    ASSERT_NE(group_id, lfs::core::NULL_NODE);
    ASSERT_NE(child_a_id, lfs::core::NULL_NODE);
    ASSERT_NE(child_b_id, lfs::core::NULL_NODE);
    scene_manager_->setPlyPath(child_a_id, std::filesystem::path{"/tmp/child_a.ply"});
    scene_manager_->setPlyPath(child_b_id, std::filesystem::path{"/tmp/child_b.ply"});
    scene_manager_->changeContentType(lfs::vis::SceneManager::ContentType::SplatFiles);

    ASSERT_EQ(scene_manager_->mergeGroupNode(group_id), "group");

    EXPECT_FALSE(scene_manager_->getPlyPath(child_a_id).has_value());
    EXPECT_FALSE(scene_manager_->getPlyPath(child_b_id).has_value());
    EXPECT_TRUE(scene_manager_->getSceneInfo().source_path.empty());
}

// Catches keep-children removal erasing source paths owned by the promoted child nodes.
TEST_F(SceneGraphIdentityTest, KeepChildrenRemovalPreservesChildPlyPaths) {
    auto& scene = scene_manager_->getScene();
    const auto group_id = scene.addGroup("group");
    const auto child_a_id = scene.addSplat("child_a", make_test_splat({0.0f, 0.0f, 0.0f}), group_id);
    const auto child_b_id = scene.addSplat("child_b", make_test_splat({1.0f, 0.0f, 0.0f}), group_id);
    ASSERT_NE(group_id, lfs::core::NULL_NODE);
    ASSERT_NE(child_a_id, lfs::core::NULL_NODE);
    ASSERT_NE(child_b_id, lfs::core::NULL_NODE);
    const std::filesystem::path child_a_path{"/tmp/child_a.ply"};
    const std::filesystem::path child_b_path{"/tmp/child_b.ply"};
    scene_manager_->setPlyPath(child_a_id, child_a_path);
    scene_manager_->setPlyPath(child_b_id, child_b_path);
    scene_manager_->changeContentType(lfs::vis::SceneManager::ContentType::SplatFiles);

    scene_manager_->removeNode(group_id, true);

    EXPECT_EQ(scene.getNodeById(group_id), nullptr);
    ASSERT_NE(scene.getNodeById(child_a_id), nullptr);
    ASSERT_NE(scene.getNodeById(child_b_id), nullptr);
    EXPECT_EQ(scene_manager_->getPlyPath(child_a_id), child_a_path);
    EXPECT_EQ(scene_manager_->getPlyPath(child_b_id), child_b_path);
}

// Catches the by-name merge overload retaining a node-owned label across subtree destruction.
TEST_F(SceneGraphIdentityTest, SceneMergeGroupByNameMergesSubtree) {
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

    const std::string merged_name = scene.mergeGroup(std::string{"group"});

    EXPECT_EQ(merged_name, "group");
    const auto* merged = scene.getNode("group");
    ASSERT_NE(merged, nullptr);
    EXPECT_EQ(merged->type, lfs::core::NodeType::SPLAT);
    EXPECT_EQ(merged->gaussian_count.load(std::memory_order_acquire), 3u);
    EXPECT_EQ(scene.getTotalGaussianCount(), 3u);
    EXPECT_EQ(scene.getNodeById(group_id), nullptr);
    EXPECT_EQ(scene.getNodeById(child_a_id), nullptr);
    EXPECT_EQ(scene.getNodeById(child_b_id), nullptr);
    EXPECT_EQ(scene.getNodeCount(), 1u);
}

// Catches path metadata being stranded under a node's old mutable label after rename.
TEST_F(SceneGraphIdentityTest, PlyPathSurvivesRename) {
    auto& scene = scene_manager_->getScene();
    const auto node_id = scene.addSplat("node", make_test_splat({0.0f, 0.0f, 0.0f}));
    ASSERT_NE(node_id, lfs::core::NULL_NODE);
    const std::filesystem::path source_path{"/tmp/a.ply"};
    scene_manager_->setPlyPath("node", source_path);

    ASSERT_TRUE(scene_manager_->renameNode(node_id, "renamed"));

    const auto renamed_path = scene_manager_->getPlyPath("renamed");
    ASSERT_TRUE(renamed_path.has_value());
    EXPECT_EQ(*renamed_path, source_path);
}

// Catches movePlyPath erasing another node's path when the renamed source has no path.
TEST_F(SceneGraphIdentityTest, RenameOntoForeignNameDoesNotDropItsPath) {
    auto& scene = scene_manager_->getScene();
    const auto source_id = scene.addSplat("source", make_test_splat({0.0f, 0.0f, 0.0f}));
    const auto owner_id = scene.addSplat("foreign", make_test_splat({1.0f, 0.0f, 0.0f}));
    ASSERT_NE(source_id, lfs::core::NULL_NODE);
    ASSERT_NE(owner_id, lfs::core::NULL_NODE);
    const std::filesystem::path foreign_path{"/tmp/foreign.ply"};
    scene_manager_->setPlyPath("foreign", foreign_path);
    scene_manager_->changeContentType(lfs::vis::SceneManager::ContentType::SplatFiles);

    ASSERT_TRUE(scene.renameNode(owner_id, "owner"));
    ASSERT_EQ(scene_manager_->getSceneInfo().source_path, foreign_path);
    ASSERT_TRUE(scene_manager_->renameNode(source_id, "foreign"));

    EXPECT_EQ(scene_manager_->getSceneInfo().source_path, foreign_path);
    const auto owner_path = scene_manager_->getPlyPath("owner");
    ASSERT_TRUE(owner_path.has_value());
    EXPECT_EQ(*owner_path, foreign_path);
}

// Catches removal leaving path metadata reachable through the deleted node's stale ID.
TEST_F(SceneGraphIdentityTest, PlyPathClearedWhenNodeRemoved) {
    auto& scene = scene_manager_->getScene();
    const auto anchor_id = scene.addGroup("anchor");
    const auto removed_id = scene.addSplat("node", make_test_splat({0.0f, 0.0f, 0.0f}));
    ASSERT_NE(anchor_id, lfs::core::NULL_NODE);
    ASSERT_NE(removed_id, lfs::core::NULL_NODE);
    scene_manager_->setPlyPath("node", std::filesystem::path{"/tmp/a.ply"});
    scene_manager_->changeContentType(lfs::vis::SceneManager::ContentType::SplatFiles);

    scene_manager_->removeNode(removed_id, false);

    ASSERT_EQ(scene.getNodeById(removed_id), nullptr);
    EXPECT_FALSE(scene_manager_->getPlyPath(removed_id).has_value());
    const auto replacement_id = scene.addSplat("node", make_test_splat({1.0f, 0.0f, 0.0f}));
    ASSERT_NE(replacement_id, lfs::core::NULL_NODE);
    EXPECT_NE(replacement_id, removed_id);
    EXPECT_FALSE(scene_manager_->getPlyPath("node").has_value());
}

// Catches topology undo restoring a path to a fresh node ID while retaining the destroyed ID's entry.
TEST_F(SceneGraphIdentityTest, PlyPathRestoredToFreshIdByTopologyUndo) {
    auto& scene = scene_manager_->getScene();
    const auto anchor_id = scene.addGroup("anchor");
    const auto removed_id = scene.addSplat("node", make_test_splat({0.0f, 0.0f, 0.0f}));
    ASSERT_NE(anchor_id, lfs::core::NULL_NODE);
    ASSERT_NE(removed_id, lfs::core::NULL_NODE);
    const std::filesystem::path source_path{"/tmp/a.ply"};
    scene_manager_->setPlyPath(removed_id, source_path);
    scene_manager_->changeContentType(lfs::vis::SceneManager::ContentType::SplatFiles);

    scene_manager_->removeNode(removed_id, false);
    ASSERT_EQ(scene.getNodeById(removed_id), nullptr);

    const auto undo_result = lfs::vis::op::undoHistory().undo();

    ASSERT_TRUE(undo_result.success);
    const auto* restored = scene.getNode("node");
    ASSERT_NE(restored, nullptr);
    EXPECT_NE(restored->id, removed_id);
    EXPECT_FALSE(scene_manager_->getPlyPath(removed_id).has_value());
    const auto restored_path = scene_manager_->getPlyPath(restored->id);
    ASSERT_TRUE(restored_path.has_value());
    EXPECT_EQ(*restored_path, source_path);
}

TEST(SceneCameraAssetPathTest, RebaseRewritesUnderOldRootAndRefreshesNodeMirrors) {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
        GTEST_SKIP() << "CUDA device unavailable";
    }

    const auto old_root = std::filesystem::temp_directory_path() / "lfs-rebase-old";
    const auto new_root = std::filesystem::temp_directory_path() / "lfs-rebase-new";
    const auto outside = std::filesystem::temp_directory_path() / "lfs-rebase-elsewhere" / "photo.jpg";

    auto make_camera = [](const std::string& name,
                          const std::filesystem::path& image,
                          const std::filesystem::path& mask,
                          const std::filesystem::path& depth,
                          const std::filesystem::path& normal,
                          const int uid) {
        return std::make_shared<lfs::core::Camera>(
            Tensor::eye(3, Device::CPU),
            Tensor::zeros({3}, Device::CPU),
            100.0f, 100.0f, 32.0f, 32.0f,
            Tensor(), Tensor(), lfs::core::CameraModelType::PINHOLE,
            name, image, mask, 64, 64, uid, 0, depth, normal);
    };

    lfs::core::Scene scene;
    const auto group = scene.addCameraGroup("Training", scene.addGroup("Cameras"), 2);
    scene.addCamera(
        "inside.png", group,
        make_camera("inside.png",
                    old_root / "images" / "inside.png",
                    old_root / "masks" / "inside.png",
                    old_root / "depth" / "inside.exr",
                    old_root / "normals" / "inside.png",
                    1));
    scene.addCamera(
        "outside.png", group,
        make_camera("outside.png", outside, {}, {}, {}, 2));

    EXPECT_EQ(scene.rebaseCameraAssetPaths(old_root / "", new_root), 2u);

    const auto cameras = scene.getAllCameras();
    ASSERT_EQ(cameras.size(), 2u);

    std::shared_ptr<lfs::core::Camera> inside;
    std::shared_ptr<lfs::core::Camera> outside_cam;
    for (const auto& cam : cameras) {
        if (cam->image_name() == "inside.png") {
            inside = cam;
        } else if (cam->image_name() == "outside.png") {
            outside_cam = cam;
        }
    }
    ASSERT_NE(inside, nullptr);
    ASSERT_NE(outside_cam, nullptr);

    EXPECT_EQ(inside->image_path(), new_root / "images" / "inside.png");
    EXPECT_EQ(inside->mask_path(), new_root / "masks" / "inside.png");
    EXPECT_EQ(inside->depth_path(), new_root / "depth" / "inside.exr");
    EXPECT_EQ(inside->normal_path(), new_root / "normals" / "inside.png");
    EXPECT_EQ(outside_cam->image_path(), outside);
    EXPECT_TRUE(outside_cam->mask_path().empty());

    const auto* inside_node = scene.getNode("inside.png");
    ASSERT_NE(inside_node, nullptr);
    EXPECT_EQ(inside_node->image_path, lfs::core::path_to_utf8(new_root / "images" / "inside.png"));
    EXPECT_EQ(inside_node->mask_path, lfs::core::path_to_utf8(new_root / "masks" / "inside.png"));
    EXPECT_EQ(inside_node->depth_path, lfs::core::path_to_utf8(new_root / "depth" / "inside.exr"));

    const auto* outside_node = scene.getNode("outside.png");
    ASSERT_NE(outside_node, nullptr);
    EXPECT_EQ(outside_node->image_path, lfs::core::path_to_utf8(outside));
    EXPECT_TRUE(outside_node->mask_path.empty());
}

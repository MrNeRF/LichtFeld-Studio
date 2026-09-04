/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/event_bridge/event_bridge.hpp"
#include "core/event_bus.hpp"
#include "core/scene.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "operation/undo_entry.hpp"
#include "operation/undo_history.hpp"
#include "rendering/rendering_manager.hpp"
#include "scene/scene_manager.hpp"
#include "visualizer/gui_capabilities.hpp"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <format>
#include <glm/gtc/matrix_transform.hpp>
#include <gtest/gtest.h>
#include <iomanip>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

    using lfs::core::DataType;
    using lfs::core::Device;
    using lfs::core::NodeId;
    using lfs::core::NodeType;
    using lfs::core::Tensor;

    std::unique_ptr<lfs::core::SplatData> make_splat(
        const size_t count,
        const float offset,
        const Device device) {
        std::vector<float> means;
        means.reserve(count * 3);
        for (size_t i = 0; i < count; ++i) {
            means.push_back(offset + static_cast<float>(i));
            means.push_back(static_cast<float>(i % 3));
            means.push_back(-static_cast<float>(i));
        }

        std::vector<float> rotation(count * 4, 0.0f);
        for (size_t i = 0; i < count; ++i)
            rotation[i * 4] = 1.0f;

        return std::make_unique<lfs::core::SplatData>(
            1,
            Tensor::from_vector(means, {count, size_t{3}}, device),
            Tensor::zeros({count, size_t{1}, size_t{3}}, device, DataType::Float32),
            Tensor::zeros({count, size_t{3}, size_t{3}}, device, DataType::Float32),
            Tensor::zeros({count, size_t{3}}, device, DataType::Float32),
            Tensor::from_vector(rotation, {count, size_t{4}}, device),
            Tensor::zeros({count, size_t{1}}, device, DataType::Float32),
            1.0f);
    }

    std::unique_ptr<lfs::core::SplatData> make_cpu_splat(const size_t count, const float offset) {
        return make_splat(count, offset, Device::CPU);
    }

    std::unique_ptr<lfs::core::SplatData> make_cuda_splat(const size_t count, const float offset) {
        return make_splat(count, offset, Device::CUDA);
    }

    uint64_t hash_tensor(const Tensor& tensor) {
        if (!tensor.is_valid())
            return 0;
        const auto values = tensor.cpu().contiguous().to_vector();
        uint64_t hash = 1469598103934665603ull;
        for (const float value : values) {
            uint32_t bits = 0;
            static_assert(sizeof(bits) == sizeof(value));
            std::memcpy(&bits, &value, sizeof(bits));
            hash ^= bits;
            hash *= 1099511628211ull;
        }
        return hash;
    }

    std::string scene_state(const lfs::vis::SceneManager& manager, const bool include_ids = true) {
        const auto& scene = manager.getScene();
        std::ostringstream out;
        out << std::setprecision(9);
        std::vector<const lfs::core::SceneNode*> ordered_nodes;
        const auto append_subtree = [&](const auto& self, const NodeId id) -> void {
            const auto* node = scene.getNodeById(id);
            if (!node)
                return;
            ordered_nodes.push_back(node);
            for (const NodeId child : node->children)
                self(self, child);
        };
        for (const NodeId root : scene.getRootNodes())
            append_subtree(append_subtree, root);

        for (const auto* node : ordered_nodes) {
            if (include_ids)
                out << node->id << ':';
            out << node->uuid.to_string() << ':' << node->name << ':'
                << static_cast<int>(node->type) << ':' << node->parent_id << ':'
                << static_cast<bool>(node->visible) << ':' << static_cast<bool>(node->locked) << ':'
                << node->gaussian_count.load(std::memory_order_acquire) << ':';
            for (const NodeId child : node->children)
                out << child << ',';
            const auto transform = scene.getNodeTransform(node->id);
            for (int col = 0; col < 4; ++col)
                for (int row = 0; row < 4; ++row)
                    out << ':' << transform[col][row];
            out << ':' << (node->model ? hash_tensor(node->model->means_raw()) : 0);
            out << ':' << (node->model && node->model->has_deleted_mask() ? hash_tensor(node->model->deleted()) : 0)
                << ';';
        }
        std::vector<NodeId> selected = manager.getSelectedNodeIds();
        std::ranges::sort(selected);
        out << "selection:";
        for (const NodeId id : selected) {
            const auto* node = scene.getNodeById(id);
            if (include_ids)
                out << id;
            else if (node)
                out << node->uuid.to_string();
            out << ',';
        }
        if (const auto mask = scene.getSelectionMask())
            out << "mask:" << hash_tensor(*mask);
        return out.str();
    }

    std::unordered_map<lfs::core::Uuid, NodeId> node_ids_by_uuid(const lfs::vis::SceneManager& manager) {
        std::unordered_map<lfs::core::Uuid, NodeId> result;
        for (const auto* node : manager.getScene().getNodes())
            result.emplace(node->uuid, node->id);
        return result;
    }

    void expect_common_node_ids(
        const lfs::vis::SceneManager& manager,
        const std::unordered_map<lfs::core::Uuid, NodeId>& expected,
        const std::unordered_map<lfs::core::Uuid, NodeId>& other,
        const int seed = -1,
        const int operation = -1,
        const int kind = -1) {
        const auto current = node_ids_by_uuid(manager);
        for (const auto& [uuid, id] : expected) {
            if (!other.contains(uuid))
                continue;
            const auto current_it = current.find(uuid);
            ASSERT_NE(current_it, current.end())
                << "seed=" << seed << " operation=" << operation << " kind=" << kind;
            EXPECT_EQ(current_it->second, id)
                << "seed=" << seed << " operation=" << operation << " kind=" << kind;
        }
    }

    void check_scene_invariants(const lfs::vis::SceneManager& manager) {
        const auto& scene = manager.getScene();
        std::unordered_set<NodeId> ids;
        std::unordered_set<lfs::core::Uuid> uuids;
        std::unordered_set<std::string> names;
        size_t visible_gaussians = 0;
        for (const auto* node : scene.getNodes()) {
            ASSERT_NE(node, nullptr);
            ASSERT_TRUE(ids.insert(node->id).second);
            ASSERT_TRUE(uuids.insert(node->uuid).second);
            ASSERT_TRUE(names.insert(node->name).second);
            if (node->parent_id == lfs::core::NULL_NODE) {
                EXPECT_FALSE(std::ranges::any_of(scene.getNodes(), [&](const auto* parent) {
                    return parent && std::ranges::count(parent->children, node->id) != 0;
                }));
            } else {
                const auto* parent = scene.getNodeById(node->parent_id);
                ASSERT_NE(parent, nullptr);
                EXPECT_EQ(std::ranges::count(parent->children, node->id), 1);
            }
            for (const NodeId child_id : node->children) {
                const auto* child = scene.getNodeById(child_id);
                ASSERT_NE(child, nullptr);
                EXPECT_EQ(child->parent_id, node->id);
                EXPECT_EQ(std::ranges::count(node->children, child_id), 1);
            }
            EXPECT_EQ(std::ranges::count(node->children, node->id), 0);
            if (node->type == NodeType::SPLAT && scene.isNodeEffectivelyVisible(node->id))
                visible_gaussians += node->gaussian_count.load(std::memory_order_acquire);
        }
        EXPECT_EQ(scene.getTotalGaussianCount(), visible_gaussians);

        for (const auto* node : scene.getNodes()) {
            std::unordered_set<NodeId> visited;
            for (NodeId current = node->id; current != lfs::core::NULL_NODE;) {
                ASSERT_TRUE(visited.insert(current).second);
                const auto* current_node = scene.getNodeById(current);
                ASSERT_NE(current_node, nullptr);
                current = current_node->parent_id;
            }
        }
        for (const NodeId id : manager.getSelectedNodeIds())
            EXPECT_NE(scene.getNodeById(id), nullptr);
    }

    class SceneGraphFuzzTest : public ::testing::Test {
    protected:
        void SetUp() override {
            lfs::event::EventBridge::instance().clear_all();
            lfs::core::event::bus().clear_all();
            lfs::vis::services().clear();
            lfs::vis::op::undoHistory().clear();
            manager_ = std::make_unique<lfs::vis::SceneManager>();
            rendering_ = std::make_unique<lfs::vis::RenderingManager>();
            lfs::vis::services().set(manager_.get());
            lfs::vis::services().set(rendering_.get());
            manager_->changeContentType(lfs::vis::SceneManager::ContentType::SplatFiles);
        }

        void TearDown() override {
            lfs::vis::op::undoHistory().clear();
            lfs::vis::services().clear();
            rendering_.reset();
            manager_.reset();
            lfs::core::event::bus().clear_all();
            lfs::event::EventBridge::instance().clear_all();
        }

        void seed_scene() {
            auto& scene = manager_->getScene();
            const NodeId left = scene.addGroup("left");
            const NodeId right = scene.addGroup("right");
            const NodeId nested = scene.addGroup("nested", left);
            scene.addSplat("a", make_cuda_splat(2, 0.0f), left);
            scene.addSplat("b", make_cuda_splat(3, 10.0f), left);
            scene.addSplat("c", make_cuda_splat(2, 20.0f), nested);
            scene.addSplat("d", make_cuda_splat(1, 30.0f), right);
        }

        std::unique_ptr<lfs::vis::SceneManager> manager_;
        std::unique_ptr<lfs::vis::RenderingManager> rendering_;
    };

    class SceneGraphRegression : public SceneGraphFuzzTest {};

    TEST_F(SceneGraphRegression, BakeTransformUndoRestoresMeans) {
        auto& scene = manager_->getScene();
        const NodeId node_id = scene.addSplat("bake", make_cpu_splat(2, 1.0f));
        ASSERT_NE(node_id, lfs::core::NULL_NODE);
        const auto* node = scene.getNodeById(node_id);
        ASSERT_NE(node, nullptr);
        ASSERT_NE(node->model, nullptr);

        const auto original_means = node->model->means_raw().cpu().to_vector();
        const auto original_transform =
            glm::translate(glm::mat4(1.0f), glm::vec3(4.0f, -2.0f, 3.0f));
        scene.setNodeTransform(node_id, original_transform);

        ASSERT_TRUE(lfs::vis::cap::bakeNodeTransforms(*manager_, {"bake"}, "Bake Transform"));
        ASSERT_EQ(lfs::vis::op::undoHistory().undoCount(), 1u);
        const auto* baked = scene.getNodeById(node_id);
        ASSERT_NE(baked, nullptr);
        ASSERT_NE(baked->model, nullptr);
        const auto baked_means = baked->model->means_raw().cpu().to_vector();
        EXPECT_NE(baked_means, original_means);
        EXPECT_EQ(baked->local_transform.get(), glm::mat4(1.0f));

        ASSERT_TRUE(lfs::vis::op::undoHistory().undo().success);
        const auto* undone = scene.getNodeByUuid(baked->uuid);
        ASSERT_NE(undone, nullptr);
        ASSERT_NE(undone->model, nullptr);
        EXPECT_EQ(undone->id, node_id);
        EXPECT_EQ(undone->model->means_raw().cpu().to_vector(), original_means);
        EXPECT_EQ(undone->local_transform.get(), original_transform);

        ASSERT_TRUE(lfs::vis::op::undoHistory().redo().success);
        const auto* redone = scene.getNodeByUuid(undone->uuid);
        ASSERT_NE(redone, nullptr);
        ASSERT_NE(redone->model, nullptr);
        EXPECT_EQ(redone->id, node_id);
        EXPECT_EQ(redone->model->means_raw().cpu().to_vector(), baked_means);
        EXPECT_EQ(redone->local_transform.get(), glm::mat4(1.0f));

        ASSERT_TRUE(lfs::vis::op::undoHistory().undo().success);
        EXPECT_EQ(scene.getNodeByUuid(redone->uuid)->model->means_raw().cpu().to_vector(), original_means);
    }

    TEST_F(SceneGraphRegression, SoftDeleteSelectedIsAtomicOnFailure) {
        auto& scene = manager_->getScene();
        const NodeId first_id = scene.addSplat("first", make_cpu_splat(2, 0.0f));
        const NodeId second_id = scene.addSplat("second", make_cpu_splat(2, 10.0f));
        ASSERT_NE(first_id, lfs::core::NULL_NODE);
        ASSERT_NE(second_id, lfs::core::NULL_NODE);

        auto* first = scene.getNodeById(first_id);
        auto* second = scene.getNodeById(second_id);
        ASSERT_NE(first, nullptr);
        ASSERT_NE(second, nullptr);
        ASSERT_NE(first->model, nullptr);
        ASSERT_NE(second->model, nullptr);

        // The second model deliberately has a pre-existing deleted mask on a
        // different device. Preparation must reject the whole operation before
        // applying the first partial slice.
        second->model->deleted() = Tensor::zeros({2}, Device::CUDA, DataType::Bool);
        second->model->notify_deleted_mask_changed();
        const auto before_second_deleted = second->model->deleted().clone();
        scene.setSelectionMask(std::make_shared<Tensor>(
            Tensor::from_vector(std::vector<bool>{true, false, true, false}, {4}, Device::CPU)));
        const auto before_selection = scene.getSelectionMask()->clone();

        const auto result = manager_->softDeleteSelectedGaussians();
        ASSERT_FALSE(result);
        EXPECT_FALSE(first->model->has_deleted_mask());
        ASSERT_TRUE(second->model->has_deleted_mask());
        EXPECT_EQ(second->model->deleted().cpu().to_vector_bool(),
                  before_second_deleted.cpu().to_vector_bool());
        EXPECT_EQ(scene.getSelectionMask()->cpu().to_vector_bool(),
                  before_selection.cpu().to_vector_bool());
        EXPECT_EQ(lfs::vis::op::undoHistory().undoCount(), 0u);
    }

    TEST_F(SceneGraphFuzzTest, GroupNodesRejectsDuplicateIdsAtomically) {
        seed_scene();
        const auto& scene = manager_->getScene();
        const NodeId a = scene.getNodeIdByName("a");
        const std::string before = scene_state(*manager_);
        const size_t history_before = lfs::vis::op::undoHistory().undoCount();

        EXPECT_FALSE(manager_->groupNodes({a, a}));
        EXPECT_EQ(scene_state(*manager_), before);
        EXPECT_EQ(lfs::vis::op::undoHistory().undoCount(), history_before);
    }

    TEST_F(SceneGraphFuzzTest, GroupSelectedUndoRestoresRootsAndRemovesGroup) {
        auto& scene = manager_->getScene();
        const NodeId a = scene.addSplat("a", make_cpu_splat(1, 0.0f));
        const NodeId b = scene.addSplat("b", make_cpu_splat(1, 1.0f));
        const auto roots_before = scene.getRootNodes();

        manager_->setNodeVisibility(a, false);
        const auto before_group = scene_state(*manager_);

        ASSERT_TRUE(manager_->groupNodes({a, b}));
        ASSERT_EQ(lfs::vis::op::undoHistory().undoCount(), 2u);
        ASSERT_EQ(scene.getRootNodes().size(), 1u);
        ASSERT_EQ(scene.getNodeById(scene.getRootNodes().front())->type, NodeType::GROUP);

        ASSERT_TRUE(lfs::vis::op::undoHistory().undo().success);
        EXPECT_EQ(scene_state(*manager_), before_group);
        EXPECT_EQ(scene.getRootNodes(), roots_before);
        EXPECT_EQ(scene.getNode("a")->id, a);
        EXPECT_EQ(scene.getNode("b")->id, b);

        ASSERT_TRUE(lfs::vis::op::undoHistory().undo().success);
        EXPECT_TRUE(scene.getNodeById(a)->visible);
    }

    TEST_F(SceneGraphFuzzTest, UngroupUndoRestoresChildrenWithStableIds) {
        auto& scene = manager_->getScene();
        const NodeId group = scene.addGroup("group");
        const NodeId a = scene.addSplat("a", make_cpu_splat(1, 0.0f), group);
        const NodeId b = scene.addSplat("b", make_cpu_splat(1, 1.0f), group);
        const NodeId other = scene.addSplat("other", make_cpu_splat(1, 2.0f));
        const auto group_uuid = scene.getNodeById(group)->uuid;

        manager_->setNodeVisibility(a, false);
        const auto node_count_before = scene.getNodeCount();
        ASSERT_TRUE(manager_->ungroupNode(group));
        ASSERT_EQ(lfs::vis::op::undoHistory().undoCount(), 2u);

        ASSERT_TRUE(lfs::vis::op::undoHistory().undo().success);
        ASSERT_EQ(scene.getNodeCount(), node_count_before);
        const auto* restored_group = scene.getNodeByUuid(group_uuid);
        ASSERT_NE(restored_group, nullptr);
        EXPECT_EQ(restored_group->children, (std::vector<NodeId>{a, b}));
        EXPECT_EQ(scene.getNodeById(a)->id, a);
        EXPECT_EQ(scene.getNodeById(b)->id, b);
        EXPECT_EQ(scene.getRootNodes().size(), 2u);
        EXPECT_EQ(scene.getRootNodes()[0], restored_group->id);
        EXPECT_EQ(scene.getRootNodes()[1], other);

        ASSERT_TRUE(lfs::vis::op::undoHistory().undo().success);
        EXPECT_TRUE(scene.getNodeById(a)->visible);
    }

    TEST_F(SceneGraphFuzzTest, GroupSelectedAllowsGroupAndSplatSiblings) {
        auto& scene = manager_->getScene();
        const NodeId first_group = scene.addGroup("first");
        const NodeId splat = scene.addSplat("splat", make_cpu_splat(1, 0.0f));

        ASSERT_TRUE(manager_->groupNodes({splat, first_group}));
        const auto roots = scene.getRootNodes();
        ASSERT_EQ(roots.size(), 1u);
        const auto* outer = scene.getNodeById(roots.front());
        ASSERT_NE(outer, nullptr);
        ASSERT_EQ(outer->children, (std::vector<NodeId>{first_group, splat}));
        EXPECT_EQ(scene.getNodeById(first_group)->parent_id, outer->id);
        EXPECT_EQ(scene.getNodeById(splat)->parent_id, outer->id);

        ASSERT_TRUE(lfs::vis::op::undoHistory().undo().success);
        EXPECT_EQ(scene.getRootNodes(), (std::vector<NodeId>{first_group, splat}));
    }

    TEST_F(SceneGraphFuzzTest, DuplicateNestedTransformedGroupIsOneUndoStep) {
        auto& scene = manager_->getScene();
        const NodeId outer = scene.addGroup("outer");
        const NodeId inner = scene.addGroup("inner", outer);
        const NodeId splat = scene.addSplat("splat", make_cpu_splat(1, 0.0f), inner);
        const auto outer_transform = glm::translate(glm::mat4(1.0f), glm::vec3(3.0f, 1.0f, -2.0f));
        const auto inner_transform = glm::scale(glm::mat4(1.0f), glm::vec3(2.0f, 3.0f, 4.0f));
        scene.setNodeTransform(outer, outer_transform);
        scene.setNodeTransform(inner, inner_transform);
        const auto before = scene_state(*manager_);

        const auto duplicate_name = manager_->duplicateNodeTree(outer);
        ASSERT_FALSE(duplicate_name.empty());
        ASSERT_EQ(lfs::vis::op::undoHistory().undoCount(), 1u);
        EXPECT_EQ(lfs::vis::op::undoHistory().undoName(), "Duplicate Node");
        const auto* duplicate = scene.getNode(duplicate_name);
        ASSERT_NE(duplicate, nullptr);
        EXPECT_EQ(scene.getNodeById(duplicate->children.front())->local_transform.get(), inner_transform);
        EXPECT_EQ(duplicate->local_transform.get(), outer_transform);

        ASSERT_TRUE(lfs::vis::op::undoHistory().undo().success);
        EXPECT_EQ(scene_state(*manager_), before);
        ASSERT_TRUE(lfs::vis::op::undoHistory().redo().success);
        ASSERT_NE(scene.getNode(duplicate_name), nullptr);
        ASSERT_TRUE(lfs::vis::op::undoHistory().undo().success);
        EXPECT_EQ(scene_state(*manager_), before);
    }

    TEST_F(SceneGraphFuzzTest, BatchRemovalRejectsDuplicateIdsAtomically) {
        seed_scene();
        const auto& scene = manager_->getScene();
        const NodeId a = scene.getNodeIdByName("a");
        const std::string before = scene_state(*manager_);
        const auto result = manager_->removeNodesByIdsWithResult({a, a});

        EXPECT_FALSE(result);
        EXPECT_EQ(scene_state(*manager_), before);
        EXPECT_EQ(lfs::vis::op::undoHistory().undoCount(), 0u);
    }

    TEST_F(SceneGraphFuzzTest, BatchRemovalIsOneUndoStep) {
        seed_scene();
        const auto& scene = manager_->getScene();
        const NodeId a = scene.getNodeIdByName("a");
        const NodeId d = scene.getNodeIdByName("d");
        const std::string before = scene_state(*manager_);

        ASSERT_TRUE(manager_->removeNodesByIdsWithResult({a, d}));
        ASSERT_EQ(lfs::vis::op::undoHistory().undoCount(), 1u);
        ASSERT_TRUE(lfs::vis::op::undoHistory().undo().success);
        EXPECT_EQ(scene_state(*manager_), before);
    }

    TEST_F(SceneGraphFuzzTest, MoveNodeIntoSplatLeafIsRejected) {
        seed_scene();
        const auto& scene = manager_->getScene();
        const NodeId a = scene.getNodeIdByName("a");
        const NodeId b = scene.getNodeIdByName("b");
        const std::string before = scene_state(*manager_);

        EXPECT_FALSE(manager_->moveNode(a, b, 0));
        EXPECT_EQ(scene_state(*manager_), before);
    }

    TEST_F(SceneGraphFuzzTest, AddGroupUnderSplatLeafIsRejected) {
        seed_scene();
        const auto& scene = manager_->getScene();
        const NodeId a = scene.getNodeIdByName("a");
        const std::string before = scene_state(*manager_);

        EXPECT_TRUE(manager_->addGroupNode("invalid", a).empty());
        EXPECT_EQ(scene_state(*manager_), before);
    }

    TEST_F(SceneGraphFuzzTest, AddSplatUnderSplatLeafIsRejected) {
        seed_scene();
        auto& scene = manager_->getScene();
        const NodeId a = scene.getNodeIdByName("a");
        const std::string before = scene_state(*manager_);

        EXPECT_EQ(scene.addSplat("invalid", make_cpu_splat(1, 0.0f), a), lfs::core::NULL_NODE);
        EXPECT_EQ(scene_state(*manager_), before);
    }

    TEST_F(SceneGraphFuzzTest, ReparentNodeIntoSplatLeafIsRejected) {
        seed_scene();
        const auto& scene = manager_->getScene();
        const NodeId a = scene.getNodeIdByName("a");
        const NodeId b = scene.getNodeIdByName("b");
        const std::string before = scene_state(*manager_);

        EXPECT_FALSE(manager_->reparentNode(a, b));
        EXPECT_EQ(scene_state(*manager_), before);
    }

    TEST_F(SceneGraphFuzzTest, FailedMergePreservesSelection) {
        const NodeId empty_group = manager_->getScene().addGroup("empty");
        ASSERT_NE(empty_group, lfs::core::NULL_NODE);
        manager_->selectNode(empty_group);
        const std::string before = scene_state(*manager_);

        EXPECT_TRUE(manager_->mergeGroupNode(empty_group).empty());
        EXPECT_EQ(scene_state(*manager_), before);
    }

    TEST_F(SceneGraphFuzzTest, CopyPastePreservesGroupHierarchy) {
        seed_scene();
        auto& scene = manager_->getScene();
        const NodeId left = scene.getNodeIdByName("left");
        const NodeId nested = scene.getNodeIdByName("nested");
        const auto left_transform = glm::translate(glm::mat4(1.0f), glm::vec3(4.0f, 0.0f, 2.0f));
        const auto nested_transform = glm::rotate(glm::mat4(1.0f), 0.35f, glm::vec3(0.0f, 1.0f, 0.0f));
        scene.setNodeTransform(left, left_transform);
        scene.setNodeTransform(nested, nested_transform);
        const auto before = scene_state(*manager_);
        manager_->selectNode(left);
        ASSERT_TRUE(manager_->copySelectedNodes());

        const auto pasted = manager_->pasteNodes();
        ASSERT_EQ(pasted.size(), 1u);
        ASSERT_EQ(lfs::vis::op::undoHistory().undoCount(), 1u);
        const auto* pasted_root = scene.getNode(pasted.front());
        ASSERT_NE(pasted_root, nullptr);
        ASSERT_EQ(pasted_root->type, NodeType::GROUP);
        ASSERT_EQ(pasted_root->children.size(), 3u);
        EXPECT_EQ(pasted_root->local_transform.get(), left_transform);
        EXPECT_NE(pasted_root->uuid, scene.getNodeById(left)->uuid);
        EXPECT_NE(pasted_root->name, scene.getNodeById(left)->name);
        for (size_t i = 0; i < pasted_root->children.size(); ++i) {
            const auto* source_child = scene.getNodeById(scene.getNodeById(left)->children[i]);
            const auto* pasted_child = scene.getNodeById(pasted_root->children[i]);
            ASSERT_NE(source_child, nullptr);
            ASSERT_NE(pasted_child, nullptr);
            EXPECT_EQ(pasted_child->type, source_child->type);
            EXPECT_EQ(pasted_child->local_transform.get(), source_child->local_transform.get());
            EXPECT_NE(pasted_child->uuid, source_child->uuid);
            EXPECT_EQ(pasted_child->name, source_child->name + " 1");
            if (source_child->type == NodeType::GROUP) {
                ASSERT_EQ(pasted_child->children.size(), source_child->children.size());
                for (size_t j = 0; j < source_child->children.size(); ++j) {
                    const auto* source_grandchild = scene.getNodeById(source_child->children[j]);
                    const auto* pasted_grandchild = scene.getNodeById(pasted_child->children[j]);
                    ASSERT_NE(source_grandchild, nullptr);
                    ASSERT_NE(pasted_grandchild, nullptr);
                    EXPECT_EQ(pasted_grandchild->type, source_grandchild->type);
                    EXPECT_EQ(pasted_grandchild->local_transform.get(), source_grandchild->local_transform.get());
                    EXPECT_NE(pasted_grandchild->uuid, source_grandchild->uuid);
                    EXPECT_EQ(pasted_grandchild->name, source_grandchild->name + " 1");
                }
            }
        }

        ASSERT_TRUE(lfs::vis::op::undoHistory().undo().success);
        EXPECT_EQ(scene_state(*manager_), before);
    }

    TEST_F(SceneGraphFuzzTest, CopyPasteGroupHierarchyCanBeRepeatedAfterSourceDelete) {
        seed_scene();
        const auto& scene = manager_->getScene();
        const NodeId left = scene.getNodeIdByName("left");
        manager_->selectNode(left);
        ASSERT_TRUE(manager_->copySelectedNodes());
        ASSERT_EQ(manager_->pasteNodes().size(), 1u);
        ASSERT_EQ(manager_->pasteNodes().size(), 1u);
        ASSERT_TRUE(manager_->removeNodeWithResult(left));
        ASSERT_EQ(manager_->pasteNodes().size(), 1u);
        check_scene_invariants(*manager_);
    }

    TEST_F(SceneGraphFuzzTest, MergeBakesChildWorldTransforms) {
        auto& scene = manager_->getScene();
        const NodeId group = scene.addGroup("merge");
        const NodeId child = scene.addSplat("child", make_cpu_splat(2, 3.0f), group);
        ASSERT_NE(group, lfs::core::NULL_NODE);
        ASSERT_NE(child, lfs::core::NULL_NODE);
        scene.setNodeTransform(group, glm::translate(glm::mat4(1.0f), glm::vec3(5.0f, 2.0f, -1.0f)));
        scene.setNodeTransform(child, glm::translate(glm::mat4(1.0f), glm::vec3(-2.0f, 4.0f, 3.0f)));

        const auto before = scene.getWorldTransform(child);
        const auto source = scene.getNodeById(child)->model->means_raw().cpu().to_vector();
        std::vector<float> expected;
        for (size_t row = 0; row < 2; ++row) {
            const glm::vec4 world = before * glm::vec4(source[row * 3], source[row * 3 + 1], source[row * 3 + 2], 1.0f);
            expected.insert(expected.end(), {world.x, world.y, world.z});
        }

        ASSERT_EQ(manager_->mergeGroupNode(group), "merge");
        const auto* merged = scene.getNode("merge");
        ASSERT_NE(merged, nullptr);
        const auto actual = merged->model->means_raw().cpu().to_vector();
        ASSERT_EQ(actual.size(), expected.size());
        for (size_t i = 0; i < actual.size(); ++i)
            EXPECT_NEAR(actual[i], expected[i], 1e-4f);
    }

    TEST_F(SceneGraphFuzzTest, MergeIncludesHiddenChildModels) {
        auto& scene = manager_->getScene();
        const NodeId group = scene.addGroup("merge_hidden");
        const NodeId visible_a = scene.addSplat("visible_a", make_cpu_splat(1, 1.0f), group);
        const NodeId hidden = scene.addSplat("hidden", make_cpu_splat(2, 10.0f), group);
        const NodeId visible_b = scene.addSplat("visible_b", make_cpu_splat(3, 20.0f), group);
        ASSERT_NE(group, lfs::core::NULL_NODE);
        ASSERT_NE(visible_a, lfs::core::NULL_NODE);
        ASSERT_NE(hidden, lfs::core::NULL_NODE);
        ASSERT_NE(visible_b, lfs::core::NULL_NODE);
        scene.setNodeVisibility(hidden, false);

        ASSERT_EQ(manager_->mergeGroupNode(group), "merge_hidden");
        const auto* merged = scene.getNode("merge_hidden");
        ASSERT_NE(merged, nullptr);
        ASSERT_NE(merged->model, nullptr);
        EXPECT_EQ(merged->model->size(), 6);
        const auto means = merged->model->means_raw().cpu().to_vector();
        ASSERT_EQ(means.size(), 18u);
        EXPECT_FLOAT_EQ(means[0], 1.0f);
        EXPECT_FLOAT_EQ(means[3], 10.0f);
        EXPECT_FLOAT_EQ(means[6], 11.0f);
        EXPECT_FLOAT_EQ(means[9], 20.0f);
        EXPECT_FLOAT_EQ(means[15], 22.0f);
    }

    TEST_F(SceneGraphFuzzTest, LockedNodeRejectsTransformMoveDeleteAndDuplicate) {
        seed_scene();
        auto& scene = manager_->getScene();
        const NodeId locked = scene.getNodeIdByName("a");
        const NodeId destination = scene.getNodeIdByName("right");
        const auto before_transform = scene.getNodeTransform(locked);
        const auto before_parent = scene.getNodeById(locked)->parent_id;
        const size_t before_count = scene.getNodeCount();
        scene.setNodeLocked("a", true);

        EXPECT_FALSE(manager_->setNodeTransform("a", glm::translate(glm::mat4(1.0f), glm::vec3(4.0f))));
        EXPECT_FALSE(manager_->moveNode(locked, destination, 0));
        EXPECT_FALSE(manager_->reparentNode(locked, lfs::core::NULL_NODE));
        const auto delete_result = manager_->removeNodeWithResult(locked);
        ASSERT_FALSE(delete_result);
        EXPECT_NE(delete_result.error().find("node is locked"), std::string::npos);
        EXPECT_TRUE(manager_->duplicateNodeTree(locked).empty());
        EXPECT_EQ(scene.getNodeCount(), before_count);
        EXPECT_EQ(scene.getNodeById(locked)->parent_id, before_parent);
        EXPECT_EQ(scene.getNodeTransform(locked), before_transform);
        EXPECT_EQ(lfs::vis::op::undoHistory().undoCount(), 0u);
    }

    TEST_F(SceneGraphFuzzTest, RenameRejectsExistingName) {
        auto& scene = manager_->getScene();
        const NodeId first = scene.addSplat("first", make_cpu_splat(1, 0.0f));
        const NodeId second = scene.addSplat("second", make_cpu_splat(1, 1.0f));
        ASSERT_NE(first, lfs::core::NULL_NODE);
        ASSERT_NE(second, lfs::core::NULL_NODE);

        EXPECT_FALSE(manager_->renameNode(first, "second"));
        EXPECT_NE(scene.getNode("first"), nullptr);
        EXPECT_NE(scene.getNode("second"), nullptr);
        EXPECT_EQ(lfs::vis::op::undoHistory().undoCount(), 0u);
    }

    TEST_F(SceneGraphFuzzTest, SeededOperationSequencesPreserveSceneInvariants) {
        constexpr int seed_count = 200;
        constexpr int operations_per_seed = 60;
        struct UndoFrame {
            std::string before;
            std::string after;
            std::string before_graph;
            std::string after_graph;
            std::unordered_map<lfs::core::Uuid, NodeId> before_ids;
            std::unordered_map<lfs::core::Uuid, NodeId> after_ids;
            int kind = -1;
            bool stable_common_ids = false;
        };
        for (int seed = 0; seed < seed_count; ++seed) {
            manager_->clearSelection();
            manager_->getScene().clear();
            lfs::vis::op::undoHistory().clear();
            seed_scene();
            std::mt19937 rng(static_cast<uint32_t>(0x51CE'0000u + seed));
            std::vector<UndoFrame> undo_frames;
            std::vector<UndoFrame> redo_frames;

            for (int operation = 0; operation < operations_per_seed; ++operation) {
                auto nodes = manager_->getScene().getNodes();
                if (nodes.empty()) {
                    seed_scene();
                    nodes = manager_->getScene().getNodes();
                }
                const auto pick = [&](const bool group_only = false) -> const lfs::core::SceneNode* {
                    std::vector<const lfs::core::SceneNode*> candidates;
                    for (const auto* node : nodes)
                        if (node && (!group_only || node->type == NodeType::GROUP))
                            candidates.push_back(node);
                    return candidates.empty() ? nullptr : candidates[rng() % candidates.size()];
                };

                std::string before = scene_state(*manager_);
                const std::string before_graph = scene_state(*manager_, false);
                const auto before_ids = node_ids_by_uuid(*manager_);
                const size_t undo_before = lfs::vis::op::undoHistory().undoCount();
                const int kind = static_cast<int>(rng() % 18);
                bool changed = false;
                bool history_candidate = false;
                bool history_navigation = false;
                bool untracked_mutation = false;
                std::string operation_error;
                switch (kind) {
                case 0:
                    changed = !manager_->addGroupNode(std::format("g_{}_{}", seed, operation)).empty();
                    history_candidate = changed;
                    break;
                case 1: {
                    const auto* node = pick();
                    changed = node && !manager_->duplicateNodeTree(node->id).empty();
                    history_candidate = changed;
                    break;
                }
                case 2: {
                    const auto* node = pick();
                    if (node) {
                        if ((rng() & 1u) != 0 && nodes.size() > 1) {
                            const auto* other = nodes[rng() % nodes.size()];
                            changed = other != node && manager_->renameNode(node->id, other->name);
                        } else {
                            changed = manager_->renameNode(node->id, std::format("r_{}_{}", seed, operation));
                        }
                    }
                    history_candidate = changed;
                    break;
                }
                case 3: {
                    const auto* node = pick();
                    if (node) {
                        const auto destination_kind = rng() % 4;
                        NodeId destination = lfs::core::NULL_NODE;
                        if (destination_kind == 1) {
                            destination = node->id;
                        } else if (destination_kind == 2 && !nodes.empty()) {
                            destination = nodes[rng() % nodes.size()]->id;
                        } else if (destination_kind == 3) {
                            if (const auto* parent = pick(true))
                                destination = parent->id;
                        }
                        changed = manager_->reparentNode(node->id, destination);
                    }
                    history_candidate = changed;
                    break;
                }
                case 4: {
                    const auto* node = pick();
                    const auto* parent = (rng() & 1u) != 0 ? pick(true) : pick();
                    if (node && parent) {
                        const auto index_kind = rng() % 5;
                        const int index = index_kind == 0
                                              ? 0
                                          : index_kind == 1
                                              ? static_cast<int>(parent->children.size() / 2)
                                          : index_kind == 2
                                              ? static_cast<int>(parent->children.size())
                                          : index_kind == 3
                                              ? static_cast<int>(parent->children.size() + 3)
                                              : -1;
                        changed = manager_->moveNode(node->id, parent->id, index);
                    }
                    history_candidate = changed;
                    break;
                }
                case 5: {
                    std::vector<NodeId> candidates;
                    for (const auto* node : nodes)
                        if (node && node->type != NodeType::GROUP)
                            candidates.push_back(node->id);
                    if (candidates.size() >= 2) {
                        std::ranges::shuffle(candidates, rng);
                        changed = manager_->groupNodes({candidates[0], candidates[1]});
                        history_candidate = changed;
                    }
                    break;
                }
                case 6: {
                    const auto* group = pick(true);
                    changed = group && manager_->ungroupNode(group->id);
                    history_candidate = changed;
                    break;
                }
                case 7: {
                    const auto* node = pick();
                    if (node) {
                        manager_->setNodeVisibility(node->id, !static_cast<bool>(node->visible));
                        changed = true;
                        history_candidate = true;
                    }
                    break;
                }
                case 8: {
                    const auto* node = pick();
                    if (node) {
                        manager_->selectNodesById({node->id});
                        manager_->copySelectedNodes();
                        if ((rng() & 1u) != 0) {
                            const auto result = manager_->removeNodeWithResult(node->id);
                            changed = result.has_value();
                            history_candidate = changed;
                        } else {
                            changed = true;
                            untracked_mutation = true;
                        }
                    }
                    break;
                }
                case 9: {
                    if (manager_->hasClipboard()) {
                        changed = !manager_->pasteNodes().empty();
                        history_candidate = changed;
                    }
                    break;
                }
                case 10: {
                    const auto* group = pick(true);
                    if (group) {
                        changed = !manager_->mergeGroupNode(group->id).empty();
                        history_candidate = changed;
                    }
                    break;
                }
                case 11: {
                    const auto* node = pick();
                    if (node) {
                        auto transform = glm::translate(glm::mat4(1.0f), glm::vec3(1.0f, -2.0f, 0.5f));
                        transform = glm::rotate(transform, 0.2f, glm::vec3(0.0f, 1.0f, 0.0f));
                        transform = glm::scale(transform, glm::vec3(1.2f, 0.8f, 1.1f));
                        manager_->setNodeTransform(node->name, transform);
                        changed = true;
                        untracked_mutation = true;
                    }
                    break;
                }
                case 12: {
                    auto& scene = manager_->getScene();
                    const size_t capacity = scene.getSelectionCapacity(lfs::core::SelectionDomain::Splat);
                    if (capacity > 0) {
                        std::vector<bool> mask(capacity, false);
                        mask[rng() % capacity] = true;
                        scene.setSelectionMask(
                            std::make_shared<Tensor>(Tensor::from_vector(mask, {capacity}, Device::CUDA)));
                        // Installing this mask is part of the operation. A
                        // failed delete must preserve the state after this
                        // intentional selection change.
                        before = scene_state(*manager_);
                        const auto result = manager_->softDeleteSelectedGaussians();
                        if (!result)
                            operation_error = result.error();
                        changed = result.has_value();
                        untracked_mutation = changed;
                    }
                    break;
                }
                case 13:
                    history_navigation = true;
                    if (!undo_frames.empty() && lfs::vis::op::undoHistory().canUndo() &&
                        (redo_frames.empty() || (rng() & 1u) == 0)) {
                        const auto frame = undo_frames.back();
                        const auto result = lfs::vis::op::undoHistory().undo();
                        ASSERT_TRUE(result.success) << result.error;
                        EXPECT_EQ(scene_state(*manager_, false), frame.before_graph)
                            << "seed=" << seed << " operation=" << operation << " kind=" << frame.kind;
                        if (frame.stable_common_ids)
                            expect_common_node_ids(*manager_, frame.before_ids, frame.after_ids,
                                                   seed, operation, frame.kind);
                        undo_frames.pop_back();
                        redo_frames.push_back(frame);
                    } else if (!redo_frames.empty() && lfs::vis::op::undoHistory().canRedo()) {
                        const auto frame = redo_frames.back();
                        const auto result = lfs::vis::op::undoHistory().redo();
                        ASSERT_TRUE(result.success) << result.error;
                        EXPECT_EQ(scene_state(*manager_, false), frame.after_graph)
                            << "seed=" << seed << " operation=" << operation << " kind=" << frame.kind;
                        if (frame.stable_common_ids)
                            expect_common_node_ids(*manager_, frame.before_ids, frame.after_ids,
                                                   seed, operation, frame.kind);
                        redo_frames.pop_back();
                        undo_frames.push_back(frame);
                    }
                    break;
                case 14: {
                    if (nodes.size() >= 2) {
                        const auto* first = nodes[rng() % nodes.size()];
                        const auto* second = nodes[rng() % nodes.size()];
                        if (first != second) {
                            const auto result = manager_->removeNodesByIdsWithResult(
                                {first->id, second->id}, (rng() & 1u) != 0);
                            changed = static_cast<bool>(result);
                            history_candidate = changed;
                        }
                    }
                    break;
                }
                case 15: {
                    const auto* parent = (rng() & 1u) != 0 ? pick() : nullptr;
                    changed = manager_->getScene().addSplat(
                                  std::format("s_{}_{}", seed, operation),
                                  make_cuda_splat(1 + rng() % 3, static_cast<float>(operation)),
                                  parent ? parent->id : lfs::core::NULL_NODE) !=
                              lfs::core::NULL_NODE;
                    untracked_mutation = changed;
                    break;
                }
                case 16: {
                    const size_t removed = manager_->applyDeleted();
                    changed = removed > 0;
                    untracked_mutation = changed;
                    break;
                }
                case 17: {
                    const auto* node = pick();
                    if (node && node->type == NodeType::SPLAT && node->model && node->model->size() > 0) {
                        const auto options = lfs::vis::op::SceneGraphCaptureOptions{
                            .mode = lfs::vis::op::SceneGraphCaptureMode::FULL,
                            .include_selected_nodes = false,
                            .include_scene_context = false,
                        };
                        auto payload_before = lfs::vis::op::SceneGraphPatchEntry::captureStateByIds(
                            *manager_, manager_->getScene().getRootNodes(), options);
                        std::vector<bool> mask(static_cast<size_t>(node->model->size()), false);
                        mask[rng() % mask.size()] = true;
                        const auto mask_tensor = Tensor::from_vector(
                                                     mask, {mask.size()}, Device::CPU)
                                                     .to(node->model->means_raw().device());
                        node->model->soft_delete(mask_tensor);
                        manager_->getScene().markPayloadDiverged(node->id);
                        manager_->getScene().notifyMutation(lfs::core::Scene::MutationType::MODEL_CHANGED);
                        auto payload_after = lfs::vis::op::SceneGraphPatchEntry::captureStateByIds(
                            *manager_, manager_->getScene().getRootNodes(), options);
                        lfs::vis::op::undoHistory().push(std::make_unique<lfs::vis::op::SceneGraphPatchEntry>(
                            *manager_,
                            "Fuzz Payload Mutation",
                            std::move(payload_before),
                            std::move(payload_after)));
                        changed = true;
                        history_candidate = true;
                    }
                    break;
                }
                }

                check_scene_invariants(*manager_);
                const std::string after = scene_state(*manager_);
                const std::string after_graph = scene_state(*manager_, false);
                const auto after_ids = node_ids_by_uuid(*manager_);
                const size_t undo_after = lfs::vis::op::undoHistory().undoCount();
                if (!changed && !history_candidate && !history_navigation)
                    EXPECT_EQ(after, before)
                        << "seed=" << seed << " operation=" << operation << " kind=" << kind
                        << " error=" << operation_error;
                if (history_candidate && undo_after == undo_before + 1) {
                    undo_frames.push_back(UndoFrame{
                        .before = before,
                        .after = after,
                        .before_graph = before_graph,
                        .after_graph = after_graph,
                        .before_ids = before_ids,
                        .after_ids = after_ids,
                        .kind = kind,
                        .stable_common_ids = kind == 3 || kind == 4 || kind == 5 || kind == 6 || kind == 17,
                    });
                    redo_frames.clear();
                }
                if (untracked_mutation) {
                    undo_frames.clear();
                    redo_frames.clear();
                }
            }
        }
    }

} // namespace

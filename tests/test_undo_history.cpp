/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/event_bridge/event_bridge.hpp"
#include "core/event_bus.hpp"
#include "core/services.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "operation/undo_history.hpp"
#include "rendering/rendering_manager.hpp"
#include "scene/scene_manager.hpp"

#include <gtest/gtest.h>
#include <memory>
#include <stdexcept>
#include <vector>

using lfs::core::DataType;
using lfs::core::Device;
using lfs::core::Tensor;

namespace {

    class CountingEntry final : public lfs::vis::op::UndoEntry {
    public:
        CountingEntry(std::string name, int& value, int delta, size_t estimated_bytes = 0)
            : name_(std::move(name)),
              value_(value),
              delta_(delta),
              estimated_bytes_(estimated_bytes) {}

        void undo() override { value_ -= delta_; }
        void redo() override { value_ += delta_; }
        [[nodiscard]] std::string name() const override { return name_; }
        [[nodiscard]] size_t estimatedBytes() const override { return estimated_bytes_; }

    private:
        std::string name_;
        int& value_;
        int delta_ = 0;
        size_t estimated_bytes_ = 0;
    };

    class ReentrantUndoEntry final : public lfs::vis::op::UndoEntry {
    public:
        explicit ReentrantUndoEntry(bool& queried) : queried_(queried) {}

        void undo() override { queried_ = lfs::vis::op::undoHistory().canUndo(); }
        void redo() override {}
        [[nodiscard]] std::string name() const override { return "reentrant.undo"; }

    private:
        bool& queried_;
    };

    class ThrowingUndoEntry final : public lfs::vis::op::UndoEntry {
    public:
        void undo() override { throw std::runtime_error("undo failed"); }
        void redo() override {}
        [[nodiscard]] std::string name() const override { return "throwing.undo"; }
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

    std::vector<bool> deleted_mask_values(const lfs::core::SplatData& splat) {
        if (!splat.has_deleted_mask()) {
            return {};
        }
        return splat.deleted().cpu().to_vector_bool();
    }

    Tensor make_uint8_mask(const std::vector<uint8_t>& values) {
        auto tensor = Tensor::empty({values.size()}, Device::CPU, DataType::UInt8);
        std::copy(values.begin(), values.end(), tensor.ptr<uint8_t>());
        return tensor.cuda();
    }

    std::vector<uint8_t> selection_mask_values(const lfs::core::Scene& scene) {
        auto mask = scene.getSelectionMask();
        if (!mask || !mask->is_valid()) {
            return {};
        }
        return mask->cpu().to_vector_uint8();
    }

} // namespace

class UndoHistoryTest : public ::testing::Test {
protected:
    void SetUp() override {
        lfs::event::EventBridge::instance().clear_all();
        lfs::core::event::bus().clear_all();
        lfs::vis::services().clear();
        lfs::vis::op::undoHistory().clear();
    }

    void TearDown() override {
        lfs::vis::op::undoHistory().clear();
        lfs::vis::services().clear();
        lfs::core::event::bus().clear_all();
        lfs::event::EventBridge::instance().clear_all();
    }
};

TEST_F(UndoHistoryTest, TransactionCommitGroupsEntriesIntoSingleUndoStep) {
    auto& history = lfs::vis::op::undoHistory();
    int value = 0;

    history.beginTransaction("grouped.change");
    history.push(std::make_unique<CountingEntry>("change.one", value, 2));
    value += 2;
    history.push(std::make_unique<CountingEntry>("change.two", value, 3));
    value += 3;
    history.commitTransaction();

    EXPECT_EQ(value, 5);
    EXPECT_EQ(history.undoCount(), 1u);
    EXPECT_EQ(history.undoName(), "grouped.change");

    history.undo();
    EXPECT_EQ(value, 0);
    EXPECT_EQ(history.redoCount(), 1u);

    history.redo();
    EXPECT_EQ(value, 5);
}

TEST_F(UndoHistoryTest, TransactionRollbackRestoresAppliedStateWithoutCreatingHistory) {
    auto& history = lfs::vis::op::undoHistory();
    int value = 1;

    history.beginTransaction("rolled.back");
    history.push(std::make_unique<CountingEntry>("change.one", value, 4));
    value += 4;
    history.push(std::make_unique<CountingEntry>("change.two", value, -2));
    value -= 2;
    history.rollbackTransaction();

    EXPECT_EQ(value, 1);
    EXPECT_EQ(history.undoCount(), 0u);
    EXPECT_EQ(history.redoCount(), 0u);
}

TEST_F(UndoHistoryTest, NestedTransactionsCollapseIntoSingleUndoStep) {
    auto& history = lfs::vis::op::undoHistory();
    int value = 0;

    history.beginTransaction("outer.group");
    history.beginTransaction("inner.group");
    history.push(std::make_unique<CountingEntry>("change.one", value, 1));
    value += 1;
    history.commitTransaction();
    history.push(std::make_unique<CountingEntry>("change.two", value, 2));
    value += 2;
    history.commitTransaction();

    ASSERT_EQ(history.undoCount(), 1u);
    EXPECT_EQ(history.undoName(), "outer.group");

    const auto result = history.undo();
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.steps_performed, 1u);
    EXPECT_EQ(value, 0);
}

TEST_F(UndoHistoryTest, EstimatedByteBudgetEvictsOldestUndoEntries) {
    auto& history = lfs::vis::op::undoHistory();
    int value = 0;

    history.push(std::make_unique<CountingEntry>("large.one", value, 1, 300ull * 1024ull * 1024ull));
    history.push(std::make_unique<CountingEntry>("large.two", value, 1, 300ull * 1024ull * 1024ull));

    EXPECT_EQ(history.undoCount(), 1u);
    EXPECT_EQ(history.undoName(), "large.two");
    EXPECT_LE(history.undoBytes(), lfs::vis::op::UndoHistory::MAX_BYTES);
}

TEST_F(UndoHistoryTest, UndoAndRedoNamesReturnNewestFirst) {
    auto& history = lfs::vis::op::undoHistory();
    int value = 0;

    history.push(std::make_unique<CountingEntry>("first", value, 1));
    history.push(std::make_unique<CountingEntry>("second", value, 1));
    history.undo();

    EXPECT_EQ(history.undoNames(), (std::vector<std::string>{"first"}));
    EXPECT_EQ(history.redoNames(), (std::vector<std::string>{"second"}));
}

TEST_F(UndoHistoryTest, StackItemsExposeStructuredMetadataAndTransactionState) {
    auto& history = lfs::vis::op::undoHistory();
    int value = 0;

    history.beginTransaction("Grouped Transaction");
    EXPECT_TRUE(history.hasActiveTransaction());
    EXPECT_EQ(history.transactionDepth(), 1u);
    EXPECT_EQ(history.activeTransactionName(), "Grouped Transaction");

    history.push(std::make_unique<CountingEntry>("first.change", value, 1, 128));
    history.commitTransaction();

    const auto items = history.undoItems();
    ASSERT_EQ(items.size(), 1u);
    EXPECT_EQ(items.front().metadata.id, "history.transaction");
    EXPECT_EQ(items.front().metadata.label, "Grouped Transaction");
    EXPECT_EQ(items.front().metadata.source, "history");
    EXPECT_EQ(items.front().metadata.scope, "grouped");
    EXPECT_EQ(items.front().estimated_bytes, 128u);
    EXPECT_FALSE(history.hasActiveTransaction());
    EXPECT_EQ(history.transactionDepth(), 0u);
    EXPECT_TRUE(history.activeTransactionName().empty());
}

TEST_F(UndoHistoryTest, UndoAndRedoMultipleSupportHistoryNavigationChains) {
    auto& history = lfs::vis::op::undoHistory();
    int value = 0;

    for (int i = 0; i < 5; ++i) {
        history.push(std::make_unique<CountingEntry>("step", value, 1));
        value += 1;
    }

    auto undo_result = history.undoMultiple(3);
    EXPECT_TRUE(undo_result.success);
    EXPECT_TRUE(undo_result.changed);
    EXPECT_EQ(undo_result.steps_performed, 3u);
    EXPECT_EQ(value, 2);

    auto redo_result = history.redoMultiple(2);
    EXPECT_TRUE(redo_result.success);
    EXPECT_EQ(redo_result.steps_performed, 2u);
    EXPECT_EQ(value, 4);

    auto single_undo = history.undo();
    EXPECT_TRUE(single_undo.success);
    EXPECT_EQ(value, 3);
}

TEST_F(UndoHistoryTest, UndoCallbacksCanQueryHistoryState) {
    auto& history = lfs::vis::op::undoHistory();
    bool queried = false;

    history.push(std::make_unique<ReentrantUndoEntry>(queried));
    history.undo();

    EXPECT_FALSE(queried);
    EXPECT_EQ(history.redoCount(), 1u);
}

TEST_F(UndoHistoryTest, FailedUndoLeavesEntryOnUndoStack) {
    auto& history = lfs::vis::op::undoHistory();

    history.push(std::make_unique<ThrowingUndoEntry>());
    history.undo();

    EXPECT_TRUE(history.canUndo());
    EXPECT_FALSE(history.canRedo());
    EXPECT_EQ(history.undoName(), "throwing.undo");
}

TEST_F(UndoHistoryTest, FailedUndoReturnsStructuredFailureResult) {
    auto& history = lfs::vis::op::undoHistory();

    history.push(std::make_unique<ThrowingUndoEntry>());
    const auto result = history.undo();

    EXPECT_FALSE(result.success);
    EXPECT_FALSE(result.changed);
    EXPECT_EQ(result.steps_performed, 0u);
    EXPECT_FALSE(result.error.empty());
    EXPECT_TRUE(history.canUndo());
}

TEST_F(UndoHistoryTest, FailedGroupedUndoCompensatesAlreadyUndoneChildren) {
    auto& history = lfs::vis::op::undoHistory();
    int value = 0;

    history.beginTransaction("compound.failure");
    history.push(std::make_unique<ThrowingUndoEntry>());
    history.push(std::make_unique<CountingEntry>("change.one", value, 1));
    value += 1;
    history.commitTransaction();

    const auto result = history.undo();
    EXPECT_FALSE(result.success);
    EXPECT_EQ(value, 1);
    EXPECT_TRUE(history.canUndo());
    EXPECT_FALSE(history.canRedo());
}

TEST_F(UndoHistoryTest, ObserversReceiveNotificationsUntilUnsubscribed) {
    auto& history = lfs::vis::op::undoHistory();
    int value = 0;
    int notifications = 0;

    const auto id = history.subscribe([&notifications]() { ++notifications; });
    history.push(std::make_unique<CountingEntry>("first", value, 1));
    value += 1;
    history.undo();
    history.redo();

    history.unsubscribe(id);
    history.clear();

    EXPECT_GE(notifications, 3);
}

TEST_F(UndoHistoryTest, TopologyUndoRestoresSoftDeletedMasks) {
    auto scene_manager = std::make_unique<lfs::vis::SceneManager>();
    auto rendering_manager = std::make_unique<lfs::vis::RenderingManager>();
    lfs::vis::services().set(scene_manager.get());
    lfs::vis::services().set(rendering_manager.get());

    scene_manager->getScene().addNode(
        "model",
        make_test_splat({
            0.0f,
            0.0f,
            0.0f,
            1.0f,
            0.0f,
            0.0f,
        }));

    auto selection = std::make_shared<Tensor>(make_uint8_mask({1, 0}));
    scene_manager->getScene().setSelectionMask(selection);

    scene_manager->deleteSelectedGaussians();

    auto* node = scene_manager->getScene().getNode("model");
    ASSERT_NE(node, nullptr);
    ASSERT_NE(node->model, nullptr);
    EXPECT_EQ(deleted_mask_values(*node->model), (std::vector<bool>{true, false}));

    lfs::vis::op::undoHistory().undo();
    EXPECT_TRUE(deleted_mask_values(*node->model).empty() ||
                deleted_mask_values(*node->model) == std::vector<bool>({false, false}));

    lfs::vis::op::undoHistory().redo();
    EXPECT_EQ(deleted_mask_values(*node->model), (std::vector<bool>{true, false}));
}

TEST_F(UndoHistoryTest, SceneResetClearsHistory) {
    auto scene_manager = std::make_unique<lfs::vis::SceneManager>();
    auto rendering_manager = std::make_unique<lfs::vis::RenderingManager>();
    lfs::vis::services().set(scene_manager.get());
    lfs::vis::services().set(rendering_manager.get());

    int value = 0;
    lfs::vis::op::undoHistory().push(std::make_unique<CountingEntry>("before.clear", value, 1));
    ASSERT_EQ(lfs::vis::op::undoHistory().undoCount(), 1u);

    scene_manager->clear();

    EXPECT_EQ(lfs::vis::op::undoHistory().undoCount(), 0u);
    EXPECT_EQ(lfs::vis::op::undoHistory().redoCount(), 0u);
}

TEST_F(UndoHistoryTest, DeletingLastNodeRemainsUndoable) {
    auto scene_manager = std::make_unique<lfs::vis::SceneManager>();
    auto rendering_manager = std::make_unique<lfs::vis::RenderingManager>();
    lfs::vis::services().set(scene_manager.get());
    lfs::vis::services().set(rendering_manager.get());

    scene_manager->getScene().addNode("model", make_test_splat({0.0f, 0.0f, 0.0f}));
    scene_manager->changeContentType(lfs::vis::SceneManager::ContentType::SplatFiles);

    scene_manager->removePLY("model");

    EXPECT_EQ(scene_manager->getScene().getNodeCount(), 0u);
    EXPECT_EQ(scene_manager->getContentType(), lfs::vis::SceneManager::ContentType::Empty);
    ASSERT_EQ(lfs::vis::op::undoHistory().undoCount(), 1u);

    lfs::vis::op::undoHistory().undo();
    EXPECT_NE(scene_manager->getScene().getNode("model"), nullptr);
    EXPECT_EQ(scene_manager->getContentType(), lfs::vis::SceneManager::ContentType::SplatFiles);

    lfs::vis::op::undoHistory().redo();
    EXPECT_EQ(scene_manager->getScene().getNode("model"), nullptr);
    EXPECT_EQ(scene_manager->getContentType(), lfs::vis::SceneManager::ContentType::Empty);
}

TEST_F(UndoHistoryTest, DeleteKeepChildrenRestoresHierarchyOnUndo) {
    auto scene_manager = std::make_unique<lfs::vis::SceneManager>();
    auto rendering_manager = std::make_unique<lfs::vis::RenderingManager>();
    lfs::vis::services().set(scene_manager.get());
    lfs::vis::services().set(rendering_manager.get());

    const auto group_id = scene_manager->getScene().addGroup("group");
    scene_manager->getScene().addSplat("child", make_test_splat({0.0f, 0.0f, 0.0f}), group_id);
    scene_manager->changeContentType(lfs::vis::SceneManager::ContentType::SplatFiles);

    scene_manager->removePLY("group", true);

    EXPECT_EQ(scene_manager->getScene().getNode("group"), nullptr);
    const auto* child = scene_manager->getScene().getNode("child");
    ASSERT_NE(child, nullptr);
    EXPECT_EQ(child->parent_id, lfs::core::NULL_NODE);

    lfs::vis::op::undoHistory().undo();
    const auto* restored_group = scene_manager->getScene().getNode("group");
    child = scene_manager->getScene().getNode("child");
    ASSERT_NE(restored_group, nullptr);
    ASSERT_NE(child, nullptr);
    EXPECT_EQ(child->parent_id, restored_group->id);

    lfs::vis::op::undoHistory().redo();
    EXPECT_EQ(scene_manager->getScene().getNode("group"), nullptr);
    child = scene_manager->getScene().getNode("child");
    ASSERT_NE(child, nullptr);
    EXPECT_EQ(child->parent_id, lfs::core::NULL_NODE);
}

TEST_F(UndoHistoryTest, RenameNodeCreatesUndoableSceneGraphEntry) {
    auto scene_manager = std::make_unique<lfs::vis::SceneManager>();
    auto rendering_manager = std::make_unique<lfs::vis::RenderingManager>();
    lfs::vis::services().set(scene_manager.get());
    lfs::vis::services().set(rendering_manager.get());

    scene_manager->getScene().addNode("old_name", make_test_splat({0.0f, 0.0f, 0.0f}));
    scene_manager->changeContentType(lfs::vis::SceneManager::ContentType::SplatFiles);

    ASSERT_TRUE(scene_manager->renamePLY("old_name", "new_name"));
    EXPECT_NE(scene_manager->getScene().getNode("new_name"), nullptr);
    ASSERT_EQ(lfs::vis::op::undoHistory().undoCount(), 1u);
    EXPECT_LT(lfs::vis::op::undoHistory().undoItems().front().estimated_bytes, 4096u);

    lfs::vis::op::undoHistory().undo();
    EXPECT_NE(scene_manager->getScene().getNode("old_name"), nullptr);
    EXPECT_EQ(scene_manager->getScene().getNode("new_name"), nullptr);

    lfs::vis::op::undoHistory().redo();
    EXPECT_EQ(scene_manager->getScene().getNode("old_name"), nullptr);
    EXPECT_NE(scene_manager->getScene().getNode("new_name"), nullptr);
}

TEST_F(UndoHistoryTest, ReparentNodeCreatesUndoableSceneGraphEntry) {
    auto scene_manager = std::make_unique<lfs::vis::SceneManager>();
    auto rendering_manager = std::make_unique<lfs::vis::RenderingManager>();
    lfs::vis::services().set(scene_manager.get());
    lfs::vis::services().set(rendering_manager.get());

    const auto parent_a = scene_manager->getScene().addGroup("A");
    const auto parent_b = scene_manager->getScene().addGroup("B");
    scene_manager->getScene().addSplat("child", make_test_splat({0.0f, 0.0f, 0.0f}), parent_a);
    scene_manager->changeContentType(lfs::vis::SceneManager::ContentType::SplatFiles);

    ASSERT_TRUE(scene_manager->reparentNode("child", "B"));
    auto* child = scene_manager->getScene().getNode("child");
    ASSERT_NE(child, nullptr);
    EXPECT_EQ(child->parent_id, parent_b);

    lfs::vis::op::undoHistory().undo();
    child = scene_manager->getScene().getNode("child");
    ASSERT_NE(child, nullptr);
    EXPECT_EQ(child->parent_id, parent_a);

    lfs::vis::op::undoHistory().redo();
    child = scene_manager->getScene().getNode("child");
    ASSERT_NE(child, nullptr);
    EXPECT_EQ(child->parent_id, parent_b);
}

TEST_F(UndoHistoryTest, AddGroupCreatesUndoableSceneGraphEntry) {
    auto scene_manager = std::make_unique<lfs::vis::SceneManager>();
    auto rendering_manager = std::make_unique<lfs::vis::RenderingManager>();
    lfs::vis::services().set(scene_manager.get());
    lfs::vis::services().set(rendering_manager.get());

    const auto group_name = scene_manager->addGroupNode("group");
    ASSERT_FALSE(group_name.empty());
    EXPECT_NE(scene_manager->getScene().getNode(group_name), nullptr);

    lfs::vis::op::undoHistory().undo();
    EXPECT_EQ(scene_manager->getScene().getNode(group_name), nullptr);

    lfs::vis::op::undoHistory().redo();
    EXPECT_NE(scene_manager->getScene().getNode(group_name), nullptr);
}

TEST_F(UndoHistoryTest, AnimatablePropertyWritesCreateUndoEntries) {
    auto scene_manager = std::make_unique<lfs::vis::SceneManager>();
    auto rendering_manager = std::make_unique<lfs::vis::RenderingManager>();
    lfs::vis::services().set(scene_manager.get());
    lfs::vis::services().set(rendering_manager.get());

    scene_manager->getScene().addNode("model", make_test_splat({0.0f, 0.0f, 0.0f}));
    auto* node = scene_manager->getScene().getMutableNode("model");
    ASSERT_NE(node, nullptr);

    node->visible = false;
    ASSERT_EQ(lfs::vis::op::undoHistory().undoCount(), 1u);
    EXPECT_FALSE(static_cast<bool>(node->visible));

    lfs::vis::op::undoHistory().undo();
    EXPECT_TRUE(static_cast<bool>(node->visible));

    lfs::vis::op::undoHistory().redo();
    EXPECT_FALSE(static_cast<bool>(node->visible));
}

TEST_F(UndoHistoryTest, DuplicateNodeCreatesUndoableSceneGraphEntry) {
    auto scene_manager = std::make_unique<lfs::vis::SceneManager>();
    auto rendering_manager = std::make_unique<lfs::vis::RenderingManager>();
    lfs::vis::services().set(scene_manager.get());
    lfs::vis::services().set(rendering_manager.get());

    scene_manager->getScene().addNode("model", make_test_splat({0.0f, 0.0f, 0.0f}));
    scene_manager->changeContentType(lfs::vis::SceneManager::ContentType::SplatFiles);

    const auto duplicate_name = scene_manager->duplicateNodeTree("model");
    ASSERT_FALSE(duplicate_name.empty());
    EXPECT_NE(scene_manager->getScene().getNode(duplicate_name), nullptr);

    lfs::vis::op::undoHistory().undo();
    EXPECT_EQ(scene_manager->getScene().getNode(duplicate_name), nullptr);

    lfs::vis::op::undoHistory().redo();
    EXPECT_NE(scene_manager->getScene().getNode(duplicate_name), nullptr);
}

TEST_F(UndoHistoryTest, SelectionSnapshotRestoresSelectionGroupsAndActiveGroup) {
    auto scene_manager = std::make_unique<lfs::vis::SceneManager>();
    auto rendering_manager = std::make_unique<lfs::vis::RenderingManager>();
    lfs::vis::services().set(scene_manager.get());
    lfs::vis::services().set(rendering_manager.get());

    scene_manager->getScene().addNode(
        "model",
        make_test_splat({
            0.0f,
            0.0f,
            0.0f,
            1.0f,
            0.0f,
            0.0f,
        }));

    const uint8_t second_group = scene_manager->getScene().addSelectionGroup("Second", {0.2f, 0.4f, 0.6f});
    scene_manager->getScene().setActiveSelectionGroup(second_group);
    scene_manager->getScene().setSelectionMask(std::make_shared<Tensor>(make_uint8_mask({1, second_group})));

    auto snapshot = std::make_unique<lfs::vis::op::SceneSnapshot>(*scene_manager, "selection.groups");
    snapshot->captureSelection();

    scene_manager->getScene().renameSelectionGroup(second_group, "Renamed");
    scene_manager->getScene().setSelectionGroupColor(second_group, {0.8f, 0.1f, 0.2f});
    scene_manager->getScene().setSelectionGroupLocked(second_group, true);
    scene_manager->getScene().setActiveSelectionGroup(1);
    scene_manager->getScene().clearSelection();

    snapshot->captureAfter();
    lfs::vis::op::undoHistory().push(std::move(snapshot));

    ASSERT_EQ(scene_manager->getScene().getActiveSelectionGroup(), 1);
    ASSERT_FALSE(scene_manager->getScene().hasSelection());
    ASSERT_TRUE(scene_manager->getScene().isSelectionGroupLocked(second_group));
    const auto* mutated_group = scene_manager->getScene().getSelectionGroup(second_group);
    ASSERT_NE(mutated_group, nullptr);
    ASSERT_EQ(mutated_group->name, "Renamed");

    lfs::vis::op::undoHistory().undo();

    const auto* restored_group = scene_manager->getScene().getSelectionGroup(second_group);
    ASSERT_NE(restored_group, nullptr);
    EXPECT_EQ(restored_group->name, "Second");
    EXPECT_FLOAT_EQ(restored_group->color.x, 0.2f);
    EXPECT_FLOAT_EQ(restored_group->color.y, 0.4f);
    EXPECT_FLOAT_EQ(restored_group->color.z, 0.6f);
    EXPECT_FALSE(restored_group->locked);
    EXPECT_EQ(scene_manager->getScene().getActiveSelectionGroup(), second_group);
    EXPECT_EQ(selection_mask_values(scene_manager->getScene()), (std::vector<uint8_t>{1, second_group}));

    lfs::vis::op::undoHistory().redo();

    const auto* redone_group = scene_manager->getScene().getSelectionGroup(second_group);
    ASSERT_NE(redone_group, nullptr);
    EXPECT_EQ(redone_group->name, "Renamed");
    EXPECT_TRUE(redone_group->locked);
    EXPECT_EQ(scene_manager->getScene().getActiveSelectionGroup(), 1);
    EXPECT_TRUE(selection_mask_values(scene_manager->getScene()).empty());
}

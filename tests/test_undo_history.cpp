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
#include "visualizer/gui_capabilities.hpp"

#include <any>
#include <condition_variable>
#include <future>
#include <glm/gtc/matrix_transform.hpp>
#include <gtest/gtest.h>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
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

    class UndoSucceedsRedoThrowsEntry final : public lfs::vis::op::UndoEntry {
    public:
        void undo() override {}
        void redo() override { throw std::runtime_error("redo failed"); }
        [[nodiscard]] std::string name() const override { return "undo.ok.redo.throws"; }
    };

    class BlockingUndoEntry final : public lfs::vis::op::UndoEntry {
    public:
        BlockingUndoEntry(std::mutex& mutex, std::condition_variable& cv, bool& started, bool& released)
            : mutex_(mutex),
              cv_(cv),
              started_(started),
              released_(released) {}

        void undo() override {
            {
                std::lock_guard lock(mutex_);
                started_ = true;
            }
            cv_.notify_all();

            std::unique_lock lock(mutex_);
            cv_.wait(lock, [this]() { return released_; });
        }

        void redo() override {}
        [[nodiscard]] std::string name() const override { return "blocking.undo"; }

    private:
        std::mutex& mutex_;
        std::condition_variable& cv_;
        bool& started_;
        bool& released_;
    };

    class TensorResidencyEntry final : public lfs::vis::op::UndoEntry {
    public:
        explicit TensorResidencyEntry(std::string name)
            : name_(std::move(name)),
              tensor_(Tensor::ones({16}, Device::CUDA, DataType::Float32)) {}

        void undo() override {
            undo_device_ = tensor_.device();
        }

        void redo() override {
            redo_device_ = tensor_.device();
        }

        [[nodiscard]] std::string name() const override { return name_; }
        [[nodiscard]] size_t estimatedBytes() const override { return tensor_.bytes(); }
        [[nodiscard]] lfs::vis::op::UndoMemoryBreakdown memoryBreakdown() const override {
            return tensor_.device() == Device::CUDA
                       ? lfs::vis::op::UndoMemoryBreakdown{.cpu_bytes = 0, .gpu_bytes = tensor_.bytes()}
                       : lfs::vis::op::UndoMemoryBreakdown{.cpu_bytes = tensor_.bytes(), .gpu_bytes = 0};
        }
        void offloadToCPU() override {
            if (tensor_.device() != Device::CPU) {
                tensor_ = tensor_.to(Device::CPU).contiguous();
            }
        }
        void restoreToPreferredDevice() override {
            if (tensor_.device() != Device::CUDA) {
                tensor_ = tensor_.to(Device::CUDA).contiguous();
            }
        }

        [[nodiscard]] Device device() const { return tensor_.device(); }
        [[nodiscard]] Device undoDevice() const { return undo_device_; }

    private:
        std::string name_;
        Tensor tensor_;
        Device undo_device_ = Device::CPU;
        Device redo_device_ = Device::CPU;
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

    std::unique_ptr<lfs::core::SplatData> make_linear_test_splat(const size_t count) {
        std::vector<float> xyz;
        xyz.reserve(count * 3);
        for (size_t i = 0; i < count; ++i) {
            xyz.push_back(static_cast<float>(i));
            xyz.push_back(0.0f);
            xyz.push_back(0.0f);
        }
        return make_test_splat(xyz);
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
        lfs::vis::op::undoHistory().setMaxBytes(lfs::vis::op::UndoHistory::MAX_BYTES);
    }

    void TearDown() override {
        lfs::vis::op::undoHistory().clear();
        lfs::vis::op::undoHistory().setMaxBytes(lfs::vis::op::UndoHistory::MAX_BYTES);
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

TEST_F(UndoHistoryTest, EmptyTransactionCommitCreatesNoUndoEntry) {
    auto& history = lfs::vis::op::undoHistory();

    history.beginTransaction("empty.transaction");
    history.commitTransaction();

    EXPECT_EQ(history.undoCount(), 0u);
    EXPECT_EQ(history.redoCount(), 0u);
    EXPECT_FALSE(history.canUndo());
    EXPECT_FALSE(history.canRedo());
    EXPECT_FALSE(history.hasActiveTransaction());
}

TEST_F(UndoHistoryTest, TransactionGuardCommitsAndRollsBackSafely) {
    auto& history = lfs::vis::op::undoHistory();
    int value = 0;

    {
        lfs::vis::op::TransactionGuard guard("guard.commit");
        history.push(std::make_unique<CountingEntry>("guard.step", value, 1));
        value += 1;
        guard.commit();
    }

    EXPECT_EQ(history.undoCount(), 1u);
    EXPECT_EQ(history.undoName(), "guard.commit");

    history.undo();
    EXPECT_EQ(value, 0);

    history.clear();
    value = 0;

    {
        lfs::vis::op::TransactionGuard guard("guard.rollback");
        history.push(std::make_unique<CountingEntry>("guard.step", value, 2));
        value += 2;
    }

    EXPECT_EQ(value, 0);
    EXPECT_EQ(history.undoCount(), 0u);
    EXPECT_FALSE(history.hasActiveTransaction());
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

    history.push(std::make_unique<CountingEntry>("large.one", value, 1, 200ull * 1024ull * 1024ull));
    history.push(std::make_unique<CountingEntry>("large.two", value, 1, 200ull * 1024ull * 1024ull));
    history.push(std::make_unique<CountingEntry>("large.three", value, 1, 200ull * 1024ull * 1024ull));

    EXPECT_EQ(history.undoCount(), 2u);
    EXPECT_EQ(history.undoNames(), (std::vector<std::string>{"large.three", "large.two"}));
    EXPECT_LE(history.undoBytes(), lfs::vis::op::UndoHistory::MAX_BYTES);
}

TEST_F(UndoHistoryTest, ConfigurableByteBudgetTrimsHistoryAndUpdatesGetter) {
    auto& history = lfs::vis::op::undoHistory();
    int value = 0;

    history.setMaxBytes(256u);
    EXPECT_EQ(history.maxBytes(), 256u);

    history.push(std::make_unique<CountingEntry>("budget.one", value, 1, 128u));
    history.push(std::make_unique<CountingEntry>("budget.two", value, 1, 128u));
    history.push(std::make_unique<CountingEntry>("budget.three", value, 1, 128u));

    EXPECT_EQ(history.undoCount(), 2u);
    EXPECT_EQ(history.undoNames(), (std::vector<std::string>{"budget.three", "budget.two"}));
    EXPECT_LE(history.undoBytes(), 256u);
}

TEST_F(UndoHistoryTest, OversizedSingleUndoEntryIsRetained) {
    auto& history = lfs::vis::op::undoHistory();
    int value = 0;

    history.push(std::make_unique<CountingEntry>("huge.entry", value, 1, 600ull * 1024ull * 1024ull));

    EXPECT_EQ(history.undoCount(), 1u);
    EXPECT_EQ(history.undoName(), "huge.entry");
    EXPECT_EQ(history.undoBytes(), 600ull * 1024ull * 1024ull);
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
    EXPECT_EQ(items.front().metadata.id, "history.transaction.grouped_transaction");
    EXPECT_EQ(items.front().metadata.label, "Grouped Transaction");
    EXPECT_EQ(items.front().metadata.source, "history");
    EXPECT_EQ(items.front().metadata.scope, "grouped");
    EXPECT_EQ(items.front().estimated_bytes, 128u);
    EXPECT_FALSE(history.hasActiveTransaction());
    EXPECT_EQ(history.transactionDepth(), 0u);
    EXPECT_TRUE(history.activeTransactionName().empty());
}

TEST_F(UndoHistoryTest, PropertyChangesCoalesceOnUndoStack) {
    auto& history = lfs::vis::op::undoHistory();
    float value = 0.0f;
    auto applier = [&value](const std::any& applied) { value = std::any_cast<float>(applied); };

    value = 0.5f;
    history.push(std::make_unique<lfs::vis::op::PropertyChangeUndoEntry>(
        "scene_node.model.opacity", 0.0f, 0.5f, applier));
    value = 1.0f;
    history.push(std::make_unique<lfs::vis::op::PropertyChangeUndoEntry>(
        "scene_node.model.opacity", 0.5f, 1.0f, applier));

    ASSERT_EQ(history.undoCount(), 1u);

    const auto undo_result = history.undo();
    EXPECT_TRUE(undo_result.success);
    EXPECT_FLOAT_EQ(value, 0.0f);

    const auto redo_result = history.redo();
    EXPECT_TRUE(redo_result.success);
    EXPECT_FLOAT_EQ(value, 1.0f);
}

TEST_F(UndoHistoryTest, PropertyChangesCoalesceInsideActiveTransaction) {
    auto& history = lfs::vis::op::undoHistory();
    float value = 0.0f;
    auto applier = [&value](const std::any& applied) { value = std::any_cast<float>(applied); };

    history.beginTransaction("drag.opacity");
    value = 0.5f;
    history.push(std::make_unique<lfs::vis::op::PropertyChangeUndoEntry>(
        "scene_node.model.opacity", 0.0f, 0.5f, applier));
    value = 1.0f;
    history.push(std::make_unique<lfs::vis::op::PropertyChangeUndoEntry>(
        "scene_node.model.opacity", 0.5f, 1.0f, applier));

    EXPECT_GT(history.transactionBytes(), 0u);

    const auto rollback = history.rollbackTransaction();
    EXPECT_TRUE(rollback.success);
    EXPECT_EQ(rollback.steps_performed, 1u);
    EXPECT_FLOAT_EQ(value, 0.0f);
    EXPECT_EQ(history.transactionBytes(), 0u);
}

TEST_F(UndoHistoryTest, TotalBytesIncludesActiveTransactionBytes) {
    auto& history = lfs::vis::op::undoHistory();
    int value = 0;

    history.beginTransaction("memory.visible");
    history.push(std::make_unique<CountingEntry>("pending.step", value, 1, 4096));

    EXPECT_EQ(history.undoBytes(), 0u);
    EXPECT_EQ(history.redoBytes(), 0u);
    EXPECT_EQ(history.transactionBytes(), 4096u);
    EXPECT_EQ(history.totalBytes(), 4096u);

    history.commitTransaction();
    EXPECT_EQ(history.transactionBytes(), 0u);
    EXPECT_EQ(history.totalBytes(), history.undoBytes());
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

TEST_F(UndoHistoryTest, RollbackFailureRestoresActiveTransactionWhenCompensationSucceeds) {
    auto& history = lfs::vis::op::undoHistory();
    int value = 1;

    history.beginTransaction("rollback.failure");
    history.push(std::make_unique<ThrowingUndoEntry>());
    history.push(std::make_unique<CountingEntry>("change.one", value, 1));
    value += 1;

    const auto result = history.rollbackTransaction();
    EXPECT_FALSE(result.success);
    EXPECT_EQ(value, 2);
    EXPECT_TRUE(history.hasActiveTransaction());
    EXPECT_EQ(history.transactionDepth(), 1u);
    EXPECT_EQ(history.activeTransactionName(), "rollback.failure");

    history.commitTransaction();
    EXPECT_EQ(history.undoCount(), 1u);
}

TEST_F(UndoHistoryTest, RollbackFailureClearsHistoryAfterCompensationFailure) {
    auto& history = lfs::vis::op::undoHistory();

    history.beginTransaction("rollback.fatal");
    history.push(std::make_unique<ThrowingUndoEntry>());
    history.push(std::make_unique<UndoSucceedsRedoThrowsEntry>());

    const auto result = history.rollbackTransaction();
    EXPECT_FALSE(result.success);
    EXPECT_NE(result.error.find("history cleared"), std::string::npos);
    EXPECT_FALSE(history.hasActiveTransaction());
    EXPECT_EQ(history.undoCount(), 0u);
    EXPECT_EQ(history.redoCount(), 0u);
}

TEST_F(UndoHistoryTest, ConcurrentPlaybackIsRejected) {
    auto& history = lfs::vis::op::undoHistory();
    std::mutex mutex;
    std::condition_variable cv;
    bool started = false;
    bool released = false;
    lfs::vis::op::HistoryResult worker_result;

    history.push(std::make_unique<BlockingUndoEntry>(mutex, cv, started, released));

    std::thread worker([&]() { worker_result = history.undo(); });

    {
        std::unique_lock lock(mutex);
        cv.wait(lock, [&]() { return started; });
    }

    const auto result = history.undo();
    EXPECT_FALSE(result.success);
    EXPECT_FALSE(result.changed);
    EXPECT_EQ(result.steps_performed, 0u);
    EXPECT_EQ(result.error, "History playback already in progress");

    {
        std::lock_guard lock(mutex);
        released = true;
    }
    cv.notify_all();
    worker.join();

    EXPECT_TRUE(worker_result.success);
    EXPECT_TRUE(history.canRedo());
}

TEST_F(UndoHistoryTest, PushWaitsForPlaybackAndAppliesAfterUndoCompletes) {
    auto& history = lfs::vis::op::undoHistory();
    int value = 0;
    std::mutex mutex;
    std::condition_variable cv;
    bool started = false;
    bool released = false;
    lfs::vis::op::HistoryResult worker_result;

    history.push(std::make_unique<CountingEntry>("step.one", value, 1));
    value += 1;
    history.push(std::make_unique<BlockingUndoEntry>(mutex, cv, started, released));

    std::thread worker([&]() { worker_result = history.undo(); });

    {
        std::unique_lock lock(mutex);
        cv.wait(lock, [&]() { return started; });
    }

    std::promise<void> push_finished;
    auto push_done = push_finished.get_future();
    std::thread pusher([&]() {
        history.push(std::make_unique<CountingEntry>("step.two", value, 2));
        value += 2;
        push_finished.set_value();
    });

    EXPECT_EQ(push_done.wait_for(std::chrono::milliseconds(50)), std::future_status::timeout);

    {
        std::lock_guard lock(mutex);
        released = true;
    }
    cv.notify_all();

    worker.join();
    pusher.join();

    EXPECT_TRUE(worker_result.success);
    EXPECT_EQ(value, 3);
    EXPECT_EQ(history.redoCount(), 0u);
    EXPECT_EQ(history.undoCount(), 2u);
    EXPECT_EQ(history.undoName(), "step.two");
}

TEST_F(UndoHistoryTest, PushInsideActiveTransactionKeepsRedoUntilCommit) {
    auto& history = lfs::vis::op::undoHistory();
    int value = 0;

    history.push(std::make_unique<CountingEntry>("step.one", value, 1));
    value += 1;
    history.push(std::make_unique<CountingEntry>("step.two", value, 1));
    value += 1;
    history.undo();
    ASSERT_TRUE(history.canRedo());

    history.beginTransaction("transactional.change");
    history.push(std::make_unique<CountingEntry>("step.three", value, 5));
    value += 5;
    EXPECT_TRUE(history.canRedo());
    history.commitTransaction();

    EXPECT_FALSE(history.canRedo());
}

TEST_F(UndoHistoryTest, RedoStackStaysWithinConfiguredBoundsAfterUndoingHistory) {
    auto& history = lfs::vis::op::undoHistory();
    int value = 0;

    for (size_t i = 0; i < lfs::vis::op::UndoHistory::MAX_ENTRIES + 5; ++i) {
        history.push(std::make_unique<CountingEntry>("bounded.step", value, 1));
        value += 1;
    }

    const auto result = history.undoMultiple(history.undoCount());
    EXPECT_TRUE(result.success);
    EXPECT_EQ(history.redoCount(), lfs::vis::op::UndoHistory::MAX_ENTRIES);
    EXPECT_LE(history.redoBytes(), lfs::vis::op::UndoHistory::MAX_BYTES);
}

TEST_F(UndoHistoryTest, OlderTensorEntriesOffloadToCPUAndRestoreBeforePlayback) {
    auto& history = lfs::vis::op::undoHistory();
    std::vector<TensorResidencyEntry*> entries;

    for (size_t i = 0; i < lfs::vis::op::UndoHistory::HOT_ENTRIES + 2; ++i) {
        auto entry = std::make_unique<TensorResidencyEntry>("tensor.entry." + std::to_string(i));
        entries.push_back(entry.get());
        history.push(std::move(entry));
    }

    ASSERT_EQ(history.undoCount(), lfs::vis::op::UndoHistory::HOT_ENTRIES + 2);
    EXPECT_EQ(entries.front()->device(), Device::CPU);
    EXPECT_EQ(entries.back()->device(), Device::CUDA);

    const auto memory = history.undoMemory();
    EXPECT_GT(memory.cpu_bytes, 0u);
    EXPECT_GT(memory.gpu_bytes, 0u);

    const auto undo_items = history.undoItems();
    ASSERT_EQ(undo_items.size(), lfs::vis::op::UndoHistory::HOT_ENTRIES + 2);
    EXPECT_EQ(undo_items.front().gpu_bytes, entries.back()->estimatedBytes());
    EXPECT_EQ(undo_items.back().gpu_bytes, 0u);
    EXPECT_GT(undo_items.back().cpu_bytes, 0u);

    const auto result = history.undoMultiple(history.undoCount());
    EXPECT_TRUE(result.success);
    EXPECT_EQ(entries.front()->undoDevice(), Device::CUDA);
}

TEST_F(UndoHistoryTest, ShrinkToFitOffloadsHistoryToMeetGpuBudget) {
    auto& history = lfs::vis::op::undoHistory();
    std::vector<TensorResidencyEntry*> entries;

    for (size_t i = 0; i < lfs::vis::op::UndoHistory::HOT_ENTRIES + 2; ++i) {
        auto entry = std::make_unique<TensorResidencyEntry>("tensor.entry." + std::to_string(i));
        entries.push_back(entry.get());
        history.push(std::move(entry));
    }

    const auto generation_before = history.generation();
    ASSERT_GT(history.totalMemory().gpu_bytes, 0u);

    history.shrinkToFit(0u);

    EXPECT_EQ(history.totalMemory().gpu_bytes, 0u);
    EXPECT_EQ(history.undoCount(), lfs::vis::op::UndoHistory::HOT_ENTRIES + 2);
    EXPECT_GT(history.generation(), generation_before);
    for (auto* entry : entries) {
        ASSERT_NE(entry, nullptr);
        EXPECT_EQ(entry->device(), Device::CPU);
    }
}

TEST_F(UndoHistoryTest, TensorUndoEntryRestoresTensorRoundTrip) {
    auto scene_manager = std::make_unique<lfs::vis::SceneManager>();
    auto rendering_manager = std::make_unique<lfs::vis::RenderingManager>();
    lfs::vis::services().set(scene_manager.get());
    lfs::vis::services().set(rendering_manager.get());

    scene_manager->getScene().addNode("model", make_linear_test_splat(2));
    auto* node = scene_manager->getScene().getMutableNode("model");
    ASSERT_NE(node, nullptr);
    ASSERT_NE(node->model, nullptr);

    const auto before = node->model->sh0().clone();
    auto entry = std::make_unique<lfs::vis::op::TensorUndoEntry>(
        "brush.saturation",
        lfs::vis::op::UndoMetadata{
            .id = "tensor.saturation",
            .label = "Saturation Brush",
            .source = "operator",
            .scope = "tensor",
        },
        "model.sh0",
        before.clone(),
        [&]() -> Tensor* { return &node->model->sh0(); });

    node->model->sh0() = Tensor::ones({2, size_t{1}, size_t{3}}, Device::CUDA, DataType::Float32);
    const auto after = node->model->sh0().clone();

    entry->captureAfter();
    ASSERT_TRUE(entry->hasChanges());
    lfs::vis::op::undoHistory().push(std::move(entry));

    const auto undo_result = lfs::vis::op::undoHistory().undo();
    EXPECT_TRUE(undo_result.success);
    EXPECT_TRUE((node->model->sh0() == before).all().item<bool>());

    const auto redo_result = lfs::vis::op::undoHistory().redo();
    EXPECT_TRUE(redo_result.success);
    EXPECT_TRUE((node->model->sh0() == after).all().item<bool>());
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

TEST_F(UndoHistoryTest, ThrowingObserversAreAutomaticallyUnsubscribed) {
    auto& history = lfs::vis::op::undoHistory();
    int value = 0;
    int throwing_notifications = 0;
    int healthy_notifications = 0;

    const auto throwing_id = history.subscribe([&]() {
        ++throwing_notifications;
        throw std::runtime_error("observer failed");
    });
    const auto healthy_id = history.subscribe([&]() { ++healthy_notifications; });

    history.push(std::make_unique<CountingEntry>("first", value, 1));
    value += 1;
    history.push(std::make_unique<CountingEntry>("second", value, 1));
    value += 1;

    EXPECT_EQ(throwing_notifications, 1);
    EXPECT_EQ(healthy_notifications, 2);

    history.unsubscribe(throwing_id);
    history.unsubscribe(healthy_id);
}

TEST_F(UndoHistoryTest, SceneGraphMetadataEntryRollsBackEarlierDiffsOnFailure) {
    auto scene_manager = std::make_unique<lfs::vis::SceneManager>();
    lfs::vis::services().set(scene_manager.get());

    scene_manager->getScene().addNode("first", make_linear_test_splat(1));
    scene_manager->getScene().addNode("second", make_linear_test_splat(1));

    const auto before = lfs::vis::op::SceneGraphMetadataEntry::captureNodes(*scene_manager, {"first", "second"});
    ASSERT_EQ(before.size(), 2u);

    auto first_after = before[0];
    first_after.name = "first_renamed";

    auto second_after = before[1];
    second_after.name = "second_renamed";
    second_after.parent_name = "missing_parent";

    lfs::vis::op::SceneGraphMetadataEntry entry(
        *scene_manager,
        "Rename Node",
        {
            lfs::vis::op::SceneGraphNodeMetadataDiff{.before = before[0], .after = first_after},
            lfs::vis::op::SceneGraphNodeMetadataDiff{.before = before[1], .after = second_after},
        });

    EXPECT_THROW(entry.redo(), std::runtime_error);
    EXPECT_NE(scene_manager->getScene().getNode("first"), nullptr);
    EXPECT_NE(scene_manager->getScene().getNode("second"), nullptr);
    EXPECT_EQ(scene_manager->getScene().getNode("first_renamed"), nullptr);
    EXPECT_EQ(scene_manager->getScene().getNode("second_renamed"), nullptr);
}

TEST_F(UndoHistoryTest, SparseSelectionUndoIsRejectedAfterTopologyChange) {
    auto scene_manager = std::make_unique<lfs::vis::SceneManager>();
    lfs::vis::services().set(scene_manager.get());

    scene_manager->getScene().addNode("model", make_linear_test_splat(32));

    auto snapshot = std::make_unique<lfs::vis::op::SceneSnapshot>(*scene_manager, "selection.change");
    snapshot->captureSelection();
    scene_manager->getScene().setSelectionMask(std::make_shared<Tensor>(make_uint8_mask(
        [] {
            std::vector<uint8_t> values(32, 0);
            values[3] = 1;
            return values;
        }())));
    snapshot->captureAfter();
    ASSERT_TRUE(snapshot->hasChanges());

    scene_manager->getScene().removeNode("model", false);
    scene_manager->getScene().addNode("model_replaced", make_linear_test_splat(48));

    EXPECT_THROW(snapshot->undo(), std::runtime_error);
}

TEST_F(UndoHistoryTest, CropBoxUndoEntryRoundTripsAndHandlesDeletedNode) {
    auto scene_manager = std::make_unique<lfs::vis::SceneManager>();
    auto rendering_manager = std::make_unique<lfs::vis::RenderingManager>();
    lfs::vis::services().set(scene_manager.get());
    lfs::vis::services().set(rendering_manager.get());

    scene_manager->getScene().addNode("model", make_linear_test_splat(2));
    const auto parent_id = scene_manager->getScene().getNodeIdByName("model");
    ASSERT_NE(parent_id, lfs::core::NULL_NODE);

    const auto cropbox_id = scene_manager->getScene().addCropBox("model_cropbox", parent_id);
    ASSERT_NE(cropbox_id, lfs::core::NULL_NODE);
    auto* cropbox_node = scene_manager->getScene().getMutableNode("model_cropbox");
    ASSERT_NE(cropbox_node, nullptr);
    ASSERT_NE(cropbox_node->cropbox, nullptr);

    const auto before_data = *cropbox_node->cropbox;
    const auto before_transform = scene_manager->getNodeTransform(cropbox_node->name);
    auto crop_settings = rendering_manager->getSettings();
    crop_settings.show_crop_box = true;
    crop_settings.use_crop_box = false;
    rendering_manager->updateSettings(crop_settings);
    lfs::vis::op::CropBoxUndoEntry noop(
        *scene_manager,
        rendering_manager.get(),
        cropbox_node->name,
        before_data,
        before_transform,
        crop_settings.show_crop_box,
        crop_settings.use_crop_box);
    EXPECT_FALSE(noop.hasChanges());
    EXPECT_GT(noop.estimatedBytes(), cropbox_node->name.size());

    auto changed_data = before_data;
    changed_data.min = glm::vec3(-2.0f);
    changed_data.max = glm::vec3(3.0f);
    scene_manager->getScene().setCropBoxData(cropbox_id, changed_data);
    scene_manager->setNodeTransform(cropbox_node->name, glm::translate(glm::mat4(1.0f), glm::vec3(1.0f, 2.0f, 3.0f)));
    crop_settings.show_crop_box = false;
    crop_settings.use_crop_box = true;
    rendering_manager->updateSettings(crop_settings);

    lfs::vis::op::CropBoxUndoEntry entry(
        *scene_manager,
        rendering_manager.get(),
        cropbox_node->name,
        before_data,
        before_transform,
        true,
        false);
    ASSERT_TRUE(entry.hasChanges());

    entry.undo();
    EXPECT_EQ(cropbox_node->cropbox->min, before_data.min);
    EXPECT_EQ(cropbox_node->cropbox->max, before_data.max);
    EXPECT_EQ(cropbox_node->cropbox->inverse, before_data.inverse);
    EXPECT_EQ(cropbox_node->cropbox->enabled, before_data.enabled);
    EXPECT_EQ(scene_manager->getNodeTransform(cropbox_node->name), before_transform);
    crop_settings = rendering_manager->getSettings();
    EXPECT_TRUE(crop_settings.show_crop_box);
    EXPECT_FALSE(crop_settings.use_crop_box);

    entry.redo();
    EXPECT_EQ(cropbox_node->cropbox->min, changed_data.min);
    EXPECT_EQ(cropbox_node->cropbox->max, changed_data.max);
    EXPECT_EQ(cropbox_node->cropbox->inverse, changed_data.inverse);
    EXPECT_EQ(cropbox_node->cropbox->enabled, changed_data.enabled);
    crop_settings = rendering_manager->getSettings();
    EXPECT_FALSE(crop_settings.show_crop_box);
    EXPECT_TRUE(crop_settings.use_crop_box);

    scene_manager->getScene().removeNode(cropbox_node->name, false);
    entry.undo();
    entry.redo();
}

TEST_F(UndoHistoryTest, EllipsoidUndoEntryRoundTripsAndHandlesDeletedNode) {
    auto scene_manager = std::make_unique<lfs::vis::SceneManager>();
    auto rendering_manager = std::make_unique<lfs::vis::RenderingManager>();
    lfs::vis::services().set(scene_manager.get());
    lfs::vis::services().set(rendering_manager.get());

    scene_manager->getScene().addNode("model", make_linear_test_splat(2));
    const auto parent_id = scene_manager->getScene().getNodeIdByName("model");
    ASSERT_NE(parent_id, lfs::core::NULL_NODE);

    const auto ellipsoid_id = scene_manager->getScene().addEllipsoid("model_ellipsoid", parent_id);
    ASSERT_NE(ellipsoid_id, lfs::core::NULL_NODE);
    auto* ellipsoid_node = scene_manager->getScene().getMutableNode("model_ellipsoid");
    ASSERT_NE(ellipsoid_node, nullptr);
    ASSERT_NE(ellipsoid_node->ellipsoid, nullptr);

    const auto before_data = *ellipsoid_node->ellipsoid;
    const auto before_transform = scene_manager->getNodeTransform(ellipsoid_node->name);
    auto ellipsoid_settings = rendering_manager->getSettings();
    ellipsoid_settings.show_ellipsoid = true;
    ellipsoid_settings.use_ellipsoid = false;
    rendering_manager->updateSettings(ellipsoid_settings);
    lfs::vis::op::EllipsoidUndoEntry noop(
        *scene_manager,
        rendering_manager.get(),
        ellipsoid_node->name,
        before_data,
        before_transform,
        ellipsoid_settings.show_ellipsoid,
        ellipsoid_settings.use_ellipsoid);
    EXPECT_FALSE(noop.hasChanges());
    EXPECT_GT(noop.estimatedBytes(), ellipsoid_node->name.size());

    auto changed_data = before_data;
    changed_data.radii = glm::vec3(4.0f, 5.0f, 6.0f);
    scene_manager->getScene().setEllipsoidData(ellipsoid_id, changed_data);
    scene_manager->setNodeTransform(ellipsoid_node->name, glm::translate(glm::mat4(1.0f), glm::vec3(2.0f, 0.0f, 1.0f)));
    ellipsoid_settings.show_ellipsoid = false;
    ellipsoid_settings.use_ellipsoid = true;
    rendering_manager->updateSettings(ellipsoid_settings);

    lfs::vis::op::EllipsoidUndoEntry entry(
        *scene_manager,
        rendering_manager.get(),
        ellipsoid_node->name,
        before_data,
        before_transform,
        true,
        false);
    ASSERT_TRUE(entry.hasChanges());

    entry.undo();
    EXPECT_EQ(ellipsoid_node->ellipsoid->radii, before_data.radii);
    EXPECT_EQ(ellipsoid_node->ellipsoid->inverse, before_data.inverse);
    EXPECT_EQ(ellipsoid_node->ellipsoid->enabled, before_data.enabled);
    EXPECT_EQ(scene_manager->getNodeTransform(ellipsoid_node->name), before_transform);
    ellipsoid_settings = rendering_manager->getSettings();
    EXPECT_TRUE(ellipsoid_settings.show_ellipsoid);
    EXPECT_FALSE(ellipsoid_settings.use_ellipsoid);

    entry.redo();
    EXPECT_EQ(ellipsoid_node->ellipsoid->radii, changed_data.radii);
    EXPECT_EQ(ellipsoid_node->ellipsoid->inverse, changed_data.inverse);
    EXPECT_EQ(ellipsoid_node->ellipsoid->enabled, changed_data.enabled);
    ellipsoid_settings = rendering_manager->getSettings();
    EXPECT_FALSE(ellipsoid_settings.show_ellipsoid);
    EXPECT_TRUE(ellipsoid_settings.use_ellipsoid);

    scene_manager->getScene().removeNode(ellipsoid_node->name, false);
    entry.undo();
    entry.redo();
}

TEST_F(UndoHistoryTest, GaussianFieldWritePushesUndoableTensorEntries) {
    auto scene_manager = std::make_unique<lfs::vis::SceneManager>();
    auto rendering_manager = std::make_unique<lfs::vis::RenderingManager>();
    lfs::vis::services().set(scene_manager.get());
    lfs::vis::services().set(rendering_manager.get());

    scene_manager->getScene().addNode("model", make_linear_test_splat(2));
    auto* node = scene_manager->getScene().getMutableNode("model");
    ASSERT_NE(node, nullptr);
    ASSERT_NE(node->model, nullptr);

    const auto original_opacity = node->model->opacity_raw().clone();
    auto write_result = lfs::vis::cap::writeGaussianField(
        *scene_manager, rendering_manager.get(), "model", "opacity_raw", {0}, {0.75f});
    ASSERT_TRUE(write_result) << write_result.error();
    EXPECT_EQ(lfs::vis::op::undoHistory().undoName(), "Edit Opacity");
    EXPECT_NE(node->model->opacity_raw().cpu().to_vector()[0], original_opacity.cpu().to_vector()[0]);

    auto undo_result = lfs::vis::op::undoHistory().undo();
    EXPECT_TRUE(undo_result.success);
    EXPECT_TRUE((node->model->opacity_raw() == original_opacity).all().item<bool>());

    auto redo_result = lfs::vis::op::undoHistory().redo();
    EXPECT_TRUE(redo_result.success);
    EXPECT_FLOAT_EQ(node->model->opacity_raw().cpu().to_vector()[0], 0.75f);

    const auto original_means = node->model->means_raw().clone();
    write_result = lfs::vis::cap::writeGaussianField(
        *scene_manager,
        rendering_manager.get(),
        "model",
        "means",
        {0},
        {1.5f, 0.0f, 0.0f});
    ASSERT_TRUE(write_result) << write_result.error();
    EXPECT_EQ(lfs::vis::op::undoHistory().undoName(), "Edit Means");

    undo_result = lfs::vis::op::undoHistory().undo();
    EXPECT_TRUE(undo_result.success);
    EXPECT_TRUE((node->model->means_raw() == original_means).all().item<bool>());
}

TEST_F(UndoHistoryTest, CropBoxCapabilityUndoRestoresRenderSettings) {
    auto scene_manager = std::make_unique<lfs::vis::SceneManager>();
    auto rendering_manager = std::make_unique<lfs::vis::RenderingManager>();
    lfs::vis::services().set(scene_manager.get());
    lfs::vis::services().set(rendering_manager.get());

    scene_manager->getScene().addNode("model", make_linear_test_splat(2));
    const auto parent_id = scene_manager->getScene().getNodeIdByName("model");
    ASSERT_NE(parent_id, lfs::core::NULL_NODE);
    const auto cropbox_id = scene_manager->getScene().addCropBox("model_cropbox", parent_id);
    ASSERT_NE(cropbox_id, lfs::core::NULL_NODE);

    auto settings = rendering_manager->getSettings();
    settings.show_crop_box = true;
    settings.use_crop_box = false;
    rendering_manager->updateSettings(settings);

    lfs::vis::cap::CropBoxUpdate update;
    update.translation = glm::vec3(1.0f, 2.0f, 3.0f);
    update.has_show = true;
    update.show = false;
    update.has_use = true;
    update.use = true;
    auto crop_result = lfs::vis::cap::updateCropBox(*scene_manager, rendering_manager.get(), cropbox_id, update);
    ASSERT_TRUE(crop_result) << crop_result.error();

    settings = rendering_manager->getSettings();
    EXPECT_FALSE(settings.show_crop_box);
    EXPECT_TRUE(settings.use_crop_box);

    auto undo_result = lfs::vis::op::undoHistory().undo();
    EXPECT_TRUE(undo_result.success);
    settings = rendering_manager->getSettings();
    EXPECT_TRUE(settings.show_crop_box);
    EXPECT_FALSE(settings.use_crop_box);

    auto redo_result = lfs::vis::op::undoHistory().redo();
    EXPECT_TRUE(redo_result.success);
    settings = rendering_manager->getSettings();
    EXPECT_FALSE(settings.show_crop_box);
    EXPECT_TRUE(settings.use_crop_box);
}

TEST_F(UndoHistoryTest, CropBoxResetUndoRestoresUseToggle) {
    auto scene_manager = std::make_unique<lfs::vis::SceneManager>();
    auto rendering_manager = std::make_unique<lfs::vis::RenderingManager>();
    lfs::vis::services().set(scene_manager.get());
    lfs::vis::services().set(rendering_manager.get());

    scene_manager->getScene().addNode("model", make_linear_test_splat(2));
    const auto parent_id = scene_manager->getScene().getNodeIdByName("model");
    ASSERT_NE(parent_id, lfs::core::NULL_NODE);
    const auto cropbox_id = scene_manager->getScene().addCropBox("model_cropbox", parent_id);
    ASSERT_NE(cropbox_id, lfs::core::NULL_NODE);

    auto settings = rendering_manager->getSettings();
    settings.show_crop_box = false;
    settings.use_crop_box = true;
    rendering_manager->updateSettings(settings);

    auto* cropbox_node = scene_manager->getScene().getMutableNode("model_cropbox");
    ASSERT_NE(cropbox_node, nullptr);
    scene_manager->setNodeTransform(cropbox_node->name, glm::translate(glm::mat4(1.0f), glm::vec3(0.5f, 0.0f, 0.0f)));

    auto reset_result = lfs::vis::cap::resetCropBox(*scene_manager, rendering_manager.get(), cropbox_id);
    ASSERT_TRUE(reset_result) << reset_result.error();
    settings = rendering_manager->getSettings();
    EXPECT_FALSE(settings.use_crop_box);

    auto undo_result = lfs::vis::op::undoHistory().undo();
    EXPECT_TRUE(undo_result.success);
    settings = rendering_manager->getSettings();
    EXPECT_TRUE(settings.use_crop_box);
    EXPECT_EQ(scene_manager->getNodeTransform(cropbox_node->name),
              glm::translate(glm::mat4(1.0f), glm::vec3(0.5f, 0.0f, 0.0f)));
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
    EXPECT_FALSE(node->model->has_deleted_mask());

    lfs::vis::op::undoHistory().redo();
    EXPECT_EQ(deleted_mask_values(*node->model), (std::vector<bool>{true, false}));
}

TEST_F(UndoHistoryTest, SceneSnapshotCompactsSparseSelectionMasks) {
    auto scene_manager = std::make_unique<lfs::vis::SceneManager>();
    auto rendering_manager = std::make_unique<lfs::vis::RenderingManager>();
    lfs::vis::services().set(scene_manager.get());
    lfs::vis::services().set(rendering_manager.get());

    scene_manager->getScene().addNode("model", make_linear_test_splat(16));

    auto snapshot = std::make_unique<lfs::vis::op::SceneSnapshot>(*scene_manager, "selection.sparse");
    snapshot->captureSelection();
    scene_manager->getScene().setSelectionMask(
        std::make_shared<Tensor>(make_uint8_mask({0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0})));
    snapshot->captureAfter();

    EXPECT_LT(snapshot->estimatedBytes(), 16u);

    lfs::vis::op::undoHistory().push(std::move(snapshot));
    lfs::vis::op::undoHistory().undo();
    EXPECT_TRUE(selection_mask_values(scene_manager->getScene()).empty());

    lfs::vis::op::undoHistory().redo();
    EXPECT_EQ(selection_mask_values(scene_manager->getScene()),
              (std::vector<uint8_t>{0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}));
}

TEST_F(UndoHistoryTest, PushSceneSnapshotIfChangedSkipsNoOpSnapshots) {
    auto scene_manager = std::make_unique<lfs::vis::SceneManager>();
    auto rendering_manager = std::make_unique<lfs::vis::RenderingManager>();
    lfs::vis::services().set(scene_manager.get());
    lfs::vis::services().set(rendering_manager.get());

    scene_manager->getScene().addNode("model", make_linear_test_splat(4));

    auto snapshot = std::make_unique<lfs::vis::op::SceneSnapshot>(*scene_manager, "selection.noop");
    snapshot->captureSelection();
    snapshot->captureAfter();

    EXPECT_FALSE(lfs::vis::op::pushSceneSnapshotIfChanged(std::move(snapshot)));
    EXPECT_EQ(lfs::vis::op::undoHistory().undoCount(), 0u);
}

TEST_F(UndoHistoryTest, SceneSnapshotSparseSelectionRoundTripsDirectly) {
    auto scene_manager = std::make_unique<lfs::vis::SceneManager>();
    auto rendering_manager = std::make_unique<lfs::vis::RenderingManager>();
    lfs::vis::services().set(scene_manager.get());
    lfs::vis::services().set(rendering_manager.get());

    scene_manager->getScene().addNode("model", make_linear_test_splat(16));

    lfs::vis::op::SceneSnapshot snapshot(*scene_manager, "selection.direct");
    snapshot.captureSelection();
    scene_manager->getScene().setSelectionMask(
        std::make_shared<Tensor>(make_uint8_mask({0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0})));
    snapshot.captureAfter();

    snapshot.undo();
    EXPECT_TRUE(selection_mask_values(scene_manager->getScene()).empty());

    snapshot.redo();
    EXPECT_EQ(selection_mask_values(scene_manager->getScene()),
              (std::vector<uint8_t>{0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}));
}

TEST_F(UndoHistoryTest, OffloadedSceneSnapshotRestoresSelectionDuringPlayback) {
    auto scene_manager = std::make_unique<lfs::vis::SceneManager>();
    auto rendering_manager = std::make_unique<lfs::vis::RenderingManager>();
    lfs::vis::services().set(scene_manager.get());
    lfs::vis::services().set(rendering_manager.get());

    scene_manager->getScene().addNode("model", make_linear_test_splat(16));

    auto snapshot = std::make_unique<lfs::vis::op::SceneSnapshot>(*scene_manager, "selection.offloaded");
    snapshot->captureSelection();
    scene_manager->getScene().setSelectionMask(
        std::make_shared<Tensor>(make_uint8_mask({0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0})));
    snapshot->captureAfter();
    ASSERT_TRUE(lfs::vis::op::pushSceneSnapshotIfChanged(std::move(snapshot)));

    int value = 0;
    for (size_t i = 0; i < lfs::vis::op::UndoHistory::HOT_ENTRIES; ++i) {
        lfs::vis::op::undoHistory().push(
            std::make_unique<CountingEntry>("padding." + std::to_string(i), value, 1));
        value += 1;
    }

    const auto items = lfs::vis::op::undoHistory().undoItems();
    ASSERT_EQ(items.size(), lfs::vis::op::UndoHistory::HOT_ENTRIES + 1);
    EXPECT_EQ(items.back().metadata.label, "selection.offloaded");
    EXPECT_EQ(items.back().gpu_bytes, 0u);
    EXPECT_GT(items.back().cpu_bytes, 0u);

    const auto undo_result = lfs::vis::op::undoHistory().undoMultiple(lfs::vis::op::undoHistory().undoCount());
    EXPECT_TRUE(undo_result.success);
    EXPECT_TRUE(selection_mask_values(scene_manager->getScene()).empty());

    const auto redo_result = lfs::vis::op::undoHistory().redoMultiple(lfs::vis::op::undoHistory().redoCount());
    EXPECT_TRUE(redo_result.success);
    EXPECT_EQ(selection_mask_values(scene_manager->getScene()),
              (std::vector<uint8_t>{0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}));
}

TEST_F(UndoHistoryTest, SceneSnapshotEstimatedBytesIncludeTransformMaps) {
    auto scene_manager = std::make_unique<lfs::vis::SceneManager>();
    auto rendering_manager = std::make_unique<lfs::vis::RenderingManager>();
    lfs::vis::services().set(scene_manager.get());
    lfs::vis::services().set(rendering_manager.get());

    scene_manager->getScene().addNode("model", make_linear_test_splat(2));

    auto snapshot = std::make_unique<lfs::vis::op::SceneSnapshot>(*scene_manager, "transform.bytes");
    snapshot->captureTransforms({"model"});

    glm::mat4 transform(1.0f);
    transform[3].x = 3.0f;
    scene_manager->setNodeTransform("model", transform);
    snapshot->captureAfter();

    const size_t minimum_expected = 2 * (sizeof(glm::mat4) + std::string("model").size());
    EXPECT_GE(snapshot->estimatedBytes(), minimum_expected);
}

TEST_F(UndoHistoryTest, SceneSnapshotFallsBackToDenseSelectionStorageForWideChanges) {
    auto scene_manager = std::make_unique<lfs::vis::SceneManager>();
    auto rendering_manager = std::make_unique<lfs::vis::RenderingManager>();
    lfs::vis::services().set(scene_manager.get());
    lfs::vis::services().set(rendering_manager.get());

    scene_manager->getScene().addNode("model", make_linear_test_splat(16));

    auto snapshot = std::make_unique<lfs::vis::op::SceneSnapshot>(*scene_manager, "selection.dense");
    snapshot->captureSelection();
    scene_manager->getScene().setSelectionMask(
        std::make_shared<Tensor>(make_uint8_mask({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})));
    snapshot->captureAfter();

    EXPECT_EQ(snapshot->estimatedBytes(), 16u);

    lfs::vis::op::undoHistory().push(std::move(snapshot));
    lfs::vis::op::undoHistory().undo();
    EXPECT_TRUE(selection_mask_values(scene_manager->getScene()).empty());

    lfs::vis::op::undoHistory().redo();
    EXPECT_EQ(selection_mask_values(scene_manager->getScene()),
              (std::vector<uint8_t>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}));
}

TEST_F(UndoHistoryTest, SceneSnapshotCompactsSparseDeletedMasksAndRestoresPresence) {
    auto scene_manager = std::make_unique<lfs::vis::SceneManager>();
    auto rendering_manager = std::make_unique<lfs::vis::RenderingManager>();
    lfs::vis::services().set(scene_manager.get());
    lfs::vis::services().set(rendering_manager.get());

    scene_manager->getScene().addNode("model", make_linear_test_splat(16));
    auto* node = scene_manager->getScene().getMutableNode("model");
    ASSERT_NE(node, nullptr);
    ASSERT_NE(node->model, nullptr);

    auto snapshot = std::make_unique<lfs::vis::op::SceneSnapshot>(*scene_manager, "delete.sparse");
    snapshot->captureTopology();
    node->model->soft_delete(make_uint8_mask({0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}).to(DataType::Bool));
    snapshot->captureAfter();

    EXPECT_LT(snapshot->estimatedBytes(), 16u);

    lfs::vis::op::undoHistory().push(std::move(snapshot));
    EXPECT_TRUE(node->model->has_deleted_mask());

    lfs::vis::op::undoHistory().undo();
    EXPECT_FALSE(node->model->has_deleted_mask());

    lfs::vis::op::undoHistory().redo();
    EXPECT_TRUE(node->model->has_deleted_mask());
    EXPECT_EQ(deleted_mask_values(*node->model),
              (std::vector<bool>{false, false, false, false, false, true, false, false,
                                 false, false, false, false, false, false, false, false}));
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

TEST_F(UndoHistoryTest, RapidVisibilityChangesMergeIntoSingleUndoStep) {
    auto scene_manager = std::make_unique<lfs::vis::SceneManager>();
    auto rendering_manager = std::make_unique<lfs::vis::RenderingManager>();
    lfs::vis::services().set(scene_manager.get());
    lfs::vis::services().set(rendering_manager.get());

    scene_manager->getScene().addNode("model", make_test_splat({0.0f, 0.0f, 0.0f}));
    auto* node = scene_manager->getScene().getMutableNode("model");
    ASSERT_NE(node, nullptr);
    ASSERT_TRUE(static_cast<bool>(node->visible));

    scene_manager->setPLYVisibility("model", false);
    scene_manager->setPLYVisibility("model", true);
    scene_manager->setPLYVisibility("model", false);

    ASSERT_EQ(lfs::vis::op::undoHistory().undoCount(), 1u);
    EXPECT_EQ(lfs::vis::op::undoHistory().undoName(), "Set Visibility");
    EXPECT_FALSE(static_cast<bool>(node->visible));

    lfs::vis::op::undoHistory().undo();
    EXPECT_TRUE(static_cast<bool>(node->visible));

    lfs::vis::op::undoHistory().redo();
    EXPECT_FALSE(static_cast<bool>(node->visible));
}

TEST_F(UndoHistoryTest, RapidLockChangesMergeIntoSingleUndoStep) {
    auto scene_manager = std::make_unique<lfs::vis::SceneManager>();
    auto rendering_manager = std::make_unique<lfs::vis::RenderingManager>();
    lfs::vis::services().set(scene_manager.get());
    lfs::vis::services().set(rendering_manager.get());

    scene_manager->getScene().addNode("model", make_test_splat({0.0f, 0.0f, 0.0f}));
    auto* node = scene_manager->getScene().getMutableNode("model");
    ASSERT_NE(node, nullptr);
    ASSERT_FALSE(static_cast<bool>(node->locked));

    lfs::core::events::cmd::SetNodeLocked{.name = "model", .locked = true}.emit();
    lfs::core::events::cmd::SetNodeLocked{.name = "model", .locked = false}.emit();
    lfs::core::events::cmd::SetNodeLocked{.name = "model", .locked = true}.emit();

    ASSERT_EQ(lfs::vis::op::undoHistory().undoCount(), 1u);
    EXPECT_EQ(lfs::vis::op::undoHistory().undoName(), "Set Lock State");
    EXPECT_TRUE(static_cast<bool>(node->locked));

    lfs::vis::op::undoHistory().undo();
    EXPECT_FALSE(static_cast<bool>(node->locked));

    lfs::vis::op::undoHistory().redo();
    EXPECT_TRUE(static_cast<bool>(node->locked));
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

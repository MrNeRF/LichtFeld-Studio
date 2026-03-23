/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"

#include "undo_entry.hpp"
#include <atomic>
#include <chrono>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace lfs::vis::op {

    struct LFS_VIS_API UndoStackItem {
        UndoMetadata metadata;
        size_t estimated_bytes = 0;
        size_t cpu_bytes = 0;
        size_t gpu_bytes = 0;
    };

    struct LFS_VIS_API HistoryResult {
        bool success = false;
        bool changed = false;
        size_t steps_performed = 0;
        std::string error;
    };

    class LFS_VIS_API TransactionGuard {
    public:
        explicit TransactionGuard(std::string name);
        ~TransactionGuard();

        void commit();
        void rollback();
        void release();

        TransactionGuard(const TransactionGuard&) = delete;
        TransactionGuard& operator=(const TransactionGuard&) = delete;
        TransactionGuard(TransactionGuard&& other) noexcept;
        TransactionGuard& operator=(TransactionGuard&& other) noexcept;

    private:
        bool active_ = false;
    };

    class LFS_VIS_API UndoHistory {
    public:
        static constexpr size_t MAX_ENTRIES = 100;
        static constexpr size_t MAX_BYTES = 512ull * 1024ull * 1024ull;
        static constexpr size_t HOT_ENTRIES = 5;
        using ObserverId = uint64_t;
        using Observer = std::function<void()>;

        static UndoHistory& instance();

        void push(UndoEntryPtr entry);
        HistoryResult undo();
        HistoryResult redo();
        HistoryResult undoMultiple(size_t count);
        HistoryResult redoMultiple(size_t count);
        void clear();

        void beginTransaction(std::string name);
        void commitTransaction();
        HistoryResult rollbackTransaction();

        [[nodiscard]] bool canUndo() const;
        [[nodiscard]] bool canRedo() const;
        [[nodiscard]] std::string undoName() const;
        [[nodiscard]] std::string redoName() const;
        [[nodiscard]] std::vector<std::string> undoNames() const;
        [[nodiscard]] std::vector<std::string> redoNames() const;
        [[nodiscard]] size_t undoCount() const;
        [[nodiscard]] size_t redoCount() const;
        [[nodiscard]] size_t undoBytes() const;
        [[nodiscard]] size_t redoBytes() const;
        [[nodiscard]] size_t transactionBytes() const;
        [[nodiscard]] size_t totalBytes() const;
        [[nodiscard]] size_t maxBytes() const;
        [[nodiscard]] UndoMemoryBreakdown undoMemory() const;
        [[nodiscard]] UndoMemoryBreakdown redoMemory() const;
        [[nodiscard]] UndoMemoryBreakdown transactionMemory() const;
        [[nodiscard]] UndoMemoryBreakdown totalMemory() const;
        [[nodiscard]] bool hasActiveTransaction() const;
        [[nodiscard]] size_t transactionDepth() const;
        [[nodiscard]] uint64_t transactionAgeMs() const;
        [[nodiscard]] std::string activeTransactionName() const;
        [[nodiscard]] std::vector<UndoStackItem> undoItems() const;
        [[nodiscard]] std::vector<UndoStackItem> redoItems() const;
        [[nodiscard]] uint64_t generation() const;

        void setMaxBytes(size_t max_bytes);
        void shrinkToFit(size_t target_gpu_bytes);
        ObserverId subscribe(Observer observer);
        void unsubscribe(ObserverId id);

    private:
        UndoHistory() = default;
        ~UndoHistory() = default;
        UndoHistory(const UndoHistory&) = delete;
        UndoHistory& operator=(const UndoHistory&) = delete;

        struct TransactionFrame {
            std::string name;
            std::vector<UndoEntryPtr> entries;
            size_t estimated_bytes = 0;
            std::chrono::steady_clock::time_point started_at;
        };

        void clearStack(std::deque<UndoEntryPtr>& stack, size_t& bytes);
        void clearLocked();
        void updateAvailabilityLocked();
        void trimUndoStack();
        void trimRedoStack();
        void resetRedoStack();
        void notifyObservers();
        void bumpGenerationLocked();
        [[nodiscard]] size_t transactionBytesLocked() const;
        [[nodiscard]] size_t totalBytesLocked() const;
        [[nodiscard]] UndoMemoryBreakdown stackMemoryLocked(const std::deque<UndoEntryPtr>& stack) const;
        [[nodiscard]] UndoMemoryBreakdown transactionMemoryLocked() const;
        [[nodiscard]] UndoMemoryBreakdown totalMemoryLocked() const;
        void refreshResidencyLocked();
        HistoryResult performPlayback(bool undo_direction, size_t count);

        std::deque<UndoEntryPtr> undo_stack_;
        std::deque<UndoEntryPtr> redo_stack_;
        std::vector<TransactionFrame> transactions_;
        std::unordered_map<ObserverId, Observer> observers_;
        size_t undo_bytes_ = 0;
        size_t redo_bytes_ = 0;
        size_t max_bytes_ = MAX_BYTES;
        std::atomic<uint64_t> generation_{0};
        std::atomic<bool> can_undo_{false};
        std::atomic<bool> can_redo_{false};
        ObserverId next_observer_id_ = 1;
        std::thread::id playback_thread_id_{};
        size_t playback_depth_ = 0;
        mutable std::mutex mutex_;
        std::recursive_mutex playback_mutex_;
    };

    inline UndoHistory& undoHistory() {
        return UndoHistory::instance();
    }

} // namespace lfs::vis::op

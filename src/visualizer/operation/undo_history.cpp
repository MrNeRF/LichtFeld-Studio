/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "undo_history.hpp"
#include "core/logger.hpp"
#include "core/services.hpp"
#include "operator/operator_registry.hpp"
#include "rendering/dirty_flags.hpp"
#include "rendering/rendering_manager.hpp"
#include <cctype>
#include <string_view>
#include <utility>

namespace lfs::vis::op {

    namespace {
        void restoreUndoneTail(const std::vector<UndoEntryPtr>& entries, const size_t undone_count) {
            if (undone_count == 0) {
                return;
            }
            const size_t start = entries.size() - undone_count;
            for (size_t idx = start; idx < entries.size(); ++idx) {
                entries[idx]->redo();
            }
        }

        void restoreRedoneHead(const std::vector<UndoEntryPtr>& entries, const size_t redone_count) {
            for (size_t idx = redone_count; idx > 0; --idx) {
                entries[idx - 1]->undo();
            }
        }

        [[nodiscard]] std::string sanitizeHistoryId(std::string_view value) {
            std::string sanitized;
            sanitized.reserve(value.size());

            bool last_was_underscore = false;
            for (const unsigned char ch : value) {
                if (std::isalnum(ch)) {
                    sanitized.push_back(static_cast<char>(std::tolower(ch)));
                    last_was_underscore = false;
                    continue;
                }

                if (!last_was_underscore) {
                    sanitized.push_back('_');
                    last_was_underscore = true;
                }
            }

            while (!sanitized.empty() && sanitized.front() == '_') {
                sanitized.erase(sanitized.begin());
            }
            while (!sanitized.empty() && sanitized.back() == '_') {
                sanitized.pop_back();
            }

            if (sanitized.empty()) {
                return "grouped_changes";
            }
            return sanitized;
        }

        class CompoundUndoEntry final : public UndoEntry {
        public:
            CompoundUndoEntry(std::string name, std::vector<UndoEntryPtr> entries, size_t estimated_bytes)
                : name_(std::move(name)),
                  entries_(std::move(entries)),
                  estimated_bytes_(estimated_bytes),
                  metadata_id_("history.transaction." + sanitizeHistoryId(name_)) {}

            void undo() override {
                size_t undone_count = 0;
                try {
                    for (auto it = entries_.rbegin(); it != entries_.rend(); ++it) {
                        (*it)->undo();
                        ++undone_count;
                    }
                } catch (...) {
                    try {
                        restoreUndoneTail(entries_, undone_count);
                    } catch (const std::exception& rollback_error) {
                        LOG_ERROR("Compound undo rollback failed for '{}': {}", name_, rollback_error.what());
                    } catch (...) {
                        LOG_ERROR("Compound undo rollback failed for '{}': unknown exception", name_);
                    }
                    throw;
                }
            }

            void redo() override {
                size_t redone_count = 0;
                try {
                    for (auto& entry : entries_) {
                        entry->redo();
                        ++redone_count;
                    }
                } catch (...) {
                    try {
                        restoreRedoneHead(entries_, redone_count);
                    } catch (const std::exception& rollback_error) {
                        LOG_ERROR("Compound redo rollback failed for '{}': {}", name_, rollback_error.what());
                    } catch (...) {
                        LOG_ERROR("Compound redo rollback failed for '{}': unknown exception", name_);
                    }
                    throw;
                }
            }

            [[nodiscard]] std::string name() const override { return name_; }
            [[nodiscard]] UndoMetadata metadata() const override {
                return UndoMetadata{
                    .id = metadata_id_,
                    .label = name_,
                    .source = "history",
                    .scope = "grouped",
                };
            }
            [[nodiscard]] size_t estimatedBytes() const override { return estimated_bytes_; }

        private:
            std::string name_;
            std::vector<UndoEntryPtr> entries_;
            size_t estimated_bytes_ = 0;
            std::string metadata_id_;
        };

        void invalidateUndoRedoPollState() {
            operators().invalidatePollCache();
        }

        void refreshAfterHistoryPlayback() {
            invalidateUndoRedoPollState();
            if (auto* rm = services().renderingOrNull()) {
                rm->markDirty(DirtyFlag::ALL);
            }
        }

        [[nodiscard]] size_t entryBytes(const UndoEntryPtr& entry) {
            return entry ? entry->estimatedBytes() : 0;
        }

        template <typename Container>
        bool tryMergeBackEntry(Container& container, size_t& total_bytes, UndoEntryPtr& incoming) {
            if (container.empty() || !incoming || !container.back()) {
                return false;
            }

            const size_t before = entryBytes(container.back());
            if (!container.back()->tryMerge(*incoming)) {
                return false;
            }

            const size_t after = entryBytes(container.back());
            if (after >= before) {
                total_bytes += after - before;
            } else {
                total_bytes -= before - after;
            }
            return true;
        }

        [[nodiscard]] UndoStackItem stackItem(const UndoEntryPtr& entry) {
            UndoStackItem item;
            if (!entry) {
                return item;
            }
            item.metadata = entry->metadata();
            item.estimated_bytes = entry->estimatedBytes();
            return item;
        }

        [[nodiscard]] HistoryResult makeEmptyPlaybackResult(const char* verb) {
            return HistoryResult{
                .success = false,
                .changed = false,
                .steps_performed = 0,
                .error = std::string("Nothing to ") + verb,
            };
        }

    } // namespace

    UndoHistory& UndoHistory::instance() {
        static UndoHistory instance;
        return instance;
    }

    void UndoHistory::clearStack(std::deque<UndoEntryPtr>& stack, size_t& bytes) {
        stack.clear();
        bytes = 0;
    }

    void UndoHistory::bumpGenerationLocked() {
        ++generation_;
    }

    void UndoHistory::notifyObservers() {
        std::vector<Observer> observers;
        {
            std::lock_guard lock(mutex_);
            observers.reserve(observers_.size());
            for (const auto& [_, observer] : observers_) {
                if (observer) {
                    observers.push_back(observer);
                }
            }
        }

        for (const auto& observer : observers) {
            try {
                observer();
            } catch (const std::exception& e) {
                LOG_ERROR("UndoHistory observer failed: {}", e.what());
            } catch (...) {
                LOG_ERROR("UndoHistory observer failed: unknown exception");
            }
        }
    }

    void UndoHistory::trimUndoStack() {
        while (undo_stack_.size() > MAX_ENTRIES || undo_bytes_ > MAX_BYTES) {
            if (undo_stack_.empty()) {
                undo_bytes_ = 0;
                break;
            }
            undo_bytes_ -= entryBytes(undo_stack_.front());
            undo_stack_.pop_front();
        }
    }

    void UndoHistory::trimRedoStack() {
        while (redo_stack_.size() > MAX_ENTRIES || redo_bytes_ > MAX_BYTES) {
            if (redo_stack_.empty()) {
                redo_bytes_ = 0;
                break;
            }
            redo_bytes_ -= entryBytes(redo_stack_.front());
            redo_stack_.pop_front();
        }
    }

    void UndoHistory::resetRedoStack() {
        clearStack(redo_stack_, redo_bytes_);
    }

    void UndoHistory::push(UndoEntryPtr entry) {
        if (!entry) {
            return;
        }

        const size_t entry_bytes = entryBytes(entry);
        bool changed = false;
        bool merged = false;
        {
            std::lock_guard lock(mutex_);

            if (!transactions_.empty()) {
                auto& frame = transactions_.back();
                if (tryMergeBackEntry(frame.entries, frame.estimated_bytes, entry)) {
                    return;
                }
                frame.estimated_bytes += entry_bytes;
                frame.entries.push_back(std::move(entry));
                return;
            }

            resetRedoStack();
            if (tryMergeBackEntry(undo_stack_, undo_bytes_, entry)) {
                trimUndoStack();
                bumpGenerationLocked();
                changed = true;
                merged = true;
            } else {
                undo_stack_.push_back(std::move(entry));
                undo_bytes_ += entry_bytes;
                trimUndoStack();
                bumpGenerationLocked();
                changed = true;
            }
            if (!undo_stack_.empty()) {
                LOG_DEBUG("{} undo entry: {} (stack size: {})",
                          merged ? "Merged" : "Pushed",
                          undo_stack_.back()->name(),
                          undo_stack_.size());
            } else {
                LOG_DEBUG("Dropped undo entry after trimming oversized history payload");
            }
        }
        if (changed) {
            invalidateUndoRedoPollState();
            notifyObservers();
        }
    }

    HistoryResult UndoHistory::performPlayback(const bool undo_direction, const size_t count) {
        if (count == 0) {
            return HistoryResult{
                .success = true,
                .changed = false,
                .steps_performed = 0,
                .error = {},
            };
        }

        std::unique_lock playback_lock(playback_mutex_, std::try_to_lock);
        if (!playback_lock.owns_lock()) {
            return HistoryResult{
                .success = false,
                .changed = false,
                .steps_performed = 0,
                .error = "History playback already in progress",
            };
        }

        HistoryResult result{
            .success = true,
            .changed = false,
            .steps_performed = 0,
            .error = {},
        };

        for (size_t step = 0; step < count; ++step) {
            UndoEntryPtr entry;
            size_t bytes = 0;
            {
                std::lock_guard lock(mutex_);
                auto& source_stack = undo_direction ? undo_stack_ : redo_stack_;
                auto& source_bytes = undo_direction ? undo_bytes_ : redo_bytes_;
                if (source_stack.empty()) {
                    if (step == 0) {
                        return makeEmptyPlaybackResult(undo_direction ? "undo" : "redo");
                    }
                    break;
                }

                entry = std::move(source_stack.back());
                source_stack.pop_back();
                bytes = entryBytes(entry);
                source_bytes -= bytes;
            }

            LOG_DEBUG("{}ing: {}", undo_direction ? "Undo" : "Redo", entry->name());

            try {
                if (undo_direction) {
                    entry->undo();
                } else {
                    entry->redo();
                }
            } catch (const std::exception& e) {
                LOG_ERROR("{} failed for '{}': {}",
                          undo_direction ? "Undo" : "Redo", entry->name(), e.what());
                {
                    std::lock_guard lock(mutex_);
                    auto& source_stack = undo_direction ? undo_stack_ : redo_stack_;
                    auto& source_bytes = undo_direction ? undo_bytes_ : redo_bytes_;
                    source_stack.push_back(std::move(entry));
                    source_bytes += bytes;
                }
                result.success = false;
                result.error = e.what();
                if (result.changed) {
                    refreshAfterHistoryPlayback();
                    notifyObservers();
                }
                return result;
            } catch (...) {
                LOG_ERROR("{} failed for '{}': unknown exception",
                          undo_direction ? "Undo" : "Redo", entry->name());
                {
                    std::lock_guard lock(mutex_);
                    auto& source_stack = undo_direction ? undo_stack_ : redo_stack_;
                    auto& source_bytes = undo_direction ? undo_bytes_ : redo_bytes_;
                    source_stack.push_back(std::move(entry));
                    source_bytes += bytes;
                }
                result.success = false;
                result.error = "unknown exception";
                if (result.changed) {
                    refreshAfterHistoryPlayback();
                    notifyObservers();
                }
                return result;
            }

            {
                std::lock_guard lock(mutex_);
                auto& target_stack = undo_direction ? redo_stack_ : undo_stack_;
                auto& target_bytes = undo_direction ? redo_bytes_ : undo_bytes_;
                target_stack.push_back(std::move(entry));
                target_bytes += bytes;
                if (undo_direction) {
                    trimRedoStack();
                } else {
                    trimUndoStack();
                }
                bumpGenerationLocked();
            }

            result.changed = true;
            ++result.steps_performed;
        }

        if (result.changed) {
            refreshAfterHistoryPlayback();
            notifyObservers();
        }
        return result;
    }

    HistoryResult UndoHistory::undo() {
        return performPlayback(true, 1);
    }

    HistoryResult UndoHistory::redo() {
        return performPlayback(false, 1);
    }

    HistoryResult UndoHistory::undoMultiple(const size_t count) {
        return performPlayback(true, count);
    }

    HistoryResult UndoHistory::redoMultiple(const size_t count) {
        return performPlayback(false, count);
    }

    void UndoHistory::clear() {
        bool changed = false;
        {
            std::lock_guard lock(mutex_);
            changed = !undo_stack_.empty() || !redo_stack_.empty() || !transactions_.empty();
            clearStack(undo_stack_, undo_bytes_);
            clearStack(redo_stack_, redo_bytes_);
            transactions_.clear();
            if (changed) {
                bumpGenerationLocked();
            }
        }
        invalidateUndoRedoPollState();
        if (changed) {
            notifyObservers();
        }
    }

    void UndoHistory::beginTransaction(std::string name) {
        {
            std::lock_guard lock(mutex_);
            transactions_.push_back(TransactionFrame{
                .name = std::move(name),
                .entries = {},
                .estimated_bytes = 0,
            });
            bumpGenerationLocked();
        }
        notifyObservers();
    }

    void UndoHistory::commitTransaction() {
        UndoEntryPtr committed_entry;
        bool changed = false;

        {
            std::lock_guard lock(mutex_);
            if (transactions_.empty()) {
                return;
            }

            TransactionFrame frame = std::move(transactions_.back());
            transactions_.pop_back();

            if (frame.entries.empty()) {
                bumpGenerationLocked();
                changed = true;
            } else {
                committed_entry = std::make_unique<CompoundUndoEntry>(std::move(frame.name),
                                                                      std::move(frame.entries),
                                                                      frame.estimated_bytes);
                if (!committed_entry) {
                    return;
                }

                const size_t committed_bytes = entryBytes(committed_entry);

                if (!transactions_.empty()) {
                    auto& parent = transactions_.back();
                    parent.estimated_bytes += committed_bytes;
                    parent.entries.push_back(std::move(committed_entry));
                    bumpGenerationLocked();
                    changed = true;
                } else {
                    resetRedoStack();
                    undo_bytes_ += committed_bytes;
                    undo_stack_.push_back(std::move(committed_entry));
                    trimUndoStack();
                    bumpGenerationLocked();
                    changed = true;
                    if (!undo_stack_.empty()) {
                        LOG_DEBUG("Committed history transaction '{}' (stack size: {})",
                                  undo_stack_.back()->name(), undo_stack_.size());
                    } else {
                        LOG_DEBUG("Dropped committed history transaction after trimming oversized payload");
                    }
                }
            }
        }

        if (changed) {
            invalidateUndoRedoPollState();
            notifyObservers();
        }
    }

    HistoryResult UndoHistory::rollbackTransaction() {
        std::vector<UndoEntryPtr> entries;

        {
            std::lock_guard lock(mutex_);
            if (transactions_.empty()) {
                return HistoryResult{
                    .success = false,
                    .changed = false,
                    .steps_performed = 0,
                    .error = "No active history transaction",
                };
            }

            entries = std::move(transactions_.back().entries);
            transactions_.pop_back();
            bumpGenerationLocked();
        }

        HistoryResult result{
            .success = true,
            .changed = false,
            .steps_performed = 0,
            .error = {},
        };

        if (entries.empty()) {
            notifyObservers();
            return result;
        }

        std::unique_lock playback_lock(playback_mutex_, std::try_to_lock);
        if (!playback_lock.owns_lock()) {
            notifyObservers();
            return HistoryResult{
                .success = false,
                .changed = false,
                .steps_performed = 0,
                .error = "History playback already in progress",
            };
        }

        size_t undone_count = 0;
        for (auto it = entries.rbegin(); it != entries.rend(); ++it) {
            try {
                (*it)->undo();
                ++undone_count;
            } catch (const std::exception& e) {
                LOG_ERROR("Rollback failed for '{}': {}", (*it)->name(), e.what());
                try {
                    restoreUndoneTail(entries, undone_count);
                } catch (const std::exception& rollback_error) {
                    LOG_ERROR("Rollback compensation failed: {}", rollback_error.what());
                } catch (...) {
                    LOG_ERROR("Rollback compensation failed: unknown exception");
                }
                result.success = false;
                result.error = e.what();
                notifyObservers();
                return result;
            } catch (...) {
                LOG_ERROR("Rollback failed for '{}': unknown exception", (*it)->name());
                try {
                    restoreUndoneTail(entries, undone_count);
                } catch (const std::exception& rollback_error) {
                    LOG_ERROR("Rollback compensation failed: {}", rollback_error.what());
                } catch (...) {
                    LOG_ERROR("Rollback compensation failed: unknown exception");
                }
                result.success = false;
                result.error = "unknown exception";
                notifyObservers();
                return result;
            }
        }

        result.changed = undone_count > 0;
        result.steps_performed = undone_count;
        refreshAfterHistoryPlayback();
        notifyObservers();
        return result;
    }

    bool UndoHistory::canUndo() const {
        std::lock_guard lock(mutex_);
        return !undo_stack_.empty();
    }

    bool UndoHistory::canRedo() const {
        std::lock_guard lock(mutex_);
        return !redo_stack_.empty();
    }

    std::string UndoHistory::undoName() const {
        std::lock_guard lock(mutex_);
        if (undo_stack_.empty()) {
            return "";
        }
        return undo_stack_.back()->metadata().label;
    }

    std::string UndoHistory::redoName() const {
        std::lock_guard lock(mutex_);
        if (redo_stack_.empty()) {
            return "";
        }
        return redo_stack_.back()->metadata().label;
    }

    std::vector<std::string> UndoHistory::undoNames() const {
        std::vector<std::string> result;
        std::lock_guard lock(mutex_);
        result.reserve(undo_stack_.size());
        for (auto it = undo_stack_.rbegin(); it != undo_stack_.rend(); ++it) {
            result.push_back((*it)->metadata().label);
        }
        return result;
    }

    std::vector<std::string> UndoHistory::redoNames() const {
        std::vector<std::string> result;
        std::lock_guard lock(mutex_);
        result.reserve(redo_stack_.size());
        for (auto it = redo_stack_.rbegin(); it != redo_stack_.rend(); ++it) {
            result.push_back((*it)->metadata().label);
        }
        return result;
    }

    size_t UndoHistory::undoCount() const {
        std::lock_guard lock(mutex_);
        return undo_stack_.size();
    }

    size_t UndoHistory::redoCount() const {
        std::lock_guard lock(mutex_);
        return redo_stack_.size();
    }

    size_t UndoHistory::undoBytes() const {
        std::lock_guard lock(mutex_);
        return undo_bytes_;
    }

    size_t UndoHistory::redoBytes() const {
        std::lock_guard lock(mutex_);
        return redo_bytes_;
    }

    size_t UndoHistory::transactionBytes() const {
        std::lock_guard lock(mutex_);
        size_t total = 0;
        for (const auto& frame : transactions_) {
            total += frame.estimated_bytes;
        }
        return total;
    }

    size_t UndoHistory::totalBytes() const {
        std::lock_guard lock(mutex_);
        return undo_bytes_ + redo_bytes_;
    }

    bool UndoHistory::hasActiveTransaction() const {
        std::lock_guard lock(mutex_);
        return !transactions_.empty();
    }

    size_t UndoHistory::transactionDepth() const {
        std::lock_guard lock(mutex_);
        return transactions_.size();
    }

    std::string UndoHistory::activeTransactionName() const {
        std::lock_guard lock(mutex_);
        return transactions_.empty() ? std::string{} : transactions_.back().name;
    }

    std::vector<UndoStackItem> UndoHistory::undoItems() const {
        std::vector<UndoStackItem> result;
        std::lock_guard lock(mutex_);
        result.reserve(undo_stack_.size());
        for (auto it = undo_stack_.rbegin(); it != undo_stack_.rend(); ++it) {
            result.push_back(stackItem(*it));
        }
        return result;
    }

    std::vector<UndoStackItem> UndoHistory::redoItems() const {
        std::vector<UndoStackItem> result;
        std::lock_guard lock(mutex_);
        result.reserve(redo_stack_.size());
        for (auto it = redo_stack_.rbegin(); it != redo_stack_.rend(); ++it) {
            result.push_back(stackItem(*it));
        }
        return result;
    }

    uint64_t UndoHistory::generation() const {
        std::lock_guard lock(mutex_);
        return generation_;
    }

    UndoHistory::ObserverId UndoHistory::subscribe(Observer observer) {
        if (!observer) {
            return 0;
        }
        std::lock_guard lock(mutex_);
        const ObserverId id = next_observer_id_++;
        observers_.emplace(id, std::move(observer));
        return id;
    }

    void UndoHistory::unsubscribe(const ObserverId id) {
        if (id == 0) {
            return;
        }
        std::lock_guard lock(mutex_);
        observers_.erase(id);
    }

} // namespace lfs::vis::op

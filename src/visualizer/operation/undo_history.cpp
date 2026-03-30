/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "undo_history.hpp"
#include "core/logger.hpp"
#include "core/services.hpp"
#include "operator/operator_registry.hpp"
#include "rendering/dirty_flags.hpp"
#include "rendering/rendering_manager.hpp"
#include <algorithm>
#include <cctype>
#include <string_view>
#include <utility>

namespace lfs::vis::op {

    namespace {
        template <typename Fn>
        class ScopeExit final {
        public:
            explicit ScopeExit(Fn fn) : fn_(std::move(fn)) {}
            ScopeExit(const ScopeExit&) = delete;
            ScopeExit& operator=(const ScopeExit&) = delete;
            ~ScopeExit() {
                if (active_) {
                    fn_();
                }
            }

        private:
            Fn fn_;
            bool active_ = true;
        };

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
                        throw HistoryCorruptionError("Compound undo rollback failed for '" + name_ +
                                                     "': " + rollback_error.what());
                    } catch (...) {
                        LOG_ERROR("Compound undo rollback failed for '{}': unknown exception", name_);
                        throw HistoryCorruptionError("Compound undo rollback failed for '" + name_ +
                                                     "': unknown exception");
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
                        throw HistoryCorruptionError("Compound redo rollback failed for '" + name_ +
                                                     "': " + rollback_error.what());
                    } catch (...) {
                        LOG_ERROR("Compound redo rollback failed for '{}': unknown exception", name_);
                        throw HistoryCorruptionError("Compound redo rollback failed for '" + name_ +
                                                     "': unknown exception");
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
            [[nodiscard]] UndoMemoryBreakdown memoryBreakdown() const override {
                UndoMemoryBreakdown total;
                for (const auto& entry : entries_) {
                    if (entry) {
                        total += entry->memoryBreakdown();
                    }
                }
                return total;
            }
            void offloadToCPU() override {
                for (auto& entry : entries_) {
                    if (entry) {
                        entry->offloadToCPU();
                    }
                }
            }
            void restoreToPreferredDevice() override {
                for (auto& entry : entries_) {
                    if (entry) {
                        entry->restoreToPreferredDevice();
                    }
                }
            }
            [[nodiscard]] DirtyMask dirtyFlags() const override {
                DirtyMask flags = 0;
                for (const auto& entry : entries_) {
                    if (entry) {
                        flags |= entry->dirtyFlags();
                    }
                }
                return flags == 0 ? DirtyFlag::ALL : flags;
            }

        private:
            std::string name_;
            std::vector<UndoEntryPtr> entries_;
            size_t estimated_bytes_ = 0;
            std::string metadata_id_;
        };

        void invalidateUndoRedoPollState() {
            operators().invalidatePollCache();
        }

        void refreshAfterHistoryPlayback(const DirtyMask flags = DirtyFlag::ALL) {
            invalidateUndoRedoPollState();
            if (auto* rm = services().renderingOrNull()) {
                rm->markDirty(flags == 0 ? DirtyFlag::ALL : flags);
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
            const auto memory = entry->memoryBreakdown();
            item.cpu_bytes = memory.cpu_bytes;
            item.gpu_bytes = memory.gpu_bytes;
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

    TransactionGuard::TransactionGuard(std::string name) {
        undoHistory().beginTransaction(std::move(name));
        active_ = true;
    }

    TransactionGuard::~TransactionGuard() {
        if (!active_) {
            return;
        }
        try {
            undoHistory().rollbackTransaction();
        } catch (...) {
            LOG_ERROR("TransactionGuard rollback failed during destruction");
        }
    }

    void TransactionGuard::commit() {
        if (!active_) {
            return;
        }
        undoHistory().commitTransaction();
        active_ = false;
    }

    void TransactionGuard::rollback() {
        if (!active_) {
            return;
        }
        undoHistory().rollbackTransaction();
        active_ = false;
    }

    void TransactionGuard::release() {
        active_ = false;
    }

    TransactionGuard::TransactionGuard(TransactionGuard&& other) noexcept
        : active_(other.active_) {
        other.active_ = false;
    }

    TransactionGuard& TransactionGuard::operator=(TransactionGuard&& other) noexcept {
        if (this == &other) {
            return *this;
        }
        if (active_) {
            try {
                undoHistory().rollbackTransaction();
            } catch (...) {
                LOG_ERROR("TransactionGuard rollback failed during move assignment");
            }
        }
        active_ = other.active_;
        other.active_ = false;
        return *this;
    }

    UndoHistory& UndoHistory::instance() {
        static UndoHistory instance;
        return instance;
    }

    void UndoHistory::clearStack(std::deque<UndoEntryPtr>& stack, size_t& bytes) {
        stack.clear();
        bytes = 0;
    }

    void UndoHistory::clearLocked() {
        clearStack(undo_stack_, undo_bytes_);
        clearStack(redo_stack_, redo_bytes_);
        transactions_.clear();
        updateAvailabilityLocked();
    }

    void UndoHistory::bumpGenerationLocked() {
        generation_.fetch_add(1, std::memory_order_release);
    }

    void UndoHistory::updateAvailabilityLocked() {
        can_undo_.store(!undo_stack_.empty(), std::memory_order_release);
        can_redo_.store(!redo_stack_.empty(), std::memory_order_release);
    }

    size_t UndoHistory::transactionBytesLocked() const {
        size_t total = 0;
        for (const auto& frame : transactions_) {
            total += frame.estimated_bytes;
        }
        return total;
    }

    size_t UndoHistory::totalBytesLocked() const {
        return undo_bytes_ + redo_bytes_ + transactionBytesLocked();
    }

    UndoMemoryBreakdown UndoHistory::stackMemoryLocked(const std::deque<UndoEntryPtr>& stack) const {
        UndoMemoryBreakdown total;
        for (const auto& entry : stack) {
            if (entry) {
                total += entry->memoryBreakdown();
            }
        }
        return total;
    }

    UndoMemoryBreakdown UndoHistory::transactionMemoryLocked() const {
        UndoMemoryBreakdown total;
        for (const auto& frame : transactions_) {
            for (const auto& entry : frame.entries) {
                if (entry) {
                    total += entry->memoryBreakdown();
                }
            }
        }
        return total;
    }

    UndoMemoryBreakdown UndoHistory::totalMemoryLocked() const {
        auto total = stackMemoryLocked(undo_stack_);
        total += stackMemoryLocked(redo_stack_);
        total += transactionMemoryLocked();
        return total;
    }

    void UndoHistory::refreshResidencyLocked() {
        const auto refresh_stack = [](std::deque<UndoEntryPtr>& stack) {
            const size_t hot_start = stack.size() > HOT_ENTRIES ? stack.size() - HOT_ENTRIES : 0;
            for (size_t index = 0; index < stack.size(); ++index) {
                auto& entry = stack[index];
                if (!entry) {
                    continue;
                }
                if (index < hot_start) {
                    entry->offloadToCPU();
                } else {
                    entry->restoreToPreferredDevice();
                }
            }
        };

        refresh_stack(undo_stack_);
        refresh_stack(redo_stack_);
    }

    void UndoHistory::notifyObservers() {
        std::vector<std::pair<ObserverId, Observer>> observers;
        {
            std::lock_guard lock(mutex_);
            observers.reserve(observers_.size());
            for (const auto& [id, observer] : observers_) {
                if (observer) {
                    observers.emplace_back(id, observer);
                }
            }
        }

        std::vector<ObserverId> failed_ids;
        for (const auto& [id, observer] : observers) {
            try {
                observer();
            } catch (const std::exception& e) {
                LOG_ERROR("UndoHistory observer failed: {}", e.what());
                failed_ids.push_back(id);
            } catch (...) {
                LOG_ERROR("UndoHistory observer failed: unknown exception");
                failed_ids.push_back(id);
            }
        }

        if (!failed_ids.empty()) {
            std::lock_guard lock(mutex_);
            for (const auto id : failed_ids) {
                if (observers_.erase(id) > 0) {
                    LOG_WARN("UndoHistory observer {} unsubscribed after throwing", id);
                }
            }
        }
    }

    void UndoHistory::trimUndoStack() {
        while (undo_stack_.size() > 1 && (undo_stack_.size() > MAX_ENTRIES || undo_bytes_ > max_bytes_)) {
            undo_bytes_ -= entryBytes(undo_stack_.front());
            undo_stack_.pop_front();
        }
        updateAvailabilityLocked();
    }

    void UndoHistory::trimRedoStack() {
        while (redo_stack_.size() > 1 && (redo_stack_.size() > MAX_ENTRIES || redo_bytes_ > max_bytes_)) {
            redo_bytes_ -= entryBytes(redo_stack_.front());
            redo_stack_.pop_front();
        }
        updateAvailabilityLocked();
    }

    void UndoHistory::resetRedoStack() {
        clearStack(redo_stack_, redo_bytes_);
        updateAvailabilityLocked();
    }

    void UndoHistory::push(UndoEntryPtr entry) {
        if (!entry) {
            return;
        }

        const size_t entry_bytes = entryBytes(entry);
        bool changed = false;
        bool merged = false;
        bool invalidate_poll = false;
        std::unique_lock playback_lock(playback_mutex_);
        {
            std::lock_guard lock(mutex_);
            if (playback_depth_ > 0 && playback_thread_id_ == std::this_thread::get_id()) {
                LOG_WARN("Ignoring history push '{}' during playback", entry->name());
                return;
            }

            if (!transactions_.empty()) {
                auto& frame = transactions_.back();
                if (tryMergeBackEntry(frame.entries, frame.estimated_bytes, entry)) {
                    bumpGenerationLocked();
                    changed = true;
                    merged = true;
                } else {
                    frame.estimated_bytes += entry_bytes;
                    frame.entries.push_back(std::move(entry));
                    bumpGenerationLocked();
                    changed = true;
                }
                LOG_DEBUG("{} transaction entry in '{}': {} (transaction size: {})",
                          merged ? "Merged" : "Queued",
                          frame.name,
                          frame.entries.back()->name(),
                          frame.entries.size());
            } else {
                resetRedoStack();
                invalidate_poll = true;
                if (tryMergeBackEntry(undo_stack_, undo_bytes_, entry)) {
                    trimUndoStack();
                    refreshResidencyLocked();
                    bumpGenerationLocked();
                    changed = true;
                    merged = true;
                } else {
                    undo_stack_.push_back(std::move(entry));
                    undo_bytes_ += entry_bytes;
                    trimUndoStack();
                    refreshResidencyLocked();
                    bumpGenerationLocked();
                    changed = true;
                }
                if (undo_stack_.empty()) {
                    LOG_DEBUG("Dropped undo entry after trimming oversized history payload");
                } else {
                    LOG_DEBUG("{} undo entry: {} (stack size: {})",
                              merged ? "Merged" : "Pushed",
                              undo_stack_.back()->name(),
                              undo_stack_.size());
                }
            }
        }
        if (changed) {
            if (invalidate_poll) {
                invalidateUndoRedoPollState();
            }
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

        {
            std::lock_guard lock(mutex_);
            if (playback_depth_ > 0 && playback_thread_id_ == std::this_thread::get_id()) {
                return HistoryResult{
                    .success = false,
                    .changed = false,
                    .steps_performed = 0,
                    .error = "History playback already in progress",
                };
            }
            ++playback_depth_;
            playback_thread_id_ = std::this_thread::get_id();
        }
        ScopeExit playback_guard([this]() {
            std::lock_guard lock(mutex_);
            if (playback_depth_ == 0) {
                return;
            }
            --playback_depth_;
            if (playback_depth_ == 0) {
                playback_thread_id_ = {};
            }
        });

        HistoryResult result{
            .success = true,
            .changed = false,
            .steps_performed = 0,
            .error = {},
        };
        DirtyMask dirty_flags = 0;

        for (size_t step = 0; step < count; ++step) {
            UndoEntryPtr entry;
            size_t bytes = 0;
            DirtyMask entry_dirty_flags = DirtyFlag::ALL;
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
                entry_dirty_flags = entry ? entry->dirtyFlags() : DirtyFlag::ALL;
                source_bytes -= bytes;
                updateAvailabilityLocked();
            }

            LOG_DEBUG("{}ing: {}", undo_direction ? "Undo" : "Redo", entry->name());

            try {
                entry->restoreToPreferredDevice();
                if (undo_direction) {
                    entry->undo();
                } else {
                    entry->redo();
                }
            } catch (const HistoryCorruptionError& e) {
                LOG_ERROR("Fatal {} failure for '{}': {}",
                          undo_direction ? "undo" : "redo", entry->name(), e.what());
                {
                    std::lock_guard lock(mutex_);
                    clearLocked();
                    bumpGenerationLocked();
                }
                result.success = false;
                result.error = e.what();
                refreshAfterHistoryPlayback();
                notifyObservers();
                return result;
            } catch (const std::exception& e) {
                LOG_ERROR("{} failed for '{}': {}",
                          undo_direction ? "Undo" : "Redo", entry->name(), e.what());
                {
                    std::lock_guard lock(mutex_);
                    auto& source_stack = undo_direction ? undo_stack_ : redo_stack_;
                    auto& source_bytes = undo_direction ? undo_bytes_ : redo_bytes_;
                    source_stack.push_back(std::move(entry));
                    source_bytes += bytes;
                    refreshResidencyLocked();
                    updateAvailabilityLocked();
                }
                result.success = false;
                result.error = e.what();
                if (result.changed) {
                    refreshAfterHistoryPlayback(dirty_flags);
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
                    refreshResidencyLocked();
                    updateAvailabilityLocked();
                }
                result.success = false;
                result.error = "unknown exception";
                if (result.changed) {
                    refreshAfterHistoryPlayback(dirty_flags);
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
                refreshResidencyLocked();
                updateAvailabilityLocked();
                bumpGenerationLocked();
            }

            dirty_flags |= entry_dirty_flags;
            result.changed = true;
            ++result.steps_performed;
        }

        if (result.changed) {
            refreshAfterHistoryPlayback(dirty_flags);
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
        std::unique_lock playback_lock(playback_mutex_);
        {
            std::lock_guard lock(mutex_);
            if (playback_depth_ > 0 && playback_thread_id_ == std::this_thread::get_id()) {
                LOG_WARN("Ignoring history clear during playback");
                return;
            }
            changed = !undo_stack_.empty() || !redo_stack_.empty() || !transactions_.empty();
            clearLocked();
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
        std::unique_lock playback_lock(playback_mutex_);
        {
            std::lock_guard lock(mutex_);
            if (playback_depth_ > 0 && playback_thread_id_ == std::this_thread::get_id()) {
                LOG_WARN("Ignoring history transaction begin during playback");
                return;
            }
            transactions_.push_back(TransactionFrame{
                .name = std::move(name),
                .entries = {},
                .estimated_bytes = 0,
                .started_at = std::chrono::steady_clock::now(),
            });
            bumpGenerationLocked();
        }
        notifyObservers();
    }

    void UndoHistory::commitTransaction() {
        UndoEntryPtr committed_entry;
        bool changed = false;
        bool invalidate_poll = false;

        std::unique_lock playback_lock(playback_mutex_);
        {
            std::lock_guard lock(mutex_);
            if (playback_depth_ > 0 && playback_thread_id_ == std::this_thread::get_id()) {
                LOG_WARN("Ignoring history transaction commit during playback");
                return;
            }
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
                    invalidate_poll = true;
                    undo_bytes_ += committed_bytes;
                    undo_stack_.push_back(std::move(committed_entry));
                    trimUndoStack();
                    refreshResidencyLocked();
                    updateAvailabilityLocked();
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
            if (invalidate_poll) {
                invalidateUndoRedoPollState();
            }
            notifyObservers();
        }
    }

    HistoryResult UndoHistory::rollbackTransaction() {
        std::unique_lock playback_lock(playback_mutex_, std::try_to_lock);
        if (!playback_lock.owns_lock()) {
            return HistoryResult{
                .success = false,
                .changed = false,
                .steps_performed = 0,
                .error = "History playback already in progress",
            };
        }

        TransactionFrame frame;
        {
            std::lock_guard lock(mutex_);
            if (playback_depth_ > 0 && playback_thread_id_ == std::this_thread::get_id()) {
                return HistoryResult{
                    .success = false,
                    .changed = false,
                    .steps_performed = 0,
                    .error = "History playback already in progress",
                };
            }
            if (transactions_.empty()) {
                return HistoryResult{
                    .success = false,
                    .changed = false,
                    .steps_performed = 0,
                    .error = "No active history transaction",
                };
            }

            ++playback_depth_;
            playback_thread_id_ = std::this_thread::get_id();
            frame = std::move(transactions_.back());
            transactions_.pop_back();
            bumpGenerationLocked();
        }
        ScopeExit playback_guard([this]() {
            std::lock_guard lock(mutex_);
            if (playback_depth_ == 0) {
                return;
            }
            --playback_depth_;
            if (playback_depth_ == 0) {
                playback_thread_id_ = {};
            }
        });

        HistoryResult result{
            .success = true,
            .changed = false,
            .steps_performed = 0,
            .error = {},
        };
        DirtyMask dirty_flags = 0;

        auto restore_transaction = [this, &frame]() {
            std::lock_guard lock(mutex_);
            transactions_.push_back(std::move(frame));
            bumpGenerationLocked();
        };
        auto clear_history = [this]() {
            std::lock_guard lock(mutex_);
            clearLocked();
            bumpGenerationLocked();
        };

        auto& entries = frame.entries;
        if (entries.empty()) {
            notifyObservers();
            return result;
        }

        size_t undone_count = 0;
        for (auto it = entries.rbegin(); it != entries.rend(); ++it) {
            try {
                (*it)->restoreToPreferredDevice();
                (*it)->undo();
                ++undone_count;
                dirty_flags |= (*it)->dirtyFlags();
            } catch (const HistoryCorruptionError& e) {
                LOG_ERROR("Rollback encountered fatal history corruption for '{}': {}", (*it)->name(), e.what());
                clear_history();
                result.success = false;
                result.error = "Rollback failed and history was cleared: " + std::string(e.what());
                refreshAfterHistoryPlayback();
                notifyObservers();
                return result;
            } catch (const std::exception& e) {
                LOG_ERROR("Rollback failed for '{}': {}", (*it)->name(), e.what());
                try {
                    restoreUndoneTail(entries, undone_count);
                    restore_transaction();
                } catch (const std::exception& rollback_error) {
                    LOG_ERROR("Rollback compensation failed: {}", rollback_error.what());
                    clear_history();
                    result.success = false;
                    result.error =
                        "Rollback failed and compensation failed; history cleared: " + std::string(rollback_error.what());
                    refreshAfterHistoryPlayback();
                    notifyObservers();
                    return result;
                } catch (...) {
                    LOG_ERROR("Rollback compensation failed: unknown exception");
                    clear_history();
                    result.success = false;
                    result.error = "Rollback failed and compensation failed; history cleared: unknown exception";
                    refreshAfterHistoryPlayback();
                    notifyObservers();
                    return result;
                }
                result.success = false;
                result.error = e.what();
                refreshAfterHistoryPlayback();
                notifyObservers();
                return result;
            } catch (...) {
                LOG_ERROR("Rollback failed for '{}': unknown exception", (*it)->name());
                try {
                    restoreUndoneTail(entries, undone_count);
                    restore_transaction();
                } catch (const std::exception& rollback_error) {
                    LOG_ERROR("Rollback compensation failed: {}", rollback_error.what());
                    clear_history();
                    result.success = false;
                    result.error =
                        "Rollback failed and compensation failed; history cleared: " + std::string(rollback_error.what());
                    refreshAfterHistoryPlayback();
                    notifyObservers();
                    return result;
                } catch (...) {
                    LOG_ERROR("Rollback compensation failed: unknown exception");
                    clear_history();
                    result.success = false;
                    result.error = "Rollback failed and compensation failed; history cleared: unknown exception";
                    refreshAfterHistoryPlayback();
                    notifyObservers();
                    return result;
                }
                result.success = false;
                result.error = "unknown exception";
                refreshAfterHistoryPlayback();
                notifyObservers();
                return result;
            }
        }

        result.changed = undone_count > 0;
        result.steps_performed = undone_count;
        refreshAfterHistoryPlayback(dirty_flags);
        notifyObservers();
        return result;
    }

    bool UndoHistory::canUndo() const {
        return can_undo_.load(std::memory_order_acquire);
    }

    bool UndoHistory::canRedo() const {
        return can_redo_.load(std::memory_order_acquire);
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
        return transactionBytesLocked();
    }

    size_t UndoHistory::totalBytes() const {
        std::lock_guard lock(mutex_);
        return totalBytesLocked();
    }

    size_t UndoHistory::maxBytes() const {
        std::lock_guard lock(mutex_);
        return max_bytes_;
    }

    UndoMemoryBreakdown UndoHistory::undoMemory() const {
        std::lock_guard lock(mutex_);
        return stackMemoryLocked(undo_stack_);
    }

    UndoMemoryBreakdown UndoHistory::redoMemory() const {
        std::lock_guard lock(mutex_);
        return stackMemoryLocked(redo_stack_);
    }

    UndoMemoryBreakdown UndoHistory::transactionMemory() const {
        std::lock_guard lock(mutex_);
        return transactionMemoryLocked();
    }

    UndoMemoryBreakdown UndoHistory::totalMemory() const {
        std::lock_guard lock(mutex_);
        return totalMemoryLocked();
    }

    bool UndoHistory::hasActiveTransaction() const {
        std::lock_guard lock(mutex_);
        return !transactions_.empty();
    }

    size_t UndoHistory::transactionDepth() const {
        std::lock_guard lock(mutex_);
        return transactions_.size();
    }

    uint64_t UndoHistory::transactionAgeMs() const {
        std::lock_guard lock(mutex_);
        if (transactions_.empty()) {
            return 0;
        }
        const auto age = std::chrono::steady_clock::now() - transactions_.back().started_at;
        return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(age).count());
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
        return generation_.load(std::memory_order_acquire);
    }

    void UndoHistory::setMaxBytes(const size_t max_bytes) {
        bool changed = false;
        std::unique_lock playback_lock(playback_mutex_);
        {
            std::lock_guard lock(mutex_);
            max_bytes_ = std::max<size_t>(1, max_bytes);
            const auto before_undo = undo_stack_.size();
            const auto before_redo = redo_stack_.size();
            trimUndoStack();
            trimRedoStack();
            refreshResidencyLocked();
            changed = before_undo != undo_stack_.size() || before_redo != redo_stack_.size();
            if (changed) {
                bumpGenerationLocked();
            }
        }

        if (changed) {
            invalidateUndoRedoPollState();
            notifyObservers();
        }
    }

    void UndoHistory::shrinkToFit(const size_t target_gpu_bytes) {
        bool changed = false;
        std::unique_lock playback_lock(playback_mutex_);
        {
            std::lock_guard lock(mutex_);
            const auto before_memory = totalMemoryLocked();

            const auto offload_entries = [](auto& entries) {
                for (auto& entry : entries) {
                    if (entry) {
                        entry->offloadToCPU();
                    }
                }
            };

            offload_entries(undo_stack_);
            offload_entries(redo_stack_);
            for (auto& frame : transactions_) {
                offload_entries(frame.entries);
            }

            while (totalMemoryLocked().gpu_bytes > target_gpu_bytes) {
                if (!undo_stack_.empty()) {
                    undo_bytes_ -= entryBytes(undo_stack_.front());
                    undo_stack_.pop_front();
                    changed = true;
                    continue;
                }
                if (!redo_stack_.empty()) {
                    redo_bytes_ -= entryBytes(redo_stack_.front());
                    redo_stack_.pop_front();
                    changed = true;
                    continue;
                }
                break;
            }

            updateAvailabilityLocked();
            const auto after_memory = totalMemoryLocked();
            changed = changed || before_memory.cpu_bytes != after_memory.cpu_bytes ||
                      before_memory.gpu_bytes != after_memory.gpu_bytes;
            if (changed) {
                bumpGenerationLocked();
            }
        }

        if (changed) {
            invalidateUndoRedoPollState();
            notifyObservers();
        }
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

/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "core/job_registry.hpp"

#include <algorithm>
#include <cassert>
#include <ranges>
#include <utility>

namespace lfs::vis {

    JobRegistry::JobRegistry()
        : JobRegistry(std::this_thread::get_id()) {}

    JobRegistry::JobRegistry(const std::thread::id main_thread)
        : main_thread_(main_thread) {}

    void JobRegistry::assertMainThread() const {
        assert(
            std::this_thread::get_id() == main_thread_ &&
            "JobRegistry main-thread operation called from a worker");
    }

    JobRegistry::Entry*
    JobRegistry::findLocked(const JobHandle handle) {
        const auto it = entries_.find(handle.id);
        if (it == entries_.end() ||
            it->second.handle.type != handle.type) {
            return nullptr;
        }
        return &it->second;
    }

    const JobRegistry::Entry*
    JobRegistry::findLocked(const JobHandle handle) const {
        const auto it = entries_.find(handle.id);
        if (it == entries_.end() ||
            it->second.handle.type != handle.type) {
            return nullptr;
        }
        return &it->second;
    }

    std::optional<JobHandle>
    JobRegistry::init(const JobType type, std::string stage) {
        assertMainThread();
        const std::lock_guard lock(mutex_);
        if (active_by_type_.contains(type)) {
            return std::nullopt;
        }

        const JobHandle handle{
            .id = next_id_++,
            .type = type,
        };
        entries_.emplace(
            handle.id,
            Entry{
                .handle = handle,
                .status = JobStatus::Initialized,
                .progress = 0.0F,
                .stage = std::move(stage),
                .error = {},
                .cancel_requested = false,
                .worker_canceled = false,
            });
        active_by_type_.emplace(type, handle.id);
        return handle;
    }

    void JobRegistry::work(const JobHandle handle) {
        assert(
            std::this_thread::get_id() != main_thread_ &&
            "JobRegistry::work must run on a worker thread");
        const std::lock_guard lock(mutex_);
        auto* const entry = findLocked(handle);
        assert(entry && "JobRegistry::work received a stale handle");
        assert(
            entry->status == JobStatus::Initialized &&
            "JobRegistry::work requires initialized state");
        entry->status = JobStatus::Running;
    }

    void JobRegistry::report(
        const JobHandle handle,
        const std::optional<float> progress,
        std::optional<std::string> stage,
        std::optional<std::string> error) {
        const std::lock_guard lock(mutex_);
        auto* const entry = findLocked(handle);
        if (!entry ||
            (entry->status != JobStatus::Initialized &&
             entry->status != JobStatus::Running &&
             entry->status != JobStatus::CompletionPending)) {
            return;
        }
        if (progress) {
            entry->progress =
                std::clamp(*progress, 0.0F, 1.0F);
        }
        if (stage) {
            entry->stage = std::move(*stage);
        }
        if (error) {
            entry->error = std::move(*error);
        }
    }

    void JobRegistry::finishWork(
        const JobHandle handle, const bool canceled,
        std::string error,
        const std::optional<lfs::ErrorCode> error_code) {
        assert(
            std::this_thread::get_id() != main_thread_ &&
            "JobRegistry::finishWork must run on a worker thread");
        const std::lock_guard lock(mutex_);
        auto* const entry = findLocked(handle);
        if (!entry ||
            (entry->status != JobStatus::Initialized &&
             entry->status != JobStatus::Running)) {
            return;
        }
        entry->worker_canceled =
            canceled || entry->cancel_requested;
        if (!error.empty()) {
            entry->error = std::move(error);
        }
        entry->error_code = error_code;
        entry->status = JobStatus::CompletionPending;
    }

    std::optional<JobSnapshot>
    JobRegistry::update(const JobHandle handle) const {
        assertMainThread();
        return peek(handle);
    }

    std::optional<JobSnapshot>
    JobRegistry::peek(const JobHandle handle) const {
        const std::lock_guard lock(mutex_);
        const auto* const entry = findLocked(handle);
        if (!entry) {
            return std::nullopt;
        }
        return JobSnapshot{
            .handle = entry->handle,
            .status = entry->status,
            .progress = entry->progress,
            .stage = entry->stage,
            .error = entry->error,
            .error_code = entry->error_code,
            .cancel_requested =
                entry->cancel_requested,
            .worker_canceled =
                entry->worker_canceled,
        };
    }

    void JobRegistry::completed(const JobHandle handle) {
        assertMainThread();
        const std::lock_guard lock(mutex_);
        auto* const entry = findLocked(handle);
        assert(entry && "JobRegistry::completed received a stale handle");
        assert(
            entry->status != JobStatus::Completed &&
            entry->status != JobStatus::Failed &&
            entry->status != JobStatus::Canceled &&
            "JobRegistry::completed requires an active job");
        entry->status = JobStatus::Completed;
        entry->progress = 1.0F;
        active_by_type_.erase(entry->handle.type);
    }

    void JobRegistry::canceled(const JobHandle handle) {
        assertMainThread();
        const std::lock_guard lock(mutex_);
        auto* const entry = findLocked(handle);
        assert(entry && "JobRegistry::canceled received a stale handle");
        assert(
            entry->status != JobStatus::Completed &&
            entry->status != JobStatus::Failed &&
            entry->status != JobStatus::Canceled &&
            "JobRegistry::canceled requires an active job");
        entry->status = JobStatus::Canceled;
        active_by_type_.erase(entry->handle.type);
    }

    void JobRegistry::failed(
        const JobHandle handle, std::string error,
        std::string stage) {
        assertMainThread();
        const std::lock_guard lock(mutex_);
        auto* const entry = findLocked(handle);
        assert(entry && "JobRegistry::failed received a stale handle");
        assert(
            entry->status != JobStatus::Completed &&
            entry->status != JobStatus::Failed &&
            entry->status != JobStatus::Canceled &&
            "JobRegistry::failed requires an active job");
        entry->error = std::move(error);
        entry->stage = std::move(stage);
        entry->status = JobStatus::Failed;
        active_by_type_.erase(entry->handle.type);
    }

    void JobRegistry::free(const JobHandle handle) {
        assertMainThread();
        const std::lock_guard lock(mutex_);
        const auto it = entries_.find(handle.id);
        if (it == entries_.end() ||
            it->second.handle.type != handle.type) {
            return;
        }
        assert(
            it->second.status != JobStatus::Initialized &&
            it->second.status != JobStatus::Running &&
            it->second.status !=
                JobStatus::CompletionPending &&
            "JobRegistry::free cannot release a running job");
        active_by_type_.erase(it->second.handle.type);
        entries_.erase(it);
    }

    void JobRegistry::requestCancel(
        const JobHandle handle, std::string stage) {
        assertMainThread();
        const std::lock_guard lock(mutex_);
        auto* const entry = findLocked(handle);
        if (!entry ||
            (entry->status != JobStatus::Initialized &&
             entry->status != JobStatus::Running &&
             entry->status != JobStatus::CompletionPending)) {
            return;
        }
        entry->cancel_requested = true;
        entry->stage = std::move(stage);
    }

    bool JobRegistry::cancelRequested(
        const JobHandle handle) const {
        const std::lock_guard lock(mutex_);
        const auto* const entry = findLocked(handle);
        return entry && entry->cancel_requested;
    }

    bool JobRegistry::anyRunning(const JobType type) const {
        const std::lock_guard lock(mutex_);
        return active_by_type_.contains(type);
    }

    std::optional<JobSnapshot>
    JobRegistry::active(const JobType type) const {
        const std::lock_guard lock(mutex_);
        const auto active = active_by_type_.find(type);
        if (active == active_by_type_.end()) {
            return std::nullopt;
        }
        const auto* entry = findLocked(JobHandle{active->second, type});
        if (!entry) {
            return std::nullopt;
        }
        return JobSnapshot{
            .handle = entry->handle,
            .status = entry->status,
            .progress = entry->progress,
            .stage = entry->stage,
            .error = entry->error,
            .error_code = entry->error_code,
            .cancel_requested = entry->cancel_requested,
            .worker_canceled = entry->worker_canceled,
        };
    }

} // namespace lfs::vis

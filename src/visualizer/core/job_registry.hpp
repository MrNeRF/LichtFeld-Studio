/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "core/error_codes.hpp"
#include "core/export.hpp"

#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>

namespace lfs::vis {

    enum class JobType : std::uint8_t {
        Export,
        Import,
        VideoExport,
        Mesh2Splat,
        SplatSimplify,
        ProjectWrite,
        ProjectOpen,
    };

    enum class JobStatus : std::uint8_t {
        Initialized,
        Running,
        CompletionPending,
        Completed,
        Failed,
        Canceled,
    };

    struct JobHandle {
        std::uint64_t id = 0;
        JobType type = JobType::Export;

        [[nodiscard]] explicit operator bool() const noexcept {
            return id != 0;
        }

        friend bool operator==(const JobHandle&,
                               const JobHandle&) = default;
    };

    struct JobSnapshot {
        JobHandle handle;
        JobStatus status = JobStatus::Initialized;
        float progress = 0.0F;
        std::string stage;
        std::string error;
        std::optional<lfs::ErrorCode> error_code;
        bool cancel_requested = false;
        bool worker_canceled = false;

        [[nodiscard]] bool running() const noexcept {
            return status == JobStatus::Initialized ||
                   status == JobStatus::Running ||
                   status == JobStatus::CompletionPending;
        }
    };

    // One registry owns the lifecycle and user-visible state for every
    // background job. Main-thread transitions are deliberately separate from
    // worker publication so a worker can never publish its own completion.
    class LFS_VIS_API JobRegistry {
    public:
        JobRegistry();
        explicit JobRegistry(std::thread::id main_thread);

        JobRegistry(const JobRegistry&) = delete;
        JobRegistry& operator=(const JobRegistry&) = delete;

        // Main thread: reserves the exclusive slot for type.
        [[nodiscard]] std::optional<JobHandle>
        init(JobType type, std::string stage = {});

        // Worker thread: enters the work phase, reports state, then records
        // that a main-thread completion transition is pending.
        void work(JobHandle handle);
        void report(JobHandle handle,
                    std::optional<float> progress,
                    std::optional<std::string> stage = std::nullopt,
                    std::optional<std::string> error = std::nullopt);
        void finishWork(JobHandle handle, bool canceled,
                        std::string error = {},
                        std::optional<lfs::ErrorCode> error_code = std::nullopt);

        // Main thread: observes and performs lifecycle transitions.
        [[nodiscard]] std::optional<JobSnapshot>
        update(JobHandle handle) const;
        [[nodiscard]] std::optional<JobSnapshot>
        peek(JobHandle handle) const;
        void completed(JobHandle handle);
        void canceled(JobHandle handle);
        void failed(JobHandle handle, std::string error,
                    std::string stage = "Failed");
        void free(JobHandle handle);
        void requestCancel(JobHandle handle,
                           std::string stage = "Cancelling");

        [[nodiscard]] bool cancelRequested(
            JobHandle handle) const;
        [[nodiscard]] bool anyRunning(JobType type) const;
        [[nodiscard]] std::optional<JobSnapshot>
        active(JobType type) const;

    private:
        struct Entry {
            JobHandle handle;
            JobStatus status = JobStatus::Initialized;
            float progress = 0.0F;
            std::string stage;
            std::string error;
            std::optional<lfs::ErrorCode> error_code;
            bool cancel_requested = false;
            bool worker_canceled = false;
        };

        void assertMainThread() const;
        [[nodiscard]] Entry* findLocked(JobHandle handle);
        [[nodiscard]] const Entry*
        findLocked(JobHandle handle) const;

        const std::thread::id main_thread_;
        mutable std::mutex mutex_;
        std::unordered_map<std::uint64_t, Entry> entries_;
        std::unordered_map<JobType, std::uint64_t> active_by_type_;
        std::uint64_t next_id_ = 1;
    };

} // namespace lfs::vis

/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/tensor_execution.hpp"
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace lfs::gpu_ops {
    struct ArenaMemoryInfo {
        size_t arena_capacity, required_bytes, current_usage, peak_usage;
    };
    struct IdleArenaBorrow {
        void* owner = nullptr;
        char* data = nullptr;
        size_t bytes = 0;
        uint64_t frame = 0;
    };
    struct SessionOps {
        void (*profile)(bool start);
        void (*resize_arena)(std::string_view boundary, bool release_all);
        void (*reset_arena)();
        void (*dump_arena_statistics)();
        std::optional<ArenaMemoryInfo> (*arena_memory_info)();
        void (*log_arena_failure)(const char* context);
        uint32_t (*set_arena_timeout)(uint32_t timeout);
        IdleArenaBorrow (*borrow_idle_arena)(size_t wanted, size_t minimum, core::TensorExecutionTarget);
        void (*release_idle_arena)(const IdleArenaBorrow&, core::TensorExecutionTarget);
        void* (*zero_idle_arena)(const IdleArenaBorrow&, size_t bytes, core::TensorExecutionTarget);
        size_t (*allocation_bytes)(size_t bytes);
        size_t (*direct_storage_live_bytes)();
        std::optional<std::string> (*last_error)();
        size_t (*device_baseline_bytes)();
        void (*sample_memory)();
        // Backends with persistent private workspaces can relinquish them
        // between training and evaluation, then recreate them on resume.
        bool release_workspaces_before_evaluation = false;
        // Row reordering temporarily duplicates model and optimizer storage.
        bool release_workspaces_before_reorder = false;
    };

    class ScopedArenaTimeout {
    public:
        ScopedArenaTimeout(const SessionOps& ops, uint32_t timeout)
            : ops_(ops), previous_(ops.set_arena_timeout(timeout)) {}
        ~ScopedArenaTimeout() { ops_.set_arena_timeout(previous_); }
        ScopedArenaTimeout(const ScopedArenaTimeout&) = delete;
        ScopedArenaTimeout& operator=(const ScopedArenaTimeout&) = delete;

    private:
        const SessionOps& ops_;
        uint32_t previous_;
    };
} // namespace lfs::gpu_ops

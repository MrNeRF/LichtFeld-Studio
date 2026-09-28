/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "metal_families.hpp"

#include "core/gpu_device_runtime.hpp"
#include "core/logger.hpp"

#include <atomic>

// Metal has no rasterizer arena: frames allocate from the tensor allocator,
// so the arena hooks have nothing to resize, report or lend.
namespace lfs::training {
    namespace {
        std::atomic<uint32_t> arena_timeout{0};

        void log_arena_failure(const char* const context) {
            LOG_ERROR("Metal training allocation failed: {}", context);
        }
    } // namespace

    const lfs::gpu_ops::SessionOps& metal_session_ops() {
        static const lfs::gpu_ops::SessionOps ops{
            .profile = [](bool) {},
            .resize_arena = [](std::string_view, bool) {},
            .reset_arena = [] {},
            .dump_arena_statistics = [] {},
            .arena_memory_info = []() -> std::optional<lfs::gpu_ops::ArenaMemoryInfo> { return std::nullopt; },
            .log_arena_failure = log_arena_failure,
            .set_arena_timeout = [](const uint32_t timeout) { return arena_timeout.exchange(timeout); },
            .borrow_idle_arena = [](size_t, size_t, core::TensorExecutionTarget) { return lfs::gpu_ops::IdleArenaBorrow{}; },
            .release_idle_arena = [](const lfs::gpu_ops::IdleArenaBorrow&, core::TensorExecutionTarget) {},
            .zero_idle_arena = [](const lfs::gpu_ops::IdleArenaBorrow&, size_t, core::TensorExecutionTarget) -> void* {
                return nullptr;
            },
            .allocation_bytes = [](const size_t bytes) { return core::gpu_allocation_bytes(core::GpuBackend::Metal, bytes); },
            .direct_storage_live_bytes = [] { return size_t{0}; },
            .last_error = []() -> std::optional<std::string> { return std::nullopt; },
            .device_baseline_bytes = [] { return size_t{0}; },
            .sample_memory = [] {},
        };
        return ops;
    }

} // namespace lfs::training

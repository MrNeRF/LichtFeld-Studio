/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "memory_arena.hpp"

#include "core/logger.hpp"
#include "diagnostics/vram_profiler.hpp"

namespace lfs::core {

    GlobalArenaManager& global_arena_manager() {
        static GlobalArenaManager manager;
        return manager;
    }

    void shutdown_global_arena_manager() {
        auto& manager = global_arena_manager();
        std::lock_guard<std::mutex> lock(manager.init_mutex_);
        if (manager.shutdown_) {
            return;
        }

        manager.shutdown_ = true;
        if (!manager.arena_) {
            return;
        }

        manager.arena_->full_reset();
        manager.arena_.reset();
    }

    void log_arena_failure_vram_snapshot(
        const std::string_view label, const size_t committed_bytes,
        const size_t frame_peak_bytes) {
        size_t free_bytes = 0;
        size_t total_bytes = 0;
        (void)cudaMemGetInfo(&free_bytes, &total_bytes);

        std::uint64_t pool_used = 0;
        std::uint64_t pool_reserved = 0;
#if CUDART_VERSION >= 12080
        int device = 0;
        cudaMemPool_t pool = nullptr;
        if (cudaGetDevice(&device) == cudaSuccess &&
            cudaDeviceGetDefaultMemPool(&pool, device) == cudaSuccess) {
            (void)cudaMemPoolGetAttribute(
                pool, cudaMemPoolAttrUsedMemCurrent, &pool_used);
            (void)cudaMemPoolGetAttribute(
                pool, cudaMemPoolAttrReservedMemCurrent, &pool_reserved);
        }
#endif

        const std::string label_text = label.empty() ? "unnamed" : std::string(label);
        const auto profiler = lfs::diagnostics::VramProfiler::instance().snapshot();
        const auto& process = profiler.process;
        LOG_ERROR(
            "Arena VRAM failure snapshot label=%s cuda_free=%zu MiB cuda_total=%zu MiB "
            "arena_committed=%zu MiB frame_peak=%zu MiB exportable_splat=%zu MiB "
            "shared_scratch=%zu MiB cuda_default_pool_reserved=%zu MiB "
            "cuda_default_pool_used=%zu MiB",
            label_text.c_str(),
            free_bytes >> 20, total_bytes >> 20,
            committed_bytes >> 20, frame_peak_bytes >> 20,
            process.exportable_splat_bytes >> 20,
            process.shared_scratch_bytes >> 20,
            static_cast<size_t>(pool_reserved >> 20),
            static_cast<size_t>(pool_used >> 20));
    }

} // namespace lfs::core

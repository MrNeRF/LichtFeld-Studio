/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs/training/ops/session_cuda.hpp"

#include "core/cuda/memory_arena.hpp"
#include "core/cuda_error.hpp"
#include "core/logger.hpp"
#include "core/tensor.hpp"
#include "core/tensor/backend/cuda/runtime/size_bucketed_pool.hpp"
#include "diagnostics/vram_profiler.hpp"
#include <algorithm>
#include <cuda_profiler_api.h>
#include <format>

namespace lfs::training {
    namespace {

        void resize_arena(std::string_view boundary,
                          bool release_all) {
            auto* arena = lfs::core::GlobalArenaManager::instance().try_get_arena();
            if (!arena) {
                return;
            }
            const bool resized = release_all
                                     ? arena->release_at_boundary()
                                     : arena->shrink_to_current_at_boundary();
            if (!resized) {
                LOG_WARN("Rasterizer arena {} boundary could not drain every CUDA device",
                         boundary);
            }
            if (release_all) {
                // The viewer may have installed its exportable block as the
                // arena backing. release_at_boundary intentionally preserves
                // such a block for the peer owner; once the trainer is idle it
                // must relinquish the arena's reference so the next resume
                // starts from a zero-sized training arena.
                if (arena->using_external_backing()) {
                    lfs::core::GlobalArenaManager::instance().clear_external_backing();
                }
            }
        }

        void reset_arena() { core::GlobalArenaManager::instance().get_arena().full_reset(); }

        std::optional<lfs::gpu_ops::ArenaMemoryInfo> arena_memory_info() {
            if (auto* arena = core::GlobalArenaManager::instance().try_get_arena()) {
                const auto info = arena->get_memory_info();
                return lfs::gpu_ops::ArenaMemoryInfo{info.arena_capacity, info.required_bytes, info.current_usage, info.peak_usage};
            }
            return std::nullopt;
        }
        void log_arena_failure(const char* context) {
            const auto info = arena_memory_info();
            core::log_arena_failure_vram_snapshot(context, info ? info->arena_capacity : 0, info ? info->peak_usage : 0);
        }
        lfs::gpu_ops::IdleArenaBorrow borrow_idle_arena(size_t wanted, size_t minimum, core::TensorExecutionTarget target) {
            auto* arena = core::GlobalArenaManager::instance().try_get_arena();
            if (!arena || wanted == 0)
                return {};
            const auto bytes = std::min(wanted, arena->get_memory_info().arena_capacity);
            if (bytes == 0 || bytes < minimum)
                return {};
            const auto frame = arena->try_begin_frame(static_cast<cudaStream_t>(target.native_handle()));
            if (!frame)
                return {};
            char* data = arena->get_allocator(*frame, "training.idle_scratch")(bytes);
            return {arena, data, data ? bytes : 0, *frame};
        }
        void release_idle_arena(const lfs::gpu_ops::IdleArenaBorrow& borrow, core::TensorExecutionTarget target) {
            if (auto* arena = static_cast<core::RasterizerMemoryArena*>(borrow.owner))
                arena->end_frame(borrow.frame, static_cast<cudaStream_t>(target.native_handle()));
        }
        void* zero_idle_arena(const lfs::gpu_ops::IdleArenaBorrow& borrow, size_t bytes, core::TensorExecutionTarget target) {
            if (bytes == 0 || bytes > borrow.bytes)
                return nullptr;
            LFS_CUDA_CHECK(cudaMemsetAsync(borrow.data, 0, bytes, static_cast<cudaStream_t>(target.native_handle())));
            return borrow.data;
        }
        std::optional<std::string> last_error() {
            const auto error = cudaGetLastError();
            if (error != cudaSuccess)
                return std::format("CUDA kernel error: {}", cudaGetErrorString(error));
            return std::nullopt;
        }

        void profile(const bool start) {
            if (start)
                cudaProfilerStart();
            else
                cudaProfilerStop();
        }

    } // namespace

    const lfs::gpu_ops::SessionOps& cuda_session_ops() {
        static const lfs::gpu_ops::SessionOps ops{
            .profile = profile,
            .resize_arena = resize_arena,
            .reset_arena = reset_arena,
            .arena_memory_info = arena_memory_info,
            .log_arena_failure = log_arena_failure,
            .set_arena_timeout = core::RasterizerMemoryArena::set_begin_frame_timeout,
            .borrow_idle_arena = borrow_idle_arena,
            .release_idle_arena = release_idle_arena,
            .zero_idle_arena = zero_idle_arena,
            .allocation_bytes = core::SizeBucketedPool::get_bucket_size,
            .direct_storage_live_bytes = core::Tensor::cuda_direct_storage_live_bytes,
            .last_error = last_error,
            .device_baseline_bytes = [] { return diagnostics::VramProfiler::instance().cudaDeviceBaselineBytes(); },
            .sample_memory = [] { diagnostics::VramProfiler::instance().sampleCudaMemory(); },
        };
        return ops;
    }

} // namespace lfs::training

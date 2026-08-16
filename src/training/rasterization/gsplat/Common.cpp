/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "Common.h"
#include "memory_pool.hpp"

#include <format>

#if LFS_CUDA_FAILURE_INJECTION_ENABLED
#include <atomic>
#endif

namespace gsplat_lfs {

    void* allocate_exact_async_storage(const size_t bytes,
                                       const cudaStream_t stream,
                                       const char* const label) {
        lfs::core::CudaMemoryPool::LabelGuard label_guard(label);
        return lfs::core::allocate_cuda_storage(
            bytes, stream, lfs::core::CudaStorageMode::ExactAsync,
            label, "gsplat exact workspace allocation");
    }

    void release_exact_async_storage(void* const ptr, const cudaStream_t stream) noexcept {
        lfs::core::safe_cuda_pool_deallocate(ptr, stream);
    }

    void record_exact_async_stream(void* const ptr, const cudaStream_t stream) noexcept {
        lfs::core::CudaMemoryPool::instance().record_stream(ptr, stream);
    }

    namespace {
        // Thread-local grow-only CUB workspace for scan/sort in the gsplat path.
        // Replaces per-call StreamOrderedDeviceBuffer alloc/free (CudaCubWorkspace).
        struct GsplatCubWorkspaceCache {
            StreamOrderedDeviceBuffer buffer;
            size_t capacity_bytes = 0;

            void* ensure(const size_t bytes, const cudaStream_t stream) {
                if (bytes == 0) {
                    return nullptr;
                }
                if (bytes <= capacity_bytes && buffer) {
                    buffer.bind_stream(stream);
                    return buffer.get();
                }
                const size_t new_cap = bytes;
                StreamOrderedDeviceBuffer replacement(
                    new_cap, stream, "rasterizer.gsplat.cub_workspace");
                buffer = std::move(replacement);
                capacity_bytes = new_cap;
                return buffer.get();
            }

            bool release() noexcept {
                buffer.reset();
                capacity_bytes = 0;
                return !buffer;
            }
        };

        GsplatCubWorkspaceCache& cub_cache() {
            static thread_local GsplatCubWorkspaceCache cache;
            return cache;
        }

        // Per-backward intermediate for rasterize_from_world_with_sh_bwd color grads.
        // Same grow-only pattern as CUB workspace (stream-ordered, TLS).
        struct GsplatColorGradWorkspaceCache {
            StreamOrderedDeviceBuffer buffer;
            size_t capacity_bytes = 0;

            void* ensure(const size_t bytes, const cudaStream_t stream) {
                if (bytes == 0) {
                    return nullptr;
                }
                if (bytes <= capacity_bytes && buffer) {
                    buffer.bind_stream(stream);
                    return buffer.get();
                }
                const size_t new_cap = bytes;
                StreamOrderedDeviceBuffer replacement(
                    new_cap, stream, "rasterizer.gsplat.color_gradients");
                buffer = std::move(replacement);
                capacity_bytes = new_cap;
                return buffer.get();
            }

            bool release() noexcept {
                buffer.reset();
                capacity_bytes = 0;
                return !buffer;
            }
        };

        GsplatColorGradWorkspaceCache& color_grad_cache() {
            static thread_local GsplatColorGradWorkspaceCache cache;
            return cache;
        }
    } // namespace

    void* ensure_gsplat_cub_workspace(const size_t bytes, const cudaStream_t stream) {
        return cub_cache().ensure(bytes, stream);
    }

    bool release_gsplat_cub_workspace() noexcept {
        return cub_cache().release();
    }

    void* ensure_gsplat_color_grad_workspace(const size_t bytes, const cudaStream_t stream) {
        return color_grad_cache().ensure(bytes, stream);
    }

    bool release_gsplat_color_grad_workspace() noexcept {
        return color_grad_cache().release();
    }

#if LFS_CUDA_FAILURE_INJECTION_ENABLED
    namespace {
        std::atomic_bool force_cuda_allocation_failure{false};
    }

    void set_cuda_allocation_failure_for_testing(const bool fail) {
        force_cuda_allocation_failure.store(fail, std::memory_order_relaxed);
    }

    bool cuda_allocation_failure_is_forced() {
        return force_cuda_allocation_failure.load(std::memory_order_relaxed);
    }
#endif

} // namespace gsplat_lfs

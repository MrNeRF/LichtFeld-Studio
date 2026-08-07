/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "Common.h"

#include <format>
#include <limits>

#if LFS_CUDA_FAILURE_INJECTION_ENABLED
#include <atomic>
#endif

namespace gsplat_lfs {

    namespace {
        size_t growth_capacity_bytes(const size_t required, const std::string_view allocation) {
            const size_t headroom = required / 4;
            LFS_ASSERT_MSG(
                required <= std::numeric_limits<size_t>::max() - headroom,
                std::format("{} capacity overflow for {} bytes", allocation, required));
            return required + headroom;
        }

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
                    return buffer.get();
                }
                const size_t new_cap = growth_capacity_bytes(bytes, "gsplat cub workspace");
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
                    return buffer.get();
                }
                const size_t new_cap = growth_capacity_bytes(bytes, "gsplat color grad workspace");
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

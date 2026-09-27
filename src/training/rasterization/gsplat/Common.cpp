/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "Common.h"
#include "core/tensor/backend/cuda/runtime/memory_pool.hpp"

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
        void* ensure_workspace(StreamOrderedDeviceBuffer& buffer, size_t& capacity,
                               size_t bytes, cudaStream_t stream, const char* label) {
            if (bytes == 0)
                return nullptr;
            if (bytes <= capacity && buffer) {
                buffer.bind_stream(stream);
                return buffer.get();
            }
            StreamOrderedDeviceBuffer replacement(bytes, stream, label);
            buffer = std::move(replacement);
            capacity = bytes;
            return buffer.get();
        }
    } // namespace

    void* ensure_gsplat_cub_workspace(Workspace& saved, size_t bytes, cudaStream_t stream) {
        return ensure_workspace(saved.cub, saved.cub_capacity, bytes, stream, "rasterizer.gsplat.cub_workspace");
    }
    void* ensure_gsplat_color_grad_workspace(Workspace& saved, size_t bytes, cudaStream_t stream) {
        return ensure_workspace(saved.color_grad, saved.color_grad_capacity, bytes, stream, "rasterizer.gsplat.color_gradients");
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

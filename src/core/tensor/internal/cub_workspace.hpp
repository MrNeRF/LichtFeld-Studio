/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/cuda_allocation.hpp"
#include "memory_pool.hpp"

#include <cuda_runtime.h>
#include <string>
#include <string_view>
#include <utility>

namespace lfs::core::tensor_ops {

    LFS_CORE_API bool cub_workspace_failure_is_forced();
    LFS_CORE_API void set_cub_workspace_failure_for_testing(bool fail);

    struct MemoryPoolCudaAllocator {
        [[nodiscard]] void* allocate(const size_t bytes,
                                     const cudaStream_t stream,
                                     const std::string_view label) const {
            CudaMemoryPool::LabelGuard label_guard(label);
            void* ptr = CudaMemoryPool::instance().allocate(bytes, stream);
            LFS_ASSERT_MSG(ptr != nullptr,
                           std::string(label) + " CUDA pool allocation failed");
            return ptr;
        }

        void deallocate(void* ptr, const cudaStream_t stream) const noexcept {
            CudaMemoryPool::instance().deallocate(ptr, stream);
        }
    };

    struct TensorCubAllocationHooks {
        void before_allocate(std::string_view) const {
            LFS_ASSERT_MSG(!cub_workspace_failure_is_forced(),
                           "tensor CUB workspace allocation failure injected");
        }

        void after_allocate(void*, size_t, std::string_view) const noexcept {}
        void before_deallocate(void*) const noexcept {}
    };

    struct TensorCubWorkspaceTraits {
        static constexpr std::string_view allocation_label = "tensor.cub_workspace";
        static constexpr std::string_view diagnostic_scope = "tensor";
    };

    using ScopedDeviceBuffer = UniqueCudaAllocation<MemoryPoolCudaAllocator>;
    using TensorCubWorkspace = CudaCubWorkspace<
        UniqueCudaAllocation<MemoryPoolCudaAllocator, TensorCubAllocationHooks>,
        TensorCubWorkspaceTraits>;

    template <typename Operation>
    void run_cub_operation(const std::string_view name,
                           const cudaStream_t stream,
                           Operation&& operation) {
        lfs::core::run_cub_operation<TensorCubWorkspace>(
            name, stream, std::forward<Operation>(operation));
    }

} // namespace lfs::core::tensor_ops

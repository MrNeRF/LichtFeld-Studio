/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/assert.hpp"
#include "memory_pool.hpp"

#include <cuda_runtime.h>
#include <format>
#include <string>
#include <string_view>
#include <utility>

namespace lfs::core::tensor_ops {

    LFS_CORE_API bool cub_workspace_failure_is_forced();
    LFS_CORE_API void set_cub_workspace_failure_for_testing(bool fail);

    inline void check_cuda_status(const cudaError_t status, const std::string_view operation) {
        if (status == cudaSuccess) {
            return;
        }
        const std::string message = std::format(
            "{} failed: {} ({})", operation, cudaGetErrorString(status), cudaGetErrorName(status));
        cudaGetLastError();
        LFS_ASSERT_MSG(status == cudaSuccess, message);
    }

    class ScopedDeviceBuffer {
    public:
        ScopedDeviceBuffer() = default;

        ScopedDeviceBuffer(const size_t bytes,
                           const cudaStream_t stream,
                           const std::string_view label) {
            allocate(bytes, stream, label);
        }

        ~ScopedDeviceBuffer() {
            reset();
        }

        ScopedDeviceBuffer(const ScopedDeviceBuffer&) = delete;
        ScopedDeviceBuffer& operator=(const ScopedDeviceBuffer&) = delete;
        ScopedDeviceBuffer(ScopedDeviceBuffer&&) = delete;
        ScopedDeviceBuffer& operator=(ScopedDeviceBuffer&&) = delete;

        void allocate(const size_t bytes,
                      const cudaStream_t stream,
                      const std::string_view label) {
            LFS_ASSERT_MSG(ptr_ == nullptr, "Scoped CUDA buffer cannot be allocated twice");
            LFS_ASSERT_MSG(bytes > 0, "Scoped CUDA buffer requires a nonzero size");

            CudaMemoryPool::LabelGuard label_guard(label);
            ptr_ = CudaMemoryPool::instance().allocate(bytes, stream);
            LFS_ASSERT_MSG(
                ptr_ != nullptr,
                std::format("CUDA allocation for '{}' failed ({} bytes)", label, bytes));
            bytes_ = bytes;
            stream_ = stream;
        }

        void reset() noexcept {
            if (!ptr_) {
                return;
            }
            CudaMemoryPool::instance().deallocate(ptr_, stream_);
            ptr_ = nullptr;
            bytes_ = 0;
        }

        void* get() const noexcept { return ptr_; }

        template <typename T>
        T* as() const noexcept {
            return static_cast<T*>(ptr_);
        }

        size_t size() const noexcept { return bytes_; }

    private:
        void* ptr_ = nullptr;
        size_t bytes_ = 0;
        cudaStream_t stream_ = nullptr;
    };

    template <typename Operation>
    void run_cub_operation(const std::string_view name,
                           const cudaStream_t stream,
                           Operation&& operation) {
        size_t workspace_bytes = 0;
        check_cuda_status(
            operation(nullptr, workspace_bytes), std::format("{} workspace query", name));
        LFS_ASSERT_MSG(
            workspace_bytes > 0,
            std::format("{} returned an empty workspace for a nonempty operation", name));

        if (cub_workspace_failure_is_forced()) {
            LFS_ASSERT_MSG(false, std::format("{} workspace allocation failure injected", name));
        }

        ScopedDeviceBuffer workspace(workspace_bytes, stream, "tensor.cub_workspace");
        LFS_ASSERT_MSG(
            workspace.get() != nullptr,
            std::format("{} cannot execute with null workspace ({} bytes)", name, workspace_bytes));
        check_cuda_status(operation(workspace.get(), workspace_bytes), name);
    }

} // namespace lfs::core::tensor_ops

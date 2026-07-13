/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/assert.hpp"
#include "core/checked_arithmetic.hpp"
#include "core/cuda_error.hpp"
#include "diagnostics/vram_profiler.hpp"

#include <cuda_runtime.h>
#include <format>
#include <limits>
#include <string>
#include <string_view>

namespace lfs::training::cuda_scratch {

    inline size_t checked_bytes(const size_t count,
                                const size_t element_size,
                                const std::string_view allocation) {
        return lfs::core::checked_product(count, element_size, allocation);
    }

    class DeviceBuffer {
    public:
        DeviceBuffer() = default;

        DeviceBuffer(const size_t bytes,
                     const cudaStream_t stream,
                     const std::string_view label) {
            allocate(bytes, stream, label);
        }

        ~DeviceBuffer() {
            reset();
        }

        DeviceBuffer(const DeviceBuffer&) = delete;
        DeviceBuffer& operator=(const DeviceBuffer&) = delete;
        DeviceBuffer(DeviceBuffer&&) = delete;
        DeviceBuffer& operator=(DeviceBuffer&&) = delete;

        void allocate(const size_t bytes,
                      const cudaStream_t stream,
                      const std::string_view label) {
            LFS_ASSERT_MSG(ptr_ == nullptr, "CUDA scratch buffer cannot be allocated twice");
            LFS_ASSERT_MSG(bytes > 0, "CUDA scratch buffer requires a nonzero size");
            void* ptr = nullptr;
#if CUDART_VERSION >= 11020
            LFS_CUDA_CHECK_MSG(cudaMallocAsync(&ptr, bytes, stream),
                               "CUDA scratch allocation '{}' ({} bytes)", label, bytes);
            constexpr auto method = diagnostics::VramAllocationMethod::Async;
#else
            LFS_CUDA_CHECK_MSG(cudaMalloc(&ptr, bytes),
                               "CUDA scratch allocation '{}' ({} bytes)", label, bytes);
            constexpr auto method = diagnostics::VramAllocationMethod::Direct;
#endif
            LFS_ASSERT_MSG(ptr != nullptr,
                           std::format("CUDA allocation for '{}' returned null ({} bytes)", label, bytes));

            ptr_ = ptr;
            bytes_ = bytes;
            stream_ = stream;
            try {
                diagnostics::VramProfiler::instance().recordAllocation(ptr_, bytes_, method, label);
            } catch (...) {
                // Allocation diagnostics must not change training behavior.
            }
        }

        void reset() noexcept {
            if (!ptr_) {
                return;
            }
            try {
                diagnostics::VramProfiler::instance().recordDeallocation(ptr_);
            } catch (...) {
                // Allocation diagnostics must not change training behavior.
            }
#if CUDART_VERSION >= 11020
            const cudaError_t status = cudaFreeAsync(ptr_, stream_);
#else
            const cudaError_t status = cudaFree(ptr_);
#endif
            if (status != cudaSuccess) {
                lfs::core::ensure_cuda_success(
                    status, "CUDA scratch buffer free",
                    std::format("ptr={}, bytes={}", ptr_, bytes_),
                    std::source_location::current(),
                    lfs::core::CudaFailureDisposition::LogOnly);
                cudaGetLastError();
            }
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

    class CubWorkspace {
    public:
        template <typename Query>
        CubWorkspace(const std::string_view operation,
                     const cudaStream_t stream,
                     Query&& query)
            : operation_(operation) {
            LFS_CUDA_CHECK_MSG(query(nullptr, bytes_),
                               "{} workspace query", operation_);
            LFS_ASSERT_MSG(
                bytes_ > 0,
                std::format("{} returned an empty workspace for a nonempty operation", operation_));
            buffer_.allocate(bytes_, stream, "training.cub_workspace");
            LFS_ASSERT_MSG(
                buffer_.get() != nullptr,
                std::format("{} cannot execute with null workspace ({} bytes)", operation_, bytes_));
        }

        template <typename Operation>
        void run(Operation&& operation) {
            LFS_CUDA_CHECK_MSG(operation(buffer_.get(), bytes_),
                               "training CUB workspace operation: {}", operation_);
        }

        CubWorkspace(const CubWorkspace&) = delete;
        CubWorkspace& operator=(const CubWorkspace&) = delete;
        CubWorkspace(CubWorkspace&&) = delete;
        CubWorkspace& operator=(CubWorkspace&&) = delete;

    private:
        std::string operation_;
        size_t bytes_ = 0;
        DeviceBuffer buffer_;
    };

} // namespace lfs::training::cuda_scratch

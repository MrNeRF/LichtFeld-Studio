/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/assert.hpp"
#include "diagnostics/vram_profiler.hpp"

#include <cuda_runtime.h>
#include <format>
#include <limits>
#include <string>
#include <string_view>

namespace lfs::training::cuda_scratch {

    inline void check_status(const cudaError_t status, const std::string_view operation) {
        if (status == cudaSuccess) {
            return;
        }
        const std::string message = std::format(
            "{} failed: {} ({})", operation, cudaGetErrorString(status), cudaGetErrorName(status));
        cudaGetLastError();
        LFS_ASSERT_MSG(status == cudaSuccess, message);
    }

    inline size_t checked_bytes(const size_t count,
                                const size_t element_size,
                                const std::string_view allocation) {
        LFS_ASSERT_MSG(
            count == 0 || element_size <= std::numeric_limits<size_t>::max() / count,
            std::format("{} size overflow: {} elements of {} bytes", allocation, count, element_size));
        return count * element_size;
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
            const cudaError_t status = cudaMallocAsync(&ptr, bytes, stream);
            constexpr auto method = diagnostics::VramAllocationMethod::Async;
#else
            const cudaError_t status = cudaMalloc(&ptr, bytes);
            constexpr auto method = diagnostics::VramAllocationMethod::Direct;
#endif
            check_status(status, std::format("CUDA allocation for '{}' ({} bytes)", label, bytes));
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
            check_status(query(nullptr, bytes_), std::format("{} workspace query", operation_));
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
            check_status(operation(buffer_.get(), bytes_), operation_);
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

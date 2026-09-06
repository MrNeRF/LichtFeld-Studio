/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/cuda_error.hpp"
#include "core/tensor/backend/gpu_backend_ops.hpp"

#include <cuda_runtime.h>
#include <memory>

namespace lfs::core {

    // RAII wrapper for short-lived CUDA device temps (shape/stride metadata,
    // small op scratch). Routes through CudaMemoryPool so classic-heap
    // cudaMalloc fragmentation is avoided and allocations stay tracked.
    template <typename T>
    class CudaDeviceMemory {
    private:
        T* ptr_ = nullptr;
        size_t size_ = 0;
        cudaStream_t stream_ = nullptr;

        void free_owned() {
            if (ptr_) {
                internal::backend_ops(GpuBackend::CUDA).deallocate(internal::StorageRef{
                                                                       .backend = GpuBackend::CUDA,
                                                                       .data = ptr_,
                                                                       .byte_offset = 0,
                                                                       .dtype = DataType::UInt8,
                                                                       .meta = nullptr,
                                                                   },
                                                                   internal::ExecContext{stream_});
                ptr_ = nullptr;
                size_ = 0;
            }
        }

    public:
        CudaDeviceMemory() = default;

        explicit CudaDeviceMemory(size_t count, cudaStream_t stream = nullptr)
            : size_(count),
              stream_(stream) {
            if (count > 0) {
                const internal::StorageRef storage =
                    internal::backend_ops(GpuBackend::CUDA).allocate(count * sizeof(T), alignof(T), internal::ExecContext{stream});
                ptr_ = static_cast<T*>(storage.data);
                if (!ptr_) {
                    ensure_cuda_success(
                        cudaErrorMemoryAllocation, "CudaMemoryPool(CudaDeviceMemory)",
                        detail::format_cuda_safe(
                            "element_count={}, element_bytes={}, requested_bytes={}",
                            count, sizeof(T), count * sizeof(T)),
                        LFS_SOURCE_SITE_CURRENT(), CudaFailureDisposition::LogOnly);
                    size_ = 0;
                }
            }
        }

        ~CudaDeviceMemory() {
            free_owned();
        }

        // Delete copy constructor and assignment
        CudaDeviceMemory(const CudaDeviceMemory&) = delete;
        CudaDeviceMemory& operator=(const CudaDeviceMemory&) = delete;

        // Move constructor
        CudaDeviceMemory(CudaDeviceMemory&& other) noexcept
            : ptr_(other.ptr_),
              size_(other.size_),
              stream_(other.stream_) {
            other.ptr_ = nullptr;
            other.size_ = 0;
            other.stream_ = nullptr;
        }

        // Move assignment
        CudaDeviceMemory& operator=(CudaDeviceMemory&& other) noexcept {
            if (this != &other) {
                free_owned();
                ptr_ = other.ptr_;
                size_ = other.size_;
                stream_ = other.stream_;
                other.ptr_ = nullptr;
                other.size_ = 0;
                other.stream_ = nullptr;
            }
            return *this;
        }

        T* get() { return ptr_; }
        const T* get() const { return ptr_; }
        size_t size() const { return size_; }
        bool valid() const { return ptr_ != nullptr; }

        T* release() {
            T* tmp = ptr_;
            ptr_ = nullptr;
            size_ = 0;
            return tmp;
        }

        void reset(T* ptr = nullptr, size_t size = 0, cudaStream_t stream = nullptr) {
            if (ptr_ && ptr_ != ptr) {
                free_owned();
            }
            ptr_ = ptr;
            size_ = size;
            stream_ = stream;
        }

        // Copy data from host
        cudaError_t copy_from_host(const T* host_ptr, size_t count) {
            if (!ptr_ || count > size_) {
                return cudaErrorInvalidValue;
            }
            internal::backend_ops(GpuBackend::CUDA).copy_host_to_device(internal::CopyRequest{
                .src = internal::StorageRef{
                    .backend = GpuBackend::CUDA,
                    .data = const_cast<T*>(host_ptr),
                    .byte_offset = 0,
                    .dtype = DataType::UInt8,
                    .meta = nullptr,
                    .flags = internal::STORAGE_REF_HOST_MEMORY,
                },
                .dst = internal::StorageRef{
                    .backend = GpuBackend::CUDA,
                    .data = ptr_,
                    .byte_offset = 0,
                    .dtype = DataType::UInt8,
                    .meta = nullptr,
                },
                .bytes = count * sizeof(T),
                .synchronous = true,
                .context = internal::ExecContext{stream_},
            });
            return cudaSuccess;
        }

        // Copy data to host
        cudaError_t copy_to_host(T* host_ptr, size_t count) const {
            if (!ptr_ || count > size_) {
                return cudaErrorInvalidValue;
            }
            internal::backend_ops(GpuBackend::CUDA).copy_device_to_host(internal::CopyRequest{
                .src = internal::StorageRef{
                    .backend = GpuBackend::CUDA,
                    .data = ptr_,
                    .byte_offset = 0,
                    .dtype = DataType::UInt8,
                    .meta = nullptr,
                },
                .dst = internal::StorageRef{
                    .backend = GpuBackend::CUDA,
                    .data = host_ptr,
                    .byte_offset = 0,
                    .dtype = DataType::UInt8,
                    .meta = nullptr,
                    .flags = internal::STORAGE_REF_HOST_MEMORY,
                },
                .bytes = count * sizeof(T),
                .synchronous = true,
                .context = internal::ExecContext{stream_},
            });
            return cudaSuccess;
        }
    };

    // Helper function to allocate multiple device arrays at once
    template <typename... Args>
    bool cuda_multi_malloc(Args&... args) {
        return ((args.valid()) && ...);
    }

} // namespace lfs::core

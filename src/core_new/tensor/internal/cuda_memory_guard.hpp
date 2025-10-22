/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cuda_runtime.h>
#include <memory>

namespace lfs::core {

    // RAII wrapper for CUDA device memory
    template <typename T>
    class CudaDeviceMemory {
    private:
        T* ptr_ = nullptr;
        size_t size_ = 0;

    public:
        CudaDeviceMemory() = default;

        explicit CudaDeviceMemory(size_t count) : size_(count) {
            if (count > 0) {
                cudaError_t err = cudaMalloc(&ptr_, count * sizeof(T));
                if (err != cudaSuccess) {
                    ptr_ = nullptr;
                    size_ = 0;
                }
            }
        }

        ~CudaDeviceMemory() {
            if (ptr_) {
                cudaFree(ptr_);
            }
        }

        // Delete copy constructor and assignment
        CudaDeviceMemory(const CudaDeviceMemory&) = delete;
        CudaDeviceMemory& operator=(const CudaDeviceMemory&) = delete;

        // Move constructor
        CudaDeviceMemory(CudaDeviceMemory&& other) noexcept
            : ptr_(other.ptr_),
              size_(other.size_) {
            other.ptr_ = nullptr;
            other.size_ = 0;
        }

        // Move assignment
        CudaDeviceMemory& operator=(CudaDeviceMemory&& other) noexcept {
            if (this != &other) {
                if (ptr_) {
                    cudaFree(ptr_);
                }
                ptr_ = other.ptr_;
                size_ = other.size_;
                other.ptr_ = nullptr;
                other.size_ = 0;
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

        void reset(T* ptr = nullptr, size_t size = 0) {
            if (ptr_ && ptr_ != ptr) {
                cudaFree(ptr_);
            }
            ptr_ = ptr;
            size_ = size;
        }

        // Copy data from host
        cudaError_t copy_from_host(const T* host_ptr, size_t count) {
            if (!ptr_ || count > size_) {
                return cudaErrorInvalidValue;
            }
            return cudaMemcpy(ptr_, host_ptr, count * sizeof(T), cudaMemcpyHostToDevice);
        }

        // Copy data to host
        cudaError_t copy_to_host(T* host_ptr, size_t count) const {
            if (!ptr_ || count > size_) {
                return cudaErrorInvalidValue;
            }
            return cudaMemcpy(host_ptr, ptr_, count * sizeof(T), cudaMemcpyDeviceToHost);
        }
    };

    // Helper function to allocate multiple device arrays at once
    template <typename... Args>
    bool cuda_multi_malloc(Args&... args) {
        return ((args.valid()) && ...);
    }

} // namespace lfs::core

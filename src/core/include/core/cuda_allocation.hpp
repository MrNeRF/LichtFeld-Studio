/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/assert.hpp"
#include "core/cuda_error.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace lfs::core {

    struct DirectCudaAllocator {
        [[nodiscard]] void* allocate(const size_t bytes,
                                     cudaStream_t,
                                     const std::string_view label) const {
            void* ptr = nullptr;
            LFS_CUDA_CHECK_MSG(cudaMalloc(&ptr, bytes),
                               "CUDA allocation '{}' ({} bytes)", label, bytes);
            return ptr;
        }

        void deallocate(void* ptr, cudaStream_t) const noexcept {
            const cudaError_t status = cudaFree(ptr);
            if (status != cudaSuccess) {
                ensure_cuda_success(
                    status, "direct CUDA allocation free", {},
                    LFS_SOURCE_SITE_CURRENT(), CudaFailureDisposition::LogOnlyNoLatch);
                cudaGetLastError();
            }
        }
    };

    struct StreamOrderedCudaAllocator {
        [[nodiscard]] void* allocate(const size_t bytes,
                                     const cudaStream_t stream,
                                     const std::string_view label) const {
            void* ptr = nullptr;
#if CUDART_VERSION >= 11020
            LFS_CUDA_CHECK_MSG(cudaMallocAsync(&ptr, bytes, stream),
                               "stream-ordered CUDA allocation '{}' ({} bytes)", label, bytes);
#else
            LFS_CUDA_CHECK_MSG(cudaMalloc(&ptr, bytes),
                               "CUDA allocation '{}' ({} bytes)", label, bytes);
#endif
            return ptr;
        }

        void deallocate(void* ptr, const cudaStream_t stream) const noexcept {
#if CUDART_VERSION >= 11020
            const cudaError_t status = cudaFreeAsync(ptr, stream);
#else
            const cudaError_t status = cudaFree(ptr);
#endif
            if (status != cudaSuccess) {
                ensure_cuda_success(
                    status, "stream-ordered CUDA allocation free", {},
                    LFS_SOURCE_SITE_CURRENT(), CudaFailureDisposition::LogOnlyNoLatch);
                cudaGetLastError();
            }
        }
    };

    struct NoCudaAllocationHooks {
        void before_allocate(std::string_view) const noexcept {}
        void after_allocate(void*, size_t, std::string_view) const noexcept {}
        void before_deallocate(void*) const noexcept {}
    };

    template <typename Allocator, typename Hooks = NoCudaAllocationHooks>
    class UniqueCudaAllocation {
    public:
        UniqueCudaAllocation() = default;

        UniqueCudaAllocation(const size_t bytes,
                             const cudaStream_t stream,
                             const std::string_view label)
            : UniqueCudaAllocation(bytes, stream, label, Allocator{}, Hooks{}) {}

        UniqueCudaAllocation(const size_t bytes,
                             const cudaStream_t stream,
                             const std::string_view label,
                             Allocator allocator)
            : UniqueCudaAllocation(bytes, stream, label, std::move(allocator), Hooks{}) {}

        UniqueCudaAllocation(const size_t bytes,
                             const cudaStream_t stream,
                             const std::string_view label,
                             Allocator allocator,
                             Hooks hooks)
            : allocator_(std::move(allocator)),
              hooks_(std::move(hooks)) {
            allocate(bytes, stream, label);
        }

        ~UniqueCudaAllocation() {
            reset();
        }

        UniqueCudaAllocation(const UniqueCudaAllocation&) = delete;
        UniqueCudaAllocation& operator=(const UniqueCudaAllocation&) = delete;

        UniqueCudaAllocation(UniqueCudaAllocation&& other) noexcept(
            std::is_nothrow_move_constructible_v<Allocator>&&
                std::is_nothrow_move_constructible_v<Hooks>)
            : ptr_(std::exchange(other.ptr_, nullptr)),
              bytes_(std::exchange(other.bytes_, 0)),
              stream_(std::exchange(other.stream_, nullptr)),
              allocator_(std::move(other.allocator_)),
              hooks_(std::move(other.hooks_)) {}

        UniqueCudaAllocation& operator=(UniqueCudaAllocation&& other) noexcept(
            std::is_nothrow_move_assignable_v<Allocator>&&
                std::is_nothrow_move_assignable_v<Hooks>) {
            if (this != &other) {
                reset();
                allocator_ = std::move(other.allocator_);
                hooks_ = std::move(other.hooks_);
                ptr_ = std::exchange(other.ptr_, nullptr);
                bytes_ = std::exchange(other.bytes_, 0);
                stream_ = std::exchange(other.stream_, nullptr);
            }
            return *this;
        }

        void allocate(const size_t bytes,
                      const cudaStream_t stream,
                      const std::string_view label) {
            LFS_ASSERT_MSG(ptr_ == nullptr, "CUDA allocation already owns memory");
            LFS_ASSERT_MSG(bytes > 0, "CUDA allocation requires a nonzero size");

            hooks_.before_allocate(label);
            ptr_ = allocator_.allocate(bytes, stream, label);
            LFS_ASSERT_MSG(
                ptr_ != nullptr,
                ::lfs::core::detail::format_cuda_safe("CUDA allocation for '{}' returned null ({} bytes)", label, bytes));
            bytes_ = bytes;
            stream_ = stream;
            try {
                hooks_.after_allocate(ptr_, bytes_, label);
            } catch (...) {
                allocator_.deallocate(ptr_, stream_);
                ptr_ = nullptr;
                bytes_ = 0;
                stream_ = nullptr;
                throw;
            }
        }

        void reset() noexcept {
            if (!ptr_) {
                return;
            }
            hooks_.before_deallocate(ptr_);
            allocator_.deallocate(ptr_, stream_);
            ptr_ = nullptr;
            bytes_ = 0;
            stream_ = nullptr;
        }

        [[nodiscard]] void* release() noexcept {
            void* ptr = std::exchange(ptr_, nullptr);
            bytes_ = 0;
            stream_ = nullptr;
            return ptr;
        }

        void swap(UniqueCudaAllocation& other) noexcept(
            std::is_nothrow_swappable_v<Allocator>&&
                std::is_nothrow_swappable_v<Hooks>) {
            using std::swap;
            swap(ptr_, other.ptr_);
            swap(bytes_, other.bytes_);
            swap(stream_, other.stream_);
            swap(allocator_, other.allocator_);
            swap(hooks_, other.hooks_);
        }

        [[nodiscard]] void* get() const noexcept { return ptr_; }

        template <typename T>
        [[nodiscard]] T* as() const noexcept {
            return static_cast<T*>(ptr_);
        }

        [[nodiscard]] size_t size() const noexcept { return bytes_; }
        [[nodiscard]] cudaStream_t stream() const noexcept { return stream_; }
        [[nodiscard]] explicit operator bool() const noexcept { return ptr_ != nullptr; }

    private:
        void* ptr_ = nullptr;
        size_t bytes_ = 0;
        cudaStream_t stream_ = nullptr;
        [[no_unique_address]] Allocator allocator_;
        [[no_unique_address]] Hooks hooks_;
    };

    struct DefaultCudaCubWorkspaceTraits {
        static constexpr std::string_view allocation_label = "cuda.cub_workspace";
        static constexpr std::string_view diagnostic_scope = "CUDA";
    };

    template <typename Allocation, typename Traits = DefaultCudaCubWorkspaceTraits>
    class CudaCubWorkspace {
    public:
        template <typename Query>
        CudaCubWorkspace(const std::string_view operation,
                         const cudaStream_t stream,
                         Query&& query)
            : operation_(operation) {
            LFS_CUDA_CHECK_MSG(query(nullptr, bytes_),
                               "{} workspace query", operation_);
            LFS_ASSERT_MSG(
                bytes_ > 0,
                std::string(operation_) + " returned an empty workspace for a nonempty operation");
            allocation_.allocate(bytes_, stream, Traits::allocation_label);
        }

        template <typename Operation>
        void run(Operation&& operation) {
            LFS_CUDA_CHECK_MSG(operation(allocation_.get(), bytes_),
                               "{} CUB workspace operation: {}",
                               Traits::diagnostic_scope, operation_);
        }

        CudaCubWorkspace(const CudaCubWorkspace&) = delete;
        CudaCubWorkspace& operator=(const CudaCubWorkspace&) = delete;
        CudaCubWorkspace(CudaCubWorkspace&&) = delete;
        CudaCubWorkspace& operator=(CudaCubWorkspace&&) = delete;

    private:
        std::string operation_;
        size_t bytes_ = 0;
        Allocation allocation_;
    };

    template <typename Workspace, typename Operation>
    void run_cub_operation(const std::string_view name,
                           const cudaStream_t stream,
                           Operation&& operation) {
        Workspace workspace(name, stream, operation);
        workspace.run(operation);
    }

} // namespace lfs::core

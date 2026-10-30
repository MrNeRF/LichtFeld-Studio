/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "../../internal/tensor_impl.hpp"
#include "../native_pressure.hpp"
#include "core/alloc_counter.hpp"
#include "core/cuda_error.hpp"
#include "core/pinned_memory_allocator.hpp"
#include "runtime/memory_pool.hpp"
#include <cuda_runtime.h>
#include <format>
namespace lfs::core {
    namespace {

        [[noreturn]] void throw_cuda_unavailable_allocation(
            const size_t bytes,
            const cudaStream_t stream,
            const char* const label,
            const char* const operation) {
            throw MemoryAllocationError(AllocationFailure{
                .domain = MemoryDomain::CudaDevice,
                .requested_bytes = bytes,
                .alignment = 0,
                .device = -1,
                .stream = reinterpret_cast<uintptr_t>(stream),
                .label = label,
                .operation = operation,
                .native_error = cudaErrorInitializationError,
            });
        }

        void* try_allocate_direct_cuda_storage(const size_t bytes,
                                               cudaError_t* const failure_status) {
            void* ptr = nullptr;
            const auto pre_call_state = sample_cuda_pre_call_state();
            const cudaError_t status = cudaMalloc(&ptr, bytes);
            if (status == cudaSuccess) {
                alloc_counter::record_site(alloc_counter::Site::ZerosDirect);
                return ptr;
            }
            if (failure_status) {
                *failure_status = status;
            }
            ensure_cuda_success(
                status, pre_call_state, "cudaMalloc(direct tensor storage)",
                std::format("requested_bytes={}", bytes),
                LFS_SOURCE_SITE_CURRENT(), CudaFailureDisposition::LogOnly);
            if (ptr != nullptr) {
                const cudaError_t cleanup_status = cudaFree(ptr);
                if (cleanup_status != cudaSuccess) {
                    ensure_cuda_success(
                        cleanup_status, "cudaFree(failed direct tensor storage allocation)", {},
                        LFS_SOURCE_SITE_CURRENT(), CudaFailureDisposition::LogOnlyNoLatch);
                }
            }
            return nullptr;
        }

        [[noreturn]] void throw_non_oom_cuda_allocation(
            const cudaError_t status,
            const size_t bytes,
            const char* const label,
            const char* const operation) {
            if (status == cudaSuccess) {
                throw std::runtime_error(std::format(
                    "CUDA allocator returned no storage without a CUDA error: "
                    "requested_bytes={}, label='{}', operation='{}'",
                    bytes, label ? label : "", operation ? operation : ""));
            }
            throw std::runtime_error(std::format(
                "CUDA allocation failed with {} ({}): requested_bytes={}, label='{}', "
                "operation='{}'",
                cudaGetErrorName(status), cudaGetErrorString(status), bytes,
                label ? label : "", operation ? operation : ""));
        }

    } // namespace
    void* allocate_cuda_storage(const size_t bytes,
                                const cudaStream_t stream,
                                const CudaStorageMode mode,
                                const char* const label,
                                const char* const operation) {

        if (bytes == 0) {
            return nullptr;
        }
        if (cuda_is_unavailable()) [[unlikely]] {
            throw_cuda_unavailable_allocation(bytes, stream, label, operation);
        }

        auto& coordinator = MemoryPressureCoordinator::instance();
        const auto try_allocate = [&](cudaError_t* const failure_status) -> void* {
            *failure_status = cudaSuccess;
            if (coordinator.probe_should_fail(MemoryDomain::CudaDevice, bytes)) {
                *failure_status = cudaErrorMemoryAllocation;
                return nullptr;
            }
            if (mode == CudaStorageMode::Direct) {
                return try_allocate_direct_cuda_storage(bytes, failure_status);
            }
            if (mode == CudaStorageMode::ExactAsync) {
                return CudaMemoryPool::instance().try_allocate_exact_async(
                    bytes, stream, failure_status);
            }
            return CudaMemoryPool::instance().try_allocate(bytes, stream, failure_status);
        };

        cudaError_t failure_status = cudaSuccess;
        void* ptr = try_allocate(&failure_status);
        if (ptr != nullptr) {
            return ptr;
        }
        if (cuda_is_unavailable()) {
            throw_cuda_unavailable_allocation(bytes, stream, label, operation);
        }
        if (failure_status != cudaErrorMemoryAllocation) {
            throw_non_oom_cuda_allocation(failure_status, bytes, label, operation);
        }

        int device = 0;
        cudaGetDevice(&device);
        const AllocationFailure failure{
            .domain = MemoryDomain::CudaDevice,
            .requested_bytes = bytes,
            .alignment = 0,
            .device = device,
            .stream = reinterpret_cast<uintptr_t>(stream),
            .label = label,
            .operation = operation,
            .native_error = cudaErrorMemoryAllocation,
        };
        if (coordinator.relieve_and_should_retry(failure)) {
            ptr = try_allocate(&failure_status);
        }
        // A handled OOM and pressure-client reclaim can leave CUDA's last-error
        // state sticky. Consume it before returning control to later CUDA work.
        (void)cudaGetLastError();
        if (ptr == nullptr) {
            if (cuda_is_unavailable()) {
                throw_cuda_unavailable_allocation(bytes, stream, label, operation);
            }
            if (failure_status != cudaErrorMemoryAllocation) {
                throw_non_oom_cuda_allocation(failure_status, bytes, label, operation);
            }
            throw MemoryAllocationError(failure);
        }
        return ptr;
    }

} // namespace lfs::core
namespace lfs::core::internal {
    MemoryInfo cuda_pressure_memory_info(bool free_query) {
        if (cuda_is_unavailable())
            return {};
        if (free_query && cudaPeekAtLastError() == cudaErrorMemoryAllocation)
            (void)cudaGetLastError();
        size_t free_bytes = 0, total_bytes = 0;
        const auto status = cudaMemGetInfo(&free_bytes, &total_bytes);
        if (status != cudaSuccess) {
            if (free_query && status == cudaErrorMemoryAllocation)
                (void)cudaGetLastError();
            return {};
        }
        MemoryInfo result;
        result.free_bytes = free_bytes;
        result.total_bytes = total_bytes;
        return result;
    }
    void register_cuda_pressure_clients(MemoryPressureCoordinator& coordinator) {

        coordinator.register_client(PressureClient{
            .name = "tensor-cuda-pool",
            .priority = 10,
            .domain = MemoryDomain::CudaDevice,
            .affinity = PressureAffinity::ImmediateThreadSafe,
            .estimate = nullptr,
            .shrink = [](const PressureRequest&) {
                CudaMemoryPool::instance().trim_cached_memory();
                return ReclaimResult{};
            },
        });
        coordinator.register_client(PressureClient{
            .name = "pinned-host-cache",
            .priority = 10,
            .domain = MemoryDomain::PinnedHost,
            .affinity = PressureAffinity::ImmediateThreadSafe,
            .estimate = [](const PressureRequest&) -> size_t {
                return PinnedMemoryAllocator::instance().get_stats().cached_bytes;
            },
            .shrink = [](const PressureRequest&) {
                auto& allocator = PinnedMemoryAllocator::instance();
                const size_t before = allocator.get_stats().cached_bytes;
                allocator.empty_cache();
                return ReclaimResult{.logical_bytes_released = before}; },
        });
    }
    void record_pressure_breadcrumb() { record_cuda_breadcrumb("memory-pressure.episode", __FILE__, __LINE__); }
} // namespace lfs::core::internal

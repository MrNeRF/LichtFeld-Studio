/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "../facade_trace.hpp"
#include "../gpu_backend_ops.hpp"

#include "../../internal/tensor_impl.hpp"
#include "core/assert.hpp"
#include "core/cuda_error.hpp"
#include "core/pinned_memory_allocator.hpp"
#include "core/tensor/backend/cuda/runtime/cuda_event_pool.hpp"
#include "core/tensor/backend/cuda/runtime/memory_pool.hpp"

#include <atomic>
#include <cuda_runtime.h>
#include <string>
#include <vector>

namespace lfs::core {
    namespace {
        std::atomic<CudaMemoryPool*> g_cuda_memory_pool_instance{nullptr};
        thread_local std::string g_pool_pending_label;
    } // namespace

    CudaMemoryPool* try_live_cuda_memory_pool() noexcept {
        return g_cuda_memory_pool_instance.load(std::memory_order_acquire);
    }

    void safe_cuda_pool_deallocate(void* const pointer, const cudaStream_t stream) noexcept {
        if (pointer == nullptr) {
            return;
        }
        if (CudaMemoryPool* const pool = try_live_cuda_memory_pool()) {
            pool->deallocate(pointer, stream);
        }
    }

    CudaMemoryPool& CudaMemoryPool::instance() {
        static_cast<void>(lfs::diagnostics::VramProfiler::instance());
        static_cast<void>(CudaEventPool::instance());
        static_cast<void>(GPUSlabAllocator::instance());
        static_cast<void>(SizeBucketedPool::instance());
        static CudaMemoryPool pool;
        g_cuda_memory_pool_instance.store(&pool, std::memory_order_release);
        return pool;
    }

    CudaMemoryPool::LabelGuard::LabelGuard(std::string_view label)
        : previous_(std::move(g_pool_pending_label)),
          active_(!label.empty()) {
        if (active_) {
            g_pool_pending_label.assign(label);
        }
    }

    CudaMemoryPool::LabelGuard::~LabelGuard() {
        g_pool_pending_label = std::move(previous_);
    }

    std::string_view CudaMemoryPool::current_label() noexcept {
        return g_pool_pending_label;
    }

    namespace internal {
        namespace {
            void* cuda_address(const StorageRef storage) {
                LFS_ASSERT_MSG(storage.backend == GpuBackend::CUDA,
                               "CUDA service received non-CUDA storage");
                return static_cast<unsigned char*>(storage.data) + storage.byte_offset;
            }

            const void* cuda_const_address(const StorageRef storage) {
                return cuda_address(storage);
            }

            void require_host(const StorageRef storage, const char* const role) {
                LFS_ASSERT_MSG((storage.flags & STORAGE_REF_HOST_MEMORY) != 0,
                               std::string("CUDA copy expects host memory for the ") + role);
            }

            void require_device(const StorageRef storage, const char* const role) {
                LFS_ASSERT_MSG((storage.flags & STORAGE_REF_HOST_MEMORY) == 0,
                               std::string("CUDA copy expects device memory for the ") + role);
            }

            void prime_stream_polling(const cudaStream_t stream, const char* const operation) {
                // Runtime scheduling side effect, not useful work: a query on the stream
                // keeps the following blocking copy on the CUDA runtime's active polling
                // path. Measured on driver 580.173 (RTX 4090): the seven-element gather
                // row, whose index validation reads seven indices back synchronously,
                // takes 20.4 us instead of 12.8 us without it once the per-call pointer
                // query of storage_ptr() is gone. Same mechanism as the scalar
                // reductions in cuda_ops_reduce.cpp; re-measure before removing.
                const cudaError_t status = cudaStreamQuery(stream);
                if (status != cudaSuccess && status != cudaErrorNotReady) {
                    LFS_CUDA_CHECK_MSG_STREAM(status, stream,
                                              "{} (stream query before the synchronous copy)",
                                              operation);
                }
            }

            void copy_cuda(const CopyRequest& request, const cudaMemcpyKind kind) {
                if (request.bytes == 0) {
                    return;
                }
                switch (kind) {
                case cudaMemcpyHostToDevice:
                    require_host(request.src, "source");
                    require_device(request.dst, "destination");
                    break;
                case cudaMemcpyDeviceToHost:
                    require_device(request.src, "source");
                    require_host(request.dst, "destination");
                    break;
                default:
                    require_device(request.src, "source");
                    require_device(request.dst, "destination");
                    break;
                }
                void* const destination = cuda_address(request.dst);
                const void* const source = cuda_const_address(request.src);
                if (request.synchronous && kind == cudaMemcpyDeviceToHost) {
                    prime_stream_polling(request.context.cuda_stream, request.operation);
                }
                if (request.synchronous && request.context.cuda_stream == nullptr) {
                    LFS_CUDA_CHECK_MSG_ARGS(
                        cudaMemcpy(destination, source, request.bytes, kind),
                        reinterpret_cast<uintptr_t>(destination),
                        reinterpret_cast<uintptr_t>(source), request.bytes,
                        "{} (kind={}, dtype={})", request.operation, static_cast<int>(kind),
                        dtype_name(request.src.dtype));
                    return;
                }
                LFS_CUDA_CHECK_MSG_STREAM_ARGS(
                    cudaMemcpyAsync(destination, source, request.bytes, kind,
                                    request.context.cuda_stream),
                    request.context.cuda_stream,
                    reinterpret_cast<uintptr_t>(destination),
                    reinterpret_cast<uintptr_t>(source), request.bytes,
                    "{} (kind={}, dtype={})", request.operation, static_cast<int>(kind),
                    dtype_name(request.src.dtype));
                if (request.synchronous) {
                    LFS_CUDA_CHECK_MSG_STREAM(
                        cudaStreamSynchronize(request.context.cuda_stream),
                        request.context.cuda_stream, "{} synchronize", request.operation);
                }
            }
        } // namespace

        StorageRef CudaBackendOps::allocate(
            const size_t bytes, const size_t alignment, const ExecContext context) {
            LFS_FACADE_TRACE(service_allocate);
            LFS_ASSERT_MSG(alignment == 0 || (alignment & (alignment - 1)) == 0,
                           "CUDA allocation alignment must be zero or a power of two");
            (void)alignment;
            const bool direct = context.allocation_class == AllocationClass::Direct;
            return StorageRef{
                .backend = GpuBackend::CUDA,
                .data = allocate_cuda_storage(
                    bytes, context.cuda_stream,
                    direct ? CudaStorageMode::Direct : CudaStorageMode::Pooled,
                    context.allocation_label, context.allocation_operation),
                .byte_offset = 0,
                .dtype = DataType::UInt8,
                .meta = nullptr,
                .flags = direct ? STORAGE_REF_DIRECT_ALLOCATION : 0,
            };
        }

        void CudaBackendOps::deallocate(
            const StorageRef storage, const ExecContext context) noexcept {
            if ((storage.flags & STORAGE_REF_DIRECT_ALLOCATION) != 0) {
                if (storage.data != nullptr) {
                    const cudaError_t status = cudaFree(storage.data);
                    if (status != cudaSuccess) {
                        ensure_cuda_success(
                            status, "CUDA backend direct deallocation", {},
                            LFS_SOURCE_SITE_CURRENT(),
                            CudaFailureDisposition::LogOnlyNoLatch);
                    }
                }
                return;
            }
            safe_cuda_pool_deallocate(storage.data, context.cuda_stream);
        }

        void CudaBackendOps::record_stream(
            const StorageRef storage, const ExecContext context) {
            CudaMemoryPool::instance().record_stream(storage.data, context.cuda_stream);
        }

        void CudaBackendOps::release_stream(const ExecContext context) {
            CudaMemoryPool::instance().release_stream(context.cuda_stream);
        }

        void CudaBackendOps::rehome_stream(
            const StorageRef storage, const ExecContext context) {
            CudaMemoryPool::instance().rehome_stream(storage.data, context.cuda_stream);
        }

        void CudaBackendOps::trim() {
            CudaMemoryPool::instance().trim_cached_memory();
        }

        void CudaBackendOps::trim_if_reserved_unused_exceeds(
            const size_t threshold_bytes) {
            CudaMemoryPool::instance().trim_cached_memory_if_reserved_unused_exceeds(
                threshold_bytes);
        }

        MemoryInfo CudaBackendOps::stats() {
            MemoryInfo result;
            LFS_CUDA_CHECK(cudaMemGetInfo(&result.free_bytes, &result.total_bytes));
            result.allocated_bytes = result.total_bytes - result.free_bytes;
            result.device_id = 0;
            return result;
        }

        void CudaBackendOps::shutdown() {
            if (CudaMemoryPool* const pool =
                    g_cuda_memory_pool_instance.exchange(nullptr, std::memory_order_acq_rel)) {
                pool->shutdown();
            }
        }

        void CudaBackendOps::set_allocation_iteration(const int iteration) {
            CudaMemoryPool::instance().set_iteration(iteration);
        }

        void CudaBackendOps::record_tensor_allocation(
            const StorageRef storage, const StridedLayout& layout, const size_t bytes) {
            if constexpr (LFS_ALLOCATION_PROFILING_ENABLED) {
                std::vector<size_t> shape(layout.dims.begin(), layout.dims.begin() + layout.rank);
                CudaMemoryPool::instance().record_tensor(
                    storage.data, shape, bytes, dtype_name(storage.dtype));
            } else {
                (void)storage;
                (void)layout;
                (void)bytes;
            }
        }

        void CudaBackendOps::copy_host_to_device(const CopyRequest& request) {
            LFS_FACADE_TRACE(service_copy_host_to_device);
            copy_cuda(request, cudaMemcpyHostToDevice);
        }

        void CudaBackendOps::copy_device_to_host(const CopyRequest& request) {
            LFS_FACADE_TRACE(service_copy_device_to_host);
            copy_cuda(request, cudaMemcpyDeviceToHost);
        }

        void CudaBackendOps::copy_device_to_device(const CopyRequest& request) {
            LFS_FACADE_TRACE(service_copy_device_to_device);
            copy_cuda(request, cudaMemcpyDeviceToDevice);
        }

        void CudaBackendOps::memset(const FillRequest& request) {
            LFS_FACADE_TRACE(service_memset);
            if (request.bytes == 0) {
                return;
            }
            void* const destination = cuda_address(request.dst);
            if (request.synchronous && request.context.cuda_stream == nullptr) {
                LFS_CUDA_CHECK_MSG_ARGS(
                    cudaMemset(destination, request.value, request.bytes),
                    reinterpret_cast<uintptr_t>(destination), request.value, request.bytes,
                    "{}", request.operation);
                return;
            }
            LFS_CUDA_CHECK_MSG_STREAM_ARGS(
                cudaMemsetAsync(destination, request.value, request.bytes,
                                request.context.cuda_stream),
                request.context.cuda_stream,
                reinterpret_cast<uintptr_t>(destination), request.value, request.bytes,
                "{}", request.operation);
            if (request.synchronous) {
                LFS_CUDA_CHECK_MSG_STREAM(
                    cudaStreamSynchronize(request.context.cuda_stream),
                    request.context.cuda_stream, "{} synchronize", request.operation);
            }
        }

        void CudaBackendOps::synchronize_stream(const ExecContext context) {
            LFS_FACADE_TRACE(service_synchronize_stream);
            LFS_CUDA_CHECK(cudaStreamSynchronize(context.cuda_stream));
        }

        void CudaBackendOps::synchronize_device() {
            LFS_CUDA_CHECK(cudaDeviceSynchronize());
        }

        void CudaBackendOps::wait_for(const SyncToken token) {
            LFS_ASSERT_MSG(token.backend == GpuBackend::CUDA,
                           "CUDA sync service received a non-CUDA token");
            LFS_CUDA_CHECK(cudaStreamSynchronize(
                reinterpret_cast<cudaStream_t>(token.native)));
        }

        SyncToken CudaBackendOps::bridge(
            const ExecContext producer, const ExecContext consumer) {
            bridgeStreams(producer.cuda_stream, consumer.cuda_stream);
            return SyncToken{
                .backend = GpuBackend::CUDA,
                .value = 0,
                .native = reinterpret_cast<uintptr_t>(consumer.cuda_stream),
            };
        }

        PointerClass CudaBackendOps::classify_pointer(const void* const pointer) {
            if (pointer == nullptr) {
                return PointerClass::Unknown;
            }
            cudaPointerAttributes attributes{};
            if (cudaPointerGetAttributes(&attributes, pointer) != cudaSuccess) {
                (void)cudaGetLastError();
                return PointerClass::Unknown;
            }
            if (attributes.type == cudaMemoryTypeDevice ||
                attributes.type == cudaMemoryTypeManaged) {
                return PointerClass::Device;
            }
            if (attributes.type == cudaMemoryTypeHost) {
                return PointerClass::Pinned;
            }
            if (attributes.type == cudaMemoryTypeUnregistered) {
                return PointerClass::Host;
            }
            return PointerClass::Unknown;
        }

        bool CudaBackendOps::stream_is_capturing(const ExecContext context) {
            cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
            const cudaError_t result = cudaStreamIsCapturing(context.cuda_stream, &status);
            if (result != cudaSuccess) {
                (void)cudaGetLastError();
                return true;
            }
            return status != cudaStreamCaptureStatusNone;
        }

    } // namespace internal
} // namespace lfs::core

/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "../../internal/expression_runtime.hpp"
#include "../facade_trace.hpp"
#include "../gpu_backend_ops.hpp"
#include "../readback_buffer.hpp"
#include "../tensor_completion.hpp"
#include "core/device_fault.hpp"

#include "../../internal/tensor_impl.hpp"
#include "core/assert.hpp"
#include "core/cuda_error.hpp"
#include "core/cuda_error_typed.hpp"
#include "core/cuda_vulkan_interop.hpp"
#include "core/gpu_device_info.hpp"
#include "core/pinned_memory_allocator.hpp"
#include "core/tensor/backend/cuda/runtime/cuda_event_pool.hpp"
#include "core/tensor/backend/cuda/runtime/cuda_stream_context.hpp"
#include "core/tensor/backend/cuda/runtime/memory_pool.hpp"
#include "core/tensor_cuda_interop.hpp"
#include "core/tensor_label.hpp"
#include "core/tensor_vulkan_interop.hpp"
#include "core/vulkan_helpers.hpp"
#include "kernels/where_scalar.cuh"
#include "runtime/size_bucketed_pool.hpp"

#include <atomic>
#include <cstring>
#include <cuda_runtime.h>
#include <string>
#include <vector>

namespace lfs::core {
    bool tensor_uses_cuda_storage(const Tensor& tensor) {
        return gpu_backend_of(tensor) == GpuBackend::CUDA;
    }

    TensorCudaStream::TensorCudaStream() {
        LFS_CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    }

    TensorCudaStream::~TensorCudaStream() {
        if (!stream_)
            return;
        LFS_CUDA_LOG_TEARDOWN(cudaStreamSynchronize(stream_), stream_, "tensor stream teardown: synchronize");
        release_cuda_stream(stream_);
        LFS_CUDA_LOG_TEARDOWN(cudaStreamDestroy(stream_), nullptr, "tensor stream teardown: destroy");
    }

    void trim_cuda_memory_pool() {
        CudaMemoryPool::instance().trim();
    }

    size_t cuda_allocation_size(size_t bytes) {
        return SizeBucketedPool::get_bucket_size(bytes);
    }

    void release_cuda_stream(cudaStream_t stream) {
        CudaMemoryPool::instance().release_stream(stream);
    }

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

    TensorLabelScope::TensorLabelScope(std::string_view label)
        : previous_(std::move(g_pool_pending_label)) {
        if (!label.empty())
            g_pool_pending_label.assign(label);
    }

    TensorLabelScope::~TensorLabelScope() {
        g_pool_pending_label = std::move(previous_);
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

            class CudaReadbackBuffer final : public ReadbackBuffer {
            public:
                ~CudaReadbackBuffer() override {
                    if (pending_) {
                        // Also covers an event-record failure after a copy was queued.
                        LFS_CUDA_LOG_TEARDOWN(cudaStreamSynchronize(stream_), stream_, "readback teardown: wait for copy");
                    }
                    if (host_)
                        PinnedMemoryAllocator::instance().deallocate(host_);
                }

                void enqueue(StorageRef source, const size_t bytes, ExecContext context) override {
                    LFS_FACADE_TRACE(service_enqueue_readback);
                    LFS_ASSERT_MSG(!pending_, "Readback staging is already in use");
                    require_device(source, "source");
                    if (capacity_ < bytes) {
                        void* replacement = PinnedMemoryAllocator::instance().allocate(bytes);
                        if (!replacement)
                            throw std::bad_alloc();
                        if (host_)
                            PinnedMemoryAllocator::instance().deallocate(host_);
                        host_ = replacement;
                        capacity_ = bytes;
                    }
                    bytes_ = bytes;
                    stream_ = context.cuda_stream;
                    // Keep the source alive and drain the stream even when CUDA reports
                    // a failure after accepting the asynchronous copy.
                    pending_ = true;
                    LFS_CUDA_CHECK(cudaMemcpyAsync(host_, cuda_const_address(source), bytes,
                                                   cudaMemcpyDeviceToHost, stream_));
                    completion_ = TensorCompletionAccess::cuda(stream_);
                }

                void wait() override {
                    completion_.wait();
                }

                bool poll(void* destination) override {
                    LFS_FACADE_TRACE(service_readback_poll);
                    if (!completion_.ready())
                        return false;
                    device_fault_registry_consume_or_throw(LFS_SOURCE_SITE_CURRENT(), false);
                    if (destination)
                        std::memcpy(destination, host_, bytes_);
                    pending_ = false;
                    return true;
                }

            private:
                void* host_ = nullptr;
                TensorCompletion completion_;
                cudaStream_t stream_ = nullptr;
                size_t capacity_ = 0;
                size_t bytes_ = 0;
                bool pending_ = false;
            };

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
            synchronize_and_unload_cuda_expressions([] { CudaMemoryPool::instance().trim_cached_memory(); });
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
            if (request.synchronous && request.bytes != 0)
                device_fault_registry_consume_or_throw(LFS_SOURCE_SITE_CURRENT(), false);
        }

        void CudaBackendOps::copy_device_to_device(const CopyRequest& request) {
            LFS_FACADE_TRACE(service_copy_device_to_device);
            copy_cuda(request, cudaMemcpyDeviceToDevice);
        }

        std::unique_ptr<ReadbackBuffer> CudaBackendOps::create_readback_buffer() {
            return std::make_unique<CudaReadbackBuffer>();
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
            device_fault_registry_consume_or_throw(LFS_SOURCE_SITE_CURRENT(), false);
        }

        void CudaBackendOps::synchronize_device() {
            synchronize_and_unload_cuda_expressions([] { LFS_CUDA_CHECK(cudaDeviceSynchronize()); });
            device_fault_registry_consume_or_throw(LFS_SOURCE_SITE_CURRENT(), true);
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

    std::optional<GpuDeviceInfo> gpu_backend_device_info(const GpuBackend backend, const int device_index) {
        if (backend != GpuBackend::CUDA || device_index < 0)
            return std::nullopt;
        cudaDeviceProp properties{};
        if (cudaGetDeviceProperties(&properties, device_index) != cudaSuccess)
            return std::nullopt;
        GpuDeviceInfo result{.name = properties.name,
                             .total_memory_bytes = properties.totalGlobalMem};
        static_assert(sizeof(properties.uuid.bytes) == result.uuid.size());
        std::memcpy(result.uuid.data(), properties.uuid.bytes, result.uuid.size());
        result.compute_capability_major = properties.major;
        return result;
    }

    namespace internal {

        void cuda_where_into(Tensor& output, const Tensor& condition, float value, const Tensor& source) {
            LFS_FACADE_TRACE(where);
            pin_operands({&output, &condition, &source});
            condition.sync_to_stream(output.stream());
            source.sync_to_stream(output.stream());
            launch_where_scalar(output.data_ptr(), condition.data_ptr(), value, source.data_ptr(),
                                output.numel(), output.dtype() == DataType::Float16, output.stream());
            condition.record_stream(output.stream());
            source.record_stream(output.stream());
        }
    } // namespace internal

    namespace cuda {
        thread_local unsigned external_memory_import_depth = 0;
        ExternalMemoryImportScope::ExternalMemoryImportScope() noexcept { ++external_memory_import_depth; }
        ExternalMemoryImportScope::~ExternalMemoryImportScope() noexcept { --external_memory_import_depth; }
        bool ExternalMemoryImportScope::active() noexcept { return external_memory_import_depth != 0; }
    } // namespace cuda

} // namespace lfs::core

#ifdef LFS_TENSOR_VULKAN
#include "../tensor_vulkan_interop.hpp"
#include "../vulkan/vk_cuda_bridge.hpp"
#include "core/exportable_storage.hpp"
#include "core/sh_value_quant.hpp"
#include "core/shareable_allocation_limit.hpp"
#include "core/splat_exportable_storage.hpp"
#include "core/tensor/backend/cuda/runtime/cuda_stream_context.hpp"
#include <algorithm>
#include <limits>
#ifndef _WIN32
#include <unistd.h>
#endif

namespace lfs::core::internal {
    namespace {
        void interop_vk_check(VkResult status, const char* operation) {
            if (status != VK_SUCCESS)
                throw TensorError(std::format("{} failed: {}", operation, static_cast<int>(status)));
        }

        struct CudaImportedBuffer {
            VulkanInteropDevice target;
            std::shared_ptr<ExportableBlock> block;
            VkBuffer buffer = VK_NULL_HANDLE;
            std::vector<VkDeviceMemory> memories;
            std::vector<size_t> bound_offsets;
            uint64_t address = 0;
            bool sparse = false;

            ~CudaImportedBuffer() { release(); }

            void release() {
                const auto device = static_cast<VkDevice>(target.device);
                if (buffer)
                    vkDestroyBuffer(device, buffer, nullptr);
                for (const auto memory : memories)
                    vkFreeMemory(device, memory, nullptr);
                memories.clear();
                buffer = VK_NULL_HANDLE;
            }

            void bind() {
                const auto unbound = unboundExportableChunkIndices(block->chunks, bound_offsets);
                if (unbound.empty())
                    return;
                const auto device = static_cast<VkDevice>(target.device);
                VkMemoryRequirements requirements{};
                vkGetBufferMemoryRequirements(device, buffer, &requirements);
                VkPhysicalDeviceMemoryProperties properties{};
                vkGetPhysicalDeviceMemoryProperties(static_cast<VkPhysicalDevice>(target.physical_device), &properties);
                const uint32_t memory_type = find_vulkan_memory_type(
                    properties, requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
                if (memory_type == std::numeric_limits<uint32_t>::max())
                    throw TensorError("CUDA block has no compatible Vulkan memory type");
                std::vector<VkSparseMemoryBind> binds;
                for (const auto i : unbound) {
                    const auto& chunk = block->chunks[i];
#ifdef _WIN32
                    VkImportMemoryWin32HandleInfoKHR imported{VK_STRUCTURE_TYPE_IMPORT_MEMORY_WIN32_HANDLE_INFO_KHR};
                    imported.handleType = kVulkanExportMemoryHandleType;
                    imported.handle = chunk.handle.native;
#else
                    VkImportMemoryFdInfoKHR imported{VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR};
                    imported.handleType = kVulkanExportMemoryHandleType;
                    imported.fd = ::dup(chunk.handle.native);
                    if (imported.fd < 0)
                        throw TensorError("Cannot duplicate CUDA memory export handle");
#endif
                    VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
                    flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
                    flags.pNext = &imported;
                    VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
                    allocation.pNext = &flags;
                    allocation.allocationSize = chunk.bytes;
                    allocation.memoryTypeIndex = memory_type;
                    VkDeviceMemory memory = VK_NULL_HANDLE;
                    VkResult status;
                    {
                        const cuda::ExternalMemoryImportScope import_scope;
                        status = vkAllocateMemory(device, &allocation, nullptr, &memory);
                    }
#ifndef _WIN32
                    if (status != VK_SUCCESS)
                        ::close(imported.fd);
#endif
                    interop_vk_check(status, "vkAllocateMemory(CUDA import)");
                    memories.push_back(memory);
                    if (sparse) {
                        binds.push_back(VkSparseMemoryBind{chunk.offset, chunk.bytes, memory, 0, 0});
                    } else {
                        interop_vk_check(vkBindBufferMemory(device, buffer, memory, 0), "vkBindBufferMemory(CUDA import)");
                    }
                }
                if (!binds.empty()) {
                    VkSparseBufferMemoryBindInfo buffer_bind{buffer, static_cast<uint32_t>(binds.size()), binds.data()};
                    VkBindSparseInfo bind{VK_STRUCTURE_TYPE_BIND_SPARSE_INFO};
                    bind.bufferBindCount = 1;
                    bind.pBufferBinds = &buffer_bind;
                    VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
                    VkFence fence = VK_NULL_HANDLE;
                    interop_vk_check(vkCreateFence(device, &fence_info, nullptr, &fence), "vkCreateFence(CUDA import)");
                    VkResult status;
                    {
                        std::unique_lock<std::mutex> lock;
                        if (target.queue_mutex)
                            lock = std::unique_lock(*target.queue_mutex);
                        status = vkQueueBindSparse(static_cast<VkQueue>(target.sparse_queue), 1, &bind, fence);
                    }
                    if (status == VK_SUCCESS) {
                        do {
                            status = vkWaitForFences(device, 1, &fence, VK_TRUE, 1'000'000'000);
                        } while (status == VK_TIMEOUT);
                    }
                    vkDestroyFence(device, fence, nullptr);
                    interop_vk_check(status, "vkQueueBindSparse(CUDA import)");
                }
                for (const auto i : unbound)
                    bound_offsets.push_back(block->chunks[i].offset);
                VkBufferDeviceAddressInfo address_info{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
                address_info.buffer = buffer;
                address = vkGetBufferDeviceAddress(device, &address_info);
            }
        };

        struct CudaImportedTimeline {
            cudaExternalSemaphore_t imported = nullptr;
#ifdef _WIN32
            HANDLE handle = nullptr;
#endif
            ~CudaImportedTimeline() {
                if (imported)
                    LFS_CUDA_LOG_TEARDOWN(cudaDestroyExternalSemaphore(imported), nullptr, "tensor interop timeline release");
#ifdef _WIN32
                if (handle)
                    CloseHandle(handle);
#endif
            }
        };

        std::shared_ptr<CudaImportedTimeline> import_cuda_timeline(VkDevice device, VkSemaphore semaphore) {
            auto result = std::make_shared<CudaImportedTimeline>();
            cudaExternalSemaphoreHandleDesc descriptor{};
#ifdef _WIN32
            const auto get = reinterpret_cast<PFN_vkGetSemaphoreWin32HandleKHR>(vkGetDeviceProcAddr(device, "vkGetSemaphoreWin32HandleKHR"));
            if (!get)
                throw TensorError("Vulkan device cannot export timeline semaphores");
            VkSemaphoreGetWin32HandleInfoKHR info{VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR};
            info.semaphore = semaphore;
            info.handleType = kVulkanExportSemaphoreHandleType;
            interop_vk_check(get(device, &info, &result->handle), "vkGetSemaphoreWin32HandleKHR");
            descriptor.type = cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32;
            descriptor.handle.win32.handle = result->handle;
#else
            const auto get = reinterpret_cast<PFN_vkGetSemaphoreFdKHR>(vkGetDeviceProcAddr(device, "vkGetSemaphoreFdKHR"));
            if (!get)
                throw TensorError("Vulkan device cannot export timeline semaphores");
            VkSemaphoreGetFdInfoKHR info{VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR};
            info.semaphore = semaphore;
            info.handleType = kVulkanExportSemaphoreHandleType;
            int fd = -1;
            interop_vk_check(get(device, &info, &fd), "vkGetSemaphoreFdKHR");
            descriptor.type = cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd;
            descriptor.handle.fd = fd;
#endif
            const auto status = cudaImportExternalSemaphore(&result->imported, &descriptor);
#ifndef _WIN32
            if (status != cudaSuccess)
                ::close(fd);
#endif
            if (status != cudaSuccess)
                throw TensorError(std::format("cudaImportExternalSemaphore failed: {}", cudaGetErrorString(status)));
            return result;
        }

        class CudaTensorVulkanInterop final : public TensorVulkanInteropBackend,
                                              public std::enable_shared_from_this<CudaTensorVulkanInterop> {
        public:
            explicit CudaTensorVulkanInterop(VulkanInteropDevice target) : target_(target) {}

            ~CudaTensorVulkanInterop() override { shutdown(); }

            void shutdown() override {
                if (stream_) {
                    LFS_CUDA_LOG_TEARDOWN(cudaStreamSynchronize(stream_), stream_, "tensor interop stream drain");
                    CudaMemoryPool::instance().release_stream(stream_);
                    LFS_CUDA_LOG_TEARDOWN(cudaStreamDestroy(stream_), stream_, "tensor interop stream release");
                    stream_ = nullptr;
                }
                if (!waits_.empty())
                    LFS_CUDA_LOG_TEARDOWN(cudaDeviceSynchronize(), nullptr, "tensor interop reverse waits drain");
                for (auto& [key, weak] : imports_) {
                    if (auto imported = weak.lock())
                        imported->release();
                }
                imports_.clear();
                waits_.clear();
                timeline_import_.reset();
                if (timeline_)
                    vkDestroySemaphore(static_cast<VkDevice>(target_.device), timeline_, nullptr);
                timeline_ = VK_NULL_HANDLE;
            }

            void release_timeline(void* handle) override {
                std::lock_guard lock(mutex_);
                const auto semaphore = static_cast<VkSemaphore>(handle);
                const auto found = waits_.find(semaphore);
                if (found != waits_.end()) {
                    for (const auto stream : found->second.streams)
                        LFS_CUDA_CHECK(cudaStreamSynchronize(stream));
                    waits_.erase(found);
                }
            }

            void drain() override {
                if (stream_)
                    LFS_CUDA_CHECK(cudaStreamSynchronize(stream_));
            }

            std::shared_ptr<void> execution_scope() override {
                std::lock_guard lock(mutex_);
                ensure_timeline();
                struct Scope {
                    GpuBackendScope backend{GpuBackend::CUDA};
                    CUDAStreamGuard stream;
                    explicit Scope(cudaStream_t value) : stream(value) {}
                };
                return std::make_shared<Scope>(stream_);
            }

            Tensor empty_splat(TensorShape shape, size_t capacity, DataType dtype,
                               std::string_view name, bool preserve_float_shN) override {
                GpuBackendScope backend(GpuBackend::CUDA);
                if (!preserve_float_shN && name == "SplatData.shN" &&
                    dtype == DataType::Float32 && sh_value_quant::enabled())
                    return Tensor::zeros_direct(std::move(shape), capacity, Device::GPU, dtype);
                return empty(std::move(shape), dtype, capacity);
            }

            Tensor empty(TensorShape shape, DataType dtype, size_t capacity) override {
                if (!target_.external_memory)
                    throw TensorError("Consumer device does not support CUDA external memory");
                const size_t rows = shape.rank() == 0 ? 1 : shape[0];
                capacity = std::max(capacity, rows);
                size_t bytes = dtype_size(dtype);
                for (size_t i = 1; i < shape.rank(); ++i) {
                    if (shape[i] == 0 || bytes > SIZE_MAX / shape[i])
                        throw TensorError("Vulkan-visible tensor has an invalid shape");
                    bytes *= shape[i];
                }
                if (capacity == 0 || bytes > SIZE_MAX / capacity)
                    throw TensorError("Vulkan-visible tensor allocation size is invalid");
                bytes *= capacity;
                int device = 0;
                LFS_CUDA_CHECK(cudaGetDevice(&device));
                auto block = allocateExportableDeviceBlock(bytes, device, false, bytes);
                if (!block) {
                    if (is_shareable_allocation_limit_message(block.error()))
                        throw ShareableAllocationLimitError(block.error());
                    throw TensorError(block.error());
                }
                auto tensor = Tensor::from_external_owner((*block)->device_ptr, std::move(shape),
                                                          Device::GPU, dtype, *block, capacity, getCurrentCUDAStream(), "vulkan_external_buffer");
                (void)buffer(tensor);
                return tensor;
            }

            std::optional<TensorVulkanBuffer> buffer(const Tensor& tensor) override {
                std::shared_ptr<ExportableBlock> block;
                size_t offset = tensor.storage_offset() * dtype_size(tensor.dtype());
                size_t bytes = tensor.bytes();
                if (tensor.has_exportable_provenance()) {
                    auto control = std::static_pointer_cast<SplatExportableStorage::Control>(tensor.exportable_control());
                    const auto region = tensor.exportable_region();
                    if (!control || region >= SplatExportableStorage::Count)
                        throw TensorError("Invalid exportable tensor region");
                    block = control->block;
                    offset += control->region_offsets[region];
                    bytes = control->region_bytes[region] - std::min(control->region_bytes[region],
                                                                     tensor.storage_offset() * dtype_size(tensor.dtype()));
                } else if (tensor.external_storage_kind() == "vulkan_external_buffer") {
                    block = std::static_pointer_cast<ExportableBlock>(tensor.external_storage_owner());
                } else {
                    return std::nullopt;
                }
                if (!block || offset > block->reserved_bytes || bytes > block->reserved_bytes - offset)
                    throw TensorError("Vulkan-visible tensor exceeds its exportable block");
                std::lock_guard lock(mutex_);
                auto imported = imports_[block.get()].lock();
                if (!imported) {
                    if (!target_.external_memory)
                        throw TensorError("Consumer device does not support CUDA external memory");
                    const auto cuda = gpu_backend_device_info(GpuBackend::CUDA);
                    VkPhysicalDeviceIDProperties identity{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
                    VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
                    properties.pNext = &identity;
                    vkGetPhysicalDeviceProperties2(static_cast<VkPhysicalDevice>(target_.physical_device), &properties);
                    if (!cuda || !std::equal(cuda->uuid.begin(), cuda->uuid.end(), identity.deviceUUID))
                        throw TensorError("Vulkan consumer and CUDA storage must use the same physical device");
                    imported = std::make_shared<CudaImportedBuffer>();
                    imported->target = target_;
                    imported->block = block;
                    imported->sparse = block->chunks.size() != 1 || block->chunks.front().bytes < block->reserved_bytes;
                    if (imported->sparse && (!target_.sparse_binding || !target_.sparse_queue))
                        throw TensorError("Sparse binding is required for a growable CUDA block");
                    VkExternalMemoryBufferCreateInfo external{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
                    external.handleTypes = kVulkanExportMemoryHandleType;
                    VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
                    info.pNext = &external;
                    info.flags = imported->sparse ? VK_BUFFER_CREATE_SPARSE_BINDING_BIT : 0;
                    info.size = block->reserved_bytes;
                    info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                 VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
                    info.sharingMode = target_.queue_family_count > 1 ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE;
                    if (target_.queue_family_count > 1) {
                        info.queueFamilyIndexCount = target_.queue_family_count;
                        info.pQueueFamilyIndices = target_.queue_families.data();
                    }
                    interop_vk_check(vkCreateBuffer(static_cast<VkDevice>(target_.device), &info, nullptr, &imported->buffer), "vkCreateBuffer(CUDA import)");
                    imported->bind();
                    imports_[block.get()] = imported;
                } else {
                    imported->bind();
                }
                const auto storage = storage_ref(tensor);
                if (storage.meta) {
                    std::lock_guard storage_lock(storage.meta->vulkan_interop_mutex);
                    storage.meta->vulkan_interop_owners[target_.device] = imported;
                }
                return TensorVulkanBuffer{.buffer = imported->buffer, .offset = offset, .device_address = imported->address + offset, .bytes = bytes, .keep_alive = imported, .device = target_.device};
            }

            TensorCompletion ready(std::span<const Tensor* const> tensors) override {
                std::lock_guard lock(mutex_);
                ensure_timeline();
                std::vector<cudaStream_t> streams;
                for (const auto* tensor : tensors) {
                    if (!tensor || !tensor->is_valid() || gpu_backend_of(*tensor) != GpuBackend::CUDA)
                        continue;
                    const auto source = tensor->stream();
                    if (source == stream_)
                        continue;
                    if (std::find(streams.begin(), streams.end(), source) == streams.end()) {
                        backend_ops_for(*tensor).bridge(ExecContext{source}, ExecContext{stream_});
                        streams.push_back(source);
                    }
                    tensor->record_stream(stream_);
                }
                cudaExternalSemaphoreSignalParams signal{};
                signal.params.fence.value = ++value_;
                LFS_CUDA_CHECK(cudaSignalExternalSemaphoresAsync(&timeline_import_->imported, &signal, 1, stream_));
                return TensorCompletionAccess::external(target_.device, {timeline_, value_, shared_from_this()});
            }

            void wait(std::span<const Tensor* const> tensors, VulkanTimelinePoint point) override {
                std::lock_guard lock(mutex_);
                const auto semaphore = static_cast<VkSemaphore>(point.semaphore);
                auto& pending = waits_[semaphore];
                if (!pending.imported)
                    pending.imported = import_cuda_timeline(static_cast<VkDevice>(target_.device), semaphore);
                std::vector<cudaStream_t> streams;
                for (const auto* tensor : tensors) {
                    const auto stream = tensor->stream();
                    if (std::find(streams.begin(), streams.end(), stream) != streams.end())
                        continue;
                    streams.push_back(stream);
                    auto& consumers = pending.streams;
                    if (std::find(consumers.begin(), consumers.end(), stream) == consumers.end())
                        consumers.push_back(stream);
                    cudaExternalSemaphoreWaitParams wait{};
                    wait.params.fence.value = point.value;
                    LFS_CUDA_CHECK(cudaWaitExternalSemaphoresAsync(&pending.imported->imported, &wait, 1, stream));
                }
                if (point.keep_alive)
                    pending.owner = std::move(point.keep_alive);
            }

        private:
            void ensure_timeline() {
                if (timeline_)
                    return;
                if (!target_.external_semaphore)
                    throw TensorError("Consumer device does not support external timeline semaphores");
                LFS_CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
                VkExportSemaphoreCreateInfo external{VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO};
                external.handleTypes = kVulkanExportSemaphoreHandleType;
                VkSemaphoreTypeCreateInfo type{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
                type.pNext = &external;
                type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
                VkSemaphoreCreateInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
                info.pNext = &type;
                interop_vk_check(vkCreateSemaphore(static_cast<VkDevice>(target_.device), &info, nullptr, &timeline_), "vkCreateSemaphore(CUDA ready)");
                timeline_import_ = import_cuda_timeline(static_cast<VkDevice>(target_.device), timeline_);
            }

            VulkanInteropDevice target_;
            std::mutex mutex_;
            std::unordered_map<ExportableBlock*, std::weak_ptr<CudaImportedBuffer>> imports_;
            cudaStream_t stream_ = nullptr;
            VkSemaphore timeline_ = VK_NULL_HANDLE;
            uint64_t value_ = 0;
            std::shared_ptr<CudaImportedTimeline> timeline_import_;
            struct PendingWait {
                std::shared_ptr<void> owner;
                std::shared_ptr<CudaImportedTimeline> imported;
                std::vector<cudaStream_t> streams;
            };
            std::unordered_map<VkSemaphore, PendingWait> waits_;
        };
    } // namespace

    std::shared_ptr<TensorVulkanInteropBackend> make_cuda_vulkan_interop(VulkanInteropDevice target) {
        return std::make_shared<CudaTensorVulkanInterop>(target);
    }
} // namespace lfs::core::internal
namespace lfs::core::cuda {
    std::shared_ptr<void> import_vulkan_semaphore(void* device, void* semaphore) {
        auto owner = internal::import_cuda_timeline(static_cast<VkDevice>(device), static_cast<VkSemaphore>(semaphore));
        return std::shared_ptr<void>(owner, owner->imported);
    }
} // namespace lfs::core::cuda
#endif

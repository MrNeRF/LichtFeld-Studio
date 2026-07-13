/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "allocation_profiler.hpp"
#include "core/export.hpp"
#include "core/logger.hpp"
#include "core/pinned_memory_allocator.hpp"
#include "cuda_event_pool.hpp"
#include "deferred_free_queue.hpp"
#include "diagnostics/vram_profiler.hpp"
#include "gpu_slab_allocator.hpp"
#include "size_bucketed_pool.hpp"
#include <algorithm>
#include <cuda_runtime.h>
#include <iomanip>
#include <memory>
#include <mutex>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace lfs::core {

    static constexpr size_t SLAB_ALLOC_THRESHOLD = 256 * 1024;
    static constexpr size_t BUCKET_ALLOC_THRESHOLD = 16ULL * 1024 * 1024 * 1024;

    enum class AllocMethod : uint8_t { Slab,
                                       Bucketed,
                                       Async,
                                       Direct };

    // Multi-tier CUDA memory pool: slab (≤256KB), bucketed (≤16GB), cudaMallocAsync.
    class LFS_CORE_API CudaMemoryPool {
    public:
        static CudaMemoryPool& instance();

        class LFS_CORE_API LabelGuard {
        public:
            explicit LabelGuard(std::string_view label);
            ~LabelGuard();
            LabelGuard(const LabelGuard&) = delete;
            LabelGuard& operator=(const LabelGuard&) = delete;
            LabelGuard(LabelGuard&&) = delete;
            LabelGuard& operator=(LabelGuard&&) = delete;

        private:
            std::string previous_;
            bool active_ = false;
        };

        static std::string_view current_label() noexcept;

        void suspend_deallocations_for_process_exit() {
            suspend_deallocations_.store(true, std::memory_order_release);
        }

        void shutdown() {
            bool expected = false;
            if (!shutdown_.compare_exchange_strong(expected, true))
                return;
            LOG_INFO("Shutting down CudaMemoryPool...");
            if (suspend_deallocations_.load(std::memory_order_acquire)) {
                return;
            }
            DeferredFreeQueue::instance().shutdown();
            SizeBucketedPool::instance().shutdown();
            GPUSlabAllocator::instance().shutdown();
            CudaEventPool::instance().shutdown();
        }

        void* allocate(size_t bytes, cudaStream_t stream = nullptr) {
            if (bytes == 0)
                return nullptr;

            if (shutdown_.load(std::memory_order_acquire)) {
                LOG_ERROR("Attempted to allocate CUDA memory after shutdown!");
                return nullptr;
            }

            void* ptr = nullptr;

            if (bytes <= SLAB_ALLOC_THRESHOLD && slab_enabled_) {
                ptr = GPUSlabAllocator::instance().allocate(bytes, stream);
                if (ptr) {
                    stats_.slab_allocs.fetch_add(1, std::memory_order_relaxed);
                    stats_.slab_bytes.fetch_add(bytes, std::memory_order_relaxed);
                    track_allocation(ptr, bytes, AllocMethod::Slab, stream);

                    if constexpr (ENABLE_ALLOCATION_PROFILING) {
                        AllocationProfiler::instance().record_allocation(bytes, 3);
                    }
                    return ptr;
                }
            }

            if (bytes <= BUCKET_ALLOC_THRESHOLD) {
                ptr = SizeBucketedPool::instance().try_allocate_cached(bytes, stream);
                if (ptr) {
                    stats_.bucket_cache_hits.fetch_add(1, std::memory_order_relaxed);
                    stats_.bucket_bytes.fetch_add(bytes, std::memory_order_relaxed);
                    track_allocation(ptr, bytes, AllocMethod::Bucketed, stream);
                    if constexpr (ENABLE_ALLOCATION_PROFILING) {
                        AllocationProfiler::instance().record_allocation(bytes, 3);
                    }
                    return ptr;
                }

                const size_t bucket_size = SizeBucketedPool::get_bucket_size(bytes);

#if CUDART_VERSION >= 12080
                cudaError_t err = cudaMallocAsync(&ptr, bucket_size, stream);
                if (err == cudaSuccess) {
                    stats_.bucket_allocs.fetch_add(1, std::memory_order_relaxed);
                    stats_.bucket_bytes.fetch_add(bytes, std::memory_order_relaxed);
                    stats_.bucket_waste.fetch_add(bucket_size - bytes, std::memory_order_relaxed);
                    track_allocation(ptr, bytes, AllocMethod::Bucketed, stream);
                    if constexpr (ENABLE_ALLOCATION_PROFILING) {
                        AllocationProfiler::instance().record_allocation(bytes, 3);
                    }
                    if ((stats_.bucket_allocs.load(std::memory_order_relaxed) +
                         stats_.async_allocs.load(std::memory_order_relaxed)) %
                            100 ==
                        0) {
                        DeferredFreeQueue::instance().process();
                    }
                    log_stats_periodically();
                    return ptr;
                }
                LOG_WARN("cudaMallocAsync failed for bucket " + std::to_string(bucket_size) + ": " + cudaGetErrorString(err));
#endif
            }

#if CUDART_VERSION >= 12080
            {
                cudaError_t err = cudaMallocAsync(&ptr, bytes, stream);
                if (err == cudaSuccess) {
                    stats_.async_allocs.fetch_add(1, std::memory_order_relaxed);
                    stats_.async_bytes.fetch_add(bytes, std::memory_order_relaxed);
                    track_allocation(ptr, bytes, AllocMethod::Async, stream);
                    if constexpr (ENABLE_ALLOCATION_PROFILING) {
                        AllocationProfiler::instance().record_allocation(bytes, 3);
                    }
                    return ptr;
                }
            }
#endif

            return allocate_direct(bytes);
        }

        // Marks `ptr` as used by `stream` beyond its home stream. The free will
        // bridge that use back into the home stream before the block is recycled.
        void record_stream(void* ptr, cudaStream_t stream) {
            if (!ptr)
                return;
            std::lock_guard<std::mutex> lock(map_mutex_);
            auto it = allocation_map_.find(ptr);
            if (it == allocation_map_.end())
                return;
            AllocationInfo& info = it->second;
            if (stream == info.home_stream)
                return;
            if (std::find(info.extra_streams.begin(), info.extra_streams.end(), stream) ==
                info.extra_streams.end()) {
                info.extra_streams.push_back(stream);
            }
        }

        // Severs every allocator reference to `stream` so it can be destroyed.
        // Waits for the stream's pending work, then drops it from recorded uses,
        // re-homes live allocations to the legacy stream, and migrates cached
        // free-list entries. Must be called before cudaStreamDestroy on any
        // stream that touched pool memory; destroying a referenced stream makes
        // later frees/reuse dereference a dead handle.
        void release_stream(cudaStream_t stream) {
            if (!stream)
                return;
            cudaStreamSynchronize(stream);
            {
                std::lock_guard<std::mutex> lock(map_mutex_);
                for (auto& [ptr, info] : allocation_map_) {
                    std::erase(info.extra_streams, stream);
                    if (info.home_stream == stream) {
                        info.home_stream = nullptr;
                    }
                }
            }
            GPUSlabAllocator::instance().merge_stream_into_virgin(stream);
            SizeBucketedPool::instance().retag_stream(stream, nullptr);
            PinnedMemoryAllocator::instance().release_stream(stream);
        }

        // Moves `ptr`'s home to `stream` (declarative re-homing for tensors whose
        // future writes happen there). The old home becomes a recorded use.
        void rehome_stream(void* ptr, cudaStream_t stream) {
            if (!ptr)
                return;
            std::lock_guard<std::mutex> lock(map_mutex_);
            auto it = allocation_map_.find(ptr);
            if (it == allocation_map_.end())
                return;
            AllocationInfo& info = it->second;
            if (stream == info.home_stream)
                return;
            if (std::find(info.extra_streams.begin(), info.extra_streams.end(), info.home_stream) ==
                info.extra_streams.end()) {
                info.extra_streams.push_back(info.home_stream);
            }
            std::erase(info.extra_streams, stream);
            info.home_stream = stream;
        }

        void deallocate(void* ptr, cudaStream_t stream = nullptr) {
            if (!ptr)
                return;
            if (shutdown_.load(std::memory_order_acquire))
                return;
            if (suspend_deallocations_.load(std::memory_order_acquire)) {
                AllocationInfo info;
                take_allocation(ptr, info);
                return;
            }

            if constexpr (ENABLE_ALLOCATION_PROFILING) {
                AllocationProfiler::instance().record_deallocation(ptr);
            }

            AllocationInfo info;
            if (take_allocation(ptr, info)) {
                free_routed(ptr, info);
                return;
            }

#if CUDART_VERSION >= 12080
            cudaFreeAsync(ptr, stream);
#else
            cudaFree(ptr);
#endif
        }

        void deallocate(void* ptr, size_t /*bytes*/, cudaStream_t stream = nullptr) {
            deallocate(ptr, stream);
        }

        void set_iteration(int iteration) {
            if constexpr (ENABLE_ALLOCATION_PROFILING) {
                AllocationProfiler::instance().set_iteration(iteration);
            }
        }

        void record_tensor(void* ptr, const std::vector<size_t>& shape, size_t bytes, const std::string& dtype) {
            if constexpr (ENABLE_ALLOCATION_PROFILING) {
                AllocationProfiler::instance().record_tensor_allocation(ptr, shape, bytes, dtype, 3);
            }
        }

        void configure() {
#if CUDART_VERSION >= 12080
            int device;
            cudaError_t err = cudaGetDevice(&device);
            if (err != cudaSuccess) {
                LOG_ERROR(std::string("cudaGetDevice failed: ") + cudaGetErrorString(err));
                return;
            }

            cudaMemPool_t pool;
            err = cudaDeviceGetDefaultMemPool(&pool, device);
            if (err != cudaSuccess) {
                LOG_ERROR(std::string("cudaDeviceGetDefaultMemPool failed: ") + cudaGetErrorString(err));
                return;
            }

            // 64 MiB headroom: keep typical reuse fast (per-iter scratch buffers stay
            // pool-resident) while letting the driver reclaim memory beyond peak
            // densification spikes. UINT64_MAX hoards indefinitely and inflates
            // cuda.pool.overhead at higher gaussian counts.
            uint64_t threshold = std::uint64_t(64) << 20;
            cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold);

            LOG_DEBUG("CUDA memory pool configured for device " + std::to_string(device) + " (CUDA " + std::to_string(CUDART_VERSION) + ")");
#else
            LOG_WARN("CUDA memory pooling not available (requires CUDA >= 12.8)");
#endif

            slab_enabled_ = true;
            LOG_DEBUG("Slab allocator enabled (lazy, ≤256KB)");
            LOG_DEBUG("Size-bucketed pool enabled (256KB-16GB, reduces fragmentation)");
        }

        std::string get_stats() const {
            std::ostringstream oss;
            oss << "Memory Pool Stats:\n";
            oss << "  Slab: " << stats_.slab_allocs.load() << " allocs ("
                << (stats_.slab_bytes.load() / 1024.0 / 1024.0) << " MB)\n";
            oss << "  Bucketed: " << stats_.bucket_allocs.load() << " allocs, "
                << stats_.bucket_cache_hits.load() << " cache hits ("
                << (stats_.bucket_bytes.load() / 1024.0 / 1024.0) << " MB, "
                << (stats_.bucket_waste.load() / 1024.0 / 1024.0) << " MB wasted)\n";
            oss << "  Async: " << stats_.async_allocs.load() << " allocs ("
                << (stats_.async_bytes.load() / 1024.0 / 1024.0) << " MB)\n";
            oss << "  Direct: " << stats_.direct_allocs.load() << " allocs ("
                << (stats_.direct_bytes.load() / 1024.0 / 1024.0) << " MB)\n";

#if CUDART_VERSION >= 12080
            int device;
            cudaGetDevice(&device);
            cudaMemPool_t pool;
            cudaDeviceGetDefaultMemPool(&pool, device);

            uint64_t used = 0, reserved = 0;
            cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemCurrent, &used);
            cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemCurrent, &reserved);

            oss << "  CUDA Pool: " << (used / 1024.0 / 1024.0) << " / "
                << (reserved / 1024.0 / 1024.0) << " MB used/reserved\n";
#endif
            return oss.str();
        }

        void trim() {
            SizeBucketedPool::instance().trim_cache();
#if CUDART_VERSION >= 12080
            int device;
            cudaGetDevice(&device);
            cudaMemPool_t pool;
            cudaDeviceGetDefaultMemPool(&pool, device);
            cudaMemPoolTrimTo(pool, 0);
#endif
        }

        void trim_cached_memory() {
            if (suspend_deallocations_.load(std::memory_order_acquire)) {
                return;
            }
            cudaDeviceSynchronize();
            DeferredFreeQueue::instance().flush();
            {
                std::lock_guard<std::mutex> lock(map_mutex_);
                for (auto& [ptr, info] : allocation_map_) {
                    info.extra_streams.clear();
                }
            }
            GPUSlabAllocator::instance().merge_all_streams_into_virgin();
            SizeBucketedPool::instance().retag_all_streams(nullptr);
            SizeBucketedPool::instance().trim_cache();

#if CUDART_VERSION >= 12080
            int device;
            cudaGetDevice(&device);
            cudaMemPool_t pool;
            if (cudaDeviceGetDefaultMemPool(&pool, device) == cudaSuccess) {
                cudaMemPoolTrimTo(pool, 0);
            }
#endif
        }

        void print_stats() const {
            LOG_DEBUG(get_stats());
            GPUSlabAllocator::instance().print_stats();
            SizeBucketedPool::instance().print_stats();
        }

        CudaMemoryPool(const CudaMemoryPool&) = delete;
        CudaMemoryPool& operator=(const CudaMemoryPool&) = delete;

    private:
        struct Stats {
            std::atomic<uint64_t> slab_allocs{0};
            std::atomic<uint64_t> slab_bytes{0};
            std::atomic<uint64_t> bucket_allocs{0};
            std::atomic<uint64_t> bucket_cache_hits{0};
            std::atomic<uint64_t> bucket_bytes{0};
            std::atomic<uint64_t> bucket_waste{0};
            std::atomic<uint64_t> async_allocs{0};
            std::atomic<uint64_t> async_bytes{0};
            std::atomic<uint64_t> direct_allocs{0};
            std::atomic<uint64_t> direct_bytes{0};
        };

        struct AllocationInfo {
            size_t size = 0;
            AllocMethod method = AllocMethod::Direct;
            cudaStream_t home_stream = nullptr;
            std::vector<cudaStream_t> extra_streams;
        };

        CudaMemoryPool() {
            configure();
        }

        ~CudaMemoryPool() {
            shutdown();
        }

        void* allocate_direct(size_t bytes) {
            void* ptr = nullptr;

            cudaError_t err = cudaMalloc(&ptr, bytes);
            if (err != cudaSuccess) {
                LOG_WARN(std::string("[MEM] cudaMalloc failed: ") + cudaGetErrorString(err) + ", trimming...");
                cudaDeviceSynchronize();
                SizeBucketedPool::instance().trim_cache();
#if CUDART_VERSION >= 12080
                int device;
                cudaGetDevice(&device);
                cudaMemPool_t pool;
                cudaDeviceGetDefaultMemPool(&pool, device);
                cudaMemPoolTrimTo(pool, 0);
#endif
                err = cudaMalloc(&ptr, bytes);
                if (err != cudaSuccess) {
                    LOG_ERROR(std::string("[MEM] cudaMalloc retry failed: ") + cudaGetErrorString(err));
                    cudaGetLastError(); // Clear sticky error state for clean recovery
                    return nullptr;
                }
            }

            stats_.direct_allocs.fetch_add(1, std::memory_order_relaxed);
            stats_.direct_bytes.fetch_add(bytes, std::memory_order_relaxed);
            direct_alloc_count_.fetch_add(1, std::memory_order_release);

            track_allocation(ptr, bytes, AllocMethod::Direct);

            if constexpr (ENABLE_ALLOCATION_PROFILING) {
                AllocationProfiler::instance().record_allocation(bytes, 3);
            }

            return ptr;
        }

        void track_allocation(void* ptr, size_t size, AllocMethod method, cudaStream_t stream = nullptr) {
            std::lock_guard<std::mutex> lock(map_mutex_);
            allocation_map_[ptr] = {size, method, stream, {}};
            try {
                lfs::diagnostics::VramProfiler::instance().recordAllocation(
                    ptr, size, to_vram_method(method), current_label());
            } catch (...) {
                // Diagnostics must never make CUDA allocation fail.
            }
        }

        bool take_allocation(void* ptr, AllocationInfo& info) {
            std::lock_guard<std::mutex> lock(map_mutex_);
            auto it = allocation_map_.find(ptr);
            if (it == allocation_map_.end())
                return false;
            info = std::move(it->second);
            allocation_map_.erase(it);
            try {
                lfs::diagnostics::VramProfiler::instance().recordDeallocation(ptr);
            } catch (...) {
                // Diagnostics must never make CUDA deallocation fail.
            }
            return true;
        }

        // Bridges every recorded cross-stream use into the home stream, then frees
        // stream-ordered on home. The block is reusable the moment the GPU passes
        // the edges — no host sync, no deferred retention.
        void free_routed(void* ptr, const AllocationInfo& info) {
            for (cudaStream_t extra : info.extra_streams) {
                bridgeStreams(extra, info.home_stream);
            }

            switch (info.method) {
            case AllocMethod::Slab:
                GPUSlabAllocator::instance().deallocate(ptr, info.size, info.home_stream);
                return;
            case AllocMethod::Bucketed:
                SizeBucketedPool::instance().deallocate(ptr, info.size, info.home_stream);
                return;
            case AllocMethod::Direct:
                cudaFree(ptr);
                direct_alloc_count_.fetch_sub(1, std::memory_order_release);
                return;
            case AllocMethod::Async:
                break;
            }

#if CUDART_VERSION >= 12080
            cudaFreeAsync(ptr, info.home_stream);
#else
            cudaFree(ptr);
#endif
        }

        static lfs::diagnostics::VramAllocationMethod to_vram_method(AllocMethod method) {
            switch (method) {
            case AllocMethod::Slab: return lfs::diagnostics::VramAllocationMethod::Slab;
            case AllocMethod::Bucketed: return lfs::diagnostics::VramAllocationMethod::Bucketed;
            case AllocMethod::Async: return lfs::diagnostics::VramAllocationMethod::Async;
            case AllocMethod::Direct: return lfs::diagnostics::VramAllocationMethod::Direct;
            }
            return lfs::diagnostics::VramAllocationMethod::Unknown;
        }

        void log_stats_periodically() {
            static std::atomic<int> log_counter{0};
            if (++log_counter % 2000 == 0) {
                if constexpr (ENABLE_ALLOCATION_PROFILING) {
                    AllocationProfiler::instance().print_top_allocators(30);
                    AllocationProfiler::instance().print_active_allocations(30);
                    AllocationProfiler::instance().print_tensor_allocations(30);
                }

#if CUDART_VERSION >= 12080
                int device;
                cudaGetDevice(&device);
                cudaMemPool_t pool;
                cudaDeviceGetDefaultMemPool(&pool, device);

                uint64_t pool_used = 0, pool_reserved = 0;
                cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemCurrent, &pool_used);
                cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemCurrent, &pool_reserved);

                constexpr double GB = 1024.0 * 1024.0 * 1024.0;
                std::ostringstream oss;
                oss << "[MEM] Slab:" << stats_.slab_allocs.load()
                    << " Bucket:" << stats_.bucket_allocs.load()
                    << " (hits:" << stats_.bucket_cache_hits.load() << ")"
                    << " Async:" << stats_.async_allocs.load()
                    << " | Pool:" << std::fixed << std::setprecision(2)
                    << (pool_used / GB) << "/" << (pool_reserved / GB) << "GB";
                LOG_DEBUG(oss.str());
#endif
            }
        }

        std::unordered_map<void*, AllocationInfo> allocation_map_;
        std::mutex map_mutex_;
        std::atomic<size_t> direct_alloc_count_{0};
        bool slab_enabled_{false};
        std::atomic<bool> shutdown_{false};
        std::atomic<bool> suspend_deallocations_{false};
        Stats stats_;
    };

} // namespace lfs::core

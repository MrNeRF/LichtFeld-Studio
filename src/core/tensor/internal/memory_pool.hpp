/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "allocation_profiler.hpp"
#include "core/export.hpp"
#include "core/logger.hpp"
#include "deferred_free_queue.hpp"
#include "gpu_slab_allocator.hpp"
#include "size_bucketed_pool.hpp"
#include <cuda_runtime.h>
#include <iomanip>
#include <memory>
#include <mutex>
#include <sstream>
#include <unordered_map>

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

        void shutdown() {
            bool expected = false;
            if (!shutdown_.compare_exchange_strong(expected, true))
                return;
            LOG_INFO("Shutting down CudaMemoryPool...");
            DeferredFreeQueue::instance().shutdown();
            SizeBucketedPool::instance().shutdown();
            GPUSlabAllocator::instance().shutdown();
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
                ptr = GPUSlabAllocator::instance().allocate(bytes);
                if (ptr) {
                    stats_.slab_allocs.fetch_add(1, std::memory_order_relaxed);
                    stats_.slab_bytes.fetch_add(bytes, std::memory_order_relaxed);
                    track_allocation(ptr, bytes, AllocMethod::Slab);

                    if constexpr (ENABLE_ALLOCATION_PROFILING) {
                        AllocationProfiler::instance().record_allocation(bytes, 3);
                    }
                    return ptr;
                }
            }

            if (bytes <= BUCKET_ALLOC_THRESHOLD) {
                ptr = SizeBucketedPool::instance().try_allocate_cached(bytes);
                if (ptr) {
                    stats_.bucket_cache_hits.fetch_add(1, std::memory_order_relaxed);
                    stats_.bucket_bytes.fetch_add(bytes, std::memory_order_relaxed);
                    track_allocation(ptr, bytes, AllocMethod::Bucketed);
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
                    track_allocation(ptr, bytes, AllocMethod::Bucketed);
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
                    if constexpr (ENABLE_ALLOCATION_PROFILING) {
                        AllocationProfiler::instance().record_allocation(bytes, 3);
                    }
                    return ptr;
                }
            }
#endif

            return allocate_direct(bytes);
        }

        void deallocate(void* ptr, cudaStream_t stream = nullptr) {
            if (!ptr)
                return;
            if (shutdown_.load(std::memory_order_acquire))
                return;

            if constexpr (ENABLE_ALLOCATION_PROFILING) {
                AllocationProfiler::instance().record_deallocation(ptr);
            }

            AllocMethod method;
            size_t size;
            if (lookup_allocation(ptr, method, size)) {
                untrack_allocation(ptr);

                switch (method) {
                case AllocMethod::Slab:
                    GPUSlabAllocator::instance().deallocate(ptr, size);
                    return;
                case AllocMethod::Bucketed:
                    SizeBucketedPool::instance().cache_free(ptr, size);
                    return;
                case AllocMethod::Direct:
                    cudaFree(ptr);
                    direct_alloc_count_.fetch_sub(1, std::memory_order_release);
                    return;
                case AllocMethod::Async:
                    break;
                }
            }

#if CUDART_VERSION >= 12080
            cudaFreeAsync(ptr, stream);
#else
            cudaFree(ptr);
#endif
        }

        void deallocate(void* ptr, size_t bytes, cudaStream_t stream = nullptr) {
            if (!ptr)
                return;
            if (shutdown_.load(std::memory_order_acquire))
                return;

            if constexpr (ENABLE_ALLOCATION_PROFILING) {
                AllocationProfiler::instance().record_deallocation(ptr);
            }

            // Fast path for slab allocations
            if (bytes <= SLAB_ALLOC_THRESHOLD && slab_enabled_) {
                if (GPUSlabAllocator::instance().owns_pointer(ptr)) {
                    GPUSlabAllocator::instance().deallocate(ptr, bytes);
                    untrack_allocation(ptr);
                    return;
                }
            }

            // Fast path for bucketed allocations - cache for reuse
            if (bytes > SLAB_ALLOC_THRESHOLD && bytes <= BUCKET_ALLOC_THRESHOLD) {
                AllocMethod method;
                size_t size;
                if (lookup_allocation(ptr, method, size) && method == AllocMethod::Bucketed) {
                    untrack_allocation(ptr);
                    SizeBucketedPool::instance().cache_free(ptr, size);
                    return;
                }
            }

            // Check for direct allocation
            AllocMethod method;
            size_t size;
            if (lookup_allocation(ptr, method, size)) {
                untrack_allocation(ptr);
                if (method == AllocMethod::Direct) {
                    cudaFree(ptr);
                    direct_alloc_count_.fetch_sub(1, std::memory_order_release);
                    return;
                }
            }

            // Async allocation
#if CUDART_VERSION >= 12080
            cudaFreeAsync(ptr, stream);
#else
            cudaFree(ptr);
#endif
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

            uint64_t threshold = UINT64_MAX;
            cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold);

            LOG_DEBUG("CUDA memory pool configured for device " + std::to_string(device) + " (CUDA " + std::to_string(CUDART_VERSION) + ")");
#else
            LOG_WARN("CUDA memory pooling not available (requires CUDA >= 12.8)");
#endif

            slab_enabled_ = GPUSlabAllocator::instance().is_enabled();
            if (slab_enabled_) {
                LOG_DEBUG("Slab allocator enabled (≤256KB)");
            }
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
            cudaDeviceSynchronize();
            DeferredFreeQueue::instance().flush();
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
            size_t size;
            AllocMethod method;
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

        void track_allocation(void* ptr, size_t size, AllocMethod method) {
            std::lock_guard<std::mutex> lock(map_mutex_);
            allocation_map_[ptr] = {size, method};
        }

        void untrack_allocation(void* ptr) {
            std::lock_guard<std::mutex> lock(map_mutex_);
            allocation_map_.erase(ptr);
        }

        bool lookup_allocation(void* ptr, AllocMethod& method, size_t& size) {
            std::lock_guard<std::mutex> lock(map_mutex_);
            auto it = allocation_map_.find(ptr);
            if (it != allocation_map_.end()) {
                method = it->second.method;
                size = it->second.size;
                return true;
            }
            return false;
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
        Stats stats_;
    };

} // namespace lfs::core

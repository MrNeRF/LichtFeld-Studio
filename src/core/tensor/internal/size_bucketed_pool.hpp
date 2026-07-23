/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/cuda_error.hpp"
#include "core/export.hpp"
#include "core/logger.hpp"
#include "cuda_event_pool.hpp"
#include "diagnostics/vram_profiler.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cuda_runtime.h>
#include <limits>
#include <mutex>
#include <vector>

namespace lfs::core {

    // Size-bucketed memory pool. Rounds allocations to bucket boundaries and caches
    // freed memory per bucket to maximize reuse and reduce fragmentation.
    //
    // Cache entries are tagged with the stream their last use is ordered on.
    // Same-stream reuse is free (stream ordering); cross-stream reuse bridges
    // with a GPU-side event edge and retags. Evictions free on the entry's tag
    // stream — freeing on any other stream would be unordered with the last use.
    class SizeBucketedPool {
    public:
        static constexpr size_t MIN_BUCKET_SIZE = 256 * 1024;
        static constexpr size_t MAX_TRACKED_SIZE = 16ULL * 1024 * 1024 * 1024;
        static constexpr size_t CACHE_SIZE_PER_BUCKET = 4;
        static constexpr size_t MIN_CACHE_BUDGET = 64ULL * 1024 * 1024;
        static constexpr size_t MAX_CACHE_BUDGET = 256ULL * 1024 * 1024;
        static constexpr size_t NUM_BUCKETS = 128;

        struct Stats {
            std::atomic<uint64_t> cache_hits{0};
            std::atomic<uint64_t> cache_misses{0};
            std::atomic<uint64_t> alloc_count{0};
            std::atomic<uint64_t> free_count{0};
            std::atomic<uint64_t> bytes_cached{0};
            std::atomic<uint64_t> bytes_wasted{0};
            std::atomic<uint64_t> cross_stream_reuse{0};
        };

        static LFS_CORE_API SizeBucketedPool& instance();

        void shutdown() {
            bool expected = false;
            if (!shutdown_.compare_exchange_strong(expected, true))
                return;
            trim_cache();
        }

        static size_t get_bucket_size(size_t bytes) {
            if (bytes <= MIN_BUCKET_SIZE)
                return MIN_BUCKET_SIZE;
            if (bytes <= 1024 * 1024)
                return ((bytes + 256 * 1024 - 1) / (256 * 1024)) * (256 * 1024);
            if (bytes <= 16 * 1024 * 1024)
                return ((bytes + 1024 * 1024 - 1) / (1024 * 1024)) * (1024 * 1024);
            if (bytes <= 256 * 1024 * 1024)
                return ((bytes + 16 * 1024 * 1024 - 1) / (16 * 1024 * 1024)) * (16 * 1024 * 1024);
            if (bytes <= 1024ULL * 1024 * 1024)
                return ((bytes + 64 * 1024 * 1024 - 1) / (64 * 1024 * 1024)) * (64 * 1024 * 1024);
            if (bytes <= 8ULL * 1024 * 1024 * 1024)
                return ((bytes + 256ULL * 1024 * 1024 - 1) / (256ULL * 1024 * 1024)) * (256ULL * 1024 * 1024);
            return ((bytes + 1024ULL * 1024 * 1024 - 1) / (1024ULL * 1024 * 1024)) * (1024ULL * 1024 * 1024);
        }

        static size_t get_bucket_index(size_t bucket_size) {
            if (bucket_size <= 1024 * 1024)
                return (bucket_size / (256 * 1024)) - 1;
            if (bucket_size <= 16 * 1024 * 1024)
                return 4 + (bucket_size / (1024 * 1024)) - 1;
            if (bucket_size <= 256 * 1024 * 1024)
                return 20 + (bucket_size / (16 * 1024 * 1024)) - 1;
            if (bucket_size <= 1024ULL * 1024 * 1024)
                return 36 + (bucket_size / (64 * 1024 * 1024)) - 4;
            if (bucket_size <= 8ULL * 1024 * 1024 * 1024)
                return 48 + (bucket_size / (256ULL * 1024 * 1024)) - 4;
            const size_t idx = 76 + (bucket_size / (1024ULL * 1024 * 1024)) - 8;
            return std::min(idx, NUM_BUCKETS - 1);
        }

        static size_t max_cached_entries_for_bucket(size_t bucket_size) {
            if (bucket_size <= 16ULL * 1024 * 1024)
                return CACHE_SIZE_PER_BUCKET;
            if (bucket_size <= 64ULL * 1024 * 1024)
                return 3;
            if (bucket_size <= 256ULL * 1024 * 1024)
                return 2;
            return 1;
        }

        static size_t cache_budget_for_total_memory(size_t total_bytes) {
            if (total_bytes == 0)
                return MAX_CACHE_BUDGET;
            return std::clamp(total_bytes / 96, MIN_CACHE_BUDGET, MAX_CACHE_BUDGET);
        }

        void* try_allocate_cached(size_t bytes, cudaStream_t stream = nullptr) {
            LFS_CUDA_BREADCRUMB_STREAM("tensor.bucket.allocate", stream);
            const size_t bucket_size = get_bucket_size(bytes);
            const size_t bucket_idx = get_bucket_index(bucket_size);
            if (bucket_idx >= NUM_BUCKETS)
                return nullptr;

            {
                std::lock_guard<std::mutex> lock(buckets_[bucket_idx].mutex);
                Bucket& bucket = buckets_[bucket_idx];
                bucket.bucket_size = bucket_size;
                if (!bucket.cache.empty()) {
                    size_t pick = bucket.cache.size() - 1;
                    for (size_t i = bucket.cache.size(); i-- > 0;) {
                        if (bucket.cache[i].stream == stream) {
                            pick = i;
                            break;
                        }
                    }
                    const CachedBlock block = bucket.cache[pick];
                    bucket.cache.erase(bucket.cache.begin() + pick);
                    if (block.stream != stream) {
                        bridgeStreams(block.stream, stream);
                        stats_.cross_stream_reuse.fetch_add(1, std::memory_order_relaxed);
                    }
                    bucket.cached_bytes -= bucket_size;
                    bucket.hits++;
                    bucket.last_hit_epoch = reuse_epoch_.fetch_add(1, std::memory_order_relaxed) + 1;
                    stats_.cache_hits.fetch_add(1, std::memory_order_relaxed);
                    stats_.bytes_cached.fetch_sub(bucket_size, std::memory_order_relaxed);
                    stats_.bytes_wasted.fetch_add(bucket_size - bytes, std::memory_order_relaxed);
                    publish_cache_bytes();
                    return block.ptr;
                }
                bucket.misses++;
            }
            stats_.cache_misses.fetch_add(1, std::memory_order_relaxed);
            return nullptr;
        }

        // `stream` must be the stream the block's last use is ordered on.
        bool cache_free(void* ptr, size_t bytes, cudaStream_t stream = nullptr) {
            const size_t bucket_size = get_bucket_size(bytes);
            const size_t bucket_idx = get_bucket_index(bucket_size);
            if (bucket_idx >= NUM_BUCKETS)
                return false;

            {
                std::lock_guard<std::mutex> lock(buckets_[bucket_idx].mutex);
                Bucket& bucket = buckets_[bucket_idx];
                bucket.bucket_size = bucket_size;

                const size_t budget = current_cache_budget();
                const bool large_probationary_buffer =
                    bucket_size > budget / 2 && bucket.hits == 0 && bucket.misses < 2;
                if (large_probationary_buffer) {
                    const cudaError_t free_status = cudaFreeAsync(ptr, stream);
                    if (free_status != cudaSuccess) {
                        ensure_cuda_success(
                            free_status, "cudaFreeAsync(size-bucket probationary block)",
                            ::lfs::core::detail::format_cuda_safe("ptr={}, bytes={}, stream={}", ptr, bucket_size,
                                                                  static_cast<void*>(stream)),
                            LFS_SOURCE_SITE_CURRENT(), CudaFailureDisposition::LogOnlyNoLatch);
                    }
                    publish_cache_bytes();
                    return true;
                }

                const size_t max_entries = max_cached_entries_for_bucket(bucket_size);
                while (bucket.cache.size() >= max_entries) {
                    const CachedBlock old = bucket.cache.front();
                    bucket.cache.erase(bucket.cache.begin());
                    bucket.cached_bytes -= bucket_size;
                    stats_.bytes_cached.fetch_sub(bucket_size, std::memory_order_relaxed);
                    LFS_CUDA_BREADCRUMB_STREAM_ARGS(
                        "tensor.bucket.evict", old.stream,
                        reinterpret_cast<uintptr_t>(old.ptr), bucket_size,
                        reinterpret_cast<uintptr_t>(old.stream));
                    const cudaError_t free_status = cudaFreeAsync(old.ptr, old.stream);
                    if (free_status != cudaSuccess) {
                        ensure_cuda_success(
                            free_status, "cudaFreeAsync(size-bucket entry eviction)",
                            ::lfs::core::detail::format_cuda_safe("ptr={}, bytes={}, stream={}", old.ptr, bucket_size,
                                                                  static_cast<void*>(old.stream)),
                            LFS_SOURCE_SITE_CURRENT(), CudaFailureDisposition::LogOnlyNoLatch);
                    }
                }

                bucket.cache.push_back({ptr, stream});
                bucket.cached_bytes += bucket_size;
                stats_.free_count.fetch_add(1, std::memory_order_relaxed);
                stats_.bytes_cached.fetch_add(bucket_size, std::memory_order_relaxed);
            }

            enforce_cache_budget();
            publish_cache_bytes();
            return true;
        }

        void* allocate(size_t bytes, cudaStream_t stream = nullptr) {
            void* ptr = try_allocate_cached(bytes, stream);
            if (ptr)
                return ptr;

            const size_t bucket_size = get_bucket_size(bytes);
            cudaError_t err = cudaMallocAsync(&ptr, bucket_size, stream);
            if (err != cudaSuccess) {
                trim_cache();
                err = cudaMallocAsync(&ptr, bucket_size, stream);
                if (err != cudaSuccess) {
                    ensure_cuda_success(err, "cudaMallocAsync(size bucket retry)",
                                        ::lfs::core::detail::format_cuda_safe("bucket_bytes={}", bucket_size),
                                        LFS_SOURCE_SITE_CURRENT(),
                                        CudaFailureDisposition::LogOnly);
                    cudaGetLastError(); // Clear sticky error state for clean recovery
                    return nullptr;
                }
            }
            stats_.alloc_count.fetch_add(1, std::memory_order_relaxed);
            stats_.bytes_wasted.fetch_add(bucket_size - bytes, std::memory_order_relaxed);
            return ptr;
        }

        void deallocate(void* ptr, size_t bytes, cudaStream_t stream = nullptr) {
            LFS_CUDA_BREADCRUMB_STREAM("tensor.bucket.free", stream);
            if (!ptr)
                return;
            if (!cache_free(ptr, bytes, stream)) {
                const cudaError_t free_status = cudaFreeAsync(ptr, stream);
                if (free_status != cudaSuccess) {
                    ensure_cuda_success(
                        free_status, "cudaFreeAsync(size-bucket uncached block)",
                        ::lfs::core::detail::format_cuda_safe("ptr={}, bytes={}, stream={}", ptr, bytes,
                                                              static_cast<void*>(stream)),
                        LFS_SOURCE_SITE_CURRENT(), CudaFailureDisposition::LogOnlyNoLatch);
                }
            }
        }

        // Retags cached entries from one stream to another. Caller must have
        // synchronized `from` first.
        void retag_stream(cudaStream_t from, cudaStream_t to) {
            for (size_t i = 0; i < NUM_BUCKETS; ++i) {
                std::lock_guard<std::mutex> lock(buckets_[i].mutex);
                for (CachedBlock& block : buckets_[i].cache) {
                    if (block.stream == from) {
                        block.stream = to;
                    }
                }
            }
        }

        // Caller must have synchronized the device first; after that cached
        // blocks no longer need their last-use stream to free safely.
        void retag_all_streams(cudaStream_t to) {
            for (size_t i = 0; i < NUM_BUCKETS; ++i) {
                std::lock_guard<std::mutex> lock(buckets_[i].mutex);
                for (CachedBlock& block : buckets_[i].cache) {
                    block.stream = to;
                }
            }
        }

        void trim_cache() {
            for (size_t i = 0; i < NUM_BUCKETS; ++i) {
                std::lock_guard<std::mutex> lock(buckets_[i].mutex);
                for (const CachedBlock& block : buckets_[i].cache) {
                    // Stream-ordered free: these were cudaMallocAsync'd and a block
                    // last used on a non-default stream may still have pending work,
                    // which a plain cudaFree would not be ordered against.
                    LFS_CUDA_BREADCRUMB_STREAM_ARGS(
                        "tensor.bucket.trim", block.stream,
                        reinterpret_cast<uintptr_t>(block.ptr), buckets_[i].bucket_size,
                        reinterpret_cast<uintptr_t>(block.stream));
                    const cudaError_t free_status = cudaFreeAsync(block.ptr, block.stream);
                    if (free_status != cudaSuccess) {
                        ensure_cuda_success(
                            free_status, "cudaFreeAsync(size-bucket cache trim)",
                            ::lfs::core::detail::format_cuda_safe("ptr={}, bytes={}, stream={}", block.ptr,
                                                                  buckets_[i].bucket_size,
                                                                  static_cast<void*>(block.stream)),
                            LFS_SOURCE_SITE_CURRENT(), CudaFailureDisposition::LogOnlyNoLatch);
                    }
                }
                buckets_[i].cache.clear();
                buckets_[i].cached_bytes = 0;
            }
            stats_.bytes_cached.store(0, std::memory_order_relaxed);
            publish_cache_bytes();
        }

        const Stats& stats() const { return stats_; }

        // Fault-isolated policy hook for allocator regression tests. Zero
        // restores automatic device-sized budgeting on the next cache use.
        void set_cache_budget_for_testing(const size_t bytes) {
            cache_budget_bytes_.store(bytes, std::memory_order_release);
            if (bytes != 0) {
                enforce_cache_budget();
                publish_cache_bytes();
            }
        }

        void print_stats() const {
            uint64_t hits = stats_.cache_hits.load();
            uint64_t misses = stats_.cache_misses.load();
            double hit_rate = (hits + misses > 0) ? (100.0 * hits / (hits + misses)) : 0.0;

            LOG_INFO("SizeBucketedPool Statistics:");
            LOG_INFO("  Cache hits: {} ({:.1f}%)", hits, hit_rate);
            LOG_INFO("  Cache misses: {}", misses);
            LOG_INFO("  Bytes cached: {:.2f} MB", stats_.bytes_cached.load() / (1024.0 * 1024.0));
            LOG_INFO("  Bytes wasted (rounding): {:.2f} MB", stats_.bytes_wasted.load() / (1024.0 * 1024.0));
        }

        // Calculate waste percentage for a given size
        static double get_waste_percentage(size_t bytes) {
            size_t bucket = get_bucket_size(bytes);
            return 100.0 * (bucket - bytes) / bucket;
        }

        SizeBucketedPool(const SizeBucketedPool&) = delete;
        SizeBucketedPool& operator=(const SizeBucketedPool&) = delete;

    private:
        // Publish the live reuse-cache size so the HUD can split it out of
        // cuda.pool.untracked_used: these are freed-but-retained cudaMallocAsync
        // buffers, still in the pool's UsedMemCurrent but dropped from the tensor
        // allocator's live map. Reclaimable via trim_cache().
        void publish_cache_bytes() const {
            lfs::diagnostics::VramProfiler::instance().setCudaPoolBucketCacheBytes(
                stats_.bytes_cached.load(std::memory_order_relaxed));
        }

        size_t current_cache_budget() {
            const size_t cached = cache_budget_bytes_.load(std::memory_order_acquire);
            if (cached != 0)
                return cached;

            size_t free_bytes = 0;
            size_t total_bytes = 0;
            size_t budget = MAX_CACHE_BUDGET;
            const cudaError_t memory_status = cudaMemGetInfo(&free_bytes, &total_bytes);
            if (memory_status == cudaSuccess) {
                budget = cache_budget_for_total_memory(total_bytes);
            } else {
                ensure_cuda_success(
                    memory_status, "cudaMemGetInfo(size-bucket cache budget)",
                    ::lfs::core::detail::format_cuda_safe("fallback_budget_bytes={}", budget),
                    LFS_SOURCE_SITE_CURRENT(), CudaFailureDisposition::LogOnly);
            }

            size_t expected = 0;
            if (cache_budget_bytes_.compare_exchange_strong(expected, budget, std::memory_order_release)) {
                return budget;
            }
            return cache_budget_bytes_.load(std::memory_order_acquire);
        }

        size_t choose_eviction_bucket() {
            size_t best = NUM_BUCKETS;
            uint64_t best_epoch = std::numeric_limits<uint64_t>::max();
            size_t best_bucket_size = 0;

            for (size_t i = 0; i < NUM_BUCKETS; ++i) {
                std::lock_guard<std::mutex> lock(buckets_[i].mutex);
                const Bucket& bucket = buckets_[i];
                if (bucket.cache.empty() || bucket.bucket_size == 0)
                    continue;

                const uint64_t epoch = bucket.last_hit_epoch;
                if (best == NUM_BUCKETS ||
                    epoch < best_epoch ||
                    (epoch == best_epoch && bucket.bucket_size > best_bucket_size)) {
                    best = i;
                    best_epoch = epoch;
                    best_bucket_size = bucket.bucket_size;
                }
            }

            return best;
        }

        void enforce_cache_budget() {
            const size_t budget = current_cache_budget();
            while (stats_.bytes_cached.load(std::memory_order_relaxed) > budget) {
                const size_t victim_idx = choose_eviction_bucket();
                if (victim_idx >= NUM_BUCKETS)
                    break;

                CachedBlock victim{};
                size_t victim_size = 0;
                {
                    std::lock_guard<std::mutex> lock(buckets_[victim_idx].mutex);
                    Bucket& bucket = buckets_[victim_idx];
                    if (bucket.cache.empty() || bucket.bucket_size == 0)
                        continue;
                    victim = bucket.cache.front();
                    bucket.cache.erase(bucket.cache.begin());
                    victim_size = bucket.bucket_size;
                    bucket.cached_bytes -= victim_size;
                    stats_.bytes_cached.fetch_sub(victim_size, std::memory_order_relaxed);
                }

                LFS_CUDA_BREADCRUMB_STREAM_ARGS(
                    "tensor.bucket.budget_evict", victim.stream,
                    reinterpret_cast<uintptr_t>(victim.ptr), victim_size,
                    reinterpret_cast<uintptr_t>(victim.stream));
                const cudaError_t free_status = cudaFreeAsync(victim.ptr, victim.stream);
                if (free_status != cudaSuccess) {
                    ensure_cuda_success(
                        free_status, "cudaFreeAsync(size-bucket budget eviction)",
                        ::lfs::core::detail::format_cuda_safe("ptr={}, bytes={}, stream={}", victim.ptr, victim_size,
                                                              static_cast<void*>(victim.stream)),
                        LFS_SOURCE_SITE_CURRENT(), CudaFailureDisposition::LogOnlyNoLatch);
                }
            }
        }

        struct CachedBlock {
            void* ptr = nullptr;
            cudaStream_t stream = nullptr;
        };

        struct Bucket {
            std::vector<CachedBlock> cache;
            std::mutex mutex;
            size_t cached_bytes{0};
            size_t bucket_size{0};
            uint64_t hits{0};
            uint64_t misses{0};
            uint64_t last_hit_epoch{0};

            Bucket() {
                cache.reserve(CACHE_SIZE_PER_BUCKET);
            }
        };

        SizeBucketedPool() = default;

        ~SizeBucketedPool() {
            shutdown();
        }

        std::array<Bucket, NUM_BUCKETS> buckets_;
        std::atomic<bool> shutdown_{false};
        std::atomic<size_t> cache_budget_bytes_{0};
        std::atomic<uint64_t> reuse_epoch_{1};
        Stats stats_;
    };

} // namespace lfs::core

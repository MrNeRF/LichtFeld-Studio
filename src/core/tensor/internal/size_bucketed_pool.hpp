/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/logger.hpp"
#include <cuda_runtime.h>
#include <atomic>
#include <array>
#include <vector>
#include <mutex>
#include <cstdint>
#include <algorithm>

namespace lfs::core {

/**
 * @brief Size-bucketed memory pool for medium and large allocations
 *
 * PROBLEM:
 * Tensors like (20M, 3, 15) = 3.6GB create fragmentation because each
 * slightly different size (3.6GB, 3.5GB, 3.7GB) fragments the memory pool.
 *
 * SOLUTION:
 * Round allocations to bucket boundaries and cache freed memory per bucket.
 * This dramatically increases reuse:
 *   - 3.58GB and 3.61GB both round to 3.75GB bucket -> same memory reused
 *   - Reduces unique allocation sizes from thousands to ~50 buckets
 *
 * BUCKET SIZES:
 *   256KB - 1MB:   256KB increments  (4 buckets)
 *   1MB - 16MB:    1MB increments    (16 buckets)
 *   16MB - 256MB:  16MB increments   (16 buckets)
 *   256MB - 1GB:   64MB increments   (12 buckets)
 *   1GB - 8GB:     256MB increments  (28 buckets)
 *   > 8GB:         1GB increments    (unlimited)
 *
 * RECENT-FREE CACHE:
 *   Each bucket keeps the last N freed pointers for instant reuse.
 *   This bypasses cudaMallocAsync entirely for repeated allocations.
 *
 * WASTE vs FRAGMENTATION TRADEOFF:
 *   - Maximum waste per allocation: ~25% (e.g., 257MB rounds to 320MB)
 *   - But fragmentation is nearly eliminated for repeated workloads
 *   - Net effect: LESS total memory used due to better reuse
 */
class SizeBucketedPool {
public:
    // Configuration
    static constexpr size_t MIN_BUCKET_SIZE = 256 * 1024;      // 256 KB minimum
    static constexpr size_t MAX_TRACKED_SIZE = 16ULL * 1024 * 1024 * 1024;  // 16 GB
    static constexpr size_t CACHE_SIZE_PER_BUCKET = 4;         // Keep 4 freed ptrs per bucket
    static constexpr size_t NUM_BUCKETS = 128;                 // Max buckets

    struct Stats {
        std::atomic<uint64_t> cache_hits{0};
        std::atomic<uint64_t> cache_misses{0};
        std::atomic<uint64_t> alloc_count{0};
        std::atomic<uint64_t> free_count{0};
        std::atomic<uint64_t> bytes_cached{0};
        std::atomic<uint64_t> bytes_wasted{0};  // Due to rounding
    };

    static SizeBucketedPool& instance() {
        static SizeBucketedPool pool;
        return pool;
    }

    /**
     * @brief Get the bucket size for a given allocation
     * Returns the rounded-up size that will actually be allocated
     */
    static size_t get_bucket_size(size_t bytes) {
        if (bytes <= MIN_BUCKET_SIZE) {
            return MIN_BUCKET_SIZE;
        }

        // Bucket boundaries with increasing granularity
        if (bytes <= 1024 * 1024) {
            // 256KB - 1MB: 256KB increments
            return ((bytes + 256 * 1024 - 1) / (256 * 1024)) * (256 * 1024);
        }
        if (bytes <= 16 * 1024 * 1024) {
            // 1MB - 16MB: 1MB increments
            return ((bytes + 1024 * 1024 - 1) / (1024 * 1024)) * (1024 * 1024);
        }
        if (bytes <= 256 * 1024 * 1024) {
            // 16MB - 256MB: 16MB increments
            return ((bytes + 16 * 1024 * 1024 - 1) / (16 * 1024 * 1024)) * (16 * 1024 * 1024);
        }
        if (bytes <= 1024ULL * 1024 * 1024) {
            // 256MB - 1GB: 64MB increments
            return ((bytes + 64 * 1024 * 1024 - 1) / (64 * 1024 * 1024)) * (64 * 1024 * 1024);
        }
        if (bytes <= 8ULL * 1024 * 1024 * 1024) {
            // 1GB - 8GB: 256MB increments
            return ((bytes + 256ULL * 1024 * 1024 - 1) / (256ULL * 1024 * 1024)) * (256ULL * 1024 * 1024);
        }
        // > 8GB: 1GB increments
        return ((bytes + 1024ULL * 1024 * 1024 - 1) / (1024ULL * 1024 * 1024)) * (1024ULL * 1024 * 1024);
    }

    /**
     * @brief Get bucket index for a given size
     */
    static size_t get_bucket_index(size_t bucket_size) {
        if (bucket_size <= 1024 * 1024) {
            // 256KB - 1MB: buckets 0-3
            return (bucket_size / (256 * 1024)) - 1;
        }
        if (bucket_size <= 16 * 1024 * 1024) {
            // 1MB - 16MB: buckets 4-19
            return 4 + (bucket_size / (1024 * 1024)) - 1;
        }
        if (bucket_size <= 256 * 1024 * 1024) {
            // 16MB - 256MB: buckets 20-35
            return 20 + (bucket_size / (16 * 1024 * 1024)) - 1;
        }
        if (bucket_size <= 1024ULL * 1024 * 1024) {
            // 256MB - 1GB: buckets 36-47
            return 36 + (bucket_size / (64 * 1024 * 1024)) - 4;
        }
        if (bucket_size <= 8ULL * 1024 * 1024 * 1024) {
            // 1GB - 8GB: buckets 48-75
            return 48 + (bucket_size / (256ULL * 1024 * 1024)) - 4;
        }
        // > 8GB: buckets 76+
        size_t idx = 76 + (bucket_size / (1024ULL * 1024 * 1024)) - 8;
        return std::min(idx, NUM_BUCKETS - 1);
    }

    /**
     * @brief Try to allocate from the cache
     * @return Pointer if found in cache, nullptr otherwise
     */
    void* try_allocate_cached(size_t bytes) {
        size_t bucket_size = get_bucket_size(bytes);
        size_t bucket_idx = get_bucket_index(bucket_size);

        if (bucket_idx >= NUM_BUCKETS) {
            return nullptr;
        }

        // Try to pop from cache
        {
            std::lock_guard<std::mutex> lock(buckets_[bucket_idx].mutex);
            if (!buckets_[bucket_idx].cache.empty()) {
                void* ptr = buckets_[bucket_idx].cache.back();
                buckets_[bucket_idx].cache.pop_back();
                buckets_[bucket_idx].cached_bytes -= bucket_size;

                stats_.cache_hits.fetch_add(1, std::memory_order_relaxed);
                stats_.bytes_cached.fetch_sub(bucket_size, std::memory_order_relaxed);
                stats_.bytes_wasted.fetch_add(bucket_size - bytes, std::memory_order_relaxed);

                return ptr;
            }
        }

        stats_.cache_misses.fetch_add(1, std::memory_order_relaxed);
        return nullptr;
    }

    /**
     * @brief Cache a freed pointer for later reuse
     * @return true if cached, false if cache is full (caller should cudaFree)
     */
    bool cache_free(void* ptr, size_t bytes) {
        size_t bucket_size = get_bucket_size(bytes);
        size_t bucket_idx = get_bucket_index(bucket_size);

        if (bucket_idx >= NUM_BUCKETS) {
            return false;
        }

        std::lock_guard<std::mutex> lock(buckets_[bucket_idx].mutex);

        // Check if cache is full
        if (buckets_[bucket_idx].cache.size() >= CACHE_SIZE_PER_BUCKET) {
            // Evict oldest entry
            void* old_ptr = buckets_[bucket_idx].cache.front();
            buckets_[bucket_idx].cache.erase(buckets_[bucket_idx].cache.begin());
            buckets_[bucket_idx].cached_bytes -= bucket_size;
            stats_.bytes_cached.fetch_sub(bucket_size, std::memory_order_relaxed);

            // Free the evicted pointer
            cudaFreeAsync(old_ptr, nullptr);
        }

        buckets_[bucket_idx].cache.push_back(ptr);
        buckets_[bucket_idx].cached_bytes += bucket_size;

        stats_.free_count.fetch_add(1, std::memory_order_relaxed);
        stats_.bytes_cached.fetch_add(bucket_size, std::memory_order_relaxed);

        return true;
    }

    /**
     * @brief Allocate with size bucketing
     * @param bytes Requested size
     * @param stream CUDA stream
     * @return Pointer to allocated memory (bucket size, may be larger than bytes)
     */
    void* allocate(size_t bytes, cudaStream_t stream = nullptr) {
        // Try cache first
        void* ptr = try_allocate_cached(bytes);
        if (ptr) {
            return ptr;
        }

        // Allocate from CUDA with bucketed size
        size_t bucket_size = get_bucket_size(bytes);
        cudaError_t err = cudaMallocAsync(&ptr, bucket_size, stream);
        if (err != cudaSuccess) {
            // Try to free some cached memory and retry
            trim_cache();
            err = cudaMallocAsync(&ptr, bucket_size, stream);
            if (err != cudaSuccess) {
                LOG_ERROR("cudaMallocAsync failed for {} bytes: {}", bucket_size, cudaGetErrorString(err));
                return nullptr;
            }
        }

        stats_.alloc_count.fetch_add(1, std::memory_order_relaxed);
        stats_.bytes_wasted.fetch_add(bucket_size - bytes, std::memory_order_relaxed);

        return ptr;
    }

    /**
     * @brief Deallocate (caches for reuse)
     */
    void deallocate(void* ptr, size_t bytes, cudaStream_t stream = nullptr) {
        if (!ptr) return;

        // Try to cache for reuse
        if (!cache_free(ptr, bytes)) {
            // Cache full, free directly
            cudaFreeAsync(ptr, stream);
        }
    }

    /**
     * @brief Trim all cached memory
     */
    void trim_cache() {
        for (size_t i = 0; i < NUM_BUCKETS; i++) {
            std::lock_guard<std::mutex> lock(buckets_[i].mutex);
            for (void* ptr : buckets_[i].cache) {
                cudaFree(ptr);
            }
            buckets_[i].cache.clear();
            buckets_[i].cached_bytes = 0;
        }
        stats_.bytes_cached.store(0, std::memory_order_relaxed);
    }

    const Stats& stats() const { return stats_; }

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
    struct Bucket {
        std::vector<void*> cache;
        std::mutex mutex;
        size_t cached_bytes{0};

        Bucket() {
            cache.reserve(CACHE_SIZE_PER_BUCKET);
        }
    };

    SizeBucketedPool() = default;

    ~SizeBucketedPool() {
        trim_cache();
    }

    std::array<Bucket, NUM_BUCKETS> buckets_;
    Stats stats_;
};

} // namespace lfs::core

/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstddef>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace lfs::core {

    /**
     * @brief Caching allocator for CUDA pinned (page-locked) host memory
     *
     * This allocator uses cudaHostAlloc to create page-locked memory that can be
     * directly accessed by the GPU via DMA, providing 2-3x faster PCIe bandwidth
     * compared to regular pageable memory.
     *
     * Features:
     * - Caches freed blocks to avoid expensive cudaHostAlloc/cudaFreeHost calls
     * - Size-based bucketing for efficient reuse
     * - Thread-safe with per-size-class locking
     * - Similar design to PyTorch's CachingHostAllocator
     *
     * Performance benefits:
     * - CPU->GPU transfer: ~7-11 GB/s (vs ~3 GB/s for regular memory)
     * - cudaMemcpyAsync can overlap with CPU work
     * - Reduced PCIe latency
     */
    class PinnedMemoryAllocator {
    public:
        /**
         * @brief Get the singleton instance
         */
        static PinnedMemoryAllocator& instance();

        /**
         * @brief Allocate pinned host memory
         *
         * First tries to reuse a cached block of the same size class.
         * If no cached block is available, allocates new pinned memory.
         *
         * @param bytes Number of bytes to allocate
         * @return void* Pointer to pinned memory, or nullptr on failure
         */
        void* allocate(size_t bytes);

        /**
         * @brief Deallocate pinned memory (caches it for reuse)
         *
         * Instead of immediately calling cudaFreeHost, the block is cached
         * for potential reuse by future allocations of similar size.
         *
         * @param ptr Pointer to pinned memory to free
         */
        void deallocate(void* ptr);

        /**
         * @brief Clear all cached blocks and free them to the system
         *
         * Useful for reducing memory footprint when no longer needed.
         * Called automatically on shutdown.
         */
        void empty_cache();

        /**
         * @brief Pre-allocate common tensor sizes to avoid cold-start penalties
         *
         * Allocates and immediately frees pinned memory for common image sizes,
         * warming up the cache so subsequent allocations are instant.
         *
         * This eliminates the cudaHostAlloc penalty (e.g., 23.8ms for 4K) on
         * first use, matching LibTorch's pre-warmed pool performance.
         *
         * Call once during application startup.
         */
        void prewarm();

        /**
         * @brief Get statistics about allocator usage
         */
        struct Stats {
            size_t allocated_bytes{0}; ///< Total bytes currently allocated
            size_t cached_bytes{0};    ///< Total bytes in cache
            size_t num_allocs{0};      ///< Number of allocations
            size_t num_deallocs{0};    ///< Number of deallocations
            size_t cache_hits{0};      ///< Number of times cache was reused
            size_t cache_misses{0};    ///< Number of new allocations
        };

        Stats get_stats() const;
        void reset_stats();

        /**
         * @brief Enable/disable pinned memory (for testing/debugging)
         *
         * When disabled, falls back to regular malloc/free.
         */
        void set_enabled(bool enabled) { enabled_ = enabled; }
        bool is_enabled() const { return enabled_; }

    private:
        PinnedMemoryAllocator() = default;
        ~PinnedMemoryAllocator();

        // Non-copyable, non-movable
        PinnedMemoryAllocator(const PinnedMemoryAllocator&) = delete;
        PinnedMemoryAllocator& operator=(const PinnedMemoryAllocator&) = delete;

        /**
         * @brief Round size up to allocation bucket size
         *
         * Uses power-of-2 rounding for sizes > 4KB to reduce fragmentation.
         * Small sizes (< 4KB) use exact matching.
         */
        static size_t round_size(size_t bytes);

        struct Block {
            void* ptr{nullptr};
            size_t size{0};
        };

        // Cache of free blocks organized by size
        // Key: rounded size, Value: list of available blocks
        std::unordered_map<size_t, std::vector<Block>> cache_;

        // Track all allocated blocks (for deallocation lookup)
        std::unordered_map<void*, size_t> allocated_blocks_;

        mutable std::mutex mutex_;
        Stats stats_;
        bool enabled_{true}; // Can disable for A/B testing
    };

} // namespace lfs::core

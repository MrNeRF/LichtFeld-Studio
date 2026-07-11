/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <list>
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
     * - LRU eviction under an LFS_PINNED_CACHE_LIMIT_MB byte budget (1 GiB default)
     * - Thread-safe allocation and cache operations
     * - Similar design to PyTorch's CachingHostAllocator
     *
     * Performance benefits:
     * - CPU->GPU transfer: ~7-11 GB/s (vs ~3 GB/s for regular memory)
     * - cudaMemcpyAsync can overlap with CPU work
     * - Reduced PCIe latency
     */
    class LFS_CORE_API PinnedMemoryAllocator {
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
         * STREAM-AWARE: Records a pooled CUDA event on `stream` and on every
         * stream registered via record_stream(). The cached block is not reused
         * until all of those events have completed.
         *
         * @param ptr Pointer to pinned memory to free
         * @param stream CUDA stream that last used this memory (nullptr = default stream)
         */
        void deallocate(void* ptr, cudaStream_t stream = nullptr);

        /**
         * @brief Marks `ptr` as used by an additional stream (H2D on one stream,
         * D2H on another). The free defers reuse until that stream passes the use.
         */
        void record_stream(void* ptr, cudaStream_t stream);

        /**
         * @brief Severs references to `stream` before it is destroyed: waits for
         * its pending work, then drops it from all recorded uses.
         */
        void release_stream(cudaStream_t stream);

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
         * @brief Explicitly shut down the allocator and release all resources
         *
         * Should be called before CUDA context destruction to ensure safe cleanup.
         * Safe to call multiple times.
         */
        void shutdown();

        /**
         * @brief Get statistics about allocator usage
         */
        struct Stats {
            size_t allocated_bytes{0}; ///< Bytes held by live allocations
            size_t cached_bytes{0};    ///< Bytes retained for reuse
            size_t peak_allocated_bytes{0};
            size_t peak_cached_bytes{0};
            size_t peak_total_bytes{0};
            size_t num_allocs{0}; ///< New backend allocations
            size_t num_deallocs{0};
            size_t cache_hits{0};
            size_t cache_misses{0};
            size_t cuda_host_allocs{0};
            size_t cuda_host_frees{0};
            size_t malloc_fallback_allocs{0};
            size_t malloc_fallback_frees{0};
            size_t evicted_blocks{0};
            size_t evicted_bytes{0};
        };

        Stats get_stats() const;
        void reset_stats();

        /**
         * @brief Enable/disable pinned memory (for testing/debugging)
         *
         * When disabled, falls back to regular malloc/free.
         */
        void set_enabled(bool enabled) { enabled_.store(enabled, std::memory_order_release); }
        bool is_enabled() const { return enabled_.load(std::memory_order_acquire); }

        /** Internal fault-injection and cache-policy hooks used by regression tests. */
        void set_force_fallback_for_testing(bool force) {
            force_fallback_for_testing_.store(force, std::memory_order_release);
        }
        void set_cache_limit_for_testing(size_t bytes);
        size_t cache_limit_bytes() const;

    private:
        PinnedMemoryAllocator();
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

        enum class Backend : uint8_t {
            CudaHost,
            MallocFallback,
        };

        struct Block {
            void* ptr{nullptr};
            size_t size{0};
            Backend backend{Backend::CudaHost};
            uint64_t last_used{0};
            // Pooled events, one per stream that used this memory; the block is
            // safe to reuse once every event has completed. Events stay valid
            // after their recording stream is destroyed.
            std::vector<cudaEvent_t> ready_events;

            Block() = default;
            Block(void* p, size_t s, Backend allocation_backend)
                : ptr(p),
                  size(s),
                  backend(allocation_backend) {}

            ~Block();

            Block(Block&& other) noexcept;
            Block& operator=(Block&& other) noexcept;
            Block(const Block&) = delete;
            Block& operator=(const Block&) = delete;

            bool all_uses_complete() const;
            void release_events();
        };

        struct AllocationInfo {
            size_t size{0};
            Backend backend{Backend::CudaHost};
            std::vector<cudaStream_t> extra_streams;
        };

        // Cache of free blocks organized by size
        // Key: rounded size, Value: list of available blocks
        std::unordered_map<size_t, std::list<Block>> cache_;

        // Track all allocated blocks (for deallocation lookup)
        std::unordered_map<void*, AllocationInfo> allocated_blocks_;

        void update_peaks_locked();
        void publish_stats_locked() const;
        std::vector<Block> take_evictions_locked();
        void release_blocks(std::vector<Block> blocks, bool count_as_evictions);
        void empty_cache_impl(bool publish_stats);
        static bool record_uses(Block& block, const std::vector<cudaStream_t>& streams);

        mutable std::mutex mutex_;
        Stats stats_;
        size_t cache_limit_bytes_{0};
        uint64_t lru_clock_{0};
        std::atomic<bool> enabled_{true}; // Can disable for A/B testing
        std::atomic<bool> force_fallback_for_testing_{false};
        std::atomic<bool> shutdown_{false};
    };

} // namespace lfs::core

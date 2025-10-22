/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core_new/pinned_memory_allocator.hpp"
#include "core/logger.hpp"
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                              \
    do {                                              \
        cudaError_t error = call;                     \
        if (error != cudaSuccess) {                   \
            LOG_ERROR("CUDA error at {}:{} - {}: {}", \
                      __FILE__, __LINE__,             \
                      cudaGetErrorName(error),        \
                      cudaGetErrorString(error));     \
        }                                             \
    } while (0)

namespace lfs::core {

    PinnedMemoryAllocator& PinnedMemoryAllocator::instance() {
        static PinnedMemoryAllocator instance;
        return instance;
    }

    PinnedMemoryAllocator::~PinnedMemoryAllocator() {
        empty_cache();
    }

    size_t PinnedMemoryAllocator::round_size(size_t bytes) {
        // Small allocations: exact size to reduce fragmentation
        if (bytes < 4096) {
            return bytes;
        }

        // Large allocations: round to next power of 2 for better reuse
        // This matches PyTorch's strategy
        if (bytes < (1 << 20)) { // < 1MB: round to 512-byte blocks
            return ((bytes + 511) / 512) * 512;
        } else { // >= 1MB: round to next power of 2
            size_t power = static_cast<size_t>(std::ceil(std::log2(bytes)));
            return 1ULL << power;
        }
    }

    void* PinnedMemoryAllocator::allocate(size_t bytes) {
        if (bytes == 0) {
            return nullptr;
        }

        // Fall back to regular malloc if disabled
        if (!enabled_) {
            return std::malloc(bytes);
        }

        size_t rounded_size = round_size(bytes);

        std::lock_guard<std::mutex> lock(mutex_);

        // Try to reuse a cached block
        auto it = cache_.find(rounded_size);
        if (it != cache_.end() && !it->second.empty()) {
            // Cache hit! Reuse existing pinned block
            Block block = it->second.back();
            it->second.pop_back();

            allocated_blocks_[block.ptr] = block.size;
            stats_.allocated_bytes += block.size;
            stats_.cache_hits++;

            LOG_TRACE("Pinned memory cache HIT: {} bytes (total allocated: {} MB)",
                      bytes, stats_.allocated_bytes / (1024.0 * 1024.0));

            return block.ptr;
        }

        // Cache miss - need to allocate new pinned memory
        void* ptr = nullptr;
        cudaError_t err = cudaHostAlloc(&ptr, rounded_size, cudaHostAllocDefault);

        if (err != cudaSuccess) {
            LOG_ERROR("cudaHostAlloc failed for {} bytes: {}",
                      rounded_size, cudaGetErrorString(err));
            // Fall back to regular malloc as last resort
            ptr = std::malloc(rounded_size);
            if (!ptr) {
                LOG_ERROR("Fallback malloc also failed for {} bytes", rounded_size);
                return nullptr;
            }
            LOG_WARN("Falling back to regular malloc for {} bytes", rounded_size);
        }

        allocated_blocks_[ptr] = rounded_size;
        stats_.allocated_bytes += rounded_size;
        stats_.num_allocs++;
        stats_.cache_misses++;

        LOG_TRACE("Pinned memory cache MISS: allocated {} bytes (total: {} MB, {} allocs)",
                  bytes, stats_.allocated_bytes / (1024.0 * 1024.0), stats_.num_allocs);

        return ptr;
    }

    void PinnedMemoryAllocator::deallocate(void* ptr) {
        if (!ptr) {
            return;
        }

        // Fall back to regular free if disabled
        if (!enabled_) {
            std::free(ptr);
            return;
        }

        std::lock_guard<std::mutex> lock(mutex_);

        // Find the block size
        auto it = allocated_blocks_.find(ptr);
        if (it == allocated_blocks_.end()) {
            LOG_WARN("Attempted to free unknown pinned memory pointer: {}", ptr);
            // Try regular free as fallback
            std::free(ptr);
            return;
        }

        size_t size = it->second;
        allocated_blocks_.erase(it);
        stats_.allocated_bytes -= size;
        stats_.num_deallocs++;

        // Cache the block for reuse instead of freeing immediately
        Block block{ptr, size};
        cache_[size].push_back(block);
        stats_.cached_bytes += size;

        LOG_TRACE("Pinned memory cached: {} bytes (cache size: {} MB, {} blocks)",
                  size, stats_.cached_bytes / (1024.0 * 1024.0),
                  cache_[size].size());
    }

    void PinnedMemoryAllocator::empty_cache() {
        std::lock_guard<std::mutex> lock(mutex_);

        size_t freed_bytes = 0;
        size_t freed_blocks = 0;

        // Free all cached blocks
        for (auto& [size, blocks] : cache_) {
            for (const auto& block : blocks) {
                CHECK_CUDA(cudaFreeHost(block.ptr));
                freed_bytes += block.size;
                freed_blocks++;
            }
        }

        cache_.clear();
        stats_.cached_bytes = 0;

        if (freed_blocks > 0) {
            LOG_DEBUG("Freed pinned memory cache: {} MB in {} blocks",
                      freed_bytes / (1024.0 * 1024.0), freed_blocks);
        }
    }

    PinnedMemoryAllocator::Stats PinnedMemoryAllocator::get_stats() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return stats_;
    }

    void PinnedMemoryAllocator::reset_stats() {
        std::lock_guard<std::mutex> lock(mutex_);
        stats_ = Stats{};
    }

    void PinnedMemoryAllocator::prewarm() {
        LOG_INFO("Pre-warming pinned memory cache with common sizes...");

        // Pre-allocate sizes matching common image resolutions (HxWxC in float32)
        // Based on profiling data from permute+upload benchmark
        std::vector<size_t> common_sizes = {
            // Small images
            540 * 540 * 3 * 4, // 3.34 MB - Square HD
            720 * 820 * 3 * 4, // 6.76 MB - Production size

            // Full HD / 2K
            1080 * 1920 * 3 * 4, // 23.73 MB - Full HD
            1088 * 1920 * 3 * 4, // 23.91 MB - Actual log size

            // 4K
            2160 * 3840 * 3 * 4, // 94.92 MB - 4K UHD

            // Additional common sizes for good measure
            1 * 1024 * 1024,   // 1 MB - Small tensors
            10 * 1024 * 1024,  // 10 MB - Medium tensors
            50 * 1024 * 1024,  // 50 MB - Large tensors
            128 * 1024 * 1024, // 128 MB - Very large tensors
        };

        size_t total_prewarmed = 0;
        auto start_time = std::chrono::high_resolution_clock::now();

        for (size_t size : common_sizes) {
            void* ptr = allocate(size);
            if (ptr) {
                deallocate(ptr); // Immediately free to cache
                total_prewarmed += round_size(size);
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        LOG_INFO("Pre-warming complete: {} MB cached in {} sizes ({} ms)",
                 total_prewarmed / (1024.0 * 1024.0),
                 common_sizes.size(),
                 duration.count());

        // Log the stats
        auto stats = get_stats();
        LOG_DEBUG("  Cache hits: {}, misses: {}, cached bytes: {} MB",
                  stats.cache_hits, stats.cache_misses,
                  stats.cached_bytes / (1024.0 * 1024.0));
    }

#undef CHECK_CUDA

} // namespace lfs::core

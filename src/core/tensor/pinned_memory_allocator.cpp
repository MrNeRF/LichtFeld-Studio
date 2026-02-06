/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/pinned_memory_allocator.hpp"
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

    // Block implementation
    PinnedMemoryAllocator::Block::Block(void* p, size_t s, cudaStream_t stream)
        : ptr(p),
          size(s),
          last_stream(stream) {
        if (ptr) {
            cudaError_t err = cudaEventCreate(&ready_event);
            if (err != cudaSuccess) {
                LOG_ERROR("cudaEventCreate failed: {}", cudaGetErrorString(err));
                ready_event = nullptr;
            }
        }
    }

    PinnedMemoryAllocator::Block::~Block() {
        if (ready_event) {
            cudaError_t err = cudaEventDestroy(ready_event);
            if (err != cudaSuccess) {
                LOG_ERROR("cudaEventDestroy failed: {}", cudaGetErrorString(err));
            }
        }
    }

    PinnedMemoryAllocator::Block::Block(Block&& other) noexcept
        : ptr(other.ptr),
          size(other.size),
          last_stream(other.last_stream),
          ready_event(other.ready_event) {
        other.ptr = nullptr;
        other.size = 0;
        other.last_stream = nullptr;
        other.ready_event = nullptr;
    }

    PinnedMemoryAllocator::Block& PinnedMemoryAllocator::Block::operator=(Block&& other) noexcept {
        if (this != &other) {
            // Clean up existing event
            if (ready_event) {
                cudaEventDestroy(ready_event);
            }

            // Move data
            ptr = other.ptr;
            size = other.size;
            last_stream = other.last_stream;
            ready_event = other.ready_event;

            // Clear other
            other.ptr = nullptr;
            other.size = 0;
            other.last_stream = nullptr;
            other.ready_event = nullptr;
        }
        return *this;
    }

    PinnedMemoryAllocator& PinnedMemoryAllocator::instance() {
        static PinnedMemoryAllocator instance;
        return instance;
    }

    PinnedMemoryAllocator::~PinnedMemoryAllocator() {
        shutdown();
    }

    void PinnedMemoryAllocator::shutdown() {
        bool expected = false;
        if (!shutdown_.compare_exchange_strong(expected, true))
            return;
        LOG_INFO("Shutting down PinnedMemoryAllocator...");
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

        if (shutdown_.load(std::memory_order_acquire)) {
            LOG_ERROR("Attempted to allocate pinned memory after shutdown!");
            return nullptr;
        }

        // Try to reuse a cached block (STREAM-SAFE VERSION)
        auto it = cache_.find(rounded_size);
        if (it != cache_.end() && !it->second.empty()) {
            // Search for a block whose stream has completed
            for (size_t i = 0; i < it->second.size(); ++i) {
                Block& block = it->second[i];

                // Check if stream has completed (non-blocking query)
                cudaError_t status = cudaSuccess;
                if (block.ready_event) {
                    status = cudaEventQuery(block.ready_event);
                }

                if (status == cudaSuccess) {
                    // Stream completed! Safe to reuse this block
                    void* ptr = block.ptr;
                    size_t size = block.size;

                    // Remove from cache (swap with last element for O(1) removal)
                    std::swap(it->second[i], it->second.back());
                    it->second.pop_back();

                    allocated_blocks_[ptr] = size;
                    stats_.allocated_bytes += size;
                    stats_.cached_bytes -= size;
                    stats_.cache_hits++;

                    LOG_TRACE("Pinned memory cache HIT (stream-safe): {} bytes (total allocated: {} MB)",
                              bytes, stats_.allocated_bytes / (1024.0 * 1024.0));

                    return ptr;
                } else if (status != cudaErrorNotReady) {
                    // Unexpected error - log and skip this block
                    LOG_ERROR("cudaEventQuery failed: {}", cudaGetErrorString(status));
                }
                // If cudaErrorNotReady, stream still running - try next block
            }

            // No ready blocks found - fall through to allocate new
            LOG_TRACE("Pinned memory cache MISS (all {} blocks busy): {} bytes", it->second.size(), bytes);
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

        LOG_TRACE("Pinned memory allocated: {} bytes (total: {} MB, {} allocs)",
                  bytes, stats_.allocated_bytes / (1024.0 * 1024.0), stats_.num_allocs);

        return ptr;
    }

    void PinnedMemoryAllocator::deallocate(void* ptr, cudaStream_t stream) {
        if (!ptr) {
            return;
        }

        if (shutdown_.load(std::memory_order_acquire))
            return;

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

        // Create block with stream tracking and record event
        Block block{ptr, size, stream};

        // Record event on the stream to track when memory is safe to reuse
        if (block.ready_event) {
            cudaError_t err = cudaEventRecord(block.ready_event, stream);
            if (err != cudaSuccess) {
                LOG_ERROR("cudaEventRecord failed: {} - memory may not be stream-safe!",
                          cudaGetErrorString(err));
            }
        }

        cache_[size].push_back(std::move(block));
        stats_.cached_bytes += size;

        LOG_TRACE("Pinned memory cached with stream sync: {} bytes (stream: {}, cache size: {} MB)",
                  size, (void*)stream, stats_.cached_bytes / (1024.0 * 1024.0));
    }

    void PinnedMemoryAllocator::empty_cache() {
        std::lock_guard<std::mutex> lock(mutex_);

        size_t freed_bytes = 0;
        size_t freed_blocks = 0;

        // Free all cached blocks
        for (auto& [size, blocks] : cache_) {
            for (auto& block : blocks) {
                // Wait for stream to complete before freeing
                if (block.ready_event) {
                    cudaError_t status = cudaEventSynchronize(block.ready_event);
                    if (status != cudaSuccess) {
                        LOG_ERROR("cudaEventSynchronize failed during cache clear: {}",
                                  cudaGetErrorString(status));
                    }
                }

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

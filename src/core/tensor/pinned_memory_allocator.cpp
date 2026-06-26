/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/pinned_memory_allocator.hpp"
#include "core/logger.hpp"
#include "diagnostics/vram_profiler.hpp"
#include "internal/cuda_event_pool.hpp"
#include <algorithm>
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
    PinnedMemoryAllocator::Block::~Block() {
        release_events();
    }

    PinnedMemoryAllocator::Block::Block(Block&& other) noexcept
        : ptr(other.ptr),
          size(other.size),
          ready_events(std::move(other.ready_events)) {
        other.ptr = nullptr;
        other.size = 0;
        other.ready_events.clear();
    }

    PinnedMemoryAllocator::Block& PinnedMemoryAllocator::Block::operator=(Block&& other) noexcept {
        if (this != &other) {
            release_events();
            ptr = other.ptr;
            size = other.size;
            ready_events = std::move(other.ready_events);
            other.ptr = nullptr;
            other.size = 0;
            other.ready_events.clear();
        }
        return *this;
    }

    bool PinnedMemoryAllocator::Block::all_uses_complete() const {
        for (cudaEvent_t event : ready_events) {
            const cudaError_t status = cudaEventQuery(event);
            if (status == cudaErrorNotReady) {
                return false;
            }
            if (status != cudaSuccess) {
                LOG_ERROR("cudaEventQuery failed: {}", cudaGetErrorString(status));
                return false;
            }
        }
        return true;
    }

    void PinnedMemoryAllocator::Block::release_events() {
        for (cudaEvent_t event : ready_events) {
            CudaEventPool::instance().release(event);
        }
        ready_events.clear();
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

        // Try to reuse a cached block whose recorded uses have all completed
        auto it = cache_.find(rounded_size);
        if (it != cache_.end() && !it->second.empty()) {
            for (size_t i = 0; i < it->second.size(); ++i) {
                Block& block = it->second[i];
                if (block.all_uses_complete()) {
                    block.release_events();
                    void* ptr = block.ptr;
                    size_t size = block.size;

                    // Remove from cache (swap with last element for O(1) removal)
                    std::swap(it->second[i], it->second.back());
                    it->second.pop_back();

                    allocated_blocks_[ptr] = {size, {}};
                    stats_.allocated_bytes += size;
                    stats_.cached_bytes -= size;
                    stats_.cache_hits++;
                    lfs::diagnostics::VramProfiler::instance().setPinnedHostUsed(stats_.allocated_bytes);
                    LFS_COUNTER_ADD("io.pinned_host.cache_hit", 1);

                    return ptr;
                }
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

        allocated_blocks_[ptr] = {rounded_size, {}};
        stats_.allocated_bytes += rounded_size;
        stats_.num_allocs++;
        stats_.cache_misses++;
        lfs::diagnostics::VramProfiler::instance().setPinnedHostUsed(stats_.allocated_bytes);
        LFS_COUNTER_ADD("io.pinned_host.cache_miss", 1);

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

        const size_t size = it->second.size;
        std::vector<cudaStream_t> use_streams = std::move(it->second.extra_streams);
        allocated_blocks_.erase(it);
        stats_.allocated_bytes -= size;
        stats_.num_deallocs++;
        lfs::diagnostics::VramProfiler::instance().setPinnedHostUsed(stats_.allocated_bytes);

        if (std::find(use_streams.begin(), use_streams.end(), stream) == use_streams.end()) {
            use_streams.push_back(stream);
        }

        // One pooled event per using stream; the block is reusable once all complete
        Block block{ptr, size};
        for (cudaStream_t use_stream : use_streams) {
            cudaEvent_t event = CudaEventPool::instance().acquire();
            if (!event) {
                cudaStreamSynchronize(use_stream);
                continue;
            }
            const cudaError_t err = cudaEventRecord(event, use_stream);
            if (err != cudaSuccess) {
                LOG_ERROR("cudaEventRecord failed: {} - memory may not be stream-safe!",
                          cudaGetErrorString(err));
                CudaEventPool::instance().release(event);
                continue;
            }
            block.ready_events.push_back(event);
        }

        cache_[size].push_back(std::move(block));
        stats_.cached_bytes += size;
    }

    void PinnedMemoryAllocator::record_stream(void* ptr, cudaStream_t stream) {
        if (!ptr || !enabled_) {
            return;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = allocated_blocks_.find(ptr);
        if (it == allocated_blocks_.end()) {
            return;
        }
        auto& extras = it->second.extra_streams;
        if (std::find(extras.begin(), extras.end(), stream) == extras.end()) {
            extras.push_back(stream);
        }
    }

    void PinnedMemoryAllocator::release_stream(cudaStream_t stream) {
        if (!stream) {
            return;
        }
        cudaStreamSynchronize(stream);
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& [ptr, info] : allocated_blocks_) {
            std::erase(info.extra_streams, stream);
        }
    }

    void PinnedMemoryAllocator::empty_cache() {
        std::lock_guard<std::mutex> lock(mutex_);

        size_t freed_bytes = 0;
        size_t freed_blocks = 0;

        // Free all cached blocks
        for (auto& [size, blocks] : cache_) {
            for (auto& block : blocks) {
                for (cudaEvent_t event : block.ready_events) {
                    const cudaError_t status = cudaEventSynchronize(event);
                    if (status != cudaSuccess) {
                        LOG_ERROR("cudaEventSynchronize failed during cache clear: {}",
                                  cudaGetErrorString(status));
                    }
                }
                block.release_events();

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

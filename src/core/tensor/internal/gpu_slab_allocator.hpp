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
#include <mutex>
#include <unordered_map>
#include <vector>

namespace lfs::core {

    // GPU slab allocator for small allocations (≤256KB). Slabs are committed on
    // first use per size class and divided into fixed-size blocks.
    //
    // Free lists are kept per stream: a block freed on stream S is immediately
    // reusable on S (safe by stream ordering). Reuse on another stream "steals"
    // the block with a GPU-side event edge from the owning stream, so cross-stream
    // reuse never needs a host sync. Fresh slab blocks live in a virgin list and
    // are stream-free.
    class GPUSlabAllocator {
    public:
        static constexpr size_t MIN_BLOCK_SIZE = 256;
        static constexpr size_t MAX_BLOCK_SIZE = 256 * 1024;
        static constexpr size_t NUM_SIZE_CLASSES = 11;
        static constexpr size_t MIN_SLAB_SIZE = 256 * 1024;
        static constexpr size_t MAX_SLAB_SIZE = 8 * 1024 * 1024;
        static constexpr size_t TARGET_BLOCKS_PER_SLAB = 1024;
        static constexpr size_t MAX_BLOCKS_PER_CLASS = 512 * 1024; // Max blocks to track

        struct Stats {
            std::atomic<uint64_t> alloc_count{0};
            std::atomic<uint64_t> free_count{0};
            std::atomic<uint64_t> miss_count{0};
            std::atomic<uint64_t> steal_count{0};
            size_t total_slab_memory{0};
            size_t blocks_per_class[NUM_SIZE_CLASSES]{0};
        };

        static LFS_CORE_API GPUSlabAllocator& instance();

        void shutdown() {
            bool expected = false;
            if (!shutdown_.compare_exchange_strong(expected, true))
                return;
            enabled_.store(false, std::memory_order_release);
            cleanup();
        }

        void* allocate(size_t bytes, cudaStream_t stream = nullptr) {
            if (!enabled_.load(std::memory_order_acquire) || bytes == 0 || bytes > MAX_BLOCK_SIZE) {
                return nullptr;
            }

            const size_t size_class = get_size_class(bytes);
            if (size_class >= NUM_SIZE_CLASSES) {
                return nullptr;
            }

            void* ptr = pop_block(size_class, stream);
            if (ptr) {
                stats_.alloc_count.fetch_add(1, std::memory_order_relaxed);
                return ptr;
            }

            if (expand_slab(size_class)) {
                ptr = pop_block(size_class, stream);
                if (ptr) {
                    stats_.alloc_count.fetch_add(1, std::memory_order_relaxed);
                    return ptr;
                }
            }

            stats_.miss_count.fetch_add(1, std::memory_order_relaxed);
            return nullptr;
        }

        // Moves `stream`'s free-list entries to the virgin list. Caller must have
        // synchronized the stream (or the device) first — entries become
        // reusable on any stream with no event edge.
        void merge_stream_into_virgin(cudaStream_t stream) {
            for (auto& lists : free_lists_) {
                std::lock_guard<std::mutex> lock(lists.mutex);
                auto it = lists.per_stream.find(stream);
                if (it == lists.per_stream.end()) {
                    continue;
                }
                lists.virgin.insert(lists.virgin.end(), it->second.begin(), it->second.end());
                lists.per_stream.erase(it);
            }
        }

        // Same, for every stream. Caller must have synchronized the device.
        void merge_all_streams_into_virgin() {
            for (auto& lists : free_lists_) {
                std::lock_guard<std::mutex> lock(lists.mutex);
                for (auto& entry : lists.per_stream) {
                    auto& blocks = entry.second;
                    lists.virgin.insert(lists.virgin.end(), blocks.begin(), blocks.end());
                }
                lists.per_stream.clear();
            }
        }

        // `stream` must be the stream the block's last use is ordered on
        // (the owner's home stream after any cross-stream edges were bridged).
        void deallocate(void* ptr, size_t bytes, cudaStream_t stream = nullptr) {
            if (!ptr || bytes == 0 || bytes > MAX_BLOCK_SIZE) {
                return;
            }

            const size_t size_class = get_size_class(bytes);
            if (size_class >= NUM_SIZE_CLASSES) {
                return;
            }

            push_block(size_class, ptr, stream);
            stats_.free_count.fetch_add(1, std::memory_order_relaxed);
        }

        bool owns_pointer(void* ptr) const {
            if (!ptr)
                return false;
            uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);

            std::lock_guard<std::mutex> lock(slabs_mutex_);
            for (const auto& slab : slabs_) {
                uintptr_t slab_start = reinterpret_cast<uintptr_t>(slab.base);
                uintptr_t slab_end = slab_start + slab.size;
                if (addr >= slab_start && addr < slab_end) {
                    return true;
                }
            }
            return false;
        }

        static size_t get_size_class(size_t bytes) {
            if (bytes <= MIN_BLOCK_SIZE)
                return 0;
            size_t size = MIN_BLOCK_SIZE;
            size_t class_idx = 0;
            while (size < bytes && class_idx < NUM_SIZE_CLASSES - 1) {
                size *= 2;
                class_idx++;
            }
            return class_idx;
        }

        static size_t get_block_size(size_t size_class) {
            return MIN_BLOCK_SIZE << size_class;
        }

        static size_t slab_size_for_class(size_t size_class) {
            const size_t block_size = get_block_size(size_class);
            const size_t target_bytes = block_size * TARGET_BLOCKS_PER_SLAB;
            const size_t slab_size = std::clamp(target_bytes, MIN_SLAB_SIZE, MAX_SLAB_SIZE);
            return (slab_size / block_size) * block_size;
        }

        bool is_enabled() const {
            return enabled_.load(std::memory_order_acquire);
        }

        const Stats& stats() const { return stats_; }

        void print_stats() const {
            LOG_INFO("GPUSlabAllocator Statistics:");
            LOG_INFO("  Total slab memory: {:.2f} MB", stats_.total_slab_memory / (1024.0 * 1024.0));
            LOG_INFO("  Allocations: {}", stats_.alloc_count.load());
            LOG_INFO("  Deallocations: {}", stats_.free_count.load());
            LOG_INFO("  Misses: {}", stats_.miss_count.load());

            for (size_t i = 0; i < NUM_SIZE_CLASSES; i++) {
                if (stats_.blocks_per_class[i] > 0) {
                    LOG_INFO("  Class {} ({} bytes): {} blocks",
                             i, get_block_size(i), stats_.blocks_per_class[i]);
                }
            }
        }

        GPUSlabAllocator(const GPUSlabAllocator&) = delete;
        GPUSlabAllocator& operator=(const GPUSlabAllocator&) = delete;

    private:
        struct Slab {
            void* base;
            size_t size;
            size_t size_class;
        };

        struct FreeLists {
            std::unordered_map<cudaStream_t, std::vector<void*>> per_stream;
            std::vector<void*> virgin;
            std::mutex mutex;
            std::atomic<size_t> count{0};
        };

        GPUSlabAllocator() {
            int device_count = 0;
            cudaError_t err = cudaGetDeviceCount(&device_count);
            if (err != cudaSuccess) {
                ensure_cuda_success(
                    err, "cudaGetDeviceCount(GPU slab allocator)",
                    "fallback=disable slab allocator", LFS_SOURCE_SITE_CURRENT(),
                    CudaFailureDisposition::LogOnly);
                enabled_.store(false, std::memory_order_release);
                return;
            }
            if (device_count == 0) {
                LOG_DEBUG("GPUSlabAllocator: No CUDA devices available");
                enabled_.store(false, std::memory_order_release);
                return;
            }

            for (size_t i = 0; i < NUM_SIZE_CLASSES; ++i) {
                const size_t initial_blocks = slab_size_for_class(i) / get_block_size(i);
                free_lists_[i].virgin.reserve(std::min(initial_blocks, MAX_BLOCKS_PER_CLASS));
            }

            enabled_.store(true, std::memory_order_release);
            LOG_DEBUG("GPUSlabAllocator: lazy initialization enabled");
        }

        ~GPUSlabAllocator() {
            shutdown();
        }

        bool allocate_slab(size_t size_class) {
            LFS_CUDA_BREADCRUMB("tensor.slab.allocate");
            const size_t block_size = get_block_size(size_class);
            const size_t slab_size = slab_size_for_class(size_class);

            void* slab_base = nullptr;
            const cudaError_t status = cudaMalloc(&slab_base, slab_size);
            if (status != cudaSuccess) {
                ensure_cuda_success(status, "cudaMalloc(GPU slab)",
                                    ::lfs::core::detail::format_cuda_safe("slab_bytes={}, size_class={}", slab_size, size_class),
                                    LFS_SOURCE_SITE_CURRENT(),
                                    CudaFailureDisposition::LogOnly);
                return false;
            }

            const size_t num_blocks = slab_size / block_size;
            {
                std::lock_guard<std::mutex> lock(free_lists_[size_class].mutex);
                for (size_t i = 0; i < num_blocks; ++i) {
                    void* block = static_cast<char*>(slab_base) + i * block_size;
                    free_lists_[size_class].virgin.push_back(block);
                }
                free_lists_[size_class].count.fetch_add(num_blocks, std::memory_order_release);
            }

            {
                std::lock_guard<std::mutex> lock(slabs_mutex_);
                slabs_.push_back({slab_base, slab_size, size_class});
            }

            stats_.total_slab_memory += slab_size;
            stats_.blocks_per_class[size_class] += num_blocks;
            publish_reserved_bytes();

            return true;
        }

        bool expand_slab(size_t size_class) {
            static std::mutex expand_mutex;
            std::lock_guard<std::mutex> lock(expand_mutex);
            if (free_lists_[size_class].count.load(std::memory_order_acquire) > 0) {
                return true;
            }
            return allocate_slab(size_class);
        }

        void cleanup() {
            LFS_CUDA_BREADCRUMB("tensor.slab.free");
            std::lock_guard<std::mutex> lock(slabs_mutex_);
            for (const auto& slab : slabs_) {
                const cudaError_t free_status = cudaFree(slab.base);
                if (free_status != cudaSuccess) {
                    ensure_cuda_success(
                        free_status, "cudaFree(GPU slab)",
                        ::lfs::core::detail::format_cuda_safe("ptr={}, bytes={}, size_class={}", slab.base, slab.size,
                                                              slab.size_class),
                        LFS_SOURCE_SITE_CURRENT(), CudaFailureDisposition::LogOnly);
                }
            }
            slabs_.clear();
            stats_.total_slab_memory = 0;
            publish_reserved_bytes();
        }

        void* pop_block(size_t size_class, cudaStream_t stream) {
            FreeLists& lists = free_lists_[size_class];
            if (lists.count.load(std::memory_order_acquire) == 0) {
                return nullptr;
            }
            std::lock_guard<std::mutex> lock(lists.mutex);

            if (auto it = lists.per_stream.find(stream);
                it != lists.per_stream.end() && !it->second.empty()) {
                void* ptr = it->second.back();
                it->second.pop_back();
                lists.count.fetch_sub(1, std::memory_order_release);
                return ptr;
            }

            if (!lists.virgin.empty()) {
                void* ptr = lists.virgin.back();
                lists.virgin.pop_back();
                lists.count.fetch_sub(1, std::memory_order_release);
                return ptr;
            }

            // Steal from the richest other stream. The class mutex orders this
            // after the owner's push, so the event edge captures the block's
            // last use on the victim stream.
            auto victim = lists.per_stream.end();
            for (auto it = lists.per_stream.begin(); it != lists.per_stream.end(); ++it) {
                if (it->second.empty()) {
                    continue;
                }
                if (victim == lists.per_stream.end() ||
                    it->second.size() > victim->second.size()) {
                    victim = it;
                }
            }
            if (victim == lists.per_stream.end()) {
                return nullptr;
            }

            void* ptr = victim->second.back();
            victim->second.pop_back();
            lists.count.fetch_sub(1, std::memory_order_release);
            bridgeStreams(victim->first, stream);
            stats_.steal_count.fetch_add(1, std::memory_order_relaxed);
            return ptr;
        }

        void push_block(size_t size_class, void* ptr, cudaStream_t stream) {
            FreeLists& lists = free_lists_[size_class];
            std::lock_guard<std::mutex> lock(lists.mutex);
            lists.per_stream[stream].push_back(ptr);
            lists.count.fetch_add(1, std::memory_order_release);
        }

        std::array<FreeLists, NUM_SIZE_CLASSES> free_lists_;
        std::vector<Slab> slabs_;
        mutable std::mutex slabs_mutex_;
        Stats stats_;
        std::atomic<bool> enabled_{false};
        std::atomic<bool> shutdown_{false};

        void publish_reserved_bytes() const {
            try {
                lfs::diagnostics::VramProfiler::instance().setCudaSlabReservedBytes(stats_.total_slab_memory);
            } catch (...) {
                // Diagnostics must never make allocator growth or shutdown fail.
            }
        }
    };

} // namespace lfs::core

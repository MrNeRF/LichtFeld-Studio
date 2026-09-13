/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/alloc_counter.hpp"
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

        // Free slabs whose every block is currently free. Caller must have
        // synchronized the device and preferably merged streams into virgin
        // (see CudaMemoryPool::trim_cached_memory). Safe only when no live
        // allocation holds a block from a reclaimed slab.
        // Serialized with expand_slab via expand_mutex_. Candidate blocks are
        // detached under the metadata locks, but the driver free is deliberately
        // performed after releasing them. A failed free remains tracked and
        // budgeted; its blocks stay quarantined because CUDA may surface an older
        // asynchronous error without making the free's outcome knowable.
        void reclaim_empty_slabs() {
            LFS_CUDA_BREADCRUMB("tensor.slab.reclaim");
            std::lock_guard<std::mutex> expand_lock(expand_mutex_);

            size_t reclaimed_bytes = 0;
            std::vector<Slab> candidates;
            {
                std::lock_guard<std::mutex> slabs_lock(slabs_mutex_);
                if (slabs_.empty()) {
                    return;
                }
                candidates.reserve(slabs_.size());

                for (const auto& slab : slabs_) {
                    const size_t size_class = slab.size_class;
                    if (size_class >= NUM_SIZE_CLASSES) {
                        continue;
                    }

                    const size_t block_size = get_block_size(size_class);
                    const size_t num_blocks = slab.size / block_size;
                    FreeLists& lists = free_lists_[size_class];
                    std::lock_guard<std::mutex> free_lock(lists.mutex);

                    std::unordered_map<void*, size_t /*refcount*/> free_counts;
                    free_counts.reserve(lists.virgin.size() + 64);
                    auto bump = [&](void* p) {
                        if (p)
                            ++free_counts[p];
                    };
                    for (void* p : lists.virgin)
                        bump(p);
                    for (auto& entry : lists.per_stream) {
                        for (void* p : entry.second)
                            bump(p);
                    }

                    const uintptr_t slab_start = reinterpret_cast<uintptr_t>(slab.base);
                    size_t free_in_slab = 0;
                    for (size_t i = 0; i < num_blocks; ++i) {
                        void* block = reinterpret_cast<void*>(slab_start + i * block_size);
                        auto it = free_counts.find(block);
                        if (it != free_counts.end() && it->second > 0) {
                            ++free_in_slab;
                        }
                    }

                    if (free_in_slab != num_blocks) {
                        continue;
                    }

                    // Fully empty: strip every block of this slab from free lists.
                    auto strip = [&](std::vector<void*>& vec) {
                        vec.erase(std::remove_if(vec.begin(), vec.end(),
                                                 [&](void* p) {
                                                     const uintptr_t addr =
                                                         reinterpret_cast<uintptr_t>(p);
                                                     return addr >= slab_start &&
                                                            addr < slab_start + slab.size;
                                                 }),
                                  vec.end());
                    };
                    strip(lists.virgin);
                    for (auto it = lists.per_stream.begin(); it != lists.per_stream.end();) {
                        strip(it->second);
                        if (it->second.empty()) {
                            it = lists.per_stream.erase(it);
                        } else {
                            ++it;
                        }
                    }
                    const size_t prev = lists.count.load(std::memory_order_relaxed);
                    lists.count.store(prev >= num_blocks ? prev - num_blocks : 0,
                                      std::memory_order_release);
                    candidates.push_back(slab);
                }
            }

            std::unordered_map<void*, bool> reclaimed_bases;
            reclaimed_bases.reserve(candidates.size());
            for (const auto& slab : candidates) {
                const cudaError_t free_status = reclaim_free_fn_
                                                    ? reclaim_free_fn_(slab.base)
                                                    : cudaFree(slab.base);
                if (free_status != cudaSuccess) {
                    ensure_cuda_success(
                        free_status, "cudaFree(GPU slab reclaim)",
                        ::lfs::core::detail::format_cuda_safe(
                            "ptr={}, bytes={}, size_class={}", slab.base, slab.size,
                            slab.size_class),
                        LFS_SOURCE_SITE_CURRENT(), CudaFailureDisposition::LogOnlyNoLatch);
                    continue;
                }

                reclaimed_bases.emplace(slab.base, true);
                reclaimed_bytes += slab.size;
                const size_t num_blocks = slab.size / get_block_size(slab.size_class);
                if (stats_.blocks_per_class[slab.size_class] >= num_blocks) {
                    stats_.blocks_per_class[slab.size_class] -= num_blocks;
                } else {
                    stats_.blocks_per_class[slab.size_class] = 0;
                }
            }

            if (!reclaimed_bases.empty()) {
                std::lock_guard<std::mutex> slabs_lock(slabs_mutex_);
                slabs_.erase(
                    std::remove_if(slabs_.begin(), slabs_.end(),
                                   [&](const Slab& slab) {
                                       return reclaimed_bases.contains(slab.base);
                                   }),
                    slabs_.end());
            }

            if (reclaimed_bytes > 0) {
                if (stats_.total_slab_memory >= reclaimed_bytes) {
                    stats_.total_slab_memory -= reclaimed_bytes;
                } else {
                    stats_.total_slab_memory = 0;
                }
                publish_reserved_bytes();
                LOG_DEBUG("GPUSlabAllocator: reclaimed {} bytes of fully-empty slabs "
                          "({:.2f} MB still reserved)",
                          reclaimed_bytes, stats_.total_slab_memory / (1024.0 * 1024.0));
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

        using ReclaimFreeFn = cudaError_t (*)(void*);
        void set_reclaim_free_fn_for_testing(ReclaimFreeFn fn) {
            std::lock_guard<std::mutex> lock(expand_mutex_);
            reclaim_free_fn_ = fn;
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
            alloc_counter::record_site(alloc_counter::Site::Slab);

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
            std::lock_guard<std::mutex> lock(expand_mutex_);
            if (free_lists_[size_class].count.load(std::memory_order_acquire) > 0) {
                return true;
            }
            return allocate_slab(size_class);
        }

        void cleanup() {
            LFS_CUDA_BREADCRUMB("tensor.slab.free");
            std::lock_guard<std::mutex> expand_lock(expand_mutex_);
            std::lock_guard<std::mutex> lock(slabs_mutex_);
            for (const auto& slab : slabs_) {
                const cudaError_t free_status = cudaFree(slab.base);
                if (free_status != cudaSuccess) {
                    ensure_cuda_success(
                        free_status, "cudaFree(GPU slab)",
                        ::lfs::core::detail::format_cuda_safe("ptr={}, bytes={}, size_class={}", slab.base, slab.size,
                                                              slab.size_class),
                        LFS_SOURCE_SITE_CURRENT(), CudaFailureDisposition::LogOnlyNoLatch);
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
        // Serializes expand_slab / reclaim_empty_slabs / cleanup growth-shrink.
        std::mutex expand_mutex_;
        ReclaimFreeFn reclaim_free_fn_ = nullptr;
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

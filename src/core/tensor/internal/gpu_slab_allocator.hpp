/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/logger.hpp"
#include <array>
#include <atomic>
#include <cstdint>
#include <cuda_runtime.h>
#include <mutex>
#include <vector>

namespace lfs::core {

    // GPU slab allocator for small allocations (≤256KB). Pre-allocates slabs and
    // divides them into fixed-size blocks. O(1) alloc/free, ~50ns per operation.
    class GPUSlabAllocator {
    public:
        static constexpr size_t MIN_BLOCK_SIZE = 256;
        static constexpr size_t MAX_BLOCK_SIZE = 256 * 1024;
        static constexpr size_t NUM_SIZE_CLASSES = 11;
        static constexpr size_t SLAB_SIZE = 32 * 1024 * 1024;      // 32 MB per slab
        static constexpr size_t MAX_BLOCKS_PER_CLASS = 512 * 1024; // Max blocks to track

        struct Stats {
            std::atomic<uint64_t> alloc_count{0};
            std::atomic<uint64_t> free_count{0};
            std::atomic<uint64_t> miss_count{0};
            size_t total_slab_memory{0};
            size_t blocks_per_class[NUM_SIZE_CLASSES]{0};
        };

        static GPUSlabAllocator& instance() {
            static GPUSlabAllocator allocator;
            return allocator;
        }

        void shutdown() {
            bool expected = false;
            if (!shutdown_.compare_exchange_strong(expected, true))
                return;
            enabled_.store(false, std::memory_order_release);
            cleanup();
        }

        void* allocate(size_t bytes) {
            if (!enabled_.load(std::memory_order_acquire) || bytes == 0 || bytes > MAX_BLOCK_SIZE) {
                return nullptr;
            }

            const size_t size_class = get_size_class(bytes);
            if (size_class >= NUM_SIZE_CLASSES) {
                return nullptr;
            }

            void* ptr = pop_free_stack(size_class);
            if (ptr) {
                stats_.alloc_count.fetch_add(1, std::memory_order_relaxed);
                return ptr;
            }

            if (expand_slab(size_class)) {
                ptr = pop_free_stack(size_class);
                if (ptr) {
                    stats_.alloc_count.fetch_add(1, std::memory_order_relaxed);
                    return ptr;
                }
            }

            stats_.miss_count.fetch_add(1, std::memory_order_relaxed);
            return nullptr;
        }

        void deallocate(void* ptr, size_t bytes) {
            if (!ptr || bytes == 0 || bytes > MAX_BLOCK_SIZE) {
                return;
            }

            const size_t size_class = get_size_class(bytes);
            if (size_class >= NUM_SIZE_CLASSES) {
                return;
            }

            push_free_stack(size_class, ptr);
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

        struct FreeStack {
            std::vector<void*> stack;
            std::mutex mutex;
            std::atomic<size_t> count{0};
        };

        GPUSlabAllocator() {
            int device_count = 0;
            cudaError_t err = cudaGetDeviceCount(&device_count);
            if (err != cudaSuccess || device_count == 0) {
                LOG_DEBUG("GPUSlabAllocator: No CUDA devices available");
                enabled_.store(false, std::memory_order_release);
                return;
            }

            for (size_t i = 0; i < NUM_SIZE_CLASSES; ++i) {
                free_stacks_[i].stack.reserve(MAX_BLOCKS_PER_CLASS);
            }

            if (!initialize_slabs()) {
                LOG_DEBUG("GPUSlabAllocator: Initialization failed or skipped");
                enabled_.store(false, std::memory_order_release);
            } else {
                enabled_.store(true, std::memory_order_release);
                LOG_INFO("GPUSlabAllocator: {:.2f} MB allocated",
                         stats_.total_slab_memory / (1024.0 * 1024.0));
            }
        }

        ~GPUSlabAllocator() {
            shutdown();
        }

        bool initialize_slabs() {
            for (size_t size_class = 2; size_class < NUM_SIZE_CLASSES; ++size_class) {
                if (!allocate_slab(size_class)) {
                    LOG_DEBUG("Skipping slab for class {}", size_class);
                }
            }
            return stats_.total_slab_memory > 0;
        }

        bool allocate_slab(size_t size_class) {
            const size_t block_size = get_block_size(size_class);

            void* slab_base = nullptr;
            if (cudaMalloc(&slab_base, SLAB_SIZE) != cudaSuccess) {
                return false;
            }

            const size_t num_blocks = SLAB_SIZE / block_size;
            {
                std::lock_guard<std::mutex> lock(free_stacks_[size_class].mutex);
                for (size_t i = 0; i < num_blocks; ++i) {
                    void* block = static_cast<char*>(slab_base) + i * block_size;
                    free_stacks_[size_class].stack.push_back(block);
                }
                free_stacks_[size_class].count.fetch_add(num_blocks, std::memory_order_release);
            }

            {
                std::lock_guard<std::mutex> lock(slabs_mutex_);
                slabs_.push_back({slab_base, SLAB_SIZE, size_class});
            }

            stats_.total_slab_memory += SLAB_SIZE;
            stats_.blocks_per_class[size_class] += num_blocks;

            return true;
        }

        bool expand_slab(size_t size_class) {
            static std::mutex expand_mutex;
            std::lock_guard<std::mutex> lock(expand_mutex);
            if (free_stacks_[size_class].count.load(std::memory_order_acquire) > 0) {
                return true;
            }
            return allocate_slab(size_class);
        }

        void cleanup() {
            std::lock_guard<std::mutex> lock(slabs_mutex_);
            for (const auto& slab : slabs_) {
                cudaFree(slab.base);
            }
            slabs_.clear();
            stats_.total_slab_memory = 0;
        }

        void* pop_free_stack(size_t size_class) {
            if (free_stacks_[size_class].count.load(std::memory_order_acquire) == 0) {
                return nullptr;
            }
            std::lock_guard<std::mutex> lock(free_stacks_[size_class].mutex);
            if (free_stacks_[size_class].stack.empty()) {
                return nullptr;
            }
            void* ptr = free_stacks_[size_class].stack.back();
            free_stacks_[size_class].stack.pop_back();
            free_stacks_[size_class].count.fetch_sub(1, std::memory_order_release);
            return ptr;
        }

        void push_free_stack(size_t size_class, void* ptr) {
            std::lock_guard<std::mutex> lock(free_stacks_[size_class].mutex);
            free_stacks_[size_class].stack.push_back(ptr);
            free_stacks_[size_class].count.fetch_add(1, std::memory_order_release);
        }

        std::array<FreeStack, NUM_SIZE_CLASSES> free_stacks_;
        std::vector<Slab> slabs_;
        mutable std::mutex slabs_mutex_;
        Stats stats_;
        std::atomic<bool> enabled_{false};
        std::atomic<bool> shutdown_{false};
    };

} // namespace lfs::core

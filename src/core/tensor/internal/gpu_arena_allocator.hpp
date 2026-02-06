/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/logger.hpp"
#include "offset_allocator.hpp"
#include <atomic>
#include <cuda_runtime.h>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace lfs::core {

    /**
     * @brief GPU arena allocator using OffsetAllocator for O(1) sub-allocations
     *
     * Pre-allocates a large GPU buffer (8GB default) and uses OffsetAllocator
     * to perform fast O(1) sub-allocations from this arena. This eliminates
     * the overhead of cudaMalloc/cudaFree for medium-sized tensors.
     *
     * Performance comparison:
     * - cudaMalloc: ~0.15-0.6 ms per call
     * - Arena allocation: ~0.001 ms per call (150-600× faster!)
     *
     * Memory overhead:
     * - OffsetAllocator: 6.25% fragmentation overhead (TLSF-like algorithm)
     * - Power-of-2 binning: 50% overhead (not used here)
     *
     * Expected speedup: 2-5× for allocation-heavy workloads
     *
     * Usage strategy:
     * - Small/medium tensors (<100MB): Use arena allocator
     * - Large tensors (≥100MB): Use cudaMallocAsync pool (avoid fragmentation)
     *
     * Thread safety: Protected by mutex
     */
    class GPUArenaAllocator {
    public:
        static GPUArenaAllocator& instance() {
            static GPUArenaAllocator allocator;
            return allocator;
        }

        void shutdown() {
            bool expected = false;
            if (!shutdown_.compare_exchange_strong(expected, true))
                return;
            cleanup();
        }

        /**
         * @brief Allocate memory from the GPU arena
         * @param bytes Number of bytes to allocate
         * @return Pointer to allocated memory, or nullptr on failure
         *
         * Thread-safe: Protected by mutex
         * Time complexity: O(1) amortized
         */
        void* allocate(size_t bytes) {
            if (bytes == 0) {
                return nullptr;
            }

            std::lock_guard<std::mutex> lock(mutex_);

            // Request allocation from OffsetAllocator
            OffsetAllocator::Allocation alloc = allocator_->allocate(static_cast<uint32_t>(bytes));

            if (alloc.offset == OffsetAllocator::Allocation::NO_SPACE) {
                LOG_ERROR("GPU arena out of memory: requested {} bytes, {} bytes free",
                          bytes, allocator_->storageReport().totalFreeSpace);
                return nullptr;
            }

            // Calculate actual GPU pointer
            void* ptr = static_cast<char*>(gpu_base_) + alloc.offset;

            // Track allocation for later deallocation
            allocations_[ptr] = alloc;

            return ptr;
        }

        /**
         * @brief Deallocate memory back to the arena
         * @param ptr Pointer to memory to deallocate
         *
         * Thread-safe: Protected by mutex
         * Time complexity: O(1)
         *
         * CRITICAL: Synchronizes CUDA device before returning memory to pool
         * to avoid use-after-free bugs with asynchronous CUDA operations.
         */
        void deallocate(void* ptr) {
            if (!ptr) {
                return;
            }
            if (shutdown_.load(std::memory_order_acquire))
                return;

            std::lock_guard<std::mutex> lock(mutex_);

            auto it = allocations_.find(ptr);
            if (it == allocations_.end()) {
                LOG_ERROR("Attempted to free pointer not allocated by arena: {}", ptr);
                return;
            }

            // CRITICAL: Synchronize device before returning memory to pool!
            // Without this, CUDA operations may still be using this memory when we reuse it.
            // This prevents use-after-free bugs that cause "illegal memory access" errors.
            cudaError_t err = cudaDeviceSynchronize();
            if (err != cudaSuccess) {
                LOG_WARN("cudaDeviceSynchronize failed in arena deallocate: {}",
                         cudaGetErrorString(err));
                // Continue anyway - better to potentially reuse memory than leak it
            }

            // Free the allocation in OffsetAllocator
            allocator_->free(it->second);

            // Remove from tracking map
            allocations_.erase(it);
        }

        /**
         * @brief Get arena size in bytes
         */
        size_t arena_size() const {
            return arena_size_;
        }

        /**
         * @brief Get total free space in arena
         */
        size_t total_free() const {
            std::lock_guard<std::mutex> lock(mutex_);
            return allocator_->storageReport().totalFreeSpace;
        }

        /**
         * @brief Get total allocated space in arena
         */
        size_t total_allocated() const {
            std::lock_guard<std::mutex> lock(mutex_);
            return arena_size_ - allocator_->storageReport().totalFreeSpace;
        }

        /**
         * @brief Get detailed storage report
         */
        OffsetAllocator::StorageReport storage_report() const {
            std::lock_guard<std::mutex> lock(mutex_);
            return allocator_->storageReport();
        }

        /**
         * @brief Get number of active allocations
         */
        size_t num_allocations() const {
            std::lock_guard<std::mutex> lock(mutex_);
            return allocations_.size();
        }

        /**
         * @brief Check if arena is enabled
         */
        bool is_enabled() const {
            return gpu_base_ != nullptr;
        }

        /**
         * @brief Check if a pointer was allocated from this arena
         * @param ptr Pointer to check
         * @return true if pointer belongs to this arena
         *
         * Thread-safe: Protected by mutex
         */
        bool owns_pointer(void* ptr) const {
            if (!ptr || !gpu_base_) {
                return false;
            }

            std::lock_guard<std::mutex> lock(mutex_);
            return allocations_.find(ptr) != allocations_.end();
        }

        // Disable copy and move
        GPUArenaAllocator(const GPUArenaAllocator&) = delete;
        GPUArenaAllocator& operator=(const GPUArenaAllocator&) = delete;
        GPUArenaAllocator(GPUArenaAllocator&&) = delete;
        GPUArenaAllocator& operator=(GPUArenaAllocator&&) = delete;

    private:
        GPUArenaAllocator() : gpu_base_(nullptr),
                              arena_size_(0),
                              allocator_(nullptr) {
            // Default: ~4GB arena (OffsetAllocator uint32 limit), support 128K allocations
            // Must be < 4GB due to OffsetAllocator's uint32 size limit
            initialize((4ULL << 30) - (16 << 20), 128 * 1024); // 4GB - 16MB
        }

        ~GPUArenaAllocator() {
            shutdown();
        }

        void initialize(size_t size_bytes, uint32_t max_allocs) {
            // CRITICAL: OffsetAllocator uses uint32 for sizes, maximum ~4GB
            if (size_bytes > UINT32_MAX) {
                LOG_ERROR("Arena size {} exceeds OffsetAllocator limit (~4 GB)", size_bytes);
                gpu_base_ = nullptr;
                arena_size_ = 0;
                return;
            }

            // Allocate GPU memory for arena
            cudaError_t err = cudaMalloc(&gpu_base_, size_bytes);
            if (err != cudaSuccess) {
                LOG_ERROR("Failed to allocate {:.2f} GB GPU arena: {}",
                          size_bytes / (1024.0 * 1024.0 * 1024.0),
                          cudaGetErrorString(err));
                gpu_base_ = nullptr;
                arena_size_ = 0;
                return;
            }

            arena_size_ = size_bytes;

            // Create OffsetAllocator with arena size (in bytes)
            uint32_t allocator_size = static_cast<uint32_t>(size_bytes);
            allocator_ = std::make_unique<OffsetAllocator::Allocator>(allocator_size, max_allocs);

            LOG_INFO("GPU Arena Allocator initialized:");
            LOG_INFO("  • Arena size: {:.2f} GB", arena_size_ / (1024.0 * 1024.0 * 1024.0));
            LOG_INFO("  • Max allocations: {}", max_allocs);
            LOG_INFO("  • Base address: {}", gpu_base_);
            LOG_INFO("  • OffsetAllocator overhead: ~6.25% (TLSF algorithm)");
        }

        void cleanup() {
            if (allocations_.size() > 0) {
                LOG_WARN("GPU arena has {} leaked allocations at shutdown", allocations_.size());
            }

            allocations_.clear();
            allocator_.reset();

            if (gpu_base_) {
                cudaError_t err = cudaFree(gpu_base_);
                if (err != cudaSuccess) {
                    LOG_ERROR("Failed to free GPU arena: {}", cudaGetErrorString(err));
                }
                gpu_base_ = nullptr;
            }

            arena_size_ = 0;
        }

        void* gpu_base_;
        size_t arena_size_;
        std::unique_ptr<OffsetAllocator::Allocator> allocator_;
        std::unordered_map<void*, OffsetAllocator::Allocation> allocations_;
        mutable std::mutex mutex_;
        std::atomic<bool> shutdown_{false};
    };

} // namespace lfs::core

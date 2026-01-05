/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <atomic>
#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace lfs::core {

    class RasterizerMemoryArena {
    public:
        struct Config {
            size_t virtual_size = 32ULL << 30; // 32GB virtual address space (free!)
            size_t initial_commit = 256 << 20; // 256MB initial physical memory
            size_t max_physical = 8ULL << 30;  // 8GB max physical memory
            size_t granularity = 2 << 20;      // 2MB allocation granularity
            size_t alignment = 256;
            bool enable_profiling = false;
            size_t log_interval = 1000; // Log every N frames
        };

        struct BufferHandle {
            char* ptr = nullptr;
            size_t size = 0;
            uint64_t generation = 0;
            int device = -1;
        };

        struct FrameContext {
            std::vector<BufferHandle> buffers;
            uint64_t frame_id = 0;
            uint64_t generation = 0;
            size_t total_allocated = 0;
            bool is_active = false;
            std::chrono::steady_clock::time_point timestamp;
        };

        struct Statistics {
            size_t current_usage = 0;
            size_t peak_usage = 0;
            size_t capacity = 0;
            size_t reallocation_count = 0;
            size_t frame_count = 0;
            float utilization_ratio = 0.0f;
        };

        struct MemoryInfo {
            size_t arena_capacity = 0;
            size_t current_usage = 0;
            size_t peak_usage = 0;
            size_t gpu_free = 0;
            size_t gpu_total = 0;
            size_t num_reallocations = 0;
            float utilization_percent = 0.0f;
        };

    private:
        struct PhysicalChunk {
            CUmemGenericAllocationHandle handle = 0;
            size_t offset = 0;
            size_t size = 0;
            bool is_mapped = false;
        };

        struct Arena {
            // VMM specific
            CUdeviceptr d_ptr = 0;             // Virtual base address
            size_t virtual_size = 0;           // Total virtual space
            size_t committed_size = 0;         // Actually committed physical memory
            size_t granularity = 0;            // Allocation granularity
            std::vector<PhysicalChunk> chunks; // Physical memory chunks
            std::mutex chunks_mutex;

            // Traditional allocation fallback
            void* fallback_buffer = nullptr; // Raw CUDA memory for non-VMM
            std::atomic<size_t> offset{0};   // Current allocation offset
            size_t capacity = 0;             // Same as committed_size for compatibility
            uint64_t generation = 0;
            int device = -1;

            // Frame synchronization event (replaces cudaDeviceSynchronize)
            cudaEvent_t frame_event = nullptr;
            bool event_recorded = false;

            // Statistics
            std::atomic<size_t> peak_usage{0};
            std::atomic<size_t> peak_usage_period{0};
            std::atomic<size_t> total_allocated{0};
            std::atomic<size_t> realloc_count{0};
            std::chrono::steady_clock::time_point last_log_time;
        };

        std::unordered_map<int, std::unique_ptr<Arena>> device_arenas_;
        std::unordered_map<uint64_t, FrameContext> frame_contexts_;
        Config config_;

        mutable std::mutex arena_mutex_;
        mutable std::mutex frame_mutex_;
        std::atomic<uint64_t> frame_counter_{0};
        std::atomic<uint64_t> generation_counter_{0};

        // Performance tracking
        std::chrono::steady_clock::time_point creation_time_;
        std::atomic<size_t> total_frames_processed_{0};

        // Simple rendering synchronization
        std::atomic<bool> rendering_active_{false};
        std::atomic<uint64_t> active_training_frames_{0};

    public:
        // Constructors
        RasterizerMemoryArena();
        explicit RasterizerMemoryArena(const Config& cfg);
        ~RasterizerMemoryArena();

        // Delete copy operations
        RasterizerMemoryArena(const RasterizerMemoryArena&) = delete;
        RasterizerMemoryArena& operator=(const RasterizerMemoryArena&) = delete;

        // Allow move operations
        RasterizerMemoryArena(RasterizerMemoryArena&&) noexcept;
        RasterizerMemoryArena& operator=(RasterizerMemoryArena&&) noexcept;

        uint64_t begin_frame(bool from_rendering = false);
        void end_frame(uint64_t frame_id, bool from_rendering = false);
        std::function<char*(size_t)> get_allocator(uint64_t frame_id);
        std::vector<BufferHandle> get_frame_buffers(uint64_t frame_id) const;
        void reset_frame(uint64_t frame_id); // Keeps allocation, resets offset
        void cleanup_frames(int keep_recent = 3);
        void emergency_cleanup();

        Statistics get_statistics() const;
        MemoryInfo get_memory_info() const;
        void dump_statistics() const;
        void log_memory_status(uint64_t frame_id, bool force = false);

        bool is_under_memory_pressure() const;
        float get_memory_pressure() const;

        bool is_rendering_active() const { return rendering_active_.load(std::memory_order_acquire); }
        void set_rendering_active(bool active) { rendering_active_.store(active, std::memory_order_release); }
        void wait_for_training() const;

    private:
        Arena& get_or_create_arena(int device);
        char* allocate_internal(Arena& arena, size_t size, uint64_t frame_id);
        bool grow_arena(Arena& arena, size_t required_size);
        size_t align_size(size_t size) const;
        void record_allocation(uint64_t frame_id, const BufferHandle& handle);
        bool commit_more_memory(Arena& arena, size_t required_size);
        void decommit_unused_memory(Arena& arena);
        bool is_vmm_supported(int device) const;
        void empty_cuda_cache();
    };

    class GlobalArenaManager {
    public:
        static GlobalArenaManager& instance();
        RasterizerMemoryArena& get_arena();
        void reset();

    private:
        GlobalArenaManager() = default;
        ~GlobalArenaManager() = default;
        std::unique_ptr<RasterizerMemoryArena> arena_;
        std::mutex init_mutex_;
    };

} // namespace lfs::core
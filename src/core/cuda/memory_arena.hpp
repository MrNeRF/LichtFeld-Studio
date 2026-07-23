/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cuda.h>
#include <cuda_runtime.h>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace lfs::core {

    class GlobalArenaManager;
    LFS_CORE_API GlobalArenaManager& global_arena_manager();
    LFS_CORE_API void shutdown_global_arena_manager();

    class RasterizerMemoryArena {
    public:
        struct Config {
            size_t virtual_size = 32ULL << 30; // 32GB virtual address space (free!)
            size_t initial_commit = 128 << 20; // 128MB initial physical memory
            size_t max_physical = 8ULL << 30;  // 8GB max physical memory
            size_t granularity = 2 << 20;      // 2MB allocation granularity
            size_t alignment = 256;
            bool enable_vmm = true; // Disable to use and validate the cudaMalloc fallback.
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

        struct ExternalBacking {
            void* device_ptr = nullptr;
            size_t size = 0;
            int device = -1;
            std::shared_ptr<void> owner;
            std::string label;
            // Grows the committed physical to >= the requested bytes in place (the
            // base pointer must stay constant), returning the new committed size or
            // 0 on failure. Invoked by the arena when training's scratch demand
            // outgrows the current capacity. Optional; if unset the arena cannot
            // grow and an over-capacity request fails.
            std::function<size_t(size_t)> grow;

            [[nodiscard]] bool valid() const noexcept {
                return device_ptr != nullptr && size > 0 && device >= 0 && static_cast<bool>(owner);
            }
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
            bool owns_fallback_buffer = true;
            bool external_backing = false;
            std::shared_ptr<void> external_owner;
            std::string external_label;
            std::function<size_t(size_t)> external_grow;
            std::atomic<size_t> offset{0}; // Current allocation offset
            size_t capacity = 0;           // Same as committed_size for compatibility
            uint64_t generation = 0;
            int device = -1;

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

        // The arena uses a single offset per device, so only one live frame can own it safely.
        // Pending render requests cover the queued-before-active window; active_frames_ covers
        // the actual lifetime once begin_frame succeeds.
        mutable std::mutex sync_mutex_;
        mutable std::condition_variable sync_cv_;
        uint64_t active_frames_ = 0;
        uint64_t pending_render_frames_ = 0;
        uint64_t active_training_frames_ = 0;

        // Completion event of the most recent stream-aware frame. Invalid when
        // the last frame was legacy (no stream) — the next begin then falls back
        // to a device sync. Guarded by last_frame_event_mutex_.
        mutable std::mutex last_frame_event_mutex_;
        cudaEvent_t last_frame_event_ = nullptr;
        bool last_frame_event_valid_ = false;

        // Pending Vulkan release of the previous frame's arena work (see
        // note_external_release). Guarded by last_frame_event_mutex_.
        cudaExternalSemaphore_t external_release_semaphore_ = nullptr;
        uint64_t external_release_value_ = 0;

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

        // Stream-aware frames chain begin→end with a GPU event edge: begin_frame
        // waits (on `stream`) for the previous frame's completion event instead of
        // a device-wide sync, and end_frame records the event on `stream` — which
        // must be the stream all of the frame's arena work was enqueued on. Frames
        // without a stream keep the legacy cudaDeviceSynchronize and invalidate
        // the chain.
        //
        // A tenant whose arena work runs on a VULKAN queue (the viewport) must
        // call note_external_release before ending its frame: neither the chain
        // event nor cudaDeviceSynchronize can see in-flight Vulkan work, so the
        // next frame waits the imported timeline value instead.
        uint64_t begin_frame(bool from_rendering = false) { return begin_frame(nullptr, from_rendering); }
        uint64_t begin_frame(cudaStream_t stream, bool from_rendering = false);
        std::optional<uint64_t> try_begin_frame(bool from_rendering = false) { return try_begin_frame(nullptr, from_rendering); }
        std::optional<uint64_t> try_begin_frame(cudaStream_t stream, bool from_rendering = false);

        // Bounded wait: with a render pending (set_rendering_active), the trainer
        // cannot START a new frame, so waiting out its current one takes ~one
        // iteration. The timeout covers the refining-iteration case where the
        // trainer holds the frame while blocked on the exclusive render lock the
        // caller's shared lock prevents — give up there instead of deadlocking.
        std::optional<uint64_t> try_begin_frame_for(uint32_t timeout_ms, bool from_rendering = false) {
            return try_begin_frame_for(timeout_ms, nullptr, from_rendering);
        }
        std::optional<uint64_t> try_begin_frame_for(uint32_t timeout_ms, cudaStream_t stream,
                                                    bool from_rendering = false);
        void end_frame(uint64_t frame_id, bool from_rendering = false) { end_frame(frame_id, nullptr, from_rendering); }
        void end_frame(uint64_t frame_id, cudaStream_t stream, bool from_rendering = false);

        // Registers the Vulkan-side completion of the current frame's arena work
        // (an imported timeline + the value its submit signals). Consumed once by
        // the next begin_frame, which waits it GPU-side (or via the legacy
        // stream ahead of its device sync on the streamless path).
        void note_external_release(cudaExternalSemaphore_t semaphore, uint64_t value);

        // Caps wait-forever begin_frame() on the CURRENT thread to a bounded
        // wait for its scope — so a cross-thread reader (GUI metric render) that
        // holds render_mutex_ can't deadlock against a refining trainer that
        // holds the arena frame and wants the exclusive lock: the reader's
        // begin_frame bails (throws), the metric is skipped, the lock releases.
        // Training threads never set this, so their acquisition stays blocking.
        class ScopedBeginFrameTimeout {
        public:
            explicit ScopedBeginFrameTimeout(uint32_t timeout_ms);
            ~ScopedBeginFrameTimeout();
            ScopedBeginFrameTimeout(const ScopedBeginFrameTimeout&) = delete;
            ScopedBeginFrameTimeout& operator=(const ScopedBeginFrameTimeout&) = delete;

        private:
            uint32_t previous_ = 0;
        };

        std::function<char*(size_t)> get_allocator(uint64_t frame_id);
        std::vector<BufferHandle> get_frame_buffers(uint64_t frame_id) const;
        void reset_frame(uint64_t frame_id); // Keeps allocation, resets offset
        void cleanup_frames(int keep_recent = 3);
        void full_reset();
        bool install_external_backing(ExternalBacking backing);
        bool try_install_external_backing(ExternalBacking backing);
        // Grows the committed size of an already-installed external backing whose
        // base pointer is `device_ptr` (which must stay constant). The arena
        // drains all frames and the device, then invokes `commit(new_size)` — which
        // performs the physical grow + Vulkan re-import — inside that safe window;
        // on success the arena's committed capacity is bumped to new_size.
        bool grow_external_backing(const void* device_ptr, size_t new_size,
                                   const std::function<bool(size_t)>& commit);
        void clear_external_backing(const void* device_ptr = nullptr);
        [[nodiscard]] bool using_external_backing() const;

        Statistics get_statistics() const;
        MemoryInfo get_memory_info() const;
        void dump_statistics() const;
        void log_memory_status(uint64_t frame_id, bool force = false);
        [[nodiscard]] bool has_last_frame_event_for_testing() const;

        bool is_under_memory_pressure() const;
        float get_memory_pressure() const;

        bool is_rendering_active() const;
        void set_rendering_active(bool active);

    private:
        Arena& get_or_create_arena(int device);
        // wait_timeout: nullopt = non-blocking try; 0 = wait forever; else bounded.
        std::optional<uint64_t> begin_frame_impl(cudaStream_t stream, bool from_rendering,
                                                 std::optional<uint32_t> wait_timeout_ms);
        cudaError_t wait_for_previous_frame(cudaStream_t stream);
        // Host-blocks on a pending Vulkan release fence (note_external_release)
        // and clears it. Must run before any path that frees or replaces arena
        // backing — a device sync cannot observe the in-flight Vulkan batch.
        void drain_external_release();
        bool install_external_backing_impl(ExternalBacking backing, bool wait);
        char* allocate_internal(Arena& arena, size_t size, uint64_t frame_id);
        void release_arena_storage(Arena& arena);
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
        static GlobalArenaManager& instance() { return global_arena_manager(); }
        RasterizerMemoryArena& get_arena();
        RasterizerMemoryArena* try_get_arena();
        bool install_external_backing(RasterizerMemoryArena::ExternalBacking backing);
        bool try_install_external_backing(RasterizerMemoryArena::ExternalBacking backing);
        bool grow_external_backing(const void* device_ptr, size_t new_size,
                                   const std::function<bool(size_t)>& commit);
        void clear_external_backing(const void* device_ptr = nullptr);
        void reset();
        void shutdown() { shutdown_global_arena_manager(); }
        /// Reconstructs the arena with a caller-supplied capacity for deterministic OOM tests.
        void reconfigure_for_testing(RasterizerMemoryArena::Config config);

    private:
        friend GlobalArenaManager& global_arena_manager();
        friend void shutdown_global_arena_manager();

        GlobalArenaManager() = default;
        ~GlobalArenaManager() = default;
        std::unique_ptr<RasterizerMemoryArena> arena_;
        std::mutex init_mutex_;
        bool shutdown_ = false;
    };

} // namespace lfs::core

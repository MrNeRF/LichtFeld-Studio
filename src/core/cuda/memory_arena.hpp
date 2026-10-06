/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "viewer_admission.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace lfs::core {

    class GlobalArenaManager;
    LFS_CORE_API GlobalArenaManager& global_arena_manager();
    LFS_CORE_API void shutdown_global_arena_manager();
    LFS_CORE_API void log_arena_failure_vram_snapshot(
        std::string_view label, size_t committed_bytes, size_t frame_peak_bytes);

    class RasterizerMemoryArena {
    public:
        struct Config {
            size_t virtual_size = 32ULL << 30; // 32GB virtual address space (free!)
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
            size_t required_bytes = 0;
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
            // Shrinks the committed physical prefix toward the requested size
            // in place. The callback must decommit only whole VMM granules and
            // return the resulting committed prefix, or 0 on failure. It runs
            // after the arena has drained CUDA and any registered external
            // release timeline. Importers of the backing must be retired by
            // the callback before its physical handles are unmapped.
            std::function<size_t(size_t)> shrink;
            // Minimum prefix that must remain committed for a peer consumer
            // (for example the viewer's shared-scratch high-water).
            std::function<size_t()> minimum_size;

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
            std::function<size_t(size_t)> external_shrink;
            std::function<size_t()> external_minimum_size;
            std::atomic<size_t> offset{0}; // Current allocation offset
            size_t capacity = 0;           // Same as committed_size for compatibility
            uint64_t generation = 0;
            int device = -1;

            // Statistics
            // Logical high-water since the last explicit shrink/reset boundary.
            std::atomic<size_t> peak_usage{0};
            // Lifetime high-water retained across boundary decommits so teardown
            // diagnostics still report the work done before backing was released.
            std::atomic<size_t> lifetime_peak_usage{0};
            std::atomic<size_t> peak_usage_period{0};
            std::atomic<size_t> total_allocated{0};
            std::atomic<size_t> realloc_count{0};
            std::chrono::steady_clock::time_point last_log_time;
            // B1 hysteresis: avoid immediately undoing a recent recommit.
            std::uint32_t boundaries_since_growth = 0;
        };

        struct GrowthTiming {
            mutable std::mutex mutex;
            uint64_t commit_attempts_warmup = 0;
            uint64_t commit_attempts_steady = 0;
            uint64_t commit_events_warmup = 0;
            uint64_t commit_events_steady = 0;
            uint64_t commit_time_us_warmup = 0;
            uint64_t commit_time_us_steady = 0;
            uint64_t growth_path_time_us_warmup = 0;
            uint64_t growth_path_time_us_steady = 0;
            uint64_t growth_sync_time_us_warmup = 0;
            uint64_t growth_sync_time_us_steady = 0;
            uint64_t b1_events = 0;
            uint64_t b1_time_us = 0;
            uint64_t b3_events = 0;
            uint64_t b3_time_us = 0;
        };

        std::unordered_map<int, std::unique_ptr<Arena>> device_arenas_;
        std::unordered_map<uint64_t, FrameContext> frame_contexts_;
        Config config_;

        mutable std::mutex arena_mutex_;
        mutable std::mutex frame_mutex_;
        std::atomic<uint64_t> frame_counter_{0};
        std::atomic<uint64_t> generation_counter_{0};
        std::shared_ptr<GrowthTiming> growth_timing_ = std::make_shared<GrowthTiming>();

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
        uint64_t last_handoff_frame_id_ = 0;
        uint64_t render_handoff_token_ = 0;
        uint64_t next_render_handoff_token_ = 1;
        std::chrono::steady_clock::time_point render_handoff_deadline_{};
        // Turn ledger: training frames that may begin before a waiting viewer
        // (live handoff token) takes the arena. Set by the viewer when it
        // releases a navigation frame; survives without a live token.
        uint32_t training_turns_owed_ = 0;
        uint32_t viewer_release_waiters_ = 0;
        ViewerAdmission viewer_admission_;
        // Turn cost measurements. stats_mutex_ is always innermost (taken under
        // sync_mutex_ by begin_frame, alone elsewhere) and never held across a
        // CUDA call.
        mutable std::mutex stats_mutex_;
        std::chrono::steady_clock::time_point last_training_begin_{};
        bool last_tenant_was_training_ = false;
        double training_step_ms_ = 0.0;
        double viewer_record_ms_ = 0.0;
        double viewer_turn_ring_[5] = {};
        uint32_t viewer_turn_count_ = 0;

        // Completion event of the most recent stream-aware frame. Invalid when
        // the last frame was legacy (no stream) — the next begin then falls back
        // to a device sync. Guarded by last_frame_event_mutex_.
        mutable std::mutex last_frame_event_mutex_;
        cudaEvent_t last_frame_event_ = nullptr;
        bool last_frame_event_valid_ = false;
        // A host-consumed viewer release completes the chain even when another
        // admission waiter or teardown consumed it. Cleared by end_frame.
        bool last_frame_gpu_complete_ = false;

        // Pending Vulkan release of the previous frame's arena work (see
        // await_external_release). Guarded by last_frame_event_mutex_.
        cudaExternalSemaphore_t external_release_semaphore_ = nullptr;
        uint64_t external_release_value_ = 0;
        // GPU completion of that release: a wait on the imported timeline is
        // enqueued on release_stream_ and release_event_ is recorded behind it,
        // so both tenants can see (query) and host-wait (synchronize) the
        // viewer's batch without a device-wide sync. release_pending_ is false
        // when event tracking could not be set up; the semaphore/value paths
        // above then run exactly as before.
        // Borrowed from the viewer; drain before its owner destroys it.
        cudaStream_t release_stream_ = nullptr;
        cudaEvent_t release_event_ = nullptr;
        bool release_pending_ = false;
        bool release_stream_failed_ = false;
        std::chrono::steady_clock::time_point release_submitted_at_{};

    public:
        // Constructors
        RasterizerMemoryArena();
        explicit RasterizerMemoryArena(const Config& cfg);
        ~RasterizerMemoryArena();

        // Delete copy operations
        RasterizerMemoryArena(const RasterizerMemoryArena&) = delete;
        RasterizerMemoryArena& operator=(const RasterizerMemoryArena&) = delete;

        RasterizerMemoryArena(RasterizerMemoryArena&&) = delete;
        RasterizerMemoryArena& operator=(RasterizerMemoryArena&&) = delete;

        // Stream-aware frames chain begin→end with a GPU event edge: begin_frame
        // waits (on `stream`) for the previous frame's completion event instead of
        // a device-wide sync, and end_frame records the event on `stream` — which
        // must be the stream all of the frame's arena work was enqueued on. Frames
        // without a stream keep the legacy cudaDeviceSynchronize and invalidate
        // the chain.
        //
        // A tenant whose arena work runs on a VULKAN queue (the viewport) must
        // call await_external_release before ending its frame: neither the chain
        // event nor cudaDeviceSynchronize can see in-flight Vulkan work, so the
        // next frame waits the imported timeline value instead.
        uint64_t begin_frame(bool from_rendering = false) { return begin_frame(nullptr, from_rendering); }
        uint64_t begin_frame(cudaStream_t stream, bool from_rendering = false);
        std::optional<uint64_t> try_begin_frame(bool from_rendering = false) { return try_begin_frame(nullptr, from_rendering); }
        std::optional<uint64_t> try_begin_frame(cudaStream_t stream, bool from_rendering = false);
        // Debug-only ownership assertion for the shared CUDA/Vulkan scratch
        // epoch. The preceding frame's CUDA event or imported Vulkan timeline
        // wait must have been installed before this frame can touch offset zero.
        void assert_frame_handoff(uint64_t frame_id) const;

        using RenderHandoffToken = uint64_t;
        // Every live holder renews within a few milliseconds (the parked preview
        // polls at 4 ms, the navigation wait at 1 ms); the lease only bounds what
        // an abandoned or late holder can cost training.
        static constexpr uint32_t kRenderHandoffLeaseMs = 25;
        // Keeps the next idle arena window for a renderer after an ordinary
        // bounded timeout. The short lease survives the caller unwinding model
        // locks, but expires on its own if the viewport is minimized, paused, or
        // otherwise abandons the retry. Supplying the current token renews only
        // that request; an old token can never replace or cancel a newer owner.
        // Training frames still begin first while the turn ledger owes them
        // (owe_training_frames).
        [[nodiscard]] RenderHandoffToken request_render_handoff(RenderHandoffToken current_token = 0);
        void cancel_render_handoff(RenderHandoffToken token);
        [[nodiscard]] bool has_render_handoff(RenderHandoffToken token) const;
        // True while this live reservation still lets training frames begin first.
        [[nodiscard]] bool render_handoff_owes_training(RenderHandoffToken token) const;
        // Stops this reservation from letting any further training frame begin.
        void withdraw_render_handoff_training_frames(RenderHandoffToken token);
        // Lets `frames` training frames begin before a waiting renderer takes the
        // arena. Called by the viewer when it releases a navigation frame.
        void owe_training_frames(uint32_t frames);
        void prepare_viewer_frame();
        void cancel_viewer_prediction();
        [[nodiscard]] RenderHandoffToken reserve_next_viewer_turn(RenderHandoffToken current_token);

        // Measured turn costs, for the viewer's cadence budget.
        struct TurnStats {
            // Median of the last five viewer turns (submit → GPU completion).
            double viewer_turn_ms = 0.0;
            // CPU recording also excludes training from the shared arena.
            double viewer_record_ms = 0.0;
            // Period between consecutive training frames with no viewer turn between.
            double training_step_ms = 0.0;
        };
        [[nodiscard]] TurnStats turn_stats() const;
        // Test seam: registers an already-recorded event as the pending viewer
        // release, standing in for the imported Vulkan timeline. Owns the event;
        // the optional viewer stream remains owned by the caller.
        void set_release_event_for_testing(cudaEvent_t event, cudaStream_t viewer_stream = nullptr);
        // Host-waits for GPU work only within timeout_ms: while the previous
        // CUDA frame is still running on the GPU it polls that frame, then
        // declines like a busy arena, and the caller's reservation keeps the
        // next training frame out until it retries.
        std::optional<uint64_t> try_begin_render_frame_for(
            uint32_t timeout_ms, RenderHandoffToken token = 0);
        // True when a render holding this reservation could begin now without
        // waiting: no frame is active, no other reservation is live, and the
        // previous CUDA frame has finished on the GPU.
        [[nodiscard]] bool render_frame_ready(RenderHandoffToken token = 0) const;
        void end_frame(uint64_t frame_id, bool from_rendering = false) { end_frame(frame_id, nullptr, from_rendering); }
        void end_frame(uint64_t frame_id, cudaStream_t stream, bool from_rendering = false);

        // Registers the Vulkan-side completion of the current frame's arena work
        // (an imported timeline + the value its submit signals). Consumed once by
        // the next begin_frame, which host-waits its release event before
        // admission (falling back to the tenant stream if setup fails).
        // The non-blocking viewer stream has already queued the input uploads.
        // Its owner must call drain_external_release before destroying it.
        void await_external_release(cudaExternalSemaphore_t semaphore, uint64_t value, cudaStream_t viewer_stream);
        // Drain before destroying the imported semaphore, including after the
        // backing has been detached. No arena/model lock may be held.
        void drain_external_release();

        // Caps wait-forever begin_frame() on the CURRENT thread to a bounded
        // wait for its scope — so a cross-thread reader (GUI metric render) that
        // holds render_mutex_ can't deadlock against a refining trainer that
        // holds the arena frame and wants the exclusive lock: the reader's
        // begin_frame bails (throws), the metric is skipped, the lock releases.
        // Training threads never set this, so their acquisition stays blocking.
        static uint32_t set_begin_frame_timeout(uint32_t timeout_ms);

        class ScopedBeginFrameTimeout {
        public:
            explicit ScopedBeginFrameTimeout(uint32_t timeout_ms);
            ~ScopedBeginFrameTimeout();
            ScopedBeginFrameTimeout(const ScopedBeginFrameTimeout&) = delete;
            ScopedBeginFrameTimeout& operator=(const ScopedBeginFrameTimeout&) = delete;

        private:
            uint32_t previous_ = 0;
        };

        std::function<char*(size_t)> get_allocator(uint64_t frame_id,
                                                   const char* label = nullptr);
        std::vector<BufferHandle> get_frame_buffers(uint64_t frame_id) const;
        void cleanup_frames(int keep_recent = 3);
        // Named EXACT-3 boundaries only. Both methods globally gate begin_frame,
        // require zero active training/render frames, and device-synchronize
        // before changing physical mappings. B1 is non-blocking because its caller
        // holds topology locks; B3 waits for an already-active frame to finish.
        bool shrink_to_current_at_boundary();
        bool release_at_boundary();
        void full_reset();
        bool install_external_backing(ExternalBacking backing);
        // With a nonzero timeout, reserve the next idle window against new training
        // frames. Zero keeps the try-only path used by callers that must not wait.
        bool try_install_external_backing(ExternalBacking backing, uint32_t timeout_ms = 0);
        // Grows the committed size of an already-installed external backing whose
        // base pointer is `device_ptr` (which must stay constant). The arena
        // drains all frames and the device, then invokes `commit(new_size)` — which
        // performs the physical grow + Vulkan re-import — inside that safe window;
        // on success the arena's committed capacity is bumped to new_size.
        // With a nonzero timeout, reserve the next idle window against new
        // training frames. The wait is bounded because callers may hold model
        // locks needed by an active training frame. Zero keeps the try-only path.
        enum class ExternalGrowFailure { None,
                                         Busy,
                                         BackingMissing,
                                         CudaFailure,
                                         CommitFailure };
        bool grow_external_backing(const void* device_ptr, size_t new_size,
                                   const std::function<bool(size_t)>& commit,
                                   uint32_t timeout_ms = 0,
                                   ExternalGrowFailure* failure = nullptr);
        void clear_external_backing(const void* device_ptr = nullptr);
        [[nodiscard]] bool using_external_backing(const void* device_ptr = nullptr) const;

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
                                                 std::optional<uint32_t> wait_timeout_ms,
                                                 RenderHandoffToken render_handoff_token = 0,
                                                 bool decline_while_previous_frame_runs = false);
        // A host-consumed release preserves a shared completion proof until the
        // next tenant ends, including when another waiter consumed the release.
        cudaError_t wait_for_previous_frame(cudaStream_t stream);
        // True while the last stream-ordered frame's completion event, or the
        // viewer's pending release, is still pending on the GPU. Caller holds
        // sync_mutex_.
        [[nodiscard]] bool previous_frame_still_running() const;
        // True while a viewer batch registered through await_external_release is
        // still running on the GPU. Caller holds last_frame_event_mutex_.
        [[nodiscard]] bool release_event_pending_locked() const;
        // Host-waits the pending viewer release event, records the turn cost and
        // clears the pending state. Returns false when nothing was pending.
        // Must not be called with sync_mutex_ held.
        bool wait_release_event();
        // Update the completed-turn history without holding a CUDA event lock.
        void update_viewer_turn_cost(double turn_ms);
        bool install_external_backing_impl(ExternalBacking backing, bool wait, uint32_t timeout_ms = 0);
        char* allocate_internal(Arena& arena, size_t size, uint64_t frame_id,
                                const char* label);
        void release_arena_storage(Arena& arena);
        bool grow_arena(Arena& arena, size_t required_size);
        size_t align_size(size_t size) const;
        void record_allocation(uint64_t frame_id, const BufferHandle& handle);
        bool commit_more_memory(Arena& arena, size_t required_size, uint64_t frame_id);
        void decommit_unused_memory(Arena& arena, bool release_all = false,
                                    bool allow_reclaim = true);
        bool shrink_at_boundary(bool release_all);
        void record_commit_timing(uint64_t frame_id, uint64_t elapsed_us, bool committed);
        void record_growth_path_timing(uint64_t frame_id, uint64_t elapsed_us,
                                       uint64_t sync_elapsed_us);
        void record_boundary_timing(bool release_all, uint64_t frame_count, uint64_t elapsed_us);
        void dump_growth_timing() const;
        bool is_vmm_supported(int device) const;
        void empty_cuda_cache();
    };

    class GlobalArenaManager {
    public:
        static GlobalArenaManager& instance() { return global_arena_manager(); }
        RasterizerMemoryArena& get_arena();
        RasterizerMemoryArena* try_get_arena();
        bool try_install_external_backing(RasterizerMemoryArena::ExternalBacking backing,
                                          uint32_t timeout_ms = 0);
        bool grow_external_backing(const void* device_ptr, size_t new_size,
                                   const std::function<bool(size_t)>& commit,
                                   uint32_t timeout_ms = 0,
                                   RasterizerMemoryArena::ExternalGrowFailure* failure = nullptr);
        void clear_external_backing(const void* device_ptr = nullptr);
        void reset();
        void shutdown() { shutdown_global_arena_manager(); }
        /// Reconstructs the arena with a caller-supplied capacity for deterministic OOM tests.
        void reconfigure_for_testing(RasterizerMemoryArena::Config config);

    private:
        friend LFS_CORE_API GlobalArenaManager& global_arena_manager();
        friend LFS_CORE_API void shutdown_global_arena_manager();

        GlobalArenaManager() = default;
        ~GlobalArenaManager() = default;
        std::unique_ptr<RasterizerMemoryArena> arena_;
        std::mutex init_mutex_;
        bool shutdown_ = false;
    };

} // namespace lfs::core

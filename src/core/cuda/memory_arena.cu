/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/assert.hpp"
#include "core/cuda_error.hpp"
#include "core/logger.hpp"
#include "diagnostics/vram_profiler.hpp"
#include "memory_arena.hpp"
#include <algorithm>
#include <cstring>
#include <cuda_runtime.h>
#include <limits>
#include <stdexcept>
#include <thread>

namespace lfs::core {

    // Default constructor implementation
    RasterizerMemoryArena::RasterizerMemoryArena()
        : RasterizerMemoryArena(Config{}) {
    }

    RasterizerMemoryArena::RasterizerMemoryArena(const Config& cfg)
        : config_(cfg),
          creation_time_(std::chrono::steady_clock::now()) {

        // Check if VMM is supported
        int device = -1;
        LFS_CUDA_CHECK_MSG(cudaGetDevice(&device),
                           "RasterizerMemoryArena construction");
        if (config_.enable_vmm && !is_vmm_supported(device)) {
            LOG_WARN("VMM not supported, falling back to smaller allocation");
            config_.initial_commit = 128 << 20;
            config_.max_physical = 4ULL << 30;
        } else if (!config_.enable_vmm) {
            LOG_DEBUG("VMM disabled; using traditional arena allocation");
        }

        LOG_DEBUG("Arena config: virtual=%zu GB, initial=%zu MB, max=%zu GB, granularity=%zu MB",
                  config_.virtual_size >> 30, config_.initial_commit >> 20,
                  config_.max_physical >> 30, config_.granularity >> 20);
    }

    RasterizerMemoryArena::~RasterizerMemoryArena() {
        dump_statistics();

        {
            std::lock_guard<std::mutex> event_lock(last_frame_event_mutex_);
            if (last_frame_event_) {
                const cudaError_t destroy_status = cudaEventDestroy(last_frame_event_);
                if (destroy_status != cudaSuccess) {
                    ensure_cuda_success(
                        destroy_status, "cudaEventDestroy(arena frame completion)",
                        "context=arena destruction", LFS_SOURCE_SITE_CURRENT(),
                        CudaFailureDisposition::LogOnlyNoLatch);
                }
                last_frame_event_ = nullptr;
                last_frame_event_valid_ = false;
            }
        }

        std::lock_guard<std::mutex> lock(arena_mutex_);
        for (auto& entry : device_arenas_) {
            auto& arena_ptr = entry.second;
            if (arena_ptr) {
                release_arena_storage(*arena_ptr);
            }
        }

        device_arenas_.clear();
        frame_contexts_.clear();
    }

    RasterizerMemoryArena::RasterizerMemoryArena(RasterizerMemoryArena&& other) noexcept {
        std::scoped_lock lock(other.arena_mutex_, other.frame_mutex_, other.sync_mutex_,
                              other.last_frame_event_mutex_);
        device_arenas_ = std::move(other.device_arenas_);
        frame_contexts_ = std::move(other.frame_contexts_);
        config_ = other.config_;
        frame_counter_ = other.frame_counter_.load();
        generation_counter_ = other.generation_counter_.load();
        creation_time_ = other.creation_time_;
        total_frames_processed_ = other.total_frames_processed_.load();
        active_frames_ = other.active_frames_;
        pending_render_frames_ = other.pending_render_frames_;
        active_training_frames_ = other.active_training_frames_;
        last_frame_event_ = other.last_frame_event_;
        last_frame_event_valid_ = other.last_frame_event_valid_;
        external_release_semaphore_ = other.external_release_semaphore_;
        external_release_value_ = other.external_release_value_;
        other.last_frame_event_ = nullptr;
        other.last_frame_event_valid_ = false;
        other.external_release_semaphore_ = nullptr;
        other.external_release_value_ = 0;
    }

    RasterizerMemoryArena& RasterizerMemoryArena::operator=(RasterizerMemoryArena&& other) noexcept {
        if (this != &other) {
            std::scoped_lock lock(
                arena_mutex_,
                other.arena_mutex_,
                frame_mutex_,
                other.frame_mutex_,
                sync_mutex_,
                other.sync_mutex_,
                last_frame_event_mutex_,
                other.last_frame_event_mutex_);
            device_arenas_ = std::move(other.device_arenas_);
            frame_contexts_ = std::move(other.frame_contexts_);
            config_ = other.config_;
            frame_counter_ = other.frame_counter_.load();
            generation_counter_ = other.generation_counter_.load();
            creation_time_ = other.creation_time_;
            total_frames_processed_ = other.total_frames_processed_.load();
            active_frames_ = other.active_frames_;
            pending_render_frames_ = other.pending_render_frames_;
            active_training_frames_ = other.active_training_frames_;
            if (last_frame_event_) {
                const cudaError_t destroy_status = cudaEventDestroy(last_frame_event_);
                if (destroy_status != cudaSuccess) {
                    ensure_cuda_success(
                        destroy_status, "cudaEventDestroy(arena frame completion)",
                        "context=arena move assignment", LFS_SOURCE_SITE_CURRENT(),
                        CudaFailureDisposition::LogOnlyNoLatch);
                }
            }
            last_frame_event_ = other.last_frame_event_;
            last_frame_event_valid_ = other.last_frame_event_valid_;
            external_release_semaphore_ = other.external_release_semaphore_;
            external_release_value_ = other.external_release_value_;
            other.last_frame_event_ = nullptr;
            other.last_frame_event_valid_ = false;
            other.external_release_semaphore_ = nullptr;
            other.external_release_value_ = 0;
        }
        return *this;
    }

    bool RasterizerMemoryArena::is_vmm_supported(int device) const {
        // Check compute capability
        int major = 0, minor = 0;
        const cudaError_t major_status =
            cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
        if (major_status != cudaSuccess) {
            ensure_cuda_success(
                major_status, "cudaDeviceGetAttribute(compute capability major)",
                detail::format_cuda_safe("device={}, fallback=disable VMM", device),
                LFS_SOURCE_SITE_CURRENT(), CudaFailureDisposition::LogOnly);
            return false;
        }
        const cudaError_t minor_status =
            cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
        if (minor_status != cudaSuccess) {
            ensure_cuda_success(
                minor_status, "cudaDeviceGetAttribute(compute capability minor)",
                detail::format_cuda_safe("device={}, fallback=disable VMM", device),
                LFS_SOURCE_SITE_CURRENT(), CudaFailureDisposition::LogOnly);
            return false;
        }

        // VMM requires compute capability 6.0+
        if (major < 6) {
            return false;
        }

        // Check if we can get allocation granularity (VMM function)
        CUmemAllocationProp prop = {};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = device;

        size_t granularity = 0;
        CUresult result = cuMemGetAllocationGranularity(&granularity, &prop,
                                                        CU_MEM_ALLOC_GRANULARITY_MINIMUM);

        // If this VMM function succeeds, VMM is supported
        if (result == CUDA_SUCCESS && granularity > 0) {
            return true;
        }

        // VMM not supported or not available
        return false;
    }

    namespace {
        // Per-thread cap on wait-forever begin_frame(); 0 = block forever.
        thread_local uint32_t tl_begin_frame_timeout_ms = 0;
    } // namespace

    RasterizerMemoryArena::ScopedBeginFrameTimeout::ScopedBeginFrameTimeout(uint32_t timeout_ms)
        : previous_(tl_begin_frame_timeout_ms) {
        tl_begin_frame_timeout_ms = timeout_ms;
    }

    RasterizerMemoryArena::ScopedBeginFrameTimeout::~ScopedBeginFrameTimeout() {
        tl_begin_frame_timeout_ms = previous_;
    }

    uint64_t RasterizerMemoryArena::begin_frame(cudaStream_t stream, bool from_rendering) {
        // A thread that opted into bounded acquisition (GUI metric render) gets
        // a timed wait instead of waiting forever, so it can't deadlock holding
        // render_mutex_ against a refining trainer that holds the arena frame.
        const std::optional<uint32_t> wait =
            tl_begin_frame_timeout_ms > 0 ? std::optional<uint32_t>(tl_begin_frame_timeout_ms)
                                          : std::optional<uint32_t>(0u);
        auto frame_id = begin_frame_impl(stream, from_rendering, wait);
        if (!frame_id) {
            throw std::runtime_error("RasterizerMemoryArena::begin_frame failed to acquire arena frame");
        }
        return *frame_id;
    }

    std::optional<uint64_t> RasterizerMemoryArena::try_begin_frame(cudaStream_t stream, bool from_rendering) {
        return begin_frame_impl(stream, from_rendering, std::nullopt);
    }

    std::optional<uint64_t> RasterizerMemoryArena::try_begin_frame_for(uint32_t timeout_ms,
                                                                       cudaStream_t stream,
                                                                       bool from_rendering) {
        return begin_frame_impl(stream, from_rendering, timeout_ms);
    }

    void RasterizerMemoryArena::note_external_release(cudaExternalSemaphore_t semaphore, uint64_t value) {
        std::lock_guard<std::mutex> lock(last_frame_event_mutex_);
        external_release_semaphore_ = semaphore;
        external_release_value_ = value;
    }

    void RasterizerMemoryArena::drain_external_release() {
        cudaExternalSemaphore_t release_semaphore = nullptr;
        uint64_t release_value = 0;
        {
            std::lock_guard<std::mutex> lock(last_frame_event_mutex_);
            release_semaphore = external_release_semaphore_;
            release_value = external_release_value_;
            external_release_semaphore_ = nullptr;
            external_release_value_ = 0;
        }
        if (release_semaphore == nullptr || release_value == 0) {
            return;
        }
        // A device sync cannot observe the in-flight Vulkan batch that signals
        // this release, so host-block on the fence before backing is freed or
        // replaced under it.
        cudaExternalSemaphoreWaitParams wait_params{};
        wait_params.params.fence.value = release_value;
        const cudaError_t wait_status =
            cudaWaitExternalSemaphoresAsync(&release_semaphore, &wait_params, 1, nullptr);
        if (wait_status != cudaSuccess) {
            ensure_cuda_success(
                wait_status, "cudaWaitExternalSemaphoresAsync(arena drain)",
                detail::format_cuda_safe("release_value={}, fallback=device sync", release_value),
                LFS_SOURCE_SITE_CURRENT(), CudaFailureDisposition::LogOnly);
            // The GPU-side fence wait couldn't be enqueued; fall back to a full
            // device sync so the caller doesn't free/replace the backing under
            // in-flight CUDA work (the Vulkan batch can't be observed here).
            const cudaError_t sync_status = cudaDeviceSynchronize();
            if (sync_status != cudaSuccess) {
                ensure_cuda_success(
                    sync_status, "cudaDeviceSynchronize(arena drain fallback)",
                    detail::format_cuda_safe("release_value={}", release_value),
                    LFS_SOURCE_SITE_CURRENT(), CudaFailureDisposition::LogOnly);
            }
            return;
        }
        const cudaError_t sync_status = cudaStreamSynchronize(nullptr);
        if (sync_status != cudaSuccess) {
            ensure_cuda_success(
                sync_status, "cudaStreamSynchronize(arena drain)",
                detail::format_cuda_safe("release_value={}", release_value),
                LFS_SOURCE_SITE_CURRENT(), CudaFailureDisposition::LogOnly);
        }
    }

    // Orders the new frame's work after the previous frame before the arena
    // offset resets and memory gets overwritten. Stream-aware frames chain via
    // the completion event; streamless frames or a broken chain fall back to a
    // device-wide sync. A pending Vulkan release is waited explicitly because
    // neither the chain event nor a device sync can see in-flight Vulkan work.
    cudaError_t RasterizerMemoryArena::wait_for_previous_frame(cudaStream_t stream) {
        cudaExternalSemaphore_t release_semaphore = nullptr;
        uint64_t release_value = 0;
        bool chain_ok = false;
        {
            std::lock_guard<std::mutex> lock(last_frame_event_mutex_);
            release_semaphore = external_release_semaphore_;
            release_value = external_release_value_;
            external_release_semaphore_ = nullptr;
            external_release_value_ = 0;
            if (stream) {
                if (last_frame_event_valid_) {
                    const cudaError_t chain_status =
                        cudaStreamWaitEvent(stream, last_frame_event_, 0);
                    chain_ok = chain_status == cudaSuccess;
                    if (!chain_ok) {
                        ensure_cuda_success(
                            chain_status, "cudaStreamWaitEvent(arena frame chain)",
                            "fallback=device sync", LFS_SOURCE_SITE_CURRENT(),
                            CudaFailureDisposition::LogOnly);
                    }
                }
            }
        }

        if (release_semaphore != nullptr && release_value != 0) {
            // Enqueue on the frame's stream (or the default stream for streamless
            // frames, where the device sync below then blocks until it passes).
            cudaExternalSemaphoreWaitParams wait_params{};
            wait_params.params.fence.value = release_value;
            const cudaStream_t wait_stream = stream;
            const cudaError_t wait_status = cudaWaitExternalSemaphoresAsync(
                &release_semaphore, &wait_params, 1, wait_stream);
            if (wait_status != cudaSuccess) {
                ensure_cuda_success(
                    wait_status, "cudaWaitExternalSemaphoresAsync(arena frame)",
                    detail::format_cuda_safe("release_value={}, fallback=default stream", release_value),
                    LFS_SOURCE_SITE_CURRENT(), CudaFailureDisposition::LogOnly);
                // The GPU-side wait wasn't enqueued; host-block on the fence so the
                // arena isn't reset/reused while the Vulkan batch still reads it
                // (the device-sync fallback below only covers in-flight CUDA work).
                const cudaError_t fallback_wait_status =
                    cudaWaitExternalSemaphoresAsync(
                        &release_semaphore, &wait_params, 1, nullptr);
                if (fallback_wait_status != cudaSuccess) {
                    return fallback_wait_status;
                }
                const cudaError_t fallback_sync_status = cudaStreamSynchronize(nullptr);
                if (fallback_sync_status != cudaSuccess) {
                    return fallback_sync_status;
                }
                chain_ok = false;
            } else if (wait_stream != nullptr) {
                // The Vulkan tenant device-synced all prior CUDA work at its own
                // streamless begin, and its arena work is Vulkan-only — this
                // wait alone re-establishes the chain GPU-side.
                chain_ok = true;
            }
        }

        if (chain_ok) {
            return cudaSuccess;
        }
        return cudaDeviceSynchronize();
    }

    std::optional<uint64_t> RasterizerMemoryArena::begin_frame_impl(cudaStream_t stream, bool from_rendering,
                                                                    std::optional<uint32_t> wait_timeout_ms) {
        LFS_CUDA_BREADCRUMB_STREAM("arena.begin_frame", stream);
        {
            std::unique_lock<std::mutex> sync_lock(sync_mutex_);
            const auto can_begin = [this, from_rendering]() {
                return active_frames_ == 0 && (from_rendering || pending_render_frames_ == 0);
            };
            if (!wait_timeout_ms.has_value()) {
                if (!can_begin()) {
                    return std::nullopt;
                }
            } else if (*wait_timeout_ms == 0u) {
                sync_cv_.wait(sync_lock, can_begin);
            } else if (!sync_cv_.wait_for(sync_lock, std::chrono::milliseconds(*wait_timeout_ms), can_begin)) {
                return std::nullopt;
            }
            ++active_frames_;
            if (!from_rendering) {
                ++active_training_frames_;
            }
        }

        uint64_t frame_id = frame_counter_.fetch_add(1, std::memory_order_relaxed);

        const cudaError_t wait_status = wait_for_previous_frame(stream);
        if (wait_status != cudaSuccess) {
            end_frame(frame_id, from_rendering);
            LFS_ENSURE_CUDA_SUCCESS_MSG(
                wait_status, "RasterizerMemoryArena::wait_for_previous_frame",
                detail::format_cuda_safe("frame_id={}, stream={}", frame_id, static_cast<void*>(stream)));
        }

        // CRITICAL FIX: Reset arena offset at the beginning of each frame!
        int device = -1;
        const cudaError_t device_status = cudaGetDevice(&device);
        if (device_status != cudaSuccess) {
            end_frame(frame_id, from_rendering);
            LFS_ENSURE_CUDA_SUCCESS_MSG(
                device_status, "cudaGetDevice(arena begin frame)",
                detail::format_cuda_safe("frame_id={}", frame_id));
        }
        {
            std::lock_guard<std::mutex> arena_lock(arena_mutex_);
            auto it = device_arenas_.find(device);
            if (it != device_arenas_.end() && it->second) {
                // Reset the offset to reuse the buffer from the beginning
                it->second->offset.store(0, std::memory_order_release);

                // Log memory status periodically (but not too often)
                bool should_log = (frame_id == 1) || (frame_id % config_.log_interval == 0);

                if (should_log) {
                    log_memory_status(frame_id, true);
                }
            }
        }

        std::lock_guard<std::mutex> lock(frame_mutex_);

        // Create new frame context
        FrameContext& ctx = frame_contexts_[frame_id];
        ctx.frame_id = frame_id;
        ctx.generation = generation_counter_.load(std::memory_order_relaxed);
        ctx.is_active = true;
        ctx.timestamp = std::chrono::steady_clock::now();
        ctx.buffers.clear();
        ctx.total_allocated = 0;

        total_frames_processed_.fetch_add(1, std::memory_order_relaxed);

        return frame_id;
    }

    void RasterizerMemoryArena::end_frame(uint64_t frame_id, cudaStream_t stream, bool from_rendering) {
        // Record the frame's completion event before releasing frame ownership:
        // once active_frames_ drops, the next begin_frame may chain on it. A
        // frame ended without a stream breaks the chain (next begin device-syncs).
        {
            std::lock_guard<std::mutex> event_lock(last_frame_event_mutex_);
            if (stream) {
                if (!last_frame_event_) {
                    const cudaError_t create_status = cudaEventCreateWithFlags(
                        &last_frame_event_, cudaEventDisableTiming);
                    if (create_status != cudaSuccess) {
                        ensure_cuda_success(
                            create_status, "cudaEventCreateWithFlags(arena frame completion)",
                            detail::format_cuda_safe("frame_id={}", frame_id),
                            LFS_SOURCE_SITE_CURRENT(), CudaFailureDisposition::LogOnly);
                        last_frame_event_ = nullptr;
                    }
                }
                if (last_frame_event_) {
                    const cudaError_t record_status = cudaEventRecord(last_frame_event_, stream);
                    last_frame_event_valid_ = record_status == cudaSuccess;
                    if (!last_frame_event_valid_) {
                        ensure_cuda_success(
                            record_status, "cudaEventRecord(arena frame completion)",
                            detail::format_cuda_safe("frame_id={}, stream={}", frame_id,
                                                     static_cast<void*>(stream)),
                            LFS_SOURCE_SITE_CURRENT(), CudaFailureDisposition::LogOnly);
                    }
                } else {
                    last_frame_event_valid_ = false;
                }
            } else {
                last_frame_event_valid_ = false;
            }
        }

        // Track peak usage before resetting
        int device;
        cudaError_t err = cudaGetDevice(&device);
        if (err == cudaSuccess) {
            std::lock_guard<std::mutex> lock(arena_mutex_);
            auto it = device_arenas_.find(device);
            if (it != device_arenas_.end() && it->second) {
                size_t frame_usage = it->second->offset.load(std::memory_order_relaxed);

                // Update overall peak
                size_t current_peak = it->second->peak_usage.load(std::memory_order_relaxed);
                while (frame_usage > current_peak) {
                    if (it->second->peak_usage.compare_exchange_weak(current_peak, frame_usage)) {
                        break;
                    }
                }

                // Update period peak (for logging)
                size_t period_peak = it->second->peak_usage_period.load(std::memory_order_relaxed);
                while (frame_usage > period_peak) {
                    if (it->second->peak_usage_period.compare_exchange_weak(period_peak, frame_usage)) {
                        break;
                    }
                }
            }
        } else {
            ensure_cuda_success(
                err, "cudaGetDevice(arena end frame)",
                detail::format_cuda_safe("frame_id={}", frame_id), LFS_SOURCE_SITE_CURRENT(),
                CudaFailureDisposition::LogOnly);
        }

        std::lock_guard<std::mutex> lock(frame_mutex_);

        auto it = frame_contexts_.find(frame_id);
        if (it != frame_contexts_.end()) {
            it->second.is_active = false;
        }

        // Cleanup old frames - keep only last 3
        cleanup_frames(3);

        {
            std::lock_guard<std::mutex> sync_lock(sync_mutex_);
            if (active_frames_ > 0) {
                --active_frames_;
            } else {
                LOG_WARN("RasterizerMemoryArena::end_frame called with no active arena frames");
            }
            if (!from_rendering) {
                if (active_training_frames_ > 0) {
                    --active_training_frames_;
                } else {
                    LOG_WARN("RasterizerMemoryArena::end_frame called with no active training frames");
                }
            }
        }
        sync_cv_.notify_all();
    }

    void RasterizerMemoryArena::log_memory_status(uint64_t frame_id, bool force) {
        // Called with arena_mutex_ already held
        int device;
        cudaError_t err = cudaGetDevice(&device);
        if (err != cudaSuccess) {
            ensure_cuda_success(
                err, "cudaGetDevice(arena memory status)",
                detail::format_cuda_safe("frame_id={}", frame_id), LFS_SOURCE_SITE_CURRENT(),
                CudaFailureDisposition::LogOnly);
            return;
        }

        auto it = device_arenas_.find(device);
        if (it == device_arenas_.end() || !it->second)
            return;

        auto& arena = *it->second;

        // Get memory info
        size_t committed_mb = arena.committed_size >> 20;
        size_t current_usage_mb = arena.offset.load() >> 20;
        size_t peak_period_mb = arena.peak_usage_period.load() >> 20;
        size_t peak_overall_mb = arena.peak_usage.load() >> 20;

        // Calculate utilization and fragmentation metrics
        float utilization = arena.committed_size > 0 ? (100.0f * arena.peak_usage_period.load() / arena.committed_size) : 0.0f;

        // Fragmentation = (committed - peak_used) / committed
        // 0% = no fragmentation (using all committed memory)
        // 100% = complete fragmentation (using none of committed memory)
        float fragmentation = arena.committed_size > 0 ? (100.0f * (arena.committed_size - arena.peak_usage_period.load()) / arena.committed_size) : 0.0f;

        size_t wasted_mb = (arena.committed_size - arena.peak_usage_period.load()) >> 20;

        // Analyze actual allocations from frame contexts
        size_t total_frame_allocations = 0;
        size_t num_active_frames = 0;
        size_t max_frame_alloc = 0;
        {
            std::lock_guard<std::mutex> lock(frame_mutex_);
            for (const auto& entry : frame_contexts_) {
                const auto fid = entry.first;
                const auto& ctx = entry.second;
                if (ctx.is_active || (frame_id - fid) < config_.log_interval) {
                    total_frame_allocations += ctx.total_allocated;
                    num_active_frames++;
                    max_frame_alloc = std::max(max_frame_alloc, ctx.total_allocated);
                }
            }
        }

        LOG_DEBUG("Arena frame %lu: %zu MB committed, %zu MB peak (%.0f%% used), %zu reallocs",
                  frame_id, committed_mb, peak_period_mb, utilization, arena.realloc_count.load());

        // Reset period peak for next logging interval
        arena.peak_usage_period.store(0, std::memory_order_release);
        arena.last_log_time = std::chrono::steady_clock::now();
    }

    std::function<char*(size_t)> RasterizerMemoryArena::get_allocator(uint64_t frame_id) {
        return [this, frame_id](size_t size) -> char* {
            if (size == 0) {
                return nullptr;
            }

            int device;
            LFS_CUDA_CHECK_MSG(cudaGetDevice(&device),
                               "RasterizerMemoryArena allocation");

            Arena& arena = get_or_create_arena(device);
            return allocate_internal(arena, size, frame_id);
        };
    }

    std::vector<RasterizerMemoryArena::BufferHandle>
    RasterizerMemoryArena::get_frame_buffers(uint64_t frame_id) const {
        std::lock_guard<std::mutex> lock(frame_mutex_);

        auto it = frame_contexts_.find(frame_id);
        if (it != frame_contexts_.end()) {
            return it->second.buffers;
        }

        return {};
    }

    void RasterizerMemoryArena::reset_frame(uint64_t frame_id) {
        std::lock_guard<std::mutex> lock(frame_mutex_);

        auto it = frame_contexts_.find(frame_id);
        if (it != frame_contexts_.end()) {
            // Keep buffers but mark as reusable
            it->second.total_allocated = 0;
        }
    }

    void RasterizerMemoryArena::cleanup_frames(int keep_recent) {
        // Called with frame_mutex_ already held

        if (frame_contexts_.size() <= static_cast<size_t>(keep_recent)) {
            return;
        }

        // Find oldest frames to remove
        std::vector<uint64_t> frame_ids;
        frame_ids.reserve(frame_contexts_.size());

        for (const auto& entry : frame_contexts_) {
            const auto id = entry.first;
            const auto& ctx = entry.second;
            if (!ctx.is_active) {
                frame_ids.push_back(id);
            }
        }

        if (frame_ids.size() <= static_cast<size_t>(keep_recent)) {
            return;
        }

        // Sort by frame ID (oldest first)
        std::sort(frame_ids.begin(), frame_ids.end());

        // Remove oldest frames
        size_t to_remove = frame_ids.size() - keep_recent;
        for (size_t i = 0; i < to_remove; ++i) {
            frame_contexts_.erase(frame_ids[i]);
        }
    }

    void RasterizerMemoryArena::empty_cuda_cache() {
        // Try to force CUDA to release cached memory
        // This is less aggressive than cudaDeviceReset()
        const cudaError_t sync_status = cudaDeviceSynchronize();
        if (sync_status != cudaSuccess) {
            ensure_cuda_success(
                sync_status, "cudaDeviceSynchronize(arena cache trim)", {},
                LFS_SOURCE_SITE_CURRENT(), CudaFailureDisposition::LogOnly);
            return;
        }

        // Allocate and immediately free a small buffer to trigger cleanup
        void* dummy = nullptr;
        const cudaError_t allocation_status = cudaMalloc(&dummy, 1);
        if (allocation_status != cudaSuccess) {
            ensure_cuda_success(
                allocation_status, "cudaMalloc(arena cache trim probe)", {},
                LFS_SOURCE_SITE_CURRENT(), CudaFailureDisposition::LogOnly);
            return;
        }
        const cudaError_t free_status = cudaFree(dummy);
        if (free_status != cudaSuccess) {
            ensure_cuda_success(
                free_status, "cudaFree(arena cache trim probe)", {},
                LFS_SOURCE_SITE_CURRENT(), CudaFailureDisposition::LogOnlyNoLatch);
        }
    }

    void RasterizerMemoryArena::release_arena_storage(Arena& arena) {
        {
            std::lock_guard<std::mutex> chunk_lock(arena.chunks_mutex);
            for (auto& chunk : arena.chunks) {
                if (chunk.is_mapped) {
                    lfs::diagnostics::VramProfiler::instance().recordDeallocation(
                        reinterpret_cast<void*>(arena.d_ptr + chunk.offset));
                    cuMemUnmap(arena.d_ptr + chunk.offset, chunk.size);
                    cuMemRelease(chunk.handle);
                    chunk.is_mapped = false;
                }
            }
            arena.chunks.clear();
        }

        if (arena.d_ptr) {
            cuMemAddressFree(arena.d_ptr, arena.virtual_size);
            arena.d_ptr = 0;
            arena.virtual_size = 0;
        }

        if (arena.fallback_buffer && arena.owns_fallback_buffer) {
            lfs::diagnostics::VramProfiler::instance().recordDeallocation(arena.fallback_buffer);
            const cudaError_t free_status = cudaFree(arena.fallback_buffer);
            if (free_status != cudaSuccess) {
                ensure_cuda_success(
                    free_status, "cudaFree(arena backing)",
                    detail::format_cuda_safe("capacity_bytes={}", arena.capacity),
                    LFS_SOURCE_SITE_CURRENT(), CudaFailureDisposition::LogOnlyNoLatch);
            }
        }
        arena.fallback_buffer = nullptr;
        arena.owns_fallback_buffer = true;
        arena.external_backing = false;
        arena.external_owner.reset();
        arena.external_label.clear();
        arena.external_grow = nullptr;
        arena.committed_size = 0;
        arena.capacity = 0;
        arena.offset.store(0, std::memory_order_release);
    }

    void RasterizerMemoryArena::full_reset() {
        std::unique_lock<std::mutex> sync_lock(sync_mutex_);

        sync_cv_.wait(sync_lock, [this]() {
            return active_frames_ == 0 && pending_render_frames_ == 0;
        });

        // A submitted viewport batch may still be reading arena scratch; drain
        // its release fence before the reset frees or decommits the backing.
        drain_external_release();

        const std::scoped_lock lock(arena_mutex_, frame_mutex_);

        frame_contexts_.clear();

        for (auto& entry : device_arenas_) {
            const auto device = entry.first;
            auto& arena = entry.second;
            if (arena) {
                LFS_CUDA_CHECK_MSG(cudaSetDevice(device),
                                   "RasterizerMemoryArena reset device={}", device);
                LFS_CUDA_CHECK_MSG(cudaDeviceSynchronize(),
                                   "RasterizerMemoryArena reset device={}", device);

                // No frame can own the arena now. Publish the empty high-water
                // before deciding which VMM chunks are unused, otherwise the old
                // frame offset pins every chunk it crossed.
                arena->offset.store(0, std::memory_order_release);
                decommit_unused_memory(*arena);
            }
        }

        active_frames_ = 0;
        pending_render_frames_ = 0;
        active_training_frames_ = 0;

        empty_cuda_cache();
        sync_lock.unlock();
        sync_cv_.notify_all();
        LOG_DEBUG("Arena reset");
    }

    bool RasterizerMemoryArena::install_external_backing(ExternalBacking backing) {
        return install_external_backing_impl(std::move(backing), true);
    }

    bool RasterizerMemoryArena::try_install_external_backing(ExternalBacking backing) {
        return install_external_backing_impl(std::move(backing), false);
    }

    bool RasterizerMemoryArena::install_external_backing_impl(ExternalBacking backing, bool wait) {
        if (!backing.valid()) {
            LOG_WARN("RasterizerMemoryArena::install_external_backing called with invalid backing");
            return false;
        }

        std::unique_lock<std::mutex> sync_lock(sync_mutex_);
        const auto can_install = [this]() {
            return active_frames_ == 0 && pending_render_frames_ == 0;
        };
        if (wait) {
            sync_cv_.wait(sync_lock, can_install);
        } else if (!can_install()) {
            return false;
        }

        // A submitted viewport batch may still be reading the current backing;
        // drain its release fence before the swap releases that storage.
        drain_external_release();

        const cudaError_t set_device_status = cudaSetDevice(backing.device);
        if (set_device_status != cudaSuccess) {
            ensure_cuda_success(
                set_device_status, "cudaSetDevice(external arena install)",
                detail::format_cuda_safe("device={}", backing.device), LFS_SOURCE_SITE_CURRENT(),
                CudaFailureDisposition::LogOnly);
            return false;
        }
        const cudaError_t sync_status = cudaDeviceSynchronize();
        if (sync_status != cudaSuccess) {
            ensure_cuda_success(
                sync_status, "cudaDeviceSynchronize(external arena install)",
                detail::format_cuda_safe("device={}", backing.device), LFS_SOURCE_SITE_CURRENT(),
                CudaFailureDisposition::LogOnly);
            return false;
        }

        std::scoped_lock lock(arena_mutex_, frame_mutex_);
        frame_contexts_.clear();

        auto& arena_ptr = device_arenas_[backing.device];
        if (!arena_ptr) {
            arena_ptr = std::make_unique<Arena>();
        } else {
            release_arena_storage(*arena_ptr);
        }

        auto& arena = *arena_ptr;
        arena.device = backing.device;
        arena.fallback_buffer = backing.device_ptr;
        arena.owns_fallback_buffer = false;
        arena.external_backing = true;
        arena.external_owner = std::move(backing.owner);
        arena.external_label = std::move(backing.label);
        arena.external_grow = std::move(backing.grow);
        arena.committed_size = backing.size;
        arena.capacity = backing.size;
        arena.granularity = std::max(config_.alignment, config_.granularity);
        arena.generation = generation_counter_.fetch_add(1, std::memory_order_relaxed);
        arena.last_log_time = std::chrono::steady_clock::now();
        arena.offset.store(0, std::memory_order_release);
        arena.peak_usage_period.store(0, std::memory_order_release);
        arena.total_allocated.store(0, std::memory_order_release);
        arena.realloc_count.fetch_add(1, std::memory_order_relaxed);

        LOG_INFO("Rasterizer arena now uses external CUDA backing '%s' size=%zu MiB ptr=%p",
                 arena.external_label.empty() ? "unnamed" : arena.external_label.c_str(),
                 static_cast<size_t>(backing.size >> 20),
                 backing.device_ptr);
        sync_lock.unlock();
        sync_cv_.notify_all();
        return true;
    }

    void RasterizerMemoryArena::clear_external_backing(const void* device_ptr) {
        std::unique_lock<std::mutex> sync_lock(sync_mutex_);
        sync_cv_.wait(sync_lock, [this]() {
            return active_frames_ == 0 && pending_render_frames_ == 0;
        });

        LFS_CUDA_CHECK_MSG(cudaDeviceSynchronize(),
                           "clearing external RasterizerMemoryArena backing");

        std::scoped_lock lock(arena_mutex_, frame_mutex_);
        for (auto it = device_arenas_.begin(); it != device_arenas_.end();) {
            if (!it->second) {
                it = device_arenas_.erase(it);
                continue;
            }
            auto& arena = *it->second;
            const bool matches =
                arena.external_backing &&
                (device_ptr == nullptr || arena.fallback_buffer == device_ptr);
            if (!matches) {
                ++it;
                continue;
            }

            LOG_INFO("Rasterizer arena released external CUDA backing '%s' size=%zu MiB ptr=%p",
                     arena.external_label.empty() ? "unnamed" : arena.external_label.c_str(),
                     static_cast<size_t>(arena.committed_size >> 20),
                     arena.fallback_buffer);
            release_arena_storage(arena);
            it = device_arenas_.erase(it);
        }
        frame_contexts_.clear();
        sync_lock.unlock();
        sync_cv_.notify_all();
    }

    bool RasterizerMemoryArena::grow_external_backing(const void* device_ptr, size_t new_size,
                                                      const std::function<bool(size_t)>& commit) {
        // Non-blocking on purpose: this is called by the render thread while it
        // holds the trainer's render_mutex_ (shared). Waiting here for the arena
        // to drain would deadlock against a refining training step that holds the
        // arena frame and is itself blocked on render_mutex_ (write). If the arena
        // is busy we bail; the caller falls back to a cached frame and retries.
        std::unique_lock<std::mutex> sync_lock(sync_mutex_, std::try_to_lock);
        if (!sync_lock.owns_lock() || active_frames_ != 0 || pending_render_frames_ != 0) {
            return false;
        }

        const cudaError_t sync_status = cudaDeviceSynchronize();
        if (sync_status != cudaSuccess) {
            ensure_cuda_success(
                sync_status, "cudaDeviceSynchronize(external arena growth)",
                detail::format_cuda_safe("requested_bytes={}", new_size),
                LFS_SOURCE_SITE_CURRENT(), CudaFailureDisposition::LogOnly);
            return false;
        }

        std::scoped_lock lock(arena_mutex_, frame_mutex_);

        Arena* target = nullptr;
        for (auto& entry : device_arenas_) {
            auto& arena_ptr = entry.second;
            if (arena_ptr && arena_ptr->external_backing && arena_ptr->fallback_buffer == device_ptr) {
                target = arena_ptr.get();
                break;
            }
        }
        if (!target) {
            sync_lock.unlock();
            sync_cv_.notify_all();
            return false;
        }
        if (new_size <= target->committed_size) {
            sync_lock.unlock();
            sync_cv_.notify_all();
            return true;
        }

        // commit() performs the in-place physical grow + Vulkan re-import while the
        // device is drained and no frame is active. device_ptr must stay constant.
        if (!commit(new_size)) {
            sync_lock.unlock();
            sync_cv_.notify_all();
            return false;
        }

        target->committed_size = new_size;
        target->capacity = new_size;
        target->realloc_count.fetch_add(1, std::memory_order_relaxed);
        target->offset.store(0, std::memory_order_release);
        frame_contexts_.clear();

        LOG_INFO("Rasterizer external arena grew in place: ptr=%p capacity=%zu MiB",
                 device_ptr, static_cast<size_t>(new_size >> 20));

        sync_lock.unlock();
        sync_cv_.notify_all();
        return true;
    }

    bool RasterizerMemoryArena::using_external_backing() const {
        std::lock_guard<std::mutex> lock(arena_mutex_);
        for (const auto& entry : device_arenas_) {
            const auto& arena_ptr = entry.second;
            if (arena_ptr && arena_ptr->external_backing) {
                return true;
            }
        }
        return false;
    }

    RasterizerMemoryArena::Arena& RasterizerMemoryArena::get_or_create_arena(int device) {
        LFS_CUDA_BREADCRUMB("arena.get_or_create");
        std::lock_guard<std::mutex> lock(arena_mutex_);

        LFS_CUDA_CHECK_MSG(cudaSetDevice(device),
                           "RasterizerMemoryArena initialization device={}", device);

        auto& arena_ptr = device_arenas_[device];
        if (!arena_ptr) {
            arena_ptr = std::make_unique<Arena>();
            arena_ptr->device = device;
            arena_ptr->last_log_time = std::chrono::steady_clock::now();

            if (config_.enable_vmm && is_vmm_supported(device)) {
                // Use VMM path
                auto& arena = *arena_ptr;

                // Get allocation granularity
                CUmemAllocationProp prop = {};
                prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
                prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                prop.location.id = device;

                size_t granularity = 0;
                CUresult result = cuMemGetAllocationGranularity(&granularity, &prop,
                                                                CU_MEM_ALLOC_GRANULARITY_MINIMUM);
                if (result != CUDA_SUCCESS) {
                    // Fall back to default granularity
                    granularity = 2 << 20; // 2MB
                }
                arena.granularity = std::max(granularity, config_.granularity);

                // Reserve virtual address space (this is FREE!)
                arena.virtual_size = config_.virtual_size;
                result = cuMemAddressReserve(&arena.d_ptr, arena.virtual_size, 0, 0, 0);
                if (result != CUDA_SUCCESS) {
                    LOG_WARN("VMM reservation failed, using traditional allocation");
                    arena.d_ptr = 0;
                    arena.virtual_size = 0;
                } else {
                    LOG_INFO("VMM initialized: device=%d, virtual=%zu GB", device, arena.virtual_size >> 30);

                    if (!commit_more_memory(arena, config_.initial_commit)) {
                        cuMemAddressFree(arena.d_ptr, arena.virtual_size);
                        arena.d_ptr = 0;
                        arena.virtual_size = 0;
                        LOG_WARN("Initial commit failed, falling back to traditional allocation");
                    } else {
                        arena.generation = generation_counter_.fetch_add(1, std::memory_order_relaxed);
                        arena.offset.store(0, std::memory_order_release);
                        return *arena_ptr;
                    }
                }
            }

            // Fallback to traditional allocation (either VMM not supported or failed)
            auto& arena = *arena_ptr;

            // Check available memory
            size_t free_memory, total_memory;
            cudaError_t err = cudaMemGetInfo(&free_memory, &total_memory);
            if (err != cudaSuccess) {
                LFS_ENSURE_CUDA_SUCCESS_MSG(
                    err, "cudaMemGetInfo(arena initialization)",
                    detail::format_cuda_safe("device={}", device));
            }

            // Start with a reasonable initial size
            size_t initial_size = std::min({config_.initial_commit,
                                            free_memory / 2,
                                            size_t(256) << 20});

            if (initial_size < (64 << 20)) {
                throw std::runtime_error("Insufficient GPU memory for arena initialization (need at least 64MB)");
            }

            // Try to allocate with fallback to smaller sizes
            bool allocated = false;
            while (initial_size >= (64 << 20) && !allocated) {
                err = cudaMalloc(&arena.fallback_buffer, initial_size);
                if (err == cudaSuccess) {
                    LOG_TRACE("Arena cudaMalloc: %zu MB", initial_size >> 20);
                    lfs::diagnostics::VramProfiler::instance().recordAllocation(
                        arena.fallback_buffer, initial_size,
                        lfs::diagnostics::VramAllocationMethod::Arena,
                        "rasterizer.arena.initial");
                    arena.capacity = initial_size;
                    arena.committed_size = initial_size;
                    arena.generation = generation_counter_.fetch_add(1, std::memory_order_relaxed);
                    arena.offset.store(0, std::memory_order_release);
                    allocated = true;

                    const cudaError_t memory_status =
                        cudaMemGetInfo(&free_memory, &total_memory);
                    if (memory_status == cudaSuccess) {
                        LOG_INFO("Traditional allocation: device=%d, size=%zu MB, GPU free=%zu MB",
                                 device, initial_size >> 20, free_memory >> 20);
                    } else {
                        ensure_cuda_success(
                            memory_status, "cudaMemGetInfo(arena initialization statistics)",
                            detail::format_cuda_safe("device={}, allocation_bytes={}", device, initial_size),
                            LFS_SOURCE_SITE_CURRENT(), CudaFailureDisposition::LogOnly);
                        LOG_INFO("Traditional allocation: device=%d, size=%zu MB",
                                 device, initial_size >> 20);
                    }
                } else {
                    initial_size /= 2;
                    LOG_DEBUG("Allocation failed, trying %zu MB", initial_size >> 20);
                }
            }

            if (!allocated) {
                LFS_ENSURE_CUDA_SUCCESS_MSG(
                    err, "cudaMalloc(arena initialization)",
                    detail::format_cuda_safe("device={}, attempts exhausted", device));
                throw std::runtime_error("Arena allocation failed without a CUDA status");
            }
        }

        return *arena_ptr;
    }

    bool RasterizerMemoryArena::commit_more_memory(Arena& arena, size_t required_size) {
        // Only for VMM-enabled arenas
        if (arena.d_ptr == 0) {
            return false;
        }

        // Round up to granularity
        size_t commit_size = ((required_size + arena.granularity - 1) /
                              arena.granularity) *
                             arena.granularity;

        // Ensure we don't exceed limits
        if (arena.committed_size + commit_size > config_.max_physical) {
            // Try to commit as much as possible
            commit_size = config_.max_physical - arena.committed_size;
            if (commit_size < arena.granularity) {
                return false; // Can't commit anything
            }
            // Round down to granularity
            commit_size = (commit_size / arena.granularity) * arena.granularity;
        }

        // Check if we'd exceed virtual space
        if (arena.committed_size + commit_size > arena.virtual_size) {
            return false;
        }

        // Check available GPU memory with larger buffer
        size_t free_memory = 0;
        size_t total_memory = 0;
        cudaError_t memory_status = cudaMemGetInfo(&free_memory, &total_memory);
        if (memory_status != cudaSuccess) {
            ensure_cuda_success(
                memory_status, "cudaMemGetInfo(arena VMM commit)",
                detail::format_cuda_safe("device={}, requested_bytes={}", arena.device, commit_size),
                LFS_SOURCE_SITE_CURRENT(), CudaFailureDisposition::LogOnly);
            return false;
        }

        // Need substantial buffer for VMM and other operations
        size_t buffer_needed = std::min(size_t(1) << 30, total_memory / 10); // 1GB or 10% of total

        if (free_memory < commit_size + buffer_needed) {
            // Try cleanup
            empty_cuda_cache();
            memory_status = cudaMemGetInfo(&free_memory, &total_memory);
            if (memory_status != cudaSuccess) {
                ensure_cuda_success(
                    memory_status, "cudaMemGetInfo(arena VMM commit retry)",
                    detail::format_cuda_safe("device={}, requested_bytes={}", arena.device, commit_size),
                    LFS_SOURCE_SITE_CURRENT(), CudaFailureDisposition::LogOnly);
                return false;
            }

            if (free_memory < commit_size + buffer_needed) {
                // Not enough memory - try to allocate as much as we can up to requested size
                size_t available = free_memory > buffer_needed ? free_memory - buffer_needed : 0;
                commit_size = std::min(commit_size, available);
                commit_size = (commit_size / arena.granularity) * arena.granularity;
                if (commit_size < arena.granularity) {
                    return false;
                }
            }
        }

        CUmemAllocationProp prop = {};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = arena.device;

        CUmemGenericAllocationHandle handle;
        CUresult result = cuMemCreate(&handle, commit_size, &prop, 0);

        // If large allocation fails, try splitting into smaller chunks
        if (result != CUDA_SUCCESS && commit_size > arena.granularity * 4) {
            LOG_DEBUG("Large allocation of %zu MB failed, trying smaller chunks", commit_size >> 20);

            // Try 4 smaller chunks (each 1/4 of requested size)
            size_t chunk_size = (commit_size / 4);
            chunk_size = (chunk_size / arena.granularity) * arena.granularity; // Round to granularity

            size_t total_allocated = 0;
            for (int i = 0; i < 4 && total_allocated < commit_size; ++i) {
                CUmemGenericAllocationHandle chunk_handle;
                result = cuMemCreate(&chunk_handle, chunk_size, &prop, 0);
                if (result == CUDA_SUCCESS) {
                    // Map this chunk
                    size_t map_offset = arena.committed_size + total_allocated;
                    map_offset = (map_offset + arena.granularity - 1) & ~(arena.granularity - 1);

                    result = cuMemMap(arena.d_ptr + map_offset, chunk_size, 0, chunk_handle, 0);
                    if (result == CUDA_SUCCESS) {
                        CUmemAccessDesc access_desc = {};
                        access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                        access_desc.location.id = arena.device;
                        access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

                        result = cuMemSetAccess(arena.d_ptr + map_offset, chunk_size, &access_desc, 1);
                        if (result == CUDA_SUCCESS) {
                            {
                                std::lock_guard<std::mutex> lock(arena.chunks_mutex);
                                arena.chunks.push_back({chunk_handle, map_offset, chunk_size, true});
                            }
                            lfs::diagnostics::VramProfiler::instance().recordAllocation(
                                reinterpret_cast<void*>(arena.d_ptr + map_offset),
                                chunk_size,
                                lfs::diagnostics::VramAllocationMethod::Arena,
                                "rasterizer.arena.vmm_chunk");
                            total_allocated += chunk_size;
                            LOG_TRACE("VMM chunk %d: %zu MB", i, chunk_size >> 20);
                        } else {
                            cuMemUnmap(arena.d_ptr + map_offset, chunk_size);
                            cuMemRelease(chunk_handle);
                        }
                    } else {
                        cuMemRelease(chunk_handle);
                    }
                }
            }

            if (total_allocated > 0) {
                arena.committed_size += total_allocated;
                arena.capacity = arena.committed_size;
                arena.realloc_count.fetch_add(1, std::memory_order_relaxed);
                LOG_DEBUG("Committed %zu MB via chunks (total: %zu MB)", total_allocated >> 20, arena.committed_size >> 20);
                return true;
            }

            return false;
        }

        if (result != CUDA_SUCCESS) {
            return false;
        }

        LOG_TRACE("VMM commit: %zu MB (total: %zu MB)", commit_size >> 20, (arena.committed_size + commit_size) >> 20);

        // Map with proper alignment
        size_t map_offset = arena.committed_size;
        map_offset = (map_offset + arena.granularity - 1) & ~(arena.granularity - 1);

        result = cuMemMap(arena.d_ptr + map_offset, commit_size, 0, handle, 0);
        if (result != CUDA_SUCCESS) {
            cuMemRelease(handle);
            return false;
        }

        CUmemAccessDesc access_desc = {};
        access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access_desc.location.id = arena.device;
        access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

        result = cuMemSetAccess(arena.d_ptr + map_offset, commit_size, &access_desc, 1);
        if (result != CUDA_SUCCESS) {
            cuMemUnmap(arena.d_ptr + map_offset, commit_size);
            cuMemRelease(handle);
            return false;
        }

        // Track the chunk
        {
            std::lock_guard<std::mutex> lock(arena.chunks_mutex);
            arena.chunks.push_back({handle, map_offset, commit_size, true});
        }
        lfs::diagnostics::VramProfiler::instance().recordAllocation(
            reinterpret_cast<void*>(arena.d_ptr + map_offset),
            commit_size,
            lfs::diagnostics::VramAllocationMethod::Arena,
            "rasterizer.arena.vmm");

        arena.committed_size = map_offset + commit_size;
        arena.capacity = arena.committed_size;
        arena.realloc_count.fetch_add(1, std::memory_order_relaxed);

        LOG_DEBUG("Committed %zu MB (total: %zu MB)", commit_size >> 20, arena.committed_size >> 20);

        return true;
    }

    void RasterizerMemoryArena::decommit_unused_memory(Arena& arena) {
        // Called with arena_mutex_ held
        // Free ALL chunks BEYOND the current offset (unused memory)
        // This is called during emergency cleanup, so be aggressive

        std::lock_guard<std::mutex> chunk_lock(arena.chunks_mutex);

        if (arena.chunks.empty()) {
            return; // Nothing to decommit
        }

        const size_t current_offset = arena.offset.load(std::memory_order_acquire);
        size_t total_freed = 0;
        size_t chunks_to_remove = 0;

        // Free ALL chunks beyond the current offset (no 1GB limit during emergency)
        // CRITICAL: Only free chunks that are completely beyond the current offset
        for (size_t i = arena.chunks.size(); i > 0; --i) {
            size_t idx = i - 1;
            auto& chunk = arena.chunks[idx];

            // Don't free the first chunk (keep at least initial allocation)
            if (idx == 0) {
                break;
            }

            // Only free chunks that are completely unused (beyond current offset)
            if (chunk.offset >= current_offset) {
                // This chunk is completely unused, safe to free
                if (chunk.is_mapped) {
                    // Unmap the physical memory from virtual address space
                    CUresult result = cuMemUnmap(arena.d_ptr + chunk.offset, chunk.size);
                    if (result == CUDA_SUCCESS) {
                        lfs::diagnostics::VramProfiler::instance().recordDeallocation(
                            reinterpret_cast<void*>(arena.d_ptr + chunk.offset));
                        // Release the physical memory allocation
                        cuMemRelease(chunk.handle);
                        chunk.is_mapped = false;
                        total_freed += chunk.size;
                        chunks_to_remove++;
                    }
                } else {
                    chunks_to_remove++;
                }
            } else {
                // This chunk (or earlier chunks) are still in use, stop here
                break;
            }
        }

        // Remove the freed chunks from the end
        if (chunks_to_remove > 0) {
            size_t new_size = arena.chunks.size() - chunks_to_remove;
            arena.chunks.erase(arena.chunks.begin() + new_size, arena.chunks.end());
            arena.committed_size -= total_freed;
            arena.capacity = arena.committed_size;

            LOG_DEBUG("Decommitted %zu MB (%zu chunks), arena now at %zu MB",
                      total_freed >> 20, chunks_to_remove, arena.committed_size >> 20);
        } else {
            LOG_TRACE("No unused chunks to decommit");
        }

        // Reset peak usage tracking
        arena.peak_usage_period.store(0, std::memory_order_release);
    }

    char* RasterizerMemoryArena::allocate_internal(Arena& arena, size_t size, uint64_t frame_id) {
        LFS_CUDA_BREADCRUMB("arena.allocate");
        size_t aligned_size = align_size(size);

        if (config_.enable_profiling && (frame_id % config_.log_interval == 0)) {
            const size_t size_mb = aligned_size >> 20;
            if (size_mb > 0) {
                LOG_TRACE("Frame %lu: requesting %zu MB (offset=%zu MB)",
                          frame_id, size_mb, arena.offset.load() >> 20);
            }
        }

        // Sanity check
        if (aligned_size > config_.max_physical) {
            LOG_ERROR("Single allocation request %zu MB exceeds max physical size %zu MB",
                      aligned_size >> 20,
                      config_.max_physical >> 20);
            return nullptr;
        }

        // Grows the shared exportable backing in place under its stable virtual
        // address (existing contents preserved), so training never fails when its
        // scratch demand outgrows the render's initial sizing. The base pointer is
        // unchanged; the render re-imports the larger range into Vulkan on its next
        // frame. Caller must hold arena_mutex_.
        const auto try_grow_external = [&](size_t need) -> bool {
            if (!arena.external_backing || !arena.external_grow) {
                return false;
            }
            // external_grow performs the in-place physical grow (stable base
            // address, contents preserved) and returns the new committed size, or
            // 0 on failure. The Vulkan side re-imports the larger range later.
            const size_t new_committed = arena.external_grow(need);
            if (new_committed < need) {
                LOG_ERROR("External rasterizer arena '%s' grow failed (need=%zu MiB, capacity=%zu MiB)",
                          arena.external_label.empty() ? "unnamed" : arena.external_label.c_str(),
                          static_cast<size_t>(need >> 20),
                          static_cast<size_t>(new_committed >> 20));
                return false;
            }
            arena.committed_size = new_committed;
            arena.capacity = new_committed;
            arena.realloc_count.fetch_add(1, std::memory_order_relaxed);
            LOG_INFO("External rasterizer arena '%s' grew in place to %zu MiB (need %zu MiB)",
                     arena.external_label.empty() ? "unnamed" : arena.external_label.c_str(),
                     static_cast<size_t>(new_committed >> 20),
                     static_cast<size_t>(need >> 20));
            return true;
        };

        // Retry loop - keep trying until we succeed or hit max retries
        const int MAX_RETRIES = 5;
        for (int retry = 0; retry < MAX_RETRIES; ++retry) {
            // Try to allocate
            size_t offset = arena.offset.fetch_add(aligned_size, std::memory_order_acq_rel);

            if (offset + aligned_size <= arena.committed_size) {
                // Success!
                char* ptr = nullptr;
                if (arena.d_ptr != 0) {
                    ptr = reinterpret_cast<char*>(arena.d_ptr) + offset;
                } else {
                    ptr = static_cast<char*>(arena.fallback_buffer) + offset;
                }

                // Update peak usage
                size_t current_usage = offset + aligned_size;
                size_t peak = arena.peak_usage.load(std::memory_order_relaxed);
                while (current_usage > peak) {
                    if (arena.peak_usage.compare_exchange_weak(peak, current_usage)) {
                        break;
                    }
                }

                size_t period_peak = arena.peak_usage_period.load(std::memory_order_relaxed);
                while (current_usage > period_peak) {
                    if (arena.peak_usage_period.compare_exchange_weak(period_peak, current_usage)) {
                        break;
                    }
                }

                BufferHandle handle;
                handle.ptr = ptr;
                handle.size = aligned_size;
                handle.generation = arena.generation;
                handle.device = arena.device;
                record_allocation(frame_id, handle);

                return ptr;
            }

            // Allocation failed - revert the offset
            arena.offset.fetch_sub(aligned_size, std::memory_order_acq_rel);

            // Try to grow the arena
            std::lock_guard<std::mutex> lock(arena_mutex_);

            // Re-check current state after getting lock
            size_t current_offset = arena.offset.load(std::memory_order_acquire);
            size_t total_needed = current_offset + aligned_size;

            // Check if someone else already grew it
            if (total_needed <= arena.committed_size) {
                continue; // Retry allocation
            }

            if (arena.external_backing) {
                if (try_grow_external(total_needed)) {
                    continue; // retry allocation against the grown capacity
                }
                LOG_ERROR("External rasterizer arena '%s' exhausted and could not grow: need=%zu MiB capacity=%zu MiB request=%zu MiB",
                          arena.external_label.empty() ? "unnamed" : arena.external_label.c_str(),
                          static_cast<size_t>(total_needed >> 20),
                          static_cast<size_t>(arena.committed_size >> 20),
                          static_cast<size_t>(aligned_size >> 20));
                return nullptr;
            }

            // We need to grow - calculate how much
            size_t growth_needed = total_needed - arena.committed_size;

            // Progressive fallback strategy
            size_t growth_amount;
            if (retry < 3) {
                growth_amount = growth_needed * 2;
                if (retry > 0) {
                    LOG_DEBUG("Retry %d: growth %zu MB (2x needed)", retry, growth_amount >> 20);
                }
            } else {
                growth_amount = (growth_needed * 3) / 2;
                LOG_DEBUG("Retry %d: minimal growth %zu MB", retry, growth_amount >> 20);
            }

            // Cap at max physical
            size_t new_committed = std::min(arena.committed_size + growth_amount, config_.max_physical);
            growth_amount = new_committed - arena.committed_size;

            if (growth_amount == 0) {
                LOG_ERROR("Capacity limit reached: max=%zu MB, requested=%zu MB",
                          config_.max_physical >> 20, aligned_size >> 20);
                return nullptr;
            }

            // Try to grow
            bool success = false;
            if (arena.d_ptr != 0) {
                // VMM path
                success = commit_more_memory(arena, growth_amount);
                // Synchronize after committing new memory to ensure no GPU kernels
                // are still accessing old memory regions before we allow new allocations
                if (success) {
                    const cudaError_t sync_status = cudaDeviceSynchronize();
                    if (sync_status != cudaSuccess) {
                        ensure_cuda_success(
                            sync_status, "cudaDeviceSynchronize(arena VMM growth)",
                            detail::format_cuda_safe("growth_bytes={}, retry={}", growth_amount, retry),
                            LFS_SOURCE_SITE_CURRENT(), CudaFailureDisposition::LogOnly);
                    }
                }
            } else {
                // Traditional path
                // grow_arena owns the fallback growth multiplier. Passing the
                // already-expanded VMM target here compounded it a second time.
                success = grow_arena(arena, total_needed);
            }

            if (!success) {
                if (retry < MAX_RETRIES - 1) {
                    LOG_DEBUG("Growth failed, cleanup and retry %d/%d", retry + 1, MAX_RETRIES);
                    empty_cuda_cache();
                    const cudaError_t sync_status = cudaDeviceSynchronize();
                    if (sync_status != cudaSuccess) {
                        ensure_cuda_success(
                            sync_status, "cudaDeviceSynchronize(arena growth retry)",
                            detail::format_cuda_safe("retry={}, requested_bytes={}", retry + 1, aligned_size),
                            LFS_SOURCE_SITE_CURRENT(), CudaFailureDisposition::LogOnly);
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    continue;
                } else {
                    LOG_ERROR("Out of memory after %d attempts: requested=%zu MB, usage=%zu MB, committed=%zu MB, max=%zu MB",
                              MAX_RETRIES, size >> 20, current_offset >> 20, arena.committed_size >> 20, config_.max_physical >> 20);
                    return nullptr;
                }
            }

            // Growth succeeded, retry allocation
        }

        LOG_ERROR("Allocation loop exhausted");
        return nullptr;
    }

    bool RasterizerMemoryArena::grow_arena(Arena& arena, size_t required_size) {
        LFS_CUDA_BREADCRUMB("arena.grow");
        // Called with arena_mutex_ held (fallback for non-VMM systems)
        const size_t old_capacity = arena.capacity;
        size_t new_capacity = std::max(required_size * 2, static_cast<size_t>(arena.capacity * 1.5f));

        // Round up to 128MB boundary
        constexpr size_t ALIGNMENT = 128 << 20;
        new_capacity = ((new_capacity + ALIGNMENT - 1) / ALIGNMENT) * ALIGNMENT;
        new_capacity = std::min(new_capacity, config_.max_physical);

        if (new_capacity <= arena.capacity) {
            LOG_ERROR("Cannot grow: capacity=%zu MB, required=%zu MB, max=%zu GB",
                      arena.capacity >> 20, required_size >> 20, config_.max_physical >> 30);
            return false;
        }

        size_t free_memory, total_memory;
        cudaError_t err = cudaMemGetInfo(&free_memory, &total_memory);
        if (err != cudaSuccess) {
            ensure_cuda_success(err, "cudaMemGetInfo(arena growth)",
                                detail::format_cuda_safe("required_bytes={}", required_size),
                                LFS_SOURCE_SITE_CURRENT(),
                                CudaFailureDisposition::LogOnly);
            return false;
        }

        const size_t additional_needed = new_capacity - arena.capacity;
        constexpr size_t MIN_FREE_BUFFER = 200 << 20;

        LOG_DEBUG("Growing arena: %zu MB -> %zu MB (need %zu MB, free %zu MB)",
                  old_capacity >> 20, new_capacity >> 20, additional_needed >> 20, free_memory >> 20);

        if (free_memory < additional_needed + MIN_FREE_BUFFER) {
            LOG_DEBUG("Low memory, attempting cleanup");
            empty_cuda_cache();
            err = cudaMemGetInfo(&free_memory, &total_memory);
            if (err != cudaSuccess) {
                ensure_cuda_success(
                    err, "cudaMemGetInfo(arena growth retry)",
                    detail::format_cuda_safe("required_bytes={}", required_size),
                    LFS_SOURCE_SITE_CURRENT(), CudaFailureDisposition::LogOnly);
                return false;
            }

            if (free_memory < additional_needed + MIN_FREE_BUFFER) {
                LOG_WARN("Insufficient memory for growth: need %zu MB, free %zu MB",
                         additional_needed >> 20, free_memory >> 20);
                return false;
            }
        }

        void* new_buffer = nullptr;
        err = cudaMalloc(&new_buffer, new_capacity);
        if (err != cudaSuccess) {
            ensure_cuda_success(err, "cudaMalloc(arena growth)",
                                detail::format_cuda_safe("old_capacity={}, new_capacity={}, required={}",
                                                         old_capacity, new_capacity, required_size),
                                LFS_SOURCE_SITE_CURRENT(),
                                CudaFailureDisposition::LogOnly);
            return false;
        }
        lfs::diagnostics::VramProfiler::instance().recordAllocation(
            new_buffer, new_capacity,
            lfs::diagnostics::VramAllocationMethod::Arena,
            "rasterizer.arena.grown");

        LOG_TRACE("Arena realloc: %zu MB", new_capacity >> 20);

        const size_t copy_size = arena.offset.load(std::memory_order_acquire);
        if (copy_size > 0 && arena.fallback_buffer) {
            err = cudaMemcpy(new_buffer, arena.fallback_buffer, copy_size, cudaMemcpyDeviceToDevice);
            if (err != cudaSuccess) {
                ensure_cuda_success(err, "cudaMemcpy(arena growth)",
                                    detail::format_cuda_safe("copy_bytes={}, old_capacity={}, new_capacity={}",
                                                             copy_size, old_capacity, new_capacity),
                                    LFS_SOURCE_SITE_CURRENT(),
                                    CudaFailureDisposition::LogOnly);
                lfs::diagnostics::VramProfiler::instance().recordDeallocation(new_buffer);
                const cudaError_t free_status = cudaFree(new_buffer);
                if (free_status != cudaSuccess) {
                    ensure_cuda_success(
                        free_status, "cudaFree(failed arena growth buffer)",
                        detail::format_cuda_safe("ptr={}, bytes={}", new_buffer, new_capacity),
                        LFS_SOURCE_SITE_CURRENT(), CudaFailureDisposition::LogOnlyNoLatch);
                }
                return false;
            }
        }

        if (arena.fallback_buffer) {
            lfs::diagnostics::VramProfiler::instance().recordDeallocation(arena.fallback_buffer);
            const cudaError_t free_status = cudaFree(arena.fallback_buffer);
            if (free_status != cudaSuccess) {
                ensure_cuda_success(
                    free_status, "cudaFree(previous arena backing)",
                    detail::format_cuda_safe("ptr={}, bytes={}", arena.fallback_buffer, old_capacity),
                    LFS_SOURCE_SITE_CURRENT(), CudaFailureDisposition::LogOnlyNoLatch);
            }
        }
        arena.fallback_buffer = new_buffer;
        arena.capacity = new_capacity;
        arena.committed_size = new_capacity;
        arena.generation = generation_counter_.fetch_add(1, std::memory_order_relaxed);
        arena.realloc_count.fetch_add(1, std::memory_order_relaxed);

        const cudaError_t memory_status = cudaMemGetInfo(&free_memory, &total_memory);
        if (memory_status == cudaSuccess) {
            LOG_DEBUG("Growth successful: new capacity=%zu MB, GPU free=%zu MB",
                      new_capacity >> 20, free_memory >> 20);
        } else {
            ensure_cuda_success(
                memory_status, "cudaMemGetInfo(arena post-growth statistics)",
                detail::format_cuda_safe("new_capacity={}", new_capacity), LFS_SOURCE_SITE_CURRENT(),
                CudaFailureDisposition::LogOnly);
            LOG_DEBUG("Growth successful: new capacity=%zu MB", new_capacity >> 20);
        }

        return true;
    }

    size_t RasterizerMemoryArena::align_size(size_t size) const {
        return (size + config_.alignment - 1) & ~(config_.alignment - 1);
    }

    void RasterizerMemoryArena::record_allocation(uint64_t frame_id, const BufferHandle& handle) {
        std::lock_guard<std::mutex> lock(frame_mutex_);

        auto it = frame_contexts_.find(frame_id);
        if (it != frame_contexts_.end()) {
            it->second.buffers.push_back(handle);
            it->second.total_allocated += handle.size;
        }
    }

    RasterizerMemoryArena::Statistics RasterizerMemoryArena::get_statistics() const {
        Statistics stats;

        std::lock_guard<std::mutex> lock(arena_mutex_);

        for (const auto& entry : device_arenas_) {
            const auto& arena_ptr = entry.second;
            if (arena_ptr) {
                stats.current_usage += arena_ptr->offset.load(std::memory_order_relaxed);
                stats.peak_usage = std::max(stats.peak_usage,
                                            arena_ptr->peak_usage.load(std::memory_order_relaxed));
                stats.capacity += arena_ptr->committed_size;
                stats.reallocation_count += arena_ptr->realloc_count.load(std::memory_order_relaxed);
            }
        }

        stats.frame_count = total_frames_processed_.load(std::memory_order_relaxed);
        stats.utilization_ratio = stats.capacity > 0 ? static_cast<float>(stats.current_usage) / static_cast<float>(stats.capacity) : 0.0f;

        return stats;
    }

    RasterizerMemoryArena::MemoryInfo RasterizerMemoryArena::get_memory_info() const {
        MemoryInfo info;

        int device;
        cudaError_t err = cudaGetDevice(&device);
        if (err == cudaSuccess) {
            std::lock_guard<std::mutex> lock(arena_mutex_);
            auto it = device_arenas_.find(device);
            if (it != device_arenas_.end() && it->second) {
                info.arena_capacity = it->second->committed_size;
                info.current_usage = it->second->offset.load(std::memory_order_relaxed);
                info.peak_usage = it->second->peak_usage.load(std::memory_order_relaxed);
                info.num_reallocations = it->second->realloc_count.load(std::memory_order_relaxed);
                info.utilization_percent = info.arena_capacity > 0 ? (100.0f * info.peak_usage / info.arena_capacity) : 0.0f;
            }
        } else {
            ensure_cuda_success(err, "cudaGetDevice(arena memory info)", {},
                                LFS_SOURCE_SITE_CURRENT(),
                                CudaFailureDisposition::LogOnly);
        }

        const cudaError_t memory_status = cudaMemGetInfo(&info.gpu_free, &info.gpu_total);
        if (memory_status != cudaSuccess) {
            ensure_cuda_success(memory_status, "cudaMemGetInfo(arena memory info)", {},
                                LFS_SOURCE_SITE_CURRENT(),
                                CudaFailureDisposition::LogOnly);
        }
        return info;
    }

    void RasterizerMemoryArena::dump_statistics() const {
        const auto stats = get_statistics();
        const auto runtime = std::chrono::steady_clock::now() - creation_time_;
        const auto seconds = std::chrono::duration_cast<std::chrono::seconds>(runtime).count();
        const float utilization = stats.capacity > 0 ? (100.0f * stats.peak_usage / stats.capacity) : 0.0f;

        LOG_INFO("Arena stats: committed=%zu MB, peak=%zu MB (%.1f%%), frames=%zu, reallocs=%zu, runtime=%lds",
                 stats.capacity >> 20, stats.peak_usage >> 20, utilization,
                 stats.frame_count, stats.reallocation_count, seconds);
    }

    bool RasterizerMemoryArena::is_under_memory_pressure() const {
        return get_memory_pressure() > 0.8f;
    }

    float RasterizerMemoryArena::get_memory_pressure() const {
        size_t free_memory, total_memory;
        cudaError_t err = cudaMemGetInfo(&free_memory, &total_memory);
        if (err != cudaSuccess) {
            ensure_cuda_success(err, "cudaMemGetInfo(arena memory pressure)",
                                "fallback=maximum pressure", LFS_SOURCE_SITE_CURRENT(),
                                CudaFailureDisposition::LogOnly);
            return 1.0f;
        }
        return 1.0f - (static_cast<float>(free_memory) / static_cast<float>(total_memory));
    }

    // Global singleton implementation
    GlobalArenaManager& GlobalArenaManager::instance() {
        static GlobalArenaManager instance;
        return instance;
    }

    RasterizerMemoryArena& GlobalArenaManager::get_arena() {
        std::lock_guard<std::mutex> lock(init_mutex_);

        if (!arena_) {
            // Auto-detect GPU VRAM size
            size_t free_mem = 0;
            size_t total_mem = 0;
            LFS_CUDA_CHECK_MSG(cudaMemGetInfo(&free_mem, &total_mem),
                               "initializing the global rasterizer arena");

            // Create with VMM-optimized settings
            RasterizerMemoryArena::Config config;
            config.virtual_size = 32ULL << 30; // 32GB virtual (costs nothing!)
            config.initial_commit = 128 << 20; // 128MB initial physical; grows lazily
            config.max_physical = total_mem;   // Auto-detected from GPU
            config.granularity = 2 << 20;      // 2MB chunks
            config.alignment = 256;
            config.enable_profiling = false; // Disable profiling for production
            config.log_interval = 1000;      // Log every 1000 frames

            arena_ = std::make_unique<RasterizerMemoryArena>(config);
        }
        return *arena_;
    }

    RasterizerMemoryArena* GlobalArenaManager::try_get_arena() {
        std::lock_guard<std::mutex> lock(init_mutex_);
        return arena_.get();
    }

    bool GlobalArenaManager::install_external_backing(RasterizerMemoryArena::ExternalBacking backing) {
        return get_arena().install_external_backing(std::move(backing));
    }

    bool GlobalArenaManager::try_install_external_backing(RasterizerMemoryArena::ExternalBacking backing) {
        return get_arena().try_install_external_backing(std::move(backing));
    }

    bool GlobalArenaManager::grow_external_backing(const void* device_ptr, size_t new_size,
                                                   const std::function<bool(size_t)>& commit) {
        return get_arena().grow_external_backing(device_ptr, new_size, commit);
    }

    void GlobalArenaManager::clear_external_backing(const void* device_ptr) {
        std::lock_guard<std::mutex> lock(init_mutex_);
        if (arena_) {
            arena_->clear_external_backing(device_ptr);
        }
    }

    void GlobalArenaManager::reset() {
        std::lock_guard<std::mutex> lock(init_mutex_);
        arena_.reset();
    }

    void GlobalArenaManager::reconfigure_for_testing(RasterizerMemoryArena::Config config) {
        std::lock_guard<std::mutex> lock(init_mutex_);
        arena_ = std::make_unique<RasterizerMemoryArena>(std::move(config));
    }

    bool RasterizerMemoryArena::is_rendering_active() const {
        std::lock_guard<std::mutex> sync_lock(sync_mutex_);
        return pending_render_frames_ > 0;
    }

    void RasterizerMemoryArena::set_rendering_active(bool active) {
        bool should_notify = false;
        {
            std::lock_guard<std::mutex> sync_lock(sync_mutex_);
            if (active) {
                ++pending_render_frames_;
            } else if (pending_render_frames_ > 0) {
                --pending_render_frames_;
                should_notify = pending_render_frames_ == 0;
            } else {
                LOG_WARN("RasterizerMemoryArena::set_rendering_active(false) called with no pending render frames");
            }
        }
        if (should_notify) {
            sync_cv_.notify_all();
        }
    }

} // namespace lfs::core

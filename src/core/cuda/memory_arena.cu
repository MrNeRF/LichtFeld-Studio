/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/logger.hpp"
#include "memory_arena.hpp"
#include <algorithm>
#include <cstring>
#include <cuda_runtime.h>
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
        int device;
        cudaGetDevice(&device);
        if (!is_vmm_supported(device)) {
            LOG_WARN("VMM not supported, falling back to smaller allocation");
            config_.initial_commit = 128 << 20;
            config_.max_physical = 4ULL << 30;
        }

        LOG_DEBUG("Arena config: virtual=%zu GB, initial=%zu MB, max=%zu GB, granularity=%zu MB",
                  config_.virtual_size >> 30, config_.initial_commit >> 20,
                  config_.max_physical >> 30, config_.granularity >> 20);
    }

    RasterizerMemoryArena::~RasterizerMemoryArena() {
        dump_statistics();

        // Clean up all arenas
        std::lock_guard<std::mutex> lock(arena_mutex_);
        for (auto& [device, arena_ptr] : device_arenas_) {
            if (!arena_ptr)
                continue;
            auto& arena = *arena_ptr;

            // Unmap and release all physical memory
            std::lock_guard<std::mutex> chunk_lock(arena.chunks_mutex);
            for (auto& chunk : arena.chunks) {
                if (chunk.is_mapped) {
                    cuMemUnmap(arena.d_ptr + chunk.offset, chunk.size);
                    cuMemRelease(chunk.handle);
                }
            }

            // Free virtual address space
            if (arena.d_ptr) {
                cuMemAddressFree(arena.d_ptr, arena.virtual_size);
            }

            // Free fallback buffer if exists
            if (arena.fallback_buffer) {
                cudaFree(arena.fallback_buffer);
            }

            // Destroy frame synchronization event
            if (arena.frame_event) {
                cudaEventDestroy(arena.frame_event);
            }
        }

        device_arenas_.clear();
        frame_contexts_.clear();
    }

    RasterizerMemoryArena::RasterizerMemoryArena(RasterizerMemoryArena&& other) noexcept
        : device_arenas_(std::move(other.device_arenas_)),
          frame_contexts_(std::move(other.frame_contexts_)),
          config_(other.config_),
          frame_counter_(other.frame_counter_.load()),
          generation_counter_(other.generation_counter_.load()),
          creation_time_(other.creation_time_),
          total_frames_processed_(other.total_frames_processed_.load()) {
    }

    RasterizerMemoryArena& RasterizerMemoryArena::operator=(RasterizerMemoryArena&& other) noexcept {
        if (this != &other) {
            std::lock_guard<std::mutex> lock1(arena_mutex_);
            std::lock_guard<std::mutex> lock2(other.arena_mutex_);

            device_arenas_ = std::move(other.device_arenas_);
            frame_contexts_ = std::move(other.frame_contexts_);
            config_ = other.config_;
            frame_counter_ = other.frame_counter_.load();
            generation_counter_ = other.generation_counter_.load();
            creation_time_ = other.creation_time_;
            total_frames_processed_ = other.total_frames_processed_.load();
        }
        return *this;
    }

    bool RasterizerMemoryArena::is_vmm_supported(int device) const {
        // Check compute capability
        int major = 0, minor = 0;
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);

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

    uint64_t RasterizerMemoryArena::begin_frame(bool from_rendering) {
        // Training waits for rendering; rendering skips this
        if (!from_rendering) {
            while (rendering_active_.load(std::memory_order_acquire)) {
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
            active_training_frames_.fetch_add(1, std::memory_order_release);
        }

        uint64_t frame_id = frame_counter_.fetch_add(1, std::memory_order_relaxed);

        // CRITICAL FIX: Reset arena offset at the beginning of each frame!
        int device;
        cudaError_t err = cudaGetDevice(&device);
        if (err == cudaSuccess) {
            std::lock_guard<std::mutex> lock(arena_mutex_);
            auto it = device_arenas_.find(device);
            if (it != device_arenas_.end() && it->second) {
                // Wait for previous frame's GPU work to complete before resetting
                // Use event-based sync (much faster than cudaDeviceSynchronize!)
                if (it->second->frame_event && it->second->event_recorded) {
                    cudaEventSynchronize(it->second->frame_event);
                    it->second->event_recorded = false;
                }

                // Reset the offset to reuse the buffer from the beginning
                size_t old_offset = it->second->offset.load(std::memory_order_acquire);
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

    void RasterizerMemoryArena::end_frame(uint64_t frame_id, bool from_rendering) {
        if (!from_rendering) {
            active_training_frames_.fetch_sub(1, std::memory_order_release);
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

                // Record event for synchronization in next begin_frame()
                // This marks when GPU work for this frame will be complete
                if (it->second->frame_event) {
                    cudaEventRecord(it->second->frame_event);
                    it->second->event_recorded = true;
                }
            }
        }

        std::lock_guard<std::mutex> lock(frame_mutex_);

        auto it = frame_contexts_.find(frame_id);
        if (it != frame_contexts_.end()) {
            it->second.is_active = false;
        }

        // Cleanup old frames - keep only last 3
        cleanup_frames(3);
    }

    void RasterizerMemoryArena::log_memory_status(uint64_t frame_id, bool force) {
        // Called with arena_mutex_ already held
        int device;
        cudaError_t err = cudaGetDevice(&device);
        if (err != cudaSuccess)
            return;

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
            for (const auto& [fid, ctx] : frame_contexts_) {
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
            cudaError_t err = cudaGetDevice(&device);
            if (err != cudaSuccess) {
                throw std::runtime_error("Failed to get CUDA device: " +
                                         std::string(cudaGetErrorString(err)));
            }

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

        for (const auto& [id, ctx] : frame_contexts_) {
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
        cudaDeviceSynchronize();

        // Allocate and immediately free a small buffer to trigger cleanup
        void* dummy;
        cudaMalloc(&dummy, 1);
        cudaFree(dummy);
    }

    void RasterizerMemoryArena::emergency_cleanup() {
        std::lock_guard<std::mutex> lock1(arena_mutex_);
        std::lock_guard<std::mutex> lock2(frame_mutex_);

        LOG_WARN("Emergency cleanup started");

        // Clear all inactive frames
        auto it = frame_contexts_.begin();
        while (it != frame_contexts_.end()) {
            if (!it->second.is_active) {
                it = frame_contexts_.erase(it);
            } else {
                ++it;
            }
        }

        // Decommit unused memory BEFORE resetting offsets
        for (auto& [device, arena] : device_arenas_) {
            if (arena) {
                decommit_unused_memory(*arena);
                arena->offset.store(0, std::memory_order_release);
                LOG_DEBUG("Reset arena on device %d (committed: %zu MB)", device, arena->committed_size >> 20);
            }
        }

        empty_cuda_cache();

        for (const auto& [device, arena] : device_arenas_) {
            cudaSetDevice(device);
            cudaDeviceSynchronize();
        }

        LOG_WARN("Emergency cleanup completed");
    }

    RasterizerMemoryArena::Arena& RasterizerMemoryArena::get_or_create_arena(int device) {
        std::lock_guard<std::mutex> lock(arena_mutex_);

        auto& arena_ptr = device_arenas_[device];
        if (!arena_ptr) {
            arena_ptr = std::make_unique<Arena>();
            arena_ptr->device = device;
            arena_ptr->last_log_time = std::chrono::steady_clock::now();

            // Set device before allocating
            cudaSetDevice(device);

            if (is_vmm_supported(device)) {
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

                        // Create frame synchronization event (replaces cudaDeviceSynchronize)
                        cudaEventCreateWithFlags(&arena.frame_event, cudaEventDisableTiming);
                        arena.event_recorded = false;

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
                throw std::runtime_error("Failed to query GPU memory: " +
                                         std::string(cudaGetErrorString(err)));
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
                    arena.capacity = initial_size;
                    arena.committed_size = initial_size;
                    arena.generation = generation_counter_.fetch_add(1, std::memory_order_relaxed);
                    arena.offset.store(0, std::memory_order_release);
                    allocated = true;

                    // Create frame synchronization event (replaces cudaDeviceSynchronize)
                    cudaEventCreateWithFlags(&arena.frame_event, cudaEventDisableTiming);
                    arena.event_recorded = false;

                    cudaMemGetInfo(&free_memory, &total_memory);
                    LOG_INFO("Traditional allocation: device=%d, size=%zu MB, GPU free=%zu MB",
                             device, initial_size >> 20, free_memory >> 20);
                } else {
                    initial_size /= 2;
                    LOG_DEBUG("Allocation failed, trying %zu MB", initial_size >> 20);
                }
            }

            if (!allocated) {
                throw std::runtime_error("Failed to allocate arena after multiple attempts");
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
        size_t free_memory, total_memory;
        cudaMemGetInfo(&free_memory, &total_memory);

        // Need substantial buffer for VMM and other operations
        size_t buffer_needed = std::min(size_t(1) << 30, total_memory / 10); // 1GB or 10% of total

        if (free_memory < commit_size + buffer_needed) {
            // Try cleanup
            empty_cuda_cache();
            cudaMemGetInfo(&free_memory, &total_memory);

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

            LOG_DEBUG("Decommitted %zu MB (%zu chunks), arena now at %zu MB",
                      total_freed >> 20, chunks_to_remove, arena.committed_size >> 20);
        } else {
            LOG_TRACE("No unused chunks to decommit");
        }

        // Reset peak usage tracking
        arena.peak_usage_period.store(0, std::memory_order_release);
    }

    char* RasterizerMemoryArena::allocate_internal(Arena& arena, size_t size, uint64_t frame_id) {
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
            throw std::runtime_error("Single allocation request " + std::to_string(aligned_size >> 20) +
                                     " MB exceeds max physical size " + std::to_string(config_.max_physical >> 30) + " GB");
        }

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
                    cudaDeviceSynchronize();
                }
            } else {
                // Traditional path
                success = grow_arena(arena, new_committed);
            }

            if (!success) {
                if (retry < MAX_RETRIES - 1) {
                    LOG_DEBUG("Growth failed, cleanup and retry %d/%d", retry + 1, MAX_RETRIES);
                    empty_cuda_cache();
                    cudaDeviceSynchronize();
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
            return false;
        }

        const size_t additional_needed = new_capacity - arena.capacity;
        constexpr size_t MIN_FREE_BUFFER = 200 << 20;

        LOG_DEBUG("Growing arena: %zu MB -> %zu MB (need %zu MB, free %zu MB)",
                  old_capacity >> 20, new_capacity >> 20, additional_needed >> 20, free_memory >> 20);

        if (free_memory < additional_needed + MIN_FREE_BUFFER) {
            LOG_DEBUG("Low memory, attempting cleanup");
            empty_cuda_cache();
            cudaMemGetInfo(&free_memory, &total_memory);

            if (free_memory < additional_needed + MIN_FREE_BUFFER) {
                LOG_WARN("Insufficient memory for growth: need %zu MB, free %zu MB",
                         additional_needed >> 20, free_memory >> 20);
                return false;
            }
        }

        void* new_buffer = nullptr;
        err = cudaMalloc(&new_buffer, new_capacity);
        if (err != cudaSuccess) {
            LOG_WARN("Growth allocation failed: %s", cudaGetErrorString(err));
            return false;
        }

        LOG_TRACE("Arena realloc: %zu MB", new_capacity >> 20);

        const size_t copy_size = arena.offset.load(std::memory_order_acquire);
        if (copy_size > 0 && arena.fallback_buffer) {
            err = cudaMemcpy(new_buffer, arena.fallback_buffer, copy_size, cudaMemcpyDeviceToDevice);
            if (err != cudaSuccess) {
                cudaFree(new_buffer);
                LOG_ERROR("Copy failed during growth: %s", cudaGetErrorString(err));
                return false;
            }
        }

        if (arena.fallback_buffer) {
            cudaFree(arena.fallback_buffer);
        }
        arena.fallback_buffer = new_buffer;
        arena.capacity = new_capacity;
        arena.committed_size = new_capacity;
        arena.generation = generation_counter_.fetch_add(1, std::memory_order_relaxed);
        arena.realloc_count.fetch_add(1, std::memory_order_relaxed);

        cudaMemGetInfo(&free_memory, &total_memory);
        LOG_DEBUG("Growth successful: new capacity=%zu MB, GPU free=%zu MB", new_capacity >> 20, free_memory >> 20);

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

        for (const auto& [device, arena_ptr] : device_arenas_) {
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
        }

        cudaMemGetInfo(&info.gpu_free, &info.gpu_total);
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
            size_t free_mem, total_mem;
            cudaMemGetInfo(&free_mem, &total_mem);

            // Create with VMM-optimized settings
            RasterizerMemoryArena::Config config;
            config.virtual_size = 32ULL << 30; // 32GB virtual (costs nothing!)
            config.initial_commit = 512 << 20; // 512MB initial physical (was 256MB)
            config.max_physical = total_mem;   // Auto-detected from GPU
            config.granularity = 2 << 20;      // 2MB chunks
            config.alignment = 256;
            config.enable_profiling = false; // Disable profiling for production
            config.log_interval = 1000;      // Log every 1000 frames

            arena_ = std::make_unique<RasterizerMemoryArena>(config);
        }
        return *arena_;
    }

    void GlobalArenaManager::reset() {
        std::lock_guard<std::mutex> lock(init_mutex_);
        arena_.reset();
    }

    void RasterizerMemoryArena::wait_for_training() const {
        while (active_training_frames_.load(std::memory_order_acquire) > 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }

} // namespace lfs::core
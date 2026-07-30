/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/pinned_memory_allocator.hpp"

#include "core/cuda_error.hpp"
#include "core/environment.hpp"
#include "core/logger.hpp"
#include "diagnostics/vram_profiler.hpp"
#include "internal/cuda_event_pool.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <limits>

namespace lfs::core {
    namespace {

        constexpr size_t MIB = 1024ULL * 1024ULL;
        constexpr size_t DEFAULT_CACHE_LIMIT = 1024ULL * MIB;

        bool is_cuda_shutdown(const cudaError_t status) {
            return status == cudaErrorCudartUnloading || is_cuda_unavailable_error(status);
        }

        size_t configured_cache_limit() {
            const auto value = environment::value("LFS_PINNED_CACHE_LIMIT_MB");
            if (!value) {
                return DEFAULT_CACHE_LIMIT;
            }

            const auto megabytes = environment::unsigned_integer<unsigned long long>(*value);
            if (!megabytes) {
                LOG_WARN("Ignoring invalid LFS_PINNED_CACHE_LIMIT_MB='{}'", *value);
                return DEFAULT_CACHE_LIMIT;
            }
            if (*megabytes > std::numeric_limits<size_t>::max() / MIB) {
                LOG_WARN("LFS_PINNED_CACHE_LIMIT_MB='{}' exceeds this platform's address range", *value);
                return std::numeric_limits<size_t>::max();
            }
            return static_cast<size_t>(*megabytes) * MIB;
        }

    } // namespace

    PinnedMemoryAllocator::Block::~Block() {
        // Explicit allocator shutdown returns events through the pool. This is
        // only the static-destruction fallback, where the later-constructed
        // event-pool singleton may already be gone.
        for (const cudaEvent_t event : ready_events) {
            const cudaError_t status = cudaEventDestroy(event);
            if (status != cudaSuccess) {
                ensure_cuda_success(
                    status, "cudaEventDestroy(pinned static teardown)", {},
                    LFS_SOURCE_SITE_CURRENT(), CudaFailureDisposition::LogOnlyNoLatch);
                cudaGetLastError();
            }
        }
    }

    PinnedMemoryAllocator::Block::Block(Block&& other) noexcept
        : ptr(other.ptr),
          size(other.size),
          backend(other.backend),
          last_used(other.last_used),
          ready_events(std::move(other.ready_events)) {
        other.ptr = nullptr;
        other.size = 0;
        other.last_used = 0;
        other.ready_events.clear();
    }

    PinnedMemoryAllocator::Block& PinnedMemoryAllocator::Block::operator=(Block&& other) noexcept {
        if (this != &other) {
            release_events();
            ptr = other.ptr;
            size = other.size;
            backend = other.backend;
            last_used = other.last_used;
            ready_events = std::move(other.ready_events);
            other.ptr = nullptr;
            other.size = 0;
            other.last_used = 0;
            other.ready_events.clear();
        }
        return *this;
    }

    bool PinnedMemoryAllocator::Block::all_uses_complete() const {
        for (const cudaEvent_t event : ready_events) {
            const cudaError_t status = cudaEventQuery(event);
            if (status == cudaErrorNotReady) {
                return false;
            }
            if (is_cuda_shutdown(status)) {
                cudaGetLastError();
                continue;
            }
            if (status != cudaSuccess) {
                ensure_cuda_success(status, "cudaEventQuery(pinned block)", {},
                                    LFS_SOURCE_SITE_CURRENT(),
                                    CudaFailureDisposition::LogOnly);
                cudaGetLastError();
                return false;
            }
        }
        return true;
    }

    void PinnedMemoryAllocator::Block::release_events() {
        for (const cudaEvent_t event : ready_events) {
            CudaEventPool::instance().release(event);
        }
        ready_events.clear();
    }

    PinnedMemoryAllocator& PinnedMemoryAllocator::instance() {
        static PinnedMemoryAllocator instance;
        return instance;
    }

    PinnedMemoryAllocator::PinnedMemoryAllocator()
        : cache_limit_bytes_(configured_cache_limit()) {}

    PinnedMemoryAllocator::~PinnedMemoryAllocator() {
        // Application and test shutdown must call shutdown() while CUDA and
        // diagnostics are alive. At static destruction, the OS/runtime owns
        // final reclamation; re-entering later-constructed singletons is unsafe.
        shutdown_.store(true, std::memory_order_release);
    }

    void PinnedMemoryAllocator::shutdown() {
        bool expected = false;
        if (!shutdown_.compare_exchange_strong(expected, true)) {
            return;
        }

        const Stats before = get_stats();
        LOG_INFO("Pinned memory stats at shutdown: active={:.2f} MiB, cached={:.2f} MiB, "
                 "peak={:.2f} MiB, cuda_host={}/{}, fallback={}/{}, evicted={:.2f} MiB/{} blocks",
                 before.allocated_bytes / static_cast<double>(MIB),
                 before.cached_bytes / static_cast<double>(MIB),
                 before.peak_total_bytes / static_cast<double>(MIB),
                 before.cuda_host_allocs,
                 before.cuda_host_frees,
                 before.malloc_fallback_allocs,
                 before.malloc_fallback_frees,
                 before.evicted_bytes / static_cast<double>(MIB),
                 before.evicted_blocks);
        empty_cache_impl(true);
    }

    size_t PinnedMemoryAllocator::round_size(const size_t bytes) {
        if (bytes < 4096) {
            return bytes;
        }
        if (bytes < MIB) {
            return ((bytes + 511) / 512) * 512;
        }

        size_t rounded = MIB;
        while (rounded < bytes) {
            if (rounded > std::numeric_limits<size_t>::max() / 2) {
                return bytes;
            }
            rounded *= 2;
        }
        return rounded;
    }

    void PinnedMemoryAllocator::update_peaks_locked() {
        stats_.peak_allocated_bytes = std::max(stats_.peak_allocated_bytes, stats_.allocated_bytes);
        stats_.peak_cached_bytes = std::max(stats_.peak_cached_bytes, stats_.cached_bytes);
        stats_.peak_total_bytes =
            std::max(stats_.peak_total_bytes, stats_.allocated_bytes + stats_.cached_bytes);
    }

    void PinnedMemoryAllocator::publish_stats_locked() const {
        lfs::diagnostics::VramProfiler::instance().setPinnedHostMemory(
            stats_.allocated_bytes, stats_.cached_bytes, stats_.peak_total_bytes);
    }

    void* PinnedMemoryAllocator::allocate(const size_t bytes) {
        if (bytes == 0) {
            return nullptr;
        }

        const bool use_pinned =
            enabled_.load(std::memory_order_acquire) &&
            !force_fallback_for_testing_.load(std::memory_order_acquire) &&
            !cuda_is_unavailable();
        const size_t allocation_size = use_pinned ? round_size(bytes) : bytes;

        std::unique_lock lock(mutex_);
        if (shutdown_.load(std::memory_order_acquire)) {
            LOG_ERROR("Attempted to allocate pinned memory after shutdown");
            return nullptr;
        }

        if (use_pinned) {
            auto cache_it = cache_.find(allocation_size);
            if (cache_it != cache_.end()) {
                auto& blocks = cache_it->second;
                for (auto block_it = blocks.begin(); block_it != blocks.end(); ++block_it) {
                    if (!block_it->all_uses_complete()) {
                        continue;
                    }

                    block_it->release_events();
                    void* ptr = block_it->ptr;
                    const size_t size = block_it->size;
                    const Backend backend = block_it->backend;
                    blocks.erase(block_it);
                    if (blocks.empty()) {
                        cache_.erase(cache_it);
                    }

                    allocated_blocks_[ptr] = {size, backend, {}};
                    stats_.allocated_bytes += size;
                    stats_.cached_bytes -= size;
                    ++stats_.cache_hits;
                    update_peaks_locked();
                    publish_stats_locked();
                    LFS_COUNTER_ADD("io.pinned_host.cache_hit", 1);
                    LFS_CUDA_BREADCRUMB_ARGS(
                        "tensor.pinned.allocate",
                        reinterpret_cast<uintptr_t>(ptr), 0, size);
                    return ptr;
                }

                LOG_TRACE("Pinned memory cache miss ({} blocks still busy): {} bytes",
                          blocks.size(), bytes);
            }
        }

        void* ptr = nullptr;
        Backend backend = Backend::MallocFallback;
        if (use_pinned) {
            const auto pre_call_state = sample_cuda_pre_call_state();
            const cudaError_t status = cudaHostAlloc(&ptr, allocation_size, cudaHostAllocDefault);
            if (status == cudaSuccess) {
                backend = Backend::CudaHost;
            } else {
                ensure_cuda_success(status, pre_call_state, "cudaHostAlloc(pinned block)",
                                    std::format("bytes={}, fallback=malloc", allocation_size),
                                    LFS_SOURCE_SITE_CURRENT(),
                                    CudaFailureDisposition::LogOnly);
                cudaGetLastError();
            }
        }

        if (!ptr) {
            ptr = std::malloc(allocation_size);
            if (!ptr) {
                LOG_ERROR("Host allocation failed for {} bytes", allocation_size);
                return nullptr;
            }
        }

        allocated_blocks_[ptr] = {allocation_size, backend, {}};
        stats_.allocated_bytes += allocation_size;
        ++stats_.num_allocs;
        ++stats_.cache_misses;
        if (backend == Backend::CudaHost) {
            ++stats_.cuda_host_allocs;
        } else {
            ++stats_.malloc_fallback_allocs;
        }
        update_peaks_locked();
        publish_stats_locked();
        LFS_COUNTER_ADD("io.pinned_host.cache_miss", 1);
        const double active_mib = stats_.allocated_bytes / static_cast<double>(MIB);
        const double cached_mib = stats_.cached_bytes / static_cast<double>(MIB);
        lock.unlock();

        register_cuda_address_range(ptr, allocation_size, "pinned-staging");

        LOG_TRACE("Pinned memory allocated: {} bytes (active: {:.2f} MiB, cache: {:.2f} MiB)",
                  bytes,
                  active_mib,
                  cached_mib);
        LFS_CUDA_BREADCRUMB_ARGS(
            "tensor.pinned.allocate",
            reinterpret_cast<uintptr_t>(ptr), 0, allocation_size);
        return ptr;
    }

    bool PinnedMemoryAllocator::record_uses(Block& block,
                                            const std::vector<cudaStream_t>& streams) {
        LFS_CUDA_BREADCRUMB("tensor.pinned.record_stream");
        bool all_streams_safe = true;
        for (const cudaStream_t stream : streams) {
            cudaEvent_t event = CudaEventPool::instance().acquire();
            if (event) {
                const cudaError_t record_status = cudaEventRecord(event, stream);
                if (record_status == cudaSuccess) {
                    block.ready_events.push_back(event);
                    continue;
                }
                CudaEventPool::instance().release(event);
                ensure_cuda_success(record_status, "cudaEventRecord(pinned block)",
                                    std::format("stream={}", static_cast<void*>(stream)),
                                    LFS_SOURCE_SITE_CURRENT(),
                                    CudaFailureDisposition::LogOnly);
                cudaGetLastError();
            }

            const cudaError_t sync_status = cudaStreamSynchronize(stream);
            if (sync_status != cudaSuccess && !is_cuda_shutdown(sync_status)) {
                ensure_cuda_success(sync_status, "cudaStreamSynchronize(pinned quarantine)",
                                    std::format("stream={}", static_cast<void*>(stream)),
                                    LFS_SOURCE_SITE_CURRENT(),
                                    CudaFailureDisposition::LogOnly);
                cudaGetLastError();
                all_streams_safe = false;
            }
        }

        if (all_streams_safe) {
            return true;
        }

        const cudaError_t device_status = cudaDeviceSynchronize();
        if (device_status == cudaSuccess || is_cuda_shutdown(device_status)) {
            if (is_cuda_shutdown(device_status)) {
                cudaGetLastError();
            }
            return true;
        }
        ensure_cuda_success(device_status, "cudaDeviceSynchronize(pinned quarantine)", {},
                            LFS_SOURCE_SITE_CURRENT(),
                            CudaFailureDisposition::LogOnly);
        cudaGetLastError();
        return false;
    }

    std::vector<PinnedMemoryAllocator::Block> PinnedMemoryAllocator::take_evictions_locked() {
        std::vector<Block> evicted;
        while (stats_.cached_bytes > cache_limit_bytes_) {
            auto oldest_class = cache_.end();
            uint64_t oldest_use = std::numeric_limits<uint64_t>::max();
            for (auto it = cache_.begin(); it != cache_.end(); ++it) {
                if (!it->second.empty() && it->second.front().last_used < oldest_use) {
                    oldest_use = it->second.front().last_used;
                    oldest_class = it;
                }
            }
            if (oldest_class == cache_.end()) {
                break;
            }

            Block& block = oldest_class->second.front();
            stats_.cached_bytes -= block.size;
            evicted.push_back(std::move(block));
            oldest_class->second.pop_front();
            if (oldest_class->second.empty()) {
                cache_.erase(oldest_class);
            }
        }
        return evicted;
    }

    void PinnedMemoryAllocator::release_blocks(std::vector<Block> blocks,
                                               const bool count_as_evictions) {
        size_t cuda_frees = 0;
        size_t fallback_frees = 0;
        size_t evicted_bytes = 0;
        size_t released_bytes = 0;

        for (Block& block : blocks) {
            bool safe_to_release = true;
            for (const cudaEvent_t event : block.ready_events) {
                const cudaError_t status = cudaEventSynchronize(event);
                if (status != cudaSuccess && !is_cuda_shutdown(status)) {
                    ensure_cuda_success(status, "cudaEventSynchronize(pinned release)", {},
                                        LFS_SOURCE_SITE_CURRENT(),
                                        CudaFailureDisposition::LogOnly);
                    cudaGetLastError();
                    safe_to_release = false;
                }
            }
            block.release_events();

            if (count_as_evictions) {
                evicted_bytes += block.size;
            }
            if (!safe_to_release) {
                LOG_ERROR("Leaking {}-byte host block rather than recycling memory still in use",
                          block.size);
                // Keep the range LIVE because the intentionally leaked block remains mapped.
                continue;
            }

            LFS_CUDA_BREADCRUMB_ARGS(
                "tensor.pinned.free",
                0, reinterpret_cast<uintptr_t>(block.ptr), block.size);
            if (block.backend == Backend::CudaHost) {
                const cudaError_t status = cudaFreeHost(block.ptr);
                if (status == cudaSuccess || is_cuda_shutdown(status)) {
                    unregister_cuda_address_range(block.ptr);
                    ++cuda_frees;
                    released_bytes += block.size;
                    if (is_cuda_shutdown(status)) {
                        cudaGetLastError();
                    }
                } else {
                    ensure_cuda_success(status, "cudaFreeHost(pinned block)",
                                        std::format("bytes={}", block.size),
                                        LFS_SOURCE_SITE_CURRENT(),
                                        CudaFailureDisposition::LogOnlyNoLatch);
                    cudaGetLastError();
                }
            } else {
                std::free(block.ptr);
                unregister_cuda_address_range(block.ptr);
                ++fallback_frees;
                released_bytes += block.size;
            }
            block.ptr = nullptr;
        }

        {
            std::lock_guard lock(mutex_);
            stats_.cuda_host_frees += cuda_frees;
            stats_.malloc_fallback_frees += fallback_frees;
            if (count_as_evictions) {
                stats_.evicted_blocks += blocks.size();
                stats_.evicted_bytes += evicted_bytes;
            }
        }

        if (!blocks.empty()) {
            LOG_DEBUG("Released {:.2f} MiB in {} host blocks",
                      released_bytes / static_cast<double>(MIB), blocks.size());
        }
    }

    void PinnedMemoryAllocator::deallocate(void* ptr, const cudaStream_t stream) {
        if (!ptr) {
            return;
        }

        AllocationInfo allocation;
        {
            std::lock_guard lock(mutex_);
            const auto it = allocated_blocks_.find(ptr);
            if (it == allocated_blocks_.end()) {
                LOG_WARN("Attempted to free unknown pinned memory pointer: {}", ptr);
                return;
            }

            allocation = std::move(it->second);
            allocated_blocks_.erase(it);
            stats_.allocated_bytes -= allocation.size;
            ++stats_.num_deallocs;
            publish_stats_locked();
        }

        if (std::find(allocation.extra_streams.begin(), allocation.extra_streams.end(), stream) ==
            allocation.extra_streams.end()) {
            allocation.extra_streams.push_back(stream);
        }

        Block block{ptr, allocation.size, allocation.backend};
        if (!record_uses(block, allocation.extra_streams)) {
            // The CUDA runtime could not establish that all users are finished.
            // Keep the storage out of both the cache and the backend free path.
            block.release_events();
            block.ptr = nullptr;
            return;
        }

        const bool cache_block =
            allocation.backend == Backend::CudaHost &&
            enabled_.load(std::memory_order_acquire) &&
            !shutdown_.load(std::memory_order_acquire);
        if (!cache_block) {
            std::vector<Block> released;
            released.push_back(std::move(block));
            release_blocks(std::move(released), false);
            std::lock_guard lock(mutex_);
            publish_stats_locked();
            return;
        }

        std::vector<Block> evicted;
        bool count_as_evictions = true;
        {
            std::lock_guard lock(mutex_);
            if (shutdown_.load(std::memory_order_acquire) ||
                !enabled_.load(std::memory_order_acquire)) {
                evicted.push_back(std::move(block));
                count_as_evictions = false;
            } else {
                block.last_used = ++lru_clock_;
                cache_[block.size].push_back(std::move(block));
                stats_.cached_bytes += allocation.size;
                update_peaks_locked();
                evicted = take_evictions_locked();
            }
            publish_stats_locked();
        }
        release_blocks(std::move(evicted), count_as_evictions);
        {
            std::lock_guard lock(mutex_);
            publish_stats_locked();
        }
    }

    void PinnedMemoryAllocator::record_stream(void* ptr, const cudaStream_t stream) {
        if (!ptr) {
            return;
        }
        std::lock_guard lock(mutex_);
        const auto it = allocated_blocks_.find(ptr);
        if (it == allocated_blocks_.end()) {
            return;
        }
        auto& streams = it->second.extra_streams;
        if (std::find(streams.begin(), streams.end(), stream) == streams.end()) {
            streams.push_back(stream);
        }
    }

    bool PinnedMemoryAllocator::is_cuda_host_allocation(const void* ptr) const {
        if (!ptr) {
            return false;
        }
        std::lock_guard lock(mutex_);
        const auto it = allocated_blocks_.find(const_cast<void*>(ptr));
        return it != allocated_blocks_.end() && it->second.backend == Backend::CudaHost;
    }

    void PinnedMemoryAllocator::release_stream(const cudaStream_t stream) {
        if (!stream) {
            return;
        }
        const cudaError_t status = cudaStreamSynchronize(stream);
        if (status != cudaSuccess && !is_cuda_shutdown(status)) {
            ensure_cuda_success(status, "cudaStreamSynchronize(pinned release_stream)",
                                std::format("stream={}", static_cast<void*>(stream)),
                                LFS_SOURCE_SITE_CURRENT(),
                                CudaFailureDisposition::LogOnly);
            cudaGetLastError();
            return;
        }
        if (is_cuda_shutdown(status)) {
            cudaGetLastError();
        }

        std::lock_guard lock(mutex_);
        for (auto& [ptr, info] : allocated_blocks_) {
            std::erase(info.extra_streams, stream);
        }
    }

    void PinnedMemoryAllocator::empty_cache_impl(const bool publish_stats) {
        std::vector<Block> blocks;
        {
            std::lock_guard lock(mutex_);
            size_t block_count = 0;
            for (const auto& [size, entries] : cache_) {
                block_count += entries.size();
            }
            blocks.reserve(block_count);
            for (auto& [size, entries] : cache_) {
                for (Block& block : entries) {
                    blocks.push_back(std::move(block));
                }
            }
            cache_.clear();
            stats_.cached_bytes = 0;
            if (publish_stats) {
                publish_stats_locked();
            }
        }

        release_blocks(std::move(blocks), false);
        if (publish_stats) {
            std::lock_guard lock(mutex_);
            publish_stats_locked();
        }
    }

    void PinnedMemoryAllocator::empty_cache() {
        empty_cache_impl(true);
    }

    PinnedMemoryAllocator::Stats PinnedMemoryAllocator::get_stats() const {
        std::lock_guard lock(mutex_);
        return stats_;
    }

    void PinnedMemoryAllocator::reset_stats() {
        std::lock_guard lock(mutex_);
        const size_t allocated = stats_.allocated_bytes;
        const size_t cached = stats_.cached_bytes;
        stats_ = Stats{};
        stats_.allocated_bytes = allocated;
        stats_.cached_bytes = cached;
        stats_.peak_allocated_bytes = allocated;
        stats_.peak_cached_bytes = cached;
        stats_.peak_total_bytes = allocated + cached;
        publish_stats_locked();
    }

    void PinnedMemoryAllocator::set_cache_limit_for_testing(const size_t bytes) {
        std::vector<Block> evicted;
        {
            std::lock_guard lock(mutex_);
            cache_limit_bytes_ = bytes;
            evicted = take_evictions_locked();
            publish_stats_locked();
        }
        release_blocks(std::move(evicted), true);
        {
            std::lock_guard lock(mutex_);
            publish_stats_locked();
        }
    }

    size_t PinnedMemoryAllocator::cache_limit_bytes() const {
        std::lock_guard lock(mutex_);
        return cache_limit_bytes_;
    }

    void PinnedMemoryAllocator::prewarm() {
        LOG_INFO("Pre-warming pinned memory cache with common sizes...");

        const std::vector<size_t> common_sizes = {
            540 * 540 * 3 * 4,
            720 * 820 * 3 * 4,
            1080 * 1920 * 3 * 4,
            1088 * 1920 * 3 * 4,
            2160 * 3840 * 3 * 4,
            1 * MIB,
            10 * MIB,
            50 * MIB,
            128 * MIB,
        };

        const auto start_time = std::chrono::steady_clock::now();
        for (const size_t size : common_sizes) {
            if (void* ptr = allocate(size)) {
                deallocate(ptr);
            }
        }
        const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start_time);
        const Stats stats = get_stats();

        LOG_INFO("Pre-warming complete: {:.2f} MiB cached in {} size requests ({} ms, "
                 "limit {:.2f} MiB)",
                 stats.cached_bytes / static_cast<double>(MIB),
                 common_sizes.size(),
                 duration.count(),
                 cache_limit_bytes() / static_cast<double>(MIB));
        LOG_DEBUG("Pinned cache hits: {}, misses: {}", stats.cache_hits, stats.cache_misses);
    }

} // namespace lfs::core

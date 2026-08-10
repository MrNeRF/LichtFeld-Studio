/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "diagnostics/export.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace lfs::diagnostics {

    enum class VramAllocationMethod : std::uint8_t {
        Unknown,
        Slab,
        Bucketed,
        Async,
        Direct,
        Arena,
        External,
    };

    /// Distinguishes real hooked allocations from disclosure re-descriptions.
    /// Only Hooked rows may contribute to a ledger root's attributed total (HUD_DESIGN §3.3).
    enum class VramRowKind : std::uint8_t {
        Hooked = 0,
        Sampled = 1,
        Static = 2,
    };

    struct VramMetricSnapshot {
        std::string scope;
        std::string label;
        std::size_t live_bytes = 0;
        std::size_t peak_bytes = 0;
        std::size_t allocated_bytes = 0;
        std::size_t freed_bytes = 0;
        std::uint64_t allocation_count = 0;
        std::uint64_t free_count = 0;
        VramAllocationMethod method = VramAllocationMethod::Unknown;
        VramRowKind kind = VramRowKind::Hooked;
    };

    struct VramTreeNodeSnapshot {
        std::string path;
        std::string name;
        std::uint32_t depth = 0;
        bool has_children = false;
        bool has_metrics = false;
        bool timer_scope = false;
        bool vram_delta_scope = false;
        bool logical_scope = false;
        std::size_t live_bytes = 0;
        std::size_t peak_bytes = 0;
        std::size_t allocated_bytes = 0;
        std::size_t freed_bytes = 0;
        std::uint64_t allocation_count = 0;
        std::uint64_t free_count = 0;
        std::uint64_t timer_call_count = 0;
        double total_ms = 0.0;
        double last_ms = 0.0;
        double max_ms = 0.0;
        double wall_p50_ms = 0.0;
        double wall_p95_ms = 0.0;
        double wall_p99_ms = 0.0;
        std::uint64_t gpu_call_count = 0;
        double gpu_total_ms = 0.0;
        double gpu_last_ms = 0.0;
        double gpu_max_ms = 0.0;
        std::uint64_t vram_delta_count = 0;
        std::int64_t last_vram_delta_bytes = 0;
        std::int64_t net_vram_delta_bytes = 0;
        std::int64_t max_vram_increase_bytes = 0;
        std::int64_t max_vram_decrease_bytes = 0;
    };

    struct NamedGauge {
        std::string key;
        double value = 0.0;
        std::uint64_t generation = 0;
    };

    struct NamedCounter {
        std::string key;
        std::uint64_t value = 0;
    };

    struct NamedHistogram {
        std::string key;
        std::uint64_t count = 0;
        double sum = 0.0;
        double min_value = 0.0;
        double max_value = 0.0;
        double p50 = 0.0;
        double p95 = 0.0;
        double p99 = 0.0;
    };

    struct TopAlloc {
        std::string scope;
        std::string label;
        std::size_t live_bytes = 0;
    };

    // Persistent training-state ledger.
    // Buckets: parameters, optimizer state, gradients, and densification data.
    // bytes_per_splat = total_bytes / live_splats (0 when N==0).
    struct TrainingStateLedger {
        std::size_t params_bytes = 0;
        std::size_t optimizer_bytes = 0;
        std::size_t gradients_or_helpers_bytes = 0;
        std::size_t densify_aux_bytes = 0;
        std::size_t total_bytes = 0;
        std::size_t live_splats = 0;
        double bytes_per_splat = 0.0;
    };

    /// Process-peak attribution. Every residual above the 938 MiB ex-cache
    /// baseline must be eliminated or listed with an owner; justified entries
    /// are budget-gated or required by design.
    enum class AttributionState : std::uint8_t {
        Unjustified = 0,
        Justified = 1,
        Nested = 2,
    };

    struct PeakSubsystemLine {
        std::string name;
        std::string owner;
        std::size_t bytes = 0;
        AttributionState state = AttributionState::Unjustified;
        std::string note;
    };

    struct PeakExCacheLedger {
        // Device-wide cudaMemGetInfo sample at the Bonsai quiet-GPU peak with
        // no GT cache. ex_cache must remain within 5%, unless every excess line
        // is justified.
        static constexpr std::size_t kExCacheBaselineBytes =
            static_cast<std::size_t>(938.3 * 1024.0 * 1024.0);
        // Logical training inventory already represented by the canonical
        // 1.5M benchmark inside kExCacheBaselineBytes.
        static constexpr std::size_t kTrainingStateBaselineBytes = 396404224;

        std::size_t peak_cuda_used_bytes = 0;     // device-wide free (legacy metric)
        std::size_t baseline_cuda_used_bytes = 0; // device-wide at train start
        std::size_t peak_pool_reserved_bytes = 0; // process-local pool HWM
        std::size_t peak_pool_used_bytes = 0;
        std::size_t pool_reserved_at_peak_bytes = 0;
        std::size_t pool_used_at_peak_bytes = 0;
        std::size_t ex_cache_net_bytes = 0;
        std::size_t training_state_bytes = 0;          // params+optim+densify (logical)
        std::size_t training_state_reserved_bytes = 0; // capacity-backed
        std::size_t training_state_baseline_bytes = kTrainingStateBaselineBytes;
        std::size_t training_state_growth_bytes = 0;
        std::size_t loss_workspace_required_bytes = 0;
        std::size_t loss_workspace_allocated_bytes = 0;
        std::size_t densify_workspace_bytes = 0;
        std::size_t mrnf_strategy_required_bytes = 0;
        std::size_t mrnf_strategy_allocated_bytes = 0;
        std::size_t mrnf_densify_n_required_bytes = 0;
        std::size_t mrnf_densify_n_allocated_bytes = 0;
        std::size_t mrnf_densify_child_required_bytes = 0;
        std::size_t mrnf_densify_child_allocated_bytes = 0;
        std::size_t mrnf_refine_peak_required_bytes = 0;
        std::size_t mrnf_refine_peak_allocated_bytes = 0;
        std::size_t mrnf_grow_peak_required_bytes = 0;
        std::size_t mrnf_grow_peak_allocated_bytes = 0;
        std::size_t pool_bucket_cache_bytes = 0; // at peak moment when possible
        std::size_t pool_bucket_live_rounding_waste_bytes = 0;
        std::size_t exportable_splat_bytes = 0;
        std::size_t fastgs_sort_required_bytes = 0;
        std::size_t fastgs_sort_allocated_bytes = 0;
        std::size_t fastgs_raster_live_bytes = 0; // per_prim+tile+sorted at peak
        std::size_t fastgs_raster_arena_live_bytes = 0;
        std::size_t fastgs_raster_sort_live_bytes = 0;
        std::size_t arena_required_bytes = 0;
        std::size_t arena_capacity_bytes = 0;
        std::size_t ex_cache_bytes = 0;
        std::size_t baseline_ex_cache_bytes = kExCacheBaselineBytes;
        std::size_t excess_over_baseline_bytes = 0;
        std::size_t justified_excess_bytes = 0;
        std::int64_t signed_residual_bytes = 0; // justified - excess, no clamp
        std::size_t unjustified_excess_bytes = 0;
        std::size_t over_attributed_bytes = 0;
        int peak_iter = 0;
        std::vector<PeakSubsystemLine> lines;
        std::vector<PeakSubsystemLine> peak_rows; // top VramProfiler rows at peak
    };

    struct VramProcessSnapshot {
        std::size_t cuda_used = 0;
        std::size_t cuda_total = 0;
        std::size_t cuda_pool_used = 0;
        std::size_t cuda_pool_reserved = 0;
        std::size_t cuda_pool_fragmentation = 0;
        // Live size-bucketed reuse cache: freed-but-retained cudaMallocAsync buffers
        // still counted in cuda_pool_used but not in the allocator's live map. Lets
        // the HUD split cuda.pool.untracked_used into reclaimable cache vs the rest.
        std::size_t cuda_pool_bucket_cache_bytes = 0;
        // Quantization waste for outstanding size-bucketed allocations only.
        // Cached free-list entries are excluded and reported by the cache field.
        std::size_t cuda_pool_bucket_live_waste_bytes = 0;
        // Device memory committed by the small-allocation slab allocator. Live
        // blocks are tracked individually; the reserve gap is free space inside
        // committed slabs that still belongs to this process.
        std::size_t cuda_slab_reserved_bytes = 0;
        std::size_t cuda_context_baseline = 0;
        std::size_t cuda_warmup_bytes = 0;
        std::size_t cuda_phase_primary_context = 0;
        std::size_t cuda_phase_default_pool = 0;
        std::size_t cuda_phase_printf_fifo = 0;
        std::size_t cuda_phase_stack_reserve = 0;
        std::size_t cuda_phase_malloc_heap = 0;
        std::size_t cuda_phase_curand_load = 0;
        std::size_t pinned_host_used = 0;
        std::size_t pinned_host_cached = 0;
        std::size_t pinned_host_peak = 0;
        // VK_EXT_memory_budget heap usage: the driver's estimate of the *whole
        // process* device heap (CUDA allocations, imported external blocks,
        // swap-chain, descriptor pools, driver internals) — NOT just VMA memory.
        std::size_t vulkan_vma_used = 0;
        // VMA blockBytes: device memory VMA actually owns. This is the real
        // Vulkan footprint the HUD attributes a residual against; vulkan_vma_used
        // overcounts it ~3x with non-Vulkan process memory.
        std::size_t vulkan_vma_block_bytes = 0;
        // CUDA-allocated exportable VMM block backing the splat tensors (shared
        // with Vulkan via VK_EXT_external_memory). Imported memory is not part of
        // vulkan_vma_block_bytes, so the per-tensor model.* rows account for it
        // without overlapping the Vulkan residual.
        std::size_t exportable_splat_bytes = 0;
        // Shared CUDA↔Vulkan scratch block (viewport arena external backing).
        // Separate from exportable_splat_bytes (C4); both form ledger root E.
        std::size_t shared_scratch_bytes = 0;
        std::size_t process_used = 0;
        std::size_t total_used = 0;
        std::size_t total = 0;
        std::string device_name;
        bool cuda_memory_valid = false;
        bool cuda_pool_valid = false;
        bool process_memory_valid = false;
    };

    struct VramProfilerSnapshot {
        bool enabled = false;
        int iteration = 0;
        std::uint64_t sequence = 0;
        std::uint64_t allocation_events = 0;
        std::uint64_t free_events = 0;
        std::uint64_t iter_allocation_events = 0;
        std::uint64_t iter_free_events = 0;
        double iter_per_second = 0.0;
        double iter_ms_p95 = 0.0;
        double iter_ms_last = 0.0;
        std::size_t accounted_live_bytes = 0;
        std::size_t accounted_cuda_pool_live_bytes = 0;
        std::size_t accounted_slab_live_bytes = 0;
        std::size_t accounted_bucketed_live_bytes = 0;
        std::size_t accounted_async_live_bytes = 0;
        std::size_t accounted_direct_live_bytes = 0;
        std::size_t accounted_arena_live_bytes = 0;
        std::size_t accounted_external_live_bytes = 0;
        std::size_t accounted_unknown_live_bytes = 0;
        std::size_t accounted_peak_bytes = 0;
        std::size_t sampled_live_bytes = 0;
        VramProcessSnapshot process;
        std::vector<VramMetricSnapshot> rows;
        std::vector<VramTreeNodeSnapshot> tree;
        std::vector<NamedGauge> gauges;
        std::vector<NamedCounter> iter_counters;
        std::vector<NamedCounter> total_counters;
        std::vector<NamedHistogram> histograms;
        std::vector<TopAlloc> top_live;
        TrainingStateLedger training_state;
    };

    class LFS_DIAGNOSTICS_API VramScope {
    public:
        explicit VramScope(std::string_view scope);
        ~VramScope();

        VramScope(const VramScope&) = delete;
        VramScope& operator=(const VramScope&) = delete;

        VramScope(VramScope&& other) noexcept;
        VramScope& operator=(VramScope&& other) noexcept;

    private:
        bool active_ = false;
    };

    class LFS_DIAGNOSTICS_API VramDeltaScope {
    public:
        explicit VramDeltaScope(std::string_view scope);
        ~VramDeltaScope();

        VramDeltaScope(const VramDeltaScope&) = delete;
        VramDeltaScope& operator=(const VramDeltaScope&) = delete;

        VramDeltaScope(VramDeltaScope&& other) noexcept;
        VramDeltaScope& operator=(VramDeltaScope&& other) noexcept;

    private:
        bool active_ = false;
        bool start_valid_ = false;
        bool pushed_scope_ = false;
        std::size_t start_used_bytes_ = 0;
    };

    class LFS_DIAGNOSTICS_API TraceScope {
    public:
        explicit TraceScope(std::string_view scope);
        ~TraceScope();

        TraceScope(const TraceScope&) = delete;
        TraceScope& operator=(const TraceScope&) = delete;
        TraceScope(TraceScope&&) = delete;
        TraceScope& operator=(TraceScope&&) = delete;

    private:
        bool active_ = false;
        std::int64_t start_ns_ = 0;
    };

    class LFS_DIAGNOSTICS_API GpuTimeScope {
    public:
        GpuTimeScope(std::string_view scope, void* stream);
        ~GpuTimeScope();

        GpuTimeScope(const GpuTimeScope&) = delete;
        GpuTimeScope& operator=(const GpuTimeScope&) = delete;
        GpuTimeScope(GpuTimeScope&&) = delete;
        GpuTimeScope& operator=(GpuTimeScope&&) = delete;

    private:
        bool active_ = false;
        std::int32_t event_pair_ = -1;
        void* stream_ = nullptr;
    };

    class LFS_DIAGNOSTICS_API VramProfiler {
    public:
        static VramProfiler& instance();

        void setEnabled(bool enabled);
        [[nodiscard]] bool enabled() const;

        void beginIteration(int iteration);
        void pushScope(std::string_view scope);
        void popScope();
        void pushTimerScope(std::string_view scope);
        void popTimerScope(double elapsed_ms);
        bool pushVramDeltaScope(std::string_view scope,
                                std::size_t& start_used_bytes,
                                bool& pushed_scope);
        void popVramDeltaScope(std::size_t start_used_bytes, bool start_valid, bool pushed_scope);

        void recordAllocation(void* ptr,
                              std::size_t bytes,
                              VramAllocationMethod method);
        void recordAllocation(void* ptr,
                              std::size_t bytes,
                              VramAllocationMethod method,
                              std::string_view label);
        void recordDeallocation(void* ptr);
        void relabelAllocation(void* ptr, std::string_view label);
        void recordCurrentBytes(std::string_view scope,
                                std::string_view label,
                                std::size_t bytes,
                                VramAllocationMethod method = VramAllocationMethod::External);
        void recordStaticBytes(std::string_view scope,
                               std::string_view label,
                               std::size_t bytes,
                               VramAllocationMethod method = VramAllocationMethod::External);
        void clearStaticScope(std::string_view scope);
        void recordTimerSample(std::string_view scope, double elapsed_ms);
        void clearScope(std::string_view scope);

        std::int32_t acquireGpuEventPair(std::string_view scope, void* stream);
        void releaseGpuEventPair(std::int32_t pair, void* stream);
        void drainGpuEvents();

        void setGauge(std::string_view key, double value);
        void addCounter(std::string_view key, std::uint64_t delta, bool per_iteration);
        void recordHistogram(std::string_view key, double value);

        void setPinnedHostMemory(std::size_t active_bytes,
                                 std::size_t cached_bytes,
                                 std::size_t peak_bytes);
        void setVulkanVmaUsed(std::size_t bytes);
        void setVulkanVmaBlockBytes(std::size_t bytes);
        void setCudaPoolBucketCacheBytes(std::size_t bytes);
        void setCudaPoolBucketLiveWasteBytes(std::size_t bytes);
        void setCudaSlabReservedBytes(std::size_t bytes);
        void setExportableSplatBytes(std::size_t bytes);
        void setSharedScratchBytes(std::size_t bytes);
        void captureCudaDeviceBaseline();
        void captureCudaWarmupDelta();
        void recordCudaPhaseBytes(std::string_view phase, std::size_t bytes);
        void setCudaContextBaselineBytes(std::size_t bytes);
        /// Device-wide cudaMemGetInfo captured at primary-context initialization.
        /// Late configure() samples include unrelated allocations in ex_cache_net.
        [[nodiscard]] std::size_t cudaDeviceBaselineBytes() const;

        void sampleCudaMemory();
        void updateProcessMemory(std::size_t process_used,
                                 std::size_t total_used,
                                 std::size_t total,
                                 std::string device_name);

        // Training-state bytes-per-splat ledger. Caller computes the
        // ledger from SplatData + AdamOptimizer; profiler stores the last value
        // for HUD / bench scripts.
        void setTrainingStateLedger(const TrainingStateLedger& ledger);
        [[nodiscard]] TrainingStateLedger trainingStateLedger() const;

        [[nodiscard]] VramProfilerSnapshot snapshot() const;

    private:
        VramProfiler();
        ~VramProfiler();
        VramProfiler(const VramProfiler&) = delete;
        VramProfiler& operator=(const VramProfiler&) = delete;

        struct Impl;
        Impl* impl_;
    };

} // namespace lfs::diagnostics

#define LFS_DIAGNOSTICS_CONCAT_INNER(a, b) a##b
#define LFS_DIAGNOSTICS_CONCAT(a, b)       LFS_DIAGNOSTICS_CONCAT_INNER(a, b)
#define LFS_VRAM_SCOPE(name) \
    ::lfs::diagnostics::VramScope LFS_DIAGNOSTICS_CONCAT(_lfs_vram_scope_, __LINE__)(name)
#define LOG_VRAM_DIFF(name) \
    ::lfs::diagnostics::VramDeltaScope LFS_DIAGNOSTICS_CONCAT(_lfs_vram_delta_scope_, __LINE__)(name)
#define LFS_TRACE(name) \
    ::lfs::diagnostics::TraceScope LFS_DIAGNOSTICS_CONCAT(_lfs_trace_scope_, __LINE__)(name)
#define LFS_GPU_TIME(name, stream)                                                           \
    ::lfs::diagnostics::GpuTimeScope LFS_DIAGNOSTICS_CONCAT(_lfs_gpu_time_scope_, __LINE__)( \
        name, static_cast<void*>(stream))
#define LFS_GAUGE(key, value) \
    ::lfs::diagnostics::VramProfiler::instance().setGauge(key, static_cast<double>(value))
#define LFS_COUNTER_ADD(key, n)                                                            \
    ::lfs::diagnostics::VramProfiler::instance().addCounter(key,                           \
                                                            static_cast<std::uint64_t>(n), \
                                                            true)
#define LFS_COUNTER_TOTAL_ADD(key, n)                                                      \
    ::lfs::diagnostics::VramProfiler::instance().addCounter(key,                           \
                                                            static_cast<std::uint64_t>(n), \
                                                            false)
#define LFS_HIST(key, value) \
    ::lfs::diagnostics::VramProfiler::instance().recordHistogram(key, static_cast<double>(value))

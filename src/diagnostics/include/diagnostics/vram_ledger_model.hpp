/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "diagnostics/export.hpp"
#include "diagnostics/vram_profiler.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace lfs::diagnostics {

    /// Independent measured reservations that may be summed against NVML process_used.
    /// Order matches HUD_DESIGN.md §3.2. Root F is included on Linux (L0: NVML compute
    /// process memory includes DEVICE_LOCAL Vulkan allocations).
    enum class VramLedgerRootId : std::uint8_t {
        CudaAsyncPool = 0,     // A
        CudaSlab = 1,          // B
        CudaDirect = 2,        // C
        RasterizerArena = 3,   // D (empty when external-backed; bytes live in E)
        ExportableVmm = 4,     // E (splat store + shared scratch)
        VulkanVma = 5,         // F (device-local VMA blockBytes only)
        VulkanExternal = 6,    // G (raw vkAllocateMemory outside VMA)
        CudaContextDriver = 7, // H
        Unattributed = 8,      // I (signed residual magnitude)
        Count = 9,
    };

    enum class LedgerClosureState : std::uint8_t {
        Closed = 0,
        Gap = 1,
        Over = 2,
    };

    struct LFS_DIAGNOSTICS_API LedgerSignedResidual {
        std::int64_t signed_residual_bytes = 0; // justified - measured
        std::size_t under_claim_bytes = 0;      // measured - justified when residual < 0
        std::size_t over_claim_bytes = 0;       // justified - measured when residual > 0
    };

    /// Signed byte arithmetic shared by the live and peak ledger models.
    [[nodiscard]] LFS_DIAGNOSTICS_API std::int64_t signed_byte_difference(std::size_t lhs,
                                                                          std::size_t rhs);

    [[nodiscard]] LFS_DIAGNOSTICS_API std::size_t signed_byte_magnitude(std::int64_t bytes) noexcept;

    /// residual = signed_byte_difference(attributed, measured), split into
    /// under-claim and over-claim magnitudes.
    [[nodiscard]] LFS_DIAGNOSTICS_API LedgerSignedResidual
    make_signed_residual(std::size_t attributed_bytes, std::size_t measured_bytes);

    [[nodiscard]] LFS_DIAGNOSTICS_API const char* vram_ledger_root_name(VramLedgerRootId id) noexcept;

    struct LFS_DIAGNOSTICS_API VramLedgerNode {
        std::string name;
        std::string owner;
        std::size_t measured_bytes = 0;
        std::size_t attributed_bytes = 0; // sum of Justified children only
        std::size_t required_bytes = 0;   // 0 when no disclosure pair
        bool has_required = false;
        AttributionState state = AttributionState::Justified;
        LedgerClosureState closure = LedgerClosureState::Closed;
        std::int64_t residual_bytes = 0;
        VramRowKind row_kind = VramRowKind::Hooked;
        VramLedgerRootId root_id = VramLedgerRootId::Unattributed;
        std::string note;
        std::vector<VramLedgerNode> children;
    };

    struct LFS_DIAGNOSTICS_API VramLedgerTree {
        std::size_t process_used_bytes = 0;
        std::size_t attributed_bytes = 0; // sum of Justified roots A–H
        LedgerSignedResidual residual;
        LedgerClosureState closure = LedgerClosureState::Closed;
        std::size_t epsilon_bytes = 0;
        std::vector<VramLedgerNode> roots; // A–H plus Unattributed child
    };

    struct LFS_DIAGNOSTICS_API VramLedgerPolicy {
        /// max(min_abs, parent * min_frac). Design starting point: 2 MiB / 0.25%.
        std::size_t epsilon_min_bytes = 2ull * 1024ull * 1024ull;
        double epsilon_frac = 0.0025;
        bool include_vulkan_in_sum = true; // L0 answer on this machine
        /// When true, arena measured under root D is zeroed if process reports
        /// external-backed arena (exportable path owns the bytes in root E).
        bool arena_external_backing = false;
    };

    [[nodiscard]] LFS_DIAGNOSTICS_API std::size_t
    ledger_epsilon(std::size_t parent_bytes, const VramLedgerPolicy& policy);

    /// Live closed ledger over a profiler snapshot. Justified children contribute to
    /// per-root attributed totals. Sampled/Static rows are Nested disclosures except
    /// root F (Vulkan VMA), where named Sampled rows justify used-inside-blocks and
    /// driver free caps free_inside; residual beyond that is Unjustified GAP.
    [[nodiscard]] LFS_DIAGNOSTICS_API VramLedgerTree
    buildLiveLedger(const VramProfilerSnapshot& snapshot, const VramLedgerPolicy& policy = {});

    /// Peak ex-cache residual + line state, shared with perf_bench so the gate and
    /// model cannot disagree on signed residual arithmetic.
    struct LFS_DIAGNOSTICS_API PeakExCacheInputs {
        std::size_t peak_cuda_used_bytes = 0;
        std::size_t baseline_cuda_used_bytes = 0;
        std::size_t baseline_ex_cache_bytes = PeakExCacheLedger::kExCacheBaselineBytes;
        std::size_t training_state_bytes = 0;
        std::size_t training_state_reserved_bytes = 0;
        std::size_t training_state_baseline_bytes =
            PeakExCacheLedger::kTrainingStateBaselineBytes;
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
        std::size_t pool_bucket_cache_bytes = 0;
        std::size_t pool_bucket_live_rounding_waste_bytes = 0;
        std::size_t exportable_splat_bytes = 0;
        std::size_t fastgs_sort_required_bytes = 0;
        std::size_t fastgs_sort_allocated_bytes = 0;
        std::size_t fastgs_raster_live_bytes = 0;
        std::size_t fastgs_raster_arena_live_bytes = 0;
        std::size_t fastgs_raster_sort_live_bytes = 0;
        std::size_t arena_required_bytes = 0;
        std::size_t arena_capacity_bytes = 0;
        std::size_t peak_io_ring_bytes = 0;
        std::size_t peak_io_external_bytes = 0;
        std::size_t peak_steady_pinned_host_bytes = 0;
    };

    /// Fills residual fields and attribution lines identically to the pre-refactor
    /// PerfBenchCollector::peak_ex_cache_ledger policy.
    [[nodiscard]] LFS_DIAGNOSTICS_API PeakExCacheLedger
    buildPeakExCacheLedger(const PeakExCacheInputs& in);

} // namespace lfs::diagnostics

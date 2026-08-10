/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "diagnostics/vram_ledger_model.hpp"
#include "diagnostics/vram_profiler.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <string>

using namespace lfs::diagnostics;

namespace {

    VramMetricSnapshot make_row(std::string scope,
                                std::string label,
                                const std::size_t live,
                                const VramRowKind kind,
                                const VramAllocationMethod method = VramAllocationMethod::Unknown) {
        VramMetricSnapshot r;
        r.scope = std::move(scope);
        r.label = std::move(label);
        r.live_bytes = live;
        r.peak_bytes = live;
        r.kind = kind;
        r.method = method;
        return r;
    }

} // namespace

TEST(VramLedger, SignedByteMagnitudeAtNegativeExtreme) {
    EXPECT_EQ(signed_byte_magnitude(0), 0u);
    EXPECT_EQ(signed_byte_magnitude(42), 42u);
    EXPECT_EQ(signed_byte_magnitude(-1), 1u);
    EXPECT_EQ(signed_byte_magnitude(std::numeric_limits<std::int64_t>::min()),
              static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()) + 1u);
}

TEST(VramLedger, SignedResidualSplitsUnderAndOver) {
    const auto under = make_signed_residual(/*attributed=*/100, /*measured=*/150);
    EXPECT_EQ(under.signed_residual_bytes, -50);
    EXPECT_EQ(under.under_claim_bytes, 50u);
    EXPECT_EQ(under.over_claim_bytes, 0u);

    const auto over = make_signed_residual(/*attributed=*/200, /*measured=*/150);
    EXPECT_EQ(over.signed_residual_bytes, 50);
    EXPECT_EQ(over.under_claim_bytes, 0u);
    EXPECT_EQ(over.over_claim_bytes, 50u);

    const auto closed = make_signed_residual(100, 100);
    EXPECT_EQ(closed.signed_residual_bytes, 0);
    EXPECT_EQ(closed.under_claim_bytes, 0u);
    EXPECT_EQ(closed.over_claim_bytes, 0u);
}

TEST(VramLedger, SampledRowsNeverContributeToSum) {
    VramProfilerSnapshot snap;
    snap.process.process_memory_valid = true;
    snap.process.process_used = 1000;
    snap.process.cuda_pool_valid = true;
    snap.process.cuda_pool_reserved = 1000;
    snap.process.cuda_pool_used = 800;
    snap.accounted_bucketed_live_bytes = 500;
    snap.accounted_async_live_bytes = 0;
    // Sampled disclosure re-describes the same 500 bytes.
    snap.rows.push_back(make_row("optimizer.adam", "means.exp_avg", 500, VramRowKind::Sampled,
                                 VramAllocationMethod::Unknown));
    snap.rows.push_back(make_row("train.step", "bucketed", 500, VramRowKind::Hooked,
                                 VramAllocationMethod::Bucketed));

    const auto tree = buildLiveLedger(snap);
    // Root A measured is pool reserved; attributed children must not double-count Sampled.
    ASSERT_FALSE(tree.roots.empty());
    const auto& pool = tree.roots.front();
    EXPECT_EQ(pool.root_id, VramLedgerRootId::CudaAsyncPool);
    std::size_t justified = 0;
    std::size_t nested = 0;
    for (const auto& c : pool.children) {
        if (c.state == AttributionState::Justified) {
            justified += c.measured_bytes;
        } else if (c.state == AttributionState::Nested) {
            nested += c.measured_bytes;
        }
    }
    EXPECT_GE(justified, 500u);
    EXPECT_GE(nested, 500u);
    // Nested must not inflate the root measured sum used for top-level residual.
    EXPECT_EQ(pool.measured_bytes, 1000u);
}

TEST(VramLedger, DeliberateDoubleCountProducesOverNotCap) {
    // Top-level residual: attributed roots sum > process_used → OVER, not capped.
    // Use multi-MiB sizes so residual exceeds the 2 MiB epsilon floor.
    constexpr std::size_t MiB = 1024ull * 1024ull;
    VramProfilerSnapshot snap;
    snap.process.process_memory_valid = true;
    snap.process.process_used = 100 * MiB;
    snap.process.cuda_pool_valid = true;
    snap.process.cuda_pool_reserved = 80 * MiB;
    snap.process.cuda_slab_reserved_bytes = 50 * MiB; // A+B = 130 MiB > 100 MiB
    snap.accounted_slab_live_bytes = 50 * MiB;

    VramLedgerPolicy policy;
    policy.epsilon_min_bytes = 2 * MiB;
    const auto tree = buildLiveLedger(snap, policy);
    EXPECT_EQ(tree.closure, LedgerClosureState::Over);
    EXPECT_GT(tree.residual.over_claim_bytes, 0u);
    EXPECT_EQ(tree.residual.under_claim_bytes, 0u);
    // Unattributed row present with over note (C6: not capped away)
    const auto& last = tree.roots.back();
    EXPECT_EQ(last.root_id, VramLedgerRootId::Unattributed);
    EXPECT_EQ(last.closure, LedgerClosureState::Over);
    EXPECT_GE(last.measured_bytes, 20 * MiB);
}

TEST(VramLedger, ExternalBackingMovesArenaToExportable) {
    VramProfilerSnapshot snap;
    snap.process.process_memory_valid = true;
    snap.process.process_used = 2000;
    snap.accounted_arena_live_bytes = 400;
    snap.process.exportable_splat_bytes = 300;
    snap.process.shared_scratch_bytes = 400; // owns arena backing

    VramLedgerPolicy policy;
    policy.arena_external_backing = true;
    const auto tree = buildLiveLedger(snap, policy);

    std::size_t arena_m = 0;
    std::size_t export_m = 0;
    for (const auto& r : tree.roots) {
        if (r.root_id == VramLedgerRootId::RasterizerArena) {
            arena_m = r.measured_bytes;
        }
        if (r.root_id == VramLedgerRootId::ExportableVmm) {
            export_m = r.measured_bytes;
        }
    }
    EXPECT_EQ(arena_m, 0u);
    EXPECT_EQ(export_m, 700u);
}

TEST(VramLedger, C2SlabNotInPoolAccounted) {
    // Simulator of snapshot post-C2: slab live is separate from pool accounted.
    VramProfilerSnapshot snap;
    snap.accounted_slab_live_bytes = 64 * 1024 * 1024;
    snap.accounted_bucketed_live_bytes = 100 * 1024 * 1024;
    snap.accounted_async_live_bytes = 0;
    // accounted_cuda_pool_live must equal bucketed+async only (not slab).
    snap.accounted_cuda_pool_live_bytes =
        snap.accounted_bucketed_live_bytes + snap.accounted_async_live_bytes;
    snap.process.cuda_pool_valid = true;
    snap.process.cuda_pool_used = 120 * 1024 * 1024;
    snap.process.cuda_pool_reserved = 128 * 1024 * 1024;
    snap.process.cuda_slab_reserved_bytes = 80 * 1024 * 1024;
    snap.process.process_memory_valid = true;
    snap.process.process_used = 300 * 1024 * 1024;

    const auto tree = buildLiveLedger(snap);
    std::size_t slab_m = 0;
    std::size_t pool_m = 0;
    for (const auto& r : tree.roots) {
        if (r.root_id == VramLedgerRootId::CudaSlab) {
            slab_m = r.measured_bytes;
        }
        if (r.root_id == VramLedgerRootId::CudaAsyncPool) {
            pool_m = r.measured_bytes;
        }
    }
    EXPECT_EQ(slab_m, 80u * 1024u * 1024u);
    EXPECT_EQ(pool_m, 128u * 1024u * 1024u);
    EXPECT_NE(slab_m, 0u);
}

TEST(VramLedger, PeakExCacheIdenticalGateNumbersFromMyConfirm3Inputs) {
    // Inputs reconstructed from .codex_tmp/exactmem/myconfirm3/perf_bench.json
    // peak_ex_cache + ledger fields so residual math is bit-identical.
    PeakExCacheInputs in;
    in.peak_cuda_used_bytes = 2399600640ull;
    in.baseline_cuda_used_bytes = 1025310720ull;
    in.baseline_ex_cache_bytes = PeakExCacheLedger::kExCacheBaselineBytes;
    in.training_state_bytes = 396404224ull;
    in.training_state_reserved_bytes = 457959120ull;
    in.loss_workspace_required_bytes = 34576640ull;
    in.loss_workspace_allocated_bytes = 34576640ull;
    in.densify_workspace_bytes = 0;
    in.pool_bucket_cache_bytes = 9437184ull;
    in.pool_bucket_live_rounding_waste_bytes = 19128948ull;
    in.exportable_splat_bytes = 0;
    in.fastgs_sort_required_bytes = 63624703ull;
    in.fastgs_sort_allocated_bytes = 63624703ull;
    in.fastgs_raster_live_bytes = 122711027ull;
    in.fastgs_raster_arena_live_bytes = 111860735ull;
    in.fastgs_raster_sort_live_bytes = 15779572ull;
    in.arena_required_bytes = 142468352ull;
    in.arena_capacity_bytes = 146800640ull;
    // Reverse-engineered exact bytes from myconfirm3 justified_new_bytes (446221888)
    // minus the other justified lines (capacity overhead, loss, pool, sort, raster).
    in.peak_io_external_bytes = 146038782ull;
    in.peak_io_ring_bytes = 32ull * 1024ull * 1024ull;
    in.peak_steady_pinned_host_bytes = 10116ull;

    const auto out = buildPeakExCacheLedger(in);

    // Gate numbers from myconfirm3 (must be bit-identical)
    EXPECT_EQ(out.ex_cache_net_bytes, 1374289920ull);
    EXPECT_EQ(out.excess_over_baseline_bytes, 390411060ull);
    EXPECT_EQ(out.signed_residual_bytes, static_cast<std::int64_t>(55810828));
    EXPECT_EQ(out.over_attributed_bytes, 55810828ull);
    EXPECT_EQ(out.unjustified_excess_bytes, 0ull);
    EXPECT_EQ(out.justified_excess_bytes, 446221888ull);
}

TEST(VramLedger, NineRootsPresentWhenVulkanIncluded) {
    VramProfilerSnapshot snap;
    snap.process.process_memory_valid = true;
    snap.process.process_used = 1;
    VramLedgerPolicy policy;
    policy.include_vulkan_in_sum = true;
    const auto tree = buildLiveLedger(snap, policy);
    // A–H + Unattributed = 9
    EXPECT_EQ(tree.roots.size(), 9u);
}

TEST(VramLedger, ShaderBytecodeCollapsedUnderRootH) {
    VramProfilerSnapshot snap;
    snap.process.process_memory_valid = true;
    snap.process.process_used = 64ull * 1024ull * 1024ull;
    snap.process.cuda_context_baseline = 32ull * 1024ull * 1024ull;
    snap.process.vulkan_vma_block_bytes = 16ull * 1024ull * 1024ull;
    snap.rows.push_back(make_row("vksplat.shaders.slang.spirv.projection_forward", "",
                                 200ull * 1024ull, VramRowKind::Static));
    snap.rows.push_back(make_row("vksplat.shaders.slang.spirv.rasterize_forward", "",
                                 300ull * 1024ull, VramRowKind::Static));
    snap.rows.push_back(make_row("vksplat.shaders.glsl.spirv.radix_sort_visible", "",
                                 100ull * 1024ull, VramRowKind::Static));
    // A real VMA-named row must still appear under root F.
    snap.rows.push_back(make_row("vulkan.image.color", "", 1024ull * 1024ull,
                                 VramRowKind::Hooked, VramAllocationMethod::External));

    VramLedgerPolicy policy;
    policy.include_vulkan_in_sum = true;
    const auto tree = buildLiveLedger(snap, policy);

    const VramLedgerNode* root_f = nullptr;
    const VramLedgerNode* root_h = nullptr;
    for (const auto& r : tree.roots) {
        if (r.root_id == VramLedgerRootId::VulkanVma)
            root_f = &r;
        if (r.root_id == VramLedgerRootId::CudaContextDriver)
            root_h = &r;
    }
    ASSERT_NE(root_f, nullptr);
    ASSERT_NE(root_h, nullptr);

    // No per-module shader flood under VMA.
    for (const auto& c : root_f->children) {
        EXPECT_EQ(c.name.find("shaders."), std::string::npos) << c.name;
        EXPECT_EQ(c.name.find("shader bytecode"), std::string::npos) << c.name;
    }

    int shader_groups = 0;
    for (const auto& c : root_h->children) {
        if (c.name.find("shader bytecode") != std::string::npos) {
            ++shader_groups;
            EXPECT_NE(c.name.find("3 modules"), std::string::npos) << c.name;
            EXPECT_EQ(c.measured_bytes, 600ull * 1024ull);
            EXPECT_EQ(c.state, AttributionState::Nested);
        }
    }
    EXPECT_EQ(shader_groups, 1);
}

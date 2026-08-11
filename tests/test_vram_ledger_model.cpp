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

TEST(VramLedger, ViewerExternalRowsLandInTheirMeasuredRoots) {
    constexpr std::size_t MiB = 1024ull * 1024ull;
    VramProfilerSnapshot snap;
    snap.process.process_memory_valid = true;
    snap.process.process_used = 100 * MiB;
    snap.rows.push_back(make_row("io.nvimagecodec", "driver_or_direct", 12 * MiB,
                                 VramRowKind::Static, VramAllocationMethod::External));
    snap.rows.push_back(make_row("vulkan.external.viewport_interop", "color", 8 * MiB,
                                 VramRowKind::Sampled, VramAllocationMethod::External));

    const auto tree = buildLiveLedger(snap);
    const auto* cuda_direct = static_cast<const VramLedgerNode*>(nullptr);
    const auto* vulkan_external = static_cast<const VramLedgerNode*>(nullptr);
    for (const auto& root : tree.roots) {
        if (root.root_id == VramLedgerRootId::CudaDirect) {
            cuda_direct = &root;
        } else if (root.root_id == VramLedgerRootId::VulkanExternal) {
            vulkan_external = &root;
        }
    }

    ASSERT_NE(cuda_direct, nullptr);
    ASSERT_NE(vulkan_external, nullptr);
    EXPECT_EQ(cuda_direct->measured_bytes, 12 * MiB);
    EXPECT_EQ(cuda_direct->attributed_bytes, 12 * MiB);
    EXPECT_EQ(vulkan_external->measured_bytes, 8 * MiB);
    EXPECT_EQ(vulkan_external->attributed_bytes, 8 * MiB);
    EXPECT_EQ(cuda_direct->closure, LedgerClosureState::Closed);
    EXPECT_EQ(vulkan_external->closure, LedgerClosureState::Closed);
}

TEST(VramLedger, WholeProcessVulkanBudgetRemainderStaysUnjustified) {
    constexpr std::size_t MiB = 1024ull * 1024ull;
    VramProfilerSnapshot snap;
    snap.process.process_memory_valid = true;
    snap.process.process_used = 200 * MiB;
    snap.process.cuda_pool_valid = true;
    snap.process.cuda_pool_reserved = 100 * MiB;
    snap.process.vulkan_vma_block_bytes = 20 * MiB;
    snap.process.vulkan_vma_used = 200 * MiB; // Whole-process heap usage, not VMA-only.
    snap.process.cuda_context_baseline = 30 * MiB;
    snap.rows.push_back(make_row("vulkan.external.viewport_interop", "color", 10 * MiB,
                                 VramRowKind::Sampled, VramAllocationMethod::External));

    const auto tree = buildLiveLedger(snap);
    const VramLedgerNode* vulkan_external = nullptr;
    for (const auto& root : tree.roots) {
        if (root.root_id == VramLedgerRootId::VulkanExternal) {
            vulkan_external = &root;
            break;
        }
    }

    ASSERT_NE(vulkan_external, nullptr);
    EXPECT_EQ(vulkan_external->measured_bytes, 180 * MiB);
    EXPECT_EQ(vulkan_external->attributed_bytes, 10 * MiB);
    EXPECT_EQ(vulkan_external->closure, LedgerClosureState::Gap);

    bool saw_unattributed_budget = false;
    for (const auto& child : vulkan_external->children) {
        EXPECT_EQ(child.name.find("render_targets_and_driver"), std::string::npos);
        if (child.name == "vulkan.memory_budget.unattributed_process_usage") {
            saw_unattributed_budget = true;
            EXPECT_EQ(child.measured_bytes, 170 * MiB);
            EXPECT_EQ(child.state, AttributionState::Unjustified);
        }
    }
    EXPECT_TRUE(saw_unattributed_budget);

    // Independent cover is pool + VMA + named external + context. The ambiguous
    // budget remainder must become a 40 MiB honest gap, never a false OVER.
    EXPECT_EQ(tree.attributed_bytes, 160 * MiB);
    EXPECT_EQ(tree.residual.under_claim_bytes, 40 * MiB);
    EXPECT_EQ(tree.residual.over_claim_bytes, 0u);
    EXPECT_EQ(tree.closure, LedgerClosureState::Gap);
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

TEST(VramLedger, PeakExCacheNamesTrainingGrowthAndDisjointMrnfWorkspaces) {
    PeakExCacheInputs in;
    in.baseline_cuda_used_bytes = 1000;
    in.baseline_ex_cache_bytes = PeakExCacheLedger::kExCacheBaselineBytes;
    in.training_state_baseline_bytes = 400;
    in.training_state_bytes = 1000;
    in.training_state_reserved_bytes = 1100;
    in.mrnf_strategy_required_bytes = 80;
    in.mrnf_strategy_allocated_bytes = 100;
    in.mrnf_densify_n_required_bytes = 20;
    in.mrnf_densify_n_allocated_bytes = 30;
    in.mrnf_densify_child_required_bytes = 40;
    in.mrnf_densify_child_allocated_bytes = 50;
    in.mrnf_refine_peak_required_bytes = 60;
    in.mrnf_refine_peak_allocated_bytes = 60;
    in.mrnf_grow_peak_required_bytes = 70;
    in.mrnf_grow_peak_allocated_bytes = 70;

    constexpr std::size_t justified = 600 + 100 + 100 + 30 + 50 + 60 + 70;
    in.peak_cuda_used_bytes =
        in.baseline_cuda_used_bytes + in.baseline_ex_cache_bytes + justified;

    const auto out = buildPeakExCacheLedger(in);
    EXPECT_EQ(out.training_state_growth_bytes, 600u);
    EXPECT_EQ(out.justified_excess_bytes, justified);
    EXPECT_EQ(out.signed_residual_bytes, 0);

    const auto find_line = [&](const std::string& name) -> const PeakSubsystemLine* {
        for (const auto& line : out.lines) {
            if (line.name == name) {
                return &line;
            }
        }
        return nullptr;
    };

    const auto* growth = find_line("training_state_growth");
    ASSERT_NE(growth, nullptr);
    EXPECT_EQ(growth->bytes, 600u);
    EXPECT_EQ(growth->state, AttributionState::Justified);

    const auto* child = find_line("mrnf_densify_child_allocated");
    ASSERT_NE(child, nullptr);
    EXPECT_EQ(child->bytes, 50u);
    EXPECT_EQ(child->state, AttributionState::Justified);

    const auto* grow = find_line("mrnf_grow_peak_exclusive_allocated");
    ASSERT_NE(grow, nullptr);
    EXPECT_EQ(grow->bytes, 70u);
    EXPECT_EQ(grow->state, AttributionState::Justified);
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

// N1: module_warmup is a device-wide delta, not inside context baseline.
TEST(VramLedger, ModuleWarmupNestedUnderRootH) {
    constexpr std::size_t MiB = 1024ull * 1024ull;
    VramProfilerSnapshot snap;
    snap.process.process_memory_valid = true;
    snap.process.process_used = 300 * MiB;
    // Live idle shape: primary_context equals the whole root H measured baseline.
    snap.process.cuda_context_baseline = 242 * MiB;
    snap.process.cuda_phase_primary_context = 242 * MiB;
    snap.process.cuda_warmup_bytes = 36 * MiB;

    VramLedgerPolicy policy;
    policy.epsilon_min_bytes = 2 * MiB;
    const auto tree = buildLiveLedger(snap, policy);

    const VramLedgerNode* root_h = nullptr;
    for (const auto& r : tree.roots) {
        if (r.root_id == VramLedgerRootId::CudaContextDriver)
            root_h = &r;
    }
    ASSERT_NE(root_h, nullptr);
    EXPECT_EQ(root_h->measured_bytes, 242 * MiB);

    bool saw_primary = false;
    bool saw_warmup = false;
    for (const auto& c : root_h->children) {
        if (c.name == "primary_context") {
            saw_primary = true;
            EXPECT_EQ(c.state, AttributionState::Justified);
            EXPECT_EQ(c.measured_bytes, 242 * MiB);
        }
        if (c.name == "module_warmup") {
            saw_warmup = true;
            EXPECT_EQ(c.state, AttributionState::Nested);
            EXPECT_EQ(c.measured_bytes, 36 * MiB);
        }
    }
    EXPECT_TRUE(saw_primary);
    EXPECT_TRUE(saw_warmup);
    // Warmup must not inflate justified sum past measured → permanent OVER.
    EXPECT_EQ(root_h->attributed_bytes, 242 * MiB);
    EXPECT_NE(root_h->closure, LedgerClosureState::Over);
    EXPECT_TRUE(root_h->closure == LedgerClosureState::Closed ||
                root_h->closure == LedgerClosureState::Gap);
}

// N2: Sampled named VMA rows justify used-inside-blocks; free is the residual.
TEST(VramLedger, VulkanVmaSampledNamedAttribution) {
    constexpr std::size_t MiB = 1024ull * 1024ull;
    // Live idle shape from designer re-validation: ~20.8 MiB free next to VMA.
    // Use exact bytes so free_inside == blockBytes - named used (no Hooked dual path).
    constexpr std::size_t kBlocks = 100 * MiB;
    constexpr std::size_t kNamedA = 40 * MiB;
    constexpr std::size_t kNamedB = 30 * MiB;
    constexpr std::size_t kNamedC = 9 * MiB + 200ull * 1024ull; // ~9.2 MiB
    constexpr std::size_t kNamed = kNamedA + kNamedB + kNamedC;
    constexpr std::size_t kExpectedFree = kBlocks - kNamed; // ~20.8 MiB

    VramProfilerSnapshot snap;
    snap.process.process_memory_valid = true;
    snap.process.process_used = 200 * MiB;
    snap.process.vulkan_vma_block_bytes = kBlocks;
    // All Vulkan rows are Sampled (recordCurrentBytes) — pre-fix only Hooked counted.
    snap.rows.push_back(make_row("vulkan.image.color", "viewport", kNamedA, VramRowKind::Sampled));
    snap.rows.push_back(make_row("vulkan.buffer.ubo", "frame", kNamedB, VramRowKind::Sampled));
    snap.rows.push_back(
        make_row("vulkan.rmlui.render_layer", "layer_color:rmlui:1920x1080@0x1", kNamedC,
                 VramRowKind::Sampled));
    // Neighbouring driver free sampler — Nested, not part of used sum.
    snap.rows.push_back(make_row("vulkan.vma", "allocator_free_in_blocks", kExpectedFree,
                                 VramRowKind::Sampled));

    VramLedgerPolicy policy;
    policy.include_vulkan_in_sum = true;
    policy.epsilon_min_bytes = 2 * MiB;
    const auto tree = buildLiveLedger(snap, policy);

    const VramLedgerNode* root_f = nullptr;
    for (const auto& r : tree.roots) {
        if (r.root_id == VramLedgerRootId::VulkanVma)
            root_f = &r;
    }
    ASSERT_NE(root_f, nullptr);
    EXPECT_EQ(root_f->measured_bytes, kBlocks);

    std::size_t free_inside = 0;
    std::size_t driver_free = 0;
    std::size_t justified_named = 0;
    int free_rows = 0;
    for (const auto& c : root_f->children) {
        if (c.name == "free_inside_blocks") {
            ++free_rows;
            free_inside = c.measured_bytes;
            EXPECT_EQ(c.state, AttributionState::Justified);
        } else if (c.name.find("allocator_free_in_blocks") != std::string::npos) {
            driver_free = c.measured_bytes;
            EXPECT_EQ(c.state, AttributionState::Nested);
        } else if (c.state == AttributionState::Justified) {
            justified_named += c.measured_bytes;
        }
    }
    EXPECT_EQ(free_rows, 1);
    EXPECT_EQ(justified_named, kNamed);
    EXPECT_EQ(free_inside, kExpectedFree);
    EXPECT_EQ(driver_free, kExpectedFree);
    // free_inside agrees with driver free within ε (exact when named coverage complete).
    EXPECT_EQ(free_inside, driver_free);
    // free + named == measured → CLOSED (not 100% free_inside retention).
    EXPECT_EQ(root_f->attributed_bytes, root_f->measured_bytes);
    EXPECT_EQ(root_f->closure, LedgerClosureState::Closed);
    // ~20.8 MiB free.
    EXPECT_NEAR(static_cast<double>(free_inside) / static_cast<double>(MiB), 20.8, 0.05);
}

// N3: serial-stamped vulkan.rmlui.texture.* collapse like shader bytecode.
TEST(VramLedger, RmlUiTexturesCollapsedUnderRootF) {
    constexpr std::size_t MiB = 1024ull * 1024ull;
    VramProfilerSnapshot snap;
    snap.process.process_memory_valid = true;
    snap.process.process_used = 64 * MiB;
    snap.process.vulkan_vma_block_bytes = 32 * MiB;
    // Three serial-stamped textures (pointer suffix differs) + one unrelated VMA row.
    snap.rows.push_back(make_row("vulkan.rmlui.texture", "texture:icon.png:32x32@0xaaa",
                                 128ull * 1024ull, VramRowKind::Sampled));
    snap.rows.push_back(make_row("vulkan.rmlui.texture", "texture:logo.png:64x64@0xbbb",
                                 256ull * 1024ull, VramRowKind::Sampled));
    snap.rows.push_back(make_row("vulkan.rmlui.texture", "texture:icon.png:32x32@0xccc",
                                 128ull * 1024ull, VramRowKind::Sampled));
    snap.rows.push_back(make_row("vulkan.image.color", "viewport", 4 * MiB, VramRowKind::Sampled));

    VramLedgerPolicy policy;
    policy.include_vulkan_in_sum = true;
    const auto tree = buildLiveLedger(snap, policy);

    const VramLedgerNode* root_f = nullptr;
    for (const auto& r : tree.roots) {
        if (r.root_id == VramLedgerRootId::VulkanVma)
            root_f = &r;
    }
    ASSERT_NE(root_f, nullptr);

    // No per-texture flood.
    int texture_groups = 0;
    int raw_texture_rows = 0;
    for (const auto& c : root_f->children) {
        if (c.name.find("RmlUi textures") != std::string::npos) {
            ++texture_groups;
            EXPECT_NE(c.name.find("3"), std::string::npos) << c.name;
            EXPECT_EQ(c.measured_bytes, 512ull * 1024ull);
            EXPECT_EQ(c.state, AttributionState::Justified);
        }
        if (c.name.find("vulkan.rmlui.texture") != std::string::npos) {
            ++raw_texture_rows;
        }
    }
    EXPECT_EQ(texture_groups, 1);
    EXPECT_EQ(raw_texture_rows, 0);

    // Without driver free authority, residual is unattributed (not false free_inside).
    const std::size_t kNamed = 4 * MiB + 512ull * 1024ull;
    const std::size_t kResidual = 32 * MiB - kNamed;
    std::size_t free_inside = 0;
    std::size_t unattributed = 0;
    for (const auto& c : root_f->children) {
        if (c.name == "free_inside_blocks")
            free_inside = c.measured_bytes;
        if (c.name == "unattributed_in_root") {
            unattributed = c.measured_bytes;
            EXPECT_EQ(c.state, AttributionState::Unjustified);
        }
    }
    EXPECT_EQ(free_inside, 0u);
    EXPECT_EQ(unattributed, kResidual);
    EXPECT_EQ(root_f->closure, LedgerClosureState::Gap);
}

// N2b: incomplete named VMA coverage must GAP (not close via free_inside sponge).
// free_inside = min(blocks - named, driver_free); remainder is unattributed_in_root.
TEST(VramLedger, IncompleteNamedVmaCoverageYieldsGap) {
    constexpr std::size_t MiB = 1024ull * 1024ull;
    // Live idle shape: blocks >> named used; driver free is the true free authority.
    constexpr std::size_t kBlocks = 23 * MiB;
    constexpr std::size_t kNamed = 5 * MiB;      // incomplete instrumentation
    constexpr std::size_t kDriverFree = 4 * MiB; // allocator_free_in_blocks
    constexpr std::size_t kResidual = kBlocks - kNamed;
    constexpr std::size_t kFreeInside = kDriverFree; // min(residual, driver_free)
    constexpr std::size_t kUnattributed = kResidual - kFreeInside;

    VramProfilerSnapshot snap;
    snap.process.process_memory_valid = true;
    snap.process.process_used = 200 * MiB;
    snap.process.vulkan_vma_block_bytes = kBlocks;
    snap.rows.push_back(make_row("vulkan.image.color", "viewport", kNamed, VramRowKind::Sampled));
    snap.rows.push_back(make_row("vulkan.vma", "allocator_free_in_blocks", kDriverFree,
                                 VramRowKind::Sampled));

    VramLedgerPolicy policy;
    policy.include_vulkan_in_sum = true;
    policy.epsilon_min_bytes = 1 * MiB;
    const auto tree = buildLiveLedger(snap, policy);

    const VramLedgerNode* root_f = nullptr;
    for (const auto& r : tree.roots) {
        if (r.root_id == VramLedgerRootId::VulkanVma)
            root_f = &r;
    }
    ASSERT_NE(root_f, nullptr);
    EXPECT_EQ(root_f->measured_bytes, kBlocks);

    std::size_t free_inside = 0;
    std::size_t unattributed = 0;
    std::size_t driver_free = 0;
    for (const auto& c : root_f->children) {
        if (c.name == "free_inside_blocks") {
            free_inside = c.measured_bytes;
            EXPECT_EQ(c.state, AttributionState::Justified);
            EXPECT_EQ(c.note, "retention");
        } else if (c.name == "unattributed_in_root") {
            unattributed = c.measured_bytes;
            EXPECT_EQ(c.state, AttributionState::Unjustified);
        } else if (c.name.find("allocator_free_in_blocks") != std::string::npos) {
            driver_free = c.measured_bytes;
            EXPECT_EQ(c.state, AttributionState::Nested);
        }
    }
    EXPECT_EQ(driver_free, kDriverFree);
    EXPECT_EQ(free_inside, kFreeInside);
    EXPECT_EQ(unattributed, kUnattributed);
    // Justified = named + free_inside; unattributed does not close the root.
    EXPECT_EQ(root_f->attributed_bytes, kNamed + kFreeInside);
    EXPECT_LT(root_f->attributed_bytes, root_f->measured_bytes);
    EXPECT_EQ(root_f->closure, LedgerClosureState::Gap);

    // Complete coverage still closes (regression guard for the complete-coverage path).
    {
        constexpr std::size_t kFullNamed = kBlocks - kDriverFree;
        VramProfilerSnapshot full;
        full.process.process_memory_valid = true;
        full.process.process_used = 200 * MiB;
        full.process.vulkan_vma_block_bytes = kBlocks;
        full.rows.push_back(
            make_row("vulkan.image.color", "viewport", kFullNamed, VramRowKind::Sampled));
        full.rows.push_back(make_row("vulkan.vma", "allocator_free_in_blocks", kDriverFree,
                                     VramRowKind::Sampled));
        const auto full_tree = buildLiveLedger(full, policy);
        const VramLedgerNode* full_f = nullptr;
        for (const auto& r : full_tree.roots) {
            if (r.root_id == VramLedgerRootId::VulkanVma)
                full_f = &r;
        }
        ASSERT_NE(full_f, nullptr);
        bool saw_unattr = false;
        for (const auto& c : full_f->children) {
            if (c.name == "unattributed_in_root")
                saw_unattr = true;
        }
        EXPECT_FALSE(saw_unattr);
        EXPECT_EQ(full_f->attributed_bytes, full_f->measured_bytes);
        EXPECT_EQ(full_f->closure, LedgerClosureState::Closed);
    }
}

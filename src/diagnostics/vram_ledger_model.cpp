/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "diagnostics/vram_ledger_model.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <limits>
#include <map>
#include <stdexcept>

namespace lfs::diagnostics {
    namespace {

        [[nodiscard]] bool starts_with(const std::string_view s, const std::string_view prefix) {
            return s.size() >= prefix.size() && s.compare(0, prefix.size(), prefix) == 0;
        }

        [[nodiscard]] bool is_fastgs_or_gsplat_logical_scope(const std::string_view scope) {
            return starts_with(scope, "rasterizer.fastgs") ||
                   starts_with(scope, "rasterizer.gsplat");
        }

        [[nodiscard]] bool is_vulkan_named_row(const VramMetricSnapshot& row) {
            return starts_with(row.scope, "vulkan.") || starts_with(row.scope, "vksplat") ||
                   starts_with(row.label, "vulkan.");
        }

        [[nodiscard]] bool is_vulkan_external_row(const VramMetricSnapshot& row) {
            return starts_with(row.scope, "vulkan.external") ||
                   starts_with(row.label, "vulkan.external");
        }

        // nvImageCodec's viewer decoder probe reports driver/direct CUDA bytes as a
        // Static External row.  They are real per-process device allocations, but do
        // not flow through the CUDA pool/direct accounting counters.
        [[nodiscard]] bool is_viewer_cuda_external_row(const VramMetricSnapshot& row) {
            return starts_with(row.scope, "io.nvimagecodec");
        }

        // recordStaticBytes SPIR-V estimates — not VMA allocations. Group under root H.
        [[nodiscard]] bool is_shader_bytecode_row(const VramMetricSnapshot& row) {
            return starts_with(row.scope, "vksplat.shaders.") ||
                   starts_with(row.scope, "vksplat.shaders");
        }

        // Serial-stamped RmlUi textures (TextureVramLabel appends @0xPTR). Group under root F.
        [[nodiscard]] bool is_rmlui_texture_row(const VramMetricSnapshot& row) {
            return starts_with(row.scope, "vulkan.rmlui.texture");
        }

        // VMA block free sampler (blockBytes − allocationBytes). Nested driver reference;
        // free_inside_blocks is derived separately and must agree with this within ε.
        [[nodiscard]] bool is_vma_free_sampler_row(const VramMetricSnapshot& row) {
            return starts_with(row.scope, "vulkan.vma") &&
                   (row.label.find("free_in_blocks") != std::string::npos ||
                    row.label.find("allocator_free") != std::string::npos ||
                    starts_with(row.label, "free"));
        }

        [[nodiscard]] std::string format_mib(const std::size_t bytes) {
            const double mib = static_cast<double>(bytes) / (1024.0 * 1024.0);
            if (mib >= 10.0)
                return std::format("{:.0f} MiB", mib);
            if (mib >= 1.0)
                return std::format("{:.1f} MiB", mib);
            return std::format("{:.2f} MiB", mib);
        }

        void apply_closure(VramLedgerNode& node, const std::size_t epsilon) {
            const auto res = make_signed_residual(node.attributed_bytes, node.measured_bytes);
            node.residual_bytes = res.signed_residual_bytes;
            if (signed_byte_magnitude(res.signed_residual_bytes) <= epsilon) {
                node.closure = LedgerClosureState::Closed;
            } else if (res.signed_residual_bytes < 0) {
                node.closure = LedgerClosureState::Gap;
            } else {
                node.closure = LedgerClosureState::Over;
            }
        }

        VramLedgerNode make_root(const VramLedgerRootId id,
                                 const std::size_t measured,
                                 const char* owner) {
            VramLedgerNode n;
            n.name = vram_ledger_root_name(id);
            n.owner = owner;
            n.measured_bytes = measured;
            n.root_id = id;
            n.state = AttributionState::Justified;
            n.row_kind = VramRowKind::Hooked;
            return n;
        }

        void add_child(VramLedgerNode& parent,
                       std::string name,
                       const std::size_t bytes,
                       const AttributionState state,
                       const VramRowKind kind,
                       std::string note = {}) {
            if (bytes == 0 && state != AttributionState::Unjustified) {
                return;
            }
            VramLedgerNode c;
            c.name = std::move(name);
            c.owner = parent.owner;
            c.measured_bytes = bytes;
            c.attributed_bytes = state == AttributionState::Justified ? bytes : 0;
            c.state = state;
            c.row_kind = kind;
            c.root_id = parent.root_id;
            c.note = std::move(note);
            if (state == AttributionState::Justified) {
                parent.attributed_bytes += bytes;
            }
            parent.children.push_back(std::move(c));
        }

        struct RequiredAllocationPair {
            std::size_t required_bytes = 0;
            std::size_t allocated_bytes = 0;
            bool has_required = false;
            bool has_allocated = false;
        };

        [[nodiscard]] std::size_t gauge_bytes(const double value) {
            if (!std::isfinite(value) || value <= 0.0) {
                return 0;
            }
            const auto max_bytes = static_cast<double>(std::numeric_limits<std::size_t>::max());
            return static_cast<std::size_t>(std::min(value, max_bytes));
        }

        [[nodiscard]] std::map<std::string, RequiredAllocationPair>
        discover_required_allocation_pairs(const VramProfilerSnapshot& snapshot) {
            constexpr std::string_view kPrefix = "vram.audit.";
            constexpr std::string_view kRequiredSuffix = ".required_bytes";
            constexpr std::string_view kAllocatedSuffix = ".allocated_bytes";

            std::map<std::string, RequiredAllocationPair> pairs;
            for (const auto& gauge : snapshot.gauges) {
                const std::string_view key = gauge.key;
                if (!starts_with(key, kPrefix)) {
                    continue;
                }

                const auto set_value = [&](const std::string_view suffix,
                                           const bool required) {
                    if (!key.ends_with(suffix) || key.size() <= kPrefix.size() + suffix.size()) {
                        return false;
                    }
                    const auto name = key.substr(
                        kPrefix.size(), key.size() - kPrefix.size() - suffix.size());
                    auto& pair = pairs[std::string(name)];
                    if (required) {
                        pair.required_bytes = gauge_bytes(gauge.value);
                        pair.has_required = true;
                    } else {
                        pair.allocated_bytes = gauge_bytes(gauge.value);
                        pair.has_allocated = true;
                    }
                    return true;
                };

                if (!set_value(kRequiredSuffix, true)) {
                    (void)set_value(kAllocatedSuffix, false);
                }
            }
            return pairs;
        }

        void add_required_allocation_disclosures(
            VramLedgerNode& pool_root,
            VramLedgerNode& arena_root,
            const std::map<std::string, RequiredAllocationPair>& pairs) {
            for (const auto& [name, pair] : pairs) {
                if (!pair.has_required || !pair.has_allocated) {
                    continue;
                }

                const bool rasterizer_pair = name.find("raster") != std::string::npos ||
                                             name.find("fastgs") != std::string::npos;
                auto& parent = rasterizer_pair ? arena_root : pool_root;
                VramLedgerNode disclosure;
                disclosure.name = name;
                disclosure.owner = parent.owner;
                disclosure.measured_bytes = pair.allocated_bytes;
                disclosure.required_bytes = pair.required_bytes;
                disclosure.has_required = true;
                disclosure.state = AttributionState::Nested;
                disclosure.row_kind = VramRowKind::Sampled;
                disclosure.root_id = parent.root_id;
                disclosure.note = name == "fastgs_raster_live" &&
                                          pair.required_bytes == pair.allocated_bytes
                                      ? "degenerate required/allocated pair"
                                      : "required/allocated disclosure";
                parent.children.push_back(std::move(disclosure));
            }
        }

    } // namespace

    std::int64_t signed_byte_difference(const std::size_t lhs, const std::size_t rhs) {
        constexpr auto kMaxSignedBytes =
            static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max());
        if (lhs > kMaxSignedBytes || rhs > kMaxSignedBytes) {
            throw std::overflow_error("VRAM ledger byte count exceeds signed residual range");
        }
        return static_cast<std::int64_t>(lhs) - static_cast<std::int64_t>(rhs);
    }

    std::size_t signed_byte_magnitude(const std::int64_t bytes) noexcept {
        return bytes >= 0 ? static_cast<std::size_t>(bytes)
                          : static_cast<std::size_t>(-(bytes + 1)) + 1;
    }

    LedgerSignedResidual make_signed_residual(const std::size_t attributed_bytes,
                                              const std::size_t measured_bytes) {
        LedgerSignedResidual out;
        out.signed_residual_bytes = signed_byte_difference(attributed_bytes, measured_bytes);
        out.under_claim_bytes = out.signed_residual_bytes < 0
                                    ? signed_byte_magnitude(out.signed_residual_bytes)
                                    : 0;
        out.over_claim_bytes = out.signed_residual_bytes > 0
                                   ? signed_byte_magnitude(out.signed_residual_bytes)
                                   : 0;
        return out;
    }

    const char* vram_ledger_root_name(const VramLedgerRootId id) noexcept {
        switch (id) {
        case VramLedgerRootId::CudaAsyncPool:
            return "CUDA async pool";
        case VramLedgerRootId::CudaSlab:
            return "CUDA slab";
        case VramLedgerRootId::CudaDirect:
            return "CUDA direct";
        case VramLedgerRootId::RasterizerArena:
            return "Rasterizer arena";
        case VramLedgerRootId::ExportableVmm:
            return "Exportable VMM";
        case VramLedgerRootId::VulkanVma:
            return "Vulkan VMA blocks";
        case VramLedgerRootId::VulkanExternal:
            return "Vulkan external";
        case VramLedgerRootId::CudaContextDriver:
            return "CUDA context and driver";
        case VramLedgerRootId::Unattributed:
            return "Unattributed";
        case VramLedgerRootId::Count:
            break;
        }
        return "Unknown";
    }

    std::size_t ledger_epsilon(const std::size_t parent_bytes, const VramLedgerPolicy& policy) {
        const auto frac = static_cast<std::size_t>(
            static_cast<double>(parent_bytes) * policy.epsilon_frac);
        return std::max(policy.epsilon_min_bytes, frac);
    }

    VramLedgerTree buildLiveLedger(const VramProfilerSnapshot& snapshot,
                                   const VramLedgerPolicy& policy) {
        VramLedgerTree tree;
        const auto disclosure_pairs = discover_required_allocation_pairs(snapshot);
        const auto& proc = snapshot.process;
        tree.process_used_bytes =
            proc.process_memory_valid ? proc.process_used : 0;
        tree.epsilon_bytes = ledger_epsilon(tree.process_used_bytes, policy);

        // --- Root measured sizes (disjoint API sources) ---
        const std::size_t pool_reserved =
            proc.cuda_pool_valid ? proc.cuda_pool_reserved : 0;
        const std::size_t slab_reserved = proc.cuda_slab_reserved_bytes;
        const std::size_t direct_live = snapshot.accounted_direct_live_bytes;
        // C3: do not also sum cuda_phase_default_pool when pool reserved is a root.
        std::size_t arena_live = snapshot.accounted_arena_live_bytes;
        if (policy.arena_external_backing) {
            arena_live = 0; // bytes live under exportable VMM / shared scratch
        }
        const std::size_t exportable =
            proc.exportable_splat_bytes + proc.shared_scratch_bytes;
        const std::size_t vma_blocks =
            policy.include_vulkan_in_sum ? proc.vulkan_vma_block_bytes : 0;

        std::size_t vulkan_named_external = 0;
        // VK_EXT_memory_budget is a whole-process heap estimate. After subtracting
        // VMA blocks it still overlaps CUDA and exportable roots, so it is only a
        // coverage ceiling for root G, never an independently summable reservation.
        const std::size_t vulkan_budget_external =
            proc.vulkan_vma_used > vma_blocks ? proc.vulkan_vma_used - vma_blocks : 0;
        std::size_t viewer_cuda_external = 0;
        std::size_t hooked_pool_bytes = 0;
        std::size_t hooked_slab_bytes = 0;
        std::size_t hooked_direct_bytes = 0;
        std::size_t hooked_arena_bytes = 0;

        for (const auto& row : snapshot.rows) {
            if (row.live_bytes == 0) {
                continue;
            }
            if (is_vulkan_external_row(row)) {
                // recordCurrentBytes rows are Sampled, while direct Vulkan imports
                // may be Hooked. Both are genuine root-G attribution.
                vulkan_named_external += row.live_bytes;
            }
            if (is_viewer_cuda_external_row(row)) {
                viewer_cuda_external += row.live_bytes;
            }
            if (row.kind == VramRowKind::Hooked) {
                switch (row.method) {
                case VramAllocationMethod::Bucketed:
                case VramAllocationMethod::Async:
                    hooked_pool_bytes += row.live_bytes;
                    break;
                case VramAllocationMethod::Slab:
                    hooked_slab_bytes += row.live_bytes;
                    break;
                case VramAllocationMethod::Direct:
                    hooked_direct_bytes += row.live_bytes;
                    break;
                case VramAllocationMethod::Arena:
                    if (!policy.arena_external_backing) {
                        hooked_arena_bytes += row.live_bytes;
                    }
                    break;
                case VramAllocationMethod::External:
                default:
                    break;
                }
            } else if (row.kind == VramRowKind::Sampled || row.kind == VramRowKind::Static) {
                if (is_vulkan_external_row(row)) {
                    // external disclosure only
                }
            }
        }

        // Prefer process method buckets for hooked totals when available (C2: slab
        // is no longer folded into accounted_cuda_pool_live_bytes).
        if (snapshot.accounted_bucketed_live_bytes + snapshot.accounted_async_live_bytes > 0) {
            hooked_pool_bytes =
                snapshot.accounted_bucketed_live_bytes + snapshot.accounted_async_live_bytes;
        }
        if (snapshot.accounted_slab_live_bytes > 0) {
            hooked_slab_bytes = snapshot.accounted_slab_live_bytes;
        }
        if (snapshot.accounted_direct_live_bytes > 0) {
            hooked_direct_bytes = snapshot.accounted_direct_live_bytes;
        }
        if (snapshot.accounted_arena_live_bytes > 0 && !policy.arena_external_backing) {
            hooked_arena_bytes = snapshot.accounted_arena_live_bytes;
        }

        auto root_a = make_root(VramLedgerRootId::CudaAsyncPool, pool_reserved, "cuda_pool");
        add_child(root_a, "hooked_pool_live", hooked_pool_bytes, AttributionState::Justified,
                  VramRowKind::Hooked);
        if (proc.cuda_pool_bucket_cache_bytes > 0) {
            add_child(root_a, "bucket_cache", proc.cuda_pool_bucket_cache_bytes,
                      AttributionState::Justified, VramRowKind::Hooked, "reclaimable");
        }
        // Nested disclosures under pool (Sampled optimizer/model tensors, etc.)
        for (const auto& row : snapshot.rows) {
            if (row.live_bytes == 0 || row.kind == VramRowKind::Hooked) {
                continue;
            }
            if (is_fastgs_or_gsplat_logical_scope(row.scope)) {
                continue; // placed under arena / untracked
            }
            if (row.method == VramAllocationMethod::Bucketed ||
                row.method == VramAllocationMethod::Async ||
                starts_with(row.scope, "optimizer.") ||
                starts_with(row.scope, "model.") ||
                starts_with(row.scope, "train.")) {
                add_child(root_a,
                          row.scope + (row.label.empty() ? "" : "." + row.label),
                          row.live_bytes,
                          AttributionState::Nested,
                          row.kind,
                          "disclosure, not counted again");
            }
        }
        {
            const auto used = proc.cuda_pool_valid ? proc.cuda_pool_used : 0;
            if (pool_reserved > used) {
                add_child(root_a, "reserved_not_used", pool_reserved - used,
                          AttributionState::Justified, VramRowKind::Hooked, "retention");
            }
            // Untracked inside pool: used - justified children so far (excluding nested)
            std::size_t justified_used = 0;
            for (const auto& c : root_a.children) {
                if (c.state == AttributionState::Justified && c.name != "reserved_not_used") {
                    justified_used += c.measured_bytes;
                }
            }
            if (used > justified_used) {
                add_child(root_a, "untracked_pool_used", used - justified_used,
                          AttributionState::Justified, VramRowKind::Hooked, "untracked");
            }
            // Recompute attributed from Justified children only
            root_a.attributed_bytes = 0;
            for (const auto& c : root_a.children) {
                if (c.state == AttributionState::Justified) {
                    root_a.attributed_bytes += c.measured_bytes;
                }
            }
            apply_closure(root_a, ledger_epsilon(root_a.measured_bytes, policy));
        }

        auto root_b = make_root(VramLedgerRootId::CudaSlab, slab_reserved, "cuda_slab");
        add_child(root_b, "hooked_slab_live", hooked_slab_bytes, AttributionState::Justified,
                  VramRowKind::Hooked);
        if (slab_reserved > hooked_slab_bytes) {
            add_child(root_b, "slab_reserve_gap", slab_reserved - hooked_slab_bytes,
                      AttributionState::Justified, VramRowKind::Hooked, "retention");
        }
        root_b.attributed_bytes = 0;
        for (const auto& c : root_b.children) {
            if (c.state == AttributionState::Justified) {
                root_b.attributed_bytes += c.measured_bytes;
            }
        }
        apply_closure(root_b, ledger_epsilon(root_b.measured_bytes, policy));

        const std::size_t cuda_direct_measured = direct_live + viewer_cuda_external;
        auto root_c = make_root(VramLedgerRootId::CudaDirect, cuda_direct_measured, "cuda_direct");
        add_child(root_c, "hooked_direct_live", hooked_direct_bytes, AttributionState::Justified,
                  VramRowKind::Hooked);
        if (viewer_cuda_external > 0) {
            for (const auto& row : snapshot.rows) {
                if (row.live_bytes == 0 || !is_viewer_cuda_external_row(row)) {
                    continue;
                }
                add_child(root_c,
                          row.scope + (row.label.empty() ? "" : "." + row.label),
                          row.live_bytes,
                          AttributionState::Justified,
                          row.kind,
                          "viewer decoder driver/direct allocation");
            }
        }
        root_c.attributed_bytes = hooked_direct_bytes + viewer_cuda_external;
        apply_closure(root_c, ledger_epsilon(root_c.measured_bytes, policy));

        auto root_d = make_root(VramLedgerRootId::RasterizerArena, arena_live, "rasterizer");
        add_child(root_d, "hooked_arena_live", hooked_arena_bytes, AttributionState::Justified,
                  VramRowKind::Hooked);
        for (const auto& row : snapshot.rows) {
            if (row.live_bytes == 0) {
                continue;
            }
            if (!is_fastgs_or_gsplat_logical_scope(row.scope)) {
                continue;
            }
            // Nested disclosures (capacity/current/peak, per_primitive, etc.)
            add_child(root_d,
                      row.scope + (row.label.empty() ? "" : "." + row.label),
                      row.live_bytes,
                      AttributionState::Nested,
                      row.kind == VramRowKind::Hooked ? VramRowKind::Sampled : row.kind,
                      "disclosure, not counted again");
        }
        root_d.attributed_bytes = hooked_arena_bytes;
        apply_closure(root_d, ledger_epsilon(root_d.measured_bytes, policy));

        add_required_allocation_disclosures(root_a, root_d, disclosure_pairs);

        auto root_e = make_root(VramLedgerRootId::ExportableVmm, exportable, "viewport");
        if (proc.exportable_splat_bytes > 0) {
            add_child(root_e, "splat_store", proc.exportable_splat_bytes,
                      AttributionState::Justified, VramRowKind::Hooked);
        }
        if (proc.shared_scratch_bytes > 0) {
            add_child(root_e, "shared_scratch", proc.shared_scratch_bytes,
                      AttributionState::Justified, VramRowKind::Hooked);
        }
        root_e.attributed_bytes =
            proc.exportable_splat_bytes + proc.shared_scratch_bytes;
        apply_closure(root_e, ledger_epsilon(root_e.measured_bytes, policy));

        // Root F: VMA blockBytes is the measured reservation. Named Vulkan rows are
        // recorded via recordCurrentBytes → Sampled (nothing else covers VMA), so they
        // justify the used portion. Driver free (allocator_free_in_blocks) is Nested
        // authority for free_inside_blocks; residual beyond that is unattributed_in_root
        // (Unjustified) so incomplete named coverage reports GAP, not a false CLOSED.
        auto root_f = make_root(VramLedgerRootId::VulkanVma, vma_blocks, "vulkan_vma");
        std::size_t vulkan_named = 0;
        std::size_t driver_free_in_blocks = 0;
        std::size_t rmlui_texture_bytes = 0;
        std::size_t rmlui_texture_count = 0;
        for (const auto& row : snapshot.rows) {
            if (row.live_bytes == 0 || !is_vulkan_named_row(row) || is_vulkan_external_row(row) ||
                is_shader_bytecode_row(row)) {
                continue;
            }
            if (is_vma_free_sampler_row(row)) {
                driver_free_in_blocks = row.live_bytes;
                add_child(root_f,
                          row.scope + (row.label.empty() ? "" : "." + row.label),
                          row.live_bytes, AttributionState::Nested, row.kind,
                          "VMA free sampler (driver); authority for free_inside_blocks");
                continue;
            }
            // N3: collapse serial-stamped RmlUi textures into one disclosure row.
            if (is_rmlui_texture_row(row)) {
                rmlui_texture_bytes += row.live_bytes;
                ++rmlui_texture_count;
                continue;
            }
            // N2: Sampled named VMA rows justify used-inside-blocks (no Hooked dual path).
            add_child(root_f,
                      row.scope + (row.label.empty() ? "" : "." + row.label),
                      row.live_bytes,
                      AttributionState::Justified,
                      row.kind);
            vulkan_named += row.live_bytes;
        }
        if (rmlui_texture_count > 0) {
            add_child(root_f,
                      std::format("RmlUi textures, {}, {}", rmlui_texture_count,
                                  format_mib(rmlui_texture_bytes)),
                      rmlui_texture_bytes, AttributionState::Justified, VramRowKind::Sampled);
            vulkan_named += rmlui_texture_bytes;
        }
        if (vma_blocks > vulkan_named) {
            const std::size_t residual = vma_blocks - vulkan_named;
            const std::size_t free_inside =
                std::min(residual, driver_free_in_blocks);
            const std::size_t unattributed_in_root = residual - free_inside;
            if (free_inside > 0) {
                add_child(root_f, "free_inside_blocks", free_inside,
                          AttributionState::Justified, VramRowKind::Hooked, "retention");
            }
            if (unattributed_in_root > 0) {
                add_child(root_f, "unattributed_in_root", unattributed_in_root,
                          AttributionState::Unjustified, VramRowKind::Sampled,
                          "incomplete named VMA coverage");
            }
        }
        root_f.attributed_bytes = 0;
        for (const auto& c : root_f.children) {
            if (c.state == AttributionState::Justified) {
                root_f.attributed_bytes += c.measured_bytes;
            }
        }
        apply_closure(root_f, ledger_epsilon(root_f.measured_bytes, policy));

        const std::size_t budget_unattributed_process =
            vulkan_budget_external > vulkan_named_external
                ? vulkan_budget_external - vulkan_named_external
                : 0;
        const std::size_t vulkan_external_coverage =
            std::max(vulkan_budget_external, vulkan_named_external);
        auto root_g =
            make_root(VramLedgerRootId::VulkanExternal, vulkan_external_coverage, "vulkan_raw");
        if (vulkan_named_external > 0) {
            for (const auto& row : snapshot.rows) {
                if (row.live_bytes == 0 || !is_vulkan_external_row(row)) {
                    continue;
                }
                add_child(root_g,
                          row.scope + (row.label.empty() ? "" : "." + row.label),
                          row.live_bytes,
                          AttributionState::Justified,
                          row.kind,
                          "viewer interop external allocation");
            }
        }
        if (budget_unattributed_process > 0) {
            add_child(root_g,
                      "vulkan.memory_budget.unattributed_process_usage",
                      budget_unattributed_process,
                      AttributionState::Unjustified,
                      VramRowKind::Sampled,
                      "whole-process budget remainder overlaps CUDA roots");
        }
        root_g.attributed_bytes = vulkan_named_external;
        apply_closure(root_g, ledger_epsilon(root_g.measured_bytes, policy));

        // Root H: context baseline minus phases that belong to other roots.
        // C3: cuda_phase_default_pool is NOT added here as a justified summand when
        // root A already carries reserved.
        const std::size_t context_baseline = proc.cuda_context_baseline;
        auto root_h =
            make_root(VramLedgerRootId::CudaContextDriver, context_baseline, "cuda_driver");
        if (proc.cuda_phase_primary_context > 0) {
            add_child(root_h, "primary_context", proc.cuda_phase_primary_context,
                      AttributionState::Justified, VramRowKind::Static);
        }
        if (proc.cuda_phase_default_pool > 0) {
            add_child(root_h, "default_pool_at_startup", proc.cuda_phase_default_pool,
                      AttributionState::Nested, VramRowKind::Static,
                      "nested: live pool reserved is root A");
        }
        if (proc.cuda_phase_curand_load > 0) {
            add_child(root_h, "curand_load", proc.cuda_phase_curand_load,
                      AttributionState::Justified, VramRowKind::Static);
        }
        // N1: cuda_warmup_bytes is a device-wide cudaMemGetInfo delta (post-warmup
        // used − device baseline), not a slice of the per-PID context baseline that
        // measures root H. Mark Nested so primary_context (= baseline) can close H.
        if (proc.cuda_warmup_bytes > 0) {
            add_child(root_h, "module_warmup", proc.cuda_warmup_bytes,
                      AttributionState::Nested, VramRowKind::Static,
                      "device-wide warmup delta; not inside context baseline");
        }
        {
            std::size_t shader_bytes = 0;
            std::size_t shader_modules = 0;
            for (const auto& row : snapshot.rows) {
                if (row.live_bytes == 0 || !is_shader_bytecode_row(row)) {
                    continue;
                }
                shader_bytes += row.live_bytes;
                ++shader_modules;
            }
            if (shader_modules > 0) {
                // One grouped row under root H (design: no decision-value per-module flood).
                // Nested: bytecode estimates must not inflate root H attribution.
                add_child(root_h,
                          std::format("shader bytecode, {} modules, {}", shader_modules,
                                      format_mib(shader_bytes)),
                          shader_bytes, AttributionState::Nested, VramRowKind::Static);
            }
        }
        root_h.attributed_bytes = 0;
        for (const auto& c : root_h.children) {
            if (c.state == AttributionState::Justified) {
                root_h.attributed_bytes += c.measured_bytes;
            }
        }
        // If we only have baseline, treat baseline as measured with nested residual.
        if (root_h.attributed_bytes == 0 && context_baseline > 0) {
            root_h.attributed_bytes = context_baseline;
        }
        apply_closure(root_h, ledger_epsilon(root_h.measured_bytes, policy));

        tree.roots.push_back(std::move(root_a));
        tree.roots.push_back(std::move(root_b));
        tree.roots.push_back(std::move(root_c));
        tree.roots.push_back(std::move(root_d));
        tree.roots.push_back(std::move(root_e));
        if (policy.include_vulkan_in_sum) {
            tree.roots.push_back(std::move(root_f));
            tree.roots.push_back(std::move(root_g));
        }
        tree.roots.push_back(std::move(root_h));

        tree.attributed_bytes = 0;
        for (const auto& r : tree.roots) {
            // Sum independent root measurements. Root G's memory-budget coverage is
            // process-wide and overlaps the CUDA roots, so only its named external
            // attribution is independent.
            if (r.root_id != VramLedgerRootId::Unattributed) {
                tree.attributed_bytes += r.root_id == VramLedgerRootId::VulkanExternal
                                             ? r.attributed_bytes
                                             : r.measured_bytes;
            }
        }

        tree.residual = make_signed_residual(tree.attributed_bytes, tree.process_used_bytes);
        if (signed_byte_magnitude(tree.residual.signed_residual_bytes) <= tree.epsilon_bytes) {
            tree.closure = LedgerClosureState::Closed;
        } else if (tree.residual.signed_residual_bytes < 0) {
            tree.closure = LedgerClosureState::Gap;
        } else {
            tree.closure = LedgerClosureState::Over;
        }

        // Always show unattributed residual row (C6: never cap away).
        VramLedgerNode unattr =
            make_root(VramLedgerRootId::Unattributed,
                      tree.residual.under_claim_bytes > 0
                          ? tree.residual.under_claim_bytes
                          : tree.residual.over_claim_bytes,
                      "audit");
        unattr.state = AttributionState::Unjustified;
        if (tree.residual.over_claim_bytes > 0) {
            unattr.note = "over-attributed";
            unattr.closure = LedgerClosureState::Over;
        } else if (tree.residual.under_claim_bytes > 0) {
            unattr.note = "honest gap";
            unattr.closure = LedgerClosureState::Gap;
        } else {
            unattr.note = "closes";
            unattr.closure = LedgerClosureState::Closed;
        }
        unattr.residual_bytes = tree.residual.signed_residual_bytes;
        tree.roots.push_back(std::move(unattr));

        return tree;
    }

    PeakExCacheLedger buildPeakExCacheLedger(const PeakExCacheInputs& in) {
        PeakExCacheLedger out;
        out.peak_cuda_used_bytes = in.peak_cuda_used_bytes;
        out.baseline_cuda_used_bytes = in.baseline_cuda_used_bytes;
        out.baseline_ex_cache_bytes = in.baseline_ex_cache_bytes;
        out.training_state_bytes = in.training_state_bytes;
        out.training_state_reserved_bytes = in.training_state_reserved_bytes;
        out.training_state_baseline_bytes = in.training_state_baseline_bytes;
        out.loss_workspace_required_bytes = in.loss_workspace_required_bytes;
        out.loss_workspace_allocated_bytes = in.loss_workspace_allocated_bytes;
        out.densify_workspace_bytes = in.densify_workspace_bytes;
        out.mrnf_strategy_required_bytes = in.mrnf_strategy_required_bytes;
        out.mrnf_strategy_allocated_bytes = in.mrnf_strategy_allocated_bytes;
        out.mrnf_densify_n_required_bytes = in.mrnf_densify_n_required_bytes;
        out.mrnf_densify_n_allocated_bytes = in.mrnf_densify_n_allocated_bytes;
        out.mrnf_densify_child_required_bytes = in.mrnf_densify_child_required_bytes;
        out.mrnf_densify_child_allocated_bytes = in.mrnf_densify_child_allocated_bytes;
        out.mrnf_refine_peak_required_bytes = in.mrnf_refine_peak_required_bytes;
        out.mrnf_refine_peak_allocated_bytes = in.mrnf_refine_peak_allocated_bytes;
        out.mrnf_grow_peak_required_bytes = in.mrnf_grow_peak_required_bytes;
        out.mrnf_grow_peak_allocated_bytes = in.mrnf_grow_peak_allocated_bytes;
        out.pool_bucket_cache_bytes = in.pool_bucket_cache_bytes;
        out.pool_bucket_live_rounding_waste_bytes = in.pool_bucket_live_rounding_waste_bytes;
        out.exportable_splat_bytes = in.exportable_splat_bytes;
        out.fastgs_sort_required_bytes = in.fastgs_sort_required_bytes;
        out.fastgs_sort_allocated_bytes = in.fastgs_sort_allocated_bytes;
        out.fastgs_raster_live_bytes = in.fastgs_raster_live_bytes;
        out.fastgs_raster_arena_live_bytes = in.fastgs_raster_arena_live_bytes;
        out.fastgs_raster_sort_live_bytes = in.fastgs_raster_sort_live_bytes;
        out.arena_required_bytes = in.arena_required_bytes;
        out.arena_capacity_bytes = in.arena_capacity_bytes;
        out.ex_cache_bytes = in.peak_cuda_used_bytes;

        const std::size_t peak_above_baseline =
            in.peak_cuda_used_bytes > in.baseline_cuda_used_bytes
                ? in.peak_cuda_used_bytes - in.baseline_cuda_used_bytes
                : 0;
        out.ex_cache_net_bytes = peak_above_baseline;
        out.excess_over_baseline_bytes =
            out.ex_cache_net_bytes > out.baseline_ex_cache_bytes
                ? out.ex_cache_net_bytes - out.baseline_ex_cache_bytes
                : 0;

        auto add = [&](const char* name,
                       const char* owner,
                       const std::size_t bytes,
                       const AttributionState state,
                       const char* note = "") {
            if (bytes == 0) {
                return;
            }
            PeakSubsystemLine line;
            line.name = name;
            line.owner = owner;
            line.bytes = bytes;
            line.state = state;
            line.note = note;
            out.lines.push_back(std::move(line));
            if (state == AttributionState::Justified) {
                out.justified_excess_bytes += bytes;
            }
        };

        const std::size_t loss_workspace_slack =
            in.loss_workspace_allocated_bytes > in.loss_workspace_required_bytes
                ? in.loss_workspace_allocated_bytes - in.loss_workspace_required_bytes
                : 0;
        const std::size_t capacity_overhead =
            in.training_state_reserved_bytes > in.training_state_bytes
                ? in.training_state_reserved_bytes - in.training_state_bytes
                : 0;
        const std::size_t training_state_growth =
            in.training_state_bytes > in.training_state_baseline_bytes
                ? in.training_state_bytes - in.training_state_baseline_bytes
                : 0;
        const std::size_t training_state_baseline_inventory =
            std::min(in.training_state_bytes, in.training_state_baseline_bytes);
        out.training_state_growth_bytes = training_state_growth;
        const std::size_t fastgs_sort_slack =
            in.fastgs_sort_allocated_bytes > in.fastgs_sort_required_bytes
                ? in.fastgs_sort_allocated_bytes - in.fastgs_sort_required_bytes
                : 0;

        add("baseline_cuda_context", "desktop+ctx", in.baseline_cuda_used_bytes,
            AttributionState::Nested,
            "subtracted before process-net excess; not justified cover");
        add("training_state", "optimizer", in.training_state_bytes, AttributionState::Nested,
            "aggregate inventory; baseline, growth, and capacity are disclosed below");
        add("training_state_baseline", "optimizer", training_state_baseline_inventory,
            AttributionState::Nested,
            "canonical 1.5M inventory already included in the ex-cache baseline");
        add("training_state_growth", "optimizer", training_state_growth,
            AttributionState::Justified,
            "logical inventory above the canonical 1.5M baseline");
        add("training_state_capacity_overhead", "capacity", capacity_overhead,
            AttributionState::Justified);
        add("loss_workspace_required", "loss_workspace", in.loss_workspace_required_bytes,
            AttributionState::Justified);
        add("loss_workspace_arena", "loss_workspace", in.loss_workspace_allocated_bytes,
            AttributionState::Nested,
            "aggregate of loss_workspace_required and loss_workspace_slack");
        add("loss_workspace_slack", "loss_workspace", loss_workspace_slack,
            AttributionState::Justified);
        const bool has_named_mrnf_densify =
            in.mrnf_densify_n_allocated_bytes > 0 ||
            in.mrnf_densify_child_allocated_bytes > 0;
        add("densify_workspace_legacy", "densification",
            has_named_mrnf_densify ? 0 : in.densify_workspace_bytes,
            AttributionState::Justified,
            "legacy aggregate used only when named densify owners are absent");

        const auto add_required_allocated = [&](const char* prefix,
                                                const char* owner,
                                                const std::size_t required,
                                                const std::size_t allocated) {
            const std::string required_name = std::string(prefix) + "_required";
            const std::string allocated_name = std::string(prefix) + "_allocated";
            const std::string slack_name = std::string(prefix) + "_slack";
            add(required_name.c_str(), owner, required, AttributionState::Nested,
                "logical requirement; disclosed inside allocated backing");
            add(allocated_name.c_str(), owner, allocated, AttributionState::Justified);
            add(slack_name.c_str(), owner,
                allocated > required ? allocated - required : 0,
                AttributionState::Nested,
                "disclosure component already included in allocated backing");
        };
        add_required_allocated("mrnf_strategy", "MRNF",
                               in.mrnf_strategy_required_bytes,
                               in.mrnf_strategy_allocated_bytes);
        add_required_allocated("mrnf_densify_n", "MRNF",
                               in.mrnf_densify_n_required_bytes,
                               in.mrnf_densify_n_allocated_bytes);
        add_required_allocated("mrnf_densify_child", "MRNF",
                               in.mrnf_densify_child_required_bytes,
                               in.mrnf_densify_child_allocated_bytes);
        add_required_allocated("mrnf_refine_peak", "MRNF",
                               in.mrnf_refine_peak_required_bytes,
                               in.mrnf_refine_peak_allocated_bytes);
        add_required_allocated("mrnf_grow_peak_exclusive", "MRNF",
                               in.mrnf_grow_peak_required_bytes,
                               in.mrnf_grow_peak_allocated_bytes);
        add("pool_bucket_cache", "allocator", in.pool_bucket_cache_bytes,
            AttributionState::Justified);
        add("pool_bucket_live_rounding_waste", "allocator",
            in.pool_bucket_live_rounding_waste_bytes, AttributionState::Justified);
        add("exportable_splat", "viewport", in.exportable_splat_bytes,
            AttributionState::Justified);
        add("fastgs_sort_required", "fastgs_sort", in.fastgs_sort_required_bytes,
            AttributionState::Nested, "disclosure component of fastgs_sort_allocated");
        add("fastgs_sort_allocated", "fastgs_sort", in.fastgs_sort_allocated_bytes,
            AttributionState::Justified);
        add("fastgs_sort_slack", "fastgs_sort", fastgs_sort_slack, AttributionState::Nested,
            "disclosure component already included in fastgs_sort_allocated");
        add("fastgs_raster_live", "FastGS", in.fastgs_raster_live_bytes,
            AttributionState::Nested, "aggregate of raster arena live and sort workspace");
        add("fastgs_raster_arena_live", "FastGS", in.fastgs_raster_arena_live_bytes,
            AttributionState::Justified);
        add("fastgs_raster_sort_live", "FastGS", in.fastgs_raster_sort_live_bytes,
            AttributionState::Nested,
            "selected sort slice already included in fastgs_sort_allocated");
        add("rasterizer_arena", "rasterizer", in.arena_capacity_bytes, AttributionState::Nested,
            "backing container; live FastGS arena bytes are covered separately");
        add("rasterizer_arena_required", "rasterizer", in.arena_required_bytes,
            AttributionState::Nested, "required portion of the rasterizer arena container");
        add("io.decoded_frame_ring", "image_loader", in.peak_io_ring_bytes,
            AttributionState::Nested, "already included in io.external_codec_and_bucketed");
        add("io.external_codec_and_bucketed", "image_loader", in.peak_io_external_bytes,
            AttributionState::Justified);
        add("pinned_host_active_cached_steady", "pinned_allocator",
            in.peak_steady_pinned_host_bytes, AttributionState::Unjustified,
            "host memory; visible inventory outside the VRAM cover sum");

        const std::size_t new_justified = out.justified_excess_bytes;
        const auto residual =
            make_signed_residual(new_justified, out.excess_over_baseline_bytes);
        out.signed_residual_bytes = residual.signed_residual_bytes;
        out.unjustified_excess_bytes = residual.under_claim_bytes;
        out.over_attributed_bytes = residual.over_claim_bytes;
        if (out.signed_residual_bytes != 0) {
            add("signed_residual", "audit", signed_byte_magnitude(out.signed_residual_bytes),
                AttributionState::Unjustified,
                "magnitude only; signed value is in signed_residual_bytes");
        }
        out.justified_excess_bytes = new_justified;
        return out;
    }

} // namespace lfs::diagnostics

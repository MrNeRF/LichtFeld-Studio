/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "diagnostics/vram_owner_model.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <string_view>

namespace lfs::diagnostics {
    namespace {
        [[nodiscard]] bool prefix(std::string_view text, std::string_view p) {
            return text == p || (text.starts_with(p) && text.size() > p.size() &&
                                 (text[p.size()] == '.' || text[p.size()] == '/'));
        }

        [[nodiscard]] bool component(std::string_view text, std::string_view name) {
            std::size_t pos = 0;
            while ((pos = text.find(name, pos)) != std::string_view::npos) {
                const auto before = pos == 0 || text[pos - 1] == '.' || text[pos - 1] == '/';
                const auto end = pos + name.size();
                const auto after = end == text.size() || text[end] == '.' || text[end] == '/';
                if (before && after)
                    return true;
                pos = end;
            }
            return false;
        }

        [[nodiscard]] std::size_t index(VramOwner owner) {
            return static_cast<std::size_t>(owner);
        }

        [[nodiscard]] std::size_t capped(std::size_t requested, std::size_t& remaining) {
            const auto result = std::min(requested, remaining);
            remaining -= result;
            return result;
        }

        [[nodiscard]] std::size_t gauge(const VramProfilerSnapshot& s, std::string_view key) {
            for (const auto& g : s.gauges) {
                if (g.key == key && std::isfinite(g.value) && g.value > 0.0) {
                    return static_cast<std::size_t>(std::min(
                        g.value, static_cast<double>(std::numeric_limits<std::size_t>::max())));
                }
            }
            return 0;
        }

        [[nodiscard]] VramOwner classify(const VramMetricSnapshot& row) {
            const auto s = std::string_view(row.scope);
            const auto label = std::string_view(row.label);
            if (prefix(s, "model.gaussians") &&
                (prefix(label, "densification_info") || prefix(label, "deleted")))
                return VramOwner::Densification;
            if (component(s, "optimizer"))
                return VramOwner::Optimizer;
            if (component(s, "model"))
                return VramOwner::Model;
            if (component(s, "MRNF") || component(s, "strategy") ||
                (prefix(s, "train.losses") && label.starts_with("densification_")))
                return VramOwner::Densification;
            if (component(s, "rasterizer") || s.find("rasterize_forward") != std::string_view::npos ||
                s.find("rasterize_backward") != std::string_view::npos || prefix(s, "shared.scratch"))
                return VramOwner::Rasterizer;
            if (component(s, "io") || s.starts_with("COLMAP "))
                return VramOwner::ImageIo;
            if (prefix(s, "vulkan") || s.starts_with("vksplat") || prefix(s, "viewer"))
                return VramOwner::Viewer;
            if (component(s, "train") || component(s, "loss") ||
                (s == "unscoped" && label.starts_with("training.snapshot.")))
                return VramOwner::LossStep;
            return VramOwner::Unattributed;
        }

        void add(VramOwnerBreakdown& out, VramOwner owner, std::size_t bytes,
                 const VramMetricSnapshot* row = nullptr) {
            out.bytes[index(owner)] += static_cast<std::int64_t>(bytes);
            if (row && bytes)
                out.rows.push_back({row->scope, row->label, owner, bytes, row->peak_bytes});
        }

        [[nodiscard]] bool hooked(const VramMetricSnapshot& row,
                                  VramAllocationMethod a, VramAllocationMethod b = VramAllocationMethod::Unknown) {
            return row.kind == VramRowKind::Hooked && (row.method == a || row.method == b);
        }

        void consume_rows(VramOwnerBreakdown& out, const VramProfilerSnapshot& s,
                          std::size_t& remaining, VramAllocationMethod a,
                          VramAllocationMethod b = VramAllocationMethod::Unknown) {
            for (const auto& row : s.rows) {
                if (hooked(row, a, b) && row.live_bytes)
                    add(out, classify(row), capped(row.live_bytes, remaining), &row);
            }
        }
    } // namespace

    std::int64_t VramOwnerBreakdown::sum() const noexcept {
        return std::accumulate(bytes.begin(), bytes.end(), std::int64_t{0});
    }

    const char* vramOwnerName(VramOwner owner) noexcept {
        switch (owner) {
        case VramOwner::Model: return "Model parameters";
        case VramOwner::Optimizer: return "Optimizer and gradients";
        case VramOwner::Rasterizer: return "Rasterizer";
        case VramOwner::LossStep: return "Loss and step buffers";
        case VramOwner::Densification: return "Densification";
        case VramOwner::ImageIo: return "Image I/O and decode";
        case VramOwner::Viewer: return "Viewer";
        case VramOwner::Slack: return "Allocator slack";
        case VramOwner::Context: return "CUDA context and driver";
        case VramOwner::Unattributed: return "Unattributed";
        case VramOwner::Count: break;
        }
        return "";
    }

    VramOwnerBreakdown buildVramOwnerBreakdown(const VramProfilerSnapshot& s,
                                               bool arena_external_backing) {
        VramOwnerBreakdown out;
        out.process_valid = s.process.process_memory_valid;
        out.process_bytes = s.process.process_used;
        const auto& p = s.process;
        const bool shared_arena_alias = arena_external_backing &&
                                        std::ranges::any_of(s.rows, [](const auto& row) {
                                            return row.scope == "shared.scratch" &&
                                                   row.label.starts_with("cuda_vulkan_arena") &&
                                                   row.live_bytes > 0;
                                        });

        if (p.cuda_pool_valid) {
            std::size_t remaining = p.cuda_pool_reserved;
            consume_rows(out, s, remaining, VramAllocationMethod::Bucketed, VramAllocationMethod::Async);
            for (const auto& row : s.rows) {
                if (row.scope == "io.nvimagecodec" && row.label == "default_pool" &&
                    row.kind == VramRowKind::Static)
                    add(out, VramOwner::ImageIo, capped(row.live_bytes, remaining), &row);
            }
            // Tensor and pipeline disclosures with an unknown allocator method may
            // occupy the unclaimed part of the CUDA pool. Keep the pool budget as
            // the authority: these rows never increase the reservation total.
            for (const auto& row : s.rows) {
                if (row.kind != VramRowKind::Sampled ||
                    row.method != VramAllocationMethod::Unknown || !row.live_bytes)
                    continue;
                if (prefix(row.scope, "optimizer") || prefix(row.scope, "train") ||
                    prefix(row.scope, "io.pipeline"))
                    add(out, classify(row), capped(row.live_bytes, remaining), &row);
            }
            for (const auto& row : s.rows) {
                if (row.kind == VramRowKind::Sampled && row.method == VramAllocationMethod::External &&
                    (prefix(row.scope, "io.pipeline") ||
                     (row.scope == "train.losses" && row.label == "loss_workspace_arena") ||
                     (p.exportable_splat_bytes && row.scope == "model.gaussians" &&
                      row.label == "densification_info")))
                    add(out, classify(row), capped(row.live_bytes, remaining), &row);
            }
            add(out, VramOwner::Slack, remaining);
        }

        {
            std::size_t remaining = p.cuda_slab_reserved_bytes;
            const auto gap = remaining > s.accounted_slab_live_bytes
                                 ? remaining - s.accounted_slab_live_bytes
                                 : 0;
            add(out, VramOwner::Slack, capped(gap, remaining));
            consume_rows(out, s, remaining, VramAllocationMethod::Slab);
            out.unattributed_roots[0] = static_cast<std::int64_t>(remaining);
            add(out, VramOwner::Unattributed, remaining);
        }

        {
            std::size_t remaining = s.accounted_direct_live_bytes;
            consume_rows(out, s, remaining, VramAllocationMethod::Direct);
            out.unattributed_roots[1] = static_cast<std::int64_t>(remaining);
            add(out, VramOwner::Unattributed, remaining);
            for (const auto& row : s.rows) {
                if (row.scope == "io.nvimagecodec" && row.label == "driver_or_direct" &&
                    row.kind == VramRowKind::Static)
                    add(out, VramOwner::ImageIo, row.live_bytes, &row);
            }
        }

        {
            std::size_t remaining = gauge(s, "vram.audit.tensor.cuda_direct_live_bytes");
            for (const auto owner : {VramOwner::Optimizer, VramOwner::Model,
                                     VramOwner::Densification, VramOwner::Rasterizer,
                                     VramOwner::LossStep, VramOwner::ImageIo}) {
                for (const auto& row : s.rows) {
                    if (row.kind == VramRowKind::Sampled &&
                        (row.method == VramAllocationMethod::Direct ||
                         (!p.exportable_splat_bytes && row.method == VramAllocationMethod::External &&
                          prefix(row.scope, "model.gaussians"))) &&
                        classify(row) == owner)
                        add(out, owner, capped(row.live_bytes, remaining), &row);
                }
            }
            for (const std::string_view name : {
                     "refine_weight_max", "refine_ratio_max", "vis_count", "free_mask",
                     "refine_counts_device", "edge.precomputed_scores", "edge.score_sum",
                     "edge.view_scores", "explore.score_sum", "explore.error_hw",
                     "explore.view_scores", "explore.means2d", "explore.radii",
                     "explore.far_field_mask", "explore.cached_image", "explore.cached_target",
                     "explore.cached_alpha", "explore.cached_depth"}) {
                const auto key = "vram.audit.mrnf." + std::string(name) + ".allocated_bytes";
                add(out, VramOwner::Densification, capped(gauge(s, key), remaining));
            }
            out.unattributed_roots[2] = static_cast<std::int64_t>(remaining);
            add(out, VramOwner::Unattributed, remaining);
        }

        if (!shared_arena_alias) {
            std::size_t remaining = s.accounted_arena_live_bytes;
            for (const auto& row : s.rows) {
                if (row.scope == "rasterizer.fastgs" && row.label == "arena.capacity" &&
                    row.kind == VramRowKind::Sampled)
                    remaining = std::max(remaining, row.live_bytes);
            }
            consume_rows(out, s, remaining, VramAllocationMethod::Arena);
            add(out, VramOwner::Rasterizer, remaining);
        }

        {
            std::size_t remaining = p.exportable_splat_bytes;
            for (const auto& row : s.rows) {
                if (prefix(row.scope, "model.gaussians") && row.kind == VramRowKind::Sampled &&
                    row.label != "densification_info")
                    add(out, classify(row), capped(row.live_bytes, remaining), &row);
            }
            add(out, VramOwner::Slack, remaining);
        }

        {
            std::size_t remaining = p.shared_scratch_bytes;
            for (const auto& row : s.rows) {
                if (prefix(row.scope, "shared.scratch") && row.kind == VramRowKind::Sampled &&
                    row.label.find("cuda_vulkan_arena") == std::string::npos &&
                    row.label != "reserve_unbound")
                    add(out, classify(row), capped(row.live_bytes, remaining), &row);
            }
            add(out, VramOwner::Rasterizer, remaining);
        }

        {
            std::size_t remaining = p.vulkan_vma_block_bytes;
            for (const auto& row : s.rows) {
                if (row.live_bytes == 0 || row.scope.starts_with("vulkan.external") ||
                    row.scope.starts_with("vksplat.shaders") ||
                    (row.scope == "vulkan.vma" && row.label.starts_with("free")) ||
                    (row.scope == "vulkan.vma" && row.label == "allocator_free_in_blocks"))
                    continue;
                if (prefix(row.scope, "vulkan") || row.scope.starts_with("vksplat"))
                    add(out, VramOwner::Viewer, capped(row.live_bytes, remaining), &row);
            }
            for (const auto& row : s.rows) {
                if (row.scope == "vulkan.vma" && row.label == "allocator_free_in_blocks")
                    add(out, VramOwner::Slack, capped(row.live_bytes, remaining));
            }
            out.unattributed_roots[3] = static_cast<std::int64_t>(remaining);
            add(out, VramOwner::Unattributed, remaining);
        }

        for (const auto& row : s.rows) {
            if (!row.scope.starts_with("vulkan.external") || row.live_bytes == 0 ||
                row.scope.starts_with("vulkan.external.imported") ||
                row.scope.starts_with("vulkan.external_tensor") ||
                row.scope == "vulkan.external.semaphore")
                continue;
            add(out, VramOwner::Viewer, row.live_bytes, &row);
        }

        std::size_t context_bytes = p.cuda_context_baseline;
        if (out.process_valid) {
            std::int64_t known = 0;
            for (std::size_t i = 0; i < index(VramOwner::Context); ++i)
                known += out.bytes[i];
            const auto remaining = known < static_cast<std::int64_t>(p.process_used)
                                       ? static_cast<std::size_t>(static_cast<std::int64_t>(p.process_used) - known)
                                       : 0;
            context_bytes = std::min(context_bytes, remaining);
        }
        add(out, VramOwner::Context, context_bytes);
        if (out.process_valid) {
            std::int64_t attributed = 0;
            for (std::size_t i = 0; i < kVramOwnerCount; ++i)
                attributed += out.bytes[i];
            const auto balance = static_cast<std::int64_t>(p.process_used) - attributed;
            if (balance > 0 && p.cuda_context_baseline > 0 && p.cuda_pool_valid) {
                out.context_inferred_bytes = balance;
                add(out, VramOwner::Context, static_cast<std::size_t>(balance));
                attributed += balance;
            }
            out.signed_residual_bytes = static_cast<std::int64_t>(p.process_used) - attributed;
            out.unattributed_roots[4] = out.signed_residual_bytes;
            out.bytes[index(VramOwner::Unattributed)] += out.signed_residual_bytes;
            out.over_attributed = out.signed_residual_bytes < 0;
        }
        return out;
    }

    std::array<std::int64_t, kVramOwnerCount>
    displayVramOwnerBytes(const VramOwnerBreakdown& breakdown) noexcept {
        auto result = breakdown.bytes;
        if (!breakdown.process_valid)
            return result;
        auto remaining = static_cast<std::int64_t>(breakdown.process_bytes);
        for (std::size_t i = 0; i < index(VramOwner::Unattributed); ++i) {
            result[i] = std::clamp(result[i], std::int64_t{0}, remaining);
            remaining -= result[i];
        }
        result[index(VramOwner::Unattributed)] = remaining;
        return result;
    }
} // namespace lfs::diagnostics

/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/vram_hud_overlay.hpp"

#include "gui/layout_state.hpp"

#include <RmlUi/Core/Context.h>
#include <RmlUi/Core/Element.h>
#include <RmlUi/Core/ElementDocument.h>
#include <RmlUi/Core/Elements/ElementFormControlInput.h>
#include <RmlUi/Core/Event.h>
#include <RmlUi/Core/ID.h>
#include <RmlUi/Core/Types.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <format>
#include <limits>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace lfs::vis::gui {

    namespace {

        constexpr int kRowIndentPx = 10;
        constexpr std::size_t kDefaultCollapseDepth = 2;
        constexpr std::size_t kMaxAnnotationRows = 512;
        constexpr float kMinHudWidthPx = 360.0f;
        constexpr float kMinHudHeightPx = 200.0f;
        constexpr float kHudViewportPaddingPx = 16.0f;

        struct SummaryRowSpec {
            std::string_view key;
            std::string_view label;
        };

        constexpr SummaryRowSpec kSummaryRows[] = {
            {"process", "Process"},
            {"cuda_context", "GPU used (all procs)"},
            {"cuda_pool_used", "CUDA pool used"},
            {"cuda_pool_reserved", "CUDA pool reserved"},
            {"cuda_pool_fragmentation", "Pool fragmentation"},
            {"pinned_host", "Pinned host"},
            {"vulkan_budget", "Vulkan budget (raw)"},
            {"vulkan_blocks", "Vulkan VMA blocks"},
            {"sampled", "Sampled subtotal (raw)"},
            {"allocator_live", "Allocator live"},
            {"process_gap", "Raw sampled gap"},
            {"allocator_peak", "Allocator peak"},
            {"events", "Events"},
            {"iter_events", "Events (iter)"},
        };

        void escapeRmlInto(std::string& out, std::string_view text) {
            out.reserve(out.size() + text.size());
            for (const char c : text) {
                switch (c) {
                case '&': out += "&amp;"; break;
                case '<': out += "&lt;"; break;
                case '>': out += "&gt;"; break;
                case '"': out += "&quot;"; break;
                case '\'': out += "&#39;"; break;
                default: out.push_back(c); break;
                }
            }
        }

        [[nodiscard]] std::string formatBytes(std::size_t bytes) {
            constexpr double kKiB = 1024.0;
            constexpr double kMiB = 1024.0 * kKiB;
            constexpr double kGiB = 1024.0 * kMiB;
            const double v = static_cast<double>(bytes);
            if (v >= kGiB)
                return std::format("{:.2f} GiB", v / kGiB);
            if (v >= kMiB)
                return std::format("{:.1f} MiB", v / kMiB);
            if (v >= kKiB)
                return std::format("{:.1f} KiB", v / kKiB);
            return std::format("{} B", bytes);
        }

        [[nodiscard]] std::string formatSignedBytes(std::int64_t bytes) {
            if (bytes == 0)
                return "0 B";
            const auto magnitude = static_cast<std::size_t>(std::llabs(bytes));
            return std::format("{}{}", bytes > 0 ? "+" : "-", formatBytes(magnitude));
        }

        [[nodiscard]] std::string formatTime(double ms) {
            if (ms <= 0.0)
                return "--";
            if (ms < 0.01)
                return std::format("{:.1f} us", ms * 1000.0);
            if (ms < 1.0)
                return std::format("{:.2f} ms", ms);
            if (ms < 100.0)
                return std::format("{:.1f} ms", ms);
            return std::format("{:.0f} ms", ms);
        }

        [[nodiscard]] std::string formatPercent(std::size_t part, std::size_t total) {
            if (part == 0 || total == 0)
                return {};
            return std::format("{:.1f}%", 100.0 * static_cast<double>(part) / static_cast<double>(total));
        }

        [[nodiscard]] std::size_t bestProcessTotal(const lfs::diagnostics::VramProfilerSnapshot& s) {
            if (s.process.process_memory_valid && s.process.total > 0)
                return s.process.total;
            if (s.process.cuda_memory_valid && s.process.cuda_total > 0)
                return s.process.cuda_total;
            return 0;
        }

        [[nodiscard]] std::size_t bestProcessUsed(const lfs::diagnostics::VramProfilerSnapshot& s) {
            if (s.process.process_memory_valid && s.process.process_used > 0)
                return s.process.process_used;
            if (s.process.cuda_memory_valid && s.process.cuda_used > 0)
                return s.process.cuda_used;
            return 0;
        }

        void setText(Rml::Element* el, std::string& cache, std::string&& value) {
            if (!el || cache == value)
                return;
            cache = std::move(value);
            std::string rml;
            escapeRmlInto(rml, cache);
            el->SetInnerRML(Rml::String(rml));
        }

        void setRawRml(Rml::Element* el, std::string& cache, std::string&& value) {
            if (!el || cache == value)
                return;
            cache = std::move(value);
            el->SetInnerRML(Rml::String(cache));
        }

        void applyRowClasses(Rml::Element* el, std::string& cache, std::string&& classes) {
            if (cache == classes)
                return;
            cache = std::move(classes);
            el->SetAttribute("class", Rml::String(cache));
        }

        [[nodiscard]] std::string buildSummaryRowValueRml(std::string_view value, std::string_view extra) {
            std::string rml;
            rml.reserve(value.size() + extra.size() + 2);
            escapeRmlInto(rml, value);
            if (!extra.empty()) {
                rml.push_back(' ');
                escapeRmlInto(rml, extra);
            }
            return rml;
        }

        Rml::Element* createSpan(Rml::ElementDocument* doc, Rml::Element* parent,
                                 std::string_view class_name) {
            auto element_ptr = doc->CreateElement("span");
            element_ptr->SetAttribute("class", Rml::String(class_name));
            return parent->AppendChild(std::move(element_ptr));
        }

        [[nodiscard]] std::string toLowerAscii(std::string_view in) {
            std::string out;
            out.reserve(in.size());
            for (const char c : in)
                out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
            return out;
        }

        [[nodiscard]] std::string_view lastTrainScopeComponent(std::string_view scope) {
            std::string_view best;
            std::size_t pos = 0;
            while (pos < scope.size()) {
                const std::size_t end = scope.find('/', pos);
                const auto part = scope.substr(pos, end == std::string_view::npos ? scope.size() - pos : end - pos);
                if (part.rfind("train.", 0) == 0 || part == "train") {
                    best = part;
                }
                if (end == std::string_view::npos) {
                    break;
                }
                pos = end + 1;
            }
            return best.empty() ? scope : best;
        }

        [[nodiscard]] std::string firstScopeSegments(std::string_view scope, const std::size_t count) {
            if (count == 0) {
                return {};
            }

            std::size_t segments = 1;
            for (std::size_t i = 0; i < scope.size(); ++i) {
                if (scope[i] != '.' && scope[i] != '/') {
                    continue;
                }
                if (segments == count) {
                    return std::string(scope.substr(0, i));
                }
                ++segments;
            }
            return std::string(scope);
        }

        [[nodiscard]] std::string topSegment(std::string_view scope) {
            const auto end = scope.find_first_of("/.");
            return std::string(end == std::string_view::npos ? scope : scope.substr(0, end));
        }

        [[nodiscard]] std::string firstDotSegments(std::string_view scope, const std::size_t count) {
            std::size_t pos = 0;
            for (std::size_t i = 0; i < count; ++i) {
                pos = scope.find('.', pos);
                if (pos == std::string_view::npos)
                    return std::string(scope);
                ++pos;
            }
            return std::string(scope.substr(0, pos - 1));
        }

        [[nodiscard]] bool isFastGsLogicalRow(const lfs::diagnostics::VramMetricSnapshot& row) {
            return row.scope.rfind("rasterizer.fastgs", 0) == 0;
        }

        [[nodiscard]] std::string breakdownLabel(const lfs::diagnostics::VramMetricSnapshot& row) {
            const std::string_view scope = row.scope;
            const std::string_view label = row.label;
            if (label.rfind("rasterizer.arena.", 0) == 0) {
                return "rasterizer.arena";
            }
            if (scope.rfind("io.", 0) == 0) {
                const auto io_group = firstDotSegments(scope, 2);
                if (io_group == "io.pipeline" && !row.label.empty()) {
                    return io_group + "." + row.label;
                }
                return io_group;
            }
            if (scope.rfind("vulkan.", 0) == 0) {
                return firstDotSegments(scope, 3);
            }
            if (scope == "shared.scratch" && !label.empty()) {
                return "shared.scratch." + std::string(label);
            }
            if (scope.rfind("train.", 0) == 0 || scope.find("/train.") != std::string_view::npos) {
                return firstScopeSegments(lastTrainScopeComponent(scope), 2);
            }
            return topSegment(scope);
        }

        struct VramBreakdownEntry {
            std::string label;
            std::size_t bytes = 0;
            bool unaccounted = false;
            bool total = false;
        };

        struct VramBreakdownAudit {
            std::size_t process_used = 0;
            std::size_t raw_sampled_subtotal = 0;
            std::size_t overview_named_rows = 0;
            std::size_t fastgs_logical_not_totaled = 0;
            std::size_t named_slab_rows = 0;
            std::size_t named_bucketed_rows = 0;
            std::size_t named_async_rows = 0;
            std::size_t named_direct_rows = 0;
            std::size_t named_arena_rows = 0;
            std::size_t named_external_rows = 0;
            std::size_t named_unknown_rows = 0;
            std::size_t cuda_pool_used = 0;
            std::size_t cuda_pool_reserved = 0;
            std::size_t cuda_pool_accounted_live = 0;
            std::size_t cuda_slab_reserved = 0;
            std::size_t cuda_slab_reserve_gap = 0;
            std::size_t accounted_slab_live = 0;
            std::size_t accounted_bucketed_live = 0;
            std::size_t accounted_async_live = 0;
            std::size_t accounted_direct_live = 0;
            std::size_t accounted_arena_live = 0;
            std::size_t accounted_external_live = 0;
            std::size_t accounted_unknown_live = 0;
            std::size_t cuda_pool_bucket_cache = 0;
            std::size_t cuda_pool_untracked_used = 0;
            std::size_t cuda_pool_overhead = 0;
            std::size_t cuda_context_baseline = 0;
            std::size_t cuda_context_phases = 0;
            std::size_t cuda_context_residual = 0;
            std::size_t vulkan_vma_blocks = 0;
            std::size_t vulkan_named_rows = 0;
            std::size_t vulkan_residual = 0;
            std::size_t breakdown_totaled = 0;
            std::size_t process_unresolved = 0;
        };

        struct VramBreakdownModel {
            std::vector<VramBreakdownEntry> entries;
            std::size_t denom = 0;
            VramBreakdownAudit audit;
        };

        [[nodiscard]] VramBreakdownModel buildBreakdownModel(
            const lfs::diagnostics::VramProfilerSnapshot& snapshot,
            std::size_t process_used) {
            VramBreakdownModel model;
            model.audit.process_used = process_used;
            model.audit.raw_sampled_subtotal = snapshot.sampled_live_bytes;
            model.audit.cuda_pool_used =
                snapshot.process.cuda_pool_valid ? snapshot.process.cuda_pool_used : 0;
            model.audit.cuda_pool_reserved =
                snapshot.process.cuda_pool_valid ? snapshot.process.cuda_pool_reserved : 0;
            model.audit.cuda_pool_accounted_live = snapshot.accounted_cuda_pool_live_bytes;
            model.audit.cuda_slab_reserved = snapshot.process.cuda_slab_reserved_bytes;
            model.audit.accounted_slab_live = snapshot.accounted_slab_live_bytes;
            model.audit.accounted_bucketed_live = snapshot.accounted_bucketed_live_bytes;
            model.audit.accounted_async_live = snapshot.accounted_async_live_bytes;
            model.audit.accounted_direct_live = snapshot.accounted_direct_live_bytes;
            model.audit.accounted_arena_live = snapshot.accounted_arena_live_bytes;
            model.audit.accounted_external_live = snapshot.accounted_external_live_bytes;
            model.audit.accounted_unknown_live = snapshot.accounted_unknown_live_bytes;
            model.audit.cuda_context_baseline = snapshot.process.cuda_context_baseline;
            model.audit.vulkan_vma_blocks = snapshot.process.vulkan_vma_block_bytes;

            const auto add_named_method = [&](const lfs::diagnostics::VramAllocationMethod method,
                                              const std::size_t bytes) {
                switch (method) {
                case lfs::diagnostics::VramAllocationMethod::Slab:
                    model.audit.named_slab_rows += bytes;
                    break;
                case lfs::diagnostics::VramAllocationMethod::Bucketed:
                    model.audit.named_bucketed_rows += bytes;
                    break;
                case lfs::diagnostics::VramAllocationMethod::Async:
                    model.audit.named_async_rows += bytes;
                    break;
                case lfs::diagnostics::VramAllocationMethod::Direct:
                    model.audit.named_direct_rows += bytes;
                    break;
                case lfs::diagnostics::VramAllocationMethod::Arena:
                    model.audit.named_arena_rows += bytes;
                    break;
                case lfs::diagnostics::VramAllocationMethod::External:
                    model.audit.named_external_rows += bytes;
                    break;
                case lfs::diagnostics::VramAllocationMethod::Unknown:
                default:
                    model.audit.named_unknown_rows += bytes;
                    break;
                }
            };

            std::unordered_map<std::string, std::size_t> groups;
            for (const auto& row : snapshot.rows) {
                if (row.live_bytes == 0)
                    continue;
                if (isFastGsLogicalRow(row)) {
                    model.audit.fastgs_logical_not_totaled += row.live_bytes;
                    continue;
                }
                const std::string label = breakdownLabel(row);
                groups[label] += row.live_bytes;
                model.audit.overview_named_rows += row.live_bytes;
                add_named_method(row.method, row.live_bytes);
            }

            model.entries.reserve(groups.size() + 16);
            std::size_t tracked_total = 0;
            for (auto& [label, bytes] : groups) {
                tracked_total += bytes;
                model.entries.push_back({label, bytes, false});
            }

            const auto synthetic_budget = [&]() -> std::size_t {
                if (process_used == 0) {
                    return std::numeric_limits<std::size_t>::max();
                }
                return process_used > tracked_total ? process_used - tracked_total : 0;
            };
            const auto add_synthetic_row =
                [&](std::string label, const std::size_t bytes, const bool unaccounted = false) -> std::size_t {
                const std::size_t capped = std::min(bytes, synthetic_budget());
                if (capped == 0) {
                    return 0;
                }
                tracked_total += capped;
                model.entries.push_back({std::move(label), capped, unaccounted});
                return capped;
            };

            const auto& proc = snapshot.process;
            if (proc.cuda_slab_reserved_bytes > snapshot.accounted_slab_live_bytes) {
                const std::size_t slab_gap =
                    proc.cuda_slab_reserved_bytes - snapshot.accounted_slab_live_bytes;
                model.audit.cuda_slab_reserve_gap = slab_gap;
                add_synthetic_row("cuda.slab.reserve_gap", slab_gap);
            }

            if (proc.cuda_pool_valid && proc.cuda_pool_reserved > proc.cuda_pool_used) {
                const std::size_t pool_overhead = proc.cuda_pool_reserved - proc.cuda_pool_used;
                model.audit.cuda_pool_overhead = pool_overhead;
                add_synthetic_row("cuda.pool.overhead", pool_overhead);
            }

            std::size_t pool_untracked = 0;
            if (proc.cuda_pool_valid &&
                proc.cuda_pool_used > snapshot.accounted_cuda_pool_live_bytes) {
                pool_untracked = proc.cuda_pool_used - snapshot.accounted_cuda_pool_live_bytes;
            }

            const auto add_phase = [&](const char* label, std::size_t bytes) {
                if (bytes == 0)
                    return;
                model.audit.cuda_context_phases += bytes;
                const std::size_t capped = std::min(bytes, synthetic_budget());
                if (capped == 0) {
                    return;
                }
                tracked_total += capped;
                model.entries.push_back({label, capped, false});
            };
            add_phase("cuda.primary_context", proc.cuda_phase_primary_context);
            add_phase("cuda.default_pool", proc.cuda_phase_default_pool);
            add_phase("cuda.curand_load", proc.cuda_phase_curand_load);

            if (proc.cuda_context_baseline > model.audit.cuda_context_phases) {
                const std::size_t residual =
                    proc.cuda_context_baseline - model.audit.cuda_context_phases;
                model.audit.cuda_context_residual = residual;
                add_synthetic_row("cuda.context.residual", residual);
            }
            if (proc.cuda_warmup_bytes > 0) {
                add_synthetic_row("cuda.modules", proc.cuda_warmup_bytes);
            }
            if (proc.vulkan_vma_block_bytes > 0) {
                const auto vksplat_it = groups.find("vksplat");
                const std::size_t vksplat_labeled =
                    vksplat_it != groups.end() ? vksplat_it->second : 0;
                std::size_t vulkan_labeled = 0;
                for (const auto& [label, bytes] : groups) {
                    if (label.rfind("vulkan.", 0) == 0) {
                        vulkan_labeled += bytes;
                    }
                }
                model.audit.vulkan_named_rows = vksplat_labeled + vulkan_labeled;
                const std::size_t vulkan_residual =
                    proc.vulkan_vma_block_bytes > model.audit.vulkan_named_rows
                        ? proc.vulkan_vma_block_bytes - model.audit.vulkan_named_rows
                        : 0;
                if (vulkan_residual > 0) {
                    model.audit.vulkan_residual = vulkan_residual;
                    add_synthetic_row("vulkan.residual", vulkan_residual, true);
                }
            }

            if (pool_untracked > 0) {
                const std::size_t bucket_cache =
                    std::min(proc.cuda_pool_bucket_cache_bytes, pool_untracked);
                if (bucket_cache > 0) {
                    model.audit.cuda_pool_bucket_cache = bucket_cache;
                    add_synthetic_row("cuda.pool.bucket_cache", bucket_cache);
                }
                const std::size_t pool_remainder = pool_untracked - bucket_cache;
                if (pool_remainder > 0) {
                    model.audit.cuda_pool_untracked_used = pool_remainder;
                    add_synthetic_row("cuda.pool.untracked_used", pool_remainder);
                }
            }

            std::size_t unattributed_balance = 0;
            if (process_used > tracked_total) {
                unattributed_balance = process_used - tracked_total;
                model.audit.process_unresolved = unattributed_balance;
                tracked_total += unattributed_balance;
            }

            std::sort(model.entries.begin(), model.entries.end(),
                      [](const VramBreakdownEntry& a, const VramBreakdownEntry& b) {
                          return a.bytes > b.bytes;
                      });

            if (unattributed_balance > 0) {
                // process_used is the NVML per-PID reading: CUDA context working set,
                // driver-internal reservations, and heap fragmentation no per-allocation
                // hook can observe. After every itemizable source is rowed above, this
                // remainder is genuinely untrackable rather than a profiler miss.
                model.entries.push_back(
                    {"driver/context reserved (NVML, untracked)", unattributed_balance, true, false});
            }

            std::size_t row_sum = tracked_total;
            if (process_used > 0) {
                const std::size_t unaccounted =
                    process_used > tracked_total ? process_used - tracked_total : 0;
                if (unaccounted > 0) {
                    model.entries.push_back({"(unaccounted)", unaccounted, true, false});
                    row_sum += unaccounted;
                }
            }

            model.audit.breakdown_totaled = row_sum;
            model.entries.push_back({"\xE2\x80\x94 Sum", row_sum, false, true});
            if (process_used > 0)
                model.entries.push_back({"\xE2\x80\x94 Process VRAM", process_used, false, true});
            model.denom = process_used > 0 ? process_used : tracked_total;
            return model;
        }

        [[nodiscard]] Rml::Vector2f contextSize(Rml::ElementDocument* document) {
            if (!document)
                return {};
            auto* context = document->GetContext();
            if (!context)
                return {};
            const auto dimensions = context->GetDimensions();
            return {
                static_cast<float>(dimensions.x),
                static_cast<float>(dimensions.y),
            };
        }

        [[nodiscard]] float finiteOr(float value, float fallback) {
            return std::isfinite(value) ? value : fallback;
        }

        [[nodiscard]] float maxHudExtent(float viewport_extent, float origin) {
            if (!std::isfinite(viewport_extent) || viewport_extent <= 0.0f)
                return std::numeric_limits<float>::infinity();
            const float leading_padding = origin >= 0.0f ? origin : kHudViewportPaddingPx;
            return std::max(1.0f, viewport_extent - leading_padding - kHudViewportPaddingPx);
        }

        [[nodiscard]] float clampHudExtent(float requested,
                                           float min_extent,
                                           float viewport_extent,
                                           float origin) {
            requested = finiteOr(requested, min_extent);
            if (requested <= 0.0f)
                requested = min_extent;

            const float max_extent = maxHudExtent(viewport_extent, origin);
            if (!std::isfinite(max_extent))
                return std::max(min_extent, requested);

            const float effective_min = std::min(min_extent, max_extent);
            return std::clamp(requested, effective_min, max_extent);
        }

        [[nodiscard]] float clampHudPosition(float requested,
                                             float extent,
                                             float viewport_extent) {
            if (!std::isfinite(requested) || requested < 0.0f)
                return -1.0f;
            if (!std::isfinite(viewport_extent) || viewport_extent <= 0.0f)
                return std::max(0.0f, requested);

            const float safe_extent = std::max(1.0f, finiteOr(extent, 1.0f));
            const float max_pos = std::max(0.0f, viewport_extent - safe_extent - kHudViewportPaddingPx);
            return std::clamp(requested, 0.0f, max_pos);
        }

    } // namespace

    VramHudOverlay::VramHudOverlay() {
        click_listener_.owner = this;
        header_drag_listener_.owner = this;
        resize_drag_listener_.owner = this;
        filter_listener_.owner = this;
        filter_clear_listener_.owner = this;
        tab_listener_.owner = this;
        anno_filter_listener_.owner = this;
        anno_filter_clear_listener_.owner = this;
        loadPersistedState();
    }

    VramHudOverlay::~VramHudOverlay() = default;

    void VramHudOverlay::loadPersistedState() {
        LayoutState ls;
        ls.load();
        pos_x_ = ls.vram_hud_x;
        pos_y_ = ls.vram_hud_y;
        size_w_ = ls.vram_hud_width;
        size_h_ = ls.vram_hud_height;
        if (ls.vram_hud_active_tab == "overview" || ls.vram_hud_active_tab == "allocations" ||
            ls.vram_hud_active_tab == "annotations" || ls.vram_hud_active_tab == "tree") {
            active_tab_ = ls.vram_hud_active_tab;
        }
        collapsed_paths_.clear();
        for (const auto& p : ls.vram_hud_collapsed_paths)
            collapsed_paths_.insert(p);
        default_collapse_applied_ = !collapsed_paths_.empty();
    }

    void VramHudOverlay::schedulePersistSave() {
        persistence_dirty_ = true;
    }

    void VramHudOverlay::persistNow() {
        if (!persistence_dirty_)
            return;
        (void)sanitizeGeometry();
        LayoutState ls;
        ls.load();
        ls.vram_hud_x = pos_x_;
        ls.vram_hud_y = pos_y_;
        ls.vram_hud_width = size_w_;
        ls.vram_hud_height = size_h_;
        ls.vram_hud_active_tab = active_tab_;
        ls.vram_hud_collapsed_paths.assign(collapsed_paths_.begin(), collapsed_paths_.end());
        ls.save();
        persistence_dirty_ = false;
    }

    void VramHudOverlay::onDocumentLoaded(Rml::ElementDocument* document) {
        document_ = document;
        listeners_attached_ = false;
        rows_by_path_.clear();
        counter_rows_by_key_.clear();
        allocs_rows_.clear();
        anno_rows_.clear();
        cached_allocs_summary_.clear();
        cached_anno_summary_.clear();
        summary_by_key_.clear();
        cached_iteration_text_.clear();
        cached_throughput_text_.clear();
        cached_device_text_.clear();
        last_sequence_ = 0;
        last_visible_ = false;
        root_ = nullptr;
        header_ = nullptr;
        resize_handle_ = nullptr;
        filter_input_ = nullptr;
        iteration_label_ = nullptr;
        throughput_label_ = nullptr;
        summary_root_ = nullptr;
        counters_root_ = nullptr;
        counters_empty_ = nullptr;
        panel_overview_ = nullptr;
        panel_allocations_ = nullptr;
        panel_tree_ = nullptr;
        tabs_root_ = nullptr;
        allocs_rows_root_ = nullptr;
        allocs_summary_value_ = nullptr;
        breakdown_root_ = nullptr;
        panel_annotations_ = nullptr;
        anno_rows_root_ = nullptr;
        anno_summary_value_ = nullptr;
        anno_filter_input_ = nullptr;
        anno_filter_clear_ = nullptr;
        rows_root_ = nullptr;
        device_label_ = nullptr;
        empty_row_ = nullptr;

        if (!document_)
            return;

        root_ = document_->GetElementById("vram-hud-overlay");
        header_ = document_->GetElementById("vram-hud-header");
        resize_handle_ = document_->GetElementById("vram-hud-resize");
        filter_input_ = document_->GetElementById("vram-hud-filter");
        filter_clear_ = document_->GetElementById("vram-hud-filter-clear");
        iteration_label_ = document_->GetElementById("vram-hud-iteration");
        throughput_label_ = document_->GetElementById("vram-hud-throughput");
        summary_root_ = document_->GetElementById("vram-hud-summary");
        counters_root_ = document_->GetElementById("vram-hud-counters");
        counters_empty_ = document_->GetElementById("vram-hud-counters-empty");
        panel_overview_ = document_->GetElementById("vram-hud-panel-overview");
        panel_allocations_ = document_->GetElementById("vram-hud-panel-allocations");
        panel_tree_ = document_->GetElementById("vram-hud-panel-tree");
        tabs_root_ = document_->GetElementById("vram-hud-tabs");
        allocs_rows_root_ = document_->GetElementById("vram-hud-allocs-rows");
        allocs_summary_value_ = document_->GetElementById("vram-hud-allocs-summary-value");
        breakdown_root_ = document_->GetElementById("vram-hud-breakdown");
        panel_annotations_ = document_->GetElementById("vram-hud-panel-annotations");
        anno_rows_root_ = document_->GetElementById("vram-hud-anno-rows");
        anno_summary_value_ = document_->GetElementById("vram-hud-anno-summary-value");
        anno_filter_input_ = document_->GetElementById("vram-hud-anno-filter");
        anno_filter_clear_ = document_->GetElementById("vram-hud-anno-filter-clear");
        rows_root_ = document_->GetElementById("vram-hud-rows");

        if (counters_root_) {
            for (auto* it = counters_root_->GetFirstChild(); it != nullptr;) {
                auto* next = it->GetNextSibling();
                if (it != counters_empty_)
                    counters_root_->RemoveChild(it);
                it = next;
            }
        }
        if (allocs_rows_root_)
            allocs_rows_root_->SetInnerRML("");
        if (breakdown_root_)
            breakdown_root_->SetInnerRML("");
        if (anno_rows_root_)
            anno_rows_root_->SetInnerRML("");

        if (summary_root_) {
            summary_root_->SetInnerRML("");
            for (const auto& spec : kSummaryRows) {
                auto row_ptr = document_->CreateElement("div");
                row_ptr->SetAttribute("class", "vram-hud-summary-row");
                auto* row = summary_root_->AppendChild(std::move(row_ptr));

                auto* label = createSpan(document_, row, "vram-hud-summary-label");
                label->SetInnerRML(Rml::String(spec.label));

                auto* value = createSpan(document_, row, "vram-hud-summary-value");
                summary_by_key_[std::string(spec.key)] = SummaryEntry{value, {}};
            }
            auto device_ptr = document_->CreateElement("div");
            device_ptr->SetAttribute("class", "vram-hud-device");
            device_label_ = summary_root_->AppendChild(std::move(device_ptr));
        }

        if (rows_root_) {
            rows_root_->SetInnerRML("");
            auto empty_ptr = document_->CreateElement("div");
            empty_ptr->SetAttribute("class", "vram-hud-empty");
            empty_ptr->SetInnerRML("Waiting for training diagnostics...");
            empty_row_ = rows_root_->AppendChild(std::move(empty_ptr));
        }

        if (filter_input_) {
            if (auto* input = dynamic_cast<Rml::ElementFormControlInput*>(filter_input_))
                input->SetValue(Rml::String(filter_text_));
        }
        updateFilterClearVisibility();

        applyPersistedGeometry();
        refreshTabClasses();
        attachListeners();
        apply();
    }

    void VramHudOverlay::onDocumentDestroyed() {
        persistNow();
        document_ = nullptr;
        root_ = nullptr;
        header_ = nullptr;
        resize_handle_ = nullptr;
        filter_input_ = nullptr;
        iteration_label_ = nullptr;
        throughput_label_ = nullptr;
        summary_root_ = nullptr;
        counters_root_ = nullptr;
        counters_empty_ = nullptr;
        panel_overview_ = nullptr;
        panel_allocations_ = nullptr;
        panel_tree_ = nullptr;
        tabs_root_ = nullptr;
        allocs_rows_root_ = nullptr;
        allocs_summary_value_ = nullptr;
        breakdown_root_ = nullptr;
        panel_annotations_ = nullptr;
        anno_rows_root_ = nullptr;
        anno_summary_value_ = nullptr;
        anno_filter_input_ = nullptr;
        anno_filter_clear_ = nullptr;
        rows_root_ = nullptr;
        device_label_ = nullptr;
        empty_row_ = nullptr;
        rows_by_path_.clear();
        counter_rows_by_key_.clear();
        allocs_rows_.clear();
        breakdown_rows_.clear();
        summary_by_key_.clear();
        listeners_attached_ = false;
        dragging_header_ = false;
        dragging_resize_ = false;
        pointer_captured_ = false;
    }

    void VramHudOverlay::attachListeners() {
        if (listeners_attached_)
            return;
        if (rows_root_)
            rows_root_->AddEventListener(Rml::EventId::Click, &click_listener_);
        if (header_) {
            header_->AddEventListener(Rml::EventId::Dragstart, &header_drag_listener_);
            header_->AddEventListener(Rml::EventId::Drag, &header_drag_listener_);
            header_->AddEventListener(Rml::EventId::Dragend, &header_drag_listener_);
        }
        if (resize_handle_) {
            resize_handle_->AddEventListener(Rml::EventId::Dragstart, &resize_drag_listener_);
            resize_handle_->AddEventListener(Rml::EventId::Drag, &resize_drag_listener_);
            resize_handle_->AddEventListener(Rml::EventId::Dragend, &resize_drag_listener_);
        }
        if (filter_input_)
            filter_input_->AddEventListener(Rml::EventId::Change, &filter_listener_);
        if (filter_clear_)
            filter_clear_->AddEventListener(Rml::EventId::Click, &filter_clear_listener_);
        if (tabs_root_)
            tabs_root_->AddEventListener(Rml::EventId::Click, &tab_listener_);
        if (anno_filter_input_)
            anno_filter_input_->AddEventListener(Rml::EventId::Change, &anno_filter_listener_);
        if (anno_filter_clear_)
            anno_filter_clear_->AddEventListener(Rml::EventId::Click, &anno_filter_clear_listener_);
        listeners_attached_ = true;
    }

    void VramHudOverlay::updateAnnoFilterClearVisibility() {
        if (anno_filter_clear_)
            anno_filter_clear_->SetClass("hidden", anno_filter_text_.empty());
    }

    void VramHudOverlay::onAnnoFilterChange(Rml::Event& event) {
        if (!anno_filter_input_)
            return;
        auto* input = dynamic_cast<Rml::ElementFormControlInput*>(anno_filter_input_);
        if (!input)
            return;
        const std::string value = input->GetValue();
        if (value == anno_filter_text_)
            return;
        anno_filter_text_ = value;
        anno_filter_text_lower_ = toLowerAscii(anno_filter_text_);
        updateAnnoFilterClearVisibility();
        applyAnnotations();
        event.StopPropagation();
    }

    void VramHudOverlay::onAnnoFilterClear() {
        if (anno_filter_text_.empty())
            return;
        anno_filter_text_.clear();
        anno_filter_text_lower_.clear();
        if (anno_filter_input_) {
            if (auto* input = dynamic_cast<Rml::ElementFormControlInput*>(anno_filter_input_))
                input->SetValue("");
        }
        updateAnnoFilterClearVisibility();
        applyAnnotations();
    }

    void VramHudOverlay::AnnoFilterListener::ProcessEvent(Rml::Event& event) {
        if (owner)
            owner->onAnnoFilterChange(event);
    }

    void VramHudOverlay::AnnoFilterClearListener::ProcessEvent(Rml::Event&) {
        if (owner)
            owner->onAnnoFilterClear();
    }

    void VramHudOverlay::TabListener::ProcessEvent(Rml::Event& event) {
        if (!owner)
            return;
        auto* target = event.GetTargetElement();
        while (target) {
            const auto key = target->GetAttribute<Rml::String>("data-vram-tab", "");
            if (!key.empty()) {
                owner->setActiveTab(std::string(key));
                event.StopPropagation();
                return;
            }
            target = target->GetParentNode();
        }
    }

    void VramHudOverlay::setActiveTab(std::string_view tab) {
        const std::string requested(tab);
        if (requested != "overview" && requested != "allocations" &&
            requested != "annotations" && requested != "tree")
            return;
        if (active_tab_ == requested)
            return;
        active_tab_ = requested;
        refreshTabClasses();
        schedulePersistSave();
        persistNow();
        apply();
    }

    void VramHudOverlay::refreshTabClasses() {
        if (tabs_root_) {
            for (auto* el = tabs_root_->GetFirstChild(); el != nullptr; el = el->GetNextSibling()) {
                const auto key = el->GetAttribute<Rml::String>("data-vram-tab", "");
                el->SetClass("active", !key.empty() && key == active_tab_);
            }
        }
        if (panel_overview_)
            panel_overview_->SetClass("hidden", active_tab_ != "overview");
        if (panel_allocations_)
            panel_allocations_->SetClass("hidden", active_tab_ != "allocations");
        if (panel_annotations_)
            panel_annotations_->SetClass("hidden", active_tab_ != "annotations");
        if (panel_tree_)
            panel_tree_->SetClass("hidden", active_tab_ != "tree");
    }

    void VramHudOverlay::updateFilterClearVisibility() {
        if (filter_clear_)
            filter_clear_->SetClass("hidden", filter_text_.empty());
    }

    void VramHudOverlay::setFilterText(std::string text) {
        if (text == filter_text_)
            return;
        filter_text_ = std::move(text);
        filter_text_lower_ = toLowerAscii(filter_text_);
        if (filter_input_) {
            if (auto* input = dynamic_cast<Rml::ElementFormControlInput*>(filter_input_))
                input->SetValue(Rml::String(filter_text_));
        }
        updateFilterClearVisibility();
        apply();
    }

    void VramHudOverlay::onFilterClear() {
        setFilterText({});
    }

    void VramHudOverlay::FilterClearListener::ProcessEvent(Rml::Event&) {
        if (owner)
            owner->onFilterClear();
    }

    void VramHudOverlay::applyPersistedGeometry() {
        if (!root_)
            return;
        if (sanitizeGeometry())
            schedulePersistSave();
        if (pos_x_ >= 0.0f && pos_y_ >= 0.0f) {
            root_->SetProperty("right", "auto");
            root_->SetProperty("left", std::format("{:.1f}px", pos_x_));
            root_->SetProperty("top", std::format("{:.1f}px", pos_y_));
        }
        if (size_w_ > 0.0f)
            root_->SetProperty("width", std::format("{:.1f}px", size_w_));
        if (size_h_ > 0.0f)
            root_->SetProperty("height", std::format("{:.1f}px", size_h_));
    }

    bool VramHudOverlay::sanitizeGeometry() {
        const float old_pos_x = pos_x_;
        const float old_pos_y = pos_y_;
        const float old_size_w = size_w_;
        const float old_size_h = size_h_;

        const auto bounds = contextSize(document_);
        if (size_w_ > 0.0f || !std::isfinite(size_w_))
            size_w_ = clampHudExtent(size_w_, kMinHudWidthPx, bounds.x, pos_x_);
        if (size_h_ > 0.0f || !std::isfinite(size_h_))
            size_h_ = clampHudExtent(size_h_, kMinHudHeightPx, bounds.y, pos_y_);
        pos_x_ = clampHudPosition(pos_x_, size_w_ > 0.0f ? size_w_ : kMinHudWidthPx, bounds.x);
        pos_y_ = clampHudPosition(pos_y_, size_h_ > 0.0f ? size_h_ : kMinHudHeightPx, bounds.y);

        return old_pos_x != pos_x_ || old_pos_y != pos_y_ ||
               old_size_w != size_w_ || old_size_h != size_h_;
    }

    void VramHudOverlay::setState(State state) {
        const bool visibility_changed = last_visible_ != state.visible;
        const bool data_changed = state.visible && last_sequence_ != state.snapshot.sequence;
        state_ = std::move(state);
        if (!visibility_changed && !data_changed)
            return;
        last_visible_ = state_.visible;
        last_sequence_ = state_.snapshot.sequence;
        apply();
    }

    bool VramHudOverlay::isDueForProcessSample(std::chrono::milliseconds interval) {
        const auto now = std::chrono::steady_clock::now();
        if (last_process_sample_ == std::chrono::steady_clock::time_point{} ||
            now - last_process_sample_ >= interval) {
            last_process_sample_ = now;
            return true;
        }
        return false;
    }

    void VramHudOverlay::apply() {
        if (!document_ || !root_)
            return;

        root_->SetClass("hidden", !state_.visible);
        if (!state_.visible)
            return;

        const auto& s = state_.snapshot;
        const auto process_used = bestProcessUsed(s);
        const auto process_total = bestProcessTotal(s);

        if (iteration_label_) {
            setText(iteration_label_, cached_iteration_text_, std::format("iter {}", s.iteration));
        }

        if (throughput_label_) {
            std::string text;
            if (s.iter_per_second > 0.0) {
                text = std::format("{:.1f} iter/s", s.iter_per_second);
                if (s.iter_ms_p95 > 0.0)
                    text += std::format(" · p95 {:.1f} ms", s.iter_ms_p95);
            }
            throughput_label_->SetClass("hidden", text.empty());
            if (!text.empty())
                setText(throughput_label_, cached_throughput_text_, std::move(text));
        }

        applySummary(process_used, process_total);
        applyBreakdown(process_used);
        applyCounters();
        applyAllocations();
        applyAnnotations();

        if (!default_collapse_applied_ && !s.tree.empty()) {
            primeDefaultCollapse();
            default_collapse_applied_ = true;
            schedulePersistSave();
        }

        applyTree(process_used);
    }

    void VramHudOverlay::applySummary(std::size_t process_used, std::size_t process_total) {
        const auto& s = state_.snapshot;
        const std::size_t gap =
            process_used > s.sampled_live_bytes ? process_used - s.sampled_live_bytes : 0;

        const auto write = [&](std::string_view key, std::string value, std::string extra = {}) {
            auto it = summary_by_key_.find(std::string(key));
            if (it == summary_by_key_.end())
                return;
            setRawRml(it->second.value, it->second.cached_text,
                      buildSummaryRowValueRml(value, extra));
        };

        write("process", formatBytes(process_used), formatPercent(process_used, process_total));
        write("cuda_context", formatBytes(s.process.cuda_used),
              formatPercent(s.process.cuda_used, s.process.cuda_total));
        write("cuda_pool_used",
              formatBytes(s.process.cuda_pool_valid ? s.process.cuda_pool_used : 0));
        write("cuda_pool_reserved",
              formatBytes(s.process.cuda_pool_valid ? s.process.cuda_pool_reserved : 0));
        write("cuda_pool_fragmentation",
              formatBytes(s.process.cuda_pool_valid ? s.process.cuda_pool_fragmentation : 0));
        write("pinned_host", formatBytes(s.process.pinned_host_used),
              std::format("{} cached / {} peak",
                          formatBytes(s.process.pinned_host_cached),
                          formatBytes(s.process.pinned_host_peak)));
        write("vulkan_budget", formatBytes(s.process.vulkan_vma_used),
              formatPercent(s.process.vulkan_vma_used, process_used));
        write("vulkan_blocks", formatBytes(s.process.vulkan_vma_block_bytes));
        write("sampled", formatBytes(s.sampled_live_bytes),
              formatPercent(s.sampled_live_bytes, process_used));
        write("allocator_live", formatBytes(s.accounted_live_bytes),
              formatPercent(s.accounted_live_bytes, process_used));
        write("process_gap", formatBytes(gap), formatPercent(gap, process_used));
        write("allocator_peak", formatBytes(s.accounted_peak_bytes));
        write("events", std::format("{} alloc / {} free", s.allocation_events, s.free_events));
        write("iter_events",
              std::format("{} alloc / {} free", s.iter_allocation_events, s.iter_free_events));

        if (device_label_) {
            const std::string device_text = s.process.device_name.empty()
                                                ? std::string{"No device"}
                                                : s.process.device_name;
            setText(device_label_, cached_device_text_, std::string(device_text));
        }
    }

    void VramHudOverlay::applyBreakdown(std::size_t process_used) {
        if (!breakdown_root_)
            return;

        const VramBreakdownModel model = buildBreakdownModel(state_.snapshot, process_used);
        const auto& entries = model.entries;
        const std::size_t denom = model.denom;

        while (breakdown_rows_.size() > entries.size()) {
            breakdown_root_->RemoveChild(breakdown_rows_.back().row);
            breakdown_rows_.pop_back();
        }
        while (breakdown_rows_.size() < entries.size()) {
            auto row_ptr = document_->CreateElement("div");
            row_ptr->SetAttribute("class", "vram-hud-breakdown-row");
            auto* row = breakdown_root_->AppendChild(std::move(row_ptr));
            BreakdownRowElements e{};
            e.row = row;
            e.name = createSpan(document_, row, "vram-hud-breakdown-name");
            e.bytes = createSpan(document_, row, "vram-hud-breakdown-bytes");
            e.pct = createSpan(document_, row, "vram-hud-breakdown-pct");
            breakdown_rows_.push_back(std::move(e));
        }

        for (std::size_t i = 0; i < entries.size(); ++i) {
            auto& row = breakdown_rows_[i];
            std::string klass = "vram-hud-breakdown-row";
            if (entries[i].unaccounted)
                klass += " unaccounted";
            if (entries[i].total)
                klass += " total";
            if (row.cached_classes != klass) {
                row.cached_classes = klass;
                row.row->SetAttribute("class", Rml::String(row.cached_classes));
            }
            setText(row.name, row.cached_name, std::string(entries[i].label));
            setText(row.bytes, row.cached_bytes, formatBytes(entries[i].bytes));
            // Hide the % column for totals (they are themselves the 100% reference).
            const std::string pct = entries[i].total ? std::string("")
                                                     : formatPercent(entries[i].bytes, denom);
            setText(row.pct, row.cached_pct, std::string(pct));
        }
    }

    void VramHudOverlay::applyCounters() {
        if (!counters_root_)
            return;

        struct Entry {
            std::string label;
            std::string value;
        };
        std::vector<Entry> entries;
        entries.reserve(state_.snapshot.iter_counters.size() + state_.snapshot.gauges.size() + 20);

        for (const auto& c : state_.snapshot.iter_counters) {
            if (c.value == 0)
                continue;
            entries.push_back({c.key + " (iter)", std::to_string(c.value)});
        }
        for (const auto& g : state_.snapshot.gauges) {
            const double v = g.value;
            std::string vs;
            if (std::abs(v) >= 1000.0 || v == std::floor(v)) {
                vs = std::format("{:.0f}", v);
            } else {
                vs = std::format("{:.3f}", v);
            }
            entries.push_back({g.key, std::move(vs)});
        }

        const std::size_t process_used = bestProcessUsed(state_.snapshot);
        if (process_used > 0 || state_.snapshot.sampled_live_bytes > 0 || !state_.snapshot.rows.empty()) {
            const VramBreakdownModel model = buildBreakdownModel(state_.snapshot, process_used);
            const auto& audit = model.audit;
            const auto add_audit = [&](std::string label, const std::size_t bytes, const bool always = false) {
                if (!always && bytes == 0) {
                    return;
                }
                entries.push_back({std::move(label), formatBytes(bytes)});
            };

            add_audit("vram.audit.process_used", audit.process_used, true);
            add_audit("vram.audit.sampled_subtotal_raw", audit.raw_sampled_subtotal, true);
            add_audit("vram.audit.sampled_gap_raw",
                      audit.process_used > audit.raw_sampled_subtotal
                          ? audit.process_used - audit.raw_sampled_subtotal
                          : 0,
                      true);
            add_audit("vram.audit.overview_named_rows", audit.overview_named_rows, true);
            add_audit("vram.audit.named.direct_rows", audit.named_direct_rows);
            add_audit("vram.audit.named.async_rows", audit.named_async_rows);
            add_audit("vram.audit.named.slab_rows", audit.named_slab_rows);
            add_audit("vram.audit.named.bucketed_rows", audit.named_bucketed_rows);
            add_audit("vram.audit.named.arena_rows", audit.named_arena_rows);
            add_audit("vram.audit.named.external_rows", audit.named_external_rows);
            add_audit("vram.audit.named.unknown_rows", audit.named_unknown_rows);
            add_audit("vram.audit.fastgs_logical_not_totaled", audit.fastgs_logical_not_totaled);
            add_audit("vram.audit.cuda_pool_used", audit.cuda_pool_used);
            add_audit("vram.audit.cuda_pool_reserved", audit.cuda_pool_reserved);
            add_audit("vram.audit.cuda_pool_accounted_live", audit.cuda_pool_accounted_live);
            add_audit("vram.audit.cuda_slab_reserved", audit.cuda_slab_reserved);
            add_audit("vram.audit.cuda_slab_reserve_gap", audit.cuda_slab_reserve_gap);
            add_audit("vram.audit.accounted.slab_live", audit.accounted_slab_live);
            add_audit("vram.audit.accounted.bucketed_live", audit.accounted_bucketed_live);
            add_audit("vram.audit.accounted.async_live", audit.accounted_async_live);
            add_audit("vram.audit.accounted.direct_live", audit.accounted_direct_live);
            add_audit("vram.audit.accounted.arena_live", audit.accounted_arena_live);
            add_audit("vram.audit.accounted.external_live", audit.accounted_external_live);
            add_audit("vram.audit.accounted.unknown_live", audit.accounted_unknown_live);
            add_audit("vram.audit.cuda_pool_bucket_cache", audit.cuda_pool_bucket_cache);
            add_audit("vram.audit.cuda_pool_untracked_used", audit.cuda_pool_untracked_used);
            add_audit("vram.audit.cuda_pool_overhead", audit.cuda_pool_overhead);
            add_audit("vram.audit.cuda_context_baseline", audit.cuda_context_baseline);
            add_audit("vram.audit.cuda_context_phases", audit.cuda_context_phases);
            add_audit("vram.audit.cuda_context_residual", audit.cuda_context_residual);
            add_audit("vram.audit.vulkan_vma_blocks", audit.vulkan_vma_blocks);
            add_audit("vram.audit.vulkan_named_rows", audit.vulkan_named_rows);
            add_audit("vram.audit.vulkan_residual", audit.vulkan_residual);
            add_audit("vram.audit.breakdown_totaled", audit.breakdown_totaled, true);
            add_audit("vram.audit.process_unresolved", audit.process_unresolved, true);
        }

        if (counters_empty_)
            counters_empty_->SetClass("hidden", !entries.empty());

        std::unordered_set<std::string> seen;
        seen.reserve(entries.size());
        Rml::Element* cursor = counters_root_->GetFirstChild();
        if (cursor == counters_empty_)
            cursor = cursor ? cursor->GetNextSibling() : nullptr;
        for (const auto& e : entries) {
            seen.insert(e.label);
            auto [it, inserted] = counter_rows_by_key_.try_emplace(e.label);
            auto& row = it->second;
            if (inserted) {
                auto row_ptr = document_->CreateElement("div");
                row_ptr->SetAttribute("class", "vram-hud-counter-row");
                Rml::Element* anchor = cursor;
                row.row = anchor ? counters_root_->InsertBefore(std::move(row_ptr), anchor)
                                 : counters_root_->AppendChild(std::move(row_ptr));
                auto* label = createSpan(document_, row.row, "vram-hud-counter-label");
                label->SetInnerRML(Rml::String(e.label));
                row.value = createSpan(document_, row.row, "vram-hud-counter-value");
            } else if (row.row != cursor) {
                auto owned = counters_root_->RemoveChild(row.row);
                row.row = cursor ? counters_root_->InsertBefore(std::move(owned), cursor)
                                 : counters_root_->AppendChild(std::move(owned));
            }
            cursor = row.row->GetNextSibling();
            setText(row.value, row.cached_value, std::string(e.value));
            row.row->SetClass("hidden", false);
        }

        for (auto it = counter_rows_by_key_.begin(); it != counter_rows_by_key_.end();) {
            if (!seen.contains(it->first)) {
                counters_root_->RemoveChild(it->second.row);
                it = counter_rows_by_key_.erase(it);
            } else {
                ++it;
            }
        }
    }

    void VramHudOverlay::applyAnnotations() {
        if (!document_ || !anno_rows_root_)
            return;

        const auto clear_rows = [this]() {
            while (!anno_rows_.empty()) {
                if (anno_rows_.back().row)
                    anno_rows_root_->RemoveChild(anno_rows_.back().row);
                anno_rows_.pop_back();
            }
        };

        if (active_tab_ != "annotations") {
            clear_rows();
            return;
        }

        struct Entry {
            std::string category;
            std::string name;
            std::size_t live_bytes = 0;
            std::size_t peak_bytes = 0;
            double total_ms = 0.0;
            double last_ms = 0.0;
            double gpu_total_ms = 0.0;
            std::uint64_t calls = 0;
        };

        std::vector<Entry> entries;
        entries.reserve(state_.snapshot.rows.size() + state_.snapshot.tree.size() +
                        state_.snapshot.gauges.size() + state_.snapshot.iter_counters.size());

        // Labeled allocator metric rows -> tensor entries.
        for (const auto& r : state_.snapshot.rows) {
            if (r.label.empty())
                continue;
            const bool method_only = r.label == "slab" || r.label == "bucketed" ||
                                     r.label == "async" || r.label == "direct" ||
                                     r.label == "external" || r.label == "arena" ||
                                     r.label == "unknown";
            if (method_only)
                continue;
            Entry e;
            e.category = "T";
            e.name = r.label;
            e.live_bytes = r.live_bytes;
            e.peak_bytes = r.peak_bytes;
            entries.push_back(std::move(e));
        }

        // Leaf timer scopes -> kernel/function entries.
        for (const auto& n : state_.snapshot.tree) {
            if (n.timer_call_count == 0 || n.has_children)
                continue;
            Entry e;
            const bool is_kernel = n.path.find("kernel.") != std::string::npos ||
                                   n.path.find("shaders.") != std::string::npos;
            e.category = is_kernel ? "K" : "F";
            e.name = n.path;
            e.live_bytes = n.live_bytes;
            e.peak_bytes = n.peak_bytes;
            e.total_ms = n.total_ms;
            e.last_ms = n.last_ms;
            e.gpu_total_ms = n.gpu_total_ms;
            e.calls = n.timer_call_count;
            entries.push_back(std::move(e));
        }

        // Gauges -> "G" entries.
        for (const auto& g : state_.snapshot.gauges) {
            Entry e;
            e.category = "G";
            e.name = g.key;
            entries.push_back(std::move(e));
        }

        // Iteration counters -> "C" entries (skip zeros for noise reduction).
        for (const auto& c : state_.snapshot.iter_counters) {
            if (c.value == 0)
                continue;
            Entry e;
            e.category = "C";
            e.name = c.key;
            e.calls = c.value;
            entries.push_back(std::move(e));
        }

        // Filter.
        if (!anno_filter_text_lower_.empty()) {
            entries.erase(
                std::remove_if(entries.begin(), entries.end(),
                               [this](const Entry& e) {
                                   return toLowerAscii(e.name).find(anno_filter_text_lower_) ==
                                          std::string::npos;
                               }),
                entries.end());
        }

        // Stable alphabetical sort — values fluctuate every frame, so byte-sort would
        // shuffle rows on every snapshot. Use the Allocations tab when you want
        // largest-first; Annotations stays stable for searching.
        std::sort(entries.begin(), entries.end(),
                  [](const Entry& a, const Entry& b) { return a.name < b.name; });

        const auto total_entries = entries.size();
        if (entries.size() > kMaxAnnotationRows)
            entries.resize(kMaxAnnotationRows);

        if (anno_summary_value_) {
            std::string text = total_entries > entries.size()
                                   ? std::format("{} / {}", entries.size(), total_entries)
                                   : std::format("{}", entries.size());
            setText(anno_summary_value_, cached_anno_summary_, std::move(text));
        }

        while (anno_rows_.size() > entries.size()) {
            anno_rows_root_->RemoveChild(anno_rows_.back().row);
            anno_rows_.pop_back();
        }
        while (anno_rows_.size() < entries.size()) {
            auto row_ptr = document_->CreateElement("div");
            row_ptr->SetAttribute("class", "vram-hud-anno-row");
            auto* row = anno_rows_root_->AppendChild(std::move(row_ptr));
            AnnotationRowElements e{};
            e.row = row;
            e.cat = createSpan(document_, row, "vram-hud-anno-cat");
            e.name = createSpan(document_, row, "vram-hud-anno-name");
            e.bytes = createSpan(document_, row, "vram-hud-anno-bytes");
            e.peak = createSpan(document_, row, "vram-hud-anno-peak");
            e.wall = createSpan(document_, row, "vram-hud-anno-wall");
            e.gpu = createSpan(document_, row, "vram-hud-anno-gpu");
            e.calls = createSpan(document_, row, "vram-hud-anno-calls");
            anno_rows_.push_back(std::move(e));
        }

        for (std::size_t i = 0; i < entries.size(); ++i) {
            auto& row = anno_rows_[i];
            setText(row.cat, row.cached_cat, std::string(entries[i].category));
            setText(row.name, row.cached_name, std::string(entries[i].name));
            setText(row.bytes, row.cached_bytes,
                    entries[i].live_bytes > 0 ? formatBytes(entries[i].live_bytes) : std::string("--"));
            setText(row.peak, row.cached_peak,
                    entries[i].peak_bytes > 0 ? formatBytes(entries[i].peak_bytes) : std::string("--"));
            setText(row.wall, row.cached_wall,
                    entries[i].calls > 0 || entries[i].total_ms > 0.0
                        ? formatTime(entries[i].last_ms > 0.0 ? entries[i].last_ms : entries[i].total_ms)
                        : std::string("--"));
            setText(row.gpu, row.cached_gpu,
                    entries[i].gpu_total_ms > 0.0 ? formatTime(entries[i].gpu_total_ms)
                                                  : std::string("--"));
            setText(row.calls, row.cached_calls,
                    entries[i].calls > 0 ? std::to_string(entries[i].calls) : std::string("--"));
        }
    }

    void VramHudOverlay::applyAllocations() {
        if (!allocs_rows_root_)
            return;

        struct Entry {
            std::string scope;
            std::string label;
            std::size_t live_bytes;
        };
        std::vector<Entry> entries;
        entries.reserve(state_.snapshot.rows.size());
        for (const auto& r : state_.snapshot.rows) {
            if (r.live_bytes == 0)
                continue;
            entries.push_back({r.scope, r.label, r.live_bytes});
        }
        std::sort(entries.begin(), entries.end(),
                  [](const Entry& a, const Entry& b) { return a.live_bytes > b.live_bytes; });

        std::size_t total_live = 0;
        for (const auto& e : entries)
            total_live += e.live_bytes;
        const std::size_t denom = bestProcessUsed(state_.snapshot) > 0
                                      ? bestProcessUsed(state_.snapshot)
                                      : total_live;

        if (allocs_summary_value_) {
            std::string text = std::format("{} · {}", entries.size(), formatBytes(total_live));
            setText(allocs_summary_value_, cached_allocs_summary_, std::move(text));
        }

        while (allocs_rows_.size() > entries.size()) {
            allocs_rows_root_->RemoveChild(allocs_rows_.back().row);
            allocs_rows_.pop_back();
        }
        while (allocs_rows_.size() < entries.size()) {
            auto row_ptr = document_->CreateElement("div");
            row_ptr->SetAttribute("class", "vram-hud-allocs-row");
            auto* row = allocs_rows_root_->AppendChild(std::move(row_ptr));
            AllocRowElements e{};
            e.row = row;
            e.name = createSpan(document_, row, "vram-hud-allocs-name");
            e.bytes = createSpan(document_, row, "vram-hud-allocs-bytes");
            e.pct = createSpan(document_, row, "vram-hud-allocs-pct");
            allocs_rows_.push_back(std::move(e));
        }

        for (std::size_t i = 0; i < entries.size(); ++i) {
            auto& row = allocs_rows_[i];
            std::string name = entries[i].label.empty()
                                   ? entries[i].scope
                                   : entries[i].scope + " \xC2\xB7 " + entries[i].label;
            setText(row.name, row.cached_name, std::move(name));
            setText(row.bytes, row.cached_bytes, formatBytes(entries[i].live_bytes));
            setText(row.pct, row.cached_pct, formatPercent(entries[i].live_bytes, denom));
        }
    }

    void VramHudOverlay::primeDefaultCollapse() {
        for (const auto& node : state_.snapshot.tree) {
            if (node.has_children && node.depth >= kDefaultCollapseDepth)
                collapsed_paths_.insert(node.path);
        }
    }

    void VramHudOverlay::applyTree(std::size_t process_used) {
        if (!rows_root_)
            return;

        const auto& tree = state_.snapshot.tree;
        visible_paths_.clear();
        visible_paths_.reserve(tree.size());
        snapshot_paths_.clear();
        snapshot_paths_.reserve(tree.size());
        filter_ancestors_.clear();

        const bool filter_active = !filter_text_lower_.empty();
        if (filter_active) {
            for (const auto& node : tree) {
                if (toLowerAscii(node.name).find(filter_text_lower_) != std::string::npos ||
                    toLowerAscii(node.path).find(filter_text_lower_) != std::string::npos) {
                    std::string_view path = node.path;
                    while (true) {
                        const auto slash = path.find_last_of('/');
                        if (slash == std::string_view::npos)
                            break;
                        path.remove_suffix(path.size() - slash);
                        filter_ancestors_.emplace(path);
                    }
                }
            }
        }

        struct VisibleEntry {
            const lfs::diagnostics::VramTreeNodeSnapshot* node;
            bool collapsed_self;
        };
        std::vector<VisibleEntry> visible_nodes;
        visible_nodes.reserve(tree.size());

        std::vector<bool> collapsed_at_depth;
        collapsed_at_depth.reserve(8);

        for (const auto& node : tree) {
            snapshot_paths_.insert(node.path);

            while (collapsed_at_depth.size() > node.depth)
                collapsed_at_depth.pop_back();

            const bool hidden_by_parent =
                std::any_of(collapsed_at_depth.begin(), collapsed_at_depth.end(),
                            [](bool b) { return b; });

            const bool collapsed_self =
                node.has_children && !filter_active && collapsed_paths_.contains(node.path);

            bool filter_pass = true;
            if (filter_active) {
                const bool self_match =
                    toLowerAscii(node.name).find(filter_text_lower_) != std::string::npos ||
                    toLowerAscii(node.path).find(filter_text_lower_) != std::string::npos;
                const bool is_ancestor = filter_ancestors_.contains(node.path);
                filter_pass = self_match || is_ancestor;
            }

            if (!hidden_by_parent && filter_pass) {
                visible_paths_.insert(node.path);
                visible_nodes.push_back({&node, collapsed_self});
            }

            if (node.has_children)
                collapsed_at_depth.push_back(collapsed_self || hidden_by_parent);
        }

        // Drop rows whose path won't be visible this frame (vanished from snapshot OR hidden by parent).
        for (auto it = rows_by_path_.begin(); it != rows_by_path_.end();) {
            if (!visible_paths_.contains(it->first)) {
                rows_root_->RemoveChild(it->second.row);
                it = rows_by_path_.erase(it);
            } else {
                ++it;
            }
        }

        Rml::Element* cursor = rows_root_->GetFirstChild();
        for (const auto& vn : visible_nodes) {
            const auto& node = *vn.node;
            const bool collapsed_self = vn.collapsed_self;

            auto [it, inserted] = rows_by_path_.try_emplace(node.path);
            auto& row = it->second;
            if (inserted) {
                auto row_ptr = document_->CreateElement("div");
                Rml::Element* anchor = cursor ? cursor : empty_row_;
                row.row = rows_root_->InsertBefore(std::move(row_ptr), anchor);
                row.name_cell = createSpan(document_, row.row, "vram-hud-row-name");
                row.toggle = createSpan(document_, row.name_cell, "expand-toggle vram-hud-expand-toggle");
                row.label = createSpan(document_, row.name_cell, "vram-hud-node-label");
                row.badges = createSpan(document_, row.name_cell, "vram-hud-row-badges");
                row.live = createSpan(document_, row.row, "vram-hud-col-live");
                row.peak = createSpan(document_, row.row, "vram-hud-col-peak");
                row.delta = createSpan(document_, row.row, "vram-hud-col-delta");
                row.time = createSpan(document_, row.row, "vram-hud-col-time");
                row.gpu = createSpan(document_, row.row, "vram-hud-col-gpu");
            } else if (row.row != cursor) {
                auto owned = rows_root_->RemoveChild(row.row);
                Rml::Element* anchor = cursor ? cursor : empty_row_;
                row.row = rows_root_->InsertBefore(std::move(owned), anchor);
            }
            cursor = row.row->GetNextSibling();

            std::string classes = "vram-hud-tree-row";
            if (node.has_children)
                classes += " has-children";
            if (collapsed_self)
                classes += " is-collapsed";
            if (node.timer_scope)
                classes += " scope-timer";
            if (node.vram_delta_scope)
                classes += " scope-delta";
            applyRowClasses(row.row, row.cached_classes, std::move(classes));

            if (node.has_children) {
                row.row->SetAttribute("data-vram-node", Rml::String(node.path));
                row.toggle->SetAttribute("data-vram-node", Rml::String(node.path));
            } else if (row.cached_has_children) {
                row.row->RemoveAttribute("data-vram-node");
                row.toggle->RemoveAttribute("data-vram-node");
            }
            row.cached_has_children = node.has_children;

            std::string padding = std::format("padding-left: {}dp;", node.depth * kRowIndentPx);
            if (row.cached_padding != padding) {
                row.name_cell->SetAttribute("style", Rml::String(padding));
                row.cached_padding = std::move(padding);
            }

            const char* toggle_glyph = node.has_children
                                           ? (collapsed_self ? "\xE2\x96\xB6" : "\xE2\x96\xBC")
                                           : " ";
            if (row.toggle && row.cached_toggle != toggle_glyph) {
                row.cached_toggle = toggle_glyph;
                row.toggle->SetInnerRML(Rml::String(toggle_glyph));
            }

            setText(row.label, row.cached_name, std::string(node.name));

            std::string badges;
            if (node.timer_scope)
                badges += "T";
            if (node.vram_delta_scope)
                badges += "D";
            if (node.has_metrics)
                badges += "M";
            if (row.cached_badges != badges) {
                row.cached_badges = badges;
                if (badges.empty()) {
                    row.badges->SetInnerRML("");
                } else {
                    std::string badge_rml = "<em>";
                    escapeRmlInto(badge_rml, badges);
                    badge_rml += "</em>";
                    row.badges->SetInnerRML(Rml::String(badge_rml));
                }
            }

            std::string live_text = formatBytes(node.live_bytes);
            const auto live_pct = formatPercent(node.live_bytes, process_used);
            std::string live_rml;
            live_rml.reserve(live_text.size() + live_pct.size() + 16);
            live_rml += live_text;
            if (!live_pct.empty()) {
                live_rml += "<em>";
                live_rml += live_pct;
                live_rml += "</em>";
            }
            setRawRml(row.live, row.cached_live, std::move(live_rml));

            setText(row.peak, row.cached_peak, formatBytes(node.peak_bytes));

            std::string delta_rml;
            if (node.vram_delta_count > 0) {
                delta_rml = formatSignedBytes(node.last_vram_delta_bytes);
                if (node.vram_delta_count > 1) {
                    delta_rml += "<em>";
                    delta_rml += formatSignedBytes(node.net_vram_delta_bytes);
                    delta_rml += "</em>";
                }
            } else {
                delta_rml = "--";
            }
            setRawRml(row.delta, row.cached_delta, std::move(delta_rml));

            std::string time_rml;
            if (node.timer_call_count > 0) {
                time_rml = formatTime(node.last_ms > 0.0 ? node.last_ms : node.total_ms);
                if (node.timer_call_count > 1) {
                    time_rml += "<em>x";
                    time_rml += std::to_string(node.timer_call_count);
                    time_rml += "</em>";
                }
            } else {
                time_rml = "--";
            }
            setRawRml(row.time, row.cached_time, std::move(time_rml));

            std::string gpu_rml;
            if (node.gpu_call_count > 0) {
                gpu_rml = formatTime(node.gpu_last_ms > 0.0 ? node.gpu_last_ms : node.gpu_total_ms);
                if (node.gpu_call_count > 1) {
                    gpu_rml += "<em>x";
                    gpu_rml += std::to_string(node.gpu_call_count);
                    gpu_rml += "</em>";
                }
            } else {
                gpu_rml = "--";
            }
            setRawRml(row.gpu, row.cached_gpu, std::move(gpu_rml));

            row.row->SetClass("hidden", false);
        }

        if (empty_row_)
            empty_row_->SetClass("hidden", !visible_paths_.empty());

        pruneCollapsedSet();
    }

    void VramHudOverlay::pruneCollapsedSet() {
        const bool changed_before = persistence_dirty_;
        for (auto it = collapsed_paths_.begin(); it != collapsed_paths_.end();) {
            if (!snapshot_paths_.contains(*it)) {
                it = collapsed_paths_.erase(it);
                persistence_dirty_ = true;
            } else {
                ++it;
            }
        }
        if (!changed_before && persistence_dirty_) {
            // pruned entries no longer match live tree — persist on next dragend or shutdown.
        }
    }

    void VramHudOverlay::toggleNode(const std::string& path) {
        if (collapsed_paths_.contains(path))
            collapsed_paths_.erase(path);
        else
            collapsed_paths_.insert(path);
        schedulePersistSave();
        apply();
    }

    void VramHudOverlay::ClickListener::ProcessEvent(Rml::Event& event) {
        if (!owner)
            return;
        auto* target = event.GetTargetElement();
        while (target) {
            const auto key = target->GetAttribute<Rml::String>("data-vram-node", "");
            if (!key.empty()) {
                owner->toggleNode(std::string(key));
                event.StopPropagation();
                return;
            }
            target = target->GetParentNode();
        }
    }

    void VramHudOverlay::HeaderDragListener::ProcessEvent(Rml::Event& event) {
        if (owner)
            owner->onHeaderDrag(event);
    }

    void VramHudOverlay::ResizeDragListener::ProcessEvent(Rml::Event& event) {
        if (owner)
            owner->onResizeDrag(event);
    }

    void VramHudOverlay::FilterListener::ProcessEvent(Rml::Event& event) {
        if (owner)
            owner->onFilterChange(event);
    }

    void VramHudOverlay::onHeaderDrag(Rml::Event& event) {
        if (!root_)
            return;
        const auto type = event.GetId();
        const float mx = event.GetParameter("mouse_x", 0.0f);
        const float my = event.GetParameter("mouse_y", 0.0f);
        if (type == Rml::EventId::Dragstart) {
            dragging_header_ = true;
            pointer_captured_ = true;
            const auto box = root_->GetAbsoluteOffset();
            drag_start_pos_x_ = box.x;
            drag_start_pos_y_ = box.y;
            drag_start_mouse_x_ = mx;
            drag_start_mouse_y_ = my;
            event.StopPropagation();
        } else if (type == Rml::EventId::Drag && dragging_header_) {
            const float dx = mx - drag_start_mouse_x_;
            const float dy = my - drag_start_mouse_y_;
            const auto bounds = contextSize(document_);
            pos_x_ = std::max(0.0f,
                              clampHudPosition(drag_start_pos_x_ + dx,
                                               size_w_ > 0.0f ? size_w_ : root_->GetBox().GetSize().x,
                                               bounds.x));
            pos_y_ = std::max(0.0f,
                              clampHudPosition(drag_start_pos_y_ + dy,
                                               size_h_ > 0.0f ? size_h_ : root_->GetBox().GetSize().y,
                                               bounds.y));
            root_->SetProperty("right", "auto");
            root_->SetProperty("left", std::format("{:.1f}px", pos_x_));
            root_->SetProperty("top", std::format("{:.1f}px", pos_y_));
            event.StopPropagation();
        } else if (type == Rml::EventId::Dragend && dragging_header_) {
            dragging_header_ = false;
            pointer_captured_ = dragging_resize_;
            schedulePersistSave();
            persistNow();
            event.StopPropagation();
        }
    }

    void VramHudOverlay::onResizeDrag(Rml::Event& event) {
        if (!root_)
            return;
        const auto type = event.GetId();
        const float mx = event.GetParameter("mouse_x", 0.0f);
        const float my = event.GetParameter("mouse_y", 0.0f);
        if (type == Rml::EventId::Dragstart) {
            dragging_resize_ = true;
            pointer_captured_ = true;
            const auto box = root_->GetBox().GetSize();
            drag_start_size_w_ = box.x;
            drag_start_size_h_ = box.y;
            drag_start_mouse_x_ = mx;
            drag_start_mouse_y_ = my;
            event.StopPropagation();
        } else if (type == Rml::EventId::Drag && dragging_resize_) {
            const float dx = mx - drag_start_mouse_x_;
            const float dy = my - drag_start_mouse_y_;
            const auto bounds = contextSize(document_);
            size_w_ = clampHudExtent(drag_start_size_w_ + dx, kMinHudWidthPx, bounds.x, pos_x_);
            size_h_ = clampHudExtent(drag_start_size_h_ + dy, kMinHudHeightPx, bounds.y, pos_y_);
            root_->SetProperty("width", std::format("{:.1f}px", size_w_));
            root_->SetProperty("height", std::format("{:.1f}px", size_h_));
            event.StopPropagation();
        } else if (type == Rml::EventId::Dragend && dragging_resize_) {
            dragging_resize_ = false;
            pointer_captured_ = dragging_header_;
            schedulePersistSave();
            persistNow();
            event.StopPropagation();
        }
    }

    void VramHudOverlay::onFilterChange(Rml::Event& event) {
        if (!filter_input_)
            return;
        auto* input = dynamic_cast<Rml::ElementFormControlInput*>(filter_input_);
        if (!input)
            return;
        const std::string value = input->GetValue();
        if (value == filter_text_)
            return;
        filter_text_ = value;
        filter_text_lower_ = toLowerAscii(filter_text_);
        updateFilterClearVisibility();
        apply();
        event.StopPropagation();
    }

} // namespace lfs::vis::gui

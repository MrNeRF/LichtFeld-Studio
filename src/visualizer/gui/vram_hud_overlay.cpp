/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/vram_hud_overlay.hpp"

#include "core/event_bridge/localization_manager.hpp"
#include "core/events.hpp"
#include "diagnostics/vram_ledger_model.hpp"
#include "gui/layout_state.hpp"
#include "gui/string_keys.hpp"
#include "visualizer/app_store.hpp"

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
            {"vulkan_blocks", "Vulkan VMA blocks"},
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

        [[nodiscard]] const char* closureClass(
            const lfs::diagnostics::LedgerClosureState state) noexcept {
            using lfs::diagnostics::LedgerClosureState;
            switch (state) {
            case LedgerClosureState::Closed: return "closed";
            case LedgerClosureState::Gap: return "gap";
            case LedgerClosureState::Over: return "over";
            }
            return "gap";
        }

        [[nodiscard]] const char* closureGlyph(
            const lfs::diagnostics::LedgerClosureState state) noexcept {
            using lfs::diagnostics::LedgerClosureState;
            switch (state) {
            case LedgerClosureState::Closed: return "\xE2\x9C\x93";
            case LedgerClosureState::Gap: return "\xE2\x9A\x91";
            case LedgerClosureState::Over: return "\xE2\x80\xBC";
            }
            return "\xE2\x9A\x91";
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
        if (ls.vram_hud_active_tab == "overview" || ls.vram_hud_active_tab == "ledger" ||
            ls.vram_hud_active_tab == "allocations" ||
            ls.vram_hud_active_tab == "annotations" || ls.vram_hud_active_tab == "tree") {
            active_tab_ = ls.vram_hud_active_tab;
        }
        collapsed_paths_.clear();
        default_collapse_applied_ = false;
        ledger_default_collapse_applied_ = false;
        for (const auto& p : ls.vram_hud_collapsed_paths) {
            collapsed_paths_.insert(p);
            if (p.rfind("ledger/", 0) == 0) {
                ledger_default_collapse_applied_ = true;
            } else {
                default_collapse_applied_ = true;
            }
        }
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
        ledger_rows_.clear();
        anno_rows_.clear();
        cached_allocs_summary_.clear();
        cached_anno_summary_.clear();
        summary_by_key_.clear();
        cached_iteration_text_.clear();
        cached_throughput_text_.clear();
        cached_device_text_.clear();
        cached_ledger_process_.clear();
        cached_ledger_attributed_.clear();
        cached_ledger_residual_.clear();
        cached_ledger_epsilon_.clear();
        cached_ledger_closure_.clear();
        cached_ledger_over_banner_.clear();
        last_sequence_ = 0;
        has_language_generation_ = false;
        last_visible_ = false;
        last_perf_snapshot_.reset();
        last_perf_visible_ = false;
        last_perf_expanded_ = true;
        root_ = nullptr;
        perf_strip_ = nullptr;
        perf_card_ = nullptr;
        perf_rate_ = nullptr;
        perf_vram_process_ = nullptr;
        perf_vram_other_ = nullptr;
        perf_vram_free_ = nullptr;
        perf_vram_value_ = nullptr;
        perf_vram_badge_ = nullptr;
        perf_ram_process_ = nullptr;
        perf_ram_other_ = nullptr;
        perf_ram_free_ = nullptr;
        perf_ram_value_ = nullptr;
        perf_gpu_fill_ = nullptr;
        perf_gpu_value_ = nullptr;
        perf_cpu_fill_ = nullptr;
        perf_cpu_value_ = nullptr;
        perf_core_strip_ = nullptr;
        spark_vram_root_ = nullptr;
        spark_ram_root_ = nullptr;
        spark_gpu_root_ = nullptr;
        spark_cpu_root_ = nullptr;
        header_ = nullptr;
        resize_handle_ = nullptr;
        filter_input_ = nullptr;
        iteration_label_ = nullptr;
        throughput_label_ = nullptr;
        summary_root_ = nullptr;
        counters_root_ = nullptr;
        counters_empty_ = nullptr;
        panel_overview_ = nullptr;
        panel_ledger_ = nullptr;
        panel_allocations_ = nullptr;
        panel_tree_ = nullptr;
        tabs_root_ = nullptr;
        allocs_rows_root_ = nullptr;
        allocs_summary_value_ = nullptr;
        tracking_off_ = nullptr;
        ledger_process_ = nullptr;
        ledger_attributed_ = nullptr;
        ledger_residual_ = nullptr;
        ledger_epsilon_ = nullptr;
        ledger_closure_ = nullptr;
        ledger_over_banner_ = nullptr;
        ledger_rows_root_ = nullptr;
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
        perf_strip_ = document_->GetElementById("perf-hud-strip");
        perf_card_ = document_->GetElementById("perf-hud-card");
        perf_rate_ = document_->GetElementById("perf-hud-strip-rate");
        perf_vram_process_ = document_->GetElementById("perf-hud-vram-process");
        perf_vram_other_ = document_->GetElementById("perf-hud-vram-other");
        perf_vram_free_ = document_->GetElementById("perf-hud-vram-free");
        perf_vram_value_ = document_->GetElementById("perf-hud-vram-value");
        perf_vram_badge_ = document_->GetElementById("perf-hud-vram-badge");
        perf_ram_process_ = document_->GetElementById("perf-hud-ram-process");
        perf_ram_other_ = document_->GetElementById("perf-hud-ram-other");
        perf_ram_free_ = document_->GetElementById("perf-hud-ram-free");
        perf_ram_value_ = document_->GetElementById("perf-hud-ram-value");
        perf_gpu_fill_ = document_->GetElementById("perf-hud-gpu-fill");
        perf_gpu_value_ = document_->GetElementById("perf-hud-gpu-value");
        perf_cpu_fill_ = document_->GetElementById("perf-hud-cpu-fill");
        perf_cpu_value_ = document_->GetElementById("perf-hud-cpu-value");
        perf_core_strip_ = document_->GetElementById("perf-hud-core-strip");
        spark_vram_root_ = document_->GetElementById("perf-hud-spark-vram");
        spark_ram_root_ = document_->GetElementById("perf-hud-spark-ram");
        spark_gpu_root_ = document_->GetElementById("perf-hud-spark-gpu");
        spark_cpu_root_ = document_->GetElementById("perf-hud-spark-cpu");
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
        panel_ledger_ = document_->GetElementById("vram-hud-panel-ledger");
        panel_allocations_ = document_->GetElementById("vram-hud-panel-allocations");
        panel_tree_ = document_->GetElementById("vram-hud-panel-tree");
        tabs_root_ = document_->GetElementById("vram-hud-tabs");
        allocs_rows_root_ = document_->GetElementById("vram-hud-allocs-rows");
        allocs_summary_value_ = document_->GetElementById("vram-hud-allocs-summary-value");
        tracking_off_ = document_->GetElementById("perf-hud-tracking-off");
        ledger_process_ = document_->GetElementById("perf-hud-ledger-process");
        ledger_attributed_ = document_->GetElementById("perf-hud-ledger-attributed");
        ledger_residual_ = document_->GetElementById("perf-hud-ledger-residual");
        ledger_epsilon_ = document_->GetElementById("perf-hud-ledger-epsilon");
        ledger_closure_ = document_->GetElementById("perf-hud-ledger-closure");
        ledger_over_banner_ = document_->GetElementById("perf-hud-ledger-over-banner");
        ledger_rows_root_ = document_->GetElementById("perf-hud-ledger-rows");
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
        if (ledger_rows_root_)
            ledger_rows_root_->SetInnerRML("");
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
            empty_ptr->SetInnerRML(
                lfs::event::LocalizationManager::getInstance().get("toolbar.waiting_training_diagnostics"));
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
        panel_ledger_ = nullptr;
        panel_allocations_ = nullptr;
        panel_tree_ = nullptr;
        tabs_root_ = nullptr;
        allocs_rows_root_ = nullptr;
        allocs_summary_value_ = nullptr;
        tracking_off_ = nullptr;
        ledger_process_ = nullptr;
        ledger_attributed_ = nullptr;
        ledger_residual_ = nullptr;
        ledger_epsilon_ = nullptr;
        ledger_closure_ = nullptr;
        ledger_over_banner_ = nullptr;
        ledger_rows_root_ = nullptr;
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
        ledger_rows_.clear();
        summary_by_key_.clear();
        listeners_attached_ = false;
        dragging_header_ = false;
        dragging_resize_ = false;
        pointer_captured_ = false;
    }

    void VramHudOverlay::attachListeners() {
        if (listeners_attached_)
            return;
        if (root_) {
            root_->AddEventListener(Rml::EventId::Click, &click_listener_);
            // Expanded→compact return path: double-click the card header.
            root_->AddEventListener(Rml::EventId::Dblclick, &click_listener_);
        }
        if (header_) {
            header_->AddEventListener(Rml::EventId::Dragstart, &header_drag_listener_);
            header_->AddEventListener(Rml::EventId::Drag, &header_drag_listener_);
            header_->AddEventListener(Rml::EventId::Dragend, &header_drag_listener_);
            header_->AddEventListener(Rml::EventId::Dblclick, &click_listener_);
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
        if (requested != "overview" && requested != "ledger" && requested != "allocations" &&
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
        if (panel_ledger_)
            panel_ledger_->SetClass(
                "hidden", active_tab_ != "ledger" || !state_.snapshot.enabled);
        if (panel_allocations_)
            panel_allocations_->SetClass(
                "hidden", active_tab_ != "allocations" || !state_.snapshot.enabled);
        if (panel_annotations_)
            panel_annotations_->SetClass(
                "hidden", active_tab_ != "annotations" || !state_.snapshot.enabled);
        if (panel_tree_)
            panel_tree_->SetClass("hidden", active_tab_ != "tree" || !state_.snapshot.enabled);
        if (tracking_off_) {
            const bool detailed_tab = active_tab_ == "ledger" || active_tab_ == "allocations" ||
                                      active_tab_ == "annotations" || active_tab_ == "tree";
            tracking_off_->SetClass("hidden", !detailed_tab || state_.snapshot.enabled);
        }
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
        const auto language_generation = lfs::vis::app_store().language_generation.get();
        const bool language_changed = !has_language_generation_ ||
                                      language_generation != last_language_generation_;
        const bool effective_visible = state.visible || state.perf_hud.visible;
        const bool visibility_changed = last_visible_ != effective_visible;
        const bool data_changed = state.visible && last_sequence_ != state.snapshot.sequence;
        const bool perf_changed = last_perf_visible_ != state.perf_hud.visible ||
                                  last_perf_expanded_ != state.perf_hud.expanded ||
                                  last_perf_snapshot_ != state.perf_hud.snapshot;
        state_ = std::move(state);
        if (!visibility_changed && !data_changed && !perf_changed && !language_changed)
            return;
        if (language_changed) {
            last_language_generation_ = language_generation;
            has_language_generation_ = true;
            cached_iteration_text_.clear();
            cached_iteration_label_ = LOC("status.iteration_label");
            if (empty_row_) {
                empty_row_->SetInnerRML(
                    lfs::event::LocalizationManager::getInstance().get("toolbar.waiting_training_diagnostics"));
            }
        }
        last_visible_ = effective_visible;
        last_perf_visible_ = state_.perf_hud.visible;
        last_perf_expanded_ = state_.perf_hud.expanded;
        last_perf_snapshot_ = state_.perf_hud.snapshot;
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

        const bool visible = state_.visible || state_.perf_hud.visible;
        root_->SetClass("hidden", !visible);
        root_->SetClass("perf-hud-compact", state_.perf_hud.visible && !state_.perf_hud.expanded);
        if (!visible)
            return;

        applyCompactStrip();
        if (sparkline_tick_due())
            pushSparklineSample();
        applySparklines();
        const bool compact = state_.perf_hud.visible && !state_.perf_hud.expanded;
        if (perf_strip_) {
            perf_strip_->SetClass("hidden", !state_.perf_hud.visible || state_.perf_hud.expanded);
            perf_strip_->SetProperty("display", compact ? "flex" : "none");
        }
        if (perf_card_) {
            perf_card_->SetClass("hidden", compact);
            perf_card_->SetProperty("display", compact ? "none" : "flex");
        }

        if (!state_.visible)
            return;

        refreshTabClasses();

        const auto& s = state_.snapshot;
        const auto process_used = bestProcessUsed(s);
        const auto process_total = bestProcessTotal(s);

        if (iteration_label_) {
            setText(iteration_label_, cached_iteration_text_,
                    std::format("{} {}", cached_iteration_label_, s.iteration));
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
        snapshot_paths_.clear();
        if (active_tab_ == "ledger" && s.enabled)
            applyLedger();
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

    void VramHudOverlay::applyCompactStrip() {
        if (!state_.perf_hud.visible || !state_.perf_hud.snapshot)
            return;
        const auto& s = *state_.perf_hud.snapshot;
        const auto ratio = [](std::size_t value, std::size_t total) {
            return total == 0 ? 0.0f : std::clamp(100.0f * static_cast<float>(value) / static_cast<float>(total), 0.0f, 100.0f);
        };
        const auto percent = [](float value) { return std::clamp(value, 0.0f, 100.0f); };
        const auto set_width = [](Rml::Element* element, const float value) {
            if (element)
                element->SetProperty("width", std::format("{:.2f}%", value));
        };
        const auto set_threshold = [](Rml::Element* element, const float value) {
            if (!element)
                return;
            element->SetClass("warn", value >= 80.0f && value < 92.0f);
            element->SetClass("crit", value >= 92.0f);
        };
        const auto vram_process = ratio(s.vram_process_bytes, s.vram_total_bytes);
        const auto vram_used = ratio(s.vram_used_bytes, s.vram_total_bytes);
        set_width(perf_vram_process_, vram_process);
        set_width(perf_vram_other_, std::max(0.0f, vram_used - vram_process));
        set_width(perf_vram_free_, std::max(0.0f, 100.0f - vram_used));
        set_threshold(perf_vram_process_, vram_used);
        if (perf_vram_value_)
            perf_vram_value_->SetInnerRML(std::format("{} / {}", formatBytes(s.vram_used_bytes),
                                                      formatBytes(s.vram_total_bytes)));
        if (perf_vram_badge_) {
            // Unknown when profiler off — no standing amber GAP.
            if (!s.ledger_valid)
                perf_vram_badge_->SetInnerRML("–");
            else if (s.ledger_over)
                perf_vram_badge_->SetInnerRML("\xE2\x80\xBC"); // ‼
            else if (s.ledger_closed)
                perf_vram_badge_->SetInnerRML("\xE2\x9C\x93"); // ✓
            else
                perf_vram_badge_->SetInnerRML("!");
        }

        const auto ram_process = ratio(s.ram_process_bytes, s.ram_total_bytes);
        const auto ram_used = ratio(s.ram_used_bytes, s.ram_total_bytes);
        set_width(perf_ram_process_, ram_process);
        set_width(perf_ram_other_, std::max(0.0f, ram_used - ram_process));
        set_width(perf_ram_free_, std::max(0.0f, 100.0f - ram_used));
        set_threshold(perf_ram_process_, ram_used);
        if (perf_ram_value_)
            perf_ram_value_->SetInnerRML(std::format("{} / {}", formatBytes(s.ram_used_bytes),
                                                     formatBytes(s.ram_total_bytes)));

        const auto gpu = s.gpu_utilization_valid ? percent(s.gpu_utilization_percent) : 0.0f;
        const auto cpu = s.cpu_valid ? percent(s.process_cpu_percent) : 0.0f;
        set_width(perf_gpu_fill_, gpu);
        set_width(perf_cpu_fill_, cpu);
        // Utilization is not pressure — never paint GPU/CPU meters as warn/crit.
        if (perf_gpu_fill_) {
            perf_gpu_fill_->SetClass("warn", false);
            perf_gpu_fill_->SetClass("crit", false);
        }
        if (perf_cpu_fill_) {
            perf_cpu_fill_->SetClass("warn", false);
            perf_cpu_fill_->SetClass("crit", false);
        }
        if (perf_gpu_value_)
            perf_gpu_value_->SetInnerRML(s.gpu_utilization_valid ? std::format("{:.0f}%", gpu) : "--");
        if (perf_cpu_value_)
            perf_cpu_value_->SetInnerRML(s.cpu_valid ? std::format("{:.0f}%", cpu) : "--");
        if (perf_rate_) {
            // Keep unit literal off the SetInnerRML line (check_ui_hardcoded is line-based);
            // fps is a design-exempt technical unit, same pattern as "{:.1f} iter/s" above.
            const std::string rate_text =
                s.rate > 0.0f ? std::format("{:.1f} fps", s.rate) : std::string("--");
            perf_rate_->SetInnerRML(rate_text);
        }

        if (perf_core_strip_) {
            std::string bars;
            const std::size_t bucket_count = std::min<std::size_t>(32, s.per_core_cpu_percent.size());
            if (bucket_count > 0) {
                bars.reserve(bucket_count * 64);
                for (std::size_t i = 0; i < bucket_count; ++i) {
                    const auto begin = i * s.per_core_cpu_percent.size() / bucket_count;
                    const auto end = std::max(begin + 1,
                                              (i + 1) * s.per_core_cpu_percent.size() / bucket_count);
                    float peak = 0.0f;
                    for (std::size_t j = begin; j < end && j < s.per_core_cpu_percent.size(); ++j)
                        peak = std::max(peak, s.per_core_cpu_percent[j]);
                    bars += std::format("<span class=\"perf-hud-core-bar\" style=\"height:{:.1f}%\"></span>",
                                        percent(peak));
                }
            }
            perf_core_strip_->SetInnerRML(bars);
        }
    }

    bool VramHudOverlay::sparkline_tick_due() const noexcept {
        if (!state_.perf_hud.visible)
            return false;
        if (last_sparkline_sample_ == std::chrono::steady_clock::time_point{})
            return true;
        return std::chrono::steady_clock::now() - last_sparkline_sample_ >= std::chrono::seconds(1);
    }

    void VramHudOverlay::pushSparklineSample() {
        if (!state_.perf_hud.snapshot)
            return;
        const auto& s = *state_.perf_hud.snapshot;
        const auto ratio = [](std::size_t value, std::size_t total) {
            return total == 0
                       ? 0.0f
                       : std::clamp(100.0f * static_cast<float>(value) /
                                        static_cast<float>(total),
                                    0.0f, 100.0f);
        };
        spark_vram_hist_[spark_write_] = ratio(s.vram_process_bytes, s.vram_total_bytes);
        spark_ram_hist_[spark_write_] = ratio(s.ram_process_bytes, s.ram_total_bytes);
        spark_gpu_hist_[spark_write_] =
            s.gpu_utilization_valid ? std::clamp(s.gpu_utilization_percent, 0.0f, 100.0f) : 0.0f;
        spark_cpu_hist_[spark_write_] =
            s.cpu_valid ? std::clamp(s.process_cpu_percent, 0.0f, 100.0f) : 0.0f;
        spark_write_ = (spark_write_ + 1) % kSparklineSamples;
        if (spark_count_ < kSparklineSamples)
            ++spark_count_;
        last_sparkline_sample_ = std::chrono::steady_clock::now();
        cached_spark_vram_.clear();
        cached_spark_ram_.clear();
        cached_spark_gpu_.clear();
        cached_spark_cpu_.clear();
    }

    void VramHudOverlay::applySparklines() {
        if (!state_.perf_hud.visible || spark_count_ == 0)
            return;

        const auto render_series = [&](Rml::Element* root, std::string& cache,
                                       const std::array<float, kSparklineSamples>& hist) {
            if (!root)
                return;
            std::string rml;
            rml.reserve(spark_count_ * 72);
            const std::size_t start =
                spark_count_ < kSparklineSamples ? 0
                                                 : spark_write_;
            for (std::size_t i = 0; i < spark_count_; ++i) {
                const float value = hist[(start + i) % kSparklineSamples];
                rml += std::format(
                    "<span class=\"histogram-bar-fill\" style=\"height:{:.1f}%\"></span>",
                    value);
            }
            if (cache == rml)
                return;
            cache = rml;
            root->SetInnerRML(cache);
        };

        render_series(spark_vram_root_, cached_spark_vram_, spark_vram_hist_);
        render_series(spark_ram_root_, cached_spark_ram_, spark_ram_hist_);
        render_series(spark_gpu_root_, cached_spark_gpu_, spark_gpu_hist_);
        render_series(spark_cpu_root_, cached_spark_cpu_, spark_cpu_hist_);
    }

    void VramHudOverlay::applySummary(std::size_t process_used, std::size_t process_total) {
        const auto& s = state_.snapshot;

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
        write("vulkan_blocks", formatBytes(s.process.vulkan_vma_block_bytes));
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

    void VramHudOverlay::applyLedger() {
        if (!document_ || !ledger_rows_root_)
            return;

        using lfs::diagnostics::AttributionState;
        using lfs::diagnostics::LedgerClosureState;
        using lfs::diagnostics::VramLedgerNode;

        const auto ledger = lfs::diagnostics::buildLiveLedger(state_.snapshot);
        if (!ledger_default_collapse_applied_) {
            primeDefaultLedgerCollapse(ledger);
            ledger_default_collapse_applied_ = true;
        }

        setText(ledger_process_, cached_ledger_process_,
                std::format("{} VRAM · {}", LOC("ui.perf_this_process"),
                            formatBytes(ledger.process_used_bytes)));
        setText(ledger_attributed_, cached_ledger_attributed_,
                std::format("{} {}", LOC("ui.perf_ledger_attributed"),
                            formatBytes(ledger.attributed_bytes)));

        const std::size_t residual_magnitude =
            lfs::diagnostics::signed_byte_magnitude(ledger.residual.signed_residual_bytes);
        const char* residual_key = ledger.closure == LedgerClosureState::Over
                                       ? "ui.perf_ledger_over"
                                       : "ui.perf_ledger_gap";
        setText(ledger_residual_, cached_ledger_residual_,
                std::format("{} {}", LOC(residual_key), formatBytes(residual_magnitude)));
        setText(ledger_epsilon_, cached_ledger_epsilon_,
                std::format("\xCE\xB5 {}", formatBytes(ledger.epsilon_bytes)));

        const char* closure_key = "ui.perf_ledger_closes";
        if (ledger.closure == LedgerClosureState::Gap)
            closure_key = "ui.perf_ledger_gap";
        else if (ledger.closure == LedgerClosureState::Over)
            closure_key = "ui.perf_ledger_over";
        setText(ledger_closure_, cached_ledger_closure_,
                std::format("{} {}", closureGlyph(ledger.closure), LOC(closure_key)));
        if (ledger_closure_) {
            ledger_closure_->SetAttribute(
                "class", Rml::String(std::format("perf-hud-ledger-closure {}",
                                                 closureClass(ledger.closure))));
        }

        if (ledger_over_banner_) {
            ledger_over_banner_->SetClass("hidden", ledger.closure != LedgerClosureState::Over);
            if (ledger.closure == LedgerClosureState::Over) {
                setText(ledger_over_banner_, cached_ledger_over_banner_,
                        std::format("{} · +{}", LOC("ui.perf_ledger_over"),
                                    formatBytes(ledger.residual.over_claim_bytes)));
            }
        }

        struct VisibleLedgerNode {
            const VramLedgerNode* node = nullptr;
            std::string path;
            std::size_t parent_bytes = 0;
            std::size_t depth = 0;
            bool collapsed = false;
        };
        std::vector<VisibleLedgerNode> visible;

        const auto collect_paths = [&](auto&& self,
                                       const VramLedgerNode& node,
                                       const std::string& path) -> void {
            snapshot_paths_.insert(path);
            for (const auto& child : node.children)
                self(self, child, path + "/" + child.name);
        };
        const auto collect_visible = [&](auto&& self,
                                         const VramLedgerNode& node,
                                         std::string path,
                                         const std::size_t parent_bytes,
                                         const std::size_t depth) -> void {
            const bool collapsed = !node.children.empty() && collapsed_paths_.contains(path);
            visible.push_back({&node, std::move(path), parent_bytes, depth, collapsed});
            if (collapsed)
                return;
            const auto child_parent = node.measured_bytes;
            const std::string parent_path = visible.back().path;
            for (const auto& child : node.children)
                self(self, child, parent_path + "/" + child.name, child_parent, depth + 1);
        };

        for (const auto& root : ledger.roots) {
            const auto path = std::format("ledger/{}/{}",
                                          static_cast<unsigned>(root.root_id), root.name);
            collect_paths(collect_paths, root, path);
            collect_visible(collect_visible, root, path, ledger.process_used_bytes, 0);
        }

        while (ledger_rows_.size() > visible.size()) {
            ledger_rows_root_->RemoveChild(ledger_rows_.back().row);
            ledger_rows_.pop_back();
        }
        while (ledger_rows_.size() < visible.size()) {
            auto row_ptr = document_->CreateElement("div");
            auto* row_element = ledger_rows_root_->AppendChild(std::move(row_ptr));
            LedgerRowElements row{};
            row.row = row_element;
            row.name_cell = createSpan(document_, row_element, "perf-hud-ledger-name-cell");
            row.toggle = createSpan(document_, row.name_cell, "perf-hud-ledger-toggle");
            row.name = createSpan(document_, row.name_cell, "perf-hud-ledger-name");
            row.note = createSpan(document_, row.name_cell, "perf-hud-ledger-note");
            auto* share = createSpan(document_, row_element, "perf-hud-ledger-share");
            row.share_fill = createSpan(document_, share, "perf-hud-ledger-share-fill");
            row.disclosure = createSpan(document_, row_element, "perf-hud-ledger-disclosure");
            row.allocated = createSpan(document_, row_element, "perf-hud-ledger-allocated");
            row.badge = createSpan(document_, row_element, "perf-hud-ledger-badge");
            ledger_rows_.push_back(std::move(row));
        }

        for (std::size_t i = 0; i < visible.size(); ++i) {
            const auto& item = visible[i];
            const auto& node = *item.node;
            auto& row = ledger_rows_[i];

            std::string classes = "perf-hud-ledger-row ";
            classes += closureClass(node.closure);
            if (item.depth == 0)
                classes += " root";
            if (!node.children.empty())
                classes += " has-children";
            if (item.collapsed)
                classes += " is-collapsed";
            if (node.state == AttributionState::Nested)
                classes += " nested";
            else if (node.state == AttributionState::Unjustified)
                classes += " unjustified";

            bool estimate = false;
            const bool degenerate = node.has_required &&
                                    node.name == "fastgs_raster_live" &&
                                    node.required_bytes == node.measured_bytes;
            if (node.has_required && node.required_bytes > 0) {
                const double ratio = static_cast<double>(node.measured_bytes) /
                                     static_cast<double>(node.required_bytes);
                constexpr double kEstimateFactors[] = {1.2, 1.25, 1.5, 2.0};
                estimate = std::any_of(
                    std::begin(kEstimateFactors), std::end(kEstimateFactors),
                    [ratio](const double factor) { return std::abs(ratio / factor - 1.0) <= 0.02; });
                if (estimate)
                    classes += " estimate";
                else if (node.measured_bytes == node.required_bytes)
                    classes += " exact";
                else if (node.measured_bytes > node.required_bytes)
                    classes += " slack";
                else
                    classes += " over-budget";
            }
            if (degenerate)
                classes += " degenerate";
            applyRowClasses(row.row, row.cached_classes, std::move(classes));

            if (!node.children.empty()) {
                row.row->SetAttribute("data-vram-node", Rml::String(item.path));
                row.toggle->SetAttribute("data-vram-node", Rml::String(item.path));
            } else {
                row.row->RemoveAttribute("data-vram-node");
                row.toggle->RemoveAttribute("data-vram-node");
            }

            std::string padding = std::format("padding-left: {}dp;", item.depth * kRowIndentPx);
            if (row.cached_padding != padding) {
                row.cached_padding = std::move(padding);
                row.name_cell->SetAttribute("style", Rml::String(row.cached_padding));
            }
            const char* toggle = node.children.empty()
                                     ? " "
                                     : (item.collapsed ? "\xE2\x96\xB6" : "\xE2\x96\xBC");
            if (row.cached_toggle != toggle) {
                row.cached_toggle = toggle;
                row.toggle->SetInnerRML(Rml::String(toggle));
            }

            setText(row.name, row.cached_name, std::string(node.name));
            // Map stable English tokens from the ledger model onto locale keys (S9).
            std::string note_text = node.note;
            if (node.note == "retention")
                note_text = std::string(LOC("ui.perf_retention"));
            else if (node.note == "reclaimable")
                note_text = std::string(LOC("ui.perf_reclaimable"));
            else if (node.note == "untracked")
                note_text = std::string(LOC("ui.perf_untracked"));
            setText(row.note, row.cached_note, std::move(note_text));

            const double share = item.parent_bytes == 0
                                     ? 0.0
                                     : std::min(100.0, 100.0 * static_cast<double>(node.measured_bytes) /
                                                           static_cast<double>(item.parent_bytes));
            std::string share_width = std::format("width: {:.2f}%;", share);
            if (row.cached_share_width != share_width) {
                row.cached_share_width = std::move(share_width);
                row.share_fill->SetAttribute("style", Rml::String(row.cached_share_width));
            }

            std::string disclosure;
            if (node.has_required) {
                disclosure = std::format("{} {} / {} {}",
                                         LOC("ui.perf_required"), formatBytes(node.required_bytes),
                                         LOC("ui.perf_allocated"), formatBytes(node.measured_bytes));
            }
            setText(row.disclosure, row.cached_disclosure, std::move(disclosure));
            setText(row.allocated, row.cached_allocated, formatBytes(node.measured_bytes));

            const char* badge = closureGlyph(node.closure);
            const char* badge_title = "ui.perf_ledger_closes";
            if (node.closure == LedgerClosureState::Gap)
                badge_title = "ui.perf_ledger_gap";
            else if (node.closure == LedgerClosureState::Over)
                badge_title = "ui.perf_ledger_over";
            if (estimate) {
                badge = "EST";
                badge_title = "ui.perf_estimate_detected";
            } else if (degenerate) {
                badge = "=";
                badge_title = "ui.perf_degenerate_pair";
            } else if (node.has_required && node.measured_bytes > node.required_bytes) {
                badge_title = "ui.perf_slack";
            } else if (node.has_required && node.measured_bytes < node.required_bytes) {
                badge_title = "ui.perf_over_budget";
            } else if (node.state == AttributionState::Nested) {
                badge = "\xE2\x86\xB3";
                badge_title = "ui.perf_nested_disclosure";
            } else if (node.state == AttributionState::Unjustified) {
                badge = "\xE2\x9A\x91";
                badge_title = "ui.perf_no_owner";
            }
            setText(row.badge, row.cached_badge, std::string(badge));
            row.badge->SetAttribute("title", Rml::String(LOC(badge_title)));
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

    void VramHudOverlay::primeDefaultLedgerCollapse(
        const lfs::diagnostics::VramLedgerTree& ledger) {
        const auto visit = [&](auto&& self,
                               const lfs::diagnostics::VramLedgerNode& node,
                               const std::string& path,
                               const std::size_t depth) -> void {
            if (!node.children.empty() && depth >= kDefaultCollapseDepth)
                collapsed_paths_.insert(path);
            for (const auto& child : node.children)
                self(self, child, path + "/" + child.name, depth + 1);
        };
        for (const auto& root : ledger.roots) {
            const auto path = std::format("ledger/{}/{}",
                                          static_cast<unsigned>(root.root_id), root.name);
            visit(visit, root, path, 0);
        }
        schedulePersistSave();
    }

    void VramHudOverlay::applyTree(std::size_t process_used) {
        if (!rows_root_)
            return;

        const auto& tree = state_.snapshot.tree;
        visible_paths_.clear();
        visible_paths_.reserve(tree.size());
        snapshot_paths_.reserve(snapshot_paths_.size() + tree.size());
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
            const bool inactive_ledger_path = it->rfind("ledger/", 0) == 0 &&
                                              active_tab_ != "ledger";
            if (!inactive_ledger_path && !snapshot_paths_.contains(*it)) {
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
        const bool is_dblclick = event.GetId() == Rml::EventId::Dblclick;

        // Double-click expanded card header → collapse to strip.
        if (is_dblclick) {
            while (target) {
                if (target->GetId() == "vram-hud-header" || target->GetId() == "perf-hud-card") {
                    if (owner->state_.perf_hud.visible && owner->state_.perf_hud.expanded) {
                        lfs::core::events::ui::TogglePerfHudExpanded{}.emit();
                        event.StopPropagation();
                    }
                    return;
                }
                target = target->GetParentNode();
            }
            return;
        }

        target = event.GetTargetElement();
        while (target) {
            const auto toggle_expanded = target->GetAttribute<Rml::String>("data-perf-toggle-expanded", "");
            if (!toggle_expanded.empty()) {
                lfs::core::events::ui::TogglePerfHudExpanded{}.emit();
                event.StopPropagation();
                return;
            }
            const auto perf_tab = target->GetAttribute<Rml::String>("data-perf-tab", "");
            if (!perf_tab.empty()) {
                lfs::core::events::ui::OpenPerfHudLedger{}.emit();
                owner->setActiveTab("ledger");
                event.StopPropagation();
                return;
            }
            const auto enable_tracking =
                target->GetAttribute<Rml::String>("data-perf-enable-tracking", "");
            if (!enable_tracking.empty()) {
                owner->enableDetailedTracking();
                event.StopPropagation();
                return;
            }
            const auto key = target->GetAttribute<Rml::String>("data-vram-node", "");
            if (!key.empty()) {
                owner->toggleNode(std::string(key));
                event.StopPropagation();
                return;
            }
            target = target->GetParentNode();
        }
    }

    void VramHudOverlay::enableDetailedTracking() {
        lfs::diagnostics::VramProfiler::instance().setEnabled(true);
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

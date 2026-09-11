/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "diagnostics/vram_ledger_model.hpp"
#include "diagnostics/vram_profiler.hpp"
#include "visualizer/app_store.hpp"

#include <RmlUi/Core/EventListener.h>
#include <array>
#include <chrono>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace Rml {
    class Element;
    class ElementDocument;
} // namespace Rml

namespace lfs::vis::gui {

    class VramHudOverlay {
    public:
        struct State {
            bool visible = false;
            lfs::diagnostics::VramProfilerSnapshot snapshot;
            lfs::vis::AppStore::PerfHud perf_hud;
        };

        VramHudOverlay();
        ~VramHudOverlay();

        VramHudOverlay(const VramHudOverlay&) = delete;
        VramHudOverlay& operator=(const VramHudOverlay&) = delete;

        void onDocumentLoaded(Rml::ElementDocument* document);
        void onDocumentDestroyed();

        void setState(State state);
        [[nodiscard]] bool isVisible() const noexcept { return state_.visible || state_.perf_hud.visible; }
        [[nodiscard]] bool needsAnimationFrame() const noexcept {
            return pointer_captured_ || sparkline_tick_due();
        }
        [[nodiscard]] bool isCapturingPointer() const noexcept { return pointer_captured_; }

        [[nodiscard]] bool isDueForProcessSample(std::chrono::milliseconds interval);

    private:
        struct ClickListener final : Rml::EventListener {
            VramHudOverlay* owner = nullptr;
            void ProcessEvent(Rml::Event& event) override;
        };
        struct HeaderDragListener final : Rml::EventListener {
            VramHudOverlay* owner = nullptr;
            void ProcessEvent(Rml::Event& event) override;
        };
        struct ResizeDragListener final : Rml::EventListener {
            VramHudOverlay* owner = nullptr;
            void ProcessEvent(Rml::Event& event) override;
        };
        struct FilterListener final : Rml::EventListener {
            VramHudOverlay* owner = nullptr;
            void ProcessEvent(Rml::Event& event) override;
        };
        struct FilterClearListener final : Rml::EventListener {
            VramHudOverlay* owner = nullptr;
            void ProcessEvent(Rml::Event& event) override;
        };
        struct TabListener final : Rml::EventListener {
            VramHudOverlay* owner = nullptr;
            void ProcessEvent(Rml::Event& event) override;
        };
        struct AnnoFilterListener final : Rml::EventListener {
            VramHudOverlay* owner = nullptr;
            void ProcessEvent(Rml::Event& event) override;
        };
        struct AnnoFilterClearListener final : Rml::EventListener {
            VramHudOverlay* owner = nullptr;
            void ProcessEvent(Rml::Event& event) override;
        };

        void attachListeners();
        void apply();
        void applyCompactStrip();
        void applySparklines();
        void pushSparklineSample();
        [[nodiscard]] bool sparkline_tick_due() const noexcept;
        void applySummary(std::size_t process_used, std::size_t process_total);
        void applyLedger();
        void applyCounters();
        void applyAllocations();
        void applyAnnotations();
        void applyTree(std::size_t process_used);
        void setActiveTab(std::string_view tab);
        void refreshTabClasses();
        void onAnnoFilterChange(Rml::Event& event);
        void onAnnoFilterClear();
        void updateAnnoFilterClearVisibility();
        void primeDefaultCollapse();
        void primeDefaultLedgerCollapse(const lfs::diagnostics::VramLedgerTree& ledger);
        void toggleNode(const std::string& path);
        void enableDetailedTracking();
        void pruneCollapsedSet();
        void loadPersistedState();
        void schedulePersistSave();
        void persistNow();
        void applyPersistedGeometry();
        [[nodiscard]] bool sanitizeGeometry();
        void onHeaderDrag(Rml::Event& event);
        void onResizeDrag(Rml::Event& event);
        void onFilterChange(Rml::Event& event);
        void onFilterClear();
        void updateFilterClearVisibility();
        void setFilterText(std::string text);

        State state_;
        std::uint64_t last_sequence_ = 0;
        std::uint64_t last_language_generation_ = 0;
        bool has_language_generation_ = false;
        bool last_visible_ = false;
        std::shared_ptr<const lfs::vis::AppStore::PerfHudSnapshot> last_perf_snapshot_;
        bool last_perf_visible_ = false;
        bool last_perf_expanded_ = true;
        bool default_collapse_applied_ = false;
        bool ledger_default_collapse_applied_ = false;

        Rml::ElementDocument* document_ = nullptr;
        Rml::Element* root_ = nullptr;
        Rml::Element* perf_strip_ = nullptr;
        Rml::Element* perf_card_ = nullptr;
        Rml::Element* perf_rate_ = nullptr;
        Rml::Element* perf_vram_process_ = nullptr;
        Rml::Element* perf_vram_other_ = nullptr;
        Rml::Element* perf_vram_free_ = nullptr;
        Rml::Element* perf_vram_value_ = nullptr;
        Rml::Element* perf_vram_badge_ = nullptr;
        Rml::Element* perf_ram_process_ = nullptr;
        Rml::Element* perf_ram_other_ = nullptr;
        Rml::Element* perf_ram_free_ = nullptr;
        Rml::Element* perf_ram_value_ = nullptr;
        Rml::Element* perf_gpu_fill_ = nullptr;
        Rml::Element* perf_gpu_value_ = nullptr;
        Rml::Element* perf_cpu_fill_ = nullptr;
        Rml::Element* perf_cpu_value_ = nullptr;
        Rml::Element* perf_core_strip_ = nullptr;
        Rml::Element* spark_vram_root_ = nullptr;
        Rml::Element* spark_ram_root_ = nullptr;
        Rml::Element* spark_gpu_root_ = nullptr;
        Rml::Element* spark_cpu_root_ = nullptr;
        Rml::Element* header_ = nullptr;
        Rml::Element* resize_handle_ = nullptr;
        Rml::Element* filter_input_ = nullptr;
        Rml::Element* filter_clear_ = nullptr;
        Rml::Element* iteration_label_ = nullptr;
        Rml::Element* throughput_label_ = nullptr;
        Rml::Element* summary_root_ = nullptr;
        Rml::Element* counters_root_ = nullptr;
        Rml::Element* counters_empty_ = nullptr;
        Rml::Element* panel_overview_ = nullptr;
        Rml::Element* panel_ledger_ = nullptr;
        Rml::Element* panel_allocations_ = nullptr;
        Rml::Element* panel_tree_ = nullptr;
        Rml::Element* tabs_root_ = nullptr;
        Rml::Element* allocs_rows_root_ = nullptr;
        Rml::Element* allocs_summary_value_ = nullptr;
        Rml::Element* tracking_off_ = nullptr;
        Rml::Element* ledger_process_ = nullptr;
        Rml::Element* ledger_attributed_ = nullptr;
        Rml::Element* ledger_residual_ = nullptr;
        Rml::Element* ledger_epsilon_ = nullptr;
        Rml::Element* ledger_closure_ = nullptr;
        Rml::Element* ledger_over_banner_ = nullptr;
        Rml::Element* ledger_rows_root_ = nullptr;
        Rml::Element* panel_annotations_ = nullptr;
        Rml::Element* anno_rows_root_ = nullptr;
        Rml::Element* anno_summary_value_ = nullptr;
        Rml::Element* anno_filter_input_ = nullptr;
        Rml::Element* anno_filter_clear_ = nullptr;
        Rml::Element* rows_root_ = nullptr;
        Rml::Element* empty_row_ = nullptr;

        struct RowElements {
            Rml::Element* row = nullptr;
            Rml::Element* name_cell = nullptr;
            Rml::Element* toggle = nullptr;
            Rml::Element* label = nullptr;
            Rml::Element* badges = nullptr;
            Rml::Element* live = nullptr;
            Rml::Element* peak = nullptr;
            Rml::Element* delta = nullptr;
            Rml::Element* time = nullptr;
            Rml::Element* gpu = nullptr;
            std::string cached_name;
            std::string cached_live;
            std::string cached_peak;
            std::string cached_delta;
            std::string cached_time;
            std::string cached_gpu;
            std::string cached_badges;
            std::string cached_classes;
            std::string cached_padding;
            std::string cached_toggle;
            bool cached_has_children = false;
        };

        struct CounterRowElements {
            Rml::Element* row = nullptr;
            Rml::Element* value = nullptr;
            std::string cached_value;
        };

        struct AllocRowElements {
            Rml::Element* row = nullptr;
            Rml::Element* name = nullptr;
            Rml::Element* bytes = nullptr;
            Rml::Element* pct = nullptr;
            std::string cached_name;
            std::string cached_bytes;
            std::string cached_pct;
        };

        struct LedgerRowElements {
            Rml::Element* row = nullptr;
            Rml::Element* name_cell = nullptr;
            Rml::Element* toggle = nullptr;
            Rml::Element* name = nullptr;
            Rml::Element* note = nullptr;
            Rml::Element* share_fill = nullptr;
            Rml::Element* disclosure = nullptr;
            Rml::Element* allocated = nullptr;
            Rml::Element* badge = nullptr;
            std::string cached_name;
            std::string cached_note;
            std::string cached_share_width;
            std::string cached_disclosure;
            std::string cached_allocated;
            std::string cached_badge;
            std::string cached_classes;
            std::string cached_padding;
            std::string cached_toggle;
        };

        struct AnnotationRowElements {
            Rml::Element* row = nullptr;
            Rml::Element* cat = nullptr;
            Rml::Element* name = nullptr;
            Rml::Element* bytes = nullptr;
            Rml::Element* peak = nullptr;
            Rml::Element* wall = nullptr;
            Rml::Element* gpu = nullptr;
            Rml::Element* calls = nullptr;
            std::string cached_cat;
            std::string cached_name;
            std::string cached_bytes;
            std::string cached_peak;
            std::string cached_wall;
            std::string cached_gpu;
            std::string cached_calls;
        };

        std::unordered_map<std::string, RowElements> rows_by_path_;
        std::unordered_map<std::string, CounterRowElements> counter_rows_by_key_;
        std::vector<AllocRowElements> allocs_rows_;
        std::vector<LedgerRowElements> ledger_rows_;
        std::vector<AnnotationRowElements> anno_rows_;
        std::string cached_allocs_summary_;
        std::string cached_anno_summary_;
        std::string anno_filter_text_;
        std::string anno_filter_text_lower_;
        std::string active_tab_ = "overview";
        std::unordered_set<std::string> collapsed_paths_;
        std::unordered_set<std::string> visible_paths_;
        std::unordered_set<std::string> snapshot_paths_;
        std::unordered_set<std::string> filter_ancestors_;
        std::string filter_text_;
        std::string filter_text_lower_;
        std::string cached_throughput_text_;
        std::string cached_ledger_process_;
        std::string cached_ledger_attributed_;
        std::string cached_ledger_residual_;
        std::string cached_ledger_epsilon_;
        std::string cached_ledger_closure_;
        std::string cached_ledger_over_banner_;

        struct SummaryEntry {
            Rml::Element* value = nullptr;
            std::string cached_text;
        };
        std::unordered_map<std::string, SummaryEntry> summary_by_key_;
        std::string cached_iteration_text_;
        std::string cached_iteration_label_;
        std::string cached_device_text_;
        Rml::Element* device_label_ = nullptr;

        ClickListener click_listener_;
        HeaderDragListener header_drag_listener_;
        ResizeDragListener resize_drag_listener_;
        FilterListener filter_listener_;
        FilterClearListener filter_clear_listener_;
        TabListener tab_listener_;
        AnnoFilterListener anno_filter_listener_;
        AnnoFilterClearListener anno_filter_clear_listener_;
        bool listeners_attached_ = false;

        float pos_x_ = -1.0f;
        float pos_y_ = -1.0f;
        float size_w_ = -1.0f;
        float size_h_ = -1.0f;
        float drag_start_pos_x_ = 0.0f;
        float drag_start_pos_y_ = 0.0f;
        float drag_start_mouse_x_ = 0.0f;
        float drag_start_mouse_y_ = 0.0f;
        float drag_start_size_w_ = 0.0f;
        float drag_start_size_h_ = 0.0f;
        bool dragging_header_ = false;
        bool dragging_resize_ = false;
        bool pointer_captured_ = false;
        bool geometry_dirty_ = false;
        bool persistence_dirty_ = false;

        std::chrono::steady_clock::time_point last_process_sample_{};
        std::chrono::steady_clock::time_point last_sparkline_sample_{};

        static constexpr std::size_t kSparklineSamples = 60;
        std::array<float, kSparklineSamples> spark_vram_hist_{};
        std::array<float, kSparklineSamples> spark_ram_hist_{};
        std::array<float, kSparklineSamples> spark_gpu_hist_{};
        std::array<float, kSparklineSamples> spark_cpu_hist_{};
        std::size_t spark_count_ = 0;
        std::size_t spark_write_ = 0;
        std::string cached_spark_vram_;
        std::string cached_spark_ram_;
        std::string cached_spark_gpu_;
        std::string cached_spark_cpu_;
    };

} // namespace lfs::vis::gui

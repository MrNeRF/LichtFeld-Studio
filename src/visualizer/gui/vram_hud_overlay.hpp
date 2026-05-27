/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "diagnostics/vram_profiler.hpp"

#include <RmlUi/Core/EventListener.h>
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
        };

        VramHudOverlay();
        ~VramHudOverlay();

        VramHudOverlay(const VramHudOverlay&) = delete;
        VramHudOverlay& operator=(const VramHudOverlay&) = delete;

        void onDocumentLoaded(Rml::ElementDocument* document);
        void onDocumentDestroyed();

        void setState(State state);
        [[nodiscard]] bool isVisible() const noexcept { return state_.visible; }
        [[nodiscard]] bool needsAnimationFrame() const noexcept { return pointer_captured_; }
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
        void applySummary(std::size_t process_used, std::size_t process_total);
        void applyBreakdown(std::size_t process_used);
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
        void toggleNode(const std::string& path);
        void pruneCollapsedSet();
        void loadPersistedState();
        void schedulePersistSave();
        void persistNow();
        void applyPersistedGeometry();
        void onHeaderDrag(Rml::Event& event);
        void onResizeDrag(Rml::Event& event);
        void onFilterChange(Rml::Event& event);
        void onFilterClear();
        void updateFilterClearVisibility();
        void setFilterText(std::string text);

        State state_;
        std::uint64_t last_sequence_ = 0;
        bool last_visible_ = false;
        bool default_collapse_applied_ = false;

        Rml::ElementDocument* document_ = nullptr;
        Rml::Element* root_ = nullptr;
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
        Rml::Element* panel_allocations_ = nullptr;
        Rml::Element* panel_tree_ = nullptr;
        Rml::Element* tabs_root_ = nullptr;
        Rml::Element* allocs_rows_root_ = nullptr;
        Rml::Element* allocs_summary_value_ = nullptr;
        Rml::Element* breakdown_root_ = nullptr;
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

        struct BreakdownRowElements {
            Rml::Element* row = nullptr;
            Rml::Element* name = nullptr;
            Rml::Element* bytes = nullptr;
            Rml::Element* pct = nullptr;
            std::string cached_name;
            std::string cached_bytes;
            std::string cached_pct;
            std::string cached_classes;
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
        std::vector<BreakdownRowElements> breakdown_rows_;
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

        struct SummaryEntry {
            Rml::Element* value = nullptr;
            std::string cached_text;
        };
        std::unordered_map<std::string, SummaryEntry> summary_by_key_;
        std::string cached_iteration_text_;
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
    };

} // namespace lfs::vis::gui

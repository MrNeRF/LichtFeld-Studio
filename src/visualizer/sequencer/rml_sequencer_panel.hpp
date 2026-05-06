/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "gui/rmlui/rml_tooltip.hpp"
#include "gui/sequencer_ui_state.hpp"
#include "sequencer_controller.hpp"
#include <RmlUi/Core/EventListener.h>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace Rml {
    class Context;
    class Element;
    class ElementDocument;
} // namespace Rml

namespace lfs::vis::gui {
    class FilmStripRenderer;
    class RmlUIManager;
} // namespace lfs::vis::gui

namespace lfs::vis {
    class RenderingManager;
    class SceneManager;

    struct Theme;

    struct PanelInputState {
        float mouse_x = 0.0f;
        float mouse_y = 0.0f;
        float screen_x = 0.0f;
        float screen_y = 0.0f;
        bool mouse_down[3] = {};
        bool mouse_clicked[3] = {};
        bool mouse_released[3] = {};
        float mouse_wheel = 0.0f;
        bool key_shift = false;
        bool key_ctrl = false;
        bool key_alt = false;
        bool key_super = false;
        bool key_delete_pressed = false;
        float time = 0.0f;
        float delta_time = 0.0f;
        bool want_capture_mouse = false;
        int screen_w = 0;
        int screen_h = 0;
        std::vector<int> keys_pressed;
        std::vector<int> keys_released;
        std::vector<uint32_t> text_codepoints;
        std::vector<std::string> text_inputs;
        std::string text_editing;
        int text_editing_start = -1;
        int text_editing_length = -1;
        bool has_text_editing = false;
    };

    namespace panel_config {
        inline constexpr float TRANSPORT_ROW_HEIGHT = 36.0f;
        inline constexpr float HEIGHT = 108.0f;
        inline constexpr float PADDING_H = 16.0f;
        inline constexpr float PADDING_BOTTOM = 18.0f;
        inline constexpr float INNER_PADDING = 8.0f;
        inline constexpr float INNER_PADDING_H = 16.0f;
        inline constexpr float RULER_HEIGHT = 16.0f;
        inline constexpr float TIMELINE_HEIGHT = 24.0f;
        inline constexpr float KEYFRAME_RADIUS = 6.0f;
        inline constexpr float PLAYHEAD_WIDTH = 2.0f;
        inline constexpr float BUTTON_SIZE = 20.0f;
        inline constexpr float BUTTON_SPACING = 4.0f;

        inline constexpr float MIN_ZOOM = 0.5f;
        inline constexpr float MAX_ZOOM = 4.0f;
        inline constexpr float ZOOM_SPEED = 0.1f;
        inline constexpr float EASING_STRIPE_HEIGHT = 36.0f;
        inline constexpr float BORDER_OVERLAP = 1.0f;
    } // namespace panel_config

    struct TimelineContextMenuState {
        bool open = false;
        float time = 0.0f;
        std::optional<size_t> keyframe;
    };

    struct TransportContextMenuRequest {
        enum class Target { NONE,
                            SNAP,
                            PREVIEW,
                            FORMAT,
                            CLEAR };
        Target target = Target::NONE;
        float screen_x = 0.0f;
        float screen_y = 0.0f;
    };

    struct TimeEditRequest {
        bool active = false;
        size_t keyframe_index = 0;
        float current_time = 0.0f;
    };

    struct FocalEditRequest {
        bool active = false;
        size_t keyframe_index = 0;
        float current_focal_mm = 0.0f;
    };

    class RmlSequencerPanel {
    public:
        RmlSequencerPanel(SequencerController& controller, gui::panels::SequencerUIState& ui_state,
                          gui::RmlUIManager* rml_manager);
        ~RmlSequencerPanel();

        RmlSequencerPanel(const RmlSequencerPanel&) = delete;
        RmlSequencerPanel& operator=(const RmlSequencerPanel&) = delete;

        void render(float panel_x, float panel_y, float panel_width, float total_height,
                    const PanelInputState& input,
                    RenderingManager* rm, SceneManager* sm,
                    gui::FilmStripRenderer& film_strip);
        void compositeToScreen(int screen_w, int screen_h);
        void clearPendingComposite();

        void setFilmStripAttached(bool attached) { film_strip_attached_ = attached; }
        void setFloating(bool floating) { floating_ = floating; }
        [[nodiscard]] bool isFloating() const { return floating_; }

        [[nodiscard]] bool consumeSavePathRequest();
        [[nodiscard]] bool consumeLoadPathRequest();
        [[nodiscard]] bool consumeExportRequest();
        [[nodiscard]] bool consumeDockToggleRequest();
        [[nodiscard]] bool consumeClosePanelRequest();
        [[nodiscard]] bool consumeClearRequest();

        void openFocalLengthEdit(size_t index, float current_focal_mm);

        [[nodiscard]] bool isHovered() const { return hovered_; }
        [[nodiscard]] bool wantsKeyboard() const { return wants_keyboard_; }

        [[nodiscard]] float zoomLevel() const { return zoom_level_; }
        [[nodiscard]] float panOffset() const { return pan_offset_; }
        [[nodiscard]] float cachedPanelX() const { return cached_panel_x_; }
        [[nodiscard]] float cachedPanelY() const { return cached_panel_y_; }
        [[nodiscard]] float cachedPanelWidth() const { return cached_panel_width_; }
        [[nodiscard]] float cachedHeight() const { return cached_height_; }
        [[nodiscard]] float cachedDpRatio() const { return cached_dp_ratio_; }
        [[nodiscard]] float cachedPlayheadScreenX() const { return cached_playhead_screen_x_; }
        [[nodiscard]] bool isPlayheadInRange() const { return playhead_in_range_; }
        [[nodiscard]] float getDisplayEndTime() const;
        [[nodiscard]] std::optional<sequencer::KeyframeId> hoveredKeyframeId() const;

        [[nodiscard]] TimelineContextMenuState consumeContextMenu();
        [[nodiscard]] TransportContextMenuRequest consumeTransportContextMenu();
        [[nodiscard]] TimeEditRequest consumeTimeEditRequest();
        [[nodiscard]] FocalEditRequest consumeFocalEditRequest();

        void destroyGraphicsResources();
        void reloadResources();

    private:
        void initContext(int width, int height);

        void syncTheme();

        void clearElementCache();
        void cacheElements();
        void updateButtonStates();
        void updatePlayhead();
        void updateTimeDisplay();
        void updateTransportSettings();
        void rebuildKeyframes();
        void rebuildRuler();
        void rebuildEasingStripe(float timeline_x, float timeline_width);
        void rebuildFilmStrip(float timeline_x, float timeline_width,
                              float strip_y, const PanelInputState& input,
                              RenderingManager* rm, SceneManager* sm,
                              gui::FilmStripRenderer& film_strip);
        void rebuildFilmStripDecor(float timeline_width);
        void ensureFilmThumbPool(size_t count);
        void clearFilmThumbPool();
        void unregisterFilmStripSources();
        void updateTimelineGuides(float timeline_x, float timeline_width,
                                  const gui::FilmStripRenderer& film_strip);
        void updateTimelineTooltip(const gui::FilmStripRenderer& film_strip,
                                   const PanelInputState& input);
        void forwardInput(const PanelInputState& input);

        struct Vec2 {
            float x, y;
        };

        void handleTimelineInteraction(const Vec2& pos, float width, float height,
                                       const PanelInputState& input);
        void handleEasingStripeInteraction(float timeline_x, float timeline_width,
                                           const PanelInputState& input);
        void handleFilmStripInteraction(float timeline_x, float timeline_width,
                                        const PanelInputState& input,
                                        gui::FilmStripRenderer& film_strip);

        void clampPanOffset();
        [[nodiscard]] float timeToX(float time, float timeline_x, float timeline_width) const;
        [[nodiscard]] float xToTime(float x, float timeline_x, float timeline_width) const;
        [[nodiscard]] float snapTime(float time) const;

        struct TransportClickListener : Rml::EventListener {
            RmlSequencerPanel* panel = nullptr;
            void ProcessEvent(Rml::Event& event) override;
        };

        struct QualityScrubListener : Rml::EventListener {
            RmlSequencerPanel* panel = nullptr;
            void ProcessEvent(Rml::Event& event) override;
        };

        struct DurationEditListener : Rml::EventListener {
            RmlSequencerPanel* panel = nullptr;
            void ProcessEvent(Rml::Event& event) override;
        };

        void syncQualityScrub();
        void applyQualityFromDrag(float mouse_x);
        void enterQualityEdit();
        void exitQualityEdit(bool commit);

        void syncDurationDisplay();
        void enterDurationEdit();
        void exitDurationEdit(bool commit);

        SequencerController& controller_;
        gui::panels::SequencerUIState& ui_state_;
        gui::RmlUIManager* rml_manager_;
        TransportClickListener transport_listener_;
        QualityScrubListener quality_scrub_listener_;
        DurationEditListener duration_listener_;

        bool quality_scrub_active_ = false;
        bool quality_scrub_dragging_ = false;
        bool quality_scrub_editing_ = false;
        float quality_scrub_start_x_ = 0.0f;
        bool duration_editing_ = false;

        Rml::Context* rml_context_ = nullptr;
        Rml::ElementDocument* document_ = nullptr;
        std::string base_rcss_;
        std::size_t last_theme_signature_ = 0;
        bool has_theme_signature_ = false;

        // Cached DOM elements
        bool elements_cached_ = false;
        Rml::Element* el_panel_ = nullptr;
        Rml::Element* el_floating_header_ = nullptr;
        Rml::Element* el_ruler_ = nullptr;
        Rml::Element* el_track_bar_ = nullptr;
        Rml::Element* el_keyframes_ = nullptr;
        Rml::Element* el_playhead_ = nullptr;
        Rml::Element* el_hint_ = nullptr;
        Rml::Element* el_current_time_ = nullptr;
        Rml::Element* el_duration_ = nullptr;
        Rml::Element* el_play_icon_ = nullptr;
        Rml::Element* el_btn_loop_ = nullptr;
        Rml::Element* el_timeline_ = nullptr;
        Rml::Element* el_header_ = nullptr;
        Rml::Element* el_easing_stripe_ = nullptr;
        Rml::Element* el_easing_segments_ = nullptr;
        Rml::Element* el_easing_curves_ = nullptr;
        Rml::Element* el_easing_indicators_ = nullptr;
        Rml::Element* el_film_strip_panel_ = nullptr;
        Rml::Element* el_film_strip_groove_ = nullptr;
        Rml::Element* el_film_strip_gaps_ = nullptr;
        Rml::Element* el_film_strip_thumbs_ = nullptr;
        Rml::Element* el_film_strip_markers_ = nullptr;
        Rml::Element* el_film_strip_dividers_ = nullptr;
        Rml::Element* el_film_strip_sprockets_top_ = nullptr;
        Rml::Element* el_film_strip_sprockets_bottom_ = nullptr;
        Rml::Element* el_panel_guides_ = nullptr;
        Rml::Element* el_guide_playhead_ = nullptr;
        Rml::Element* el_guide_selected_ = nullptr;
        Rml::Element* el_guide_hovered_ = nullptr;
        Rml::Element* el_guide_strip_hover_ = nullptr;
        Rml::Element* el_timeline_tooltip_ = nullptr;
        gui::RmlTooltipController tooltip_;

        // Transport settings elements
        Rml::Element* el_btn_camera_path_ = nullptr;
        Rml::Element* el_btn_snap_ = nullptr;
        Rml::Element* el_btn_follow_ = nullptr;
        Rml::Element* el_btn_film_strip_ = nullptr;
        Rml::Element* el_btn_preview_ = nullptr;
        Rml::Element* el_speed_label_ = nullptr;
        Rml::Element* el_format_label_ = nullptr;
        Rml::Element* el_resolution_info_ = nullptr;
        Rml::Element* el_quality_scrub_ = nullptr;
        Rml::Element* el_quality_fill_ = nullptr;
        Rml::Element* el_quality_display_ = nullptr;
        Rml::Element* el_quality_input_ = nullptr;
        Rml::Element* el_duration_field_ = nullptr;
        Rml::Element* el_duration_input_ = nullptr;
        Rml::Element* el_btn_equirect_ = nullptr;
        Rml::Element* el_btn_save_ = nullptr;
        Rml::Element* el_btn_load_ = nullptr;
        Rml::Element* el_btn_export_ = nullptr;
        Rml::Element* el_btn_clear_ = nullptr;
        Rml::Element* el_transport_dock_sep_ = nullptr;
        Rml::Element* el_btn_dock_toggle_ = nullptr;
        Rml::Element* el_dock_toggle_label_ = nullptr;
        Rml::Element* el_btn_close_panel_ = nullptr;
        Rml::Element* el_close_panel_label_ = nullptr;

        // Keyframe element pool
        std::vector<Rml::Element*> keyframe_elements_;
        std::vector<Rml::Element*> film_thumb_elements_;
        std::set<std::string> registered_film_strip_sources_;

        // Dirty tracking
        size_t last_keyframe_count_ = 0;
        float last_zoom_level_ = -1.0f;
        float last_pan_offset_ = -1.0f;
        float last_kf_width_ = -1.0f;
        float last_ruler_zoom_ = -1.0f;
        float last_ruler_pan_ = -1.0f;
        float last_ruler_width_ = -1.0f;
        float last_ruler_display_end_ = -1.0f;
        uint64_t last_timeline_revision_ = 0;
        uint64_t last_selection_revision_ = 0;
        uint64_t last_selected_keyframes_signature_ = 0;

        float timelineWidth() const;

        // Layout cache for interaction
        float cached_panel_x_ = 0.0f;
        float cached_panel_y_ = 0.0f;
        float cached_panel_width_ = 0.0f;
        float cached_playhead_screen_x_ = 0.0f;
        bool playhead_in_range_ = false;
        float cached_dp_ratio_ = 1.0f;
        float cached_height_ = panel_config::HEIGHT;
        float cached_total_height_ = panel_config::HEIGHT + panel_config::EASING_STRIPE_HEIGHT;

        // Interaction state
        bool dragging_playhead_ = false;
        bool dragging_keyframe_ = false;
        bool dragged_keyframe_changed_ = false;
        bool film_strip_scrubbing_ = false;
        sequencer::KeyframeId dragged_keyframe_id_ = sequencer::INVALID_KEYFRAME_ID;
        float drag_start_mouse_x_ = 0.0f;
        std::optional<size_t> hovered_keyframe_;
        std::vector<sequencer::KeyframeId> selected_keyframes_;

        float zoom_level_ = 1.0f;
        float pan_offset_ = 0.0f;

        bool film_strip_attached_ = false;
        bool last_film_strip_attached_ = false;
        bool floating_ = false;
        bool last_floating_ = false;

        // Request flags consumed by SequencerUIManager
        bool save_path_requested_ = false;
        bool load_path_requested_ = false;
        bool export_requested_ = false;
        bool dock_toggle_requested_ = false;
        bool close_panel_requested_ = false;
        bool clear_requested_ = false;

        TransportContextMenuRequest transport_ctx_request_;

        // Time editing
        bool editing_keyframe_time_ = false;
        size_t editing_keyframe_index_ = 0;
        std::string time_edit_buffer_;

        // Focal length editing
        bool editing_focal_length_ = false;
        size_t editing_focal_index_ = 0;
        std::string focal_edit_buffer_;

        // Context menu state
        bool context_menu_open_ = false;
        float context_menu_time_ = 0.0f;
        float context_menu_x_ = 0.0f;
        float context_menu_y_ = 0.0f;
        std::optional<size_t> context_menu_keyframe_;

        // Double-click detection
        float last_click_time_ = 0.0f;
        std::optional<size_t> last_clicked_keyframe_;

        bool hovered_ = false;
        bool wants_keyboard_ = false;
        bool last_hovered_ = false;
        std::string last_language_;
    };

} // namespace lfs::vis

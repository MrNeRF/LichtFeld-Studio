/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "gui/rmlui/rml_fbo.hpp"
#include "sequencer_controller.hpp"
#include <RmlUi/Core/EventListener.h>
#include <optional>
#include <set>
#include <string>

namespace Rml {
    class Context;
    class Element;
    class ElementDocument;
} // namespace Rml

namespace lfs::vis::gui {
    class RmlUIManager;
} // namespace lfs::vis::gui

namespace lfs::vis {

    struct PanelInputState {
        float mouse_x = 0.0f;
        float mouse_y = 0.0f;
        bool mouse_down[3] = {};
        bool mouse_clicked[3] = {};
        float mouse_wheel = 0.0f;
        bool key_shift = false;
        bool key_ctrl = false;
        bool key_delete_pressed = false;
        float time = 0.0f;
        float delta_time = 0.0f;
        bool want_capture_mouse = false;
        int screen_w = 0;
        int screen_h = 0;
    };

    namespace panel_config {
        inline constexpr float HEIGHT = 72.0f;
        inline constexpr float PADDING_H = 16.0f;
        inline constexpr float PADDING_BOTTOM = 18.0f;
        inline constexpr float INNER_PADDING = 8.0f;
        inline constexpr float RULER_HEIGHT = 16.0f;
        inline constexpr float TIMELINE_HEIGHT = 24.0f;
        inline constexpr float KEYFRAME_RADIUS = 6.0f;
        inline constexpr float PLAYHEAD_WIDTH = 2.0f;
        inline constexpr float BUTTON_SIZE = 20.0f;
        inline constexpr float BUTTON_SPACING = 4.0f;
        inline constexpr float TRANSPORT_WIDTH = 176.0f;
        inline constexpr float TIME_DISPLAY_WIDTH = 100.0f;

        inline constexpr float MIN_ZOOM = 0.5f;
        inline constexpr float MAX_ZOOM = 4.0f;
        inline constexpr float ZOOM_SPEED = 0.1f;
    } // namespace panel_config

    struct TimelineContextMenuState {
        bool open = false;
        float time = 0.0f;
        std::optional<size_t> keyframe;
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
        RmlSequencerPanel(SequencerController& controller, gui::RmlUIManager* rml_manager);
        ~RmlSequencerPanel();

        RmlSequencerPanel(const RmlSequencerPanel&) = delete;
        RmlSequencerPanel& operator=(const RmlSequencerPanel&) = delete;

        void render(float viewport_x, float viewport_width, float viewport_y_bottom,
                    const PanelInputState& input);

        void setSnapEnabled(bool enabled) { snap_enabled_ = enabled; }
        void setSnapInterval(float interval) { snap_interval_ = interval; }

        void openFocalLengthEdit(size_t index, float current_focal_mm);

        [[nodiscard]] bool isHovered() const { return hovered_; }

        [[nodiscard]] TimelineContextMenuState consumeContextMenu();
        [[nodiscard]] TimeEditRequest consumeTimeEditRequest();
        [[nodiscard]] FocalEditRequest consumeFocalEditRequest();

        void destroyGLResources();

    private:
        void initContext(int width, int height);

        void syncTheme();
        std::string generateThemeRCSS() const;

        void cacheElements();
        void updateButtonStates();
        void updatePlayhead();
        void updateTimeDisplay();
        void rebuildKeyframes();
        void rebuildRuler();
        void forwardInput(const PanelInputState& input);

        struct Vec2 {
            float x, y;
        };

        void handleTimelineInteraction(const Vec2& pos, float width, float height,
                                       const PanelInputState& input);

        [[nodiscard]] float getDisplayEndTime() const;
        [[nodiscard]] float timeToX(float time, float timeline_x, float timeline_width) const;
        [[nodiscard]] float xToTime(float x, float timeline_x, float timeline_width) const;
        [[nodiscard]] float snapTime(float time) const;

        struct TransportClickListener : Rml::EventListener {
            RmlSequencerPanel* panel = nullptr;
            void ProcessEvent(Rml::Event& event) override;
        };

        SequencerController& controller_;
        gui::RmlUIManager* rml_manager_;
        TransportClickListener transport_listener_;

        Rml::Context* rml_context_ = nullptr;
        Rml::ElementDocument* document_ = nullptr;
        std::string base_rcss_;
        float last_synced_text_[4] = {};

        // Cached DOM elements
        bool elements_cached_ = false;
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

        // Keyframe element pool
        std::vector<Rml::Element*> keyframe_elements_;

        // Dirty tracking
        size_t last_keyframe_count_ = 0;
        float last_zoom_level_ = -1.0f;
        float last_pan_offset_ = -1.0f;
        float last_kf_width_ = -1.0f;
        float last_ruler_zoom_ = -1.0f;
        float last_ruler_pan_ = -1.0f;
        float last_ruler_width_ = -1.0f;

        // Layout cache for interaction
        float cached_panel_x_ = 0.0f;
        float cached_panel_y_ = 0.0f;
        float cached_panel_width_ = 0.0f;

        gui::RmlFBO fbo_;

        // Interaction state
        bool dragging_playhead_ = false;
        bool dragging_keyframe_ = false;
        size_t dragged_keyframe_index_ = 0;
        float drag_start_time_ = 0.0f;
        float drag_start_mouse_x_ = 0.0f;
        std::optional<size_t> hovered_keyframe_;
        std::set<size_t> selected_keyframes_;

        float zoom_level_ = 1.0f;
        float pan_offset_ = 0.0f;

        bool snap_enabled_ = false;
        float snap_interval_ = 0.5f;

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
    };

} // namespace lfs::vis

/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "sequencer_controller.hpp"
#include <glad/glad.h>
#include <optional>
#include <set>
#include <string>
#include <imgui.h>

namespace Rml {
    class Context;
    class Element;
    class ElementDocument;
} // namespace Rml

namespace lfs::vis::gui {
    class RmlUIManager;
} // namespace lfs::vis::gui

namespace lfs::vis {

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
        inline constexpr float TRANSPORT_WIDTH = 152.0f;
        inline constexpr float TIME_DISPLAY_WIDTH = 100.0f;

        inline constexpr float MIN_ZOOM = 0.5f;
        inline constexpr float MAX_ZOOM = 4.0f;
        inline constexpr float ZOOM_SPEED = 0.1f;
    } // namespace panel_config

    class RmlSequencerPanel {
    public:
        RmlSequencerPanel(SequencerController& controller, gui::RmlUIManager* rml_manager);
        ~RmlSequencerPanel();

        RmlSequencerPanel(const RmlSequencerPanel&) = delete;
        RmlSequencerPanel& operator=(const RmlSequencerPanel&) = delete;

        void render(float viewport_x, float viewport_width, float viewport_y_bottom);

        void setSnapEnabled(bool enabled) { snap_enabled_ = enabled; }
        void setSnapInterval(float interval) { snap_interval_ = interval; }

        void openFocalLengthEdit(size_t index, float current_focal_mm);

        void destroyGLResources();

    private:
        void initContext(int width, int height);
        void initFBO(int width, int height);
        void destroyFBO();

        void syncTheme();
        std::string generateThemeRCSS() const;

        void cacheElements();
        void updateButtonStates();
        void updatePlayhead();
        void updateTimeDisplay();
        void rebuildKeyframes();
        void rebuildRuler();
        void forwardInput();

        void handleTimelineInteraction(const ImVec2& pos, float width, float height);
        void renderTimeEditPopup();
        void renderFocalLengthEditPopup();

        [[nodiscard]] float getDisplayEndTime() const;
        [[nodiscard]] float timeToX(float time, float timeline_x, float timeline_width) const;
        [[nodiscard]] float xToTime(float x, float timeline_x, float timeline_width) const;
        [[nodiscard]] float snapTime(float time) const;

        SequencerController& controller_;
        gui::RmlUIManager* rml_manager_;

        Rml::Context* rml_context_ = nullptr;
        Rml::ElementDocument* document_ = nullptr;
        std::string base_rcss_;
        ImVec4 last_synced_text_{};

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

        GLuint fbo_ = 0;
        GLuint fbo_texture_ = 0;
        GLuint fbo_depth_stencil_ = 0;
        int fbo_width_ = 0;
        int fbo_height_ = 0;

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

        // Time editing popup (ImGui modal)
        bool editing_keyframe_time_ = false;
        size_t editing_keyframe_index_ = 0;
        char time_edit_buffer_[32] = {};

        // Focal length editing popup (ImGui modal)
        bool editing_focal_length_ = false;
        size_t editing_focal_index_ = 0;
        char focal_edit_buffer_[32] = {};

        // Context menu state
        bool context_menu_open_ = false;
        float context_menu_time_ = 0.0f;
        ImVec2 context_menu_pos_ = {0, 0};
        std::optional<size_t> context_menu_keyframe_;

        // Double-click detection
        float last_click_time_ = 0.0f;
        std::optional<size_t> last_clicked_keyframe_;
    };

} // namespace lfs::vis

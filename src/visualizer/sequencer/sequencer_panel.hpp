/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "sequencer_controller.hpp"
#include <set>
#include <imgui.h>

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

    class SequencerPanel {
    public:
        explicit SequencerPanel(SequencerController& controller);
        void render(float viewport_x, float viewport_width, float viewport_y_bottom);

        // External settings
        void setSnapEnabled(bool enabled) { snap_enabled_ = enabled; }
        void setSnapInterval(float interval) { snap_interval_ = interval; }

        void openFocalLengthEdit(size_t index, float current_focal_mm);

    private:
        void renderTransportControls(const ImVec2& pos, float height);
        void renderTimeline(const ImVec2& pos, float width, float height);
        void renderTimeRuler(ImDrawList* dl, const ImVec2& pos, float width);
        void renderTimeDisplay(const ImVec2& pos, float height);
        void renderTimeEditPopup();
        void renderFocalLengthEditPopup();

        void drawKeyframeMarker(ImDrawList* dl, const ImVec2& pos, bool selected, bool hovered, float time, bool is_loop_point) const;
        void drawPlayhead(ImDrawList* dl, const ImVec2& top, const ImVec2& bottom) const;

        [[nodiscard]] float getDisplayEndTime() const;
        [[nodiscard]] float timeToX(float time, float timeline_x, float timeline_width) const;
        [[nodiscard]] float xToTime(float x, float timeline_x, float timeline_width) const;
        [[nodiscard]] float snapTime(float time) const;

        // Playhead/keyframe dragging
        bool dragging_playhead_ = false;
        bool dragging_keyframe_ = false;
        size_t dragged_keyframe_index_ = 0;
        float drag_start_time_ = 0.0f;
        float drag_start_mouse_x_ = 0.0f;
        std::optional<size_t> hovered_keyframe_;

        // Multi-selection
        std::set<size_t> selected_keyframes_;

        // Timeline zoom
        float zoom_level_ = 1.0f;
        float pan_offset_ = 0.0f;

        // Snap to grid
        bool snap_enabled_ = false;
        float snap_interval_ = 0.5f;

        // Time editing popup
        bool editing_keyframe_time_ = false;
        size_t editing_keyframe_index_ = 0;
        char time_edit_buffer_[32] = {};

        // Focal length editing popup
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

        SequencerController& controller_;
    };

} // namespace lfs::vis

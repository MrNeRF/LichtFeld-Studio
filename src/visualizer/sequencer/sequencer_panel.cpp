/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "sequencer_panel.hpp"
#include "core/event_bridge/localization_manager.hpp"
#include "core/events.hpp"
#include "core/services.hpp"
#include "gui/string_keys.hpp"
#include "rendering/render_constants.hpp"
#include "theme/theme.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <format>

namespace lfs::vis {

    namespace {
        constexpr float DEFAULT_TIMELINE_DURATION = 10.0f;
        constexpr float TIMELINE_END_PADDING = 1.0f;
        constexpr float MIN_KEYFRAME_SPACING = 0.1f;
        constexpr float ICON_SIZE = 7.0f;
        constexpr float PLAY_ICON_SIZE = 8.0f;
        constexpr float PAUSE_BAR_W = 2.5f;
        constexpr float PAUSE_BAR_H = 9.0f;
        constexpr float PAUSE_GAP = 3.0f;
        constexpr float PLAYHEAD_HANDLE_SIZE = 7.0f;
        constexpr float TIMELINE_ROUNDING = 4.0f;
        constexpr float SKIP_ICON_SIZE = 5.0f;
        constexpr float MAJOR_TICK_HEIGHT = 8.0f;
        constexpr float MINOR_TICK_HEIGHT = 4.0f;
        constexpr float DOUBLE_CLICK_TIME = 0.3f;
        constexpr float DRAG_THRESHOLD_PX = 3.0f;

        // Loop marker (infinity symbol)
        constexpr float LOOP_MARKER_SCALE = 0.7f;
        constexpr float LOOP_MARKER_OFFSET = 0.5f;
        constexpr float LOOP_MARKER_RADIUS = 0.6f;
        constexpr int LOOP_MARKER_SEGMENTS = 8;
        constexpr float LOOP_MARKER_OUTLINE = 1.5f;
        constexpr float TWO_PI = 6.283185f;

        constexpr ImGuiWindowFlags PANEL_FLAGS =
            ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
            ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoBringToFrontOnFocus |
            ImGuiWindowFlags_NoFocusOnAppearing;

        constexpr const char* EASING_NAMES[] = {"Linear", "Ease In", "Ease Out", "Ease In-Out"};

        [[nodiscard]] std::string formatTime(const float seconds) {
            const int mins = static_cast<int>(seconds) / 60;
            const float secs = seconds - static_cast<float>(mins * 60);
            return std::format("{}:{:05.2f}", mins, secs);
        }

        [[nodiscard]] std::string formatTimeShort(const float seconds) {
            const int mins = static_cast<int>(seconds) / 60;
            const int secs = static_cast<int>(seconds) % 60;
            if (mins > 0) {
                return std::format("{}:{:02d}", mins, secs);
            }
            return std::format("{}s", secs);
        }
    } // namespace

    using namespace panel_config;

    SequencerPanel::SequencerPanel(SequencerController& controller)
        : controller_(controller) {}

    void SequencerPanel::render(const float viewport_x, const float viewport_width, const float viewport_y_bottom) {
        const auto& t = theme();

        const float panel_x = viewport_x + PADDING_H;
        const float panel_width = viewport_width - 2.0f * PADDING_H;
        const ImVec2 panel_pos = {panel_x, viewport_y_bottom - HEIGHT - PADDING_BOTTOM};
        const ImVec2 panel_size = {panel_width, HEIGHT};

        ImGui::SetNextWindowPos(panel_pos);
        ImGui::SetNextWindowSize(panel_size);
        ImGui::SetNextWindowBgAlpha(0.95f);
        ImGui::PushStyleColor(ImGuiCol_WindowBg, toU32(t.palette.surface));
        ImGui::PushStyleColor(ImGuiCol_Border, toU32WithAlpha(t.palette.border, 0.4f));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, t.sizes.window_rounding);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 1.0f);

        if (!ImGui::Begin("##SequencerPanel", nullptr, PANEL_FLAGS)) {
            ImGui::PopStyleVar(2);
            ImGui::PopStyleColor(2);
            ImGui::End();
            return;
        }

        const float content_width = panel_size.x - 2.0f * INNER_PADDING;
        const float timeline_width = content_width - TRANSPORT_WIDTH - TIME_DISPLAY_WIDTH;
        const float content_height = HEIGHT - 2.0f * INNER_PADDING;

        const ImVec2 transport_pos = {panel_pos.x + INNER_PADDING, panel_pos.y + INNER_PADDING};
        const ImVec2 timeline_pos = {transport_pos.x + TRANSPORT_WIDTH, panel_pos.y + INNER_PADDING};
        const ImVec2 time_display_pos = {timeline_pos.x + timeline_width, panel_pos.y + INNER_PADDING};

        renderTransportControls(transport_pos, content_height);
        renderTimeline(timeline_pos, timeline_width, content_height);
        renderTimeDisplay(time_display_pos, content_height);
        renderTimeEditPopup();
        renderFocalLengthEditPopup();

        ImGui::End();
        ImGui::PopStyleVar(2);
        ImGui::PopStyleColor(2);
    }

    void SequencerPanel::renderTransportControls(const ImVec2& pos, const float height) {
        const auto& t = theme();
        const float y_center = pos.y + height / 2.0f;
        const float btn_half = BUTTON_SIZE / 2.0f;
        float x_offset = 0.0f;

        ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, btn_half);
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, {0, 0});
        ImGui::PushStyleColor(ImGuiCol_Button, t.button_normal());
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, t.button_hovered());
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, t.button_active());

        // |◀ First keyframe
        ImGui::SetCursorScreenPos({pos.x + x_offset, y_center - btn_half});
        if (ImGui::Button("##first", {BUTTON_SIZE, BUTTON_SIZE})) {
            controller_.seekToFirstKeyframe();
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Go to first keyframe");
        }
        {
            ImDrawList* const dl = ImGui::GetWindowDrawList();
            const ImVec2 center = {pos.x + x_offset + btn_half, y_center};
            dl->AddRectFilled(
                {center.x - SKIP_ICON_SIZE - 1, center.y - SKIP_ICON_SIZE},
                {center.x - SKIP_ICON_SIZE + 1, center.y + SKIP_ICON_SIZE},
                t.text_u32());
            dl->AddTriangleFilled(
                {center.x + SKIP_ICON_SIZE, center.y - SKIP_ICON_SIZE},
                {center.x + SKIP_ICON_SIZE, center.y + SKIP_ICON_SIZE},
                {center.x - SKIP_ICON_SIZE + 2, center.y},
                t.text_u32());
        }
        x_offset += BUTTON_SIZE + BUTTON_SPACING;

        // ■ Stop
        ImGui::SetCursorScreenPos({pos.x + x_offset, y_center - btn_half});
        if (ImGui::Button("##stop", {BUTTON_SIZE, BUTTON_SIZE})) {
            controller_.stop();
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Stop");
        }
        {
            ImDrawList* const dl = ImGui::GetWindowDrawList();
            const ImVec2 center = {pos.x + x_offset + btn_half, y_center};
            dl->AddRectFilled(
                {center.x - ICON_SIZE / 2, center.y - ICON_SIZE / 2},
                {center.x + ICON_SIZE / 2, center.y + ICON_SIZE / 2},
                t.text_u32());
        }
        x_offset += BUTTON_SIZE + BUTTON_SPACING;

        // ▶/❚❚ Play/Pause
        ImGui::SetCursorScreenPos({pos.x + x_offset, y_center - btn_half});
        if (ImGui::Button("##playpause", {BUTTON_SIZE, BUTTON_SIZE})) {
            controller_.togglePlayPause();
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip(controller_.isPlaying() ? "Pause (Space)" : "Play (Space)");
        }
        {
            ImDrawList* const dl = ImGui::GetWindowDrawList();
            const ImVec2 center = {pos.x + x_offset + btn_half, y_center};

            if (controller_.isPlaying()) {
                dl->AddRectFilled(
                    {center.x - PAUSE_GAP - PAUSE_BAR_W, center.y - PAUSE_BAR_H / 2},
                    {center.x - PAUSE_GAP, center.y + PAUSE_BAR_H / 2},
                    t.text_u32());
                dl->AddRectFilled(
                    {center.x + PAUSE_GAP - PAUSE_BAR_W, center.y - PAUSE_BAR_H / 2},
                    {center.x + PAUSE_GAP, center.y + PAUSE_BAR_H / 2},
                    t.text_u32());
            } else {
                dl->AddTriangleFilled(
                    {center.x - PLAY_ICON_SIZE * 0.4f, center.y - PLAY_ICON_SIZE},
                    {center.x - PLAY_ICON_SIZE * 0.4f, center.y + PLAY_ICON_SIZE},
                    {center.x + PLAY_ICON_SIZE * 0.8f, center.y},
                    t.text_u32());
            }
        }
        x_offset += BUTTON_SIZE + BUTTON_SPACING;

        // ▶| Last keyframe
        ImGui::SetCursorScreenPos({pos.x + x_offset, y_center - btn_half});
        if (ImGui::Button("##last", {BUTTON_SIZE, BUTTON_SIZE})) {
            controller_.seekToLastKeyframe();
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Go to last keyframe");
        }
        {
            ImDrawList* const dl = ImGui::GetWindowDrawList();
            const ImVec2 center = {pos.x + x_offset + btn_half, y_center};
            dl->AddTriangleFilled(
                {center.x - SKIP_ICON_SIZE, center.y - SKIP_ICON_SIZE},
                {center.x - SKIP_ICON_SIZE, center.y + SKIP_ICON_SIZE},
                {center.x + SKIP_ICON_SIZE - 2, center.y},
                t.text_u32());
            dl->AddRectFilled(
                {center.x + SKIP_ICON_SIZE - 1, center.y - SKIP_ICON_SIZE},
                {center.x + SKIP_ICON_SIZE + 1, center.y + SKIP_ICON_SIZE},
                t.text_u32());
        }
        x_offset += BUTTON_SIZE + BUTTON_SPACING + 4.0f;

        // ↻ Loop toggle
        const bool is_looping = controller_.loopMode() != LoopMode::ONCE;
        if (is_looping) {
            ImGui::PushStyleColor(ImGuiCol_Button, t.primary_u32());
        }
        ImGui::SetCursorScreenPos({pos.x + x_offset, y_center - btn_half});
        if (ImGui::Button("##loop", {BUTTON_SIZE, BUTTON_SIZE})) {
            controller_.toggleLoop();
        }
        if (is_looping) {
            ImGui::PopStyleColor();
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip(is_looping ? "Loop: ON" : "Loop: OFF");
        }
        {
            ImDrawList* const dl = ImGui::GetWindowDrawList();
            const ImVec2 center = {pos.x + x_offset + btn_half, y_center};
            const float r = ICON_SIZE * 0.8f;
            const ImU32 col = is_looping ? toU32(t.palette.text) : t.text_dim_u32();
            dl->PathArcTo(center, r, 0.5f, 2.5f, 8);
            dl->PathStroke(col, 0, 1.5f);
            dl->PathArcTo(center, r, 3.64f, 5.64f, 8);
            dl->PathStroke(col, 0, 1.5f);
            const float ah = 3.0f;
            dl->AddTriangleFilled(
                {center.x + r - ah, center.y - ah},
                {center.x + r + ah, center.y},
                {center.x + r - ah, center.y + ah},
                col);
            dl->AddTriangleFilled(
                {center.x - r + ah, center.y + ah},
                {center.x - r - ah, center.y},
                {center.x - r + ah, center.y - ah},
                col);
        }
        x_offset += BUTTON_SIZE + BUTTON_SPACING;

        // + Add keyframe
        ImGui::SetCursorScreenPos({pos.x + x_offset, y_center - btn_half});
        if (ImGui::Button("+##addkf", {BUTTON_SIZE, BUTTON_SIZE})) {
            lfs::core::events::cmd::SequencerAddKeyframe{}.emit();
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Add keyframe (K)");
        }

        ImGui::PopStyleColor(3);
        ImGui::PopStyleVar(2);
    }

    void SequencerPanel::renderTimeline(const ImVec2& pos, const float width, const float height) {
        const auto& t = theme();
        ImDrawList* const dl = ImGui::GetWindowDrawList();

        const float ruler_y = pos.y;
        const float timeline_y = pos.y + RULER_HEIGHT + 4.0f;
        const float timeline_height = height - RULER_HEIGHT - 4.0f;
        const float bar_half = std::min(timeline_height, TIMELINE_HEIGHT) / 2.0f;
        const float y_center = timeline_y + timeline_height / 2.0f;

        const ImVec2 bar_min = {pos.x, y_center - bar_half};
        const ImVec2 bar_max = {pos.x + width, y_center + bar_half};

        dl->AddRectFilled(bar_min, bar_max, toU32WithAlpha(t.palette.background, 0.8f), TIMELINE_ROUNDING);
        dl->AddRect(bar_min, bar_max, toU32WithAlpha(t.palette.border, 0.3f), TIMELINE_ROUNDING, 0, 1.0f);

        const auto& timeline = controller_.timeline();

        renderTimeRuler(dl, {pos.x, ruler_y}, width);

        if (timeline.empty()) {
            constexpr const char* HINT = "Position camera and press K to add keyframes";
            const ImVec2 text_size = ImGui::CalcTextSize(HINT);
            dl->AddText({pos.x + (width - text_size.x) / 2, y_center - text_size.y / 2},
                        toU32WithAlpha(t.palette.text_dim, 0.5f), HINT);
            return;
        }

        const ImVec2 mouse = ImGui::GetMousePos();
        const bool mouse_in_timeline = mouse.x >= bar_min.x && mouse.x <= bar_max.x &&
                                       mouse.y >= bar_min.y - RULER_HEIGHT && mouse.y <= bar_max.y;

        // Timeline zoom with scroll wheel
        if (mouse_in_timeline && !ImGui::GetIO().WantCaptureMouse) {
            const float wheel = ImGui::GetIO().MouseWheel;
            if (std::abs(wheel) > 0.01f) {
                const float old_zoom = zoom_level_;
                zoom_level_ = std::clamp(zoom_level_ + wheel * ZOOM_SPEED, MIN_ZOOM, MAX_ZOOM);

                // Adjust pan to zoom towards mouse position
                if (zoom_level_ != old_zoom) {
                    const float mouse_time = xToTime(mouse.x, pos.x, width);
                    pan_offset_ += (mouse_time - pan_offset_) * (1.0f - old_zoom / zoom_level_) * 0.5f;
                    pan_offset_ = std::max(0.0f, pan_offset_);
                }
            }
        }

        // Playhead dragging
        if (ImGui::IsMouseClicked(ImGuiMouseButton_Left) && mouse_in_timeline && !dragging_keyframe_ &&
            !hovered_keyframe_.has_value()) {
            dragging_playhead_ = true;
            controller_.beginScrub();
        }
        if (dragging_playhead_) {
            if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
                float time = xToTime(mouse.x, pos.x, width);
                time = std::clamp(time, 0.0f, timeline.endTime());
                if (snap_enabled_) {
                    time = snapTime(time);
                }
                controller_.scrub(time);
            } else {
                dragging_playhead_ = false;
                controller_.endScrub();
            }
        }

        // Keyframes
        hovered_keyframe_ = std::nullopt;
        const auto& keyframes = timeline.keyframes();
        for (size_t i = 0; i < keyframes.size(); ++i) {
            const float x = timeToX(keyframes[i].time, pos.x, width);
            const ImVec2 kf_pos = {x, y_center};

            const float dist = std::abs(mouse.x - x);
            const bool hovered = mouse_in_timeline && dist < KEYFRAME_RADIUS * 2;
            if (hovered) {
                hovered_keyframe_ = i;
            }

            const bool selected = controller_.selectedKeyframe() == i ||
                                  selected_keyframes_.contains(i);
            const bool is_first = (i == 0);
            drawKeyframeMarker(dl, kf_pos, selected, hovered, keyframes[i].time, keyframes[i].is_loop_point);

            if (hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                const float current_time = static_cast<float>(ImGui::GetTime());

                // Double-click detection
                if (last_clicked_keyframe_ == i &&
                    (current_time - last_click_time_) < DOUBLE_CLICK_TIME) {
                    // Double-click: open time edit popup
                    editing_keyframe_time_ = true;
                    editing_keyframe_index_ = i;
                    std::snprintf(time_edit_buffer_, sizeof(time_edit_buffer_), "%.2f", keyframes[i].time);
                    last_clicked_keyframe_ = std::nullopt;
                } else {
                    last_click_time_ = current_time;
                    last_clicked_keyframe_ = i;

                    // Multi-select with Shift
                    if (ImGui::GetIO().KeyShift && controller_.hasSelection()) {
                        const size_t first_sel = *controller_.selectedKeyframe();
                        const size_t lo = std::min(first_sel, i);
                        const size_t hi = std::max(first_sel, i);
                        selected_keyframes_.clear();
                        for (size_t j = lo; j <= hi; ++j) {
                            selected_keyframes_.insert(j);
                        }
                    } else if (ImGui::GetIO().KeyCtrl) {
                        // Toggle selection with Ctrl
                        if (selected_keyframes_.contains(i)) {
                            selected_keyframes_.erase(i);
                        } else {
                            selected_keyframes_.insert(i);
                        }
                    } else {
                        selected_keyframes_.clear();
                        lfs::core::events::cmd::SequencerSelectKeyframe{.keyframe_index = i}.emit();
                        if (!is_first) {
                            dragging_keyframe_ = true;
                            dragged_keyframe_index_ = i;
                            drag_start_time_ = keyframes[i].time;
                            drag_start_mouse_x_ = mouse.x;
                        } else {
                            lfs::core::events::cmd::SequencerGoToKeyframe{.keyframe_index = i}.emit();
                        }
                    }
                }
            }
        }

        // Keyframe dragging with undo support
        if (dragging_keyframe_) {
            if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
                float new_time = xToTime(mouse.x, pos.x, width);
                new_time = std::max(new_time, MIN_KEYFRAME_SPACING);
                if (snap_enabled_) {
                    new_time = snapTime(new_time);
                }
                controller_.timeline().setKeyframeTime(dragged_keyframe_index_, new_time, false);
            } else {
                if (std::abs(mouse.x - drag_start_mouse_x_) < DRAG_THRESHOLD_PX) {
                    lfs::core::events::cmd::SequencerGoToKeyframe{.keyframe_index = dragged_keyframe_index_}.emit();
                }
                controller_.timeline().sortKeyframes();
                dragging_keyframe_ = false;
            }
        }

        // Delete keyframe(s)
        if ((controller_.hasSelection() || !selected_keyframes_.empty()) &&
            ImGui::IsKeyPressed(ImGuiKey_Delete)) {
            // Delete selected keyframes (in reverse order to maintain indices)
            std::vector<size_t> to_delete;
            if (!selected_keyframes_.empty()) {
                to_delete.assign(selected_keyframes_.begin(), selected_keyframes_.end());
            } else if (controller_.hasSelection()) {
                to_delete.push_back(*controller_.selectedKeyframe());
            }

            // Sort in reverse order
            std::sort(to_delete.begin(), to_delete.end(), std::greater<>());

            for (const size_t idx : to_delete) {
                if (idx == 0)
                    continue;
                controller_.timeline().removeKeyframe(idx);
            }
            selected_keyframes_.clear();
            controller_.deselectKeyframe();
        }

        // Context menu
        if (mouse_in_timeline && ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
            context_menu_time_ = xToTime(mouse.x, pos.x, width);
            context_menu_keyframe_ = hovered_keyframe_;
            context_menu_open_ = true;
            context_menu_pos_ = mouse;
            ImGui::OpenPopup("TimelineContextMenu");
        }

        ImGui::SetNextWindowPos(context_menu_pos_, ImGuiCond_Always, {0.0f, 1.0f});
        if (ImGui::BeginPopup("TimelineContextMenu")) {
            if (context_menu_keyframe_.has_value()) {
                const size_t idx = *context_menu_keyframe_;
                const bool is_first = (idx == 0);

                if (ImGui::MenuItem("Update to Current View", "U")) {
                    lfs::core::events::cmd::SequencerSelectKeyframe{.keyframe_index = idx}.emit();
                    lfs::core::events::cmd::SequencerUpdateKeyframe{}.emit();
                }
                if (ImGui::MenuItem("Go to Keyframe")) {
                    lfs::core::events::cmd::SequencerGoToKeyframe{.keyframe_index = idx}.emit();
                }
                if (ImGui::MenuItem("Edit Time...", nullptr)) {
                    editing_keyframe_time_ = true;
                    editing_keyframe_index_ = idx;
                    std::snprintf(time_edit_buffer_, sizeof(time_edit_buffer_), "%.2f", keyframes[idx].time);
                }
                if (ImGui::MenuItem(LOC(lichtfeld::Strings::Sequencer::EDIT_FOCAL_LENGTH), nullptr)) {
                    editing_focal_length_ = true;
                    editing_focal_index_ = idx;
                    std::snprintf(focal_edit_buffer_, sizeof(focal_edit_buffer_), "%.1f", keyframes[idx].focal_length_mm);
                }

                // Easing submenu (only for non-last keyframes - easing controls outgoing segment)
                const bool is_last = (idx == keyframes.size() - 1);
                if (ImGui::BeginMenu("Easing", !is_last)) {
                    const auto current_easing = keyframes[idx].easing;
                    for (int e = 0; e < 4; ++e) {
                        const auto easing = static_cast<sequencer::EasingType>(e);
                        if (ImGui::MenuItem(EASING_NAMES[e], nullptr, current_easing == easing)) {
                            if (current_easing != easing) {
                                controller_.timeline().setKeyframeEasing(idx, easing);
                            }
                        }
                    }
                    ImGui::EndMenu();
                }
                if (is_last && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
                    ImGui::SetTooltip("Easing controls outgoing motion\n(last keyframe has no outgoing segment)");
                }

                ImGui::Separator();
                if (ImGui::MenuItem("Delete Keyframe", "Del", false, !is_first)) {
                    controller_.timeline().removeKeyframe(idx);
                }
            } else {
                if (ImGui::MenuItem("Add Keyframe Here", "K")) {
                    lfs::core::events::cmd::SequencerAddKeyframe{}.emit();
                }
            }
            ImGui::EndPopup();
        } else {
            context_menu_open_ = false;
        }

        // Playhead
        const float playhead_x = timeToX(controller_.playhead(), pos.x, width);
        drawPlayhead(dl, {playhead_x, ruler_y}, {playhead_x, bar_max.y + 4});
    }

    void SequencerPanel::renderTimeEditPopup() {
        if (!editing_keyframe_time_)
            return;

        // Open popup if not already open (must be done outside context menu scope)
        if (!ImGui::IsPopupOpen("EditKeyframeTime")) {
            ImGui::OpenPopup("EditKeyframeTime");
        }

        ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Appearing, {0.5f, 0.5f});
        if (ImGui::BeginPopupModal("EditKeyframeTime", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::Text("Edit Keyframe Time");
            ImGui::Separator();

            auto applyTimeChange = [this]() {
                const float new_time = std::strtof(time_edit_buffer_, nullptr);
                if (new_time >= 0.0f) {
                    const auto& kfs = controller_.timeline().keyframes();
                    if (editing_keyframe_index_ < kfs.size()) {
                        controller_.timeline().setKeyframeTime(editing_keyframe_index_, new_time);
                    }
                }
            };

            ImGui::SetNextItemWidth(120);
            if (ImGui::InputText("Time (s)", time_edit_buffer_, sizeof(time_edit_buffer_),
                                 ImGuiInputTextFlags_CharsDecimal | ImGuiInputTextFlags_EnterReturnsTrue)) {
                applyTimeChange();
                editing_keyframe_time_ = false;
                ImGui::CloseCurrentPopup();
            }

            if (ImGui::Button("OK", {60, 0})) {
                applyTimeChange();
                editing_keyframe_time_ = false;
                ImGui::CloseCurrentPopup();
            }

            ImGui::SameLine();
            if (ImGui::Button("Cancel", {60, 0})) {
                editing_keyframe_time_ = false;
                ImGui::CloseCurrentPopup();
            }

            ImGui::EndPopup();
        }
    }

    void SequencerPanel::openFocalLengthEdit(const size_t index, const float current_focal_mm) {
        editing_focal_length_ = true;
        editing_focal_index_ = index;
        std::snprintf(focal_edit_buffer_, sizeof(focal_edit_buffer_), "%.1f", current_focal_mm);
    }

    void SequencerPanel::renderFocalLengthEditPopup() {
        if (!editing_focal_length_)
            return;

        if (!ImGui::IsPopupOpen("EditKeyframeFocalLength")) {
            ImGui::OpenPopup("EditKeyframeFocalLength");
        }

        ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Appearing, {0.5f, 0.5f});
        if (ImGui::BeginPopupModal("EditKeyframeFocalLength", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::TextUnformatted(LOC(lichtfeld::Strings::Sequencer::EDIT_FOCAL_LENGTH_TITLE));
            ImGui::Separator();
            auto applyFocalChange = [this]() {
                float new_focal = std::strtof(focal_edit_buffer_, nullptr);
                new_focal = std::clamp(new_focal,
                                       lfs::rendering::MIN_FOCAL_LENGTH_MM,
                                       lfs::rendering::MAX_FOCAL_LENGTH_MM);
                if (editing_focal_index_ < controller_.timeline().keyframes().size()) {
                    controller_.timeline().setKeyframeFocalLength(editing_focal_index_, new_focal);
                    controller_.updateLoopKeyframe();
                }
            };

            ImGui::SetNextItemWidth(120);
            if (ImGui::InputText(LOC(lichtfeld::Strings::Sequencer::FOCAL_LENGTH_MM), focal_edit_buffer_, sizeof(focal_edit_buffer_),
                                 ImGuiInputTextFlags_CharsDecimal | ImGuiInputTextFlags_EnterReturnsTrue)) {
                applyFocalChange();
                editing_focal_length_ = false;
                ImGui::CloseCurrentPopup();
            }

            if (ImGui::Button(LOC(lichtfeld::Strings::Common::OK), {60, 0})) {
                applyFocalChange();
                editing_focal_length_ = false;
                ImGui::CloseCurrentPopup();
            }

            ImGui::SameLine();
            if (ImGui::Button(LOC(lichtfeld::Strings::Common::CANCEL), {60, 0})) {
                editing_focal_length_ = false;
                ImGui::CloseCurrentPopup();
            }

            ImGui::EndPopup();
        }
    }

    void SequencerPanel::renderTimeRuler(ImDrawList* const dl, const ImVec2& pos, const float width) {
        const auto& t = theme();
        const float end_time = getDisplayEndTime();

        float major_interval = 1.0f;
        if (end_time > 60.0f) {
            major_interval = 10.0f;
        } else if (end_time > 30.0f) {
            major_interval = 5.0f;
        } else if (end_time > 10.0f) {
            major_interval = 2.0f;
        } else if (end_time <= 2.0f) {
            major_interval = 0.5f;
        }

        // Adjust for zoom
        major_interval /= zoom_level_;

        const float minor_interval = major_interval / 4.0f;

        for (float t_val = 0.0f; t_val <= end_time; t_val += minor_interval) {
            const float x = pos.x + (t_val / end_time) * width;
            if (x < pos.x || x > pos.x + width)
                continue;

            const bool is_major = std::fmod(t_val + 0.001f, major_interval) < 0.01f;

            if (is_major) {
                dl->AddLine({x, pos.y + RULER_HEIGHT - MAJOR_TICK_HEIGHT},
                            {x, pos.y + RULER_HEIGHT},
                            t.text_dim_u32(), 1.0f);

                const std::string label = formatTimeShort(t_val);
                const ImVec2 text_size = ImGui::CalcTextSize(label.c_str());
                dl->AddText({x - text_size.x / 2, pos.y}, t.text_dim_u32(), label.c_str());
            } else {
                dl->AddLine({x, pos.y + RULER_HEIGHT - MINOR_TICK_HEIGHT},
                            {x, pos.y + RULER_HEIGHT},
                            toU32WithAlpha(t.palette.text_dim, 0.5f), 1.0f);
            }
        }

        dl->AddLine({pos.x, pos.y + RULER_HEIGHT},
                    {pos.x + width, pos.y + RULER_HEIGHT},
                    toU32WithAlpha(t.palette.border, 0.5f), 1.0f);
    }

    void SequencerPanel::renderTimeDisplay(const ImVec2& pos, const float height) {
        const auto& t = theme();
        const float y_center = pos.y + height / 2.0f;

        const std::string time_str = formatTime(controller_.playhead());
        const ImVec2 text_size = ImGui::CalcTextSize(time_str.c_str());

        ImDrawList* const dl = ImGui::GetWindowDrawList();
        dl->AddText({pos.x + 8.0f, y_center - text_size.y / 2}, t.text_u32(), time_str.c_str());

        if (!controller_.timeline().empty()) {
            const std::string dur_str = " / " + formatTime(controller_.timeline().endTime());
            dl->AddText({pos.x + 8.0f + text_size.x, y_center - text_size.y / 2}, t.text_dim_u32(), dur_str.c_str());
        }
    }

    void SequencerPanel::drawKeyframeMarker(ImDrawList* const dl, const ImVec2& pos,
                                            const bool selected, const bool hovered,
                                            const float time, const bool is_loop_point) const {
        const auto& t = theme();
        const auto base_color = is_loop_point ? t.palette.info : t.palette.primary;

        ImU32 fill = toU32(base_color);
        if (selected) {
            fill = toU32(lighten(base_color, 0.2f));
        } else if (hovered) {
            fill = toU32(lighten(base_color, 0.1f));
        }

        if (is_loop_point) {
            const float r = KEYFRAME_RADIUS * LOOP_MARKER_SCALE;
            const float offset = r * LOOP_MARKER_OFFSET;
            const float arc_r = r * LOOP_MARKER_RADIUS;

            dl->PathArcTo({pos.x - offset, pos.y}, arc_r, 0.0f, TWO_PI, LOOP_MARKER_SEGMENTS);
            dl->PathFillConvex(fill);
            dl->PathArcTo({pos.x + offset, pos.y}, arc_r, 0.0f, TWO_PI, LOOP_MARKER_SEGMENTS);
            dl->PathFillConvex(fill);

            if (selected) {
                const ImU32 outline = toU32(t.palette.text);
                dl->AddCircle({pos.x - offset, pos.y}, arc_r + LOOP_MARKER_OUTLINE, outline, LOOP_MARKER_SEGMENTS, LOOP_MARKER_OUTLINE);
                dl->AddCircle({pos.x + offset, pos.y}, arc_r + LOOP_MARKER_OUTLINE, outline, LOOP_MARKER_SEGMENTS, LOOP_MARKER_OUTLINE);
            }
        } else {
            dl->AddQuadFilled(
                {pos.x, pos.y - KEYFRAME_RADIUS},
                {pos.x + KEYFRAME_RADIUS, pos.y},
                {pos.x, pos.y + KEYFRAME_RADIUS},
                {pos.x - KEYFRAME_RADIUS, pos.y},
                fill);

            if (selected) {
                dl->AddQuad(
                    {pos.x, pos.y - KEYFRAME_RADIUS - 1},
                    {pos.x + KEYFRAME_RADIUS + 1, pos.y},
                    {pos.x, pos.y + KEYFRAME_RADIUS + 1},
                    {pos.x - KEYFRAME_RADIUS - 1, pos.y},
                    toU32(t.palette.text), LOOP_MARKER_OUTLINE);
            }
        }

        if (hovered) {
            const char* tooltip = is_loop_point
                                      ? "Loop Point @ %s (returns to start)"
                                      : "Keyframe @ %s (double-click to edit)";
            ImGui::SetTooltip(tooltip, formatTime(time).c_str());
        }
    }

    void SequencerPanel::drawPlayhead(ImDrawList* const dl, const ImVec2& top, const ImVec2& bottom) const {
        const auto& t = theme();
        dl->AddLine(top, bottom, t.error_u32(), PLAYHEAD_WIDTH);
        dl->AddTriangleFilled(
            {top.x - PLAYHEAD_HANDLE_SIZE, top.y},
            {top.x + PLAYHEAD_HANDLE_SIZE, top.y},
            {top.x, top.y + PLAYHEAD_HANDLE_SIZE},
            t.error_u32());
    }

    float SequencerPanel::getDisplayEndTime() const {
        const auto& timeline = controller_.timeline();
        if (timeline.size() < 2) {
            return DEFAULT_TIMELINE_DURATION / zoom_level_;
        }
        return std::max(timeline.endTime() + TIMELINE_END_PADDING, DEFAULT_TIMELINE_DURATION) / zoom_level_;
    }

    float SequencerPanel::timeToX(const float time, const float timeline_x, const float timeline_width) const {
        const float end = getDisplayEndTime();
        const float adjusted_time = (time - pan_offset_) * zoom_level_;
        return timeline_x + (adjusted_time / (end * zoom_level_)) * timeline_width;
    }

    float SequencerPanel::xToTime(const float x, const float timeline_x, const float timeline_width) const {
        const float end = getDisplayEndTime();
        const float t = ((x - timeline_x) / timeline_width) * end;
        return t / zoom_level_ + pan_offset_;
    }

    float SequencerPanel::snapTime(const float time) const {
        if (!snap_enabled_ || snap_interval_ <= 0.0f) {
            return time;
        }
        return std::round(time / snap_interval_) * snap_interval_;
    }

} // namespace lfs::vis

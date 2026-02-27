/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// clang-format off
#include <glad/glad.h>
// clang-format on

#include "sequencer/rml_sequencer_panel.hpp"
#include "core/event_bridge/localization_manager.hpp"
#include "core/events.hpp"
#include "core/logger.hpp"
#include "gui/rmlui/rml_theme.hpp"
#include "gui/rmlui/rmlui_manager.hpp"
#include "gui/rmlui/rmlui_render_interface.hpp"
#include "gui/string_keys.hpp"
#include "internal/resource_paths.hpp"
#include "rendering/render_constants.hpp"
#include "theme/theme.hpp"

#include <RmlUi/Core.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <format>

namespace lfs::vis {

    namespace {
        constexpr float DEFAULT_TIMELINE_DURATION = 10.0f;
        constexpr float TIMELINE_END_PADDING = 1.0f;
        constexpr float MIN_KEYFRAME_SPACING = 0.1f;
        constexpr float DOUBLE_CLICK_TIME = 0.3f;
        constexpr float DRAG_THRESHOLD_PX = 3.0f;

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

    using gui::rml_theme::colorToRml;
    using gui::rml_theme::colorToRmlAlpha;
    using namespace panel_config;

    RmlSequencerPanel::RmlSequencerPanel(SequencerController& controller, gui::RmlUIManager* rml_manager)
        : controller_(controller),
          rml_manager_(rml_manager) {
        assert(rml_manager_);
        transport_listener_.panel = this;
    }

    RmlSequencerPanel::~RmlSequencerPanel() = default;

    void RmlSequencerPanel::TransportClickListener::ProcessEvent(Rml::Event& event) {
        assert(panel);
        auto* el = event.GetCurrentElement();
        if (!el)
            return;

        const auto& id = el->GetId();
        auto& ctrl = panel->controller_;

        if (id == "btn-skip-back")
            ctrl.seekToFirstKeyframe();
        else if (id == "btn-stop")
            ctrl.stop();
        else if (id == "btn-play")
            ctrl.togglePlayPause();
        else if (id == "btn-skip-forward")
            ctrl.seekToLastKeyframe();
        else if (id == "btn-loop")
            ctrl.toggleLoop();
        else if (id == "btn-add")
            lfs::core::events::cmd::SequencerAddKeyframe{}.emit();
    }

    TimelineContextMenuState RmlSequencerPanel::consumeContextMenu() {
        TimelineContextMenuState state;
        if (context_menu_open_) {
            state.open = true;
            state.time = context_menu_time_;
            state.keyframe = context_menu_keyframe_;
            context_menu_open_ = false;
        }
        return state;
    }

    TimeEditRequest RmlSequencerPanel::consumeTimeEditRequest() {
        TimeEditRequest req;
        if (editing_keyframe_time_) {
            const auto& keyframes = controller_.timeline().keyframes();
            if (editing_keyframe_index_ < keyframes.size()) {
                req.active = true;
                req.keyframe_index = editing_keyframe_index_;
                req.current_time = keyframes[editing_keyframe_index_].time;
            }
            editing_keyframe_time_ = false;
        }
        return req;
    }

    FocalEditRequest RmlSequencerPanel::consumeFocalEditRequest() {
        FocalEditRequest req;
        if (editing_focal_length_) {
            req.active = true;
            req.keyframe_index = editing_focal_index_;
            req.current_focal_mm = std::stof(focal_edit_buffer_);
            editing_focal_length_ = false;
        }
        return req;
    }

    void RmlSequencerPanel::destroyGLResources() {
        fbo_.destroy();
    }

    void RmlSequencerPanel::initContext(const int width, const int height) {
        if (rml_context_)
            return;

        rml_context_ = rml_manager_->createContext("sequencer", width, height);
        if (!rml_context_)
            return;

        try {
            const auto full_path = lfs::vis::getAssetPath("rmlui/sequencer.rml");
            document_ = rml_context_->LoadDocument(full_path.string());
            if (document_) {
                document_->Show();
                cacheElements();
            } else {
                LOG_ERROR("RmlUI: failed to load sequencer.rml");
            }
        } catch (const std::exception& e) {
            LOG_ERROR("RmlUI: sequencer resource not found: {}", e.what());
        }
    }

    void RmlSequencerPanel::cacheElements() {
        assert(document_);
        el_ruler_ = document_->GetElementById("ruler");
        el_track_bar_ = document_->GetElementById("track-bar");
        el_keyframes_ = document_->GetElementById("keyframes");
        el_playhead_ = document_->GetElementById("playhead");
        el_hint_ = document_->GetElementById("hint");
        el_current_time_ = document_->GetElementById("current-time");
        el_duration_ = document_->GetElementById("duration");
        el_play_icon_ = document_->GetElementById("play-icon");
        el_btn_loop_ = document_->GetElementById("btn-loop");
        el_timeline_ = document_->GetElementById("timeline");
        elements_cached_ = el_ruler_ && el_keyframes_ && el_playhead_ &&
                           el_current_time_ && el_duration_ && el_play_icon_ &&
                           el_btn_loop_ && el_timeline_;
        if (!elements_cached_) {
            LOG_ERROR("RmlUI sequencer: missing DOM elements");
            return;
        }

        for (const char* btn_id : {"btn-skip-back", "btn-stop", "btn-play",
                                   "btn-skip-forward", "btn-loop", "btn-add"}) {
            auto* el = document_->GetElementById(btn_id);
            if (el)
                el->AddEventListener(Rml::EventId::Click, &transport_listener_);
        }
    }

    std::string RmlSequencerPanel::generateThemeRCSS() const {
        const auto& p = lfs::vis::theme().palette;
        const auto& t = lfs::vis::theme();

        const auto surface_alpha = colorToRmlAlpha(p.surface, 0.95f);
        const auto border = colorToRmlAlpha(p.border, 0.4f);
        const auto text = colorToRml(p.text);
        const auto text_dim = colorToRml(p.text_dim);
        const auto text_dim_half = colorToRmlAlpha(p.text_dim, 0.5f);
        const auto bg_alpha = colorToRmlAlpha(p.background, 0.8f);
        const auto border_dim = colorToRmlAlpha(p.border, 0.3f);
        const auto error = colorToRml(p.error);
        const int rounding = static_cast<int>(t.sizes.window_rounding);

        return std::format(
            "#panel {{ background-color: {}; border-width: 1dp; border-color: {}; "
            "border-radius: {}dp; }}\n"
            ".transport-icon {{ image-color: {}; }}\n"
            "#track-bar {{ background-color: {}; border-width: 1dp; border-color: {}; }}\n"
            "#hint {{ color: {}; }}\n"
            ".ruler-tick.major {{ background-color: {}; }}\n"
            ".ruler-tick.minor {{ background-color: {}; }}\n"
            ".ruler-label {{ color: {}; }}\n"
            "#playhead-line {{ background-color: {}; }}\n"
            "#playhead-handle {{ background-color: {}; }}\n"
            "#current-time {{ color: {}; }}\n"
            "#duration {{ color: {}; }}\n",
            surface_alpha, border, rounding,
            text,
            bg_alpha, border_dim,
            text_dim_half,
            text_dim,
            text_dim_half,
            text_dim,
            error,
            error,
            text,
            text_dim);
    }

    void RmlSequencerPanel::syncTheme() {
        if (!document_)
            return;

        const auto& p = lfs::vis::theme().palette;
        if (std::memcmp(last_synced_text_, &p.text, sizeof(last_synced_text_)) == 0)
            return;
        std::memcpy(last_synced_text_, &p.text, sizeof(last_synced_text_));

        if (base_rcss_.empty())
            base_rcss_ = gui::rml_theme::loadBaseRCSS("rmlui/sequencer.rcss");

        gui::rml_theme::applyTheme(document_, base_rcss_, generateThemeRCSS());
    }

    void RmlSequencerPanel::updateButtonStates() {
        if (!elements_cached_)
            return;

        const bool playing = controller_.isPlaying();
        el_play_icon_->SetAttribute("src",
                                    playing ? "../icon/sequencer/pause.png"
                                            : "../icon/sequencer/play.png");

        const bool looping = controller_.loopMode() != LoopMode::ONCE;
        if (looping)
            el_btn_loop_->SetClass("active", true);
        else
            el_btn_loop_->SetClass("active", false);
    }

    void RmlSequencerPanel::updatePlayhead() {
        if (!elements_cached_)
            return;

        const float timeline_width = cached_panel_width_ - 2.0f * INNER_PADDING -
                                     TRANSPORT_WIDTH - TIME_DISPLAY_WIDTH;
        if (timeline_width <= 0.0f)
            return;

        const float x = timeToX(controller_.playhead(), 0.0f, timeline_width);
        el_playhead_->SetProperty("left", std::format("{:.1f}dp", x));
    }

    void RmlSequencerPanel::updateTimeDisplay() {
        if (!elements_cached_)
            return;

        el_current_time_->SetInnerRML(formatTime(controller_.playhead()));

        if (!controller_.timeline().empty()) {
            el_duration_->SetInnerRML(" / " + formatTime(controller_.timeline().endTime()));
        } else {
            el_duration_->SetInnerRML("");
        }
    }

    void RmlSequencerPanel::rebuildKeyframes() {
        if (!elements_cached_)
            return;

        const auto& timeline = controller_.timeline();
        const auto& keyframes = timeline.keyframes();
        const size_t count = keyframes.size();

        const float timeline_width = cached_panel_width_ - 2.0f * INNER_PADDING -
                                     TRANSPORT_WIDTH - TIME_DISPLAY_WIDTH;

        if (count == last_keyframe_count_ &&
            zoom_level_ == last_zoom_level_ &&
            pan_offset_ == last_pan_offset_ &&
            timeline_width == last_kf_width_) {
            return;
        }
        last_keyframe_count_ = count;
        last_zoom_level_ = zoom_level_;
        last_pan_offset_ = pan_offset_;
        last_kf_width_ = timeline_width;
        if (timeline_width <= 0.0f)
            return;

        const auto& p = lfs::vis::theme().palette;

        if (count == 0) {
            while (!keyframe_elements_.empty()) {
                el_keyframes_->RemoveChild(keyframe_elements_.back());
                keyframe_elements_.pop_back();
            }
            if (el_hint_)
                el_hint_->SetInnerRML("Position camera and press K to add keyframes");
            return;
        }

        if (el_hint_)
            el_hint_->SetInnerRML("");

        while (keyframe_elements_.size() < count) {
            auto new_elem = document_->CreateElement("div");
            assert(new_elem);
            Rml::Element* raw = new_elem.get();
            el_keyframes_->AppendChild(std::move(new_elem));
            keyframe_elements_.push_back(raw);
        }
        while (keyframe_elements_.size() > count) {
            el_keyframes_->RemoveChild(keyframe_elements_.back());
            keyframe_elements_.pop_back();
        }

        for (size_t i = 0; i < count; ++i) {
            auto* el = keyframe_elements_[i];
            const float x = timeToX(keyframes[i].time, 0.0f, timeline_width);
            const bool selected = controller_.selectedKeyframe() == i ||
                                  selected_keyframes_.contains(i);
            const bool is_loop = keyframes[i].is_loop_point;

            const auto base = is_loop ? p.info : p.primary;
            auto fill = base;
            if (selected)
                fill = lighten(base, 0.2f);

            el->SetClassNames("keyframe");
            el->SetClass("loop-point", is_loop);
            el->SetClass("selected", selected);
            el->SetProperty("left", std::format("{:.1f}dp", x));
            el->SetProperty("background-color", colorToRml(fill));
            el->SetProperty("border-color", selected ? colorToRml(p.text) : colorToRml(fill));
        }
    }

    void RmlSequencerPanel::rebuildRuler() {
        if (!elements_cached_)
            return;

        const float timeline_width = cached_panel_width_ - 2.0f * INNER_PADDING -
                                     TRANSPORT_WIDTH - TIME_DISPLAY_WIDTH;

        if (zoom_level_ == last_ruler_zoom_ &&
            pan_offset_ == last_ruler_pan_ &&
            timeline_width == last_ruler_width_)
            return;
        last_ruler_zoom_ = zoom_level_;
        last_ruler_pan_ = pan_offset_;
        last_ruler_width_ = timeline_width;
        if (timeline_width <= 0.0f)
            return;

        const float end_time = getDisplayEndTime();

        float major_interval = 1.0f;
        if (end_time > 60.0f)
            major_interval = 10.0f;
        else if (end_time > 30.0f)
            major_interval = 5.0f;
        else if (end_time > 10.0f)
            major_interval = 2.0f;
        else if (end_time <= 2.0f)
            major_interval = 0.5f;

        major_interval /= zoom_level_;
        const float minor_interval = major_interval / 4.0f;

        std::string html;
        html.reserve(2048);

        for (float t_val = 0.0f; t_val <= end_time; t_val += minor_interval) {
            const float x = (t_val / end_time) * timeline_width;
            if (x < 0.0f || x > timeline_width)
                continue;

            const bool is_major = std::fmod(t_val + 0.001f, major_interval) < 0.01f;

            if (is_major) {
                html += std::format(
                    "<div class=\"ruler-tick major\" style=\"left: {:.1f}dp;\" />"
                    "<span class=\"ruler-label\" style=\"left: {:.1f}dp;\">{}</span>",
                    x, x + 4.0f, formatTimeShort(t_val));
            } else {
                html += std::format(
                    "<div class=\"ruler-tick minor\" style=\"left: {:.1f}dp;\" />",
                    x);
            }
        }

        el_ruler_->SetInnerRML(html);
    }

    void RmlSequencerPanel::forwardInput(const PanelInputState& input) {
        if (!rml_context_)
            return;

        const float local_x = input.mouse_x - cached_panel_x_;
        const float local_y = input.mouse_y - cached_panel_y_;

        hovered_ = local_x >= 0 && local_y >= 0 &&
                   local_x < cached_panel_width_ && local_y < HEIGHT;
        if (!hovered_)
            return;

        const float dp_ratio = rml_manager_->getDpRatio();
        rml_context_->ProcessMouseMove(static_cast<int>(local_x * dp_ratio),
                                       static_cast<int>(local_y * dp_ratio), 0);

        if (input.mouse_clicked[0])
            rml_context_->ProcessMouseButtonDown(0, 0);
        if (!input.mouse_down[0])
            rml_context_->ProcessMouseButtonUp(0, 0);
    }

    void RmlSequencerPanel::render(const float viewport_x, const float viewport_width,
                                   const float viewport_y_bottom,
                                   const PanelInputState& input) {
        const float panel_x = viewport_x + PADDING_H;
        const float panel_width = viewport_width - 2.0f * PADDING_H;
        const float panel_y = viewport_y_bottom - HEIGHT - PADDING_BOTTOM;

        cached_panel_x_ = panel_x;
        cached_panel_y_ = panel_y;
        cached_panel_width_ = panel_width;

        const float dp_ratio = rml_manager_->getDpRatio();
        const int w = static_cast<int>(panel_width * dp_ratio);
        const int h = static_cast<int>(HEIGHT * dp_ratio);

        if (w <= 0 || h <= 0)
            return;

        if (!rml_context_)
            initContext(w, h);
        if (!rml_context_ || !document_)
            return;

        syncTheme();

        if (elements_cached_) {
            const float timeline_width = panel_width - 2.0f * INNER_PADDING -
                                         TRANSPORT_WIDTH - TIME_DISPLAY_WIDTH;
            el_timeline_->SetProperty("width", std::format("{:.1f}dp", timeline_width));

            updateButtonStates();
            updatePlayhead();
            updateTimeDisplay();
            rebuildKeyframes();
            rebuildRuler();
        }

        forwardInput(input);

        rml_context_->SetDimensions(Rml::Vector2i(w, h));
        rml_context_->Update();

        fbo_.ensure(w, h);
        if (!fbo_.valid())
            return;

        auto* render_iface = rml_manager_->getRenderInterface();
        assert(render_iface);
        render_iface->SetViewport(w, h);

        GLint prev_fbo = 0;
        fbo_.bind(&prev_fbo);

        render_iface->BeginFrame();
        rml_context_->Render();
        render_iface->EndFrame();

        fbo_.unbind(prev_fbo);

        fbo_.blitToScreen(panel_x, panel_y, panel_width, HEIGHT,
                          input.screen_w, input.screen_h);

        const float content_height = HEIGHT - 2.0f * INNER_PADDING;
        const float timeline_width = panel_width - 2.0f * INNER_PADDING -
                                     TRANSPORT_WIDTH - TIME_DISPLAY_WIDTH;

        const Vec2 timeline_pos = {panel_x + INNER_PADDING + TRANSPORT_WIDTH,
                                   panel_y + INNER_PADDING};
        handleTimelineInteraction(timeline_pos, timeline_width, content_height, input);
    }

    void RmlSequencerPanel::handleTimelineInteraction(const Vec2& pos, const float width,
                                                      const float height,
                                                      const PanelInputState& input) {
        const float timeline_y = pos.y + RULER_HEIGHT + 4.0f;
        const float timeline_height = height - RULER_HEIGHT - 4.0f;
        const float bar_half = std::min(timeline_height, TIMELINE_HEIGHT) / 2.0f;
        const float y_center = timeline_y + timeline_height / 2.0f;

        const Vec2 bar_min = {pos.x, y_center - bar_half};
        const Vec2 bar_max = {pos.x + width, y_center + bar_half};

        const auto& timeline = controller_.timeline();
        if (timeline.empty())
            return;

        const float mx = input.mouse_x;
        const float my = input.mouse_y;
        const bool mouse_in_timeline = mx >= bar_min.x && mx <= bar_max.x &&
                                       my >= bar_min.y - RULER_HEIGHT && my <= bar_max.y;

        if (mouse_in_timeline && !input.want_capture_mouse) {
            const float wheel = input.mouse_wheel;
            if (std::abs(wheel) > 0.01f) {
                const float old_zoom = zoom_level_;
                zoom_level_ = std::clamp(zoom_level_ + wheel * ZOOM_SPEED, MIN_ZOOM, MAX_ZOOM);

                if (zoom_level_ != old_zoom) {
                    const float mouse_time = xToTime(mx, pos.x, width);
                    pan_offset_ += (mouse_time - pan_offset_) * (1.0f - old_zoom / zoom_level_) * 0.5f;
                    pan_offset_ = std::max(0.0f, pan_offset_);
                }
            }
        }

        if (input.mouse_clicked[0] && mouse_in_timeline && !dragging_keyframe_ &&
            !hovered_keyframe_.has_value()) {
            dragging_playhead_ = true;
            controller_.beginScrub();
        }
        if (dragging_playhead_) {
            if (input.mouse_down[0]) {
                float time = xToTime(mx, pos.x, width);
                time = std::clamp(time, 0.0f, timeline.endTime());
                if (snap_enabled_)
                    time = snapTime(time);
                controller_.scrub(time);
            } else {
                dragging_playhead_ = false;
                controller_.endScrub();
            }
        }

        hovered_keyframe_ = std::nullopt;
        const auto& keyframes = timeline.keyframes();
        for (size_t i = 0; i < keyframes.size(); ++i) {
            const float x = timeToX(keyframes[i].time, pos.x, width);
            const float dist = std::abs(mx - x);
            const bool hovered = mouse_in_timeline && dist < KEYFRAME_RADIUS * 2;
            if (hovered)
                hovered_keyframe_ = i;

            if (hovered && input.mouse_clicked[0]) {
                const float current_time = input.time;

                if (last_clicked_keyframe_ == i &&
                    (current_time - last_click_time_) < DOUBLE_CLICK_TIME) {
                    editing_keyframe_time_ = true;
                    editing_keyframe_index_ = i;
                    time_edit_buffer_ = std::format("{:.2f}", keyframes[i].time);
                    last_clicked_keyframe_ = std::nullopt;
                } else {
                    last_click_time_ = current_time;
                    last_clicked_keyframe_ = i;

                    if (input.key_shift && controller_.hasSelection()) {
                        const size_t first_sel = *controller_.selectedKeyframe();
                        const size_t lo = std::min(first_sel, i);
                        const size_t hi = std::max(first_sel, i);
                        selected_keyframes_.clear();
                        for (size_t j = lo; j <= hi; ++j)
                            selected_keyframes_.insert(j);
                    } else if (input.key_ctrl) {
                        if (selected_keyframes_.contains(i))
                            selected_keyframes_.erase(i);
                        else
                            selected_keyframes_.insert(i);
                    } else {
                        selected_keyframes_.clear();
                        lfs::core::events::cmd::SequencerSelectKeyframe{.keyframe_index = i}.emit();
                        const bool is_first = (i == 0);
                        if (!is_first) {
                            dragging_keyframe_ = true;
                            dragged_keyframe_index_ = i;
                            drag_start_time_ = keyframes[i].time;
                            drag_start_mouse_x_ = mx;
                        } else {
                            lfs::core::events::cmd::SequencerGoToKeyframe{.keyframe_index = i}.emit();
                        }
                    }
                }
            }
        }

        if (dragging_keyframe_) {
            if (input.mouse_down[0]) {
                float new_time = xToTime(mx, pos.x, width);
                new_time = std::max(new_time, MIN_KEYFRAME_SPACING);
                if (snap_enabled_)
                    new_time = snapTime(new_time);
                controller_.timeline().setKeyframeTime(dragged_keyframe_index_, new_time, false);
            } else {
                if (std::abs(mx - drag_start_mouse_x_) < DRAG_THRESHOLD_PX) {
                    lfs::core::events::cmd::SequencerGoToKeyframe{.keyframe_index = dragged_keyframe_index_}.emit();
                }
                controller_.timeline().sortKeyframes();
                dragging_keyframe_ = false;
            }
        }

        if ((controller_.hasSelection() || !selected_keyframes_.empty()) &&
            input.key_delete_pressed) {
            std::vector<size_t> to_delete;
            if (!selected_keyframes_.empty())
                to_delete.assign(selected_keyframes_.begin(), selected_keyframes_.end());
            else if (controller_.hasSelection())
                to_delete.push_back(*controller_.selectedKeyframe());

            std::sort(to_delete.begin(), to_delete.end(), std::greater<>());

            for (const size_t idx : to_delete) {
                if (idx == 0)
                    continue;
                controller_.timeline().removeKeyframe(idx);
            }
            selected_keyframes_.clear();
            controller_.deselectKeyframe();
        }

        if (mouse_in_timeline && input.mouse_clicked[1]) {
            context_menu_time_ = xToTime(mx, pos.x, width);
            context_menu_keyframe_ = hovered_keyframe_;
            context_menu_open_ = true;
            context_menu_x_ = mx;
            context_menu_y_ = my;
        }

        // Context menu rendering is handled in sequencer_ui_manager for now
        // (still uses ImGui for context menus and tooltips as part of the viewport layer)
    }

    void RmlSequencerPanel::openFocalLengthEdit(const size_t index, const float current_focal_mm) {
        editing_focal_length_ = true;
        editing_focal_index_ = index;
        focal_edit_buffer_ = std::format("{:.1f}", current_focal_mm);
    }

    float RmlSequencerPanel::getDisplayEndTime() const {
        const auto& timeline = controller_.timeline();
        if (timeline.size() < 2)
            return DEFAULT_TIMELINE_DURATION / zoom_level_;
        return std::max(timeline.endTime() + TIMELINE_END_PADDING, DEFAULT_TIMELINE_DURATION) / zoom_level_;
    }

    float RmlSequencerPanel::timeToX(const float time, const float timeline_x, const float timeline_width) const {
        const float end = getDisplayEndTime();
        const float adjusted_time = (time - pan_offset_) * zoom_level_;
        return timeline_x + (adjusted_time / (end * zoom_level_)) * timeline_width;
    }

    float RmlSequencerPanel::xToTime(const float x, const float timeline_x, const float timeline_width) const {
        const float end = getDisplayEndTime();
        const float t = ((x - timeline_x) / timeline_width) * end;
        return t / zoom_level_ + pan_offset_;
    }

    float RmlSequencerPanel::snapTime(const float time) const {
        if (!snap_enabled_ || snap_interval_ <= 0.0f)
            return time;
        return std::round(time / snap_interval_) * snap_interval_;
    }

} // namespace lfs::vis

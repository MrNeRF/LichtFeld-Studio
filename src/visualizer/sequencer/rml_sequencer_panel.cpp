/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// clang-format off
#include <glad/glad.h>
// clang-format on

#include "sequencer/rml_sequencer_panel.hpp"
#include "core/events.hpp"
#include "core/logger.hpp"
#include "gui/rmlui/rml_input_utils.hpp"
#include "gui/rmlui/rml_panel_host.hpp"
#include "gui/rmlui/rml_theme.hpp"
#include "gui/rmlui/rml_tooltip.hpp"
#include "gui/rmlui/rmlui_manager.hpp"
#include "gui/rmlui/rmlui_render_interface.hpp"
#include "gui/rmlui/sdl_rml_key_mapping.hpp"
#include "gui/string_keys.hpp"
#include "internal/resource_paths.hpp"
#include "io/video/video_export_options.hpp"
#include "rendering/render_constants.hpp"
#include "sequencer/timeline_view_math.hpp"
#include "theme/theme.hpp"

#include <RmlUi/Core.h>
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <format>

namespace lfs::vis {

    namespace {
        constexpr float MIN_KEYFRAME_SPACING = 0.1f;
        constexpr float DOUBLE_CLICK_TIME = 0.3f;
        constexpr float DRAG_THRESHOLD_PX = 3.0f;
        constexpr float PLAYHEAD_HIT_RADIUS = 6.0f;

        constexpr std::array<float, 5> SPEED_PRESETS = {0.25f, 0.5f, 1.0f, 2.0f, 4.0f};

        [[nodiscard]] size_t findSpeedIndex(const float speed) {
            size_t best = 2;
            float best_diff = 100.0f;
            for (size_t i = 0; i < SPEED_PRESETS.size(); ++i) {
                const float diff = std::abs(SPEED_PRESETS[i] - speed);
                if (diff < best_diff) {
                    best_diff = diff;
                    best = i;
                }
            }
            return best;
        }

        [[nodiscard]] std::string formatSpeed(const float speed) {
            if (speed >= 1.0f)
                return std::format("{}x", static_cast<int>(speed));
            return std::format("{:.2g}x", speed);
        }

        [[nodiscard]] std::string formatPresetShort(const lfs::io::video::VideoPreset preset) {
            return lfs::io::video::getPresetInfo(preset).name;
        }

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

        [[nodiscard]] uint64_t selectedKeyframeSignature(const std::set<sequencer::KeyframeId>& selected_keyframes) {
            uint64_t signature = 1469598103934665603ull;
            for (const auto id : selected_keyframes) {
                signature ^= id;
                signature *= 1099511628211ull;
            }
            return signature;
        }

        void forwardFocusedKeyboardInput(Rml::Context* const context,
                                         const PanelInputState& input) {
            const int mods = gui::sdlModsToRml(input.key_ctrl, input.key_shift,
                                               input.key_alt, input.key_super);
            for (const int sc : input.keys_pressed) {
                const auto rml_key = gui::sdlScancodeToRml(static_cast<SDL_Scancode>(sc));
                if (rml_key != Rml::Input::KI_UNKNOWN)
                    context->ProcessKeyDown(rml_key, mods);
            }
            for (const int sc : input.keys_released) {
                const auto rml_key = gui::sdlScancodeToRml(static_cast<SDL_Scancode>(sc));
                if (rml_key != Rml::Input::KI_UNKNOWN)
                    context->ProcessKeyUp(rml_key, mods);
            }
        }

    } // namespace

    using gui::rml_theme::colorToRml;
    using gui::rml_theme::colorToRmlAlpha;
    using namespace panel_config;

    RmlSequencerPanel::RmlSequencerPanel(SequencerController& controller, gui::panels::SequencerUIState& ui_state,
                                         gui::RmlUIManager* rml_manager)
        : controller_(controller),
          ui_state_(ui_state),
          rml_manager_(rml_manager) {
        assert(rml_manager_);
        transport_listener_.panel = this;
        quality_scrub_listener_.panel = this;
    }

    RmlSequencerPanel::~RmlSequencerPanel() = default;

    void RmlSequencerPanel::TransportClickListener::ProcessEvent(Rml::Event& event) {
        assert(panel);
        auto* el = event.GetCurrentElement();
        if (!el)
            return;

        const auto& id = el->GetId();
        auto& ctrl = panel->controller_;
        auto& ui = panel->ui_state_;

        if (id == "btn-skip-back")
            ctrl.seekToFirstKeyframe();
        else if (id == "btn-stop")
            ctrl.stop();
        else if (id == "btn-play")
            ctrl.togglePlayPause();
        else if (id == "btn-skip-forward")
            ctrl.seekToLastKeyframe();
        else if (id == "btn-loop") {
            ctrl.toggleLoop();
            lfs::core::events::state::KeyframeListChanged{.count = ctrl.timeline().realKeyframeCount()}.emit();
        } else if (id == "btn-add")
            lfs::core::events::cmd::SequencerAddKeyframe{}.emit();
        else if (id == "btn-camera-path")
            ui.show_camera_path = !ui.show_camera_path;
        else if (id == "btn-snap")
            ui.snap_to_grid = !ui.snap_to_grid;
        else if (id == "btn-follow")
            ui.follow_playback = !ui.follow_playback;
        else if (id == "btn-film-strip")
            ui.show_film_strip = !ui.show_film_strip;
        else if (id == "btn-preview")
            ui.show_pip_preview = !ui.show_pip_preview;
        else if (id == "btn-equirect") {
            ui.equirectangular = !ui.equirectangular;
            auto event = lfs::core::events::ui::RenderSettingsChanged{};
            event.equirectangular = ui.equirectangular;
            event.emit();
        } else if (id == "btn-speed") {
            const size_t idx = findSpeedIndex(ui.playback_speed);
            const size_t next = (idx + 1) % SPEED_PRESETS.size();
            ui.playback_speed = SPEED_PRESETS[next];
            ctrl.setPlaybackSpeed(ui.playback_speed);
        } else if (id == "btn-format") {
            using lfs::io::video::VideoPreset;
            auto p = static_cast<int>(ui.preset);
            p = (p + 1) % static_cast<int>(VideoPreset::CUSTOM);
            ui.preset = static_cast<VideoPreset>(p);
            const auto info = lfs::io::video::getPresetInfo(ui.preset);
            ui.custom_width = info.width;
            ui.custom_height = info.height;
            ui.framerate = info.framerate;
        } else if (id == "btn-save-path")
            panel->save_path_requested_ = true;
        else if (id == "btn-load-path")
            panel->load_path_requested_ = true;
        else if (id == "btn-export")
            panel->export_requested_ = true;
        else if (id == "btn-clear") {
            float sx = panel->cached_panel_x_;
            float sy = panel->cached_panel_y_;
            auto abs_offset = el->GetAbsoluteOffset(Rml::BoxArea::Border);
            sx = panel->cached_panel_x_ + abs_offset.x;
            sy = panel->cached_panel_y_ + abs_offset.y + el->GetBox().GetSize().y;
            panel->transport_ctx_request_ = {TransportContextMenuRequest::Target::CLEAR, sx, sy};
        }
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

    TransportContextMenuRequest RmlSequencerPanel::consumeTransportContextMenu() {
        auto req = transport_ctx_request_;
        transport_ctx_request_ = {};
        return req;
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

        cached_dp_ratio_ = rml_manager_->getDpRatio();
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

        el_btn_camera_path_ = document_->GetElementById("btn-camera-path");
        el_btn_snap_ = document_->GetElementById("btn-snap");
        el_btn_follow_ = document_->GetElementById("btn-follow");
        el_btn_film_strip_ = document_->GetElementById("btn-film-strip");
        el_btn_preview_ = document_->GetElementById("btn-preview");
        el_speed_label_ = document_->GetElementById("speed-label");
        el_format_label_ = document_->GetElementById("format-label");
        el_resolution_info_ = document_->GetElementById("resolution-info");
        el_quality_scrub_ = document_->GetElementById("quality-scrub");
        el_quality_fill_ = document_->GetElementById("quality-fill");
        el_quality_display_ = document_->GetElementById("quality-display");
        el_quality_input_ = document_->GetElementById("quality-input");
        el_btn_equirect_ = document_->GetElementById("btn-equirect");
        el_btn_save_ = document_->GetElementById("btn-save-path");
        el_btn_load_ = document_->GetElementById("btn-load-path");
        el_btn_export_ = document_->GetElementById("btn-export");
        el_btn_clear_ = document_->GetElementById("btn-clear");

        elements_cached_ = el_ruler_ && el_keyframes_ && el_playhead_ &&
                           el_current_time_ && el_duration_ && el_play_icon_ &&
                           el_btn_loop_ && el_timeline_;
        if (!elements_cached_) {
            LOG_ERROR("RmlUI sequencer: missing DOM elements");
            return;
        }

        for (const char* btn_id : {"btn-skip-back", "btn-stop", "btn-play",
                                   "btn-skip-forward", "btn-loop", "btn-add",
                                   "btn-camera-path", "btn-snap", "btn-follow",
                                   "btn-film-strip", "btn-preview", "btn-equirect", "btn-speed",
                                   "btn-format", "btn-save-path", "btn-load-path",
                                   "btn-export", "btn-clear"}) {
            auto* el = document_->GetElementById(btn_id);
            if (el)
                el->AddEventListener(Rml::EventId::Click, &transport_listener_);
        }

        if (el_quality_scrub_) {
            el_quality_scrub_->AddEventListener(Rml::EventId::Mousedown, &quality_scrub_listener_);
            if (auto* body = document_->GetElementById("body")) {
                body->AddEventListener(Rml::EventId::Mousemove, &quality_scrub_listener_);
                body->AddEventListener(Rml::EventId::Mouseup, &quality_scrub_listener_);
            }
        }
        if (el_quality_input_) {
            el_quality_input_->AddEventListener(Rml::EventId::Change, &quality_scrub_listener_);
            el_quality_input_->AddEventListener(Rml::EventId::Blur, &quality_scrub_listener_);
        }
    }

    std::string RmlSequencerPanel::generateThemeRCSS(const lfs::vis::Theme& t) const {
        const auto& p = t.palette;

        const auto surface_alpha = colorToRmlAlpha(p.surface, 0.95f);
        const auto border = colorToRmlAlpha(p.border, 0.4f);
        const auto text = colorToRml(p.text);
        const auto text_dim = colorToRml(p.text_dim);
        const auto text_dim_half = colorToRmlAlpha(p.text_dim, 0.5f);
        const auto bg_alpha = colorToRmlAlpha(p.background, 0.8f);
        const auto border_dim = colorToRmlAlpha(p.border, 0.3f);
        const auto error = colorToRml(p.error);
        const auto primary_active = colorToRmlAlpha(p.primary, 0.20f);
        const auto primary_btn = colorToRmlAlpha(p.primary, 0.15f);
        const auto primary_btn_hover = colorToRmlAlpha(p.primary, 0.25f);
        const auto error_btn = colorToRmlAlpha(p.error, 0.15f);
        const auto error_btn_hover = colorToRmlAlpha(p.error, 0.30f);
        const auto primary_cam_bg = colorToRmlAlpha(p.primary, 0.30f);
        const auto primary_cam_border = colorToRmlAlpha(p.primary, 0.50f);
        const auto primary_export_border = colorToRmlAlpha(p.primary, 0.40f);
        const auto surface_bright_alpha = colorToRmlAlpha(p.surface_bright, 0.30f);
        const auto primary_color = colorToRml(p.primary);
        const int rounding = static_cast<int>(t.sizes.window_rounding);

        const std::string radius_str = film_strip_attached_
                                           ? std::format("{}dp {}dp 0dp 0dp", rounding, rounding)
                                           : std::format("{}dp", rounding);

        return std::format(
            "#panel {{ background-color: {}; border-width: 1dp; border-color: {}; "
            "border-radius: {}; }}\n"
            ".transport-icon {{ image-color: {}; }}\n"
            "#track-bar {{ background-color: {}; border-width: 1dp; border-color: {}; }}\n"
            "#hint {{ color: {}; }}\n"
            ".ruler-tick.major {{ background-color: {}; }}\n"
            ".ruler-tick.minor {{ background-color: {}; }}\n"
            ".ruler-label {{ color: {}; }}\n"
            "#playhead-handle {{ background-color: {}; }}\n"
            "#current-time {{ color: {}; }}\n"
            "#duration {{ color: {}; }}\n"
            "#easing-stripe {{ border-top: 1dp {}; }}\n"
            "#transport-row {{ border-bottom: 1dp {}; }}\n"
            ".transport-sep {{ background-color: {}; }}\n"
            ".transport-label {{ color: {}; }}\n"
            ".transport-info {{ color: {}; }}\n"
            ".transport-btn.toggle.active {{ background-color: {}; }}\n"
            ".transport-btn.primary {{ background-color: {}; }}\n"
            ".transport-btn.primary:hover {{ background-color: {}; }}\n"
            ".transport-btn.error:hover {{ background-color: {}; }}\n"
            "#btn-camera-path.active {{ background-color: {}; border-width: 1dp; border-color: {}; }}\n"
            "#btn-add .transport-icon {{ image-color: {}; }}\n"
            ".speed-val {{ color: {}; }}\n"
            ".speed-text {{ color: {}; }}\n"
            ".dropdown-arrow {{ color: {}; }}\n"
            ".snap-check {{ border-color: {}; }}\n"
            "#btn-snap.active .snap-check {{ background-color: {}; border-color: {}; }}\n"
            ".format-badge {{ background-color: {}; }}\n"
            "#btn-export {{ border-width: 1dp; border-color: {}; }}\n"
            "#btn-clear {{ background-color: {}; border-width: 1dp; border-color: {}; }}\n"
            "#btn-clear .transport-icon {{ image-color: {}; }}\n"
            ".ctx-indicator {{ color: {}; }}\n",
            surface_alpha, border, radius_str,
            text,
            bg_alpha, border_dim,
            text_dim_half,
            text_dim,
            text_dim_half,
            text_dim,
            error,
            text,
            text_dim,
            border_dim,
            border_dim,
            border_dim,
            text,
            text_dim,
            primary_active,
            primary_btn,
            primary_btn_hover,
            error_btn_hover,
            primary_cam_bg, primary_cam_border,
            error,
            text,
            text_dim,
            text_dim,
            text_dim,
            primary_color, primary_color,
            surface_bright_alpha,
            primary_export_border,
            error_btn, error_btn,
            error,
            text_dim_half);
    }

    void RmlSequencerPanel::syncTheme() {
        if (!document_)
            return;

        const auto& p = lfs::vis::theme().palette;
        const bool layout_changed = film_strip_attached_ != last_film_strip_attached_;
        if (!layout_changed && std::memcmp(last_synced_text_, &p.text, sizeof(last_synced_text_)) == 0)
            return;
        std::memcpy(last_synced_text_, &p.text, sizeof(last_synced_text_));
        last_film_strip_attached_ = film_strip_attached_;

        if (base_rcss_.empty())
            base_rcss_ = gui::rml_theme::loadBaseRCSS("rmlui/sequencer.rcss");

        gui::rml_theme::applyTheme(document_, base_rcss_, gui::rml_theme::generateAllThemeMedia([this](const auto& th) { return generateThemeRCSS(th); }));
    }

    void RmlSequencerPanel::updateButtonStates() {
        if (!elements_cached_)
            return;

        const bool playing = controller_.isPlaying();
        el_play_icon_->SetAttribute("src",
                                    playing ? "../icon/sequencer/pause.png"
                                            : "../icon/sequencer/play.png");

        auto* btn_play = document_->GetElementById("btn-play");
        if (btn_play)
            btn_play->SetAttribute("data-tooltip",
                                   playing ? "tooltip.seq_pause" : "tooltip.seq_play");

        const bool looping = controller_.loopMode() != LoopMode::ONCE;
        el_btn_loop_->SetClass("active", looping);
        el_btn_loop_->SetAttribute("data-tooltip",
                                   looping ? "tooltip.seq_loop_on" : "tooltip.seq_loop_off");
    }

    void RmlSequencerPanel::updatePlayhead() {
        if (!elements_cached_)
            return;

        const float tl_width = timelineWidth();
        if (tl_width <= 0.0f)
            return;

        const float x = timeToX(controller_.playhead(), 0.0f, tl_width);
        el_playhead_->SetProperty("left", std::format("{:.1f}px", x));
    }

    void RmlSequencerPanel::updateTimeDisplay() {
        if (!elements_cached_)
            return;

        el_current_time_->SetInnerRML(formatTime(controller_.playhead()));

        const float end = controller_.timeline().empty()
                              ? sequencer_ui::DEFAULT_TIMELINE_DURATION
                              : controller_.timeline().endTime();
        el_duration_->SetInnerRML(" / " + formatTime(end));
    }

    void RmlSequencerPanel::rebuildKeyframes() {
        if (!elements_cached_)
            return;

        const auto& timeline = controller_.timeline();
        const auto& keyframes = timeline.keyframes();
        const size_t count = keyframes.size();

        for (auto it = selected_keyframes_.begin(); it != selected_keyframes_.end();) {
            if (!timeline.findKeyframeIndex(*it).has_value())
                it = selected_keyframes_.erase(it);
            else
                ++it;
        }

        const float timeline_width = timelineWidth();
        const uint64_t timeline_revision = controller_.timelineRevision();
        const uint64_t selection_revision = controller_.selectionRevision();
        const uint64_t selected_keyframes_signature = selectedKeyframeSignature(selected_keyframes_);

        if (!dragging_keyframe_ &&
            count == last_keyframe_count_ &&
            zoom_level_ == last_zoom_level_ &&
            pan_offset_ == last_pan_offset_ &&
            timeline_revision == last_timeline_revision_ &&
            selection_revision == last_selection_revision_ &&
            selected_keyframes_signature == last_selected_keyframes_signature_ &&
            timeline_width == last_kf_width_) {
            return;
        }
        last_keyframe_count_ = count;
        last_zoom_level_ = zoom_level_;
        last_pan_offset_ = pan_offset_;
        last_kf_width_ = timeline_width;
        last_timeline_revision_ = timeline_revision;
        last_selection_revision_ = selection_revision;
        last_selected_keyframes_signature_ = selected_keyframes_signature;
        if (timeline_width <= 0.0f)
            return;

        const auto& p = lfs::vis::theme().palette;

        if (count == 0) {
            while (!keyframe_elements_.empty()) {
                el_keyframes_->RemoveChild(keyframe_elements_.back());
                keyframe_elements_.pop_back();
            }
            if (el_hint_)
                el_hint_->SetInnerRML(LOC(lichtfeld::Strings::Sequencer::EMPTY_HINT));
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
                                  selected_keyframes_.contains(keyframes[i].id);
            const bool is_loop = keyframes[i].is_loop_point;

            const auto base = is_loop ? p.info : (i % 2 == 0 ? p.primary : p.secondary);
            auto fill = base;
            if (selected)
                fill = lighten(base, 0.2f);

            el->SetClassNames("keyframe");
            el->SetClass("loop-point", is_loop);
            el->SetClass("selected", selected);
            el->SetProperty("left", std::format("{:.1f}px", x));
            el->SetProperty("background-color", colorToRml(fill));
            el->SetProperty("border-color", selected ? colorToRml(p.text) : colorToRml(fill));
        }
    }

    void RmlSequencerPanel::rebuildRuler() {
        if (!elements_cached_)
            return;

        const float timeline_width = timelineWidth();
        const float display_end_time = getDisplayEndTime();

        if (zoom_level_ == last_ruler_zoom_ &&
            pan_offset_ == last_ruler_pan_ &&
            timeline_width == last_ruler_width_ &&
            display_end_time == last_ruler_display_end_)
            return;
        last_ruler_zoom_ = zoom_level_;
        last_ruler_pan_ = pan_offset_;
        last_ruler_width_ = timeline_width;
        last_ruler_display_end_ = display_end_time;
        if (timeline_width <= 0.0f)
            return;

        const float visible_duration = display_end_time;
        const float visible_start = pan_offset_;
        const float visible_end = visible_start + visible_duration;

        float major_interval = 1.0f;
        if (visible_duration > 60.0f)
            major_interval = 10.0f;
        else if (visible_duration > 30.0f)
            major_interval = 5.0f;
        else if (visible_duration > 10.0f)
            major_interval = 2.0f;
        else if (visible_duration <= 2.0f)
            major_interval = 0.5f;

        major_interval /= zoom_level_;
        const float minor_interval = major_interval / 4.0f;

        std::string html;
        html.reserve(2048);

        const float label_margin = 30.0f * cached_dp_ratio_;

        const float first_tick = std::floor(visible_start / minor_interval) * minor_interval;
        for (float t_val = first_tick; t_val <= visible_end + minor_interval * 0.5f; t_val += minor_interval) {
            if (t_val < 0.0f)
                continue;

            const float x = timeToX(t_val, 0.0f, timeline_width);
            if (x < 0.0f || x > timeline_width)
                continue;

            const float major_phase = std::fmod(t_val, major_interval);
            const bool is_major = major_phase < 0.01f || (major_interval - major_phase) < 0.01f;

            if (is_major) {
                html += std::format(
                    "<div class=\"ruler-tick major\" style=\"left: {:.1f}px;\" />", x);
                if (x + label_margin <= timeline_width) {
                    html += std::format(
                        "<span class=\"ruler-label\" style=\"left: {:.1f}px;\">{}</span>",
                        x + 4.0f * cached_dp_ratio_, formatTimeShort(t_val));
                }
            } else {
                html += std::format(
                    "<div class=\"ruler-tick minor\" style=\"left: {:.1f}px;\" />",
                    x);
            }
        }

        el_ruler_->SetInnerRML(html);
    }

    void RmlSequencerPanel::forwardInput(const PanelInputState& input) {
        if (!rml_context_)
            return;
        if (rml_manager_) {
            rml_manager_->trackContextFrame(rml_context_,
                                            static_cast<int>(cached_panel_x_ - input.screen_x),
                                            static_cast<int>(cached_panel_y_ - input.screen_y));
        }

        const float local_x = input.mouse_x - cached_panel_x_;
        const float local_y = input.mouse_y - cached_panel_y_;

        const float total_h = (HEIGHT + EASING_STRIPE_HEIGHT) * cached_dp_ratio_;
        hovered_ = local_x >= 0 && local_y >= 0 &&
                   local_x < cached_panel_width_ && local_y < total_h;

        if (!hovered_) {
            tooltip_.clear();
            gui::RmlPanelHost::setFrameTooltip({}, nullptr);
            if (last_hovered_)
                rml_context_->ProcessMouseLeave();
            last_hovered_ = false;

            if (input.mouse_clicked[0]) {
                if (auto* const focused = rml_context_->GetFocusElement())
                    focused->Blur();
            }

            auto* const focused = rml_context_->GetFocusElement();
            if (gui::rml_input::hasFocusedKeyboardTarget(focused))
                forwardFocusedKeyboardInput(rml_context_, input);

            wants_keyboard_ = gui::rml_input::hasFocusedKeyboardTarget(focused) ||
                              dragging_playhead_ || dragging_keyframe_ ||
                              controller_.hasSelection() || !selected_keyframes_.empty();
            return;
        }

        last_hovered_ = true;

        rml_context_->ProcessMouseMove(static_cast<int>(local_x),
                                       static_cast<int>(local_y), 0);

        if (input.mouse_clicked[0])
            rml_context_->ProcessMouseButtonDown(0, 0);
        if (!input.mouse_down[0])
            rml_context_->ProcessMouseButtonUp(0, 0);

        tooltip_.clear();
        auto* hover = rml_context_->GetHoverElement();
        if (hover) {
            gui::RmlPanelHost::setFrameTooltip(gui::resolveRmlTooltip(hover), hover);

            if (input.mouse_clicked[1]) {
                for (auto* el = hover; el; el = el->GetParentNode()) {
                    const auto& id = el->GetId();
                    TransportContextMenuRequest::Target target = TransportContextMenuRequest::Target::NONE;
                    if (id == "btn-snap")
                        target = TransportContextMenuRequest::Target::SNAP;
                    else if (id == "btn-preview")
                        target = TransportContextMenuRequest::Target::PREVIEW;
                    else if (id == "btn-format")
                        target = TransportContextMenuRequest::Target::FORMAT;

                    if (target != TransportContextMenuRequest::Target::NONE) {
                        transport_ctx_request_ = {target, input.mouse_x, input.mouse_y};
                        break;
                    }
                }
            }
        } else {
            gui::RmlPanelHost::setFrameTooltip({}, nullptr);
        }

        auto* const focused = rml_context_->GetFocusElement();
        if (gui::rml_input::hasFocusedKeyboardTarget(focused))
            forwardFocusedKeyboardInput(rml_context_, input);

        wants_keyboard_ = gui::rml_input::hasFocusedKeyboardTarget(focused) ||
                          dragging_playhead_ || dragging_keyframe_ ||
                          controller_.hasSelection() || !selected_keyframes_.empty();
    }

    std::string RmlSequencerPanel::consumeTooltip() {
        std::string result;
        result.swap(tooltip_);
        return result;
    }

    bool RmlSequencerPanel::consumeSavePathRequest() {
        const bool r = save_path_requested_;
        save_path_requested_ = false;
        return r;
    }

    bool RmlSequencerPanel::consumeLoadPathRequest() {
        const bool r = load_path_requested_;
        load_path_requested_ = false;
        return r;
    }

    bool RmlSequencerPanel::consumeExportRequest() {
        const bool request = export_requested_;
        export_requested_ = false;
        return request;
    }

    bool RmlSequencerPanel::consumeClearRequest() {
        const bool r = clear_requested_;
        clear_requested_ = false;
        return r;
    }

    void RmlSequencerPanel::updateTransportSettings() {
        if (!elements_cached_)
            return;

        const bool has_camera_keyframes = controller_.timeline().realKeyframeCount() > 0;
        const bool has_any_state = has_camera_keyframes || controller_.timeline().hasAnimationClip();

        if (el_btn_camera_path_)
            el_btn_camera_path_->SetClass("active", ui_state_.show_camera_path);
        if (el_btn_snap_)
            el_btn_snap_->SetClass("active", ui_state_.snap_to_grid);
        if (el_btn_follow_)
            el_btn_follow_->SetClass("active", ui_state_.follow_playback);
        if (el_btn_film_strip_)
            el_btn_film_strip_->SetClass("active", ui_state_.show_film_strip);
        if (el_btn_preview_)
            el_btn_preview_->SetClass("active", ui_state_.show_pip_preview);
        if (el_btn_equirect_)
            el_btn_equirect_->SetClass("active", ui_state_.equirectangular);
        if (el_speed_label_)
            el_speed_label_->SetInnerRML(formatSpeed(ui_state_.playback_speed));
        if (el_format_label_)
            el_format_label_->SetInnerRML(formatPresetShort(ui_state_.preset));
        if (el_resolution_info_) {
            const auto info = lfs::io::video::getPresetInfo(ui_state_.preset);
            const bool custom = ui_state_.preset == lfs::io::video::VideoPreset::CUSTOM;
            const int w = custom ? ui_state_.custom_width : info.width;
            const int h = custom ? ui_state_.custom_height : info.height;
            const int fps = custom ? ui_state_.framerate : info.framerate;
            el_resolution_info_->SetInnerRML(std::format("{}x{} @ {}fps", w, h, fps));
        }
        if (!quality_scrub_editing_)
            syncQualityScrub();

        if (el_btn_save_)
            el_btn_save_->SetClass("disabled", !has_camera_keyframes);
        if (el_btn_export_)
            el_btn_export_->SetClass("disabled", !has_camera_keyframes);
        if (el_btn_clear_)
            el_btn_clear_->SetClass("disabled", !has_any_state);
    }

    float RmlSequencerPanel::timelineWidth() const {
        const float s = cached_dp_ratio_;
        return cached_panel_width_ - 2.0f * INNER_PADDING_H * s;
    }

    void RmlSequencerPanel::render(const float viewport_x, const float viewport_width,
                                   const float viewport_y_bottom,
                                   const PanelInputState& input) {
        const float dp = rml_manager_->getDpRatio();
        cached_dp_ratio_ = dp;
        cached_height_ = HEIGHT * dp;
        const float total_height = (HEIGHT + EASING_STRIPE_HEIGHT) * dp;

        const float padding_h = PADDING_H * dp;
        const float padding_bottom = PADDING_BOTTOM * dp;

        const float panel_x = viewport_x + padding_h;
        const float panel_width = viewport_width - 2.0f * padding_h;
        const float panel_y = viewport_y_bottom - total_height - padding_bottom;

        cached_panel_x_ = panel_x;
        cached_panel_y_ = panel_y;
        cached_panel_width_ = panel_width;

        const int w = static_cast<int>(panel_width);
        const int h = static_cast<int>(cached_height_);

        if (w <= 0 || h <= 0)
            return;

        if (!rml_context_)
            initContext(w, h);
        if (!rml_context_ || !document_)
            return;

        syncTheme();

        const auto& lang = lfs::event::LocalizationManager::getInstance().getCurrentLanguage();
        if (lang != last_language_) {
            last_language_ = lang;
            last_keyframe_count_ = static_cast<size_t>(-1);
        }

        if (elements_cached_) {
            el_timeline_->SetProperty("width", std::format("{:.1f}px", timelineWidth()));

            updateButtonStates();
            updateTransportSettings();
            updatePlayhead();
            updateTimeDisplay();
            rebuildKeyframes();
            rebuildRuler();
        }

        forwardInput(input);

        if (!rml_manager_->shouldDeferFboUpdate(fbo_)) {
            if (rml_manager_) {
                rml_manager_->trackContextFrame(rml_context_,
                                                static_cast<int>(panel_x - input.screen_x),
                                                static_cast<int>(panel_y - input.screen_y));
            }
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
            render_iface->SetTargetFramebuffer(fbo_.fbo());

            render_iface->BeginFrame();
            rml_context_->Render();
            render_iface->EndFrame();

            render_iface->SetTargetFramebuffer(0);
            fbo_.unbind(prev_fbo);
        }

        if (fbo_.valid())
            fbo_.blitToScreen(panel_x, panel_y, panel_width, cached_height_,
                              input.screen_w, input.screen_h);

        const float inner_pad_h = INNER_PADDING_H * dp;
        const float inner_pad = INNER_PADDING * dp;
        const float transport_row_h = TRANSPORT_ROW_HEIGHT * dp;
        const float content_height = cached_height_ - 2.0f * inner_pad - transport_row_h;
        const float tl_width = timelineWidth();

        const Vec2 timeline_pos = {panel_x + inner_pad_h,
                                   panel_y + inner_pad + transport_row_h};

        cached_playhead_screen_x_ = timeToX(controller_.playhead(), timeline_pos.x, tl_width);
        playhead_in_range_ = cached_playhead_screen_x_ >= timeline_pos.x &&
                             cached_playhead_screen_x_ <= timeline_pos.x + tl_width;

        handleTimelineInteraction(timeline_pos, tl_width, content_height, input);
    }

    void RmlSequencerPanel::handleTimelineInteraction(const Vec2& pos, const float width,
                                                      const float height,
                                                      const PanelInputState& input) {
        const float s = cached_dp_ratio_;
        const float timeline_y = pos.y + RULER_HEIGHT * s + 4.0f * s;
        const float timeline_height = height - RULER_HEIGHT * s - 4.0f * s;
        const float bar_half = std::min(timeline_height, TIMELINE_HEIGHT * s) / 2.0f;
        const float y_center = timeline_y + timeline_height / 2.0f;

        const Vec2 bar_min = {pos.x, y_center - bar_half};
        const Vec2 bar_max = {pos.x + width, y_center + bar_half};

        const auto& timeline = controller_.timeline();
        if (timeline.empty())
            return;

        const float mx = input.mouse_x;
        const float my = input.mouse_y;
        const bool mouse_in_timeline = mx >= bar_min.x && mx <= bar_max.x &&
                                       my >= bar_min.y - RULER_HEIGHT * s && my <= bar_max.y;

        if (mouse_in_timeline && !input.want_capture_mouse) {
            const float wheel = input.mouse_wheel;
            if (std::abs(wheel) > 0.01f) {
                if (input.key_ctrl || input.key_super) {
                    const float mouse_time = xToTime(mx, pos.x, width);
                    const float anchor_ratio = std::clamp((mx - pos.x) / width, 0.0f, 1.0f);
                    const float old_zoom = zoom_level_;
                    const float zoom_factor = std::pow(1.0f + ZOOM_SPEED, wheel);
                    zoom_level_ = std::clamp(old_zoom * zoom_factor, MIN_ZOOM, MAX_ZOOM);

                    if (zoom_level_ != old_zoom) {
                        const float new_visible_duration = getDisplayEndTime();
                        pan_offset_ = mouse_time - anchor_ratio * new_visible_duration;
                        clampPanOffset();
                    }
                } else {
                    const float pan_step = std::max(getDisplayEndTime() * 0.12f, 0.1f);
                    pan_offset_ -= wheel * pan_step;
                    clampPanOffset();
                }
            }
        }

        hovered_keyframe_ = std::nullopt;
        const auto& keyframes = timeline.keyframes();
        for (size_t i = 0; i < keyframes.size(); ++i) {
            if (keyframes[i].is_loop_point)
                continue;
            const float x = timeToX(keyframes[i].time, pos.x, width);
            const float dist = std::abs(mx - x);
            const bool hovered = mouse_in_timeline && dist < KEYFRAME_RADIUS * s * 2;
            if (hovered)
                hovered_keyframe_ = i;
        }

        const float playhead_x = timeToX(controller_.playhead(), pos.x, width);
        const float playhead_dist = std::abs(mx - playhead_x);
        bool on_playhead_handle = playhead_dist < PLAYHEAD_HIT_RADIUS * s;

        if (on_playhead_handle && hovered_keyframe_.has_value()) {
            const float kf_x = timeToX(keyframes[*hovered_keyframe_].time, pos.x, width);
            if (std::abs(mx - kf_x) < playhead_dist)
                on_playhead_handle = false;
        }

        for (size_t i = 0; i < keyframes.size(); ++i) {
            const bool hovered = hovered_keyframe_.has_value() && *hovered_keyframe_ == i;

            if (hovered && input.mouse_clicked[0] && !on_playhead_handle) {
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
                        for (size_t j = lo; j <= hi; ++j) {
                            if (!keyframes[j].is_loop_point)
                                selected_keyframes_.insert(keyframes[j].id);
                        }
                    } else if (input.key_ctrl) {
                        const auto id = keyframes[i].id;
                        if (selected_keyframes_.contains(id))
                            selected_keyframes_.erase(id);
                        else
                            selected_keyframes_.insert(id);
                    } else {
                        selected_keyframes_.clear();
                        lfs::core::events::cmd::SequencerSelectKeyframe{.keyframe_index = i}.emit();
                        const bool is_first = (i == 0);
                        if (!is_first) {
                            dragging_keyframe_ = true;
                            dragged_keyframe_changed_ = false;
                            dragged_keyframe_id_ = keyframes[i].id;
                            drag_start_mouse_x_ = mx;
                        } else {
                            lfs::core::events::cmd::SequencerGoToKeyframe{.keyframe_index = i}.emit();
                        }
                    }
                }
            }
        }

        if (input.mouse_clicked[0] && mouse_in_timeline && !dragging_keyframe_ &&
            (on_playhead_handle || !hovered_keyframe_.has_value())) {
            dragging_playhead_ = true;
            controller_.beginScrub();
        }
        if (dragging_playhead_) {
            if (input.mouse_down[0]) {
                float time = xToTime(mx, pos.x, width);
                time = std::clamp(time, 0.0f, timeline.endTime());
                if (ui_state_.snap_to_grid)
                    time = snapTime(time);
                controller_.scrub(time);
            } else {
                dragging_playhead_ = false;
                controller_.endScrub();
            }
        }

        if (dragging_keyframe_) {
            if (input.mouse_down[0]) {
                float new_time = xToTime(mx, pos.x, width);
                new_time = std::max(new_time, MIN_KEYFRAME_SPACING);
                if (ui_state_.snap_to_grid)
                    new_time = snapTime(new_time);
                if (controller_.previewKeyframeTimeById(dragged_keyframe_id_, new_time))
                    dragged_keyframe_changed_ = true;
            } else {
                if (dragged_keyframe_changed_)
                    controller_.commitKeyframeTimeById(dragged_keyframe_id_);

                if (!dragged_keyframe_changed_ && std::abs(mx - drag_start_mouse_x_) < DRAG_THRESHOLD_PX) {
                    if (const auto index = controller_.timeline().findKeyframeIndex(dragged_keyframe_id_); index.has_value()) {
                        lfs::core::events::cmd::SequencerGoToKeyframe{.keyframe_index = *index}.emit();
                    }
                }
                dragging_keyframe_ = false;
                const bool emit_keyframe_change = dragged_keyframe_changed_;
                dragged_keyframe_changed_ = false;
                dragged_keyframe_id_ = sequencer::INVALID_KEYFRAME_ID;
                if (emit_keyframe_change)
                    lfs::core::events::state::KeyframeListChanged{.count = controller_.timeline().realKeyframeCount()}.emit();
            }
        }

        if ((controller_.hasSelection() || !selected_keyframes_.empty()) &&
            input.key_delete_pressed) {
            std::vector<sequencer::KeyframeId> to_delete;
            if (!selected_keyframes_.empty())
                to_delete.assign(selected_keyframes_.begin(), selected_keyframes_.end());
            else if (auto selected_id = controller_.selectedKeyframeId(); selected_id.has_value())
                to_delete.push_back(*selected_id);

            const auto& keyframes = controller_.timeline().keyframes();
            const auto first_real_it = std::find_if(
                keyframes.begin(), keyframes.end(),
                [](const sequencer::Keyframe& keyframe) { return !keyframe.is_loop_point; });
            if (first_real_it != keyframes.end())
                std::erase(to_delete, first_real_it->id);

            bool removed_any = false;
            for (const auto id : to_delete)
                removed_any |= controller_.removeKeyframeById(id);
            for (const auto id : to_delete)
                selected_keyframes_.erase(id);
            if (removed_any)
                lfs::core::events::state::KeyframeListChanged{.count = controller_.timeline().realKeyframeCount()}.emit();
        }

        if (mouse_in_timeline && input.mouse_clicked[1]) {
            context_menu_time_ = std::max(0.0f, xToTime(mx, pos.x, width));
            if (ui_state_.snap_to_grid)
                context_menu_time_ = snapTime(context_menu_time_);
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
        return sequencer_ui::displayEndTime(controller_.timeline(), zoom_level_);
    }

    std::optional<sequencer::KeyframeId> RmlSequencerPanel::hoveredKeyframeId() const {
        if (!hovered_keyframe_.has_value())
            return std::nullopt;
        const auto* const keyframe = controller_.timeline().getKeyframe(*hovered_keyframe_);
        if (!keyframe || keyframe->is_loop_point)
            return std::nullopt;
        return keyframe->id;
    }

    void RmlSequencerPanel::clampPanOffset() {
        pan_offset_ = std::clamp(pan_offset_, 0.0f,
                                 sequencer_ui::maxPanOffset(controller_.timeline(), zoom_level_));
    }

    float RmlSequencerPanel::timeToX(const float time, const float timeline_x, const float timeline_width) const {
        return sequencer_ui::timeToScreenX(time, timeline_x, timeline_width, getDisplayEndTime(), pan_offset_);
    }

    float RmlSequencerPanel::xToTime(const float x, const float timeline_x, const float timeline_width) const {
        return sequencer_ui::screenXToTime(x, timeline_x, timeline_width, getDisplayEndTime(), pan_offset_);
    }

    float RmlSequencerPanel::snapTime(const float time) const {
        if (!ui_state_.snap_to_grid || ui_state_.snap_interval <= 0.0f)
            return time;
        return std::round(time / ui_state_.snap_interval) * ui_state_.snap_interval;
    }

    // ── Quality Scrub Field ──────────────────────────────────

    namespace {
        constexpr int QUALITY_MIN = 15;
        constexpr int QUALITY_MAX = 28;
        constexpr float SCRUB_DRAG_THRESHOLD_PX = 4.0f;
    } // namespace

    void RmlSequencerPanel::QualityScrubListener::ProcessEvent(Rml::Event& event) {
        assert(panel);
        const auto event_id = event.GetId();
        auto* el = event.GetCurrentElement();

        if (event_id == Rml::EventId::Mousedown && el && el->GetId() == "quality-scrub") {
            const int button = event.GetParameter<int>("button", 0);
            if (button != 0)
                return;
            panel->quality_scrub_active_ = true;
            panel->quality_scrub_dragging_ = false;
            panel->quality_scrub_start_x_ = event.GetParameter<float>("mouse_x", 0.0f);
        } else if (event_id == Rml::EventId::Mousemove && panel->quality_scrub_active_) {
            const float mx = event.GetParameter<float>("mouse_x", 0.0f);
            const float dx = mx - panel->quality_scrub_start_x_;
            if (!panel->quality_scrub_dragging_ && std::abs(dx) < SCRUB_DRAG_THRESHOLD_PX)
                return;

            if (!panel->quality_scrub_dragging_) {
                panel->quality_scrub_dragging_ = true;
                if (panel->el_quality_scrub_)
                    panel->el_quality_scrub_->SetClass("is-dragging", true);
            }
            panel->applyQualityFromDrag(mx);
            event.StopPropagation();
        } else if (event_id == Rml::EventId::Mouseup && panel->quality_scrub_active_) {
            const bool was_dragging = panel->quality_scrub_dragging_;
            panel->quality_scrub_active_ = false;
            panel->quality_scrub_dragging_ = false;
            if (panel->el_quality_scrub_)
                panel->el_quality_scrub_->SetClass("is-dragging", false);

            if (!was_dragging)
                panel->enterQualityEdit();
            event.StopPropagation();
        } else if (event_id == Rml::EventId::Change && el && el->GetId() == "quality-input") {
            const bool linebreak = event.GetParameter<bool>("linebreak", false);
            if (linebreak)
                panel->exitQualityEdit(true);
        } else if (event_id == Rml::EventId::Blur && el && el->GetId() == "quality-input") {
            panel->exitQualityEdit(false);
        }
    }

    void RmlSequencerPanel::syncQualityScrub() {
        if (!el_quality_display_ || !el_quality_fill_)
            return;

        const int value = std::clamp(ui_state_.quality, QUALITY_MIN, QUALITY_MAX);
        const float t = static_cast<float>(value - QUALITY_MIN) /
                        static_cast<float>(QUALITY_MAX - QUALITY_MIN);
        const std::string pct = std::format("{:.1f}%", t * 100.0f);
        el_quality_fill_->SetProperty("width", pct);
        el_quality_display_->SetInnerRML(std::to_string(value));
    }

    void RmlSequencerPanel::applyQualityFromDrag(const float mouse_x) {
        if (!el_quality_scrub_)
            return;

        const float left = el_quality_scrub_->GetAbsoluteLeft();
        const float width = std::max(el_quality_scrub_->GetBox().GetSize().x, 1.0f);
        const float t = std::clamp((mouse_x - left) / width, 0.0f, 1.0f);
        const int value = QUALITY_MIN + static_cast<int>(std::round(t * (QUALITY_MAX - QUALITY_MIN)));
        ui_state_.quality = std::clamp(value, QUALITY_MIN, QUALITY_MAX);
        syncQualityScrub();
    }

    void RmlSequencerPanel::enterQualityEdit() {
        if (!el_quality_scrub_ || !el_quality_input_ || quality_scrub_editing_)
            return;

        quality_scrub_editing_ = true;
        el_quality_scrub_->SetClass("is-editing", true);
        el_quality_input_->SetAttribute("value", std::to_string(ui_state_.quality));
        el_quality_input_->Focus();
    }

    void RmlSequencerPanel::exitQualityEdit(const bool commit) {
        if (!quality_scrub_editing_)
            return;

        if (commit && el_quality_input_) {
            const auto text = el_quality_input_->GetAttribute<Rml::String>("value", "");
            try {
                const int val = std::stoi(text);
                ui_state_.quality = std::clamp(val, QUALITY_MIN, QUALITY_MAX);
            } catch (...) {
            }
        }

        quality_scrub_editing_ = false;
        if (el_quality_scrub_)
            el_quality_scrub_->SetClass("is-editing", false);
        syncQualityScrub();
    }

} // namespace lfs::vis

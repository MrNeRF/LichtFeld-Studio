/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// clang-format off
#include <glad/glad.h>
// clang-format on

#include "sequencer/rml_sequencer_panel.hpp"
#include "core/event_bridge/localization_manager.hpp"
#include "core/events.hpp"
#include "core/logger.hpp"
#include "gui/film_strip_renderer.hpp"
#include "gui/rmlui/rml_document_utils.hpp"
#include "gui/rmlui/rml_input_utils.hpp"
#include "gui/rmlui/rml_panel_host.hpp"
#include "gui/rmlui/rml_theme.hpp"
#include "gui/rmlui/rml_tooltip.hpp"
#include "gui/rmlui/rmlui_manager.hpp"
#include "gui/rmlui/rmlui_render_interface.hpp"
#include "gui/rmlui/sdl_rml_key_mapping.hpp"
#include "gui/string_keys.hpp"
#include "gui/ui_widgets.hpp"
#include "internal/resource_paths.hpp"
#include "io/video/video_export_options.hpp"
#include "rendering/render_constants.hpp"
#include "sequencer/interpolation.hpp"
#include "sequencer/timeline_view_math.hpp"
#include "theme/theme.hpp"

#include <RmlUi/Core.h>
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fmt/format.h>
#include <imgui_impl_opengl3.h>
#include <imgui.h>

namespace lfs::vis {

    namespace {
        constexpr float MIN_KEYFRAME_SPACING = 0.1f;
        constexpr float DOUBLE_CLICK_TIME = 0.3f;
        constexpr float DRAG_THRESHOLD_PX = 3.0f;
        constexpr float PLAYHEAD_HIT_RADIUS = 6.0f;
        constexpr float PLAYHEAD_HANDLE_WIDTH = 8.0f;

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
                return fmt::format("{}x", static_cast<int>(speed));
            return fmt::format("{:.2g}x", speed);
        }

        [[nodiscard]] std::string formatPresetShort(const lfs::io::video::VideoPreset preset) {
            return lfs::io::video::getPresetInfo(preset).name;
        }

        [[nodiscard]] std::string formatTime(const float seconds) {
            const int mins = static_cast<int>(seconds) / 60;
            const float secs = seconds - static_cast<float>(mins * 60);
            return fmt::format("{}:{:05.2f}", mins, secs);
        }

        [[nodiscard]] std::string formatTimeShort(const float seconds) {
            const int mins = static_cast<int>(seconds) / 60;
            const int secs = static_cast<int>(seconds) % 60;
            if (mins > 0) {
                return fmt::format("{}:{:02d}", mins, secs);
            }
            return fmt::format("{}s", secs);
        }

        [[nodiscard]] bool hasSelectedKeyframe(const std::vector<sequencer::KeyframeId>& selected_keyframes,
                                               const sequencer::KeyframeId id) {
            return std::find(selected_keyframes.begin(), selected_keyframes.end(), id) !=
                   selected_keyframes.end();
        }

        [[nodiscard]] uint64_t selectedKeyframeSignature(std::vector<sequencer::KeyframeId> selected_keyframes) {
            std::sort(selected_keyframes.begin(), selected_keyframes.end());
            uint64_t signature = 1469598103934665603ull;
            for (const auto id : selected_keyframes) {
                signature ^= id;
                signature *= 1099511628211ull;
            }
            return signature;
        }

        [[nodiscard]] float clampCenteredSpan(const float center,
                                              const float extent,
                                              const float span) {
            if (extent <= 0.0f)
                return 0.0f;

            const float half_span = std::max(span * 0.5f, 0.0f);
            if (extent <= span)
                return extent * 0.5f;

            return std::clamp(center, half_span, extent - half_span);
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
        else if (id == "btn-prev-keyframe")
            ctrl.seekToPreviousKeyframe();
        else if (id == "btn-stop")
            ctrl.stop();
        else if (id == "btn-play")
            ctrl.togglePlayPause();
        else if (id == "btn-next-keyframe")
            ctrl.seekToNextKeyframe();
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
        else if (id == "btn-follow") {
            ui.follow_playback = !ui.follow_playback;
            if (ui.follow_playback)
                ui.show_pip_preview = false;
        } else if (id == "btn-film-strip")
            ui.show_film_strip = !ui.show_film_strip;
        else if (id == "btn-preview") {
            ui.show_pip_preview = !ui.show_pip_preview;
            if (ui.show_pip_preview)
                ui.follow_playback = false;
        } else if (id == "btn-equirect") {
            ui.equirectangular = !ui.equirectangular;
            auto settings_event = lfs::core::events::ui::RenderSettingsChanged{};
            settings_event.equirectangular = ui.equirectangular;
            settings_event.emit();
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
        else if (id == "btn-dock-toggle")
            panel->dock_toggle_requested_ = true;
        else if (id == "btn-close-panel")
            panel->close_panel_requested_ = true;
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
        clearPendingComposite();
        unregisterFilmStripSources();
        clearFilmThumbPool();
        if (el_film_strip_gaps_)
            el_film_strip_gaps_->SetInnerRML("");
        if (el_film_strip_markers_)
            el_film_strip_markers_->SetInnerRML("");
        if (el_film_strip_dividers_)
            el_film_strip_dividers_->SetInnerRML("");
        if (el_film_strip_sprockets_top_)
            el_film_strip_sprockets_top_->SetInnerRML("");
        if (el_film_strip_sprockets_bottom_)
            el_film_strip_sprockets_bottom_->SetInnerRML("");
        fbo_.destroy();
    }

    void RmlSequencerPanel::clearPendingComposite() {
        pending_foreground_composite_ = false;
        pending_composite_x_ = 0.0f;
        pending_composite_y_ = 0.0f;
        pending_composite_width_ = 0.0f;
        pending_composite_height_ = 0.0f;
    }

    void RmlSequencerPanel::clearElementCache() {
        elements_cached_ = false;
        el_panel_ = nullptr;
        el_floating_header_ = nullptr;
        el_ruler_ = nullptr;
        el_track_bar_ = nullptr;
        el_keyframes_ = nullptr;
        el_playhead_ = nullptr;
        el_hint_ = nullptr;
        el_current_time_ = nullptr;
        el_duration_ = nullptr;
        el_play_icon_ = nullptr;
        el_btn_loop_ = nullptr;
        el_timeline_ = nullptr;
        el_header_ = nullptr;
        el_easing_stripe_ = nullptr;
        el_easing_segments_ = nullptr;
        el_easing_curves_ = nullptr;
        el_easing_indicators_ = nullptr;
        el_film_strip_panel_ = nullptr;
        el_film_strip_groove_ = nullptr;
        el_film_strip_gaps_ = nullptr;
        el_film_strip_thumbs_ = nullptr;
        el_film_strip_markers_ = nullptr;
        el_film_strip_dividers_ = nullptr;
        el_film_strip_sprockets_top_ = nullptr;
        el_film_strip_sprockets_bottom_ = nullptr;
        el_panel_guides_ = nullptr;
        el_guide_playhead_ = nullptr;
        el_guide_selected_ = nullptr;
        el_guide_hovered_ = nullptr;
        el_guide_strip_hover_ = nullptr;
        el_timeline_tooltip_ = nullptr;
        el_btn_camera_path_ = nullptr;
        el_btn_snap_ = nullptr;
        el_btn_follow_ = nullptr;
        el_btn_film_strip_ = nullptr;
        el_btn_preview_ = nullptr;
        el_speed_label_ = nullptr;
        el_format_label_ = nullptr;
        el_resolution_info_ = nullptr;
        el_quality_scrub_ = nullptr;
        el_quality_fill_ = nullptr;
        el_quality_display_ = nullptr;
        el_quality_input_ = nullptr;
        el_btn_equirect_ = nullptr;
        el_btn_save_ = nullptr;
        el_btn_load_ = nullptr;
        el_btn_export_ = nullptr;
        el_btn_clear_ = nullptr;
        el_transport_dock_sep_ = nullptr;
        el_btn_dock_toggle_ = nullptr;
        el_dock_toggle_label_ = nullptr;
        el_btn_close_panel_ = nullptr;
        el_close_panel_label_ = nullptr;
        keyframe_elements_.clear();
        film_thumb_elements_.clear();
    }

    void RmlSequencerPanel::reloadResources() {
        if (!rml_context_)
            return;

        clearPendingComposite();
        unregisterFilmStripSources();
        clearFilmThumbPool();
        if (document_) {
            rml_context_->UnloadDocument(document_);
            rml_context_->Update();
        }

        document_ = nullptr;
        base_rcss_.clear();
        has_theme_signature_ = false;
        last_theme_signature_ = 0;
        clearElementCache();
        last_keyframe_count_ = static_cast<size_t>(-1);
        last_zoom_level_ = -1.0f;
        last_pan_offset_ = -1.0f;
        last_kf_width_ = -1.0f;
        last_ruler_zoom_ = -1.0f;
        last_ruler_pan_ = -1.0f;
        last_ruler_width_ = -1.0f;
        last_ruler_display_end_ = -1.0f;
        last_timeline_revision_ = 0;
        last_selection_revision_ = 0;
        last_selected_keyframes_signature_ = 0;
        quality_scrub_active_ = false;
        quality_scrub_dragging_ = false;
        quality_scrub_editing_ = false;
        last_language_.clear();

        try {
            const auto full_path = lfs::vis::getAssetPath("rmlui/sequencer.rml");
            document_ = lfs::vis::gui::rml_documents::loadDocument(rml_context_, full_path);
            if (!document_) {
                LOG_ERROR("RmlUI: failed to reload sequencer.rml");
                return;
            }
            document_->Show();
            cacheElements();
        } catch (const std::exception& e) {
            LOG_ERROR("RmlUI: sequencer resource not found during reload: {}", e.what());
        }
    }

    void RmlSequencerPanel::compositeToScreen(const int screen_w, const int screen_h) {
        if (!pending_foreground_composite_ || !fbo_.valid() || screen_w <= 0 || screen_h <= 0) {
            clearPendingComposite();
            return;
        }

        ImDrawList draw_list(ImGui::GetDrawListSharedData());
        draw_list._ResetForNewFrame();
        draw_list.PushTextureID(ImGui::GetIO().Fonts->TexID);
        draw_list.PushClipRectFullScreen();
        gui::widgets::DrawFloatingWindowShadow(
            &draw_list,
            {pending_composite_x_, pending_composite_y_},
            {pending_composite_width_, pending_composite_height_},
            theme().sizes.window_rounding);
        draw_list.PopClipRect();

        if (!draw_list.CmdBuffer.empty() && !draw_list.VtxBuffer.empty()) {
            ImDrawData draw_data{};
            draw_data.DisplayPos = ImVec2(0.0f, 0.0f);
            draw_data.DisplaySize = ImVec2(static_cast<float>(screen_w),
                                           static_cast<float>(screen_h));
            draw_data.FramebufferScale = ImGui::GetIO().DisplayFramebufferScale;
            draw_data.Valid = true;
            draw_data.AddDrawList(&draw_list);
            ImGui_ImplOpenGL3_RenderDrawData(&draw_data);
        }

        fbo_.blitToScreen(pending_composite_x_, pending_composite_y_,
                          pending_composite_width_, pending_composite_height_,
                          screen_w, screen_h);
        clearPendingComposite();
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
            document_ = lfs::vis::gui::rml_documents::loadDocument(rml_context_, full_path);
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
        clearElementCache();
        el_panel_ = document_->GetElementById("panel");
        el_floating_header_ = document_->GetElementById("floating-header");
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
        el_header_ = document_->GetElementById("header");
        el_easing_stripe_ = document_->GetElementById("easing-stripe");
        el_easing_segments_ = document_->GetElementById("easing-segments");
        el_easing_curves_ = document_->GetElementById("easing-curves");
        el_easing_indicators_ = document_->GetElementById("easing-indicators");
        el_film_strip_panel_ = document_->GetElementById("film-strip-panel");
        el_film_strip_groove_ = document_->GetElementById("film-strip-groove");
        el_film_strip_gaps_ = document_->GetElementById("film-strip-gaps");
        el_film_strip_thumbs_ = document_->GetElementById("film-strip-thumbs");
        el_film_strip_markers_ = document_->GetElementById("film-strip-markers");
        el_film_strip_dividers_ = document_->GetElementById("film-strip-dividers");
        el_film_strip_sprockets_top_ = document_->GetElementById("film-strip-sprockets-top");
        el_film_strip_sprockets_bottom_ = document_->GetElementById("film-strip-sprockets-bottom");
        el_panel_guides_ = document_->GetElementById("panel-guides");
        el_guide_playhead_ = document_->GetElementById("guide-playhead");
        el_guide_selected_ = document_->GetElementById("guide-selected");
        el_guide_hovered_ = document_->GetElementById("guide-hovered");
        el_guide_strip_hover_ = document_->GetElementById("guide-strip-hover");
        el_timeline_tooltip_ = document_->GetElementById("timeline-tooltip");

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
        el_transport_dock_sep_ = document_->GetElementById("dock-toggle-sep");
        el_btn_dock_toggle_ = document_->GetElementById("btn-dock-toggle");
        el_dock_toggle_label_ = document_->GetElementById("dock-toggle-label");
        el_btn_close_panel_ = document_->GetElementById("btn-close-panel");
        el_close_panel_label_ = document_->GetElementById("close-panel-label");

        elements_cached_ = el_ruler_ && el_keyframes_ && el_playhead_ &&
                           el_current_time_ && el_duration_ && el_play_icon_ &&
                           el_btn_loop_ && el_timeline_ && el_header_ &&
                           el_easing_stripe_ && el_easing_segments_ &&
                           el_easing_curves_ && el_easing_indicators_ &&
                           el_film_strip_panel_ && el_film_strip_groove_ &&
                           el_film_strip_gaps_ && el_film_strip_thumbs_ &&
                           el_film_strip_markers_ && el_film_strip_dividers_ &&
                           el_film_strip_sprockets_top_ && el_film_strip_sprockets_bottom_ &&
                           el_panel_guides_ && el_guide_playhead_ &&
                           el_guide_selected_ && el_guide_hovered_ &&
                           el_guide_strip_hover_ && el_timeline_tooltip_;
        if (!elements_cached_) {
            LOG_ERROR("RmlUI sequencer: missing DOM elements");
            return;
        }

        for (const char* btn_id : {"btn-skip-back", "btn-stop", "btn-play",
                                   "btn-prev-keyframe", "btn-next-keyframe", "btn-skip-forward",
                                   "btn-loop", "btn-add",
                                   "btn-camera-path", "btn-snap", "btn-follow",
                                   "btn-film-strip", "btn-preview", "btn-equirect", "btn-speed",
                                   "btn-format", "btn-save-path", "btn-load-path",
                                   "btn-export", "btn-clear", "btn-dock-toggle",
                                   "btn-close-panel"}) {
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

    void RmlSequencerPanel::syncTheme() {
        if (!document_)
            return;

        const std::size_t theme_signature = gui::rml_theme::currentThemeSignature();
        const bool layout_changed = film_strip_attached_ != last_film_strip_attached_ ||
                                    floating_ != last_floating_;
        if (!layout_changed && has_theme_signature_ && theme_signature == last_theme_signature_)
            return;
        last_theme_signature_ = theme_signature;
        has_theme_signature_ = true;
        last_film_strip_attached_ = film_strip_attached_;
        last_floating_ = floating_;

        if (base_rcss_.empty())
            base_rcss_ = gui::rml_theme::loadBaseRCSS("rmlui/sequencer.rcss");

        gui::rml_theme::applyTheme(document_, base_rcss_, gui::rml_theme::loadBaseRCSS("rmlui/sequencer.theme.rcss"));
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

        const float x = clampCenteredSpan(
            timeToX(controller_.playhead(), 0.0f, tl_width),
            tl_width,
            PLAYHEAD_HANDLE_WIDTH * cached_dp_ratio_);
        el_playhead_->SetProperty("left", fmt::format("{:.1f}px", x));
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

        std::erase_if(selected_keyframes_, [&timeline](const sequencer::KeyframeId id) {
            return !timeline.findKeyframeIndex(id).has_value();
        });

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
                                  hasSelectedKeyframe(selected_keyframes_, keyframes[i].id);
            const bool is_loop = keyframes[i].is_loop_point;

            const auto base = is_loop ? p.info : (i % 2 == 0 ? p.primary : p.secondary);
            auto fill = base;
            if (selected)
                fill = lighten(base, 0.2f);

            el->SetClassNames("keyframe");
            el->SetClass("loop-point", is_loop);
            el->SetClass("selected", selected);
            el->SetProperty("left", fmt::format("{:.1f}px", x));
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
                html += fmt::format(
                    "<div class=\"ruler-tick major\" style=\"left: {:.1f}px;\" />", x);
                if (x + label_margin <= timeline_width) {
                    html += fmt::format(
                        "<span class=\"ruler-label\" style=\"left: {:.1f}px;\">{}</span>",
                        x + 4.0f * cached_dp_ratio_, formatTimeShort(t_val));
                }
            } else {
                html += fmt::format(
                    "<div class=\"ruler-tick minor\" style=\"left: {:.1f}px;\" />",
                    x);
            }
        }

        el_ruler_->SetInnerRML(html);
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

    bool RmlSequencerPanel::consumeDockToggleRequest() {
        const bool request = dock_toggle_requested_;
        dock_toggle_requested_ = false;
        return request;
    }

    bool RmlSequencerPanel::consumeClosePanelRequest() {
        const bool request = close_panel_requested_;
        close_panel_requested_ = false;
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
            el_resolution_info_->SetInnerRML(fmt::format("{}x{} @ {}fps", w, h, fps));
        }
        if (!quality_scrub_editing_)
            syncQualityScrub();

        if (el_btn_save_)
            el_btn_save_->SetClass("disabled", !has_camera_keyframes);
        if (el_btn_export_)
            el_btn_export_->SetClass("disabled", !has_camera_keyframes);
        if (el_btn_clear_)
            el_btn_clear_->SetClass("disabled", !has_any_state);
        if (el_panel_) {
            el_panel_->SetClass("is-floating", floating_);
            el_panel_->SetClass("film-strip-attached", film_strip_attached_);
        }
        if (el_floating_header_)
            el_floating_header_->SetClass("hidden", !floating_);
        if (el_film_strip_panel_)
            el_film_strip_panel_->SetProperty("display", film_strip_attached_ ? "block" : "none");
        if (el_transport_dock_sep_)
            el_transport_dock_sep_->SetClass("hidden", false);
        if (el_btn_dock_toggle_) {
            el_btn_dock_toggle_->SetAttribute("data-tooltip",
                                              floating_ ? "tooltip.seq_dock" : "tooltip.seq_undock");
            el_btn_dock_toggle_->SetClass("active", false);
            el_btn_dock_toggle_->SetClass("hidden", false);
        }
        if (el_dock_toggle_label_)
            el_dock_toggle_label_->SetInnerRML(floating_ ? "Dock" : "Undock");
        if (el_btn_close_panel_) {
            el_btn_close_panel_->SetAttribute("data-tooltip", "common.close");
            el_btn_close_panel_->SetClass("hidden", !floating_);
        }
        if (el_close_panel_label_)
            el_close_panel_label_->SetInnerRML(lfs::event::LocalizationManager::getInstance().get("common.close"));
    }

    float RmlSequencerPanel::timelineWidth() const {
        const float s = cached_dp_ratio_;
        return cached_panel_width_ - 2.0f * INNER_PADDING_H * s;
    }

    void RmlSequencerPanel::render(const float panel_x, const float panel_y,
                                   const float panel_width, const float total_height,
                                   const PanelInputState& input,
                                   RenderingManager* rm, SceneManager* sm,
                                   gui::FilmStripRenderer& film_strip) {
        clearPendingComposite();
        const float dp = rml_manager_->getDpRatio();
        cached_dp_ratio_ = dp;

        const float strip_height = film_strip_attached_ ? gui::FilmStripRenderer::STRIP_HEIGHT : 0.0f;
        const float easing_height = EASING_STRIPE_HEIGHT * dp;
        cached_total_height_ = std::max(0.0f, total_height);
        cached_height_ = std::max(0.0f, total_height - easing_height - strip_height);

        cached_panel_x_ = panel_x;
        cached_panel_y_ = panel_y;
        cached_panel_width_ = panel_width;

        const int w = static_cast<int>(panel_width);
        const int h = static_cast<int>(cached_total_height_);

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
            el_timeline_->SetProperty("width", fmt::format("{:.1f}px", timelineWidth()));
        }

        forwardInput(input);

        const float inner_pad_h = INNER_PADDING_H * dp;
        const float inner_pad = INNER_PADDING * dp;
        const float transport_row_h = TRANSPORT_ROW_HEIGHT * dp;
        const float content_height = cached_height_ - 2.0f * inner_pad - transport_row_h;
        const float tl_width = timelineWidth();

        const Vec2 timeline_pos = {panel_x + inner_pad_h,
                                   panel_y + inner_pad + transport_row_h};

        if (elements_cached_) {
            handleTimelineInteraction(timeline_pos, tl_width, content_height, input);
            handleEasingStripeInteraction(timeline_pos.x, tl_width, input);
            rebuildFilmStrip(timeline_pos.x, tl_width,
                             panel_y + cached_height_ + easing_height - BORDER_OVERLAP * dp,
                             input, rm, sm, film_strip);

            cached_playhead_screen_x_ = timeline_pos.x + clampCenteredSpan(
                                                             timeToX(controller_.playhead(), 0.0f, tl_width),
                                                             tl_width,
                                                             PLAYHEAD_HANDLE_WIDTH * dp);
            playhead_in_range_ = cached_playhead_screen_x_ >= timeline_pos.x &&
                                 cached_playhead_screen_x_ <= timeline_pos.x + tl_width;

            updateButtonStates();
            updateTransportSettings();
            updatePlayhead();
            updateTimeDisplay();
            rebuildKeyframes();
            rebuildRuler();
            rebuildEasingStripe(timeline_pos.x, tl_width);
            updateTimelineGuides(timeline_pos.x, tl_width, film_strip);
        }

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

            if (!floating_) {
                const auto& shell_bg = theme().menu_background();
                glClearColor(shell_bg.x, shell_bg.y, shell_bg.z, 1.0f);
                glClear(GL_COLOR_BUFFER_BIT);
            }

            render_iface->BeginFrame();
            rml_context_->Render();
            render_iface->EndFrame();

            render_iface->SetTargetFramebuffer(0);
            fbo_.unbind(prev_fbo);
        }

        if (!fbo_.valid())
            return;

        if (floating_) {
            pending_foreground_composite_ = true;
            pending_composite_x_ = panel_x;
            pending_composite_y_ = panel_y;
            pending_composite_width_ = panel_width;
            pending_composite_height_ = cached_total_height_;
            return;
        }

        fbo_.blitToScreen(panel_x, panel_y, panel_width, cached_total_height_,
                          input.screen_w, input.screen_h);
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
        const std::string pct = fmt::format("{:.1f}%", t * 100.0f);
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

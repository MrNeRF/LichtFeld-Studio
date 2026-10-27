/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "sequencer/rml_sequencer_panel.hpp"
#include "core/event_bridge/localization_manager.hpp"
#include "core/events.hpp"
#include "core/logger.hpp"
#include "gui/film_strip_renderer.hpp"
#include "gui/rmlui/rml_document_utils.hpp"
#include "gui/rmlui/rml_input_utils.hpp"
#include "gui/rmlui/rml_theme.hpp"
#include "gui/rmlui/rml_tooltip.hpp"
#include "gui/rmlui/rmlui_manager.hpp"
#include "gui/rmlui/sdl_rml_key_mapping.hpp"
#include "gui/string_keys.hpp"
#include "gui/ui_widgets.hpp"
#include "internal/resource_paths.hpp"
#include "io/video/video_export_options.hpp"
#include "rendering/render_constants.hpp"
#include "rendering/rendering_manager.hpp"
#include "sequencer/interpolation.hpp"
#include "sequencer/reveal_effect_i18n.hpp"
#include "sequencer/timeline_view_math.hpp"
#include "theme/theme.hpp"

#include <RmlUi/Core.h>
#include <RmlUi/Core/Input.h>
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fmt/format.h>
#include <stdexcept>

namespace lfs::vis {

    namespace {
        constexpr float MIN_KEYFRAME_SPACING = 0.1f;
        constexpr float DOUBLE_CLICK_TIME = 0.3f;
        constexpr float DRAG_THRESHOLD_PX = 3.0f;

        [[nodiscard]] std::string formatSpeed(const float speed) {
            return fmt::format("{:.3g}x", speed);
        }

        [[nodiscard]] std::string formatSequenceFps(const float fps) {
            const float clamped = std::clamp(fps, MIN_SEQUENCE_FPS, MAX_SEQUENCE_FPS);
            const float rounded = std::round(clamped);
            if (std::abs(clamped - rounded) < 0.01f)
                return fmt::format("{}", static_cast<int>(rounded));
            return fmt::format("{:.2f}", clamped);
        }

        [[nodiscard]] std::string formatTime(const float seconds) {
            const int mins = static_cast<int>(seconds) / 60;
            const float secs = seconds - static_cast<float>(mins * 60);
            return fmt::format("{}:{:05.2f}", mins, secs);
        }

        [[nodiscard]] std::string formatTimeShort(const float seconds, const float major_interval) {
            if (seconds >= 60.0f) {
                const int total = static_cast<int>(seconds);
                return fmt::format("{}:{:02d}", total / 60, total % 60);
            }
            if (major_interval < 1.0f) {
                const int decimals = (major_interval < 0.5f) ? 2 : 1;
                std::string body = fmt::format("{:.{}f}", seconds, decimals);
                const auto dot = body.find('.');
                if (dot != std::string::npos) {
                    while (body.size() > dot && body.back() == '0')
                        body.pop_back();
                    if (!body.empty() && body.back() == '.')
                        body.pop_back();
                }
                return body + "s";
            }
            return fmt::format("{}s", static_cast<int>(seconds));
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

        [[nodiscard]] int milli(const float value) {
            return static_cast<int>(std::lround(value * 1000.0f));
        }

        [[nodiscard]] bool hasInputActivity(const PanelInputState& input) {
            for (int i = 0; i < 3; ++i) {
                if (input.mouse_down[i] || input.mouse_clicked[i] || input.mouse_released[i])
                    return true;
            }
            return input.mouse_wheel != 0.0f ||
                   !input.keys_pressed.empty() ||
                   !input.input_events.empty();
        }

        [[nodiscard]] bool mouseOverPanel(const PanelInputState& input,
                                          const float panel_x, const float panel_y,
                                          const float panel_width, const float panel_height) {
            return input.mouse_x >= panel_x && input.mouse_x < panel_x + panel_width &&
                   input.mouse_y >= panel_y && input.mouse_y < panel_y + panel_height;
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
        duration_listener_.panel = this;
        reveal_listener_.panel = this;
        sequence_fps_listener_.panel = this;
    }

    RmlSequencerPanel::~RmlSequencerPanel() = default;

    void RmlSequencerPanel::clearKeyframeSelection() {
        selected_keyframes_.clear();
        controller_.deselectKeyframe();
    }

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
        } else if (id == "btn-speed" || id == "btn-path-menu" || id == "btn-load-sequence" ||
                   id == "btn-snap-options" || id == "btn-preview-options" || id == "btn-equirect" || id == "btn-camera-aspect") {
            using Target = TransportContextMenuRequest::Target;
            const auto target = id == "btn-camera-aspect" ? Target::CAMERA_ASPECT : id == "btn-equirect"    ? Target::PROJECTION
                                                                                : id == "btn-speed"         ? Target::SPEED
                                                                                : id == "btn-path-menu"     ? Target::PATH
                                                                                : id == "btn-load-sequence" ? Target::PLY
                                                                                : id == "btn-snap-options"  ? Target::SNAP
                                                                                                            : Target::PREVIEW;
            const auto offset = el->GetAbsoluteOffset(Rml::BoxArea::Border);
            panel->transport_ctx_request_ = {target, panel->cached_panel_x_ + offset.x,
                                             panel->cached_panel_y_ + offset.y + el->GetBox().GetSize().y};
        } else if (id == "btn-add-reveal") {
            const auto offset = el->GetAbsoluteOffset(Rml::BoxArea::Border);
            panel->transport_ctx_request_ = {TransportContextMenuRequest::Target::REVEAL,
                                             panel->cached_panel_x_ + offset.x,
                                             panel->cached_panel_y_ + offset.y + el->GetBox().GetSize().y};
        } else if (id == "btn-remove-reveal") {
            ctrl.setRevealClip(std::nullopt);
            panel->selectReveal(false);
        } else if (id == "ply-sequence-clip") {
            panel->selectReveal(false);
            panel->ply_sequence_selected_ = true;
            panel->selected_keyframes_.clear();
            ctrl.deselectKeyframe();
            event.StopPropagation();
        } else if (id == "reveal-clip") {
            panel->selectReveal(true);
            event.StopPropagation();
        } else if (id == "btn-export")
            panel->export_requested_ = true;
        else if (id == "btn-dock-toggle")
            panel->dock_toggle_requested_ = true;
        else if (id == "btn-close-panel")
            panel->close_panel_requested_ = true;
    }

    TimelineContextMenuState RmlSequencerPanel::consumeContextMenu() {
        TimelineContextMenuState state;
        if (context_menu_open_) {
            state.open = true;
            state.time = context_menu_time_;
            state.keyframe = context_menu_keyframe_;
            state.reveal = context_menu_reveal_;
            context_menu_open_ = false;
            context_menu_reveal_ = false;
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

    void RmlSequencerPanel::destroyGraphicsResources() {
        timeline_markup_.clear();
        clearPendingComposite();
        if (rml_manager_)
            rml_manager_->releaseCachedVulkanContext(direct_cache_);
        direct_cache_dirty_ = true;
        last_render_signature_.reset();
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
        if (el_sequence_strip_)
            el_sequence_strip_->SetInnerRML("");
    }

    void RmlSequencerPanel::clearPendingComposite() {
    }

    bool RmlSequencerPanel::needsLocalizationFrame() const {
        const std::string current_language =
            lfs::event::LocalizationManager::getInstance().getCurrentLanguage();
        return !current_language.empty() && current_language != last_language_;
    }

    void RmlSequencerPanel::clearElementCache() {
        timeline_markup_.clear();
        elements_cached_ = false;
        el_panel_ = nullptr;
        el_floating_header_ = nullptr;
        el_ruler_ = nullptr;
        el_track_bar_ = nullptr;
        el_sequence_strip_ = nullptr;
        el_keyframes_ = nullptr;
        el_playhead_ = nullptr;
        el_playhead_handle_ = nullptr;
        el_timeline_scrollbar_ = nullptr;
        el_timeline_scrollbar_thumb_ = nullptr;
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
        el_sequence_fps_field_ = nullptr;
        el_sequence_fps_display_ = nullptr;
        el_sequence_fps_input_ = nullptr;
        el_duration_field_ = nullptr;
        el_duration_input_ = nullptr;
        duration_editing_ = false;
        sequence_fps_editing_ = false;
        el_btn_equirect_ = nullptr;
        el_btn_load_sequence_ = nullptr;
        el_btn_dock_toggle_ = nullptr;
        el_btn_close_panel_ = nullptr;
        el_close_panel_label_ = nullptr;
        keyframe_elements_.clear();
        film_thumb_elements_.clear();
    }

    void RmlSequencerPanel::reloadResources() {
        if (!rml_context_)
            return;

        clearPendingComposite();
        if (rml_manager_)
            rml_manager_->releaseCachedVulkanContext(direct_cache_);
        direct_cache_dirty_ = true;
        last_render_signature_.reset();
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
        last_ruler_dp_ratio_ = -1.0f;
        last_ruler_display_end_ = -1.0f;
        last_timeline_revision_ = 0;
        last_selection_revision_ = 0;
        last_selected_keyframes_signature_ = 0;
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

    RmlSequencerPanel::RenderSignature RmlSequencerPanel::makeRenderSignature(
        const int width,
        const int height,
        const std::size_t theme_signature,
        std::string language) const {
        return {
            .width = width,
            .height = height,
            .dp_milli = milli(cached_dp_ratio_),
            .floating = floating_,
            .film_strip_attached = film_strip_attached_,
            .theme_signature = theme_signature,
            .language = std::move(language),
            .timeline_revision = controller_.timelineRevision(),
            .selection_revision = controller_.selectionRevision(),
            .selected_keyframes_signature = selectedKeyframeSignature(selected_keyframes_),
            .playhead_milli = milli(controller_.playhead()),
            .zoom_milli = milli(zoom_level_),
            .pan_milli = milli(pan_offset_),
            .playback_speed_milli = milli(ui_state_.playback_speed),
            .sequence_fps_milli = milli(ui_state_.sequence_fps),
            .sequence_frame_count = controller_.hasPlySequence() ? controller_.plySequence()->frames.size() : 0,
            .sequence_node_name = controller_.hasPlySequence() ? controller_.plySequence()->node_name : "",
            .snap_interval_milli = milli(ui_state_.snap_interval),
            .pip_scale_milli = milli(ui_state_.pip_preview_scale),
            .state = static_cast<int>(controller_.state()),
            .loop_mode = static_cast<int>(controller_.loopMode()),
            .follow_playback = ui_state_.follow_playback,
            .show_camera_path = ui_state_.show_camera_path,
            .camera_width = ui_state_.camera_width,
            .camera_height = ui_state_.camera_height,
            .snap_to_grid = ui_state_.snap_to_grid,
            .show_film_strip = ui_state_.show_film_strip,
            .show_pip_preview = ui_state_.show_pip_preview,
            .equirectangular = ui_state_.equirectangular,
        };
    }

    bool RmlSequencerPanel::canReuseCachedRender(const RenderSignature& signature,
                                                 const PanelInputState& input,
                                                 const int width,
                                                 const int height) const {
        const bool mouse_over_panel = mouseOverPanel(input, cached_panel_x_, cached_panel_y_,
                                                     cached_panel_width_, cached_total_height_);
        const bool mouse_moved = std::abs(input.mouse_x - last_render_mouse_x_) >= 0.5f ||
                                 std::abs(input.mouse_y - last_render_mouse_y_) >= 0.5f;
        // Hover classes (playhead handle, scrollbar) only update on a full
        // render. Invalidate while the cursor is over the panel, and once more
        // when it leaves so hot classes clear. Mouse motion over the viewport
        // must not bust the cache.
        const bool hover_needs_frame = (mouse_over_panel && mouse_moved) ||
                                       (!mouse_over_panel && last_render_mouse_over_panel_);

        return !direct_cache_dirty_ &&
               direct_cache_.texture != 0 &&
               direct_cache_.width == width &&
               direct_cache_.height == height &&
               last_render_signature_.has_value() &&
               *last_render_signature_ == signature &&
               !hasInputActivity(input) &&
               !hover_needs_frame &&
               !tooltip_.needsFrame() &&
               !duration_editing_ &&
               !sequence_fps_editing_;
    }

    void RmlSequencerPanel::queueCachedRender(const float context_x,
                                              const float context_y,
                                              const float panel_width,
                                              const float total_height,
                                              const int width,
                                              const int height,
                                              const bool refresh) {
        if (!rml_manager_ || !rml_context_)
            return;
        rml_manager_->queueCachedVulkanContext({
            .context = rml_context_,
            .cache = &direct_cache_,
            .cache_width = width,
            .cache_height = height,
            .offset_x = context_x,
            .offset_y = context_y,
            .draw_width = panel_width,
            .draw_height = total_height,
            .refresh = refresh,
            .foreground = floating_,
            .clip_enabled = true,
            .clip = {
                .x1 = context_x,
                .y1 = context_y,
                .x2 = context_x + panel_width,
                .y2 = context_y + total_height,
            },
        });
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
        el_sequence_strip_ = document_->GetElementById("sequence-strip");
        el_keyframes_ = document_->GetElementById("keyframes");
        el_playhead_ = document_->GetElementById("playhead");
        el_playhead_handle_ = document_->GetElementById("playhead-handle");
        el_timeline_scrollbar_ = document_->GetElementById("timeline-scrollbar");
        el_timeline_scrollbar_thumb_ = document_->GetElementById("timeline-scrollbar-thumb");
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
        el_sequence_fps_field_ = document_->GetElementById("sequence-fps-field");
        el_sequence_fps_display_ = document_->GetElementById("sequence-fps-display");
        el_sequence_fps_input_ = document_->GetElementById("sequence-fps-input");
        el_duration_field_ = document_->GetElementById("duration-field");
        el_duration_input_ = document_->GetElementById("duration-input");
        el_btn_equirect_ = document_->GetElementById("btn-equirect");
        el_btn_load_sequence_ = document_->GetElementById("btn-load-sequence");
        el_btn_dock_toggle_ = document_->GetElementById("btn-dock-toggle");
        el_btn_close_panel_ = document_->GetElementById("btn-close-panel");
        el_close_panel_label_ = document_->GetElementById("close-panel-label");

        elements_cached_ = el_ruler_ && el_sequence_strip_ && el_keyframes_ && el_playhead_ && el_playhead_handle_ &&
                           el_timeline_scrollbar_ && el_timeline_scrollbar_thumb_ &&
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
                                   "btn-path-menu", "btn-snap-options", "btn-preview-options", "btn-camera-aspect",
                                   "btn-load-sequence", "btn-add-reveal", "btn-remove-reveal",
                                   "btn-export", "btn-dock-toggle",
                                   "btn-close-panel"}) {
            auto* el = document_->GetElementById(btn_id);
            if (el)
                el->AddEventListener(Rml::EventId::Click, &transport_listener_);
        }

        for (const char* id : {"reveal-start", "reveal-end", "reveal-softness"}) {
            if (auto* el = document_->GetElementById(id)) {
                el->AddEventListener(Rml::EventId::Change, &reveal_listener_);
                el->AddEventListener(Rml::EventId::Blur, &reveal_listener_);
            }
        }

        if (el_duration_field_)
            el_duration_field_->AddEventListener(Rml::EventId::Click, &duration_listener_);
        if (el_duration_input_) {
            el_duration_input_->AddEventListener(Rml::EventId::Change, &duration_listener_);
            el_duration_input_->AddEventListener(Rml::EventId::Blur, &duration_listener_);
        }

        if (el_sequence_fps_field_)
            el_sequence_fps_field_->AddEventListener(Rml::EventId::Click, &sequence_fps_listener_);
        if (el_sequence_fps_input_) {
            el_sequence_fps_input_->AddEventListener(Rml::EventId::Change, &sequence_fps_listener_);
            el_sequence_fps_input_->AddEventListener(Rml::EventId::Blur, &sequence_fps_listener_);
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

    void RmlSequencerPanel::updateTimelineScrollbar() {
        if (!elements_cached_ || !el_timeline_scrollbar_ || !el_timeline_scrollbar_thumb_)
            return;

        const float duration = controller_.timeline().clipDuration();
        const float max_pan = sequencer_ui::maxPanOffset(controller_.timeline(), zoom_level_);
        const bool visible = max_pan > 0.0f && duration > 0.0f;
        el_timeline_scrollbar_->SetClass("visible", visible);
        if (!visible) {
            el_timeline_scrollbar_->SetClass("hot", false);
            return;
        }

        const float left_frac = std::clamp(pan_offset_ / duration, 0.0f, 1.0f);
        const float width_frac = std::clamp(getDisplayEndTime() / duration, 0.0f, 1.0f);
        el_timeline_scrollbar_thumb_->SetProperty("left", fmt::format("{:.3f}%", left_frac * 100.0f));
        el_timeline_scrollbar_thumb_->SetProperty("width", fmt::format("{:.3f}%", width_frac * 100.0f));
    }

    void RmlSequencerPanel::updateTimeDisplay() {
        if (!elements_cached_)
            return;

        el_current_time_->SetInnerRML(formatTime(controller_.playhead()));
        syncDurationDisplay();
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
            if (el_hint_) {
                if (controller_.hasPlySequence() || controller_.timeline().revealClip()) {
                    el_hint_->SetInnerRML("");
                } else {
                    el_hint_->SetInnerRML(LOC(lichtfeld::Strings::Sequencer::EMPTY_HINT));
                }
            }
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

    void RmlSequencerPanel::rebuildPlySequenceClip() {
        if (!elements_cached_ || !el_sequence_strip_)
            return;

        const auto* const sequence = controller_.plySequence();
        const auto& reveal = controller_.timeline().revealClip();
        const float timeline_width = timelineWidth();
        const bool has_sequence = sequence && !sequence->frames.empty();
        if (auto* track = document_->GetElementById("track-area"))
            track->SetClass("two-clips", has_sequence && reveal.has_value());
        if (auto* strip = document_->GetElementById("reveal-strip")) {
            if (reveal && timeline_width > 0.0f) {
                const float start_x = timeToX(reveal->start, 0.0f, timeline_width);
                const float end_x = timeToX(reveal->end, 0.0f, timeline_width);
                const auto html = fmt::format(
                    "<div id=\"reveal-clip\" class=\"sequence-clip reveal-clip\" style=\"left: {:.1f}px; width: {:.1f}px;\"><span class=\"sequence-clip-label\">{}</span></div>",
                    start_x, std::max(2.0f * cached_dp_ratio_, end_x - start_x),
                    Rml::StringUtilities::EncodeRml(LOC(revealEffectKey(reveal->effect))));
                setTimelineMarkup(strip, html);
                if (auto* el = document_->GetElementById("reveal-clip")) {
                    el->RemoveEventListener(Rml::EventId::Click, &transport_listener_);
                    el->AddEventListener(Rml::EventId::Click, &transport_listener_);
                }
            } else {
                setTimelineMarkup(strip, "");
            }
        }
        if (!sequence || sequence->frames.empty() || timeline_width <= 0.0f) {
            setTimelineMarkup(el_sequence_strip_, "");
            return;
        }

        const float duration = sequence->duration();
        if (duration <= 0.0f) {
            setTimelineMarkup(el_sequence_strip_, "");
            return;
        }

        const float start_x = timeToX(0.0f, 0.0f, timeline_width);
        const float end_x = timeToX(duration, 0.0f, timeline_width);
        const float width = std::max(2.0f * cached_dp_ratio_, end_x - start_x);
        const std::string label = Rml::StringUtilities::EncodeRml(
            fmt::format("{}  {} frames @ {}",
                        sequence->node_name.empty() ? "PLY Sequence" : sequence->node_name,
                        sequence->frames.size(),
                        formatSequenceFps(sequence->fps)));

        std::string html;
        html.reserve(2048);
        html += fmt::format(
            "<div id=\"ply-sequence-clip\" class=\"sequence-clip{}\" style=\"left: {:.1f}px; width: {:.1f}px;\">",
            ply_sequence_selected_ ? " selected" : "",
            start_x,
            width);
        html += fmt::format("<span class=\"sequence-clip-label\">{}</span>", label);

        const float tick_spacing = width / static_cast<float>(sequence->frames.size());
        if (sequence->frames.size() <= 240 && tick_spacing >= 3.0f * cached_dp_ratio_) {
            for (size_t i = 1; i < sequence->frames.size(); ++i) {
                html += fmt::format(
                    "<div class=\"sequence-frame-tick\" style=\"left: {:.1f}px;\"></div>",
                    static_cast<float>(i) * tick_spacing);
            }
        }

        html += "</div>";
        setTimelineMarkup(el_sequence_strip_, html);
        if (auto* el = document_->GetElementById("ply-sequence-clip")) {
            el->RemoveEventListener(Rml::EventId::Click, &transport_listener_);
            el->AddEventListener(Rml::EventId::Click, &transport_listener_);
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
            cached_dp_ratio_ == last_ruler_dp_ratio_ &&
            display_end_time == last_ruler_display_end_)
            return;
        last_ruler_zoom_ = zoom_level_;
        last_ruler_pan_ = pan_offset_;
        last_ruler_width_ = timeline_width;
        last_ruler_dp_ratio_ = cached_dp_ratio_;
        last_ruler_display_end_ = display_end_time;
        if (timeline_width <= 0.0f)
            return;

        const float visible_duration = display_end_time;
        const float visible_start = pan_offset_;
        const float visible_end = visible_start + visible_duration;

        const float label_margin = 30.0f * cached_dp_ratio_;
        // Leave room for the label span and the inward clamp at the ruler's ends.
        const float major_interval = sequencer_ui::rulerMajorInterval(visible_duration, timeline_width, 2.0f * label_margin);
        const float minor_interval = major_interval / 4.0f;

        std::string html;
        html.reserve(2048);

        const float half_label = label_margin * 0.5f;

        int first_index = static_cast<int>(std::floor(visible_start / minor_interval));
        if (first_index < 0)
            first_index = 0;

        for (int i = 0;; ++i) {
            const int index = first_index + i;
            const float t_val = static_cast<float>(index) * minor_interval;
            if (t_val > visible_end)
                break;

            const float x = timeToX(t_val, 0.0f, timeline_width);
            if (x < 0.0f || x > timeline_width)
                continue;

            const bool is_major = (index % 4) == 0;
            if (is_major) {
                html += fmt::format(
                    "<div class=\"ruler-tick major\" style=\"left: {:.1f}px;\" />", x);
                if (timeline_width > label_margin) {
                    const float label_x = std::clamp(x, half_label, timeline_width - label_margin);
                    html += fmt::format(
                        "<span class=\"ruler-label\" style=\"left: {:.1f}px;\">{}</span>",
                        label_x, formatTimeShort(t_val, major_interval));
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

    bool RmlSequencerPanel::consumeLoadSequenceRequest() {
        const bool request = load_sequence_requested_;
        load_sequence_requested_ = false;
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

    void RmlSequencerPanel::updateTransportSettings() {
        if (!elements_cached_)
            return;

        const bool has_sequence = controller_.hasPlySequence();
        const bool has_reveal = controller_.timeline().revealClip().has_value();
        if (auto* label = document_->GetElementById("reveal-button-label")) {
            const auto text = Rml::StringUtilities::EncodeRml(lfs::event::LocalizationManager::getInstance().get(
                has_reveal ? "sequencer.change_reveal" : "sequencer.add_reveal"));
            if (label->GetInnerRML() != text)
                label->SetInnerRML(text);
        }
        if ((!has_reveal || controller_.hasSelection()) && reveal_selected_)
            selectReveal(false);
        if (!has_sequence || controller_.hasSelection())
            ply_sequence_selected_ = false;

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
        if (!sequence_fps_editing_)
            syncSequenceFpsDisplay();
        if (el_sequence_fps_field_)
            el_sequence_fps_field_->SetClass("hidden", !has_sequence);

        if (auto* frame = document_->GetElementById("btn-camera-aspect")) {
            frame->SetClass("disabled", ui_state_.equirectangular);
            if (auto* label = document_->GetElementById("camera-aspect-label"))
                label->SetInnerRML(ui_state_.equirectangular ? "2:1" : std::format("{}:{}", ui_state_.camera_width, ui_state_.camera_height));
        }
        if (el_btn_load_sequence_)
            el_btn_load_sequence_->SetClass("active", has_sequence);
        if (el_panel_) {
            el_panel_->SetClass("is-floating", floating_);
            el_panel_->SetClass("film-strip-attached", film_strip_attached_);
        }
        if (el_floating_header_)
            el_floating_header_->SetClass("hidden", !floating_);
        if (el_film_strip_panel_)
            el_film_strip_panel_->SetProperty("display", film_strip_attached_ ? "block" : "none");
        if (el_btn_dock_toggle_) {
            el_btn_dock_toggle_->SetAttribute("data-tooltip",
                                              floating_ ? "tooltip.seq_dock" : "tooltip.seq_undock");
            el_btn_dock_toggle_->SetClass("active", false);
            el_btn_dock_toggle_->SetClass("hidden", false);
        }
        if (el_btn_close_panel_) {
            el_btn_close_panel_->SetAttribute("data-tooltip", "common.close");
            el_btn_close_panel_->SetClass("hidden", !floating_);
        }
        if (el_close_panel_label_)
            el_close_panel_label_->SetInnerRML(lfs::event::LocalizationManager::getInstance().get("common.close"));
    }

    float RmlSequencerPanel::timelineWidth() const {
        if (el_header_ && el_header_->GetClientWidth() > 0.0f)
            return el_header_->GetClientWidth();
        return cached_panel_width_ - 2.0f * INNER_PADDING_H * cached_dp_ratio_;
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

        const std::size_t theme_signature = gui::rml_theme::currentThemeSignature();
        auto language = lfs::event::LocalizationManager::getInstance().getCurrentLanguage();
        const float context_x = panel_x - input.screen_x;
        const float context_y = panel_y - input.screen_y;
        rml_manager_->trackContextFrame(rml_context_,
                                        static_cast<int>(context_x),
                                        static_cast<int>(context_y));

        const RenderSignature signature =
            makeRenderSignature(w, h, theme_signature, language);
        if (canReuseCachedRender(signature, input, w, h)) {
            queueCachedRender(context_x, context_y, panel_width, cached_total_height_, w, h, false);
            return;
        }

        syncTheme();

        if (language != last_language_) {
            last_language_ = std::move(language);
            last_keyframe_count_ = static_cast<size_t>(-1);
        }

        if (elements_cached_) {
            el_timeline_->SetProperty("width", "100%");
        }

        updateTransportSettings();
        rml_context_->SetDimensions(Rml::Vector2i(w, h));
        rml_context_->Update();
        forwardInput(input);

        const auto timeline_offset = el_timeline_->GetAbsoluteOffset(Rml::BoxArea::Border);
        const float content_height = std::max(0.0f, el_timeline_->GetClientHeight());
        const float tl_width = timelineWidth();

        const Vec2 timeline_pos = {panel_x + timeline_offset.x,
                                   panel_y + timeline_offset.y};

        if (elements_cached_) {
            handleTimelineInteraction(timeline_pos, tl_width, content_height, input);
            handleEasingStripeInteraction(timeline_pos.x, tl_width, input);
            const float strip_y = panel_y + el_film_strip_panel_->GetAbsoluteOffset(Rml::BoxArea::Border).y;
            rebuildFilmStrip(timeline_pos.x, tl_width, strip_y, input, rm, sm, film_strip);

            cached_playhead_screen_x_ = timeline_pos.x + clampCenteredSpan(
                                                             timeToX(controller_.playhead(), 0.0f, tl_width),
                                                             tl_width,
                                                             PLAYHEAD_HANDLE_WIDTH * dp);
            playhead_in_range_ = cached_playhead_screen_x_ >= timeline_pos.x &&
                                 cached_playhead_screen_x_ <= timeline_pos.x + tl_width;

            updateButtonStates();
            updateTransportSettings();
            updatePlayhead();
            updateTimelineScrollbar();
            updateTimeDisplay();
            rebuildKeyframes();
            rebuildPlySequenceClip();
            rebuildRuler();
            rebuildEasingStripe(timeline_pos.x, tl_width);
            updateTimelineGuides(timeline_pos.x, tl_width, film_strip);
        }

        if (!rml_manager_ || !rml_manager_->getVulkanRenderInterface())
            return;

        if (document_) {
            Rml::Element* body = document_->GetElementById("body");
            if (!body)
                body = document_;
            const int local_mx = static_cast<int>(input.mouse_x - cached_panel_x_);
            const int local_my = static_cast<int>(input.mouse_y - cached_panel_y_);
            tooltip_.apply(body, local_mx, local_my, w, h);
        }

        rml_context_->SetDimensions(Rml::Vector2i(w, h));
        rml_context_->Update();
        queueCachedRender(context_x, context_y, panel_width, cached_total_height_, w, h, true);
        last_render_signature_ = signature;
        last_render_mouse_x_ = input.mouse_x;
        last_render_mouse_y_ = input.mouse_y;
        last_render_mouse_over_panel_ = mouseOverPanel(input, cached_panel_x_, cached_panel_y_,
                                                       cached_panel_width_, cached_total_height_);
        direct_cache_dirty_ = false;
    }

    void RmlSequencerPanel::selectReveal(const bool selected) {
        reveal_selected_ = selected && controller_.timeline().revealClip().has_value();
        if (reveal_selected_)
            ply_sequence_selected_ = false;
        if (!document_)
            return;
        if (auto* settings = document_->GetElementById("reveal-settings"))
            settings->SetClass("hidden", !reveal_selected_);
        if (!reveal_selected_)
            return;
        controller_.deselectKeyframe();
        selected_keyframes_.clear();
        const auto& clip = *controller_.timeline().revealClip();
        for (const auto& [id, value] : {std::pair{"reveal-start", clip.start},
                                        std::pair{"reveal-end", clip.end},
                                        std::pair{"reveal-softness", clip.softness}}) {
            if (auto* input = document_->GetElementById(id))
                input->SetAttribute("value", fmt::format("{:.2f}", value));
        }
    }

    void RmlSequencerPanel::RevealEditListener::ProcessEvent(Rml::Event& event) {
        if (!panel || !panel->reveal_selected_ || !panel->controller_.timeline().revealClip())
            return;
        if (event.GetId() == Rml::EventId::Change && !event.GetParameter<bool>("linebreak", false))
            return;
        auto* el = event.GetCurrentElement();
        if (!el)
            return;
        const auto text = el->GetAttribute<Rml::String>("value", "");
        char* end = nullptr;
        const float value = std::strtof(text.c_str(), &end);
        auto clip = *panel->controller_.timeline().revealClip();
        if (end != text.c_str() && std::isfinite(value)) {
            if (el->GetId() == "reveal-start")
                clip.start = value;
            else if (el->GetId() == "reveal-end")
                clip.end = value;
            else if (el->GetId() == "reveal-softness")
                clip.softness = value;
            try {
                panel->controller_.setRevealClip(clip);
            } catch (const std::invalid_argument&) {}
        }
        panel->selectReveal(true);
    }

    // ── Clip Duration Field ─────────────────────────────────

    void RmlSequencerPanel::DurationEditListener::ProcessEvent(Rml::Event& event) {
        assert(panel);
        const auto event_id = event.GetId();
        auto* el = event.GetCurrentElement();
        if (!el)
            return;

        if (event_id == Rml::EventId::Click && el->GetId() == "duration-field") {
            if (event.GetParameter<int>("button", 0) != 0)
                return;
            panel->enterDurationEdit();
            event.StopPropagation();
        } else if (event_id == Rml::EventId::Change && el->GetId() == "duration-input") {
            if (event.GetParameter<bool>("linebreak", false))
                panel->exitDurationEdit(true);
        } else if (event_id == Rml::EventId::Blur && el->GetId() == "duration-input") {
            panel->exitDurationEdit(true);
        }
    }

    void RmlSequencerPanel::syncDurationDisplay() {
        if (!el_duration_ || duration_editing_)
            return;
        el_duration_->SetInnerRML(" / " + formatTime(controller_.clipDuration()));
    }

    void RmlSequencerPanel::enterDurationEdit() {
        if (!el_duration_field_ || !el_duration_input_ || duration_editing_)
            return;

        duration_editing_ = true;
        el_duration_field_->SetClass("is-editing", true);
        el_duration_input_->SetAttribute("value", fmt::format("{:.2f}", controller_.clipDuration()));
        el_duration_input_->Focus();
    }

    void RmlSequencerPanel::exitDurationEdit(const bool commit) {
        if (!duration_editing_)
            return;

        if (commit && el_duration_input_) {
            const auto text = el_duration_input_->GetAttribute<Rml::String>("value", "");
            char* end = nullptr;
            const float parsed = std::strtof(text.c_str(), &end);
            if (end != text.c_str())
                controller_.editClipDuration(parsed);
        }

        duration_editing_ = false;
        if (el_duration_field_)
            el_duration_field_->SetClass("is-editing", false);
        syncDurationDisplay();
    }

    // ── PLY Sequence FPS Field ─────────────────────────────

    void RmlSequencerPanel::SequenceFpsEditListener::ProcessEvent(Rml::Event& event) {
        assert(panel);
        const auto event_id = event.GetId();
        auto* el = event.GetCurrentElement();
        if (!el)
            return;

        if (event_id == Rml::EventId::Click && el->GetId() == "sequence-fps-field") {
            if (event.GetParameter<int>("button", 0) != 0)
                return;
            panel->enterSequenceFpsEdit();
            event.StopPropagation();
        } else if (event_id == Rml::EventId::Change && el->GetId() == "sequence-fps-input") {
            if (event.GetParameter<bool>("linebreak", false))
                panel->exitSequenceFpsEdit(true);
        } else if (event_id == Rml::EventId::Blur && el->GetId() == "sequence-fps-input") {
            panel->exitSequenceFpsEdit(true);
        }
    }

    void RmlSequencerPanel::syncSequenceFpsDisplay() {
        if (!el_sequence_fps_display_ || sequence_fps_editing_)
            return;
        el_sequence_fps_display_->SetInnerRML(formatSequenceFps(ui_state_.sequence_fps));
    }

    void RmlSequencerPanel::enterSequenceFpsEdit() {
        if (!el_sequence_fps_field_ || !el_sequence_fps_input_ || sequence_fps_editing_)
            return;

        sequence_fps_editing_ = true;
        el_sequence_fps_field_->SetClass("is-editing", true);
        el_sequence_fps_input_->SetAttribute("value", fmt::format("{:.2f}", ui_state_.sequence_fps));
        el_sequence_fps_input_->Focus();
    }

    void RmlSequencerPanel::exitSequenceFpsEdit(const bool commit) {
        if (!sequence_fps_editing_)
            return;

        if (commit && el_sequence_fps_input_) {
            const auto text = el_sequence_fps_input_->GetAttribute<Rml::String>("value", "");
            char* end = nullptr;
            const float parsed = std::strtof(text.c_str(), &end);
            if (end != text.c_str()) {
                ui_state_.sequence_fps = std::clamp(parsed, MIN_SEQUENCE_FPS, MAX_SEQUENCE_FPS);
                controller_.setPlySequenceFps(ui_state_.sequence_fps);
            }
        }

        sequence_fps_editing_ = false;
        if (el_sequence_fps_field_)
            el_sequence_fps_field_->SetClass("is-editing", false);
        syncSequenceFpsDisplay();
    }

} // namespace lfs::vis

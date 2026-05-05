/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Split from rml_sequencer_panel_timeline.cpp — MSVC 14.42 ICE workaround.
// Designated initializers inside a lambda returning std::optional<struct>
// trigger fatal error C1001 (msc1.cpp line 1599). Replaced with member assignment.

#include "core/events.hpp"
#include "gui/film_strip_renderer.hpp"
#include "gui/rmlui/rml_input_utils.hpp"
#include "gui/rmlui/rml_panel_host.hpp"
#include "gui/rmlui/rml_tooltip.hpp"
#include "gui/rmlui/rmlui_manager.hpp"
#include "gui/rmlui/sdl_rml_key_mapping.hpp"
#include "sequencer/interpolation.hpp"
#include "sequencer/rml_sequencer_panel.hpp"
#include "sequencer/timeline_view_math.hpp"

#include <RmlUi/Core.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fmt/format.h>

namespace lfs::vis {

    namespace {
        constexpr float MIN_KEYFRAME_SPACING = 0.1f;
        constexpr float DOUBLE_CLICK_TIME = 0.3f;
        constexpr float DRAG_THRESHOLD_PX = 3.0f;
        constexpr float PLAYHEAD_HIT_RADIUS = 6.0f;
        constexpr float PLAYHEAD_HANDLE_WIDTH = 8.0f;

        [[nodiscard]] std::string formatTime(const float seconds) {
            const int mins = static_cast<int>(seconds) / 60;
            const float secs = seconds - static_cast<float>(mins * 60);
            return fmt::format("{}:{:05.2f}", mins, secs);
        }

        [[nodiscard]] bool hasSelectedKeyframe(const std::vector<sequencer::KeyframeId>& selected_keyframes,
                                               const sequencer::KeyframeId id) {
            return std::find(selected_keyframes.begin(), selected_keyframes.end(), id) !=
                   selected_keyframes.end();
        }

        void addSelectedKeyframe(std::vector<sequencer::KeyframeId>& selected_keyframes,
                                 const sequencer::KeyframeId id) {
            if (!hasSelectedKeyframe(selected_keyframes, id))
                selected_keyframes.push_back(id);
        }

        void removeSelectedKeyframe(std::vector<sequencer::KeyframeId>& selected_keyframes,
                                    const sequencer::KeyframeId id) {
            if (const auto it = std::find(selected_keyframes.begin(), selected_keyframes.end(), id);
                it != selected_keyframes.end()) {
                selected_keyframes.erase(it);
            }
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

        struct ElementBounds {
            float x = 0.0f;
            float y = 0.0f;
            float width = 0.0f;
            float height = 0.0f;
        };

        bool measureElement(Rml::Element* const el,
                            const Rml::Vector2f& document_offset,
                            ElementBounds& out) {
            if (!el)
                return false;
            const auto offset = el->GetAbsoluteOffset(Rml::BoxArea::Border);
            const auto size = el->GetBox().GetSize(Rml::BoxArea::Border);
            out.x = offset.x - document_offset.x;
            out.y = offset.y - document_offset.y;
            out.width = size.x;
            out.height = size.y;
            return true;
        }
    } // namespace

    using namespace panel_config;

    void RmlSequencerPanel::updateTimelineGuides(const float timeline_x, const float timeline_width,
                                                 const gui::FilmStripRenderer& film_strip) {
        if (!elements_cached_ || timeline_width <= 0.0f)
            return;

        const auto document_offset = document_->GetAbsoluteOffset(Rml::BoxArea::Border);
        ElementBounds timeline_bounds;
        ElementBounds easing_bounds;
        if (!measureElement(el_timeline_, document_offset, timeline_bounds))
            return;
        if (!measureElement(el_easing_stripe_, document_offset, easing_bounds))
            return;

        float guide_left = timeline_bounds.x;
        float guide_top = timeline_bounds.y;
        float guide_width = timeline_bounds.width;
        float guide_bottom = std::max(timeline_bounds.y + timeline_bounds.height,
                                      easing_bounds.y + easing_bounds.height);

        if (film_strip_attached_) {
            ElementBounds film_strip_bounds;
            if (measureElement(el_film_strip_panel_, document_offset, film_strip_bounds)) {
                guide_bottom = std::max(guide_bottom,
                                        film_strip_bounds.y + film_strip_bounds.height);
            }
        }

        guide_width = std::max(guide_width, 0.0f);
        el_panel_guides_->SetProperty("left", fmt::format("{:.1f}px", guide_left));
        el_panel_guides_->SetProperty("top", fmt::format("{:.1f}px", guide_top));
        el_panel_guides_->SetProperty("width", fmt::format("{:.1f}px", guide_width));
        el_panel_guides_->SetProperty("height", fmt::format("{:.1f}px", std::max(0.0f, guide_bottom - guide_top)));

        const auto set_guide = [guide_width](Rml::Element* const el,
                                             const std::optional<float> x,
                                             const float width_px = 1.0f) {
            if (!el)
                return;
            if (!x.has_value()) {
                el->SetProperty("display", "none");
                return;
            }
            const float clamped_center = clampCenteredSpan(*x, guide_width, width_px);
            el->SetProperty("display", "block");
            el->SetProperty("left", fmt::format("{:.1f}px", clamped_center - width_px * 0.5f));
            el->SetProperty("width", fmt::format("{:.1f}px", width_px));
        };

        std::optional<float> strip_hover_x;
        if (film_strip_attached_) {
            if (const auto& hover = film_strip.hoverState(); hover.has_value())
                strip_hover_x = hover->guide_x - timeline_x;
        }

        std::optional<float> hovered_x;
        if (const auto hovered_id = hoveredKeyframeId(); hovered_id.has_value()) {
            if (const auto* const keyframe = controller_.timeline().getKeyframeById(*hovered_id))
                hovered_x = timeToX(keyframe->time, 0.0f, timeline_width);
        }

        std::optional<float> selected_x;
        if (const auto selected = controller_.selectedKeyframe(); selected.has_value()) {
            if (const auto* const keyframe = controller_.timeline().getKeyframe(*selected))
                selected_x = timeToX(keyframe->time, 0.0f, timeline_width);
        }

        std::optional<float> playhead_x;
        if (playhead_in_range_)
            playhead_x = cached_playhead_screen_x_ - timeline_x;

        set_guide(el_guide_strip_hover_, strip_hover_x, 1.0f);
        set_guide(el_guide_hovered_, hovered_x, 1.5f);
        set_guide(el_guide_selected_, selected_x, 2.0f);
        set_guide(el_guide_playhead_, playhead_x, PLAYHEAD_WIDTH);
    }

    void RmlSequencerPanel::updateTimelineTooltip(const gui::FilmStripRenderer& film_strip,
                                                  const PanelInputState& input) {
        if (!elements_cached_ || !el_timeline_tooltip_) {
            return;
        }

        const auto& hover = film_strip.hoverState();
        if (!film_strip_attached_ || !hover.has_value()) {
            el_timeline_tooltip_->SetProperty("display", "none");
            return;
        }

        std::string html = "<span class=\"timeline-tooltip-line title\">Time ";
        html += formatTime(hover->exact_time);
        html += "</span>";
        if (hover->over_thumbnail) {
            html += "<span class=\"timeline-tooltip-line\">Sample ";
            html += formatTime(hover->sample_time);
            html += "</span>";
            html += "<span class=\"timeline-tooltip-line\">Covers ";
            html += formatTime(hover->interval_start_time);
            html += " - ";
            html += formatTime(hover->interval_end_time);
            html += "</span>";
        }

        const float dp = cached_dp_ratio_;
        const float local_x = input.mouse_x - cached_panel_x_;
        const float local_y = input.mouse_y - cached_panel_y_;
        const float offset_x = 14.0f * dp;
        const float offset_y = 10.0f * dp;
        const bool align_right = local_x > cached_panel_width_ - 180.0f * dp;
        const bool place_below = local_y < 48.0f * dp;

        const float approx_width = 170.0f * dp;
        const float approx_height = hover->over_thumbnail ? 54.0f * dp : 32.0f * dp;
        float left = align_right ? local_x - approx_width - offset_x : local_x + offset_x;
        float top = place_below ? local_y + offset_y : local_y - approx_height - offset_y;
        left = std::clamp(left, 8.0f * dp, std::max(8.0f * dp, cached_panel_width_ - approx_width - 8.0f * dp));
        top = std::clamp(top, 8.0f * dp, std::max(8.0f * dp, cached_total_height_ - approx_height - 8.0f * dp));

        el_timeline_tooltip_->SetInnerRML(html);
        char left_buffer[32];
        char top_buffer[32];
        std::snprintf(left_buffer, sizeof(left_buffer), "%.1fpx", left);
        std::snprintf(top_buffer, sizeof(top_buffer), "%.1fpx", top);
        el_timeline_tooltip_->SetProperty("left", left_buffer);
        el_timeline_tooltip_->SetProperty("top", top_buffer);
        el_timeline_tooltip_->SetProperty("display", "block");
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

        const float total_h = cached_total_height_;
        hovered_ = local_x >= 0 && local_y >= 0 &&
                   local_x < cached_panel_width_ && local_y < total_h;

        if (!hovered_) {
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

        const float playhead_x = pos.x + clampCenteredSpan(
                                             timeToX(controller_.playhead(), 0.0f, width),
                                             width,
                                             PLAYHEAD_HANDLE_WIDTH * s);
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
                    time_edit_buffer_ = fmt::format("{:.2f}", keyframes[i].time);
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
                                addSelectedKeyframe(selected_keyframes_, keyframes[j].id);
                        }
                    } else if (input.key_ctrl) {
                        const auto id = keyframes[i].id;
                        if (hasSelectedKeyframe(selected_keyframes_, id))
                            removeSelectedKeyframe(selected_keyframes_, id);
                        else
                            addSelectedKeyframe(selected_keyframes_, id);
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

            for (const auto& keyframe : keyframes) {
                if (!keyframe.is_loop_point) {
                    std::erase(to_delete, keyframe.id);
                    break;
                }
            }

            bool removed_any = false;
            for (const auto id : to_delete)
                removed_any |= controller_.removeKeyframeById(id);
            for (const auto id : to_delete)
                removeSelectedKeyframe(selected_keyframes_, id);
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

        // Context menu display is handled by the sequencer RML overlay.
    }

    void RmlSequencerPanel::handleEasingStripeInteraction(const float timeline_x, const float timeline_width,
                                                          const PanelInputState& input) {
        const float dp = cached_dp_ratio_;
        const float stripe_y = cached_panel_y_ + cached_height_;
        const float stripe_h = EASING_STRIPE_HEIGHT * dp;
        const float mx = input.mouse_x;
        const float my = input.mouse_y;

        if (timeline_width <= 0.0f || controller_.timeline().keyframes().empty())
            return;

        if (mx < timeline_x || mx > timeline_x + timeline_width ||
            my < stripe_y || my > stripe_y + stripe_h) {
            return;
        }

        if (input.mouse_clicked[1]) {
            std::optional<size_t> nearest;
            float best_dist = KEYFRAME_RADIUS * 3.0f * dp;
            for (size_t i = 0; i < controller_.timeline().keyframes().size(); ++i) {
                const float dist = std::abs(mx - timeToX(controller_.timeline().keyframes()[i].time,
                                                         timeline_x, timeline_width));
                if (dist < best_dist) {
                    best_dist = dist;
                    nearest = i;
                }
            }

            context_menu_time_ = nearest.has_value()
                                     ? controller_.timeline().keyframes()[*nearest].time
                                     : controller_.playhead();
            context_menu_keyframe_ = nearest;
            context_menu_open_ = true;
        }

        if (input.mouse_clicked[0]) {
            std::optional<size_t> nearest;
            float best_dist = KEYFRAME_RADIUS * 2.0f * dp;
            for (size_t i = 0; i < controller_.timeline().keyframes().size(); ++i) {
                const float dist = std::abs(mx - timeToX(controller_.timeline().keyframes()[i].time,
                                                         timeline_x, timeline_width));
                if (dist < best_dist) {
                    best_dist = dist;
                    nearest = i;
                }
            }
            if (nearest.has_value())
                lfs::core::events::cmd::SequencerSelectKeyframe{.keyframe_index = *nearest}.emit();
        }
    }

    void RmlSequencerPanel::handleFilmStripInteraction(const float timeline_x, const float timeline_width,
                                                       const PanelInputState& input,
                                                       gui::FilmStripRenderer& film_strip) {
        if (!film_strip_attached_) {
            if (film_strip_scrubbing_) {
                film_strip_scrubbing_ = false;
                controller_.endScrub();
            }
            return;
        }

        const bool can_scrub = controller_.timeline().size() >= 2;
        const float scrub_time = can_scrub
                                     ? std::clamp(
                                           sequencer_ui::screenXToTime(input.mouse_x, timeline_x, timeline_width,
                                                                       getDisplayEndTime(), pan_offset_),
                                           controller_.timeline().startTime(), controller_.timeline().endTime())
                                     : 0.0f;

        if (film_strip_scrubbing_) {
            if (input.mouse_down[0] && can_scrub) {
                controller_.scrub(scrub_time);
            } else {
                film_strip_scrubbing_ = false;
                controller_.endScrub();
            }
        }

        if (const auto& hover = film_strip.hoverState(); hover.has_value()) {
            if (!film_strip_scrubbing_ && can_scrub && !input.want_capture_mouse && input.mouse_clicked[0]) {
                film_strip_scrubbing_ = true;
                controller_.beginScrub();
                controller_.scrub(scrub_time);
            }
        }
    }

    void RmlSequencerPanel::openFocalLengthEdit(const size_t index, const float current_focal_mm) {
        editing_focal_length_ = true;
        editing_focal_index_ = index;
        focal_edit_buffer_ = fmt::format("{:.1f}", current_focal_mm);
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

} // namespace lfs::vis

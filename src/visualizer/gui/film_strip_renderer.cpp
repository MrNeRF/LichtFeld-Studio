/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/film_strip_renderer.hpp"
#include "rendering/rendering_manager.hpp"
#include "scene/scene_manager.hpp"
#include "sequencer/sequencer_controller.hpp"
#include "sequencer/timeline_view_math.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>

namespace lfs::vis::gui {

    int FilmStripRenderer::findSlot(const float time, const float tolerance,
                                    const std::array<bool, MAX_SLOTS>& claimed_slots) const {
        int stale_match = -1;
        for (int i = 0; i < MAX_SLOTS; ++i) {
            if (claimed_slots[i])
                continue;
            if (slots_[i].valid && std::abs(slots_[i].time - time) < tolerance) {
                if (slots_[i].generation == generation_)
                    return i;
                if (stale_match < 0)
                    stale_match = i;
            }
        }
        return stale_match;
    }

    int FilmStripRenderer::allocateSlot(const std::array<bool, MAX_SLOTS>& claimed_slots) {
        for (int i = 0; i < MAX_SLOTS; ++i) {
            if (!claimed_slots[i] && !slots_[i].valid)
                return i;
        }

        int lru = 0;
        uint32_t oldest = std::numeric_limits<uint32_t>::max();
        for (int i = 0; i < MAX_SLOTS; ++i) {
            if (claimed_slots[i])
                continue;
            if (slots_[i].frame_used < oldest) {
                oldest = slots_[i].frame_used;
                lru = i;
            }
        }
        slots_[lru].valid = false;
        return lru;
    }

    bool FilmStripRenderer::renderThumbnail(const int slot_idx, const float time,
                                            const SequencerController& controller,
                                            RenderingManager* rm, SceneManager* sm) {
        assert(slot_idx >= 0 && slot_idx < MAX_SLOTS);
        const auto& timeline = controller.timeline();

        if (timeline.size() < 2)
            return false;

        const auto state = timeline.evaluate(time);
        const glm::mat3 cam_rot = glm::mat3_cast(state.rotation);

        const auto image = rm->renderPreviewImage(sm, cam_rot, state.position, state.focal_length_mm,
                                                  THUMB_WIDTH, THUMB_HEIGHT);
        const bool ok = image && slots_[slot_idx].texture.upload(*image, THUMB_WIDTH, THUMB_HEIGHT);

        if (ok) {
            slots_[slot_idx].time = time;
            slots_[slot_idx].frame_used = frame_counter_;
            slots_[slot_idx].valid = true;
        }

        return ok;
    }

    void FilmStripRenderer::render(const SequencerController& controller,
                                   RenderingManager* rm, SceneManager* sm,
                                   const RenderOptions& options) {
        const float timeline_x = options.timeline_x;
        const float timeline_width = options.timeline_width;
        const float strip_y = options.strip_y;
        const float mouse_x = options.mouse_x;
        const float mouse_y = options.mouse_y;
        const float zoom_level = options.zoom_level;
        const float pan_offset = options.pan_offset;
        const float display_end_time = options.display_end_time;

        hover_state_.reset();
        if (timeline_width <= 0.0f)
            return;

        const auto& timeline = controller.timeline();
        const bool has_animation = timeline.size() >= 2;

        const float thumb_display_h = STRIP_HEIGHT - THUMB_PADDING * 2.0f;
        const float base_thumb_w = thumb_display_h * (static_cast<float>(THUMB_WIDTH) / static_cast<float>(THUMB_HEIGHT));
        const int num_thumbs = sequencer_ui::thumbnailCount(timeline_width, base_thumb_w, zoom_level);
        const float groove_min_y = strip_y + THUMB_PADDING;
        const float groove_max_y = strip_y + STRIP_HEIGHT - THUMB_PADDING;

        const bool mouse_in_strip = mouse_x >= timeline_x && mouse_x <= timeline_x + timeline_width &&
                                    mouse_y >= strip_y && mouse_y <= strip_y + STRIP_HEIGHT;
        if (mouse_in_strip) {
            HoverState hover;
            hover.exact_time = std::clamp(
                sequencer_ui::screenXToTime(mouse_x, timeline_x, timeline_width, display_end_time, pan_offset),
                pan_offset, pan_offset + display_end_time);
            hover.sample_time = hover.exact_time;
            hover.interval_start_time = hover.exact_time;
            hover.interval_end_time = hover.exact_time;
            hover.guide_x = std::clamp(mouse_x, timeline_x, timeline_x + timeline_width);
            hover.thumb_min_x = hover.guide_x;
            hover.thumb_max_x = hover.guide_x;
            hover_state_ = hover;
        }

        thumbs_.clear();
        exact_markers_.clear();
        render_requests_.clear();

        float anim_start = 0.0f;
        float anim_end = 0.0f;

        if (has_animation && rm && sm && num_thumbs > 0) {
            ++frame_counter_;
                visible_slot_assignments_.resize(static_cast<size_t>(num_thumbs), -1);
                std::array<bool, MAX_SLOTS> claimed_slots{};

                anim_start = timeline.startTime();
                anim_end = timeline.endTime();
                const float visible_center_time = sequencer_ui::screenXToTime(
                    timeline_x + timeline_width * 0.5f, timeline_x, timeline_width, display_end_time, pan_offset);
                const float playhead_time = controller.playhead();
                const float time_per_thumb = display_end_time / static_cast<float>(num_thumbs);

                if (hover_state_.has_value()) {
                    if (!last_hover_focus_time_.has_value() ||
                        std::abs(*last_hover_focus_time_ - hover_state_->exact_time) > time_per_thumb * 0.5f) {
                        burst_remaining_ = BURST_FRAMES;
                        last_hover_focus_time_ = hover_state_->exact_time;
                    }
                } else {
                    last_hover_focus_time_.reset();
                }

                thumbs_.reserve(num_thumbs);
                render_requests_.reserve(static_cast<size_t>(num_thumbs));

                for (int i = 0; i < num_thumbs; ++i) {
                    const auto slot = sequencer_ui::thumbnailSlotAt(
                        i, num_thumbs, timeline_x, timeline_width, display_end_time, pan_offset);
                    if (slot.interval_end_time < anim_start || slot.interval_start_time > anim_end)
                        continue;

                    const float clamped_sample_time = sequencer_ui::resolvedThumbnailSampleTime(
                        slot.sample_time, slot.interval_start_time, slot.interval_end_time, anim_start, anim_end);
                    const float half_interval = std::max((slot.interval_end_time - slot.interval_start_time) * 0.5f, 0.001f);
                    int assigned_slot = -1;
                    const int preferred_slot = visible_slot_assignments_[static_cast<size_t>(i)];
                    if (preferred_slot >= 0 && preferred_slot < MAX_SLOTS &&
                        !claimed_slots[preferred_slot]) {
                        assigned_slot = preferred_slot;
                    } else {
                        assigned_slot = findSlot(clamped_sample_time, half_interval, claimed_slots);
                    }

                    if (assigned_slot >= 0) {
                        claimed_slots[assigned_slot] = true;
                        visible_slot_assignments_[static_cast<size_t>(i)] = assigned_slot;
                    }

                    ThumbInfo thumb;
                    thumb.time = clamped_sample_time;
                    thumb.interval_start_time = slot.interval_start_time;
                    thumb.interval_end_time = slot.interval_end_time;
                    thumb.screen_x = slot.screen_x;
                    thumb.screen_width = slot.screen_width;
                    thumb.screen_center_x = slot.screen_center_x;
                    thumb.slot_idx = assigned_slot;
                    thumb.contains_selected = options.selected_keyframe_time.has_value() &&
                                              *options.selected_keyframe_time >= slot.interval_start_time &&
                                              *options.selected_keyframe_time <= slot.interval_end_time;
                    thumb.contains_hovered_keyframe = options.hovered_keyframe_time.has_value() &&
                                                      *options.hovered_keyframe_time >= slot.interval_start_time &&
                                                      *options.hovered_keyframe_time <= slot.interval_end_time;

                    float priority = std::abs(clamped_sample_time - visible_center_time);
                    priority = std::min(priority, std::abs(clamped_sample_time - playhead_time) * 0.85f);
                    if (hover_state_.has_value())
                        priority = std::min(priority, std::abs(clamped_sample_time - hover_state_->exact_time) * 0.35f);
                    if (options.hovered_keyframe_time.has_value())
                        priority = std::min(priority, std::abs(clamped_sample_time - *options.hovered_keyframe_time) * 0.45f);
                    if (options.selected_keyframe_time.has_value())
                        priority = std::min(priority, std::abs(clamped_sample_time - *options.selected_keyframe_time) * 0.60f);
                    thumb.priority = priority;

                    const bool mouse_in_thumb = hover_state_.has_value() &&
                                                mouse_y >= groove_min_y && mouse_y <= groove_max_y &&
                                                mouse_x >= slot.screen_x && mouse_x <= slot.screen_x + slot.screen_width;
                    if (mouse_in_thumb) {
                        thumb.hovered = true;
                        hover_state_->over_thumbnail = true;
                        hover_state_->sample_time = clamped_sample_time;
                        hover_state_->interval_start_time = slot.interval_start_time;
                        hover_state_->interval_end_time = slot.interval_end_time;
                        hover_state_->thumb_min_x = slot.screen_x;
                        hover_state_->thumb_max_x = slot.screen_x + slot.screen_width;
                    }

                    const bool slot_matches_time = thumb.slot_idx >= 0 &&
                                                   slots_[thumb.slot_idx].valid &&
                                                   std::abs(slots_[thumb.slot_idx].time - thumb.time) < half_interval;
                    thumb.stale = thumb.slot_idx < 0 || !slot_matches_time ||
                                  slots_[thumb.slot_idx].generation != generation_;

                    thumbs_.push_back(thumb);
                    if (thumb.slot_idx >= 0)
                        slots_[thumb.slot_idx].frame_used = frame_counter_;

                    if (thumb.stale) {
                        render_requests_.push_back({
                            .index = thumbs_.size() - 1,
                            .visible_index = i,
                            .time = thumb.time,
                            .tolerance = half_interval,
                            .priority = thumb.priority,
                            .preferred_slot = thumb.slot_idx,
                        });
                    }
                }

                std::sort(render_requests_.begin(), render_requests_.end(), [](const RenderRequest& lhs, const RenderRequest& rhs) {
                    return lhs.priority < rhs.priority;
                });

                const bool has_visible_current_thumb = std::any_of(thumbs_.begin(), thumbs_.end(), [&](const ThumbInfo& thumb) {
                    return thumb.slot_idx >= 0 &&
                           slots_[thumb.slot_idx].valid &&
                           slots_[thumb.slot_idx].generation == generation_;
                });
                const int max_renders = !has_visible_current_thumb
                                            ? BURST_RENDERS_PER_FRAME
                                        : burst_remaining_ > 0
                                            ? BURST_RENDERS_PER_FRAME
                                            : MAX_RENDERS_PER_FRAME;

                const auto assign_request_slot = [&](const RenderRequest& request, const int slot) {
                    thumbs_[request.index].slot_idx = slot;
                    slots_[slot].frame_used = frame_counter_;
                };

                int renders = 0;
                for (const auto& request : render_requests_) {
                    if (renders >= max_renders)
                        break;

                    int slot = request.preferred_slot;
                    if (slot < 0 || slot >= MAX_SLOTS) {
                        slot = findSlot(request.time, request.tolerance, claimed_slots);
                    }
                    if (slot < 0) {
                        slot = allocateSlot(claimed_slots);
                        if (request.visible_index >= 0 &&
                            static_cast<size_t>(request.visible_index) < visible_slot_assignments_.size()) {
                            visible_slot_assignments_[static_cast<size_t>(request.visible_index)] = slot;
                        }
                    }
                    claimed_slots[slot] = true;

                    if (renderThumbnail(slot, request.time, controller, rm, sm)) {
                        slots_[slot].generation = generation_;
                        assign_request_slot(request, slot);
                        thumbs_[request.index].stale = false;
                        ++renders;
                    }
                }
                if (burst_remaining_ > 0)
                    --burst_remaining_;
        } else {
            last_hover_focus_time_.reset();
            visible_slot_assignments_.clear();
        }

        if (has_animation) {
            for (const auto& keyframe : timeline.keyframes()) {
                if (keyframe.is_loop_point)
                    continue;

                const float screen_x = sequencer_ui::timeToScreenX(
                    keyframe.time, timeline_x, timeline_width, display_end_time, pan_offset);
                if (screen_x < timeline_x - 1.0f || screen_x > timeline_x + timeline_width + 1.0f)
                    continue;

                const bool selected = options.selected_keyframe_id.has_value() &&
                                      *options.selected_keyframe_id == keyframe.id;
                const bool hovered = options.hovered_keyframe_id.has_value() &&
                                     *options.hovered_keyframe_id == keyframe.id;

                exact_markers_.push_back({
                    .time = keyframe.time,
                    .screen_x = screen_x,
                    .selected = selected,
                    .hovered = hovered,
                });
            }
        }
    }

    std::uintptr_t FilmStripRenderer::textureIdForSlot(const int slot_idx) const {
        if (slot_idx < 0 || slot_idx >= MAX_SLOTS)
            return 0;
        const auto& slot = slots_[slot_idx];
        return slot.valid ? slot.texture.textureId() : 0;
    }

    bool FilmStripRenderer::slotIsCurrentGeneration(const int slot_idx) const {
        if (slot_idx < 0 || slot_idx >= MAX_SLOTS)
            return false;
        const auto& slot = slots_[slot_idx];
        return slot.valid && slot.generation == generation_;
    }

    void FilmStripRenderer::invalidateAll() {
        ++generation_;
        burst_remaining_ = BURST_FRAMES;
    }

    void FilmStripRenderer::destroyGraphicsResources() {
        for (auto& slot : slots_) {
            slot.texture.reset();
            slot.valid = false;
            slot.generation = 0;
        }
        generation_ = 0;
        burst_remaining_ = 0;
        hover_state_.reset();
        last_hover_focus_time_.reset();
        visible_slot_assignments_.clear();
    }

} // namespace lfs::vis::gui

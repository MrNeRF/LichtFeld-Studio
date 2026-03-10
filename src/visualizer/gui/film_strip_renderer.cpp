/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// clang-format off
#include <glad/glad.h>
// clang-format on

#include "gui/film_strip_renderer.hpp"
#include "core/logger.hpp"
#include "rendering/rendering_manager.hpp"
#include "scene/scene_manager.hpp"
#include "sequencer/sequencer_controller.hpp"
#include "sequencer/timeline_view_math.hpp"
#include "theme/theme.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <imgui.h>

namespace lfs::vis::gui {

    void FilmStripRenderer::initGL() {
        if (gl_initialized_ || gl_init_failed_)
            return;

        glGenFramebuffers(1, fbo_.ptr());
        glGenRenderbuffers(1, depth_rbo_.ptr());

        glBindRenderbuffer(GL_RENDERBUFFER, depth_rbo_.get());
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, THUMB_WIDTH, THUMB_HEIGHT);
        glBindRenderbuffer(GL_RENDERBUFFER, 0);

        glBindFramebuffer(GL_FRAMEBUFFER, fbo_.get());
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth_rbo_.get());
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        for (auto& slot : slots_) {
            glGenTextures(1, slot.texture.ptr());
            glBindTexture(GL_TEXTURE_2D, slot.texture.get());
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, THUMB_WIDTH, THUMB_HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        }
        glBindTexture(GL_TEXTURE_2D, 0);

        glBindFramebuffer(GL_FRAMEBUFFER, fbo_.get());
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, slots_[0].texture.get(), 0);
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            LOG_ERROR("Film strip FBO incomplete");
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            gl_init_failed_ = true;
            return;
        }
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        gl_initialized_ = true;
    }

    int FilmStripRenderer::findSlot(const float time, const float tolerance) const {
        int stale_match = -1;
        for (int i = 0; i < MAX_SLOTS; ++i) {
            if (slots_[i].valid && std::abs(slots_[i].time - time) < tolerance) {
                if (slots_[i].generation == generation_)
                    return i;
                if (stale_match < 0)
                    stale_match = i;
            }
        }
        return stale_match;
    }

    int FilmStripRenderer::allocateSlot() {
        for (int i = 0; i < MAX_SLOTS; ++i) {
            if (!slots_[i].valid)
                return i;
        }

        int lru = 0;
        uint32_t oldest = slots_[0].frame_used;
        for (int i = 1; i < MAX_SLOTS; ++i) {
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

        glBindFramebuffer(GL_FRAMEBUFFER, fbo_.get());
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                               slots_[slot_idx].texture.get(), 0);

        const bool ok = rm->renderPreviewFrame(sm, cam_rot, state.position, state.focal_length_mm,
                                               fbo_, slots_[slot_idx].texture,
                                               THUMB_WIDTH, THUMB_HEIGHT);

        glBindFramebuffer(GL_FRAMEBUFFER, 0);

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
        const float panel_x = options.panel_x;
        const float panel_width = options.panel_width;
        const float timeline_x = options.timeline_x;
        const float timeline_width = options.timeline_width;
        const float strip_y = options.strip_y;
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
        const float actual_thumb_w = (num_thumbs > 0) ? timeline_width / static_cast<float>(num_thumbs) : 0.0f;

        const float groove_x = timeline_x - THUMB_PADDING;
        const float groove_w = timeline_width + THUMB_PADDING * 2.0f;
        const ImVec2 groove_min(groove_x, strip_y + THUMB_PADDING);
        const ImVec2 groove_max(groove_x + groove_w, strip_y + STRIP_HEIGHT - THUMB_PADDING);

        const auto& io = ImGui::GetIO();
        const float mouse_x = io.MousePos.x;
        const float mouse_y = io.MousePos.y;
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
            if (!gl_initialized_)
                initGL();

            if (gl_initialized_) {
                ++frame_counter_;

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
                    const int existing = findSlot(clamped_sample_time, half_interval);

                    ThumbInfo thumb;
                    thumb.time = clamped_sample_time;
                    thumb.interval_start_time = slot.interval_start_time;
                    thumb.interval_end_time = slot.interval_end_time;
                    thumb.screen_x = slot.screen_x;
                    thumb.screen_width = slot.screen_width;
                    thumb.screen_center_x = slot.screen_center_x;
                    thumb.slot_idx = existing;
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
                                                mouse_y >= groove_min.y && mouse_y <= groove_max.y &&
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

                    thumbs_.push_back(thumb);
                    if (thumb.slot_idx >= 0)
                        slots_[thumb.slot_idx].frame_used = frame_counter_;

                    if (thumb.slot_idx < 0 || slots_[thumb.slot_idx].generation != generation_) {
                        render_requests_.push_back({
                            .index = thumbs_.size() - 1,
                            .time = thumb.time,
                            .tolerance = half_interval,
                            .priority = thumb.priority,
                        });
                    }
                }

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

                    const int reusable_slot = findSlot(request.time, request.tolerance);
                    if (reusable_slot >= 0 && slots_[reusable_slot].generation == generation_ &&
                        slots_[reusable_slot].valid) {
                        assign_request_slot(request, reusable_slot);
                        continue;
                    }

                    const int slot = allocateSlot();
                    if (renderThumbnail(slot, request.time, controller, rm, sm)) {
                        slots_[slot].generation = generation_;
                        assign_request_slot(request, slot);
                        ++renders;
                    }
                }
                if (burst_remaining_ > 0)
                    --burst_remaining_;
            }
        } else {
            last_hover_focus_time_.reset();
        }

        const auto& t = theme();
        auto* dl = ImGui::GetForegroundDrawList();
        const float rounding = t.sizes.window_rounding;

        // Panel-matching background (bottom half only, top connects to panel above)
        const ImVec2 strip_min(panel_x, strip_y);
        const ImVec2 strip_max(panel_x + panel_width, strip_y + STRIP_HEIGHT);

        const ImU32 bg_color = toU32WithAlpha(t.palette.surface, 0.95f);
        dl->AddRectFilled(strip_min, strip_max, bg_color,
                          rounding, ImDrawFlags_RoundCornersBottom);

        const ImU32 border_color = toU32WithAlpha(t.palette.border, 0.4f);
        // Left, right, bottom borders only (top connects to panel)
        dl->AddLine({panel_x, strip_y}, {panel_x, strip_max.y - rounding}, border_color);
        dl->AddLine({strip_max.x, strip_y}, {strip_max.x, strip_max.y - rounding}, border_color);
        dl->AddRect({panel_x, strip_max.y - rounding * 2.0f}, strip_max, border_color,
                    rounding, ImDrawFlags_RoundCornersBottom);

        const ImU32 groove_color = toU32WithAlpha(t.palette.background, 0.85f);
        dl->AddRectFilled(groove_min, groove_max, groove_color, 4.0f);

        // Clip thumbnails to groove
        dl->PushClipRect(groove_min, groove_max, true);

        for (const auto& thumb : thumbs_) {
            if (thumb.slot_idx < 0)
                continue;

            const auto& slot = slots_[thumb.slot_idx];
            if (!slot.valid || slot.texture.get() == 0)
                continue;

            const ImVec2 img_min(thumb.screen_x, groove_min.y);
            const ImVec2 img_max(thumb.screen_x + thumb.screen_width, groove_max.y);
            const bool current_generation = slot.generation == generation_;
            const float image_alpha = current_generation ? 1.0f : 0.58f;

            dl->AddImage(static_cast<ImTextureID>(static_cast<uintptr_t>(slot.texture.get())),
                         img_min, img_max, {0, 1}, {1, 0},
                         IM_COL32(255, 255, 255, static_cast<int>(255.0f * image_alpha)));

            if (thumb.contains_hovered_keyframe) {
                dl->AddRectFilled(img_min, img_max,
                                  toU32WithAlpha(t.palette.secondary, 0.14f), 3.0f);
            }
            if (thumb.contains_selected) {
                dl->AddRectFilled(img_min, img_max,
                                  toU32WithAlpha(t.palette.primary, 0.18f), 3.0f);
            }

            dl->AddLine({thumb.screen_center_x, groove_min.y + SPROCKET_H + 1.0f},
                        {thumb.screen_center_x, groove_max.y - SPROCKET_H - 1.0f},
                        IM_COL32(0, 0, 0, 70), 3.0f);

            const float tick_thickness = thumb.hovered || thumb.contains_selected ? 1.5f : 1.0f;
            const ImU32 tick_color = thumb.contains_selected
                                         ? toU32WithAlpha(t.palette.primary, 0.95f)
                                     : thumb.contains_hovered_keyframe
                                         ? toU32WithAlpha(t.palette.secondary, 0.85f)
                                     : thumb.hovered
                                         ? toU32WithAlpha(t.palette.text, 0.85f)
                                         : toU32WithAlpha(t.palette.text_dim, 0.35f);
            dl->AddLine({thumb.screen_center_x, groove_min.y + SPROCKET_H + 1.0f},
                        {thumb.screen_center_x, groove_max.y - SPROCKET_H - 1.0f},
                        tick_color, tick_thickness);

            if (thumb.contains_selected) {
                const ImU32 selected_color = toU32WithAlpha(t.palette.primary, 0.65f);
                dl->AddRect({thumb.screen_x, groove_min.y},
                            {thumb.screen_x + thumb.screen_width, groove_max.y},
                            selected_color, 3.0f, 0, 2.0f);
                dl->AddRectFilled({thumb.screen_x, groove_min.y},
                                  {thumb.screen_x + thumb.screen_width, groove_min.y + 3.0f},
                                  toU32WithAlpha(t.palette.primary, 0.9f));
                dl->AddRectFilled({thumb.screen_x, groove_max.y - 3.0f},
                                  {thumb.screen_x + thumb.screen_width, groove_max.y},
                                  toU32WithAlpha(t.palette.primary, 0.75f));
            }

            if (thumb.contains_hovered_keyframe) {
                const ImU32 hovered_keyframe_color = toU32WithAlpha(t.palette.secondary, 0.7f);
                dl->AddRectFilled({thumb.screen_x, groove_min.y},
                                  {thumb.screen_x + thumb.screen_width, groove_min.y + 2.5f},
                                  hovered_keyframe_color);
                dl->AddRectFilled({thumb.screen_x, groove_max.y - 2.0f},
                                  {thumb.screen_x + thumb.screen_width, groove_max.y},
                                  hovered_keyframe_color);
                dl->AddRect({thumb.screen_x + 1.0f, groove_min.y + 1.0f},
                            {thumb.screen_x + thumb.screen_width - 1.0f, groove_max.y - 1.0f},
                            toU32WithAlpha(t.palette.secondary, 0.5f), 3.0f, 0, 1.0f);
            }

            if (thumb.hovered) {
                const ImU32 hover_border = toU32WithAlpha(t.palette.text, 0.85f);
                dl->AddRect({thumb.screen_x, groove_min.y},
                            {thumb.screen_x + thumb.screen_width, groove_max.y},
                            hover_border, 3.0f, 0, 1.5f);
            }
        }

        if (has_animation) {
            const float visible_left_x = timeline_x;
            const float visible_right_x = timeline_x + timeline_width;
            const float anim_start_x = std::clamp(
                sequencer_ui::timeToScreenX(anim_start, timeline_x, timeline_width, display_end_time, pan_offset),
                visible_left_x, visible_right_x);
            const float anim_end_x = std::clamp(
                sequencer_ui::timeToScreenX(anim_end, timeline_x, timeline_width, display_end_time, pan_offset),
                visible_left_x, visible_right_x);

            const auto draw_gap_region = [&](const float x_min, const float x_max, const float alpha_scale) {
                if (x_max <= x_min)
                    return;
                dl->AddRectFilled({x_min, groove_min.y}, {x_max, groove_max.y},
                                  toU32WithAlpha(t.palette.surface, 0.30f * alpha_scale), 0.0f);
                const float stripe_span = groove_max.y - groove_min.y;
                for (float stripe_x = x_min - stripe_span; stripe_x < x_max; stripe_x += 10.0f) {
                    dl->AddLine({stripe_x, groove_max.y},
                                {stripe_x + stripe_span, groove_min.y},
                                toU32WithAlpha(t.palette.border, 0.18f * alpha_scale), 1.0f);
                }
            };

            if (anim_start_x > visible_left_x)
                draw_gap_region(visible_left_x, anim_start_x, 0.85f);
            if (anim_end_x < visible_right_x)
                draw_gap_region(anim_end_x, visible_right_x, 1.0f);

            dl->AddLine({anim_start_x, groove_min.y}, {anim_start_x, groove_max.y},
                        toU32WithAlpha(t.palette.border, 0.70f), 1.5f);
            dl->AddLine({anim_end_x, groove_min.y}, {anim_end_x, groove_max.y},
                        toU32WithAlpha(t.palette.text, 0.90f), 2.0f);
        }

        const float marker_min_y = groove_min.y + SPROCKET_H + 1.0f;
        const float marker_max_y = groove_max.y - SPROCKET_H - 1.0f;
        for (const auto& marker : exact_markers_) {
            const ImU32 marker_color = marker.selected
                                           ? toU32WithAlpha(t.palette.primary, 0.95f)
                                       : marker.hovered
                                           ? toU32WithAlpha(t.palette.secondary, 0.88f)
                                           : toU32WithAlpha(t.palette.text, 0.55f);
            const float marker_thickness = marker.selected ? 2.2f : marker.hovered ? 1.8f
                                                                                   : 1.2f;
            dl->AddLine({marker.screen_x, marker_min_y}, {marker.screen_x, marker_max_y},
                        IM_COL32(0, 0, 0, 78), marker_thickness + 1.6f);
            dl->AddLine({marker.screen_x, marker_min_y}, {marker.screen_x, marker_max_y},
                        marker_color, marker_thickness);
            dl->AddRectFilled({marker.screen_x - 2.0f, marker_min_y},
                              {marker.screen_x + 2.0f, marker_min_y + 4.0f},
                              marker_color, 1.5f);
            dl->AddRectFilled({marker.screen_x - 2.0f, marker_max_y - 4.0f},
                              {marker.screen_x + 2.0f, marker_max_y},
                              marker_color, 1.5f);
        }

        dl->PopClipRect();

        const ImU32 sprocket_color = toU32WithAlpha(t.palette.text_dim, 0.3f);

        const float sprocket_start = groove_min.x + SPROCKET_SPACING * 0.5f;
        const int sprocket_count = static_cast<int>((groove_max.x - groove_min.x) / SPROCKET_SPACING);
        for (int i = 0; i < sprocket_count; ++i) {
            const float cx = sprocket_start + static_cast<float>(i) * SPROCKET_SPACING;
            const float sx = cx - SPROCKET_W * 0.5f;
            dl->AddRectFilled({sx, groove_min.y + SPROCKET_INSET},
                              {sx + SPROCKET_W, groove_min.y + SPROCKET_INSET + SPROCKET_H},
                              sprocket_color, SPROCKET_ROUNDING);
            dl->AddRectFilled({sx, groove_max.y - SPROCKET_INSET - SPROCKET_H},
                              {sx + SPROCKET_W, groove_max.y - SPROCKET_INSET},
                              sprocket_color, SPROCKET_ROUNDING);
        }

        // Frame divider lines between thumbnail slots
        const ImU32 divider_color = toU32WithAlpha(t.palette.text_dim, 0.15f);
        for (int i = 1; i < num_thumbs; ++i) {
            const float dx = timeline_x + actual_thumb_w * static_cast<float>(i);
            dl->AddLine({dx, groove_min.y + SPROCKET_H + 1.0f},
                        {dx, groove_max.y - SPROCKET_H - 1.0f}, divider_color);
        }
    }

    void FilmStripRenderer::invalidateAll() {
        ++generation_;
        burst_remaining_ = BURST_FRAMES;
    }

    void FilmStripRenderer::destroyGLResources() {
        for (auto& slot : slots_) {
            slot.texture = {};
            slot.valid = false;
            slot.generation = 0;
        }
        fbo_ = {};
        depth_rbo_ = {};
        gl_initialized_ = false;
        gl_init_failed_ = false;
        generation_ = 0;
        burst_remaining_ = 0;
        hover_state_.reset();
        last_hover_focus_time_.reset();
    }

} // namespace lfs::vis::gui

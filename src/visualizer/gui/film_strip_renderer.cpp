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
        for (int i = 0; i < MAX_SLOTS; ++i) {
            if (slots_[i].valid && std::abs(slots_[i].time - time) < tolerance)
                return i;
        }
        return -1;
    }

    int FilmStripRenderer::allocateSlot(const uint32_t current_frame) {
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
                                   const float panel_x, const float panel_width,
                                   const float timeline_x, const float timeline_width,
                                   const float strip_y,
                                   const float zoom_level, const float pan_offset,
                                   const float display_end_time) {
        if (timeline_width <= 0.0f)
            return;

        const auto& timeline = controller.timeline();
        const bool has_animation = timeline.size() >= 2;

        // Thumbnail layout
        const float thumb_display_h = STRIP_HEIGHT - THUMB_PADDING * 2.0f;
        const float thumb_display_w = thumb_display_h * (static_cast<float>(THUMB_WIDTH) / static_cast<float>(THUMB_HEIGHT));
        const int num_thumbs = (thumb_display_w > 0.0f) ? std::max(1, static_cast<int>(timeline_width / thumb_display_w)) : 0;
        const float actual_thumb_w = (num_thumbs > 0) ? timeline_width / static_cast<float>(num_thumbs) : 0.0f;

        thumbs_.clear();
        uncached_.clear();

        if (has_animation && rm && sm && num_thumbs > 0) {
            if (!gl_initialized_)
                initGL();

            if (gl_initialized_) {
                ++frame_counter_;

                const auto xToTime = [&](float x) -> float {
                    return ((x - timeline_x) / timeline_width) * display_end_time / zoom_level + pan_offset;
                };

                const float time_per_thumb = (display_end_time / zoom_level) / static_cast<float>(num_thumbs);
                const float half_interval = time_per_thumb * 0.5f;
                const float anim_start = timeline.startTime();
                const float anim_end = timeline.endTime();
                const float visible_center_time = xToTime(timeline_x + timeline_width * 0.5f);

                thumbs_.reserve(num_thumbs);

                for (int i = 0; i < num_thumbs; ++i) {
                    const float sx = timeline_x + actual_thumb_w * static_cast<float>(i);
                    const float thumb_center_x = sx + actual_thumb_w * 0.5f;
                    const float t = xToTime(thumb_center_x);

                    if (t < anim_start - half_interval || t > anim_end + half_interval)
                        continue;

                    const float clamped_t = std::clamp(t, anim_start, anim_end);
                    const int existing = findSlot(clamped_t, half_interval);
                    thumbs_.push_back({clamped_t, sx, existing, std::abs(clamped_t - visible_center_time)});
                }

                for (auto& thumb : thumbs_) {
                    if (thumb.slot_idx >= 0)
                        slots_[thumb.slot_idx].frame_used = frame_counter_;
                }

                for (size_t i = 0; i < thumbs_.size(); ++i) {
                    if (thumbs_[i].slot_idx < 0)
                        uncached_.push_back(i);
                }

                std::sort(uncached_.begin(), uncached_.end(), [&](size_t a, size_t b) {
                    return thumbs_[a].dist_from_center < thumbs_[b].dist_from_center;
                });

                int renders = 0;
                for (const size_t idx : uncached_) {
                    if (renders >= MAX_RENDERS_PER_FRAME)
                        break;

                    const int slot = allocateSlot(frame_counter_);
                    if (renderThumbnail(slot, thumbs_[idx].time, controller, rm, sm)) {
                        thumbs_[idx].slot_idx = slot;
                        ++renders;
                    }
                }
            }
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

        // Dark inset groove for thumbnails in the timeline area
        const float groove_x = timeline_x - THUMB_PADDING;
        const float groove_w = timeline_width + THUMB_PADDING * 2.0f;
        const ImVec2 groove_min(groove_x, strip_y + THUMB_PADDING);
        const ImVec2 groove_max(groove_x + groove_w, strip_y + STRIP_HEIGHT - THUMB_PADDING);

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
            const ImVec2 img_max(thumb.screen_x + actual_thumb_w, groove_max.y);

            dl->AddImage(static_cast<ImTextureID>(static_cast<uintptr_t>(slot.texture.get())),
                         img_min, img_max, {0, 1}, {1, 0});
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
        for (auto& slot : slots_)
            slot.valid = false;
    }

    void FilmStripRenderer::destroyGLResources() {
        for (auto& slot : slots_) {
            slot.texture = {};
            slot.valid = false;
        }
        fbo_ = {};
        depth_rbo_ = {};
        gl_initialized_ = false;
        gl_init_failed_ = false;
    }

} // namespace lfs::vis::gui

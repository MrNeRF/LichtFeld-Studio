/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// clang-format off
#include <glad/glad.h>
// clang-format on

#include "sequencer/rml_sequencer_panel.hpp"
#include "core/event_bridge/localization_manager.hpp"
#include "core/events.hpp"
#include "core/logger.hpp"
#include "gui/rmlui/rmlui_manager.hpp"
#include "gui/rmlui/rmlui_render_interface.hpp"
#include "gui/string_keys.hpp"
#include "internal/resource_paths.hpp"
#include "rendering/render_constants.hpp"
#include "theme/theme.hpp"

#include <RmlUi/Core.h>
#include <RmlUi/Core/Factory.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <format>
#include <fstream>
#include <imgui.h>

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

        std::string colorToRml(const ImVec4& c) {
            const auto r = static_cast<int>(c.x * 255.0f);
            const auto g = static_cast<int>(c.y * 255.0f);
            const auto b = static_cast<int>(c.z * 255.0f);
            const auto a = static_cast<int>(c.w * 255.0f);
            return std::format("rgba({},{},{},{})", r, g, b, a);
        }

        std::string colorToRmlWithAlpha(const ImVec4& c, const float alpha) {
            const auto r = static_cast<int>(c.x * 255.0f);
            const auto g = static_cast<int>(c.y * 255.0f);
            const auto b = static_cast<int>(c.z * 255.0f);
            const auto a = static_cast<int>(alpha * 255.0f);
            return std::format("rgba({},{},{},{})", r, g, b, a);
        }

    } // namespace

    using namespace panel_config;

    RmlSequencerPanel::RmlSequencerPanel(SequencerController& controller, gui::RmlUIManager* rml_manager)
        : controller_(controller),
          rml_manager_(rml_manager) {
        assert(rml_manager_);
    }

    RmlSequencerPanel::~RmlSequencerPanel() {
        destroyFBO();
    }

    void RmlSequencerPanel::destroyGLResources() {
        destroyFBO();
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
        }
    }

    void RmlSequencerPanel::initFBO(const int width, const int height) {
        if (fbo_ && fbo_width_ == width && fbo_height_ == height)
            return;

        destroyFBO();

        fbo_width_ = width;
        fbo_height_ = height;

        glGenFramebuffers(1, &fbo_);
        glGenTextures(1, &fbo_texture_);
        glGenRenderbuffers(1, &fbo_depth_stencil_);

        glBindTexture(GL_TEXTURE_2D, fbo_texture_);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glBindTexture(GL_TEXTURE_2D, 0);

        glBindRenderbuffer(GL_RENDERBUFFER, fbo_depth_stencil_);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, width, height);
        glBindRenderbuffer(GL_RENDERBUFFER, 0);

        glBindFramebuffer(GL_FRAMEBUFFER, fbo_);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, fbo_texture_, 0);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, fbo_depth_stencil_);

        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            LOG_ERROR("Sequencer panel FBO incomplete");
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            destroyFBO();
            return;
        }
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    void RmlSequencerPanel::destroyFBO() {
        if (fbo_texture_) {
            glDeleteTextures(1, &fbo_texture_);
            fbo_texture_ = 0;
        }
        if (fbo_depth_stencil_) {
            glDeleteRenderbuffers(1, &fbo_depth_stencil_);
            fbo_depth_stencil_ = 0;
        }
        if (fbo_) {
            glDeleteFramebuffers(1, &fbo_);
            fbo_ = 0;
        }
        fbo_width_ = 0;
        fbo_height_ = 0;
    }

    std::string RmlSequencerPanel::generateThemeRCSS() const {
        const auto& p = lfs::vis::theme().palette;
        const auto& t = lfs::vis::theme();

        const auto surface_alpha = colorToRmlWithAlpha(p.surface, 0.95f);
        const auto border = colorToRmlWithAlpha(p.border, 0.4f);
        const auto text = colorToRml(p.text);
        const auto text_dim = colorToRml(p.text_dim);
        const auto text_dim_half = colorToRmlWithAlpha(p.text_dim, 0.5f);
        const auto bg_alpha = colorToRmlWithAlpha(p.background, 0.8f);
        const auto border_dim = colorToRmlWithAlpha(p.border, 0.3f);
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
        if (std::memcmp(&last_synced_text_, &p.text, sizeof(ImVec4)) == 0)
            return;
        last_synced_text_ = p.text;

        if (base_rcss_.empty()) {
            try {
                auto rcss_path = lfs::vis::getAssetPath("rmlui/sequencer.rcss");
                std::ifstream f(rcss_path);
                if (f) {
                    base_rcss_.assign(std::istreambuf_iterator<char>(f),
                                      std::istreambuf_iterator<char>());
                }
            } catch (...) {
            }
        }

        const std::string combined = base_rcss_ + "\n" + generateThemeRCSS();
        auto sheet = Rml::Factory::InstanceStyleSheetString(combined);
        if (sheet)
            document_->SetStyleSheetContainer(std::move(sheet));
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

    void RmlSequencerPanel::forwardInput() {
        if (!rml_context_)
            return;

        const ImVec2 mouse = ImGui::GetMousePos();
        const float local_x = mouse.x - cached_panel_x_;
        const float local_y = mouse.y - cached_panel_y_;

        const bool hovered = local_x >= 0 && local_y >= 0 &&
                             local_x < cached_panel_width_ && local_y < HEIGHT;
        if (!hovered)
            return;

        const float dp_ratio = rml_manager_->getDpRatio();
        rml_context_->ProcessMouseMove(static_cast<int>(local_x * dp_ratio),
                                       static_cast<int>(local_y * dp_ratio), 0);
    }

    void RmlSequencerPanel::render(const float viewport_x, const float viewport_width,
                                   const float viewport_y_bottom) {
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

        forwardInput();

        rml_context_->SetDimensions(Rml::Vector2i(w, h));
        rml_context_->Update();

        initFBO(w, h);
        if (!fbo_)
            return;

        auto* render_iface = rml_manager_->getRenderInterface();
        assert(render_iface);
        render_iface->SetViewport(fbo_width_, fbo_height_);

        GLint prev_fbo = 0;
        glGetIntegerv(GL_FRAMEBUFFER_BINDING, &prev_fbo);

        glBindFramebuffer(GL_FRAMEBUFFER, fbo_);
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

        render_iface->BeginFrame();
        rml_context_->Render();
        render_iface->EndFrame();

        glBindFramebuffer(GL_FRAMEBUFFER, prev_fbo);

        constexpr ImGuiWindowFlags PANEL_FLAGS =
            ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
            ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoBringToFrontOnFocus |
            ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoBackground;

        const ImVec2 panel_pos = {panel_x, panel_y};
        const ImVec2 panel_size = {panel_width, HEIGHT};

        ImGui::SetNextWindowPos(panel_pos);
        ImGui::SetNextWindowSize(panel_size);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {0, 0});

        if (ImGui::Begin("##RmlSequencerPanel", nullptr, PANEL_FLAGS)) {
            ImGui::Image(static_cast<ImTextureID>(static_cast<uintptr_t>(fbo_texture_)),
                         ImVec2(panel_width, HEIGHT), ImVec2(0, 1), ImVec2(1, 0));

            const float content_height = HEIGHT - 2.0f * INNER_PADDING;
            const float timeline_width = panel_size.x - 2.0f * INNER_PADDING -
                                         TRANSPORT_WIDTH - TIME_DISPLAY_WIDTH;

            // Transport button interaction (invisible ImGui buttons over RmlUI icons)
            {
                const float y_center = panel_pos.y + INNER_PADDING + content_height / 2.0f;
                const float btn_half = BUTTON_SIZE / 2.0f;
                const float btn_y = y_center - btn_half;
                float x_offset = panel_pos.x + INNER_PADDING;

                ImGui::SetCursorScreenPos({x_offset, btn_y});
                if (ImGui::InvisibleButton("##first", {BUTTON_SIZE, BUTTON_SIZE}))
                    controller_.seekToFirstKeyframe();
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("Go to first keyframe");
                x_offset += BUTTON_SIZE + BUTTON_SPACING;

                ImGui::SetCursorScreenPos({x_offset, btn_y});
                if (ImGui::InvisibleButton("##stop", {BUTTON_SIZE, BUTTON_SIZE}))
                    controller_.stop();
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("Stop");
                x_offset += BUTTON_SIZE + BUTTON_SPACING;

                ImGui::SetCursorScreenPos({x_offset, btn_y});
                if (ImGui::InvisibleButton("##playpause", {BUTTON_SIZE, BUTTON_SIZE}))
                    controller_.togglePlayPause();
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip(controller_.isPlaying() ? "Pause (Space)" : "Play (Space)");
                x_offset += BUTTON_SIZE + BUTTON_SPACING;

                ImGui::SetCursorScreenPos({x_offset, btn_y});
                if (ImGui::InvisibleButton("##last", {BUTTON_SIZE, BUTTON_SIZE}))
                    controller_.seekToLastKeyframe();
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("Go to last keyframe");
                x_offset += BUTTON_SIZE + BUTTON_SPACING + 4.0f;

                ImGui::SetCursorScreenPos({x_offset, btn_y});
                if (ImGui::InvisibleButton("##loop", {BUTTON_SIZE, BUTTON_SIZE}))
                    controller_.toggleLoop();
                if (ImGui::IsItemHovered()) {
                    const bool looping = controller_.loopMode() != LoopMode::ONCE;
                    ImGui::SetTooltip(looping ? "Loop: ON" : "Loop: OFF");
                }
                x_offset += BUTTON_SIZE + BUTTON_SPACING;

                ImGui::SetCursorScreenPos({x_offset, btn_y});
                if (ImGui::InvisibleButton("##addkf", {BUTTON_SIZE, BUTTON_SIZE}))
                    lfs::core::events::cmd::SequencerAddKeyframe{}.emit();
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("Add keyframe (K)");
            }

            // Timeline interaction
            const ImVec2 timeline_pos = {panel_pos.x + INNER_PADDING + TRANSPORT_WIDTH,
                                         panel_pos.y + INNER_PADDING};
            handleTimelineInteraction(timeline_pos, timeline_width, content_height);
        }
        ImGui::End();
        ImGui::PopStyleVar();

        renderTimeEditPopup();
        renderFocalLengthEditPopup();
    }

    void RmlSequencerPanel::handleTimelineInteraction(const ImVec2& pos, const float width,
                                                      const float height) {
        const float timeline_y = pos.y + RULER_HEIGHT + 4.0f;
        const float timeline_height = height - RULER_HEIGHT - 4.0f;
        const float bar_half = std::min(timeline_height, TIMELINE_HEIGHT) / 2.0f;
        const float y_center = timeline_y + timeline_height / 2.0f;

        const ImVec2 bar_min = {pos.x, y_center - bar_half};
        const ImVec2 bar_max = {pos.x + width, y_center + bar_half};

        const auto& timeline = controller_.timeline();
        if (timeline.empty())
            return;

        const ImVec2 mouse = ImGui::GetMousePos();
        const bool mouse_in_timeline = mouse.x >= bar_min.x && mouse.x <= bar_max.x &&
                                       mouse.y >= bar_min.y - RULER_HEIGHT && mouse.y <= bar_max.y;

        if (mouse_in_timeline && !ImGui::GetIO().WantCaptureMouse) {
            const float wheel = ImGui::GetIO().MouseWheel;
            if (std::abs(wheel) > 0.01f) {
                const float old_zoom = zoom_level_;
                zoom_level_ = std::clamp(zoom_level_ + wheel * ZOOM_SPEED, MIN_ZOOM, MAX_ZOOM);

                if (zoom_level_ != old_zoom) {
                    const float mouse_time = xToTime(mouse.x, pos.x, width);
                    pan_offset_ += (mouse_time - pan_offset_) * (1.0f - old_zoom / zoom_level_) * 0.5f;
                    pan_offset_ = std::max(0.0f, pan_offset_);
                }
            }
        }

        if (ImGui::IsMouseClicked(ImGuiMouseButton_Left) && mouse_in_timeline && !dragging_keyframe_ &&
            !hovered_keyframe_.has_value()) {
            dragging_playhead_ = true;
            controller_.beginScrub();
        }
        if (dragging_playhead_) {
            if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
                float time = xToTime(mouse.x, pos.x, width);
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
            const float dist = std::abs(mouse.x - x);
            const bool hovered = mouse_in_timeline && dist < KEYFRAME_RADIUS * 2;
            if (hovered)
                hovered_keyframe_ = i;

            if (hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                const float current_time = static_cast<float>(ImGui::GetTime());

                if (last_clicked_keyframe_ == i &&
                    (current_time - last_click_time_) < DOUBLE_CLICK_TIME) {
                    editing_keyframe_time_ = true;
                    editing_keyframe_index_ = i;
                    std::snprintf(time_edit_buffer_, sizeof(time_edit_buffer_), "%.2f", keyframes[i].time);
                    last_clicked_keyframe_ = std::nullopt;
                } else {
                    last_click_time_ = current_time;
                    last_clicked_keyframe_ = i;

                    if (ImGui::GetIO().KeyShift && controller_.hasSelection()) {
                        const size_t first_sel = *controller_.selectedKeyframe();
                        const size_t lo = std::min(first_sel, i);
                        const size_t hi = std::max(first_sel, i);
                        selected_keyframes_.clear();
                        for (size_t j = lo; j <= hi; ++j)
                            selected_keyframes_.insert(j);
                    } else if (ImGui::GetIO().KeyCtrl) {
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
                            drag_start_mouse_x_ = mouse.x;
                        } else {
                            lfs::core::events::cmd::SequencerGoToKeyframe{.keyframe_index = i}.emit();
                        }
                    }
                }
            }
        }

        if (dragging_keyframe_) {
            if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
                float new_time = xToTime(mouse.x, pos.x, width);
                new_time = std::max(new_time, MIN_KEYFRAME_SPACING);
                if (snap_enabled_)
                    new_time = snapTime(new_time);
                controller_.timeline().setKeyframeTime(dragged_keyframe_index_, new_time, false);
            } else {
                if (std::abs(mouse.x - drag_start_mouse_x_) < DRAG_THRESHOLD_PX) {
                    lfs::core::events::cmd::SequencerGoToKeyframe{.keyframe_index = dragged_keyframe_index_}.emit();
                }
                controller_.timeline().sortKeyframes();
                dragging_keyframe_ = false;
            }
        }

        if ((controller_.hasSelection() || !selected_keyframes_.empty()) &&
            ImGui::IsKeyPressed(ImGuiKey_Delete)) {
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

                const bool is_last = (idx == keyframes.size() - 1);
                if (ImGui::BeginMenu("Easing", !is_last)) {
                    const auto current_easing = keyframes[idx].easing;
                    for (int e = 0; e < 4; ++e) {
                        const auto easing = static_cast<sequencer::EasingType>(e);
                        if (ImGui::MenuItem(EASING_NAMES[e], nullptr, current_easing == easing)) {
                            if (current_easing != easing)
                                controller_.timeline().setKeyframeEasing(idx, easing);
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

        if (hovered_keyframe_.has_value()) {
            const size_t hi = *hovered_keyframe_;
            if (hi < keyframes.size()) {
                const bool is_loop = keyframes[hi].is_loop_point;
                const char* tooltip = is_loop
                                          ? "Loop Point @ %s (returns to start)"
                                          : "Keyframe @ %s (double-click to edit)";
                ImGui::SetTooltip(tooltip, formatTime(keyframes[hi].time).c_str());
            }
        }
    }

    void RmlSequencerPanel::renderTimeEditPopup() {
        if (!editing_keyframe_time_)
            return;

        if (!ImGui::IsPopupOpen("EditKeyframeTime"))
            ImGui::OpenPopup("EditKeyframeTime");

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

    void RmlSequencerPanel::openFocalLengthEdit(const size_t index, const float current_focal_mm) {
        editing_focal_length_ = true;
        editing_focal_index_ = index;
        std::snprintf(focal_edit_buffer_, sizeof(focal_edit_buffer_), "%.1f", current_focal_mm);
    }

    void RmlSequencerPanel::renderFocalLengthEditPopup() {
        if (!editing_focal_length_)
            return;

        if (!ImGui::IsPopupOpen("EditKeyframeFocalLength"))
            ImGui::OpenPopup("EditKeyframeFocalLength");

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

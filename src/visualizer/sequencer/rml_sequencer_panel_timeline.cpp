/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/events.hpp"
#include "gui/film_strip_renderer.hpp"
#include "gui/rmlui/rml_panel_host.hpp"
#include "gui/rmlui/rml_tooltip.hpp"
#include "gui/rmlui/rmlui_manager.hpp"
#include "sequencer/interpolation.hpp"
#include "sequencer/rml_sequencer_panel.hpp"
#include "sequencer/timeline_view_math.hpp"

#include <RmlUi/Core.h>
#include <algorithm>
#include <cmath>
#include <fmt/format.h>

namespace lfs::vis {

    namespace {
        constexpr float MIN_KEYFRAME_SPACING = 0.1f;
        constexpr float DOUBLE_CLICK_TIME = 0.3f;
        constexpr float DRAG_THRESHOLD_PX = 3.0f;
        constexpr float PLAYHEAD_HIT_RADIUS = 6.0f;
        constexpr float PLAYHEAD_HANDLE_WIDTH = 8.0f;
    } // namespace

    using namespace panel_config;

    void RmlSequencerPanel::rebuildEasingStripe(const float timeline_x, const float timeline_width) {
        if (!elements_cached_)
            return;

        const auto& keyframes = controller_.timeline().keyframes();
        if (timeline_width <= 0.0f || keyframes.empty()) {
            el_easing_segments_->SetInnerRML("");
            el_easing_curves_->SetInnerRML("");
            el_easing_indicators_->SetInnerRML("");
            return;
        }

        constexpr int CURVE_SAMPLES = 20;
        const float stripe_h = EASING_STRIPE_HEIGHT * cached_dp_ratio_;
        const float y_center = stripe_h * 0.5f;
        const float amplitude = stripe_h * 0.35f;
        const float display_end = getDisplayEndTime();
        const float pan = pan_offset_;

        const auto localTimeToX = [&](const float time) -> float {
            return sequencer_ui::timeToScreenX(time, timeline_x, timeline_width, display_end, pan) - timeline_x;
        };

        std::string segments_html;
        std::string curves_html;
        std::string indicators_html;
        segments_html.reserve(512);
        curves_html.reserve(4096);
        indicators_html.reserve(1024);

        for (size_t i = 0; i + 1 < keyframes.size(); ++i) {
            const float x0 = localTimeToX(keyframes[i].time);
            const float x1 = localTimeToX(keyframes[i + 1].time);
            if (x1 <= x0)
                continue;

            segments_html += fmt::format(
                "<div class=\"easing-segment {}\" style=\"left:{:.1f}px;width:{:.1f}px;\"></div>",
                (i % 2 == 0) ? "primary" : "secondary",
                x0, x1 - x0);

            const auto easing = keyframes[i].easing;
            if (easing == sequencer::EasingType::LINEAR) {
                const float len = x1 - x0;
                curves_html += fmt::format(
                    "<div class=\"easing-curve-segment\" style=\"left:{:.1f}px;top:{:.1f}px;width:{:.1f}px;transform:rotate(0deg);\"></div>",
                    x0, y_center, len);
                continue;
            }

            for (int s = 0; s < CURVE_SAMPLES; ++s) {
                const float t0 = static_cast<float>(s) / static_cast<float>(CURVE_SAMPLES);
                const float t1 = static_cast<float>(s + 1) / static_cast<float>(CURVE_SAMPLES);
                const float eased0 = sequencer::applyEasing(t0, easing);
                const float eased1 = sequencer::applyEasing(t1, easing);
                const float px0 = x0 + t0 * (x1 - x0);
                const float px1 = x0 + t1 * (x1 - x0);
                const float py0 = y_center - (eased0 - t0) * amplitude;
                const float py1 = y_center - (eased1 - t1) * amplitude;
                const float dx = px1 - px0;
                const float dy = py1 - py0;
                const float len = std::sqrt(dx * dx + dy * dy);
                if (len < 0.25f)
                    continue;

                const float angle_deg = std::atan2(dy, dx) * 57.2957795f;
                curves_html += fmt::format(
                    "<div class=\"easing-curve-segment\" style=\"left:{:.1f}px;top:{:.1f}px;width:{:.1f}px;transform:rotate({:.2f}deg);\"></div>",
                    px0, py0, len, angle_deg);
            }
        }

        for (size_t i = 0; i < keyframes.size(); ++i) {
            const float kx = localTimeToX(keyframes[i].time);
            const char* tone = (i % 2 == 0) ? "primary" : "secondary";
            indicators_html += fmt::format(
                "<div class=\"easing-dot {}\" style=\"left:{:.1f}px;top:{:.1f}px;\"></div>",
                tone, kx, y_center);

            const auto easing = keyframes[i].easing;
            if (easing == sequencer::EasingType::LINEAR)
                continue;

            const float iy = y_center - stripe_h * 0.3f;
            const char* easing_class = "";
            switch (easing) {
            case sequencer::EasingType::EASE_IN: easing_class = "ease-in"; break;
            case sequencer::EasingType::EASE_OUT: easing_class = "ease-out"; break;
            case sequencer::EasingType::EASE_IN_OUT: easing_class = "ease-in-out"; break;
            default: break;
            }

            indicators_html += fmt::format(
                "<div class=\"easing-indicator {} {}\" style=\"left:{:.1f}px;top:{:.1f}px;\"></div>",
                easing_class, tone, kx, iy);
        }

        el_easing_segments_->SetInnerRML(segments_html);
        el_easing_curves_->SetInnerRML(curves_html);
        el_easing_indicators_->SetInnerRML(indicators_html);
    }

    void RmlSequencerPanel::ensureFilmThumbPool(const size_t count) {
        if (!elements_cached_ || !el_film_strip_thumbs_)
            return;

        while (film_thumb_elements_.size() < count) {
            auto thumb = document_->CreateElement("div");
            auto* thumb_raw = thumb.get();
            thumb_raw->SetClassNames("film-thumb");

            auto image = document_->CreateElement("img");
            image->SetClassNames("film-thumb-image");
            thumb_raw->AppendChild(std::move(image));

            auto tint_hover = document_->CreateElement("div");
            tint_hover->SetClassNames("film-thumb-tint hovered-keyframe");
            thumb_raw->AppendChild(std::move(tint_hover));

            auto tint_selected = document_->CreateElement("div");
            tint_selected->SetClassNames("film-thumb-tint selected");
            thumb_raw->AppendChild(std::move(tint_selected));

            auto edge_top = document_->CreateElement("div");
            edge_top->SetClassNames("film-thumb-edge top");
            thumb_raw->AppendChild(std::move(edge_top));

            auto edge_bottom = document_->CreateElement("div");
            edge_bottom->SetClassNames("film-thumb-edge bottom");
            thumb_raw->AppendChild(std::move(edge_bottom));

            auto outline = document_->CreateElement("div");
            outline->SetClassNames("film-thumb-outline");
            thumb_raw->AppendChild(std::move(outline));

            auto mid_shadow = document_->CreateElement("div");
            mid_shadow->SetClassNames("film-thumb-midline shadow");
            thumb_raw->AppendChild(std::move(mid_shadow));

            auto mid_main = document_->CreateElement("div");
            mid_main->SetClassNames("film-thumb-midline main");
            thumb_raw->AppendChild(std::move(mid_main));

            el_film_strip_thumbs_->AppendChild(std::move(thumb));
            film_thumb_elements_.push_back(thumb_raw);
        }
    }

    void RmlSequencerPanel::clearFilmThumbPool() {
        if (!el_film_strip_thumbs_)
            return;

        while (!film_thumb_elements_.empty()) {
            auto* el = film_thumb_elements_.back();
            if (el && el->GetNumChildren() > 0) {
                if (auto* image = el->GetChild(0))
                    image->SetAttribute("src", "");
            }
            el_film_strip_thumbs_->RemoveChild(el);
            film_thumb_elements_.pop_back();
        }
    }

    void RmlSequencerPanel::unregisterFilmStripSources() {
        registered_film_strip_sources_.clear();
    }

    void RmlSequencerPanel::rebuildFilmStripDecor(const float timeline_width) {
        if (!elements_cached_)
            return;

        const float thumb_display_h = gui::FilmStripRenderer::STRIP_HEIGHT -
                                      gui::FilmStripRenderer::THUMB_PADDING * 2.0f;
        const float base_thumb_w = thumb_display_h * (static_cast<float>(gui::FilmStripRenderer::THUMB_WIDTH) /
                                                      static_cast<float>(gui::FilmStripRenderer::THUMB_HEIGHT));
        const int num_thumbs = sequencer_ui::thumbnailCount(timeline_width, base_thumb_w, zoom_level_);
        const float actual_thumb_w = num_thumbs > 0 ? timeline_width / static_cast<float>(num_thumbs) : 0.0f;
        const float groove_w = timeline_width + gui::FilmStripRenderer::THUMB_PADDING * 2.0f;

        std::string divider_html;
        divider_html.reserve(256);
        for (int i = 1; i < num_thumbs; ++i) {
            divider_html += fmt::format(
                "<div class=\"film-strip-divider\" style=\"left:{:.1f}px;\"></div>",
                gui::FilmStripRenderer::THUMB_PADDING + actual_thumb_w * static_cast<float>(i));
        }
        el_film_strip_dividers_->SetInnerRML(divider_html);

        std::string sprocket_top_html;
        std::string sprocket_bottom_html;
        const float sprocket_start = gui::FilmStripRenderer::SPROCKET_SPACING * 0.5f;
        const int sprocket_count = static_cast<int>(groove_w / gui::FilmStripRenderer::SPROCKET_SPACING);
        sprocket_top_html.reserve(static_cast<size_t>(sprocket_count) * 48);
        sprocket_bottom_html.reserve(static_cast<size_t>(sprocket_count) * 48);
        for (int i = 0; i < sprocket_count; ++i) {
            const float cx = sprocket_start + static_cast<float>(i) * gui::FilmStripRenderer::SPROCKET_SPACING;
            const float sx = cx - gui::FilmStripRenderer::SPROCKET_W * 0.5f;
            sprocket_top_html += fmt::format(
                "<div class=\"film-strip-sprocket top\" style=\"left:{:.1f}px;\"></div>", sx);
            sprocket_bottom_html += fmt::format(
                "<div class=\"film-strip-sprocket bottom\" style=\"left:{:.1f}px;\"></div>", sx);
        }
        el_film_strip_sprockets_top_->SetInnerRML(sprocket_top_html);
        el_film_strip_sprockets_bottom_->SetInnerRML(sprocket_bottom_html);
    }

    void RmlSequencerPanel::rebuildFilmStrip(float timeline_x, const float timeline_width,
                                             const float strip_y, const PanelInputState& input,
                                             RenderingManager* rm, SceneManager* sm,
                                             gui::FilmStripRenderer& film_strip) {
        if (!elements_cached_)
            return;

        if (!film_strip_attached_) {
            if (film_strip_scrubbing_) {
                film_strip_scrubbing_ = false;
                controller_.endScrub();
            }
            unregisterFilmStripSources();
            clearFilmThumbPool();
            el_film_strip_gaps_->SetInnerRML("");
            el_film_strip_markers_->SetInnerRML("");
            el_film_strip_dividers_->SetInnerRML("");
            el_film_strip_sprockets_top_->SetInnerRML("");
            el_film_strip_sprockets_bottom_->SetInnerRML("");
            updateTimelineTooltip(film_strip, input);
            return;
        }

        std::optional<float> selected_keyframe_time;
        if (const auto selected = controller_.selectedKeyframe(); selected.has_value()) {
            if (const auto* const keyframe = controller_.timeline().getKeyframe(*selected))
                selected_keyframe_time = keyframe->time;
        }

        std::optional<float> hovered_keyframe_time;
        if (const auto hovered_id = hoveredKeyframeId(); hovered_id.has_value()) {
            if (const auto* const keyframe = controller_.timeline().getKeyframeById(*hovered_id))
                hovered_keyframe_time = keyframe->time;
        }

        gui::FilmStripRenderer::RenderOptions options;
        options.panel_x = cached_panel_x_;
        options.panel_width = cached_panel_width_;
        options.timeline_x = timeline_x;
        options.timeline_width = timeline_width;
        options.strip_y = strip_y;
        options.mouse_x = input.mouse_x;
        options.mouse_y = input.mouse_y;
        options.zoom_level = zoom_level_;
        options.pan_offset = pan_offset_;
        options.display_end_time = getDisplayEndTime();
        options.selected_keyframe_id = controller_.selectedKeyframeId();
        options.hovered_keyframe_id = hoveredKeyframeId();
        options.selected_keyframe_time = selected_keyframe_time;
        options.hovered_keyframe_time = hovered_keyframe_time;
        film_strip.render(controller_, rm, sm, options);

        handleFilmStripInteraction(timeline_x, timeline_width, input, film_strip);
        rebuildFilmStripDecor(timeline_width);

        const float groove_origin_x = timeline_x - gui::FilmStripRenderer::THUMB_PADDING;

        std::string gaps_html;
        if (controller_.timeline().size() >= 2) {
            const float visible_left_x = gui::FilmStripRenderer::THUMB_PADDING;
            const float visible_right_x = gui::FilmStripRenderer::THUMB_PADDING + timeline_width;
            const float anim_start_x = std::clamp(
                timeToX(controller_.timeline().startTime(), timeline_x, timeline_width) - timeline_x +
                    gui::FilmStripRenderer::THUMB_PADDING,
                visible_left_x, visible_right_x);
            const float anim_end_x = std::clamp(
                timeToX(controller_.timeline().endTime(), timeline_x, timeline_width) - timeline_x +
                    gui::FilmStripRenderer::THUMB_PADDING,
                visible_left_x, visible_right_x);

            const auto append_gap_region = [&](const float x_min, const float x_max) {
                if (x_max <= x_min)
                    return;

                gaps_html += fmt::format(
                    "<div class=\"film-strip-gap\" style=\"left:{:.1f}px;width:{:.1f}px;\">",
                    x_min, x_max - x_min);
                const float stripe_span = gui::FilmStripRenderer::STRIP_HEIGHT -
                                          gui::FilmStripRenderer::THUMB_PADDING * 2.0f;
                for (float stripe_x = -stripe_span; stripe_x < (x_max - x_min) + stripe_span;
                     stripe_x += 10.0f) {
                    gaps_html += fmt::format(
                        "<div class=\"film-strip-gap-stripe\" style=\"left:{:.1f}px;top:{:.1f}px;height:{:.1f}px;transform:rotate(45deg);\"></div>",
                        stripe_x, stripe_span, stripe_span * 1.4142f);
                }
                gaps_html += "</div>";
            };

            if (anim_start_x > visible_left_x)
                append_gap_region(visible_left_x, anim_start_x);
            if (anim_end_x < visible_right_x)
                append_gap_region(anim_end_x, visible_right_x);
        }
        el_film_strip_gaps_->SetInnerRML(gaps_html);

        ensureFilmThumbPool(film_strip.thumbs().size());
        std::set<std::string> active_sources;
        for (size_t i = 0; i < film_thumb_elements_.size(); ++i) {
            auto* thumb_el = film_thumb_elements_[i];
            auto* image_el = thumb_el && thumb_el->GetNumChildren() > 0 ? thumb_el->GetChild(0) : nullptr;
            if (!thumb_el || !image_el)
                continue;

            if (i >= film_strip.thumbs().size()) {
                thumb_el->SetProperty("display", "none");
                image_el->SetAttribute("src", "");
                continue;
            }

            const auto& thumb = film_strip.thumbs()[i];
            const std::uintptr_t texture_id = film_strip.textureIdForSlot(thumb.slot_idx);
            if (texture_id == 0) {
                thumb_el->SetProperty("display", "none");
                image_el->SetAttribute("src", "");
                continue;
            }

            thumb_el->SetProperty("display", "block");
            thumb_el->SetProperty("left", fmt::format("{:.1f}px", thumb.screen_x - groove_origin_x));
            thumb_el->SetProperty("width", fmt::format("{:.1f}px", thumb.screen_width));
            thumb_el->SetClassNames("film-thumb");
            thumb_el->SetClass("hovered", thumb.hovered);
            thumb_el->SetClass("contains-selected", thumb.contains_selected);
            thumb_el->SetClass("contains-hovered-keyframe", thumb.contains_hovered_keyframe);
            thumb_el->SetClass("stale", thumb.stale);
            image_el->SetAttribute("src", "");
        }

        for (auto it = registered_film_strip_sources_.begin(); it != registered_film_strip_sources_.end();) {
            if (!active_sources.contains(*it))
                it = registered_film_strip_sources_.erase(it);
            else
                ++it;
        }
        registered_film_strip_sources_.insert(active_sources.begin(), active_sources.end());

        std::string markers_html;
        markers_html.reserve(film_strip.markers().size() * 196);
        for (const auto& marker : film_strip.markers()) {
            markers_html += fmt::format(
                "<div class=\"film-strip-marker{}{}\" style=\"left:{:.1f}px;\">"
                "<div class=\"film-strip-marker-line shadow\"></div>"
                "<div class=\"film-strip-marker-line main\"></div>"
                "<div class=\"film-strip-marker-cap top\"></div>"
                "<div class=\"film-strip-marker-cap bottom\"></div>"
                "</div>",
                marker.selected ? " selected" : "",
                marker.hovered ? " hovered" : "",
                marker.screen_x - groove_origin_x);
        }
        el_film_strip_markers_->SetInnerRML(markers_html);

        updateTimelineTooltip(film_strip, input);
    }

} // namespace lfs::vis

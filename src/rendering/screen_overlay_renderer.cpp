/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "screen_overlay_renderer.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace lfs::rendering {

    namespace {
        constexpr int kMinCircleSegments = 12;
        constexpr int kMaxCircleSegments = 64;

        int adaptiveSegments(const float radius) {
            const float r = std::max(radius, 1.0f);
            const int seg = static_cast<int>(std::round(r * 0.75f)) + 8;
            return std::clamp(seg, kMinCircleSegments, kMaxCircleSegments);
        }

        TextMeasureFn& textMeasureFn() {
            static TextMeasureFn fn;
            return fn;
        }
    } // namespace

    void ScreenOverlayRenderer::setTextMeasureFn(TextMeasureFn fn) {
        textMeasureFn() = std::move(fn);
    }

    void ScreenOverlayRenderer::beginFrame() {
        frame_active_ = true;
        commands_.clear();
        clip_stack_.clear();
    }

    void ScreenOverlayRenderer::endFrame() {
        frame_active_ = false;
    }

    void ScreenOverlayRenderer::pushClipRect(const glm::vec2 min, const glm::vec2 max,
                                             const bool intersect_with_current) {
        OverlayClipRect next{.min = glm::min(min, max), .max = glm::max(min, max)};
        if (intersect_with_current && !clip_stack_.empty()) {
            const auto& cur = clip_stack_.back();
            next.min = glm::max(next.min, cur.min);
            next.max = glm::min(next.max, cur.max);
            next.max = glm::max(next.max, next.min);
        }
        clip_stack_.push_back(next);
    }

    void ScreenOverlayRenderer::popClipRect() {
        if (!clip_stack_.empty()) {
            clip_stack_.pop_back();
        }
    }

    std::optional<OverlayClipRect> ScreenOverlayRenderer::currentClip() const {
        if (clip_stack_.empty()) {
            return std::nullopt;
        }
        return clip_stack_.back();
    }

    glm::vec4 ScreenOverlayRenderer::toPremul(const OverlayColor c) const {
        const float a = std::clamp(c.a, 0.0f, 1.0f);
        return {c.r * a, c.g * a, c.b * a, a};
    }

    void ScreenOverlayRenderer::addLine(const glm::vec2 a, const glm::vec2 b, const OverlayColor c,
                                        const float thickness) {
        if (c.a <= 0.0f) {
            return;
        }
        commands_.push_back({
            .type = OverlayCommandType::Line,
            .p0 = a,
            .p1 = b,
            .color_premul = toPremul(c),
            .thickness = std::max(thickness, 1.0f),
            .clip = currentClip(),
        });
    }

    void ScreenOverlayRenderer::addPolyline(const std::span<const glm::vec2> pts, const OverlayColor c,
                                            const bool closed, const float thickness) {
        if (pts.size() < 2 || c.a <= 0.0f) {
            return;
        }
        for (std::size_t i = 0; i + 1 < pts.size(); ++i) {
            addLine(pts[i], pts[i + 1], c, thickness);
        }
        if (closed && pts.size() > 2) {
            addLine(pts.back(), pts.front(), c, thickness);
        }
    }

    void ScreenOverlayRenderer::addConvexPolyFilled(const std::span<const glm::vec2> pts,
                                                    const OverlayColor c) {
        if (pts.size() < 3 || c.a <= 0.0f) {
            return;
        }
        for (std::size_t i = 1; i + 1 < pts.size(); ++i) {
            addTriangleFilled(pts[0], pts[i], pts[i + 1], c);
        }
    }

    void ScreenOverlayRenderer::addCircle(const glm::vec2 center, const float radius,
                                          const OverlayColor c, const int segments, const float thickness) {
        if (radius <= 0.0f || c.a <= 0.0f) {
            return;
        }
        commands_.push_back({
            .type = OverlayCommandType::CircleOutline,
            .p0 = center,
            .color_premul = toPremul(c),
            .thickness = std::max(thickness, 1.0f),
            .radius = radius,
            .segments = segments > 0 ? segments : adaptiveSegments(radius),
            .clip = currentClip(),
        });
    }

    void ScreenOverlayRenderer::addCircleFilled(const glm::vec2 center, const float radius,
                                                const OverlayColor c, const int segments) {
        if (radius <= 0.0f || c.a <= 0.0f) {
            return;
        }
        commands_.push_back({
            .type = OverlayCommandType::CircleFilled,
            .p0 = center,
            .color_premul = toPremul(c),
            .thickness = radius,
            .radius = radius,
            .segments = segments > 0 ? segments : adaptiveSegments(radius),
            .clip = currentClip(),
        });
    }

    void ScreenOverlayRenderer::addRect(const glm::vec2 a, const glm::vec2 b, const OverlayColor c,
                                        const float thickness) {
        const glm::vec2 min = glm::min(a, b);
        const glm::vec2 max = glm::max(a, b);
        const std::array<glm::vec2, 4> pts = {
            glm::vec2{min.x, min.y}, glm::vec2{max.x, min.y},
            glm::vec2{max.x, max.y}, glm::vec2{min.x, max.y}};
        addPolyline(pts, c, true, thickness);
    }

    void ScreenOverlayRenderer::addRectFilled(const glm::vec2 a, const glm::vec2 b, const OverlayColor c) {
        const glm::vec2 min = glm::min(a, b);
        const glm::vec2 max = glm::max(a, b);
        const std::array<glm::vec2, 4> pts = {
            glm::vec2{min.x, min.y}, glm::vec2{max.x, min.y},
            glm::vec2{max.x, max.y}, glm::vec2{min.x, max.y}};
        addConvexPolyFilled(pts, c);
    }

    void ScreenOverlayRenderer::addTriangleFilled(const glm::vec2 p0, const glm::vec2 p1,
                                                  const glm::vec2 p2, const OverlayColor c) {
        if (c.a <= 0.0f) {
            return;
        }
        commands_.push_back({
            .type = OverlayCommandType::Triangle,
            .p0 = p0,
            .p1 = p1,
            .p2 = p2,
            .color_premul = toPremul(c),
            .clip = currentClip(),
        });
    }

    void ScreenOverlayRenderer::addText(const glm::vec2 top_left_px, const std::string_view text,
                                        const OverlayColor c, const float size_px) {
        if (text.empty() || size_px <= 0.0f || c.a <= 0.0f) {
            return;
        }
        commands_.push_back({
            .type = OverlayCommandType::Text,
            .p0 = top_left_px,
            .color_premul = toPremul(c),
            .clip = currentClip(),
            .text = std::string(text),
            .font_size = size_px,
        });
    }

    void ScreenOverlayRenderer::addTextWithShadow(const glm::vec2 top_left_px, const std::string_view text,
                                                  const OverlayColor c, const OverlayColor shadow_c,
                                                  const float size_px, const glm::vec2 shadow_offset) {
        addText(top_left_px + shadow_offset, text, shadow_c, size_px);
        addText(top_left_px, text, c, size_px);
    }

    glm::vec2 ScreenOverlayRenderer::measureText(const std::string_view text, const float size_px) const {
        if (const auto& fn = textMeasureFn(); fn) {
            return fn(text, size_px);
        }
        return {0.0f, 0.0f};
    }

    std::vector<OverlayCommand> ScreenOverlayRenderer::consumeCommands() {
        std::vector<OverlayCommand> out;
        out.swap(commands_);
        return out;
    }

} // namespace lfs::rendering

/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <functional>
#include <glm/glm.hpp>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace lfs::rendering {

    struct OverlayColor {
        float r = 0.0f, g = 0.0f, b = 0.0f, a = 0.0f;
    };

    enum class OverlayCommandType {
        Line,
        Triangle,
        CircleFilled,
        CircleOutline,
        Text,
    };

    struct OverlayClipRect {
        glm::vec2 min{0.0f};
        glm::vec2 max{0.0f};
    };

    struct OverlayCommand {
        OverlayCommandType type = OverlayCommandType::Line;
        glm::vec2 p0{0.0f};
        glm::vec2 p1{0.0f};
        glm::vec2 p2{0.0f};
        glm::vec4 color_premul{0.0f};
        float thickness = 1.0f;
        float radius = 0.0f;
        int segments = 0;
        std::optional<OverlayClipRect> clip;
        std::string text;
        float font_size = 0.0f;
    };

    using TextMeasureFn = std::function<glm::vec2(std::string_view, float)>;

    class ScreenOverlayRenderer {
    public:
        static void setTextMeasureFn(TextMeasureFn fn);

        void beginFrame();
        void endFrame();
        bool isFrameActive() const { return frame_active_; }

        void pushClipRect(glm::vec2 min, glm::vec2 max, bool intersect_with_current = true);
        void popClipRect();

        void addLine(glm::vec2 a, glm::vec2 b, OverlayColor c, float thickness = 1.0f);
        void addPolyline(std::span<const glm::vec2> pts, OverlayColor c, bool closed, float thickness);
        void addConvexPolyFilled(std::span<const glm::vec2> pts, OverlayColor c);
        void addCircle(glm::vec2 center, float radius, OverlayColor c, int segments, float thickness);
        void addCircleFilled(glm::vec2 center, float radius, OverlayColor c, int segments = 0);
        void addRect(glm::vec2 a, glm::vec2 b, OverlayColor c, float thickness = 1.0f);
        void addRectFilled(glm::vec2 a, glm::vec2 b, OverlayColor c);
        void addTriangleFilled(glm::vec2 p0, glm::vec2 p1, glm::vec2 p2, OverlayColor c);

        void addText(glm::vec2 top_left_px, std::string_view text, OverlayColor c, float size_px);
        void addTextWithShadow(glm::vec2 top_left_px, std::string_view text,
                               OverlayColor c, OverlayColor shadow_c, float size_px,
                               glm::vec2 shadow_offset = {1.0f, 1.0f});
        [[nodiscard]] glm::vec2 measureText(std::string_view text, float size_px) const;

        std::vector<OverlayCommand> consumeCommands();

        class ScopedClipRect {
        public:
            ScopedClipRect(ScreenOverlayRenderer& r, glm::vec2 min, glm::vec2 max,
                           bool intersect_with_current = true)
                : r_(r) {
                r_.pushClipRect(min, max, intersect_with_current);
            }
            ~ScopedClipRect() { r_.popClipRect(); }
            ScopedClipRect(const ScopedClipRect&) = delete;
            ScopedClipRect& operator=(const ScopedClipRect&) = delete;

        private:
            ScreenOverlayRenderer& r_;
        };

    private:
        glm::vec4 toPremul(OverlayColor c) const;
        std::optional<OverlayClipRect> currentClip() const;

        bool frame_active_ = false;
        std::vector<OverlayCommand> commands_;
        std::vector<OverlayClipRect> clip_stack_;
    };

} // namespace lfs::rendering

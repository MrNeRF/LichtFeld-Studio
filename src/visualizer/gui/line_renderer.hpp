/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <core/export.hpp>
#include <glm/glm.hpp>

#include <cstdint>
#include <optional>
#include <vector>

namespace lfs::vis::gui {

    struct ClipRect {
        int x = 0;
        int y = 0;
        int width = 0;
        int height = 0;
    };

    enum class LineRendererCommandType {
        Line,
        Triangle,
        Circle,
        CircleOutline
    };

    struct LineRendererCommand {
        LineRendererCommandType type = LineRendererCommandType::Line;
        std::optional<ClipRect> clip_rect;
        glm::vec2 p0{0.0f};
        glm::vec2 p1{0.0f};
        glm::vec2 p2{0.0f};
        glm::vec4 color{1.0f};
        float thickness = 1.0f;
        float radius = 0.0f;
        int segments = 16;
    };

    [[nodiscard]] LFS_VIS_API std::vector<LineRendererCommand> consumeLineRendererCommands();
    LFS_VIS_API void clearLineRendererCommands();

    using OverlayColor = std::uint32_t;

    constexpr OverlayColor overlayColor(const int r, const int g, const int b, const int a) {
        return (static_cast<OverlayColor>(a) << 24u) |
               (static_cast<OverlayColor>(b) << 16u) |
               (static_cast<OverlayColor>(g) << 8u) |
               static_cast<OverlayColor>(r);
    }

    constexpr OverlayColor OVERLAY_COL32_WHITE = overlayColor(255, 255, 255, 255);
    constexpr OverlayColor OVERLAY_COL32_BLACK = overlayColor(0, 0, 0, 255);

    [[nodiscard]] LFS_VIS_API glm::vec4 overlayColorToVec4(OverlayColor color);
    [[nodiscard]] LFS_VIS_API OverlayColor overlayColorWithAlpha(OverlayColor color, float alpha);

    struct NativeGizmoInput {
        glm::vec2 mouse_pos{0.0f};
        bool mouse_left_down = false;
        bool mouse_left_clicked = false;
    };

    class LFS_VIS_API NativeOverlayDrawList {
    public:
        void PushClipRect(glm::vec2 min, glm::vec2 max, bool intersect_with_current_clip = false);
        void PopClipRect();

        void AddLine(glm::vec2 p0, glm::vec2 p1, OverlayColor color, float thickness = 1.0f);
        void AddTriangleFilled(glm::vec2 p0, glm::vec2 p1, glm::vec2 p2, OverlayColor color);
        void AddConvexPolyFilled(const glm::vec2* points, int count, OverlayColor color);
        void AddPolyline(const glm::vec2* points, int count, OverlayColor color, bool closed, float thickness = 1.0f);
        void AddRectFilled(glm::vec2 min, glm::vec2 max, OverlayColor color, float rounding = 0.0f);
        void AddRect(glm::vec2 min, glm::vec2 max, OverlayColor color, float rounding = 0.0f, float thickness = 1.0f);
        void AddCircleFilled(glm::vec2 center, float radius, OverlayColor color, int segments = 16);
        void AddCircle(glm::vec2 center, float radius, OverlayColor color, int segments = 16, float thickness = 1.0f);

    private:
        [[nodiscard]] std::optional<ClipRect> currentClipRect() const;
        void queue(LineRendererCommand command) const;

        std::vector<std::optional<ClipRect>> clip_stack_;
    };

    class LFS_VIS_API LineRenderer {
    public:
        LineRenderer() = default;

        void begin(int screen_w, int screen_h, int fb_w, int fb_h,
                   std::optional<ClipRect> clip_rect = std::nullopt);
        void addLine(glm::vec2 p0, glm::vec2 p1, glm::vec4 color, float thickness = 1.0f);
        void addTriangleFilled(glm::vec2 p0, glm::vec2 p1, glm::vec2 p2, glm::vec4 color);
        void addCircleFilled(glm::vec2 center, float radius, glm::vec4 color, int segments = 16);
        void end();

        void destroyResources();

    private:
        std::vector<LineRendererCommand> commands_;
        std::optional<ClipRect> clip_rect_;
    };

} // namespace lfs::vis::gui

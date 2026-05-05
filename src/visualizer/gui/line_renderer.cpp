/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/line_renderer.hpp"

#include <algorithm>
#include <cmath>
#include <mutex>
#include <utility>

namespace lfs::vis::gui {

    namespace {
        std::mutex g_pending_commands_mutex;
        std::vector<LineRendererCommand> g_pending_commands;
    } // namespace

    std::vector<LineRendererCommand> consumeLineRendererCommands() {
        std::lock_guard lock(g_pending_commands_mutex);
        std::vector<LineRendererCommand> out;
        out.swap(g_pending_commands);
        return out;
    }

    void clearLineRendererCommands() {
        std::lock_guard lock(g_pending_commands_mutex);
        g_pending_commands.clear();
    }

    namespace {
        void queueCommand(LineRendererCommand command) {
            std::lock_guard lock(g_pending_commands_mutex);
            g_pending_commands.push_back(std::move(command));
        }

        void queueCommands(std::vector<LineRendererCommand>& commands,
                           const std::optional<ClipRect>& clip_rect) {
            if (commands.empty()) {
                return;
            }
            std::lock_guard lock(g_pending_commands_mutex);
            g_pending_commands.reserve(g_pending_commands.size() + commands.size());
            for (auto& command : commands) {
                command.clip_rect = clip_rect;
                g_pending_commands.push_back(std::move(command));
            }
            commands.clear();
        }

        [[nodiscard]] ClipRect clipRectFromMinMax(const glm::vec2 min, const glm::vec2 max) {
            const int x0 = static_cast<int>(std::floor(std::min(min.x, max.x)));
            const int y0 = static_cast<int>(std::floor(std::min(min.y, max.y)));
            const int x1 = static_cast<int>(std::ceil(std::max(min.x, max.x)));
            const int y1 = static_cast<int>(std::ceil(std::max(min.y, max.y)));
            return {
                .x = x0,
                .y = y0,
                .width = std::max(0, x1 - x0),
                .height = std::max(0, y1 - y0),
            };
        }

        [[nodiscard]] ClipRect intersectClipRects(const ClipRect& a, const ClipRect& b) {
            const int ax1 = a.x + a.width;
            const int ay1 = a.y + a.height;
            const int bx1 = b.x + b.width;
            const int by1 = b.y + b.height;
            const int x0 = std::max(a.x, b.x);
            const int y0 = std::max(a.y, b.y);
            const int x1 = std::min(ax1, bx1);
            const int y1 = std::min(ay1, by1);
            return {
                .x = x0,
                .y = y0,
                .width = std::max(0, x1 - x0),
                .height = std::max(0, y1 - y0),
            };
        }
    } // namespace

    glm::vec4 overlayColorToVec4(const OverlayColor color) {
        return {
            static_cast<float>(color & 0xFFu) / 255.0f,
            static_cast<float>((color >> 8u) & 0xFFu) / 255.0f,
            static_cast<float>((color >> 16u) & 0xFFu) / 255.0f,
            static_cast<float>((color >> 24u) & 0xFFu) / 255.0f,
        };
    }

    OverlayColor overlayColorWithAlpha(const OverlayColor color, const float alpha) {
        const int r = static_cast<int>(color & 0xFFu);
        const int g = static_cast<int>((color >> 8u) & 0xFFu);
        const int b = static_cast<int>((color >> 16u) & 0xFFu);
        const int a = static_cast<int>(std::clamp(alpha, 0.0f, 1.0f) * 255.0f);
        return overlayColor(r, g, b, a);
    }

    void NativeOverlayDrawList::PushClipRect(const glm::vec2 min,
                                             const glm::vec2 max,
                                             const bool intersect_with_current_clip) {
        ClipRect next = clipRectFromMinMax(min, max);
        if (intersect_with_current_clip) {
            if (const auto current = currentClipRect()) {
                next = intersectClipRects(*current, next);
            }
        }
        clip_stack_.push_back(next);
    }

    void NativeOverlayDrawList::PopClipRect() {
        if (!clip_stack_.empty()) {
            clip_stack_.pop_back();
        }
    }

    std::optional<ClipRect> NativeOverlayDrawList::currentClipRect() const {
        for (auto it = clip_stack_.rbegin(); it != clip_stack_.rend(); ++it) {
            if (*it) {
                return *it;
            }
        }
        return std::nullopt;
    }

    void NativeOverlayDrawList::queue(LineRendererCommand command) const {
        command.clip_rect = currentClipRect();
        queueCommand(std::move(command));
    }

    void NativeOverlayDrawList::AddLine(const glm::vec2 p0,
                                        const glm::vec2 p1,
                                        const OverlayColor color,
                                        const float thickness) {
        queue({
            .type = LineRendererCommandType::Line,
            .clip_rect = std::nullopt,
            .p0 = p0,
            .p1 = p1,
            .color = overlayColorToVec4(color),
            .thickness = thickness,
        });
    }

    void NativeOverlayDrawList::AddTriangleFilled(const glm::vec2 p0,
                                                  const glm::vec2 p1,
                                                  const glm::vec2 p2,
                                                  const OverlayColor color) {
        queue({
            .type = LineRendererCommandType::Triangle,
            .clip_rect = std::nullopt,
            .p0 = p0,
            .p1 = p1,
            .p2 = p2,
            .color = overlayColorToVec4(color),
        });
    }

    void NativeOverlayDrawList::AddConvexPolyFilled(const glm::vec2* const points,
                                                    const int count,
                                                    const OverlayColor color) {
        if (!points || count < 3) {
            return;
        }
        for (int i = 1; i + 1 < count; ++i) {
            AddTriangleFilled(points[0], points[i], points[i + 1], color);
        }
    }

    void NativeOverlayDrawList::AddPolyline(const glm::vec2* const points,
                                            const int count,
                                            const OverlayColor color,
                                            const bool closed,
                                            const float thickness) {
        if (!points || count < 2) {
            return;
        }
        for (int i = 0; i + 1 < count; ++i) {
            AddLine(points[i], points[i + 1], color, thickness);
        }
        if (closed && count > 2) {
            AddLine(points[count - 1], points[0], color, thickness);
        }
    }

    void NativeOverlayDrawList::AddRectFilled(const glm::vec2 min,
                                              const glm::vec2 max,
                                              const OverlayColor color,
                                              const float) {
        const glm::vec2 points[4] = {
            {min.x, min.y},
            {max.x, min.y},
            {max.x, max.y},
            {min.x, max.y},
        };
        AddConvexPolyFilled(points, 4, color);
    }

    void NativeOverlayDrawList::AddRect(const glm::vec2 min,
                                        const glm::vec2 max,
                                        const OverlayColor color,
                                        const float,
                                        const float thickness) {
        const glm::vec2 points[4] = {
            {min.x, min.y},
            {max.x, min.y},
            {max.x, max.y},
            {min.x, max.y},
        };
        AddPolyline(points, 4, color, true, thickness);
    }

    void NativeOverlayDrawList::AddCircleFilled(const glm::vec2 center,
                                                const float radius,
                                                const OverlayColor color,
                                                const int segments) {
        queue({
            .type = LineRendererCommandType::Circle,
            .clip_rect = std::nullopt,
            .p0 = center,
            .color = overlayColorToVec4(color),
            .thickness = radius,
            .radius = radius,
            .segments = segments,
        });
    }

    void NativeOverlayDrawList::AddCircle(const glm::vec2 center,
                                          const float radius,
                                          const OverlayColor color,
                                          const int segments,
                                          const float thickness) {
        queue({
            .type = LineRendererCommandType::CircleOutline,
            .clip_rect = std::nullopt,
            .p0 = center,
            .color = overlayColorToVec4(color),
            .thickness = thickness,
            .radius = radius,
            .segments = segments,
        });
    }

    void LineRenderer::begin(const int, const int, const int, const int,
                             const std::optional<ClipRect> clip_rect) {
        clip_rect_ = clip_rect;
        commands_.clear();
    }

    void LineRenderer::addLine(const glm::vec2 p0,
                               const glm::vec2 p1,
                               const glm::vec4 color,
                               const float thickness) {
        commands_.push_back({
            .type = LineRendererCommandType::Line,
            .clip_rect = std::nullopt,
            .p0 = p0,
            .p1 = p1,
            .color = color,
            .thickness = thickness,
        });
    }

    void LineRenderer::addTriangleFilled(const glm::vec2 p0,
                                         const glm::vec2 p1,
                                         const glm::vec2 p2,
                                         const glm::vec4 color) {
        commands_.push_back({
            .type = LineRendererCommandType::Triangle,
            .clip_rect = std::nullopt,
            .p0 = p0,
            .p1 = p1,
            .p2 = p2,
            .color = color,
        });
    }

    void LineRenderer::addCircleFilled(const glm::vec2 center,
                                       const float radius,
                                       const glm::vec4 color,
                                       const int segments) {
        commands_.push_back({
            .type = LineRendererCommandType::Circle,
            .clip_rect = std::nullopt,
            .p0 = center,
            .color = color,
            .thickness = radius,
            .radius = radius,
            .segments = segments,
        });
    }

    void LineRenderer::end() {
        queueCommands(commands_, clip_rect_);
    }

    void LineRenderer::destroyResources() {
        commands_.clear();
    }

} // namespace lfs::vis::gui

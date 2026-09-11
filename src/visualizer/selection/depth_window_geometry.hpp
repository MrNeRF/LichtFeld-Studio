/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "rendering/rendering_types.hpp"
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <glm/glm.hpp>

namespace lfs::vis::op {

    inline constexpr float DEPTH_WINDOW_HANDLE_DRAW_RADIUS_DP = 6.0f;
    inline constexpr float DEPTH_WINDOW_HANDLE_HIT_RADIUS_PX = 11.0f;

    struct DepthWindowPanelMapping {
        SplitViewPanelId panel = SplitViewPanelId::Left;
        float x = 0.0f;
        float y = 0.0f;
        float width = 0.0f;
        float height = 0.0f;
        int render_width = 0;
        int render_height = 0;

        [[nodiscard]] bool valid() const {
            return width > 0.0f && height > 0.0f &&
                   render_width > 0 && render_height > 0;
        }
    };

    struct DepthWindowRect {
        glm::vec2 min{0.0f};
        glm::vec2 max{0.0f};

        [[nodiscard]] glm::vec2 size() const { return max - min; }
        [[nodiscard]] glm::vec2 center() const { return (min + max) * 0.5f; }
    };

    enum class DepthWindowHandle : std::uint8_t {
        None,
        TopLeft,
        TopRight,
        BottomRight,
        BottomLeft,
        Center,
        EdgeTop,
        EdgeRight,
        EdgeBottom,
        EdgeLeft,
    };

    enum class DepthWindowCursor : std::uint8_t {
        Default,
        ResizeNwse,
        ResizeNesw,
        ResizeEw,
        ResizeNs,
        Move,
        Crosshair,
    };

    struct DepthWindowHandleGeometry {
        std::array<glm::vec2, 4> corners{};
        std::array<glm::vec2, 4> edge_midpoints{}; // top, right, bottom, left
        glm::vec2 center{0.0f};
    };

    struct DepthWindowOverlayState {
        bool visible = false;
        bool has_hovered_panel = false;
        // While a NEW box is being rubber-banded the handles are hidden -
        // they only apply to a committed/released window.
        bool hide_handles = false;
        SplitViewPanelId hovered_panel = SplitViewPanelId::Left;
        DepthWindowHandle hovered_handle = DepthWindowHandle::None;
    };

    // KEEP IN SYNC: this screen->render mapping is formula-identical to the two
    // screenToRender overloads in selection_service.cpp (ViewerLayout and
    // ViewportInfo). A change to any one of the three must be mirrored in the
    // other two.
    template <typename PanelInfo>
    [[nodiscard]] glm::vec2 screenToRender(const glm::vec2& screen, const PanelInfo& info) {
        const float scale_x = static_cast<float>(info.render_width) / info.width;
        const float scale_y = static_cast<float>(info.render_height) / info.height;
        return {
            (screen.x - info.x) * scale_x,
            (screen.y - info.y) * scale_y,
        };
    }

    [[nodiscard]] inline glm::vec2 renderToScreen(const glm::vec2& render,
                                                  const DepthWindowPanelMapping& panel) {
        const float scale_x = panel.width / static_cast<float>(panel.render_width);
        const float scale_y = panel.height / static_cast<float>(panel.render_height);
        return {
            panel.x + render.x * scale_x,
            panel.y + render.y * scale_y,
        };
    }

    // KEEP IN SYNC: this is the single CPU UI definition of the screen-window
    // containment rectangle. The shader, CUDA selection kernel, and CPU reference
    // test implement the same formula for their respective execution domains.
    [[nodiscard]] inline DepthWindowRect depthWindowRenderRect(
        const int render_width,
        const int render_height,
        const float scale_x,
        const float scale_y,
        const float offset_x,
        const float offset_y) {
        const float width = static_cast<float>(render_width);
        const float height = static_cast<float>(render_height);
        const float half_width = 0.5f * scale_x * width;
        const float half_height = 0.5f * scale_y * height;
        const float center_x = 0.5f * width + offset_x * (0.5f * width - half_width);
        const float center_y = 0.5f * height + offset_y * (0.5f * height - half_height);
        return {
            .min = {center_x - half_width, center_y - half_height},
            .max = {center_x + half_width, center_y + half_height},
        };
    }

    [[nodiscard]] inline DepthWindowRect depthWindowScreenRect(
        const DepthWindowPanelMapping& panel,
        const float scale_x,
        const float scale_y,
        const float offset_x,
        const float offset_y) {
        const auto render_rect = depthWindowRenderRect(
            panel.render_width, panel.render_height,
            scale_x, scale_y, offset_x, offset_y);
        return {
            .min = renderToScreen(render_rect.min, panel),
            .max = renderToScreen(render_rect.max, panel),
        };
    }

    [[nodiscard]] inline DepthWindowHandleGeometry depthWindowHandleGeometry(
        const DepthWindowRect& screen_rect) {
        const glm::vec2 mid = screen_rect.center();
        return {
            .corners = {
                glm::vec2{screen_rect.min.x, screen_rect.min.y},
                glm::vec2{screen_rect.max.x, screen_rect.min.y},
                glm::vec2{screen_rect.max.x, screen_rect.max.y},
                glm::vec2{screen_rect.min.x, screen_rect.max.y},
            },
            .edge_midpoints = {
                glm::vec2{mid.x, screen_rect.min.y},
                glm::vec2{screen_rect.max.x, mid.y},
                glm::vec2{mid.x, screen_rect.max.y},
                glm::vec2{screen_rect.min.x, mid.y},
            },
            .center = mid,
        };
    }

    [[nodiscard]] inline DepthWindowHandle hitTestDepthWindowHandles(
        const glm::vec2 pointer,
        const DepthWindowHandleGeometry& geometry,
        const float hit_radius_px = DEPTH_WINDOW_HANDLE_HIT_RADIUS_PX) {
        const float radius_squared = hit_radius_px * hit_radius_px;
        constexpr std::array handles{
            DepthWindowHandle::TopLeft,
            DepthWindowHandle::TopRight,
            DepthWindowHandle::BottomRight,
            DepthWindowHandle::BottomLeft,
        };
        for (size_t index = 0; index < geometry.corners.size(); ++index) {
            const glm::vec2 delta = pointer - geometry.corners[index];
            if (glm::dot(delta, delta) <= radius_squared) {
                return handles[index];
            }
        }

        constexpr std::array edge_handles{
            DepthWindowHandle::EdgeTop,
            DepthWindowHandle::EdgeRight,
            DepthWindowHandle::EdgeBottom,
            DepthWindowHandle::EdgeLeft,
        };
        for (size_t index = 0; index < geometry.edge_midpoints.size(); ++index) {
            const glm::vec2 delta = pointer - geometry.edge_midpoints[index];
            if (glm::dot(delta, delta) <= radius_squared) {
                return edge_handles[index];
            }
        }

        // Proportional move zone (supersedes the old 11px center hotspot):
        // dim < 48 → 0.5*dim cap wins; 48 ≤ dim < 60 → 24px floor; dim ≥ 60 → 0.4*dim.
        const float dim_x = geometry.corners[1].x - geometry.corners[0].x;
        const float dim_y = geometry.corners[3].y - geometry.corners[0].y;
        const auto zone_for_axis = [](const float dim) {
            return std::min(std::max(0.4f * dim, 24.0f), 0.5f * dim);
        };
        const float zone_x = zone_for_axis(std::abs(dim_x));
        const float zone_y = zone_for_axis(std::abs(dim_y));
        const glm::vec2 center_delta = pointer - geometry.center;
        if (std::abs(center_delta.x) <= 0.5f * zone_x &&
            std::abs(center_delta.y) <= 0.5f * zone_y) {
            return DepthWindowHandle::Center;
        }
        return DepthWindowHandle::None;
    }

    [[nodiscard]] inline DepthWindowCursor depthWindowCursorForHandle(
        const DepthWindowHandle handle) {
        switch (handle) {
        case DepthWindowHandle::TopLeft:
        case DepthWindowHandle::BottomRight:
            return DepthWindowCursor::ResizeNwse;
        case DepthWindowHandle::TopRight:
        case DepthWindowHandle::BottomLeft:
            return DepthWindowCursor::ResizeNesw;
        case DepthWindowHandle::EdgeTop:
        case DepthWindowHandle::EdgeBottom:
            return DepthWindowCursor::ResizeNs;
        case DepthWindowHandle::EdgeLeft:
        case DepthWindowHandle::EdgeRight:
            return DepthWindowCursor::ResizeEw;
        case DepthWindowHandle::Center:
            return DepthWindowCursor::Move;
        case DepthWindowHandle::None:
        default:
            return DepthWindowCursor::Default;
        }
    }

    // While GT comparison MODE is active the depth window draws on NO panel —
    // normal panels and the GT/compare panel alike, including the loading window
    // where the engaged GT context is still unpublished. Suppression deliberately
    // keys on the broad mode predicate, never on context engagement, and never
    // mutates settings; toggling GT off restores the overlay exactly. Direct
    // depth-window interaction in GT view is deferred as future work.
    [[nodiscard]] constexpr bool depthWindowOverlaySuppressed(const bool gt_comparison_mode_active) {
        return gt_comparison_mode_active;
    }

    [[nodiscard]] inline bool pointInDepthWindowPanel(
        const glm::vec2 point,
        const DepthWindowPanelMapping& panel) {
        return point.x >= panel.x && point.y >= panel.y &&
               point.x < panel.x + panel.width && point.y < panel.y + panel.height;
    }

} // namespace lfs::vis::op

/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "screen/screen.hpp"

#include <cstdint>
#include <optional>

// Pointer gestures on the screen's chrome:
// dragging a divider resizes its neighbours; dragging from an area corner into
// the area splits it, out of the area across an edge joins the neighbour, and
// with the swap modifier held exchanges two areas. The state machine only
// reads geometry and returns commands; the caller applies them.
namespace lfs::vis::screen {

    enum class Corner : std::uint8_t { TopLeft,
                                       TopRight,
                                       BottomLeft,
                                       BottomRight };

    struct GestureMetrics {
        float corner_size = 12.0f;   // legs of the triangular corner zone
        float drag_threshold = 8.0f; // travel before a corner drag commits to a direction
        float divider_slop = 4.0f;   // grab tolerance on both sides of a divider
        float min_split_extent = 48.0f;
    };

    enum class GestureCursor : std::uint8_t {
        Default,
        ResizeColumns, // a vertical divider: drag left/right
        ResizeRows,    // a horizontal divider: drag up/down
        Crosshair,     // over a corner, or splitting
        Move,          // swapping
        NotAllowed,
    };

    struct GesturePreview {
        enum class Kind : std::uint8_t { None,
                                         Divider,
                                         Corner,
                                         Split,
                                         Join,
                                         Swap };
        Kind kind = Kind::None;
        AreaId source;
        AreaId target;
        Rect line;                    // Divider/Split: the divider being moved or proposed
        Rect first;                   // Split: the half that keeps the source
        Rect second;                  // Split: the new area
        Side direction = Side::Right; // Join: from the source into the target
        bool allowed = true;
        Corner corner = Corner::TopLeft; // Corner: the zone under the pointer
    };

    struct GestureCommand {
        enum class Kind : std::uint8_t { None,
                                         MoveDivider,
                                         Split,
                                         Join,
                                         Swap };
        Kind kind = Kind::None;
        // MoveDivider
        DividerGeometry divider;
        float position = 0.0f;
        // Split
        AreaId area;
        SplitAxis axis = SplitAxis::Columns;
        float fraction = 0.5f;
        bool new_first = false;
        // Join (kept grows over absorbed) and Swap (area <-> other)
        AreaId other;
    };

    struct CornerHit {
        AreaId area;
        Corner corner;
    };

    [[nodiscard]] std::optional<CornerHit> cornerAt(const LayoutGeometry& geometry, float x, float y,
                                                    float size);

    class AreaGestures {
    public:
        explicit AreaGestures(GestureMetrics metrics = {}) : metrics_(metrics) {}

        void setMetrics(const GestureMetrics& metrics) { metrics_ = metrics; }
        [[nodiscard]] const GestureMetrics& metrics() const { return metrics_; }

        // Cursor and zone feedback while no gesture runs.
        [[nodiscard]] GestureCursor hoverCursor(const LayoutGeometry& geometry, float x, float y) const;
        [[nodiscard]] GesturePreview hoverPreview(const LayoutGeometry& geometry, float x, float y) const;

        // Starts a gesture if the press hits a divider or a corner zone.
        // Returns true when the press was taken.
        bool press(const LayoutGeometry& geometry, float x, float y, bool swap_modifier);
        // Divider drags return a MoveDivider command to apply at once; the
        // other gestures only update the preview until release.
        [[nodiscard]] GestureCommand move(const LayoutGeometry& geometry, const Screen& screen, float x, float y);
        [[nodiscard]] GestureCommand release(const LayoutGeometry& geometry, const Screen& screen, float x,
                                             float y);
        void cancel();

        [[nodiscard]] bool active() const { return mode_ != Mode::Idle; }
        [[nodiscard]] const GesturePreview& preview() const { return preview_; }
        [[nodiscard]] GestureCursor cursor() const;

    private:
        enum class Mode : std::uint8_t { Idle,
                                         Divider,
                                         CornerPending,
                                         Split,
                                         Join,
                                         Swap };

        void updateCorner(const LayoutGeometry& geometry, const Screen& screen, float x, float y);
        void updateSplit(const LayoutGeometry& geometry, float x, float y);
        void updateJoin(const LayoutGeometry& geometry, const Screen& screen);
        void updateSwap(const LayoutGeometry& geometry, float x, float y);

        GestureMetrics metrics_;
        Mode mode_ = Mode::Idle;
        GesturePreview preview_;
        float press_x_ = 0.0f;
        float press_y_ = 0.0f;
        // Divider
        SplitId divider_split_;
        std::uint32_t divider_index_ = 0;
        float divider_start_ = 0.0f;
        // Corner
        AreaId source_;
        Corner corner_ = Corner::TopLeft;
        SplitAxis split_axis_ = SplitAxis::Columns;
        Side join_side_ = Side::Right;
    };

} // namespace lfs::vis::screen

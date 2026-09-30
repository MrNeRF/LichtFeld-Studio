/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <array>
#include <compare>
#include <cstdint>
#include <limits>
#include <nlohmann/json_fwd.hpp>
#include <optional>
#include <vector>

// The screen layout partitions the window's work area into non-overlapping
// areas, the way Blender's screen does. It is a pure value type: no UI, no
// GPU, no editor knowledge. Areas are referenced by id; what an area shows is
// owned by the Screen that holds this layout.
//
// Representation: an n-ary tree. A split node lays its children out along one
// axis with relative weights; leaves are areas. Invariants, restored by every
// mutation: splits have at least two children, a split never has a child split
// with the same axis (those are flattened into it), weights are positive and
// sum to one, and every area id appears exactly once.
namespace lfs::vis::screen {

    struct AreaId {
        std::uint32_t value = 0;

        [[nodiscard]] bool valid() const { return value != 0; }
        auto operator<=>(const AreaId&) const = default;
    };

    struct SplitId {
        std::uint32_t value = 0;

        [[nodiscard]] bool valid() const { return value != 0; }
        auto operator<=>(const SplitId&) const = default;
    };

    enum class SplitAxis : std::uint8_t {
        Columns, // children side by side, left to right; dividers are vertical
        Rows,    // children stacked top to bottom; dividers are horizontal
    };

    enum class Side : std::uint8_t { Left,
                                     Right,
                                     Top,
                                     Bottom };

    [[nodiscard]] SplitAxis axisAcross(Side side);

    struct Rect {
        float x = 0.0f;
        float y = 0.0f;
        float w = 0.0f;
        float h = 0.0f;

        [[nodiscard]] float right() const { return x + w; }
        [[nodiscard]] float bottom() const { return y + h; }
        [[nodiscard]] bool empty() const { return w <= 0.0f || h <= 0.0f; }
        // Half-open, so a point on a shared edge belongs to exactly one rect.
        [[nodiscard]] bool contains(float px, float py) const {
            return px >= x && px < x + w && py >= y && py < y + h;
        }
        bool operator==(const Rect&) const = default;
    };

    struct LayoutMetrics {
        float divider = 2.0f;     // gap between neighbouring areas
        float min_width = 48.0f;  // no area is solved narrower than this
        float min_height = 32.0f; // nor lower than this (a header fits)
    };

    struct AreaGeometry {
        AreaId area;
        Rect rect;
    };

    // The gap between child `index` and child `index + 1` of a split.
    struct DividerGeometry {
        SplitId split;
        std::uint32_t index = 0;
        SplitAxis axis = SplitAxis::Columns;
        Rect rect;                 // the gap itself
        Rect parent;               // the split's rect
        float min_position = 0.0f; // allowed range of the gap's leading edge
        float max_position = 0.0f;
        float leading_extent = 0.0f;  // solved extent of child `index`
        float trailing_extent = 0.0f; // solved extent of child `index + 1`
    };

    struct LayoutGeometry {
        Rect bounds;
        std::vector<AreaGeometry> areas;
        std::vector<DividerGeometry> dividers;
        AreaId maximized;

        [[nodiscard]] const AreaGeometry* find(AreaId area) const;
        [[nodiscard]] AreaId areaAt(float x, float y) const;
        // Nearest divider whose gap, widened by `slop` on both sides, contains
        // the point.
        [[nodiscard]] const DividerGeometry* dividerAt(float x, float y, float slop) const;
    };

    class ScreenLayout {
    public:
        struct Node {
            SplitId split; // valid for split nodes
            AreaId area;   // valid for leaves
            SplitAxis axis = SplitAxis::Columns;
            std::vector<Node> children;
            std::vector<float> weights;

            [[nodiscard]] bool isArea() const { return area.valid(); }
        };

        ScreenLayout() = default;
        explicit ScreenLayout(AreaId only);

        [[nodiscard]] bool empty() const { return !root_.has_value(); }
        [[nodiscard]] const Node* root() const { return root_ ? &*root_ : nullptr; }
        [[nodiscard]] std::vector<AreaId> areas() const;
        [[nodiscard]] bool contains(AreaId area) const;
        [[nodiscard]] std::size_t areaCount() const;

        // Splits `target`, giving `fraction` of its extent along `axis` to the
        // new area, which goes after the target (right/below) unless
        // `new_first`. Fails if the ids are invalid, `added` already exists or
        // the fraction is not in (0, 1).
        bool split(AreaId target, AreaId added, SplitAxis axis, float fraction, bool new_first = false);

        // Adds an area along one edge of the whole screen, spanning its full
        // height (Left/Right) or width (Top/Bottom), taking `fraction` of it.
        bool insertAtEdge(AreaId added, Side side, float fraction);

        // Removes an area; its space goes to the neighbour before it (or
        // after it, for the first child). The last area cannot be removed.
        bool remove(AreaId area);

        // `absorbed` is removed and `kept` grows over its space. Only allowed
        // when the two are neighbouring areas sharing their full edge, which in
        // this tree means adjacent leaves of the same split.
        [[nodiscard]] bool canJoin(AreaId kept, AreaId absorbed) const;
        bool join(AreaId kept, AreaId absorbed);

        // The area sharing `area`'s full edge on `side`, if any. This is the
        // area a Blender-style corner drag across that edge would join.
        [[nodiscard]] AreaId joinableNeighbour(AreaId area, Side side) const;

        // The four areas of a 2x2 block containing `area`: a split of two
        // splits of the other axis, each holding two areas. Order: first
        // split's children, then the second's.
        [[nodiscard]] std::optional<std::array<AreaId, 4>> quadAround(AreaId area) const;

        // Exchanges the positions of two areas.
        bool swap(AreaId a, AreaId b);

        // Moves a divider so that its leading edge lands at `position` (in the
        // same units as the geometry), clamped so both neighbours keep their
        // minimum size. Needs the geometry the drag is based on.
        bool moveDivider(const DividerGeometry& divider, float position);

        // Sets the relative weights of a split's children directly.
        bool setWeights(SplitId split, const std::vector<float>& weights);

        [[nodiscard]] const Node* findSplit(SplitId split) const;

        // Solves the layout into pixel rects inside `bounds`. Edges are
        // snapped to whole units so neighbouring areas never overlap or leave
        // seams. With `maximized` set, that area alone covers the bounds.
        [[nodiscard]] LayoutGeometry solve(const Rect& bounds, const LayoutMetrics& metrics,
                                           AreaId maximized = {}) const;

        [[nodiscard]] nlohmann::json toJson() const;
        // Rejects malformed input as a whole; the returned layout satisfies
        // every invariant.
        [[nodiscard]] static std::optional<ScreenLayout> fromJson(const nlohmann::json& json);

        [[nodiscard]] std::uint32_t nextSplitId() const { return next_split_id_; }

        bool operator==(const ScreenLayout& other) const;

    private:
        void normalize();
        SplitId allocateSplit() {
            if (next_split_id_ == 0 || next_split_id_ == std::numeric_limits<std::uint32_t>::max())
                return {};
            return SplitId{next_split_id_++};
        }

        std::optional<Node> root_;
        std::uint32_t next_split_id_ = 1;
    };

} // namespace lfs::vis::screen

/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "screen/area_gestures.hpp"

#include <algorithm>
#include <cmath>

namespace lfs::vis::screen {

    namespace {

        bool leftCorner(const Corner c) { return c == Corner::TopLeft || c == Corner::BottomLeft; }
        bool topCorner(const Corner c) { return c == Corner::TopLeft || c == Corner::TopRight; }

        const DividerGeometry* findDivider(const LayoutGeometry& geometry, const SplitId split,
                                           const std::uint32_t index) {
            for (const auto& d : geometry.dividers) {
                if (d.split == split && d.index == index)
                    return &d;
            }
            return nullptr;
        }

        Side sideTowards(const Rect& from, const Rect& to) {
            const float dx = (to.x + to.w * 0.5f) - (from.x + from.w * 0.5f);
            const float dy = (to.y + to.h * 0.5f) - (from.y + from.h * 0.5f);
            if (std::abs(dx) >= std::abs(dy))
                return dx < 0.0f ? Side::Left : Side::Right;
            return dy < 0.0f ? Side::Top : Side::Bottom;
        }

    } // namespace

    std::optional<CornerHit> cornerAt(const LayoutGeometry& geometry, const float x, const float y,
                                      const float size) {
        if (geometry.maximized.valid())
            return std::nullopt;
        for (const auto& a : geometry.areas) {
            const Rect& r = a.rect;
            if (!r.contains(x, y) || r.w < size * 3.0f || r.h < size * 3.0f)
                continue;
            const float left = x - r.x;
            const float right = r.right() - x;
            const float top = y - r.y;
            const float bottom = r.bottom() - y;
            if (left + top < size)
                return CornerHit{a.area, Corner::TopLeft};
            if (right + top < size)
                return CornerHit{a.area, Corner::TopRight};
            if (left + bottom < size)
                return CornerHit{a.area, Corner::BottomLeft};
            if (right + bottom < size)
                return CornerHit{a.area, Corner::BottomRight};
        }
        return std::nullopt;
    }

    GestureCursor AreaGestures::hoverCursor(const LayoutGeometry& geometry, const float x, const float y) const {
        if (geometry.maximized.valid())
            return GestureCursor::Default;
        // Corners win over the divider tolerance next to them.
        if (cornerAt(geometry, x, y, metrics_.corner_size))
            return GestureCursor::Crosshair;
        if (const auto* d = geometry.dividerAt(x, y, metrics_.divider_slop))
            return d->axis == SplitAxis::Columns ? GestureCursor::ResizeColumns : GestureCursor::ResizeRows;
        return GestureCursor::Default;
    }

    GesturePreview AreaGestures::hoverPreview(const LayoutGeometry& geometry, const float x, const float y) const {
        GesturePreview hover;
        if (geometry.maximized.valid())
            return hover;
        if (const auto hit = cornerAt(geometry, x, y, metrics_.corner_size)) {
            hover.kind = GesturePreview::Kind::Corner;
            hover.source = hit->area;
            hover.corner = hit->corner;
            if (const auto* a = geometry.find(hit->area))
                hover.first = a->rect;
            return hover;
        }
        if (const auto* d = geometry.dividerAt(x, y, metrics_.divider_slop)) {
            hover.kind = GesturePreview::Kind::Divider;
            hover.line = d->rect;
        }
        return hover;
    }

    bool AreaGestures::press(const LayoutGeometry& geometry, const float x, const float y, const bool swap_modifier) {
        cancel();
        if (geometry.maximized.valid())
            return false;
        press_x_ = x;
        press_y_ = y;
        const auto hit = cornerAt(geometry, x, y, metrics_.corner_size);
        if (!hit) {
            const auto* d = geometry.dividerAt(x, y, metrics_.divider_slop);
            if (!d)
                return false;
            mode_ = Mode::Divider;
            divider_split_ = d->split;
            divider_index_ = d->index;
            divider_start_ = d->axis == SplitAxis::Columns ? d->rect.x : d->rect.y;
            preview_.kind = GesturePreview::Kind::Divider;
            preview_.line = d->rect;
            return true;
        }
        source_ = hit->area;
        corner_ = hit->corner;
        preview_.source = source_;
        preview_.corner = corner_;
        if (swap_modifier) {
            mode_ = Mode::Swap;
            preview_.kind = GesturePreview::Kind::Swap;
            preview_.allowed = false;
        } else {
            mode_ = Mode::CornerPending;
            preview_.kind = GesturePreview::Kind::Corner;
        }
        if (const auto* a = geometry.find(source_))
            preview_.first = a->rect;
        return true;
    }

    GestureCommand AreaGestures::move(const LayoutGeometry& geometry, const Screen& screen, const float x,
                                      const float y) {
        GestureCommand command;
        switch (mode_) {
        case Mode::Idle:
            break;
        case Mode::Divider: {
            const DividerGeometry* current = findDivider(geometry, divider_split_, divider_index_);
            if (!current) {
                cancel();
                break;
            }
            const bool columns = current->axis == SplitAxis::Columns;
            command.kind = GestureCommand::Kind::MoveDivider;
            command.divider = *current;
            command.position = divider_start_ + (columns ? x - press_x_ : y - press_y_);
            preview_.line = current->rect;
            break;
        }
        case Mode::CornerPending:
            updateCorner(geometry, screen, x, y);
            break;
        case Mode::Split:
            updateSplit(geometry, x, y);
            break;
        case Mode::Join:
            preview_.target = geometry.areaAt(x, y);
            updateJoin(geometry, screen);
            break;
        case Mode::Swap:
            updateSwap(geometry, x, y);
            break;
        }
        return command;
    }

    GestureCommand AreaGestures::release(const LayoutGeometry& geometry, const Screen& screen, const float x,
                                         const float y) {
        GestureCommand command;
        if (mode_ != Mode::Idle && mode_ != Mode::Divider)
            (void)move(geometry, screen, x, y);
        if (mode_ == Mode::Split && preview_.allowed) {
            const auto* source = geometry.find(source_);
            if (source) {
                const Rect& r = source->rect;
                const bool columns = split_axis_ == SplitAxis::Columns;
                const float extent = columns ? r.w : r.h;
                const float added = columns ? preview_.second.w : preview_.second.h;
                command.kind = GestureCommand::Kind::Split;
                command.area = source_;
                command.axis = split_axis_;
                command.fraction = extent > 0.0f ? std::clamp(added / extent, 0.01f, 0.99f) : 0.5f;
                command.new_first = columns ? leftCorner(corner_) : topCorner(corner_);
            }
        } else if (mode_ == Mode::Join && preview_.allowed) {
            command.kind = GestureCommand::Kind::Join;
            command.area = source_;
            command.other = preview_.target;
        } else if (mode_ == Mode::Swap && preview_.allowed) {
            command.kind = GestureCommand::Kind::Swap;
            command.area = source_;
            command.other = preview_.target;
        }
        cancel();
        return command;
    }

    void AreaGestures::cancel() {
        mode_ = Mode::Idle;
        preview_ = {};
        source_ = {};
    }

    GestureCursor AreaGestures::cursor() const {
        switch (mode_) {
        case Mode::Idle: return GestureCursor::Default;
        case Mode::Divider:
            return preview_.line.w <= preview_.line.h ? GestureCursor::ResizeColumns : GestureCursor::ResizeRows;
        case Mode::CornerPending: return GestureCursor::Crosshair;
        case Mode::Split:
            if (!preview_.allowed)
                return GestureCursor::NotAllowed;
            return split_axis_ == SplitAxis::Columns ? GestureCursor::ResizeColumns : GestureCursor::ResizeRows;
        case Mode::Join:
        case Mode::Swap:
            return preview_.allowed ? GestureCursor::Move : GestureCursor::NotAllowed;
        }
        return GestureCursor::Default;
    }

    void AreaGestures::updateCorner(const LayoutGeometry& geometry, const Screen& screen, const float x,
                                    const float y) {
        const float dx = x - press_x_;
        const float dy = y - press_y_;
        if (std::hypot(dx, dy) < metrics_.drag_threshold)
            return;
        const bool horizontal = std::abs(dx) >= std::abs(dy);
        const float inward_x = leftCorner(corner_) ? 1.0f : -1.0f;
        const float inward_y = topCorner(corner_) ? 1.0f : -1.0f;
        const bool inward = horizontal ? dx * inward_x > 0.0f : dy * inward_y > 0.0f;
        if (inward) {
            mode_ = Mode::Split;
            split_axis_ = horizontal ? SplitAxis::Columns : SplitAxis::Rows;
            preview_.kind = GesturePreview::Kind::Split;
            updateSplit(geometry, x, y);
        } else {
            mode_ = Mode::Join;
            join_side_ = horizontal ? (dx < 0.0f ? Side::Left : Side::Right) : (dy < 0.0f ? Side::Top : Side::Bottom);
            preview_.kind = GesturePreview::Kind::Join;
            preview_.target = screen.layout().joinableNeighbour(source_, join_side_);
            if (!preview_.target.valid())
                preview_.target = geometry.areaAt(x, y);
            updateJoin(geometry, screen);
        }
    }

    void AreaGestures::updateSplit(const LayoutGeometry& geometry, const float x, const float y) {
        const auto* source = geometry.find(source_);
        if (!source) {
            cancel();
            return;
        }
        const Rect& r = source->rect;
        const bool columns = split_axis_ == SplitAxis::Columns;
        const float lead = columns ? r.x : r.y;
        const float extent = columns ? r.w : r.h;
        const float min_extent = metrics_.min_split_extent;
        preview_.allowed = extent >= min_extent * 2.0f;
        const float pointer = columns ? x : y;
        const float position = preview_.allowed
                                   ? std::round(std::clamp(pointer, lead + min_extent, lead + extent - min_extent))
                                   : lead + extent * 0.5f;
        const bool new_first = columns ? leftCorner(corner_) : topCorner(corner_);
        const Rect before = columns ? Rect{r.x, r.y, position - r.x, r.h} : Rect{r.x, r.y, r.w, position - r.y};
        const Rect after = columns ? Rect{position, r.y, r.right() - position, r.h}
                                   : Rect{r.x, position, r.w, r.bottom() - position};
        preview_.first = new_first ? after : before;
        preview_.second = new_first ? before : after;
        preview_.line = columns ? Rect{position - 1.0f, r.y, 2.0f, r.h} : Rect{r.x, position - 1.0f, r.w, 2.0f};
    }

    void AreaGestures::updateJoin(const LayoutGeometry& geometry, const Screen& screen) {
        const AreaId target = preview_.target;
        const auto* source = geometry.find(source_);
        const auto* target_geometry = geometry.find(target);
        preview_.allowed = source && target_geometry && target != source_ && screen.canJoin(source_, target);
        if (source && target_geometry && target != source_) {
            preview_.direction = sideTowards(source->rect, target_geometry->rect);
            preview_.second = target_geometry->rect;
        } else {
            preview_.second = {};
        }
        if (source)
            preview_.first = source->rect;
    }

    void AreaGestures::updateSwap(const LayoutGeometry& geometry, const float x, const float y) {
        preview_.target = geometry.areaAt(x, y);
        const auto* target = geometry.find(preview_.target);
        preview_.allowed = target && preview_.target != source_;
        preview_.second = target ? target->rect : Rect{};
    }

} // namespace lfs::vis::screen

/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "corner_split_interaction.hpp"

#include <algorithm>
#include <cmath>

namespace lfs::vis {
    namespace {

        [[nodiscard]] bool finitePositive(const float value) noexcept {
            return std::isfinite(value) && value > 0.0f;
        }

        [[nodiscard]] bool containsCornerHandle(const ViewRect& rect,
                                                const glm::vec2 pointer,
                                                const float radius) noexcept {
            if (rect.empty() || !finitePositive(radius))
                return false;
            // The handle lives inside the pane. Keeping the test half-open
            // avoids claiming a neighbouring pane's pixels at a divider.
            return pointer.x >= static_cast<float>(rect.right()) - radius &&
                   pointer.x < static_cast<float>(rect.right()) &&
                   pointer.y >= static_cast<float>(rect.bottom()) - radius &&
                   pointer.y < static_cast<float>(rect.bottom()) &&
                   pointer.x >= static_cast<float>(rect.x) &&
                   pointer.y >= static_cast<float>(rect.y);
        }

        [[nodiscard]] std::optional<CornerSplitPreview> makePreview(
            const ViewId source, const ViewRect rect, const SplitAxis axis, const float ratio,
            const int min_pane_pixels, const int divider_pixels) noexcept {
            if (rect.empty() || !std::isfinite(ratio))
                return std::nullopt;

            const int extent = axis == SplitAxis::Horizontal ? rect.width : rect.height;
            const int divider = std::clamp(divider_pixels, 0, std::max(extent - 1, 0));
            const int available = extent - divider;
            const int min_pane = std::max(min_pane_pixels, 0);
            if (available <= 0 || (min_pane > 0 && available < min_pane * 2))
                return std::nullopt;

            const float minimum = min_pane > 0
                                      ? static_cast<float>(min_pane) / static_cast<float>(available)
                                      : 0.0f;
            const float maximum = min_pane > 0 ? 1.0f - minimum : 1.0f;
            const float clamped_ratio = std::clamp(ratio, minimum, maximum);
            const int first_extent = std::clamp(
                static_cast<int>(std::lround(static_cast<double>(available) * clamped_ratio)),
                min_pane, available - min_pane);
            const float actual_ratio = static_cast<float>(first_extent) /
                                       static_cast<float>(available);

            CornerSplitPreview result{.source = source,
                                      .axis = axis,
                                      .ratio = actual_ratio};
            if (axis == SplitAxis::Horizontal) {
                result.first = {rect.x, rect.y, first_extent, rect.height};
                result.splitter = {rect.x + first_extent, rect.y,
                                   divider, rect.height};
                result.second = {rect.x + first_extent + divider, rect.y,
                                 available - first_extent, rect.height};
            } else {
                result.first = {rect.x, rect.y, rect.width, first_extent};
                result.splitter = {rect.x, rect.y + first_extent,
                                   rect.width, divider};
                result.second = {rect.x, rect.y + first_extent + divider,
                                 rect.width, available - first_extent};
            }
            return result;
        }

    } // namespace

    CornerSplitInteraction::CornerSplitInteraction(
        const CornerSplitInteractionConfig config) noexcept
        : config_(config) {
        config_.handle_radius_pixels =
            finitePositive(config_.handle_radius_pixels) ? config_.handle_radius_pixels : kWorkspaceCornerHandlePixels;
        config_.activation_threshold_pixels =
            finitePositive(config_.activation_threshold_pixels)
                ? config_.activation_threshold_pixels
                : 10.0f;
        config_.min_pane_pixels = std::max(config_.min_pane_pixels, 0);
        config_.divider_pixels = std::max(config_.divider_pixels, 0);
    }

    std::optional<ViewId> CornerSplitInteraction::handleHit(
        const WorkspaceFrameSnapshot& snapshot, const glm::vec2 pointer,
        const float radius_pixels) noexcept {
        if (!finitePositive(radius_pixels))
            return std::nullopt;
        // Pane order is layout order. A handle cannot overlap another visible
        // pane except in a malformed snapshot; the first matching pane keeps
        // the result deterministic.
        if (!snapshot.areas.empty()) {
            for (const auto& area : snapshot.areas) {
                if (isValidViewId(area.id) && containsCornerHandle(area.rect, pointer, radius_pixels))
                    return area.id;
            }
        } else {
            for (const auto& pane : snapshot.panes) {
                if (isValidViewId(pane.id) && containsCornerHandle(pane.rect, pointer, radius_pixels))
                    return pane.id;
            }
        }
        return std::nullopt;
    }

    std::optional<ViewId> CornerSplitInteraction::handleHit(
        const WorkspaceFrameSnapshot& snapshot, const glm::vec2 pointer) const noexcept {
        return handleHit(snapshot, pointer, config_.handle_radius_pixels);
    }

    bool CornerSplitInteraction::begin(const WorkspaceFrameSnapshot& snapshot,
                                       const glm::vec2 pointer) noexcept {
        const auto source = handleHit(snapshot, pointer);
        if (!source)
            return false;
        state_ = State{.source = *source, .start = pointer};
        return true;
    }

    std::optional<CornerSplitPreview> CornerSplitInteraction::update(
        const WorkspaceFrameSnapshot& snapshot, const glm::vec2 pointer) noexcept {
        if (!state_)
            return std::nullopt;
        const auto area = findVisibleArea(snapshot, state_->source);
        if (!area) {
            state_.reset();
            return std::nullopt;
        }

        if (!state_->preview) {
            const glm::vec2 inward = state_->start - pointer;
            const float threshold = config_.activation_threshold_pixels;
            if (std::max(inward.x, inward.y) < threshold)
                return std::nullopt;
            const SplitAxis axis = inward.x >= inward.y
                                       ? SplitAxis::Horizontal
                                       : SplitAxis::Vertical;
            const float coordinate = axis == SplitAxis::Horizontal
                                         ? pointer.x - static_cast<float>(area->rect.x)
                                         : pointer.y - static_cast<float>(area->rect.y);
            const float extent = static_cast<float>(axis == SplitAxis::Horizontal
                                                        ? area->rect.width
                                                        : area->rect.height);
            if (extent <= 0.0f)
                return std::nullopt;
            state_->preview = makePreview(
                state_->source, area->rect, axis, coordinate / extent,
                config_.min_pane_pixels, config_.divider_pixels);
        } else {
            const auto axis = state_->preview->axis;
            const float coordinate = axis == SplitAxis::Horizontal
                                         ? pointer.x - static_cast<float>(area->rect.x)
                                         : pointer.y - static_cast<float>(area->rect.y);
            const float extent = static_cast<float>(axis == SplitAxis::Horizontal
                                                        ? area->rect.width
                                                        : area->rect.height);
            if (extent > 0.0f)
                state_->preview = makePreview(state_->source, area->rect, axis, coordinate / extent,
                                              config_.min_pane_pixels, config_.divider_pixels);
        }
        return state_->preview;
    }

    std::optional<CornerSplitRequest> CornerSplitInteraction::release(
        const WorkspaceFrameSnapshot& snapshot, const glm::vec2 pointer) noexcept {
        const auto preview = update(snapshot, pointer);
        if (!preview) {
            state_.reset();
            return std::nullopt;
        }
        const auto request = preview->request();
        state_.reset();
        return request;
    }

    bool CornerSplitInteraction::cancel() noexcept {
        if (!state_)
            return false;
        state_.reset();
        return true;
    }

    bool CornerSplitInteraction::reconcile(
        const WorkspaceFrameSnapshot& snapshot) noexcept {
        if (!state_ || findVisibleArea(snapshot, state_->source))
            return false;
        state_.reset();
        return true;
    }

    std::optional<ViewId> CornerSplitInteraction::source() const noexcept {
        if (!state_)
            return std::nullopt;
        return state_->source;
    }

    std::optional<CornerSplitPreview> CornerSplitInteraction::preview() const noexcept {
        if (!state_)
            return std::nullopt;
        return state_->preview;
    }

    std::optional<CornerSplitInteraction::VisibleArea> CornerSplitInteraction::findVisibleArea(
        const WorkspaceFrameSnapshot& snapshot, const ViewId view) noexcept {
        for (const auto& area : snapshot.areas) {
            if (area.id == view && isValidViewId(area.id) && !area.rect.empty())
                return VisibleArea{area.id, area.rect};
        }
        // Keep the helper compatible with callers/tests that construct the
        // original viewport-only snapshot directly.
        for (const auto& pane : snapshot.panes) {
            if (pane.id == view && isValidViewId(pane.id) && !pane.rect.empty())
                return VisibleArea{pane.id, pane.rect};
        }
        return std::nullopt;
    }

} // namespace lfs::vis

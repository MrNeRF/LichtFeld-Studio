/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "pane_interaction.hpp"

namespace lfs::vis {
    namespace {

        [[nodiscard]] bool containsHalfOpen(const ViewRect& rect,
                                            const glm::vec2 pointer) noexcept {
            return !rect.empty() && pointer.x >= static_cast<float>(rect.x) &&
                   pointer.y >= static_cast<float>(rect.y) &&
                   pointer.x < static_cast<float>(rect.right()) &&
                   pointer.y < static_cast<float>(rect.bottom());
        }

    } // namespace

    std::optional<PaneHit> PaneInteraction::hitTest(
        const WorkspaceFrameSnapshot& snapshot, const glm::vec2 pointer) noexcept {
        if (!containsHalfOpen(snapshot.outer, pointer)) {
            return std::nullopt;
        }
        // Splitters have priority so a malformed/overlapping pane snapshot
        // still treats a divider as a miss.
        for (const SplitterRect& splitter : snapshot.splitters) {
            if (containsHalfOpen(splitter.rect, pointer)) {
                return std::nullopt;
            }
        }
        for (const PaneSnapshot& pane : snapshot.panes) {
            if (containsHalfOpen(pane.rect, pointer)) {
                return makeHit(pane, pointer);
            }
        }
        return std::nullopt;
    }

    std::optional<PaneHit> PaneInteraction::beginCapture(
        const WorkspaceFrameSnapshot& snapshot, const glm::vec2 pointer,
        const PaneGestureKind kind) {
        const auto hit = hitTest(snapshot, pointer);
        if (!hit) {
            return std::nullopt;
        }
        capture_ = PaneCapture{hit->view, kind};
        return hit;
    }

    std::optional<PaneHit> PaneInteraction::updateCapture(
        const WorkspaceFrameSnapshot& snapshot, const glm::vec2 pointer) {
        if (!capture_) {
            return std::nullopt;
        }
        const PaneSnapshot* pane = findPane(snapshot, capture_->view);
        if (!pane) {
            capture_.reset();
            return std::nullopt;
        }
        return makeHit(*pane, pointer);
    }

    bool PaneInteraction::releaseCapture() noexcept {
        if (!capture_) {
            return false;
        }
        capture_.reset();
        return true;
    }

    bool PaneInteraction::cancelCapture() noexcept {
        return releaseCapture();
    }

    bool PaneInteraction::reconcile(const WorkspaceFrameSnapshot& snapshot) noexcept {
        if (!capture_ || findPane(snapshot, capture_->view)) {
            return false;
        }
        capture_.reset();
        return true;
    }

    std::optional<ViewId> PaneInteraction::keyboardTarget(
        const WorkspaceFrameSnapshot& snapshot) noexcept {
        if (!snapshot.focused || !isValidViewId(*snapshot.focused)) {
            return std::nullopt;
        }
        return snapshot.focused;
    }

    const PaneSnapshot* PaneInteraction::findPane(
        const WorkspaceFrameSnapshot& snapshot, const ViewId view) noexcept {
        if (!isValidViewId(view)) {
            return nullptr;
        }
        for (const PaneSnapshot& pane : snapshot.panes) {
            if (pane.id == view && !pane.rect.empty()) {
                return &pane;
            }
        }
        return nullptr;
    }

    PaneHit PaneInteraction::makeHit(const PaneSnapshot& pane,
                                     const glm::vec2 pointer) noexcept {
        PaneHit hit;
        hit.view = pane.id;
        hit.rect = pane.rect;
        hit.pointer = pointer;
        hit.local = pointer - glm::vec2(static_cast<float>(pane.rect.x),
                                        static_cast<float>(pane.rect.y));
        hit.camera.projection = pane.projection;
        hit.camera.depth = pane.depth;
        hit.camera.grid_plane = pane.grid_plane;
        hit.camera.rotation = pane.rotation;
        hit.camera.translation = pane.translation;
        hit.camera.pivot = pane.pivot;
        hit.camera.window_size = pane.window_size;
        hit.camera.framebuffer_size = pane.framebuffer_size;
        hit.camera.state_generation = pane.state_generation;
        hit.focused = pane.focused;
        hit.maximized = pane.maximized;
        return hit;
    }

} // namespace lfs::vis

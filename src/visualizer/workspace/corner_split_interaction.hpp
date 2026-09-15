/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "workspace/viewport_workspace.hpp"

#include <glm/glm.hpp>

#include <cstdint>
#include <optional>

namespace lfs::vis {

    inline constexpr float kWorkspaceCornerHandlePixels = 24.0f;

    struct CornerSplitInteractionConfig {
        float handle_radius_pixels = kWorkspaceCornerHandlePixels;
        float activation_threshold_pixels = 10.0f;
        int min_pane_pixels = kDefaultMinPanePixels;
        int divider_pixels = kDefaultDividerPixels;
    };

    struct CornerSplitRequest {
        ViewId source = kInvalidViewId;
        SplitAxis axis = SplitAxis::Horizontal;
        float ratio = 0.5f;

        [[nodiscard]] friend bool operator==(const CornerSplitRequest&,
                                             const CornerSplitRequest&) noexcept = default;
    };

    // Pure presentation data for the uncommitted split. The first rectangle
    // remains the source view; the second rectangle is the view to be created
    // when the caller commits CornerSplitRequest through ViewportWorkspace.
    struct CornerSplitPreview {
        ViewId source = kInvalidViewId;
        SplitAxis axis = SplitAxis::Horizontal;
        float ratio = 0.5f;
        ViewRect first{};
        ViewRect splitter{};
        ViewRect second{};

        [[nodiscard]] CornerSplitRequest request() const noexcept {
            return {source, axis, ratio};
        }
    };

    // Capture the source pane until release. The GUI commits the returned split
    // request; pointer movement does not mutate the layout.
    class LFS_VIS_API CornerSplitInteraction {
    public:
        explicit CornerSplitInteraction(CornerSplitInteractionConfig config = {}) noexcept;

        [[nodiscard]] static std::optional<ViewId> handleHit(
            const WorkspaceFrameSnapshot& snapshot, glm::vec2 pointer,
            float radius_pixels) noexcept;

        [[nodiscard]] std::optional<ViewId> handleHit(
            const WorkspaceFrameSnapshot& snapshot, glm::vec2 pointer) const noexcept;

        // Starts capture on a visible pane's lower-right handle. No layout
        // mutation occurs and no split is active until update crosses the
        // activation threshold.
        bool begin(const WorkspaceFrameSnapshot& snapshot, glm::vec2 pointer) noexcept;

        // Updates the preview. The dominant positive inward displacement
        // chooses the axis once: Horizontal means x/width, Vertical means
        // y/height. The axis remains locked for the rest of the gesture.
        [[nodiscard]] std::optional<CornerSplitPreview> update(
            const WorkspaceFrameSnapshot& snapshot, glm::vec2 pointer) noexcept;

        // Releases and returns a commit request only when a valid preview was
        // established. Invalid, too-small, closed, or hidden sources cancel.
        [[nodiscard]] std::optional<CornerSplitRequest> release(
            const WorkspaceFrameSnapshot& snapshot, glm::vec2 pointer) noexcept;

        // Clears the gesture without producing a request (including Escape).
        bool cancel() noexcept;

        // Call after each new frame snapshot. A closed or hidden source
        // cancels capture; retained live IDs are insufficient because hidden
        // panes are absent from snapshot.panes.
        bool reconcile(const WorkspaceFrameSnapshot& snapshot) noexcept;

        [[nodiscard]] bool active() const noexcept { return state_.has_value(); }
        [[nodiscard]] std::optional<ViewId> source() const noexcept;
        [[nodiscard]] std::optional<CornerSplitPreview> preview() const noexcept;
        [[nodiscard]] const CornerSplitInteractionConfig& config() const noexcept {
            return config_;
        }

    private:
        struct VisibleArea {
            ViewId id = kInvalidViewId;
            ViewRect rect{};
        };

        struct State {
            ViewId source = kInvalidViewId;
            glm::vec2 start{0.0f};
            std::optional<CornerSplitPreview> preview;
        };

        [[nodiscard]] static std::optional<VisibleArea> findVisibleArea(
            const WorkspaceFrameSnapshot& snapshot, ViewId view) noexcept;

        CornerSplitInteractionConfig config_{};
        std::optional<State> state_;
    };

} // namespace lfs::vis

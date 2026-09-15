/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "workspace/viewport_workspace.hpp"

#include <glm/glm.hpp>

#include <cstdint>
#include <optional>

namespace lfs::vis {

    enum class PaneGestureKind : std::uint8_t {
        Camera,
        Selection,
        Pan,
        Orbit,
    };

    // Pointer-free camera data copied from the frame snapshot. This keeps a
    // hit usable after the workspace or registry changes and gives callers the
    // exact camera state that belonged to the hit-tested frame.
    struct PaneCameraSnapshot {
        ViewProjectionState projection{};
        DepthWindowState depth{};
        int grid_plane = 1;
        glm::mat3 rotation{1.0f};
        glm::vec3 translation{0.0f};
        glm::vec3 pivot{0.0f};
        glm::ivec2 window_size{0, 0};
        glm::ivec2 framebuffer_size{0, 0};
        std::uint64_t state_generation = 0;
    };

    struct PaneHit {
        ViewId view = kInvalidViewId;
        ViewRect rect{};
        glm::vec2 pointer{0.0f};
        glm::vec2 local{0.0f};
        PaneCameraSnapshot camera{};
        bool focused = false;
        bool maximized = false;
    };

    struct PaneCapture {
        ViewId view = kInvalidViewId;
        PaneGestureKind kind = PaneGestureKind::Camera;
    };

    // Stateless hit testing plus explicit pointer capture. The only mutable
    // state is the captured ViewId and gesture kind; snapshots remain owned by
    // the caller and are never cached or overwritten here.
    class LFS_VIS_API PaneInteraction {
    public:
        [[nodiscard]] static std::optional<PaneHit> hitTest(
            const WorkspaceFrameSnapshot& snapshot, glm::vec2 pointer) noexcept;

        [[nodiscard]] std::optional<PaneHit> beginCapture(
            const WorkspaceFrameSnapshot& snapshot, glm::vec2 pointer,
            PaneGestureKind kind);

        // A captured gesture stays with its initiating ViewId even when the
        // pointer crosses other panes or a divider. Local coordinates are
        // recomputed against that ViewId's current rectangle.
        [[nodiscard]] std::optional<PaneHit> updateCapture(
            const WorkspaceFrameSnapshot& snapshot, glm::vec2 pointer);

        // Explicit lifecycle operations. releaseCapture() and cancelCapture()
        // both clear the capture and report whether one was active.
        bool releaseCapture() noexcept;
        bool cancelCapture() noexcept;

        // Cancel safely when the captured view is no longer visible in the
        // current immutable frame snapshot (closed, hidden, or maximized out).
        bool reconcile(const WorkspaceFrameSnapshot& snapshot) noexcept;

        [[nodiscard]] std::optional<PaneCapture> activeCapture() const noexcept {
            return capture_;
        }
        [[nodiscard]] bool hasCapture() const noexcept { return capture_.has_value(); }

        // Keyboard focus is independent from the pointer hit. It routes to
        // the snapshot's focused ViewId without mutating global workspace
        // focus or consulting a mutable layout hit-test cache.
        [[nodiscard]] static std::optional<ViewId> keyboardTarget(
            const WorkspaceFrameSnapshot& snapshot) noexcept;

    private:
        [[nodiscard]] static const PaneSnapshot* findPane(
            const WorkspaceFrameSnapshot& snapshot, ViewId view) noexcept;
        [[nodiscard]] static PaneHit makeHit(const PaneSnapshot& pane,
                                             glm::vec2 pointer) noexcept;

        std::optional<PaneCapture> capture_;
    };

} // namespace lfs::vis

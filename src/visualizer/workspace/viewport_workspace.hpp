/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/error.hpp"
#include "core/export.hpp"
#include "workspace/view_id.hpp"
#include "workspace/view_registry.hpp"
#include "workspace/view_types.hpp"
#include "workspace/workspace_layout.hpp"

#include <glm/glm.hpp>
#include <optional>
#include <string>
#include <vector>

namespace lfs::vis {

    struct PaneSnapshot {
        ViewId id = kInvalidViewId;
        LayoutNodeId leaf = kInvalidLayoutNodeId;
        ViewRect rect{};
        ViewProjectionState projection{};
        DepthWindowState depth{};
        int grid_plane = 1;
        glm::mat3 rotation{1.0f};
        glm::vec3 translation{0.0f};
        glm::vec3 pivot{0.0f};
        glm::ivec2 window_size{0, 0};
        glm::ivec2 framebuffer_size{0, 0};
        bool focused = false;
        bool maximized = false;
        std::uint64_t state_generation = 0;
        std::optional<int> camera_uid;
        std::uint64_t camera_binding_generation = 0;
    };

    // Full UI area boundary. Viewport panes below intentionally expose only
    // the content rectangle so existing renderer/input consumers remain
    // unchanged when another editor occupies an area.
    struct AreaSnapshot {
        ViewId id = kInvalidViewId;
        LayoutNodeId leaf = kInvalidLayoutNodeId;
        ViewRect rect{};
        ViewRect content_rect{};
        std::string editor_id = "viewport";
        bool focused = false;
        bool maximized = false;
    };

    // Copied, pointer-free frame boundary for GUI, input, and render requests.
    struct WorkspaceFrameSnapshot {
        ViewRect outer{};
        std::vector<AreaSnapshot> areas; // all visible UI areas, including non-viewports
        // All live viewport identities, including viewport areas hidden by
        // maximize or another editor. GPU owners use this list for lifecycle
        // retirement while preserving cached outputs for restoration.
        std::vector<ViewId> live_viewport_ids;
        std::vector<PaneSnapshot> panes;
        std::vector<SplitterRect> splitters;
        std::vector<ViewId> live_ids;
        std::optional<ViewId> focused;
        // The last viewport interacted with is independent from UI-area
        // focus. It remains usable when focus moves to a panel or when that
        // viewport is temporarily hidden by maximize.
        std::optional<ViewId> active_viewport;
        std::optional<ViewId> maximized;
        ViewId primary = kInvalidViewId;
        std::uint64_t layout_generation = 0;
        std::uint64_t registry_generation = 0;
    };

    struct WorkspacePersistentState {
        std::uint32_t format_version = 1;
        WorkspaceLayoutState layout;
        std::vector<ViewPersistentState> views;
        // Optional in serialized state for compatibility with workspaces
        // written before the last-focused viewport was persisted.
        std::optional<ViewId> active_viewport;
        ViewId next_view_id = 1;
        std::uint64_t registry_generation = 0;
    };

    [[nodiscard]] LFS_VIS_API lfs::Status validateWorkspaceState(const WorkspacePersistentState& state);

    class LFS_VIS_API ViewportWorkspace {
    public:
        explicit ViewportWorkspace(glm::ivec2 initial_size = {1280, 720});

        ViewportWorkspace(const ViewportWorkspace&) = delete;
        ViewportWorkspace& operator=(const ViewportWorkspace&) = delete;
        ViewportWorkspace(ViewportWorkspace&&) noexcept = default;
        ViewportWorkspace& operator=(ViewportWorkspace&&) noexcept = default;
        ~ViewportWorkspace() = default;

        [[nodiscard]] ViewId primaryView() const noexcept;
        [[nodiscard]] std::optional<ViewId> activeViewport() const noexcept;
        [[nodiscard]] const WorkspaceLayout& layout() const noexcept { return layout_; }
        [[nodiscard]] const ViewRegistry& views() const noexcept { return views_; }
        [[nodiscard]] ViewRecord* findView(ViewId id) noexcept;
        [[nodiscard]] const ViewRecord* findView(ViewId id) const noexcept;
        [[nodiscard]] Viewport* findCamera(ViewId id) noexcept;
        [[nodiscard]] const Viewport* findCamera(ViewId id) const noexcept;
        [[nodiscard]] lfs::Status setViewProjection(ViewId id, const ViewProjectionState& projection);
        [[nodiscard]] lfs::Status setViewDepthWindow(ViewId id, const DepthWindowState& depth);
        [[nodiscard]] lfs::Status setViewGridPlane(ViewId id, int grid_plane);
        [[nodiscard]] lfs::Status setAreaEditor(ViewId id, std::string editor_id);
        // Record a mutation made through the public ViewRecord accessors. This
        // advances both the per-view snapshot generation and the registry
        // generation used by frame consumers to invalidate cached state.
        [[nodiscard]] lfs::Status bumpViewState(ViewId id);

        [[nodiscard]] lfs::Result<ViewId> split(ViewId source, SplitAxis axis, float ratio = 0.5f);
        [[nodiscard]] lfs::Status close(ViewId id);
        [[nodiscard]] lfs::Status focus(ViewId id);
        [[nodiscard]] lfs::Status maximize(ViewId id);
        [[nodiscard]] lfs::Status restoreMaximized();
        [[nodiscard]] lfs::Status resize(LayoutNodeId split, float ratio);
        [[nodiscard]] lfs::Status setPreset(LayoutPreset preset);

        // Restore Viewport, Scene and Rendering areas, keeping the active camera
        // and allocating fresh IDs for new leaves.
        [[nodiscard]] lfs::Status resetToDefaultThreeAreas();

        [[nodiscard]] WorkspaceFrameSnapshot snapshot(
            ViewRect outer,
            int min_pane_pixels = kDefaultMinPanePixels,
            int divider_pixels = kDefaultDividerPixels,
            glm::vec2 framebuffer_scale = {1.0f, 1.0f},
            int header_pixels = 0) const;

        void syncExtents(ViewRect outer,
                         int min_pane_pixels = kDefaultMinPanePixels,
                         int divider_pixels = kDefaultDividerPixels,
                         glm::vec2 framebuffer_scale = {1.0f, 1.0f},
                         int header_pixels = 0);

        [[nodiscard]] WorkspacePersistentState exportState() const;
        [[nodiscard]] lfs::Status importState(const WorkspacePersistentState& state);

    private:
        void seedNewQuadPanes(const std::vector<ViewId>& ids,
                              const std::vector<ViewId>& created);
        [[nodiscard]] std::optional<ViewId> firstViewport() const noexcept;
        void repairActiveViewport() noexcept;
        [[nodiscard]] PaneSnapshot copyPane(const LayoutPane& pane,
                                            bool focused,
                                            bool maximized,
                                            glm::vec2 framebuffer_scale) const;

        glm::ivec2 default_size_{1280, 720};
        ViewRegistry views_;
        WorkspaceLayout layout_;
        std::optional<ViewId> active_viewport_;
    };

} // namespace lfs::vis

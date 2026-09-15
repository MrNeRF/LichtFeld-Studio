/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "render_pass.hpp"
#include "workspace/viewport_workspace.hpp"

namespace lfs::vis {

    [[nodiscard]] LFS_VIS_API RenderSettings workspaceRenderSettings(
        const RenderSettings& shared_settings, const PaneSnapshot& pane);

    // Unjittered projection shared by picking, overlays and temporal motion.
    // Extent is explicit: raster resolution can differ from the pane's display size.
    [[nodiscard]] LFS_VIS_API lfs::rendering::FrameView makeWorkspaceFrameView(
        const PaneSnapshot& pane, glm::ivec2 extent, glm::vec3 background_color);

    // Scene data and committed selection are shared. Camera, projection, depth
    // window and cursor preview come from the pane owning this request.
    [[nodiscard]] LFS_VIS_API lfs::rendering::ViewportRenderRequest buildWorkspaceRenderRequest(
        const FrameContext& shared_scene, const PaneSnapshot& pane,
        glm::ivec2 render_extent, glm::vec2 jitter_pixels = {},
        std::optional<ViewId> interaction_view = std::nullopt);

} // namespace lfs::vis

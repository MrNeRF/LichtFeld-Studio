/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "render_pass.hpp"

namespace lfs::vis {

    // Centralized request builders for the public renderer boundary.
    // Visualizer-side code should prefer building FrameView/GpuFrame contracts first
    // and only translate to renderer request types here.
    [[nodiscard]] LFS_VIS_API lfs::rendering::ViewportRenderRequest buildViewportRenderRequest(
        const FrameContext& ctx, glm::ivec2 render_size,
        const Viewport* source_viewport = nullptr,
        std::optional<SplitViewPanelId> render_panel = std::nullopt);

    [[nodiscard]] LFS_VIS_API lfs::rendering::HoveredGaussianQueryRequest buildHoveredGaussianQueryRequest(
        const FrameContext& ctx, glm::ivec2 render_size,
        const Viewport* source_viewport = nullptr);

    [[nodiscard]] LFS_VIS_API lfs::rendering::SplitViewGaussianPanelRenderState buildSplitViewGaussianPanelRenderState(
        const FrameContext& ctx, glm::ivec2 render_size,
        const Viewport* source_viewport = nullptr,
        std::optional<SplitViewPanelId> render_panel = std::nullopt);

    [[nodiscard]] LFS_VIS_API lfs::rendering::SplitViewPointCloudPanelRenderState buildSplitViewPointCloudPanelRenderState(
        const FrameContext& ctx, glm::ivec2 render_size,
        const Viewport* source_viewport = nullptr);

    [[nodiscard]] LFS_VIS_API lfs::rendering::PointCloudRenderRequest buildPointCloudRenderRequest(
        const FrameContext& ctx, glm::ivec2 render_size, const std::vector<glm::mat4>& model_transforms);

} // namespace lfs::vis

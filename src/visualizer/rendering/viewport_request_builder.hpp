/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "core/scene.hpp"
#include "render_pass.hpp"

namespace lfs::vis {

    // Centralized request builders for the public renderer boundary.
    // Visualizer-side code should prefer building FrameView/GpuFrame contracts first
    // and only translate to renderer request types here.
    [[nodiscard]] LFS_VIS_API lfs::rendering::ViewportRenderRequest buildViewportRenderRequest(
        const FrameContext& ctx, glm::ivec2 render_size,
        const Viewport* source_viewport = nullptr,
        std::optional<SplitViewPanelId> render_panel = std::nullopt);

    [[nodiscard]] LFS_VIS_API lfs::rendering::ViewportRenderRequest buildViewportRenderRequest(
        const FrameContext& ctx, glm::ivec2 render_size,
        const Viewport* source_viewport,
        std::optional<SplitViewPanelId> render_panel,
        glm::ivec2 subregion_origin,
        glm::ivec2 subregion_full_size);

    [[nodiscard]] LFS_VIS_API lfs::rendering::SplitViewGaussianPanelRenderState buildSplitViewGaussianPanelRenderState(
        const FrameContext& ctx, glm::ivec2 render_size,
        const Viewport* source_viewport = nullptr,
        std::optional<SplitViewPanelId> render_panel = std::nullopt);

    [[nodiscard]] LFS_VIS_API lfs::rendering::SplitViewPointCloudPanelRenderState buildSplitViewPointCloudPanelRenderState(
        const FrameContext& ctx, glm::ivec2 render_size,
        const Viewport* source_viewport = nullptr);

    [[nodiscard]] LFS_VIS_API lfs::rendering::PointCloudRenderRequest buildPointCloudRenderRequest(
        const FrameContext& ctx, glm::ivec2 render_size, const std::vector<glm::mat4>& model_transforms);

    // Visible splat node shown in a PLY-comparison panel, or null when the
    // scene has fewer than two visible splat slots.
    [[nodiscard]] LFS_VIS_API const core::SceneNode* plyComparisonNodeForPanel(
        const core::Scene& scene,
        size_t split_view_offset,
        SplitViewPanelId panel);

    // Scope crop/ellipsoid/selection overlay state to one visible splat node so
    // a per-node comparison render can use identity transform indices.
    LFS_VIS_API void applyPlyComparisonNodeScope(
        lfs::rendering::GaussianFilterState& filters,
        lfs::rendering::GaussianOverlayState& overlay,
        const FrameContext& ctx,
        const core::SceneNode& node,
        int visible_index);

    struct PlyComparisonDepthSample {
        const core::SceneNode* node = nullptr;
        const core::SplatData* model = nullptr;
        int visible_index = -1;
        bool uses_owned_node_model = false;
    };

    // Clicked comparison panel's splat node and the model expected-depth should
    // sample. Prefers the node's own storage; peeks a prepared combined model
    // only when the node no longer owns one.
    [[nodiscard]] LFS_VIS_API PlyComparisonDepthSample resolvePlyComparisonDepthSample(
        const core::Scene& scene,
        size_t split_view_offset,
        SplitViewPanelId panel);

    // Rewrite metadata render state so a single comparison node renders with
    // identity transform indices, that node's crop/ellipsoid, and its selection slice.
    LFS_VIS_API void scopeSceneRenderStateToVisibleSplatNode(
        SceneRenderState& state,
        const core::Scene& scene,
        const core::SceneNode& node,
        int visible_index,
        const glm::mat4& visualizer_world_transform);

} // namespace lfs::vis

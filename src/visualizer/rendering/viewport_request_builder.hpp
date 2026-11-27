/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "core/scene.hpp"
#include "passes/vulkan_mesh_pass.hpp"
#include "render_pass.hpp"
#include <optional>
#include <vector>

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

    // Visible meshes shaded as the viewport draws them; exports reuse this to match it.
    [[nodiscard]] LFS_VIS_API std::vector<VulkanMeshDrawItem> buildViewportMeshDrawItems(
        const SceneRenderState& scene_state,
        const RenderSettings& settings,
        const glm::vec3& camera_backward);

    // The transforms argument must outlive the returned request.
    [[nodiscard]] LFS_VIS_API lfs::rendering::PointCloudRenderRequest buildPointCloudNodeRenderRequest(
        const FrameContext& ctx, const core::Scene& scene, const core::SceneNode& node,
        glm::ivec2 render_size, const std::vector<glm::mat4>& transforms);

    struct NodeComparisonNode {
        const core::SceneNode* node = nullptr;
        // Combined-model slot of a splat node; empty for point clouds, meshes, and hidden pinned splats.
        std::optional<size_t> splat_slot;
    };

    struct NodeComparisonPair {
        NodeComparisonNode left;
        NodeComparisonNode right;
    };

    // Whether a node has renderable geometry eligible for NodeComparison (splat, point cloud, or mesh).
    [[nodiscard]] LFS_VIS_API bool isNodeComparable(const core::Scene& scene, const core::SceneNode& node);

    // Visible comparison-eligible nodes in depth-first scene-tree order.
    [[nodiscard]] LFS_VIS_API std::vector<NodeComparisonNode> nodeComparisonNodes(const core::Scene& scene);

    // Resolve the Left and Right comparison nodes, honoring optional explicit node IDs
    // and falling back to split_view_offset over visible comparable nodes when unset or invalid.
    [[nodiscard]] LFS_VIS_API std::optional<NodeComparisonPair> resolveNodeComparisonPair(
        const core::Scene& scene,
        size_t split_view_offset,
        std::optional<core::NodeId> left_node_id = std::nullopt,
        std::optional<core::NodeId> right_node_id = std::nullopt);

    // Comparable splat, point cloud, or mesh shown in a comparison panel, or null when
    // no valid 2-node comparison pair can be resolved.
    [[nodiscard]] LFS_VIS_API const core::SceneNode* nodeComparisonNodeForPanel(
        const core::Scene& scene,
        size_t split_view_offset,
        SplitViewPanelId panel,
        std::optional<core::NodeId> left_node_id = std::nullopt,
        std::optional<core::NodeId> right_node_id = std::nullopt);

    // Scope scene/crop/ellipsoid/selection overlay state to one splat node so
    // a per-node comparison render can use identity transform indices.
    LFS_VIS_API void applyNodeComparisonNodeScope(
        lfs::rendering::GaussianSceneState& scene,
        lfs::rendering::GaussianFilterState& filters,
        lfs::rendering::GaussianOverlayState& overlay,
        const FrameContext& ctx,
        const core::SceneNode& node,
        int visible_index);

    // Keeps the object record of one slot, for a draw of that slot's model
    // alone; drops the effect when no binding applies to it.
    void scopeCustomEffectToSlot(std::optional<lfs::rendering::GaussianCustomEffectState>& effect,
                                 int visible_index);

    struct NodeComparisonDepthSample {
        const core::SceneNode* node = nullptr;
        const core::SplatData* model = nullptr;
        int visible_index = -1;
        bool uses_owned_node_model = false;
    };

    // Clicked comparison panel's splat node and the model expected-depth should
    // sample. Prefers the node's own storage; peeks a prepared combined model
    // only when the node no longer owns one.
    [[nodiscard]] LFS_VIS_API NodeComparisonDepthSample resolveNodeComparisonDepthSample(
        const core::Scene& scene,
        size_t split_view_offset,
        SplitViewPanelId panel,
        std::optional<core::NodeId> left_node_id = std::nullopt,
        std::optional<core::NodeId> right_node_id = std::nullopt);

    // Rewrite metadata render state so a single comparison node renders with
    // identity transform indices, that node's crop/ellipsoid, and its selection slice.
    LFS_VIS_API void scopeSceneRenderStateToVisibleSplatNode(
        SceneRenderState& state,
        const core::Scene& scene,
        const core::SceneNode& node,
        int visible_index,
        const glm::mat4& visualizer_world_transform);

} // namespace lfs::vis

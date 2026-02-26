/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "mesh_pass.hpp"
#include "core/logger.hpp"
#include "core/mesh_data.hpp"
#include "scene/scene_manager.hpp"
#include <cassert>
#include <glad/glad.h>

namespace lfs::vis {

    bool MeshPass::shouldExecute(DirtyMask, const FrameContext& ctx) const {
        if (!ctx.scene_manager)
            return false;
        return !ctx.scene_state.meshes.empty();
    }

    void MeshPass::execute(lfs::rendering::RenderingEngine& engine,
                           const FrameContext& ctx,
                           FrameResources& res) {
        assert(ctx.scene_manager);
        if (res.split_view_executed)
            return;

        const auto& scene_state = ctx.scene_state;
        const bool mesh_dirty = (ctx.frame_dirty & MESH_GEOMETRY_MASK) != 0;

        if (mesh_dirty) {
            const auto mesh_viewport = ctx.makeViewportData();

            const float flash_intensity = ctx.selection_flash_intensity;
            const bool any_selected = std::any_of(
                                          scene_state.meshes.begin(), scene_state.meshes.end(),
                                          [](const auto& vm) { return vm.is_selected; }) ||
                                      (!scene_state.selected_node_mask.empty() &&
                                       std::any_of(scene_state.selected_node_mask.begin(),
                                                   scene_state.selected_node_mask.end(),
                                                   [](bool b) { return b; }));

            const lfs::rendering::MeshRenderOptions mesh_opts{
                .wireframe_overlay = ctx.settings.mesh_wireframe,
                .wireframe_color = ctx.settings.mesh_wireframe_color,
                .wireframe_width = ctx.settings.mesh_wireframe_width,
                .light_dir = ctx.settings.mesh_light_dir,
                .light_intensity = ctx.settings.mesh_light_intensity,
                .ambient = ctx.settings.mesh_ambient,
                .backface_culling = ctx.settings.mesh_backface_culling,
                .shadow_enabled = ctx.settings.mesh_shadow_enabled,
                .shadow_map_resolution = ctx.settings.mesh_shadow_resolution,
                .desaturate_unselected = ctx.settings.desaturate_unselected && any_selected,
                .selection_flash_intensity = flash_intensity,
                .background_color = ctx.settings.background_color};

            glEnable(GL_DEPTH_TEST);
            glDepthFunc(GL_LESS);

            engine.resetMeshFrameState();
            for (const auto& vm : scene_state.meshes) {
                auto per_mesh_opts = mesh_opts;
                per_mesh_opts.is_selected = vm.is_selected;
                const auto result = engine.renderMesh(
                    *vm.mesh, mesh_viewport, vm.transform, per_mesh_opts, res.splats_presented);
                if (!result)
                    LOG_ERROR("Failed to render mesh: {}", result.error());
            }
        }

        if (engine.hasMeshRender()) {
            glViewport(ctx.viewport_pos.x, ctx.viewport_pos.y, ctx.render_size.x, ctx.render_size.y);

            if (res.splats_presented) {
                const auto composite_result = engine.compositeMeshAndSplat(
                    res.cached_result, ctx.render_size);
                if (!composite_result)
                    LOG_ERROR("Failed to composite: {}", composite_result.error());
            } else {
                glClearColor(ctx.settings.background_color.r, ctx.settings.background_color.g,
                             ctx.settings.background_color.b, 1.0f);
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

                const auto present_result = engine.presentMeshOnly();
                if (!present_result)
                    LOG_ERROR("Failed to present mesh: {}", present_result.error());
            }
        }
    }

} // namespace lfs::vis

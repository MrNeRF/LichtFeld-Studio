/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "split_view_composition.hpp"
#include "core/event_bridge/localization_manager.hpp"
#include "gui/string_keys.hpp"
#include "rendering/coordinate_conventions.hpp"
#include "scene/scene_manager.hpp"
#include "viewport_request_builder.hpp"
#include "visualizer/scene_coordinate_utils.hpp"
#include <cassert>
#include <format>

namespace lfs::vis {

    namespace {
        constexpr glm::vec4 kSplitDividerColor(0.29f, 0.33f, 0.42f, 1.0f);

        [[nodiscard]] lfs::rendering::SplitViewPanelContent buildModelPanelContent(
            const FrameContext& ctx,
            const Viewport& viewport,
            const glm::ivec2 render_size,
            const lfs::core::SplatData& model,
            const glm::mat4& model_transform,
            const bool use_full_scene,
            const std::optional<SplitViewPanelId> render_panel = std::nullopt) {
            lfs::rendering::SplitViewPanelContent content{
                .type = lfs::rendering::PanelContentType::Model3D,
                .model = &model,
                .model_transform = model_transform,
                .gaussian_render = std::nullopt,
                .point_cloud_render = std::nullopt,
                .texture_id = 0};

            if (ctx.settings.point_cloud_mode) {
                content.point_cloud_render =
                    buildSplitViewPointCloudPanelRenderState(ctx, render_size, &viewport);
                if (!use_full_scene) {
                    content.point_cloud_render->scene = {};
                }
                return content;
            }

            content.gaussian_render =
                buildSplitViewGaussianPanelRenderState(ctx, render_size, &viewport, render_panel);
            if (!use_full_scene) {
                content.gaussian_render->scene = {};
            }
            return content;
        }

        [[nodiscard]] lfs::rendering::SplitViewPanelContent buildMaskedPLYComparisonPanelContent(
            const FrameContext& ctx,
            const Viewport& viewport,
            const glm::ivec2 render_size,
            const size_t visible_node_count,
            const size_t visible_node_index) {
            assert(ctx.model);
            assert(visible_node_index < visible_node_count);

            auto content = buildModelPanelContent(
                ctx,
                viewport,
                render_size,
                *ctx.model,
                glm::mat4(1.0f),
                true,
                std::nullopt);

            if (content.gaussian_render.has_value()) {
                auto& state = *content.gaussian_render;
                state.scene.node_visibility_mask.assign(visible_node_count, false);
                state.scene.node_visibility_mask[visible_node_index] = true;

                // The compare wipe should reflect the scene render, not a live brush preview that only
                // exists in the interactive editor state.
                state.overlay.cursor = {};
                state.overlay.emphasis.transient_mask = {};
                state.overlay.emphasis.focused_gaussian_id = -1;
            }

            return content;
        }

        [[nodiscard]] SplitViewCompositionPlan makePlan(
            const std::array<SplitViewPanelPlan, 2>& panels,
            const glm::ivec2 output_size,
            const glm::vec3& background_color,
            const bool prefer_batched_gaussian_render,
            const bool letterbox = false,
            const glm::ivec2 content_size = {0, 0}) {
            return SplitViewCompositionPlan{
                .mode_label = {},
                .detail_label = {},
                .panels = panels,
                .composite = {.output_size = output_size, .background_color = background_color},
                .presentation =
                    {.divider_color = kSplitDividerColor,
                     .letterbox = letterbox,
                     .content_size = content_size},
                .prefer_batched_gaussian_render = prefer_batched_gaussian_render,
            };
        }

        [[nodiscard]] std::optional<SplitViewCompositionPlan> buildGTComparisonPlan(
            const FrameContext& ctx,
            const FrameResources& res) {
            if (!res.gt_context || !res.gt_context->valid() ||
                !res.cached_gpu_frame || !res.cached_gpu_frame->valid()) {
                return std::nullopt;
            }

            std::string gt_label = LOC(lichtfeld::Strings::StatusBar::GROUND_TRUTH);
            if (ctx.scene_manager && ctx.current_camera_id >= 0) {
                const auto disabled_uids = ctx.scene_manager->getScene().getTrainingDisabledCameraUids();
                if (disabled_uids.count(ctx.current_camera_id) > 0) {
                    gt_label = LOC(lichtfeld::Strings::StatusBar::GROUND_TRUTH_EXCLUDED);
                }
            }

            const std::string detail_label =
                ctx.current_camera_id >= 0
                    ? std::vformat(
                          LOC(lichtfeld::Strings::StatusBar::CAMERA),
                          std::make_format_args(ctx.current_camera_id))
                    : std::string{};

            auto plan = makePlan(
                std::array<SplitViewPanelPlan, 2>{
                    SplitViewPanelPlan{
                        .label = gt_label,
                        .panel =
                            {.content =
                                 {.type = lfs::rendering::PanelContentType::Image2D,
                                  .model = nullptr,
                                  .model_transform = glm::mat4(1.0f),
                                  .gaussian_render = std::nullopt,
                                  .point_cloud_render = std::nullopt,
                                  .texture_id = res.gt_context->gt_texture_id},
                             .presentation =
                                 {.start_position = 0.0f,
                                  .end_position = ctx.settings.split_position,
                                  .texcoord_scale = res.gt_context->gt_texcoord_scale,
                                  .flip_y = lfs::rendering::presentationFlipYFromTextureOrigin(
                                      res.gt_context->gt_texture_origin)}}},
                    SplitViewPanelPlan{
                        .label = LOC(lichtfeld::Strings::StatusBar::RENDERED),
                        .panel =
                            {.content =
                                 {.type = lfs::rendering::PanelContentType::CachedRender,
                                  .model = nullptr,
                                  .model_transform = glm::mat4(1.0f),
                                  .gaussian_render = std::nullopt,
                                  .point_cloud_render = std::nullopt,
                                  .texture_id = res.cached_gpu_frame->color.id},
                             .presentation =
                                 {.start_position = ctx.settings.split_position,
                                  .end_position = 1.0f,
                                  .texcoord_scale = res.gt_context->render_texcoord_scale,
                                  .flip_y = std::nullopt}}}},
                ctx.render_size,
                ctx.settings.background_color,
                false,
                true,
                res.gt_context->dimensions);
            plan.mode_label = LOC(lichtfeld::Strings::StatusBar::GT_COMPARE);
            plan.detail_label = detail_label;
            return plan;
        }

        [[nodiscard]] std::optional<SplitViewCompositionPlan> buildPLYComparisonPlan(
            const FrameContext& ctx) {
            if (!ctx.scene_manager) {
                return std::nullopt;
            }

            const auto& scene = ctx.scene_manager->getScene();
            const auto visible_nodes = scene.getVisibleNodes();
            if (visible_nodes.size() < 2) {
                return std::nullopt;
            }

            const size_t left_idx = ctx.settings.split_view_offset % visible_nodes.size();
            const size_t right_idx = (ctx.settings.split_view_offset + 1) % visible_nodes.size();
            assert(visible_nodes[left_idx]->model);
            assert(visible_nodes[right_idx]->model);

            const bool use_combined_scene_masks = ctx.model && !ctx.settings.point_cloud_mode;

            auto plan = makePlan(
                std::array<SplitViewPanelPlan, 2>{
                    SplitViewPanelPlan{
                        .label = visible_nodes[left_idx]->name,
                        .panel =
                            {.content =
                                 use_combined_scene_masks
                                     ? buildMaskedPLYComparisonPanelContent(
                                           ctx,
                                           ctx.viewport,
                                           ctx.render_size,
                                           visible_nodes.size(),
                                           left_idx)
                                     : buildModelPanelContent(
                                           ctx,
                                           ctx.viewport,
                                           ctx.render_size,
                                           *visible_nodes[left_idx]->model,
                                           scene_coords::nodeVisualizerWorldTransform(
                                               scene, visible_nodes[left_idx]->id),
                                           false,
                                           std::nullopt),
                             .presentation =
                                 {.start_position = 0.0f,
                                  .end_position = ctx.settings.split_position,
                                  .texcoord_scale = glm::vec2(1.0f, 1.0f),
                                  .flip_y = std::nullopt}}},
                    SplitViewPanelPlan{
                        .label = visible_nodes[right_idx]->name,
                        .panel =
                            {.content =
                                 use_combined_scene_masks
                                     ? buildMaskedPLYComparisonPanelContent(
                                           ctx,
                                           ctx.viewport,
                                           ctx.render_size,
                                           visible_nodes.size(),
                                           right_idx)
                                     : buildModelPanelContent(
                                           ctx,
                                           ctx.viewport,
                                           ctx.render_size,
                                           *visible_nodes[right_idx]->model,
                                           scene_coords::nodeVisualizerWorldTransform(
                                               scene, visible_nodes[right_idx]->id),
                                           false,
                                           std::nullopt),
                             .presentation =
                                 {.start_position = ctx.settings.split_position,
                                  .end_position = 1.0f,
                                  .texcoord_scale = glm::vec2(1.0f, 1.0f),
                                  .flip_y = std::nullopt}}}},
                ctx.makeViewportData().size,
                ctx.settings.background_color,
                false);
            plan.mode_label = LOC(lichtfeld::Strings::StatusBar::SPLIT_VIEW);
            plan.detail_label = std::format("{} | {}", plan.panels[0].label, plan.panels[1].label);
            return plan;
        }

        [[nodiscard]] std::optional<SplitViewCompositionPlan> buildIndependentDualPlan(
            const FrameContext& ctx) {
            if (!ctx.model || ctx.render_size.x <= 1 || ctx.render_size.y <= 0) {
                return std::nullopt;
            }

            const auto* left_panel = ctx.findViewPanel(SplitViewPanelId::Left);
            const auto* right_panel = ctx.findViewPanel(SplitViewPanelId::Right);
            if (!left_panel || !right_panel) {
                return std::nullopt;
            }

            auto plan = makePlan(
                std::array<SplitViewPanelPlan, 2>{
                    SplitViewPanelPlan{
                        .label = LOC(lichtfeld::Strings::StatusBar::PRIMARY_VIEW),
                        .panel =
                            {.content =
                                 buildModelPanelContent(
                                     ctx,
                                     *left_panel->viewport,
                                     left_panel->render_size,
                                     *ctx.model,
                                     glm::mat4(1.0f),
                                     true,
                                     SplitViewPanelId::Left),
                             .presentation =
                                 {.start_position = left_panel->start_position,
                                  .end_position = left_panel->end_position,
                                  .texcoord_scale = glm::vec2(1.0f, 1.0f),
                                  .flip_y = std::nullopt,
                                  .normalize_x_to_panel = true}}},
                    SplitViewPanelPlan{
                        .label = LOC(lichtfeld::Strings::StatusBar::SECONDARY_VIEW),
                        .panel =
                            {.content =
                                 buildModelPanelContent(
                                     ctx,
                                     *right_panel->viewport,
                                     right_panel->render_size,
                                     *ctx.model,
                                     glm::mat4(1.0f),
                                     true,
                                     SplitViewPanelId::Right),
                             .presentation =
                                 {.start_position = right_panel->start_position,
                                  .end_position = right_panel->end_position,
                                  .texcoord_scale = glm::vec2(1.0f, 1.0f),
                                  .flip_y = std::nullopt,
                                  .normalize_x_to_panel = true}}}},
                ctx.render_size,
                ctx.settings.background_color,
                true);
            plan.mode_label = LOC(lichtfeld::Strings::StatusBar::SPLIT_VIEW);
            plan.detail_label = std::format("{} | {}", plan.panels[0].label, plan.panels[1].label);
            return plan;
        }
    } // namespace

    lfs::rendering::SplitViewRequest SplitViewCompositionPlan::toRequest() const {
        return lfs::rendering::SplitViewRequest{
            .panels = {panels[0].panel, panels[1].panel},
            .composite = composite,
            .presentation = presentation,
            .prefer_batched_gaussian_render = prefer_batched_gaussian_render,
        };
    }

    SplitViewInfo SplitViewCompositionPlan::toInfo() const {
        return SplitViewInfo{
            .enabled = true,
            .mode_label = mode_label,
            .detail_label = detail_label,
            .left_name = panels[0].label,
            .right_name = panels[1].label,
        };
    }

    std::optional<SplitViewCompositionPlan> buildSplitViewCompositionPlan(
        const FrameContext& ctx,
        const FrameResources& res) {
        switch (ctx.settings.split_view_mode) {
        case SplitViewMode::GTComparison:
            return buildGTComparisonPlan(ctx, res);
        case SplitViewMode::PLYComparison:
            return buildPLYComparisonPlan(ctx);
        case SplitViewMode::IndependentDual:
            return buildIndependentDualPlan(ctx);
        case SplitViewMode::Disabled:
        default:
            return std::nullopt;
        }
    }

} // namespace lfs::vis

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
#include "window/vulkan_result.hpp"
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
                .image_handle = 0};

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
            LFS_VK_DEBUG_ASSERT(
                ctx.model != nullptr,
                "PLY comparison panel requires a combined scene model (model={:#x}, visible_node_index={}, visible_node_count={})",
                reinterpret_cast<std::uintptr_t>(ctx.model),
                visible_node_index,
                visible_node_count);
            LFS_VK_DEBUG_ASSERT(
                visible_node_index < visible_node_count,
                "PLY comparison panel node index must be inside the visibility mask (visible_node_index={}, visible_node_count={})",
                visible_node_index,
                visible_node_count);

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
                if (!ctx.scene_manager->getScene().isCameraTrainingEnabled(ctx.current_camera_id)) {
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
                                  .image_handle = res.gt_context->gt_image_handle},
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
                                  .image_handle = res.cached_gpu_frame->color.id},
                             .presentation =
                                 {.start_position = ctx.settings.split_position,
                                  .end_position = 1.0f,
                                  .texcoord_scale = res.gt_context->render_texcoord_scale,
                                  .flip_y = std::nullopt}}}},
                ctx.render_size,
                ctx.settings.background_color,
                true,
                res.gt_context->dimensions);
            plan.mode_label = LOC(lichtfeld::Strings::StatusBar::GT_COMPARE);
            plan.detail_label = detail_label;
            return plan;
        }

        [[nodiscard]] lfs::rendering::SplitViewPanelContent buildOwnedPLYComparisonPanelContent(
            const FrameContext& ctx,
            const Viewport& viewport,
            const glm::ivec2 render_size,
            const core::Scene& scene,
            const core::SceneNode& node,
            const int visible_index) {
            LFS_VK_DEBUG_ASSERT(
                node.model != nullptr,
                "PLY comparison node must own a renderable model (node_id={}, model={:#x})",
                node.id,
                reinterpret_cast<std::uintptr_t>(node.model.get()));

            auto content = buildModelPanelContent(
                ctx,
                viewport,
                render_size,
                *node.model,
                scene_coords::nodeVisualizerWorldTransform(scene, node.id),
                false,
                std::nullopt);

            if (content.gaussian_render.has_value()) {
                applyPlyComparisonNodeScope(
                    content.gaussian_render->scene,
                    content.gaussian_render->filters,
                    content.gaussian_render->overlay,
                    ctx,
                    node,
                    visible_index);
            }
            return content;
        }

        [[nodiscard]] std::optional<SplitViewCompositionPlan> buildPLYComparisonPlan(
            const FrameContext& ctx) {
            if (!ctx.scene_manager) {
                return std::nullopt;
            }

            const auto& scene = ctx.scene_manager->getScene();
            const auto visible_nodes = scene.getVisibleSplatNodeSlots();
            const auto pair = plyComparisonPairForOffset(visible_nodes.size(), ctx.settings.split_view_offset);
            if (!pair) {
                return std::nullopt;
            }

            const auto& left_slot = visible_nodes[pair->first];
            const auto& right_slot = visible_nodes[pair->second];
            if (!left_slot.node || !right_slot.node) {
                return std::nullopt;
            }

            const bool left_owns_model = left_slot.node->model != nullptr;
            const bool right_owns_model = right_slot.node->model != nullptr;
            // Prefer owned node models so comparison does not require a concatenated
            // combined model. Fall back to combined-scene masks only when the nodes
            // no longer own storage (consolidated scenes) and a model is already
            // present on the frame — never start a combined build from here.
            const bool use_combined_scene_masks =
                ctx.model && !ctx.settings.point_cloud_mode &&
                (!left_owns_model || !right_owns_model);
            if ((use_combined_scene_masks && !ctx.model) ||
                (!use_combined_scene_masks && (!left_owns_model || !right_owns_model))) {
                return std::nullopt;
            }
            if (use_combined_scene_masks) {
                LFS_VK_DEBUG_ASSERT(
                    ctx.model != nullptr,
                    "PLY comparison mask path requires a prepared combined model");
            } else {
                LFS_VK_DEBUG_ASSERT(
                    left_slot.node->model != nullptr,
                    "PLY comparison left node must own a renderable model (left_index={}, visible_node_count={}, node_id={}, model={:#x})",
                    pair->first,
                    visible_nodes.size(),
                    left_slot.node->id,
                    reinterpret_cast<std::uintptr_t>(left_slot.node->model.get()));
                LFS_VK_DEBUG_ASSERT(
                    right_slot.node->model != nullptr,
                    "PLY comparison right node must own a renderable model (right_index={}, visible_node_count={}, node_id={}, model={:#x})",
                    pair->second,
                    visible_nodes.size(),
                    right_slot.node->id,
                    reinterpret_cast<std::uintptr_t>(right_slot.node->model.get()));
            }

            const size_t mask_slot_count = std::max(visible_nodes.size(),
                                                    ctx.scene_state.node_visibility_mask.size());

            auto plan = makePlan(
                std::array<SplitViewPanelPlan, 2>{
                    SplitViewPanelPlan{
                        .label = left_slot.node->name,
                        .panel =
                            {.content =
                                 use_combined_scene_masks
                                     ? buildMaskedPLYComparisonPanelContent(
                                           ctx,
                                           ctx.viewport,
                                           ctx.render_size,
                                           mask_slot_count,
                                           left_slot.slot_index)
                                     : buildOwnedPLYComparisonPanelContent(
                                           ctx,
                                           ctx.viewport,
                                           ctx.render_size,
                                           scene,
                                           *left_slot.node,
                                           static_cast<int>(left_slot.slot_index)),
                             .presentation =
                                 {.start_position = 0.0f,
                                  .end_position = ctx.settings.split_position,
                                  .texcoord_scale = glm::vec2(1.0f, 1.0f),
                                  .flip_y = std::nullopt}}},
                    SplitViewPanelPlan{
                        .label = right_slot.node->name,
                        .panel =
                            {.content =
                                 use_combined_scene_masks
                                     ? buildMaskedPLYComparisonPanelContent(
                                           ctx,
                                           ctx.viewport,
                                           ctx.render_size,
                                           mask_slot_count,
                                           right_slot.slot_index)
                                     : buildOwnedPLYComparisonPanelContent(
                                           ctx,
                                           ctx.viewport,
                                           ctx.render_size,
                                           scene,
                                           *right_slot.node,
                                           static_cast<int>(right_slot.slot_index)),
                             .presentation =
                                 {.start_position = ctx.settings.split_position,
                                  .end_position = 1.0f,
                                  .texcoord_scale = glm::vec2(1.0f, 1.0f),
                                  .flip_y = std::nullopt}}}},
                ctx.makeViewportData().size,
                ctx.settings.background_color);
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
                ctx.settings.background_color);
            plan.mode_label = LOC(lichtfeld::Strings::StatusBar::SPLIT_VIEW);
            plan.detail_label = std::format("{} | {}", plan.panels[0].label, plan.panels[1].label);
            return plan;
        }
    } // namespace

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

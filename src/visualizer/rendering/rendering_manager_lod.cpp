/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Level-of-detail selection for a viewport render request. Backend-neutral:
// shared by the Vulkan and tensor rendering managers.

#include "core/logger.hpp"
#include "core/splat_data.hpp"
#include "rendering_manager.hpp"
#include "scene_renderer.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace lfs::vis {
    namespace {
        constexpr double kGpuLodRenderCapacityOverhead = 1.20;

        struct LodObjectFrame {
            glm::mat4 object_to_view{1.0f};
            float object_scale = 1.0f;
        };

        [[nodiscard]] LodObjectFrame makeLodObjectFrame(
            const lfs::rendering::FrameView& frame_view,
            const lfs::rendering::GaussianSceneState& scene) {
            glm::mat4 object_to_world(1.0f);
            if (scene.model_transforms && !scene.model_transforms->empty()) {
                object_to_world = scene.model_transforms->front();
            }

            const float sx = glm::length(glm::vec3(object_to_world[0]));
            const float sy = glm::length(glm::vec3(object_to_world[1]));
            const float sz = glm::length(glm::vec3(object_to_world[2]));
            const float object_scale = std::max({sx, sy, sz, 1.0f});

            return {.object_to_view = frame_view.getViewMatrix() * object_to_world,
                    .object_scale = object_scale};
        }

        [[nodiscard]] lfs::rendering::GaussianLodGpuTraversalState makeLodGpuTraversalState(
            const LodObjectFrame& lod_frame,
            const SparkLodController::LodParameters& params,
            const std::size_t node_count) {
            const glm::mat4 view_to_object = glm::inverse(lod_frame.object_to_view);
            glm::vec3 forward = -glm::vec3(view_to_object[2]);
            const float forward_length = glm::length(forward);
            if (forward_length > 1.0e-6f) {
                forward /= forward_length;
            } else {
                forward = {0.0f, 0.0f, -1.0f};
            }

            lfs::rendering::GaussianLodGpuTraversalState state;
            state.enabled = true;
            state.node_count = node_count;
            state.pixel_scale_limit = params.pixel_scale_limit;
            state.object_scale = params.object_scale;
            state.behind_camera_penalty = params.behind_camera_penalty;
            state.cone_foveation = params.cone_foveation;
            state.cone_inner_degrees = params.cone_inner_degrees;
            state.cone_outer_degrees = params.cone_outer_degrees;
            state.outside_view_foveation = params.outside_view_foveation;
            state.viewport_half_tan_x = params.viewport_half_tan_x;
            state.viewport_half_tan_y = params.viewport_half_tan_y;
            state.ortho_half_width = params.ortho_half_width;
            state.ortho_half_height = params.ortho_half_height;
            state.view_origin = glm::vec3(view_to_object[3]);
            state.view_forward = forward;
            state.object_to_view = lod_frame.object_to_view;
            state.viewport_foveation = params.viewport_foveation;
            state.orthographic = params.orthographic;
            return state;
        }
    } // namespace

    void RenderingManager::prepareLodRequest(const RenderSettings& frame_settings,
                                             const lfs::core::SplatData* model,
                                             lfs::rendering::ViewportRenderRequest& request,
                                             std::vector<std::uint32_t>& lod_touched_chunks) {
        const bool has_lod_tree = model && model->lod_tree && model->lod_tree->has_tree();
        if (has_lod_tree) {
            // Debug colors stay on the GPU path: the selector emits
            // per-node levels alongside indices.
            const bool prefer_gpu_lod =
                frame_settings.lod_enabled &&
                lfs::rendering::isVkSplatBackend(request.raster_backend);
            const auto create_lod_controller = [this]() {
                auto controller = std::make_unique<SparkLodController>();
                controller->setReadyCallback([this] {
                    notifyAsyncLodResultsReady();
                });
                return controller;
            };
            if (!lod_controller_) {
                lod_controller_ = create_lod_controller();
            }
            if (lod_controller_model_ != model) {
                lod_controller_.reset();
                lod_controller_ = create_lod_controller();
                lod_controller_->attach(*model);
                lod_controller_model_ = model;
                lod_controller_needs_sync_traversal_ = true;
                lod_controller_page_map_generation_ = 0;
            }
            // Spark-style quality scaler: the rendered cut targets
            // LOD Budget x Render Scale splats.
            const std::size_t effective_lod_budget = std::max<std::size_t>(
                1,
                static_cast<std::size_t>(
                    std::llround(static_cast<double>(frame_settings.lod_max_splats) *
                                 std::max(frame_settings.lod_render_scale, 0.1f))));
            if (scene_renderer_) {
                // Bounded page pool only matters while a LoD cut is rendered;
                // with LoD off the full-quality reference needs every page.
                std::size_t pool_budget_splats = 0;
                if (frame_settings.lod_enabled) {
                    constexpr std::size_t kAutoPoolFactor = 4;
                    const std::size_t floor_splats =
                        2 * effective_lod_budget + lfs::core::SplatLodTree::kChunkSplats;
                    pool_budget_splats =
                        frame_settings.lod_page_pool_splats > 0
                            ? frame_settings.lod_page_pool_splats
                            : kAutoPoolFactor * effective_lod_budget;
                    if (pool_budget_splats < floor_splats) {
                        static std::size_t last_warned_budget = 0;
                        if (last_warned_budget != pool_budget_splats) {
                            last_warned_budget = pool_budget_splats;
                            LOG_WARN("LOD page pool budget {} below working-set floor {}; clamping",
                                     pool_budget_splats,
                                     floor_splats);
                        }
                        pool_budget_splats = floor_splats;
                    }
                }
                scene_renderer_->configureLod({pool_budget_splats, frame_settings.lod_pool_vram_fraction,
                                               static_cast<std::uint32_t>(std::max(frame_settings.lod_fade_frames, 0))});
                if (auto page_snapshot = scene_renderer_->ensureLodPageCacheSnapshot(*model);
                    page_snapshot &&
                    page_snapshot->generation != lod_controller_page_map_generation_) {
                    lod_controller_->applyPageMaps(page_snapshot->page_to_chunk,
                                                   page_snapshot->chunk_to_page,
                                                   !prefer_gpu_lod);
                    lod_controller_page_map_generation_ = page_snapshot->generation;
                    notifyAsyncLodResultsReady();
                }
            }

            std::optional<lfs::rendering::GaussianLodGpuTraversalState> lod_gpu_traversal;
            if (frame_settings.lod_enabled) {
                SparkLodController::LodParameters params;
                params.max_splats = effective_lod_budget;
                params.lod_render_scale = frame_settings.lod_render_scale;
                params.behind_camera_penalty = frame_settings.lod_behind_camera_penalty;
                params.cone_foveation = frame_settings.lod_cone_foveation;
                params.cone_inner_degrees = frame_settings.lod_cone_inner_degrees;
                params.cone_outer_degrees = frame_settings.lod_cone_outer_degrees;
                const LodObjectFrame lod_frame = makeLodObjectFrame(request.frame_view, request.scene);
                params.object_scale = lod_frame.object_scale;

                // Compute pixel_scale_limit dynamically from camera FOV and viewport size,
                // matching Spark's runtime computation.
                {
                    const auto& fv = request.frame_view;
                    if (fv.orthographic) {
                        params.pixel_scale_limit = fv.ortho_scale / static_cast<float>(fv.size.y);
                        if (fv.ortho_scale > 0.0f) {
                            params.ortho_half_width =
                                static_cast<float>(fv.size.x) / (2.0f * fv.ortho_scale);
                            params.ortho_half_height =
                                static_cast<float>(fv.size.y) / (2.0f * fv.ortho_scale);
                        }
                    } else {
                        float vfov = lfs::rendering::focalLengthToVFov(fv.focal_length_mm);
                        float half_tan_fov = std::tan(glm::radians(vfov) * 0.5f);
                        params.pixel_scale_limit = (2.0f * half_tan_fov) / static_cast<float>(fv.size.y);
                        params.viewport_half_tan_y = half_tan_fov;
                        params.viewport_half_tan_x =
                            half_tan_fov * (static_cast<float>(fv.size.x) / static_cast<float>(fv.size.y));
                    }
                    // Spark multiplies each node's pixel scale by lod_scale
                    // (bigger scale = finer cut); dividing the stop limit is
                    // equivalent.
                    params.pixel_scale_limit /= std::max(params.lod_render_scale, 0.1f);
                    params.orthographic = fv.orthographic;
                }

                if (lod_controller_needs_sync_traversal_) {
                    // One-time sync traversal even in GPU mode: gives the
                    // renderer a valid static fallback cut for failure-mode
                    // frames before the GPU selector has produced output.
                    LOG_TIMER("lod_controller.update_sync");
                    lod_controller_->update(lod_frame.object_to_view, params);
                    lod_controller_needs_sync_traversal_ = false;
                } else if (!prefer_gpu_lod) {
                    {
                        LOG_TIMER("lod_controller.swap_async_results");
                        lod_controller_->swapAsyncResults(true, true, true);
                    }
                    LOG_TIMER("lod_controller.update_async_request");
                    lod_controller_->updateAsync(lod_frame.object_to_view, params);
                }
                if (prefer_gpu_lod) {
                    lod_gpu_traversal = makeLodGpuTraversalState(
                        lod_frame,
                        params,
                        model->lod_tree ? model->lod_tree->total_nodes() : 0u);
                    if (lod_gpu_traversal->node_count > 0) {
                        const auto budget_capacity = static_cast<std::size_t>(
                            std::ceil(static_cast<double>(effective_lod_budget) *
                                      kGpuLodRenderCapacityOverhead));
                        // Out-of-core RAD models keep only a coarse prefix in
                        // model->size(); the GPU cut selects logical tree nodes.
                        const std::size_t capacity_limit = std::max<std::size_t>(
                            model->size(),
                            model->lod_tree ? model->lod_tree->total_nodes() : 0u);
                        lod_gpu_traversal->output_capacity =
                            std::clamp<std::size_t>(budget_capacity, 1u, capacity_limit);
                        request.lod_gpu_traversal = *lod_gpu_traversal;
                    }
                }
            } else {
                lod_controller_->activateFullQualityReference();
            }

            lod_controller_->advanceTransition();
            const bool lod_transition_active = lod_controller_->transitionActive();
            if (lod_transition_active) {
                notifyAsyncLodResultsReady();
            }
            const auto& selected = frame_settings.lod_enabled
                                       ? lod_controller_->selectedIndices()
                                       : lod_controller_->fullQualityIndices();
            if (!selected.empty()) {
                request.lod_indices = selected.data();
                if (frame_settings.lod_enabled && lod_transition_active) {
                    const auto& weights = lod_controller_->selectedWeights();
                    if (weights.size() == selected.size()) {
                        request.lod_weights = weights.data();
                    }
                }
                if (lod_controller_->pageMappingActive()) {
                    const auto& logical = frame_settings.lod_enabled
                                              ? lod_controller_->selectedLogicalIndices()
                                              : lod_controller_->fullQualityLogicalIndices();
                    if (logical.size() == selected.size()) {
                        request.lod_logical_indices = logical.data();
                    }
                }
                if (frame_settings.lod_debug_colors) {
                    const auto& levels = frame_settings.lod_enabled
                                             ? lod_controller_->selectedLevels()
                                             : lod_controller_->fullQualityLevels();
                    if (levels.size() == selected.size()) {
                        request.lod_levels = levels.data();
                    }
                }
                request.lod_count = selected.size();
                request.lod_selection_hash = lod_controller_->selectionHash();
                request.lod_generation = lod_controller_->statsGeneration();
                if (!prefer_gpu_lod) {
                    // GPU mode derives prefetch priorities from the
                    // selector's chunk-touch readback instead. Passing
                    // lod_gpu_traversal here would activate GPU selection
                    // in the renderer and override the CPU cut (breaking
                    // debug colors), so the CPU path sends indices only.
                    lod_touched_chunks = lod_controller_->touchedChunks();
                    request.lod_touched_chunks = lod_touched_chunks.data();
                    request.lod_touched_chunk_count = lod_touched_chunks.size();
                }
            }
            request.lod_debug_mode = frame_settings.lod_debug_colors;
        } else {
            lod_controller_.reset();
            lod_controller_model_ = nullptr;
            lod_controller_needs_sync_traversal_ = false;
            lod_controller_page_map_generation_ = 0;
        }
    }

    void RenderingManager::noteLodPageGeneration(const std::uint64_t generation) {
        if (generation == 0 || generation == lod_controller_page_map_generation_)
            return;
        LOG_DEBUG("LOD page map generation advanced: renderer={} controller={}",
                  generation,
                  lod_controller_page_map_generation_);
        notifyAsyncLodResultsReady();
    }
} // namespace lfs::vis

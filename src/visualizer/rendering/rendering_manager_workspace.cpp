/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/camera.hpp"
#include "core/cuda/memory_arena.hpp"
#include "core/logger.hpp"
#include "core/memory_pressure.hpp"
#include "core/splat_data.hpp"
#include "core/tensor/backend/cuda/runtime/cuda_stream_context.hpp"
#include "model_renderability.hpp"
#include "nvidia_dlss_plugin.hpp"
#include "output_image_pool.hpp"
#include "passes/vulkan_scene_dlss_pipeline.hpp"
#include "point_cloud_vulkan_renderer.hpp"
#include "rendering_manager.hpp"
#include "scene/scene_manager.hpp"
#include "scene_temporal_resolve.hpp"
#include "scene_upscaler_registry.hpp"
#include "training/trainer.hpp"
#include "training/training_manager.hpp"
#include "view_output_key.hpp"
#include "viewport_error.hpp"
#include "viewport_request_builder.hpp"
#include "vksplat_viewport_renderer.hpp"
#include "workspace_render_request.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <expected>
#include <format>
#include <glm/gtc/matrix_inverse.hpp>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace lfs::vis {
    namespace {
        constexpr double kGpuLodRenderCapacityOverhead = 1.20;
        constexpr float kInteractiveResizeRenderScale = 0.33f;
        constexpr auto kTrainingOutputResizeStableDelay = std::chrono::milliseconds(500);

        [[nodiscard]] std::optional<glm::ivec2> nvidiaDlssOptimalRenderExtent(
            const glm::ivec2 output_extent, const std::uint32_t quality) {
            if (output_extent.x <= 0 || output_extent.y <= 0)
                return std::nullopt;
            const auto optimal = NvidiaDlssPlugin::instance().optimalSettings(
                static_cast<std::uint32_t>(output_extent.x),
                static_cast<std::uint32_t>(output_extent.y),
                quality);
            if (!optimal || optimal->render_width == 0 || optimal->render_height == 0 ||
                optimal->render_width > static_cast<std::uint32_t>(output_extent.x) ||
                optimal->render_height > static_cast<std::uint32_t>(output_extent.y)) {
                return std::nullopt;
            }
            return glm::ivec2{static_cast<int>(optimal->render_width),
                              static_cast<int>(optimal->render_height)};
        }

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
            return {.object_to_view = frame_view.getViewMatrix() * object_to_world,
                    .object_scale = std::max({sx, sy, sz, 1.0f})};
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

        void applyLodPixelScale(SparkLodController::LodParameters& params,
                                const lfs::rendering::FrameView& fv) {
            params.pixel_scale_limit = lodPixelScaleLimit(fv);
            if (fv.orthographic) {
                if (fv.ortho_scale > 0.0f) {
                    params.ortho_half_width =
                        static_cast<float>(fv.size.x) / (2.0f * fv.ortho_scale);
                    params.ortho_half_height =
                        static_cast<float>(fv.size.y) / (2.0f * fv.ortho_scale);
                }
            } else {
                const float half_tan_fov = std::tan(
                    glm::radians(lfs::rendering::focalLengthToVFov(fv.focal_length_mm)) * 0.5f);
                params.viewport_half_tan_y = half_tan_fov;
                params.viewport_half_tan_x =
                    half_tan_fov * (static_cast<float>(fv.size.x) / static_cast<float>(fv.size.y));
            }
            params.pixel_scale_limit /= std::max(params.lod_render_scale, 0.1f);
            params.orthographic = fv.orthographic;
        }

        struct LiveModelLockBundle {
            std::shared_lock<std::shared_mutex> lock;
            const lfs::core::Scene* scene = nullptr;
            LiveModelLockBundle() = default;
            LiveModelLockBundle(std::shared_lock<std::shared_mutex>&& l, const lfs::core::Scene* s)
                : lock(std::move(l)),
                  scene(s) {
                if (scene && lock.owns_lock()) {
                    scene->noteLiveModelLockAcquired();
                }
            }
            LiveModelLockBundle(LiveModelLockBundle&& other) noexcept
                : lock(std::move(other.lock)),
                  scene(other.scene) {
                other.scene = nullptr;
            }
            LiveModelLockBundle& operator=(LiveModelLockBundle&& other) noexcept {
                if (this != &other) {
                    if (scene && lock.owns_lock()) {
                        scene->noteLiveModelLockReleased();
                    }
                    lock = std::move(other.lock);
                    scene = other.scene;
                    other.scene = nullptr;
                }
                return *this;
            }
            ~LiveModelLockBundle() {
                if (scene && lock.owns_lock()) {
                    scene->noteLiveModelLockReleased();
                }
            }
            LiveModelLockBundle(const LiveModelLockBundle&) = delete;
            LiveModelLockBundle& operator=(const LiveModelLockBundle&) = delete;
            [[nodiscard]] bool owns_lock() const { return lock.owns_lock(); }
        };

        [[nodiscard]] std::optional<LiveModelLockBundle> acquireLiveModelRenderLock(
            const SceneManager* const scene_manager,
            const bool try_lock = false) {
            if (const auto* tm = scene_manager ? scene_manager->getTrainerManager() : nullptr) {
                if (const auto* trainer = tm->getTrainer()) {
                    const lfs::core::Scene* scene = trainer->getScene();
                    if (try_lock) {
                        std::shared_lock<std::shared_mutex> candidate(
                            trainer->getRenderMutex(), std::try_to_lock);
                        if (!candidate.owns_lock()) {
                            return std::nullopt;
                        }
                        return LiveModelLockBundle(std::move(candidate), scene);
                    }
                    return LiveModelLockBundle(
                        std::shared_lock<std::shared_mutex>(trainer->getRenderMutex()), scene);
                }
            }
            return std::nullopt;
        }

        [[nodiscard]] bool isRetryableSharedScratchUnavailable(const std::string_view error) {
            return error.find("shared scratch") != std::string_view::npos &&
                   (error.find("busy") != std::string_view::npos ||
                    error.find("capacity insufficient") != std::string_view::npos);
        }

        void hashCombine(std::uint64_t& seed, const std::uint64_t value) {
            seed ^= value + 0x9e3779b97f4a7c15ull + (seed << 6) + (seed >> 2);
        }

        [[nodiscard]] std::uint64_t floatBits(const float value) {
            std::uint32_t bits = 0;
            std::memcpy(&bits, &value, sizeof(bits));
            return bits;
        }

        [[nodiscard]] std::uint64_t fingerprintRenderSettings(const RenderSettings& settings) {
            std::uint64_t hash = static_cast<std::uint64_t>(settings.raster_backend);
            hashCombine(hash, static_cast<std::uint64_t>(settings.gut));
            hashCombine(hash, static_cast<std::uint64_t>(settings.lod_enabled));
            hashCombine(hash, static_cast<std::uint64_t>(settings.lod_max_splats));
            hashCombine(hash, floatBits(settings.lod_render_scale));
            hashCombine(hash, static_cast<std::uint64_t>(settings.lod_page_pool_splats));
            hashCombine(hash, floatBits(settings.lod_pool_vram_fraction));
            hashCombine(hash, static_cast<std::uint64_t>(settings.lod_fade_frames));
            hashCombine(hash, static_cast<std::uint64_t>(settings.lod_debug_colors));
            hashCombine(hash, std::hash<std::string>{}(settings.scene_upscaler));
            hashCombine(hash, std::hash<std::string>{}(settings.scene_upscaler_preset));
            hashCombine(hash, floatBits(settings.scene_upscaler_scale));
            hashCombine(hash, floatBits(settings.render_scale));
            hashCombine(hash, floatBits(settings.background_color.r));
            hashCombine(hash, floatBits(settings.background_color.g));
            hashCombine(hash, floatBits(settings.background_color.b));
            hashCombine(hash, static_cast<std::uint64_t>(settings.depth_view));
            hashCombine(hash, static_cast<std::uint64_t>(settings.antialiasing));
            hashCombine(hash, static_cast<std::uint64_t>(settings.mip_filter));
            hashCombine(hash, static_cast<std::uint64_t>(settings.point_cloud_mode));
            hashCombine(hash, static_cast<std::uint64_t>(settings.environment_mode));
            hashCombine(hash, std::hash<std::string>{}(settings.environment_map_path));
            hashCombine(hash, floatBits(settings.environment_exposure));
            hashCombine(hash, floatBits(settings.environment_rotation_degrees));
            hashCombine(hash, static_cast<std::uint64_t>(settings.apply_appearance_correction));
            hashCombine(hash, static_cast<std::uint64_t>(settings.desaturate_unselected));
            return hash;
        }

        [[nodiscard]] glm::ivec2 paneOutputExtent(const PaneSnapshot& pane) {
            if (pane.framebuffer_size.x > 0 && pane.framebuffer_size.y > 0) {
                return pane.framebuffer_size;
            }
            return {std::max(pane.rect.width, 0), std::max(pane.rect.height, 0)};
        }

        [[nodiscard]] bool paneCameraOrProjectionChanged(const PaneSnapshot& previous,
                                                         const PaneSnapshot& current) {
            return previous.rotation != current.rotation ||
                   previous.translation != current.translation ||
                   previous.pivot != current.pivot ||
                   previous.projection != current.projection ||
                   previous.depth != current.depth ||
                   previous.grid_plane != current.grid_plane;
        }

        [[nodiscard]] bool paneHasRenderableExtent(const glm::ivec2 extent) {
            return extent.x > 0 && extent.y > 0;
        }

        struct PaneLodStorage {
            std::vector<std::uint32_t> indices;
            std::vector<std::uint32_t> logical;
            std::vector<std::uint32_t> levels;
            std::vector<float> weights;
            std::vector<std::uint32_t> touched;
        };

        void copyLodSelection(PaneLodStorage& storage,
                              const SparkLodController& controller,
                              const bool lod_enabled,
                              const bool debug_colors,
                              const bool include_touched) {
            storage.indices = lod_enabled ? controller.selectedIndices()
                                          : controller.fullQualityIndices();
            if (controller.pageMappingActive()) {
                storage.logical = lod_enabled ? controller.selectedLogicalIndices()
                                              : controller.fullQualityLogicalIndices();
            } else {
                storage.logical.clear();
            }
            if (debug_colors) {
                storage.levels = lod_enabled ? controller.selectedLevels()
                                             : controller.fullQualityLevels();
            } else {
                storage.levels.clear();
            }
            if (lod_enabled && controller.transitionActive()) {
                storage.weights = controller.selectedWeights();
            } else {
                storage.weights.clear();
            }
            if (include_touched) {
                storage.touched = controller.touchedChunks();
            } else {
                storage.touched.clear();
            }
        }

        void bindLodStorage(lfs::rendering::ViewportRenderRequest& request,
                            const PaneLodStorage& storage,
                            const SparkLodController& controller,
                            const bool debug_colors) {
            if (storage.indices.empty()) {
                return;
            }
            request.lod_indices = storage.indices.data();
            request.lod_count = storage.indices.size();
            request.lod_selection_hash = controller.selectionHash();
            request.lod_generation = controller.statsGeneration();
            if (storage.logical.size() == storage.indices.size()) {
                request.lod_logical_indices = storage.logical.data();
            }
            if (debug_colors && storage.levels.size() == storage.indices.size()) {
                request.lod_levels = storage.levels.data();
            }
            if (storage.weights.size() == storage.indices.size()) {
                request.lod_weights = storage.weights.data();
            }
            if (!storage.touched.empty()) {
                request.lod_touched_chunks = storage.touched.data();
                request.lod_touched_chunk_count = storage.touched.size();
            }
            request.lod_debug_mode = debug_colors;
        }

        void applySplatDepth(RenderingManager::VulkanMeshFrame& mesh,
                             const VksplatViewportRenderer::RenderResult& result,
                             const lfs::rendering::FrameView& view) {
            if (result.depth_image_view == VK_NULL_HANDLE) {
                return;
            }
            mesh.depth_blit.external_image = result.depth_image;
            mesh.depth_blit.external_image_view = result.depth_image_view;
            mesh.depth_blit.external_image_layout = result.depth_image_layout;
            mesh.depth_blit.external_image_format = VK_FORMAT_R32_SFLOAT;
            mesh.depth_blit.external_image_generation = result.depth_generation;
            mesh.depth_blit.external_image_size = result.size;
            mesh.depth_blit.external_image_allocation_size = result.alloc_size;
            mesh.depth_blit.depth_is_ndc = false;
            mesh.depth_blit.flip_y = result.flip_y;
            mesh.depth_blit.near_plane = view.near_plane > 0.0f ? view.near_plane : 0.1f;
            mesh.depth_blit.far_plane = view.far_plane > 0.0f ? view.far_plane : 1000.0f;
            const glm::ivec2 depth_valid = result.size;
            const glm::ivec2 depth_alloc =
                result.alloc_size.x > 0 && result.alloc_size.y > 0 ? result.alloc_size : depth_valid;
            mesh.depth_blit.uv_scale = outputUvScale(depth_valid, depth_alloc);
            mesh.depth_blit.uv_clamp_max = outputUvClampMax(depth_valid, depth_alloc);
        }

        [[nodiscard]] RenderingManager::VulkanFrameResult colorFromRenderResult(
            const VksplatViewportRenderer::RenderResult& result,
            const std::uint64_t image_generation) {
            return {.image = {},
                    .external_image = result.image,
                    .external_image_view = result.image_view,
                    .external_image_layout = result.image_layout,
                    .external_image_generation = result.generation,
                    .completion_semaphore = result.completion_semaphore,
                    .completion_value = result.completion_value,
                    .image_generation = image_generation,
                    .size = result.size,
                    .alloc_size = result.alloc_size,
                    .flip_y = result.flip_y,
                    .matches_viewport_extent = true};
        }

        [[nodiscard]] RenderingManager::VulkanFrameResult colorFromPointCloudResult(
            const PointCloudVulkanRenderer::RenderResult& result,
            const std::uint64_t image_generation) {
            const glm::ivec2 alloc =
                result.alloc_size.x > 0 && result.alloc_size.y > 0 ? result.alloc_size : result.size;
            return {.image = {},
                    .external_image = result.image,
                    .external_image_view = result.image_view,
                    .external_image_layout = result.image_layout,
                    .external_image_generation = result.generation,
                    .completion_semaphore = VK_NULL_HANDLE,
                    .completion_value = 0,
                    .image_generation = image_generation,
                    .size = result.size,
                    .alloc_size = alloc,
                    .flip_y = result.flip_y,
                    .matches_viewport_extent = true};
        }

        void applyPointCloudDepth(RenderingManager::VulkanMeshFrame& mesh,
                                  const PointCloudVulkanRenderer::RenderResult& result,
                                  const lfs::rendering::FrameView& view) {
            if (result.depth_image_view == VK_NULL_HANDLE) {
                return;
            }
            mesh.depth_blit.external_image = result.depth_image;
            mesh.depth_blit.external_image_view = result.depth_image_view;
            mesh.depth_blit.external_image_layout = result.depth_image_layout;
            mesh.depth_blit.external_image_format = VK_FORMAT_R32_SFLOAT;
            mesh.depth_blit.external_image_generation = result.depth_generation;
            mesh.depth_blit.external_image_size = result.size;
            mesh.depth_blit.external_image_allocation_size =
                result.alloc_size.x > 0 && result.alloc_size.y > 0 ? result.alloc_size : result.size;
            mesh.depth_blit.depth_is_ndc = true;
            mesh.depth_blit.flip_y = result.flip_y;
            mesh.depth_blit.near_plane = view.near_plane > 0.0f ? view.near_plane : 0.1f;
            mesh.depth_blit.far_plane = view.far_plane > 0.0f ? view.far_plane : 1000.0f;
            const glm::ivec2 depth_valid = result.size;
            const glm::ivec2 depth_alloc = mesh.depth_blit.external_image_allocation_size;
            mesh.depth_blit.uv_scale = outputUvScale(depth_valid, depth_alloc);
            mesh.depth_blit.uv_clamp_max = outputUvClampMax(depth_valid, depth_alloc);
        }

        [[nodiscard]] PointCloudVulkanRenderer::RenderRequest makePointCloudVulkanRequest(
            const lfs::rendering::PointCloudRenderRequest& pc_request,
            const lfs::core::Tensor* positions,
            const lfs::core::Tensor* colors,
            const std::uint64_t data_revision,
            const std::uint64_t selection_revision,
            const lfs::core::SplatData* deleted_source,
            const RenderSettings& settings) {
            PointCloudVulkanRenderer::RenderRequest vk_req{};
            vk_req.positions = positions;
            vk_req.colors = colors;
            vk_req.positions_revision = data_revision;
            vk_req.colors_revision = data_revision;
            vk_req.model_transforms = pc_request.scene.model_transforms;
            vk_req.transform_indices = pc_request.scene.transform_indices.get();
            vk_req.node_visibility_mask = &pc_request.scene.node_visibility_mask;
            if (deleted_source && deleted_source->has_deleted_mask() &&
                deleted_source->deleted_mask_matches_size()) {
                vk_req.deleted_mask = &deleted_source->deleted();
                vk_req.deleted_mask_revision = deleted_source->deleted_mask_version();
            }
            vk_req.selection_mask = pc_request.overlay.selection_mask.get();
            vk_req.preview_selection_mask = pc_request.overlay.transient_mask.mask;
            vk_req.selection_colors = &pc_request.overlay.selection_colors;
            vk_req.preview_selection_additive = pc_request.overlay.transient_mask.additive;
            vk_req.selection_revision = selection_revision;
            vk_req.preview_selection_revision = selection_revision;
            if (pc_request.filters.crop_box.has_value()) {
                PointCloudVulkanRenderer::CropBox crop{};
                crop.to_local = pc_request.filters.crop_box->transform;
                crop.min = pc_request.filters.crop_box->min;
                crop.max = pc_request.filters.crop_box->max;
                crop.inverse = pc_request.filters.crop_inverse;
                crop.desaturate = pc_request.filters.crop_desaturate;
                vk_req.crop = crop;
            } else if (pc_request.filters.crop_ellipsoid.has_value()) {
                PointCloudVulkanRenderer::CropEllipsoid crop{};
                crop.to_local = pc_request.filters.crop_ellipsoid->transform;
                crop.radii = pc_request.filters.crop_ellipsoid->radii;
                crop.inverse = pc_request.filters.crop_inverse;
                crop.desaturate = pc_request.filters.crop_desaturate;
                vk_req.crop_ellipsoid = crop;
            }
            const glm::mat4 view = pc_request.frame_view.getViewMatrix();
            const glm::mat4 projection = lfs::rendering::createProjectionMatrix(
                pc_request.frame_view.size,
                lfs::rendering::focalLengthToVFov(pc_request.frame_view.focal_length_mm),
                pc_request.frame_view.orthographic,
                pc_request.frame_view.ortho_scale,
                pc_request.frame_view.near_plane,
                pc_request.frame_view.far_plane);
            glm::mat4 clip_y_flip(1.0f);
            clip_y_flip[1][1] = -1.0f;
            vk_req.view = view;
            vk_req.view_projection = clip_y_flip * projection * view;
            vk_req.size = pc_request.frame_view.size;
            vk_req.background_color = pc_request.frame_view.background_color;
            vk_req.transparent_background = pc_request.transparent_background;
            vk_req.orthographic = pc_request.frame_view.orthographic;
            vk_req.ortho_scale = pc_request.frame_view.ortho_scale;
            vk_req.focal_y = lfs::core::fov2focal(
                lfs::rendering::focalLengthToVFovRad(pc_request.frame_view.focal_length_mm),
                pc_request.frame_view.size.y);
            vk_req.voxel_size = pc_request.render.voxel_size;
            vk_req.scaling_modifier = pc_request.render.scaling_modifier;
            vk_req.depth_view = settings.depth_view;
            vk_req.depth_view_min = settings.depth_view_min;
            vk_req.depth_view_max = settings.depth_view_max;
            vk_req.depth_visualization_mode = settings.depth_visualization_mode;
            return vk_req;
        }

        [[nodiscard]] std::shared_ptr<lfs::core::Tensor> ensureCudaViewportImage(
            std::shared_ptr<lfs::core::Tensor> image,
            const std::string_view label) {
            if (!image || !image->is_valid()) {
                return {};
            }
            if (image->device() == lfs::core::Device::GPU) {
                return image;
            }
            auto cuda_image = image->cuda();
            if (!cuda_image.is_valid() || cuda_image.device() != lfs::core::Device::GPU) {
                LOG_WARN("{} produced a non-CUDA tensor; keeping the uncorrected external image",
                         label);
                return {};
            }
            return std::make_shared<lfs::core::Tensor>(std::move(cuda_image));
        }
    } // namespace

    void RenderingManager::retireClosedWorkspaceViews(const WorkspaceFrameSnapshot& snapshot) {
        std::unordered_set<ViewId> live(snapshot.live_viewport_ids.begin(), snapshot.live_viewport_ids.end());
        std::vector<ViewId> closed;
        {
            std::lock_guard lock(workspace_frames_mutex_);
            closed.reserve(workspace_views_.size());
            for (const auto& [id, runtime] : workspace_views_) {
                if (!live.contains(id)) {
                    closed.push_back(id);
                }
            }
            for (const ViewId id : closed) {
                workspace_views_.erase(id);
            }
        }
        if (vksplat_viewport_renderer_) {
            for (const ViewId id : closed) {
                if (auto released = vksplat_viewport_renderer_->releaseViewOutput(id); !released) {
                    LOG_DEBUG("releaseViewOutput({}): {}", id, viewportErrorText(released.error()));
                }
            }
        }
        if (point_cloud_vulkan_renderer_) {
            for (const ViewId id : closed) {
                if (auto released = point_cloud_vulkan_renderer_->releaseViewOutput(id); !released) {
                    LOG_DEBUG("point-cloud releaseViewOutput({}): {}", id, viewportErrorText(released.error()));
                }
            }
        }
    }

    void RenderingManager::clearWorkspacePublishedFrames() {
        std::vector<ViewId> ids;
        {
            std::lock_guard lock(workspace_frames_mutex_);
            ids.reserve(workspace_views_.size());
            for (auto& [id, runtime] : workspace_views_) {
                ids.push_back(id);
                runtime.published = {};
                runtime.has_published = false;
                runtime.convergence.cancelSettle();
            }
        }
        if (vksplat_viewport_renderer_) {
            for (const ViewId id : ids) {
                if (auto released = vksplat_viewport_renderer_->releaseViewOutput(id); !released) {
                    LOG_DEBUG("releaseViewOutput({}) during workspace clear: {}", id, viewportErrorText(released.error()));
                }
            }
        }
        if (point_cloud_vulkan_renderer_) {
            for (const ViewId id : ids) {
                if (auto released = point_cloud_vulkan_renderer_->releaseViewOutput(id); !released) {
                    LOG_DEBUG("point-cloud releaseViewOutput({}) during workspace clear: {}",
                              id,
                              viewportErrorText(released.error()));
                }
            }
        }
    }

    RenderingManager::WorkspaceVulkanFrame RenderingManager::previousWorkspaceFrame(
        const ViewId id, const PaneSnapshot& pane, std::string diagnostic) const {
        std::lock_guard lock(workspace_frames_mutex_);
        const auto it = workspace_views_.find(id);
        if (it == workspace_views_.end() || !it->second.has_published) {
            WorkspaceVulkanFrame frame;
            frame.pane = pane;
            frame.fresh = false;
            frame.diagnostic = std::move(diagnostic);
            return frame;
        }
        WorkspaceVulkanFrame frame = it->second.published;
        frame.fresh = false;
        if (frame.color.size != glm::ivec2(0, 0) &&
            (it->second.last_output_extent.x != paneOutputExtent(pane).x ||
             it->second.last_output_extent.y != paneOutputExtent(pane).y)) {
            frame.color.matches_viewport_extent = false;
        }
        frame.diagnostic = std::move(diagnostic);
        return frame;
    }

    std::optional<RenderingManager::WorkspaceVulkanFrame>
    RenderingManager::getWorkspaceVulkanFrame(const ViewId id) const {
        std::lock_guard lock(workspace_frames_mutex_);
        const auto it = workspace_views_.find(id);
        if (it == workspace_views_.end() || !it->second.has_published) {
            return std::nullopt;
        }
        return it->second.published;
    }

    std::vector<RenderingManager::WorkspaceVulkanFrame>
    RenderingManager::getWorkspaceVulkanFrames() const {
        std::lock_guard lock(workspace_frames_mutex_);
        std::vector<WorkspaceVulkanFrame> frames;
        frames.reserve(workspace_views_.size());
        for (const auto& [id, runtime] : workspace_views_) {
            if (runtime.has_published) {
                frames.push_back(runtime.published);
            }
        }
        std::sort(frames.begin(), frames.end(), [](const WorkspaceVulkanFrame& a, const WorkspaceVulkanFrame& b) {
            return a.pane.id < b.pane.id;
        });
        return frames;
    }

    void RenderingManager::reportWorkspaceSceneUpscalerRuntimeSelection(
        const ViewId view, const SceneUpscalerSelection selection) {
        {
            std::lock_guard lock(workspace_frames_mutex_);
            workspace_views_[view].upscaler_status = selection;
        }
        if (view != kInvalidViewId &&
            (workspace_focused_view_ == kInvalidViewId || view == workspace_focused_view_)) {
            reportSceneUpscalerRuntimeSelection(selection);
        }
    }

    std::vector<RenderingManager::WorkspaceVulkanFrame> RenderingManager::renderWorkspaceVulkanFrames(
        const RenderContext& context,
        const WorkspaceFrameSnapshot& snapshot,
        const std::optional<ViewId> interaction_view) {
        LOG_TIMER("renderWorkspaceVulkanFrames");
        if (vksplat_stale_frame_guard_.takeRecoveryRequest() && vksplat_viewport_renderer_) {
            vksplat_viewport_renderer_->cancelArenaHandoff();
            lfs::core::GlobalArenaManager::instance().clear_external_backing();
            vksplat_viewport_renderer_->releaseScratchOnIdle(true);
        }

        const RenderSettings frame_settings = [this] {
            std::lock_guard lock(settings_mutex_);
            return settings_;
        }();
        SceneManager* const scene_manager = context.scene_manager;
        auto* const trainer_manager = scene_manager ? scene_manager->getTrainerManager() : nullptr;
        const bool is_training = scene_manager && scene_manager->hasDataset() &&
                                 trainer_manager && trainer_manager->isTrainingActive();
        const bool training_initializing = trainer_manager &&
                                           trainer_manager->getState() == TrainingState::Starting;
        if (!is_training && vksplat_viewport_renderer_) {
            vksplat_viewport_renderer_->setLiveSubmitCallback({});
            vksplat_viewport_renderer_->cancelArenaHandoff();
        }
        if (context.vulkan_context) {
            last_vulkan_context_ = context.vulkan_context;
        }
        if (!is_training && vksplat_viewport_renderer_ &&
            vksplat_terminal_release_pending_.exchange(false, std::memory_order_acq_rel)) {
            vksplat_viewport_renderer_->releaseScratchOnIdle(true);
            vksplat_idle_frame_count_ = 0;
        }

        workspace_focused_view_ = snapshot.active_viewport.value_or(snapshot.primary);
        if (std::ranges::none_of(snapshot.panes, [this](const PaneSnapshot& pane) {
                return pane.id == workspace_focused_view_;
            })) {
            workspace_focused_view_ = snapshot.panes.empty() ? kInvalidViewId : snapshot.panes.front().id;
        }
        retireClosedWorkspaceViews(snapshot);

        const ViewId cut_target = interaction_view.value_or(
            snapshot.active_viewport.value_or(snapshot.primary));
        const std::uint64_t temporal_camera_cut_generation =
            temporal_camera_cut_generation_.load(std::memory_order_acquire);

        std::vector<WorkspaceVulkanFrame> results;
        results.reserve(snapshot.panes.size());
        if (snapshot.panes.empty()) {
            if (vksplat_viewport_renderer_ && !is_training) {
                vksplat_viewport_renderer_->setLiveSubmitCallback({});
            }
            return results;
        }
        initialized_ = true;

        const bool training_try_lock = is_training;
        auto render_lock = acquireLiveModelRenderLock(scene_manager, training_try_lock);
        bool render_lock_contended = training_try_lock && !render_lock.has_value() &&
                                     scene_manager && scene_manager->getTrainerManager() &&
                                     scene_manager->getTrainerManager()->getTrainer();

        const lfs::core::SplatData* model = nullptr;
        SceneRenderState scene_state;
        bool has_renderable_model = false;
        bool has_visible_gaussian_model = false;
        bool has_point_cloud = false;
        bool has_meshes = false;
        bool has_environment = environmentBackgroundEnabled(frame_settings);
        bool has_render_content = false;
        bool point_cloud_path = false;
        const auto sample_model_and_content = [&]() {
            if (scene_manager) {
                model = scene_manager->getModelForRendering();
                scene_state = scene_manager->buildRenderState();
            } else {
                model = nullptr;
                scene_state = {};
            }
            has_renderable_model = hasRenderableGaussians(model);
            has_visible_gaussian_model =
                has_renderable_model && scene_state.visible_splat_count > 0;
            has_point_cloud =
                scene_state.point_cloud != nullptr && scene_state.point_cloud->size() > 0;
            has_meshes = std::any_of(scene_state.meshes.begin(),
                                     scene_state.meshes.end(),
                                     [](const auto& mesh) { return mesh.mesh != nullptr; });
            has_render_content =
                has_visible_gaussian_model || has_point_cloud || has_meshes || has_environment;
            point_cloud_path =
                frame_settings.point_cloud_mode ||
                (!has_visible_gaussian_model && has_point_cloud);
        };
        const auto return_previous_all = [&](const std::string& diagnostic) {
            for (const auto& pane : snapshot.panes) {
                results.push_back(previousWorkspaceFrame(pane.id, pane, diagnostic));
            }
            if (vksplat_viewport_renderer_ && !is_training) {
                vksplat_viewport_renderer_->setLiveSubmitCallback({});
            }
            return results;
        };

        if (render_lock_contended) {
            const bool has_any_published = [this] {
                std::lock_guard lock(workspace_frames_mutex_);
                return std::any_of(workspace_views_.begin(), workspace_views_.end(),
                                   [](const auto& entry) { return entry.second.has_published; });
            }();
            if (has_any_published || training_initializing) {
                dirty_mask_.fetch_or(DirtyFlag::SPLATS, std::memory_order_relaxed);
                LOG_PERF("renderWorkspaceVulkanFrames: {} lock contended (retaining per-view cache)",
                         training_initializing ? "training initialization" : "step-boundary");
                render_lock.reset();
                return return_previous_all(training_initializing
                                               ? "deferred: training initialization"
                                               : "deferred: render lock contended");
            }
            render_lock = acquireLiveModelRenderLock(scene_manager, /*try_lock=*/false);
            render_lock_contended = !render_lock.has_value();
        }
        if (render_lock_contended) {
            render_lock.reset();
            return return_previous_all("deferred: render lock contended");
        }

        std::optional<std::shared_lock<std::shared_mutex>> model_read_lock;
        // Take both read locks before SceneManager can rebuild or inspect the model.
        // Non-refining optimizer steps only exclude readers through this mutex.
        if (is_training && trainer_manager && trainer_manager->getTrainer()) {
            auto* const model_trainer = trainer_manager->getTrainer();
            std::shared_lock<std::shared_mutex> candidate(
                model_trainer->getModelAccessMutex(), std::try_to_lock);
            if (!candidate.owns_lock()) {
                const bool has_any_published = [this] {
                    std::lock_guard lock(workspace_frames_mutex_);
                    return std::any_of(workspace_views_.begin(), workspace_views_.end(),
                                       [](const auto& entry) { return entry.second.has_published; });
                }();
                if (has_any_published) {
                    dirty_mask_.fetch_or(DirtyFlag::SPLATS, std::memory_order_relaxed);
                    LOG_PERF("renderWorkspaceVulkanFrames: model-access lock contended");
                    if (vksplat_viewport_renderer_) {
                        vksplat_viewport_renderer_->setLiveSubmitCallback({});
                    }
                    render_lock.reset();
                    return return_previous_all("deferred: model-access lock contended");
                }
                candidate = std::shared_lock<std::shared_mutex>(model_trainer->getModelAccessMutex());
            }
            model_read_lock.emplace(std::move(candidate));
        }
        sample_model_and_content();

        const std::size_t model_ptr = reinterpret_cast<std::size_t>(model);
        const auto model_source = scene_manager && scene_manager->hasDataset()
                                      ? ViewportFrameLifecycleService::ModelSource::Training
                                      : ViewportFrameLifecycleService::ModelSource::Scene;
        if (!render_lock_contended) {
            const bool model_pointer_changed = workspace_model_ptr_ != model_ptr;
            if (const auto model_change =
                    frame_lifecycle_service_.handleModelChange(model_ptr, viewport_artifact_service_,
                                                               model_source);
                model_change.changed || model_pointer_changed) {
                workspace_model_ptr_ = model_ptr;
                clearWorkspacePublishedFrames();
                last_logged_vksplat_render_error_.clear();
                if (vksplat_viewport_renderer_ &&
                    !(is_training && lfs::rendering::isVkSplatBackend(frame_settings.raster_backend))) {
                    if (trainer_manager) {
                        if (auto* trainer = trainer_manager->getTrainer()) {
                            trainer->setViewerReleaseFence(nullptr);
                        }
                    }
                    vksplat_viewport_renderer_->reset();
                }
                if (++workspace_scene_revision_ == 0) {
                    ++workspace_scene_revision_;
                }
                markDirty(DirtyFlag::ALL);
            } else {
                workspace_model_ptr_ = model_ptr;
            }
        }

        if (!has_render_content) {
            clearWorkspacePublishedFrames();
            render_lock.reset();
            for (const auto& pane : snapshot.panes) {
                WorkspaceVulkanFrame frame;
                frame.pane = pane;
                frame.fresh = false;
                frame.diagnostic = "no renderable content";
                results.push_back(std::move(frame));
            }
            if (vksplat_viewport_renderer_) {
                vksplat_viewport_renderer_->setLiveSubmitCallback({});
            }
            return results;
        }

        const DirtyMask training_refresh_dirty = frame_lifecycle_service_.handleTrainingRefresh(
            is_training,
            framerate_controller_.getSettings().training_frame_refresh_time_sec);
        const DirtyMask consumed_dirty = dirty_mask_.exchange(0) | training_refresh_dirty;
        constexpr DirtyMask kSharedSceneDirty =
            DirtyFlag::SPLATS | DirtyFlag::MESH | DirtyFlag::BACKGROUND | DirtyFlag::PPISP;
        if ((consumed_dirty & kSharedSceneDirty) != 0) {
            if (++workspace_scene_revision_ == 0) {
                ++workspace_scene_revision_;
            }
        }
        if ((consumed_dirty & DirtyFlag::SELECTION) != 0) {
            if (++workspace_selection_revision_ == 0) {
                ++workspace_selection_revision_;
            }
        }
        const std::uint64_t settings_fingerprint = fingerprintRenderSettings(frame_settings);
        if (settings_fingerprint != workspace_settings_fingerprint_) {
            workspace_settings_fingerprint_ = settings_fingerprint;
            if (++workspace_scene_revision_ == 0) {
                ++workspace_scene_revision_;
            }
        }

        const bool lod_follow =
            std::exchange(workspace_lod_follow_up_, false) ||
            workspace_retry_raster_ ||
            (lod_controller_ &&
             (lod_controller_->hasReadyResults() || lod_controller_->transitionActive()));
        workspace_retry_raster_ = false;

        const auto requested_upscaler =
            sceneUpscalerBackendFromId(frame_settings.scene_upscaler).value_or(SceneUpscalerBackend::Native);
        const auto reported_upscaler = sceneUpscalerRuntimeSelection();
        const bool reconstruction_runtime_ready =
            reported_upscaler.requested == requested_upscaler &&
            requested_upscaler != SceneUpscalerBackend::Native &&
            reported_upscaler.effective == requested_upscaler &&
            !reported_upscaler.fellBack();
        const bool memory_pressure_active =
            lfs::core::MemoryPressureCoordinator::instance().pressure_active();
        const auto resize_result = frame_lifecycle_service_.handleViewportResize(
            context.viewport.frameBufferSize);
        if (resize_result.dirty) {
            markDirty(resize_result.dirty);
        }
        const std::uint32_t nvidia_dlss_quality =
            frame_settings.scene_upscaler_preset == "performance"
                ? LFS_SCENE_UPSCALER_PLUGIN_PERFORMANCE
            : frame_settings.scene_upscaler_preset == "quality"
                ? LFS_SCENE_UPSCALER_PLUGIN_QUALITY
                : LFS_SCENE_UPSCALER_PLUGIN_BALANCED;
        const SceneTemporalQuality temporal_quality =
            frame_settings.scene_upscaler_preset == "performance"
                ? SceneTemporalQuality::Performance
            : frame_settings.scene_upscaler_preset == "quality"
                ? SceneTemporalQuality::Quality
                : SceneTemporalQuality::Balanced;
        const bool temporal_backend_requested =
            requested_upscaler == SceneUpscalerBackend::Temporal ||
            requested_upscaler == SceneUpscalerBackend::NvidiaDlss;

        struct PanePlan {
            const PaneSnapshot* pane = nullptr;
            glm::ivec2 output_extent{0, 0};
            glm::ivec2 render_size{0, 0};
            float scale = 1.0f;
            bool needs_raster = false;
            bool camera_cut = false;
            bool temporal_eligible = false;
            bool upscaler_ready = false;
        };
        std::vector<PanePlan> plans;
        plans.reserve(snapshot.panes.size());
        bool any_raster = false;
        {
            std::lock_guard lock(workspace_frames_mutex_);
            for (const auto& pane : snapshot.panes) {
                PanePlan plan;
                plan.pane = &pane;
                plan.output_extent = paneOutputExtent(pane);
                if (!isValidViewId(pane.id) || !paneHasRenderableExtent(plan.output_extent)) {
                    plans.push_back(plan);
                    continue;
                }
                auto& runtime = workspace_views_[pane.id];
                float scale = effectiveSceneRenderScale(
                    frame_settings.render_scale,
                    frame_settings.scene_upscaler_scale,
                    reconstruction_runtime_ready);
                if (resize_result.use_interactive_render_scale) {
                    scale = std::min(scale, kInteractiveResizeRenderScale);
                }
                if (memory_pressure_active) {
                    scale = std::clamp(scale * 0.5f, 0.25f, 1.0f);
                }
                glm::ivec2 render_size(
                    std::max(static_cast<int>(std::lround(static_cast<float>(plan.output_extent.x) * scale)), 1),
                    std::max(static_cast<int>(std::lround(static_cast<float>(plan.output_extent.y) * scale)), 1));
                const bool nvidia_dlss_optimal_query_allowed =
                    requested_upscaler == SceneUpscalerBackend::NvidiaDlss &&
                    reconstruction_runtime_ready && !resize_result.use_interactive_render_scale &&
                    !memory_pressure_active;
                if (nvidia_dlss_optimal_query_allowed) {
                    if (const auto optimal =
                            nvidiaDlssOptimalRenderExtent(plan.output_extent, nvidia_dlss_quality)) {
                        render_size = *optimal;
                        scale = std::min(static_cast<float>(render_size.x) /
                                             static_cast<float>(plan.output_extent.x),
                                         static_cast<float>(render_size.y) /
                                             static_cast<float>(plan.output_extent.y));
                    }
                }
                plan.render_size = render_size;
                plan.scale = scale;

                const SceneUpscalerSelection view_status =
                    runtime.upscaler_status.requested != SceneUpscalerBackend::Native
                        ? runtime.upscaler_status
                        : reported_upscaler;
                plan.upscaler_ready =
                    view_status.requested == requested_upscaler &&
                    requested_upscaler != SceneUpscalerBackend::Native &&
                    view_status.effective == requested_upscaler &&
                    !view_status.fellBack();
                const bool dlss_output_extent_supported =
                    requested_upscaler != SceneUpscalerBackend::NvidiaDlss ||
                    nvidiaDlssSupportsOutputExtent(plan.output_extent);
                const bool dlss_interactive_or_pressure =
                    requested_upscaler == SceneUpscalerBackend::NvidiaDlss &&
                    (resize_result.use_interactive_render_scale || memory_pressure_active);
                plan.temporal_eligible =
                    temporal_backend_requested && dlss_output_extent_supported &&
                    !dlss_interactive_or_pressure &&
                    !pane.projection.equirectangular && !frame_settings.apply_appearance_correction &&
                    lfs::rendering::isVkSplatBackend(frame_settings.raster_backend);

                const bool camera_changed =
                    !runtime.has_published ||
                    paneCameraOrProjectionChanged(runtime.last_pane, pane) ||
                    runtime.last_output_extent != plan.output_extent ||
                    runtime.last_render_extent != plan.render_size;
                const bool scene_changed = runtime.last_scene_revision != workspace_scene_revision_;
                const bool selection_changed =
                    runtime.last_selection_revision != workspace_selection_revision_;
                const bool overlay_this_view =
                    (consumed_dirty & DirtyFlag::OVERLAY) != 0 &&
                    interaction_view == pane.id;
                const bool temporal_settle =
                    (consumed_dirty & DirtyFlag::TEMPORAL) != 0 &&
                    runtime.convergence.enabled() && runtime.convergence.remaining() > 0;
                plan.needs_raster = camera_changed || scene_changed || selection_changed ||
                                    overlay_this_view || temporal_settle || lod_follow;
                plan.camera_cut =
                    (temporal_camera_cut_generation != runtime.consumed_camera_cut_generation &&
                     pane.id == cut_target);
                if (plan.needs_raster) {
                    any_raster = true;
                }
                plans.push_back(plan);
            }
        }

        if (!any_raster) {
            if (vksplat_viewport_renderer_) {
                vksplat_viewport_renderer_->setLiveSubmitCallback({});
            }
            render_lock.reset();
            for (const auto& plan : plans) {
                const auto& pane = *plan.pane;
                if (!isValidViewId(pane.id) || !paneHasRenderableExtent(plan.output_extent)) {
                    WorkspaceVulkanFrame frame;
                    frame.pane = pane;
                    frame.fresh = false;
                    frame.diagnostic = "empty pane";
                    results.push_back(std::move(frame));
                    continue;
                }
                results.push_back(previousWorkspaceFrame(pane.id, pane, "cached"));
            }
            return results;
        }

        const bool vksplat_backend = lfs::rendering::isVkSplatBackend(frame_settings.raster_backend);
        const bool raster_gaussians = has_visible_gaussian_model && vksplat_backend && !point_cloud_path;
        std::optional<lfs::core::CUDAStreamGuard> frame_stream_guard;
        if (raster_gaussians && context.vulkan_context) {
            if (!vksplat_viewport_renderer_) {
                vksplat_viewport_renderer_ = std::make_unique<VksplatViewportRenderer>();
            }
            if (is_training) {
                if (const auto ok =
                        vksplat_viewport_renderer_->ensureHandshakeReady(*context.vulkan_context);
                    !ok) {
                    LOG_WARN("Workspace frame deferred: VkSplat handshake failed: {}", ok.error());
                    vksplat_viewport_renderer_->setLiveSubmitCallback({});
                    dirty_mask_.fetch_or(DirtyFlag::SPLATS, std::memory_order_relaxed);
                    workspace_retry_raster_ = true;
                    return return_previous_all("deferred: training handshake unavailable");
                }
            }
            if (vksplat_viewport_renderer_->renderStream()) {
                frame_stream_guard.emplace(vksplat_viewport_renderer_->renderStream());
            }
        }

        lfs::training::Trainer* live_trainer = nullptr;
        if (is_training && trainer_manager && vksplat_viewport_renderer_ &&
            vksplat_viewport_renderer_->renderStream() &&
            vksplat_viewport_renderer_->renderCompleteFence()) {
            live_trainer = trainer_manager->getTrainer();
        }
        if (live_trainer) {
            live_trainer->setViewerReleaseFence(vksplat_viewport_renderer_->renderCompleteFence());
            live_trainer->beginModelRead(vksplat_viewport_renderer_->renderStream());
            lfs::training::Trainer* const trainer = live_trainer;
            vksplat_viewport_renderer_->setLiveSubmitCallback(
                [trainer](const std::uint64_t value) { trainer->publishViewerBorrow(value); });
        } else if (vksplat_viewport_renderer_) {
            vksplat_viewport_renderer_->setLiveSubmitCallback({});
        }
        struct ViewerBorrowPublisher {
            lfs::training::Trainer* trainer;
            VksplatViewportRenderer* renderer;
            ~ViewerBorrowPublisher() {
                if (trainer && renderer) {
                    try {
                        trainer->endModelRead(renderer->renderStream());
                        trainer->publishViewerBorrow(renderer->renderCompleteValue());
                    } catch (const std::exception& e) {
                        LOG_ERROR("ViewerBorrowPublisher: endModelRead/publishViewerBorrow failed "
                                  "during workspace frame teardown: {}",
                                  e.what());
                    } catch (...) {
                        LOG_ERROR("ViewerBorrowPublisher: endModelRead/publishViewerBorrow failed "
                                  "during workspace frame teardown with an unknown error");
                    }
                }
            }
        } viewer_borrow_publisher{live_trainer, vksplat_viewport_renderer_.get()};

        framerate_controller_.beginFrame();

        FrameContext shared_scene{
            .viewport = context.viewport,
            .viewport_region = context.viewport_region,
            .render_lock_held = render_lock.has_value(),
            .scene_manager = scene_manager,
            .model = model,
            .scene_state = std::move(scene_state),
            .settings = frame_settings,
            .render_size = context.viewport.frameBufferSize,
            .viewport_pos = {0, 0},
            .frame_dirty = consumed_dirty,
            .training_active = is_training,
            .depth_window_drag_preview = false,
            .cursor_preview = viewport_overlay_service_.cursorPreview(),
            .gizmo = viewport_overlay_service_.makeFrameGizmoState(),
            .hovered_camera_id = camera_interaction_service_.hoveredCameraId(),
            .current_camera_id = camera_interaction_service_.currentCameraId(),
            .hovered_gaussian_id = viewport_overlay_service_.hoveredGaussianId(),
            .selection_flash_intensity = getSelectionFlashIntensity(),
            .view_panels = {},
        };

        const bool has_lod_tree = model && model->lod_tree && model->lod_tree->has_tree();
        const bool prefer_gpu_lod = frame_settings.lod_enabled && raster_gaussians;
        PaneLodStorage full_quality_lod;
        std::size_t effective_lod_budget = 1;
        if (has_lod_tree && raster_gaussians) {
            const auto create_lod_controller = [this]() {
                auto controller = std::make_unique<SparkLodController>();
                controller->setReadyCallback([this] { notifyAsyncLodResultsReady(); });
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
            effective_lod_budget = std::max<std::size_t>(
                1,
                static_cast<std::size_t>(
                    std::llround(static_cast<double>(frame_settings.lod_max_splats) *
                                 std::max(frame_settings.lod_render_scale, 0.1f))));
            if (vksplat_viewport_renderer_) {
                std::size_t pool_budget_splats = 0;
                if (frame_settings.lod_enabled) {
                    constexpr std::size_t kAutoPoolFactor = 4;
                    const std::size_t visible_count = std::max<std::size_t>(1, snapshot.panes.size());
                    const std::size_t floor_splats =
                        visible_count *
                        (2 * effective_lod_budget + lfs::core::SplatLodTree::kChunkSplats);
                    pool_budget_splats =
                        frame_settings.lod_page_pool_splats > 0
                            ? frame_settings.lod_page_pool_splats
                            : kAutoPoolFactor * effective_lod_budget;
                    if (pool_budget_splats < floor_splats) {
                        pool_budget_splats = floor_splats;
                    }
                }
                vksplat_viewport_renderer_->setLodPagePoolBudget(pool_budget_splats);
                vksplat_viewport_renderer_->setLodPoolVramFraction(frame_settings.lod_pool_vram_fraction);
                vksplat_viewport_renderer_->setLodFadeFrames(
                    static_cast<std::uint32_t>(std::max(frame_settings.lod_fade_frames, 0)));
                if (auto page_snapshot = vksplat_viewport_renderer_->ensureLodPageCacheSnapshot(*model);
                    page_snapshot &&
                    page_snapshot->generation != lod_controller_page_map_generation_) {
                    lod_controller_->applyPageMaps(page_snapshot->page_to_chunk,
                                                   page_snapshot->chunk_to_page,
                                                   !prefer_gpu_lod);
                    lod_controller_page_map_generation_ = page_snapshot->generation;
                    notifyAsyncLodResultsReady();
                }
            }
            if (!frame_settings.lod_enabled) {
                lod_controller_->activateFullQualityReference();
                copyLodSelection(full_quality_lod, *lod_controller_, false,
                                 frame_settings.lod_debug_colors, false);
            }
            lod_controller_->advanceTransition();
            if (lod_controller_->transitionActive()) {
                notifyAsyncLodResultsReady();
            }
        } else if (!has_lod_tree) {
            lod_controller_.reset();
            lod_controller_model_ = nullptr;
            lod_controller_needs_sync_traversal_ = false;
            lod_controller_page_map_generation_ = 0;
        }

        bool force_input_upload = (consumed_dirty & DirtyFlag::SPLATS) != 0;
        bool any_temporal_follow = false;
        bool any_streaming = false;
        bool gpu_fallback_sync_done = false;
        PaneLodStorage gpu_fallback_lod;

        if (point_cloud_path && force_input_upload) {
            point_cloud_colors_cache_key_ = nullptr;
            point_cloud_colors_cache_size_ = 0;
            point_cloud_colors_cache_ = lfs::core::Tensor{};
            ++point_cloud_data_revision_;
        }
        if (point_cloud_path && (consumed_dirty & DirtyFlag::SELECTION) != 0) {
            ++point_cloud_preview_selection_revision_;
        }

        lfs::core::Tensor splat_positions;
        const lfs::core::Tensor* point_cloud_positions = nullptr;
        const lfs::core::Tensor* point_cloud_colors = nullptr;
        const lfs::core::SplatData* point_cloud_deleted_source = nullptr;
        std::vector<glm::mat4> point_cloud_transforms_storage;
        const std::vector<glm::mat4>* point_cloud_transforms = nullptr;
        std::string point_cloud_setup_error;
        if (point_cloud_path) {
            if (!context.vulkan_context) {
                point_cloud_setup_error =
                    "Point-cloud viewer rendering requires an active Vulkan context";
            } else {
                if (!point_cloud_vulkan_renderer_) {
                    point_cloud_vulkan_renderer_ = std::make_unique<PointCloudVulkanRenderer>();
                }
                if (frame_settings.point_cloud_mode && has_visible_gaussian_model && model) {
                    constexpr float SH_C0 = 0.28209479177387814f;
                    const auto& sh0 = model->sh0_raw();
                    const void* sh0_key = sh0.is_valid() ? sh0.ptr<float>() : nullptr;
                    const std::size_t sh0_count =
                        sh0.is_valid() ? static_cast<std::size_t>(sh0.size(0)) : 0;
                    if (sh0_key != point_cloud_colors_cache_key_ ||
                        sh0_count != point_cloud_colors_cache_size_ ||
                        !point_cloud_colors_cache_.is_valid()) {
                        try {
                            point_cloud_colors_cache_ =
                                (sh0.slice(1, 0, 1).squeeze(1) * SH_C0 + 0.5f).clamp(0.0f, 1.0f);
                            point_cloud_colors_cache_key_ = sh0_key;
                            point_cloud_colors_cache_size_ = sh0_count;
                        } catch (const std::exception& e) {
                            LOG_WARN("Point cloud color derivation failed: {}", e.what());
                            point_cloud_setup_error =
                                std::format("Point cloud color derivation failed: {}", e.what());
                        }
                    }
                    if (point_cloud_setup_error.empty()) {
                        splat_positions = model->get_means();
                        point_cloud_positions = &splat_positions;
                        point_cloud_colors = &point_cloud_colors_cache_;
                        point_cloud_deleted_source = model;
                        point_cloud_transforms = &shared_scene.scene_state.model_transforms;
                    }
                } else if (has_point_cloud && shared_scene.scene_state.point_cloud) {
                    point_cloud_positions = &shared_scene.scene_state.point_cloud->means;
                    point_cloud_colors = &shared_scene.scene_state.point_cloud->colors;
                    point_cloud_transforms_storage = {shared_scene.scene_state.point_cloud_transform};
                    point_cloud_transforms = &point_cloud_transforms_storage;
                } else if (point_cloud_setup_error.empty()) {
                    point_cloud_setup_error = "Point-cloud workspace raster has no positions";
                }
            }
        }

        // Only visible panes contribute LOD demand. Hidden views retain their output
        // and feedback so restoration and late readbacks still work.
        struct VisibleWorkspaceDemandScope {
            VksplatViewportRenderer* renderer = nullptr;
            ~VisibleWorkspaceDemandScope() {
                if (renderer) {
                    renderer->clearVisibleWorkspaceViews();
                }
            }
        } visible_demand_scope;
        if (vksplat_viewport_renderer_) {
            std::vector<ViewOutputKey> visible_output_keys;
            visible_output_keys.reserve(snapshot.panes.size());
            for (const auto& pane : snapshot.panes) {
                if (isValidViewId(pane.id)) {
                    visible_output_keys.push_back(sceneOutputKey(pane.id));
                }
            }
            vksplat_viewport_renderer_->setVisibleWorkspaceViews(visible_output_keys);
            visible_demand_scope.renderer = vksplat_viewport_renderer_.get();
        }

        for (const auto& plan : plans) {
            const auto& pane = *plan.pane;
            if (!isValidViewId(pane.id) || !paneHasRenderableExtent(plan.output_extent)) {
                WorkspaceVulkanFrame frame;
                frame.pane = pane;
                frame.fresh = false;
                frame.diagnostic = "empty pane";
                results.push_back(std::move(frame));
                continue;
            }
            if (!plan.needs_raster) {
                results.push_back(previousWorkspaceFrame(pane.id, pane, "cached"));
                continue;
            }
            if (point_cloud_path) {
                if (!point_cloud_setup_error.empty() || !point_cloud_positions ||
                    !point_cloud_colors || !point_cloud_transforms ||
                    !point_cloud_vulkan_renderer_ || !context.vulkan_context) {
                    results.push_back(previousWorkspaceFrame(
                        pane.id,
                        pane,
                        point_cloud_setup_error.empty()
                            ? "Point-cloud workspace raster is not ready"
                            : point_cloud_setup_error));
                    continue;
                }

                lfs::rendering::FrameView unjittered;
                try {
                    unjittered = makeWorkspaceFrameView(
                        pane, plan.output_extent, frame_settings.background_color);
                } catch (const std::exception& e) {
                    LOG_WARN("Viewport {} camera setup failed: {}", pane.id, e.what());
                    results.push_back(previousWorkspaceFrame(pane.id, pane, e.what()));
                    continue;
                }

                Viewport camera(plan.render_size.x, plan.render_size.y);
                camera.setViewMatrix(pane.rotation, pane.translation);
                camera.camera.setPivot(pane.pivot);
                camera.ortho_scale_override = pane.projection.ortho_scale;
                camera.frameBufferSize = plan.render_size;
                auto pane_settings = workspaceRenderSettings(frame_settings, pane);
                const bool owns_interaction = interaction_view == pane.id;
                auto cursor = owns_interaction ? shared_scene.cursor_preview : CursorPreviewState{};
                cursor.panel.reset();
                const FrameContext pane_context{
                    .viewport = camera,
                    .render_lock_held = shared_scene.render_lock_held,
                    .scene_manager = shared_scene.scene_manager,
                    .model = shared_scene.model,
                    .scene_state = shared_scene.scene_state,
                    .settings = pane_settings,
                    .render_size = plan.render_size,
                    .viewport_pos = {pane.rect.x, pane.rect.y},
                    .frame_dirty = shared_scene.frame_dirty,
                    .training_active = shared_scene.training_active,
                    .depth_window_drag_preview =
                        owns_interaction && shared_scene.depth_window_drag_preview,
                    .cursor_preview = cursor,
                    .gizmo = shared_scene.gizmo,
                    .hovered_camera_id = owns_interaction ? shared_scene.hovered_camera_id : -1,
                    .current_camera_id = owns_interaction ? shared_scene.current_camera_id : -1,
                    .hovered_gaussian_id = owns_interaction ? shared_scene.hovered_gaussian_id : -1,
                    .selection_flash_intensity = shared_scene.selection_flash_intensity,
                    .view_panels = {},
                };
                auto pc_request = buildPointCloudRenderRequest(
                    pane_context, plan.render_size, *point_cloud_transforms);
                pc_request.frame_view = makeWorkspaceFrameView(
                    pane, plan.render_size, frame_settings.background_color);
                const auto vk_req = makePointCloudVulkanRequest(
                    pc_request,
                    point_cloud_positions,
                    point_cloud_colors,
                    point_cloud_data_revision_,
                    point_cloud_preview_selection_revision_,
                    point_cloud_deleted_source,
                    pane_settings);

                if (auto registered = point_cloud_vulkan_renderer_->registerViewOutput(pane.id);
                    !registered) {
                    results.push_back(previousWorkspaceFrame(pane.id, pane, viewportErrorText(registered.error())));
                    continue;
                }
                if (is_training &&
                    point_cloud_vulkan_renderer_->nextOutputImagesNeedResize(
                        plan.render_size, sceneOutputKey(pane.id)) &&
                    !resize_result.require_immediate_output_resize &&
                    frame_lifecycle_service_.resizeRecentlyChanged(kTrainingOutputResizeStableDelay)) {
                    workspace_retry_raster_ = true;
                    dirty_mask_.fetch_or(DirtyFlag::VIEWPORT, std::memory_order_relaxed);
                    results.push_back(previousWorkspaceFrame(
                        pane.id, pane, "deferred: training output resize"));
                    continue;
                }

                lfs::Result<PointCloudVulkanRenderer::RenderResult> render_result =
                    viewportError("Point-cloud workspace render was not executed");
                try {
                    render_result = point_cloud_vulkan_renderer_->render(
                        *context.vulkan_context, vk_req, sceneOutputKey(pane.id));
                } catch (const std::exception& e) {
                    LOG_WARN("Viewport operation failed: {}", e.what());
                    render_result = viewportError(
                        std::format("Point-cloud render threw: {}", e.what()));
                }
                if (!render_result) {
                    results.push_back(previousWorkspaceFrame(pane.id, pane, viewportErrorText(render_result.error())));
                    continue;
                }

                WorkspaceVulkanFrame frame;
                frame.pane = pane;
                std::uint64_t image_generation = 0;
                {
                    std::lock_guard lock(workspace_frames_mutex_);
                    auto& runtime = workspace_views_[pane.id];
                    image_generation = runtime.published.color.image_generation + 1;
                    if (image_generation == 0) {
                        ++image_generation;
                    }
                }
                frame.source = WorkspaceVulkanFrame::Source::PointCloud;
                frame.color = colorFromPointCloudResult(*render_result, image_generation);
                frame.geometry = makePaneMeshFrame(shared_scene, pane_settings, unjittered);
                applyPointCloudDepth(frame.geometry, *render_result, unjittered);
                frame.unjittered_view = unjittered;
                frame.fresh = true;
                frame.diagnostic = "ok";
                {
                    std::lock_guard lock(workspace_frames_mutex_);
                    auto& runtime = workspace_views_[pane.id];
                    runtime.last_pane = pane;
                    runtime.last_render_extent = plan.render_size;
                    runtime.last_output_extent = plan.output_extent;
                    runtime.last_scene_revision = workspace_scene_revision_;
                    runtime.last_selection_revision = workspace_selection_revision_;
                    runtime.last_settings_fingerprint = workspace_settings_fingerprint_;
                    runtime.consumed_camera_cut_generation = temporal_camera_cut_generation;
                    if (runtime.convergence.enabled()) {
                        runtime.convergence.cancelSettle();
                    }
                    runtime.published = frame;
                    runtime.has_published = true;
                }
                results.push_back(std::move(frame));
                continue;
            }
            if (has_visible_gaussian_model && !vksplat_backend) {
                results.push_back(previousWorkspaceFrame(
                    pane.id, pane, "Gaussian workspace raster requires a VkSplat backend"));
                continue;
            }
            if (raster_gaussians && !context.vulkan_context) {
                results.push_back(previousWorkspaceFrame(
                    pane.id, pane, "VkSplat backend requires an active Vulkan context"));
                continue;
            }

            if (raster_gaussians && is_training &&
                vksplat_viewport_renderer_->nextOutputImagesNeedResize(
                    plan.render_size, sceneOutputKey(pane.id)) &&
                !resize_result.require_immediate_output_resize &&
                frame_lifecycle_service_.resizeRecentlyChanged(kTrainingOutputResizeStableDelay)) {
                workspace_retry_raster_ = true;
                dirty_mask_.fetch_or(DirtyFlag::VIEWPORT, std::memory_order_relaxed);
                results.push_back(previousWorkspaceFrame(
                    pane.id, pane, "deferred: training output resize"));
                continue;
            }

            glm::vec2 jitter_pixels{0.0f};
            {
                std::lock_guard lock(workspace_frames_mutex_);
                auto& runtime = workspace_views_[pane.id];
                const bool allow_temporal_settle =
                    (consumed_dirty & (DirtyFlag::SPLATS | DirtyFlag::MESH | DirtyFlag::BACKGROUND)) == 0 &&
                    !lod_follow;
                runtime.convergence.prepare(
                    plan.temporal_eligible,
                    plan.camera_cut || paneCameraOrProjectionChanged(runtime.last_pane, pane) ||
                        runtime.last_output_extent != plan.output_extent ||
                        runtime.last_render_extent != plan.render_size ||
                        runtime.last_scene_revision != workspace_scene_revision_,
                    allow_temporal_settle);
                jitter_pixels = runtime.convergence.jitter();
                if (temporal_backend_requested && !plan.upscaler_ready) {
                    jitter_pixels = glm::vec2(0.0f);
                }
            }

            lfs::rendering::FrameView unjittered;
            try {
                unjittered = makeWorkspaceFrameView(
                    pane, plan.output_extent, frame_settings.background_color);
            } catch (const std::exception& e) {
                LOG_WARN("Viewport {} camera setup failed: {}", pane.id, e.what());
                results.push_back(previousWorkspaceFrame(pane.id, pane, e.what()));
                continue;
            }

            auto request = buildWorkspaceRenderRequest(
                shared_scene, pane, plan.render_size, jitter_pixels, interaction_view);
            PaneLodStorage pane_lod;
            if (has_lod_tree && raster_gaussians && lod_controller_) {
                SparkLodController::LodParameters params;
                params.max_splats = effective_lod_budget;
                params.lod_render_scale = frame_settings.lod_render_scale;
                params.behind_camera_penalty = frame_settings.lod_behind_camera_penalty;
                params.cone_foveation = frame_settings.lod_cone_foveation;
                params.cone_inner_degrees = frame_settings.lod_cone_inner_degrees;
                params.cone_outer_degrees = frame_settings.lod_cone_outer_degrees;
                const LodObjectFrame lod_frame = makeLodObjectFrame(request.frame_view, request.scene);
                params.object_scale = lod_frame.object_scale;
                applyLodPixelScale(params, request.frame_view);
                if (frame_settings.lod_enabled) {
                    if (prefer_gpu_lod) {
                        if (lod_controller_needs_sync_traversal_ && !gpu_fallback_sync_done) {
                            lod_controller_->update(lod_frame.object_to_view, params);
                            lod_controller_needs_sync_traversal_ = false;
                            gpu_fallback_sync_done = true;
                            copyLodSelection(gpu_fallback_lod, *lod_controller_, true,
                                             frame_settings.lod_debug_colors, false);
                        }
                        auto lod_gpu_traversal = makeLodGpuTraversalState(
                            lod_frame, params,
                            model->lod_tree ? model->lod_tree->total_nodes() : 0u);
                        if (lod_gpu_traversal.node_count > 0) {
                            const auto budget_capacity = static_cast<std::size_t>(
                                std::ceil(static_cast<double>(effective_lod_budget) *
                                          kGpuLodRenderCapacityOverhead));
                            const std::size_t capacity_limit = std::max<std::size_t>(
                                model->size(),
                                model->lod_tree ? model->lod_tree->total_nodes() : 0u);
                            lod_gpu_traversal.output_capacity =
                                std::clamp<std::size_t>(budget_capacity, 1u, capacity_limit);
                            request.lod_gpu_traversal = lod_gpu_traversal;
                        }
                        bindLodStorage(request, gpu_fallback_lod, *lod_controller_,
                                       frame_settings.lod_debug_colors);
                    } else {
                        lod_controller_->update(lod_frame.object_to_view, params);
                        copyLodSelection(pane_lod, *lod_controller_, true,
                                         frame_settings.lod_debug_colors, true);
                        bindLodStorage(request, pane_lod, *lod_controller_,
                                       frame_settings.lod_debug_colors);
                    }
                } else {
                    bindLodStorage(request, full_quality_lod, *lod_controller_,
                                   frame_settings.lod_debug_colors);
                }
            }

            auto pane_settings = workspaceRenderSettings(frame_settings, pane);

            const auto publish_mesh = [&](const VksplatViewportRenderer::RenderResult* result) {
                auto mesh = makePaneMeshFrame(shared_scene, pane_settings, unjittered);
                if (result) {
                    applySplatDepth(mesh, *result, unjittered);
                    if (plan.temporal_eligible && result->depth_image_view != VK_NULL_HANDLE) {
                        std::uint64_t scene_generation =
                            static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(model));
                        if (model) {
                            hashCombine(scene_generation, static_cast<std::uint64_t>(model->size()));
                            hashCombine(scene_generation, model->param_layout_generation());
                        }
                        hashCombine(scene_generation, workspace_scene_revision_);
                        mesh.temporal = VulkanMeshFrame::TemporalFrame{
                            .input =
                                {
                                    .view = unjittered,
                                    .output_extent = plan.output_extent,
                                    .jitter = request.frame_view.orthographic
                                                  ? glm::vec2(0.0f)
                                                  : temporalJitterNdc(jitter_pixels, plan.render_size),
                                    .render_scale = plan.scale,
                                    .scene_generation = scene_generation,
                                    .backend_key =
                                        (static_cast<std::uint64_t>(requested_upscaler) << 32) |
                                        (static_cast<std::uint64_t>(temporal_quality) + 1),
                                    .camera_cut = plan.camera_cut,
                                },
                            .resolve_settings = sceneTemporalQualitySettings(temporal_quality),
                            .quality = temporal_quality,
                        };
                    }
                }
                return mesh;
            };

            if (!raster_gaussians) {
                WorkspaceVulkanFrame frame;
                frame.pane = pane;
                frame.geometry = publish_mesh(nullptr);
                frame.unjittered_view = unjittered;
                frame.fresh = true;
                frame.diagnostic = "mesh/environment";
                {
                    std::lock_guard lock(workspace_frames_mutex_);
                    auto& runtime = workspace_views_[pane.id];
                    runtime.last_pane = pane;
                    runtime.last_render_extent = plan.render_size;
                    runtime.last_output_extent = plan.output_extent;
                    runtime.last_scene_revision = workspace_scene_revision_;
                    runtime.last_selection_revision = workspace_selection_revision_;
                    runtime.last_settings_fingerprint = workspace_settings_fingerprint_;
                    runtime.consumed_camera_cut_generation = temporal_camera_cut_generation;
                    runtime.published = frame;
                    runtime.has_published = true;
                }
                results.push_back(std::move(frame));
                continue;
            }

            if (auto registered = vksplat_viewport_renderer_->registerViewOutput(pane.id); !registered) {
                results.push_back(previousWorkspaceFrame(pane.id, pane, viewportErrorText(registered.error())));
                continue;
            }

            const bool selection_only =
                // Projected/sorted raster scratch is shared across submissions.
                // Another live pane may have overwritten it since this image.
                snapshot.live_viewport_ids.size() == 1 &&
                ((consumed_dirty & DirtyFlag::SELECTION) != 0) &&
                (consumed_dirty & ~(DirtyFlag::SELECTION | DirtyFlag::OVERLAY)) == 0 &&
                !plan.camera_cut;
            lfs::Result<VksplatViewportRenderer::RenderResult> render_result =
                viewportError("VkSplat workspace render was not executed");
            bool used_overlay = false;
            if (selection_only) {
                const auto previous = previousWorkspaceFrame(pane.id, pane, {});
                if (previous.color.external_image != VK_NULL_HANDLE &&
                    previous.color.size == plan.render_size) {
                    try {
                        render_result = vksplat_viewport_renderer_->rerenderSelectionOverlay(
                            *context.vulkan_context,
                            *model,
                            request,
                            sceneOutputKey(pane.id),
                            is_training);
                        used_overlay = render_result.has_value();
                    } catch (const std::exception& e) {
                        LOG_WARN("Viewport operation failed: {}", e.what());
                        render_result = viewportError(
                            std::format("VkSplat selection overlay threw: {}", e.what()));
                    }
                }
            }
            if (!used_overlay) {
                try {
                    render_result = vksplat_viewport_renderer_->render(
                        *context.vulkan_context,
                        *model,
                        request,
                        force_input_upload,
                        sceneOutputKey(pane.id),
                        is_training);
                } catch (const std::exception& e) {
                    LOG_WARN("Viewport operation failed: {}", e.what());
                    render_result = viewportError(
                        std::format("VkSplat render threw: {}", e.what()));
                    lfs::core::Tensor::trim_memory_pool();
                }
            }
            if (!render_result) {
                if (isRetryableSharedScratchUnavailable(viewportErrorText(render_result.error()))) {
                    if (viewportErrorText(render_result.error()).find("arena is busy") != std::string::npos) {
                        vksplat_viewport_renderer_->requestArenaHandoff();
                    }
                    workspace_retry_raster_ = true;
                    dirty_mask_.fetch_or(DirtyFlag::SPLATS, std::memory_order_relaxed);
                    results.push_back(previousWorkspaceFrame(
                        pane.id, pane,
                        std::format("deferred: {}", viewportErrorText(render_result.error()))));
                    continue;
                }
                results.push_back(previousWorkspaceFrame(pane.id, pane, viewportErrorText(render_result.error())));
                continue;
            }

            force_input_upload = false;
            if (render_result->lod_streaming_active) {
                any_streaming = true;
            }
            if (render_result->lod_page_generation != 0 &&
                render_result->lod_page_generation != lod_controller_page_map_generation_) {
                notifyAsyncLodResultsReady();
            }

            WorkspaceVulkanFrame frame;
            frame.pane = pane;
            std::uint64_t image_generation = 0;
            {
                std::lock_guard lock(workspace_frames_mutex_);
                auto& runtime = workspace_views_[pane.id];
                image_generation = runtime.published.color.image_generation + 1;
                if (image_generation == 0) {
                    ++image_generation;
                }
            }
            frame.source = WorkspaceVulkanFrame::Source::Gaussian;
            frame.color = colorFromRenderResult(*render_result, image_generation);
            std::string diagnostic = used_overlay ? "selection overlay" : "ok";
            if (frame_settings.apply_appearance_correction && context.vulkan_context &&
                vksplat_viewport_renderer_) {
                const bool transparent_compositing =
                    environmentBackgroundUsesTransparentViewerCompositing(pane_settings);
                auto image = transparent_compositing
                                 ? vksplat_viewport_renderer_->readOutputImageRgba(
                                       *context.vulkan_context, sceneOutputKey(pane.id))
                                 : vksplat_viewport_renderer_->readOutputImage(
                                       *context.vulkan_context, sceneOutputKey(pane.id));
                if (image && *image) {
                    // Use this pane's camera calibration while its pose and camera-list generation
                    // match; otherwise use PPISP evaluation correction.
                    const int correction_camera_uid =
                        pane.camera_uid && scene_manager &&
                                pane.camera_binding_generation ==
                                    scene_manager->getScene().cameraListGeneration()
                            ? *pane.camera_uid
                            : -1;
                    auto corrected = applyViewportAppearanceCorrection(
                        std::move(*image),
                        scene_manager,
                        frame_settings,
                        correction_camera_uid);
                    corrected = ensureCudaViewportImage(
                        std::move(corrected), "VkSplat workspace PPISP correction");
                    if (corrected && corrected->is_valid()) {
                        frame.color.image = std::move(corrected);
                        frame.color.external_image = VK_NULL_HANDLE;
                        frame.color.external_image_view = VK_NULL_HANDLE;
                        frame.color.external_image_layout = VK_IMAGE_LAYOUT_UNDEFINED;
                        frame.color.external_image_generation = 0;
                        frame.color.completion_semaphore = VK_NULL_HANDLE;
                        frame.color.completion_value = 0;
                        diagnostic = used_overlay ? "selection overlay+ppisp" : "ppisp";
                    } else {
                        LOG_WARN("Workspace PPISP correction produced no valid viewport image; "
                                 "falling back to uncorrected external image");
                    }
                } else {
                    LOG_WARN("Workspace PPISP readback failed: {}",
                             image ? "missing image payload" : viewportErrorText(image.error()));
                }
            }
            const bool temporal_frame_published =
                plan.temporal_eligible && render_result->depth_image_view != VK_NULL_HANDLE;
            frame.geometry = publish_mesh(&*render_result);
            frame.unjittered_view = unjittered;
            frame.fresh = true;
            frame.diagnostic = std::move(diagnostic);
            {
                std::lock_guard lock(workspace_frames_mutex_);
                auto& runtime = workspace_views_[pane.id];
                runtime.last_pane = pane;
                runtime.last_render_extent = plan.render_size;
                runtime.last_output_extent = plan.output_extent;
                runtime.last_scene_revision = workspace_scene_revision_;
                runtime.last_selection_revision = workspace_selection_revision_;
                runtime.last_settings_fingerprint = workspace_settings_fingerprint_;
                runtime.consumed_camera_cut_generation = temporal_camera_cut_generation;
                if (temporal_frame_published) {
                    if (runtime.convergence.completeSuccessfulFrame()) {
                        any_temporal_follow = true;
                    }
                } else if (runtime.convergence.enabled()) {
                    runtime.convergence.cancelSettle();
                }
                runtime.published = frame;
                runtime.has_published = true;
            }
            results.push_back(std::move(frame));
        }

        if (any_temporal_follow) {
            requestTemporalFollowUp();
        }
        if (any_streaming) {
            workspace_lod_follow_up_ = true;
            requestRenderFollowUp();
        }
        // Keep topology and model reads protected until ViewerBorrowPublisher
        // has recorded the GPU read completion during scope teardown.
        return results;
    }

} // namespace lfs::vis

/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/async_task_manager.hpp"
#include "core/data_loading_service.hpp"
#include "core/error_bus.hpp"
#include "core/event_bridge/localization_manager.hpp"
#include "core/events.hpp"
#include "core/logger.hpp"
#include "core/parameter_manager.hpp"
#include "core/parameters.hpp"
#include "core/path_utils.hpp"
#include "core/provenance.hpp"
#include "core/scene.hpp"
#include "core/services.hpp"
#include "gui/error_event_bridge.hpp"
#include "gui/gui_manager.hpp"
#include "gui/panel_registry.hpp"
#include "gui/string_keys.hpp"
#include "gui/utils/native_file_dialog.hpp"
#include "gui/video_export_utils.hpp"
#include "internal/resource_paths.hpp"
#include "io/exporter.hpp"
#include "io/formats/colmap.hpp"
#include "python/runner.hpp"
#include "rendering/mesh2splat.hpp"
#include "rendering/mesh_offscreen_renderer.hpp"
#include "rendering/passes/vulkan_mesh_pass.hpp"
#include "rendering/rendering.hpp"
#include "rendering/rendering_manager.hpp"
#include "scene/scene_manager.hpp"
#include "scene/scene_render_state.hpp"
#include "sequencer/keyframe.hpp"
#include "sequencer/sequencer_controller.hpp"
#include "training/training_manager.hpp"
#include "visualizer/app_store.hpp"
#include "visualizer/gui/video_widget_interface.hpp"
#include "visualizer/scene_coordinate_utils.hpp"
#include "visualizer_impl.hpp"
#include "window/vulkan_context.hpp"
#include "window/window_manager.hpp"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <condition_variable>
#include <filesystem>
#include <format>
#include <functional>
#include <future>
#include <shared_mutex>
#include <string_view>
#include <type_traits>

namespace lfs::vis::gui {

    using ExportFormat = lfs::core::ExportFormat;

    namespace {

        template <typename Fn>
        class ScopeExit final {
        public:
            explicit ScopeExit(Fn fn) : fn_(std::move(fn)) {}
            ScopeExit(const ScopeExit&) = delete;
            ScopeExit& operator=(const ScopeExit&) = delete;
            ~ScopeExit() noexcept { fn_(); }

        private:
            Fn fn_;
        };

        // Fills project/commit/node/dataset once the .licht project session (#1525) is merged.
        void populate_project_identity([[maybe_unused]] core::ProvenanceStamp& stamp) {}

        [[nodiscard]] core::ProvenanceStamp make_gui_export_stamp(const lfs::vis::SceneManager& scene_manager) {
            auto stamp = core::make_provenance_stamp();
            populate_project_identity(stamp);

            const auto* const trainer_manager = scene_manager.getTrainerManager();
            if (trainer_manager) {
                const int iteration = trainer_manager->getCurrentIteration();
                // iteration 0 means an untrained scene, deliberately not stamped.
                if (iteration > 0)
                    stamp.iteration = iteration;
            }

            const auto* const trainer = trainer_manager ? trainer_manager->getTrainer() : nullptr;
            if (trainer) {
                const auto strategy = lfs::core::param::canonical_strategy_name(
                    trainer->getParams().optimization.strategy);
                if (!strategy.empty())
                    stamp.strategy = std::string(strategy);
            }
            return stamp;
        }

    } // namespace

    [[nodiscard]] const char* getDatasetTypeName(const std::filesystem::path& path) {
        switch (lfs::io::Loader::getDatasetType(path)) {
        case lfs::io::DatasetType::COLMAP: return "COLMAP";
        case lfs::io::DatasetType::Transforms: return "NeRF/Blender";
        default: return "Dataset";
        }
    }

    [[nodiscard]] const char* exportProgressFormatName(const ExportFormat format) noexcept {
        switch (format) {
        case ExportFormat::PLY: return "PLY";
        case ExportFormat::SOG: return "SOG";
        case ExportFormat::SPZ: return "SPZ";
        case ExportFormat::HTML_VIEWER: return "HTML";
        case ExportFormat::USD: return "USD";
        case ExportFormat::NUREC_USDZ: return "USDZ";
        case ExportFormat::RAD: return "RAD";
        case ExportFormat::COLMAP: return "COLMAP";
        default: return "file";
        }
    }

    void wakeMainThreadForAsyncWork() {
        if (auto* const window_manager = services().windowOrNull())
            window_manager->wakeEventLoop();
    }

    void truncateSHDegree(lfs::core::SplatData& splat, const int target_degree) {
        splat.set_sh_degree(target_degree);
    }

    struct BorrowExportPlan {
        core::Scene::MergeStorageMode storage_mode = core::Scene::MergeStorageMode::Clone;
        std::shared_mutex* model_mutex = nullptr;
    };

    [[nodiscard]] BorrowExportPlan makeBorrowSingleIdentityExportPlan(const lfs::vis::SceneManager& scene_manager,
                                                                      const std::vector<std::string>& node_names) {
        BorrowExportPlan plan;
        if (node_names.size() != 1)
            return plan;

        const auto& scene = scene_manager.getScene();
        const auto* const node = scene.getNode(node_names.front());
        if (!node || node->type != core::NodeType::SPLAT || !node->model)
            return plan;

        if (node->model->has_deleted_mask())
            return plan;

        if (node->uuid == scene.getTrainingModelNodeUuid()) {
            const auto* const trainer_manager = scene_manager.getTrainerManager();
            const auto* const trainer = trainer_manager ? trainer_manager->getTrainer() : nullptr;
            if (trainer && trainer->is_running() && !trainer->is_paused())
                return plan;
            if (trainer)
                plan.model_mutex = &trainer->getRenderMutex();
        }

        plan.storage_mode = core::Scene::MergeStorageMode::BorrowSingleIdentity;
        return plan;
    }

    struct ColmapExportSnapshot {
        std::filesystem::path source_path;
        std::vector<io::ColmapCameraWriteData> cameras;
        std::shared_ptr<const core::PointCloud> point_cloud;
        glm::mat4 point_cloud_transform{1.0f};
    };

    [[nodiscard]] std::expected<ColmapExportSnapshot, std::string>
    makeColmapExportSnapshot(const lfs::vis::SceneManager& scene_manager) {
        if (!scene_manager.hasDataset()) {
            return std::unexpected(LOC(lichtfeld::Strings::Runtime::COLMAP_REQUIRES_DATASET));
        }

        const auto source_path = scene_manager.getDatasetPath();
        if (source_path.empty()) {
            return std::unexpected(LOC(lichtfeld::Strings::Runtime::COLMAP_REQUIRES_SOURCE_PATH));
        }

        const auto& scene = scene_manager.getScene();
        auto cameras = scene.getAllCameras();
        if (cameras.empty()) {
            return std::unexpected(LOC(lichtfeld::Strings::Runtime::COLMAP_REQUIRES_CAMERAS));
        }

        ColmapExportSnapshot snapshot;
        snapshot.source_path = source_path;
        snapshot.cameras.reserve(cameras.size());
        for (const auto& camera : cameras) {
            if (!camera)
                continue;
            snapshot.cameras.push_back(io::ColmapCameraWriteData{
                .camera = camera,
                .data_world_transform = scene.getCameraSceneTransformByUid(camera->uid()).value_or(glm::mat4(1.0f)),
            });
        }

        for (const auto* node : scene.getNodes()) {
            if (!node || node->type != core::NodeType::POINTCLOUD || !node->point_cloud ||
                !scene.isNodeEffectivelyVisible(node->id)) {
                continue;
            }
            snapshot.point_cloud = node->point_cloud;
            snapshot.point_cloud_transform = scene.getWorldTransform(node->id);
            break;
        }

        // If no live POINTCLOUD node exists (e.g. the splat model replaced it
        // once training started), the export will fall through to the source
        // COLMAP points3D file. Those points are in the original COLMAP frame
        // and the writer otherwise leaves them untransformed, which makes the
        // exported cameras (which DO get a world transform) inconsistent with
        // the points after the user reorients the scene. Anchor the point
        // transform to the DATASET node so points and cameras share the same
        // user-applied orientation.
        if (!snapshot.point_cloud) {
            for (const auto* node : scene.getNodes()) {
                if (node && node->type == core::NodeType::DATASET) {
                    snapshot.point_cloud_transform = scene.getWorldTransform(node->id);
                    break;
                }
            }
        }

        return snapshot;
    }

    template <typename F>
    auto postToViewerAndWait(VisualizerImpl* viewer, F&& fn) -> std::invoke_result_t<F> {
        using ResultT = std::invoke_result_t<F>;
        const std::string shutdown_error = LOC(lichtfeld::Strings::Runtime::VIEWER_SHUTTING_DOWN);
        const std::string task_error = LOC(lichtfeld::Strings::Runtime::VIEWER_WORK_FAILED);

        if (viewer->isOnViewerThread()) {
            if (!viewer->acceptsPostedWork()) {
                return std::unexpected(std::string(shutdown_error));
            }
            try {
                return std::invoke(std::forward<F>(fn));
            } catch (const std::exception& e) {
                return std::unexpected(std::format("{}: {}", task_error, e.what()));
            } catch (...) {
                return std::unexpected(std::string(task_error));
            }
        }

        auto task = std::make_shared<std::decay_t<F>>(std::forward<F>(fn));
        auto promise = std::make_shared<std::promise<ResultT>>();
        auto completed = std::make_shared<std::atomic_bool>(false);
        auto future = promise->get_future();

        auto finish_with_value = [promise, completed](ResultT value) mutable {
            if (!completed->exchange(true)) {
                promise->set_value(std::move(value));
            }
        };
        auto finish_with_exception = [promise, completed](std::exception_ptr error) {
            if (!completed->exchange(true)) {
                promise->set_exception(std::move(error));
            }
        };

        const bool posted = viewer->postWork(VisualizerImpl::WorkItem{
            .run =
                [task, finish_with_value, finish_with_exception]() mutable {
                    try {
                        finish_with_value(std::invoke(*task));
                    } catch (...) {
                        finish_with_exception(std::current_exception());
                    }
                },
            .cancel =
                [finish_with_value, shutdown_error]() mutable {
                    finish_with_value(std::unexpected(std::string(shutdown_error)));
                }});

        if (!posted) {
            return std::unexpected(std::string(shutdown_error));
        }

        try {
            return future.get();
        } catch (const std::exception& e) {
            return std::unexpected(std::format("{}: {}", task_error, e.what()));
        } catch (...) {
            return std::unexpected(std::string(task_error));
        }
    }

    rendering::ViewportData makeVideoExportViewport(const lfs::sequencer::CameraState& cam_state,
                                                    const RenderSettings& render_settings,
                                                    const int width,
                                                    const int height) {
        rendering::ViewportData viewport;
        viewport.rotation = glm::mat3_cast(cam_state.rotation);
        viewport.translation = cam_state.position;
        viewport.size = {width, height};
        viewport.focal_length_mm = cam_state.focal_length_mm;
        viewport.orthographic = render_settings.orthographic;
        viewport.ortho_scale = render_settings.ortho_scale;
        return viewport;
    }

    rendering::FrameView makeVideoExportFrameView(const lfs::sequencer::CameraState& cam_state,
                                                  const RenderSettings& render_settings,
                                                  const int width,
                                                  const int height) {
        return rendering::FrameView{
            .rotation = glm::mat3_cast(cam_state.rotation),
            .translation = cam_state.position,
            .size = {width, height},
            .focal_length_mm = cam_state.focal_length_mm,
            .intrinsics_override = std::nullopt,
            .far_plane = render_settings.depth_clip_enabled ? render_settings.depth_clip_far
                                                            : rendering::DEFAULT_FAR_PLANE,
            .orthographic = render_settings.orthographic,
            .ortho_scale = render_settings.ortho_scale,
            .background_color = render_settings.background_color};
    }

    struct VideoExportEnvironmentState {
        std::string cached_environment_path_value;
        std::filesystem::path cached_environment_resolved_path;
    };

    struct VideoExportMeshRendererState {
        std::unique_ptr<MeshOffscreenRenderer> renderer;

        void shutdown() {
            if (renderer) {
                renderer->shutdown();
                renderer.reset();
            }
        }
    };

    [[nodiscard]] bool isValidVideoExportMeshLayer(
        const MeshLayer& layer,
        const int width,
        const int height) {
        return layer.rgba.is_valid() && layer.rgba.dtype() == lfs::core::DataType::Float32 &&
               layer.rgba.ndim() == 3 && layer.rgba.size(0) == 4u &&
               layer.rgba.size(1) == static_cast<std::size_t>(height) &&
               layer.rgba.size(2) == static_cast<std::size_t>(width) &&
               layer.view_depth.is_valid() &&
               layer.view_depth.dtype() == lfs::core::DataType::Float32 &&
               layer.view_depth.ndim() == 2 &&
               layer.view_depth.size(0) == static_cast<std::size_t>(height) &&
               layer.view_depth.size(1) == static_cast<std::size_t>(width);
    }

    [[nodiscard]] std::filesystem::path resolveVideoExportEnvironmentPath(
        VideoExportEnvironmentState& state,
        const std::string& path_value) {
        if (path_value == state.cached_environment_path_value) {
            return state.cached_environment_resolved_path;
        }

        state.cached_environment_path_value = path_value;
        const std::filesystem::path requested(path_value);
        if (requested.empty() || requested.is_absolute()) {
            state.cached_environment_resolved_path = requested;
            return state.cached_environment_resolved_path;
        }

        try {
            state.cached_environment_resolved_path = getAssetPath(path_value);
        } catch (const std::exception&) {
            state.cached_environment_resolved_path = lfs::core::getAssetsDir() / requested;
        }
        return state.cached_environment_resolved_path;
    }

    [[nodiscard]] const VideoExportCropBoxSnapshot* activeVideoExportPointCloudCropBox(
        const VideoExportSceneSnapshot& snapshot) {
        const VideoExportCropBoxSnapshot* selected = nullptr;
        if (snapshot.selected_cropbox_index >= 0) {
            const size_t idx = static_cast<size_t>(snapshot.selected_cropbox_index);
            if (idx < snapshot.cropboxes.size() && snapshot.cropboxes[idx].has_data &&
                snapshot.cropboxes[idx].data.enabled) {
                selected = &snapshot.cropboxes[idx];
            }
        }
        if (selected) {
            return selected;
        }

        const VideoExportCropBoxSnapshot* single = nullptr;
        for (const auto& cb : snapshot.cropboxes) {
            if (!cb.has_data || !cb.data.enabled) {
                continue;
            }
            if (single) {
                return nullptr;
            }
            single = &cb;
        }
        return single;
    }

    [[nodiscard]] const VideoExportEllipsoidSnapshot* activeVideoExportPointCloudEllipsoid(
        const VideoExportSceneSnapshot& snapshot) {
        if (!snapshot.active_ellipsoid || !snapshot.active_ellipsoid->data.enabled) {
            return nullptr;
        }
        return &*snapshot.active_ellipsoid;
    }

    void applyVideoExportPointCloudFilters(rendering::PointCloudFilterState& filters,
                                           const VideoExportSceneSnapshot& snapshot,
                                           const RenderSettings& render_settings) {
        if (const auto* const cb = activeVideoExportPointCloudCropBox(snapshot)) {
            filters.crop_box = rendering::BoundingBox{
                .min = cb->data.min,
                .max = cb->data.max,
                .transform = glm::inverse(cb->world_transform)};
            filters.crop_ellipsoid.reset();
            filters.crop_inverse = cb->data.inverse;
            filters.crop_desaturate = render_settings.desaturate_cropping;
            return;
        }

        if (const auto* const el = activeVideoExportPointCloudEllipsoid(snapshot)) {
            filters.crop_ellipsoid = rendering::Ellipsoid{
                .radii = el->data.radii,
                .transform = glm::inverse(el->world_transform)};
            filters.crop_box.reset();
            filters.crop_inverse = el->data.inverse;
            filters.crop_desaturate = render_settings.desaturate_cropping;
        }
    }

    rendering::MeshRenderOptions makeVideoExportMeshOptions(const RenderSettings& render_settings,
                                                            const bool any_selected,
                                                            const bool is_selected) {
        return rendering::MeshRenderOptions{
            .wireframe_overlay = render_settings.mesh_wireframe,
            .wireframe_color = render_settings.mesh_wireframe_color,
            .wireframe_width = render_settings.mesh_wireframe_width,
            .light_dir = render_settings.mesh_light_dir,
            .light_intensity = render_settings.mesh_light_intensity,
            .ambient = render_settings.mesh_ambient,
            .backface_culling = render_settings.mesh_backface_culling,
            .shadow_enabled = render_settings.mesh_shadow_enabled,
            .shadow_map_resolution = render_settings.mesh_shadow_resolution,
            .is_emphasized = is_selected,
            .dim_non_emphasized = render_settings.desaturate_unselected && any_selected,
            .flash_intensity = 0.0f,
            .background_color = render_settings.background_color,
            .transparent_background = environmentBackgroundEnabled(render_settings)};
    }

    SceneRenderState makeVideoExportGaussianSceneState(const VideoExportSceneSnapshot& snapshot) {
        SceneRenderState state;
        state.combined_model = snapshot.combined_model.get();
        state.model_transforms = snapshot.model_transforms;
        state.transform_indices = snapshot.transform_indices;
        state.selection_mask = snapshot.selection_mask;
        state.selected_node_mask = snapshot.selected_node_mask;
        state.node_visibility_mask = snapshot.node_visibility_mask;
        state.selected_cropbox_index = snapshot.selected_cropbox_index;
        state.has_selection = state.selection_mask && state.selection_mask->is_valid();
        state.visible_splat_count = snapshot.model_transforms.size();

        state.cropboxes.reserve(snapshot.cropboxes.size());
        for (const auto& cb : snapshot.cropboxes) {
            state.cropboxes.push_back(lfs::core::Scene::RenderableCropBox{
                .node_id = cb.node_id,
                .parent_splat_id = cb.parent_splat_id,
                .parent_node_index = cb.parent_node_index,
                .data = cb.has_data ? &cb.data : nullptr,
                .world_transform = cb.world_transform,
                .local_transform = glm::mat4(1.0f),
                .effectively_visible = true,
            });
        }

        if (snapshot.active_ellipsoid) {
            const auto& el = *snapshot.active_ellipsoid;
            state.ellipsoids.push_back(lfs::core::Scene::RenderableEllipsoid{
                .node_id = el.node_id,
                .parent_splat_id = el.parent_splat_id,
                .parent_node_index = el.parent_node_index,
                .data = &el.data,
                .world_transform = el.world_transform,
                .local_transform = glm::mat4(1.0f),
                .effectively_visible = true,
            });
        }
        return state;
    }

    std::expected<lfs::core::Tensor, std::string> makeGaussianPreviewVideoFrame(
        const std::shared_ptr<lfs::core::Tensor>& image) {
        if (!image || !image->is_valid() || image->ndim() != 3) {
            return std::unexpected(LOC(lichtfeld::Strings::Runtime::RENDERED_GAUSSIAN_INVALID));
        }
        if (image->size(0) <= 0 || image->size(1) <= 0 ||
            (image->size(2) != 3 && image->size(2) != 4)) {
            return std::unexpected(LOC(lichtfeld::Strings::Runtime::RENDERED_GAUSSIAN_SHAPE_INVALID));
        }

        auto frame = *image;
        // Preview readbacks currently arrive as uint8 HWC and need normalization;
        // float preview tensors are expected to already be in normalized color space.
        const bool normalize_uint8 = frame.dtype() == lfs::core::DataType::UInt8;
        if (frame.dtype() != lfs::core::DataType::Float32) {
            frame = frame.to(lfs::core::DataType::Float32);
        }
        if (normalize_uint8) {
            frame = frame / 255.0f;
        }
        frame = frame.permute({2, 0, 1}).contiguous();
        if (frame.device() != lfs::core::Device::CUDA) {
            frame = frame.cuda();
        }
        return frame.contiguous();
    }

    rendering::FrameMetadata makeVideoExportFrameMetadata(const rendering::FrameView& frame_view) {
        return rendering::FrameMetadata{
            .valid = true,
            .far_plane = frame_view.far_plane,
            .orthographic = frame_view.orthographic};
    }

    std::expected<lfs::core::Tensor, std::string> renderVideoExportFrame(
        RenderingManager& rendering_manager,
        rendering::RenderingEngine& engine,
        VideoExportEnvironmentState& environment_state,
        VideoExportMeshRendererState* mesh_renderer_state,
        VulkanContext* vulkan_context,
        const VideoExportSceneSnapshot& snapshot,
        const RenderSettings& render_settings,
        const lfs::sequencer::CameraState& cam_state,
        const int width,
        const int height) {
        const auto viewport = makeVideoExportViewport(cam_state, render_settings, width, height);
        const auto frame_view = makeVideoExportFrameView(cam_state, render_settings, width, height);
        const bool render_environment = environmentBackgroundEnabled(render_settings);
        const bool requires_composite_pass = render_environment || !snapshot.meshes.empty();

        std::optional<rendering::GpuFrame> primary_frame;

        if (snapshot.combined_model && snapshot.combined_model->size() > 0) {
            if (render_settings.point_cloud_mode) {
                rendering::PointCloudRenderRequest request{
                    .frame_view = frame_view,
                    .render =
                        {.scaling_modifier = render_settings.scaling_modifier,
                         .voxel_size = render_settings.voxel_size,
                         .equirectangular = render_settings.equirectangular},
                    .scene =
                        {.model_transforms = &snapshot.model_transforms,
                         .transform_indices = snapshot.transform_indices,
                         .node_visibility_mask = snapshot.node_visibility_mask},
                    .filters = {},
                    .overlay = {},
                    .transparent_background = render_environment};
                applyVideoExportPointCloudFilters(request.filters, snapshot, render_settings);

                if (!requires_composite_pass) {
                    auto render_result = engine.renderPointCloudImage(*snapshot.combined_model, request);
                    if (!render_result || !render_result->image) {
                        return std::unexpected(render_result ? LOC(lichtfeld::Strings::Runtime::RENDERED_POINT_CLOUD_INVALID)
                                                             : render_result.error());
                    }
                    return *render_result->image;
                }

                auto render_result = engine.renderPointCloudGpuFrame(*snapshot.combined_model, request);
                if (!render_result || !render_result->valid()) {
                    return std::unexpected(render_result ? LOC(lichtfeld::Strings::Runtime::RENDERED_POINT_CLOUD_INVALID)
                                                         : render_result.error());
                }
                primary_frame = std::move(*render_result);
            } else {
                auto scene_state = makeVideoExportGaussianSceneState(snapshot);
                const auto camera_rotation = glm::mat3_cast(cam_state.rotation);
                auto preview_image = render_environment
                                         ? rendering_manager.renderPreviewImageRgba8(
                                               *snapshot.combined_model,
                                               std::move(scene_state),
                                               camera_rotation,
                                               cam_state.position,
                                               cam_state.focal_length_mm,
                                               width,
                                               height)
                                         : rendering_manager.renderPreviewImage(
                                               *snapshot.combined_model,
                                               std::move(scene_state),
                                               camera_rotation,
                                               cam_state.position,
                                               cam_state.focal_length_mm,
                                               width,
                                               height);
                auto video_frame = makeGaussianPreviewVideoFrame(preview_image);
                if (!video_frame) {
                    return std::unexpected(video_frame.error());
                }

                if (!requires_composite_pass) {
                    return std::move(*video_frame);
                }

                auto frame_image = std::make_shared<lfs::core::Tensor>(std::move(*video_frame));
                auto materialized = engine.materializeGpuFrame(
                    frame_image,
                    makeVideoExportFrameMetadata(frame_view),
                    {width, height});
                if (!materialized || !materialized->valid()) {
                    return std::unexpected(materialized ? LOC(lichtfeld::Strings::Runtime::RENDERED_GAUSSIAN_INVALID)
                                                        : materialized.error());
                }
                primary_frame = std::move(*materialized);
            }
        } else if (snapshot.point_cloud && snapshot.point_cloud->size() > 0) {
            const std::vector<glm::mat4> point_cloud_transforms = {snapshot.point_cloud_transform};
            rendering::PointCloudRenderRequest request{
                .frame_view = frame_view,
                .render =
                    {.scaling_modifier = render_settings.scaling_modifier,
                     .voxel_size = render_settings.voxel_size,
                     .equirectangular = render_settings.equirectangular},
                .scene =
                    {.model_transforms = &point_cloud_transforms,
                     .transform_indices = nullptr,
                     .node_visibility_mask = {}},
                .filters = {},
                .overlay = {},
                .transparent_background = render_environment};
            applyVideoExportPointCloudFilters(request.filters, snapshot, render_settings);

            auto render_result = engine.renderPointCloudGpuFrame(*snapshot.point_cloud, request);
            if (!render_result || !render_result->valid()) {
                return std::unexpected(render_result ? LOC(lichtfeld::Strings::Runtime::RENDERED_POINT_CLOUD_INVALID)
                                                     : render_result.error());
            }

            if (!requires_composite_pass) {
                auto readback_result = engine.readbackGpuFrameColor(*render_result);
                if (!readback_result || !*readback_result) {
                    return std::unexpected(readback_result ? LOC(lichtfeld::Strings::Runtime::RENDERED_POINT_CLOUD_INVALID)
                                                           : readback_result.error());
                }
                return *(*readback_result);
            }

            primary_frame = std::move(*render_result);
        }

        if (!requires_composite_pass) {
            return std::unexpected(LOC(lichtfeld::Strings::Runtime::VIDEO_FRAME_MISSING));
        }

        const bool any_selected = std::any_of(snapshot.meshes.begin(), snapshot.meshes.end(),
                                              [](const auto& mesh) { return mesh.is_selected; }) ||
                                  std::any_of(snapshot.selected_node_mask.begin(),
                                              snapshot.selected_node_mask.end(),
                                              [](const bool selected) { return selected; });

        std::optional<MeshLayer> prerendered_meshes;
        if (!snapshot.meshes.empty()) {
            if (mesh_renderer_state == nullptr) {
                return std::unexpected(
                    "Failed to render GPU mesh layer: renderer state is unavailable");
            }
            if (vulkan_context == nullptr) {
                return std::unexpected(
                    "Failed to render GPU mesh layer: no Vulkan context is available");
            }

            try {
                if (!mesh_renderer_state->renderer) {
                    mesh_renderer_state->renderer = std::make_unique<MeshOffscreenRenderer>();
                }

                const glm::mat4 projection = viewport.getProjectionMatrix();
                VulkanMeshPassParams mesh_params{
                    .view_projection = projection * viewport.getViewMatrix(),
                    .camera_position = viewport.translation,
                    .items = {},
                    .frame_slot = 0,
                    .draw_group = 0,
                    .draw_group_count = 1,
                };
                mesh_params.items.reserve(snapshot.meshes.size());
                for (const auto& mesh_snapshot : snapshot.meshes) {
                    if (!mesh_snapshot.mesh) {
                        return std::unexpected(
                            "Failed to render GPU mesh layer: mesh snapshot is invalid");
                    }

                    const auto options = makeVideoExportMeshOptions(
                        render_settings, any_selected, mesh_snapshot.is_selected);
                    mesh_params.items.push_back(VulkanMeshDrawItem{
                        .mesh = mesh_snapshot.mesh.get(),
                        .model = mesh_snapshot.transform,
                        .light_dir = options.light_dir,
                        .light_intensity = options.light_intensity,
                        .ambient = options.ambient,
                        .backface_culling = options.backface_culling,
                        .is_emphasized = options.is_emphasized,
                        .dim_non_emphasized = options.dim_non_emphasized,
                        .flash_intensity = options.flash_intensity,
                        .wireframe_overlay = options.wireframe_overlay,
                        .wireframe_color = options.wireframe_color,
                        .wireframe_width = options.wireframe_width,
                        .shadow_enabled = options.shadow_enabled,
                        .shadow_map_resolution = options.shadow_map_resolution,
                    });
                }

                auto mesh_layer = mesh_renderer_state->renderer->render(
                    *vulkan_context, mesh_params, projection, width, height);
                if (!mesh_layer) {
                    return std::unexpected(std::format(
                        "Failed to render GPU mesh layer: {}", mesh_layer.error()));
                }
                if (!isValidVideoExportMeshLayer(*mesh_layer, width, height)) {
                    return std::unexpected(
                        "Failed to render GPU mesh layer: renderer returned an invalid mesh layer");
                }
                prerendered_meshes = std::move(*mesh_layer);
            } catch (const std::exception& error) {
                return std::unexpected(std::format(
                    "Failed to render GPU mesh layer: {}", error.what()));
            } catch (...) {
                return std::unexpected(
                    "Failed to render GPU mesh layer: an unknown exception occurred");
            }
        }

        rendering::VideoCompositeFrameRequest composite_request{
            .viewport = viewport,
            .frame_view = frame_view,
            .background_color = render_settings.background_color,
            .environment =
                {.enabled = render_environment,
                 .map_path = render_environment
                                 ? resolveVideoExportEnvironmentPath(
                                       environment_state, render_settings.environment_map_path)
                                 : std::filesystem::path{},
                 .exposure = render_settings.environment_exposure,
                 .rotation_degrees = render_settings.environment_rotation_degrees,
                 .equirectangular = render_settings.equirectangular},
            .prerendered_meshes = prerendered_meshes ? &*prerendered_meshes : nullptr,
        };
        return engine.renderVideoCompositeFrame(primary_frame, composite_request);
    }

    AsyncTaskManager::AsyncTaskManager(VisualizerImpl* viewer)
        : viewer_(viewer),
          jobs_(viewer->jobs()) {}

    AsyncTaskManager::~AsyncTaskManager() {
        shutdown();
    }

    float AsyncTaskManager::jobProgress(
        const JobHandle handle) const {
        const auto snapshot = jobs_.peek(handle);
        return snapshot ? snapshot->progress : 0.0F;
    }

    std::string AsyncTaskManager::jobStage(
        const JobHandle handle) const {
        const auto snapshot = jobs_.peek(handle);
        return snapshot ? snapshot->stage : std::string{};
    }

    std::string AsyncTaskManager::jobError(
        const JobHandle handle) const {
        const auto snapshot = jobs_.peek(handle);
        return snapshot ? snapshot->error : std::string{};
    }

    std::string AsyncTaskManager::jobOutcome(
        const JobHandle handle) const {
        const auto snapshot = jobs_.peek(handle);
        if (!snapshot) {
            return "idle";
        }
        switch (snapshot->status) {
        case JobStatus::Initialized:
        case JobStatus::Running:
        case JobStatus::CompletionPending:
            return snapshot->cancel_requested
                       ? "cancelling"
                       : "running";
        case JobStatus::Completed:
            return "completed";
        case JobStatus::Failed:
            return "failed";
        case JobStatus::Canceled:
            return "cancelled";
        }
        return "idle";
    }

    bool AsyncTaskManager::beginJob(
        JobHandle& handle, const JobType type,
        std::string stage) {
        if (handle) {
            const auto prior = jobs_.update(handle);
            if (prior && prior->running()) {
                return false;
            }
            jobs_.free(handle);
            handle = {};
        }
        const auto created =
            jobs_.init(type, std::move(stage));
        if (!created) {
            return false;
        }
        handle = *created;
        return true;
    }

    void AsyncTaskManager::settlePendingJobs() {
        const auto settle =
            [this](const JobHandle handle) {
                const auto snapshot =
                    jobs_.update(handle);
                if (!snapshot ||
                    snapshot->status !=
                        JobStatus::CompletionPending) {
                    return false;
                }
                if (snapshot->worker_canceled) {
                    jobs_.canceled(handle);
                } else if (!snapshot->error.empty()) {
                    jobs_.failed(
                        handle, snapshot->error,
                        snapshot->stage);
                } else {
                    jobs_.completed(handle);
                }
                return true;
            };
        if (settle(export_state_.job)) {
            publishExportState();
        }
        if (settle(video_export_state_.job)) {
            publishVideoExportOverlayState();
        }
    }

    void AsyncTaskManager::resetVideoExportEnvironmentState() {
        video_export_environment_state_.reset();
    }

    void AsyncTaskManager::resetVideoExportMeshRendererState() {
        if (video_export_mesh_renderer_state_) {
            video_export_mesh_renderer_state_->shutdown();
            video_export_mesh_renderer_state_.reset();
        }
    }

    void AsyncTaskManager::shutdown() {
        if (isExporting())
            cancelExport();
        if (export_state_.thread && export_state_.thread->joinable())
            export_state_.thread->join();
        export_state_.thread.reset();

        if (isExportingVideo())
            cancelVideoExport();
        if (video_export_state_.thread && video_export_state_.thread->joinable())
            video_export_state_.thread->join();
        video_export_state_.thread.reset();
        if (viewer_ && viewer_->isOnViewerThread()) {
            resetVideoExportEnvironmentState();
            resetVideoExportMeshRendererState();
        }

        if (import_state_.thread) {
            import_state_.thread->request_stop();
            if (import_state_.thread->joinable())
                import_state_.thread->join();
            import_state_.thread.reset();
        }
        cancelImportCompletionDismiss();

        mesh2splat_state_.pending.store(false);
        if (isMesh2SplatActive()) {
            jobs_.canceled(mesh2splat_state_.job);
        }

        if (isSplatSimplifyActive())
            cancelSplatSimplify();
        if (splat_simplify_state_.thread && splat_simplify_state_.thread->joinable())
            splat_simplify_state_.thread->join();
        splat_simplify_state_.thread.reset();
        settlePendingJobs();
    }

    void AsyncTaskManager::setupEvents() {
        using namespace lfs::core::events;

        cmd::LoadFile::when([this](const auto& cmd) {
            if (!cmd.is_dataset)
                return;
            if (viewer_->preflightLoadFileWipe(cmd))
                return;
            if (viewer_->deferLoadFileForTraining(cmd))
                return;
            if (!viewer_->resetUntitledSessionForReplaceLoad())
                return;
            auto* const data_loader = viewer_->getDataLoader();
            if (!data_loader) {
                LOG_ERROR("LoadFile: no data loader");
                return;
            }
            const auto output_path =
                cmd.output_path.empty()
                    ? lfs::core::param::default_dataset_output_path(cmd.path)
                    : cmd.output_path;
            lfs::core::param::TrainingParameters params;
            if (auto* const param_mgr = viewer_->getParameterManager();
                param_mgr && param_mgr->ensureLoaded()) {
                params = param_mgr->createForDataset(cmd.path, output_path);
            } else {
                params = data_loader->getParameters();
                params.dataset.data_path = cmd.path;
                params.dataset.output_path = output_path;
            }
            params.init_path = std::nullopt;
            params.resume_checkpoint = std::nullopt;
            params.dataset.output_path = output_path;
            if (!cmd.init_path.empty())
                params.init_path = lfs::core::path_to_utf8(cmd.init_path);
            if (!cmd.centralize_dataset.empty())
                params.dataset.centralize_dataset = cmd.centralize_dataset;
            if (cmd.max_width.has_value() && *cmd.max_width >= 0)
                params.dataset.max_width = *cmd.max_width;
            if (cmd.min_track_length.has_value() && *cmd.min_track_length >= 0)
                params.dataset.min_track_length = *cmd.min_track_length;
            if (auto* const param_mgr = viewer_->getParameterManager()) {
                param_mgr->getDatasetConfig() = params.dataset;
            }
            data_loader->setParameters(params);
            import_state_.apply_auto_crop.store(cmd.apply_auto_crop);
            startAsyncImport(cmd.path, params);
        });

        state::DatasetLoadStarted::when([this](const auto& e) {
            if (isImporting())
                return;
            cancelImportCompletionDismiss();
            if (!beginJob(
                    import_state_.job, JobType::Import,
                    LOC(lichtfeld::Strings::Runtime::TASK_INITIALIZING))) {
                return;
            }
            {
                const std::lock_guard lock(import_state_.mutex);
                import_state_.path = e.path;
                import_state_.num_images = 0;
                import_state_.num_points = 0;
                import_state_.success = false;
                import_state_.dataset_type = getDatasetTypeName(e.path);
            }
            publishImportOverlayState();
        });

        state::DatasetLoadProgress::when([this](const auto& e) {
            jobs_.report(
                import_state_.job,
                e.progress / 100.0F, e.step);
            publishImportOverlayState();
        });

        state::DatasetLoadCompleted::when([this](const auto& e) {
            // Consume the flag exchange-style so the auto-crop fires at most
            // once per load — DatasetLoadCompleted is also emitted from the
            // scene_manager path, which bypasses the import_state_ updates
            // below.
            if (e.success && import_state_.apply_auto_crop.exchange(false))
                applyAutoCropToLoadedScene();

            if (import_state_.show_completion.load())
                return;
            const auto job =
                jobs_.update(import_state_.job);
            if (!job || !job->running()) {
                return;
            }
            {
                const std::lock_guard lock(import_state_.mutex);
                import_state_.success = e.success;
                import_state_.num_images = e.num_images;
                import_state_.num_points = e.num_points;
                import_state_.completion_time = std::chrono::steady_clock::now();
            }
            if (e.success) {
                jobs_.report(
                    import_state_.job, 1.0F,
                    LOC(lichtfeld::Strings::Runtime::TASK_COMPLETE));
                jobs_.completed(import_state_.job);
            } else {
                jobs_.failed(
                    import_state_.job,
                    e.error.value_or("Dataset load failed"));
            }
            import_state_.show_completion.store(true);
            if (e.success)
                scheduleImportCompletionDismiss();
            publishImportOverlayState();
        });

        cmd::SequencerExportVideo::when([this](const auto& evt) {
            const auto path = evt.path.empty()
                                  ? SaveMp4FileDialog("camera_path")
                                  : std::filesystem::path(evt.path);
            if (path.empty())
                return;

            io::video::VideoExportOptions options;
            options.width = evt.width;
            options.height = evt.height;
            options.framerate = evt.framerate;
            options.crf = evt.crf;
            if (evt.include_provenance) {
                if (const auto* const scene_manager = viewer_->getSceneManager()) {
                    options.provenance = make_gui_export_stamp(*scene_manager);
                } else {
                    auto stamp = core::make_provenance_stamp();
                    populate_project_identity(stamp);
                    options.provenance = std::move(stamp);
                }
            } else {
                options.provenance = core::make_minimal_provenance_stamp();
            }
            startVideoExport(path, options);
        });
    }

    void AsyncTaskManager::pollImportCompletion() {
        checkAsyncImportCompletion();
        settlePendingJobs();
    }

    bool AsyncTaskManager::hasPendingMainThreadCompletions() const {
        return import_state_.load_complete.load(std::memory_order_acquire) ||
               mesh2splat_state_.pending.load(std::memory_order_acquire) ||
               splat_simplify_state_.apply_pending.load(std::memory_order_acquire) ||
               splat_simplify_state_.completed.load(std::memory_order_acquire) ||
               (jobs_.peek(export_state_.job) &&
                jobs_.peek(export_state_.job)->status ==
                    JobStatus::CompletionPending) ||
               (jobs_.peek(video_export_state_.job) &&
                jobs_.peek(video_export_state_.job)->status ==
                    JobStatus::CompletionPending);
    }

    void AsyncTaskManager::performExport(ExportFormat format, const std::filesystem::path& path,
                                         const std::vector<std::string>& node_names, int sh_degree,
                                         bool rad_flip_y,
                                         bool rad_streamable,
                                         int spz_version,
                                         bool include_provenance) {
        if (isExporting())
            return;

        if (viewer_) {
            if (viewer_->projectContainsEmbeddedSecrets()) {
                lfs::ErrorBus::instance().publish(
                    lfs::ErrorNotification{
                        .error = lfs::make_error(
                                     lfs::ErrorInit{
                                         .code =
                                             lfs::ErrorCode::
                                                 FailedPrecondition,
                                         .domain =
                                             lfs::ErrorDomain::App,
                                         .severity =
                                             lfs::Severity::Warning,
                                         .retryability =
                                             lfs::Retryability::
                                                 NotRetryable,
                                         .operation_id = {},
                                         .user_message =
                                             "This project contains unsaved editor buffers marked as potentially secret.",
                                         .detail =
                                             "Review the embedded EDTR content before sharing the project or derived exports.",
                                         .detection =
                                             LFS_SOURCE_SITE_CURRENT(),
                                         .fields = {},
                                         .native = std::nullopt,
                                     })
                                     .with_context(
                                         error_op::kExport,
                                         LFS_SOURCE_SITE_CURRENT()),
                        .surface =
                            lfs::ErrorSurface::Toast,
                        .actions = {},
                        .operation_id =
                            lfs::OperationId::
                                generate(),
                    });
            }
        }

        if (format == ExportFormat::COLMAP) {
            startColmapExport(path);
            return;
        }

        auto* const scene_manager = viewer_->getSceneManager();
        if (!scene_manager) {
            publishExportFailureState(format, path, LOC(lichtfeld::Strings::Runtime::SCENE_MANAGER_UNAVAILABLE));
            return;
        }
        if (node_names.empty()) {
            publishExportFailureState(format, path, LOC(lichtfeld::Strings::Runtime::NO_MODEL_SELECTED));
            return;
        }

        const auto& scene = scene_manager->getScene();
        std::vector<ExportSplatSource> splats;
        splats.reserve(node_names.size());
        for (const auto& name : node_names) {
            const auto* node = scene.getNode(name);
            if (node && node->type == core::NodeType::SPLAT && node->model) {
                splats.push_back(ExportSplatSource{
                    .data = node->model.get(),
                    .transform = scene_coords::nodeDataWorldTransform(scene, node->id)});
            }
        }
        if (splats.empty()) {
            publishExportFailureState(format, path, LOC(lichtfeld::Strings::Runtime::NO_SPLAT_DATA));
            return;
        }

        auto borrow_plan = makeBorrowSingleIdentityExportPlan(*scene_manager, node_names);

        auto provenance = include_provenance ? make_gui_export_stamp(*scene_manager)
                                             : core::make_minimal_provenance_stamp();

        startAsyncExport(format,
                         path,
                         std::move(splats),
                         sh_degree,
                         borrow_plan.storage_mode == core::Scene::MergeStorageMode::BorrowSingleIdentity,
                         borrow_plan.model_mutex,
                         rad_flip_y,
                         rad_streamable,
                         spz_version,
                         std::move(provenance));
    }

    void AsyncTaskManager::startColmapExport(const std::filesystem::path& path) {
        if (isExporting())
            return;

        auto* const scene_manager = viewer_->getSceneManager();
        if (!scene_manager) {
            std::string error = LOC(lichtfeld::Strings::Runtime::SCENE_MANAGER_NOT_INITIALIZED);
            LOG_ERROR("COLMAP export failed: {}", error);
            publishExportFailureState(ExportFormat::COLMAP, path, std::move(error));
            return;
        }

        auto snapshot_result = makeColmapExportSnapshot(*scene_manager);
        if (!snapshot_result) {
            LOG_ERROR("COLMAP export failed: {}", snapshot_result.error());
            publishExportFailureState(ExportFormat::COLMAP, path, snapshot_result.error());
            lfs::core::events::state::ExportFailed{.error = snapshot_result.error()}.emit();
            return;
        }

        if (!beginJob(
                export_state_.job, JobType::Export,
                LOC(lichtfeld::Strings::Runtime::TASK_STARTING))) {
            return;
        }
        {
            const std::lock_guard lock(export_state_.mutex);
            export_state_.format = ExportFormat::COLMAP;
            export_state_.path = path;
        }
        publishExportState();

        LOG_INFO("COLMAP export started: {}", lfs::core::path_to_utf8(path));

        const auto job = export_state_.job;
        export_state_.thread.emplace(
            [this, job, path, snapshot = std::move(*snapshot_result)](std::stop_token stop_token) mutable {
                jobs_.work(job);
                bool success = false;
                bool cancelled = false;
                std::string error_msg;

                auto update_stage = [this, job](float progress, const std::string& stage) {
                    jobs_.report(job, progress, stage);
                    publishExportState();
                    if (auto* window_manager = services().windowOrNull()) {
                        window_manager->wakeEventLoop();
                    }
                };

                try {
                    if (stop_token.stop_requested() || jobs_.cancelRequested(job)) {
                        cancelled = true;
                        error_msg = LOC(lichtfeld::Strings::Runtime::EXPORT_CANCELLED);
                    } else {
                        update_stage(0.1f, LOC(lichtfeld::Strings::Runtime::EXPORT_WRITING_COLMAP));
                        auto result = io::write_colmap_reconstruction(
                            snapshot.source_path,
                            path,
                            snapshot.cameras,
                            snapshot.point_cloud.get(),
                            snapshot.point_cloud_transform,
                            io::ColmapWriteOptions{.format = io::ColmapWriteFormat::Auto});
                        if (result) {
                            success = true;
                            update_stage(1.0f, LOC(lichtfeld::Strings::Runtime::TASK_COMPLETE));
                        } else {
                            error_msg = result.error().message;
                        }
                    }
                } catch (const std::exception& e) {
                    error_msg = LOCF(lichtfeld::Strings::Runtime::TASK_FAILED_DETAIL, e.what());
                } catch (...) {
                    error_msg = LOC(lichtfeld::Strings::Runtime::COLMAP_UNKNOWN_EXCEPTION);
                }

                if (success && (stop_token.stop_requested() || jobs_.cancelRequested(job))) {
                    success = false;
                    cancelled = true;
                    error_msg = LOC(lichtfeld::Strings::Runtime::EXPORT_CANCELLED);
                }

                if (success) {
                    LOG_INFO("COLMAP export completed: {}", lfs::core::path_to_utf8(path));
                    lfs::core::events::state::ExportCompleted{
                        .path = path,
                        .format = ExportFormat::COLMAP}
                        .emit();
                } else if (cancelled) {
                    LOG_INFO("COLMAP export cancelled: {}", lfs::core::path_to_utf8(path));
                    jobs_.report(
                        job, std::nullopt, LOC(lichtfeld::Strings::Runtime::TASK_CANCELLED),
                        error_msg);
                    publishExportState();
                    lfs::core::events::state::ExportFailed{.error = error_msg, .cancelled = true}.emit();
                } else {
                    LOG_ERROR("COLMAP export failed: {}", error_msg);
                    jobs_.report(
                        job, std::nullopt, LOC(lichtfeld::Strings::Runtime::TASK_FAILED),
                        error_msg);
                    publishExportState();
                    lfs::core::events::state::ExportFailed{.error = error_msg}.emit();
                }

                lfs::core::Tensor::trim_memory_pool();
                jobs_.finishWork(
                    job, cancelled, error_msg);
                publishExportState();
                if (auto* window_manager = services().windowOrNull()) {
                    window_manager->wakeEventLoop();
                }
            });
    }

    void AsyncTaskManager::startAsyncExport(ExportFormat format,
                                            const std::filesystem::path& path,
                                            std::vector<ExportSplatSource> splats,
                                            int sh_degree,
                                            bool borrow_single_identity,
                                            std::shared_mutex* model_mutex,
                                            bool rad_flip_y,
                                            bool rad_streamable,
                                            int spz_version,
                                            core::ProvenanceStamp provenance) {
        if (splats.empty()) {
            LOG_ERROR("No splat data to export");
            publishExportFailureState(format, path, LOC(lichtfeld::Strings::Runtime::NO_SPLAT_DATA));
            return;
        }

        if (!beginJob(
                export_state_.job, JobType::Export,
                LOC(lichtfeld::Strings::Runtime::TASK_STARTING))) {
            return;
        }
        {
            const std::lock_guard lock(export_state_.mutex);
            export_state_.format = format;
            export_state_.path = path;
        }
        publishExportState();

        LOG_INFO("Export started: {} (format: {})", lfs::core::path_to_utf8(path), static_cast<int>(format));

        const auto job = export_state_.job;
        export_state_.thread.emplace(
            [this,
             job,
             format,
             path,
             splats = std::move(splats),
             sh_degree,
             borrow_single_identity,
             model_mutex,
             rad_flip_y,
             rad_streamable,
             spz_version,
             provenance](
                std::stop_token stop_token) mutable {
                bool cancellation_logged = false;
                jobs_.work(job);
                auto update_progress = [this, job, &stop_token, &cancellation_logged](float progress, const std::string& stage) -> bool {
                    if (stop_token.stop_requested() || jobs_.cancelRequested(job)) {
                        if (!cancellation_logged) {
                            LOG_INFO("Export cancelled");
                            cancellation_logged = true;
                        }
                        jobs_.report(
                            job, std::nullopt,
                            LOC(lichtfeld::Strings::Runtime::TASK_CANCELLED));
                        publishExportState();
                        if (auto* window_manager = services().windowOrNull()) {
                            window_manager->wakeEventLoop();
                        }
                        return false;
                    }
                    jobs_.report(job, progress, stage);
                    publishExportState();
                    if (auto* window_manager = services().windowOrNull()) {
                        window_manager->wakeEventLoop();
                    }
                    return true;
                };

                bool success = false;
                bool cancelled = false;
                std::string error_msg;
                std::unique_ptr<lfs::core::SplatData> splat_data;
                std::optional<std::shared_lock<std::shared_mutex>> model_lock;

                try {
                    if (!update_progress(0.0f, LOC(lichtfeld::Strings::Runtime::EXPORT_PREPARING_DATA))) {
                        cancelled = true;
                        error_msg = LOC(lichtfeld::Strings::Runtime::EXPORT_CANCELLED);
                    }

                    if (!cancelled && model_mutex) {
                        model_lock.emplace(*model_mutex);
                    }

                    if (!cancelled) {
                        std::vector<std::pair<const lfs::core::SplatData*, glm::mat4>> merge_inputs;
                        merge_inputs.reserve(splats.size());
                        for (const auto& source : splats) {
                            if (source.data) {
                                merge_inputs.emplace_back(source.data, source.transform);
                            }
                        }

                        const auto storage_mode = borrow_single_identity
                                                      ? core::Scene::MergeStorageMode::BorrowSingleIdentity
                                                      : core::Scene::MergeStorageMode::Clone;
                        splat_data = core::Scene::mergeSplatsWithTransforms(merge_inputs, storage_mode);
                        if (!splat_data) {
                            error_msg = LOC(lichtfeld::Strings::Runtime::NO_SPLAT_DATA);
                        } else if (sh_degree < splat_data->get_max_sh_degree()) {
                            truncateSHDegree(*splat_data, sh_degree);
                        }
                        model_lock.reset();
                    }

                    if (!cancelled && splat_data &&
                        !update_progress(0.0f, LOC(lichtfeld::Strings::Runtime::EXPORT_DATA_PREPARED))) {
                        cancelled = true;
                        error_msg = LOC(lichtfeld::Strings::Runtime::EXPORT_CANCELLED);
                    }

                    if (!cancelled && splat_data) {
                        switch (format) {
                        case ExportFormat::PLY: {
                            const lfs::io::PlySaveOptions options{
                                .output_path = path,
                                .binary = true,
                                .async = false,
                                .progress_callback = update_progress,
                                .extra_attributes = {},
                                .provenance = provenance};
                            if (auto result = lfs::io::save_ply(*splat_data, options); result) {
                                success = true;
                            } else {
                                error_msg = result.error().message;
                                cancelled = result.error().code == lfs::io::ErrorCode::CANCELLED;
                                if (result.error().code == lfs::io::ErrorCode::INSUFFICIENT_DISK_SPACE) {
                                    lfs::core::events::state::DiskSpaceSaveFailed{
                                        .iteration = 0,
                                        .path = path,
                                        .error = result.error().message,
                                        .required_bytes = result.error().required_bytes,
                                        .available_bytes = result.error().available_bytes,
                                        .is_disk_space_error = true}
                                        .emit();
                                }
                            }
                            break;
                        }
                        case ExportFormat::SOG: {
                            const lfs::io::SogSaveOptions options{
                                .output_path = path,
                                .kmeans_iterations = 10,
                                .progress_callback = update_progress,
                                .provenance = provenance};
                            if (auto result = lfs::io::save_sog(*splat_data, options); result) {
                                success = true;
                            } else {
                                error_msg = result.error().message;
                                cancelled = result.error().code == lfs::io::ErrorCode::CANCELLED;
                            }
                            break;
                        }
                        case ExportFormat::SPZ: {
                            const lfs::io::SpzSaveOptions options{
                                .output_path = path,
                                .version = spz_version,
                                .progress_callback = update_progress,
                                .provenance = provenance};
                            if (auto result = lfs::io::save_spz(*splat_data, options); result) {
                                success = true;
                            } else {
                                error_msg = result.error().message;
                                cancelled = result.error().code == lfs::io::ErrorCode::CANCELLED;
                            }
                            break;
                        }
                        case ExportFormat::HTML_VIEWER: {
                            const lfs::io::HtmlExportOptions options{
                                .output_path = path,
                                .kmeans_iterations = 10,
                                .progress_callback = update_progress,
                                .provenance = provenance};
                            if (auto result = lfs::io::export_html(*splat_data, options); result) {
                                success = true;
                            } else {
                                error_msg = result.error().message;
                                cancelled = result.error().code == lfs::io::ErrorCode::CANCELLED;
                            }
                            break;
                        }
                        case ExportFormat::USD: {
                            const lfs::io::UsdSaveOptions options{
                                .output_path = path,
                                .progress_callback = update_progress,
                                .provenance = provenance};
                            if (auto result = lfs::io::save_usd(*splat_data, options); result) {
                                success = true;
                            } else {
                                error_msg = result.error().message;
                                cancelled = result.error().code == lfs::io::ErrorCode::CANCELLED;
                            }
                            break;
                        }
                        case ExportFormat::NUREC_USDZ: {
                            const lfs::io::NurecUsdzSaveOptions options{
                                .output_path = path,
                                .progress_callback = update_progress,
                                .provenance = provenance};
                            if (auto result = lfs::io::save_nurec_usdz(*splat_data, options); result) {
                                success = true;
                            } else {
                                error_msg = result.error().message;
                                cancelled = result.error().code == lfs::io::ErrorCode::CANCELLED;
                            }
                            break;
                        }
                        case ExportFormat::RAD: {
                            const lfs::io::RadSaveOptions options{
                                .output_path = path,
                                .compression_level = 6,
                                .flip_y = rad_flip_y,
                                .chunk_size = rad_streamable
                                                  ? lfs::io::kRadStreamableChunkSplats
                                                  : lfs::io::kRadNativeChunkSplats,
                                .progress_callback = update_progress,
                                .provenance = provenance};
                            if (auto result = lfs::io::save_rad(*splat_data, options); result) {
                                success = true;
                            } else {
                                error_msg = result.error().message;
                                cancelled = result.error().code == lfs::io::ErrorCode::CANCELLED;
                            }
                            break;
                        }
                        case ExportFormat::COLMAP:
                            error_msg = LOC(lichtfeld::Strings::Runtime::COLMAP_WRITE_BACK_PATH);
                            break;
                        }
                    }

                } catch (const std::exception& e) {
                    error_msg = LOCF(lichtfeld::Strings::Runtime::TASK_FAILED_DETAIL, e.what());
                    LOG_ERROR("{}", error_msg);
                } catch (...) {
                    error_msg = LOC(lichtfeld::Strings::Runtime::EXPORT_UNKNOWN_EXCEPTION);
                    LOG_ERROR("{}", error_msg);
                }

                if (success && (stop_token.stop_requested() || jobs_.cancelRequested(job))) {
                    success = false;
                    cancelled = true;
                    error_msg = LOC(lichtfeld::Strings::Runtime::EXPORT_CANCELLED);
                }

                if (success) {
                    LOG_INFO("Export completed: {}", lfs::core::path_to_utf8(path));
                    jobs_.report(
                        job, 1.0F, LOC(lichtfeld::Strings::Runtime::TASK_COMPLETE));
                    publishExportState();
                    lfs::core::events::state::ExportCompleted{
                        .path = path,
                        .format = format}
                        .emit();
                } else if (cancelled) {
                    LOG_INFO("Export cancelled: {}", lfs::core::path_to_utf8(path));
                    jobs_.report(
                        job, std::nullopt, LOC(lichtfeld::Strings::Runtime::TASK_CANCELLED),
                        error_msg);
                    publishExportState();
                    lfs::core::events::state::ExportFailed{
                        .error = error_msg,
                        .cancelled = true}
                        .emit();
                } else {
                    LOG_ERROR("Export failed: {}", error_msg);
                    jobs_.report(
                        job, std::nullopt, LOC(lichtfeld::Strings::Runtime::TASK_FAILED),
                        error_msg);
                    publishExportState();
                    lfs::core::events::state::ExportFailed{
                        .error = error_msg}
                        .emit();
                }

                splat_data.reset();
                lfs::core::Tensor::trim_memory_pool();
                jobs_.finishWork(
                    job, cancelled, error_msg);
                publishExportState();
                wakeMainThreadForAsyncWork();
            });
    }

    void AsyncTaskManager::cancelExport() {
        if (!isExporting())
            return;
        LOG_INFO("Cancelling export");
        jobs_.requestCancel(
            export_state_.job, LOC(lichtfeld::Strings::Runtime::TASK_CANCELLING));
        publishExportState();
        if (export_state_.thread && export_state_.thread->joinable()) {
            export_state_.thread->request_stop();
        }
    }

    void AsyncTaskManager::publishExportFailureState(const ExportFormat format,
                                                     const std::filesystem::path& path,
                                                     std::string error) {
        if (!beginJob(
                export_state_.job, JobType::Export,
                LOC(lichtfeld::Strings::Runtime::TASK_STARTING))) {
            return;
        }
        {
            const std::lock_guard lock(export_state_.mutex);
            export_state_.format = format;
            export_state_.path = path;
        }
        jobs_.failed(
            export_state_.job, std::move(error));
        publishExportState();
    }

    void AsyncTaskManager::publishExportState() {
        lfs::vis::AppStore::ExportProgressState state;
        const auto job =
            jobs_.peek(export_state_.job);
        state.active = job && job->running();
        state.progress =
            job ? job->progress : 0.0F;
        state.stage =
            job ? job->stage : std::string{};
        state.error =
            job ? job->error : std::string{};
        state.outcome = jobOutcome(export_state_.job);
        {
            const std::lock_guard lock(export_state_.mutex);
            state.format = exportProgressFormatName(export_state_.format);
            state.path = lfs::core::path_to_utf8(export_state_.path);
        }
        lfs::vis::app_store().export_progress_state.set(std::move(state));
    }

    void AsyncTaskManager::publishImportOverlayState() {
        lfs::vis::AppStore::ImportOverlayState state;
        const auto job =
            jobs_.peek(import_state_.job);
        state.active = job && job->running();
        state.show_completion = import_state_.show_completion.load();
        state.progress =
            job ? job->progress : 0.0F;
        state.stage =
            job ? job->stage : std::string{};
        state.error =
            job ? job->error : std::string{};
        {
            const std::lock_guard lock(import_state_.mutex);
            state.dataset_type = import_state_.dataset_type;
            state.path = lfs::core::path_to_utf8(import_state_.path.filename());
            state.success = import_state_.success;
            state.num_images = static_cast<std::uint64_t>(import_state_.num_images);
            state.num_points = static_cast<std::uint64_t>(import_state_.num_points);
            if (state.show_completion &&
                import_state_.completion_time != std::chrono::steady_clock::time_point{}) {
                const auto elapsed = std::chrono::steady_clock::now() - import_state_.completion_time;
                state.seconds_since_completion = std::chrono::duration<float>(elapsed).count();
            }
        }
        lfs::vis::app_store().import_overlay_state.set(std::move(state));
    }

    void AsyncTaskManager::setImportNumImages(const size_t num_images) {
        {
            const std::lock_guard lock(import_state_.mutex);
            import_state_.num_images = num_images;
        }
        publishImportOverlayState();
    }

    void AsyncTaskManager::publishVideoExportOverlayState() {
        lfs::vis::AppStore::VideoExportOverlayState state;
        const auto job =
            jobs_.peek(video_export_state_.job);
        state.active = job && job->running();
        state.progress =
            job ? job->progress : 0.0F;
        state.current_frame = video_export_state_.current_frame.load();
        state.total_frames = video_export_state_.total_frames.load();
        state.stage =
            job ? job->stage : std::string{};
        lfs::vis::app_store().video_export_overlay_state.set(std::move(state));
    }

    void AsyncTaskManager::publishMesh2SplatState() {
        lfs::vis::AppStore::TaskProgressState state;
        const auto job =
            jobs_.peek(mesh2splat_state_.job);
        state.active = job && job->running();
        state.progress =
            job ? job->progress : 0.0F;
        state.stage =
            job ? job->stage : std::string{};
        state.error =
            job ? job->error : std::string{};
        {
            const std::lock_guard lock(mesh2splat_state_.mutex);
            state.source_name = mesh2splat_state_.source_name;
        }
        lfs::vis::app_store().mesh2splat_state.set(std::move(state));
    }

    void AsyncTaskManager::publishSplatSimplifyState() {
        lfs::vis::AppStore::TaskProgressState state;
        const auto job =
            jobs_.peek(splat_simplify_state_.job);
        state.active = job && job->running();
        state.progress =
            job ? job->progress : 0.0F;
        state.stage =
            job ? job->stage : std::string{};
        state.error =
            job ? job->error : std::string{};
        {
            const std::lock_guard lock(splat_simplify_state_.mutex);
            state.source_name = splat_simplify_state_.source_name;
            state.output_name = splat_simplify_state_.output_name;
        }
        lfs::vis::app_store().splat_simplify_state.set(std::move(state));
    }

    void AsyncTaskManager::cancelImportCompletionDismiss() {
        import_state_.completion_generation.fetch_add(1, std::memory_order_acq_rel);
        if (import_state_.completion_dismiss_thread) {
            import_state_.completion_dismiss_thread->request_stop();
            if (import_state_.completion_dismiss_thread->joinable())
                import_state_.completion_dismiss_thread->join();
            import_state_.completion_dismiss_thread.reset();
        }
    }

    void AsyncTaskManager::scheduleImportCompletionDismiss() {
        cancelImportCompletionDismiss();

        const auto generation =
            import_state_.completion_generation.fetch_add(1, std::memory_order_acq_rel) + 1;
        import_state_.completion_dismiss_thread.emplace(
            [this, generation](std::stop_token stop_token) {
                std::mutex mutex;
                std::condition_variable_any cv;
                std::unique_lock lock(mutex);
                cv.wait_for(lock, stop_token, std::chrono::milliseconds(3000), [] { return false; });
                if (stop_token.stop_requested())
                    return;
                if (import_state_.completion_generation.load(std::memory_order_acquire) != generation)
                    return;
                if (isImporting() || !import_state_.show_completion.load())
                    return;

                bool success = false;
                {
                    const std::lock_guard state_lock(import_state_.mutex);
                    success = import_state_.success;
                }
                if (!success)
                    return;

                import_state_.show_completion.store(false);
                publishImportOverlayState();
            });
    }

    void AsyncTaskManager::dismissImport() {
        cancelImportCompletionDismiss();
        import_state_.show_completion.store(false);
        publishImportOverlayState();
    }

    void AsyncTaskManager::cancelImport() {
        const bool had_activity = isImporting() ||
                                  import_state_.show_completion.load() ||
                                  import_state_.thread.has_value();
        if (!had_activity) {
            return;
        }

        LOG_INFO("Cancelling import");
        cancelImportCompletionDismiss();
        if (isImporting()) {
            jobs_.requestCancel(
                import_state_.job, LOC(lichtfeld::Strings::Runtime::TASK_CANCELLING));
        }
        if (import_state_.thread) {
            import_state_.thread->request_stop();
            if (import_state_.thread->joinable()) {
                import_state_.thread->join();
            }
            import_state_.thread.reset();
        }

        if (const auto state =
                jobs_.update(import_state_.job);
            state && state->running()) {
            jobs_.canceled(import_state_.job);
        }
        import_state_.load_complete.store(false);
        import_state_.show_completion.store(false);
        {
            const std::lock_guard lock(import_state_.mutex);
            import_state_.path.clear();
            import_state_.dataset_type.clear();
            import_state_.num_images = 0;
            import_state_.num_points = 0;
            import_state_.success = false;
            import_state_.is_mesh = false;
            import_state_.load_result.reset();
            import_state_.params = {};
        }
        PanelRegistry::instance().invalidate_poll_cache();
        publishImportOverlayState();
    }

    void AsyncTaskManager::startAsyncImport(const std::filesystem::path& path,
                                            const lfs::core::param::TrainingParameters& params) {
        if (isImporting()) {
            LOG_WARN("Import already in progress");
            return;
        }

        cancelImportCompletionDismiss();
        if (!beginJob(
                import_state_.job, JobType::Import,
                LOC(lichtfeld::Strings::Runtime::TASK_INITIALIZING))) {
            return;
        }
        import_state_.load_complete.store(false);
        import_state_.show_completion.store(false);
        PanelRegistry::instance().invalidate_poll_cache();
        {
            const std::lock_guard lock(import_state_.mutex);
            import_state_.path = path;
            import_state_.num_images = 0;
            import_state_.num_points = 0;
            import_state_.success = false;
            import_state_.is_mesh = false;
            import_state_.load_result.reset();
            import_state_.params = params;
            import_state_.dataset_type = getDatasetTypeName(path);
        }
        publishImportOverlayState();

        LOG_INFO("Async import: {}", lfs::core::path_to_utf8(path));

        const auto job = import_state_.job;
        import_state_.thread.emplace(
            [this, job, path](const std::stop_token stop_token) noexcept {
                jobs_.work(job);
                const auto record_failure = [this, job](const char* detail) noexcept {
                    try {
                        const std::lock_guard lock(import_state_.mutex);
                        import_state_.success = false;
                        import_state_.load_result.reset();
                    } catch (...) {
                    }

                    try {
                        const std::string message = detail && *detail
                                                        ? LOCF(lichtfeld::Strings::Runtime::IMPORT_FAILED_DETAIL, detail)
                                                        : LOC(lichtfeld::Strings::Runtime::IMPORT_UNKNOWN_EXCEPTION);
                        jobs_.report(
                            job, std::nullopt,
                            LOC(lichtfeld::Strings::Runtime::TASK_FAILED),
                            message);
                        LOG_ERROR("{}", message);
                    } catch (...) {
                        // The worker boundary must remain no-throw even when
                        // reporting an allocation failure. success was reset
                        // before launch, so completion still follows the
                        // failure path if diagnostic allocation also fails.
                    }
                };

                const ScopeExit publish_terminal([this, job, &stop_token]() noexcept {
                    if (!stop_token.stop_requested()) {
                        jobs_.report(
                            job, 1.0F, std::nullopt);
                        import_state_.load_complete.store(true, std::memory_order_release);
                    }
                    jobs_.finishWork(
                        job, stop_token.stop_requested());

                    try {
                        publishImportOverlayState();
                    } catch (...) {
                        // Publishing is best-effort during teardown/error
                        // handling; the atomic terminal state remains visible.
                    }
                    try {
                        wakeMainThreadForAsyncWork();
                    } catch (...) {
                    }
                });

                try {
                    lfs::core::param::TrainingParameters local_params;
                    {
                        const std::lock_guard lock(import_state_.mutex);
                        local_params = import_state_.params;
                    }

                    const auto parse_centralize = [](const std::string& s) {
                        if (s == "off")
                            return lfs::io::CentralizeDataset::Off;
                        if (s == "by_pointcloud")
                            return lfs::io::CentralizeDataset::ByPointCloud;
                        if (s == "by_cameras")
                            return lfs::io::CentralizeDataset::ByCameras;
                        return lfs::io::CentralizeDataset::Off;
                    };
                    int effective_min_track_length = local_params.dataset.min_track_length;
                    if (effective_min_track_length > 0 &&
                        local_params.init_path.has_value() &&
                        !local_params.init_path->empty()) {
                        LOG_WARN("{}", LOCF(lichtfeld::Strings::Runtime::COLMAP_MIN_TRACK_LENGTH,
                                            *local_params.init_path));
                        effective_min_track_length = 0;
                    }
                    const lfs::io::LoadOptions load_options{
                        .resize_factor = local_params.dataset.resize_factor,
                        .max_width = local_params.dataset.max_width,
                        .images_folder = local_params.dataset.images,
                        .min_track_length = effective_min_track_length,
                        .validate_only = false,
                        .centralize = parse_centralize(local_params.dataset.centralize_dataset),
                        .progress = [this, job, &stop_token](const float pct, const std::string& msg) {
                        if (stop_token.stop_requested())
                            return;
                        jobs_.report(
                            job, pct / 100.0F, msg);
                        publishImportOverlayState(); },
                        .cancel_requested = [&stop_token]() { return stop_token.stop_requested(); }};

                    auto loader = lfs::io::Loader::create();
                    auto result = loader->load(path, load_options);

                    if (stop_token.stop_requested()) {
                        return;
                    }

                    {
                        const std::lock_guard lock(import_state_.mutex);
                        if (result) {
                            import_state_.load_result = std::move(*result);
                            import_state_.success = true;
                            jobs_.report(
                                job, std::nullopt,
                                LOC(lichtfeld::Strings::Runtime::TASK_APPLYING));
                            std::visit([this](const auto& data) {
                                using T = std::decay_t<decltype(data)>;
                                if constexpr (std::is_same_v<T, std::shared_ptr<lfs::core::SplatData>>) {
                                    import_state_.num_points = data->size();
                                    import_state_.num_images = 0;
                                } else if constexpr (std::is_same_v<T, lfs::io::LoadedScene>) {
                                    import_state_.num_images = data.cameras.size();
                                    import_state_.num_points = data.point_cloud ? data.point_cloud->size() : 0;
                                } else if constexpr (std::is_same_v<T, std::shared_ptr<lfs::core::MeshData>>) {
                                    import_state_.num_points = data ? data->vertex_count() : 0;
                                    import_state_.num_images = 0;
                                    import_state_.is_mesh = true;
                                }
                            },
                                       import_state_.load_result->data);
                        } else {
                            import_state_.success = false;
                            const auto message =
                                result.error().format();
                            jobs_.report(
                                job, std::nullopt,
                                LOC(lichtfeld::Strings::Runtime::TASK_FAILED), message);
                            LOG_ERROR(
                                "Import failed: {}",
                                message);
                        }
                    }
                } catch (const std::exception& e) {
                    record_failure(e.what());
                } catch (...) {
                    record_failure(nullptr);
                }
            });
    }

    void AsyncTaskManager::checkAsyncImportCompletion() {
        if (!import_state_.load_complete.exchange(false, std::memory_order_acq_rel))
            return;

        bool success;
        {
            const std::lock_guard lock(import_state_.mutex);
            success = import_state_.success;
        }

        if (success) {
            applyLoadedDataToScene();
        } else {
            auto error =
                jobError(import_state_.job);
            if (error.empty()) {
                error = "Dataset import failed";
            }
            jobs_.failed(
                import_state_.job,
                std::move(error));
            import_state_.show_completion.store(true);
            {
                const std::lock_guard lock(import_state_.mutex);
                import_state_.completion_time = std::chrono::steady_clock::now();
            }
            publishImportOverlayState();
        }
        PanelRegistry::instance().invalidate_poll_cache();

        if (import_state_.thread && import_state_.thread->joinable()) {
            import_state_.thread->join();
            import_state_.thread.reset();
        }
    }

    void AsyncTaskManager::applyLoadedDataToScene() {
        auto* const scene_manager = viewer_->getSceneManager();
        if (!scene_manager) {
            LOG_ERROR("No scene manager");
            jobs_.failed(
                import_state_.job,
                "No scene manager");
            publishImportOverlayState();
            return;
        }

        std::optional<lfs::io::LoadResult> load_result;
        lfs::core::param::TrainingParameters params;
        std::filesystem::path path;
        {
            const std::lock_guard lock(import_state_.mutex);
            load_result = std::move(import_state_.load_result);
            params = import_state_.params;
            path = import_state_.path;
            import_state_.load_result.reset();
        }

        if (!load_result) {
            LOG_ERROR("No load result");
            jobs_.failed(
                import_state_.job,
                "No load result");
            publishImportOverlayState();
            return;
        }

        const auto result = scene_manager->applyLoadedDataset(path, params, std::move(*load_result));

        if (result) {
            if (auto* data_loader = viewer_->getDataLoader())
                data_loader->setParameters(params);
        }

        bool success_val;
        std::string error_val;
        size_t num_images_val, num_points_val;
        {
            const std::lock_guard lock(import_state_.mutex);
            import_state_.completion_time = std::chrono::steady_clock::now();
            import_state_.success = result.has_value();
            success_val = import_state_.success;
            error_val =
                result ? std::string{} : result.error();
            num_images_val = import_state_.num_images;
            num_points_val = import_state_.num_points;
        }

        if (success_val) {
            jobs_.report(
                import_state_.job, 1.0F, LOC(lichtfeld::Strings::Runtime::TASK_COMPLETE));
            jobs_.completed(import_state_.job);
        } else {
            jobs_.failed(
                import_state_.job, error_val);
        }
        bool is_mesh_load;
        {
            const std::lock_guard lock(import_state_.mutex);
            is_mesh_load = import_state_.is_mesh;
        }
        import_state_.show_completion.store(!(success_val && is_mesh_load));
        if (success_val && !is_mesh_load)
            scheduleImportCompletionDismiss();
        publishImportOverlayState();

        lfs::core::events::state::DatasetLoadCompleted{
            .path = path,
            .success = success_val,
            .error = success_val ? std::nullopt : std::optional<std::string>(error_val),
            .num_images = num_images_val,
            .num_points = num_points_val}
            .emit();
    }

    void AsyncTaskManager::applyAutoCropToLoadedScene() {
        auto* const scene_manager = viewer_->getSceneManager();
        if (!scene_manager)
            return;

        // Highest-id pointcloud/splat root = the one the import just produced.
        const core::SceneNode* target = nullptr;
        for (const auto* node : scene_manager->getScene().getNodes()) {
            if (node->type != core::NodeType::POINTCLOUD && node->type != core::NodeType::SPLAT)
                continue;
            if (!target || node->id > target->id)
                target = node;
        }
        if (!target) {
            LOG_WARN("Auto-crop requested but no pointcloud/splat node was found after load");
            return;
        }

        // AddCropBox selects the new node; FitCropBoxToScene then operates
        // on that selection. Both handlers run synchronously inside emit().
        lfs::core::events::cmd::AddCropBox{.node_name = target->name}.emit();
        lfs::core::events::cmd::FitCropBoxToScene{.use_percentile = true}.emit();
    }

    void AsyncTaskManager::cancelVideoExport() {
        if (!isExportingVideo())
            return;
        LOG_INFO("Cancelling video export");
        jobs_.requestCancel(
            video_export_state_.job, LOC(lichtfeld::Strings::Runtime::TASK_CANCELLING));
        publishVideoExportOverlayState();
        if (video_export_state_.thread) {
            video_export_state_.thread->request_stop();
        }
    }

    void AsyncTaskManager::startVideoExport(const std::filesystem::path& path,
                                            const io::video::VideoExportOptions& options) {
        auto fail_start = [this, &path](std::string error) {
            LOG_ERROR("Cannot export video: {}", error);
            if (!beginJob(
                    video_export_state_.job,
                    JobType::VideoExport,
                    LOC(lichtfeld::Strings::Runtime::TASK_INITIALIZING))) {
                return;
            }
            video_export_state_.total_frames.store(0);
            video_export_state_.current_frame.store(0);
            {
                std::lock_guard lock(video_export_state_.mutex);
                video_export_state_.path = path;
            }
            jobs_.failed(
                video_export_state_.job, error);
            publishVideoExportOverlayState();
            lfs::core::events::state::VideoExportFailed{.error = std::move(error)}.emit();
        };

        if (isExportingVideo()) {
            LOG_WARN("Video export already in progress");
            return;
        }
        if (video_export_state_.thread && video_export_state_.thread->joinable()) {
            video_export_state_.thread->join();
            video_export_state_.thread.reset();
        }

        auto* const scene_manager = viewer_->getSceneManager();
        auto* const rendering_manager = viewer_->getRenderingManager();
        if (!scene_manager || !rendering_manager) {
            fail_start(LOC(lichtfeld::Strings::Runtime::VIDEO_MISSING_SCENE_OR_RENDERING));
            return;
        }

        auto* gui_manager = viewer_->getGuiManager();
        if (!gui_manager) {
            fail_start(LOC(lichtfeld::Strings::Runtime::VIDEO_GUI_MANAGER_UNAVAILABLE));
            return;
        }
        const auto& timeline = gui_manager->sequencer().timeline();
        if (timeline.empty()) {
            fail_start(LOC(lichtfeld::Strings::Runtime::VIDEO_NO_KEYFRAMES));
            return;
        }

        const auto validated_options = validateVideoExportOptions(options);
        if (!validated_options) {
            fail_start(validated_options.error());
            return;
        }

        const auto snapshot_result = captureVideoExportSceneSnapshot(*scene_manager);
        if (!snapshot_result) {
            fail_start(snapshot_result.error());
            return;
        }

        auto* const engine = rendering_manager->getRenderingEngine();
        if (!engine) {
            fail_start(LOC(lichtfeld::Strings::Runtime::VIDEO_RENDERING_ENGINE_UNAVAILABLE));
            return;
        }

        const auto export_options = *validated_options;
        const auto render_settings = rendering_manager->getSettings();
        const float duration = timeline.duration();
        const int total_frames = static_cast<int>(std::ceil(duration * export_options.framerate)) + 1;
        const int width = export_options.width;
        const int height = export_options.height;

        std::vector<lfs::sequencer::CameraState> frame_states;
        frame_states.reserve(total_frames);
        const float start_time = timeline.startTime();
        const float time_step = 1.0f / static_cast<float>(export_options.framerate);
        for (int i = 0; i < total_frames; ++i)
            frame_states.push_back(timeline.evaluate(start_time + static_cast<float>(i) * time_step));

        if (!beginJob(
                video_export_state_.job,
                JobType::VideoExport,
                LOC(lichtfeld::Strings::Runtime::TASK_INITIALIZING))) {
            return;
        }
        video_export_state_.total_frames.store(total_frames);
        video_export_state_.current_frame.store(0);
        {
            std::lock_guard lock(video_export_state_.mutex);
            video_export_state_.path = path;
        }
        publishVideoExportOverlayState();

        resetVideoExportEnvironmentState();
        video_export_environment_state_ = std::make_unique<VideoExportEnvironmentState>();
        resetVideoExportMeshRendererState();
        if (!snapshot_result->meshes.empty()) {
            video_export_mesh_renderer_state_ = std::make_unique<VideoExportMeshRendererState>();
        }

        LOG_INFO("Starting video export: {} frames at {}x{}", total_frames, width, height);

        const auto job = video_export_state_.job;
        video_export_state_.thread.emplace(
            [this, job, viewer = viewer_, path, export_options, total_frames, width, height,
             engine, scene_manager, rendering_manager, render_settings, start_time, time_step,
             environment_state = video_export_environment_state_.get(),
             mesh_renderer_state = video_export_mesh_renderer_state_.get(),
             snapshot = *snapshot_result,
             frame_states = std::move(frame_states)](std::stop_token stop_token) mutable {
                jobs_.work(job);
                bool cancelled = false;
                std::string error_msg;
                auto cleanup_video_export_state = [this, viewer]() {
                    if (!video_export_environment_state_ && !video_export_mesh_renderer_state_) {
                        return;
                    }
                    auto cleanup_result = postToViewerAndWait(
                        viewer,
                        [this]() -> std::expected<void, std::string> {
                            resetVideoExportMeshRendererState();
                            resetVideoExportEnvironmentState();
                            return {};
                        });
                    if (!cleanup_result) {
                        LOG_DEBUG("Skipping video export state cleanup: {}", cleanup_result.error());
                    }
                };

                auto encoder = lfs::gui::createVideoEncoder();
                if (!encoder) {
                    error_msg = LOC(lichtfeld::Strings::Runtime::VIDEO_ENCODER_UNAVAILABLE);
                    jobs_.report(
                        job, std::nullopt,
                        LOC(lichtfeld::Strings::Runtime::TASK_FAILED),
                        error_msg);
                    publishVideoExportOverlayState();
                    lfs::core::events::state::VideoExportFailed{
                        .error = LOC(lichtfeld::Strings::Runtime::VIDEO_ENCODER_UNAVAILABLE)}
                        .emit();
                    cleanup_video_export_state();
                    jobs_.finishWork(
                        job, false, error_msg);
                    wakeMainThreadForAsyncWork();
                    return;
                }

                jobs_.report(
                    job, std::nullopt,
                    LOC(lichtfeld::Strings::Runtime::TASK_OPENING_ENCODER));
                publishVideoExportOverlayState();

                auto result = encoder->open(path, export_options);
                if (!result) {
                    error_msg = result.error();
                    jobs_.report(
                        job, std::nullopt,
                        LOCF(lichtfeld::Strings::Runtime::TASK_FAILED_DETAIL, result.error()),
                        error_msg);
                    LOG_ERROR("Failed to open encoder: {}", result.error());
                    lfs::core::events::state::VideoExportFailed{
                        .error = result.error()}
                        .emit();
                    publishVideoExportOverlayState();
                    cleanup_video_export_state();
                    jobs_.finishWork(
                        job, false, error_msg);
                    wakeMainThreadForAsyncWork();
                    return;
                }

                for (int frame = 0; frame < total_frames; ++frame) {
                    if (stop_token.stop_requested() || jobs_.cancelRequested(job)) {
                        LOG_INFO("Video export cancelled at frame {}", frame);
                        cancelled = true;
                        break;
                    }

                    auto frame_tensor = postToViewerAndWait(
                        viewer,
                        [viewer, engine, scene_manager, rendering_manager, environment_state,
                         mesh_renderer_state, snapshot_ptr = &snapshot, render_settings, width, height,
                         cam_state = frame_states[frame],
                         clip_time = start_time + static_cast<float>(frame) * time_step]()
                            -> std::expected<lfs::core::Tensor, std::string> {
                            if (lfs::python::has_scene_time_callback()) {
                                lfs::python::tick_scene_time_callback(clip_time);
                                refreshVideoExportMeshTransforms(
                                    *snapshot_ptr, scene_manager->getScene());
                            }
                            auto* const window_manager = viewer->getWindowManager();
                            auto* const vulkan_context =
                                window_manager != nullptr ? window_manager->getVulkanContext() : nullptr;
                            return renderVideoExportFrame(
                                *rendering_manager,
                                *engine,
                                *environment_state,
                                mesh_renderer_state,
                                vulkan_context,
                                *snapshot_ptr,
                                render_settings,
                                cam_state,
                                width,
                                height);
                        });

                    if (!frame_tensor) {
                        LOG_ERROR("Failed to render frame {}: {}", frame, frame_tensor.error());
                        error_msg = LOCF(lichtfeld::Strings::Runtime::TASK_FAILED_DETAIL, frame_tensor.error());
                        jobs_.report(
                            job, std::nullopt,
                            LOC(lichtfeld::Strings::Runtime::TASK_RENDER_ERROR),
                            error_msg);
                        publishVideoExportOverlayState();
                        break;
                    }

                    auto export_frame = frame_tensor->contiguous();
                    auto image_hwc = export_frame.permute({1, 2, 0}).contiguous();

                    if (frame == 0) {
                        LOG_INFO("Video export: CHW shape=[{},{},{}] -> HWC shape=[{},{},{}]",
                                 export_frame.shape()[0], export_frame.shape()[1], export_frame.shape()[2],
                                 image_hwc.shape()[0], image_hwc.shape()[1], image_hwc.shape()[2]);
                    }

                    const auto* const gpu_ptr = image_hwc.data_ptr();
                    auto write_result = encoder->writeFrameGpu(gpu_ptr, width, height, nullptr);
                    if (!write_result) {
                        error_msg =
                            write_result.error();
                        jobs_.report(
                            job, std::nullopt,
                            LOC(lichtfeld::Strings::Runtime::TASK_ENCODE_ERROR),
                            error_msg);
                        publishVideoExportOverlayState();
                        LOG_ERROR("Failed to encode frame {}: {}", frame, write_result.error());
                        break;
                    }

                    video_export_state_.current_frame.store(frame + 1);
                    jobs_.report(
                        job,
                        static_cast<float>(frame + 1) /
                            static_cast<float>(
                                total_frames),
                        LOCF(lichtfeld::Strings::Runtime::VIDEO_ENCODING_FRAME, frame + 1, total_frames));
                    publishVideoExportOverlayState();
                }

                if (cancelled) {
                    jobs_.report(
                        job, std::nullopt,
                        LOC(lichtfeld::Strings::Runtime::TASK_CANCELLED));
                } else if (error_msg.empty()) {
                    jobs_.report(
                        job, std::nullopt,
                        LOC(lichtfeld::Strings::Runtime::TASK_FINALIZING));
                }
                publishVideoExportOverlayState();

                if (auto close_result = encoder->close(); !close_result) {
                    error_msg = close_result.error();
                    jobs_.report(
                        job, std::nullopt, LOC(lichtfeld::Strings::Runtime::TASK_FAILED),
                        error_msg);
                    publishVideoExportOverlayState();
                    LOG_ERROR("Failed to close encoder: {}", close_result.error());
                } else {
                    bool emit_completed = false;
                    if (cancelled) {
                        jobs_.report(
                            job, std::nullopt,
                            LOC(lichtfeld::Strings::Runtime::TASK_CANCELLED));
                    } else if (
                        error_msg.empty() &&
                        !jobs_.cancelRequested(job)) {
                        jobs_.report(
                            job, 1.0F, LOC(lichtfeld::Strings::Runtime::TASK_COMPLETE));
                        LOG_INFO("Video export completed: {}", lfs::core::path_to_utf8(path));
                        emit_completed = true;
                    }
                    publishVideoExportOverlayState();
                    if (emit_completed) {
                        lfs::core::events::state::VideoExportCompleted{
                            .path = path,
                            .total_frames = total_frames}
                            .emit();
                    }
                }

                if (!error_msg.empty()) {
                    lfs::core::events::state::VideoExportFailed{
                        .error = error_msg}
                        .emit();
                }
                cleanup_video_export_state();
                jobs_.finishWork(
                    job, cancelled, error_msg);
                publishVideoExportOverlayState();
                wakeMainThreadForAsyncWork();
            });
    }

    void AsyncTaskManager::startMesh2Splat(std::shared_ptr<lfs::core::MeshData> mesh,
                                           const std::string& source_name,
                                           const lfs::core::Mesh2SplatOptions& options) {
        if (isMesh2SplatActive()) {
            LOG_WARN("Mesh2Splat conversion already in progress");
            return;
        }

        if (!mesh) {
            LOG_ERROR("Mesh2Splat: null mesh pointer");
            return;
        }

        if (!beginJob(
                mesh2splat_state_.job,
                JobType::Mesh2Splat,
                LOC(lichtfeld::Strings::Runtime::TASK_STARTING_ELLIPSIS))) {
            return;
        }
        {
            const std::lock_guard lock(mesh2splat_state_.mutex);
            mesh2splat_state_.source_name = source_name;
            mesh2splat_state_.pending_mesh = std::move(mesh);
            mesh2splat_state_.pending_options = options;
            mesh2splat_state_.result.reset();
        }

        LOG_INFO("Mesh2Splat conversion started: {} (resolution={}, sigma={})",
                 source_name, options.resolution_target, options.sigma);

        publishMesh2SplatState();
        mesh2splat_state_.pending.store(true);
        wakeMainThreadForAsyncWork();
    }

    void AsyncTaskManager::pollMesh2SplatCompletion() {
        if (!mesh2splat_state_.pending.exchange(false, std::memory_order_acq_rel))
            return;

        executeMesh2SplatOnGraphicsThread();

        bool has_result;
        {
            const std::lock_guard lock(mesh2splat_state_.mutex);
            has_result = mesh2splat_state_.result != nullptr;
        }

        if (has_result) {
            applyMesh2SplatResult();
            const auto apply_error =
                jobError(mesh2splat_state_.job);
            if (apply_error.empty()) {
                jobs_.report(
                    mesh2splat_state_.job, 1.0F,
                    LOC(lichtfeld::Strings::Runtime::TASK_COMPLETE));
                jobs_.completed(mesh2splat_state_.job);
            } else {
                jobs_.failed(
                    mesh2splat_state_.job, apply_error,
                    LOC(lichtfeld::Strings::Runtime::TASK_FAILED));
            }
        } else {
            const auto err =
                jobError(mesh2splat_state_.job);
            if (!err.empty()) {
                lfs::core::events::state::Mesh2SplatFailed{
                    .error = err}
                    .emit();
            }
            jobs_.failed(
                mesh2splat_state_.job, err,
                LOC(lichtfeld::Strings::Runtime::TASK_FAILED));
        }
        publishMesh2SplatState();
    }

    void AsyncTaskManager::executeMesh2SplatOnGraphicsThread() {
        std::shared_ptr<lfs::core::MeshData> mesh;
        lfs::core::Mesh2SplatOptions options;
        {
            const std::lock_guard lock(mesh2splat_state_.mutex);
            mesh = std::move(mesh2splat_state_.pending_mesh);
            options = mesh2splat_state_.pending_options;
        }

        if (!mesh) {
            jobs_.report(
                mesh2splat_state_.job, std::nullopt,
                std::nullopt,
                LOC(lichtfeld::Strings::Runtime::TASK_FAILED));
            return;
        }

        auto result = lfs::rendering::mesh_to_splat(
            *mesh,
            options,
            [this](const float progress, const std::string& stage) {
                jobs_.report(
                    mesh2splat_state_.job,
                    progress, stage);
                publishMesh2SplatState();
                return isMesh2SplatActive();
            });

        {
            const std::lock_guard lock(mesh2splat_state_.mutex);
            if (result) {
                mesh2splat_state_.result = std::move(*result);
                jobs_.report(
                    mesh2splat_state_.job, 1.0F,
                    LOC(lichtfeld::Strings::Runtime::TASK_COMPLETE));
            } else {
                mesh2splat_state_.result.reset();
                jobs_.report(
                    mesh2splat_state_.job,
                    std::nullopt, LOC(lichtfeld::Strings::Runtime::TASK_FAILED),
                    result.error());
                LOG_ERROR(
                    "Mesh2Splat conversion failed: {}",
                    result.error());
            }
        }
        publishMesh2SplatState();
    }

    void AsyncTaskManager::applyMesh2SplatResult() {
        auto* const scene_manager = viewer_->getSceneManager();
        if (!scene_manager) {
            LOG_ERROR("Mesh2Splat: no scene manager");
            jobs_.report(
                mesh2splat_state_.job, std::nullopt,
                std::nullopt,
                LOC(lichtfeld::Strings::Runtime::TASK_FAILED));
            return;
        }

        std::unique_ptr<lfs::core::SplatData> splat_data;
        std::string source_name;
        {
            const std::lock_guard lock(mesh2splat_state_.mutex);
            splat_data = std::move(mesh2splat_state_.result);
            source_name = mesh2splat_state_.source_name;
        }

        if (!splat_data) {
            LOG_ERROR("Mesh2Splat: no result data");
            jobs_.report(
                mesh2splat_state_.job, std::nullopt,
                std::nullopt,
                LOC(lichtfeld::Strings::Runtime::TASK_FAILED));
            return;
        }

        const std::string node_name = source_name + " (splat)";
        auto& scene = scene_manager->getScene();

        const core::NodeId existing_id = scene.getNodeIdByName(node_name);
        if (existing_id != core::NULL_NODE) {
            scene_manager->clearPlyPath(existing_id);
            scene.removeNode(node_name);
        }

        const std::string added_name =
            scene_manager->addGeneratedSplatNode(std::move(splat_data), source_name, node_name, true);
        if (added_name.empty()) {
            LOG_ERROR("Mesh2Splat: failed to add splat node '{}'", node_name);
            jobs_.report(
                mesh2splat_state_.job, std::nullopt,
                std::nullopt,
                LOC(lichtfeld::Strings::Runtime::TASK_FAILED));
            return;
        }

        jobs_.report(
            mesh2splat_state_.job, 1.0F,
            LOC(lichtfeld::Strings::Runtime::TASK_COMPLETE));
        publishMesh2SplatState();

        const auto* const added_node = scene.getNode(added_name);
        const size_t num_gaussians =
            added_node && added_node->model ? added_node->model->size() : 0;

        lfs::core::events::state::Mesh2SplatCompleted{
            .source_name = source_name,
            .node_name = added_name,
            .num_gaussians = num_gaussians}
            .emit();

        LOG_INFO("Mesh2Splat: added splat node '{}'", added_name);
    }

    void AsyncTaskManager::startSplatSimplify(const std::string& source_name,
                                              const lfs::core::SplatSimplifyOptions& options) {
        if (isSplatSimplifyActive()) {
            LOG_WARN("Splat simplification already in progress");
            return;
        }

        struct SimplifyCapture {
            std::unique_ptr<lfs::core::SplatData> model;
            std::string source_name;
            std::string output_name;
        };

        auto capture = postToViewerAndWait(
            viewer_,
            [this, source_name, options]() -> std::expected<SimplifyCapture, std::string> {
                auto* const scene_manager = viewer_->getSceneManager();
                if (!scene_manager) {
                    return std::unexpected(LOC(lichtfeld::Strings::Runtime::NO_SCENE_MANAGER));
                }

                const auto* const node = scene_manager->getScene().getNode(source_name);
                if (!node || node->type != core::NodeType::SPLAT || !node->model) {
                    return std::unexpected(LOCF(lichtfeld::Strings::Runtime::NO_SPLAT_NODE_NAMED, source_name));
                }

                const auto input_count = static_cast<int64_t>(node->model->size());
                const auto target_count = std::clamp<int64_t>(
                    static_cast<int64_t>(std::ceil(std::clamp(options.ratio, 0.0, 1.0) * static_cast<double>(input_count))),
                    int64_t{1},
                    std::max<int64_t>(int64_t{1}, input_count));
                return SimplifyCapture{
                    .model = std::make_unique<lfs::core::SplatData>(node->model->clone()),
                    .source_name = source_name,
                    .output_name = std::format("{}_{}", source_name, target_count),
                };
            });

        if (!capture) {
            LOG_ERROR("Splat simplify capture failed: {}", capture.error());
            return;
        }

        if (splat_simplify_state_.thread && splat_simplify_state_.thread->joinable()) {
            splat_simplify_state_.thread->join();
            splat_simplify_state_.thread.reset();
        }

        if (!beginJob(
                splat_simplify_state_.job,
                JobType::SplatSimplify,
                LOC(lichtfeld::Strings::Runtime::TASK_STARTING_ELLIPSIS))) {
            return;
        }
        splat_simplify_state_.completed.store(false);
        splat_simplify_state_.apply_pending.store(false);
        {
            const std::lock_guard lock(splat_simplify_state_.mutex);
            splat_simplify_state_.source_name = capture->source_name;
            splat_simplify_state_.output_name = capture->output_name;
            splat_simplify_state_.result.reset();
        }

        auto input = std::move(capture->model);
        auto opts = options;
        publishSplatSimplifyState();
        const auto job = splat_simplify_state_.job;
        splat_simplify_state_.thread.emplace([this, job, opts, input = std::move(input)](std::stop_token stop_token) mutable {
            jobs_.work(job);
            bool worker_cancelled = false;
            std::string worker_error;
            auto progress_cb = [this, job, &stop_token](const float progress, const std::string& stage) -> bool {
                if (stop_token.stop_requested() || jobs_.cancelRequested(job))
                    return false;
                jobs_.report(job, progress, stage);
                publishSplatSimplifyState();
                return true;
            };

            auto result = lfs::core::simplify_splats(*input, opts, progress_cb);
            if (result) {
                {
                    const std::lock_guard lock(splat_simplify_state_.mutex);
                    splat_simplify_state_.result = std::move(*result);
                }
                jobs_.report(
                    job, 1.0F, LOC(lichtfeld::Strings::Runtime::TASK_APPLYING));
                splat_simplify_state_.apply_pending.store(true, std::memory_order_release);
                publishSplatSimplifyState();
            } else {
                worker_cancelled =
                    jobs_.cancelRequested(job) ||
                    stop_token.stop_requested() ||
                    result.error() == "Cancelled";
                worker_error = worker_cancelled
                                   ? std::string{}
                                   : result.error();
                jobs_.report(
                    job, std::nullopt,
                    worker_cancelled ? LOC(lichtfeld::Strings::Runtime::TASK_CANCELLED)
                                     : LOC(lichtfeld::Strings::Runtime::TASK_FAILED),
                    worker_error);
                publishSplatSimplifyState();
            }
            jobs_.finishWork(
                job, worker_cancelled,
                worker_error);
            splat_simplify_state_.completed.store(true, std::memory_order_release);
            wakeMainThreadForAsyncWork();
        });
    }

    void AsyncTaskManager::pollSplatSimplifyCompletion() {
        if (splat_simplify_state_.apply_pending.exchange(false, std::memory_order_acq_rel)) {
            if (splat_simplify_state_.thread && splat_simplify_state_.thread->joinable()) {
                splat_simplify_state_.thread->join();
                splat_simplify_state_.thread.reset();
            }

            auto* const scene_manager = viewer_->getSceneManager();
            if (!scene_manager) {
                LOG_ERROR("Splat simplify: no scene manager");
                jobs_.failed(
                    splat_simplify_state_.job,
                    "No scene manager");
                splat_simplify_state_.completed.store(false);
                publishSplatSimplifyState();
                return;
            }

            std::unique_ptr<lfs::core::SplatData> result;
            std::string source_name;
            std::string output_name;
            {
                const std::lock_guard lock(splat_simplify_state_.mutex);
                result = std::move(splat_simplify_state_.result);
                source_name = splat_simplify_state_.source_name;
                output_name = splat_simplify_state_.output_name;
            }

            if (!result) {
                LOG_ERROR("Splat simplify: missing result payload");
                jobs_.failed(
                    splat_simplify_state_.job,
                    "Missing result payload");
                splat_simplify_state_.completed.store(false);
                publishSplatSimplifyState();
                return;
            }

            const auto added_name = scene_manager->addGeneratedSplatNode(std::move(result), source_name, output_name, true);
            if (added_name.empty()) {
                jobs_.failed(
                    splat_simplify_state_.job,
                    LOC(lichtfeld::Strings::Runtime::SIMPLIFIED_SPLAT_ADD_FAILED),
                    LOC(lichtfeld::Strings::Runtime::TASK_FAILED));
            } else {
                jobs_.report(
                    splat_simplify_state_.job,
                    1.0F, LOC(lichtfeld::Strings::Runtime::TASK_COMPLETE));
                jobs_.completed(
                    splat_simplify_state_.job);
            }
            splat_simplify_state_.completed.store(false);
            publishSplatSimplifyState();
            return;
        }

        if (!splat_simplify_state_.completed.load())
            return;

        if (splat_simplify_state_.thread && splat_simplify_state_.thread->joinable()) {
            splat_simplify_state_.thread->join();
            splat_simplify_state_.thread.reset();
        }
        splat_simplify_state_.completed.store(false);
        if (const auto state =
                jobs_.update(
                    splat_simplify_state_.job);
            state &&
            state->status ==
                JobStatus::CompletionPending) {
            if (state->worker_canceled) {
                jobs_.canceled(
                    splat_simplify_state_.job);
            } else if (!state->error.empty()) {
                jobs_.failed(
                    splat_simplify_state_.job,
                    state->error, state->stage);
            } else {
                jobs_.completed(
                    splat_simplify_state_.job);
            }
        }
        publishSplatSimplifyState();
    }

    void AsyncTaskManager::cancelSplatSimplify() {
        jobs_.requestCancel(
            splat_simplify_state_.job,
            LOC(lichtfeld::Strings::Runtime::TASK_CANCELLING));
        publishSplatSimplifyState();
        if (splat_simplify_state_.thread) {
            splat_simplify_state_.thread->request_stop();
        }
    }

} // namespace lfs::vis::gui

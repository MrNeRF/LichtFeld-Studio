/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/async_task_manager.hpp"
#include "core/data_loading_service.hpp"
#include "core/events.hpp"
#include "core/logger.hpp"
#include "core/parameters.hpp"
#include "core/path_utils.hpp"
#include "core/scene.hpp"
#include "core/services.hpp"
#include "gui/gui_manager.hpp"
#include "gui/html_viewer_export.hpp"
#include "gui/panel_registry.hpp"
#include "gui/utils/native_file_dialog.hpp"
#include "gui/video_export_utils.hpp"
#include "internal/resource_paths.hpp"
#include "io/exporter.hpp"
#include "io/formats/colmap.hpp"
#include "rendering/mesh2splat.hpp"
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
#include "window/window_manager.hpp"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <condition_variable>
#include <format>
#include <functional>
#include <future>
#include <shared_mutex>
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

    [[nodiscard]] std::unique_ptr<lfs::core::SplatData> cloneSplatData(const lfs::core::SplatData& src) {
        auto cloned = std::make_unique<lfs::core::SplatData>(
            src.get_max_sh_degree(),
            src.means_raw().clone(),
            src.sh0_raw().clone(),
            src.shN_raw().is_valid() ? src.shN_raw().clone() : lfs::core::Tensor{},
            src.scaling_raw().clone(),
            src.rotation_raw().clone(),
            src.opacity_raw().clone(),
            src.get_scene_scale(),
            lfs::core::SplatData::ShNLayout::Swizzled);
        cloned->set_active_sh_degree(src.get_active_sh_degree());
        cloned->set_max_sh_degree(src.get_max_sh_degree());
        if (src.has_deleted_mask()) {
            cloned->deleted() = src.deleted().clone();
        }
        if (src._densification_info.is_valid()) {
            cloned->_densification_info = src._densification_info.clone();
        }
        return cloned;
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

        if (node->name == scene.getTrainingModelNodeName()) {
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
            return std::unexpected("COLMAP export requires a loaded dataset");
        }

        const auto source_path = scene_manager.getDatasetPath();
        if (source_path.empty()) {
            return std::unexpected("COLMAP export requires a source dataset path");
        }

        const auto& scene = scene_manager.getScene();
        auto cameras = scene.getAllCameras();
        if (cameras.empty()) {
            return std::unexpected("COLMAP export requires scene cameras");
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
        constexpr std::string_view shutdown_error = "Viewer is shutting down";
        constexpr std::string_view task_error = "Viewer work failed";

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

    lfs::core::Tensor orientVideoExportFrameForEncoder(const lfs::core::Tensor& image) {
        if (!image.is_valid() || image.ndim() != 3) {
            return image;
        }
        return image.contiguous();
    }

    void applyVideoExportPointCloudFilters(rendering::PointCloudFilterState& filters,
                                           const VideoExportSceneSnapshot& snapshot,
                                           const RenderSettings& render_settings) {
        if (!(render_settings.use_crop_box || render_settings.show_crop_box) || snapshot.cropboxes.empty()) {
            return;
        }

        const size_t idx = (snapshot.selected_cropbox_index >= 0)
                               ? static_cast<size_t>(snapshot.selected_cropbox_index)
                               : 0;
        if (idx >= snapshot.cropboxes.size() || !snapshot.cropboxes[idx].has_data) {
            return;
        }

        const auto& cb = snapshot.cropboxes[idx];
        filters.crop_box = rendering::BoundingBox{
            .min = cb.data.min,
            .max = cb.data.max,
            .transform = glm::inverse(cb.world_transform)};
        filters.crop_inverse = cb.data.inverse;
        filters.crop_desaturate = render_settings.show_crop_box &&
                                  !render_settings.use_crop_box &&
                                  render_settings.desaturate_cropping;
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
            });
        }
        return state;
    }

    std::expected<lfs::core::Tensor, std::string> makeGaussianPreviewVideoFrame(
        const std::shared_ptr<lfs::core::Tensor>& image) {
        if (!image || !image->is_valid() || image->ndim() != 3) {
            return std::unexpected("Rendered Gaussian frame is invalid");
        }
        if (image->size(0) <= 0 || image->size(1) <= 0 ||
            (image->size(2) != 3 && image->size(2) != 4)) {
            return std::unexpected("Rendered Gaussian frame must have shape [H, W, 3] or [H, W, 4]");
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

    rendering::FrameMetadata makeVideoExportFrameMetadata(const rendering::FrameView& frame_view,
                                                          const bool color_has_alpha) {
        return rendering::FrameMetadata{
            .valid = true,
            .far_plane = frame_view.far_plane,
            .orthographic = frame_view.orthographic,
            .color_has_alpha = color_has_alpha};
    }

    std::expected<lfs::core::Tensor, std::string> renderVideoExportFrame(
        RenderingManager& rendering_manager,
        rendering::RenderingEngine& engine,
        VideoExportEnvironmentState& environment_state,
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
                        return std::unexpected(render_result ? "Rendered point cloud frame is invalid"
                                                             : render_result.error());
                    }
                    return *render_result->image;
                }

                auto render_result = engine.renderPointCloudGpuFrame(*snapshot.combined_model, request);
                if (!render_result || !render_result->valid()) {
                    return std::unexpected(render_result ? "Rendered point cloud frame is invalid"
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
                    makeVideoExportFrameMetadata(frame_view, render_environment),
                    {width, height});
                if (!materialized || !materialized->valid()) {
                    return std::unexpected(materialized ? "Rendered Gaussian frame is invalid"
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
                return std::unexpected(render_result ? "Rendered point cloud frame is invalid"
                                                     : render_result.error());
            }

            if (!requires_composite_pass) {
                auto readback_result = engine.readbackGpuFrameColor(*render_result);
                if (!readback_result || !*readback_result) {
                    return std::unexpected(readback_result ? "Rendered point cloud frame is invalid"
                                                           : readback_result.error());
                }
                return *(*readback_result);
            }

            primary_frame = std::move(*render_result);
        }

        if (!requires_composite_pass) {
            return std::unexpected("No rendered image produced for video export");
        }

        const bool any_selected = std::any_of(snapshot.meshes.begin(), snapshot.meshes.end(),
                                              [](const auto& mesh) { return mesh.is_selected; }) ||
                                  std::any_of(snapshot.selected_node_mask.begin(),
                                              snapshot.selected_node_mask.end(),
                                              [](const bool selected) { return selected; });

        std::vector<rendering::MeshFrameItem> mesh_items;
        mesh_items.reserve(snapshot.meshes.size());
        for (const auto& mesh_snapshot : snapshot.meshes) {
            if (!mesh_snapshot.mesh)
                continue;
            mesh_items.push_back(rendering::MeshFrameItem{
                .mesh = mesh_snapshot.mesh.get(),
                .transform = mesh_snapshot.transform,
                .options = makeVideoExportMeshOptions(
                    render_settings, any_selected, mesh_snapshot.is_selected),
            });
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
            .meshes = std::move(mesh_items),
        };
        return engine.renderVideoCompositeFrame(primary_frame, composite_request);
    }

    AsyncTaskManager::AsyncTaskManager(VisualizerImpl* viewer)
        : viewer_(viewer) {}

    AsyncTaskManager::~AsyncTaskManager() {
        shutdown();
    }

    void AsyncTaskManager::resetVideoExportEnvironmentState() {
        video_export_environment_state_.reset();
    }

    void AsyncTaskManager::shutdown() {
        if (export_state_.active.load())
            cancelExport();
        if (export_state_.thread && export_state_.thread->joinable())
            export_state_.thread->join();
        export_state_.thread.reset();

        if (video_export_state_.active.load())
            cancelVideoExport();
        if (video_export_state_.thread && video_export_state_.thread->joinable())
            video_export_state_.thread->join();
        video_export_state_.thread.reset();
        if (viewer_ && viewer_->isOnViewerThread()) {
            resetVideoExportEnvironmentState();
        }

        if (import_state_.thread) {
            import_state_.thread->request_stop();
            if (import_state_.thread->joinable())
                import_state_.thread->join();
            import_state_.thread.reset();
        }
        cancelImportCompletionDismiss();

        mesh2splat_state_.active.store(false);
        mesh2splat_state_.pending.store(false);

        if (splat_simplify_state_.active.load())
            cancelSplatSimplify();
        if (splat_simplify_state_.thread && splat_simplify_state_.thread->joinable())
            splat_simplify_state_.thread->join();
        splat_simplify_state_.thread.reset();
    }

    void AsyncTaskManager::setupEvents() {
        using namespace lfs::core::events;

        cmd::LoadFile::when([this](const auto& cmd) {
            if (!cmd.is_dataset)
                return;
            const auto* const data_loader = viewer_->getDataLoader();
            if (!data_loader) {
                LOG_ERROR("LoadFile: no data loader");
                return;
            }
            auto params = data_loader->getParameters();
            params.init_path = std::nullopt;
            params.resume_checkpoint = std::nullopt;
            params.dataset.output_path =
                cmd.output_path.empty() ? lfs::core::param::default_dataset_output_path(cmd.path) : cmd.output_path;
            if (!cmd.init_path.empty())
                params.init_path = lfs::core::path_to_utf8(cmd.init_path);
            if (!cmd.centralize_dataset.empty())
                params.dataset.centralize_dataset = cmd.centralize_dataset;
            if (cmd.max_width.has_value() && *cmd.max_width >= 0)
                params.dataset.max_width = *cmd.max_width;
            if (cmd.min_track_length.has_value() && *cmd.min_track_length >= 0)
                params.dataset.min_track_length = *cmd.min_track_length;
            import_state_.apply_auto_crop.store(cmd.apply_auto_crop);
            startAsyncImport(cmd.path, params);
        });

        state::DatasetLoadStarted::when([this](const auto& e) {
            if (import_state_.active.load())
                return;
            cancelImportCompletionDismiss();
            import_state_.active.store(true);
            import_state_.progress.store(0.0f);
            {
                const std::lock_guard lock(import_state_.mutex);
                import_state_.path = e.path;
                import_state_.stage = "Initializing...";
                import_state_.error.clear();
                import_state_.num_images = 0;
                import_state_.num_points = 0;
                import_state_.success = false;
                import_state_.dataset_type = getDatasetTypeName(e.path);
            }
            publishImportOverlayState();
        });

        state::DatasetLoadProgress::when([this](const auto& e) {
            import_state_.progress.store(e.progress / 100.0f);
            {
                const std::lock_guard lock(import_state_.mutex);
                import_state_.stage = e.step;
            }
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
            {
                const std::lock_guard lock(import_state_.mutex);
                import_state_.success = e.success;
                import_state_.num_images = e.num_images;
                import_state_.num_points = e.num_points;
                import_state_.completion_time = std::chrono::steady_clock::now();
                import_state_.error = e.error.value_or("");
                import_state_.stage = e.success ? "Complete" : "Failed";
                import_state_.progress.store(1.0f);
            }
            import_state_.active.store(false);
            import_state_.show_completion.store(true);
            if (e.success)
                scheduleImportCompletionDismiss();
            publishImportOverlayState();
        });

        cmd::SequencerExportVideo::when([this](const auto& evt) {
            const auto path = SaveMp4FileDialog("camera_path");
            if (path.empty())
                return;

            io::video::VideoExportOptions options;
            options.width = evt.width;
            options.height = evt.height;
            options.framerate = evt.framerate;
            options.crf = evt.crf;
            startVideoExport(path, options);
        });
    }

    void AsyncTaskManager::pollImportCompletion() {
        checkAsyncImportCompletion();
    }

    bool AsyncTaskManager::hasPendingMainThreadCompletions() const {
        return import_state_.load_complete.load(std::memory_order_acquire) ||
               mesh2splat_state_.pending.load(std::memory_order_acquire) ||
               splat_simplify_state_.apply_pending.load(std::memory_order_acquire) ||
               splat_simplify_state_.completed.load(std::memory_order_acquire);
    }

    void AsyncTaskManager::performExport(ExportFormat format, const std::filesystem::path& path,
                                         const std::vector<std::string>& node_names, int sh_degree,
                                         bool rad_flip_y,
                                         bool rad_streamable) {
        if (isExporting())
            return;

        if (format == ExportFormat::COLMAP) {
            startColmapExport(path);
            return;
        }

        auto* const scene_manager = viewer_->getSceneManager();
        if (!scene_manager) {
            publishExportFailureState(format, path, "Scene manager is not available");
            return;
        }
        if (node_names.empty()) {
            publishExportFailureState(format, path, "No model selected for export");
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
            publishExportFailureState(format, path, "No splat data to export");
            return;
        }

        auto borrow_plan = makeBorrowSingleIdentityExportPlan(*scene_manager, node_names);
        startAsyncExport(format,
                         path,
                         std::move(splats),
                         sh_degree,
                         borrow_plan.storage_mode == core::Scene::MergeStorageMode::BorrowSingleIdentity,
                         borrow_plan.model_mutex,
                         rad_flip_y,
                         rad_streamable);
    }

    void AsyncTaskManager::startColmapExport(const std::filesystem::path& path) {
        if (isExporting())
            return;

        auto* const scene_manager = viewer_->getSceneManager();
        if (!scene_manager) {
            std::string error = "Scene manager not initialized";
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

        export_state_.active.store(true);
        export_state_.cancel_requested.store(false);
        export_state_.progress.store(0.0f);
        {
            const std::lock_guard lock(export_state_.mutex);
            export_state_.format = ExportFormat::COLMAP;
            export_state_.stage = "Starting";
            export_state_.error.clear();
            export_state_.path = path;
        }
        publishExportState();

        LOG_INFO("COLMAP export started: {}", lfs::core::path_to_utf8(path));

        export_state_.thread.emplace(
            [this, path, snapshot = std::move(*snapshot_result)](std::stop_token stop_token) mutable {
                bool success = false;
                bool cancelled = false;
                std::string error_msg;

                auto update_stage = [this](float progress, const std::string& stage) {
                    export_state_.progress.store(progress);
                    {
                        const std::lock_guard lock(export_state_.mutex);
                        export_state_.stage = stage;
                    }
                    publishExportState();
                    if (auto* window_manager = services().windowOrNull()) {
                        window_manager->wakeEventLoop();
                    }
                };

                try {
                    if (stop_token.stop_requested() || export_state_.cancel_requested.load()) {
                        cancelled = true;
                        error_msg = "Export cancelled by user";
                    } else {
                        update_stage(0.1f, "Writing COLMAP sparse files");
                        auto result = io::write_colmap_reconstruction(
                            snapshot.source_path,
                            path,
                            snapshot.cameras,
                            snapshot.point_cloud.get(),
                            snapshot.point_cloud_transform,
                            io::ColmapWriteOptions{.format = io::ColmapWriteFormat::Auto});
                        if (result) {
                            success = true;
                            update_stage(1.0f, "Complete");
                        } else {
                            error_msg = result.error().message;
                        }
                    }
                } catch (const std::exception& e) {
                    error_msg = std::string("COLMAP export crashed with exception: ") + e.what();
                } catch (...) {
                    error_msg = "COLMAP export crashed with unknown exception";
                }

                if (success && (stop_token.stop_requested() || export_state_.cancel_requested.load())) {
                    success = false;
                    cancelled = true;
                    error_msg = "Export cancelled by user";
                }

                if (success) {
                    LOG_INFO("COLMAP export completed: {}", lfs::core::path_to_utf8(path));
                    lfs::core::events::state::ExportCompleted{
                        .path = path,
                        .format = ExportFormat::COLMAP}
                        .emit();
                } else if (cancelled) {
                    LOG_INFO("COLMAP export cancelled: {}", lfs::core::path_to_utf8(path));
                    {
                        const std::lock_guard lock(export_state_.mutex);
                        export_state_.error = error_msg;
                        export_state_.stage = "Cancelled";
                    }
                    publishExportState();
                    lfs::core::events::state::ExportFailed{.error = error_msg, .cancelled = true}.emit();
                } else {
                    LOG_ERROR("COLMAP export failed: {}", error_msg);
                    {
                        const std::lock_guard lock(export_state_.mutex);
                        export_state_.error = error_msg;
                        export_state_.stage = "Failed";
                    }
                    publishExportState();
                    lfs::core::events::state::ExportFailed{.error = error_msg}.emit();
                }

                lfs::core::Tensor::trim_memory_pool();
                export_state_.active.store(false);
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
                                            bool rad_streamable) {
        if (splats.empty()) {
            LOG_ERROR("No splat data to export");
            publishExportFailureState(format, path, "No splat data to export");
            return;
        }

        export_state_.active.store(true);
        export_state_.cancel_requested.store(false);
        export_state_.progress.store(0.0f);
        {
            const std::lock_guard lock(export_state_.mutex);
            export_state_.format = format;
            export_state_.stage = "Starting";
            export_state_.error.clear();
            export_state_.path = path;
        }
        publishExportState();

        LOG_INFO("Export started: {} (format: {})", lfs::core::path_to_utf8(path), static_cast<int>(format));

        export_state_.thread.emplace(
            [this,
             format,
             path,
             splats = std::move(splats),
             sh_degree,
             borrow_single_identity,
             model_mutex,
             rad_flip_y,
             rad_streamable](
                std::stop_token stop_token) mutable {
                bool cancellation_logged = false;
                auto update_progress = [this, &stop_token, &cancellation_logged](float progress, const std::string& stage) -> bool {
                    if (stop_token.stop_requested() || export_state_.cancel_requested.load()) {
                        if (!cancellation_logged) {
                            LOG_INFO("Export cancelled");
                            cancellation_logged = true;
                        }
                        {
                            const std::lock_guard lock(export_state_.mutex);
                            export_state_.stage = "Cancelled";
                        }
                        publishExportState();
                        if (auto* window_manager = services().windowOrNull()) {
                            window_manager->wakeEventLoop();
                        }
                        return false;
                    }
                    export_state_.progress.store(progress);
                    {
                        const std::lock_guard lock(export_state_.mutex);
                        export_state_.stage = stage;
                    }
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
                    if (!update_progress(0.0f, "Preparing export data")) {
                        cancelled = true;
                        error_msg = "Export cancelled by user";
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
                            error_msg = "No splat data to export";
                        } else if (sh_degree < splat_data->get_max_sh_degree()) {
                            truncateSHDegree(*splat_data, sh_degree);
                        }
                        model_lock.reset();
                    }

                    if (!cancelled && splat_data && !update_progress(0.0f, "Export data prepared")) {
                        cancelled = true;
                        error_msg = "Export cancelled by user";
                    }

                    if (!cancelled && splat_data) {
                        switch (format) {
                        case ExportFormat::PLY: {
                            const lfs::io::PlySaveOptions options{
                                .output_path = path,
                                .binary = true,
                                .async = false,
                                .progress_callback = update_progress,
                                .extra_attributes = {}};
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
                                        .is_disk_space_error = true,
                                        .is_checkpoint = false}
                                        .emit();
                                }
                            }
                            break;
                        }
                        case ExportFormat::SOG: {
                            const lfs::io::SogSaveOptions options{
                                .output_path = path,
                                .kmeans_iterations = 10,
                                .progress_callback = update_progress};
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
                                .progress_callback = update_progress};
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
                                .progress_callback = update_progress};
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
                                .progress_callback = update_progress};
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
                                .progress_callback = update_progress};
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
                                .progress_callback = update_progress};
                            if (auto result = lfs::io::save_rad(*splat_data, options); result) {
                                success = true;
                            } else {
                                error_msg = result.error().message;
                                cancelled = result.error().code == lfs::io::ErrorCode::CANCELLED;
                            }
                            break;
                        }
                        case ExportFormat::COLMAP:
                            error_msg = "COLMAP export uses the dataset write-back path";
                            break;
                        }
                    }

                } catch (const std::exception& e) {
                    error_msg = std::string("Export crashed with exception: ") + e.what();
                    LOG_ERROR("{}", error_msg);
                } catch (...) {
                    error_msg = "Export crashed with unknown exception";
                    LOG_ERROR("{}", error_msg);
                }

                if (success && (stop_token.stop_requested() || export_state_.cancel_requested.load())) {
                    success = false;
                    cancelled = true;
                    error_msg = "Export cancelled by user";
                }

                if (success) {
                    LOG_INFO("Export completed: {}", lfs::core::path_to_utf8(path));
                    {
                        const std::lock_guard lock(export_state_.mutex);
                        export_state_.stage = "Complete";
                    }
                    publishExportState();
                    lfs::core::events::state::ExportCompleted{
                        .path = path,
                        .format = format}
                        .emit();
                } else if (cancelled) {
                    LOG_INFO("Export cancelled: {}", lfs::core::path_to_utf8(path));
                    {
                        const std::lock_guard lock(export_state_.mutex);
                        export_state_.error = error_msg;
                        export_state_.stage = "Cancelled";
                    }
                    publishExportState();
                    lfs::core::events::state::ExportFailed{
                        .error = error_msg,
                        .cancelled = true}
                        .emit();
                } else {
                    LOG_ERROR("Export failed: {}", error_msg);
                    {
                        const std::lock_guard lock(export_state_.mutex);
                        export_state_.error = error_msg;
                        export_state_.stage = "Failed";
                    }
                    publishExportState();
                    lfs::core::events::state::ExportFailed{
                        .error = error_msg}
                        .emit();
                }

                splat_data.reset();
                lfs::core::Tensor::trim_memory_pool();
                export_state_.active.store(false);
                publishExportState();
            });
    }

    void AsyncTaskManager::cancelExport() {
        if (!export_state_.active.load())
            return;
        LOG_INFO("Cancelling export");
        export_state_.cancel_requested.store(true);
        {
            const std::lock_guard lock(export_state_.mutex);
            export_state_.stage = "Cancelling";
        }
        publishExportState();
        if (export_state_.thread && export_state_.thread->joinable()) {
            export_state_.thread->request_stop();
        }
    }

    void AsyncTaskManager::publishExportFailureState(const ExportFormat format,
                                                     const std::filesystem::path& path,
                                                     std::string error) {
        export_state_.active.store(false);
        export_state_.cancel_requested.store(false);
        export_state_.progress.store(0.0f);
        {
            const std::lock_guard lock(export_state_.mutex);
            export_state_.format = format;
            export_state_.stage = "Failed";
            export_state_.error = std::move(error);
            export_state_.path = path;
        }
        publishExportState();
    }

    void AsyncTaskManager::publishExportState() {
        lfs::vis::AppStore::ExportProgressState state;
        state.active = export_state_.active.load();
        state.progress = export_state_.progress.load();
        {
            const std::lock_guard lock(export_state_.mutex);
            state.stage = export_state_.stage;
            state.format = exportProgressFormatName(export_state_.format);
            state.error = export_state_.error;
            state.path = lfs::core::path_to_utf8(export_state_.path);
        }
        lfs::vis::app_store().export_progress_state.set(std::move(state));
    }

    void AsyncTaskManager::publishImportOverlayState() {
        lfs::vis::AppStore::ImportOverlayState state;
        state.active = import_state_.active.load();
        state.show_completion = import_state_.show_completion.load();
        state.progress = import_state_.progress.load();
        {
            const std::lock_guard lock(import_state_.mutex);
            state.stage = import_state_.stage;
            state.dataset_type = import_state_.dataset_type;
            state.path = lfs::core::path_to_utf8(import_state_.path.filename());
            state.success = import_state_.success;
            state.error = import_state_.error;
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

    void AsyncTaskManager::publishVideoExportOverlayState() {
        lfs::vis::AppStore::VideoExportOverlayState state;
        state.active = video_export_state_.active.load();
        state.progress = video_export_state_.progress.load();
        state.current_frame = video_export_state_.current_frame.load();
        state.total_frames = video_export_state_.total_frames.load();
        {
            const std::lock_guard lock(video_export_state_.mutex);
            state.stage = video_export_state_.stage;
        }
        lfs::vis::app_store().video_export_overlay_state.set(std::move(state));
    }

    void AsyncTaskManager::publishMesh2SplatState() {
        lfs::vis::AppStore::TaskProgressState state;
        state.active = mesh2splat_state_.active.load();
        state.progress = mesh2splat_state_.progress.load();
        {
            const std::lock_guard lock(mesh2splat_state_.mutex);
            state.stage = mesh2splat_state_.stage;
            state.error = mesh2splat_state_.error;
            state.source_name = mesh2splat_state_.source_name;
        }
        lfs::vis::app_store().mesh2splat_state.set(std::move(state));
    }

    void AsyncTaskManager::publishSplatSimplifyState() {
        lfs::vis::AppStore::TaskProgressState state;
        state.active = splat_simplify_state_.active.load();
        state.progress = splat_simplify_state_.progress.load();
        {
            const std::lock_guard lock(splat_simplify_state_.mutex);
            state.stage = splat_simplify_state_.stage;
            state.error = splat_simplify_state_.error;
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
                if (import_state_.active.load() || !import_state_.show_completion.load())
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
        const bool had_activity = import_state_.active.load() ||
                                  import_state_.show_completion.load() ||
                                  import_state_.thread.has_value();
        if (!had_activity) {
            return;
        }

        LOG_INFO("Cancelling import");
        cancelImportCompletionDismiss();
        if (import_state_.thread) {
            import_state_.thread->request_stop();
            if (import_state_.thread->joinable()) {
                import_state_.thread->join();
            }
            import_state_.thread.reset();
        }

        import_state_.active.store(false);
        import_state_.load_complete.store(false);
        import_state_.show_completion.store(false);
        import_state_.progress.store(0.0f);
        {
            const std::lock_guard lock(import_state_.mutex);
            import_state_.path.clear();
            import_state_.stage.clear();
            import_state_.dataset_type.clear();
            import_state_.error.clear();
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
        if (import_state_.active.load()) {
            LOG_WARN("Import already in progress");
            return;
        }

        cancelImportCompletionDismiss();
        import_state_.active.store(true);
        import_state_.load_complete.store(false);
        import_state_.show_completion.store(false);
        import_state_.progress.store(0.0f);
        PanelRegistry::instance().invalidate_poll_cache();
        {
            const std::lock_guard lock(import_state_.mutex);
            import_state_.path = path;
            import_state_.stage = "Initializing...";
            import_state_.error.clear();
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

        import_state_.thread.emplace(
            [this, path](const std::stop_token stop_token) noexcept {
                const auto record_failure = [this](const char* detail) noexcept {
                    try {
                        const std::lock_guard lock(import_state_.mutex);
                        import_state_.success = false;
                        import_state_.load_result.reset();
                        import_state_.stage = "Failed";
                        import_state_.error.clear();
                    } catch (...) {
                    }

                    try {
                        const std::string message = detail && *detail
                                                        ? std::format("Import failed with exception: {}", detail)
                                                        : "Import failed with an unknown exception";
                        {
                            const std::lock_guard lock(import_state_.mutex);
                            import_state_.error = message;
                        }
                        LOG_ERROR("{}", message);
                    } catch (...) {
                        // The worker boundary must remain no-throw even when
                        // reporting an allocation failure. success was reset
                        // before launch, so completion still follows the
                        // failure path if diagnostic allocation also fails.
                    }
                };

                const ScopeExit publish_terminal([this, &stop_token]() noexcept {
                    if (stop_token.stop_requested()) {
                        import_state_.active.store(false);
                    } else {
                        import_state_.progress.store(1.0f);
                        import_state_.load_complete.store(true, std::memory_order_release);
                    }

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
                        LOG_WARN(
                            "min-track-length cannot be used with --init; COLMAP sparse point filtering will not be applied because initialization uses '{}'",
                            *local_params.init_path);
                        effective_min_track_length = 0;
                    }
                    const lfs::io::LoadOptions load_options{
                        .resize_factor = local_params.dataset.resize_factor,
                        .max_width = local_params.dataset.max_width,
                        .images_folder = local_params.dataset.images,
                        .min_track_length = effective_min_track_length,
                        .validate_only = false,
                        .centralize = parse_centralize(local_params.dataset.centralize_dataset),
                        .progress = [this, &stop_token](const float pct, const std::string& msg) {
                        if (stop_token.stop_requested())
                            return;
                        import_state_.progress.store(pct / 100.0f);
                        {
                            const std::lock_guard lock(import_state_.mutex);
                            import_state_.stage = msg;
                        }
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
                            import_state_.stage = "Applying...";
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
                            import_state_.error = result.error().format();
                            import_state_.stage = "Failed";
                            LOG_ERROR("Import failed: {}", import_state_.error);
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
            import_state_.active.store(false);
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
            import_state_.active.store(false);
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
            import_state_.active.store(false);
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
            import_state_.stage = result ? "Complete" : "Failed";
            if (!result)
                import_state_.error = result.error();
            success_val = import_state_.success;
            error_val = import_state_.error;
            num_images_val = import_state_.num_images;
            num_points_val = import_state_.num_points;
        }

        import_state_.active.store(false);
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
        if (!video_export_state_.active.load())
            return;
        LOG_INFO("Cancelling video export");
        video_export_state_.cancel_requested.store(true);
        {
            std::lock_guard lock(video_export_state_.mutex);
            video_export_state_.stage = "Cancelling";
        }
        publishVideoExportOverlayState();
        if (video_export_state_.thread) {
            video_export_state_.thread->request_stop();
        }
    }

    void AsyncTaskManager::startVideoExport(const std::filesystem::path& path,
                                            const io::video::VideoExportOptions& options) {
        auto fail_start = [this, &path](std::string error) {
            LOG_ERROR("Cannot export video: {}", error);
            video_export_state_.active.store(false);
            video_export_state_.cancel_requested.store(false);
            video_export_state_.progress.store(0.0f);
            video_export_state_.total_frames.store(0);
            video_export_state_.current_frame.store(0);
            {
                std::lock_guard lock(video_export_state_.mutex);
                video_export_state_.stage = "Failed";
                video_export_state_.error = error;
                video_export_state_.path = path;
            }
            publishVideoExportOverlayState();
            lfs::core::events::state::VideoExportFailed{.error = std::move(error)}.emit();
        };

        if (video_export_state_.active.load()) {
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
            fail_start("Missing scene or rendering manager");
            return;
        }

        auto* gui_manager = viewer_->getGuiManager();
        if (!gui_manager) {
            fail_start("GUI manager is not available");
            return;
        }
        const auto& timeline = gui_manager->sequencer().timeline();
        if (timeline.empty()) {
            fail_start("No keyframes to export");
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
            fail_start("Rendering engine is not available");
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

        video_export_state_.active.store(true);
        video_export_state_.cancel_requested.store(false);
        video_export_state_.progress.store(0.0f);
        video_export_state_.total_frames.store(total_frames);
        video_export_state_.current_frame.store(0);
        {
            std::lock_guard lock(video_export_state_.mutex);
            video_export_state_.stage = "Initializing";
            video_export_state_.error.clear();
            video_export_state_.path = path;
        }
        publishVideoExportOverlayState();

        resetVideoExportEnvironmentState();
        video_export_environment_state_ = std::make_unique<VideoExportEnvironmentState>();

        LOG_INFO("Starting video export: {} frames at {}x{}", total_frames, width, height);

        video_export_state_.thread.emplace(
            [this, viewer = viewer_, path, export_options, total_frames, width, height,
             engine, rendering_manager, render_settings,
             environment_state = video_export_environment_state_.get(),
             snapshot = *snapshot_result,
             frame_states = std::move(frame_states)](std::stop_token stop_token) mutable {
                bool cancelled = false;
                auto cleanup_environment_state = [this, viewer]() {
                    if (!video_export_environment_state_) {
                        return;
                    }
                    auto cleanup_result = postToViewerAndWait(
                        viewer,
                        [this]() -> std::expected<void, std::string> {
                            resetVideoExportEnvironmentState();
                            return {};
                        });
                    if (!cleanup_result) {
                        LOG_DEBUG("Skipping video export environment cleanup: {}", cleanup_result.error());
                    }
                };

                auto encoder = lfs::gui::createVideoEncoder();
                if (!encoder) {
                    {
                        std::lock_guard lock(video_export_state_.mutex);
                        video_export_state_.error = "Video encoder not available";
                        video_export_state_.stage = "Failed";
                    }
                    video_export_state_.active.store(false);
                    publishVideoExportOverlayState();
                    lfs::core::events::state::VideoExportFailed{
                        .error = "Video encoder not available"}
                        .emit();
                    cleanup_environment_state();
                    return;
                }

                {
                    std::lock_guard lock(video_export_state_.mutex);
                    video_export_state_.stage = "Opening encoder";
                }
                publishVideoExportOverlayState();

                auto result = encoder->open(path, export_options);
                if (!result) {
                    {
                        std::lock_guard lock(video_export_state_.mutex);
                        video_export_state_.error = result.error();
                        video_export_state_.stage = "Failed: " + result.error();
                    }
                    LOG_ERROR("Failed to open encoder: {}", result.error());
                    lfs::core::events::state::VideoExportFailed{
                        .error = result.error()}
                        .emit();
                    video_export_state_.active.store(false);
                    publishVideoExportOverlayState();
                    cleanup_environment_state();
                    return;
                }

                for (int frame = 0; frame < total_frames; ++frame) {
                    if (stop_token.stop_requested() || video_export_state_.cancel_requested.load()) {
                        LOG_INFO("Video export cancelled at frame {}", frame);
                        cancelled = true;
                        break;
                    }

                    auto frame_tensor = postToViewerAndWait(
                        viewer,
                        [engine, rendering_manager, environment_state, snapshot, render_settings, width, height,
                         cam_state = frame_states[frame]]() -> std::expected<lfs::core::Tensor, std::string> {
                            return renderVideoExportFrame(
                                *rendering_manager,
                                *engine,
                                *environment_state,
                                snapshot,
                                render_settings,
                                cam_state,
                                width,
                                height);
                        });

                    if (!frame_tensor) {
                        LOG_ERROR("Failed to render frame {}: {}", frame, frame_tensor.error());
                        {
                            std::lock_guard lock(video_export_state_.mutex);
                            video_export_state_.error = std::format(
                                "Failed to render frame {}: {}", frame + 1, frame_tensor.error());
                            video_export_state_.stage = "Render error";
                        }
                        publishVideoExportOverlayState();
                        break;
                    }

                    auto export_frame = orientVideoExportFrameForEncoder(*frame_tensor);
                    auto image_hwc = export_frame.permute({1, 2, 0}).contiguous();

                    if (frame == 0) {
                        LOG_INFO("Video export: CHW shape=[{},{},{}] -> HWC shape=[{},{},{}]",
                                 export_frame.shape()[0], export_frame.shape()[1], export_frame.shape()[2],
                                 image_hwc.shape()[0], image_hwc.shape()[1], image_hwc.shape()[2]);
                    }

                    const auto* const gpu_ptr = image_hwc.data_ptr();
                    auto write_result = encoder->writeFrameGpu(gpu_ptr, width, height, nullptr);
                    if (!write_result) {
                        {
                            std::lock_guard lock(video_export_state_.mutex);
                            video_export_state_.error = write_result.error();
                            video_export_state_.stage = "Encode error";
                        }
                        publishVideoExportOverlayState();
                        LOG_ERROR("Failed to encode frame {}: {}", frame, write_result.error());
                        break;
                    }

                    video_export_state_.current_frame.store(frame + 1);
                    video_export_state_.progress.store(
                        static_cast<float>(frame + 1) / static_cast<float>(total_frames));
                    {
                        std::lock_guard lock(video_export_state_.mutex);
                        video_export_state_.stage = std::format("Encoding frame {}/{}", frame + 1, total_frames);
                    }
                    publishVideoExportOverlayState();
                }

                {
                    std::lock_guard lock(video_export_state_.mutex);
                    if (cancelled) {
                        video_export_state_.stage = "Cancelled";
                    } else if (video_export_state_.error.empty()) {
                        video_export_state_.stage = "Finalizing";
                    }
                }
                publishVideoExportOverlayState();

                if (auto close_result = encoder->close(); !close_result) {
                    {
                        std::lock_guard lock(video_export_state_.mutex);
                        video_export_state_.error = close_result.error();
                        video_export_state_.stage = "Failed";
                    }
                    publishVideoExportOverlayState();
                    LOG_ERROR("Failed to close encoder: {}", close_result.error());
                } else {
                    bool emit_completed = false;
                    {
                        std::lock_guard lock(video_export_state_.mutex);
                        if (cancelled) {
                            video_export_state_.stage = "Cancelled";
                        } else if (video_export_state_.error.empty() && !video_export_state_.cancel_requested.load()) {
                            video_export_state_.stage = "Complete";
                            LOG_INFO("Video export completed: {}", lfs::core::path_to_utf8(path));
                            emit_completed = true;
                        }
                    }
                    publishVideoExportOverlayState();
                    if (emit_completed) {
                        lfs::core::events::state::VideoExportCompleted{
                            .path = path,
                            .total_frames = total_frames}
                            .emit();
                    }
                }

                {
                    std::string err;
                    {
                        std::lock_guard lock(video_export_state_.mutex);
                        err = video_export_state_.error;
                    }
                    if (!err.empty()) {
                        lfs::core::events::state::VideoExportFailed{
                            .error = std::move(err)}
                            .emit();
                    }
                }
                cleanup_environment_state();
                video_export_state_.active.store(false);
                publishVideoExportOverlayState();
            });
    }

    void AsyncTaskManager::startMesh2Splat(std::shared_ptr<lfs::core::MeshData> mesh,
                                           const std::string& source_name,
                                           const lfs::core::Mesh2SplatOptions& options) {
        if (mesh2splat_state_.active.load()) {
            LOG_WARN("Mesh2Splat conversion already in progress");
            return;
        }

        if (!mesh) {
            LOG_ERROR("Mesh2Splat: null mesh pointer");
            return;
        }

        mesh2splat_state_.active.store(true);
        mesh2splat_state_.progress.store(0.0f);
        {
            const std::lock_guard lock(mesh2splat_state_.mutex);
            mesh2splat_state_.stage = "Starting...";
            mesh2splat_state_.error.clear();
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
        } else {
            std::string err;
            {
                std::lock_guard lock(mesh2splat_state_.mutex);
                err = mesh2splat_state_.error;
            }
            if (!err.empty()) {
                lfs::core::events::state::Mesh2SplatFailed{
                    .error = std::move(err)}
                    .emit();
            }
        }

        mesh2splat_state_.active.store(false);
        mesh2splat_state_.progress.store(has_result ? 1.0f : 0.0f);
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

        if (!mesh)
            return;

        auto result = lfs::rendering::mesh_to_splat(
            *mesh,
            options,
            [this](const float progress, const std::string& stage) {
                mesh2splat_state_.progress.store(progress);
                {
                    const std::lock_guard lock(mesh2splat_state_.mutex);
                    mesh2splat_state_.stage = stage;
                }
                publishMesh2SplatState();
                return mesh2splat_state_.active.load();
            });

        {
            const std::lock_guard lock(mesh2splat_state_.mutex);
            if (result) {
                mesh2splat_state_.result = std::move(*result);
                mesh2splat_state_.error.clear();
                mesh2splat_state_.stage = "Complete";
            } else {
                mesh2splat_state_.result.reset();
                mesh2splat_state_.error = result.error();
                mesh2splat_state_.stage = "Failed";
                LOG_ERROR("Mesh2Splat conversion failed: {}", mesh2splat_state_.error);
            }
        }
        publishMesh2SplatState();
    }

    void AsyncTaskManager::applyMesh2SplatResult() {
        auto* const scene_manager = viewer_->getSceneManager();
        if (!scene_manager) {
            LOG_ERROR("Mesh2Splat: no scene manager");
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
            return;
        }

        const std::string node_name = source_name + " (splat)";
        auto& scene = scene_manager->getScene();

        if (scene.getNode(node_name))
            scene.removeNode(node_name);

        const std::string added_name =
            scene_manager->addGeneratedSplatNode(std::move(splat_data), source_name, node_name, true);
        if (added_name.empty()) {
            LOG_ERROR("Mesh2Splat: failed to add splat node '{}'", node_name);
            return;
        }

        {
            const std::lock_guard lock(mesh2splat_state_.mutex);
            mesh2splat_state_.stage = "Complete";
        }
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
        if (splat_simplify_state_.active.load()) {
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
                    return std::unexpected("No scene manager");
                }

                const auto* const node = scene_manager->getScene().getNode(source_name);
                if (!node || node->type != core::NodeType::SPLAT || !node->model) {
                    return std::unexpected(std::format("No splat node named '{}'", source_name));
                }

                const auto input_count = static_cast<int64_t>(node->model->size());
                const auto target_count = std::clamp<int64_t>(
                    static_cast<int64_t>(std::ceil(std::clamp(options.ratio, 0.0, 1.0) * static_cast<double>(input_count))),
                    int64_t{1},
                    std::max<int64_t>(int64_t{1}, input_count));
                return SimplifyCapture{
                    .model = cloneSplatData(*node->model),
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

        splat_simplify_state_.active.store(true);
        splat_simplify_state_.cancel_requested.store(false);
        splat_simplify_state_.completed.store(false);
        splat_simplify_state_.apply_pending.store(false);
        splat_simplify_state_.progress.store(0.0f);
        {
            const std::lock_guard lock(splat_simplify_state_.mutex);
            splat_simplify_state_.stage = "Starting...";
            splat_simplify_state_.error.clear();
            splat_simplify_state_.source_name = capture->source_name;
            splat_simplify_state_.output_name = capture->output_name;
            splat_simplify_state_.result.reset();
        }

        auto input = std::move(capture->model);
        auto opts = options;
        publishSplatSimplifyState();
        splat_simplify_state_.thread.emplace([this, opts, input = std::move(input)](std::stop_token stop_token) mutable {
            auto progress_cb = [this, &stop_token](const float progress, const std::string& stage) -> bool {
                if (stop_token.stop_requested() || splat_simplify_state_.cancel_requested.load())
                    return false;
                splat_simplify_state_.progress.store(progress);
                {
                    const std::lock_guard lock(splat_simplify_state_.mutex);
                    splat_simplify_state_.stage = stage;
                }
                publishSplatSimplifyState();
                return true;
            };

            auto result = lfs::core::simplify_splats(*input, opts, progress_cb);
            if (result) {
                {
                    const std::lock_guard lock(splat_simplify_state_.mutex);
                    splat_simplify_state_.result = std::move(*result);
                    splat_simplify_state_.stage = "Applying...";
                }
                splat_simplify_state_.progress.store(1.0f);
                splat_simplify_state_.apply_pending.store(true, std::memory_order_release);
                publishSplatSimplifyState();
            } else {
                const bool cancelled = splat_simplify_state_.cancel_requested.load() || stop_token.stop_requested() ||
                                       result.error() == "Cancelled";
                {
                    const std::lock_guard lock(splat_simplify_state_.mutex);
                    splat_simplify_state_.error = cancelled ? std::string{} : result.error();
                    splat_simplify_state_.stage = cancelled ? "Cancelled" : "Failed";
                }
                splat_simplify_state_.active.store(false);
                publishSplatSimplifyState();
            }
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
                splat_simplify_state_.active.store(false);
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
                splat_simplify_state_.active.store(false);
                splat_simplify_state_.completed.store(false);
                publishSplatSimplifyState();
                return;
            }

            const auto added_name = scene_manager->addGeneratedSplatNode(std::move(result), source_name, output_name, true);
            {
                const std::lock_guard lock(splat_simplify_state_.mutex);
                if (added_name.empty()) {
                    splat_simplify_state_.error = "Failed to add simplified splat node";
                    splat_simplify_state_.stage = "Failed";
                } else {
                    splat_simplify_state_.stage = "Complete";
                }
            }
            splat_simplify_state_.active.store(false);
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
        publishSplatSimplifyState();
    }

    void AsyncTaskManager::cancelSplatSimplify() {
        splat_simplify_state_.cancel_requested.store(true);
        {
            const std::lock_guard lock(splat_simplify_state_.mutex);
            splat_simplify_state_.stage = "Cancelling...";
            splat_simplify_state_.error.clear();
        }
        publishSplatSimplifyState();
        if (splat_simplify_state_.thread) {
            splat_simplify_state_.thread->request_stop();
        }
    }

} // namespace lfs::vis::gui

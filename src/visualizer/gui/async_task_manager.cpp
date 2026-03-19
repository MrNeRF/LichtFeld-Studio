/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/async_task_manager.hpp"
#include "core/data_loading_service.hpp"
#include "core/events.hpp"
#include "core/logger.hpp"
#include "core/scene.hpp"
#include "gui/gui_manager.hpp"
#include "gui/html_viewer_export.hpp"
#include "gui/panel_registry.hpp"
#include "gui/utils/native_file_dialog.hpp"
#include "gui/video_export_utils.hpp"
#include "io/exporter.hpp"
#include "rendering/framebuffer.hpp"
#include "rendering/mesh2splat.hpp"
#include "rendering/rendering.hpp"
#include "rendering/rendering_manager.hpp"
#include "scene/scene_manager.hpp"
#include "sequencer/keyframe.hpp"
#include "sequencer/sequencer_controller.hpp"
#include "training/training_manager.hpp"
#include "visualizer/gui/video_widget_interface.hpp"
#include "visualizer_impl.hpp"
#include <algorithm>
#include <format>
#include <functional>
#include <future>
#include <type_traits>

namespace lfs::vis::gui {

    using ExportFormat = lfs::core::ExportFormat;

    [[nodiscard]] const char* getDatasetTypeName(const std::filesystem::path& path) {
        switch (lfs::io::Loader::getDatasetType(path)) {
        case lfs::io::DatasetType::COLMAP: return "COLMAP";
        case lfs::io::DatasetType::Transforms: return "NeRF/Blender";
        default: return "Dataset";
        }
    }

    void truncateSHDegree(lfs::core::SplatData& splat, const int target_degree) {
        if (target_degree >= splat.get_max_sh_degree())
            return;

        if (target_degree == 0) {
            splat.shN() = lfs::core::Tensor{};
        } else {
            const size_t keep_coeffs = static_cast<size_t>((target_degree + 1) * (target_degree + 1) - 1);
            auto& shN = splat.shN();
            if (shN.is_valid() && shN.ndim() >= 2 && shN.shape()[1] > keep_coeffs) {
                if (shN.ndim() == 3) {
                    shN = shN.slice(1, 0, static_cast<int64_t>(keep_coeffs)).contiguous();
                } else {
                    constexpr size_t CHANNELS = 3;
                    shN = shN.slice(1, 0, static_cast<int64_t>(keep_coeffs * CHANNELS)).contiguous();
                }
            }
        }
        splat.set_max_sh_degree(target_degree);
        splat.set_active_sh_degree(target_degree);
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

    void applyVideoExportCropState(rendering::RenderRequest& request,
                                   const VideoExportSceneSnapshot& snapshot,
                                   const RenderSettings& render_settings) {
        if ((render_settings.use_crop_box || render_settings.show_crop_box) && !snapshot.cropboxes.empty()) {
            const size_t idx = (snapshot.selected_cropbox_index >= 0)
                                   ? static_cast<size_t>(snapshot.selected_cropbox_index)
                                   : 0;
            if (idx < snapshot.cropboxes.size() && snapshot.cropboxes[idx].has_data) {
                const auto& cb = snapshot.cropboxes[idx];
                request.crop_box = rendering::BoundingBox{
                    .min = cb.data.min,
                    .max = cb.data.max,
                    .transform = glm::inverse(cb.world_transform)};
                request.crop_inverse = cb.data.inverse;
                request.crop_desaturate = render_settings.show_crop_box &&
                                          !render_settings.use_crop_box &&
                                          render_settings.desaturate_cropping;
                request.crop_parent_node_index = cb.parent_node_index;
            }
        }

        if ((render_settings.use_ellipsoid || render_settings.show_ellipsoid) &&
            snapshot.active_ellipsoid.has_value()) {
            const auto& ellipsoid = *snapshot.active_ellipsoid;
            request.ellipsoid = rendering::Ellipsoid{
                .radii = ellipsoid.data.radii,
                .transform = glm::inverse(ellipsoid.world_transform)};
            request.ellipsoid_inverse = ellipsoid.data.inverse;
            request.ellipsoid_desaturate = render_settings.show_ellipsoid &&
                                           !render_settings.use_ellipsoid &&
                                           render_settings.desaturate_cropping;
            request.ellipsoid_parent_node_index = ellipsoid.parent_node_index;
        }

        if (render_settings.depth_filter_enabled) {
            request.depth_filter = rendering::BoundingBox{
                .min = render_settings.depth_filter_min,
                .max = render_settings.depth_filter_max,
                .transform = render_settings.depth_filter_transform.inv().toMat4()};
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
            .is_selected = is_selected,
            .desaturate_unselected = render_settings.desaturate_unselected && any_selected,
            .selection_flash_intensity = 0.0f,
            .background_color = render_settings.background_color};
    }

    std::expected<lfs::core::Tensor, std::string> renderVideoExportFrame(
        rendering::RenderingEngine& engine,
        const VideoExportSceneSnapshot& snapshot,
        const RenderSettings& render_settings,
        const lfs::sequencer::CameraState& cam_state,
        const int width,
        const int height) {
        const auto viewport = makeVideoExportViewport(cam_state, render_settings, width, height);

        std::optional<rendering::RenderResult> primary_result;

        if (snapshot.combined_model && snapshot.combined_model->size() > 0) {
            rendering::RenderRequest request;
            request.viewport = viewport;
            request.scaling_modifier = render_settings.scaling_modifier;
            request.antialiasing = render_settings.antialiasing;
            request.mip_filter = render_settings.mip_filter;
            request.sh_degree = render_settings.sh_degree;
            request.background_color = render_settings.background_color;
            request.point_cloud_mode = render_settings.point_cloud_mode;
            request.voxel_size = render_settings.voxel_size;
            request.gut = render_settings.gut;
            request.equirectangular = render_settings.equirectangular;
            request.show_rings = render_settings.show_rings;
            request.ring_width = render_settings.ring_width;
            request.show_center_markers = render_settings.show_center_markers;
            request.model_transforms = &snapshot.model_transforms;
            request.transform_indices = snapshot.transform_indices;
            request.selection_mask = snapshot.selection_mask;
            request.selected_node_mask = render_settings.desaturate_unselected
                                             ? snapshot.selected_node_mask
                                             : std::vector<bool>{};
            request.node_visibility_mask = snapshot.node_visibility_mask;
            request.desaturate_unselected = render_settings.desaturate_unselected;
            request.selection_flash_intensity = 0.0f;
            request.far_plane = render_settings.depth_clip_enabled
                                    ? render_settings.depth_clip_far
                                    : rendering::DEFAULT_FAR_PLANE;
            request.orthographic = render_settings.orthographic;
            request.ortho_scale = render_settings.ortho_scale;
            applyVideoExportCropState(request, snapshot, render_settings);

            auto render_result = engine.renderGaussians(*snapshot.combined_model, request);
            if (!render_result || !render_result->valid || !render_result->image) {
                return std::unexpected(render_result ? "Rendered frame is invalid"
                                                     : render_result.error());
            }
            primary_result = std::move(*render_result);
        } else if (snapshot.point_cloud && snapshot.point_cloud->size() > 0) {
            const std::vector<glm::mat4> point_cloud_transforms = {snapshot.point_cloud_transform};
            rendering::RenderRequest request;
            request.viewport = viewport;
            request.scaling_modifier = render_settings.scaling_modifier;
            request.mip_filter = render_settings.mip_filter;
            request.sh_degree = 0;
            request.background_color = render_settings.background_color;
            request.point_cloud_mode = true;
            request.voxel_size = render_settings.voxel_size;
            request.equirectangular = render_settings.equirectangular;
            request.model_transforms = &point_cloud_transforms;
            request.far_plane = render_settings.depth_clip_enabled
                                    ? render_settings.depth_clip_far
                                    : rendering::DEFAULT_FAR_PLANE;
            request.orthographic = render_settings.orthographic;
            request.ortho_scale = render_settings.ortho_scale;
            applyVideoExportCropState(request, snapshot, render_settings);

            auto render_result = engine.renderPointCloud(*snapshot.point_cloud, request);
            if (!render_result || !render_result->valid || !render_result->image) {
                return std::unexpected(render_result ? "Rendered point cloud frame is invalid"
                                                     : render_result.error());
            }
            primary_result = std::move(*render_result);
        }

        if (snapshot.meshes.empty()) {
            if (primary_result && primary_result->image) {
                return *primary_result->image;
            }
            return std::unexpected("No rendered image produced for video export");
        }

        GLint saved_draw_fbo = 0;
        GLint saved_read_fbo = 0;
        GLint saved_viewport[4] = {0, 0, 0, 0};
        const GLboolean scissor_was_enabled = glIsEnabled(GL_SCISSOR_TEST);
        glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &saved_draw_fbo);
        glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &saved_read_fbo);
        glGetIntegerv(GL_VIEWPORT, saved_viewport);

        rendering::FrameBuffer composite_fbo;
        composite_fbo.resize(width, height);
        composite_fbo.bind();
        glDisable(GL_SCISSOR_TEST);
        glViewport(0, 0, width, height);
        glClearColor(render_settings.background_color.r,
                     render_settings.background_color.g,
                     render_settings.background_color.b,
                     1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        auto restore_state = [&]() {
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, static_cast<GLuint>(saved_draw_fbo));
            glBindFramebuffer(GL_READ_FRAMEBUFFER, static_cast<GLuint>(saved_read_fbo));
            glViewport(saved_viewport[0], saved_viewport[1], saved_viewport[2], saved_viewport[3]);
            if (scissor_was_enabled) {
                glEnable(GL_SCISSOR_TEST);
            } else {
                glDisable(GL_SCISSOR_TEST);
            }
        };

        if (primary_result.has_value()) {
            if (auto present_result = engine.presentToScreen(*primary_result, {0, 0}, {width, height});
                !present_result) {
                restore_state();
                return std::unexpected(present_result.error());
            }
        }

        const bool any_selected = std::any_of(snapshot.meshes.begin(), snapshot.meshes.end(),
                                              [](const auto& mesh) { return mesh.is_selected; }) ||
                                  std::any_of(snapshot.selected_node_mask.begin(),
                                              snapshot.selected_node_mask.end(),
                                              [](const bool selected) { return selected; });

        engine.resetMeshFrameState();
        for (const auto& mesh_snapshot : snapshot.meshes) {
            if (!mesh_snapshot.mesh)
                continue;
            const auto mesh_options = makeVideoExportMeshOptions(
                render_settings, any_selected, mesh_snapshot.is_selected);
            auto mesh_result = engine.renderMesh(
                *mesh_snapshot.mesh,
                viewport,
                mesh_snapshot.transform,
                mesh_options,
                primary_result.has_value());
            if (!mesh_result) {
                restore_state();
                return std::unexpected(mesh_result.error());
            }
        }

        if (engine.hasMeshRender()) {
            if (primary_result.has_value()) {
                if (auto composite_result = engine.compositeMeshAndSplat(*primary_result, {width, height});
                    !composite_result) {
                    restore_state();
                    return std::unexpected(composite_result.error());
                }
            } else {
                if (auto present_result = engine.presentMeshOnly(); !present_result) {
                    restore_state();
                    return std::unexpected(present_result.error());
                }
            }
        }

        std::vector<float> pixels(static_cast<size_t>(width) * static_cast<size_t>(height) * 3);
        glReadPixels(0, 0, width, height, GL_RGB, GL_FLOAT, pixels.data());

        restore_state();

        const auto image_cpu = lfs::core::Tensor::from_vector(
            pixels,
            {static_cast<size_t>(height), static_cast<size_t>(width), size_t{3}},
            lfs::core::Device::CPU);
        return image_cpu.permute({2, 0, 1}).cuda();
    }

    AsyncTaskManager::AsyncTaskManager(VisualizerImpl* viewer)
        : viewer_(viewer) {}

    AsyncTaskManager::~AsyncTaskManager() {
        shutdown();
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

        if (import_state_.thread) {
            import_state_.thread->request_stop();
            if (import_state_.thread->joinable())
                import_state_.thread->join();
            import_state_.thread.reset();
        }

        mesh2splat_state_.active.store(false);
        mesh2splat_state_.pending.store(false);
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
            if (!cmd.output_path.empty())
                params.dataset.output_path = cmd.output_path;
            if (!cmd.init_path.empty())
                params.init_path = lfs::core::path_to_utf8(cmd.init_path);
            startAsyncImport(cmd.path, params);
        });

        state::DatasetLoadStarted::when([this](const auto& e) {
            if (import_state_.active.load())
                return;
            const std::lock_guard lock(import_state_.mutex);
            import_state_.active.store(true);
            import_state_.progress.store(0.0f);
            import_state_.path = e.path;
            import_state_.stage = "Initializing...";
            import_state_.error.clear();
            import_state_.num_images = 0;
            import_state_.num_points = 0;
            import_state_.success = false;
            import_state_.dataset_type = getDatasetTypeName(e.path);
        });

        state::DatasetLoadProgress::when([this](const auto& e) {
            import_state_.progress.store(e.progress / 100.0f);
            const std::lock_guard lock(import_state_.mutex);
            import_state_.stage = e.step;
        });

        state::DatasetLoadCompleted::when([this](const auto& e) {
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

    void AsyncTaskManager::performExport(ExportFormat format, const std::filesystem::path& path,
                                         const std::vector<std::string>& node_names, int sh_degree) {
        if (isExporting())
            return;

        auto* const scene_manager = viewer_->getSceneManager();
        if (!scene_manager || node_names.empty())
            return;

        const auto& scene = scene_manager->getScene();
        std::vector<std::pair<const lfs::core::SplatData*, glm::mat4>> splats;
        splats.reserve(node_names.size());
        for (const auto& name : node_names) {
            const auto* node = scene.getNode(name);
            if (node && node->type == core::NodeType::SPLAT && node->model) {
                splats.emplace_back(node->model.get(), scene.getWorldTransform(node->id));
            }
        }
        if (splats.empty())
            return;

        auto merged = core::Scene::mergeSplatsWithTransforms(splats);
        if (!merged)
            return;

        if (sh_degree < merged->get_max_sh_degree()) {
            truncateSHDegree(*merged, sh_degree);
        }

        startAsyncExport(format, path, std::move(merged));
    }

    void AsyncTaskManager::startAsyncExport(ExportFormat format,
                                            const std::filesystem::path& path,
                                            std::unique_ptr<lfs::core::SplatData> data) {
        if (!data) {
            LOG_ERROR("No splat data to export");
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

        auto splat_data = std::shared_ptr<lfs::core::SplatData>(std::move(data));
        LOG_INFO("Export started: {} (format: {})", lfs::core::path_to_utf8(path), static_cast<int>(format));

        export_state_.thread.emplace(
            [this, format, path, splat_data](std::stop_token stop_token) {
                auto update_progress = [this, &stop_token](float progress, const std::string& stage) -> bool {
                    export_state_.progress.store(progress);
                    {
                        const std::lock_guard lock(export_state_.mutex);
                        export_state_.stage = stage;
                    }
                    if (stop_token.stop_requested() || export_state_.cancel_requested.load()) {
                        LOG_INFO("Export cancelled");
                        return false;
                    }
                    return true;
                };

                bool success = false;
                std::string error_msg;

                switch (format) {
                case ExportFormat::PLY: {
                    update_progress(0.1f, "Writing PLY");
                    const lfs::io::PlySaveOptions options{
                        .output_path = path,
                        .binary = true,
                        .async = false};
                    if (auto result = lfs::io::save_ply(*splat_data, options); result) {
                        success = true;
                        update_progress(1.0f, "Complete");
                    } else {
                        error_msg = result.error().message;
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
                    }
                    break;
                }
                case ExportFormat::SPZ: {
                    update_progress(0.1f, "Writing SPZ");
                    const lfs::io::SpzSaveOptions options{.output_path = path};
                    if (auto result = lfs::io::save_spz(*splat_data, options); result) {
                        success = true;
                        update_progress(1.0f, "Complete");
                    } else {
                        error_msg = result.error().message;
                    }
                    break;
                }
                case ExportFormat::HTML_VIEWER: {
                    const HtmlViewerExportOptions options{
                        .output_path = path,
                        .progress_callback = [&update_progress](float p, const std::string& s) {
                            update_progress(p, s);
                        }};
                    if (auto result = export_html_viewer(*splat_data, options); result) {
                        success = true;
                    } else {
                        error_msg = result.error();
                    }
                    break;
                }
                }

                if (success) {
                    LOG_INFO("Export completed: {}", lfs::core::path_to_utf8(path));
                    {
                        const std::lock_guard lock(export_state_.mutex);
                        export_state_.stage = "Complete";
                    }
                    lfs::core::events::state::ExportCompleted{
                        .path = path,
                        .format = format}
                        .emit();
                } else {
                    LOG_ERROR("Export failed: {}", error_msg);
                    {
                        const std::lock_guard lock(export_state_.mutex);
                        export_state_.error = error_msg;
                        export_state_.stage = "Failed";
                    }
                    lfs::core::events::state::ExportFailed{
                        .error = error_msg}
                        .emit();
                }

                export_state_.active.store(false);
            });
    }

    void AsyncTaskManager::cancelExport() {
        if (!export_state_.active.load())
            return;
        LOG_INFO("Cancelling export");
        export_state_.cancel_requested.store(true);
        if (export_state_.thread && export_state_.thread->joinable()) {
            export_state_.thread->request_stop();
        }
    }

    void AsyncTaskManager::startAsyncImport(const std::filesystem::path& path,
                                            const lfs::core::param::TrainingParameters& params) {
        if (import_state_.active.load()) {
            LOG_WARN("Import already in progress");
            return;
        }

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

        LOG_INFO("Async import: {}", lfs::core::path_to_utf8(path));

        import_state_.thread.emplace(
            [this, path](const std::stop_token stop_token) {
                lfs::core::param::TrainingParameters local_params;
                {
                    const std::lock_guard lock(import_state_.mutex);
                    local_params = import_state_.params;
                }

                const lfs::io::LoadOptions load_options{
                    .resize_factor = local_params.dataset.resize_factor,
                    .max_width = local_params.dataset.max_width,
                    .images_folder = local_params.dataset.images,
                    .validate_only = false,
                    .progress = [this, &stop_token](const float pct, const std::string& msg) {
                        if (stop_token.stop_requested())
                            return;
                        import_state_.progress.store(pct / 100.0f);
                        const std::lock_guard lock(import_state_.mutex);
                        import_state_.stage = msg;
                    }};

                auto loader = lfs::io::Loader::create();
                auto result = loader->load(path, load_options);

                if (stop_token.stop_requested()) {
                    import_state_.active.store(false);
                    return;
                }

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
                import_state_.progress.store(1.0f);
                import_state_.load_complete.store(true);
            });
    }

    void AsyncTaskManager::checkAsyncImportCompletion() {
        if (!import_state_.load_complete.load())
            return;
        import_state_.load_complete.store(false);

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
            const std::lock_guard lock(import_state_.mutex);
            import_state_.completion_time = std::chrono::steady_clock::now();
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

        lfs::core::events::state::DatasetLoadCompleted{
            .path = path,
            .success = success_val,
            .error = success_val ? std::nullopt : std::optional<std::string>(error_val),
            .num_images = num_images_val,
            .num_points = num_points_val}
            .emit();
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

        LOG_INFO("Starting video export: {} frames at {}x{}", total_frames, width, height);

        video_export_state_.thread.emplace(
            [this, viewer = viewer_, path, export_options, total_frames, width, height,
             engine, render_settings,
             snapshot = *snapshot_result,
             frame_states = std::move(frame_states)](std::stop_token stop_token) mutable {
                bool cancelled = false;

                auto encoder = lfs::gui::createVideoEncoder();
                if (!encoder) {
                    {
                        std::lock_guard lock(video_export_state_.mutex);
                        video_export_state_.error = "Video encoder not available";
                        video_export_state_.stage = "Failed";
                    }
                    video_export_state_.active.store(false);
                    lfs::core::events::state::VideoExportFailed{
                        .error = "Video encoder not available"}
                        .emit();
                    return;
                }

                {
                    std::lock_guard lock(video_export_state_.mutex);
                    video_export_state_.stage = "Opening encoder";
                }

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
                        [engine, snapshot, render_settings, width, height,
                         cam_state = frame_states[frame]]() -> std::expected<lfs::core::Tensor, std::string> {
                            return renderVideoExportFrame(
                                *engine, snapshot, render_settings, cam_state, width, height);
                        });

                    if (!frame_tensor) {
                        LOG_ERROR("Failed to render frame {}: {}", frame, frame_tensor.error());
                        {
                            std::lock_guard lock(video_export_state_.mutex);
                            video_export_state_.error = std::format(
                                "Failed to render frame {}: {}", frame + 1, frame_tensor.error());
                            video_export_state_.stage = "Render error";
                        }
                        break;
                    }

                    auto image_hwc = frame_tensor->permute({1, 2, 0}).contiguous();

                    if (frame == 0) {
                        LOG_INFO("Video export: CHW shape=[{},{},{}] -> HWC shape=[{},{},{}]",
                                 frame_tensor->shape()[0], frame_tensor->shape()[1], frame_tensor->shape()[2],
                                 image_hwc.shape()[0], image_hwc.shape()[1], image_hwc.shape()[2]);
                    }

                    const auto* const gpu_ptr = image_hwc.data_ptr();
                    auto write_result = encoder->writeFrameGpu(gpu_ptr, width, height, nullptr);
                    if (!write_result) {
                        std::lock_guard lock(video_export_state_.mutex);
                        video_export_state_.error = write_result.error();
                        video_export_state_.stage = "Encode error";
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
                }

                {
                    std::lock_guard lock(video_export_state_.mutex);
                    if (cancelled) {
                        video_export_state_.stage = "Cancelled";
                    } else if (video_export_state_.error.empty()) {
                        video_export_state_.stage = "Finalizing";
                    }
                }

                if (auto close_result = encoder->close(); !close_result) {
                    std::lock_guard lock(video_export_state_.mutex);
                    video_export_state_.error = close_result.error();
                    video_export_state_.stage = "Failed";
                    LOG_ERROR("Failed to close encoder: {}", close_result.error());
                } else {
                    bool emit_completed = false;
                    {
                        std::lock_guard lock(video_export_state_.mutex);
                        if (cancelled) {
                            video_export_state_.stage = "Cancelled";
                        } else if (video_export_state_.error.empty() && !video_export_state_.cancel_requested.load()) {
                            video_export_state_.stage = "Complete";
                            LOG_INFO("Video export completed: {}", path.string());
                            emit_completed = true;
                        }
                    }
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
                video_export_state_.active.store(false);
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

        mesh2splat_state_.pending.store(true);
    }

    void AsyncTaskManager::pollMesh2SplatCompletion() {
        if (!mesh2splat_state_.pending.load())
            return;
        mesh2splat_state_.pending.store(false);

        executeMesh2SplatOnGlThread();

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
    }

    void AsyncTaskManager::executeMesh2SplatOnGlThread() {
        std::shared_ptr<lfs::core::MeshData> mesh;
        lfs::core::Mesh2SplatOptions options;
        {
            const std::lock_guard lock(mesh2splat_state_.mutex);
            mesh = std::move(mesh2splat_state_.pending_mesh);
            options = mesh2splat_state_.pending_options;
        }

        if (!mesh)
            return;

        auto progress_cb = [this](float progress, const std::string& stage) -> bool {
            mesh2splat_state_.progress.store(progress);
            {
                const std::lock_guard lock(mesh2splat_state_.mutex);
                mesh2splat_state_.stage = stage;
            }
            return true;
        };

        auto result = lfs::rendering::mesh_to_splat(*mesh, options, progress_cb);

        const std::lock_guard lock(mesh2splat_state_.mutex);
        if (result) {
            mesh2splat_state_.result = std::move(*result);
            mesh2splat_state_.stage = "Applying...";
            LOG_INFO("Mesh2Splat conversion produced {} gaussians",
                     mesh2splat_state_.result->size());
        } else {
            mesh2splat_state_.error = result.error();
            mesh2splat_state_.stage = "Failed";
            LOG_ERROR("Mesh2Splat conversion failed: {}", result.error());
        }
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

        scene.addSplat(node_name, std::move(splat_data));

        {
            const std::lock_guard lock(mesh2splat_state_.mutex);
            mesh2splat_state_.stage = "Complete";
        }

        const auto* const added_node = scene.getNode(node_name);
        const size_t num_gaussians =
            added_node && added_node->model ? added_node->model->size() : 0;

        lfs::core::events::state::Mesh2SplatCompleted{
            .source_name = source_name,
            .node_name = node_name,
            .num_gaussians = num_gaussians}
            .emit();

        LOG_INFO("Mesh2Splat: added splat node '{}'", node_name);
    }

} // namespace lfs::vis::gui

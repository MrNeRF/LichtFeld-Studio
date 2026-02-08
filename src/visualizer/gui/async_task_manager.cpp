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
#include "gui/utils/windows_utils.hpp"
#include "io/exporter.hpp"
#include "io/video/video_encoder.hpp"
#include "rendering/rendering.hpp"
#include "rendering/rendering_manager.hpp"
#include "scene/scene_manager.hpp"
#include "sequencer/keyframe.hpp"
#include "sequencer/sequencer_controller.hpp"
#include "training/training_manager.hpp"
#include "visualizer_impl.hpp"
#include <algorithm>
#include <format>

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
                    const std::lock_guard lock(export_state_.mutex);
                    export_state_.stage = "Complete";
                } else {
                    LOG_ERROR("Export failed: {}", error_msg);
                    const std::lock_guard lock(export_state_.mutex);
                    export_state_.error = error_msg;
                    export_state_.stage = "Failed";
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
        {
            const std::lock_guard lock(import_state_.mutex);
            import_state_.path = path;
            import_state_.stage = "Initializing...";
            import_state_.error.clear();
            import_state_.num_images = 0;
            import_state_.num_points = 0;
            import_state_.success = false;
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
        import_state_.show_completion.store(true);

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
        if (video_export_state_.thread) {
            video_export_state_.thread->request_stop();
        }
    }

    void AsyncTaskManager::startVideoExport(const std::filesystem::path& path,
                                            const io::video::VideoExportOptions& options) {
        auto* const scene_manager = viewer_->getSceneManager();
        auto* const rendering_manager = viewer_->getRenderingManager();
        if (!scene_manager || !rendering_manager) {
            LOG_ERROR("Cannot export video: missing components");
            return;
        }

        auto* gui_manager = viewer_->getGuiManager();
        if (!gui_manager) {
            LOG_ERROR("Cannot export video: no gui manager");
            return;
        }
        const auto& timeline = gui_manager->sequencer().timeline();
        if (timeline.empty()) {
            LOG_ERROR("Cannot export video: no keyframes");
            return;
        }

        auto* const scene_ptr = &scene_manager->getScene();
        scene_ptr->pinForExport();
        auto export_pin = std::shared_ptr<void>(nullptr, [scene_ptr](void*) { scene_ptr->unpinForExport(); });

        const auto render_state = scene_manager->buildRenderState();
        if (!render_state.combined_model) {
            LOG_ERROR("No splat data to render");
            return;
        }

        auto* const engine = rendering_manager->getRenderingEngine();
        if (!engine) {
            LOG_ERROR("Rendering engine not available");
            return;
        }

        const float duration = timeline.duration();
        const int total_frames = static_cast<int>(std::ceil(duration * options.framerate)) + 1;
        const int width = options.width;
        const int height = options.height;

        std::vector<lfs::sequencer::CameraState> frame_states;
        frame_states.reserve(total_frames);
        const float start_time = timeline.startTime();
        const float time_step = 1.0f / static_cast<float>(options.framerate);
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
        }

        LOG_INFO("Starting video export: {} frames at {}x{}", total_frames, width, height);

        const auto render_settings = rendering_manager->getSettings();
        const lfs::core::SplatData* splat_ptr = render_state.combined_model;

        video_export_state_.thread.emplace(
            [this, path, options, total_frames, width, height,
             splat_ptr, engine, render_settings,
             frame_states = std::move(frame_states),
             export_pin = std::move(export_pin)](std::stop_token stop_token) {
                io::video::VideoEncoder encoder;

                {
                    std::lock_guard lock(video_export_state_.mutex);
                    video_export_state_.stage = "Opening encoder";
                }

                auto result = encoder.open(path, options);
                if (!result) {
                    std::lock_guard lock(video_export_state_.mutex);
                    video_export_state_.error = result.error();
                    video_export_state_.stage = "Failed: " + result.error();
                    video_export_state_.active.store(false);
                    LOG_ERROR("Failed to open encoder: {}", result.error());
                    return;
                }

                for (int frame = 0; frame < total_frames; ++frame) {
                    if (stop_token.stop_requested() || video_export_state_.cancel_requested.load()) {
                        LOG_INFO("Video export cancelled at frame {}", frame);
                        break;
                    }

                    const auto& cam_state = frame_states[frame];

                    rendering::RenderRequest request;
                    request.viewport.rotation = glm::mat3_cast(cam_state.rotation);
                    request.viewport.translation = cam_state.position;
                    request.viewport.size = {width, height};
                    request.viewport.focal_length_mm = lfs::rendering::vFovToFocalLength(cam_state.fov);
                    request.background_color = render_settings.background_color;
                    request.sh_degree = render_settings.sh_degree;
                    request.scaling_modifier = render_settings.scaling_modifier;
                    request.antialiasing = true;

                    auto render_result = engine->renderGaussians(*splat_ptr, request);
                    if (!render_result.has_value() || !render_result->valid || !render_result->image) {
                        LOG_ERROR("Failed to render frame {}", frame);
                        continue;
                    }

                    auto image_hwc = render_result->image->permute({1, 2, 0}).contiguous();

                    if (frame == 0) {
                        LOG_INFO("Video export: CHW shape=[{},{},{}] -> HWC shape=[{},{},{}]",
                                 render_result->image->shape()[0], render_result->image->shape()[1], render_result->image->shape()[2],
                                 image_hwc.shape()[0], image_hwc.shape()[1], image_hwc.shape()[2]);
                    }

                    const auto* const gpu_ptr = image_hwc.data_ptr();
                    auto write_result = encoder.writeFrameGpu(gpu_ptr, width, height, nullptr);
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
                    video_export_state_.stage = "Finalizing";
                }

                if (auto close_result = encoder.close(); !close_result) {
                    std::lock_guard lock(video_export_state_.mutex);
                    video_export_state_.error = close_result.error();
                    video_export_state_.stage = "Failed";
                    LOG_ERROR("Failed to close encoder: {}", close_result.error());
                } else {
                    std::lock_guard lock(video_export_state_.mutex);
                    if (video_export_state_.error.empty() && !video_export_state_.cancel_requested.load()) {
                        video_export_state_.stage = "Complete";
                        LOG_INFO("Video export completed: {}", path.string());
                    }
                }

                video_export_state_.active.store(false);
            });
    }

} // namespace lfs::vis::gui

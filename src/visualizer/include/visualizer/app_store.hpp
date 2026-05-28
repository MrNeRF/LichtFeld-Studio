/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "core/reactive/observable.hpp"
#include "core/reactive/store.hpp"

#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace lfs::diagnostics {
    struct VramProfilerSnapshot;
} // namespace lfs::diagnostics

namespace lfs::vis {

    class LFS_VIS_API AppStore {
    public:
        struct LFS_VIS_API CameraMetrics {
            int camera_id = -1;
            int iteration = -1;
            float psnr = 0.0f;
            std::optional<float> ssim;
            bool used_mask = false;

            bool operator==(const CameraMetrics&) const = default;
        };

        struct LFS_VIS_API GTMetricsOverlayConfig {
            bool visible = false;
            float x = 16.0f;
            float y = 16.0f;
            bool show_ssim = false;
            int current_camera_id = -1;

            [[nodiscard]] bool operator==(const GTMetricsOverlayConfig& other) const noexcept {
                return visible == other.visible &&
                       std::abs(x - other.x) <= 0.5f &&
                       std::abs(y - other.y) <= 0.5f &&
                       show_ssim == other.show_ssim &&
                       current_camera_id == other.current_camera_id;
            }
        };

        struct LFS_VIS_API VramHud {
            bool visible = false;
            std::shared_ptr<const lfs::diagnostics::VramProfilerSnapshot> snapshot;

            [[nodiscard]] bool operator==(const VramHud& other) const noexcept {
                return visible == other.visible && snapshot == other.snapshot;
            }
        };

        struct LFS_VIS_API ImportOverlayState {
            bool active = false;
            bool show_completion = false;
            float progress = 0.0f;
            std::string stage;
            std::string dataset_type;
            std::string path;
            bool success = false;
            std::string error;
            std::uint64_t num_images = 0;
            std::uint64_t num_points = 0;
            float seconds_since_completion = 0.0f;

            bool operator==(const ImportOverlayState&) const = default;
        };

        struct LFS_VIS_API VideoExportOverlayState {
            bool active = false;
            float progress = 0.0f;
            int current_frame = 0;
            int total_frames = 0;
            std::string stage;

            bool operator==(const VideoExportOverlayState&) const = default;
        };

        struct LFS_VIS_API ExportProgressState {
            bool active = false;
            float progress = 0.0f;
            std::string stage;
            std::string format;
            std::string error;
            std::string path;

            [[nodiscard]] bool operator==(const ExportProgressState& other) const noexcept {
                return active == other.active &&
                       std::abs(progress - other.progress) <= 0.0005f &&
                       stage == other.stage &&
                       format == other.format &&
                       error == other.error &&
                       path == other.path;
            }
        };

        struct LFS_VIS_API TaskProgressState {
            bool active = false;
            float progress = 0.0f;
            std::string stage;
            std::string error;
            std::string source_name;
            std::string output_name;

            [[nodiscard]] bool operator==(const TaskProgressState& other) const noexcept {
                return active == other.active &&
                       std::abs(progress - other.progress) <= 0.0005f &&
                       stage == other.stage &&
                       error == other.error &&
                       source_name == other.source_name &&
                       output_name == other.output_name;
            }
        };

        enum Field : std::uint32_t {
            Iteration = 1,
            TotalIterations,
            Loss,
            NumGaussians,
            MaxGaussians,
            TrainingRunning,
            TrainingState,
            TrainerLoaded,
            EvalPsnr,
            EvalSsim,
            SceneGeneration,
            SelectionGeneration,
            Fps,
            ModeText,
            CameraMetricsValue,
            GTMetricsOverlayConfigValue,
            VramHudValue,
            ActiveTool,
            ActiveSubmode,
            TransformSpaceValue,
            PivotModeValue,
            ImportOverlayStateValue,
            VideoExportOverlayStateValue,
            ExportProgressStateValue,
            Mesh2SplatStateValue,
            SplatSimplifyStateValue,
            ScriptsGeneration,
            LanguageGeneration,
        };

        AppStore();

        [[nodiscard]] lfs::core::reactive::Store& store() noexcept { return store_; }
        [[nodiscard]] const lfs::core::reactive::Store& store() const noexcept { return store_; }

        lfs::core::reactive::Observable<int> iteration;
        lfs::core::reactive::Observable<int> total_iterations;
        lfs::core::reactive::Observable<float> loss;
        lfs::core::reactive::Observable<std::int64_t> num_gaussians;
        lfs::core::reactive::Observable<std::int64_t> max_gaussians;
        lfs::core::reactive::Observable<bool> training_running;
        lfs::core::reactive::Observable<std::string> training_state;
        lfs::core::reactive::Observable<bool> trainer_loaded;
        lfs::core::reactive::Observable<std::optional<float>> eval_psnr;
        lfs::core::reactive::Observable<std::optional<float>> eval_ssim;
        lfs::core::reactive::Observable<std::uint64_t> scene_generation;
        lfs::core::reactive::Observable<std::uint64_t> selection_generation;
        lfs::core::reactive::Observable<float> fps;
        lfs::core::reactive::Observable<std::string> mode_text;
        lfs::core::reactive::Observable<std::optional<CameraMetrics>> camera_metrics;
        lfs::core::reactive::Observable<GTMetricsOverlayConfig> gt_metrics_overlay_config;
        lfs::core::reactive::Observable<VramHud> vram_hud;
        lfs::core::reactive::Observable<std::string> active_tool;
        lfs::core::reactive::Observable<std::string> active_submode;
        lfs::core::reactive::Observable<int> transform_space;
        lfs::core::reactive::Observable<int> pivot_mode;
        lfs::core::reactive::Observable<ImportOverlayState> import_overlay_state;
        lfs::core::reactive::Observable<VideoExportOverlayState> video_export_overlay_state;
        lfs::core::reactive::Observable<ExportProgressState> export_progress_state;
        lfs::core::reactive::Observable<TaskProgressState> mesh2splat_state;
        lfs::core::reactive::Observable<TaskProgressState> splat_simplify_state;
        lfs::core::reactive::Observable<std::uint64_t> scripts_generation;
        lfs::core::reactive::Observable<std::uint64_t> language_generation;

    private:
        lfs::core::reactive::Store store_;
    };

    LFS_VIS_API AppStore& app_store();
    LFS_VIS_API void publish_language_generation();

} // namespace lfs::vis

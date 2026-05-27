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
        struct CameraMetrics {
            int camera_id = -1;
            int iteration = -1;
            float psnr = 0.0f;
            std::optional<float> ssim;
            bool used_mask = false;

            bool operator==(const CameraMetrics&) const = default;
        };

        struct GTMetricsOverlayConfig {
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

        struct VramHud {
            bool visible = false;
            std::shared_ptr<const lfs::diagnostics::VramProfilerSnapshot> snapshot;

            [[nodiscard]] bool operator==(const VramHud& other) const noexcept {
                return visible == other.visible && snapshot == other.snapshot;
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

    private:
        lfs::core::reactive::Store store_;
    };

    LFS_VIS_API AppStore& app_store();

} // namespace lfs::vis

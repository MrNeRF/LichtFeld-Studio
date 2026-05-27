/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "visualizer/app_store.hpp"

namespace lfs::vis {

    AppStore::AppStore()
        : iteration(store_, Field::Iteration, "iteration", 0),
          total_iterations(store_, Field::TotalIterations, "total_iterations", 0),
          loss(store_, Field::Loss, "loss", 0.0f),
          num_gaussians(store_, Field::NumGaussians, "num_gaussians", 0),
          max_gaussians(store_, Field::MaxGaussians, "max_gaussians", 0),
          training_running(store_, Field::TrainingRunning, "training_running", false),
          training_state(store_, Field::TrainingState, "training_state", std::string("idle")),
          trainer_loaded(store_, Field::TrainerLoaded, "trainer_loaded", false),
          eval_psnr(store_, Field::EvalPsnr, "eval_psnr", std::optional<float>{}),
          eval_ssim(store_, Field::EvalSsim, "eval_ssim", std::optional<float>{}),
          scene_generation(store_, Field::SceneGeneration, "scene_generation", 0),
          selection_generation(store_, Field::SelectionGeneration, "selection_generation", 0),
          fps(store_, Field::Fps, "fps", 0.0f),
          mode_text(store_, Field::ModeText, "mode_text", std::string{}),
          camera_metrics(store_, Field::CameraMetricsValue, "camera_metrics", std::optional<CameraMetrics>{}),
          gt_metrics_overlay_config(store_,
                                    Field::GTMetricsOverlayConfigValue,
                                    "gt_metrics_overlay_config",
                                    GTMetricsOverlayConfig{}),
          vram_hud(store_, Field::VramHudValue, "vram_hud", VramHud{}),
          active_tool(store_, Field::ActiveTool, "active_tool", std::string{}),
          active_submode(store_, Field::ActiveSubmode, "active_submode", std::string{}),
          transform_space(store_, Field::TransformSpaceValue, "transform_space", 0),
          pivot_mode(store_, Field::PivotModeValue, "pivot_mode", 0),
          import_overlay_state(store_, Field::ImportOverlayStateValue, "import_overlay_state", ImportOverlayState{}),
          video_export_overlay_state(store_,
                                     Field::VideoExportOverlayStateValue,
                                     "video_export_overlay_state",
                                     VideoExportOverlayState{}),
          export_progress_state(store_,
                                Field::ExportProgressStateValue,
                                "export_progress_state",
                                ExportProgressState{}),
          mesh2splat_state(store_, Field::Mesh2SplatStateValue, "mesh2splat_state", TaskProgressState{}),
          splat_simplify_state(store_,
                               Field::SplatSimplifyStateValue,
                               "splat_simplify_state",
                               TaskProgressState{}),
          scripts_generation(store_, Field::ScriptsGeneration, "scripts_generation", 0),
          language_generation(store_, Field::LanguageGeneration, "language_generation", 0) {}

    AppStore& app_store() {
        static AppStore instance;
        return instance;
    }

    void publish_language_generation() {
        auto& signal = app_store().language_generation;
        signal.set(signal.get() + 1);
    }

} // namespace lfs::vis

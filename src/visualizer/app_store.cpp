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
          mode_text(store_, Field::ModeText, "mode_text", std::string{}) {}

    AppStore& app_store() {
        static AppStore instance;
        return instance;
    }

} // namespace lfs::vis

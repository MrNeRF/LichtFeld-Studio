/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "core/reactive/observable.hpp"
#include "core/reactive/store.hpp"

#include <cstdint>
#include <optional>
#include <string>

namespace lfs::vis {

    class LFS_VIS_API AppStore {
    public:
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

    private:
        lfs::core::reactive::Store store_;
    };

    LFS_VIS_API AppStore& app_store();

} // namespace lfs::vis

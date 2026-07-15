/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <memory>

namespace lfs::training {
    class TrainingProgress {
    public:
        enum class Phase {
            Train,
            Refine,
            Controller,
            Sparse
        };

        TrainingProgress(int total_iterations, int update_frequency = 100);
        ~TrainingProgress();

        TrainingProgress(const TrainingProgress&) = delete;
        TrainingProgress& operator=(const TrainingProgress&) = delete;
        TrainingProgress(TrainingProgress&&) = delete;
        TrainingProgress& operator=(TrainingProgress&&) = delete;

        void update(int current_iteration, float loss, int splat_count, Phase phase = Phase::Train);
        void pause();
        void resume(int current_iteration, float loss, int splat_count, Phase phase = Phase::Train);
        void complete();
        void print_final_summary(int final_splats, int actual_iterations = -1);

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace lfs::training

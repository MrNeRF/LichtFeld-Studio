/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/events.hpp"
#include "core/training_manager.hpp"

#include <gtest/gtest.h>

namespace {
    using namespace lfs::core::events;

    // Fails if a destroyed manager leaves its handlers on the process-wide bridge: the next training run would
    // deliver its progress to the freed manager.
    TEST(TrainingManagerEvents, DestroyedManagerLeavesNoHandlers) {
        const auto progress = lfs::event::subscriber_count<state::TrainingProgress>();
        const auto evaluation = lfs::event::subscriber_count<state::EvaluationCompleted>();
        const auto start = lfs::event::subscriber_count<cmd::StartTraining>();
        const auto stop = lfs::event::subscriber_count<cmd::StopTraining>();
        {
            lfs::vis::TrainerManager manager;
            EXPECT_EQ(lfs::event::subscriber_count<state::TrainingProgress>(), progress + 1);
            state::TrainingProgress{.iteration = 1, .loss = 0.5f, .num_gaussians = 16}.emit();
        }
        EXPECT_EQ(lfs::event::subscriber_count<state::TrainingProgress>(), progress);
        EXPECT_EQ(lfs::event::subscriber_count<state::EvaluationCompleted>(), evaluation);
        EXPECT_EQ(lfs::event::subscriber_count<cmd::StartTraining>(), start);
        EXPECT_EQ(lfs::event::subscriber_count<cmd::StopTraining>(), stop);
        state::TrainingProgress{.iteration = 2, .loss = 0.25f, .num_gaussians = 16}.emit();
    }
} // namespace

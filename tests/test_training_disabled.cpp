/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/event_bridge/command_api.hpp"
#include "core/training_manager.hpp"

#include <gtest/gtest.h>

namespace {
    using namespace lfs::vis;

    TEST(TrainingDisabled, RejectsTrainingAndKeepsEditingAvailable) {
        TrainerManager manager;
        EXPECT_FALSE(manager.hasTrainer());
        EXPECT_FALSE(manager.hasLiveTrainingThread());
        EXPECT_FALSE(manager.startTraining());
        EXPECT_FALSE(manager.resumeTraining());
        EXPECT_FALSE(manager.waitForInitialization());
        EXPECT_EQ(manager.getState(), TrainingState::Idle);
        EXPECT_EQ(manager.getLastError(), "Training is not included in this build");
        for (const auto action : {TrainingAction::Start, TrainingAction::Pause,
                                  TrainingAction::Resume, TrainingAction::Stop})
            EXPECT_FALSE(manager.canPerform(action));
        for (const auto action : {TrainingAction::LoadDataset, TrainingAction::LoadCheckpoint,
                                  TrainingAction::ClearScene, TrainingAction::DeleteTrainingNode})
            EXPECT_TRUE(manager.canPerform(action));
    }

    TEST(TrainingDisabled, PreservesStoredProjectMetricsWithoutStartingAWorker) {
        TrainerManager manager;
        lfs::io::project::MetricsChapter stored;
        stored.loss_history = {{1, 0.25f}, {12, 0.125f}};
        stored.psnr_history = {{12, 27.5f}};
        stored.last_evaluation = lfs::io::project::LastEvaluationMetrics{12, 27.5f, 0.75f};
        stored.accumulated_training_seconds = 123.5;
        stored.finish_reason = lfs::io::project::TrainingFinishReason::Completed;
        manager.restoreProjectMetrics(stored);
        const auto restored = manager.captureProjectMetrics();
        ASSERT_EQ(restored.loss_history.size(), 2);
        EXPECT_EQ(restored.loss_history.back().iteration, 12);
        EXPECT_FLOAT_EQ(restored.loss_history.back().value, 0.125f);
        ASSERT_EQ(restored.psnr_history.size(), 1);
        EXPECT_FLOAT_EQ(restored.psnr_history.front().value, 27.5f);
        ASSERT_TRUE(restored.last_evaluation);
        EXPECT_FLOAT_EQ(restored.last_evaluation->ssim, 0.75f);
        EXPECT_DOUBLE_EQ(restored.accumulated_training_seconds, 123.5);
        EXPECT_EQ(restored.finish_reason, stored.finish_reason);
        EXPECT_FALSE(manager.hasLiveTrainingThread());
    }
} // namespace

/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <filesystem>
#include <gtest/gtest.h>

#include "core/logger.hpp"
#include "core/parameters.hpp"
#include "training/checkpoint.hpp"
#include "training/trainer.hpp"
#include "training/training_setup.hpp"
#include "visualizer/scene/scene.hpp"

namespace {

    constexpr const char* TEST_DATASET = "data/bicycle";
    constexpr const char* TEST_IMAGES = "images_4";
    constexpr int CHECKPOINT_ITER = 1200;
    constexpr int RESUME_ITER = 2100;

    class CheckpointResumeTest : public ::testing::TestWithParam<std::tuple<std::string, int>> {
    protected:
        void SetUp() override {
            auto [strategy, sh_degree] = GetParam();
            strategy_ = strategy;
            sh_degree_ = sh_degree;

            // Create unique output directory for this test
            output_path_ = std::filesystem::temp_directory_path() /
                           std::format("lfs_test_checkpoint_{}_{}", strategy_, sh_degree_);
            std::filesystem::create_directories(output_path_);
            std::filesystem::create_directories(output_path_ / "checkpoints");
        }

        void TearDown() override {
            std::error_code ec;
            std::filesystem::remove_all(output_path_, ec);
        }

        lfs::core::param::TrainingParameters createParams(int iterations) {
            lfs::core::param::TrainingParameters params;
            params.dataset.data_path = TEST_DATASET;
            params.dataset.images = TEST_IMAGES;
            params.dataset.output_path = output_path_;
            params.optimization.iterations = iterations;
            params.optimization.strategy = strategy_;
            params.optimization.sh_degree = sh_degree_;
            params.optimization.headless = true;
            params.optimization.max_cap = 100000;
            params.optimization.refine_every = 100;
            params.optimization.start_refine = 500;
            params.optimization.stop_refine = iterations;
            return params;
        }

        std::string strategy_;
        int sh_degree_;
        std::filesystem::path output_path_;
    };

    TEST_P(CheckpointResumeTest, TrainSaveLoadResume) {
        auto [strategy, sh_degree] = GetParam();
        LOG_INFO("Testing checkpoint resume: strategy={}, sh_degree={}", strategy, sh_degree);

        // Phase 1: Train to checkpoint iteration
        {
            auto params = createParams(CHECKPOINT_ITER);
            lfs::vis::Scene scene;

            auto load_result = lfs::training::loadTrainingDataIntoScene(params, scene);
            ASSERT_TRUE(load_result.has_value()) << "Failed to load training data: " << load_result.error();

            auto model_result = lfs::training::initializeTrainingModel(params, scene);
            ASSERT_TRUE(model_result.has_value()) << "Failed to init model: " << model_result.error();

            auto trainer = std::make_unique<lfs::training::Trainer>(scene);
            auto init_result = trainer->initialize(params);
            ASSERT_TRUE(init_result.has_value()) << "Failed to init trainer: " << init_result.error();

            auto train_result = trainer->train();
            ASSERT_TRUE(train_result.has_value()) << "Training failed: " << train_result.error();

            EXPECT_EQ(trainer->get_current_iteration(), CHECKPOINT_ITER);

            auto save_result = trainer->save_checkpoint(CHECKPOINT_ITER);
            ASSERT_TRUE(save_result.has_value()) << "Failed to save checkpoint: " << save_result.error();

            trainer->shutdown();
        }

        // Verify checkpoint file exists
        auto checkpoint_path = output_path_ / "checkpoints" /
                               std::format("checkpoint_{}.resume", CHECKPOINT_ITER);
        ASSERT_TRUE(std::filesystem::exists(checkpoint_path))
            << "Checkpoint file not found: " << checkpoint_path;

        // Phase 2: Load checkpoint and resume to final iteration
        {
            auto params = createParams(RESUME_ITER);
            params.resume_checkpoint = checkpoint_path;

            lfs::vis::Scene scene;

            auto load_result = lfs::training::loadTrainingDataIntoScene(params, scene);
            ASSERT_TRUE(load_result.has_value()) << "Failed to load training data: " << load_result.error();

            auto model_result = lfs::training::initializeTrainingModel(params, scene);
            ASSERT_TRUE(model_result.has_value()) << "Failed to init model: " << model_result.error();

            auto trainer = std::make_unique<lfs::training::Trainer>(scene);
            auto init_result = trainer->initialize(params);
            ASSERT_TRUE(init_result.has_value()) << "Failed to init trainer: " << init_result.error();

            // After loading checkpoint, iteration should be at checkpoint point
            EXPECT_EQ(trainer->get_current_iteration(), CHECKPOINT_ITER);

            auto train_result = trainer->train();
            ASSERT_TRUE(train_result.has_value()) << "Resume training failed: " << train_result.error();

            EXPECT_EQ(trainer->get_current_iteration(), RESUME_ITER);

            trainer->shutdown();
        }

        LOG_INFO("Checkpoint resume test passed: strategy={}, sh_degree={}", strategy, sh_degree);
    }

    std::string TestName(const ::testing::TestParamInfo<CheckpointResumeTest::ParamType>& info) {
        return std::format("{}_{}", std::get<0>(info.param), std::get<1>(info.param));
    }

    INSTANTIATE_TEST_SUITE_P(
        CheckpointStrategies,
        CheckpointResumeTest,
        ::testing::Combine(
            ::testing::Values("mcmc", "default"),
            ::testing::Values(0, 1, 2, 3)),
        TestName);

} // namespace

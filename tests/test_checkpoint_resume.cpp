/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <gtest/gtest.h>

#include "core/logger.hpp"
#include "core/parameters.hpp"
#include "core/scene.hpp"
#include "training/checkpoint.hpp"
#include "training/trainer.hpp"
#include "training/training_setup.hpp"

namespace {

    constexpr const char* TEST_IMAGES = "images_4";
    constexpr int CHECKPOINT_ITER = 1200;
    constexpr int TOTAL_ITER = 2100;

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
            params.dataset.data_path = std::filesystem::path(TEST_DATA_DIR) / "bicycle";
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
        const bool fixed_horizon_resume = strategy == "igs+";
        const int phase_one_iterations = fixed_horizon_resume ? TOTAL_ITER : CHECKPOINT_ITER + 1;
        // Phase 1 always leaves the rotating checkpoint at the completed iteration because the
        // final save path writes a .resume alongside the final PLY.
        const int checkpoint_iteration = phase_one_iterations;

        // Phase 1: Write multiple checkpoints and verify the latest save is the only one retained.
        {
            auto params = createParams(phase_one_iterations);
            params.optimization.save_steps = fixed_horizon_resume
                                                 ? std::vector<size_t>{static_cast<size_t>(CHECKPOINT_ITER)}
                                                 : std::vector<size_t>{static_cast<size_t>(CHECKPOINT_ITER / 2),
                                                                       static_cast<size_t>(CHECKPOINT_ITER)};
            lfs::core::Scene scene;

            auto load_result = lfs::training::loadTrainingDataIntoScene(params, scene);
            ASSERT_TRUE(load_result.has_value()) << "Failed to load training data: " << load_result.error();

            auto model_result = lfs::training::initializeTrainingModel(params, scene);
            ASSERT_TRUE(model_result.has_value()) << "Failed to init model: " << model_result.error();

            auto trainer = std::make_unique<lfs::training::Trainer>(scene);
            auto init_result = trainer->initialize(params);
            ASSERT_TRUE(init_result.has_value()) << "Failed to init trainer: " << init_result.error();

            auto train_result = trainer->train();
            ASSERT_TRUE(train_result.has_value()) << "Training failed: " << train_result.error();

            EXPECT_EQ(trainer->get_current_iteration(), phase_one_iterations);

            trainer->shutdown();
        }

        // Verify the rotating checkpoint exists and is the only checkpoint file.
        auto checkpoint_path = lfs::training::checkpoint_output_path(output_path_);
        ASSERT_TRUE(std::filesystem::exists(checkpoint_path))
            << "Checkpoint file not found: " << checkpoint_path;
        EXPECT_EQ(checkpoint_path.filename(), "checkpoint.resume");

        size_t resume_file_count = 0;
        for (const auto& entry : std::filesystem::directory_iterator(output_path_ / "checkpoints")) {
            if (entry.path().extension() == ".resume") {
                ++resume_file_count;
            }
            EXPECT_EQ(entry.path().filename(), checkpoint_path.filename())
                << "Unexpected stale checkpoint file left behind: " << entry.path();
        }
        EXPECT_EQ(resume_file_count, 1u);

        // Phase 2: Load checkpoint and resume to final iteration
        {
            auto checkpoint_params_result = lfs::core::load_checkpoint_params(checkpoint_path);
            ASSERT_TRUE(checkpoint_params_result.has_value())
                << "Failed to load checkpoint params: " << checkpoint_params_result.error();

            auto params = std::move(*checkpoint_params_result);
            params.resume_checkpoint = checkpoint_path;
            params.dataset.data_path = std::filesystem::path(TEST_DATA_DIR) / "bicycle";
            params.dataset.output_path = output_path_;
            auto resumed_params = params;
            if (!fixed_horizon_resume) {
                resumed_params.optimization.iterations = TOTAL_ITER;
                resumed_params.optimization.stop_refine = TOTAL_ITER;
            }

            lfs::core::Scene scene;

            auto load_result = lfs::training::loadTrainingDataIntoScene(params, scene);
            ASSERT_TRUE(load_result.has_value()) << "Failed to load training data: " << load_result.error();

            auto model_result = lfs::training::initializeTrainingModel(params, scene);
            ASSERT_TRUE(model_result.has_value()) << "Failed to init model: " << model_result.error();

            auto trainer = std::make_unique<lfs::training::Trainer>(scene);
            auto init_result = trainer->initialize(params);
            ASSERT_TRUE(init_result.has_value()) << "Failed to init trainer: " << init_result.error();
            if (!fixed_horizon_resume) {
                trainer->get_strategy_mutable().set_optimization_params(resumed_params.optimization);
                trainer->setParams(resumed_params);
            }

            // After loading checkpoint, iteration should be at checkpoint point
            EXPECT_EQ(trainer->get_current_iteration(), checkpoint_iteration);
            EXPECT_EQ(trainer->getParams().optimization.iterations, static_cast<size_t>(TOTAL_ITER));
            EXPECT_EQ(trainer->getParams().optimization.refine_every, static_cast<size_t>(100));
            EXPECT_EQ(trainer->getParams().optimization.stop_refine, static_cast<size_t>(TOTAL_ITER));
            EXPECT_TRUE(trainer->getParams().optimization.headless);

            auto train_result = trainer->train();
            ASSERT_TRUE(train_result.has_value()) << "Resume training failed: " << train_result.error();

            EXPECT_EQ(trainer->get_current_iteration(), TOTAL_ITER);

            trainer->shutdown();
        }

        LOG_INFO("Checkpoint resume test passed: strategy={}, sh_degree={}", strategy, sh_degree);
    }

    std::string TestName(const ::testing::TestParamInfo<CheckpointResumeTest::ParamType>& info) {
        auto name = std::format("{}_{}", std::get<0>(info.param), std::get<1>(info.param));
        std::replace_if(name.begin(), name.end(), [](const unsigned char c) { return !std::isalnum(c); }, '_');
        return name;
    }

    INSTANTIATE_TEST_SUITE_P(
        CheckpointStrategies,
        CheckpointResumeTest,
        ::testing::Values(
            std::make_tuple("mcmc", 0),
            std::make_tuple("mcmc", 1),
            std::make_tuple("mcmc", 2),
            std::make_tuple("mcmc", 3),
            std::make_tuple("mrnf", 3),
            std::make_tuple("igs+", 3)),
        TestName);

} // namespace

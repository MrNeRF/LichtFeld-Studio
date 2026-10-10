/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/events.hpp"
#include "core/parameter_manager.hpp"
#include "core/scene.hpp"
#include "core/services.hpp"
#include "core/splat_data.hpp"
#include "core/uuid.hpp"
#include "training/training_manager.hpp"

#include <cuda_runtime.h>
#include <filesystem>
#include <gtest/gtest.h>
#include <vector>

namespace {
    using namespace lfs::core::events;

    class TrainingManagerPresentationTest : public testing::Test {
    protected:
        void SetUp() override {
            previous_params_ = lfs::vis::services().paramsOrNull();
            int device_count = 0;
            if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0)
                GTEST_SKIP() << "CUDA device unavailable";
            lfs::vis::services().set(&params_);
            ASSERT_TRUE(params_.ensureLoaded());
            const auto cameras = scene_.addGroup("Cameras");
            scene_.addCamera("camera.png", cameras, std::make_shared<lfs::core::Camera>(lfs::core::Tensor::eye(3, lfs::core::Device::CPU), lfs::core::Tensor::zeros({3}, lfs::core::Device::CPU), 100.0f, 100.0f, 32.0f, 32.0f, lfs::core::Tensor(), lfs::core::Tensor(), lfs::core::CameraModelType::PINHOLE, "camera.png", std::filesystem::path{}, std::filesystem::path{}, 64, 64, 0));
        }

        void TearDown() override {
            EXPECT_TRUE(manager_.clearTrainer());
            lfs::vis::services().set(previous_params_);
            if (!output_.empty()) {
                std::error_code error;
                std::filesystem::remove_all(output_, error);
            }
        }

        void installTrainer(bool checkpoint = false) {
            auto trainer = std::make_unique<lfs::training::Trainer>(scene_);
            auto params = trainer->getParams();
            params.optimization.strategy = "mrnf";
            params.optimization.gut = false;
            params.optimization.max_cap = 5000000;
            trainer->setParams(params);
            if (checkpoint)
                manager_.setTrainerFromCheckpoint(std::move(trainer), 12);
            else
                manager_.setTrainer(std::move(trainer));
        }

        void checkInitializedUpdate(const bool shared_params) {
            using namespace lfs::core;
            output_ = std::filesystem::temp_directory_path() /
                      ("lfs_live_params_" + generate_uuid_v4().to_string());
            std::filesystem::create_directories(output_);
            scene_.addSplat("Model", std::make_unique<SplatData>(
                                         0, Tensor::zeros({1, 3}, Device::CUDA),
                                         Tensor::zeros({1, 1, 3}, Device::CUDA),
                                         Tensor::zeros({1, 0, 3}, Device::CUDA),
                                         Tensor::zeros({1, 3}, Device::CUDA),
                                         Tensor::from_vector(std::vector<float>{1, 0, 0, 0}, {1, 4}, Device::CUDA),
                                         Tensor::zeros({1, 1}, Device::CUDA), 1.0f));
            scene_.setTrainingModelNode("Model");
            installTrainer();
            auto initial = manager_.getTrainer()->getParams();
            initial.dataset.output_path = output_;
            initial.optimization.iterations = 3000;
            initial.optimization.max_cap = 1000;
            initial.optimization.sh_degree = 0;
            initial.optimization.enable_eval = false;
            initial.optimization.enable_sparsity = false;
            initial.optimization.use_ppisp = false;
            initial.optimization.use_exposure_correction = false;
            initial.optimization.headless = true;
            const auto initialized = manager_.getTrainer()->initialize(initial);
            ASSERT_TRUE(initialized) << initialized.error();
            ASSERT_TRUE(manager_.getTrainer()->isInitialized());
            const auto original = manager_.getTrainer()->getParams().optimization;
            params_.importTrainingParams(manager_.getTrainer()->getParams());
            auto edited = original;
            edited.iterations = 1500;
            edited.max_cap = 2000;
            edited.gut = true;
            edited.enable_sparsity = true;
            edited.means_lr = 0.0001f;
            edited.save_steps = {100, 200};
            edited.eval_steps = {150};
            if (shared_params) {
                params_.modifyActiveParams([&](auto& params) { params = edited; });
            } else {
                lfs::vis::services().set(static_cast<lfs::vis::ParameterManager*>(nullptr));
                manager_.getEditableOptParams() = edited;
            }

            // Exercise the real initialized-run boundary, not just its property helper.
            manager_.applyPendingParams();
            const auto actual = manager_.getTrainer()->getParams().optimization;
            EXPECT_EQ(actual.iterations, original.iterations);
            EXPECT_EQ(actual.max_cap, original.max_cap);
            EXPECT_EQ(actual.gut, original.gut);
            EXPECT_EQ(actual.enable_sparsity, original.enable_sparsity);
            EXPECT_FLOAT_EQ(actual.means_lr, edited.means_lr);
            EXPECT_EQ(actual.save_steps, edited.save_steps);
            EXPECT_EQ(actual.eval_steps, edited.eval_steps);
            auto expected = original;
            expected.means_lr = edited.means_lr;
            expected.save_steps = edited.save_steps;
            expected.eval_steps = edited.eval_steps;
            EXPECT_EQ(actual.to_json(), expected.to_json());

            // Editing another strategy must not replace the current run or its live defaults.
            edited.strategy = "mcmc";
            edited.means_lr = 0.0002f;
            if (shared_params) {
                params_.setActiveStrategy("mcmc");
                params_.modifyActiveParams([&](auto& params) { params = edited; });
            } else {
                manager_.getEditableOptParams() = edited;
            }
            manager_.applyPendingParams();
            EXPECT_EQ(manager_.getTrainer()->getParams().optimization.to_json(), expected.to_json());
        }

        std::filesystem::path output_;
        lfs::vis::ParameterManager* previous_params_ = nullptr;
        lfs::vis::ParameterManager params_;
        lfs::core::Scene scene_;
        lfs::vis::TrainerManager manager_;
    };

    TEST_F(TrainingManagerPresentationTest, ReadyReflectsEachEditableStrategyAndMode) {
        installTrainer();
        ASSERT_EQ(manager_.getState(), lfs::vis::TrainingState::Ready);
        for (const auto* strategy : {"mcmc", "igs+", "mrnf"}) {
            params_.setActiveStrategy(strategy);
            for (const bool gut : {false, true}) {
                params_.modifyActiveParams([gut](auto& params) {
                    params.gut = gut;
                    params.max_cap = gut ? 123456 : 654321;
                });
                EXPECT_STREQ(manager_.getStrategyType(), strategy);
                EXPECT_EQ(manager_.isGutEnabled(), gut);
                EXPECT_EQ(manager_.getMaxGaussians(), gut ? 123456 : 654321);
            }
        }
        // Presentation must not apply the edits to the placeholder trainer.
        EXPECT_FALSE(manager_.getTrainer()->isInitialized());
        EXPECT_EQ(manager_.getTrainer()->getParams().optimization.strategy, "mrnf");
        EXPECT_FALSE(manager_.getTrainer()->getParams().optimization.gut);
        EXPECT_EQ(manager_.getTrainer()->getParams().optimization.max_cap, 5000000);
    }

    TEST_F(TrainingManagerPresentationTest, ReadyStrategyNameOutlivesLaterEdits) {
        installTrainer();
        params_.setActiveStrategy("mcmc");
        const char* const reported = manager_.getStrategyType();
        params_.setActiveStrategy("igs+");
        EXPECT_STREQ(reported, "mcmc");
        EXPECT_STREQ(manager_.getStrategyType(), "igs+");
    }

    TEST_F(TrainingManagerPresentationTest, ReadyWithoutParameterServiceUsesPendingEdits) {
        installTrainer();
        lfs::vis::services().set(static_cast<lfs::vis::ParameterManager*>(nullptr));
        auto& pending = manager_.getEditableOptParams();
        pending.strategy = "lfs";
        pending.gut = true;
        pending.max_cap = 234567;
        EXPECT_STREQ(manager_.getStrategyType(), "mrnf");
        EXPECT_TRUE(manager_.isGutEnabled());
        EXPECT_EQ(manager_.getMaxGaussians(), 234567);
    }

    TEST_F(TrainingManagerPresentationTest, EmptyManagerKeepsEmptyPresentation) {
        params_.setActiveStrategy("mcmc");
        params_.modifyActiveParams([](auto& params) { params.gut = true; });
        EXPECT_STREQ(manager_.getStrategyType(), "unknown");
        EXPECT_FALSE(manager_.isGutEnabled());
        EXPECT_EQ(manager_.getMaxGaussians(), 0);
    }

    TEST_F(TrainingManagerPresentationTest, BeforeInitializationAppliesAllEditableSettings) {
        installTrainer();
        params_.modifyActiveParams([](auto& params) {
            params.iterations = 1500;
            params.max_cap = 200000;
            params.enable_sparsity = true;
            params.sparsify_steps = 2000;
        });
        manager_.applyPendingParams();
        EXPECT_EQ(manager_.getTrainer()->getParams().optimization.to_json(),
                  params_.copyActiveParams().to_json());
    }

    TEST_F(TrainingManagerPresentationTest, InitializedRunFiltersSharedParameterUpdates) {
        checkInitializedUpdate(true);
    }

    TEST_F(TrainingManagerPresentationTest, InitializedRunFiltersPendingParameterUpdates) {
        checkInitializedUpdate(false);
    }

    TEST_F(TrainingManagerPresentationTest, CheckpointStateDoesNotUseNextRunEdits) {
        installTrainer(true);
        ASSERT_EQ(manager_.getState(), lfs::vis::TrainingState::Paused);
        params_.setActiveStrategy("mcmc");
        params_.modifyActiveParams([](auto& params) {
            params.gut = true;
            params.max_cap = 123456;
        });
        EXPECT_STREQ(manager_.getStrategyType(), "unknown");
        EXPECT_FALSE(manager_.isGutEnabled());
        EXPECT_EQ(manager_.getMaxGaussians(), 5000000);
    }

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

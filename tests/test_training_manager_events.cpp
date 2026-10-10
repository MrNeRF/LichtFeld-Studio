/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/events.hpp"
#include "core/parameter_manager.hpp"
#include "core/scene.hpp"
#include "core/services.hpp"
#include "training/training_manager.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

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

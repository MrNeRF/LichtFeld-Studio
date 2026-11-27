/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/event_bridge/control_boundary.hpp"
#include "core/event_bridge/scoped_handler.hpp"
#include "core/events.hpp"
#include "core/parameter_manager.hpp"
#include "core/parameters.hpp"
#include "core/scene.hpp"
#include "core/services.hpp"
#include "licht_test_support.hpp"
#include "training/trainer.hpp"
#include "training/training_manager.hpp"
#include "training/training_setup.hpp"

#include <chrono>
#include <condition_variable>
#include <cuda_runtime.h>
#include <filesystem>
#include <gtest/gtest.h>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <utility>
#include <vector>

namespace lfs::training {
    struct TrainerSaveHooksTestAccess {
        // As the app takes a save with no training thread: the writer starts, and the
        // save's hooks are left for a later frame.
        static void take_requested_save(Trainer& trainer) {
            trainer.consume_requested_project_snapshot(trainer.project_snapshot_iteration());
        }
    };
} // namespace lfs::training

namespace {
    using namespace lfs::core::events;

    std::mutex trim_test_mutex;
    std::condition_variable trim_test_condition;
    bool trim_test_ran = false;
    std::thread::id trim_test_thread;

    void signal_heap_trim() noexcept {
        {
            std::lock_guard lock(trim_test_mutex);
            trim_test_ran = true;
            trim_test_thread = std::this_thread::get_id();
        }
        trim_test_condition.notify_one();
    }

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
            lfs::vis::training_memory_detail::setClearTrainerTrimHookForTesting(nullptr);
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

    TEST_F(TrainingManagerPresentationTest, PausedTrainerRejectsDifferentStrategyParameters) {
        installTrainer(true);
        const auto before = manager_.getTrainer()->getParams();
        params_.setActiveStrategy("mcmc");
        params_.modifyActiveParams([](auto& params) {
            params.iterations = 12345;
            params.max_cap = 123456;
        });
        manager_.applyPendingParams();
        EXPECT_EQ(manager_.getTrainer()->getParams().optimization.to_json(),
                  before.optimization.to_json());
    }

    TEST_F(TrainingManagerPresentationTest, PausedTrainerKeepsSameStrategyUpdates) {
        installTrainer(true);
        params_.modifyActiveParams([](auto& params) {
            params.means_lr = 0.00012f;
        });
        const auto expected = params_.copyActiveParams();
        manager_.applyPendingParams();
        EXPECT_EQ(manager_.getTrainer()->getParams().optimization.to_json(), expected.to_json());
    }

    TEST_F(TrainingManagerPresentationTest, ReadyTrainerAcceptsDifferentStrategyParameters) {
        installTrainer();
        params_.setActiveStrategy("mcmc");
        params_.modifyActiveParams([](auto& params) {
            params.iterations = 12345;
            params.max_cap = 123456;
        });
        const auto expected = params_.copyActiveParams();
        manager_.applyPendingParams();
        EXPECT_EQ(manager_.getTrainer()->getParams().optimization.to_json(), expected.to_json());
    }

    // The panel unlocks after a finished run, and Start resets to a fresh trainer that takes the edited preset.
    TEST_F(TrainingManagerPresentationTest, FinishedTrainerAcceptsDifferentStrategyForTheNextRun) {
        installTrainer(true);
        EXPECT_FALSE(manager_.canChangeStrategy("mcmc"));
        auto& state_machine = const_cast<lfs::vis::TrainingStateMachine&>(manager_.getStateMachine());
        ASSERT_TRUE(state_machine.transitionTo(lfs::vis::TrainingState::Stopping));
        ASSERT_TRUE(state_machine.transitionToFinished(lfs::vis::FinishReason::Completed));
        EXPECT_TRUE(manager_.canChangeStrategy("mcmc"));
        EXPECT_TRUE(manager_.canChangeStrategy("mrnf"));
    }

    TEST_F(TrainingManagerPresentationTest, ClearTrainerRunsHeapTrimOffThread) {
#ifdef __GLIBC__
        installTrainer();
        {
            std::lock_guard lock(trim_test_mutex);
            trim_test_ran = false;
            trim_test_thread = {};
        }
        lfs::vis::training_memory_detail::setClearTrainerTrimHookForTesting(signal_heap_trim);

        const bool cleared = manager_.clearTrainer();
        ASSERT_TRUE(cleared);
        {
            std::unique_lock lock(trim_test_mutex);
            ASSERT_TRUE(trim_test_condition.wait_for(lock, std::chrono::seconds(5), [] { return trim_test_ran; }));
            EXPECT_NE(trim_test_thread, std::this_thread::get_id());
        }
        lfs::vis::training_memory_detail::setClearTrainerTrimHookForTesting(nullptr);
        lfs::vis::training_memory_detail::shutdownHeapTrimWorker();
        EXPECT_EQ(manager_.getTrainer(), nullptr);
        EXPECT_EQ(manager_.getState(), lfs::vis::TrainingState::Idle);
        EXPECT_TRUE(manager_.clearTrainer());
#else
        GTEST_SKIP() << "Requires glibc heap trim";
#endif
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

    // A manager holding a trainer that can save, paused with no training thread: as the app
    // is when it saves after a checkpoint is loaded.
    class TrainingManagerSaveHooksTest : public testing::Test {
    protected:
        void SetUp() override {
            int device_count = 0;
            if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0)
                GTEST_SKIP() << "CUDA device unavailable";
            const auto dataset = std::filesystem::path(TEST_DATA_DIR) / "bicycle";
            if (!std::filesystem::is_directory(dataset / "images_4"))
                GTEST_SKIP() << "bicycle dataset is required for a project save";

            params_.dataset.data_path = dataset;
            params_.dataset.images = "images_4";
            params_.dataset.output_path = temporary_.path;
            params_.optimization.iterations = 1;
            params_.optimization.strategy = "mcmc";
            params_.optimization.sh_degree = 0;
            params_.optimization.headless = true;
            params_.optimization.max_cap = 100000;
            params_.optimization.start_refine = 1;
            params_.optimization.stop_refine = 1;
            params_.optimization.enable_eval = false;

            ASSERT_TRUE(lfs::training::loadTrainingDataIntoScene(params_, scene_));
            ASSERT_TRUE(lfs::training::initializeTrainingModel(params_, scene_));
            auto trainer = std::make_unique<lfs::training::Trainer>(scene_);
            ASSERT_TRUE(trainer->initialize(params_));
            lfs::training::grant_headless_project_saves(*trainer, params_);
            // At 0 of 1 iterations: a checkpoint at its total restores as finished, not paused.
            manager_.setTrainerFromCheckpoint(std::move(trainer), 0);
            ASSERT_NE(manager_.getTrainer(), nullptr);
            ASSERT_TRUE(manager_.isPaused());
            ASSERT_FALSE(manager_.hasLiveTrainingThread());
        }

        // Starts a save as the app does, and leaves its hooks for a frame, which these tests
        // never run: only clearing the trainer can run them.
        void startSave() {
            auto& trainer = *manager_.getTrainer();
            ASSERT_NE(trainer.request_project_save(params_.dataset.project_file()), 0u);
            ASSERT_TRUE(trainer.can_flush_project_snapshot());
            lfs::training::TrainerSaveHooksTestAccess::take_requested_save(trainer);
        }

        lfs::test::licht::TemporaryDirectory temporary_;
        lfs::core::param::TrainingParameters params_;
        lfs::core::Scene scene_;
        lfs::vis::TrainerManager manager_;
    };

    // Fails if clearing the trainer, as a dataset load or a scene reset does, drops a save it is
    // still writing: with no training thread, only a later frame would run the save's hooks.
    TEST_F(TrainingManagerSaveHooksTest, ClearingTheTrainerEndsTheSaveItIsWriting) {
        using lfs::training::ControlHook;
        using lfs::training::HookContext;
        std::vector<std::pair<ControlHook, HookContext>> heard;
        {
            lfs::event::ScopedHandler hooks;
            for (const auto hook : {ControlHook::SaveStart, ControlHook::SaveEnd}) {
                hooks.subscribe_hook(hook, [&heard, hook](const HookContext& ctx) {
                    heard.emplace_back(hook, ctx);
                });
            }
            ASSERT_NO_FATAL_FAILURE(startSave());
            EXPECT_TRUE(heard.empty());
            ASSERT_TRUE(manager_.clearTrainer());
        }

        ASSERT_EQ(heard.size(), 2u);
        const auto& [start_hook, start] = heard[0];
        const auto& [end_hook, end] = heard[1];
        EXPECT_EQ(start_hook, ControlHook::SaveStart);
        EXPECT_EQ(end_hook, ControlHook::SaveEnd);
        ASSERT_TRUE(start.save);
        ASSERT_TRUE(end.save);
        EXPECT_EQ(end.save->snapshot_id, start.save->snapshot_id);
        EXPECT_TRUE(end.save->ok) << end.save->error;
        EXPECT_EQ(end.save->bytes, std::filesystem::file_size(params_.dataset.project_file()));
    }

    // Fails if a save hook that asks for its trainer to be cleared, as lf.clear_scene() from
    // one does, frees the trainer while its hooks still run, or stops the paused session
    // before the clear is refused.
    TEST_F(TrainingManagerSaveHooksTest, ASaveHookCannotClearItsTrainer) {
        using lfs::training::ControlHook;
        using lfs::training::HookContext;
        std::optional<bool> cleared_from_hook;
        bool paused_after_refusal = false;
        int ends = 0;
        {
            lfs::event::ScopedHandler hooks;
            hooks.subscribe_hook(ControlHook::SaveStart, [&](const HookContext&) {
                cleared_from_hook = manager_.clearTrainer();
                paused_after_refusal = manager_.isPaused();
            });
            hooks.subscribe_hook(ControlHook::SaveEnd, [&](const HookContext&) { ++ends; });
            ASSERT_NO_FATAL_FAILURE(startSave());
            // As a frame runs the hooks, once the save is written.
            manager_.getTrainer()->finish_saves();
        }

        ASSERT_TRUE(cleared_from_hook.has_value());
        EXPECT_FALSE(*cleared_from_hook);
        EXPECT_TRUE(paused_after_refusal);
        EXPECT_EQ(ends, 1);
        // Outside its hooks, the trainer can go.
        EXPECT_TRUE(manager_.clearTrainer());
        EXPECT_EQ(manager_.getTrainer(), nullptr);
    }

    // Fails if a save made while paused ends after the run that resumes from it starts: a
    // resume calls train() without initialize(), and train() runs the save's hooks first.
    TEST_F(TrainingManagerSaveHooksTest, ResumingRunsTheSaveHooksFirst) {
        using lfs::training::ControlHook;
        using lfs::training::HookContext;
        std::vector<ControlHook> heard;
        {
            lfs::event::ScopedHandler hooks;
            for (const auto hook : {ControlHook::SaveStart, ControlHook::SaveEnd, ControlHook::TrainingStart}) {
                hooks.subscribe_hook(hook, [&heard, hook](const HookContext&) { heard.push_back(hook); });
            }
            ASSERT_NO_FATAL_FAILURE(startSave());
            const auto trained = manager_.getTrainer()->train();
            ASSERT_TRUE(trained) << lfs::format_for_developer(trained.error());
        }
        ASSERT_GE(heard.size(), 3u);
        EXPECT_EQ(heard[0], ControlHook::SaveStart);
        EXPECT_EQ(heard[1], ControlHook::SaveEnd);
        EXPECT_EQ(heard[2], ControlHook::TrainingStart);
    }
} // namespace

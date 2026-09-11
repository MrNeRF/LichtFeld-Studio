/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/camera.hpp"
#include "core/event_bridge/control_boundary.hpp"
#include "core/events.hpp"
#include "core/scene.hpp"
#include "core/tensor.hpp"
#include "training/control/command_api.hpp"
#include "training/trainer.hpp"

#include <gtest/gtest.h>

#include <filesystem>
#include <limits>
#include <memory>

namespace {

    [[nodiscard]] std::shared_ptr<lfs::core::Camera> make_command_camera() {
        return std::make_shared<lfs::core::Camera>(
            lfs::core::Tensor::eye(3, lfs::core::Device::CPU),
            lfs::core::Tensor::zeros({3}, lfs::core::Device::CPU),
            100.0f, 100.0f, 32.0f, 32.0f,
            lfs::core::Tensor(), lfs::core::Tensor(),
            lfs::core::CameraModelType::PINHOLE,
            "camera.png", std::filesystem::path{}, std::filesystem::path{},
            64, 64, 0);
    }

    class TrainingTerminalStateTest : public testing::Test {
    protected:
        void SetUp() override {
            const auto cameras = scene_.addGroup("Cameras");
            scene_.addCamera("camera.png", cameras, make_command_camera());
            trainer_ = std::make_unique<lfs::training::Trainer>(scene_);
            lfs::training::CommandCenter::instance().bind_state_events();
        }

        void TearDown() override {
            auto& command_center = lfs::training::CommandCenter::instance();
            command_center.clear_snapshot(command_center.snapshot().trainer);
            trainer_.reset();
        }

        lfs::core::Scene scene_;
        std::unique_ptr<lfs::training::Trainer> trainer_;
    };

    TEST_F(TrainingTerminalStateTest, UnregisterCancelsAlreadyPendingCallback) {
        auto& boundary = lfs::training::ControlBoundary::instance();
        boundary.clear_all();

        int calls = 0;
        const auto handle = boundary.register_callback(
            lfs::training::ControlHook::TrainingEnd,
            [&](const lfs::training::HookContext&) { ++calls; });
        ASSERT_NE(handle, 0U);

        boundary.notify(lfs::training::ControlHook::TrainingEnd, {});
        boundary.unregister_callback(lfs::training::ControlHook::TrainingEnd, handle);
        boundary.drain_callbacks();

        EXPECT_EQ(calls, 0);
        boundary.clear_all();
    }

    TEST_F(TrainingTerminalStateTest, TerminalSnapshotIsInvalidatedOnlyByOwningTrainer) {
        auto& command_center = lfs::training::CommandCenter::instance();
        auto* const trainer = trainer_.get();
        const lfs::training::HookContext context{
            .iteration = 17,
            .loss = 0.25f,
            .num_gaussians = 42,
            .trainer = trainer};

        command_center.update_snapshot(
            context, 100, false, true, false, lfs::training::TrainingPhase::SafeControl);
        command_center.clear_snapshot(nullptr);
        EXPECT_EQ(command_center.snapshot().trainer, trainer);

        command_center.clear_snapshot(trainer);
        const auto snapshot = command_center.snapshot();
        EXPECT_EQ(snapshot.trainer, nullptr);
        EXPECT_FALSE(snapshot.is_running);
        EXPECT_EQ(snapshot.phase, lfs::training::TrainingPhase::Idle);
    }

    TEST_F(TrainingTerminalStateTest, ModelCommandsQueueWithoutDereferencingCallerThreadSnapshot) {
        auto& command_center = lfs::training::CommandCenter::instance();
        auto* const trainer = trainer_.get();
        const lfs::training::HookContext context{.trainer = trainer};
        command_center.update_snapshot(
            context, 100, false, true, false, lfs::training::TrainingPhase::SafeControl);

        lfs::training::Command command{
            .target = lfs::training::CommandTarget::Model,
            .op = "scale_attribute",
            .selection = {.kind = lfs::training::SelectionKind::All},
            .args = {{"attribute", std::string("means")}, {"factor", 2.0}}};

        EXPECT_TRUE(command_center.execute(command));
        command_center.clear_snapshot(trainer);
    }

    TEST_F(TrainingTerminalStateTest, QueuedCommandsRejectInvalidArgumentsSynchronously) {
        auto& command_center = lfs::training::CommandCenter::instance();
        auto* const trainer = trainer_.get();
        const lfs::training::HookContext context{.trainer = trainer};
        command_center.update_snapshot(
            context, 100, false, true, false, lfs::training::TrainingPhase::Forward);

        lfs::training::Command command{
            .target = lfs::training::CommandTarget::Optimizer,
            .op = "set_lr",
            .selection = {.kind = lfs::training::SelectionKind::All},
            .args = {{"value", std::numeric_limits<double>::quiet_NaN()}}};

        const auto result = command_center.execute(command);
        ASSERT_FALSE(result);
        EXPECT_NE(result.error().find("Non-finite"), std::string::npos);
        command_center.clear_snapshot(trainer);
    }

    TEST_F(TrainingTerminalStateTest, TrainingPausedSeedsSnapshotIterationAndPaused) {
        auto& command_center = lfs::training::CommandCenter::instance();
        command_center.clear_snapshot(command_center.snapshot().trainer);

        lfs::core::events::state::TrainingPaused{.iteration = 8200}.emit();

        const auto snapshot = command_center.snapshot();
        EXPECT_EQ(snapshot.iteration, 8200);
        EXPECT_TRUE(snapshot.is_paused);
        EXPECT_FALSE(snapshot.is_running);
    }

} // namespace

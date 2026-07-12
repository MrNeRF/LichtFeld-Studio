/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/event_bridge/control_boundary.hpp"
#include "training/control/command_api.hpp"

#include <gtest/gtest.h>

namespace {

    TEST(TrainingTerminalStateTest, UnregisterCancelsAlreadyPendingCallback) {
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

    TEST(TrainingTerminalStateTest, TerminalSnapshotIsInvalidatedOnlyByOwningTrainer) {
        auto& command_center = lfs::training::CommandCenter::instance();
        auto* const trainer = reinterpret_cast<lfs::training::Trainer*>(0x1);
        auto* const other = reinterpret_cast<lfs::training::Trainer*>(0x2);
        const lfs::training::HookContext context{
            .iteration = 17,
            .loss = 0.25f,
            .num_gaussians = 42,
            .trainer = trainer};

        command_center.update_snapshot(
            context, 100, false, true, false, lfs::training::TrainingPhase::SafeControl);
        command_center.clear_snapshot(other);
        EXPECT_EQ(command_center.snapshot().trainer, trainer);

        command_center.clear_snapshot(trainer);
        const auto snapshot = command_center.snapshot();
        EXPECT_EQ(snapshot.trainer, nullptr);
        EXPECT_FALSE(snapshot.is_running);
        EXPECT_EQ(snapshot.phase, lfs::training::TrainingPhase::Idle);
    }

} // namespace

/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/event_bridge/control_boundary.hpp"
#include <gtest/gtest.h>
#include <vector>

TEST(TrainingHookContext, DeferredCallbacksKeepTheirTerminalOutcome) {
    using namespace lfs::training;
    auto& boundary = ControlBoundary::instance();
    std::vector<TrainingOutcome> observed;
    const auto registration = boundary.register_callback(ControlHook::TrainingEnd,
                                                         [&](const HookContext& ctx) { observed.push_back(ctx.terminal_outcome); });
    const std::vector<TrainingOutcome> expected{TrainingOutcome::Completed, TrainingOutcome::Stopped, TrainingOutcome::Failed};
    for (auto outcome : expected) {
        HookContext context{.iteration = 7, .terminal_outcome = outcome};
        boundary.notify(ControlHook::TrainingEnd, context);
        context.terminal_outcome = TrainingOutcome::Unspecified;
    }
    EXPECT_TRUE(observed.empty());
    boundary.drain_callbacks();
    boundary.unregister_callback(ControlHook::TrainingEnd, registration);
    EXPECT_EQ(observed, expected);
}

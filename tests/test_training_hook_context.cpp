/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/event_bridge/control_boundary.hpp"
#include <gtest/gtest.h>
#include <memory>
#include <utility>
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

TEST(TrainingHookContext, DeferredSaveHooksKeepTheirRecordInOrder) {
    using namespace lfs::training;
    auto& boundary = ControlBoundary::instance();
    std::vector<std::pair<ControlHook, SaveRecord>> observed;
    const auto observe = [&](const ControlHook hook) {
        return [&observed, hook](const HookContext& ctx) {
            ASSERT_TRUE(ctx.save);
            observed.emplace_back(hook, *ctx.save);
        };
    };
    const auto start_registration = boundary.register_callback(ControlHook::SaveStart, observe(ControlHook::SaveStart));
    const auto end_registration = boundary.register_callback(ControlHook::SaveEnd, observe(ControlHook::SaveEnd));

    auto start = std::make_shared<SaveRecord>(
        SaveRecord{.snapshot_id = "a1f3", .path = "out/project.licht", .type = SaveType::Final});
    boundary.notify(ControlHook::SaveStart, HookContext{.iteration = 7000, .save = start});
    auto end = std::make_shared<SaveRecord>(*start);
    end->ok = true;
    end->generation = 3;
    end->bytes = 1024;
    // The writer thread queues the end; the queue keeps its own copy of each record.
    boundary.notify(ControlHook::SaveEnd, HookContext{.iteration = 7000, .save = std::move(end)});
    start.reset();

    EXPECT_TRUE(observed.empty());
    boundary.drain_callbacks();
    boundary.unregister_callback(ControlHook::SaveStart, start_registration);
    boundary.unregister_callback(ControlHook::SaveEnd, end_registration);

    ASSERT_EQ(observed.size(), 2u);
    EXPECT_EQ(observed[0].first, ControlHook::SaveStart);
    EXPECT_EQ(observed[0].second.snapshot_id, "a1f3");
    EXPECT_FALSE(observed[0].second.ok);
    EXPECT_EQ(observed[1].first, ControlHook::SaveEnd);
    EXPECT_EQ(observed[1].second.snapshot_id, "a1f3");
    EXPECT_EQ(observed[1].second.type, SaveType::Final);
    EXPECT_TRUE(observed[1].second.ok);
    EXPECT_EQ(observed[1].second.generation, 3u);
    EXPECT_EQ(observed[1].second.bytes, 1024u);
}

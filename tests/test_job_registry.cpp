/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "core/job_registry.hpp"

#include <gtest/gtest.h>

#include <thread>

namespace {

    TEST(JobRegistryTest,
         EnforcesTypeExclusivityAndMainThreadCompletion) {
        lfs::vis::JobRegistry registry;
        const auto first = registry.init(
            lfs::vis::JobType::ProjectWrite,
            "Preparing");
        ASSERT_TRUE(first);
        EXPECT_FALSE(registry.init(
            lfs::vis::JobType::ProjectWrite,
            "Compacting"));
        EXPECT_TRUE(registry.anyRunning(
            lfs::vis::JobType::ProjectWrite));

        std::jthread worker([&] {
            registry.work(*first);
            registry.report(
                *first, 0.5F, "Writing", std::nullopt);
            registry.finishWork(*first, false);
        });
        worker.join();

        auto snapshot = registry.update(*first);
        ASSERT_TRUE(snapshot);
        EXPECT_EQ(
            snapshot->status,
            lfs::vis::JobStatus::CompletionPending);
        EXPECT_FLOAT_EQ(snapshot->progress, 0.5F);
        EXPECT_EQ(snapshot->stage, "Writing");

        registry.completed(*first);
        EXPECT_FALSE(registry.anyRunning(
            lfs::vis::JobType::ProjectWrite));
        snapshot = registry.update(*first);
        ASSERT_TRUE(snapshot);
        EXPECT_EQ(
            snapshot->status,
            lfs::vis::JobStatus::Completed);
        EXPECT_FALSE(snapshot->running());
        EXPECT_FLOAT_EQ(snapshot->progress, 1.0F);

        registry.free(*first);
        EXPECT_FALSE(registry.update(*first));

        const auto failed = registry.init(
            lfs::vis::JobType::ProjectWrite,
            "Will fail");
        ASSERT_TRUE(failed);
        registry.failed(*failed, "injected failure");
        snapshot = registry.update(*failed);
        ASSERT_TRUE(snapshot);
        EXPECT_EQ(snapshot->status,
                  lfs::vis::JobStatus::Failed);
        EXPECT_FALSE(snapshot->running());
        EXPECT_EQ(snapshot->error,
                  "injected failure");
        EXPECT_FALSE(registry.anyRunning(
            lfs::vis::JobType::ProjectWrite));
        registry.free(*failed);
    }

    TEST(JobRegistryTest,
         CancellationAndErrorStayInRegistry) {
        lfs::vis::JobRegistry registry;
        const auto handle = registry.init(
            lfs::vis::JobType::Export, "Starting");
        ASSERT_TRUE(handle);
        registry.requestCancel(*handle);

        std::jthread worker([&] {
            registry.work(*handle);
            EXPECT_TRUE(
                registry.cancelRequested(*handle));
            registry.finishWork(
                *handle, true, "worker detail");
        });
        worker.join();

        auto snapshot = registry.update(*handle);
        ASSERT_TRUE(snapshot);
        EXPECT_TRUE(snapshot->cancel_requested);
        EXPECT_TRUE(snapshot->worker_canceled);
        EXPECT_EQ(snapshot->stage, "Cancelling");
        EXPECT_EQ(snapshot->error, "worker detail");

        registry.canceled(*handle);
        snapshot = registry.update(*handle);
        ASSERT_TRUE(snapshot);
        EXPECT_EQ(
            snapshot->status,
            lfs::vis::JobStatus::Canceled);
        EXPECT_FALSE(snapshot->running());
        EXPECT_FALSE(registry.anyRunning(
            lfs::vis::JobType::Export));
        registry.free(*handle);
    }

} // namespace

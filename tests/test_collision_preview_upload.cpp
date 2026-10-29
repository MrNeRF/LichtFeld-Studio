/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "rendering/passes/collision_preview_upload.hpp"
#include <gtest/gtest.h>
#include <stdexcept>

namespace {
    using Status = lfs::vis::detail::CollisionUploadStatus;
    struct Resource {};
    using Job = lfs::vis::detail::CollisionPreviewUploadJob<Resource>;

    TEST(CollisionPreviewUploadJob, PollAndCancellationNeverJoinABlockedWorker) {
        Job job;
        std::promise<void> entered, release;
        auto gate = release.get_future().share();
        auto resource = std::make_shared<Resource>();
        std::weak_ptr<Resource> weak = resource;
        ASSERT_TRUE(job.start(resource, [&](Resource&, std::stop_token stop) {
            entered.set_value();
            gate.wait();
            return stop.stop_requested() ? Status::Canceled : Status::Ready;
        }));
        entered.get_future().wait();
        resource.reset();
        EXPECT_TRUE(job.active());
        EXPECT_FALSE(job.poll());
        job.cancel();
        EXPECT_FALSE(job.poll());
        EXPECT_FALSE(weak.expired());
        EXPECT_FALSE(job.start(std::make_shared<Resource>(), [](Resource&, std::stop_token) { return Status::Ready; }));
        release.set_value();
        auto completed = job.finish();
        ASSERT_TRUE(completed);
        EXPECT_EQ(completed->status, Status::Canceled);
        EXPECT_FALSE(job.active());
        EXPECT_FALSE(job.poll());
        EXPECT_FALSE(weak.expired());
        completed.reset();
        EXPECT_TRUE(weak.expired());
    }

    TEST(CollisionPreviewUploadJob, CompletionIsPublishedOnceAndWorkerCanBeReused) {
        Job job;
        for (int i = 0; i < 2; ++i) {
            auto resource = std::make_shared<Resource>();
            ASSERT_TRUE(job.start(resource, [](Resource&, std::stop_token) { return Status::Ready; }));
            auto completion = job.finish();
            ASSERT_TRUE(completion);
            EXPECT_EQ(completion->status, Status::Ready);
            EXPECT_EQ(completion->resource, resource);
            EXPECT_FALSE(job.finish());
        }
    }

    TEST(CollisionPreviewUploadJob, UnexpectedExceptionRetainsTheResourceForFenceRetirement) {
        Job job;
        auto resource = std::make_shared<Resource>();
        ASSERT_TRUE(job.start(resource, [](Resource&, std::stop_token) -> Status {
            throw std::runtime_error("injected worker failure");
        }));
        auto completion = job.finish();
        ASSERT_TRUE(completion);
        EXPECT_EQ(completion->status, Status::PendingFailure);
        EXPECT_EQ(completion->resource, resource);
        EXPECT_FALSE(job.active());
    }

    TEST(CollisionPreviewUploadJob, WakeObservesPublishedResultWithoutWaitingForCallback) {
        Job job;
        std::promise<void> entered, release;
        auto gate = release.get_future().share();
        ASSERT_TRUE(job.start(std::make_shared<Resource>(), [](Resource&, std::stop_token) { return Status::Ready; }, [&] {
            entered.set_value();
            gate.wait(); }));
        entered.get_future().wait();
        // The callback remains blocked. poll must already publish the result;
        // waking before promise publication would lose an idle viewer's wake.
        auto completion = job.poll();
        EXPECT_TRUE(completion);
        if (completion)
            EXPECT_EQ(completion->status, Status::Ready);
        release.set_value();
        EXPECT_FALSE(job.finish()); // join callback before its captured promises die
    }
} // namespace

/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "rendering/frame_demand.hpp"
#include "rendering/passes/collision_preview_cache.hpp"
#include <gtest/gtest.h>
#include <limits>

namespace {
    using Status = lfs::vis::detail::CollisionUploadStatus;
    struct Source {};
    struct Resource {
        std::shared_ptr<const Source> owner;
        bool retired = true;
    };
    using Cache = lfs::vis::detail::CollisionPreviewCache<Source, Resource>;

    class CollisionPreviewCache : public ::testing::Test {
    protected:
        Cache cache{100, std::chrono::seconds(5)};
        std::shared_ptr<const Source> source = std::make_shared<Source>();
        Cache::Time now{};
        uint64_t frame = 0;
        Status outcome = Status::Ready;
        std::weak_ptr<Resource> last_upload;
        static bool retired(const Resource& resource) { return resource.retired; }
        std::shared_ptr<Resource> request(size_t bytes = 40) {
            return cache.request(source, bytes, ++frame, now, [&](auto& resource) {
                resource = std::make_shared<Resource>();
                resource->owner = source;
                resource->retired = outcome != Status::PendingFailure;
                last_upload = resource;
                return outcome; }, retired);
        }
        void maintain() { cache.maintain(now, retired); }
    };

    TEST_F(CollisionPreviewCache, DisabledDepthAndExcludedPanelsNeedNoPreparation) {
        for (const auto opacity : {0.f, -1.f, std::numeric_limits<float>::quiet_NaN()})
            EXPECT_FALSE(Cache::needsPreview(true, opacity, false, 1));
        EXPECT_FALSE(Cache::needsPreview(false, .4f, false, 1));
        EXPECT_FALSE(Cache::needsPreview(true, .4f, true, 1));
        EXPECT_FALSE(Cache::needsPreview(true, .4f, false, 0));
        EXPECT_TRUE(Cache::needsPreview(true, .4f, false, 2));
        EXPECT_EQ(cache.stats().attempts, 0u);
    }

    TEST_F(CollisionPreviewCache, ReusesImmutableVolumeAcrossFramesAndInstances) {
        const auto first = request();
        ASSERT_TRUE(first);
        EXPECT_EQ(request(), first);
        EXPECT_EQ(request(), first);
        EXPECT_EQ(cache.ready(source.get()), first);
        EXPECT_EQ(cache.stats().attempts, 1u);
        EXPECT_EQ(cache.stats().hits, 2u);
    }

    TEST_F(CollisionPreviewCache, ShortHideOrZeroOpacityRetainsBudgetedResource) {
        auto first = request();
        const std::weak_ptr<Resource> weak = first;
        first.reset();
        frame += 3; // no requests while hidden
        now += std::chrono::seconds(1);
        maintain();
        EXPECT_FALSE(weak.expired());
        EXPECT_EQ(request(), weak.lock());
        EXPECT_EQ(cache.stats().attempts, 1u);
        EXPECT_EQ(cache.stats().resident_bytes, 40u);
    }

    TEST_F(CollisionPreviewCache, IdleEvictionReleasesCacheButNotInFlightFrame) {
        auto in_flight = request();
        const std::weak_ptr<Resource> weak = in_flight;
        now += std::chrono::seconds(5);
        maintain();
        EXPECT_EQ(cache.stats().resident_bytes, 0u);
        EXPECT_EQ(cache.stats().evictions, 1u);
        ASSERT_FALSE(weak.expired());
        EXPECT_EQ(request(), in_flight); // recover via weak entry
        now += std::chrono::seconds(5);
        maintain();
        in_flight.reset();
        EXPECT_TRUE(weak.expired());
        EXPECT_TRUE(request());
        EXPECT_EQ(cache.stats().attempts, 2u);
    }

    TEST_F(CollisionPreviewCache, OversizedResourceRemainsReusableThroughFrameOwnership) {
        auto in_flight = request(200);
        maintain();
        EXPECT_EQ(cache.stats().resident_bytes, 0u);
        EXPECT_EQ(request(200), in_flight);
        maintain();
        EXPECT_EQ(cache.stats().attempts, 1u);
        const std::weak_ptr<Resource> weak = in_flight;
        in_flight.reset();
        EXPECT_TRUE(weak.expired());
    }

    TEST_F(CollisionPreviewCache, BudgetEvictsLeastRecentlyUsedWithoutInvalidatingFrames) {
        auto a = request(60);
        const auto source_a = source;
        source = std::make_shared<Source>();
        now += std::chrono::seconds(1);
        auto b = request(60);
        const std::weak_ptr<Resource> weak_a = a, weak_b = b;
        maintain();
        EXPECT_EQ(cache.stats().resident_bytes, 60u);
        a.reset();
        b.reset();
        EXPECT_TRUE(weak_a.expired());
        EXPECT_FALSE(weak_b.expired());
        EXPECT_FALSE(cache.ready(source_a.get()));
    }

    TEST_F(CollisionPreviewCache, TransientFailureRetriesWithBackoffAndRecovers) {
        outcome = Status::RetryableFailure;
        EXPECT_FALSE(request());
        EXPECT_TRUE(last_upload.expired());
        EXPECT_FALSE(cache.ready(source.get()));
        EXPECT_FALSE(request());
        EXPECT_EQ(cache.stats().attempts, 1u);
        now += std::chrono::seconds(1);
        outcome = Status::Ready;
        EXPECT_TRUE(request());
        EXPECT_EQ(cache.stats().attempts, 2u);
        EXPECT_EQ(cache.stats().failures, 1u);
    }

    TEST_F(CollisionPreviewCache, RepeatedFailuresStopUntilHiddenAndShown) {
        outcome = Status::RetryableFailure;
        EXPECT_FALSE(request());
        now += std::chrono::seconds(1);
        EXPECT_FALSE(request());
        now += std::chrono::seconds(2);
        EXPECT_FALSE(request());
        now += std::chrono::seconds(100);
        EXPECT_FALSE(request());
        EXPECT_EQ(cache.stats().attempts, 3u);
        frame += 2;
        outcome = Status::Ready;
        EXPECT_TRUE(request());
        EXPECT_EQ(cache.stats().attempts, 4u);
    }

    TEST_F(CollisionPreviewCache, PermanentFailureNeverRetriesIncludingAfterReactivation) {
        outcome = Status::PermanentFailure;
        EXPECT_FALSE(request());
        EXPECT_TRUE(last_upload.expired());
        now += std::chrono::seconds(100);
        frame += 3;
        outcome = Status::Ready;
        EXPECT_FALSE(request());
        EXPECT_EQ(cache.stats().attempts, 1u);
        EXPECT_FALSE(cache.ready(source.get()));
    }

    TEST_F(CollisionPreviewCache, PendingUploadCannotRetryOrEvictBeforeRetirement) {
        outcome = Status::PendingFailure;
        EXPECT_FALSE(request(200));
        const auto pending = last_upload;
        now += std::chrono::seconds(100);
        frame += 2;
        maintain();
        EXPECT_EQ(cache.stats().pending_bytes, 200u);
        EXPECT_FALSE(request(200));
        EXPECT_EQ(cache.stats().attempts, 1u);
        ASSERT_FALSE(pending.expired());
        pending.lock()->retired = true;
        maintain();
        EXPECT_TRUE(pending.expired());
        EXPECT_EQ(cache.stats().pending_bytes, 0u);
        outcome = Status::Ready;
        EXPECT_TRUE(request(200));
        EXPECT_EQ(cache.stats().attempts, 2u);
    }

    TEST_F(CollisionPreviewCache, ClearPreservesFrameOwnerAndCannotReturnStaleResource) {
        auto in_flight = request();
        const std::weak_ptr<Resource> weak = in_flight;
        cache.clear();
        EXPECT_FALSE(cache.ready(source.get()));
        EXPECT_FALSE(weak.expired());
        const auto replacement = request();
        EXPECT_NE(replacement, in_flight);
        in_flight.reset();
        EXPECT_TRUE(weak.expired());
    }

    TEST_F(CollisionPreviewCache, SourceRemovalEventuallyReleasesCachedOwnership) {
        auto resource = request();
        const std::weak_ptr<const Source> weak_source = source;
        const std::weak_ptr<Resource> weak_resource = resource;
        source.reset();
        resource.reset();
        EXPECT_FALSE(weak_source.expired());
        now += std::chrono::seconds(5);
        maintain();
        EXPECT_TRUE(weak_source.expired());
        EXPECT_TRUE(weak_resource.expired());
        EXPECT_EQ(cache.stats().resident_bytes, 0u);
    }

    TEST_F(CollisionPreviewCache, RemovedSourceWithPendingUploadStaysAliveUntilFenceSignals) {
        outcome = Status::PendingFailure;
        EXPECT_FALSE(request());
        const auto pending = last_upload;
        const std::weak_ptr<const Source> weak_source = source;
        source.reset();
        now += std::chrono::seconds(100);
        maintain();
        EXPECT_FALSE(weak_source.expired());
        ASSERT_FALSE(pending.expired());
        pending.lock()->retired = true;
        maintain();
        EXPECT_TRUE(weak_source.expired());
        EXPECT_TRUE(pending.expired());
    }

    TEST_F(CollisionPreviewCache, RecycledSourceAddressDoesNotInheritPermanentFailure) {
        Source storage;
        source = std::shared_ptr<const Source>(&storage, [](auto*) {});
        outcome = Status::PermanentFailure;
        EXPECT_FALSE(request());
        source.reset();
        // Reuse the address with a new ownership identity before maintenance.
        source = std::shared_ptr<const Source>(&storage, [](auto*) {});
        outcome = Status::Ready;
        EXPECT_TRUE(request());
        EXPECT_EQ(cache.stats().attempts, 2u);
    }

    TEST_F(CollisionPreviewCache, SharedInstanceCannotRearmFailedUploadInSameFrame) {
        outcome = Status::RetryableFailure;
        EXPECT_FALSE(request());
        --frame; // the second instance is prepared in the same viewer frame
        EXPECT_FALSE(request());
        EXPECT_EQ(cache.stats().attempts, 1u);
        EXPECT_EQ(cache.stats().failures, 1u);
    }

    TEST_F(CollisionPreviewCache, RetryDeadlineWakesAnIdleViewWithoutNewInput) {
        outcome = Status::RetryableFailure;
        EXPECT_FALSE(request());
        EXPECT_EQ(cache.nextDeadline(frame, now), now + std::chrono::seconds(1));
        EXPECT_FALSE(cache.retryDue(frame, now));
        // No request/frame advances while the viewer is asleep.
        now += std::chrono::seconds(1);
        maintain();
        EXPECT_TRUE(cache.retryDue(frame, now));
        outcome = Status::Ready;
        EXPECT_TRUE(request());
        EXPECT_FALSE(cache.retryDue(frame, now));
    }

    TEST_F(CollisionPreviewCache, IdleMaintenanceEvictsWithoutPreparingAnotherFrame) {
        auto uploaded = request();
        const std::weak_ptr<Resource> weak = uploaded;
        uploaded.reset();
        EXPECT_EQ(cache.nextDeadline(frame, now), now + std::chrono::seconds(5));
        now += std::chrono::seconds(5);
        maintain();
        EXPECT_TRUE(weak.expired());
        EXPECT_FALSE(cache.retryDue(frame, now));
        EXPECT_FALSE(cache.nextDeadline(frame, now));
    }

    TEST_F(CollisionPreviewCache, HiddenOrExhaustedFailuresDoNotScheduleViewRetries) {
        outcome = Status::RetryableFailure;
        EXPECT_FALSE(request());
        ++frame; // prepare a frame without requesting the hidden volume
        now += std::chrono::seconds(1);
        EXPECT_FALSE(cache.retryDue(frame, now));
        EXPECT_FALSE(cache.nextDeadline(frame, now));
        outcome = Status::PermanentFailure;
        EXPECT_FALSE(request());
        EXPECT_FALSE(cache.nextDeadline(frame, now));
    }

    TEST_F(CollisionPreviewCache, PendingFencePollSchedulesNoViewUntilRetirement) {
        outcome = Status::PendingFailure;
        EXPECT_FALSE(request());
        EXPECT_EQ(cache.nextDeadline(frame, now), now + std::chrono::milliseconds(250));
        now += std::chrono::seconds(10);
        maintain();
        EXPECT_FALSE(cache.retryDue(frame, now));
        last_upload.lock()->retired = true;
        maintain();
        EXPECT_TRUE(cache.retryDue(frame, now));
        EXPECT_EQ(cache.nextDeadline(frame, now), Cache::Time{} + std::chrono::seconds(1));
    }

    TEST_F(CollisionPreviewCache, RetryProducesAViewDemandButIdleEvictionDoesNot) {
        lfs::vis::FrameDemandLedger ledger;
        outcome = Status::RetryableFailure;
        EXPECT_FALSE(request());
        ASSERT_TRUE(cache.nextDeadline(frame, now));
        now = *cache.nextDeadline(frame, now);
        maintain();
        if (cache.retryDue(frame, now))
            ledger.request({.reason = lfs::vis::FrameReason::AsyncCompletion,
                            .scope = lfs::vis::FrameScope::View,
                            .views = 1,
                            .flags = lfs::vis::DirtyFlag::OVERLAY,
                            .detail = "collision_preview_retry"});
        const auto plan = ledger.plan(now);
        EXPECT_TRUE(plan.present);
        EXPECT_EQ(plan.render_views, 1u);
        outcome = Status::Ready;
        EXPECT_TRUE(request());
        now += std::chrono::seconds(5);
        maintain();
        EXPECT_FALSE(cache.retryDue(frame, now));
        EXPECT_FALSE(cache.nextDeadline(frame, now));
    }
} // namespace

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

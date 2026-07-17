/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/error.hpp"
#include "core/error_bus.hpp"
#include "core/error_codes.hpp"
#include "core/error_reporter.hpp"
#include "core/events.hpp"
#include "core/source_site.hpp"
#include "gui/error_event_bridge.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <mutex>
#include <thread>
#include <vector>

namespace {

    lfs::Error makeError(const lfs::ErrorCode code, const lfs::ErrorDomain domain,
                         const std::string& message, const char* operation = "unit_test") {
        return lfs::make_legacy_error(message, lfs::LegacyErrorContext{
                                                   .code = code,
                                                   .domain = domain,
                                                   .operation = operation,
                                                   .source = LFS_SOURCE_SITE_CURRENT(),
                                               });
    }

    lfs::ErrorNotification makeNotification(lfs::Error error, const lfs::OperationId op,
                                            const lfs::ErrorSurface surface = lfs::ErrorSurface::Modal) {
        // lfs::Error has no public default ctor, so aggregate-init with the error
        // provided rather than default-constructing the notification.
        return lfs::ErrorNotification{
            .error = std::move(error),
            .surface = surface,
            .actions = {},
            .operation_id = op,
        };
    }

    class RecordingConsumer final : public lfs::NativeErrorConsumer {
    public:
        void on_error(const lfs::ErrorNotification& notification) noexcept override {
            count_.fetch_add(1, std::memory_order_relaxed);
            std::lock_guard lock(mutex_);
            codes_.push_back(notification.error.code());
            surfaces_.push_back(notification.surface);
            operation_ids_.push_back(notification.operation_id);
            action_counts_.push_back(notification.actions.size());
        }

        [[nodiscard]] int count() const { return count_.load(std::memory_order_relaxed); }

        std::mutex mutex_;
        std::vector<lfs::ErrorCode> codes_;
        std::vector<lfs::ErrorSurface> surfaces_;
        std::vector<lfs::OperationId> operation_ids_;
        std::vector<std::size_t> action_counts_;

    private:
        std::atomic<int> count_{0};
    };

} // namespace

TEST(ErrorBusTest, PublishDeliversToSingleSubscriber) {
    auto bus = lfs::ErrorBus::create_isolated_for_testing();
    RecordingConsumer consumer;
    auto subscription = bus->subscribe(consumer);

    const lfs::OperationId op = lfs::OperationId::generate();
    lfs::ErrorNotification notification =
        makeNotification(makeError(lfs::ErrorCode::DataLoss, lfs::ErrorDomain::IO, "bad dataset"), op);
    notification.actions.push_back(lfs::ErrorAction{.kind = lfs::ErrorActionKind::Dismiss});

    bus->publish(std::move(notification));

    ASSERT_EQ(consumer.count(), 1);
    EXPECT_EQ(consumer.codes_.front(), lfs::ErrorCode::DataLoss);
    EXPECT_EQ(consumer.surfaces_.front(), lfs::ErrorSurface::Modal);
    EXPECT_EQ(consumer.operation_ids_.front(), op);
    EXPECT_EQ(consumer.action_counts_.front(), 1u);
}

TEST(ErrorBusTest, PublishDeliversToEverySubscriber) {
    auto bus = lfs::ErrorBus::create_isolated_for_testing();
    RecordingConsumer a;
    RecordingConsumer b;
    auto sub_a = bus->subscribe(a);
    auto sub_b = bus->subscribe(b);

    bus->publish(makeNotification(makeError(lfs::ErrorCode::Internal, lfs::ErrorDomain::Training, "boom"),
                                  lfs::OperationId::generate()));

    EXPECT_EQ(a.count(), 1);
    EXPECT_EQ(b.count(), 1);
}

TEST(ErrorBusTest, PublishWithoutSubscriberTakesDurableFallback) {
    auto bus = lfs::ErrorBus::create_isolated_for_testing();
    // No consumer: publish must route to the ErrorReporter durable fallback and
    // return without blocking or crashing. Nothing observes the log here; the
    // contract under test is that publish is safe with zero subscribers.
    bus->publish(makeNotification(makeError(lfs::ErrorCode::NotFound, lfs::ErrorDomain::IO, "missing"),
                                  lfs::OperationId::generate()));
    SUCCEED();
}

TEST(ErrorBusTest, SubscriptionRaiiStopsDelivery) {
    auto bus = lfs::ErrorBus::create_isolated_for_testing();
    RecordingConsumer consumer;
    {
        auto subscription = bus->subscribe(consumer);
        bus->publish(makeNotification(
            makeError(lfs::ErrorCode::Internal, lfs::ErrorDomain::IO, "one"),
            lfs::OperationId::generate()));
        EXPECT_EQ(consumer.count(), 1);
    }
    // After the Subscription is destroyed the consumer no longer receives.
    bus->publish(makeNotification(makeError(lfs::ErrorCode::Internal, lfs::ErrorDomain::IO, "two"),
                                  lfs::OperationId::generate()));
    EXPECT_EQ(consumer.count(), 1);
}

TEST(ErrorBusTest, DedupCollapsesSameFingerprintAndOperation) {
    auto bus = lfs::ErrorBus::create_isolated_for_testing();
    RecordingConsumer consumer;
    auto subscription = bus->subscribe(consumer);

    const lfs::Error error = makeError(lfs::ErrorCode::DeviceLost, lfs::ErrorDomain::Vulkan, "lost");
    const lfs::OperationId op = lfs::OperationId::generate();

    bus->publish(makeNotification(error, op));
    bus->publish(makeNotification(error, op)); // same fingerprint + op -> suppressed
    EXPECT_EQ(consumer.count(), 1);

    // A different operation id is a distinct operation and must surface.
    bus->publish(makeNotification(error, lfs::OperationId::generate()));
    EXPECT_EQ(consumer.count(), 2);
}

TEST(ErrorBusTest, ThreadMarshalDeliversAllFromWorkerThreads) {
    auto bus = lfs::ErrorBus::create_isolated_for_testing();
    RecordingConsumer consumer;
    auto subscription = bus->subscribe(consumer);

    constexpr int kThreads = 8;
    constexpr int kPerThread = 50;
    std::vector<std::thread> workers;
    workers.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t) {
        workers.emplace_back([&bus] {
            for (int i = 0; i < kPerThread; ++i) {
                // Fresh op id per publish keeps every notification distinct so
                // dedup does not collapse legitimate concurrent failures.
                bus->publish(makeNotification(
                    makeError(lfs::ErrorCode::Internal, lfs::ErrorDomain::Training, "worker"),
                    lfs::OperationId::generate()));
            }
        });
    }
    for (std::thread& worker : workers) {
        worker.join();
    }

    EXPECT_EQ(consumer.count(), kThreads * kPerThread);
}

TEST(ErrorBusTest, RegisteredConsumerSurfacesWithoutPython) {
    // The native consumer subscribes at the core level with no Python symbols
    // involved, proving availability-independence from Python init.
    auto bus = lfs::ErrorBus::create_isolated_for_testing();
    RecordingConsumer consumer;
    auto subscription = bus->subscribe(consumer);
    bus->publish(makeNotification(
        makeError(lfs::ErrorCode::InvalidArgument, lfs::ErrorDomain::App, "bad config"),
        lfs::OperationId::generate()));
    EXPECT_EQ(consumer.count(), 1);
}

TEST(ErrorFingerprintTest, EqualForSameDimensionsDiffersOtherwise) {
    const lfs::Error a = makeError(lfs::ErrorCode::DataLoss, lfs::ErrorDomain::IO, "x", "op");
    const lfs::Error b = makeError(lfs::ErrorCode::DataLoss, lfs::ErrorDomain::IO, "y", "op");
    const lfs::Error c = makeError(lfs::ErrorCode::Internal, lfs::ErrorDomain::IO, "x", "op");

    EXPECT_EQ(lfs::core::error_fingerprint(a), lfs::core::error_fingerprint(b));
    EXPECT_NE(lfs::core::error_fingerprint(a), lfs::core::error_fingerprint(c));
}

TEST(ErrorEventBridgeTest, TrainingStopSuppressesModal) {
    lfs::core::events::state::TrainingCompleted stopped{};
    stopped.success = false;
    stopped.user_stopped = true;
    stopped.error = "interrupted";
    EXPECT_FALSE(lfs::vis::gui::translateTrainingCompleted(stopped).has_value());
}

TEST(ErrorEventBridgeTest, TrainingFailureSurfacesAsModalError) {
    lfs::core::events::state::TrainingCompleted failed{};
    failed.success = false;
    failed.user_stopped = false;
    failed.resource_exhausted = false;
    failed.error = "kernel launch failed";

    const auto notification = lfs::vis::gui::translateTrainingCompleted(failed);
    ASSERT_TRUE(notification.has_value());
    EXPECT_EQ(notification->surface, lfs::ErrorSurface::Modal);
    EXPECT_EQ(notification->error.domain(), lfs::ErrorDomain::Training);
    EXPECT_EQ(notification->error.code(), lfs::ErrorCode::Internal);
}

TEST(ErrorEventBridgeTest, TrainingOomMapsToResourceExhausted) {
    lfs::core::events::state::TrainingCompleted oom{};
    oom.success = false;
    oom.user_stopped = false;
    oom.resource_exhausted = true;
    oom.error = "out of memory (12.0 GB)";

    const auto notification = lfs::vis::gui::translateTrainingCompleted(oom);
    ASSERT_TRUE(notification.has_value());
    EXPECT_EQ(notification->error.code(), lfs::ErrorCode::ResourceExhausted);
}

TEST(ErrorEventBridgeTest, TrainingSuccessDoesNotSurface) {
    lfs::core::events::state::TrainingCompleted success{};
    success.success = true;
    success.user_stopped = false;
    EXPECT_FALSE(lfs::vis::gui::translateTrainingCompleted(success).has_value());
}

TEST(ErrorEventBridgeTest, DiskSpaceCaseIsLeftToNativeHandler) {
    lfs::core::events::state::DiskSpaceSaveFailed disk{};
    disk.is_disk_space_error = true;
    disk.error = "no space";
    EXPECT_FALSE(lfs::vis::gui::translateDiskSpaceSaveFailed(disk).has_value());

    lfs::core::events::state::DiskSpaceSaveFailed other{};
    other.is_disk_space_error = false;
    other.is_checkpoint = true;
    other.error = "write failed";
    const auto notification = lfs::vis::gui::translateDiskSpaceSaveFailed(other);
    ASSERT_TRUE(notification.has_value());
    EXPECT_EQ(notification->error.domain(), lfs::ErrorDomain::IO);
}

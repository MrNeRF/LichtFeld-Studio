/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/error.hpp"
#include "core/error_bus.hpp"
#include "core/error_codes.hpp"
#include "core/error_reporter.hpp"
#include "core/events.hpp"
#include "core/frame_state_machine.hpp"
#include "core/modal_request.hpp"
#include "core/source_site.hpp"
#include "gui/error_event_bridge.hpp"
#include "gui/error_surface_types.hpp"
#include "gui/gui_error_consumer.hpp"
#include "gui/rml_status_bar.hpp"
#include "gui/rml_toast_overlay.hpp"
#include "gui/string_keys.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
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
        void on_error(const lfs::ErrorNotification& notification,
                      const lfs::ErrorDeliveryInfo& delivery) noexcept override {
            count_.fetch_add(1, std::memory_order_relaxed);
            std::lock_guard lock(mutex_);
            codes_.push_back(notification.error.code());
            surfaces_.push_back(notification.surface);
            operation_ids_.push_back(notification.operation_id);
            action_counts_.push_back(notification.actions.size());
            repeats_.push_back(delivery.suppressed_repeats);
        }

        [[nodiscard]] int count() const { return count_.load(std::memory_order_relaxed); }

        std::mutex mutex_;
        std::vector<lfs::ErrorCode> codes_;
        std::vector<lfs::ErrorSurface> surfaces_;
        std::vector<lfs::OperationId> operation_ids_;
        std::vector<std::size_t> action_counts_;
        std::vector<std::uint32_t> repeats_;

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
    EXPECT_EQ(consumer.repeats_.front(), 0u);

    // A different operation id is a distinct operation and must surface.
    bus->publish(makeNotification(error, lfs::OperationId::generate()));
    EXPECT_EQ(consumer.count(), 2);
    EXPECT_EQ(consumer.repeats_.back(), 0u);
}

TEST(ErrorBusTest, DifferentOperationIdBypassesDedup) {
    auto bus = lfs::ErrorBus::create_isolated_for_testing();
    RecordingConsumer consumer;
    auto subscription = bus->subscribe(consumer);

    const lfs::Error error = makeError(lfs::ErrorCode::DeviceLost, lfs::ErrorDomain::Vulkan, "lost");
    bus->publish(makeNotification(error, lfs::OperationId::generate()));
    bus->publish(makeNotification(error, lfs::OperationId::generate()));
    bus->publish(makeNotification(error, lfs::OperationId::generate()));

    EXPECT_EQ(consumer.count(), 3);
    for (const std::uint32_t repeats : consumer.repeats_) {
        EXPECT_EQ(repeats, 0u);
    }
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

TEST(ErrorEventBridgeTest, CancelledExportMapsToStatusOnly) {
    lfs::core::events::state::ExportFailed e{};
    e.error = "Export cancelled by user";
    e.cancelled = true;

    const auto notification = lfs::vis::gui::translateExportFailed(e);
    ASSERT_TRUE(notification.has_value());
    EXPECT_EQ(notification->surface, lfs::ErrorSurface::StatusOnly);
    EXPECT_EQ(notification->error.severity(), lfs::Severity::Info);
    EXPECT_EQ(notification->error.code(), lfs::ErrorCode::Cancelled);
    EXPECT_TRUE(notification->actions.empty());
}

TEST(ErrorEventBridgeTest, FailedExportStaysModal) {
    lfs::core::events::state::ExportFailed e{};
    e.error = "disk write error";
    e.cancelled = false;

    const auto notification = lfs::vis::gui::translateExportFailed(e);
    ASSERT_TRUE(notification.has_value());
    EXPECT_EQ(notification->surface, lfs::ErrorSurface::Modal);
    bool has_open_log = false;
    for (const lfs::ErrorAction& action : notification->actions) {
        if (action.kind == lfs::ErrorActionKind::OpenLog)
            has_open_log = true;
    }
    EXPECT_TRUE(has_open_log);
}

TEST(ErrorEventBridgeTest, CudaVersionUnsupportedMapsToToast) {
    lfs::core::events::state::CudaVersionUnsupported e{};
    e.major = 11;
    e.minor = 0;
    e.min_major = 12;
    e.min_minor = 0;

    const auto notification = lfs::vis::gui::translateCudaVersionUnsupported(e);
    ASSERT_TRUE(notification.has_value());
    EXPECT_EQ(notification->surface, lfs::ErrorSurface::Toast);
    EXPECT_TRUE(notification->actions.empty());
}

namespace {
    constexpr std::chrono::steady_clock::time_point kBase{std::chrono::seconds(100)};
} // namespace

TEST(ErrorDedupTest, RedeliveryAfterWindowCarriesSuppressedCount) {
    lfs::ErrorDedup dedup;
    constexpr std::uint64_t key = 0x1234;

    EXPECT_FALSE(dedup.check(key, kBase).suppress); // first delivery
    constexpr int kSuppressed = 4;
    for (int i = 1; i <= kSuppressed; ++i) {
        const auto decision = dedup.check(key, kBase + std::chrono::milliseconds(500 * i));
        EXPECT_TRUE(decision.suppress);
        EXPECT_EQ(decision.repeats, 0u);
    }

    const auto redelivery = dedup.check(key, kBase + lfs::ErrorDedup::kWindow);
    EXPECT_FALSE(redelivery.suppress);
    EXPECT_EQ(redelivery.repeats, static_cast<std::uint32_t>(kSuppressed));

    // Counter reset: the next window boundary re-delivers with zero repeats.
    const auto after_reset = dedup.check(key, kBase + lfs::ErrorDedup::kWindow + lfs::ErrorDedup::kWindow);
    EXPECT_FALSE(after_reset.suppress);
    EXPECT_EQ(after_reset.repeats, 0u);
}

TEST(ErrorDedupTest, WindowDoesNotSlideOnSuppressedRepeats) {
    lfs::ErrorDedup dedup;
    constexpr std::uint64_t key = 0xABCD;

    EXPECT_FALSE(dedup.check(key, kBase).suppress);
    EXPECT_TRUE(dedup.check(key, kBase + std::chrono::milliseconds(4000)).suppress);
    EXPECT_TRUE(dedup.check(key, kBase + std::chrono::milliseconds(4900)).suppress);

    // The P1 sliding bug would push delivered_at forward and suppress forever;
    // the fixed window re-delivers exactly at kBase + kWindow.
    const auto decision = dedup.check(key, kBase + std::chrono::milliseconds(5000));
    EXPECT_FALSE(decision.suppress);
    EXPECT_EQ(decision.repeats, 2u);
}

TEST(ErrorDedupTest, IdleEntriesExpire) {
    lfs::ErrorDedup dedup;
    constexpr std::uint64_t key = 0x55;

    EXPECT_FALSE(dedup.check(key, kBase).suppress);
    EXPECT_TRUE(dedup.check(key, kBase + std::chrono::seconds(2)).suppress); // count=1, window unmoved

    // After kIdleExpiry with no re-delivery the entry is swept, so the next check
    // is a fresh first delivery (repeats 0), not a windowed re-delivery (repeats 1).
    const auto decision = dedup.check(key, kBase + lfs::ErrorDedup::kIdleExpiry + std::chrono::seconds(1));
    EXPECT_FALSE(decision.suppress);
    EXPECT_EQ(decision.repeats, 0u);
}

TEST(ToastStackTest, PushCollapsesSameFingerprint) {
    lfs::vis::gui::ToastStack stack;
    stack.push(lfs::vis::gui::ToastRequest{.title = "A", .message = "m", .fingerprint = 42}, kBase);
    const bool changed = stack.push(
        lfs::vis::gui::ToastRequest{.title = "A", .message = "m", .fingerprint = 42},
        kBase + std::chrono::seconds(1));

    EXPECT_TRUE(changed);
    ASSERT_EQ(stack.entries.size(), 1u);
    EXPECT_EQ(stack.entries[0].count, 2u);
    EXPECT_EQ(stack.entries[0].shown_at, kBase + std::chrono::seconds(1)); // timer reset
}

TEST(ToastStackTest, OverflowEvictsOldest) {
    lfs::vis::gui::ToastStack stack;
    for (int i = 0; i < 5; ++i) {
        // fingerprint 0 => no collapse, each pushes a fresh entry.
        stack.push(lfs::vis::gui::ToastRequest{.title = std::to_string(i)}, kBase);
    }

    ASSERT_EQ(stack.entries.size(), lfs::vis::gui::ToastStack::kMaxVisible);
    EXPECT_EQ(stack.entries.front().request.title, "1"); // "0" evicted
    EXPECT_EQ(stack.entries.back().request.title, "4");
}

TEST(ToastStackTest, ExpireRemovesAfterDuration) {
    lfs::vis::gui::ToastStack stack;
    stack.push(lfs::vis::gui::ToastRequest{.title = "x"}, kBase);

    EXPECT_FALSE(stack.expire(kBase + std::chrono::milliseconds(3000)));
    EXPECT_EQ(stack.entries.size(), 1u);
    EXPECT_TRUE(stack.expire(kBase + lfs::vis::gui::ToastStack::kDuration));
    EXPECT_TRUE(stack.entries.empty());
}

TEST(ToastStackTest, AlphaFadesInFinalWindow) {
    lfs::vis::gui::ToastStack::Entry entry;
    entry.shown_at = kBase;

    EXPECT_FLOAT_EQ(lfs::vis::gui::ToastStack::alpha(entry, kBase), 1.0f);
    const auto near_end =
        kBase + lfs::vis::gui::ToastStack::kDuration - std::chrono::milliseconds(250);
    EXPECT_NEAR(lfs::vis::gui::ToastStack::alpha(entry, near_end), 0.5f, 1e-3f);
    EXPECT_FLOAT_EQ(
        lfs::vis::gui::ToastStack::alpha(entry, kBase + lfs::vis::gui::ToastStack::kDuration), 0.0f);
}

TEST(StatusMessageStateTest, PostThenSnapshotVisible) {
    lfs::vis::gui::StatusMessageState state;
    const auto before = std::chrono::steady_clock::now();
    state.post("hello", lfs::vis::gui::ErrorNoticeLevel::Warning);

    // Snapshot at/just-before the internal post time => full alpha, visible.
    const auto snap = state.snapshot(before);
    EXPECT_TRUE(snap.visible);
    EXPECT_EQ(snap.text, "hello");
    EXPECT_EQ(snap.level, lfs::vis::gui::ErrorNoticeLevel::Warning);
    EXPECT_FLOAT_EQ(snap.alpha, 1.0f);
}

TEST(StatusMessageStateTest, ExpiresAndFades) {
    lfs::vis::gui::StatusMessageState state;
    const auto before = std::chrono::steady_clock::now();
    state.post("bye", lfs::vis::gui::ErrorNoticeLevel::Error);

    // Inside the final fade window: alpha strictly between 0 and 1.
    const auto fading =
        before + lfs::vis::gui::StatusMessageState::kDuration - std::chrono::milliseconds(200);
    const auto fade_snap = state.snapshot(fading);
    EXPECT_TRUE(fade_snap.visible);
    EXPECT_GT(fade_snap.alpha, 0.0f);
    EXPECT_LT(fade_snap.alpha, 1.0f);

    // Past the duration: self-cleared.
    const auto expired =
        before + lfs::vis::gui::StatusMessageState::kDuration + std::chrono::seconds(1);
    EXPECT_FALSE(state.snapshot(expired).visible);
}

namespace {

    lfs::ErrorNotification consumerNotification(lfs::Error error, const lfs::ErrorSurface surface) {
        return makeNotification(std::move(error), lfs::OperationId::generate(), surface);
    }

} // namespace

TEST(GuiErrorConsumerTest, ToastSurfaceRoutesToToastSink) {
    std::optional<lfs::vis::gui::ToastRequest> toast;
    int modal_calls = 0;
    lfs::vis::gui::GuiErrorConsumer consumer(lfs::vis::gui::GuiErrorConsumer::Sinks{
        .modal = [&](lfs::core::ModalRequest) { ++modal_calls; },
        .toast = [&](lfs::vis::gui::ToastRequest r) { toast = std::move(r); },
        .status = {},
    });

    const auto n = consumerNotification(
        makeError(lfs::ErrorCode::FailedPrecondition, lfs::ErrorDomain::CUDA, "old driver"),
        lfs::ErrorSurface::Toast);
    consumer.on_error(n, lfs::ErrorDeliveryInfo{});

    ASSERT_TRUE(toast.has_value());
    EXPECT_FALSE(toast->title.empty());
    EXPECT_EQ(toast->message, "old driver");
    EXPECT_EQ(toast->fingerprint, lfs::core::error_fingerprint(n.error));
    EXPECT_EQ(modal_calls, 0);
}

TEST(GuiErrorConsumerTest, ToastFallsBackToModalWithoutSink) {
    int modal_calls = 0;
    lfs::vis::gui::GuiErrorConsumer consumer(lfs::vis::gui::GuiErrorConsumer::Sinks{
        .modal = [&](lfs::core::ModalRequest) { ++modal_calls; },
        .toast = {},
        .status = {},
    });

    const auto n = consumerNotification(
        makeError(lfs::ErrorCode::FailedPrecondition, lfs::ErrorDomain::CUDA, "old driver"),
        lfs::ErrorSurface::Toast);
    consumer.on_error(n, lfs::ErrorDeliveryInfo{});

    EXPECT_EQ(modal_calls, 1);
}

TEST(GuiErrorConsumerTest, StatusOnlyRoutesFirstLineToStatusSink) {
    std::optional<std::string> status_text;
    int modal_calls = 0;
    lfs::vis::gui::GuiErrorConsumer consumer(lfs::vis::gui::GuiErrorConsumer::Sinks{
        .modal = [&](lfs::core::ModalRequest) { ++modal_calls; },
        .toast = {},
        .status = [&](std::string t, lfs::vis::gui::ErrorNoticeLevel) { status_text = std::move(t); },
    });

    const auto n = consumerNotification(
        makeError(lfs::ErrorCode::Cancelled, lfs::ErrorDomain::IO, "Export cancelled\nsecond line"),
        lfs::ErrorSurface::StatusOnly);
    consumer.on_error(n, lfs::ErrorDeliveryInfo{});

    ASSERT_TRUE(status_text.has_value());
    EXPECT_EQ(*status_text, "Export cancelled"); // truncated at the first newline
    EXPECT_EQ(modal_calls, 0);
}

TEST(GuiErrorConsumerTest, PanelBuildsDetailsModal) {
    std::optional<lfs::core::ModalRequest> modal;
    lfs::vis::gui::GuiErrorConsumer consumer(lfs::vis::gui::GuiErrorConsumer::Sinks{
        .modal = [&](lfs::core::ModalRequest r) { modal = std::move(r); },
        .toast = {},
        .status = {},
    });

    const auto n = consumerNotification(
        makeError(lfs::ErrorCode::Internal, lfs::ErrorDomain::Training, "boom"),
        lfs::ErrorSurface::Panel);
    consumer.on_error(n, lfs::ErrorDeliveryInfo{});

    ASSERT_TRUE(modal.has_value());
    EXPECT_EQ(modal->width_dp, 640);
    EXPECT_NE(modal->body_rml.find("details-block"), std::string::npos);
    EXPECT_NE(modal->body_rml.find("boom"), std::string::npos);
}

TEST(GuiErrorConsumerTest, ModalGainsDetailsButton) {
    std::optional<lfs::core::ModalRequest> modal;
    lfs::vis::gui::GuiErrorConsumer consumer(lfs::vis::gui::GuiErrorConsumer::Sinks{
        .modal = [&](lfs::core::ModalRequest r) { modal = std::move(r); },
        .toast = {},
        .status = {},
    });

    const auto n = consumerNotification(
        makeError(lfs::ErrorCode::Internal, lfs::ErrorDomain::IO, "x"), lfs::ErrorSurface::Modal);
    consumer.on_error(n, lfs::ErrorDeliveryInfo{});

    ASSERT_TRUE(modal.has_value());
    ASSERT_EQ(modal->buttons.size(), 2u); // OK + Details
    EXPECT_EQ(modal->buttons.back().style, "secondary");
}

TEST(GuiErrorConsumerTest, DetailsButtonEnqueuesDetailsModal) {
    std::vector<lfs::core::ModalRequest> modals;
    modals.reserve(2);
    lfs::vis::gui::GuiErrorConsumer consumer(lfs::vis::gui::GuiErrorConsumer::Sinks{
        .modal = [&](lfs::core::ModalRequest r) { modals.push_back(std::move(r)); },
        .toast = {},
        .status = {},
    });

    const auto n = consumerNotification(
        makeError(lfs::ErrorCode::Internal, lfs::ErrorDomain::Training, "boom"),
        lfs::ErrorSurface::Modal);
    consumer.on_error(n, lfs::ErrorDeliveryInfo{});
    ASSERT_EQ(modals.size(), 1u);
    ASSERT_TRUE(modals[0].on_result);

    const std::string details_label = modals[0].buttons.back().label;
    modals[0].on_result(lfs::core::ModalResult{.button_label = details_label});

    ASSERT_EQ(modals.size(), 2u);
    EXPECT_EQ(modals[1].width_dp, 640);
    EXPECT_NE(modals[1].body_rml.find("details-block"), std::string::npos);
}

TEST(GuiErrorConsumerTest, ActionInvokeReceivesFreshOperationId) {
    std::optional<lfs::core::ModalRequest> modal;
    lfs::vis::gui::GuiErrorConsumer consumer(lfs::vis::gui::GuiErrorConsumer::Sinks{
        .modal = [&](lfs::core::ModalRequest r) { modal = std::move(r); },
        .toast = {},
        .status = {},
    });

    std::vector<lfs::OperationId> seen;
    auto n = consumerNotification(
        makeError(lfs::ErrorCode::Internal, lfs::ErrorDomain::IO, "x"), lfs::ErrorSurface::Modal);
    n.actions.push_back(lfs::ErrorAction{.kind = lfs::ErrorActionKind::OpenLog,
                                         .label = "Act",
                                         .on_invoke = [&](lfs::OperationId id) { seen.push_back(id); }});
    consumer.on_error(n, lfs::ErrorDeliveryInfo{});

    ASSERT_TRUE(modal.has_value());
    ASSERT_TRUE(modal->on_result);
    modal->on_result(lfs::core::ModalResult{.button_label = "Act"});
    modal->on_result(lfs::core::ModalResult{.button_label = "Act"});

    ASSERT_EQ(seen.size(), 2u);
    EXPECT_TRUE(seen[0].has_value());
    EXPECT_NE(seen[0].value(), seen[1].value());
    EXPECT_EQ(n.error.code(), lfs::ErrorCode::Internal); // source error untouched
}

TEST(GuiErrorConsumerTest, SuppressedRepeatsRenderInBody) {
    std::optional<lfs::core::ModalRequest> modal;
    lfs::vis::gui::GuiErrorConsumer consumer(lfs::vis::gui::GuiErrorConsumer::Sinks{
        .modal = [&](lfs::core::ModalRequest r) { modal = std::move(r); },
        .toast = {},
        .status = {},
    });

    const auto n = consumerNotification(
        makeError(lfs::ErrorCode::Internal, lfs::ErrorDomain::IO, "x"), lfs::ErrorSurface::Modal);
    consumer.on_error(n, lfs::ErrorDeliveryInfo{.suppressed_repeats = 12});

    ASSERT_TRUE(modal.has_value());
    EXPECT_NE(modal->body_rml.find("x12"), std::string::npos);
}

// P3 §4.1: the new titleKeyFor Vulkan/Rendering mapping, verified through the
// public consumer boundary. LOC headless returns the key, so assert the key.
TEST(GuiErrorConsumerTest, VulkanDeviceLostMapsToDeviceLostTitle) {
    std::optional<lfs::core::ModalRequest> modal;
    lfs::vis::gui::GuiErrorConsumer consumer(lfs::vis::gui::GuiErrorConsumer::Sinks{
        .modal = [&](lfs::core::ModalRequest r) { modal = std::move(r); },
        .toast = {},
        .status = {}});
    consumer.on_error(consumerNotification(
                          makeError(lfs::ErrorCode::DeviceLost, lfs::ErrorDomain::Vulkan, "lost"),
                          lfs::ErrorSurface::Modal),
                      lfs::ErrorDeliveryInfo{});
    ASSERT_TRUE(modal.has_value());
    EXPECT_EQ(modal->title, std::string(lichtfeld::Strings::ErrorModal::RENDERER_DEVICE_LOST));
}

TEST(GuiErrorConsumerTest, VulkanDeadlineMapsToStalledTitle) {
    std::optional<lfs::core::ModalRequest> modal;
    lfs::vis::gui::GuiErrorConsumer consumer(lfs::vis::gui::GuiErrorConsumer::Sinks{
        .modal = [&](lfs::core::ModalRequest r) { modal = std::move(r); },
        .toast = {},
        .status = {}});
    consumer.on_error(consumerNotification(makeError(lfs::ErrorCode::DeadlineExceeded,
                                                     lfs::ErrorDomain::Vulkan, "stalled"),
                                           lfs::ErrorSurface::Modal),
                      lfs::ErrorDeliveryInfo{});
    ASSERT_TRUE(modal.has_value());
    EXPECT_EQ(modal->title, std::string(lichtfeld::Strings::ErrorModal::RENDERER_STALLED));
}

TEST(GuiErrorConsumerTest, RenderFrameOomMapsToGpuMemoryTitle) {
    std::optional<lfs::core::ModalRequest> modal;
    lfs::vis::gui::GuiErrorConsumer consumer(lfs::vis::gui::GuiErrorConsumer::Sinks{
        .modal = [&](lfs::core::ModalRequest r) { modal = std::move(r); },
        .toast = {},
        .status = {}});
    consumer.on_error(
        consumerNotification(makeError(lfs::ErrorCode::ResourceExhausted, lfs::ErrorDomain::Rendering,
                                       "vram", lfs::vis::gui::error_op::kRenderFrame),
                             lfs::ErrorSurface::Modal),
        lfs::ErrorDeliveryInfo{});
    ASSERT_TRUE(modal.has_value());
    EXPECT_EQ(modal->title, std::string(lichtfeld::Strings::ErrorModal::OUT_OF_GPU_MEMORY));
}

using lfs::vis::FrameFault;
using lfs::vis::FrameStateMachine;
using lfs::vis::RendererTerminalState;
using FsmState = lfs::vis::FrameStateMachine::State;

TEST(RendererTerminalStateTest, MappingDominance) {
    using lfs::vis::renderer_terminal_state;
    EXPECT_EQ(renderer_terminal_state(true, false), RendererTerminalState::DeviceLost);
    EXPECT_EQ(renderer_terminal_state(true, true), RendererTerminalState::DeviceLost);
    EXPECT_EQ(renderer_terminal_state(false, true), RendererTerminalState::Quarantined);
    EXPECT_EQ(renderer_terminal_state(false, false), RendererTerminalState::Running);
}

TEST(FrameStateMachineTest, OomWithinBudgetReclaimsAndToastsOnce) {
    FrameStateMachine m;
    for (int i = 1; i <= 8; ++i) {
        const auto fx = m.on_fault(FrameFault::OomPressure);
        EXPECT_TRUE(fx.run_reclaim_episode) << "fault " << i;
        EXPECT_EQ(fx.publish_pressure_toast, i == 1) << "fault " << i;
        EXPECT_FALSE(fx.publish_oom_modal) << "fault " << i;
    }
    EXPECT_EQ(m.state(), FsmState::PressureRetry);
    EXPECT_FALSE(m.scene_render_suspended());
}

TEST(FrameStateMachineTest, OomBudgetExhaustionSuspendsOnce) {
    FrameStateMachine m;
    for (int i = 0; i < 8; ++i)
        (void)m.on_fault(FrameFault::OomPressure);

    const auto ninth = m.on_fault(FrameFault::OomPressure);
    EXPECT_TRUE(ninth.publish_oom_modal);
    EXPECT_FALSE(ninth.run_reclaim_episode);
    EXPECT_TRUE(m.scene_render_suspended());

    const auto tenth = m.on_fault(FrameFault::OomPressure);
    EXPECT_FALSE(tenth.publish_oom_modal);
    EXPECT_FALSE(tenth.run_reclaim_episode);
    EXPECT_FALSE(tenth.publish_pressure_toast);
}

TEST(FrameStateMachineTest, SuccessResetsBothBudgets) {
    {
        FrameStateMachine m;
        for (int i = 0; i < 7; ++i)
            (void)m.on_fault(FrameFault::OomPressure);
        m.on_frame_success();
        for (int i = 0; i < 8; ++i)
            EXPECT_FALSE(m.on_fault(FrameFault::OomPressure).publish_oom_modal) << "fault " << i;
    }
    {
        FrameStateMachine m;
        (void)m.on_fault(FrameFault::RendererInternal);
        (void)m.on_fault(FrameFault::RendererInternal);
        m.on_frame_success();
        EXPECT_FALSE(m.on_fault(FrameFault::RendererInternal).publish_internal_modal);
        EXPECT_FALSE(m.on_fault(FrameFault::RendererInternal).publish_internal_modal);
    }
}

TEST(FrameStateMachineTest, InternalTransientThenTerminal) {
    FrameStateMachine m;
    EXPECT_FALSE(m.on_fault(FrameFault::RendererInternal).publish_internal_modal);
    EXPECT_FALSE(m.on_fault(FrameFault::RendererInternal).publish_internal_modal);
    EXPECT_FALSE(m.scene_render_suspended());

    const auto third = m.on_fault(FrameFault::RendererInternal);
    EXPECT_TRUE(third.publish_internal_modal);
    EXPECT_TRUE(m.scene_render_suspended());

    EXPECT_FALSE(m.on_fault(FrameFault::RendererInternal).publish_internal_modal);
}

TEST(FrameStateMachineTest, MixedCountersIndependent) {
    FrameStateMachine m;
    EXPECT_FALSE(m.on_fault(FrameFault::OomPressure).publish_oom_modal);           // oom 1
    EXPECT_FALSE(m.on_fault(FrameFault::RendererInternal).publish_internal_modal); // int 1
    EXPECT_FALSE(m.on_fault(FrameFault::OomPressure).publish_oom_modal);           // oom 2
    EXPECT_FALSE(m.on_fault(FrameFault::RendererInternal).publish_internal_modal); // int 2
    EXPECT_FALSE(m.on_fault(FrameFault::OomPressure).publish_oom_modal);           // oom 3 (< 8)

    const auto crosses = m.on_fault(FrameFault::RendererInternal); // int 3 -> escalates first
    EXPECT_TRUE(crosses.publish_internal_modal);
    EXPECT_TRUE(m.scene_render_suspended());
}

TEST(FrameStateMachineTest, RetryRearmsExactlyOnce) {
    FrameStateMachine m;
    for (int i = 0; i < 3; ++i)
        (void)m.on_fault(FrameFault::RendererInternal);
    ASSERT_TRUE(m.scene_render_suspended());

    m.on_retry_action();
    EXPECT_EQ(m.state(), FsmState::Healthy);
    EXPECT_FALSE(m.scene_render_suspended());

    EXPECT_FALSE(m.on_fault(FrameFault::RendererInternal).publish_internal_modal);
    EXPECT_FALSE(m.on_fault(FrameFault::RendererInternal).publish_internal_modal);
    EXPECT_TRUE(m.on_fault(FrameFault::RendererInternal).publish_internal_modal);
}

TEST(FrameStateMachineTest, DismissDoesNotRearm) {
    FrameStateMachine m;
    for (int i = 0; i < 3; ++i)
        (void)m.on_fault(FrameFault::RendererInternal);
    ASSERT_TRUE(m.scene_render_suspended());

    for (int i = 0; i < 5; ++i) {
        const auto fx = m.on_fault(FrameFault::RendererInternal);
        EXPECT_FALSE(fx.publish_internal_modal);
        EXPECT_FALSE(fx.publish_oom_modal);
    }
}

TEST(FrameStateMachineTest, StopRendererIsSticky) {
    FrameStateMachine m;
    for (int i = 0; i < 3; ++i)
        (void)m.on_fault(FrameFault::RendererInternal);
    ASSERT_TRUE(m.scene_render_suspended());

    m.on_stop_renderer_action();
    EXPECT_TRUE(m.scene_render_suspended());

    EXPECT_FALSE(m.on_fault(FrameFault::RendererInternal).publish_internal_modal);
    m.on_retry_action(); // no-op: renderer stopped for the session, no modal to press
    EXPECT_EQ(m.state(), FsmState::SceneSuspended);
    EXPECT_TRUE(m.scene_render_suspended());
}

TEST(FrameStateMachineTest, DeviceLostFaultIsImmediatelyTerminal) {
    {
        FrameStateMachine m;
        const auto fx = m.on_fault(FrameFault::DeviceLost);
        EXPECT_TRUE(fx.publish_renderer_dead_modal);
        EXPECT_EQ(fx.dead_cause, RendererTerminalState::DeviceLost);
        EXPECT_EQ(m.state(), FsmState::RendererDead);

        const auto after = m.on_fault(FrameFault::OomPressure);
        EXPECT_FALSE(after.publish_renderer_dead_modal);
        EXPECT_FALSE(after.run_reclaim_episode);
    }
    {
        FrameStateMachine m;
        (void)m.on_fault(FrameFault::OomPressure); // PressureRetry
        const auto fx = m.on_fault(FrameFault::DeviceLost);
        EXPECT_TRUE(fx.publish_renderer_dead_modal);
        EXPECT_EQ(fx.dead_cause, RendererTerminalState::DeviceLost);
        EXPECT_EQ(m.state(), FsmState::RendererDead);
    }
}

TEST(FrameStateMachineTest, PollQuarantinePublishesOnceWithCause) {
    FrameStateMachine m;
    const auto fx = m.on_renderer_terminal(RendererTerminalState::Quarantined);
    EXPECT_TRUE(fx.publish_renderer_dead_modal);
    EXPECT_EQ(fx.dead_cause, RendererTerminalState::Quarantined);
    EXPECT_EQ(m.state(), FsmState::RendererDead);

    EXPECT_FALSE(m.on_renderer_terminal(RendererTerminalState::Quarantined).publish_renderer_dead_modal);
    EXPECT_FALSE(m.on_renderer_terminal(RendererTerminalState::Running).publish_renderer_dead_modal);

    FrameStateMachine healthy;
    const auto running = healthy.on_renderer_terminal(RendererTerminalState::Running);
    EXPECT_FALSE(running.publish_renderer_dead_modal);
    EXPECT_EQ(healthy.state(), FsmState::Healthy);
}

TEST(FrameStateMachineTest, DeadDominatesSuspended) {
    FrameStateMachine m;
    for (int i = 0; i < 3; ++i)
        (void)m.on_fault(FrameFault::RendererInternal); // internal modal fires + disarms
    ASSERT_TRUE(m.scene_render_suspended());

    const auto fx = m.on_renderer_terminal(RendererTerminalState::DeviceLost);
    EXPECT_TRUE(fx.publish_renderer_dead_modal); // own one-shot, still armed
    EXPECT_EQ(fx.dead_cause, RendererTerminalState::DeviceLost);
    EXPECT_EQ(m.state(), FsmState::RendererDead);
}

TEST(FrameStateMachineTest, CustomLimits) {
    FrameStateMachine m(FrameStateMachine::Limits{.oom_retry_budget = 2, .internal_retry_budget = 1});
    EXPECT_TRUE(m.on_fault(FrameFault::OomPressure).run_reclaim_episode);
    EXPECT_TRUE(m.on_fault(FrameFault::OomPressure).run_reclaim_episode);
    const auto third = m.on_fault(FrameFault::OomPressure);
    EXPECT_TRUE(third.publish_oom_modal);
    EXPECT_TRUE(m.scene_render_suspended());

    FrameStateMachine n(FrameStateMachine::Limits{.oom_retry_budget = 2, .internal_retry_budget = 1});
    const auto first_internal = n.on_fault(FrameFault::RendererInternal);
    EXPECT_TRUE(first_internal.publish_internal_modal);
    EXPECT_TRUE(n.scene_render_suspended());
}

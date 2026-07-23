/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/error.hpp"
#include "core/error_reporter.hpp"
#include "core/failure_report.hpp"
#include "core/logger.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

using namespace lfs;

namespace {

    class LogHandlerGuard {
    public:
        explicit LogHandlerGuard(core::LogHandler handler)
            : token_(core::Logger::get().add_log_handler(std::move(handler))) {}

        ~LogHandlerGuard() {
            core::Logger::get().remove_log_handler(token_);
        }

        LogHandlerGuard(const LogHandlerGuard&) = delete;
        LogHandlerGuard& operator=(const LogHandlerGuard&) = delete;

    private:
        core::LogHandlerToken token_;
    };

    // Restores the process-global Logger singleton to its normal init()
    // state after a test forces it back to a pre-init() state.
    class LoggerInitGuard {
    public:
        LoggerInitGuard() = default;
        ~LoggerInitGuard() { core::Logger::get().init(); }

        LoggerInitGuard(const LoggerInitGuard&) = delete;
        LoggerInitGuard& operator=(const LoggerInitGuard&) = delete;
    };

    // Restores Logger's global display level after a test changes it.
    class LogLevelGuard {
    public:
        LogLevelGuard() : previous_(core::Logger::get().level()) {}
        ~LogLevelGuard() { core::Logger::get().set_level(previous_); }

        LogLevelGuard(const LogLevelGuard&) = delete;
        LogLevelGuard& operator=(const LogLevelGuard&) = delete;

    private:
        core::LogLevel previous_;
    };

    std::string next_marker(const std::string& label) {
        static std::atomic<uint64_t> counter{0};
        return "error_reporter_test_marker_" + label + "_" + std::to_string(counter.fetch_add(1));
    }

    // Single shared detection site (matches test_error_core.cpp's own
    // make_test_error convention): LFS_SOURCE_SITE_CURRENT() captures where
    // this helper is textually written, not where it is called from.
    Error make_reporter_test_error(const ErrorCode code, const ErrorDomain domain,
                                   const std::string_view detail,
                                   const std::optional<NativeError>& native = std::nullopt) {
        return make_error(ErrorInit{
            .code = code,
            .domain = domain,
            .severity = Severity::Error,
            .retryability = Retryability::Retryable,
            .operation_id = OperationId::generate(),
            .user_message = "error reporter test failure",
            .detail = std::string(detail),
            .detection = LFS_SOURCE_SITE_CURRENT(),
            .fields = SmallFields{},
            .native = native,
        });
    }

    std::atomic<bool> g_section_provider_should_throw{false};

    void maybe_throwing_section_provider(std::ostream&, core::FailureReportSectionPosition,
                                         const core::FailureReport&) {
        if (g_section_provider_should_throw.load(std::memory_order_acquire)) {
            throw std::runtime_error("fault-injected section provider failure");
        }
    }

} // namespace

// ---------------------------------------------------------------------------
// Stack-capture policy matrix (Section 5.4)
// ---------------------------------------------------------------------------

TEST(ErrorReporterTest, StackCapturePolicyMatchesFrozenMatrix) {
    EXPECT_TRUE(core::should_capture_stack_for_testing(ErrorCode::ContractViolation));
    EXPECT_TRUE(core::should_capture_stack_for_testing(ErrorCode::Internal));
    EXPECT_TRUE(core::should_capture_stack_for_testing(ErrorCode::DeviceLost));
    EXPECT_FALSE(core::should_capture_stack_for_testing(ErrorCode::InvalidArgument));
    EXPECT_FALSE(core::should_capture_stack_for_testing(ErrorCode::ResourceExhausted));
    EXPECT_FALSE(core::should_capture_stack_for_testing(ErrorCode::PermissionDenied));

    const LogLevelGuard level_guard;
    core::Logger::get().set_level(core::LogLevel::Info);
    EXPECT_FALSE(core::should_capture_stack_for_testing(ErrorCode::NotFound));
    EXPECT_FALSE(core::should_capture_stack_for_testing(ErrorCode::Cancelled));

    core::Logger::get().set_level(core::LogLevel::Debug);
    EXPECT_TRUE(core::should_capture_stack_for_testing(ErrorCode::NotFound));
    EXPECT_TRUE(core::should_capture_stack_for_testing(ErrorCode::Cancelled));
}

// ---------------------------------------------------------------------------
// Structured record: thread id / operation id / domain / code / native /
// retry action all land in the memory-sink snapshot for a single report().
// ---------------------------------------------------------------------------

TEST(ErrorReporterTest, MemorySinkRecordCarriesStructuredFields) {
    core::reset_failure_report_dedup_for_testing();
    const LogLevelGuard level_guard;
    core::Logger::get().set_level(core::LogLevel::Info);

    const std::string marker = next_marker("structured_fields");
    const NativeError native{
        .domain = ErrorDomain::CUDA,
        .code = 2,
        .name = "cudaErrorMemoryAllocation"};
    Error error = make_reporter_test_error(ErrorCode::ResourceExhausted, ErrorDomain::CUDA, marker, native);
    error = std::move(error).with_context("allocate splat buffer", LFS_SOURCE_SITE_CURRENT());

    core::ErrorReporter::get().report(error, core::ReportChannel::OwnerLog);

    const auto entries = core::Logger::get().buffered_logs();
    const auto found = std::find_if(entries.begin(), entries.end(), [&](const core::LogEntrySnapshot& e) {
        return e.message.find(marker) != std::string::npos;
    });
    ASSERT_NE(found, entries.end());
    const std::string& message = found->message;

    EXPECT_NE(message.find("thread_id=0x"), std::string::npos);
    EXPECT_NE(message.find("channel=OwnerLog"), std::string::npos);
    EXPECT_NE(message.find("ResourceExhausted"), std::string::npos);
    EXPECT_NE(message.find("CUDA"), std::string::npos);
    EXPECT_NE(message.find("cudaErrorMemoryAllocation"), std::string::npos);
    EXPECT_NE(message.find("retryability="), std::string::npos);
    EXPECT_NE(message.find("operation_id="), std::string::npos);
    EXPECT_NE(message.find("allocate splat buffer"), std::string::npos);
}

// ---------------------------------------------------------------------------
// Dedup count/time behavior: fingerprint extends the existing family/code/
// site mechanism, including its cap + periodic full-repeat semantics.
// ---------------------------------------------------------------------------

TEST(ErrorReporterTest, DedupEmitsFullThenRepeatsThenFullAgainAtCountThreshold) {
    core::reset_failure_report_dedup_for_testing();
    const LogLevelGuard level_guard;
    core::Logger::get().set_level(core::LogLevel::Info);

    const std::string marker = next_marker("dedup");
    const Error error = make_reporter_test_error(ErrorCode::Unavailable, ErrorDomain::Rendering, marker);

    std::vector<std::string> messages;
    LogHandlerGuard guard([&](core::LogLevel, const core::SourceSite&, const std::string_view message) {
        messages.emplace_back(message);
    });

    for (int i = 0; i < 100; ++i) {
        core::ErrorReporter::get().report(error, core::ReportChannel::OwnerLog);
    }

    ASSERT_EQ(messages.size(), 100u);
    std::size_t full_reports = 0;
    std::size_t repeat_notices = 0;
    for (const auto& msg : messages) {
        if (msg.find(marker) != std::string::npos) {
            ++full_reports;
        } else if (msg.find("repeated x") != std::string::npos) {
            ++repeat_notices;
        }
    }
    // decide_failure_report: call 1 is always full; calls 2-99 are repeats;
    // call 100 crosses the 100-count full-repeat threshold.
    EXPECT_EQ(full_reports, 2u);
    EXPECT_EQ(repeat_notices, 98u);
}

// ---------------------------------------------------------------------------
// Disabled-log-level fast path: nothing observable happens.
// ---------------------------------------------------------------------------

TEST(ErrorReporterTest, DisabledLogLevelFastPathProducesNoHandlerInvocationOrStderr) {
    core::reset_failure_report_dedup_for_testing();
    const LogLevelGuard level_guard;
    core::Logger::get().set_level(core::LogLevel::Critical);

    std::atomic<int> handler_invocations{0};
    LogHandlerGuard guard([&](core::LogLevel, const core::SourceSite&, std::string_view) {
        ++handler_invocations;
    });

    const std::string marker = next_marker("disabled_level");
    const Error error = make_reporter_test_error(ErrorCode::InvalidArgument, ErrorDomain::IO, marker);

    testing::internal::CaptureStderr();
    core::ErrorReporter::get().report(error, core::ReportChannel::OwnerLog);
    const std::string captured = testing::internal::GetCapturedStderr();

    EXPECT_EQ(handler_invocations.load(), 0);
    EXPECT_TRUE(captured.empty());
}

// ---------------------------------------------------------------------------
// ProcessBoundary channel guarantees stderr visibility even when the
// OwnerLog path would otherwise be suppressed by level gating.
// ---------------------------------------------------------------------------

TEST(ErrorReporterTest, ProcessBoundaryChannelGuaranteesStderrEvenWhenLevelSuppressesOwnerLog) {
    core::reset_failure_report_dedup_for_testing();
    const LogLevelGuard level_guard;
    core::Logger::get().set_level(core::LogLevel::Critical);

    const std::string marker = next_marker("process_boundary");
    const Error error = make_reporter_test_error(ErrorCode::Internal, ErrorDomain::App, marker);

    testing::internal::CaptureStderr();
    core::ErrorReporter::get().report(error, core::ReportChannel::ProcessBoundary);
    const std::string captured = testing::internal::GetCapturedStderr();

    EXPECT_NE(captured.find(marker), std::string::npos);
    EXPECT_NE(captured.find("process boundary"), std::string::npos);
}

// ---------------------------------------------------------------------------
// Pre-init fallback: Logger not yet initialized falls back to stderr.
// ---------------------------------------------------------------------------

TEST(ErrorReporterTest, PreInitLoggerFallsBackToStderr) {
    const LoggerInitGuard restore_guard;
    core::Logger::get().reset_for_testing();
    ASSERT_FALSE(core::Logger::get().is_ready());

    const std::string marker = next_marker("preinit");
    const Error error = make_reporter_test_error(ErrorCode::Unavailable, ErrorDomain::Core, marker);

    testing::internal::CaptureStderr();
    core::ErrorReporter::get().report(error, core::ReportChannel::OwnerLog);
    const std::string captured = testing::internal::GetCapturedStderr();

    EXPECT_NE(captured.find(marker), std::string::npos);
    EXPECT_NE(captured.find("logger not initialized"), std::string::npos);
}

// ---------------------------------------------------------------------------
// A throwing FailureReport section provider (a "sink" in the reporting
// pipeline) falls back to stderr instead of propagating or crashing.
// ---------------------------------------------------------------------------

TEST(ErrorReporterTest, ThrowingFailureReportSectionProviderFallsBackToStderr) {
    core::reset_failure_report_dedup_for_testing();
    const LogLevelGuard level_guard;
    core::Logger::get().set_level(core::LogLevel::Info);

    core::register_failure_report_section_provider(to_string(ErrorDomain::Sequencer),
                                                   maybe_throwing_section_provider);
    g_section_provider_should_throw.store(true, std::memory_order_release);

    const std::string marker = next_marker("throwing_provider");
    const Error error = make_reporter_test_error(ErrorCode::Internal, ErrorDomain::Sequencer, marker);

    testing::internal::CaptureStderr();
    EXPECT_NO_THROW(core::ErrorReporter::get().report(error, core::ReportChannel::OwnerLog));
    const std::string captured = testing::internal::GetCapturedStderr();

    g_section_provider_should_throw.store(false, std::memory_order_release);

    EXPECT_NE(captured.find(marker), std::string::npos);
    EXPECT_NE(captured.find("reporting pipeline threw"), std::string::npos);
}

// ---------------------------------------------------------------------------
// Recursion guard: a log handler that itself calls report() must not
// recurse into the full pipeline; it takes the fallback path instead.
// ---------------------------------------------------------------------------

TEST(ErrorReporterTest, RecursiveReportTakesFallbackPathWithoutDeadlock) {
    core::reset_failure_report_dedup_for_testing();
    const LogLevelGuard level_guard;
    core::Logger::get().set_level(core::LogLevel::Info);

    const std::string outer_marker = next_marker("recursion_outer");
    const std::string inner_marker = next_marker("recursion_inner");
    const Error inner_error =
        make_reporter_test_error(ErrorCode::Internal, ErrorDomain::Core, inner_marker);

    std::atomic<int> handler_invocations{0};
    LogHandlerGuard guard([&](core::LogLevel, const core::SourceSite&, const std::string_view message) {
        if (message.find(outer_marker) != std::string_view::npos) {
            ++handler_invocations;
            core::ErrorReporter::get().report(inner_error, core::ReportChannel::OwnerLog);
        }
    });

    const Error outer_error =
        make_reporter_test_error(ErrorCode::Internal, ErrorDomain::Core, outer_marker);

    testing::internal::CaptureStderr();
    core::ErrorReporter::get().report(outer_error, core::ReportChannel::OwnerLog);
    const std::string captured = testing::internal::GetCapturedStderr();

    EXPECT_EQ(handler_invocations.load(), 1);
    EXPECT_NE(captured.find("recursive report suppressed"), std::string::npos);
    EXPECT_NE(captured.find(inner_marker), std::string::npos);

    // The inner error never reached Logger's memory sink: report() detected
    // re-entrancy and short-circuited to the stderr fallback.
    const auto entries = core::Logger::get().buffered_logs();
    const bool inner_reached_memory_sink =
        std::any_of(entries.begin(), entries.end(), [&](const core::LogEntrySnapshot& e) {
            return e.message.find(inner_marker) != std::string::npos;
        });
    EXPECT_FALSE(inner_reached_memory_sink);
}

// ---------------------------------------------------------------------------
// Logger handler containment: a throwing handler is disabled without
// affecting the caller or later handlers.
// ---------------------------------------------------------------------------

TEST(ErrorReporterTest, ThrowingLogHandlerIsDisabledWithoutAffectingLaterHandlersOrCaller) {
    core::reset_failure_report_dedup_for_testing();
    const LogLevelGuard level_guard;
    core::Logger::get().set_level(core::LogLevel::Info);

    std::atomic<int> throwing_invocations{0};
    std::atomic<int> later_invocations{0};

    LogHandlerGuard throwing_guard([&](core::LogLevel, const core::SourceSite&, std::string_view) {
        ++throwing_invocations;
        throw std::runtime_error("handler fault injection");
    });
    LogHandlerGuard later_guard([&](core::LogLevel, const core::SourceSite&, std::string_view) {
        ++later_invocations;
    });

    const std::string marker1 = next_marker("containment_1");
    const Error error1 = make_reporter_test_error(ErrorCode::Internal, ErrorDomain::Rendering, marker1);

    testing::internal::CaptureStderr();
    EXPECT_NO_THROW(core::ErrorReporter::get().report(error1, core::ReportChannel::OwnerLog));
    const std::string captured1 = testing::internal::GetCapturedStderr();

    EXPECT_EQ(throwing_invocations.load(), 1);
    EXPECT_EQ(later_invocations.load(), 1);
    EXPECT_NE(captured1.find("disabling it"), std::string::npos);

    // The throwing handler is now disabled: a second report neither invokes
    // it again nor throws, while the still-registered handler keeps working.
    // Reset dedup so this second call is independently a fresh report rather
    // than folding into a "repeated x2" notice for error1's fingerprint.
    core::reset_failure_report_dedup_for_testing();
    const std::string marker2 = next_marker("containment_2");
    const Error error2 = make_reporter_test_error(ErrorCode::Internal, ErrorDomain::Rendering, marker2);
    core::ErrorReporter::get().report(error2, core::ReportChannel::OwnerLog);

    EXPECT_EQ(throwing_invocations.load(), 1);
    EXPECT_EQ(later_invocations.load(), 2);
}

// ---------------------------------------------------------------------------
// Concurrent handler add/remove during reporting must not crash or corrupt
// the handler list.
// ---------------------------------------------------------------------------

TEST(ErrorReporterTest, ConcurrentHandlerAddRemoveDuringReportingDoesNotCrash) {
    core::reset_failure_report_dedup_for_testing();
    const LogLevelGuard level_guard;
    core::Logger::get().set_level(core::LogLevel::Info);

    std::atomic<bool> stop{false};
    std::vector<std::thread> churners;
    churners.reserve(4);
    for (int i = 0; i < 4; ++i) {
        churners.emplace_back([&stop] {
            auto& logger = core::Logger::get();
            while (!stop.load(std::memory_order_relaxed)) {
                const auto token =
                    logger.add_log_handler([](core::LogLevel, const core::SourceSite&, std::string_view) {});
                logger.remove_log_handler(token);
            }
        });
    }

    const Error error =
        make_reporter_test_error(ErrorCode::Unavailable, ErrorDomain::Rendering, next_marker("concurrent"));
    for (int i = 0; i < 500; ++i) {
        core::ErrorReporter::get().report(error, core::ReportChannel::OwnerLog);
    }

    stop.store(true, std::memory_order_relaxed);
    for (auto& churner : churners) {
        churner.join();
    }

    // Sanity check after the churn: the handler list is not corrupted.
    std::atomic<int> final_invocations{0};
    LogHandlerGuard guard([&](core::LogLevel, const core::SourceSite&, std::string_view) {
        ++final_invocations;
    });
    core::ErrorReporter::get().report(error, core::ReportChannel::OwnerLog);

    EXPECT_EQ(final_invocations.load(), 1);
}

// ---------------------------------------------------------------------------
// OOM seed: an immortal, frame-less Error must still report cleanly.
// ---------------------------------------------------------------------------

TEST(ErrorReporterTest, ImmortalOomSeedReportsWithoutCrashing) {
    core::reset_failure_report_dedup_for_testing();
    const LogLevelGuard level_guard;
    core::Logger::get().set_level(core::LogLevel::Info);

    const Error seed = make_immortal_error_for_testing();
    ASSERT_TRUE(seed.is_immortal());
    ASSERT_TRUE(seed.frames().empty());

    EXPECT_NO_THROW(core::ErrorReporter::get().report(seed, core::ReportChannel::OwnerLog));
    EXPECT_NO_THROW(core::ErrorReporter::get().report(seed, core::ReportChannel::ProcessBoundary));
}

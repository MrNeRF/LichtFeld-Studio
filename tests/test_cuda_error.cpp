/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/cuda_error.hpp"
#include "core/device_fault.hpp"
#include "core/failure_report.hpp"
#include "core/logger.hpp"
#include "core/tensor.hpp"

#include <atomic>
#include <gtest/gtest.h>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {

    class ErrorLogCapture {
    public:
        ErrorLogCapture()
            : token_(lfs::core::Logger::get().add_log_handler(
                  [this](const lfs::core::LogLevel level,
                         const lfs::core::SourceSite&,
                         const std::string_view message) {
                      if (level != lfs::core::LogLevel::Error) {
                          return;
                      }
                      std::scoped_lock lock(mutex_);
                      messages_.emplace_back(message);
                  })) {}

        ~ErrorLogCapture() {
            lfs::core::Logger::get().remove_log_handler(token_);
        }

        [[nodiscard]] std::string joined() const {
            std::scoped_lock lock(mutex_);
            std::string result;
            for (const auto& message : messages_) {
                result += message;
                result += '\n';
            }
            return result;
        }

    private:
        lfs::core::LogHandlerToken token_;
        mutable std::mutex mutex_;
        std::vector<std::string> messages_;
    };

    class CudaErrorDiagnostics : public ::testing::Test {
    protected:
        void SetUp() override {
            lfs::core::reset_cuda_diagnostics_for_testing();
        }

        void TearDown() override {
            lfs::core::reset_cuda_diagnostics_for_testing();
        }
    };

    TEST_F(CudaErrorDiagnostics, CudaErrorClassifier) {
        EXPECT_TRUE(lfs::core::is_cuda_unavailable_error(cudaErrorInitializationError));
        EXPECT_TRUE(lfs::core::is_cuda_unavailable_error(cudaErrorNoDevice));
        EXPECT_TRUE(lfs::core::is_cuda_unavailable_error(cudaErrorInsufficientDriver));
        EXPECT_FALSE(lfs::core::is_cuda_unavailable_error(cudaErrorMemoryAllocation));
        EXPECT_FALSE(lfs::core::is_cuda_unavailable_error(cudaSuccess));
        EXPECT_FALSE(lfs::core::is_cuda_unavailable_error(cudaErrorInvalidDevice));
    }

    TEST_F(CudaErrorDiagnostics, CudaUnavailableLatchIsOnceAndTerminal) {
        EXPECT_FALSE(lfs::core::cuda_is_unavailable());
        EXPECT_TRUE(lfs::core::latch_cuda_unavailable(cudaErrorInitializationError));
        EXPECT_FALSE(lfs::core::latch_cuda_unavailable(cudaErrorInitializationError));
        EXPECT_TRUE(lfs::core::cuda_is_unavailable());
    }

    TEST_F(CudaErrorDiagnostics, FailureReportDedupEmitsFullThenRepeats) {
        uint64_t count = 0;
        EXPECT_TRUE(lfs::core::decide_failure_report_for_testing("F", 7, "site.cpp:10", count));
        EXPECT_EQ(count, 1u);
        EXPECT_FALSE(lfs::core::decide_failure_report_for_testing("F", 7, "site.cpp:10", count));
        EXPECT_EQ(count, 2u);
        EXPECT_FALSE(lfs::core::decide_failure_report_for_testing("F", 7, "site.cpp:10", count));
        EXPECT_EQ(count, 3u);
        EXPECT_TRUE(lfs::core::decide_failure_report_for_testing("F", 7, "other.cpp:10", count));
        EXPECT_EQ(count, 1u);
    }

    TEST_F(CudaErrorDiagnostics, BreadcrumbRingWrapsMostRecentFirst) {
        lfs::core::clear_cuda_breadcrumbs_for_testing();
        for (size_t i = 0; i < lfs::core::CUDA_BREADCRUMB_CAPACITY + 9; ++i) {
            lfs::core::record_cuda_breadcrumb("wrap", __FILE__, static_cast<uint32_t>(i + 1));
        }

        const auto breadcrumbs = lfs::core::cuda_breadcrumbs_most_recent_first();

        ASSERT_EQ(breadcrumbs.size(), lfs::core::CUDA_BREADCRUMB_CAPACITY);
        EXPECT_EQ(breadcrumbs.front().sequence, lfs::core::CUDA_BREADCRUMB_CAPACITY + 9);
        EXPECT_EQ(breadcrumbs.front().line, lfs::core::CUDA_BREADCRUMB_CAPACITY + 9);
        EXPECT_EQ(breadcrumbs.back().sequence, 10);
        EXPECT_EQ(breadcrumbs.back().line, 10);
    }

    TEST_F(CudaErrorDiagnostics, BreadcrumbRingThreadSafetySmoke) {
        lfs::core::clear_cuda_breadcrumbs_for_testing();
        constexpr size_t THREAD_COUNT = 8;
        constexpr size_t WRITES_PER_THREAD = 512;
        std::vector<std::thread> threads;
        threads.reserve(THREAD_COUNT);
        for (size_t thread = 0; thread < THREAD_COUNT; ++thread) {
            threads.emplace_back([] {
                for (size_t i = 0; i < WRITES_PER_THREAD; ++i) {
                    LFS_CUDA_BREADCRUMB("thread-smoke");
                }
            });
        }
        for (auto& thread : threads) {
            thread.join();
        }

        const auto breadcrumbs = lfs::core::cuda_breadcrumbs_most_recent_first();
        ASSERT_EQ(breadcrumbs.size(), lfs::core::CUDA_BREADCRUMB_CAPACITY);
        for (size_t i = 1; i < breadcrumbs.size(); ++i) {
            EXPECT_GT(breadcrumbs[i - 1].sequence, breadcrumbs[i].sequence);
            EXPECT_STREQ(breadcrumbs[i].tag, "thread-smoke");
            EXPECT_NE(breadcrumbs[i].thread_id, 0);
        }
    }

    TEST_F(CudaErrorDiagnostics, BreadcrumbArgumentsRoundTripAndFormatting) {
        constexpr uint64_t SOURCE = 0x1306000000ULL;
        constexpr uint64_t DESTINATION = 0x7f12000000ULL;
        lfs::core::clear_cuda_breadcrumbs_for_testing();
        lfs::core::record_cuda_breadcrumb(
            "with-args", __FILE__, __LINE__, nullptr, DESTINATION, SOURCE);
        lfs::core::record_cuda_breadcrumb("zero-args", __FILE__, __LINE__);

        const auto breadcrumbs = lfs::core::cuda_breadcrumbs_most_recent_first();
        ASSERT_EQ(breadcrumbs.size(), 2u);
        EXPECT_EQ(breadcrumbs[0].a0, 0u);
        EXPECT_EQ(breadcrumbs[0].a1, 0u);
        EXPECT_EQ(breadcrumbs[0].a2, 0u);
        EXPECT_EQ(breadcrumbs[1].a0, DESTINATION);
        EXPECT_EQ(breadcrumbs[1].a1, SOURCE);
        EXPECT_EQ(breadcrumbs[1].a2, 0u);

        lfs::core::initialize_cuda_diagnostics();
        const std::string report = lfs::core::format_failure_report(
            lfs::core::FailureReport{
                .family = "CUDA",
                .error = "synthetic",
                .expression = "breadcrumb formatting",
                .location = LFS_SOURCE_SITE_CURRENT(),
            },
            "  #0 breadcrumb-args-test\n");
        EXPECT_NE(report.find("with-args"), std::string::npos);
        EXPECT_NE(report.find("args=0x7f12000000,0x1306000000"), std::string::npos);
        EXPECT_EQ(report.find("args=0x7f12000000,0x1306000000,0x0"), std::string::npos);

        const size_t zero_position = report.find("zero-args");
        ASSERT_NE(zero_position, std::string::npos);
        const size_t zero_line_end = report.find('\n', zero_position);
        ASSERT_NE(zero_line_end, std::string::npos);
        const std::string_view zero_line(
            report.data() + zero_position, zero_line_end - zero_position);
        EXPECT_EQ(zero_line.find("args="), std::string_view::npos);
    }

    TEST_F(CudaErrorDiagnostics, ContractReportFormattingExcludesCudaSections) {
        const std::string report = lfs::core::format_contract_failure_report(
            "test contract", "lhs.dtype() == rhs.dtype()", "dtype mismatch",
            LFS_SOURCE_SITE_CURRENT(), "  #0 formatting_test\n");

        EXPECT_NE(report.find("========== LFS FAILURE REPORT =========="), std::string::npos);
        EXPECT_NE(report.find("Family: tensor contract violation"), std::string::npos);
        EXPECT_NE(report.find("Failed expression: lhs.dtype() == rhs.dtype()"), std::string::npos);
        EXPECT_NE(report.find("Host stack trace:"), std::string::npos);
        EXPECT_EQ(report.find("Thread:"), std::string::npos);
        EXPECT_EQ(report.find("CUDA device:"), std::string::npos);
        EXPECT_EQ(report.find("VRAM:"), std::string::npos);
        EXPECT_EQ(report.find("CUDA breadcrumbs (most recent first):"), std::string::npos);
        EXPECT_EQ(report.find("LFS_CUDA_SYNC_DEBUG=cuda-sync"), std::string::npos);
    }

    TEST_F(CudaErrorDiagnostics, SuccessfulCheckDoesNotFormatFailureContext) {
        if (lfs::core::cuda_sync_debug_enabled()) {
            GTEST_SKIP() << "sync-debug mode deliberately runs the full completion path";
        }

        int format_evaluations = 0;
        LFS_CUDA_CHECK_MSG(cudaSuccess, "unused failure context {}", ++format_evaluations);
        LFS_CUDA_CHECK_MSG_ARGS(
            cudaSuccess, 1, 2, 3,
            "unused argument-carrying failure context {}", ++format_evaluations);
        LFS_CUDA_CHECK_MSG_STREAM_ARGS(
            cudaSuccess, nullptr, 1, 2, 3,
            "unused stream-and-argument failure context {}", ++format_evaluations);

        EXPECT_EQ(format_evaluations, 0);
    }

    TEST_F(CudaErrorDiagnostics, DeadAddressRangeIsAnnotatedFromTransferBreadcrumb) {
        constexpr uintptr_t RANGE_BASE = 0x1306000000ULL;
        constexpr size_t RANGE_BYTES = 100ULL * 1024ULL * 1024ULL;
        constexpr uintptr_t SOURCE = RANGE_BASE + 0x4000ULL;
        constexpr uintptr_t DESTINATION = 0x7f12000000ULL;
        cudaStream_t failure_stream = nullptr;
        ASSERT_EQ(cudaStreamCreate(&failure_stream), cudaSuccess);
        lfs::core::clear_cuda_breadcrumbs_for_testing();
        lfs::core::register_cuda_address_range(
            reinterpret_cast<const void*>(RANGE_BASE), RANGE_BYTES,
            "exportable-splat-block");
        lfs::core::unregister_cuda_address_range(
            reinterpret_cast<const void*>(RANGE_BASE));
        ErrorLogCapture capture;

        EXPECT_THROW(
            LFS_CUDA_CHECK_MSG_STREAM_ARGS(
                cudaSetDevice(-1), failure_stream,
                DESTINATION, SOURCE, RANGE_BYTES,
                "while copying tensor '{}' shape={} dtype={} dst={} src={} to CPU",
                "SplatData.means", "[1000000,3]", "float32",
                reinterpret_cast<const void*>(DESTINATION),
                reinterpret_cast<const void*>(SOURCE)),
            std::runtime_error);

        const std::string report = capture.joined();
        EXPECT_NE(report.find("Context: while copying tensor 'SplatData.means' "
                              "shape=[1000000,3] dtype=float32 dst=0x7f12000000 "
                              "src=0x1306004000 to CPU"),
                  std::string::npos);
        EXPECT_NE(report.find("args=0x7f12000000,0x1306004000,0x6400000"),
                  std::string::npos);
        EXPECT_NE(report.find("Address annotations:"), std::string::npos);
        EXPECT_NE(report.find("0x1306004000 → inside DEAD range "
                              "'exportable-splat-block' [0x1306000000 +100 MiB], "
                              "unmapped "),
                  std::string::npos);
        EXPECT_NE(report.find(" ms ago"), std::string::npos);
        const auto breadcrumbs = lfs::core::cuda_breadcrumbs_most_recent_first();
        ASSERT_EQ(breadcrumbs.size(), 1u);
        EXPECT_EQ(breadcrumbs.front().stream,
                  reinterpret_cast<uintptr_t>(failure_stream));
        EXPECT_EQ(breadcrumbs.front().a0, DESTINATION);
        EXPECT_EQ(breadcrumbs.front().a1, SOURCE);
        EXPECT_EQ(breadcrumbs.front().a2, RANGE_BYTES);
        (void)cudaGetLastError();
        EXPECT_EQ(cudaStreamDestroy(failure_stream), cudaSuccess);
    }

    TEST_F(CudaErrorDiagnostics, ResetClearsAddressRangesAndNoHitOmitsAnnotationSection) {
        constexpr uintptr_t RANGE_BASE = 0x1306000000ULL;
        constexpr size_t RANGE_BYTES = 100ULL * 1024ULL * 1024ULL;
        constexpr uintptr_t SOURCE = RANGE_BASE + 0x4000ULL;
        constexpr uintptr_t DESTINATION = 0x7f12000000ULL;
        lfs::core::register_cuda_address_range(
            reinterpret_cast<const void*>(RANGE_BASE), RANGE_BYTES,
            "exportable-splat-block");
        lfs::core::unregister_cuda_address_range(
            reinterpret_cast<const void*>(RANGE_BASE));
        lfs::core::reset_cuda_diagnostics_for_testing();
        lfs::core::clear_cuda_breadcrumbs_for_testing();
        ErrorLogCapture capture;

        EXPECT_THROW(
            LFS_CUDA_CHECK_MSG_ARGS(
                cudaSetDevice(-1), DESTINATION, SOURCE, RANGE_BYTES,
                "synthetic transfer dst={} src={} bytes={} after diagnostic reset",
                reinterpret_cast<const void*>(DESTINATION),
                reinterpret_cast<const void*>(SOURCE), RANGE_BYTES),
            std::runtime_error);

        EXPECT_EQ(capture.joined().find("Address annotations:"), std::string::npos);
        (void)cudaGetLastError();
    }

    TEST_F(CudaErrorDiagnostics, ForcedCudaErrorReportHasExpectedSections) {
        ErrorLogCapture capture;

        EXPECT_THROW(LFS_CUDA_CHECK(cudaSetDevice(-1)), std::runtime_error);

        const std::string report = capture.joined();
        EXPECT_NE(report.find("Family: CUDA"), std::string::npos);
        EXPECT_NE(report.find("Error: cudaErrorInvalidDevice"), std::string::npos);
        EXPECT_NE(report.find("Failed expression: cudaSetDevice(-1)"), std::string::npos);
        EXPECT_NE(report.find("Detection site:"), std::string::npos);
        EXPECT_NE(report.find("Attribution:"), std::string::npos);
        EXPECT_NE(report.find("Thread:"), std::string::npos);
        EXPECT_NE(report.find("CUDA device:"), std::string::npos);
        EXPECT_NE(report.find("VRAM:"), std::string::npos);
        EXPECT_NE(report.find("Host stack trace:"), std::string::npos);
        EXPECT_NE(report.find("CUDA breadcrumbs (most recent first):"), std::string::npos);
        EXPECT_NE(report.find("LFS_CUDA_SYNC_DEBUG=cuda-sync"), std::string::npos);
        (void)cudaGetLastError();
    }

    TEST(DiagnosticModeParsing, AbsentValueIsNoModes) {
        const auto parsed = lfs::core::parse_diagnostic_modes(std::nullopt, std::nullopt);
        EXPECT_EQ(parsed.modes, 0u);
        EXPECT_FALSE(parsed.unknown_tokens_present);
        EXPECT_FALSE(parsed.legacy_alias_present);
    }

    TEST(DiagnosticModeParsing, EmptyValueIsNoModes) {
        const auto parsed = lfs::core::parse_diagnostic_modes("", "");
        EXPECT_EQ(parsed.modes, 0u);
        EXPECT_FALSE(parsed.unknown_tokens_present);
        EXPECT_FALSE(parsed.legacy_alias_present);
    }

    TEST(DiagnosticModeParsing, LegacyFalsyIsNoModes) {
        const auto parsed = lfs::core::parse_diagnostic_modes("0", std::nullopt);
        EXPECT_EQ(parsed.modes, 0u);
    }

    TEST(DiagnosticModeParsing, LegacyTruthyOneMeansCudaSync) {
        const auto parsed = lfs::core::parse_diagnostic_modes("1", std::nullopt);
        EXPECT_EQ(parsed.modes, static_cast<unsigned>(lfs::core::DiagnosticMode::CudaSync));
    }

    TEST(DiagnosticModeParsing, LegacyTruthyTrueMeansCudaSync) {
        const auto parsed = lfs::core::parse_diagnostic_modes("true", std::nullopt);
        EXPECT_EQ(parsed.modes, static_cast<unsigned>(lfs::core::DiagnosticMode::CudaSync));
    }

    TEST(DiagnosticModeParsing, SingleModeToken) {
        const auto parsed = lfs::core::parse_diagnostic_modes("cuda-sync", std::nullopt);
        EXPECT_EQ(parsed.modes, static_cast<unsigned>(lfs::core::DiagnosticMode::CudaSync));
        EXPECT_FALSE(parsed.unknown_tokens_present);
    }

    TEST(DiagnosticModeParsing, ModeListUnionsRequestedModes) {
        const auto parsed = lfs::core::parse_diagnostic_modes("cuda-sync,vk-fatal", std::nullopt);
        EXPECT_EQ(parsed.modes,
                  static_cast<unsigned>(lfs::core::DiagnosticMode::CudaSync) |
                      static_cast<unsigned>(lfs::core::DiagnosticMode::VkFatal));
    }

    TEST(DiagnosticModeParsing, DeviceTrapTokenIsParsedAndConsumedByLaunchPrepHelper) {
        // device-trap still parses into DiagnosticMode::DeviceTrap (Phase 0).
        const auto parsed = lfs::core::parse_diagnostic_modes("device-trap", std::nullopt);
        EXPECT_EQ(parsed.modes, static_cast<unsigned>(lfs::core::DiagnosticMode::DeviceTrap));
        // Phase 6C-P2 consumer: host launch-prep helper reads the process diagnostic
        // bitmask once and yields trap_after_record for kernels (Ruling 1: device
        // never parses modes). Both surfaces share the same DeviceTrap bit.
        EXPECT_EQ(lfs::core::device_fault_trap_after_record_for_launch(),
                  lfs::core::diagnostic_mode_enabled(lfs::core::DiagnosticMode::DeviceTrap));
    }

    TEST(DiagnosticModeParsing, UnknownTokenWarnsAndIsIgnored) {
        const auto parsed = lfs::core::parse_diagnostic_modes("cuda-sync,bogus-mode", std::nullopt);
        EXPECT_EQ(parsed.modes, static_cast<unsigned>(lfs::core::DiagnosticMode::CudaSync));
        EXPECT_TRUE(parsed.unknown_tokens_present);
        EXPECT_NE(parsed.unknown_tokens.find("bogus-mode"), std::string::npos);
    }

    TEST(DiagnosticModeParsing, WhitespaceAroundTokensIsTrimmed) {
        const auto parsed = lfs::core::parse_diagnostic_modes(" cuda-sync , vk-fatal ", std::nullopt);
        EXPECT_EQ(parsed.modes,
                  static_cast<unsigned>(lfs::core::DiagnosticMode::CudaSync) |
                      static_cast<unsigned>(lfs::core::DiagnosticMode::VkFatal));
        EXPECT_FALSE(parsed.unknown_tokens_present);
    }

    TEST(DiagnosticModeParsing, LegacyVkValidationFatalAliasAloneSetsVkFatal) {
        const auto parsed = lfs::core::parse_diagnostic_modes(std::nullopt, "1");
        EXPECT_EQ(parsed.modes, static_cast<unsigned>(lfs::core::DiagnosticMode::VkFatal));
        EXPECT_TRUE(parsed.legacy_alias_present);
    }

    TEST(DiagnosticModeParsing, BothSpellingsSetUnionModesWithOneAliasWarning) {
        const auto parsed = lfs::core::parse_diagnostic_modes("cuda-sync", "1");
        EXPECT_EQ(parsed.modes,
                  static_cast<unsigned>(lfs::core::DiagnosticMode::CudaSync) |
                      static_cast<unsigned>(lfs::core::DiagnosticMode::VkFatal));
        EXPECT_TRUE(parsed.legacy_alias_present);
    }

    TEST(DiagnosticModeParsing, LegacyAliasFalsyStillWarnsButAddsNoMode) {
        const auto parsed = lfs::core::parse_diagnostic_modes("cuda-sync", "0");
        EXPECT_EQ(parsed.modes, static_cast<unsigned>(lfs::core::DiagnosticMode::CudaSync));
        EXPECT_TRUE(parsed.legacy_alias_present);
    }

    TEST_F(CudaErrorDiagnostics, ContractReportNamesTensorCallerInStack) {
        ErrorLogCapture capture;
        const auto values = lfs::core::Tensor::from_vector(
            {1.0f, 2.0f}, {2}, lfs::core::Device::CPU);
        const auto invalid_mask = lfs::core::Tensor::from_vector(
            {1.0f, 0.0f}, {2}, lfs::core::Device::CPU);

        EXPECT_THROW((void)values.masked_select(invalid_mask), std::runtime_error);

        const std::string report = capture.joined();
        EXPECT_NE(report.find("Family: tensor contract violation"), std::string::npos);
        EXPECT_NE(report.find("masked_select"), std::string::npos);
        EXPECT_NE(report.find("ContractReportNamesTensorCallerInStack"), std::string::npos);
    }

} // namespace

/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/cuda_error.hpp"
#include "core/failure_report.hpp"
#include "core/logger.hpp"
#include "core/tensor.hpp"

#include <atomic>
#include <gtest/gtest.h>
#include <mutex>
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
        EXPECT_EQ(report.find("LFS_CUDA_SYNC_DEBUG=1"), std::string::npos);
    }

    TEST_F(CudaErrorDiagnostics, SuccessfulCheckDoesNotFormatFailureContext) {
        if (lfs::core::cuda_sync_debug_enabled()) {
            GTEST_SKIP() << "sync-debug mode deliberately runs the full completion path";
        }

        int format_evaluations = 0;
        LFS_CUDA_CHECK_MSG(cudaSuccess, "unused failure context {}", ++format_evaluations);

        EXPECT_EQ(format_evaluations, 0);
    }

    TEST_F(CudaErrorDiagnostics, ForcedCudaErrorReportHasExpectedSections) {
        ErrorLogCapture capture;

        EXPECT_THROW(LFS_CUDA_CHECK(cudaSetDevice(-1)), std::runtime_error);

        const std::string report = capture.joined();
        EXPECT_NE(report.find("Family: CUDA runtime error"), std::string::npos);
        EXPECT_NE(report.find("Error: cudaErrorInvalidDevice"), std::string::npos);
        EXPECT_NE(report.find("Failed expression: cudaSetDevice(-1)"), std::string::npos);
        EXPECT_NE(report.find("Detection site:"), std::string::npos);
        EXPECT_NE(report.find("Attribution:"), std::string::npos);
        EXPECT_NE(report.find("Thread:"), std::string::npos);
        EXPECT_NE(report.find("CUDA device:"), std::string::npos);
        EXPECT_NE(report.find("VRAM:"), std::string::npos);
        EXPECT_NE(report.find("Host stack trace:"), std::string::npos);
        EXPECT_NE(report.find("CUDA breadcrumbs (most recent first):"), std::string::npos);
        EXPECT_NE(report.find("LFS_CUDA_SYNC_DEBUG=1"), std::string::npos);
        (void)cudaGetLastError();
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

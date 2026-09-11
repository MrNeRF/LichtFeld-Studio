/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/cuda_error_typed.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <string_view>
#include <variant>

namespace {

    class CudaLaunchCheckTest : public ::testing::Test {
    protected:
        void SetUp() override {
            lfs::core::reset_cuda_diagnostics_for_testing();
        }

        void TearDown() override {
            (void)cudaGetLastError();
            lfs::core::reset_cuda_diagnostics_for_testing();
        }
    };

    class CudaLaunchCheckDeathTest : public CudaLaunchCheckTest {};

    [[nodiscard]] const lfs::SmallFields::Entry* find_field(
        const lfs::Error& error, const std::string_view key) {
        if (error.frames().empty()) {
            return nullptr;
        }
        for (const auto& entry : error.frames().front().fields.entries()) {
            if (entry.key == key) {
                return &entry;
            }
        }
        return nullptr;
    }

    void enable_cuda_sync_debug_for_subprocess() {
#if defined(_WIN32)
        (void)_putenv_s("LFS_CUDA_SYNC_DEBUG", "cuda-sync");
#else
        (void)setenv("LFS_CUDA_SYNC_DEBUG", "cuda-sync", 1);
#endif
    }

    TEST_F(CudaLaunchCheckTest, DirectSeedFailureThrowsTypedCudaError) {
        const lfs::core::CudaFailureSeed seed{
            .native = cudaErrorMemoryAllocation,
            .predecessor = cudaSuccess,
            .stream = 0x1234,
            .first_sequence = 7,
            .last_sequence = 11,
            .expression = "direct seed launch",
            .source = LFS_SOURCE_SITE_CURRENT(),
        };

        try {
            lfs::core::report_cuda_launch_check_failure(seed);
            FAIL() << "report_cuda_launch_check_failure returned";
        } catch (const lfs::Exception& exception) {
            const lfs::Error& error = exception.error();
            EXPECT_EQ(error.domain(), lfs::ErrorDomain::CUDA);
            EXPECT_EQ(error.code(), lfs::ErrorCode::ResourceExhausted);
            ASSERT_TRUE(error.native().has_value());
            EXPECT_EQ(error.native()->code, cudaErrorMemoryAllocation);
            EXPECT_NE(error.detail().find("kernel launch check failed"), std::string_view::npos);
            ASSERT_NE(find_field(error, "stream"), nullptr);
            EXPECT_EQ(std::get<std::int64_t>(find_field(error, "stream")->value), 0x1234);
        }
    }

    TEST_F(CudaLaunchCheckTest, CleanSlowPathReturnsNormally) {
        if (lfs::core::cuda_sync_debug_enabled()) {
            GTEST_SKIP() << "cuda-sync mode deliberately performs a real CUDA synchronization";
        }

        EXPECT_NO_THROW(lfs::core::handle_cuda_launch_check_slow_path(
            cudaSuccess, nullptr, "clean launch", LFS_SOURCE_SITE_CURRENT(), 1));
    }

    TEST_F(CudaLaunchCheckTest, FailureLatchUsesMostRecentPredecessorAcrossStreams) {
        EXPECT_EQ(lfs::core::exchange_last_cuda_check_failure(cudaErrorInvalidConfiguration),
                  cudaSuccess);
        EXPECT_EQ(lfs::core::exchange_last_cuda_check_failure(cudaErrorIllegalAddress),
                  cudaErrorInvalidConfiguration);
        EXPECT_EQ(lfs::core::exchange_last_cuda_check_failure(cudaErrorLaunchFailure),
                  cudaErrorIllegalAddress);
    }

    TEST_F(CudaLaunchCheckTest, BreadcrumbSequenceAndAwaitRangeAdvanceMonotonically) {
        const std::uint64_t before = lfs::core::current_cuda_breadcrumb_sequence();
        const std::uint64_t first = lfs::core::record_cuda_breadcrumb(
            "private launch one", __FILE__, __LINE__);
        const std::uint64_t second = lfs::core::record_cuda_breadcrumb(
            "private launch two", __FILE__, __LINE__);

        EXPECT_EQ(first, before + 1);
        EXPECT_EQ(second, before + 2);
        EXPECT_EQ(lfs::core::current_cuda_breadcrumb_sequence(), second);

        const auto ticket = lfs::core::cuda_record_range(nullptr, "submitted operation");
        const std::uint64_t after_ticket = lfs::core::record_cuda_breadcrumb(
            "private launch three", __FILE__, __LINE__);
        EXPECT_EQ(ticket.first_sequence, second);
        EXPECT_LT(ticket.first_sequence, after_ticket);
        EXPECT_STREQ(ticket.operation_tag, "submitted operation");
    }

    TEST_F(CudaLaunchCheckTest, AwaitSuccessPathWithRealCudaCallDoesNotThrow) {
        int device = -1;
        if (cudaGetDevice(&device) != cudaSuccess) {
            (void)cudaGetLastError();
            GTEST_SKIP() << "a live CUDA device is required";
        }

        const auto ticket = lfs::core::cuda_record_range(nullptr, "query current device");
        EXPECT_NO_THROW(LFS_CUDA_AWAIT(ticket, cudaGetDevice(&device), "query current device"));
    }

    TEST_F(CudaLaunchCheckTest, AwaitFailureThrowsWithSubmittedRangeFraming) {
        const auto ticket = lfs::core::cuda_record_range(nullptr, "await injected failure");
        const std::uint64_t last_sequence = lfs::core::record_cuda_breadcrumb(
            "await observation", __FILE__, __LINE__);

        try {
            lfs::core::handle_cuda_await_failure(
                cudaErrorInvalidValue, ticket, "await injected failure",
                LFS_SOURCE_SITE_CURRENT(), last_sequence);
            FAIL() << "handle_cuda_await_failure returned";
        } catch (const lfs::Exception& exception) {
            const lfs::Error& error = exception.error();
            EXPECT_NE(error.detail().find("observed at await"), std::string_view::npos);
            EXPECT_NE(error.detail().find(ticket.operation_tag), std::string_view::npos);
            EXPECT_EQ(error.code(), lfs::ErrorCode::InvalidArgument);
        }
    }

    TEST_F(CudaLaunchCheckTest, NativeLaunchStatusHasPriorityOnSlowPath) {
        try {
            lfs::core::handle_cuda_launch_check_slow_path(
                cudaErrorInvalidConfiguration, nullptr, "priority launch",
                LFS_SOURCE_SITE_CURRENT(), 1);
            FAIL() << "handle_cuda_launch_check_slow_path returned";
        } catch (const lfs::Exception& exception) {
            ASSERT_TRUE(exception.error().native().has_value());
            EXPECT_EQ(exception.error().native()->code, cudaErrorInvalidConfiguration);
        }
    }

    TEST_F(CudaLaunchCheckDeathTest, PoisonedLaunchStateIsCaughtByLaunchCheck) {
        EXPECT_EXIT(
            {
                (void)cudaSetDevice(-1);
                (void)cudaLaunchKernel(nullptr, dim3(1), dim3(1), nullptr, 0, nullptr);
                try {
                    LFS_CUDA_LAUNCH_CHECK(nullptr, "poisoned launch check");
                    std::_Exit(2);
                } catch (const lfs::Exception&) {
                    std::_Exit(0);
                } catch (...) {
                    std::_Exit(3);
                }
            },
            ::testing::ExitedWithCode(0), "");
    }

    TEST_F(CudaLaunchCheckDeathTest, PoisonedAwaitCallThrowsCatchableException) {
        EXPECT_EXIT(
            {
                const auto ticket =
                    lfs::core::cuda_record_range(nullptr, "poisoned await submission");
                try {
                    LFS_CUDA_AWAIT(ticket, cudaSetDevice(-1), "poisoned await observation");
                    std::_Exit(2);
                } catch (const lfs::Exception&) {
                    std::_Exit(0);
                } catch (...) {
                    std::_Exit(3);
                }
            },
            ::testing::ExitedWithCode(0), "");
    }

    TEST_F(CudaLaunchCheckDeathTest, CudaSyncModeCleanSlowPathReturnsNormally) {
        GTEST_FLAG_SET(death_test_style, "threadsafe");
        EXPECT_EXIT(
            {
                enable_cuda_sync_debug_for_subprocess();
                int device = -1;
                if (cudaGetDevice(&device) != cudaSuccess) {
                    std::_Exit(4);
                }
                try {
                    lfs::core::handle_cuda_launch_check_slow_path(
                        cudaSuccess, nullptr, "cuda-sync clean launch",
                        LFS_SOURCE_SITE_CURRENT(), 1);
                    std::_Exit(0);
                } catch (...) {
                    std::_Exit(3);
                }
            },
            ::testing::ExitedWithCode(0), "");
    }

    TEST_F(CudaLaunchCheckDeathTest, CudaSyncModeNativeStatusWinsOverSyncAndPeek) {
        GTEST_FLAG_SET(death_test_style, "threadsafe");
        EXPECT_EXIT(
            {
                enable_cuda_sync_debug_for_subprocess();
                (void)cudaSetDevice(-1);
                try {
                    lfs::core::handle_cuda_launch_check_slow_path(
                        cudaErrorInvalidConfiguration, nullptr, "cuda-sync priority launch",
                        LFS_SOURCE_SITE_CURRENT(), 1);
                    std::_Exit(2);
                } catch (const lfs::Exception& exception) {
                    if (exception.error().native().has_value() &&
                        exception.error().native()->code == cudaErrorInvalidConfiguration) {
                        std::_Exit(0);
                    }
                    std::_Exit(5);
                } catch (...) {
                    std::_Exit(3);
                }
            },
            ::testing::ExitedWithCode(0), "");
    }

} // namespace

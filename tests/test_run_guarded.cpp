/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/error.hpp"
#include "core/guarded_task.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <thread>

namespace {

    lfs::core::TaskContext make_context(const std::string& name = "test.run_guarded") {
        return {
            .name = name,
            .domain = lfs::ErrorDomain::Core,
            .operation_id = lfs::OperationId::generate(),
            .site = LFS_SOURCE_SITE_CURRENT(),
        };
    }

    lfs::Error make_test_error(const lfs::ErrorCode code,
                               const std::string& detail = "test failure") {
        return lfs::make_error(lfs::ErrorInit{
            .code = code,
            .domain = lfs::ErrorDomain::Core,
            .operation_id = lfs::OperationId::generate(),
            .detail = detail,
            .detection = LFS_SOURCE_SITE_CURRENT(),
        });
    }

} // namespace

TEST(RunGuardedTest, ImmediateStandardThrowSettlesInternalFailureExactlyOnce) {
    int completions = 0;
    lfs::core::run_guarded<void>(
        make_context(),
        []() -> lfs::Result<void> { throw std::runtime_error("x"); },
        [&completions](lfs::Result<void>&& result) {
            ++completions;
            ASSERT_FALSE(result);
            EXPECT_EQ(result.error().code(), lfs::ErrorCode::Internal);
            EXPECT_EQ(result.error().detail(), "x");
        });
    EXPECT_EQ(completions, 1);
}

TEST(RunGuardedTest, PartialWorkRemainsVisibleWhenBodyThrows) {
    bool partial_work = false;
    int completions = 0;
    lfs::core::run_guarded<void>(
        make_context(),
        [&partial_work]() -> lfs::Result<void> {
            partial_work = true;
            throw std::runtime_error("inside");
        },
        [&completions](lfs::Result<void>&& result) {
            ++completions;
            ASSERT_FALSE(result);
            EXPECT_EQ(result.error().detail(), "inside");
        });
    EXPECT_TRUE(partial_work);
    EXPECT_EQ(completions, 1);
}

TEST(RunGuardedTest, LichtfeldExceptionPreservesClassificationAndAddsContext) {
    lfs::core::run_guarded<void>(
        make_context("preserved-context"),
        []() -> lfs::Result<void> {
            throw lfs::Exception(make_test_error(lfs::ErrorCode::ResourceExhausted));
        },
        [](lfs::Result<void>&& result) {
            ASSERT_FALSE(result);
            EXPECT_EQ(result.error().code(), lfs::ErrorCode::ResourceExhausted);
            ASSERT_FALSE(result.error().frames().empty());
            EXPECT_EQ(result.error().frames().back().operation, "preserved-context");
        });
}

TEST(RunGuardedTest, CancellationIsASettledResult) {
    int completions = 0;
    lfs::core::run_guarded<void>(
        make_context(),
        [] { return lfs::Result<void>::failure(make_test_error(lfs::ErrorCode::Cancelled)); },
        [&completions](lfs::Result<void>&& result) {
            ++completions;
            ASSERT_FALSE(result);
            EXPECT_EQ(result.error().code(), lfs::ErrorCode::Cancelled);
        });
    EXPECT_EQ(completions, 1);
}

TEST(RunGuardedTest, ThrowingCompletionIsReportedAndSettlementRejectsSecondCall) {
    auto settlement = std::make_shared<lfs::core::TaskSettlement>();
    auto context = make_context("throwing-completion");
    context.settlement = settlement;

    testing::internal::CaptureStderr();
    lfs::core::run_guarded<void>(
        context,
        [] { return lfs::Result<void>{}; },
        [](lfs::Result<void>&&) { throw std::runtime_error("completion bug"); });
    const std::string captured = testing::internal::GetCapturedStderr();

    EXPECT_NE(captured.find("TaskCompletion sink threw during settlement"), std::string::npos);
    EXPECT_EQ(captured.find("TaskCompletion sink threw during settlement"),
              captured.rfind("TaskCompletion sink threw during settlement"));

    bool second_body_ran = false;
    bool second_completion_ran = false;
    lfs::core::run_guarded<void>(
        context,
        [&second_body_ran] {
            second_body_ran = true;
            return lfs::Result<void>{};
        },
        [&second_completion_ran](lfs::Result<void>&&) { second_completion_ran = true; });
    EXPECT_FALSE(second_body_ran);
    EXPECT_FALSE(second_completion_ran);
}

TEST(RunGuardedTest, ExceptionNormalizationRemainsNoThrowWhenAllocationFails) {
    try {
        throw std::runtime_error("allocation fallback");
    } catch (...) {
        lfs::force_next_error_allocation_to_fail_for_testing(true);
        const lfs::Error normalized =
            lfs::core::detail::normalize_current_exception(make_context());
        EXPECT_TRUE(normalized.is_immortal());
        EXPECT_EQ(normalized.code(), lfs::ErrorCode::ResourceExhausted);
    }
}

TEST(RunGuardedTest, NonVoidInstantiationSettlesSuccessAndFailure) {
    int success = 0;
    lfs::core::run_guarded<int>(
        make_context(),
        [] { return lfs::Result<int>(42); },
        [&success](lfs::Result<int>&& result) {
            ASSERT_TRUE(result);
            success = *result;
        });
    EXPECT_EQ(success, 42);

    lfs::core::run_guarded<int>(
        make_context(),
        [] { return lfs::Result<int>(make_test_error(lfs::ErrorCode::InvalidArgument)); },
        [](lfs::Result<int>&& result) {
            ASSERT_FALSE(result);
            EXPECT_EQ(result.error().code(), lfs::ErrorCode::InvalidArgument);
        });
}

TEST(RunGuardedDeathTest, UnexpectedThreadTerminatesBeforeBodyExecution) {
    std::thread::id other_thread;
    std::thread thread([&other_thread] { other_thread = std::this_thread::get_id(); });
    thread.join();

    auto context = make_context();
    context.expected_thread = other_thread;
    EXPECT_DEATH(
        lfs::core::run_guarded<void>(
            context,
            [] { return lfs::Result<void>{}; },
            [](lfs::Result<void>&&) {}),
        "run_guarded: body invoked on unexpected thread");
}

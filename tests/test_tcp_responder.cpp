/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/guarded_task.hpp"

#include <gtest/gtest.h>

TEST(TcpResponderTest, UnknownThreadEntryExceptionSettlesInsteadOfEscaping) {
    bool completed = false;
    lfs::core::run_guarded<void>(
        lfs::core::TaskContext{
            .name = "tcp.responder-thread",
            .domain = lfs::ErrorDomain::TCP,
            .operation_id = lfs::OperationId::generate(),
            .site = LFS_SOURCE_SITE_CURRENT(),
        },
        []() -> lfs::Result<void> { throw 7; },
        [&completed](lfs::Result<void>&& result) {
            completed = true;
            ASSERT_FALSE(result);
            EXPECT_EQ(result.error().code(), lfs::ErrorCode::Internal);
            EXPECT_EQ(result.error().domain(), lfs::ErrorDomain::TCP);
            EXPECT_EQ(result.error().detail(), "unknown exception");
        });
    EXPECT_TRUE(completed);
}

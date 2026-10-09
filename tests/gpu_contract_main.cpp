/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/crash_handler.hpp"
#include <gtest/gtest.h>

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    const int result = RUN_ALL_TESTS();
    // Release GPU holders before pools, then avoid static destructors entering
    // storage that the ordered teardown has already released.
    lfs::core::teardown_gpu_before_exit();
    lfs::core::flush_and_exit(result);
}

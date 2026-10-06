/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include <cstdio>
#ifndef LFS_METAL_TEST_REQUIRE_DEVICE
#define LFS_METAL_TEST_REQUIRE_DEVICE 0
#endif
namespace lfs::metal_test {
    inline int unavailableMetal4() {
        std::fprintf(stderr, "%s: resident Metal contracts require macOS 26 and a Metal 4 device\n",
                     LFS_METAL_TEST_REQUIRE_DEVICE ? "FAIL" : "SKIP");
        return LFS_METAL_TEST_REQUIRE_DEVICE ? 1 : 77;
    }
} // namespace lfs::metal_test

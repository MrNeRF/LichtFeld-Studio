/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/abi.hpp"
#include "lfs_core_abi_stamp.h"

#include <cstring>

extern "C" const char* lfs_core_abi_stamp() noexcept {
    return LFS_CORE_ABI_STAMP;
}

extern "C" bool lfs_core_abi_matches(const char* expected_stamp) noexcept {
    return expected_stamp != nullptr && std::strcmp(expected_stamp, LFS_CORE_ABI_STAMP) == 0;
}

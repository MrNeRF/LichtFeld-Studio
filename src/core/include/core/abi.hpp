/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/export.hpp"

extern "C" LFS_CORE_API const char* lfs_core_abi_stamp() noexcept;
extern "C" LFS_CORE_API bool lfs_core_abi_matches(const char* expected_stamp) noexcept;

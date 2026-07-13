/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/failure_report.hpp"

#include <cassert>
#include <string_view>

// Public/API-boundary contract. This remains enabled in every build type and
// throws before invalid state reaches an implementation or kernel.
#define LFS_ASSERT(condition)                                                        \
    do {                                                                             \
        if (!(condition)) [[unlikely]] {                                             \
            ::lfs::core::detail::assertion_failed(                                   \
                "LFS boundary contract", #condition, {}, LFS_SOURCE_SITE_CURRENT()); \
        }                                                                            \
    } while (false)

#define LFS_ASSERT_MSG(condition, message)                                                  \
    do {                                                                                    \
        if (!(condition)) [[unlikely]] {                                                    \
            ::lfs::core::detail::assertion_failed(                                          \
                "LFS boundary contract", #condition, (message), LFS_SOURCE_SITE_CURRENT()); \
        }                                                                                   \
    } while (false)

// Internal hot-loop/kernel invariant. This compiles to no code when NDEBUG is
// defined. Keep input validation at the public boundary in LFS_ASSERT instead.
#ifndef NDEBUG
#if defined(__CUDA_ARCH__)
// Device code cannot format or throw. Native CUDA assert still supplies the
// failed expression and source location. Caller-provided formatting is not
// evaluated in the device hot path.
#define LFS_DEBUG_ASSERT_MSG(condition, message) assert(condition)
#else
#define LFS_DEBUG_ASSERT_MSG(condition, message)                                          \
    do {                                                                                  \
        if (!(condition)) [[unlikely]] {                                                  \
            ::lfs::core::detail::assertion_failed(                                        \
                "LFS debug invariant", #condition, (message), LFS_SOURCE_SITE_CURRENT()); \
        }                                                                                 \
    } while (false)
#endif
#define LFS_DEBUG_ASSERT(condition) LFS_DEBUG_ASSERT_MSG(condition, std::string_view{})
#else
#define LFS_DEBUG_ASSERT_MSG(condition, message) ((void)0)
#define LFS_DEBUG_ASSERT(condition)              ((void)0)
#endif

/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include <cassert>
#include <stdexcept>
#include <string>
#include <string_view>

namespace lfs::core::detail {

    [[noreturn]] inline void assertion_failed(const char* expression,
                                              const std::string_view message = {}) {
        std::string error = "LFS assertion failed: ";
        error += expression;
        if (!message.empty()) {
            error += " (";
            error += message;
            error += ')';
        }
        throw std::runtime_error(error);
    }

} // namespace lfs::core::detail

// Public/API-boundary contract. This remains enabled in every build type and
// throws before invalid state reaches an implementation or kernel.
#define LFS_ASSERT(condition)                                  \
    do {                                                       \
        if (!(condition)) [[unlikely]] {                       \
            ::lfs::core::detail::assertion_failed(#condition); \
        }                                                      \
    } while (false)

#define LFS_ASSERT_MSG(condition, message)                                \
    do {                                                                  \
        if (!(condition)) [[unlikely]] {                                  \
            ::lfs::core::detail::assertion_failed(#condition, (message)); \
        }                                                                 \
    } while (false)

// Internal hot-loop/kernel invariant. This compiles to no code when NDEBUG is
// defined. Keep input validation at the public boundary in LFS_ASSERT instead.
#ifndef NDEBUG
#define LFS_DEBUG_ASSERT(condition) assert(condition)
#else
#define LFS_DEBUG_ASSERT(condition) ((void)0)
#endif

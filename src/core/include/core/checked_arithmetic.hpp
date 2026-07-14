/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/assert.hpp"
#include "core/cuda_safe_format.hpp"

#include <cstddef>
#include <limits>
#include <string_view>

namespace lfs::core {

    [[nodiscard]] inline size_t checked_product(const size_t left,
                                                const size_t right,
                                                const std::string_view context) {
        LFS_ASSERT_MSG(
            left == 0 || right <= std::numeric_limits<size_t>::max() / left,
            detail::format_cuda_safe(
                "{} size overflow (left={}, right={})", context, left, right));
        return left * right;
    }

    [[nodiscard]] constexpr size_t saturating_add(const size_t left,
                                                  const size_t right) noexcept {
        return right > std::numeric_limits<size_t>::max() - left
                   ? std::numeric_limits<size_t>::max()
                   : left + right;
    }

    [[nodiscard]] constexpr size_t saturating_multiply(const size_t left,
                                                       const size_t right) noexcept {
        return left != 0 && right > std::numeric_limits<size_t>::max() / left
                   ? std::numeric_limits<size_t>::max()
                   : left * right;
    }

} // namespace lfs::core

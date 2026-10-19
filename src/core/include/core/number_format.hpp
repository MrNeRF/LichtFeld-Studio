/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <concepts>
#include <cstddef>
#include <string>

namespace lfs::core {

    // Digits in groups of three for counts shown in the UI: 1234567 -> "1,234,567".
    template <std::integral T>
        requires(!std::same_as<T, bool>)
    [[nodiscard]] std::string format_count(const T value) {
        std::string text = std::to_string(value);
        const std::ptrdiff_t first_digit = text.front() == '-' ? 1 : 0;
        for (auto i = static_cast<std::ptrdiff_t>(text.size()) - 3; i > first_digit; i -= 3)
            text.insert(static_cast<std::size_t>(i), 1, ',');
        return text;
    }

} // namespace lfs::core

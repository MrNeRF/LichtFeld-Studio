/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <charconv>
#include <concepts>
#include <cstdlib>
#include <optional>
#include <string_view>

namespace lfs::core::environment {

    [[nodiscard]] inline std::optional<std::string_view> value(const char* const name) noexcept {
        const char* const raw = std::getenv(name);
        if (!raw || *raw == '\0') {
            return std::nullopt;
        }
        return std::string_view(raw);
    }

    namespace detail {
        [[nodiscard]] inline bool equals_ignore_ascii_case(const std::string_view lhs,
                                                           const std::string_view rhs) noexcept {
            if (lhs.size() != rhs.size()) {
                return false;
            }
            for (std::size_t i = 0; i < lhs.size(); ++i) {
                const auto lower = [](const char ch) {
                    return ch >= 'A' && ch <= 'Z' ? static_cast<char>(ch - 'A' + 'a') : ch;
                };
                if (lower(lhs[i]) != lower(rhs[i])) {
                    return false;
                }
            }
            return true;
        }
    } // namespace detail

    [[nodiscard]] inline bool flag(const char* const name,
                                   const bool default_value = false) noexcept {
        const auto raw = value(name);
        if (!raw) {
            return default_value;
        }
        if (*raw == "1" || detail::equals_ignore_ascii_case(*raw, "true") ||
            detail::equals_ignore_ascii_case(*raw, "yes") || detail::equals_ignore_ascii_case(*raw, "on")) {
            return true;
        }
        if (*raw == "0" || detail::equals_ignore_ascii_case(*raw, "false") ||
            detail::equals_ignore_ascii_case(*raw, "no") || detail::equals_ignore_ascii_case(*raw, "off")) {
            return false;
        }
        return default_value;
    }

    template <std::unsigned_integral T>
    [[nodiscard]] inline std::optional<T> unsigned_integer(const std::string_view raw) noexcept {
        T parsed{};
        const auto [end, error] = std::from_chars(raw.data(), raw.data() + raw.size(), parsed);
        if (error != std::errc{} || end != raw.data() + raw.size()) {
            return std::nullopt;
        }
        return parsed;
    }

    template <std::unsigned_integral T>
    [[nodiscard]] inline std::optional<T> unsigned_integer(const char* const name) noexcept {
        const auto raw = value(name);
        return raw ? unsigned_integer<T>(*raw) : std::nullopt;
    }

} // namespace lfs::core::environment

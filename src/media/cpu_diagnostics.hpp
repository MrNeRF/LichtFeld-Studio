// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <cstdio>
#include <format>
#include <string_view>
namespace lfs::media::detail {
    template <class... Args>
    void warning(std::format_string<Args...> text, Args&&... args) {
        const auto message = std::format(text, std::forward<Args>(args)...);
        std::fprintf(stderr, "%s\n", message.c_str());
    }
} // namespace lfs::media::detail
// Standalone diagnostics never initialize an app logger or write user files.
#define LOG_INFO(...)  ((void)0)
#define LOG_DEBUG(...) ((void)0)
#define LOG_WARN(...)  ::lfs::media::detail::warning(__VA_ARGS__)
#define LOG_ERROR(...) ::lfs::media::detail::warning(__VA_ARGS__)

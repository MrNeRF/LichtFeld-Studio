/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include <cstdint>
#if !defined(__CUDACC__)
#include <source_location>
#endif

namespace lfs::core {

    // CUDA-safe caller metadata. This type has one scalar-only definition in
    // every translation unit; only the caller-capture expression is guarded.
    class SourceSite {
    public:
        constexpr SourceSite(const char* file,
                             const uint_least32_t line,
                             const char* function) noexcept
            : file_(file),
              function_(function),
              line_(line) {
        }

        [[nodiscard]] constexpr const char* file_name() const noexcept { return file_; }
        [[nodiscard]] constexpr const char* function_name() const noexcept { return function_; }
        [[nodiscard]] constexpr uint_least32_t line() const noexcept { return line_; }

    private:
        const char* file_;
        const char* function_;
        uint_least32_t line_;
    };

#if !defined(__CUDACC__)
    [[nodiscard]] constexpr SourceSite source_site_from_location(
        const std::source_location& location) noexcept {
        return SourceSite{
            location.file_name(), location.line(), location.function_name()};
    }
#endif

} // namespace lfs::core

// std::source_location remains the exact host capture mechanism. nvcc never
// parses that branch and receives only scalar built-ins at the call site.
#if defined(__CUDACC__)
#if defined(_MSC_VER)
#define LFS_SOURCE_SITE_CURRENT() \
    ::lfs::core::SourceSite(__builtin_FILE(), __builtin_LINE(), __builtin_FUNCSIG())
#else
#define LFS_SOURCE_SITE_CURRENT() \
    ::lfs::core::SourceSite(__builtin_FILE(), __builtin_LINE(), __builtin_FUNCTION())
#endif
#else
#define LFS_SOURCE_SITE_CURRENT() \
    ::lfs::core::source_site_from_location(std::source_location::current())
#endif

/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <algorithm>
#include <filesystem>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

namespace lfs::io::project::detail::project_fs {

    // Keep user-visible paths unchanged. Only the Windows I/O boundary uses the
    // extended namespace, independently of the machine's LongPathsEnabled policy.
    [[nodiscard]] inline std::filesystem::path native_path(const std::filesystem::path& path) {
#ifdef _WIN32
        if (path.empty()) {
            return path;
        }
        const auto& name = path.native();
        if (name.starts_with(L"\\\\?\\") || name.starts_with(L"\\\\.\\")) {
            return path;
        }
        std::error_code error;
        auto absolute = std::filesystem::absolute(path, error);
        if (error || absolute.native().size() < 248) {
            return path;
        }
        absolute = absolute.lexically_normal();
        absolute.make_preferred();
        const auto& normalized = absolute.native();
        if (normalized.starts_with(L"\\\\")) {
            return std::filesystem::path(L"\\\\?\\UNC\\" + normalized.substr(2));
        }
        return std::filesystem::path(L"\\\\?\\" + normalized);
#else
        return path;
#endif
    }

    [[nodiscard]] inline std::filesystem::path display_path(const std::filesystem::path& path) {
#ifdef _WIN32
        const auto& name = path.native();
        if (name.starts_with(L"\\\\?\\UNC\\")) {
            return std::filesystem::path(L"\\\\" + name.substr(8));
        }
        if (name.starts_with(L"\\\\?\\") && name.size() >= 7 && name[5] == L':') {
            return std::filesystem::path(name.substr(4));
        }
#endif
        return path;
    }

    template <typename... Args>
    inline auto exists(const std::filesystem::path& path, Args&&... args) {
        return std::filesystem::exists(native_path(path), std::forward<Args>(args)...);
    }

    template <typename... Args>
    inline auto is_regular_file(const std::filesystem::path& path, Args&&... args) {
        return std::filesystem::is_regular_file(native_path(path), std::forward<Args>(args)...);
    }

    template <typename... Args>
    inline auto is_directory(const std::filesystem::path& path, Args&&... args) {
        return std::filesystem::is_directory(native_path(path), std::forward<Args>(args)...);
    }

    template <typename... Args>
    inline auto remove(const std::filesystem::path& path, Args&&... args) {
        return std::filesystem::remove(native_path(path), std::forward<Args>(args)...);
    }

    template <typename... Args>
    inline auto create_directories(const std::filesystem::path& path, Args&&... args) {
        return std::filesystem::create_directories(native_path(path), std::forward<Args>(args)...);
    }

    template <typename... Args>
    inline auto space(const std::filesystem::path& path, Args&&... args) {
        return std::filesystem::space(native_path(path), std::forward<Args>(args)...);
    }

    template <typename... Args>
    inline auto last_write_time(const std::filesystem::path& path, Args&&... args) {
        return std::filesystem::last_write_time(native_path(path), std::forward<Args>(args)...);
    }

#ifdef _WIN32
    // Length of the part of an extended path that has no parent to search: \\?\C:\ or \\?\UNC\server\share.
    [[nodiscard]] inline std::size_t extended_root_length(const std::wstring& name) {
        constexpr std::wstring_view unc = L"\\\\?\\UNC\\";
        if (name.starts_with(unc)) {
            const auto server_end = std::min(name.find(L'\\', unc.size()), name.size());
            return std::min(name.find(L'\\', server_end + 1), name.size());
        }
        return std::min(name.find(L'\\', 4), name.size()) + 1;
    }
#endif

    // std::filesystem::weakly_canonical, except for missing extended-length paths on Windows: MSVC walks
    // those down from \\?\, which it parses as the root of \\?\UNC\server\share, so the walk opens the
    // network redirector and fails with ERROR_INVALID_FUNCTION. They are resolved from their deepest
    // existing folder instead, never searching above the drive or share.
    inline std::filesystem::path weakly_canonical(const std::filesystem::path& path, std::error_code& error) {
        auto native = native_path(path);
#ifdef _WIN32
        if (native.native().starts_with(L"\\\\?\\")) {
            const auto root = extended_root_length(native.native());
            std::filesystem::path missing;
            while (!std::filesystem::exists(native, error)) {
                if (error) {
                    return {};
                }
                if (native.native().size() <= root) {
                    error = std::make_error_code(std::errc::no_such_file_or_directory);
                    return {};
                }
                missing = missing.empty() ? native.filename() : native.filename() / missing;
                native = native.parent_path();
            }
            auto resolved = std::filesystem::canonical(native, error);
            if (error) {
                return {};
            }
            if (!missing.empty()) {
                resolved /= missing;
            }
            return display_path(resolved.lexically_normal());
        }
#endif
        return display_path(std::filesystem::weakly_canonical(native, error));
    }

    template <typename... Args>
    inline auto copy_file(const std::filesystem::path& source, const std::filesystem::path& destination, Args&&... args) {
        return std::filesystem::copy_file(native_path(source), native_path(destination), std::forward<Args>(args)...);
    }

    inline void rename(const std::filesystem::path& source, const std::filesystem::path& destination, std::error_code& error) {
        std::filesystem::rename(native_path(source), native_path(destination), error);
    }

} // namespace lfs::io::project::detail::project_fs

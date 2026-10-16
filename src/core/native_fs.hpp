/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <algorithm>
#include <filesystem>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <utility>

namespace lfs::core::native_fs {

    // The \\?\ or \\?\UNC\ spelling of an absolute path once it reaches `threshold` UTF-16 units.
    [[nodiscard]] inline std::filesystem::path extended_path(const std::filesystem::path& path,
                                                             [[maybe_unused]] const std::size_t threshold) {
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
        if (error || absolute.native().size() < threshold) {
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

    // Keep user-visible paths unchanged. Only the Windows I/O boundary uses the
    // extended namespace, independently of the machine's LongPathsEnabled policy.
    [[nodiscard]] inline std::filesystem::path native_path(const std::filesystem::path& path) {
        return extended_path(path, 248);
    }

    // Directory walks build their descendants' paths from the root, so the root is extended whatever its
    // length: a short folder can hold entries far past the limit.
    [[nodiscard]] inline std::filesystem::path traversal_path(const std::filesystem::path& path) {
        return extended_path(path, 0);
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

    // An entry found by iterating traversal_path(root), spelled under `root` as the caller gave it.
    [[nodiscard]] inline std::filesystem::path entry_path(const std::filesystem::path& root,
                                                          const std::filesystem::path& entry) {
        return root / entry.lexically_relative(traversal_path(root));
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

    // std::filesystem::relative with the resolution above, so extended operands never reach MSVC's walk.
    inline std::filesystem::path relative(const std::filesystem::path& path, const std::filesystem::path& base,
                                          std::error_code& error) {
#ifdef _WIN32
        const auto resolved = lfs::core::native_fs::weakly_canonical(path, error);
        if (error) {
            return {};
        }
        const auto resolved_base = lfs::core::native_fs::weakly_canonical(base, error);
        if (error) {
            return {};
        }
        return resolved.lexically_relative(resolved_base);
#else
        return std::filesystem::relative(path, base, error);
#endif
    }

    template <typename... Args>
    inline constexpr bool reports_error_code = (std::is_same_v<std::remove_cvref_t<Args>, std::error_code> || ...);

    // Runs the standard throwing operation on native operands and reports a failure with the caller's paths, so
    // error text never shows the extended spelling while every other behaviour stays the standard one.
    template <typename Operation>
    decltype(auto) rethrow_logical([[maybe_unused]] const char* const name,
                                   [[maybe_unused]] const std::filesystem::path& first,
                                   [[maybe_unused]] const std::filesystem::path* const second, Operation&& operation) {
#ifdef _WIN32
        try {
            return operation();
        } catch (const std::filesystem::filesystem_error& error) {
            if (second != nullptr)
                throw std::filesystem::filesystem_error(name, first, *second, error.code());
            throw std::filesystem::filesystem_error(name, first, error.code());
        }
#else
        return operation();
#endif
    }

    // Throwing queries on an entry found by iterating traversal_path(root), reported under the caller's root.
    inline bool entry_is_regular_file(const std::filesystem::path& root, const std::filesystem::directory_entry& entry) {
        return rethrow_logical("is_regular_file", entry_path(root, entry.path()), nullptr,
                               [&] { return entry.is_regular_file(); });
    }

    inline std::filesystem::file_status entry_symlink_status(const std::filesystem::path& root,
                                                             const std::filesystem::directory_entry& entry) {
        return rethrow_logical("symlink_status", entry_path(root, entry.path()), nullptr,
                               [&] { return entry.symlink_status(); });
    }

    template <typename... Args>
    inline auto exists(const std::filesystem::path& path, Args&&... args) {
        if constexpr (reports_error_code<Args...>)
            return std::filesystem::exists(native_path(path), std::forward<Args>(args)...);
        else
            return rethrow_logical("exists", path, nullptr, [&] {
                return std::filesystem::exists(native_path(path), std::forward<Args>(args)...);
            });
    }

    template <typename... Args>
    inline auto is_regular_file(const std::filesystem::path& path, Args&&... args) {
        if constexpr (reports_error_code<Args...>)
            return std::filesystem::is_regular_file(native_path(path), std::forward<Args>(args)...);
        else
            return rethrow_logical("is_regular_file", path, nullptr, [&] {
                return std::filesystem::is_regular_file(native_path(path), std::forward<Args>(args)...);
            });
    }

    template <typename... Args>
    inline auto is_directory(const std::filesystem::path& path, Args&&... args) {
        if constexpr (reports_error_code<Args...>)
            return std::filesystem::is_directory(native_path(path), std::forward<Args>(args)...);
        else
            return rethrow_logical("is_directory", path, nullptr, [&] {
                return std::filesystem::is_directory(native_path(path), std::forward<Args>(args)...);
            });
    }

    template <typename... Args>
    inline auto is_empty(const std::filesystem::path& path, Args&&... args) {
        if constexpr (reports_error_code<Args...>)
            return std::filesystem::is_empty(native_path(path), std::forward<Args>(args)...);
        else
            return rethrow_logical("is_empty", path, nullptr, [&] {
                return std::filesystem::is_empty(native_path(path), std::forward<Args>(args)...);
            });
    }

    template <typename... Args>
    inline auto status(const std::filesystem::path& path, Args&&... args) {
        if constexpr (reports_error_code<Args...>)
            return std::filesystem::status(native_path(path), std::forward<Args>(args)...);
        else
            return rethrow_logical("status", path, nullptr, [&] {
                return std::filesystem::status(native_path(path), std::forward<Args>(args)...);
            });
    }

    template <typename... Args>
    inline auto file_size(const std::filesystem::path& path, Args&&... args) {
        if constexpr (reports_error_code<Args...>)
            return std::filesystem::file_size(native_path(path), std::forward<Args>(args)...);
        else
            return rethrow_logical("file_size", path, nullptr, [&] {
                return std::filesystem::file_size(native_path(path), std::forward<Args>(args)...);
            });
    }

    template <typename... Args>
    inline auto last_write_time(const std::filesystem::path& path, Args&&... args) {
        if constexpr (reports_error_code<Args...>)
            return std::filesystem::last_write_time(native_path(path), std::forward<Args>(args)...);
        else
            return rethrow_logical("last_write_time", path, nullptr, [&] {
                return std::filesystem::last_write_time(native_path(path), std::forward<Args>(args)...);
            });
    }

    template <typename... Args>
    inline auto space(const std::filesystem::path& path, Args&&... args) {
        if constexpr (reports_error_code<Args...>)
            return std::filesystem::space(native_path(path), std::forward<Args>(args)...);
        else
            return rethrow_logical("space", path, nullptr, [&] {
                return std::filesystem::space(native_path(path), std::forward<Args>(args)...);
            });
    }

    template <typename... Args>
    inline auto resize_file(const std::filesystem::path& path, Args&&... args) {
        if constexpr (reports_error_code<Args...>)
            return std::filesystem::resize_file(native_path(path), std::forward<Args>(args)...);
        else
            return rethrow_logical("resize_file", path, nullptr, [&] {
                return std::filesystem::resize_file(native_path(path), std::forward<Args>(args)...);
            });
    }

    template <typename... Args>
    inline auto create_directory(const std::filesystem::path& path, Args&&... args) {
        if constexpr (reports_error_code<Args...>)
            return std::filesystem::create_directory(native_path(path), std::forward<Args>(args)...);
        else
            return rethrow_logical("create_directory", path, nullptr, [&] {
                return std::filesystem::create_directory(native_path(path), std::forward<Args>(args)...);
            });
    }

    template <typename... Args>
    inline auto create_directories(const std::filesystem::path& path, Args&&... args) {
        if constexpr (reports_error_code<Args...>)
            return std::filesystem::create_directories(native_path(path), std::forward<Args>(args)...);
        else
            return rethrow_logical("create_directories", path, nullptr, [&] {
                return std::filesystem::create_directories(native_path(path), std::forward<Args>(args)...);
            });
    }

    template <typename... Args>
    inline auto remove(const std::filesystem::path& path, Args&&... args) {
        if constexpr (reports_error_code<Args...>)
            return std::filesystem::remove(native_path(path), std::forward<Args>(args)...);
        else
            return rethrow_logical("remove", path, nullptr, [&] {
                return std::filesystem::remove(native_path(path), std::forward<Args>(args)...);
            });
    }

    template <typename... Args>
    inline auto remove_all(const std::filesystem::path& path, Args&&... args) {
        if constexpr (reports_error_code<Args...>)
            return std::filesystem::remove_all(traversal_path(path), std::forward<Args>(args)...);
        else
            return rethrow_logical("remove_all", path, nullptr, [&] {
                return std::filesystem::remove_all(traversal_path(path), std::forward<Args>(args)...);
            });
    }

    template <typename... Args>
    inline auto copy(const std::filesystem::path& source, const std::filesystem::path& destination, Args&&... args) {
        if constexpr (reports_error_code<Args...>)
            return std::filesystem::copy(traversal_path(source), traversal_path(destination), std::forward<Args>(args)...);
        else
            return rethrow_logical("copy", source, &destination, [&] {
                return std::filesystem::copy(traversal_path(source), traversal_path(destination), std::forward<Args>(args)...);
            });
    }

    template <typename... Args>
    inline auto copy_file(const std::filesystem::path& source, const std::filesystem::path& destination, Args&&... args) {
        if constexpr (reports_error_code<Args...>)
            return std::filesystem::copy_file(native_path(source), native_path(destination), std::forward<Args>(args)...);
        else
            return rethrow_logical("copy_file", source, &destination, [&] {
                return std::filesystem::copy_file(native_path(source), native_path(destination), std::forward<Args>(args)...);
            });
    }

    template <typename... Args>
    inline auto rename(const std::filesystem::path& source, const std::filesystem::path& destination, Args&&... args) {
        if constexpr (reports_error_code<Args...>)
            return std::filesystem::rename(native_path(source), native_path(destination), std::forward<Args>(args)...);
        else
            return rethrow_logical("rename", source, &destination, [&] {
                return std::filesystem::rename(native_path(source), native_path(destination), std::forward<Args>(args)...);
            });
    }

    template <typename... Args>
    inline auto equivalent(const std::filesystem::path& source, const std::filesystem::path& destination, Args&&... args) {
        if constexpr (reports_error_code<Args...>)
            return std::filesystem::equivalent(native_path(source), native_path(destination), std::forward<Args>(args)...);
        else
            return rethrow_logical("equivalent", source, &destination, [&] {
                return std::filesystem::equivalent(native_path(source), native_path(destination), std::forward<Args>(args)...);
            });
    }

    // Iterate traversal_path(root) and spell entries with entry_path(root, entry.path()).
    inline std::filesystem::directory_iterator
    directory_iterator(const std::filesystem::path& root,
                       const std::filesystem::directory_options options = std::filesystem::directory_options::none) {
        return rethrow_logical("directory_iterator", root, nullptr, [&] {
            return std::filesystem::directory_iterator(traversal_path(root), options);
        });
    }

    inline std::filesystem::recursive_directory_iterator
    recursive_directory_iterator(const std::filesystem::path& root,
                                 const std::filesystem::directory_options options = std::filesystem::directory_options::none) {
        return rethrow_logical("recursive_directory_iterator", root, nullptr, [&] {
            return std::filesystem::recursive_directory_iterator(traversal_path(root), options);
        });
    }

} // namespace lfs::core::native_fs

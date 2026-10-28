/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <algorithm>
#include <fstream>
#include <optional>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif
#include <filesystem>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <utility>

namespace lfs::core::native_fs {
    using namespace std::filesystem;
    namespace fs = lfs::core::native_fs;
    [[nodiscard]] inline path display_path(const path& value);
    [[nodiscard]] inline std::optional<bool> process_long_paths_enabled() noexcept {
#ifdef _WIN32
        using Query = BOOLEAN(NTAPI*)();
        const auto module = GetModuleHandleW(L"ntdll.dll");
        const auto query = module ? reinterpret_cast<Query>(GetProcAddress(module, "RtlAreLongPathsEnabled")) : nullptr;
        if (query)
            return query() != FALSE;
#endif
        return std::nullopt;
    }
#ifdef _WIN32
    namespace detail {
        inline bool extended_unc_path(const std::wstring_view name) noexcept {
            return name.size() >= 8 && name.starts_with(L"\\\\?\\") &&
                   (name[4] == L'U' || name[4] == L'u') &&
                   (name[5] == L'N' || name[5] == L'n') &&
                   (name[6] == L'C' || name[6] == L'c') && name[7] == L'\\';
        }
        inline bool dos_device_name(std::wstring_view value) {
            value = value.substr(0, value.find(L'.'));
            while (!value.empty() && (value.back() == L' ' || value.back() == L'.'))
                value.remove_suffix(1);
            if (value.size() != 3 && value.size() != 4)
                return false;
            const auto at = [&](size_t i) { auto ch = value[i]; return ch >= L'a' && ch <= L'z' ? wchar_t(ch - (L'a' - L'A')) : ch; };
            if (value.size() == 3)
                return (at(0) == L'C' && at(1) == L'O' && at(2) == L'N') ||
                       (at(0) == L'P' && at(1) == L'R' && at(2) == L'N') ||
                       (at(0) == L'A' && at(1) == L'U' && at(2) == L'X') ||
                       (at(0) == L'N' && at(1) == L'U' && at(2) == L'L');
            return ((at(0) == L'C' && at(1) == L'O' && at(2) == L'M') ||
                    (at(0) == L'L' && at(1) == L'P' && at(2) == L'T')) &&
                   ((value[3] >= L'1' && value[3] <= L'9') || value[3] == L'\u00b9' || value[3] == L'\u00b2' || value[3] == L'\u00b3');
        }
        inline bool bare_device(const path& value) {
            return !value.has_parent_path() && !value.has_extension() && dos_device_name(value.native());
        }
        inline bool needs_verbatim(const path& value) {
            const std::wstring_view name = value.native();
            size_t begin = 0;
            while (begin < name.size()) {
                const auto end = name.find_first_of(L"\\/", begin);
                const auto part = name.substr(begin, end == name.npos ? name.npos : end - begin);
                if (!part.empty() && part != L"." && part != L".." &&
                    (part.back() == L'.' || part.back() == L' ' || dos_device_name(part)))
                    return true;
                if (end == name.npos)
                    break;
                begin = end + 1;
            }
            return false;
        }
        inline path absolute_preserving_names(const path& value, std::error_code& error) {
            if (value.is_absolute())
                return display_path(value);
            if (value.has_root_name() && !value.has_root_directory()) {
                const auto base = display_path(std::filesystem::absolute(value.root_name() / L".", error));
                return error ? path{} : base / value.relative_path();
            }
            const auto base = display_path(std::filesystem::current_path(error));
            return error ? path{} : base / value;
        }
    } // namespace detail
#endif

    // The \\?\ or \\?\UNC\ spelling of an absolute path once it reaches `threshold` UTF-16 units.
    [[nodiscard]] inline path extended_path(const path& value, [[maybe_unused]] const std::size_t threshold) {
#ifdef _WIN32
        const bool force_extended = threshold == 0;
        if (value.empty())
            return value;
        const auto& name = value.native();
        if (name.starts_with(L"\\\\?\\") || name.starts_with(L"\\\\.\\"))
            return value;
        // Keep explicit device syntax and legacy bare NUL/CON destinations.
        if (!force_extended && detail::bare_device(value))
            return value;
        const bool verbatim = force_extended || detail::needs_verbatim(value);
        if (!verbatim && value.is_absolute() && name.size() < threshold)
            return value;
        std::error_code error;
        auto absolute = detail::absolute_preserving_names(value, error);
        if (error || (!verbatim && !detail::needs_verbatim(absolute.parent_path()) && absolute.native().size() < threshold))
            return value;
        absolute = absolute.lexically_normal();
        absolute.make_preferred();
        const auto& normalized = absolute.native();
        if (normalized.starts_with(L"\\\\"))
            return path(L"\\\\?\\UNC\\" + normalized.substr(2));
        return path(L"\\\\?\\" + normalized);
#else
        return value;
#endif
    }

    // Keep user-visible paths unchanged. Only the Windows I/O boundary uses the
    // extended namespace, independently of the machine's LongPathsEnabled policy.
    [[nodiscard]] inline path native_path(const path& value, const bool force_extended = false) {
        return extended_path(value, force_extended ? 0 : 248);
    }

    // Directory walks build their descendants' paths from the root, so the root is extended whatever its
    // length: a short folder can hold entries far past the limit.
    [[nodiscard]] inline std::filesystem::path traversal_path(const std::filesystem::path& path) {
        return extended_path(path, 0);
    }

    [[nodiscard]] inline path display_path(const path& value) {
#ifdef _WIN32
        const auto& name = value.native();
        if (detail::extended_unc_path(name))
            return path(L"\\\\" + name.substr(8));
        if (name.starts_with(L"\\\\?\\") && name.size() >= 7 && name[5] == L':')
            return path(name.substr(4));
#endif
        return value;
    }

    // An entry found by iterating traversal_path(root), spelled under `root` as the caller gave it.
    [[nodiscard]] inline std::filesystem::path entry_path([[maybe_unused]] const std::filesystem::path& root,
                                                          const std::filesystem::path& entry) {
#ifdef _WIN32
        return root / entry.lexically_relative(traversal_path(root));
#else
        // POSIX walks the root unchanged, so the entry already is spelled under it.
        return entry;
#endif
    }

#ifdef _WIN32
    // Length of the part of an extended path that has no parent to search: \\?\C:\ or \\?\UNC\server\share.
    [[nodiscard]] inline std::size_t extended_root_length(const std::wstring& name) {
        constexpr std::wstring_view unc = L"\\\\?\\UNC\\";
        if (detail::extended_unc_path(name)) {
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
            // An existing path resolves directly. Only a missing tail needs the walk.
            if (auto resolved = std::filesystem::canonical(native, error); !error) {
                return display_path(resolved.lexically_normal());
            }
            error.clear();
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
    // The caller's spelling is only built for an error: walks call these once per entry.
    inline bool entry_is_regular_file([[maybe_unused]] const std::filesystem::path& root,
                                      const std::filesystem::directory_entry& entry) {
#ifdef _WIN32
        try {
            return entry.is_regular_file();
        } catch (const std::filesystem::filesystem_error& error) {
            throw std::filesystem::filesystem_error("is_regular_file", entry_path(root, entry.path()), error.code());
        }
#else
        return entry.is_regular_file();
#endif
    }

    inline bool entry_is_directory([[maybe_unused]] const std::filesystem::path& root,
                                   const std::filesystem::directory_entry& entry) {
#ifdef _WIN32
        try {
            return entry.is_directory();
        } catch (const std::filesystem::filesystem_error& error) {
            throw std::filesystem::filesystem_error("is_directory", entry_path(root, entry.path()), error.code());
        }
#else
        return entry.is_directory();
#endif
    }

    inline std::filesystem::file_status entry_symlink_status([[maybe_unused]] const std::filesystem::path& root,
                                                             const std::filesystem::directory_entry& entry) {
#ifdef _WIN32
        try {
            return entry.symlink_status();
        } catch (const std::filesystem::filesystem_error& error) {
            throw std::filesystem::filesystem_error("symlink_status", entry_path(root, entry.path()), error.code());
        }
#else
        return entry.symlink_status();
#endif
    }

    template <typename... Args>
    inline auto exists(const std::filesystem::path& path, Args&&... args) {
#ifndef _WIN32
        // This hot path is used by per-sample Camera sidecar checks.
        // POSIX paths already are native: avoid copying their storage.
        return std::filesystem::exists(path, std::forward<Args>(args)...);
#else
        if constexpr (reports_error_code<Args...>)
            return std::filesystem::exists(native_path(path), std::forward<Args>(args)...);
        else
            return rethrow_logical("exists", path, nullptr, [&] {
                return std::filesystem::exists(native_path(path), std::forward<Args>(args)...);
            });
#endif
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
#ifdef _WIN32
        // Scanners and SMB servers can hold a just-closed file without delete sharing for a moment, and MoveFileExW
        // then fails with access denied or a sharing violation. Retry those for up to a second before reporting.
        const auto native_source = native_path(source), native_destination = native_path(destination);
        std::error_code error;
        for (int attempt = 1;; ++attempt) {
            std::filesystem::rename(native_source, native_destination, error);
            const bool held = error.category() == std::system_category() &&
                              (error.value() == ERROR_ACCESS_DENIED || error.value() == ERROR_SHARING_VIOLATION ||
                               error.value() == ERROR_LOCK_VIOLATION);
            if (!held || attempt == 20)
                break;
            Sleep(50);
        }
        if constexpr (reports_error_code<Args...>)
            ((args = error), ...);
        else if (error)
            throw std::filesystem::filesystem_error("rename", source, destination, error);
#else
        return std::filesystem::rename(source, destination, std::forward<Args>(args)...);
#endif
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

    inline path weakly_canonical(const path& value) {
        std::error_code error;
        auto result = lfs::core::native_fs::weakly_canonical(value, error);
        if (error)
            throw filesystem_error("weakly_canonical", value, error);
        return result;
    }
    inline path relative(const path& value, const path& base = std::filesystem::current_path()) {
        std::error_code error;
        auto result = lfs::core::native_fs::relative(value, base, error);
        if (error)
            throw filesystem_error("relative", value, base, error);
        return result;
    }
    [[nodiscard]] inline path temp_directory_path(std::error_code& error) {
#ifdef _WIN32
        // GetTempPathW may silently fall back to USERPROFILE when TMP exceeds
        // MAX_PATH with opt-in disabled. Read its documented precedence directly.
        path result;
        for (const auto* key : {L"TMP", L"TEMP", L"USERPROFILE"}) {
            const DWORD required = GetEnvironmentVariableW(key, nullptr, 0);
            if (!required)
                continue;
            std::wstring buffer(required, L'\0');
            const DWORD length = GetEnvironmentVariableW(key, buffer.data(), required);
            if (!length || length >= required) {
                error = std::error_code(length == 0 ? GetLastError() : ERROR_INSUFFICIENT_BUFFER,
                                        std::system_category());
                return {};
            }
            buffer.resize(length);
            result = display_path(path(buffer));
            break;
        }
        if (result.empty()) {
            std::wstring buffer(32768, L'\0');
            const UINT length = GetWindowsDirectoryW(buffer.data(), static_cast<UINT>(buffer.size()));
            if (!length || length >= buffer.size()) {
                error = std::error_code(length == 0 ? GetLastError() : ERROR_INSUFFICIENT_BUFFER,
                                        std::system_category());
                return {};
            }
            buffer.resize(length);
            result = path(buffer);
        }
        result = display_path(std::filesystem::absolute(result, error));
        if (error)
            return {};
        if (!std::filesystem::is_directory(native_path(result), error)) {
            if (!error)
                error = std::make_error_code(std::errc::not_a_directory);
            return {};
        }
        error.clear();
        return result;
#else
        return std::filesystem::temp_directory_path(error);
#endif
    }
    [[nodiscard]] inline path temp_directory_path() {
        std::error_code error;
        auto result = temp_directory_path(error);
        if (error)
            throw filesystem_error("Cannot locate temporary directory", error);
        return result;
    }
    template <typename... Args>
    inline auto is_symlink(const path& value, Args&&... args) {
        if constexpr (reports_error_code<Args...>)
            return std::filesystem::is_symlink(native_path(value), std::forward<Args>(args)...);
        else
            return rethrow_logical("is_symlink", value, nullptr, [&] {
                return std::filesystem::is_symlink(native_path(value), std::forward<Args>(args)...);
            });
    }
    template <typename... Args>
    inline auto is_block_file(const path& value, Args&&... args) {
        if constexpr (reports_error_code<Args...>)
            return std::filesystem::is_block_file(native_path(value), std::forward<Args>(args)...);
        else
            return rethrow_logical("is_block_file", value, nullptr, [&] {
                return std::filesystem::is_block_file(native_path(value), std::forward<Args>(args)...);
            });
    }
    template <typename... Args>
    inline auto is_character_file(const path& value, Args&&... args) {
        if constexpr (reports_error_code<Args...>)
            return std::filesystem::is_character_file(native_path(value), std::forward<Args>(args)...);
        else
            return rethrow_logical("is_character_file", value, nullptr, [&] {
                return std::filesystem::is_character_file(native_path(value), std::forward<Args>(args)...);
            });
    }
    template <typename... Args>
    inline auto is_fifo(const path& value, Args&&... args) {
        if constexpr (reports_error_code<Args...>)
            return std::filesystem::is_fifo(native_path(value), std::forward<Args>(args)...);
        else
            return rethrow_logical("is_fifo", value, nullptr, [&] {
                return std::filesystem::is_fifo(native_path(value), std::forward<Args>(args)...);
            });
    }
    template <typename... Args>
    inline auto is_socket(const path& value, Args&&... args) {
        if constexpr (reports_error_code<Args...>)
            return std::filesystem::is_socket(native_path(value), std::forward<Args>(args)...);
        else
            return rethrow_logical("is_socket", value, nullptr, [&] {
                return std::filesystem::is_socket(native_path(value), std::forward<Args>(args)...);
            });
    }
    template <typename... Args>
    inline auto is_other(const path& value, Args&&... args) {
        if constexpr (reports_error_code<Args...>)
            return std::filesystem::is_other(native_path(value), std::forward<Args>(args)...);
        else
            return rethrow_logical("is_other", value, nullptr, [&] {
                return std::filesystem::is_other(native_path(value), std::forward<Args>(args)...);
            });
    }

    template <typename... Args>
    inline auto symlink_status(const path& value, Args&&... args) {
        if constexpr (reports_error_code<Args...>)
            return std::filesystem::symlink_status(native_path(value), std::forward<Args>(args)...);
        else
            return rethrow_logical("symlink_status", value, nullptr, [&] {
                return std::filesystem::symlink_status(native_path(value), std::forward<Args>(args)...);
            });
    }
    template <typename... Args>
    inline auto hard_link_count(const path& value, Args&&... args) {
        if constexpr (reports_error_code<Args...>)
            return std::filesystem::hard_link_count(native_path(value), std::forward<Args>(args)...);
        else
            return rethrow_logical("hard_link_count", value, nullptr, [&] {
                return std::filesystem::hard_link_count(native_path(value), std::forward<Args>(args)...);
            });
    }
    template <typename... Args>
    inline auto permissions(const path& value, Args&&... args) {
        if constexpr (reports_error_code<Args...>)
            return std::filesystem::permissions(native_path(value), std::forward<Args>(args)...);
        else
            return rethrow_logical("permissions", value, nullptr, [&] {
                return std::filesystem::permissions(native_path(value), std::forward<Args>(args)...);
            });
    }

    template <typename... Args>
    inline auto copy_symlink(const path& source, const path& destination, Args&&... args) {
        if constexpr (reports_error_code<Args...>)
            return std::filesystem::copy_symlink(native_path(source), native_path(destination), std::forward<Args>(args)...);
        else
            return rethrow_logical("copy_symlink", source, &destination, [&] {
                return std::filesystem::copy_symlink(native_path(source), native_path(destination), std::forward<Args>(args)...);
            });
    }
    template <typename... Args>
    inline auto create_hard_link(const path& source, const path& destination, Args&&... args) {
        if constexpr (reports_error_code<Args...>)
            return std::filesystem::create_hard_link(native_path(source), native_path(destination), std::forward<Args>(args)...);
        else
            return rethrow_logical("create_hard_link", source, &destination, [&] {
                return std::filesystem::create_hard_link(native_path(source), native_path(destination), std::forward<Args>(args)...);
            });
    }
    template <typename... Args>
    inline auto create_symlink(const path& source, const path& destination, Args&&... args) {
        if constexpr (reports_error_code<Args...>)
            return std::filesystem::create_symlink(source, native_path(destination), std::forward<Args>(args)...);
        else
            return rethrow_logical("create_symlink", source, &destination, [&] {
                return std::filesystem::create_symlink(source, native_path(destination), std::forward<Args>(args)...);
            });
    }
    template <typename... Args>
    inline auto create_directory_symlink(const path& source, const path& destination, Args&&... args) {
        if constexpr (reports_error_code<Args...>)
            return std::filesystem::create_directory_symlink(source, native_path(destination), std::forward<Args>(args)...);
        else
            return rethrow_logical("create_directory_symlink", source, &destination, [&] {
                return std::filesystem::create_directory_symlink(source, native_path(destination), std::forward<Args>(args)...);
            });
    }
    inline bool exists(file_status value) noexcept { return std::filesystem::exists(value); }
    inline bool is_regular_file(file_status value) noexcept { return std::filesystem::is_regular_file(value); }
    inline bool is_directory(file_status value) noexcept { return std::filesystem::is_directory(value); }
    inline bool is_symlink(file_status value) noexcept { return std::filesystem::is_symlink(value); }
    inline bool is_block_file(file_status value) noexcept { return std::filesystem::is_block_file(value); }
    inline bool is_character_file(file_status value) noexcept { return std::filesystem::is_character_file(value); }
    inline bool is_fifo(file_status value) noexcept { return std::filesystem::is_fifo(value); }
    inline bool is_socket(file_status value) noexcept { return std::filesystem::is_socket(value); }
    inline bool is_other(file_status value) noexcept { return std::filesystem::is_other(value); }
    inline bool create_directory(const path& value, const path& attributes) {
        return rethrow_logical("create_directory", value, &attributes, [&] {
            return std::filesystem::create_directory(native_path(value), native_path(attributes));
        });
    }
    inline bool create_directory(const path& value, const path& attributes, std::error_code& error) {
        return std::filesystem::create_directory(native_path(value), native_path(attributes), error);
    }
    inline path canonical(const path& value, std::error_code& error) {
        return display_path(std::filesystem::canonical(native_path(value), error));
    }
    inline path canonical(const path& value) {
        return rethrow_logical("canonical", value, nullptr, [&] {
            return display_path(std::filesystem::canonical(native_path(value)));
        });
    }
    inline path read_symlink(const path& value, std::error_code& error) {
        return display_path(std::filesystem::read_symlink(native_path(value), error));
    }
    inline path read_symlink(const path& value) {
        return rethrow_logical("read_symlink", value, nullptr, [&] {
            return display_path(std::filesystem::read_symlink(native_path(value)));
        });
    }
    inline path current_path() { return display_path(std::filesystem::current_path()); }
    inline path current_path(std::error_code& error) { return display_path(std::filesystem::current_path(error)); }
    inline void current_path(const path& value) {
        rethrow_logical("current_path", value, nullptr, [&] {
            std::filesystem::current_path(native_path(value));
        });
    }
    inline void current_path(const path& value, std::error_code& error) { std::filesystem::current_path(native_path(value), error); }

    [[nodiscard]] inline path from_utf8(std::string_view value) {
        return path(std::u8string_view(reinterpret_cast<const char8_t*>(value.data()), value.size()));
    }

    inline std::filesystem::directory_iterator directory_iterator(const path& root, std::error_code& error) {
        return std::filesystem::directory_iterator(traversal_path(root), error);
    }
    inline std::filesystem::directory_iterator directory_iterator(const path& root, directory_options options, std::error_code& error) {
        return std::filesystem::directory_iterator(traversal_path(root), options, error);
    }
    inline std::filesystem::recursive_directory_iterator recursive_directory_iterator(const path& root, std::error_code& error) {
        return std::filesystem::recursive_directory_iterator(traversal_path(root), error);
    }
    inline std::filesystem::recursive_directory_iterator recursive_directory_iterator(const path& root, directory_options options, std::error_code& error) {
        return std::filesystem::recursive_directory_iterator(traversal_path(root), options, error);
    }

} // namespace lfs::core::native_fs

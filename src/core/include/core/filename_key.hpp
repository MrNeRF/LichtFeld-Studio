/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/native_fs.hpp"
namespace lfs::core::detail {
    // Stable sibling keys must follow the containing directory's case semantics.
    // In ordinary Windows directories, casing aliases must share one key even
    // before the source exists. Case-sensitive directories keep distinct keys.
    [[nodiscard]] inline native_fs::path artifact_filename_key(const native_fs::path& source) {
#ifdef _WIN32
        auto directory = source.parent_path();
        if (directory.empty())
            directory = native_fs::current_path();
        std::error_code error;
        while (!native_fs::exists(directory, error) && !error && directory.has_relative_path())
            directory = directory.parent_path();
        bool case_sensitive = false;
        if (directory.empty())
            directory = native_fs::current_path();
        const auto handle = CreateFileW(native_fs::native_path(directory).c_str(), FILE_READ_ATTRIBUTES,
                                        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                                        nullptr, OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS, nullptr);
        if (handle != INVALID_HANDLE_VALUE) {
            FILE_CASE_SENSITIVE_INFO info{};
            if (GetFileInformationByHandleEx(handle, FileCaseSensitiveInfo, &info, sizeof(info)))
                case_sensitive = (info.Flags & FILE_CS_FLAG_CASE_SENSITIVE_DIR) != 0;
            CloseHandle(handle);
        }
        if (!case_sensitive) {
            auto units = source.filename().native();
            using Upcase = WCHAR(NTAPI*)(WCHAR);
            const auto upcase = reinterpret_cast<Upcase>(GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "RtlUpcaseUnicodeChar"));
            for (auto& unit : units) {
                if (upcase)
                    unit = upcase(unit);
                else {
                    wchar_t mapped = unit;
                    if (LCMapStringEx(LOCALE_NAME_INVARIANT, LCMAP_UPPERCASE, &unit, 1, &mapped, 1, nullptr, nullptr, 0) == 1)
                        unit = mapped;
                }
            }
            return native_fs::path(units);
        }
#endif
        return source.filename();
    }
} // namespace lfs::core::detail

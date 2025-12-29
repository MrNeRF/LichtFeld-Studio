/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <filesystem>
#include <string>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace lfs::core {

    /**
     * @brief Convert filesystem path to UTF-8 string for use with external libraries
     *
     * On Windows, std::filesystem::path::string() returns a string in the system codepage,
     * not UTF-8. This function ensures the returned string is always UTF-8 encoded.
     * On Linux/Mac, the native encoding is already UTF-8.
     *
     * @param p The filesystem path to convert
     * @return UTF-8 encoded string representation of the path
     */
    inline std::string path_to_utf8(const std::filesystem::path& p) {
#ifdef _WIN32
        // On Windows, convert wide string to UTF-8
        const std::wstring wstr = p.wstring();
        if (wstr.empty()) {
            return std::string();
        }

        const int size_needed = WideCharToMultiByte(CP_UTF8, 0, wstr.c_str(),
                                                    static_cast<int>(wstr.size()),
                                                    nullptr, 0, nullptr, nullptr);
        if (size_needed <= 0) {
            return std::string();
        }

        std::string utf8_str(size_needed, 0);
        const int converted = WideCharToMultiByte(CP_UTF8, 0, wstr.c_str(),
                                                  static_cast<int>(wstr.size()),
                                                  &utf8_str[0], size_needed, nullptr, nullptr);
        if (converted <= 0) {
            return std::string();
        }
        utf8_str.resize(converted);
        return utf8_str;
#else
        // On Linux/Mac, native encoding is UTF-8
        return p.string();
#endif
    }

    /**
     * @brief Convert UTF-8 string to filesystem path
     *
     * On Windows, std::filesystem::path constructor from std::string interprets
     * the string as being in the system codepage. This function properly converts
     * UTF-8 strings to paths by converting to wide string first on Windows.
     * On Linux/Mac, the native encoding is already UTF-8.
     *
     * @param utf8_str UTF-8 encoded string to convert
     * @return filesystem path constructed from the UTF-8 string
     */
    inline std::filesystem::path utf8_to_path(const std::string& utf8_str) {
#ifdef _WIN32
        // On Windows, convert UTF-8 string to wide string, then to path
        if (utf8_str.empty()) {
            return std::filesystem::path();
        }

        const int size_needed = MultiByteToWideChar(CP_UTF8, 0, utf8_str.c_str(),
                                                    static_cast<int>(utf8_str.size()),
                                                    nullptr, 0);
        if (size_needed <= 0) {
            return std::filesystem::path();
        }

        std::wstring wstr(size_needed, 0);
        const int converted = MultiByteToWideChar(CP_UTF8, 0, utf8_str.c_str(),
                                                  static_cast<int>(utf8_str.size()),
                                                  &wstr[0], size_needed);
        if (converted <= 0) {
            return std::filesystem::path();
        }
        wstr.resize(converted);
        return std::filesystem::path(wstr);
#else
        // On Linux/Mac, native encoding is UTF-8, so use string directly
        return std::filesystem::path(utf8_str);
#endif
    }

} // namespace lfs::core

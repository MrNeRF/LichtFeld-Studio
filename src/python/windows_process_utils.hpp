/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <filesystem>
#include <sstream>
#include <string>
#include <vector>

#include <core/path_utils.hpp>

namespace lfs::python::detail {

#ifdef _WIN32
    inline std::wstring quote_windows_cmd_arg(const std::wstring& arg) {
        std::wstring quoted;
        quoted.reserve(arg.size() + 2);
        quoted.push_back(L'"');

        size_t backslash_count = 0;
        for (const wchar_t ch : arg) {
            if (ch == L'\\') {
                ++backslash_count;
                continue;
            }

            if (ch == L'"') {
                quoted.append(backslash_count * 2 + 1, L'\\');
                quoted.push_back(L'"');
                backslash_count = 0;
                continue;
            }

            if (backslash_count > 0) {
                quoted.append(backslash_count, L'\\');
                backslash_count = 0;
            }
            quoted.push_back(ch);
        }

        if (backslash_count > 0) {
            quoted.append(backslash_count * 2, L'\\');
        }
        quoted.push_back(L'"');
        return quoted;
    }

    inline std::wstring build_win32_cmdline(const std::filesystem::path& program,
                                            const std::vector<std::string>& args) {
        std::wstring cmdline = quote_windows_cmd_arg(program.wstring());
        for (const auto& arg : args) {
            cmdline.push_back(L' ');
            cmdline += quote_windows_cmd_arg(lfs::core::utf8_to_wstring(arg));
        }
        return cmdline;
    }
#endif

    inline std::string format_command_for_log(const std::filesystem::path& program,
                                              const std::vector<std::string>& args) {
        std::ostringstream stream;
        stream << '"' << lfs::core::path_to_utf8(program) << '"';
        for (const auto& arg : args) {
            stream << ' ' << '"' << arg << '"';
        }
        return stream.str();
    }

} // namespace lfs::python::detail

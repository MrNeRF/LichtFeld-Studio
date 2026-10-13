/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <filesystem>
#include <string_view>
#include <system_error>
#include <vector>

#ifndef _WIN32
#include <unistd.h>
#endif

namespace lfs::python::detail {

    inline std::filesystem::path resolve_uv_path(
        const std::vector<std::filesystem::path>& bundled_candidates, std::string_view path_list) {
        for (const auto& candidate : bundled_candidates) {
            std::error_code ec;
            if (std::filesystem::is_regular_file(candidate, ec) && !ec)
                return candidate;
        }

#ifdef _WIN32
        constexpr const char* UV_BINARY = "uv.exe";
        constexpr char PATH_SEPARATOR = ';';
#else
        constexpr const char* UV_BINARY = "uv";
        constexpr char PATH_SEPARATOR = ':';
#endif
        size_t start = 0;
        while (start <= path_list.size()) {
            const size_t end = path_list.find(PATH_SEPARATOR, start);
            const auto part = path_list.substr(
                start, end == std::string_view::npos ? path_list.size() - start : end - start);
            if (!part.empty()) {
                const std::filesystem::path directory(part);
                if (!directory.is_absolute()) {
                    if (end == std::string_view::npos)
                        break;
                    start = end + 1;
                    continue;
                }
                const auto candidate = directory / UV_BINARY;
                std::error_code ec;
                if (std::filesystem::is_regular_file(candidate, ec) && !ec) {
#ifndef _WIN32
                    if (access(candidate.c_str(), X_OK) != 0) {
                        if (end == std::string_view::npos)
                            break;
                        start = end + 1;
                        continue;
                    }
#endif
                    return candidate;
                }
            }
            if (end == std::string_view::npos)
                break;
            start = end + 1;
        }
        return {};
    }

} // namespace lfs::python::detail

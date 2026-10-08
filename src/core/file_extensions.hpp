/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/path_utils.hpp"
#include <algorithm>
#include <cctype>
#include <string_view>

namespace lfs::core {
    [[nodiscard]] inline bool has_extension(const std::filesystem::path& path,
                                            const std::string_view extension) {
        const auto suffix = path_to_utf8(path.extension());
        return std::ranges::equal(suffix, extension, [](const unsigned char a, const unsigned char b) {
            return std::tolower(a) == std::tolower(b);
        });
    }
} // namespace lfs::core

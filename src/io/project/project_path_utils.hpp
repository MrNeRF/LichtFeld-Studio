/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/path_utils.hpp"

#include <filesystem>
#include <format>
#include <string_view>

namespace lfs::io::project::detail {

    [[nodiscard]] inline std::filesystem::path save_as_staging_path(
        const std::filesystem::path& destination,
        const std::string_view unique_suffix) {
        auto filename = lfs::core::utf8_to_path(".");
        filename += destination.filename();
        filename += lfs::core::utf8_to_path(
            std::format(".saveas-{}.tmp", unique_suffix));
        return destination.parent_path() / filename;
    }

} // namespace lfs::io::project::detail

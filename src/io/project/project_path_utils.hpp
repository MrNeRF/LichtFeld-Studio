/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/path_utils.hpp"

#include <filesystem>
#include <format>
#include <string>
#include <string_view>

namespace lfs::io::project::detail {

    [[nodiscard]] inline std::filesystem::path save_as_staging_path(
        const std::filesystem::path& destination,
        const std::string_view unique_suffix) {
        auto filename = lfs::core::utf8_to_path(".");
        // Compaction adds another suffix to this name. Reserve room for that
        // suffix and its lock file within a 255-byte filesystem component.
        // Keep the saveas marker so recovery still recognizes stale staging files.
        const auto destination_name = destination.filename();
        filename += lfs::core::path_to_utf8(destination_name).size() <= 128
                        ? destination_name
                        : lfs::core::utf8_to_path(std::string(unique_suffix));
        filename += lfs::core::utf8_to_path(
            std::format(".saveas-{}.tmp", unique_suffix));
        return destination.parent_path() / filename;
    }

} // namespace lfs::io::project::detail

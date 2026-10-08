/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/file_extensions.hpp"
#include <filesystem>
#include <string>
#include <string_view>

namespace lfs::vis::gui::detail {

    [[nodiscard]] inline std::filesystem::path appendRequiredExtension(
        std::filesystem::path path,
        const std::string_view extension) {
        if (!path.empty() && !extension.empty() && !core::has_extension(path, extension)) {
            path += std::string(extension);
        }
        return path;
    }

    [[nodiscard]] inline std::string saveDialogDefaultName(const std::string& defaultName,
                                                           const std::string_view extension) {
        auto filename = core::utf8_to_path(defaultName).filename();
        // Native pickers may select only the stem. Keep the suggested name free
        // of the format suffix so typing a complete filename cannot leave one behind.
        if (!extension.empty() && core::has_extension(filename, extension)) {
            filename.replace_extension();
        }
        return core::path_to_utf8(filename);
    }
} // namespace lfs::vis::gui::detail

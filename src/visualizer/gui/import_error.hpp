/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once
#include <algorithm>
#include <cctype>
#include <cstddef>
#include <string>
#include <string_view>

namespace lfs::vis::gui {
    // Matches the one/few/other convention in lfs_plugins/localization.py.
    inline std::string_view importFailureTitleKey(std::string_view language, std::size_t count) {
        if (count == 1)
            return "runtime.import_batch_failed.one";
        if (language == "pl" && count % 10 >= 2 && count % 10 <= 4 &&
            !(count % 100 >= 12 && count % 100 <= 14))
            return "runtime.import_batch_failed.few";
        return "runtime.import_batch_failed.other";
    }

    inline bool isImportOutOfMemory(std::string_view message) {
        std::string lower(message);
        std::transform(lower.begin(), lower.end(), lower.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return lower.contains("out of memory") || lower.contains("out_of_memory") ||
               lower.contains("out of device memory") || lower.contains("out_of_device_memory") || lower.contains("out_of_host_memory") || lower.contains("cudaerrormemoryallocation") ||
               lower.contains("gpu memory exhausted") || lower.contains("not enough gpu memory") || lower.contains("exceeded available gpu memory");
    }
} // namespace lfs::vis::gui

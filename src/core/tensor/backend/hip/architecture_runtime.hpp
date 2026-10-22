/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include <string_view>

inline std::string_view lfs_hip_base_architecture(std::string_view architecture) {
    return architecture.substr(0, architecture.find(':'));
}
inline bool lfs_hip_has_architecture(std::string_view targets, std::string_view architecture) {
    architecture = lfs_hip_base_architecture(architecture);
    if (architecture.empty()) return false;
    while (!targets.empty()) {
        const auto end = targets.find(',');
        if (targets.substr(0, end) == architecture) return true;
        if (end == std::string_view::npos) break;
        targets.remove_prefix(end + 1);
    }
    return false;
}

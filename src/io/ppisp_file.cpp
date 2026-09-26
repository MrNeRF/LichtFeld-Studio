/* SPDX-License-Identifier: GPL-3.0-or-later */
#include "io/ppisp_file.hpp"

namespace lfs::training {
    std::filesystem::path find_ppisp_companion(const std::filesystem::path& splat_path) {
        auto companion = get_ppisp_companion_path(splat_path);
        if (std::filesystem::exists(companion)) {
            return companion;
        }
        return {};
    }

    std::filesystem::path get_ppisp_companion_path(const std::filesystem::path& splat_path) {
        auto path = splat_path;
        path.replace_extension(".ppisp");
        return path;
    }

} // namespace lfs::training

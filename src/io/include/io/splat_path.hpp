/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/path_utils.hpp"
#include <filesystem>
#include <string>

namespace lfs::io {
    // Use the asset folder, rather than its generic JSON manifest, in the scene.
    inline std::string splat_import_name(const std::filesystem::path& source) {
        namespace fs = std::filesystem;
        auto path = fs::absolute(source).lexically_normal();
        if (path.filename().empty())
            path = path.parent_path();
        const bool manifest = path.filename() == "lod-meta.json" || path.filename() == "meta.json";
        std::error_code error;
        if (!manifest && !fs::is_directory(path, error))
            return core::path_to_utf8(source.stem());
        const auto folder = manifest ? path.parent_path() : path;
        if ((path.filename() == "meta.json" || fs::exists(folder / "meta.json", error)) &&
            fs::exists(folder.parent_path() / "lod-meta.json", error))
            return core::path_to_utf8(folder.parent_path().filename()) + "_" + core::path_to_utf8(folder.filename());
        return core::path_to_utf8(folder.filename());
    }
} // namespace lfs::io

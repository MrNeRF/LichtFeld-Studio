/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/splat_data.hpp"
#include "io/error.hpp"
#include <expected>
#include <filesystem>
#include <functional>
#include <string>

namespace lfs::io {

    using lfs::core::SplatData;

    // Forward declaration for progress callback
    using ExportProgressCallback = std::function<bool(float progress, const std::string& stage)>;

    // Load RAD (Random Access Dynamic) format - chunked hierarchical Gaussian splat format
    std::expected<SplatData, std::string> load_rad(const std::filesystem::path& filepath);

    // Save RAD format
    struct RadSaveOptions {
        std::filesystem::path output_path;
        int compression_level = 6;                          // deflate compression level (0-9, default 6)
        std::vector<float> lod_ratios;                      // Custom LOD ratios (e.g., {0.2, 0.5, 1.0}), empty = use defaults
        bool flip_y = true;                                 // Flip Y axis on export (enabled by default)
        ExportProgressCallback progress_callback = nullptr; // Progress callback
    };

    [[nodiscard]] Result<void> save_rad(const SplatData& splat_data, const RadSaveOptions& options);

} // namespace lfs::io

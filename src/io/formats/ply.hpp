/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

// Re-export public API
#include "core/error.hpp"
#include "core/point_cloud.hpp"
#include "io/exporter.hpp"
#include "io/loader.hpp"

namespace lfs::io {

    namespace ply_constants {
        constexpr float DEFAULT_LOG_SCALE = -5.0f;
        constexpr float SCENE_SCALE_FACTOR = 0.5f;
        constexpr float SH_C0 = 0.28209479177387814f;
    } // namespace ply_constants

    // Check if PLY contains triangle face elements
    bool ply_has_faces(const std::filesystem::path& filepath);

    // Check if PLY contains Gaussian splat properties (opacity, scaling, rotation)
    bool is_gaussian_splat_ply(const std::filesystem::path& filepath);

    // Load PLY as Gaussian splat (with opacity, scaling, rotation, SH). Internal
    // Result end to end (Section 5.4); PLYLoader adapts to the legacy io::Result
    // surface at the loader boundary.
    lfs::Result<LoadOutcome<SplatData>> load_ply(const std::filesystem::path& filepath,
                                                 const LoadOptions& options = {});

    // Load PLY as simple point cloud (xyz + optional colors and normals)
    std::expected<lfs::core::PointCloud, std::string> load_ply_point_cloud(const std::filesystem::path& filepath,
                                                                           const LoadOptions& options);

} // namespace lfs::io

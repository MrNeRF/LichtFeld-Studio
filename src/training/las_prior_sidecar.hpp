/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/parameters.hpp"

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace lfs::training {

    // <dataset>/supervision.json written next to depth and normal maps rendered
    // from a LiDAR scan (GeosunPre). Only the fields that change how the maps
    // must be read are kept.
    struct LasPriorSidecar {
        bool depth_millimetres = false;     // depth_format "uint16_png_mm"
        bool ray_distance = false;          // depth_type "ray_distance", as GUT renders non-pinhole depth
        bool normals_camera_opencv = false; // normal_frame "opencv_camera..."
    };

    [[nodiscard]] std::optional<LasPriorSidecar> read_las_prior_sidecar(
        const std::filesystem::path& dataset_path);

    // Points auto settings at the scan's maps and keeps the monocular
    // estimators from overwriting them. Explicit non-default choices stay.
    // Returns one line per changed setting.
    std::vector<std::string> apply_las_prior_defaults(
        lfs::core::param::OptimizationParameters& opt,
        const LasPriorSidecar& sidecar);

} // namespace lfs::training

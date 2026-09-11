/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once
#include "colmap.hpp"
#include "core/camera.hpp"
#include "core/point_cloud.hpp"
#include "io/loader.hpp"

#include <filesystem>
#include <vector>

namespace lfs::io {

    std::filesystem::path GetTransformImagePath(
        const std::filesystem::path& dir_path,
        const std::string& frame_file_path);

    std::tuple<std::vector<CameraData>, lfs::core::Tensor, std::optional<std::tuple<std::vector<std::string>, std::vector<std::string>>>> read_transforms_cameras_and_images(
        const std::filesystem::path& transPath,
        const LoadOptions& options = {});

    PointCloud generate_random_point_cloud();

    PointCloud load_simple_ply_point_cloud(const std::filesystem::path& filepath,
                                           const LoadOptions& options = {});

    PointCloud convert_transforms_point_cloud_to_colmap_world(PointCloud point_cloud);

} // namespace lfs::io

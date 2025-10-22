/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once
#include "colmap.hpp"
#include "core/camera.hpp"
#include "core/point_cloud.hpp"

#include <filesystem>
#include <vector>

namespace gs::loader {

    std::tuple<std::vector<CameraData>, torch::Tensor, std::optional<std::tuple<std::vector<std::string>, std::vector<std::string>>>> read_transforms_cameras_and_images(
        const std::filesystem::path& transPath);

    PointCloud generate_random_point_cloud();

    PointCloud load_simple_ply_point_cloud(const std::filesystem::path& filepath);

} // namespace gs::loader
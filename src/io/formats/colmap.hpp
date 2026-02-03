/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/camera.hpp"
#include "core/point_cloud.hpp"
#include "core/tensor.hpp"
#include "io/error.hpp"
#include <filesystem>
#include <memory>
#include <vector>

namespace lfs::io {

    // Import types from lfs::core for convenience
    using lfs::core::Camera;
    using lfs::core::DataType;
    using lfs::core::Device;
    using lfs::core::PointCloud;
    using lfs::core::Tensor;

    // Camera data structure used for intermediate loading before Camera creation
    struct CameraData {
        // Static data loaded from COLMAP/transforms
        uint32_t _camera_ID = 0;
        Tensor _R;
        Tensor _T;
        float _focal_x = 0.f;
        float _focal_y = 0.f;
        float _center_x = 0.f;
        float _center_y = 0.f;
        std::string _image_name;
        std::filesystem::path _image_path;
        lfs::core::CameraModelType _camera_model_type = lfs::core::CameraModelType::PINHOLE;
        int _width = 0;
        int _height = 0;
        Tensor _radial_distortion;
        Tensor _tangential_distortion;

        // Default constructor - tensors will be assigned later
        CameraData() = default;

        // Explicitly defaulted copy/move to ensure tensor semantics are preserved
        CameraData(const CameraData&) = default;
        CameraData(CameraData&&) = default;
        CameraData& operator=(const CameraData&) = default;
        CameraData& operator=(CameraData&&) = default;
    };

    /**
     * @brief Read COLMAP cameras and images
     * @param base Base directory containing COLMAP data
     * @param images_folder Folder containing images (default: "images")
     * @return Result containing tuple of (vector of Camera, scene_center tensor [3])
     */
    Result<std::tuple<std::vector<std::shared_ptr<Camera>>, Tensor>>
    read_colmap_cameras_and_images(
        const std::filesystem::path& base,
        const std::string& images_folder = "images");

    /**
     * @brief Read COLMAP point cloud (binary format)
     * @param filepath Base directory containing points3D.bin
     * @return PointCloud
     */
    PointCloud read_colmap_point_cloud(const std::filesystem::path& filepath);

    /**
     * @brief Read COLMAP cameras and images from text files
     * @param base Base directory containing COLMAP data
     * @param images_folder Folder containing images (default: "images")
     * @return Result containing tuple of (vector of Camera, scene_center tensor [3])
     */
    Result<std::tuple<std::vector<std::shared_ptr<Camera>>, Tensor>>
    read_colmap_cameras_and_images_text(
        const std::filesystem::path& base,
        const std::string& images_folder = "images");

    /**
     * @brief Read COLMAP point cloud from text file
     * @param filepath Base directory containing points3D.txt
     * @return PointCloud
     */
    PointCloud read_colmap_point_cloud_text(const std::filesystem::path& filepath);

} // namespace lfs::io
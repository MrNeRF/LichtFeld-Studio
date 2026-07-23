/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/camera.hpp"
#include "core/point_cloud.hpp"
#include "core/tensor.hpp"
#include "io/error.hpp"
#include "io/loader.hpp"
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <glm/glm.hpp>
#include <memory>
#include <vector>

namespace lfs::io {

    // Import types from lfs::core for convenience
    using lfs::core::Camera;
    using lfs::core::DataType;
    using lfs::core::Device;
    using lfs::core::PointCloud;
    using lfs::core::Tensor;

    struct ColmapPointCloudLoadStats {
        PointCloud point_cloud;
        std::size_t total_points = 0;
        std::size_t points_after_filtering = 0;
        bool track_filter_applied = false;
    };

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
    Result<LoadOutcome<std::tuple<std::vector<std::shared_ptr<Camera>>, Tensor>>>
    read_colmap_cameras_and_images(
        const std::filesystem::path& base,
        const std::string& images_folder = "images",
        const LoadOptions& options = {});

    /**
     * @brief Read COLMAP point cloud (binary format)
     * @param filepath Base directory containing points3D.bin
     * @return PointCloud
     */
    Result<LoadOutcome<PointCloud>> read_colmap_point_cloud(
        const std::filesystem::path& filepath,
        const LoadOptions& options = {});

    Result<LoadOutcome<ColmapPointCloudLoadStats>> read_colmap_point_cloud_with_stats(
        const std::filesystem::path& filepath,
        const LoadOptions& options = {});

    /**
     * @brief Read COLMAP cameras and images from text files
     * @param base Base directory containing COLMAP data
     * @param images_folder Folder containing images (default: "images")
     * @return Result containing tuple of (vector of Camera, scene_center tensor [3])
     */
    Result<LoadOutcome<std::tuple<std::vector<std::shared_ptr<Camera>>, Tensor>>>
    read_colmap_cameras_and_images_text(
        const std::filesystem::path& base,
        const std::string& images_folder = "images",
        const LoadOptions& options = {});

    /**
     * @brief Validate COLMAP dataset image and mask layout against metadata
     * @param base Base directory containing COLMAP data
     * @param images_folder Folder containing images (default: "images")
     * @return Success if the dataset layout matches the COLMAP metadata contract
     *
     * For nested image folders, COLMAP metadata must preserve the relative path
     * under the images root. Duplicate basenames across subdirectories are valid
     * only when images.bin/images.txt stores entries like "cam_a/frame.png"
     * instead of just "frame.png". Masks follow the same relative-path contract.
     */
    Result<void> validate_colmap_dataset_layout(
        const std::filesystem::path& base,
        const std::string& images_folder = "images",
        const LoadOptions& options = {});

    /**
     * @brief Read COLMAP point cloud from text file
     * @param filepath Base directory containing points3D.txt
     * @return PointCloud
     */
    Result<LoadOutcome<PointCloud>> read_colmap_point_cloud_text(
        const std::filesystem::path& filepath,
        const LoadOptions& options = {});

    Result<LoadOutcome<ColmapPointCloudLoadStats>> read_colmap_point_cloud_text_with_stats(
        const std::filesystem::path& filepath,
        const LoadOptions& options = {});

    /**
     * @brief Read COLMAP cameras only (no image file validation required)
     * @param sparse_path Path to COLMAP sparse reconstruction folder
     * @param scale_factor Scale factor for camera intrinsics (default: 1.0)
     * @return Result containing tuple of (vector of Camera, scene_center tensor [3])
     *
     * Unlike read_colmap_cameras_and_images, this function:
     * - Does not require actual image files to exist
     * - Skips mask file lookup
     * - Sets empty image_path and mask_path on cameras
     * - Computes scene center from camera positions
     */
    Result<std::tuple<std::vector<std::shared_ptr<Camera>>, Tensor>>
    read_colmap_cameras_only(const std::filesystem::path& sparse_path,
                             float scale_factor = 1.0f);

    enum class ColmapWriteFormat {
        Auto,
        Binary,
        Text
    };

    struct ColmapCameraWriteData {
        std::shared_ptr<const Camera> camera;
        glm::mat4 data_world_transform{1.0f};
    };

    struct ColmapWriteOptions {
        ColmapWriteFormat format = ColmapWriteFormat::Auto;
    };

    /**
     * @brief Find the COLMAP sparse model directory for a dataset or sparse path.
     * @param source_base Existing COLMAP dataset/sparse directory.
     * @return Directory containing cameras/images sparse metadata.
     *
     * Binary sparse metadata is preferred when both binary and text files exist.
     */
    Result<std::filesystem::path> find_colmap_sparse_model_path(
        const std::filesystem::path& source_base);

    /**
     * @brief Write transformed cameras and point cloud back to COLMAP sparse files
     * @param source_base Existing COLMAP dataset/sparse directory to preserve camera models,
     *        image ids, 2D observations, point ids, tracks, and reprojection errors.
     * @param output_sparse_path Directory where cameras/images/points3D files are written.
     * @param cameras Current camera objects with their scene data-world transforms.
     * @param point_cloud Optional current sparse point cloud.
     * @param point_cloud_transform Scene data-world transform for point_cloud.
     * @param options Output format. Auto follows the source sparse model format.
     * @return Success or structured error.
     */
    Result<void> write_colmap_reconstruction(
        const std::filesystem::path& source_base,
        const std::filesystem::path& output_sparse_path,
        const std::vector<ColmapCameraWriteData>& cameras,
        const PointCloud* point_cloud,
        const glm::mat4& point_cloud_transform = glm::mat4(1.0f),
        const ColmapWriteOptions& options = {});

} // namespace lfs::io

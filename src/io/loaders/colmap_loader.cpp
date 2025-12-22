/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "io/loaders/colmap_loader.hpp"
#include "core/camera.hpp"
#include "core/logger.hpp"
#include "core/point_cloud.hpp"
#include "formats/colmap.hpp"
#include "io/error.hpp"
#include "io/filesystem_utils.hpp"
#include "training/dataset.hpp"
#include <algorithm>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <format>
#include <system_error>

namespace lfs::io {

    // Import types from lfs::core for convenience
    using lfs::core::DataType;
    using lfs::core::Device;
    using lfs::core::PointCloud;
    using lfs::core::Tensor;

    Result<LoadResult> ColmapLoader::load(
        const std::filesystem::path& path,
        const LoadOptions& options) {

        LOG_TIMER("COLMAP Loading");
        auto start_time = std::chrono::high_resolution_clock::now();

        // Validate directory exists
        if (!std::filesystem::exists(path)) {
            return make_error(ErrorCode::PATH_NOT_FOUND,
                              "COLMAP dataset path does not exist", path);
        }

        if (!std::filesystem::is_directory(path)) {
            return make_error(ErrorCode::NOT_A_DIRECTORY,
                              "COLMAP dataset must be a directory", path);
        }

        // Report initial progress
        if (options.progress) {
            options.progress(0.0f, "Loading COLMAP dataset...");
        }

        // Get search paths for COLMAP files
        auto search_paths = get_colmap_search_paths(path);

        // Check for required COLMAP files in any of the search paths
        auto cameras_bin = find_file_in_paths(search_paths, "cameras.bin");
        auto images_bin = find_file_in_paths(search_paths, "images.bin");
        auto points_bin = find_file_in_paths(search_paths, "points3D.bin");

        auto cameras_txt = find_file_in_paths(search_paths, "cameras.txt");
        auto images_txt = find_file_in_paths(search_paths, "images.txt");
        auto points_txt = find_file_in_paths(search_paths, "points3D.txt");

        bool has_cameras = !cameras_bin.empty();
        bool has_images = !images_bin.empty();
        bool has_points = !points_bin.empty();

        bool has_cameras_text = !cameras_txt.empty();
        bool has_images_text = !images_txt.empty();
        bool has_points_text = !points_txt.empty();

        if ((has_cameras || has_images || has_points) &&
            (has_cameras_text || has_images_text || has_points_text)) {
            LOG_WARN("Found both binary and text COLMAP files. Prioritizing binary files.");
        }

        bool trying_text = !(has_cameras && has_images) && (has_cameras_text && has_images_text);
        LOG_INFO("Loading COLMAP in {} format", trying_text ? "text" : "binary");

        // Validate we have required files
        if ((!has_cameras || !has_images) && !trying_text) {
            return make_error(ErrorCode::MISSING_REQUIRED_FILES,
                              std::format("Missing required COLMAP files. cameras.bin: {}, images.bin: {}",
                                          has_cameras ? "found" : "missing",
                                          has_images ? "found" : "missing"),
                              path);
        }

        if ((!has_cameras_text || !has_images_text) && trying_text) {
            return make_error(ErrorCode::MISSING_REQUIRED_FILES,
                              std::format("Missing required COLMAP text files. cameras.txt: {}, images.txt: {}",
                                          has_cameras_text ? "found" : "missing",
                                          has_images_text ? "found" : "missing"),
                              path);
        }

        // Determine images folder
        std::string actual_images_folder = options.images_folder;
        std::filesystem::path image_dir = path / actual_images_folder;

        auto is_dataset_root = [&](const std::filesystem::path& candidate) {
            if (candidate.empty()) {
                return false;
            }

            std::error_code ec;
            bool equivalent = std::filesystem::equivalent(candidate, path, ec);
            if (!ec) {
                return equivalent;
            }

            return candidate.lexically_normal() == path.lexically_normal();
        };

        // If specified folder doesn't exist, check for flat structure
        if (!std::filesystem::exists(image_dir)) {
            const auto cameras_txt_parent =
                cameras_txt.empty() ? std::filesystem::path{} : cameras_txt.parent_path();
            const auto cameras_bin_parent =
                cameras_bin.empty() ? std::filesystem::path{} : cameras_bin.parent_path();

            bool is_flat_structure = is_dataset_root(cameras_txt_parent) ||
                                     is_dataset_root(cameras_bin_parent);

            if (is_flat_structure) {
                bool has_images_in_root = false;
                for (const auto& entry : std::filesystem::directory_iterator(path)) {
                    if (entry.is_regular_file() && is_image_file(entry.path())) {
                        has_images_in_root = true;
                        break;
                    }
                }

                if (has_images_in_root) {
                    actual_images_folder = ".";
                    image_dir = path;
                    LOG_INFO("Detected flat structure - using root directory for images");
                } else {
                    return make_error(ErrorCode::MISSING_REQUIRED_FILES,
                                      std::format("Images directory '{}' not found and no images in root",
                                                  options.images_folder),
                                      path);
                }
            } else {
                return make_error(ErrorCode::MISSING_REQUIRED_FILES,
                                  std::format("Images directory '{}' not found", options.images_folder), path);
            }
        }

        // Validation only mode
        if (options.validate_only) {
            if (options.progress) {
                options.progress(100.0f, "COLMAP validation complete");
            }

            LOG_DEBUG("COLMAP validation successful");

            auto end_time = std::chrono::high_resolution_clock::now();
            return LoadResult{
                .data = LoadedScene{
                    .cameras = nullptr,
                    .point_cloud = nullptr},
                .scene_center = Tensor::zeros({3}, Device::CPU),
                .loader_used = name(),
                .load_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time),
                .warnings = (has_points || has_points_text) ? std::vector<std::string>{} : std::vector<std::string>{"No sparse point cloud found (points3D.bin|txt) - will use random initialization"}};
        }

        // Load cameras and images
        if (options.progress) {
            options.progress(20.0f, "Reading camera parameters...");
        }

        try {
            std::vector<std::shared_ptr<Camera>> cameras;
            Tensor scene_center;

            if (has_cameras && has_images) {
                LOG_DEBUG("Reading binary COLMAP data");
                auto result = read_colmap_cameras_and_images(path, actual_images_folder);
                if (!result) {
                    return std::unexpected(result.error());
                }
                std::tie(cameras, scene_center) = std::move(*result);
            } else if (has_cameras_text && has_images_text) {
                LOG_DEBUG("Reading text COLMAP data");
                auto result = read_colmap_cameras_and_images_text(path, actual_images_folder);
                if (!result) {
                    return std::unexpected(result.error());
                }
                std::tie(cameras, scene_center) = std::move(*result);
            } else {
                return make_error(ErrorCode::MISSING_REQUIRED_FILES,
                                  "No valid COLMAP camera and image data found", path);
            }

            if (options.progress) {
                options.progress(40.0f, std::format("Creating {} cameras...", cameras.size()));
            }

            LOG_DEBUG("Creating {} camera objects", cameras.size());

            // Create dataset configuration
            lfs::training::DatasetConfig dataset_config;
            dataset_config.resize_factor = options.resize_factor;
            dataset_config.max_width = options.max_width;
            dataset_config.test_every = 8; // Default split behavior

            // Create dataset with ALL images
            auto dataset = std::make_shared<lfs::training::CameraDataset>(
                std::move(cameras), dataset_config, lfs::training::CameraDataset::Split::ALL);

            if (options.progress) {
                options.progress(60.0f, "Loading point cloud...");
            }

            // Load point cloud if it exists
            std::shared_ptr<PointCloud> point_cloud;
            if (has_points) {
                LOG_DEBUG("Loading binary point cloud");
                auto loaded_pc = read_colmap_point_cloud(path);
                point_cloud = std::make_shared<PointCloud>(std::move(loaded_pc));
                LOG_INFO("Loaded {} points from COLMAP", point_cloud->size());
            } else if (has_points_text) {
                LOG_DEBUG("Loading text point cloud");
                auto loaded_pc = read_colmap_point_cloud_text(path);
                point_cloud = std::make_shared<PointCloud>(std::move(loaded_pc));
                LOG_INFO("Loaded {} points from COLMAP text file", point_cloud->size());
            } else {
                LOG_WARN("No point cloud found - will use random initialization");
                point_cloud = std::make_shared<PointCloud>();
            }

            if (options.progress) {
                options.progress(100.0f, "COLMAP loading complete");
            }

            auto end_time = std::chrono::high_resolution_clock::now();
            auto load_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                end_time - start_time);

            // Get scene center values and dataset size for logging BEFORE moving
            auto scene_center_cpu = scene_center.cpu();
            const float* sc_ptr = scene_center_cpu.ptr<float>();
            size_t num_cameras = dataset->size();

            LoadResult result{
                .data = LoadedScene{
                    .cameras = std::move(dataset),
                    .point_cloud = std::move(point_cloud)},
                .scene_center = scene_center,
                .loader_used = name(),
                .load_time = load_time,
                .warnings = (has_points || has_points_text) ? std::vector<std::string>{} : std::vector<std::string>{"No sparse point cloud found - using random initialization"}};

            LOG_INFO("COLMAP dataset loaded successfully in {}ms", load_time.count());
            LOG_INFO("  - {} cameras", num_cameras);
            LOG_DEBUG("  - Scene center: [{:.3f}, {:.3f}, {:.3f}]",
                      sc_ptr[0], sc_ptr[1], sc_ptr[2]);

            return result;

        } catch (const std::exception& e) {
            return make_error(ErrorCode::CORRUPTED_DATA,
                              std::format("Failed to load COLMAP dataset: {}", e.what()), path);
        }
    }

    bool ColmapLoader::canLoad(const std::filesystem::path& path) const {
        if (!safe_exists(path) || !safe_is_directory(path)) {
            return false;
        }

        auto search_paths = get_colmap_search_paths(path);

        // Check for COLMAP files in any location
        const std::vector<std::string> colmap_files = {
            "cameras.bin", "cameras.txt",
            "images.bin", "images.txt"};

        for (const auto& filename : colmap_files) {
            if (!find_file_in_paths(search_paths, filename).empty()) {
                return true;
            }
        }

        return false;
    }

    std::string ColmapLoader::name() const {
        return "COLMAP";
    }

    std::vector<std::string> ColmapLoader::supportedExtensions() const {
        return {}; // Directory-based, no file extensions
    }

    int ColmapLoader::priority() const {
        return 5; // Medium priority
    }

} // namespace lfs::io

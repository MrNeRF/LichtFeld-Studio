/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "loader/loaders/blender_loader.hpp"
#include "core/camera.hpp"
#include "core/logger.hpp"
#include "core/point_cloud.hpp"
#include "formats/transforms.hpp"
#include "training/dataset.hpp"
#include <chrono>
#include <filesystem>
#include <format>
#include <fstream>
#include <nlohmann/json.hpp>

namespace gs::loader {

    std::expected<LoadResult, std::string> BlenderLoader::load(
        const std::filesystem::path& path,
        const LoadOptions& options) {

        LOG_TIMER("Blender/NeRF Loading");
        auto start_time = std::chrono::high_resolution_clock::now();

        // Validate path exists
        if (!std::filesystem::exists(path)) {
            std::string error_msg = std::format("Path does not exist: {}", path.string());
            LOG_ERROR("{}", error_msg);
            throw std::runtime_error(error_msg);
        }

        // Report initial progress
        if (options.progress) {
            options.progress(0.0f, "Loading Blender/NeRF dataset...");
        }

        // Determine transforms file path
        std::filesystem::path transforms_file;

        if (std::filesystem::is_directory(path)) {
            // Look for transforms files in directory
            if (std::filesystem::exists(path / "transforms_train.json")) {
                transforms_file = path / "transforms_train.json";
                LOG_DEBUG("Found transforms_train.json");
            } else if (std::filesystem::exists(path / "transforms.json")) {
                transforms_file = path / "transforms.json";
                LOG_DEBUG("Found transforms.json");
            } else {
                LOG_ERROR("No transforms file found in directory: {}", path.string());
                throw std::runtime_error(
                    "No transforms file found (expected 'transforms.json' or 'transforms_train.json')");
            }
        } else if (path.extension() == ".json") {
            // Direct path to transforms file
            transforms_file = path;
            LOG_DEBUG("Using direct transforms file: {}", transforms_file.string());
        } else {
            LOG_ERROR("Path must be a directory or a JSON file: {}", path.string());
            throw std::runtime_error("Path must be a directory or a JSON file");
        }

        // Validation only mode
        if (options.validate_only) {
            LOG_DEBUG("Validation only mode for Blender/NeRF: {}", transforms_file.string());
            // Check if the transforms file is valid JSON
            std::ifstream file(transforms_file);
            if (!file) {
                LOG_ERROR("Cannot open transforms file: {}", transforms_file.string());
                throw std::runtime_error("Cannot open transforms file");
            }

            // Try to parse as JSON (basic validation)
            try {
                nlohmann::json j;
                file >> j;

                if (!j.contains("frames") || !j["frames"].is_array()) {
                    LOG_ERROR("Invalid transforms file: missing 'frames' array");
                    throw std::runtime_error("Invalid transforms file: missing 'frames' array");
                }
            } catch (const std::exception& e) {
                std::string error_msg = std::format("Invalid JSON: {}", e.what());
                LOG_ERROR("{}", error_msg);
                throw std::runtime_error(error_msg);
            }

            if (options.progress) {
                options.progress(100.0f, "Blender/NeRF validation complete");
            }

            LOG_DEBUG("Blender/NeRF validation successful");

            auto end_time = std::chrono::high_resolution_clock::now();
            return LoadResult{
                .data = LoadedScene{
                    .cameras = nullptr,
                    .point_cloud = nullptr},
                .scene_center = torch::zeros({3}),
                .loader_used = name(),
                .load_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time),
                .warnings = {"Validation mode - point cloud not loaded"}};
        }

        // Load the dataset
        if (options.progress) {
            options.progress(20.0f, "Reading transforms file...");
        }

        try {
            LOG_INFO("Loading Blender/NeRF dataset from: {}", transforms_file.string());

            // Read transforms and create cameras
            auto [camera_infos, scene_center, splits] = read_transforms_cameras_and_images(transforms_file);

            if (options.progress) {
                options.progress(40.0f, std::format("Creating {} cameras...", camera_infos.size()));
            }

            LOG_DEBUG("Creating {} camera objects", camera_infos.size());

            // Create Camera objects
            std::vector<std::shared_ptr<Camera>> cameras;
            cameras.reserve(camera_infos.size());

            for (size_t i = 0; i < camera_infos.size(); ++i) {
                const auto& info = camera_infos[i];

                auto cam = std::make_shared<Camera>(
                    info._R,
                    info._T,
                    info._focal_x,
                    info._focal_y,
                    info._center_x,
                    info._center_y,
                    info._radial_distortion,
                    info._tangential_distortion,
                    info._camera_model_type,
                    info._image_name,
                    info._image_path,
                    info._width,
                    info._height,
                    static_cast<int>(i));

                cameras.push_back(std::move(cam));
            }

            // Create dataset configuration
            gs::param::DatasetConfig dataset_config;
            dataset_config.data_path = path;
            dataset_config.images = options.images_folder;
            dataset_config.resize_factor = options.resize_factor;
            dataset_config.max_width = options.max_width;

            // Create dataset with ALL images
            auto dataset = std::make_shared<gs::training::CameraDataset>(
                std::move(cameras), dataset_config, gs::training::CameraDataset::Split::ALL);

            if (options.progress) {
                options.progress(60.0f, "Loading point cloud...");
            }

            // Try to load point cloud from PLY file, fallback to random generation
            std::filesystem::path pointcloud_path = transforms_file.parent_path() / "pointcloud.ply";
            std::shared_ptr<PointCloud> point_cloud;
            std::vector<std::string> warnings;

            LOG_DEBUG("Looking for point cloud at: {}", pointcloud_path.string());

            if (std::filesystem::exists(pointcloud_path)) {
                LOG_DEBUG("Loading point cloud from PLY file");
                auto loaded_pc = load_simple_ply_point_cloud(pointcloud_path);
                point_cloud = std::make_shared<PointCloud>(std::move(loaded_pc));
                LOG_INFO("Loaded point cloud with {} points from pointcloud.ply", point_cloud->size());
            } else {
                LOG_WARN("No pointcloud.ply found in dataset directory: {}", transforms_file.parent_path().string());
                LOG_DEBUG("Generating random point cloud for initialization");
                auto random_pc = generate_random_point_cloud();
                point_cloud = std::make_shared<PointCloud>(std::move(random_pc));
                LOG_INFO("Generated random point cloud with {} points", point_cloud->size());
                warnings.push_back("Using random point cloud initialization (pointcloud.ply not found)");
            }

            if (options.progress) {
                options.progress(100.0f, "Blender/NeRF loading complete");
            }

            auto end_time = std::chrono::high_resolution_clock::now();
            auto load_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                end_time - start_time);

            // Create result with shared_ptr
            LoadResult result{
                .data = LoadedScene{
                    .cameras = std::move(dataset),
                    .point_cloud = std::move(point_cloud)},
                .scene_center = scene_center,
                .loader_used = name(),
                .load_time = load_time,
                .warnings = std::move(warnings),
                .provided_splits = splits};

            LOG_INFO("Blender/NeRF dataset loaded successfully in {}ms", load_time.count());
            LOG_INFO("  - {} cameras", camera_infos.size());
            LOG_DEBUG("  - Scene center: [{:.3f}, {:.3f}, {:.3f}]",
                      scene_center[0].item<float>(),
                      scene_center[1].item<float>(),
                      scene_center[2].item<float>());

            return result;

        } catch (const std::exception& e) {
            std::string error_msg = std::format("Failed to load Blender/NeRF dataset: {}", e.what());
            LOG_ERROR("{}", error_msg);
            throw std::runtime_error(error_msg);
        }
    }

    bool BlenderLoader::canLoad(const std::filesystem::path& path) const {
        if (!std::filesystem::exists(path)) {
            return false;
        }

        if (std::filesystem::is_directory(path)) {
            // Check for transforms files in directory
            return std::filesystem::exists(path / "transforms.json") ||
                   std::filesystem::exists(path / "transforms_train.json");
        } else {
            // Check if it's a JSON file
            return path.extension() == ".json";
        }
    }

    std::string BlenderLoader::name() const {
        return "Blender/NeRF";
    }

    std::vector<std::string> BlenderLoader::supportedExtensions() const {
        return {".json"}; // Can load JSON files directly
    }

    int BlenderLoader::priority() const {
        return 5; // Medium priority
    }
} // namespace gs::loader
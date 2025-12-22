/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "io/loaders/blender_loader.hpp"
#include "core/camera.hpp"
#include "core/image_io.hpp"
#include "core/logger.hpp"
#include "core/point_cloud.hpp"
#include "formats/transforms.hpp"
#include "io/error.hpp"
#include "training/dataset.hpp"
#include <chrono>
#include <filesystem>
#include <format>
#include <fstream>
#include <nlohmann/json.hpp>

namespace lfs::io {

    // Import types from lfs::core for convenience
    using lfs::core::DataType;
    using lfs::core::Device;
    using lfs::core::PointCloud;
    using lfs::core::Tensor;

    namespace {
        constexpr std::array MASK_FOLDERS = {"masks", "mask", "segmentation"};
        constexpr std::array MASK_EXTENSIONS = {".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG", ".mask.png"};
    } // namespace

    // Searches for mask file matching image_name in mask folders.
    // Priority: exact match, stem+ext (e.g., img.png), full+ext (e.g., img.jpg.png)
    static std::filesystem::path find_mask_path(const std::filesystem::path& base_path,
                                                const std::string& image_name) {
        const std::filesystem::path img_path(image_name);
        const std::filesystem::path stem_path = img_path.parent_path() / img_path.stem();

        for (const auto& folder : MASK_FOLDERS) {
            const std::filesystem::path mask_dir = base_path / folder;
            if (!std::filesystem::exists(mask_dir))
                continue;

            if (const auto exact = mask_dir / image_name; std::filesystem::exists(exact))
                return exact;

            for (const auto& ext : MASK_EXTENSIONS) {
                auto path = mask_dir / stem_path;
                path += ext;
                if (std::filesystem::exists(path))
                    return path;
            }

            for (const auto& ext : MASK_EXTENSIONS) {
                auto path = mask_dir / image_name;
                path += ext;
                if (std::filesystem::exists(path))
                    return path;
            }
        }
        return {};
    }

    Result<LoadResult> BlenderLoader::load(
        const std::filesystem::path& path,
        const LoadOptions& options) {

        LOG_TIMER("Blender/NeRF Loading");
        auto start_time = std::chrono::high_resolution_clock::now();

        // Validate path exists
        if (!std::filesystem::exists(path)) {
            return make_error(ErrorCode::PATH_NOT_FOUND,
                              "Blender/NeRF dataset path does not exist", path);
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
                return make_error(ErrorCode::MISSING_REQUIRED_FILES,
                                  "No transforms file found (expected 'transforms.json' or 'transforms_train.json')", path);
            }
        } else if (path.extension() == ".json") {
            // Direct path to transforms file
            transforms_file = path;
            LOG_DEBUG("Using direct transforms file: {}", transforms_file.string());
        } else {
            return make_error(ErrorCode::UNSUPPORTED_FORMAT,
                              "Path must be a directory or a JSON file", path);
        }

        // Validation only mode
        if (options.validate_only) {
            LOG_DEBUG("Validation only mode for Blender/NeRF: {}", transforms_file.string());
            // Check if the transforms file is valid JSON
            std::ifstream file(transforms_file);
            if (!file) {
                return make_error(ErrorCode::PERMISSION_DENIED,
                                  "Cannot open transforms file for reading", transforms_file);
            }

            // Try to parse as JSON (basic validation)
            try {
                nlohmann::json j;
                file >> j;

                if (!j.contains("frames") || !j["frames"].is_array()) {
                    return make_error(ErrorCode::INVALID_DATASET,
                                      "Invalid transforms file: missing 'frames' array", transforms_file);
                }
            } catch (const std::exception& e) {
                return make_error(ErrorCode::MALFORMED_JSON,
                                  std::format("Invalid JSON: {}", e.what()), transforms_file);
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
                .scene_center = Tensor::zeros({3}, Device::CPU, DataType::Float32),
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
            auto [camera_infos, scene_center, train_val_split] = read_transforms_cameras_and_images(transforms_file);

            if (options.progress) {
                options.progress(40.0f, std::format("Creating {} cameras...", camera_infos.size()));
            }

            LOG_DEBUG("Creating {} camera objects", camera_infos.size());

            // Convert CameraData to Camera objects
            std::vector<std::shared_ptr<lfs::core::Camera>> cameras;
            cameras.reserve(camera_infos.size());

            // Get base path for mask lookup
            std::filesystem::path base_path = transforms_file.parent_path();

            for (size_t i = 0; i < camera_infos.size(); ++i) {
                const auto& info = camera_infos[i];

                try {
                    // Find mask path if available
                    std::filesystem::path mask_path = find_mask_path(base_path, info._image_name);

                    // Validate mask dimensions match image dimensions
                    if (!mask_path.empty()) {
                        auto [img_w, img_h, img_c] = lfs::core::get_image_info(info._image_path);
                        auto [mask_w, mask_h, mask_c] = lfs::core::get_image_info(mask_path);
                        if (img_w != mask_w || img_h != mask_h) {
                            return make_error(ErrorCode::MASK_SIZE_MISMATCH,
                                              std::format("Mask '{}' is {}x{} but image '{}' is {}x{}",
                                                          mask_path.filename().string(), mask_w, mask_h,
                                                          info._image_name, img_w, img_h),
                                              mask_path);
                        }
                    }

                    auto cam = std::make_shared<lfs::core::Camera>(
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
                        mask_path,
                        info._width,
                        info._height,
                        static_cast<int>(i));

                    cameras.push_back(cam);
                } catch (const std::exception& e) {
                    LOG_ERROR("Failed to create camera {}: {}", i, e.what());
                    throw;
                }
            }

            // Create dataset configuration
            lfs::training::DatasetConfig dataset_config;
            dataset_config.resize_factor = options.resize_factor;
            dataset_config.max_width = options.max_width;
            dataset_config.test_every = 8; // Default split behavior

            // Create dataset with ALL images
            auto dataset = std::make_shared<lfs::training::CameraDataset>(
                std::move(cameras), dataset_config, lfs::training::CameraDataset::Split::ALL);

            if (options.progress) {
                options.progress(60.0f, "Generating initialization point cloud...");
            }

            // Generate random point cloud for initialization
            // Note: Blender/NeRF datasets don't typically include sparse point clouds
            // If a pointcloud.ply exists, users should load it separately via the PLY loader
            LOG_DEBUG("Generating random point cloud for initialization");
            auto random_pc = generate_random_point_cloud();
            auto point_cloud = std::make_shared<PointCloud>(std::move(random_pc));
            LOG_INFO("Generated random point cloud with {} points", point_cloud->size());

            if (options.progress) {
                options.progress(100.0f, "Blender/NeRF loading complete");
            }

            auto end_time = std::chrono::high_resolution_clock::now();
            auto load_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                end_time - start_time);

            // Get scene center values for logging
            auto scene_center_cpu = scene_center.cpu();
            const float* sc_ptr = scene_center_cpu.template ptr<float>();

            // Save dataset size before moving
            size_t num_cameras = dataset->size();

            LoadResult result{
                .data = LoadedScene{
                    .cameras = std::move(dataset),
                    .point_cloud = std::move(point_cloud)},
                .scene_center = scene_center,
                .loader_used = name(),
                .load_time = load_time,
                .warnings = {"Using random point cloud initialization"}};

            LOG_INFO("Blender/NeRF dataset loaded successfully in {}ms", load_time.count());
            LOG_INFO("  - {} cameras", num_cameras);
            LOG_DEBUG("  - Scene center: [{:.3f}, {:.3f}, {:.3f}]",
                      sc_ptr[0], sc_ptr[1], sc_ptr[2]);

            return result;

        } catch (const std::exception& e) {
            return make_error(ErrorCode::CORRUPTED_DATA,
                              std::format("Failed to load Blender/NeRF dataset: {}", e.what()), path);
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

} // namespace lfs::io
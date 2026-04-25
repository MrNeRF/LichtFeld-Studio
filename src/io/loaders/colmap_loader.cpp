/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "io/loaders/colmap_loader.hpp"
#include "core/camera.hpp"
#include "core/image_io.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "core/point_cloud.hpp"
#include "formats/colmap.hpp"
#include "formats/ply.hpp"
#include "geometry/geometry_misc.hpp"
#include "io/error.hpp"
#include "io/filesystem_utils.hpp"
#include "io/loaders/loader_utils.hpp"
#include <chrono>
#include <filesystem>
#include <format>
#include <glm/glm.hpp>
#include <random>
#include <system_error>
#include <vector>

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

        if (is_load_cancel_requested(options)) {
            return make_error(ErrorCode::CANCELLED, "COLMAP dataset load cancelled", path);
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

        auto points_ply = find_file_in_paths(search_paths, "points3D.ply");

        bool has_cameras = !cameras_bin.empty();
        bool has_images = !images_bin.empty();
        bool has_points = !points_bin.empty();

        bool has_cameras_text = !cameras_txt.empty();
        bool has_images_text = !images_txt.empty();
        bool has_points_text = !points_txt.empty();

        bool has_points_ply = !points_ply.empty();

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
        std::filesystem::path image_dir = path / lfs::core::utf8_to_path(actual_images_folder);

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

        try {
            if (options.progress) {
                options.progress(10.0f, "Validating COLMAP dataset layout...");
            }

            throw_if_load_cancel_requested(options, "COLMAP dataset validation cancelled");

            if (auto validation_result = validate_colmap_dataset_layout(path, actual_images_folder, options); !validation_result) {
                return std::unexpected(validation_result.error());
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
                        .cameras = {},
                        .point_cloud = nullptr},
                    .scene_center = Tensor::zeros({3}, Device::CPU),
                    .loader_used = name(),
                    .load_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time),
                    .warnings = (has_points || has_points_text || has_points_ply) ? std::vector<std::string>{} : std::vector<std::string>{"No sparse point cloud found (points3D.bin|txt|ply) - will use random initialization"}};
            }

            // Load cameras and images
            if (options.progress) {
                options.progress(20.0f, "Reading camera parameters...");
            }

            std::vector<std::shared_ptr<Camera>> cameras;
            Tensor scene_center;

            if (has_cameras && has_images) {
                LOG_DEBUG("Reading binary COLMAP data");
                auto result = read_colmap_cameras_and_images(path, actual_images_folder, options);
                if (!result) {
                    return std::unexpected(result.error());
                }
                std::tie(cameras, scene_center) = std::move(*result);
            } else if (has_cameras_text && has_images_text) {
                LOG_DEBUG("Reading text COLMAP data");
                auto result = read_colmap_cameras_and_images_text(path, actual_images_folder, options);
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

            const bool images_have_alpha = detect_camera_alpha(cameras, options.cancel_requested);

            if (options.progress) {
                options.progress(60.0f, "Loading point cloud...");
            }

            throw_if_load_cancel_requested(options, "COLMAP point cloud load cancelled");

            // Load point cloud: points3D.ply > points3D.bin > points3D.txt
            std::shared_ptr<PointCloud> point_cloud;
            if (has_points_ply) {
                LOG_INFO("Loading custom point cloud from points3D.ply");
                auto pc_result = load_ply_point_cloud(points_ply, options);
                if (pc_result) {
                    point_cloud = std::make_shared<PointCloud>(std::move(*pc_result));
                    LOG_INFO("Loaded {} points from points3D.ply", point_cloud->size());
                } else {
                    if (is_load_cancel_requested(options)) {
                        return std::unexpected(make_error(ErrorCode::CANCELLED, pc_result.error(), points_ply));
                    }
                    LOG_WARN("Failed to load points3D.ply: {}, falling back", pc_result.error());
                }
            }
            if (!point_cloud && has_points) {
                LOG_DEBUG("Loading binary point cloud");
                auto loaded_pc = read_colmap_point_cloud(path, options);
                point_cloud = std::make_shared<PointCloud>(std::move(loaded_pc));
                LOG_INFO("Loaded {} points from COLMAP", point_cloud->size());
            } else if (!point_cloud && has_points_text) {
                LOG_DEBUG("Loading text point cloud");
                auto loaded_pc = read_colmap_point_cloud_text(path, options);
                point_cloud = std::make_shared<PointCloud>(std::move(loaded_pc));
                LOG_INFO("Loaded {} points from COLMAP text file", point_cloud->size());
            } else if (!point_cloud) {
                LOG_WARN("No point cloud found - will use random initialization");
                point_cloud = std::make_shared<PointCloud>();
            }

            // Centralize scene
            {
                const auto mode = options.centralize;
                Tensor center;

                auto build_pts = [&]() {
                    auto positions = point_cloud->means.cpu().contiguous();
                    const int64_t N = static_cast<int64_t>(positions.shape()[0]);
                    auto pos_acc = positions.accessor<float, 2>();
                    constexpr int64_t MAX_SAMPLES = 50000;
                    std::vector<glm::vec3> pts;
                    if (N <= MAX_SAMPLES) {
                        pts.resize(static_cast<size_t>(N));
                        for (int64_t i = 0; i < N; ++i)
                            pts[static_cast<size_t>(i)] = {pos_acc(i, 0), pos_acc(i, 1), pos_acc(i, 2)};
                    } else {
                        pts.resize(MAX_SAMPLES);
                        std::mt19937 rng(33550336);
                        std::uniform_int_distribution<int64_t> dist(0, N - 1);
                        for (int64_t i = 0; i < MAX_SAMPLES; ++i) {
                            int64_t idx = dist(rng);
                            pts[static_cast<size_t>(i)] = {pos_acc(idx, 0), pos_acc(idx, 1), pos_acc(idx, 2)};
                        }
                    }
                    return pts;
                };

                auto build_camera_pts = [&]() {
                    std::vector<glm::vec3> pts;
                    pts.reserve(cameras.size());
                    for (const auto& cam : cameras) {
                        auto pos = cam->cam_position().cpu();
                        const float* p = pos.ptr<float>();
                        pts.push_back({p[0], p[1], p[2]});
                    }
                    return pts;
                };

                if (mode == CentralizeDataset::Auto && point_cloud && point_cloud->size() > 0) {
                    // Shift only when the origin is clearly outside the scene distribution.
                    // factor=2.0 -> threshold ~= 2xMAD ~= 3sigma under normality (MAD ~= 0.6745sigma),
                    // meaning we centre only when the origin is a strong outlier w.r.t. the
                    // point cloud spread - robust against near-origin scenes that need no shift.
                    constexpr float k = 2.0f;

                    auto pts = build_pts();
                    auto med = lfs::geometry::geometric_median(pts);
                    float mad = lfs::geometry::geometric_mad(pts, med);
                    float origin_dist = glm::length(med);

                    if (origin_dist > k * mad) {
                        std::vector<float> center_data = {med.x, med.y, med.z};
                        center = Tensor::from_vector(center_data, {3}, Device::CPU);
                        LOG_INFO("Auto-centralizing: origin {:.3f} > {:.1f}xMAD ({:.3f}) - shifting by median [{:.3f}, {:.3f}, {:.3f}]",
                                 origin_dist, k, k * mad, med.x, med.y, med.z);
                    } else {
                        LOG_INFO("Auto-centralizing: origin {:.3f} <= {:.1f}xMAD ({:.3f}) - no shift needed",
                                 origin_dist, k, k * mad);
                    }

                } else if (mode == CentralizeDataset::ByPointCloud &&
                           point_cloud && point_cloud->size() > 0) {
                    auto pts = build_pts();
                    auto med = lfs::geometry::geometric_median(pts);
                    std::vector<float> center_data = {med.x, med.y, med.z};
                    center = Tensor::from_vector(center_data, {3}, Device::CPU);
                    LOG_INFO("Centralizing by point cloud (geometric median): center=[{:.3f}, {:.3f}, {:.3f}]",
                             med.x, med.y, med.z);

                } else if (mode == CentralizeDataset::ByCameras && !cameras.empty()) {
                    auto cam_pts = build_camera_pts();
                    auto med = lfs::geometry::geometric_median(cam_pts);
                    std::vector<float> center_data = {med.x, med.y, med.z};
                    center = Tensor::from_vector(center_data, {3}, Device::CPU);
                    LOG_INFO("Centralizing by cameras (geometric median): center=[{:.3f}, {:.3f}, {:.3f}]",
                             med.x, med.y, med.z);
                }

                if (center.is_valid()) {
                    // Shift point cloud positions by -center
                    if (point_cloud && point_cloud->size() > 0 && point_cloud->means.is_valid()) {
                        auto center_dev = center.to(point_cloud->means.device());
                        point_cloud->means = point_cloud->means - center_dev;
                    }

                    // Shift all cameras by -center
                    auto neg_center = center.neg();
                    for (auto& cam : cameras)
                        cam->translate(neg_center);

                    scene_center = Tensor::zeros({3}, Device::CPU);
                }
            }

            if (options.progress) {
                options.progress(100.0f, "COLMAP loading complete");
            }

            auto end_time = std::chrono::high_resolution_clock::now();
            auto load_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                end_time - start_time);

            auto scene_center_cpu = scene_center.cpu();
            const float* sc_ptr = scene_center_cpu.ptr<float>();
            size_t num_cameras = cameras.size();

            LoadResult result{
                .data = LoadedScene{
                    .cameras = std::move(cameras),
                    .point_cloud = std::move(point_cloud)},
                .scene_center = scene_center,
                .images_have_alpha = images_have_alpha,
                .loader_used = name(),
                .load_time = load_time,
                .warnings = (has_points || has_points_text || has_points_ply) ? std::vector<std::string>{} : std::vector<std::string>{"No sparse point cloud found - using random initialization"}};

            LOG_INFO("COLMAP dataset loaded successfully in {}ms", load_time.count());
            LOG_INFO("  - {} cameras", num_cameras);
            LOG_DEBUG("  - Scene center: [{:.3f}, {:.3f}, {:.3f}]",
                      sc_ptr[0], sc_ptr[1], sc_ptr[2]);

            return result;

        } catch (const LoadCancelledError& e) {
            return make_error(ErrorCode::CANCELLED, e.what(), path);
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

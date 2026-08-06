/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "io/error.hpp"
#include "io/filesystem_utils.hpp"
#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace lfs::io::opf {

    struct Resource {
        std::string uri;
        std::string format;
        std::filesystem::path resolved_path;
    };

    struct Source {
        std::string id;
        std::string type;
    };

    struct Item {
        std::string id;
        std::string type;
        std::vector<Source> sources;
        std::vector<Resource> resources;
    };

    struct Project {
        std::filesystem::path path;
        std::string format;
        std::string version;
        std::string id;
        std::string name;
        std::string description;
        std::vector<Item> items;
        std::vector<std::string> warnings;
    };

    struct CameraImage {
        std::uint64_t id;
        std::string uri;
        std::filesystem::path resolved_path;
    };

    struct InputSensor {
        std::uint64_t id;
        std::string name;
        std::uint32_t width;
        std::uint32_t height;
        std::string model;
        std::vector<double> principal_point;
        double focal_length = 0.0;
        std::vector<double> radial_distortion;
        std::vector<double> tangential_distortion;
    };

    struct InputCapture {
        std::uint64_t id;
        std::uint64_t reference_camera_id;
        std::vector<std::uint64_t> camera_ids;
    };

    struct PointCloudManifest {
        std::filesystem::path gltf_path;
        std::vector<std::filesystem::path> buffer_paths;
    };

    struct CalibratedCamera {
        std::uint64_t id;
        std::uint64_t sensor_id;
        std::array<double, 3> position;
        std::array<double, 3> orientation_deg;
    };

    // Parses and validates the OPF project graph. Resource files are resolved
    // relative to the directory containing the project file and must remain
    // inside that directory. Unknown extension items/resources are retained
    // as warnings and never make an otherwise valid project fail.
    [[nodiscard]] Result<Project> read_project(const std::filesystem::path& path);
    [[nodiscard]] Result<std::vector<CameraImage>> read_camera_list(const Resource& resource,
                                                                    const std::filesystem::path& project_root);
    [[nodiscard]] Result<std::filesystem::path> resolve_image_uri(const std::filesystem::path& project_root,
                                                                  std::string_view uri);
    [[nodiscard]] Result<std::vector<InputSensor>> read_input_cameras(const Resource& resource);
    [[nodiscard]] Result<std::vector<InputCapture>> read_input_captures(const Resource& resource);
    [[nodiscard]] Result<PointCloudManifest> read_point_cloud_manifest(
        const Resource& gltf_resource, const std::filesystem::path& project_root);
    [[nodiscard]] Result<std::vector<CalibratedCamera>> read_calibrated_cameras(
        const Resource& resource);

} // namespace lfs::io::opf

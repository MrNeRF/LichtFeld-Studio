/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "io/loaders/opf_loader.hpp"

#include "core/path_utils.hpp"
#include "io/formats/opf.hpp"

#include <algorithm>
#include <chrono>

namespace lfs::io {
    namespace {
        template <typename Predicate>
        const opf::Resource* find_resource(const opf::Project& project, Predicate predicate) {
            for (const auto& item : project.items)
                for (const auto& resource : item.resources)
                    if (predicate(item, resource))
                        return &resource;
            return nullptr;
        }
    } // namespace

    bool OpfLoader::canLoad(const std::filesystem::path& path) const {
        if (std::filesystem::is_regular_file(path))
            return path.extension().string() == ".opf";
        if (!std::filesystem::is_directory(path))
            return false;
        std::error_code ec;
        for (std::filesystem::directory_iterator it(path, ec), end; !ec && it != end; it.increment(ec)) {
            if (it->is_regular_file(ec) && it->path().extension() == ".opf")
                return true;
        }
        return false;
    }

    Result<LoadResult> OpfLoader::load(const std::filesystem::path& path,
                                       const LoadOptions& options) {
        const auto started = std::chrono::steady_clock::now();
        std::filesystem::path project_path = path;
        if (std::filesystem::is_directory(path)) {
            std::error_code ec;
            std::vector<std::filesystem::path> projects;
            for (std::filesystem::directory_iterator it(path, ec), end; !ec && it != end; it.increment(ec)) {
                if (it->is_regular_file(ec) && it->path().extension() == ".opf")
                    projects.push_back(it->path());
            }
            if (projects.size() != 1)
                return make_error(ErrorCode::INVALID_DATASET,
                                  "OPF dataset folder must contain exactly one .opf project file.", path);
            project_path = projects.front();
        }
        auto project = opf::read_project(project_path);
        if (!project)
            return std::unexpected(project.error());

        const auto* camera_list_resource = find_resource(
            *project, [](const auto& item, const auto& resource) {
                return item.type == "camera_list" &&
                       resource.format == "application/opf-camera-list+json";
            });
        const auto* input_resource = find_resource(
            *project, [](const auto& item, const auto& resource) {
                return item.type == "input_cameras" &&
                       resource.format == "application/opf-input-cameras+json";
            });
        const auto* calibrated_resource = find_resource(
            *project, [](const auto& item, const auto& resource) {
                return item.type == "calibration" &&
                       resource.format == "application/opf-calibrated-cameras+json";
            });
        const auto* reference_frame_resource = find_resource(
            *project, [](const auto& item, const auto& resource) {
                return item.type == "scene_reference_frame" &&
                       resource.format == "application/opf-scene-reference-frame+json";
            });
        const auto* sparse_resource = find_resource(
            *project, [](const auto& item, const auto& resource) {
                return (item.type == "calibration" || item.type == "point_cloud") &&
                       resource.format == "model/gltf+json" &&
                       resource.uri.find("sparse/") != std::string::npos;
            });
        if (!camera_list_resource || !input_resource || !calibrated_resource)
            return make_error(ErrorCode::MISSING_REQUIRED_FILES,
                              "OPF project requires camera_list, input_cameras, and calibrated cameras resources.",
                              project_path);

        auto images = opf::read_camera_list(*camera_list_resource, project_path.parent_path());
        if (!images)
            return std::unexpected(images.error());
        auto sensors = opf::read_input_cameras(*input_resource);
        if (!sensors)
            return std::unexpected(sensors.error());
        auto calibrated = opf::read_calibrated_cameras(*calibrated_resource);
        if (!calibrated)
            return std::unexpected(calibrated.error());
        auto imported = opf::assemble_cameras(*images, *sensors, *calibrated);
        if (!imported)
            return std::unexpected(imported.error());
        if (reference_frame_resource) {
            auto parsed_reference_frame = opf::read_scene_reference_frame(*reference_frame_resource);
            if (!parsed_reference_frame)
                return std::unexpected(parsed_reference_frame.error());
        }

        std::shared_ptr<PointCloud> point_cloud;
        if (sparse_resource) {
            auto manifest = opf::read_point_cloud_manifest(*sparse_resource, project_path.parent_path());
            if (!manifest)
                return std::unexpected(manifest.error());
            // Calibrated cameras and OPF sparse coordinates are already in the
            // canonical dataset frame. The scene reference frame's shift is
            // georeference metadata, not an additional local-scene transform.
            auto sparse = opf::read_sparse_point_cloud(*manifest, nullptr);
            if (!sparse)
                return std::unexpected(sparse.error());
            for (auto& camera : *imported)
                opf::apply_gltf_node_transform(camera, manifest->node_matrix);
            point_cloud = std::make_shared<PointCloud>(std::move(*sparse));
        }
        for (auto& camera : *imported)
            opf::apply_lichtfeld_coordinate_convention(camera);

        LoadedScene scene;
        scene.point_cloud = std::move(point_cloud);
        if (!options.validate_only) {
            for (const auto& camera : *imported) {
                auto loaded_camera = opf::make_camera(camera);
                if (!loaded_camera)
                    return std::unexpected(loaded_camera.error());
                scene.cameras.push_back(std::move(*loaded_camera));
            }
        }

        LoadResult result;
        result.data = std::move(scene);
        result.loader_used = "OPF";
        result.warnings = project->warnings;
        result.load_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - started);
        return result;
    }
} // namespace lfs::io

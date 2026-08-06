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
        return std::filesystem::is_regular_file(path) &&
               path.extension().string() == ".opf";
    }

    Result<LoadResult> OpfLoader::load(const std::filesystem::path& path,
                                       const LoadOptions& options) {
        const auto started = std::chrono::steady_clock::now();
        auto project = opf::read_project(path);
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
        if (!camera_list_resource || !input_resource || !calibrated_resource)
            return make_error(ErrorCode::MISSING_REQUIRED_FILES,
                              "OPF project requires camera_list, input_cameras, and calibrated cameras resources.",
                              path);

        auto images = opf::read_camera_list(*camera_list_resource, path.parent_path());
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

        LoadedScene scene;
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

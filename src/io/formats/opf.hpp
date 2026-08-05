/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "io/error.hpp"
#include <filesystem>
#include <string>
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

    // Parses and validates the OPF project graph. Resource files are resolved
    // relative to the directory containing the project file and must remain
    // inside that directory. Unknown extension items/resources are retained
    // as warnings and never make an otherwise valid project fail.
    [[nodiscard]] Result<Project> read_project(const std::filesystem::path& path);
    [[nodiscard]] Result<std::vector<CameraImage>> read_camera_list(const Resource& resource,
                                                                    const std::filesystem::path& project_root);

} // namespace lfs::io::opf

/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "io/loader_service.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "io/error.hpp"
#include "io/loaders/blender_loader.hpp"
#include "io/loaders/checkpoint_loader.hpp"
#include "io/loaders/colmap_loader.hpp"
#include "io/loaders/ply_loader.hpp"
#include "io/loaders/sogs_loader.hpp"
#include "io/loaders/spz_loader.hpp"
#include <format>

namespace lfs::io {

    LoaderService::LoaderService()
        : registry_(std::make_unique<DataLoaderRegistry>()) {

        // Register default loaders
        registry_->registerLoader(std::make_unique<PLYLoader>());
        registry_->registerLoader(std::make_unique<SogLoader>());
        registry_->registerLoader(std::make_unique<SpzLoader>());
        registry_->registerLoader(std::make_unique<CheckpointLoader>());
        registry_->registerLoader(std::make_unique<ColmapLoader>());
        registry_->registerLoader(std::make_unique<BlenderLoader>());

        LOG_DEBUG("LoaderService initialized with {} loaders", registry_->size());
    }

    Result<LoadResult> LoaderService::load(
        const std::filesystem::path& path,
        const LoadOptions& options) {

        // Check if path exists first
        std::error_code ec;
        if (!std::filesystem::exists(path, ec)) {
            return make_error(ErrorCode::PATH_NOT_FOUND, "", path);
        }

        // Find appropriate loader
        auto* loader = registry_->findLoader(path);
        if (!loader) {
            // Build user-friendly error message
            const bool is_directory = std::filesystem::is_directory(path, ec);
            const std::string path_type = is_directory ? "folder" : "file";
            const std::string filename = lfs::core::path_to_utf8(path.filename());

            std::string message;
            if (is_directory) {
                message = std::format(
                    "The folder '{}' is not a recognized dataset.\n\n"
                    "For COLMAP datasets, ensure the folder contains:\n"
                    "  - cameras.bin (or cameras.txt)\n"
                    "  - images.bin (or images.txt)\n"
                    "  - An 'images' folder with your photos\n\n"
                    "For NeRF/Blender datasets, ensure the folder contains:\n"
                    "  - transforms.json (or transforms_train.json)",
                    filename);
            } else {
                auto ext = path.extension().string();
                message = std::format(
                    "Cannot open '{}' - unsupported file format.\n\n"
                    "Supported formats:\n"
                    "  - Gaussian Splat files: .ply, .sog, .spz\n"
                    "  - Training checkpoints: .resume\n"
                    "  - NeRF transforms: .json",
                    filename);
            }

            LOG_ERROR("Unsupported format: {} ({})", lfs::core::path_to_utf8(path), path_type);
            return make_error(ErrorCode::UNSUPPORTED_FORMAT, message, path);
        }

        LOG_INFO("Using {} loader for: {}", loader->name(), lfs::core::path_to_utf8(path));

        // Perform the load - let the loader return proper errors
        return loader->load(path, options);
    }

    std::vector<std::string> LoaderService::getAvailableLoaders() const {
        std::vector<std::string> names;
        for (const auto& info : registry_->getLoaderInfo()) {
            names.push_back(info.name);
        }
        return names;
    }

    std::vector<std::string> LoaderService::getSupportedExtensions() const {
        return registry_->getAllSupportedExtensions();
    }

} // namespace lfs::io

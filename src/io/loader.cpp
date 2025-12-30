/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "io/loader.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "io/filesystem_utils.hpp"
#include "loader_service.hpp"
#include <algorithm>
#include <cctype>
#include <filesystem>
#include <memory>
#include <system_error>

namespace lfs::io {

    namespace {
        // Implementation class that hides all internal details
        class LoaderImpl : public Loader {
        public:
            LoaderImpl() : service_(std::make_unique<LoaderService>()) {
                LOG_TRACE("LoaderImpl created");
            }

            [[nodiscard]] Result<LoadResult> load(
                const std::filesystem::path& path,
                const LoadOptions& options) override {

                LOG_DEBUG("Loading from path: {}", lfs::core::path_to_utf8(path));
                // Just delegate to the service
                return service_->load(path, options);
            }

            bool canLoad(const std::filesystem::path& path) const override {
                // Check if any registered loader can handle this path
                if (!safe_exists(path)) {
                    LOG_TRACE("Path does not exist: {}", lfs::core::path_to_utf8(path));
                    return false;
                }

                // Check for SOG files
                if (path.extension() == ".sog" || path.extension() == ".SOG") {
                    LOG_TRACE("SOG file detected: {}", lfs::core::path_to_utf8(path));
                    return true;
                }

                // Check for directory with meta.json (SOG directory format)
                if (safe_is_directory(path) && safe_exists(path / "meta.json")) {
                    LOG_TRACE("SOG directory detected: {}", lfs::core::path_to_utf8(path));
                    return true;
                }

                // Check for meta.json file directly (SOG)
                if (path.filename() == "meta.json") {
                    LOG_TRACE("SOG meta.json detected: {}", lfs::core::path_to_utf8(path));
                    return true;
                }

                // Check other file extensions
                auto ext = path.extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

                // Check common extensions
                if (ext == ".ply") {
                    LOG_TRACE("PLY file detected: {}", lfs::core::path_to_utf8(path));
                    return true;
                }

                if (ext == ".resume") {
                    LOG_TRACE("Checkpoint file detected: {}", lfs::core::path_to_utf8(path));
                    return true;
                }

                if (ext == ".json") {
                    LOG_TRACE("JSON file detected (potential transforms): {}", lfs::core::path_to_utf8(path));
                    return true;
                }

                // Check for COLMAP dataset
                if (safe_is_directory(path)) {
                    auto colmap_paths = get_colmap_search_paths(path);
                    const std::vector<std::string> colmap_markers = {
                        "cameras.bin", "cameras.txt", "images.bin", "images.txt"};

                    for (const auto& marker : colmap_markers) {
                        if (!find_file_in_paths(colmap_paths, marker).empty()) {
                            LOG_TRACE("COLMAP dataset detected: {}", lfs::core::path_to_utf8(path));
                            return true;
                        }
                    }

                    // Check for Blender/NeRF dataset
                    if (safe_exists(path / "transforms.json") ||
                        safe_exists(path / "transforms_train.json")) {
                        LOG_TRACE("Blender/NeRF dataset detected: {}", lfs::core::path_to_utf8(path));
                        return true;
                    }
                }

                LOG_TRACE("No compatible loader found for: {}", lfs::core::path_to_utf8(path));
                return false;
            }

            std::vector<std::string> getSupportedFormats() const override {
                return service_->getAvailableLoaders();
            }

            std::vector<std::string> getSupportedExtensions() const override {
                return service_->getSupportedExtensions();
            }

        private:
            std::unique_ptr<LoaderService> service_;
        };
    } // namespace

    // Factory method implementation
    std::unique_ptr<Loader> Loader::create() {
        LOG_DEBUG("Creating Loader instance");
        return std::make_unique<LoaderImpl>();
    }

    bool Loader::isDatasetPath(const std::filesystem::path& path) {
        if (!safe_exists(path)) {
            LOG_TRACE("Path does not exist for dataset check: {}", lfs::core::path_to_utf8(path));
            return false;
        }

        if (!safe_is_directory(path)) {
            auto ext = path.extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

            // JSON files might be datasets (transforms.json)
            if (ext == ".json") {
                LOG_TRACE("JSON file detected, treating as potential dataset: {}", lfs::core::path_to_utf8(path));
                return true;
            }

            // SOG files are NOT datasets - they're single splat files like PLY
            if (ext == ".sog") {
                LOG_TRACE("SOG file detected, not a dataset: {}", lfs::core::path_to_utf8(path));
                return false;
            }

            // PLY files are definitely not datasets
            if (ext == ".ply") {
                LOG_TRACE("PLY file detected, not a dataset: {}", lfs::core::path_to_utf8(path));
                return false;
            }

            // Checkpoint files are not datasets
            if (ext == ".resume") {
                LOG_TRACE("Checkpoint file detected, not a dataset: {}", lfs::core::path_to_utf8(path));
                return false;
            }

            LOG_TRACE("Non-dataset file detected: {}", lfs::core::path_to_utf8(path));
            return false;
        }

        // Check for SOG directory format (with meta.json)
        // This is a special case - SOG directories are treated as single files, not datasets
        if (safe_exists(path / "meta.json")) {
            // Need to check if this is a SOG directory or something else
            // SOG directories have specific WebP files
            bool has_sog_files = safe_exists(path / "means_l.webp") ||
                                 safe_exists(path / "means_u.webp") ||
                                 safe_exists(path / "quats.webp") ||
                                 safe_exists(path / "scales.webp") ||
                                 safe_exists(path / "sh0.webp");

            if (has_sog_files) {
                LOG_TRACE("SOG directory detected, not a dataset: {}", lfs::core::path_to_utf8(path));
                return false; // SOG directories are treated as single splat files
            }
        }

        // Check for COLMAP markers in any standard location
        auto colmap_paths = get_colmap_search_paths(path);
        const std::vector<std::string> colmap_markers = {
            "cameras.bin", "cameras.txt", "images.bin", "images.txt"};

        for (const auto& marker : colmap_markers) {
            if (!find_file_in_paths(colmap_paths, marker).empty()) {
                LOG_TRACE("COLMAP dataset detected at: {}", lfs::core::path_to_utf8(path));
                return true;
            }
        }

        // Blender/NeRF markers
        if (safe_exists(path / "transforms.json") ||
            safe_exists(path / "transforms_train.json")) {
            LOG_TRACE("Blender/NeRF dataset detected at: {}", lfs::core::path_to_utf8(path));
            return true;
        }

        LOG_TRACE("No dataset markers found in directory: {}", lfs::core::path_to_utf8(path));
        return false;
    }

    // Static method to determine dataset type
    DatasetType Loader::getDatasetType(const std::filesystem::path& path) {
        if (!safe_exists(path)) {
            return DatasetType::Unknown;
        }

        if (!safe_is_directory(path)) {
            auto ext = path.extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".json") {
                return DatasetType::Transforms;
            }
            // SOG files are not datasets
            if (ext == ".sog") {
                return DatasetType::Unknown;
            }
            return DatasetType::Unknown;
        }

        // Check if it's a SOG directory (not a dataset)
        if (safe_exists(path / "meta.json")) {
            bool has_sog_files = safe_exists(path / "means_l.webp") ||
                                 safe_exists(path / "means_u.webp") ||
                                 safe_exists(path / "quats.webp") ||
                                 safe_exists(path / "scales.webp") ||
                                 safe_exists(path / "sh0.webp");

            if (has_sog_files) {
                return DatasetType::Unknown; // SOG is not a dataset type
            }
        }

        // Check for COLMAP markers
        auto colmap_paths = get_colmap_search_paths(path);
        const std::vector<std::string> colmap_markers = {
            "cameras.bin", "cameras.txt", "images.bin", "images.txt"};

        for (const auto& marker : colmap_markers) {
            if (!find_file_in_paths(colmap_paths, marker).empty()) {
                return DatasetType::COLMAP;
            }
        }

        // Check for Transforms markers
        if (safe_exists(path / "transforms.json") ||
            safe_exists(path / "transforms_train.json")) {
            return DatasetType::Transforms;
        }

        return DatasetType::Unknown;
    }

} // namespace lfs::io

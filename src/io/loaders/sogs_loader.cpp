/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "sogs_loader.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "core/splat_data.hpp"
#include "formats/sogs.hpp"
#include "io/error.hpp"
#include <chrono>
#include <filesystem>
#include <format>
#include <fstream>

namespace lfs::io {

    // Import types from lfs::core for convenience
    using lfs::core::DataType;
    using lfs::core::Device;
    using lfs::core::SplatData;
    using lfs::core::Tensor;

    Result<LoadResult> SogLoader::load(
        const std::filesystem::path& path,
        const LoadOptions& options) {

        LOG_TIMER("SOG Loading");
        auto start_time = std::chrono::high_resolution_clock::now();

        // Report progress if callback provided
        if (options.progress) {
            options.progress(0.0f, "Loading SOG file...");
        }

        // Validate path exists
        if (!std::filesystem::exists(path)) {
            return make_error(ErrorCode::PATH_NOT_FOUND,
                              "SOG file/directory does not exist", path);
        }

        // Validation only mode
        if (options.validate_only) {
            LOG_DEBUG("Validation only mode for SOG: {}", lfs::core::path_to_utf8(path));

            bool valid = false;

            // Check if it's a .sog bundle
            if (path.extension() == ".sog" && std::filesystem::is_regular_file(path)) {
                // Basic validation - check if it's a valid archive
                std::ifstream file;
                if (lfs::core::open_file_for_read(path, std::ios::binary, file)) {
                    // Check for ZIP/archive header (PK)
                    char header[2];
                    file.read(header, 2);
                    if (header[0] == 'P' && header[1] == 'K') {
                        valid = true;
                    }
                }
            }
            // Check if it's a directory with meta.json
            else if (std::filesystem::is_directory(path)) {
                if (std::filesystem::exists(path / "meta.json")) {
                    valid = true;
                }
            }
            // Check if it's a meta.json file directly
            else if (path.filename() == "meta.json" && std::filesystem::is_regular_file(path)) {
                valid = true;
            }

            if (!valid) {
                return make_error(ErrorCode::INVALID_HEADER,
                                  "Invalid SOG format (expected ZIP archive or directory with meta.json)", path);
            }

            if (options.progress) {
                options.progress(100.0f, "SOG validation complete");
            }

            LOG_DEBUG("SOG validation successful");

            // Return empty result for validation only
            LoadResult result;
            result.data = std::shared_ptr<SplatData>{}; // Empty shared_ptr
            result.scene_center = Tensor::zeros({3}, Device::CPU);
            result.loader_used = name();
            result.load_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - start_time);
            result.warnings = {};

            return result;
        }

        // Load the SOG file
        if (options.progress) {
            options.progress(50.0f, "Parsing SOG data...");
        }

        LOG_INFO("Loading SOG file: {}", lfs::core::path_to_utf8(path));
        auto splat_result = load_sog(path);
        if (!splat_result) {
            return make_error(ErrorCode::CORRUPTED_DATA,
                              std::format("Failed to load SOG: {}", splat_result.error()), path);
        }

        if (options.progress) {
            options.progress(100.0f, "SOG loading complete");
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto load_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time);

        LoadResult result{
            .data = std::make_shared<SplatData>(std::move(*splat_result)),
            .scene_center = Tensor::zeros({3}, Device::CPU),
            .loader_used = name(),
            .load_time = load_time,
            .warnings = {}};

        LOG_INFO("SOG loaded successfully in {}ms", load_time.count());

        return result;
    }

    bool SogLoader::canLoad(const std::filesystem::path& path) const {
        if (!std::filesystem::exists(path)) {
            return false;
        }

        // Check for .sog bundle file
        if (path.extension() == ".sog" && std::filesystem::is_regular_file(path)) {
            return true;
        }

        // Check for directory with meta.json
        if (std::filesystem::is_directory(path)) {
            return std::filesystem::exists(path / "meta.json");
        }

        // Check if it's a meta.json file directly
        if (path.filename() == "meta.json" && std::filesystem::is_regular_file(path)) {
            return true;
        }

        return false;
    }

    std::string SogLoader::name() const {
        return "SOG";
    }

    std::vector<std::string> SogLoader::supportedExtensions() const {
        return {".sog", ".SOG"};
    }

    int SogLoader::priority() const {
        return 15; // Higher priority than PLY since it's more compact
    }

} // namespace lfs::io
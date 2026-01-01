/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "spz_loader.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "core/splat_data.hpp"
#include "formats/spz.hpp"
#include "io/error.hpp"
#include <chrono>
#include <filesystem>
#include <format>
#include <fstream>

namespace lfs::io {

    using lfs::core::Device;
    using lfs::core::SplatData;
    using lfs::core::Tensor;

    Result<LoadResult> SpzLoader::load(
        const std::filesystem::path& path,
        const LoadOptions& options) {

        LOG_TIMER("SPZ Loading");
        auto start_time = std::chrono::high_resolution_clock::now();

        if (options.progress) {
            options.progress(0.0f, "Loading SPZ file...");
        }

        if (!std::filesystem::exists(path)) {
            return make_error(ErrorCode::PATH_NOT_FOUND,
                              "SPZ file does not exist", path);
        }

        // Validation only mode
        if (options.validate_only) {
            LOG_DEBUG("Validation only mode for SPZ: {}", lfs::core::path_to_utf8(path));

            // Check for gzip magic bytes
            std::ifstream file;
            if (!lfs::core::open_file_for_read(path, std::ios::binary, file)) {
                return make_error(ErrorCode::READ_FAILURE,
                                  "Cannot open SPZ file", path);
            }

            uint8_t header[2];
            file.read(reinterpret_cast<char*>(header), 2);

            // Gzip magic: 0x1f 0x8b
            if (header[0] != 0x1f || header[1] != 0x8b) {
                return make_error(ErrorCode::INVALID_HEADER,
                                  "Invalid SPZ format (expected gzip compressed data)", path);
            }

            if (options.progress) {
                options.progress(100.0f, "SPZ validation complete");
            }

            LoadResult result;
            result.data = std::shared_ptr<SplatData>{};
            result.scene_center = Tensor::zeros({3}, Device::CPU);
            result.loader_used = name();
            result.load_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - start_time);
            result.warnings = {};

            return result;
        }

        if (options.progress) {
            options.progress(50.0f, "Decompressing SPZ data...");
        }

        LOG_INFO("Loading SPZ file: {}", lfs::core::path_to_utf8(path));
        auto splat_result = load_spz(path);
        if (!splat_result) {
            return make_error(ErrorCode::CORRUPTED_DATA,
                              std::format("Failed to load SPZ: {}", splat_result.error()), path);
        }

        if (options.progress) {
            options.progress(100.0f, "SPZ loading complete");
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

        LOG_INFO("SPZ loaded successfully in {}ms", load_time.count());

        return result;
    }

    bool SpzLoader::canLoad(const std::filesystem::path& path) const {
        if (!std::filesystem::exists(path)) {
            return false;
        }

        if (!std::filesystem::is_regular_file(path)) {
            return false;
        }

        auto ext = path.extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        return ext == ".spz";
    }

    std::string SpzLoader::name() const {
        return "SPZ";
    }

    std::vector<std::string> SpzLoader::supportedExtensions() const {
        return {".spz", ".SPZ"};
    }

    int SpzLoader::priority() const {
        return 16; // Higher priority than SOG since it's more compact
    }

} // namespace lfs::io

/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "usd_loader.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "core/splat_data.hpp"
#include "formats/usd.hpp"
#include "io/error.hpp"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <format>

namespace lfs::io {

    using lfs::core::Device;
    using lfs::core::SplatData;
    using lfs::core::Tensor;

    Result<LoadResult> USDLoader::load(
        const std::filesystem::path& path,
        const LoadOptions& options) {

        LOG_TIMER("USD Loading");
        const auto start_time = std::chrono::high_resolution_clock::now();

        if (options.progress) {
            options.progress(0.0f, "Loading USD gaussian file...");
        }

        if (!std::filesystem::exists(path)) {
            return make_error(ErrorCode::PATH_NOT_FOUND,
                              "USD file does not exist",
                              path);
        }

        if (!std::filesystem::is_regular_file(path)) {
            return make_error(ErrorCode::NOT_A_FILE,
                              "Path is not a regular file",
                              path);
        }

        if (options.validate_only) {
            LOG_DEBUG("Validation only mode for USD: {}", lfs::core::path_to_utf8(path));

            auto validation_result = validate_usd(path);
            if (!validation_result) {
                return make_error(ErrorCode::INVALID_HEADER,
                                  std::format("Invalid USD gaussian file: {}", validation_result.error()),
                                  path);
            }

            if (options.progress) {
                options.progress(100.0f, "USD validation complete");
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
            options.progress(50.0f, "Parsing OpenUSD ParticleField...");
        }

        auto splat_result = load_usd(path);
        if (!splat_result) {
            return make_error(ErrorCode::CORRUPTED_DATA,
                              std::format("Failed to load USD gaussian file: {}", splat_result.error()),
                              path);
        }

        if (options.progress) {
            options.progress(100.0f, "USD loading complete");
        }

        const auto load_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start_time);

        LoadResult result{
            .data = std::make_shared<SplatData>(std::move(*splat_result)),
            .scene_center = Tensor::zeros({3}, Device::CPU),
            .loader_used = name(),
            .load_time = load_time,
            .warnings = {}};

        LOG_INFO("USD gaussian file loaded successfully in {}ms", load_time.count());
        return result;
    }

    bool USDLoader::canLoad(const std::filesystem::path& path) const {
        if (!std::filesystem::exists(path) || std::filesystem::is_directory(path)) {
            return false;
        }

        auto ext = path.extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        return ext == ".usd" || ext == ".usda" || ext == ".usdc" || ext == ".usdz";
    }

    std::string USDLoader::name() const {
        return "OpenUSD";
    }

    std::vector<std::string> USDLoader::supportedExtensions() const {
        return {".usd", ".USD", ".usda", ".USDA", ".usdc", ".USDC", ".usdz", ".USDZ"};
    }

    int USDLoader::priority() const {
        return 17;
    }

} // namespace lfs::io

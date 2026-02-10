/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "ply_loader.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "core/splat_data.hpp"
#include "formats/ply.hpp"
#include "io/error.hpp"
#include <cassert>
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

    Result<LoadResult> PLYLoader::load(
        const std::filesystem::path& path,
        const LoadOptions& options) {

        LOG_TIMER("PLY Loading");
        auto start_time = std::chrono::high_resolution_clock::now();

        // Report progress if callback provided
        if (options.progress) {
            options.progress(0.0f, "Loading PLY file...");
        }

        // Validate file exists
        if (!std::filesystem::exists(path)) {
            return make_error(ErrorCode::PATH_NOT_FOUND,
                              "PLY file does not exist", path);
        }

        if (!std::filesystem::is_regular_file(path)) {
            return make_error(ErrorCode::NOT_A_FILE,
                              "Path is not a regular file", path);
        }

        // Validation only mode
        if (options.validate_only) {
            LOG_DEBUG("Validation only mode for PLY: {}", lfs::core::path_to_utf8(path));
            // Basic validation - check if it's a PLY file
            std::ifstream file;
            if (!lfs::core::open_file_for_read(path, std::ios::binary, file)) {
                return make_error(ErrorCode::PERMISSION_DENIED,
                                  "Cannot open file for reading", path);
            }

            std::string header;
            std::getline(file, header);
            if (header != "ply" && header != "ply\r") {
                return make_error(ErrorCode::INVALID_HEADER,
                                  "File does not start with 'ply' header", path);
            }

            if (options.progress) {
                options.progress(100.0f, "PLY validation complete");
            }

            LOG_DEBUG("PLY validation successful");

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

        if (options.progress) {
            options.progress(50.0f, "Parsing PLY data...");
        }

        const bool is_gaussian = is_gaussian_splat_ply(path);
        LOG_INFO("Loading PLY: {} ({})", lfs::core::path_to_utf8(path), is_gaussian ? "gaussian" : "point cloud");

        auto splat_result = is_gaussian
                                ? load_ply(path)
                                : [&]() -> std::expected<SplatData, std::string> {
            const auto pc_result = load_ply_point_cloud(path);
            if (!pc_result)
                return std::unexpected(pc_result.error());

            const auto& pc = *pc_result;
            const size_t N = static_cast<size_t>(pc.size());
            assert(N > 0);

            Tensor means = pc.means.to(Device::CUDA);
            assert(means.shape()[0] == N && means.shape()[1] == 3);

            Tensor sh0 = (pc.colors.is_valid() && pc.colors.numel() > 0)
                             ? (((pc.colors.dtype() == DataType::UInt8
                                      ? pc.colors.to(DataType::Float32) / 255.0f
                                      : pc.colors) -
                                 0.5f) /
                                ply_constants::SH_C0)
                                   .reshape({static_cast<int>(N), 1, 3})
                                   .to(Device::CUDA)
                             : Tensor::zeros({N, 1, 3}, Device::CUDA);

            Tensor shN = Tensor::zeros({N, 0, 3}, Device::CUDA);
            Tensor scaling = Tensor::full({N, 3}, ply_constants::DEFAULT_LOG_SCALE, Device::CUDA);

            auto rot_cpu = Tensor::zeros({N, 4}, Device::CPU);
            float* const rot_ptr = rot_cpu.ptr<float>();
            for (size_t i = 0; i < N; ++i)
                rot_ptr[i * 4] = 1.0f;
            Tensor rotation = rot_cpu.to(Device::CUDA);

            Tensor opacity = Tensor::zeros({N, 1}, Device::CUDA);

            return SplatData(
                0,
                std::move(means),
                std::move(sh0),
                std::move(shN),
                std::move(scaling),
                std::move(rotation),
                std::move(opacity),
                ply_constants::SCENE_SCALE_FACTOR);
        }();

        if (!splat_result) {
            return make_error(ErrorCode::CORRUPTED_DATA,
                              std::format("Failed to load PLY: {}", splat_result.error()), path);
        }

        if (options.progress) {
            options.progress(100.0f, "PLY loading complete");
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

        LOG_INFO("PLY loaded successfully in {}ms", load_time.count());

        return result;
    }

    bool PLYLoader::canLoad(const std::filesystem::path& path) const {
        if (!std::filesystem::exists(path) || std::filesystem::is_directory(path)) {
            return false;
        }

        auto ext = path.extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext != ".ply")
            return false;

        if (ply_has_faces(path))
            return false;

        return true;
    }

    std::string PLYLoader::name() const {
        return "PLY";
    }

    std::vector<std::string> PLYLoader::supportedExtensions() const {
        return {".ply", ".PLY"};
    }

    int PLYLoader::priority() const {
        return 10; // Higher priority for PLY files
    }

} // namespace lfs::io
/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "spz.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "core/tensor.hpp"
#include "load-spz.h"
#include <chrono>
#include <format>

namespace lfs::io {

    using lfs::core::DataType;
    using lfs::core::Device;
    using lfs::core::SplatData;
    using lfs::core::Tensor;

    namespace {
        // SH coefficient count per degree: 0->0, 1->3, 2->8, 3->15
        constexpr int SH_COEFFS_FOR_DEGREE[] = {0, 3, 8, 15};
        constexpr float SCENE_SCALE = 0.5f; // Match PLY loader

        SplatData convert_from_spz(const spz::GaussianCloud& cloud) {
            const auto num_points = static_cast<size_t>(cloud.numPoints);
            const int sh_degree = cloud.shDegree;
            const auto sh_coeffs = sh_degree > 0 ? static_cast<size_t>(SH_COEFFS_FOR_DEGREE[sh_degree]) : 0;

            // sh0 must be [N, 1, 3] to match PLY loader format
            auto means = Tensor::empty({num_points, 3}, Device::CPU, DataType::Float32);
            auto sh0 = Tensor::empty({num_points, 1, 3}, Device::CPU, DataType::Float32);
            auto scaling = Tensor::empty({num_points, 3}, Device::CPU, DataType::Float32);
            auto rotation = Tensor::empty({num_points, 4}, Device::CPU, DataType::Float32);
            auto opacity = Tensor::empty({num_points, 1}, Device::CPU, DataType::Float32);

            Tensor shN;
            if (sh_coeffs > 0) {
                shN = Tensor::empty({num_points, sh_coeffs, 3}, Device::CPU, DataType::Float32);
            }

            auto* const means_ptr = static_cast<float*>(means.data_ptr());
            auto* const sh0_ptr = static_cast<float*>(sh0.data_ptr());
            auto* const scaling_ptr = static_cast<float*>(scaling.data_ptr());
            auto* const rotation_ptr = static_cast<float*>(rotation.data_ptr());
            auto* const opacity_ptr = static_cast<float*>(opacity.data_ptr());
            auto* const shN_ptr = sh_coeffs > 0 ? static_cast<float*>(shN.data_ptr()) : nullptr;

            std::copy(cloud.positions.begin(), cloud.positions.end(), means_ptr);
            std::copy(cloud.scales.begin(), cloud.scales.end(), scaling_ptr);
            std::copy(cloud.colors.begin(), cloud.colors.end(), sh0_ptr);

            // Rotation: SPZ xyzw -> SplatData wxyz
            for (size_t i = 0; i < num_points; ++i) {
                rotation_ptr[i * 4 + 0] = cloud.rotations[i * 4 + 3]; // w
                rotation_ptr[i * 4 + 1] = cloud.rotations[i * 4 + 0]; // x
                rotation_ptr[i * 4 + 2] = cloud.rotations[i * 4 + 1]; // y
                rotation_ptr[i * 4 + 3] = cloud.rotations[i * 4 + 2]; // z
            }

            for (size_t i = 0; i < num_points; ++i) {
                opacity_ptr[i] = cloud.alphas[i];
            }

            if (sh_coeffs > 0 && !cloud.sh.empty()) {
                std::copy(cloud.sh.begin(), cloud.sh.end(), shN_ptr);
            }

            return SplatData(
                sh_degree,
                std::move(means),
                std::move(sh0),
                std::move(shN),
                std::move(scaling),
                std::move(rotation),
                std::move(opacity),
                SCENE_SCALE);
        }

        spz::GaussianCloud convert_to_spz(const SplatData& splat) {
            const auto num_points = static_cast<int>(splat.size());
            const int sh_degree = splat.get_max_sh_degree();
            const int sh_coeffs = sh_degree > 0 ? SH_COEFFS_FOR_DEGREE[sh_degree] : 0;

            spz::GaussianCloud cloud;
            cloud.numPoints = num_points;
            cloud.shDegree = sh_degree;
            cloud.antialiased = false;

            const auto means = splat.means().contiguous().to(Device::CPU);
            const auto scaling = splat.scaling_raw().contiguous().to(Device::CPU);
            const auto rotation = splat.rotation_raw().contiguous().to(Device::CPU);
            const auto opacity = splat.opacity_raw().contiguous().to(Device::CPU);
            const auto sh0 = splat.sh0().contiguous().to(Device::CPU);

            cloud.positions.resize(num_points * 3);
            cloud.scales.resize(num_points * 3);
            cloud.rotations.resize(num_points * 4);
            cloud.alphas.resize(num_points);
            cloud.colors.resize(num_points * 3);

            const auto* const means_ptr = static_cast<const float*>(means.data_ptr());
            const auto* const scaling_ptr = static_cast<const float*>(scaling.data_ptr());
            const auto* const rotation_ptr = static_cast<const float*>(rotation.data_ptr());
            const auto* const opacity_ptr = static_cast<const float*>(opacity.data_ptr());
            const auto* const sh0_ptr = static_cast<const float*>(sh0.data_ptr());

            std::copy(means_ptr, means_ptr + num_points * 3, cloud.positions.begin());
            std::copy(scaling_ptr, scaling_ptr + num_points * 3, cloud.scales.begin());
            std::copy(sh0_ptr, sh0_ptr + num_points * 3, cloud.colors.begin());

            // Rotation: SplatData wxyz -> SPZ xyzw
            for (int i = 0; i < num_points; ++i) {
                cloud.rotations[i * 4 + 0] = rotation_ptr[i * 4 + 1]; // x
                cloud.rotations[i * 4 + 1] = rotation_ptr[i * 4 + 2]; // y
                cloud.rotations[i * 4 + 2] = rotation_ptr[i * 4 + 3]; // z
                cloud.rotations[i * 4 + 3] = rotation_ptr[i * 4 + 0]; // w
            }

            for (int i = 0; i < num_points; ++i) {
                cloud.alphas[i] = opacity_ptr[i];
            }

            if (sh_coeffs > 0 && splat.shN().is_valid()) {
                const auto shN = splat.shN().contiguous().to(Device::CPU);
                cloud.sh.resize(num_points * sh_coeffs * 3);
                const auto* const shN_ptr = static_cast<const float*>(shN.data_ptr());
                std::copy(shN_ptr, shN_ptr + num_points * sh_coeffs * 3, cloud.sh.begin());
            }

            return cloud;
        }
    } // namespace

    std::expected<SplatData, std::string> load_spz(const std::filesystem::path& filepath) {
        auto start = std::chrono::high_resolution_clock::now();

        LOG_INFO("Loading SPZ file: {}", lfs::core::path_to_utf8(filepath));

        // Load using Niantic's library (outputs RDF coordinate system like PLY)
        spz::UnpackOptions options;
        options.to = spz::CoordinateSystem::RDF;
        auto cloud = spz::loadSpz(lfs::core::path_to_utf8(filepath), options);

        if (cloud.numPoints == 0) {
            return std::unexpected(std::format("Failed to load SPZ file: {}", lfs::core::path_to_utf8(filepath)));
        }

        LOG_DEBUG("SPZ loaded: {} points, SH degree {}", cloud.numPoints, cloud.shDegree);

        auto splat = convert_from_spz(cloud);

        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start);
        LOG_INFO("SPZ loaded: {} gaussians with SH degree {} in {}ms",
                 splat.size(), splat.get_max_sh_degree(), elapsed.count());

        return splat;
    }

    Result<void> save_spz(const SplatData& splat_data, const SpzSaveOptions& options) {
        auto start = std::chrono::high_resolution_clock::now();

        LOG_INFO("Saving SPZ file: {}", lfs::core::path_to_utf8(options.output_path));

        auto cloud = convert_to_spz(splat_data);

        // Save using Niantic's library (input is RDF coordinate system like PLY)
        spz::PackOptions pack_options;
        pack_options.from = spz::CoordinateSystem::RDF;

        // Pack to memory first, then write to file ourselves to handle Unicode paths correctly
        std::vector<uint8_t> data;
        if (!spz::saveSpz(cloud, pack_options, &data)) {
            return make_error(ErrorCode::WRITE_FAILURE,
                              "Failed to pack SPZ data", options.output_path);
        }

        // Write file using std::ofstream with Unicode path handling
        std::ofstream out;
        if (!lfs::core::open_file_for_write(options.output_path, std::ios::binary | std::ios::out, out)) {
            return make_error(ErrorCode::WRITE_FAILURE,
                              "Failed to open SPZ file for writing", options.output_path);
        }
        out.write(reinterpret_cast<const char*>(data.data()), data.size());
        out.close();
        if (!out.good()) {
            return make_error(ErrorCode::WRITE_FAILURE,
                              "Failed to write SPZ file", options.output_path);
        }

        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start);

        // Get file size
        auto file_size = std::filesystem::file_size(options.output_path);
        LOG_INFO("SPZ saved: {} gaussians, {:.1f} MB in {}ms",
                 splat_data.size(),
                 static_cast<double>(file_size) / (1024.0 * 1024.0),
                 elapsed.count());

        return {};
    }

} // namespace lfs::io

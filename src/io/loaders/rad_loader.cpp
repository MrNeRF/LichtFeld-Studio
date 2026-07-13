/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "rad_loader.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "core/splat_data.hpp"
#include "formats/rad.hpp"
#include "io/error.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <string>

namespace lfs::io {

    using lfs::core::Device;
    using lfs::core::SplatData;
    using lfs::core::Tensor;

    namespace {
        bool radPagedGpuUploadRequested(const SplatData& data) {
            return rad_paged_load_recommended(data);
        }
    } // namespace

    Result<LoadResult> RadLoader::load(
        const std::filesystem::path& path,
        const LoadOptions& options) {

        LOG_TIMER("RAD Loading");
        auto start_time = std::chrono::high_resolution_clock::now();

        if (options.progress) {
            options.progress(0.0f, "Loading RAD file...");
        }

        if (!std::filesystem::exists(path)) {
            return make_error(ErrorCode::PATH_NOT_FOUND,
                              "RAD file does not exist", path);
        }

        // Validation only mode
        if (options.validate_only) {
            LOG_DEBUG("Validation only mode for RAD: {}", lfs::core::path_to_utf8(path));

            std::ifstream file;
            if (!lfs::core::open_file_for_read(path, std::ios::binary, file)) {
                return make_error(ErrorCode::READ_FAILURE,
                                  "Cannot open RAD file", path);
            }

            uint8_t header[4];
            file.read(reinterpret_cast<char*>(header), 4);

            // RAD magic: "RAD0" in little-endian = 0x30444152
            if (header[0] != 0x52 || header[1] != 0x41 || header[2] != 0x44 || header[3] != 0x30) {
                return make_error(ErrorCode::INVALID_HEADER,
                                  "Invalid RAD format (expected 'RAD0' magic)", path);
            }

            if (options.progress) {
                options.progress(100.0f, "RAD validation complete");
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
            options.progress(50.0f, "Decoding RAD data...");
        }

        LOG_INFO("Loading RAD file: {}", lfs::core::path_to_utf8(path));
        auto splat_result = load_rad(path);
        if (!splat_result) {
            return make_error(ErrorCode::CORRUPTED_DATA,
                              std::format("Failed to load RAD: {}", splat_result.error()), path);
        }

        SplatData& data = *splat_result;
        if (radPagedGpuUploadRequested(data)) {
            // Paged residency streams chunk payloads from disk, and that path
            // is sidecar-backed end to end (upload engine, page sink, and
            // selector metadata all gate on the meta view). Out-of-core loads
            // attach it inside load_rad; in-core paged loads ensure it here.
            if (!data.lod_tree->meta_view.valid()) {
                auto view = open_rad_meta_sidecar(path);
                if (!view) {
                    if (auto built = build_rad_meta_sidecar(path); !built) {
                        return make_error(ErrorCode::CORRUPTED_DATA,
                                          std::format("RAD paged load requires the metadata "
                                                      "sidecar: {}",
                                                      built.error().message),
                                          path);
                    }
                    view = open_rad_meta_sidecar(path);
                }
                if (!view) {
                    return make_error(ErrorCode::CORRUPTED_DATA,
                                      std::format("RAD paged load requires the metadata "
                                                  "sidecar: {}",
                                                  view.error()),
                                      path);
                }
                data.lod_tree->meta_view = *view;
            }
            LOG_INFO("RAD paged LOD active: deferring full CUDA tensor migration (chunks={})",
                     data.lod_tree->chunk_count());
        } else {
            // Move tensors to CUDA for Vulkan renderer compatibility.
            data.means_raw() = data.means_raw().to(Device::CUDA);
            data.sh0_raw() = data.sh0_raw().to(Device::CUDA);
            if (data.shN_raw().is_valid() && data.shN_raw().numel() > 0) {
                data.shN_raw() = data.shN_raw().to(Device::CUDA);
            }
            data.scaling_raw() = data.scaling_raw().to(Device::CUDA);
            data.rotation_raw() = data.rotation_raw().to(Device::CUDA);
            data.opacity_raw() = data.opacity_raw().to(Device::CUDA);
            if (data.has_deleted_mask()) {
                data.deleted() = data.deleted().to(Device::CUDA);
            }
        }

        if (options.progress) {
            options.progress(100.0f, "RAD loading complete");
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto load_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time);

        LoadResult result{
            .data = std::make_shared<SplatData>(std::move(data)),
            .scene_center = Tensor::zeros({3}, Device::CPU),
            .loader_used = name(),
            .load_time = load_time,
            .warnings = {}};

        LOG_INFO("RAD loaded successfully in {}ms", load_time.count());

        return result;
    }

    bool RadLoader::canLoad(const std::filesystem::path& path) const {
        if (!std::filesystem::exists(path)) {
            return false;
        }

        if (std::filesystem::is_directory(path)) {
            return false;
        }

        auto ext = path.extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        return ext == ".rad";
    }

    std::string RadLoader::name() const {
        return "RAD";
    }

    std::vector<std::string> RadLoader::supportedExtensions() const {
        return {".rad", ".RAD"};
    }

    int RadLoader::priority() const {
        return 20; // High priority since it's our native format with LOD
    }

} // namespace lfs::io

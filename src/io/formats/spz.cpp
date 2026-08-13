/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "spz.hpp"
#include "coordinate-system-adobe.h"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "core/provenance.hpp"
#include "core/tensor.hpp"
#include "io/atomic_output.hpp"
#include "load-spz.h"
#include "provenance-lichtfeld.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <format>
#include <fstream>
#include <memory>
#include <zlib.h>

namespace lfs::io {

    using lfs::core::DataType;
    using lfs::core::Device;
    using lfs::core::SplatData;
    using lfs::core::Tensor;

    namespace {
        // SH coefficient count per degree: 0->0, 1->3, 2->8, 3->15
        constexpr int SH_COEFFS_FOR_DEGREE[] = {0, 3, 8, 15};
        constexpr float SCENE_SCALE = 0.5f; // Match PLY loader

        const char* coordinate_system_name(spz::CoordinateSystem system) {
            switch (system) {
            case spz::CoordinateSystem::UNSPECIFIED: return "UNSPECIFIED";
            case spz::CoordinateSystem::LDB: return "LDB";
            case spz::CoordinateSystem::RDB: return "RDB";
            case spz::CoordinateSystem::LUB: return "LUB";
            case spz::CoordinateSystem::RUB: return "RUB";
            case spz::CoordinateSystem::LDF: return "LDF";
            case spz::CoordinateSystem::RDF: return "RDF";
            case spz::CoordinateSystem::LUF: return "LUF";
            case spz::CoordinateSystem::RUF: return "RUF";
            case spz::CoordinateSystem::LFD: return "LFD";
            case spz::CoordinateSystem::RFD: return "RFD";
            case spz::CoordinateSystem::LFU: return "LFU";
            case spz::CoordinateSystem::RFU: return "RFU";
            case spz::CoordinateSystem::LBD: return "LBD";
            case spz::CoordinateSystem::RBD: return "RBD";
            case spz::CoordinateSystem::LBU: return "LBU";
            case spz::CoordinateSystem::RBU: return "RBU";
            }
            return "UNKNOWN";
        }

        void log_coordinate_system_extension(const spz::GaussianCloud& cloud) {
#ifdef SPZ_BUILD_EXTENSIONS
            const auto coord_ext =
                spz::findExtensionByType<spz::SpzExtensionCoordinateSystemAdobe>(cloud.extensions);
            if (!coord_ext) {
                return;
            }
            const spz::CoordinateSystem system = coord_ext->resolve();
            if (system == spz::CoordinateSystem::RUB) {
                LOG_INFO("SPZ: file declares coordinate system RUB (matches default assumption)");
            } else {
                LOG_WARN(
                    "SPZ: file declares coordinate system {} (differs from historical RUB "
                    "assumption); declared system was honoured",
                    coordinate_system_name(system));
            }
#else
            (void)cloud;
#endif
        }

        std::string unsupported_spz_version_message(uint32_t version) {
            return std::format(
                "unsupported SPZ version {} (this build supports 1-{})",
                version, spz::LATEST_SPZ_HEADER_VERSION);
        }

        // Partially inflate a gzip container to recover the 16-byte SPZ header.
        // Returns true only when all 16 header bytes are produced.
        bool try_inflate_gzip_spz_header(
            const std::vector<uint8_t>& data, uint8_t out[16]) {
            z_stream stream{};
            if (inflateInit2(&stream, 16 + MAX_WBITS) != Z_OK) {
                return false;
            }
            stream.next_in =
                const_cast<Bytef*>(reinterpret_cast<const Bytef*>(data.data()));
            stream.avail_in = static_cast<uInt>(data.size());
            stream.next_out = out;
            stream.avail_out = 16;
            const int status = inflate(&stream, Z_NO_FLUSH);
            const size_t produced = 16u - static_cast<size_t>(stream.avail_out);
            inflateEnd(&stream);
            return (status == Z_OK || status == Z_STREAM_END) && produced == 16u;
        }

        std::expected<void, std::string> validate_spz_cloud(
            const spz::GaussianCloud& cloud) {
            if (cloud.numPoints <= 0 ||
                static_cast<uint32_t>(cloud.numPoints) > spz::kMaxSpzPoints) {
                return std::unexpected("SPZ header contains invalid point or SH metadata");
            }
            if (cloud.shDegree < 0 || cloud.shDegree > 3) {
                return std::unexpected(
                    "SPZ SH degree-4 files are not supported by LichtFeld yet "
                    "(supported degrees: 0-3)");
            }

            const size_t count = static_cast<size_t>(cloud.numPoints);
            const size_t sh_coefficients =
                cloud.shDegree > 0
                    ? static_cast<size_t>(SH_COEFFS_FOR_DEGREE[cloud.shDegree])
                    : 0;
            if (cloud.positions.size() != count * 3 ||
                cloud.scales.size() != count * 3 ||
                cloud.rotations.size() != count * 4 ||
                cloud.alphas.size() != count ||
                cloud.colors.size() != count * 3 ||
                cloud.sh.size() != count * sh_coefficients * 3) {
                return std::unexpected("SPZ decoded attribute sizes do not match the point count");
            }

            const auto finite = [](const std::vector<float>& values) {
                return std::ranges::all_of(values, [](const float value) {
                    return std::isfinite(value);
                });
            };
            if (!finite(cloud.positions) || !finite(cloud.scales) ||
                !finite(cloud.rotations) || !finite(cloud.alphas) ||
                !finite(cloud.colors) || !finite(cloud.sh)) {
                return std::unexpected("SPZ decoded attributes contain a non-finite value");
            }
            return {};
        }

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

            if (sh_coeffs > 0 && splat.shN().is_valid() && splat.shN().numel() > 0) {
                // shN is stored swizzled; unpack on CPU to avoid a canonical CUDA copy.
                const auto shN = splat.shN_canonical_cpu().contiguous();
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

        try {
            std::ifstream in;
            if (!lfs::core::open_file_for_read(filepath, std::ios::binary | std::ios::ate, in)) {
                return std::unexpected(std::format("Failed to open SPZ file: {}", lfs::core::path_to_utf8(filepath)));
            }

            const auto size = in.tellg();
            if (size <= 0 ||
                static_cast<uint64_t>(size) > spz::kMaxSpzCompressedBytes) {
                return std::unexpected(std::format(
                    "SPZ file must contain 1..{} compressed bytes: {}",
                    spz::kMaxSpzCompressedBytes,
                    lfs::core::path_to_utf8(filepath)));
            }

            std::vector<uint8_t> data(static_cast<size_t>(size));
            in.seekg(0, std::ios::beg);
            if (!in.read(reinterpret_cast<char*>(data.data()),
                         static_cast<std::streamsize>(data.size()))) {
                return std::unexpected(std::format("Failed to read SPZ file: {}", lfs::core::path_to_utf8(filepath)));
            }

            // Sniff the container so unsupported versions fail with a clear message.
            if (data.size() >= 2 && data[0] == 0x1f && data[1] == 0x8b) {
                // Legacy gzip container (v1–v3): partially inflate only the first
                // 16 header bytes so future versions get a clear error.
                uint8_t header[16]{};
                if (try_inflate_gzip_spz_header(data, header)) {
                    uint32_t magic = 0;
                    uint32_t version = 0;
                    std::memcpy(&magic, header, sizeof(magic));
                    std::memcpy(&version, header + 4, sizeof(version));
                    if (magic != spz::NGSP_MAGIC) {
                        return std::unexpected(std::format(
                            "not an SPZ file: {}", lfs::core::path_to_utf8(filepath)));
                    }
                    if (version > static_cast<uint32_t>(spz::LATEST_SPZ_HEADER_VERSION)) {
                        return std::unexpected(unsupported_spz_version_message(version));
                    }
                }
                // Truncated/corrupt gzip: fall through to spz::loadSpz.
            } else if (data.size() >= 8) {
                uint32_t magic = 0;
                std::memcpy(&magic, data.data(), sizeof(magic));
                if (magic == spz::NGSP_MAGIC) {
                    uint32_t version = 0;
                    std::memcpy(&version, data.data() + 4, sizeof(version));
                    if (version > static_cast<uint32_t>(spz::LATEST_SPZ_HEADER_VERSION)) {
                        return std::unexpected(unsupported_spz_version_message(version));
                    }
                } else {
                    return std::unexpected(std::format(
                        "not an SPZ file: {}", lfs::core::path_to_utf8(filepath)));
                }
            } else {
                return std::unexpected(std::format(
                    "not an SPZ file: {}", lfs::core::path_to_utf8(filepath)));
            }

            // Load through the in-memory API to avoid narrow-path handling in the bundled SPZ library.
            spz::UnpackOptions options;
            options.to = spz::CoordinateSystem::RDF;
            auto cloud = spz::loadSpz(data, options);
            if (auto validation = validate_spz_cloud(cloud); !validation) {
                return std::unexpected(std::format(
                    "Failed to load SPZ file '{}': {}",
                    lfs::core::path_to_utf8(filepath),
                    validation.error()));
            }

            log_coordinate_system_extension(cloud);

            LOG_DEBUG("SPZ loaded: {} points, SH degree {}", cloud.numPoints, cloud.shDegree);

            auto splat = convert_from_spz(cloud);

            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - start);
            LOG_INFO("SPZ loaded: {} gaussians with SH degree {} in {}ms",
                     splat.size(), splat.get_max_sh_degree(), elapsed.count());

            return splat;
        } catch (const std::bad_alloc&) {
            return std::unexpected("SPZ input exceeds available memory");
        } catch (const std::exception& error) {
            return std::unexpected(std::format("Failed to load SPZ: {}", error.what()));
        }
    }

    Result<void> save_spz(const SplatData& splat_data, const SpzSaveOptions& options_in) {
        SpzSaveOptions options = options_in;
        // v3 has no extension zone; leave the slot empty so legacy files stay clean.
        if (options.version == 4 && !options.provenance) {
            options.provenance = core::make_minimal_provenance_stamp();
        }

        auto start = std::chrono::high_resolution_clock::now();

        LOG_INFO("Saving SPZ file: {}", lfs::core::path_to_utf8(options.output_path));

        if (!report_export_progress(options.progress_callback, 0.0f, "Preparing SPZ")) {
            return make_error(ErrorCode::CANCELLED, "SPZ export cancelled", options.output_path);
        }
        if (options.version != 3 && options.version != 4) {
            return make_error(
                ErrorCode::UNSUPPORTED_FORMAT,
                std::format("SPZ export version must be 3 or 4 (got {})", options.version),
                options.output_path);
        }
        if (splat_data.size() == 0 || splat_data.size() > spz::kMaxSpzPoints) {
            return make_error(
                ErrorCode::INVALID_DATASET,
                std::format("SPZ export supports 1..{} splats", spz::kMaxSpzPoints),
                options.output_path);
        }

        auto cloud = convert_to_spz(splat_data);

        if (!report_export_progress(options.progress_callback, 0.4f, "Packing SPZ")) {
            return make_error(ErrorCode::CANCELLED, "SPZ export cancelled", options.output_path);
        }

        // Save using Niantic's library (input is RDF coordinate system like PLY).
        // Version 4 attaches the coordinate-system extension declaring RUB on disk.
        // Version 3 must attach no extensions: legacy readers hard-reject trailing bytes.
        spz::PackOptions pack_options;
        pack_options.from = spz::CoordinateSystem::RDF;
        pack_options.version = static_cast<uint32_t>(options.version);

#ifdef SPZ_BUILD_EXTENSIONS
        if (options.version == 4) {
            auto coord_ext = std::make_shared<spz::SpzExtensionCoordinateSystemAdobe>();
            coord_ext->coordinateSystem = spz::CoordinateSystem::RUB;
            cloud.extensions.push_back(coord_ext);
            if (options.provenance) {
                auto provenance_ext = std::make_shared<spz::SpzExtensionProvenanceLichtFeld>();
                provenance_ext->json = core::provenance_to_json(*options.provenance);
                cloud.extensions.push_back(std::move(provenance_ext));
            }
        }
#endif

        // Pack to memory first, then write to file ourselves to handle Unicode paths correctly
        std::vector<uint8_t> data;
        if (!spz::saveSpz(cloud, pack_options, &data)) {
            return make_error(ErrorCode::WRITE_FAILURE,
                              "Failed to pack SPZ data", options.output_path);
        }

        if (!report_export_progress(options.progress_callback, 0.9f, "Writing SPZ")) {
            return make_error(ErrorCode::CANCELLED, "SPZ export cancelled", options.output_path);
        }

        if (auto dir_result = ensure_output_parent_directory(options.output_path); !dir_result) {
            return std::unexpected(dir_result.error());
        }

        ScopedAtomicOutputFile atomic_output(options.output_path);
        std::ofstream file;
        if (!lfs::core::open_file_for_write(atomic_output.temp_path(), std::ios::binary | std::ios::out, file)) {
            return make_error(ErrorCode::WRITE_FAILURE,
                              "Failed to open temporary SPZ file for writing",
                              atomic_output.temp_path());
        }

        file.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
        file.close();

        if (!file.good()) {
            return make_error(ErrorCode::WRITE_FAILURE,
                              "Failed to write SPZ file", atomic_output.temp_path());
        }

        if (!report_export_progress(options.progress_callback, 1.0f, "SPZ export complete")) {
            return make_error(ErrorCode::CANCELLED, "SPZ export cancelled", options.output_path);
        }

        if (auto commit_result = atomic_output.commit(); !commit_result) {
            return std::unexpected(commit_result.error());
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

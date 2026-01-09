/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#ifdef _WIN32
#define NOMINMAX
#endif

#include "sogs.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "core/tensor.hpp"
#include "cuda/kmeans.hpp"
#include "cuda/morton_encoding.hpp"
#include "io/error.hpp"
#include <archive.h>
#include <archive_entry.h>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <webp/decode.h>
#include <webp/encode.h>

namespace lfs::io {

    // Import types from lfs::core for convenience
    using lfs::core::DataType;
    using lfs::core::Device;
    using lfs::core::SplatData;
    using lfs::core::Tensor;

    namespace {

#ifdef _WIN32
        using ssize_t = std::ptrdiff_t;
#endif

        // Identity layout matching the exporter
        int identity_layout(int index, [[maybe_unused]] int width) {
            return index;
        }

        // SH coefficient counts per degree
        constexpr int SH_COEFFS[] = {0, 3, 8, 15};

        float inverse_log_transform(float value) {
            float sign = value >= 0 ? 1.0f : -1.0f;
            return sign * (std::exp(std::abs(value)) - 1.0f);
        }

        std::array<float, 4> unpack_quaternion(
            uint8_t a, uint8_t b, uint8_t c, uint8_t type) {

            // Determine which component was largest during packing
            int largest = type - 252; // 0=w, 1=x, 2=y, 3=z
            if (largest < 0 || largest > 3) {
                LOG_WARN("Invalid quaternion type: {}, defaulting to w", type);
                largest = 0;
            }

            // Unpack the three stored components with sqrt(2) scaling
            constexpr float sqrt2 = 1.41421356237f;
            float v0 = (a / 255.0f - 0.5f) * sqrt2;
            float v1 = (b / 255.0f - 0.5f) * sqrt2;
            float v2 = (c / 255.0f - 0.5f) * sqrt2;

            // Reconstruct the largest component
            float largest_val = std::sqrt(std::clamp(1.0f - (v0 * v0 + v1 * v1 + v2 * v2), 0.0f, 1.0f));

            // Build the quaternion [x, y, z, w] based on what was packed
            std::array<float, 4> quat; // [x, y, z, w]

            if (largest == 0) {
                // w was largest, stored [x, y, z]
                quat[0] = v0;          // x
                quat[1] = v1;          // y
                quat[2] = v2;          // z
                quat[3] = largest_val; // w
            } else if (largest == 1) {
                // x was largest, stored [w, y, z]
                quat[0] = largest_val; // x
                quat[1] = v1;          // y
                quat[2] = v2;          // z
                quat[3] = v0;          // w
            } else if (largest == 2) {
                // y was largest, stored [w, x, z]
                quat[0] = v1;          // x
                quat[1] = largest_val; // y
                quat[2] = v2;          // z
                quat[3] = v0;          // w
            } else {                   // largest == 3
                // z was largest, stored [w, x, y]
                quat[0] = v1;          // x
                quat[1] = v2;          // y
                quat[2] = largest_val; // z
                quat[3] = v0;          // w
            }

            // Normalize quaternion
            float len = std::sqrt(quat[0] * quat[0] + quat[1] * quat[1] +
                                  quat[2] * quat[2] + quat[3] * quat[3]);
            if (len > 0) {
                for (auto& v : quat)
                    v /= len;
            }

            return quat; // Returns [x, y, z, w]
        }

        std::expected<std::vector<uint8_t>, std::string> decode_webp(
            const uint8_t* data, size_t size, int& width, int& height) {

            if (!data || size == 0) {
                return std::unexpected("Invalid WebP data");
            }

            // Get image info
            if (!WebPGetInfo(data, size, &width, &height)) {
                return std::unexpected("Failed to get WebP info");
            }

            // Decode RGBA
            std::vector<uint8_t> rgba(width * height * 4);
            uint8_t* result = WebPDecodeRGBA(data, size, &width, &height);

            if (!result) {
                return std::unexpected("Failed to decode WebP image");
            }

            // Copy to vector
            std::memcpy(rgba.data(), result, width * height * 4);
            WebPFree(result);

            return rgba;
        }

        struct SogMetadata {
            int version = 0;
            int count = 0;
            int width = 0;
            int height = 0;

            // Position bounds
            std::vector<float> means_mins;
            std::vector<float> means_maxs;
            std::vector<std::string> means_files;

            // Scale codebook
            std::vector<float> scales_codebook;
            std::vector<std::string> scales_files;

            // Quaternion files
            std::vector<std::string> quats_files;

            // Color codebook
            std::vector<float> sh0_codebook;
            std::vector<std::string> sh0_files;

            // Optional spherical harmonics
            struct SHData {
                std::vector<float> codebook;
                int palette_size = 0;
                int bands = 0;
                int coeffs = 0;
                std::vector<std::string> files;
            };
            std::optional<SHData> shN;
        };

        std::expected<SogMetadata, std::string> parse_metadata(
            const std::string& json_str) {

            try {
                auto json = nlohmann::json::parse(json_str);
                SogMetadata meta;

                // Basic fields
                meta.version = json.at("version").get<int>();
                meta.count = json.at("count").get<int>();

                // Width and height might not be in meta.json (calculated from texture)
                if (json.contains("width")) {
                    meta.width = json["width"].get<int>();
                }
                if (json.contains("height")) {
                    meta.height = json["height"].get<int>();
                }

                // Position bounds
                auto means = json.at("means");
                meta.means_mins = means.at("mins").get<std::vector<float>>();
                meta.means_maxs = means.at("maxs").get<std::vector<float>>();
                meta.means_files = means.at("files").get<std::vector<std::string>>();

                // Scales
                auto scales = json.at("scales");
                meta.scales_codebook = scales.at("codebook").get<std::vector<float>>();
                meta.scales_files = scales.at("files").get<std::vector<std::string>>();

                // Quaternions
                auto quats = json.at("quats");
                meta.quats_files = quats.at("files").get<std::vector<std::string>>();

                // Colors
                auto sh0 = json.at("sh0");
                meta.sh0_codebook = sh0.at("codebook").get<std::vector<float>>();
                meta.sh0_files = sh0.at("files").get<std::vector<std::string>>();

                // Optional spherical harmonics
                if (json.contains("shN")) {
                    auto shN = json.at("shN");
                    SogMetadata::SHData sh_data;
                    sh_data.codebook = shN.at("codebook").get<std::vector<float>>();
                    sh_data.files = shN.at("files").get<std::vector<std::string>>();

                    // Optional fields
                    if (shN.contains("count")) {
                        sh_data.palette_size = shN["count"].get<int>();
                    } else if (shN.contains("palette_size")) {
                        sh_data.palette_size = shN["palette_size"].get<int>();
                    }
                    if (shN.contains("bands")) {
                        sh_data.bands = shN["bands"].get<int>();
                    }
                    if (shN.contains("coeffs")) {
                        sh_data.coeffs = shN["coeffs"].get<int>();
                    }

                    meta.shN = sh_data;
                }

                return meta;

            } catch (const std::exception& e) {
                return std::unexpected(std::format("Failed to parse metadata: {}", e.what()));
            }
        }

        std::expected<SplatData, std::string> reconstruct_splat_data(
            const SogMetadata& meta,
            const std::unordered_map<std::string, std::vector<uint8_t>>& images) {

            const int num_splats = meta.count;

            // If width/height not in metadata, calculate from count
            int width = meta.width;
            int height = meta.height;
            if (width == 0 || height == 0) {
                width = ((int)std::ceil(std::sqrt(num_splats) / 4.0)) * 4;
                height = ((int)std::ceil(num_splats / (float)width / 4.0)) * 4;
            }

            LOG_DEBUG("Reconstructing {} splats from {}x{} textures", num_splats, width, height);

            // Create host buffers
            std::vector<float> host_means(num_splats * 3);
            std::vector<float> host_scales(num_splats * 3);
            std::vector<float> host_rotations(num_splats * 4);
            std::vector<float> host_opacity(num_splats);

            // Determine SH dimensions
            int sh0_dim1 = 1, sh0_dim2 = 3;
            int shN_dim1 = 0, shN_dim2 = 3;

            if (meta.shN.has_value()) {
                const auto& sh_meta = meta.shN.value();
                int sh_degree = sh_meta.bands > 0 ? sh_meta.bands : (sh_meta.coeffs == 3 ? 1 : sh_meta.coeffs == 8 ? 2
                                                                                           : sh_meta.coeffs == 15  ? 3
                                                                                                                   : 0);
                shN_dim1 = SH_COEFFS[sh_degree];
            }

            std::vector<float> host_sh0(num_splats * sh0_dim1 * sh0_dim2);
            std::vector<float> host_shN(num_splats * shN_dim1 * shN_dim2);

            // 1. Decode positions from means_l and means_u
            {
                auto it_l = images.find("means_l.webp");
                auto it_u = images.find("means_u.webp");

                if (it_l == images.end() || it_u == images.end()) {
                    return std::unexpected("Missing position textures");
                }

                const auto& means_l = it_l->second;
                const auto& means_u = it_u->second;

                for (int i = 0; i < num_splats; ++i) {
                    int ti = identity_layout(i, width) * 4;

                    // Reconstruct 16-bit values
                    uint16_t x16 = means_l[ti + 0] | (means_u[ti + 0] << 8);
                    uint16_t y16 = means_l[ti + 1] | (means_u[ti + 1] << 8);
                    uint16_t z16 = means_l[ti + 2] | (means_u[ti + 2] << 8);

                    // Normalize and inverse transform
                    float x_norm = x16 / 65535.0f;
                    float y_norm = y16 / 65535.0f;
                    float z_norm = z16 / 65535.0f;

                    float x_log = x_norm * (meta.means_maxs[0] - meta.means_mins[0]) + meta.means_mins[0];
                    float y_log = y_norm * (meta.means_maxs[1] - meta.means_mins[1]) + meta.means_mins[1];
                    float z_log = z_norm * (meta.means_maxs[2] - meta.means_mins[2]) + meta.means_mins[2];

                    host_means[i * 3 + 0] = inverse_log_transform(x_log);
                    host_means[i * 3 + 1] = inverse_log_transform(y_log);
                    host_means[i * 3 + 2] = inverse_log_transform(z_log);
                }
            }

            // 2. Decode quaternions
            {
                auto it = images.find("quats.webp");
                if (it == images.end()) {
                    return std::unexpected("Missing quaternion texture");
                }

                const auto& quats = it->second;

                for (int i = 0; i < num_splats; ++i) {
                    int ti = identity_layout(i, width) * 4;

                    auto quat = unpack_quaternion(
                        quats[ti + 0],
                        quats[ti + 1],
                        quats[ti + 2],
                        quats[ti + 3]);

                    // unpack_quaternion returns [x, y, z, w]
                    // Store as [w, x, y, z] for SplatData format
                    host_rotations[i * 4 + 0] = quat[3]; // w
                    host_rotations[i * 4 + 1] = quat[0]; // x
                    host_rotations[i * 4 + 2] = quat[1]; // y
                    host_rotations[i * 4 + 3] = quat[2]; // z
                }
            }

            // 3. Decode scales
            {
                auto it = images.find("scales.webp");
                if (it == images.end()) {
                    return std::unexpected("Missing scales texture");
                }

                const auto& scales_img = it->second;

                for (int i = 0; i < num_splats; ++i) {
                    int ti = identity_layout(i, width) * 4;

                    // Get indices and validate
                    uint8_t idx0 = scales_img[ti + 0];
                    uint8_t idx1 = scales_img[ti + 1];
                    uint8_t idx2 = scales_img[ti + 2];

                    // Ensure indices are within codebook bounds
                    if (idx0 >= meta.scales_codebook.size() ||
                        idx1 >= meta.scales_codebook.size() ||
                        idx2 >= meta.scales_codebook.size()) {
                        LOG_ERROR("Scale codebook index out of bounds: {}, {}, {} (codebook size: {})",
                                  idx0, idx1, idx2, meta.scales_codebook.size());
                        return std::unexpected("Invalid scale codebook index");
                    }

                    // Look up from codebook (already in log space)
                    host_scales[i * 3 + 0] = meta.scales_codebook[idx0];
                    host_scales[i * 3 + 1] = meta.scales_codebook[idx1];
                    host_scales[i * 3 + 2] = meta.scales_codebook[idx2];
                }
            }

            // 4. Decode colors and opacity
            {
                auto it = images.find("sh0.webp");
                if (it == images.end()) {
                    return std::unexpected("Missing color texture");
                }

                const auto& sh0_img = it->second;

                for (int i = 0; i < num_splats; ++i) {
                    int ti = identity_layout(i, width) * 4;

                    // Get indices and validate
                    uint8_t idx0 = sh0_img[ti + 0];
                    uint8_t idx1 = sh0_img[ti + 1];
                    uint8_t idx2 = sh0_img[ti + 2];

                    // Ensure indices are within codebook bounds
                    if (idx0 >= meta.sh0_codebook.size() ||
                        idx1 >= meta.sh0_codebook.size() ||
                        idx2 >= meta.sh0_codebook.size()) {
                        LOG_ERROR("Color codebook index out of bounds: {}, {}, {} (codebook size: {})",
                                  idx0, idx1, idx2, meta.sh0_codebook.size());
                        return std::unexpected("Invalid color codebook index");
                    }

                    // Look up colors from codebook
                    host_sh0[i * sh0_dim1 * sh0_dim2 + 0] = meta.sh0_codebook[idx0];
                    host_sh0[i * sh0_dim1 * sh0_dim2 + 1] = meta.sh0_codebook[idx1];
                    host_sh0[i * sh0_dim1 * sh0_dim2 + 2] = meta.sh0_codebook[idx2];

                    // Decode opacity (inverse sigmoid)
                    // Alpha=1 encodes opacity=0 (prevents WebP discarding RGB)
                    const uint8_t alpha = sh0_img[ti + 3];
                    float opacity_norm = (alpha <= 1) ? 1e-5f : alpha / 255.0f;
                    opacity_norm = std::clamp(opacity_norm, 1e-5f, 1.0f - 1e-5f);
                    host_opacity[i] = std::log(opacity_norm / (1.0f - opacity_norm));
                }
            }

            // 5. Decode spherical harmonics if present
            if (meta.shN.has_value() && shN_dim1 > 0) {
                const auto& sh_meta = meta.shN.value();

                auto it_centroids = images.find("shN_centroids.webp");
                auto it_labels = images.find("shN_labels.webp");

                if (it_centroids != images.end() && it_labels != images.end()) {
                    const auto& centroids_img = it_centroids->second;
                    const auto& labels_img = it_labels->second;

                    // Determine SH configuration
                    int sh_degree = sh_meta.bands > 0 ? sh_meta.bands : (sh_meta.coeffs == 3 ? 1 : sh_meta.coeffs == 8 ? 2
                                                                                               : sh_meta.coeffs == 15  ? 3
                                                                                                                       : 0);

                    int num_coeffs = SH_COEFFS[sh_degree];
                    int palette_size = static_cast<int>(sh_meta.palette_size > 0 ? sh_meta.palette_size : centroids_img.size() / (64 * num_coeffs * 4));

                    LOG_DEBUG("Decoding SH: degree={}, coeffs={}, palette_size={}",
                              sh_degree, num_coeffs, palette_size);

                    // Decode centroids from texture
                    std::vector<std::vector<float>> centroids(palette_size);
                    for (int i = 0; i < palette_size; ++i) {
                        centroids[i].resize(num_coeffs * 3);

                        for (int j = 0; j < num_coeffs; ++j) {
                            int pixel_idx = i * num_coeffs + j;

                            // Decode from codebook
                            for (int c = 0; c < 3; ++c) {
                                uint8_t idx = centroids_img[pixel_idx * 4 + c];

                                // Validate index
                                if (idx >= sh_meta.codebook.size()) {
                                    LOG_ERROR("SH codebook index out of bounds: {} (codebook size: {})",
                                              idx, sh_meta.codebook.size());
                                    return std::unexpected("Invalid SH codebook index");
                                }

                                // Band-major ordering
                                int coeff_idx = j + c * num_coeffs;
                                centroids[i][coeff_idx] = sh_meta.codebook[idx];
                            }
                        }
                    }

                    // Apply labels
                    for (int i = 0; i < num_splats; ++i) {
                        int ti = identity_layout(i, width) * 4;

                        // Reconstruct label from 16-bit value
                        int label = labels_img[ti + 0] | (labels_img[ti + 1] << 8);

                        if (label < palette_size) {
                            const auto& centroid = centroids[label];

                            // Unpack in band-major order
                            for (int c = 0; c < 3; ++c) {
                                for (int j = 0; j < num_coeffs; ++j) {
                                    host_shN[i * shN_dim1 * shN_dim2 + j * shN_dim2 + c] =
                                        centroid[j + c * num_coeffs];
                                }
                            }
                        }
                    }
                }
            }

            // Create Tensors directly from host vectors (uploads to CUDA)
            const size_t N = num_splats;

            Tensor means = Tensor::from_vector(host_means, {N, 3}, Device::CUDA);
            Tensor scales = Tensor::from_vector(host_scales, {N, 3}, Device::CUDA);
            Tensor rotations = Tensor::from_vector(host_rotations, {N, 4}, Device::CUDA);
            Tensor opacity = Tensor::from_vector(host_opacity, {N, 1}, Device::CUDA);
            Tensor sh0 = Tensor::from_vector(host_sh0, {N, static_cast<size_t>(sh0_dim1), static_cast<size_t>(sh0_dim2)}, Device::CUDA);

            Tensor shN;
            if (shN_dim1 > 0) {
                shN = Tensor::from_vector(host_shN, {N, static_cast<size_t>(shN_dim1), static_cast<size_t>(shN_dim2)}, Device::CUDA);
            } else {
                shN = Tensor::zeros({N, 0, 3}, Device::CUDA);
            }

            // Calculate SH degree
            int sh_degree = meta.shN.has_value() ? meta.shN->bands : 0;

            // Create SplatData
            SplatData splat_data(
                sh_degree,
                std::move(means),
                std::move(sh0),
                std::move(shN),
                std::move(scales),
                std::move(rotations),
                std::move(opacity),
                1.0f); // scene_scale

            LOG_INFO("Successfully reconstructed {} splats", num_splats);

            return splat_data;
        }

        std::expected<SplatData, std::string> read_sog_bundle(
            const std::filesystem::path& path) {

            LOG_INFO("Reading SOG bundle: {}", lfs::core::path_to_utf8(path));

            struct archive* a = archive_read_new();
            archive_read_support_format_zip(a);
            archive_read_support_filter_all(a);

            // Use wide-character API on Windows for proper Unicode path handling
            int result;
#ifdef _WIN32
            result = archive_read_open_filename_w(a, path.wstring().c_str(), 10240);
#else
            result = archive_read_open_filename(a, path.c_str(), 10240);
#endif
            if (result != ARCHIVE_OK) {
                archive_read_free(a);
                return std::unexpected(std::format("Failed to open archive: {}",
                                                   archive_error_string(a)));
            }

            struct archive_entry* entry;
            std::string metadata_json;
            std::unordered_map<std::string, std::vector<uint8_t>> images;

            // Read all files from archive
            while (archive_read_next_header(a, &entry) == ARCHIVE_OK) {
                std::string filename = archive_entry_pathname(entry);
                size_t size = archive_entry_size(entry);

                LOG_DEBUG("Reading {} ({} bytes)", filename, size);

                std::vector<uint8_t> data(size);
                ssize_t r = archive_read_data(a, data.data(), size);

                if (r != static_cast<ssize_t>(size)) {
                    archive_read_free(a);
                    return std::unexpected(std::format("Failed to read {} from archive", filename));
                }

                if (filename == "meta.json") {
                    metadata_json = std::string(data.begin(), data.end());
                } else if (filename.ends_with(".webp")) {
                    // Decode WebP
                    int width, height;
                    auto decoded = decode_webp(data.data(), data.size(), width, height);
                    if (!decoded) {
                        archive_read_free(a);
                        return std::unexpected(decoded.error());
                    }
                    images[filename] = std::move(decoded.value());
                }
            }

            archive_read_free(a);

            if (metadata_json.empty()) {
                return std::unexpected("Missing meta.json in archive");
            }

            // Parse metadata
            auto meta_result = parse_metadata(metadata_json);
            if (!meta_result) {
                return std::unexpected(meta_result.error());
            }

            // Reconstruct SplatData
            return reconstruct_splat_data(meta_result.value(), images);
        }

        std::expected<SplatData, std::string> read_sog_directory(
            const std::filesystem::path& path) {

            LOG_INFO("Reading SOG from directory: {}", lfs::core::path_to_utf8(path));

            // Read meta.json
            auto meta_path = path / "meta.json";
            if (!std::filesystem::exists(meta_path)) {
                return std::unexpected("Missing meta.json");
            }

            std::ifstream meta_file;
            if (!lfs::core::open_file_for_read(meta_path, meta_file)) {
                return std::unexpected("Failed to open meta.json");
            }

            std::stringstream buffer;
            buffer << meta_file.rdbuf();

            auto meta_result = parse_metadata(buffer.str());
            if (!meta_result) {
                return std::unexpected(meta_result.error());
            }

            auto& meta = meta_result.value();
            std::unordered_map<std::string, std::vector<uint8_t>> images;

            // Helper to read and decode WebP files
            auto read_webp = [&](const std::string& filename) -> bool {
                auto file_path = path / filename;

                // Also check with .webp extension if not present
                if (!std::filesystem::exists(file_path) && !filename.ends_with(".webp")) {
                    file_path = path / (filename + ".webp");
                }

                if (!std::filesystem::exists(file_path)) {
                    LOG_ERROR("Missing file: {}", lfs::core::path_to_utf8(file_path));
                    return false;
                }

                // Read file
                std::ifstream file;
                if (!lfs::core::open_file_for_read(file_path, std::ios::binary, file)) {
                    LOG_ERROR("Failed to open: {}", lfs::core::path_to_utf8(file_path));
                    return false;
                }

                file.seekg(0, std::ios::end);
                size_t size = file.tellg();
                file.seekg(0, std::ios::beg);

                std::vector<uint8_t> data(size);
                file.read(reinterpret_cast<char*>(data.data()), size);

                // Decode WebP
                int width, height;
                auto decoded = decode_webp(data.data(), data.size(), width, height);
                if (!decoded) {
                    LOG_ERROR("Failed to decode {}: {}", filename, decoded.error());
                    return false;
                }

                images[filename] = std::move(decoded.value());
                return true;
            };

            // Read all required files
            for (const auto& file : meta.means_files) {
                if (!read_webp(file))
                    return std::unexpected("Failed to read " + file);
            }
            for (const auto& file : meta.scales_files) {
                if (!read_webp(file))
                    return std::unexpected("Failed to read " + file);
            }
            for (const auto& file : meta.quats_files) {
                if (!read_webp(file))
                    return std::unexpected("Failed to read " + file);
            }
            for (const auto& file : meta.sh0_files) {
                if (!read_webp(file))
                    return std::unexpected("Failed to read " + file);
            }

            // Read optional SH files
            if (meta.shN.has_value()) {
                for (const auto& file : meta.shN->files) {
                    if (!read_webp(file)) {
                        LOG_WARN("Failed to read SH file {}, continuing without SH", file);
                        meta.shN.reset();
                        break;
                    }
                }
            }

            // Reconstruct SplatData
            return reconstruct_splat_data(meta, images);
        }

    } // anonymous namespace

    std::expected<SplatData, std::string> load_sog(const std::filesystem::path& path) {
        LOG_TIMER("SOG File Loading");

        if (!std::filesystem::exists(path)) {
            std::string error_msg = std::format("SOG file/directory does not exist: {}", lfs::core::path_to_utf8(path));
            LOG_ERROR("{}", error_msg);
            return std::unexpected(error_msg);
        }

        std::expected<SplatData, std::string> result;

        // Check if it's a .sog bundle
        if (path.extension() == ".sog") {
            result = read_sog_bundle(path);
        }
        // Check if it's a meta.json file
        else if (path.filename() == "meta.json") {
            result = read_sog_directory(path.parent_path());
        }
        // Check if it's a directory
        else if (std::filesystem::is_directory(path)) {
            result = read_sog_directory(path);
        } else {
            return std::unexpected(std::format("Unknown SOG format: {}", lfs::core::path_to_utf8(path)));
        }

        return result;
    }

    // ============================================================================
    // SOG Save Implementation
    // ============================================================================

    namespace {

        double log_transform(double value) {
            return std::copysign(std::log(std::abs(value) + 1.0), value);
        }

        double sigmoid(double x) {
            return 1.0 / (1.0 + std::exp(-x));
        }

        class SogArchive {
            struct archive* a_ = nullptr;
            std::filesystem::path output_path_;
            std::string last_error_;
            bool valid_ = false;

        public:
            explicit SogArchive(const std::filesystem::path& output_path)
                : output_path_(output_path) {
                a_ = archive_write_new();
                if (!a_) {
                    last_error_ = "Failed to allocate archive structure";
                    return;
                }

                if (archive_write_set_format_zip(a_) != ARCHIVE_OK) {
                    last_error_ = std::format("Failed to set ZIP format: {}",
                                              archive_error_string(a_) ? archive_error_string(a_) : "unknown error");
                    return;
                }

                // Use wide-character API on Windows for proper Unicode path handling
                int result;
#ifdef _WIN32
                result = archive_write_open_filename_w(a_, output_path.wstring().c_str());
#else
                result = archive_write_open_filename(a_, output_path.c_str());
#endif
                if (result != ARCHIVE_OK) {
                    last_error_ = std::format("Failed to create archive '{}': {}",
                                              lfs::core::path_to_utf8(output_path),
                                              archive_error_string(a_) ? archive_error_string(a_) : "unknown error");
                    return;
                }

                valid_ = true;
            }

            ~SogArchive() {
                if (a_) {
                    if (valid_) {
                        archive_write_close(a_);
                    }
                    archive_write_free(a_);
                }
            }

            // Non-copyable, non-movable
            SogArchive(const SogArchive&) = delete;
            SogArchive& operator=(const SogArchive&) = delete;
            SogArchive(SogArchive&&) = delete;
            SogArchive& operator=(SogArchive&&) = delete;

            [[nodiscard]] bool is_valid() const { return valid_; }
            [[nodiscard]] const std::string& last_error() const { return last_error_; }

            [[nodiscard]] Result<void> add_file(const std::string& filename, const void* data, size_t size) {
                if (!valid_) {
                    return make_error(ErrorCode::ARCHIVE_CREATION_FAILED, last_error_, output_path_);
                }

                auto* entry = archive_entry_new();
                if (!entry) {
                    return make_error(ErrorCode::INTERNAL_ERROR,
                                      std::format("Failed to create archive entry for '{}'", filename), output_path_);
                }

                const auto now = std::chrono::system_clock::now();
                const auto time_t = std::chrono::system_clock::to_time_t(now);

                archive_entry_set_pathname(entry, filename.c_str());
                archive_entry_set_size(entry, static_cast<la_int64_t>(size));
                archive_entry_set_filetype(entry, AE_IFREG);
                archive_entry_set_perm(entry, 0644);
                archive_entry_set_mtime(entry, time_t, 0);

                if (archive_write_header(a_, entry) != ARCHIVE_OK) {
                    const char* err = archive_error_string(a_);
                    archive_entry_free(entry);
                    return make_error(ErrorCode::WRITE_FAILURE,
                                      std::format("Failed to write header for '{}': {}",
                                                  filename, err ? err : "unknown error"),
                                      output_path_);
                }

                const ssize_t written = archive_write_data(a_, data, size);
                archive_entry_free(entry);

                if (written != static_cast<ssize_t>(size)) {
                    const char* err = archive_error_string(a_);
                    // Check if this might be a disk space issue
                    if (written < 0) {
                        return make_error(ErrorCode::WRITE_FAILURE,
                                          std::format("Failed to write '{}': {} (wrote {} of {} bytes)",
                                                      filename, err ? err : "write error", written, size),
                                          output_path_);
                    }
                    return make_error(ErrorCode::INSUFFICIENT_DISK_SPACE,
                                      std::format("Partial write for '{}': wrote {} of {} bytes (disk full?)",
                                                  filename, written, size),
                                      output_path_);
                }

                return {};
            }

            [[nodiscard]] Result<void> add_webp(const std::string& filename,
                                                const uint8_t* data, int width, int height) {
                if (!valid_) {
                    return make_error(ErrorCode::ARCHIVE_CREATION_FAILED, last_error_, output_path_);
                }

                uint8_t* output = nullptr;
                const size_t output_size = WebPEncodeLosslessRGBA(data, width, height, width * 4, &output);

                if (output_size == 0 || !output) {
                    if (output) {
                        WebPFree(output);
                    }
                    return make_error(ErrorCode::ENCODING_FAILED,
                                      std::format("WebP encoding failed for '{}' ({}x{} image)",
                                                  filename, width, height),
                                      output_path_);
                }

                auto result = add_file(filename, output, output_size);
                WebPFree(output);
                return result;
            }
        };

        struct Cluster1dResult {
            std::vector<float> centroids;
            std::vector<uint8_t> labels;
        };

        Cluster1dResult cluster1d(const float* data, int num_rows, int num_columns, int iterations) {
            const int total_points = num_rows * num_columns;
            std::vector<float> flat_data(total_points);

            for (int col = 0; col < num_columns; ++col) {
                for (int row = 0; row < num_rows; ++row) {
                    flat_data[col * num_rows + row] = data[row * num_columns + col];
                }
            }

            auto data_tensor = Tensor::from_blob(flat_data.data(), {static_cast<size_t>(total_points), 1},
                                                 Device::CPU, DataType::Float32)
                                   .cuda();
            auto [centroids_tensor, labels_tensor] = lfs::io::kmeans(data_tensor, 256, iterations);

            auto centroids_cpu = centroids_tensor.cpu();
            auto labels_cpu = labels_tensor.cpu();
            const auto* centroids_ptr = static_cast<const float*>(centroids_cpu.data_ptr());
            const auto* labels_ptr = static_cast<const int32_t*>(labels_cpu.data_ptr());

            std::vector<int> order(256);
            for (int i = 0; i < 256; ++i)
                order[i] = i;
            std::sort(order.begin(), order.end(), [&](int a, int b) {
                return centroids_ptr[a] < centroids_ptr[b];
            });

            std::vector<float> ordered_centroids(256);
            for (int i = 0; i < 256; ++i) {
                ordered_centroids[i] = centroids_ptr[order[i]];
            }

            std::vector<int> inv_order(256);
            for (int i = 0; i < 256; ++i) {
                inv_order[order[i]] = i;
            }

            Cluster1dResult result;
            result.centroids = ordered_centroids;
            result.labels.resize(total_points);
            for (int i = 0; i < total_points; ++i) {
                result.labels[i] = static_cast<uint8_t>(inv_order[labels_ptr[i]]);
            }

            return result;
        }

    } // anonymous namespace

    Result<void> save_sog(const SplatData& splat_data, const SogSaveOptions& options) {
        LOG_INFO("SOG write: {}", lfs::core::path_to_utf8(options.output_path));

        const auto report_progress = [&](float progress, const std::string& stage) -> bool {
            return !options.progress_callback || options.progress_callback(progress, stage);
        };

        if (!report_progress(0.0f, "Initializing")) {
            return make_error(ErrorCode::CANCELLED, "Export cancelled by user");
        }

        const int64_t num_rows = splat_data.size();
        if (num_rows == 0) {
            return make_error(ErrorCode::EMPTY_DATASET, "No splats to write", options.output_path);
        }

        // Estimate output size: 5 base textures + optional SH, ~40% compression
        const int width = static_cast<int>(std::ceil(std::sqrt(num_rows) / 4.0)) * 4;
        const int height = static_cast<int>(std::ceil(static_cast<double>(num_rows) / width / 4.0)) * 4;
        constexpr int CHANNELS = 4;
        constexpr double COMPRESSION_RATIO = 0.4;
        constexpr size_t OVERHEAD = 4096;

        const size_t texture_size = static_cast<size_t>(width) * height * CHANNELS;
        const int sh_degree = splat_data.get_max_sh_degree();

        size_t estimated_size = texture_size * 5;
        if (sh_degree > 0) {
            estimated_size += texture_size * 2;
        }
        estimated_size = static_cast<size_t>(estimated_size * COMPRESSION_RATIO) + OVERHEAD;

        if (auto result = check_disk_space(options.output_path, estimated_size); !result) {
            return std::unexpected(result.error());
        }

        if (auto result = verify_writable(options.output_path); !result) {
            return std::unexpected(result.error());
        }

        auto means_cuda = splat_data.means_raw().cuda();
        auto morton_codes = morton_encode(means_cuda);
        auto sort_indices_tensor = morton_sort_indices(morton_codes);
        auto sort_indices_cpu = sort_indices_tensor.cpu();
        const auto* indices = sort_indices_cpu.ptr<int64_t>();

        auto means_cpu = means_cuda.cpu();
        SogArchive archive(options.output_path);

        // Check archive was created successfully
        if (!archive.is_valid()) {
            return make_error(ErrorCode::ARCHIVE_CREATION_FAILED, archive.last_error(), options.output_path);
        }

        const auto write_webp = [&](const std::string& filename,
                                    const uint8_t* data, int w, int h) -> Result<void> {
            return archive.add_webp(filename, data, w, h);
        };

        if (!report_progress(0.10f, "Positions")) {
            return make_error(ErrorCode::CANCELLED, "Export cancelled by user");
        }

        const auto* means_ptr = means_cpu.ptr<float>();

        std::array<std::array<double, 2>, 3> means_min_max = {{{std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity()},
                                                               {std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity()},
                                                               {std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity()}}};

        for (int64_t i = 0; i < num_rows; ++i) {
            const int64_t idx = indices[i];
            for (int d = 0; d < 3; ++d) {
                const double v = log_transform(static_cast<double>(means_ptr[idx * 3 + d]));
                means_min_max[d][0] = std::min(means_min_max[d][0], v);
                means_min_max[d][1] = std::max(means_min_max[d][1], v);
            }
        }

        std::vector<uint8_t> means_l(width * height * CHANNELS, 0);
        std::vector<uint8_t> means_u(width * height * CHANNELS, 0);

        for (int64_t i = 0; i < num_rows; ++i) {
            const int64_t idx = indices[i];
            const double x = 65535.0 * (log_transform(static_cast<double>(means_ptr[idx * 3 + 0])) - means_min_max[0][0]) /
                             (means_min_max[0][1] - means_min_max[0][0]);
            const double y = 65535.0 * (log_transform(static_cast<double>(means_ptr[idx * 3 + 1])) - means_min_max[1][0]) /
                             (means_min_max[1][1] - means_min_max[1][0]);
            const double z = 65535.0 * (log_transform(static_cast<double>(means_ptr[idx * 3 + 2])) - means_min_max[2][0]) /
                             (means_min_max[2][1] - means_min_max[2][0]);

            const auto x16 = static_cast<uint16_t>(std::clamp(x, 0.0, 65535.0));
            const auto y16 = static_cast<uint16_t>(std::clamp(y, 0.0, 65535.0));
            const auto z16 = static_cast<uint16_t>(std::clamp(z, 0.0, 65535.0));

            const auto ti = static_cast<int>(i);
            means_l[ti * 4 + 0] = x16 & 0xff;
            means_l[ti * 4 + 1] = y16 & 0xff;
            means_l[ti * 4 + 2] = z16 & 0xff;
            means_l[ti * 4 + 3] = 0xff;

            means_u[ti * 4 + 0] = (x16 >> 8) & 0xff;
            means_u[ti * 4 + 1] = (y16 >> 8) & 0xff;
            means_u[ti * 4 + 2] = (z16 >> 8) & 0xff;
            means_u[ti * 4 + 3] = 0xff;
        }

        if (auto result = write_webp("means_l.webp", means_l.data(), width, height); !result) {
            return std::unexpected(result.error());
        }
        if (auto result = write_webp("means_u.webp", means_u.data(), width, height); !result) {
            return std::unexpected(result.error());
        }

        if (!report_progress(0.20f, "Rotations")) {
            return make_error(ErrorCode::CANCELLED, "Export cancelled by user");
        }

        auto rotations = splat_data.rotation_raw().cpu();
        const auto* rot_ptr = rotations.ptr<float>();

        std::vector<uint8_t> quats(width * height * CHANNELS, 0);

        for (int64_t i = 0; i < num_rows; ++i) {
            const int64_t idx = indices[i];
            float q[4] = {rot_ptr[idx * 4 + 0], rot_ptr[idx * 4 + 1], rot_ptr[idx * 4 + 2], rot_ptr[idx * 4 + 3]};

            const float len = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
            for (float& j : q)
                j /= len;

            int max_comp = 0;
            for (int j = 1; j < 4; ++j) {
                if (std::abs(q[j]) > std::abs(q[max_comp]))
                    max_comp = j;
            }

            if (q[max_comp] < 0) {
                for (float& j : q)
                    j *= -1;
            }

            constexpr float SQRT2 = 1.41421356237f;
            for (float& j : q)
                j *= SQRT2;

            static const int IDX_TABLE[4][3] = {{1, 2, 3}, {0, 2, 3}, {0, 1, 3}, {0, 1, 2}};
            const int* other_idx = IDX_TABLE[max_comp];

            const auto ti = static_cast<int>(i);
            quats[ti * 4 + 0] = static_cast<uint8_t>(255.0f * (q[other_idx[0]] * 0.5f + 0.5f));
            quats[ti * 4 + 1] = static_cast<uint8_t>(255.0f * (q[other_idx[1]] * 0.5f + 0.5f));
            quats[ti * 4 + 2] = static_cast<uint8_t>(255.0f * (q[other_idx[2]] * 0.5f + 0.5f));
            quats[ti * 4 + 3] = static_cast<uint8_t>(252 + max_comp);
        }

        if (auto result = write_webp("quats.webp", quats.data(), width, height); !result) {
            return std::unexpected(result.error());
        }

        if (!report_progress(0.30f, "Scales k-means")) {
            return make_error(ErrorCode::CANCELLED, "Export cancelled by user");
        }

        auto scales = splat_data.scaling_raw().cpu();
        const auto* scales_ptr = scales.ptr<float>();

        auto scale_result = cluster1d(scales_ptr, static_cast<int>(num_rows), 3, options.kmeans_iterations);

        std::vector<uint8_t> scales_data(width * height * CHANNELS, 0);
        for (int64_t i = 0; i < num_rows; ++i) {
            const int64_t idx = indices[i];
            const auto ti = static_cast<int>(i);

            scales_data[ti * 4 + 0] = scale_result.labels[0 * num_rows + idx];
            scales_data[ti * 4 + 1] = scale_result.labels[1 * num_rows + idx];
            scales_data[ti * 4 + 2] = scale_result.labels[2 * num_rows + idx];
            scales_data[ti * 4 + 3] = 0xff;
        }

        if (auto result = write_webp("scales.webp", scales_data.data(), width, height); !result) {
            return std::unexpected(result.error());
        }

        if (!report_progress(0.45f, "Colors k-means")) {
            return make_error(ErrorCode::CANCELLED, "Export cancelled by user");
        }

        auto sh0 = splat_data.sh0_raw().cpu();
        const auto* sh0_ptr = sh0.ptr<float>();

        auto color_result = cluster1d(sh0_ptr, static_cast<int>(num_rows), 3, options.kmeans_iterations);

        auto opacity = splat_data.opacity_raw().cpu();
        const auto* opacity_ptr = opacity.ptr<float>();

        std::vector<uint8_t> sh0_data(width * height * CHANNELS, 0);
        for (int64_t i = 0; i < num_rows; ++i) {
            const int64_t idx = indices[i];
            const auto ti = static_cast<int>(i);

            sh0_data[ti * 4 + 0] = color_result.labels[0 * num_rows + idx];
            sh0_data[ti * 4 + 1] = color_result.labels[1 * num_rows + idx];
            sh0_data[ti * 4 + 2] = color_result.labels[2 * num_rows + idx];
            sh0_data[ti * 4 + 3] = static_cast<uint8_t>(
                std::max(0.0, std::min(255.0, sigmoid(static_cast<double>(opacity_ptr[idx])) * 255.0)));
        }

        if (auto result = write_webp("sh0.webp", sh0_data.data(), width, height); !result) {
            return std::unexpected(result.error());
        }

        nlohmann::json sh_n_meta;

        if (sh_degree > 0) {
            if (!report_progress(0.60f, "SH k-means")) {
                return make_error(ErrorCode::CANCELLED, "Export cancelled by user");
            }

            auto shN = splat_data.shN_raw().cpu();
            const auto* shN_ptr = shN.ptr<float>();

            static const int SH_COEFFS_TABLE[] = {0, 3, 8, 15};
            const int sh_coeffs = SH_COEFFS_TABLE[sh_degree];
            const int sh_dims = sh_coeffs * 3;

            int palette_size = std::min(64, static_cast<int>(std::pow(2, std::floor(std::log2(num_rows / 1024.0))))) * 1024;
            palette_size = std::clamp(palette_size, 1024, static_cast<int>(num_rows));

            std::vector<float> shN_flat(num_rows * sh_dims);
            for (int64_t i = 0; i < num_rows; ++i) {
                for (int c = 0; c < 3; ++c) {
                    for (int j = 0; j < sh_coeffs; ++j) {
                        const int their_col = c * sh_coeffs + j;
                        shN_flat[i * sh_dims + their_col] = shN_ptr[i * sh_coeffs * 3 + j * 3 + c];
                    }
                }
            }

            auto shN_tensor = Tensor::from_blob(shN_flat.data(),
                                                {static_cast<size_t>(num_rows), static_cast<size_t>(sh_dims)}, Device::CPU, DataType::Float32)
                                  .cuda();
            auto [sh_centroids, sh_labels] = lfs::io::kmeans(shN_tensor, palette_size, options.kmeans_iterations);

            auto sh_centroids_cpu = sh_centroids.cpu();
            const auto* sh_centroids_ptr = static_cast<const float*>(sh_centroids_cpu.data_ptr());
            const int actual_palette_size = static_cast<int>(sh_centroids.size(0));

            auto codebook_result = cluster1d(sh_centroids_ptr, actual_palette_size, sh_dims, options.kmeans_iterations);

            const int centroids_width = 64 * sh_coeffs;
            const int centroids_height = (actual_palette_size + 63) / 64;

            std::vector<uint8_t> centroids_buf(centroids_width * centroids_height * CHANNELS, 0);

            for (int i = 0; i < actual_palette_size; ++i) {
                for (int j = 0; j < sh_coeffs; ++j) {
                    const int pixel_idx = i * sh_coeffs + j;
                    for (int c = 0; c < 3; ++c) {
                        const int col_idx = sh_coeffs * c + j;
                        const int label_idx = col_idx * actual_palette_size + i;
                        centroids_buf[pixel_idx * 4 + c] = codebook_result.labels[label_idx];
                    }
                    centroids_buf[pixel_idx * 4 + 3] = 0xff;
                }
            }

            if (auto result = write_webp("shN_centroids.webp", centroids_buf.data(), centroids_width, centroids_height); !result) {
                return std::unexpected(result.error());
            }

            auto sh_labels_cpu = sh_labels.cpu();
            const auto* sh_labels_ptr = static_cast<const int32_t*>(sh_labels_cpu.data_ptr());

            std::vector<uint8_t> labels_buf(width * height * CHANNELS, 0);
            for (int64_t i = 0; i < num_rows; ++i) {
                const int64_t idx = indices[i];
                const int32_t label = sh_labels_ptr[idx];
                const auto ti = static_cast<int>(i);

                labels_buf[ti * 4 + 0] = label & 0xff;
                labels_buf[ti * 4 + 1] = (label >> 8) & 0xff;
                labels_buf[ti * 4 + 2] = 0;
                labels_buf[ti * 4 + 3] = 0xff;
            }

            if (auto result = write_webp("shN_labels.webp", labels_buf.data(), width, height); !result) {
                return std::unexpected(result.error());
            }

            sh_n_meta["count"] = actual_palette_size;
            sh_n_meta["bands"] = sh_degree;
            sh_n_meta["codebook"] = codebook_result.centroids;
            sh_n_meta["files"] = {"shN_centroids.webp", "shN_labels.webp"};
        }

        if (!report_progress(0.90f, "Writing meta")) {
            return make_error(ErrorCode::CANCELLED, "Export cancelled by user");
        }

        nlohmann::json meta;
        meta["version"] = 2;
        meta["asset"]["generator"] = "LichtFeld Studio";
        meta["count"] = num_rows;

        meta["means"]["mins"] = {means_min_max[0][0], means_min_max[1][0], means_min_max[2][0]};
        meta["means"]["maxs"] = {means_min_max[0][1], means_min_max[1][1], means_min_max[2][1]};
        meta["means"]["files"] = {"means_l.webp", "means_u.webp"};

        meta["scales"]["codebook"] = scale_result.centroids;
        meta["scales"]["files"] = {"scales.webp"};

        meta["quats"]["files"] = {"quats.webp"};

        meta["sh0"]["codebook"] = color_result.centroids;
        meta["sh0"]["files"] = {"sh0.webp"};

        if (sh_degree > 0) {
            meta["shN"] = sh_n_meta;
        }

        std::string meta_json = meta.dump();
        if (auto result = archive.add_file("meta.json", meta_json.c_str(), meta_json.size()); !result) {
            return std::unexpected(result.error());
        }

        LOG_INFO("SOG export complete: {} splats", num_rows);
        report_progress(1.0f, "Complete");
        return {};
    }

} // namespace lfs::io
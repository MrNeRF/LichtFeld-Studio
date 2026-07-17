/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#include "sogs.hpp"
#include "core/error_reporter.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "core/tensor.hpp"
#include "cuda/kmeans.hpp"
#include "cuda/morton_encoding.hpp"
#include "io/atomic_output.hpp"
#include "io/error.hpp"
#include <algorithm>
#include <archive.h>
#include <archive_entry.h>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <webp/decode.h>
#include <webp/encode.h>

namespace lfs::io {

    // Import types from lfs::core for convenience
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

        // Bound allocations derived from untrusted SOG metadata.
        constexpr size_t MAX_SOG_SPLATS = 100'000'000;
        constexpr size_t MAX_METADATA_BYTES = 16ULL * 1024 * 1024;
        constexpr size_t MAX_ENCODED_IMAGE_BYTES = 512ULL * 1024 * 1024;
        constexpr size_t MAX_ARCHIVE_BYTES = 4ULL * 1024 * 1024 * 1024;
        constexpr size_t MAX_DECODED_IMAGE_BYTES = 2ULL * 1024 * 1024 * 1024;
        constexpr size_t MAX_TOTAL_DECODED_BYTES = 8ULL * 1024 * 1024 * 1024;
        constexpr size_t MAX_RECONSTRUCTION_BYTES = 8ULL * 1024 * 1024 * 1024;
        constexpr size_t MAX_ARCHIVE_ENTRIES = 128;
        constexpr size_t MAX_CODEBOOK_SIZE = 256;

        struct DecodedImage {
            std::vector<uint8_t> rgba;
            int width = 0;
            int height = 0;
        };

        using DecodedImages = std::unordered_map<std::string, DecodedImage>;

        std::expected<size_t, std::string> checked_product(
            const size_t lhs,
            const size_t rhs,
            const std::string_view description) {
            if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
                return std::unexpected(std::format("SOG {} size overflows", description));
            }
            return lhs * rhs;
        }

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

        std::expected<DecodedImage, std::string> decode_webp(
            const uint8_t* data, const size_t size) {

            if (!data || size == 0) {
                return std::unexpected("Invalid WebP data");
            }
            if (size > MAX_ENCODED_IMAGE_BYTES) {
                return std::unexpected(std::format(
                    "Encoded WebP exceeds the {} byte SOG limit", MAX_ENCODED_IMAGE_BYTES));
            }

            // Get image info
            int width = 0;
            int height = 0;
            if (!WebPGetInfo(data, size, &width, &height)) {
                return std::unexpected("Failed to get WebP info");
            }
            if (width <= 0 || height <= 0) {
                return std::unexpected("WebP has invalid dimensions");
            }

            const auto pixel_count = checked_product(
                static_cast<size_t>(width), static_cast<size_t>(height), "image pixel");
            if (!pixel_count) {
                return std::unexpected(pixel_count.error());
            }
            const auto decoded_bytes = checked_product(*pixel_count, 4, "decoded image");
            if (!decoded_bytes) {
                return std::unexpected(decoded_bytes.error());
            }
            if (*decoded_bytes > MAX_DECODED_IMAGE_BYTES) {
                return std::unexpected(std::format(
                    "Decoded WebP exceeds the {} byte SOG limit", MAX_DECODED_IMAGE_BYTES));
            }

            DecodedImage image{
                .rgba = std::vector<uint8_t>(*decoded_bytes),
                .width = width,
                .height = height};
            if (!WebPDecodeRGBAInto(data,
                                    size,
                                    image.rgba.data(),
                                    image.rgba.size(),
                                    width * 4)) {
                return std::unexpected("Failed to decode WebP image");
            }

            return image;
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

        std::expected<int, std::string> resolve_sh_degree(
            const SogMetadata::SHData& sh_data) {
            int degree = sh_data.bands;
            if (degree == 0) {
                degree = sh_data.coeffs == 3 ? 1 : sh_data.coeffs == 8 ? 2
                                               : sh_data.coeffs == 15  ? 3
                                                                       : 0;
            }
            if (degree < 1 || degree > 3) {
                return std::unexpected(std::format(
                    "Unsupported SOG SH degree {} (supported range is 1..3)", degree));
            }
            if (sh_data.coeffs != 0 && sh_data.coeffs != SH_COEFFS[degree]) {
                return std::unexpected(std::format(
                    "SOG SH coeff count {} does not match degree {}",
                    sh_data.coeffs,
                    degree));
            }
            return degree;
        }

        std::expected<void, std::string> validate_files(
            const std::vector<std::string>& actual,
            const std::initializer_list<std::string_view> expected,
            const std::string_view field) {
            if (actual.size() != expected.size()) {
                return std::unexpected(std::format(
                    "SOG {} must list exactly {} texture(s)", field, expected.size()));
            }
            size_t index = 0;
            for (const auto name : expected) {
                if (actual[index] != name) {
                    return std::unexpected(std::format(
                        "SOG {} texture {} must be '{}'", field, index, name));
                }
                ++index;
            }
            return {};
        }

        std::expected<void, std::string> validate_codebook(
            const std::vector<float>& codebook,
            const std::string_view field) {
            if (codebook.empty() || codebook.size() > MAX_CODEBOOK_SIZE) {
                return std::unexpected(std::format(
                    "SOG {} codebook must contain 1..{} entries",
                    field,
                    MAX_CODEBOOK_SIZE));
            }
            if (!std::ranges::all_of(codebook, [](const float value) {
                    return std::isfinite(value);
                })) {
                return std::unexpected(std::format(
                    "SOG {} codebook contains a non-finite value", field));
            }
            return {};
        }

        std::expected<void, std::string> validate_metadata(SogMetadata& meta) {
            if (meta.count <= 0 || static_cast<size_t>(meta.count) > MAX_SOG_SPLATS) {
                return std::unexpected(std::format(
                    "SOG splat count must be in the range 1..{}", MAX_SOG_SPLATS));
            }

            if ((meta.width == 0) != (meta.height == 0) ||
                meta.width < 0 || meta.height < 0) {
                return std::unexpected(
                    "SOG width and height must both be omitted or both be positive");
            }
            if (meta.width == 0) {
                const double count = static_cast<double>(meta.count);
                meta.width = static_cast<int>(std::ceil(std::sqrt(count) / 4.0)) * 4;
                meta.height = static_cast<int>(
                                  std::ceil(count / static_cast<double>(meta.width) / 4.0)) *
                              4;
            }

            const auto pixel_count = checked_product(
                static_cast<size_t>(meta.width),
                static_cast<size_t>(meta.height),
                "texture pixel");
            if (!pixel_count || *pixel_count < static_cast<size_t>(meta.count)) {
                return std::unexpected(pixel_count
                                           ? "SOG texture dimensions cannot hold the declared splat count"
                                           : pixel_count.error());
            }
            const auto texture_bytes = checked_product(*pixel_count, 4, "texture");
            if (!texture_bytes) {
                return std::unexpected(texture_bytes.error());
            }
            if (*texture_bytes > MAX_DECODED_IMAGE_BYTES) {
                return std::unexpected("SOG texture dimensions exceed the decoded image limit");
            }

            if (meta.means_mins.size() != 3 || meta.means_maxs.size() != 3) {
                return std::unexpected("SOG means mins and maxs must each contain three values");
            }
            const float max_finite_log_coordinate =
                std::log(std::numeric_limits<float>::max());
            for (size_t axis = 0; axis < 3; ++axis) {
                const float min_value = meta.means_mins[axis];
                const float max_value = meta.means_maxs[axis];
                if (!std::isfinite(min_value) || !std::isfinite(max_value) ||
                    min_value > max_value ||
                    std::abs(min_value) > max_finite_log_coordinate ||
                    std::abs(max_value) > max_finite_log_coordinate) {
                    return std::unexpected(std::format(
                        "SOG means bounds for axis {} are invalid", axis));
                }
            }

            if (auto result = validate_files(
                    meta.means_files, {"means_l.webp", "means_u.webp"}, "means");
                !result) {
                return result;
            }
            if (auto result = validate_files(
                    meta.scales_files, {"scales.webp"}, "scales");
                !result) {
                return result;
            }
            if (auto result = validate_files(
                    meta.quats_files, {"quats.webp"}, "quats");
                !result) {
                return result;
            }
            if (auto result = validate_files(meta.sh0_files, {"sh0.webp"}, "sh0");
                !result) {
                return result;
            }
            if (auto result = validate_codebook(meta.scales_codebook, "scales"); !result) {
                return result;
            }
            if (auto result = validate_codebook(meta.sh0_codebook, "sh0"); !result) {
                return result;
            }

            if (meta.shN) {
                auto& sh_data = *meta.shN;
                const auto degree = resolve_sh_degree(sh_data);
                if (!degree) {
                    return std::unexpected(degree.error());
                }
                sh_data.bands = *degree;
                sh_data.coeffs = SH_COEFFS[*degree];

                if (auto result = validate_files(
                        sh_data.files,
                        {"shN_centroids.webp", "shN_labels.webp"},
                        "shN");
                    !result) {
                    return result;
                }
                if (auto result = validate_codebook(sh_data.codebook, "shN"); !result) {
                    return result;
                }
                if (sh_data.palette_size <= 0 || sh_data.palette_size > 65'536 ||
                    sh_data.palette_size > meta.count) {
                    return std::unexpected(
                        "SOG SH palette size must be positive, fit the 16-bit label, and not exceed the splat count");
                }
            }

            const size_t floats_per_splat =
                14 + (meta.shN ? static_cast<size_t>(meta.shN->coeffs) * 3 : 0);
            const auto reconstruction_floats = checked_product(
                static_cast<size_t>(meta.count),
                floats_per_splat,
                "reconstruction buffer");
            if (!reconstruction_floats) {
                return std::unexpected(reconstruction_floats.error());
            }
            const auto reconstruction_bytes = checked_product(
                *reconstruction_floats,
                sizeof(float),
                "reconstruction buffer");
            if (!reconstruction_bytes ||
                *reconstruction_bytes > MAX_RECONSTRUCTION_BYTES) {
                return std::unexpected(std::format(
                    "SOG reconstruction exceeds the {} byte host-buffer limit",
                    MAX_RECONSTRUCTION_BYTES));
            }

            return {};
        }

        std::expected<const DecodedImage*, std::string> require_image(
            const DecodedImages& images,
            const std::string_view name,
            const int expected_width,
            const int expected_height) {
            const auto it = images.find(std::string(name));
            if (it == images.end()) {
                return std::unexpected(std::format("Missing SOG texture '{}'", name));
            }
            const auto& image = it->second;
            if (image.width != expected_width || image.height != expected_height) {
                return std::unexpected(std::format(
                    "SOG texture '{}' is {}x{}, expected {}x{}",
                    name,
                    image.width,
                    image.height,
                    expected_width,
                    expected_height));
            }
            const auto pixel_count = checked_product(
                static_cast<size_t>(expected_width),
                static_cast<size_t>(expected_height),
                "decoded texture pixel");
            if (!pixel_count) {
                return std::unexpected(pixel_count.error());
            }
            const auto byte_count = checked_product(*pixel_count, 4, "decoded texture");
            if (!byte_count || image.rgba.size() != *byte_count) {
                return std::unexpected(std::format(
                    "SOG texture '{}' has an invalid decoded byte count", name));
            }
            return &image;
        }

        std::expected<void, std::string> validate_decoded_payload(
            const SogMetadata& meta,
            const DecodedImages& images) {
            for (const std::string_view name : {
                     "means_l.webp",
                     "means_u.webp",
                     "quats.webp",
                     "scales.webp",
                     "sh0.webp"}) {
                if (auto result = require_image(images, name, meta.width, meta.height); !result) {
                    return std::unexpected(result.error());
                }
            }

            const auto& scales = images.at("scales.webp").rgba;
            const auto& sh0 = images.at("sh0.webp").rgba;
            for (size_t i = 0; i < static_cast<size_t>(meta.count); ++i) {
                const size_t offset = i * 4;
                for (size_t channel = 0; channel < 3; ++channel) {
                    if (scales[offset + channel] >= meta.scales_codebook.size()) {
                        return std::unexpected("SOG scale texture contains an invalid codebook index");
                    }
                    if (sh0[offset + channel] >= meta.sh0_codebook.size()) {
                        return std::unexpected("SOG sh0 texture contains an invalid codebook index");
                    }
                }
            }

            if (meta.shN) {
                const auto& sh_data = *meta.shN;
                const int coefficients = SH_COEFFS[sh_data.bands];
                const int centroid_width = 64 * coefficients;
                const int centroid_height = (sh_data.palette_size + 63) / 64;
                const auto centroids = require_image(
                    images, "shN_centroids.webp", centroid_width, centroid_height);
                if (!centroids) {
                    return std::unexpected(centroids.error());
                }
                const auto labels = require_image(
                    images, "shN_labels.webp", meta.width, meta.height);
                if (!labels) {
                    return std::unexpected(labels.error());
                }

                const auto& centroid_bytes = (*centroids)->rgba;
                for (size_t pixel = 0;
                     pixel < static_cast<size_t>(sh_data.palette_size) * coefficients;
                     ++pixel) {
                    for (size_t channel = 0; channel < 3; ++channel) {
                        if (centroid_bytes[pixel * 4 + channel] >= sh_data.codebook.size()) {
                            return std::unexpected(
                                "SOG SH centroid texture contains an invalid codebook index");
                        }
                    }
                }

                const auto& label_bytes = (*labels)->rgba;
                for (size_t i = 0; i < static_cast<size_t>(meta.count); ++i) {
                    const size_t offset = i * 4;
                    const uint16_t label = static_cast<uint16_t>(label_bytes[offset]) |
                                           (static_cast<uint16_t>(label_bytes[offset + 1]) << 8);
                    if (label >= sh_data.palette_size) {
                        return std::unexpected("SOG SH label is outside the declared palette");
                    }
                }
            }

            return {};
        }

        std::expected<SplatData, std::string> reconstruct_splat_data(
            const SogMetadata& meta,
            const DecodedImages& images) {

            if (auto result = validate_decoded_payload(meta, images); !result) {
                return std::unexpected(result.error());
            }

            const int num_splats = meta.count;
            const int width = meta.width;
            const int height = meta.height;

            LOG_DEBUG("Reconstructing {} splats from {}x{} textures", num_splats, width, height);

            // Create host buffers
            const size_t splat_count = static_cast<size_t>(num_splats);
            std::vector<float> host_means(splat_count * 3);
            std::vector<float> host_scales(splat_count * 3);
            std::vector<float> host_rotations(splat_count * 4);
            std::vector<float> host_opacity(splat_count);

            // Determine SH dimensions
            int sh0_dim1 = 1, sh0_dim2 = 3;
            int shN_dim1 = 0, shN_dim2 = 3;

            if (meta.shN.has_value()) {
                const auto& sh_meta = meta.shN.value();
                shN_dim1 = SH_COEFFS[sh_meta.bands];
            }

            std::vector<float> host_sh0(splat_count * sh0_dim1 * sh0_dim2);
            std::vector<float> host_shN(splat_count * shN_dim1 * shN_dim2);

            // 1. Decode positions from means_l and means_u
            {
                auto it_l = images.find("means_l.webp");
                auto it_u = images.find("means_u.webp");

                if (it_l == images.end() || it_u == images.end()) {
                    return std::unexpected("Missing position textures");
                }

                const auto& means_l = it_l->second.rgba;
                const auto& means_u = it_u->second.rgba;

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

                const auto& quats = it->second.rgba;

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

                const auto& scales_img = it->second.rgba;

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

                const auto& sh0_img = it->second.rgba;

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
                    const auto& centroids_img = it_centroids->second.rgba;
                    const auto& labels_img = it_labels->second.rgba;

                    // Determine SH configuration
                    const int num_coeffs = SH_COEFFS[sh_meta.bands];
                    const int palette_size = sh_meta.palette_size;

                    LOG_DEBUG("Decoding SH: degree={}, coeffs={}, palette_size={}",
                              sh_meta.bands, num_coeffs, palette_size);

                    // Decode centroids from texture
                    std::vector<float> centroids(
                        static_cast<size_t>(palette_size) * num_coeffs * 3);
                    for (int i = 0; i < palette_size; ++i) {
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
                                centroids[(static_cast<size_t>(i) * num_coeffs * 3) + coeff_idx] =
                                    sh_meta.codebook[idx];
                            }
                        }
                    }

                    // Apply labels
                    for (int i = 0; i < num_splats; ++i) {
                        int ti = identity_layout(i, width) * 4;

                        // Reconstruct label from 16-bit value
                        int label = labels_img[ti + 0] | (labels_img[ti + 1] << 8);

                        if (label < palette_size) {
                            // Unpack in band-major order
                            for (int c = 0; c < 3; ++c) {
                                for (int j = 0; j < num_coeffs; ++j) {
                                    host_shN[i * shN_dim1 * shN_dim2 + j * shN_dim2 + c] =
                                        centroids[(static_cast<size_t>(label) * num_coeffs * 3) +
                                                  j + c * num_coeffs];
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

            std::error_code file_error;
            const uintmax_t archive_size = std::filesystem::file_size(path, file_error);
            if (file_error) {
                return std::unexpected(std::format(
                    "Failed to inspect SOG archive: {}", file_error.message()));
            }
            if (archive_size > MAX_ARCHIVE_BYTES) {
                return std::unexpected(std::format(
                    "SOG archive exceeds the {} byte limit", MAX_ARCHIVE_BYTES));
            }

            struct ArchiveReadDeleter {
                void operator()(struct archive* value) const {
                    if (value) {
                        archive_read_free(value);
                    }
                }
            };
            std::unique_ptr<struct archive, ArchiveReadDeleter> archive_reader(
                archive_read_new());
            if (!archive_reader) {
                return std::unexpected("Failed to allocate SOG archive reader");
            }
            struct archive* const a = archive_reader.get();
            if (archive_read_support_format_zip(a) != ARCHIVE_OK ||
                archive_read_support_filter_all(a) != ARCHIVE_OK) {
                const char* detail = archive_error_string(a);
                return std::unexpected(std::format(
                    "Failed to configure SOG archive reader: {}",
                    detail ? detail : "unknown error"));
            }

            // Use wide-character API on Windows for proper Unicode path handling
            int result;
#ifdef _WIN32
            result = archive_read_open_filename_w(a, path.wstring().c_str(), 10240);
#else
            result = archive_read_open_filename(a, path.c_str(), 10240);
#endif
            if (result != ARCHIVE_OK) {
                const char* detail = archive_error_string(a);
                return std::unexpected(std::format("Failed to open archive: {}",
                                                   detail ? detail : "unknown error"));
            }

            struct archive_entry* entry;
            std::string metadata_json;
            DecodedImages images;
            std::unordered_set<std::string> seen_entries;
            size_t entry_count = 0;
            size_t total_entry_bytes = 0;
            size_t total_decoded_bytes = 0;

            const auto is_sog_entry = [](const std::string_view filename) {
                return filename == "meta.json" || filename == "means_l.webp" ||
                       filename == "means_u.webp" || filename == "scales.webp" ||
                       filename == "quats.webp" || filename == "sh0.webp" ||
                       filename == "shN_centroids.webp" || filename == "shN_labels.webp";
            };

            // Read all files from archive
            int header_result = ARCHIVE_OK;
            while ((header_result = archive_read_next_header(a, &entry)) == ARCHIVE_OK) {
                if (++entry_count > MAX_ARCHIVE_ENTRIES) {
                    return std::unexpected(std::format(
                        "SOG archive contains more than {} entries", MAX_ARCHIVE_ENTRIES));
                }
                const char* const pathname = archive_entry_pathname(entry);
                if (!pathname) {
                    return std::unexpected("SOG archive entry has no valid pathname");
                }
                const std::string filename(pathname);

                if (!archive_entry_size_is_set(entry)) {
                    return std::unexpected(std::format(
                        "SOG archive entry '{}' has no declared size", filename));
                }
                const la_int64_t signed_size = archive_entry_size(entry);
                if (signed_size < 0) {
                    return std::unexpected(std::format(
                        "SOG archive entry '{}' has a negative size", filename));
                }
                const size_t size = static_cast<size_t>(signed_size);
                if (size > MAX_ENCODED_IMAGE_BYTES) {
                    return std::unexpected(std::format(
                        "SOG archive entry '{}' exceeds the {} byte limit",
                        filename,
                        MAX_ENCODED_IMAGE_BYTES));
                }
                if (total_entry_bytes > MAX_ARCHIVE_BYTES - size) {
                    return std::unexpected(std::format(
                        "SOG archive entries exceed the {} byte total limit",
                        MAX_ARCHIVE_BYTES));
                }
                total_entry_bytes += size;

                if (!is_sog_entry(filename)) {
                    if (archive_read_data_skip(a) != ARCHIVE_OK) {
                        const char* detail = archive_error_string(a);
                        return std::unexpected(std::format(
                            "Failed to skip SOG archive entry '{}': {}",
                            filename,
                            detail ? detail : "unknown error"));
                    }
                    continue;
                }
                if (!seen_entries.emplace(filename).second) {
                    return std::unexpected(std::format(
                        "SOG archive contains duplicate entry '{}'", filename));
                }
                if (filename == "meta.json" && size > MAX_METADATA_BYTES) {
                    return std::unexpected(std::format(
                        "SOG metadata exceeds the {} byte limit", MAX_METADATA_BYTES));
                }

                LOG_DEBUG("Reading {} ({} bytes)", filename, size);

                std::vector<uint8_t> data(size);
                size_t offset = 0;
                while (offset < size) {
                    const ssize_t bytes_read = archive_read_data(
                        a, data.data() + offset, size - offset);
                    if (bytes_read <= 0) {
                        const char* detail = archive_error_string(a);
                        return std::unexpected(std::format(
                            "Failed to read '{}' from SOG archive: {}",
                            filename,
                            detail ? detail : "truncated entry"));
                    }
                    offset += static_cast<size_t>(bytes_read);
                }

                if (filename == "meta.json") {
                    metadata_json = std::string(data.begin(), data.end());
                } else {
                    auto decoded = decode_webp(data.data(), data.size());
                    if (!decoded) {
                        return std::unexpected(std::format(
                            "Failed to decode '{}': {}", filename, decoded.error()));
                    }
                    if (total_decoded_bytes >
                        MAX_TOTAL_DECODED_BYTES - decoded->rgba.size()) {
                        return std::unexpected(std::format(
                            "Decoded SOG textures exceed the {} byte total limit",
                            MAX_TOTAL_DECODED_BYTES));
                    }
                    total_decoded_bytes += decoded->rgba.size();
                    images.emplace(filename, std::move(*decoded));
                }
            }
            if (header_result != ARCHIVE_EOF) {
                const char* detail = archive_error_string(a);
                return std::unexpected(std::format(
                    "Failed while reading SOG archive headers: {}",
                    detail ? detail : "unknown error"));
            }

            if (metadata_json.empty()) {
                return std::unexpected("Missing meta.json in archive");
            }

            // Parse metadata
            auto meta_result = parse_metadata(metadata_json);
            if (!meta_result) {
                return std::unexpected(meta_result.error());
            }
            if (auto validation = validate_metadata(*meta_result); !validation) {
                return std::unexpected(validation.error());
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

            std::error_code file_error;
            const uintmax_t metadata_size = std::filesystem::file_size(meta_path, file_error);
            if (file_error) {
                return std::unexpected(std::format(
                    "Failed to inspect meta.json: {}", file_error.message()));
            }
            if (metadata_size == 0 || metadata_size > MAX_METADATA_BYTES) {
                return std::unexpected(std::format(
                    "SOG metadata must contain 1..{} bytes", MAX_METADATA_BYTES));
            }

            std::ifstream meta_file;
            if (!lfs::core::open_file_for_read(meta_path, std::ios::binary, meta_file)) {
                return std::unexpected("Failed to open meta.json");
            }
            std::string metadata_json(static_cast<size_t>(metadata_size), '\0');
            if (!meta_file.read(metadata_json.data(),
                                static_cast<std::streamsize>(metadata_json.size()))) {
                return std::unexpected("Failed to read complete meta.json");
            }

            auto meta_result = parse_metadata(metadata_json);
            if (!meta_result) {
                return std::unexpected(meta_result.error());
            }
            if (auto validation = validate_metadata(*meta_result); !validation) {
                return std::unexpected(validation.error());
            }

            auto& meta = meta_result.value();
            DecodedImages images;
            size_t total_decoded_bytes = 0;

            // Helper to read and decode WebP files
            auto read_webp = [&](const std::string& filename)
                -> std::expected<void, std::string> {
                if (images.contains(filename)) {
                    return std::unexpected(std::format(
                        "SOG metadata references duplicate texture '{}'", filename));
                }
                auto file_path = path / filename;

                if (!std::filesystem::exists(file_path)) {
                    return std::unexpected(std::format(
                        "Missing SOG texture '{}'", lfs::core::path_to_utf8(file_path)));
                }

                std::error_code image_error;
                const uintmax_t image_size = std::filesystem::file_size(file_path, image_error);
                if (image_error) {
                    return std::unexpected(std::format(
                        "Failed to inspect '{}': {}", filename, image_error.message()));
                }
                if (image_size == 0 || image_size > MAX_ENCODED_IMAGE_BYTES) {
                    return std::unexpected(std::format(
                        "Encoded SOG texture '{}' must contain 1..{} bytes",
                        filename,
                        MAX_ENCODED_IMAGE_BYTES));
                }

                std::ifstream file;
                if (!lfs::core::open_file_for_read(file_path, std::ios::binary, file)) {
                    return std::unexpected(std::format(
                        "Failed to open SOG texture '{}'", filename));
                }

                std::vector<uint8_t> data(static_cast<size_t>(image_size));
                if (!file.read(reinterpret_cast<char*>(data.data()),
                               static_cast<std::streamsize>(data.size()))) {
                    return std::unexpected(std::format(
                        "Failed to read complete SOG texture '{}'", filename));
                }

                // Decode WebP
                auto decoded = decode_webp(data.data(), data.size());
                if (!decoded) {
                    return std::unexpected(std::format(
                        "Failed to decode '{}': {}", filename, decoded.error()));
                }
                if (total_decoded_bytes >
                    MAX_TOTAL_DECODED_BYTES - decoded->rgba.size()) {
                    return std::unexpected(std::format(
                        "Decoded SOG textures exceed the {} byte total limit",
                        MAX_TOTAL_DECODED_BYTES));
                }
                total_decoded_bytes += decoded->rgba.size();
                images.emplace(filename, std::move(*decoded));
                return {};
            };

            // Read all required files
            for (const auto& file : meta.means_files) {
                if (auto result = read_webp(file); !result)
                    return std::unexpected(result.error());
            }
            for (const auto& file : meta.scales_files) {
                if (auto result = read_webp(file); !result)
                    return std::unexpected(result.error());
            }
            for (const auto& file : meta.quats_files) {
                if (auto result = read_webp(file); !result)
                    return std::unexpected(result.error());
            }
            for (const auto& file : meta.sh0_files) {
                if (auto result = read_webp(file); !result)
                    return std::unexpected(result.error());
            }

            // Read optional SH files
            if (meta.shN.has_value()) {
                for (const auto& file : meta.shN->files) {
                    if (auto result = read_webp(file); !result)
                        return std::unexpected(result.error());
                }
            }

            // Reconstruct SplatData
            return reconstruct_splat_data(meta, images);
        }

    } // anonymous namespace

    std::expected<SplatData, std::string> load_sog(const std::filesystem::path& path) {
        LOG_TIMER("SOG File Loading");

        try {
            if (!std::filesystem::exists(path)) {
                std::string error_msg = std::format("SOG file/directory does not exist: {}", lfs::core::path_to_utf8(path));
                LOG_ERROR("{}", error_msg);
                return std::unexpected(error_msg);
            }

            // Check if it's a .sog bundle
            if (path.extension() == ".sog") {
                return read_sog_bundle(path);
            }
            // Check if it's a meta.json file
            if (path.filename() == "meta.json") {
                return read_sog_directory(path.parent_path());
            }
            // Check if it's a directory
            if (std::filesystem::is_directory(path)) {
                return read_sog_directory(path);
            }
            return std::unexpected(std::format("Unknown SOG format: {}", lfs::core::path_to_utf8(path)));
        } catch (const std::bad_alloc&) {
            return std::unexpected("SOG input exceeds available memory");
        } catch (const std::exception& error) {
            return std::unexpected(std::format("Failed to load SOG: {}", error.what()));
        }
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

        Tensor as_cuda_contiguous(const Tensor& tensor) {
            if (tensor.device() == Device::CUDA) {
                return tensor.is_contiguous() ? tensor : tensor.contiguous();
            }
            return tensor.cuda().contiguous();
        }

        int nearest_centroid_1d(const std::vector<float>& centroids, float value) {
            auto it = std::lower_bound(centroids.begin(), centroids.end(), value);
            int best = static_cast<int>(std::distance(centroids.begin(), it));
            if (best >= static_cast<int>(centroids.size())) {
                best = static_cast<int>(centroids.size()) - 1;
            }

            float best_dist = std::abs(value - centroids[best]);
            if (best > 0) {
                const float prev_dist = std::abs(value - centroids[best - 1]);
                if (prev_dist < best_dist) {
                    best = best - 1;
                    best_dist = prev_dist;
                }
            }
            if (best + 1 < static_cast<int>(centroids.size())) {
                const float next_dist = std::abs(value - centroids[best + 1]);
                if (next_dist < best_dist) {
                    best = best + 1;
                }
            }
            return best;
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

            [[nodiscard]] Result<void> close() {
                if (!a_ || !valid_) {
                    return {};
                }

                const int result = archive_write_close(a_);
                valid_ = false;

                if (result != ARCHIVE_OK) {
                    return make_error(ErrorCode::ARCHIVE_CREATION_FAILED,
                                      std::format("Failed to close SOG archive '{}': {}",
                                                  lfs::core::path_to_utf8(output_path_),
                                                  archive_error_string(a_) ? archive_error_string(a_) : "unknown error"),
                                      output_path_);
                }

                return {};
            }

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
            constexpr int K = 256;
            const size_t total_points = static_cast<size_t>(num_rows) * static_cast<size_t>(num_columns);

            float min_val = std::numeric_limits<float>::infinity();
            float max_val = -std::numeric_limits<float>::infinity();
            for (int col = 0; col < num_columns; ++col) {
                for (int row = 0; row < num_rows; ++row) {
                    const float value = data[row * num_columns + col];
                    min_val = std::min(min_val, value);
                    max_val = std::max(max_val, value);
                }
            }

            std::vector<float> centroid_vals(K);
            const float step = (K > 1) ? (max_val - min_val) / (K - 1) : 0.0f;
            for (int i = 0; i < K; ++i) {
                centroid_vals[i] = min_val + i * step;
            }

            Cluster1dResult result;
            result.labels.assign(total_points, 0);

            struct LocalAccum {
                std::array<double, K> sums{};
                std::array<int64_t, K> counts{};
            };

            const unsigned int hw_threads = std::max(1u, std::thread::hardware_concurrency());
            const size_t worker_count = std::max<size_t>(
                1, std::min<size_t>(hw_threads, (total_points + 65535) / 65536));

            auto accumulate_range = [&](size_t begin, size_t end, const bool write_labels, LocalAccum& accum) {
                for (size_t linear = begin; linear < end; ++linear) {
                    const int col = static_cast<int>(linear / static_cast<size_t>(num_rows));
                    const int row = static_cast<int>(linear - static_cast<size_t>(col) * static_cast<size_t>(num_rows));
                    const float value = data[row * num_columns + col];
                    const int label = nearest_centroid_1d(centroid_vals, value);

                    if (write_labels) {
                        result.labels[linear] = static_cast<uint8_t>(label);
                    }
                    accum.sums[label] += static_cast<double>(value);
                    accum.counts[label]++;
                }
            };

            const int effective_iterations = std::max(0, iterations);
            for (int iter = 0; iter < effective_iterations; ++iter) {
                std::vector<LocalAccum> accumulators(worker_count);
                const bool write_labels = (iter == effective_iterations - 1);

                if (worker_count == 1) {
                    accumulate_range(0, total_points, write_labels, accumulators[0]);
                } else {
                    std::vector<std::thread> workers;
                    workers.reserve(worker_count);
                    for (size_t worker = 0; worker < worker_count; ++worker) {
                        const size_t begin = total_points * worker / worker_count;
                        const size_t end = total_points * (worker + 1) / worker_count;
                        workers.emplace_back(accumulate_range, begin, end, write_labels, std::ref(accumulators[worker]));
                    }
                    for (auto& worker : workers) {
                        worker.join();
                    }
                }

                for (int c = 0; c < K; ++c) {
                    double sum = 0.0;
                    int64_t count = 0;
                    for (const auto& accum : accumulators) {
                        sum += accum.sums[c];
                        count += accum.counts[c];
                    }
                    if (count > 0) {
                        centroid_vals[c] = static_cast<float>(sum / static_cast<double>(count));
                    }
                }
            }

            std::vector<int> order(K);
            for (int i = 0; i < K; ++i)
                order[i] = i;
            std::sort(order.begin(), order.end(), [&](int a, int b) {
                return centroid_vals[a] < centroid_vals[b];
            });

            std::vector<float> ordered_centroids(K);
            for (int i = 0; i < K; ++i) {
                ordered_centroids[i] = centroid_vals[order[i]];
            }

            std::vector<int> inv_order(K);
            for (int i = 0; i < K; ++i) {
                inv_order[order[i]] = i;
            }

            result.centroids = ordered_centroids;
            for (uint8_t& label : result.labels) {
                label = static_cast<uint8_t>(inv_order[label]);
            }

            return result;
        }

    } // anonymous namespace

    Result<void> save_sog(const SplatData& splat_data, const SogSaveOptions& options) {
        try {
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

            if (num_rows > static_cast<int64_t>(std::numeric_limits<int>::max())) {
                return make_error(ErrorCode::INVALID_DATASET,
                                  "SOG export supports at most INT_MAX splats",
                                  options.output_path);
            }

            auto means_cuda = as_cuda_contiguous(splat_data.means_raw());
            auto sort_indices_tensor = morton_sort_indices_for_positions(means_cuda);
            if (!sort_indices_tensor.is_valid()) {
                return make_error(ErrorCode::ENCODING_FAILED,
                                  "Failed to compute Morton order for SOG export",
                                  options.output_path);
            }
            auto sort_indices_cpu = sort_indices_tensor.cpu();
            const auto* indices = sort_indices_cpu.ptr<int32_t>();

            auto means_cpu = means_cuda.cpu();
            const auto* means_ptr = means_cpu.ptr<float>();
            const auto source_index = [&](int64_t sorted_index) -> int64_t {
                return static_cast<int64_t>(indices[sorted_index]);
            };

            ScopedAtomicOutputFile atomic_output(options.output_path);
            SogArchive archive(atomic_output.temp_path());

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

            std::array<std::array<double, 2>, 3> means_min_max = {{{std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity()},
                                                                   {std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity()},
                                                                   {std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity()}}};

            for (int64_t i = 0; i < num_rows; ++i) {
                const int64_t idx = source_index(i);
                for (int d = 0; d < 3; ++d) {
                    const double v = log_transform(static_cast<double>(means_ptr[idx * 3 + d]));
                    means_min_max[d][0] = std::min(means_min_max[d][0], v);
                    means_min_max[d][1] = std::max(means_min_max[d][1], v);
                }
            }

            std::vector<uint8_t> means_l(width * height * CHANNELS, 0);
            std::vector<uint8_t> means_u(width * height * CHANNELS, 0);

            for (int64_t i = 0; i < num_rows; ++i) {
                const int64_t idx = source_index(i);
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
                const int64_t idx = source_index(i);
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
                const int64_t idx = source_index(i);
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
                const int64_t idx = source_index(i);
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

                static const int SH_COEFFS_TABLE[] = {0, 3, 8, 15};
                const int sh_coeffs = SH_COEFFS_TABLE[sh_degree];
                const int sh_dims = sh_coeffs * 3;

                int palette_size = std::min(64, static_cast<int>(std::pow(2, std::floor(std::log2(num_rows / 1024.0))))) * 1024;
                palette_size = std::clamp(palette_size, 1024, static_cast<int>(num_rows));

                const auto& shN_swizzled = splat_data.shN_raw();
                if (!shN_swizzled.is_valid() || shN_swizzled.ndim() != 1 || shN_swizzled.numel() == 0) {
                    return make_error(ErrorCode::INVALID_DATASET,
                                      "Invalid swizzled SH tensor for SOG export",
                                      options.output_path);
                }

                // Run k-means directly on resident swizzled shN so SOG export does not allocate
                // a full canonical [N, K, 3] CUDA tensor.
                auto [sh_centroids, sh_labels] = lfs::io::kmeans_sh_swizzled(
                    shN_swizzled, static_cast<int>(num_rows), sh_coeffs,
                    palette_size, options.kmeans_iterations);
                if (!sh_centroids.is_valid() || !sh_labels.is_valid()) {
                    return make_error(ErrorCode::ENCODING_FAILED,
                                      "Failed to cluster swizzled SH tensor for SOG export",
                                      options.output_path);
                }

                auto sh_centroids_cpu = sh_centroids.cpu();
                const auto* sh_centroids_ptr = static_cast<const float*>(sh_centroids_cpu.data_ptr());
                const int actual_palette_size = static_cast<int>(sh_centroids.size(0));

                // Keep the full SH tensor in source layout for k-means, then regroup only
                // the much smaller centroid table into the SOG texture/codebook layout.
                std::vector<float> sh_centroids_grouped(actual_palette_size * sh_dims);
                for (int i = 0; i < actual_palette_size; ++i) {
                    for (int c = 0; c < 3; ++c) {
                        for (int j = 0; j < sh_coeffs; ++j) {
                            sh_centroids_grouped[i * sh_dims + c * sh_coeffs + j] =
                                sh_centroids_ptr[i * sh_dims + j * 3 + c];
                        }
                    }
                }

                auto codebook_result = cluster1d(sh_centroids_grouped.data(), actual_palette_size, sh_dims, options.kmeans_iterations);

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
                    const int64_t idx = source_index(i);
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

            if (auto result = archive.close(); !result) {
                return std::unexpected(result.error());
            }

            if (!report_progress(1.0f, "Complete")) {
                return make_error(ErrorCode::CANCELLED, "Export cancelled by user");
            }

            if (auto result = atomic_output.commit(); !result) {
                return std::unexpected(result.error());
            }

            LOG_INFO("SOG export complete: {} splats", num_rows);
            return {};
        } catch (const lfs::Exception& e) {
            lfs::Error error = lfs::Error(e.error())
                                   .with_context("save SOG", LFS_SOURCE_SITE_CURRENT(),
                                                 lfs::SmallFields{}.add("path", lfs::core::path_to_utf8(options.output_path)));
            lfs::core::ErrorReporter::get().report(error, lfs::core::ReportChannel::OwnerLog);
            return std::unexpected(from_lfs_error(error));
        } catch (const std::exception& e) {
            return make_error(ErrorCode::ENCODING_FAILED,
                              std::format("Failed to save SOG: {}", e.what()),
                              options.output_path);
        }
    }

} // namespace lfs::io

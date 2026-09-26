/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "spz.hpp"
#include "coordinate-system-adobe.h"
#include "core/error.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "core/provenance.hpp"
#include "core/splat_data_transform.hpp"
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
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <limits>
#include <memory>
#include <nlohmann/json.hpp>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <thread>
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

        template <typename Function>
        void parallel_for_chunks(size_t count, Function&& function) {
            if (count < 4096) {
                function(0, count);
                return;
            }

            const unsigned hardware_threads = std::thread::hardware_concurrency();
            const size_t thread_count = std::min<size_t>(
                count, std::min(16u, hardware_threads == 0 ? 1u : hardware_threads));
            if (thread_count == 1) {
                function(0, count);
                return;
            }

            const size_t chunk_size = (count + thread_count - 1) / thread_count;
            std::vector<std::thread> workers;
            workers.reserve(thread_count);
            for (size_t begin = 0; begin < count; begin += chunk_size) {
                const size_t end = std::min(begin + chunk_size, count);
                workers.emplace_back([begin, end, &function]() {
                    function(begin, end);
                });
            }
            for (auto& worker : workers) {
                worker.join();
            }
        }

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

        template <typename Cloud>
        void log_coordinate_system_extension(const Cloud& cloud) {
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

        std::expected<void, std::string> validate_spz_output(
            const spz::PackedGaussians& packed,
            const spz::GaussianCloudOutput& output) {
            if (packed.numPoints <= 0 ||
                static_cast<uint32_t>(packed.numPoints) > spz::kMaxSpzPoints) {
                return std::unexpected("SPZ header contains invalid point or SH metadata");
            }
            if (packed.shDegree < 0 || packed.shDegree > 3) {
                return std::unexpected(
                    "SPZ SH degree-4 files are not supported by LichtFeld yet "
                    "(supported degrees: 0-3)");
            }

            const size_t count = static_cast<size_t>(packed.numPoints);
            const size_t sh_coefficients = packed.shDegree > 0
                                               ? static_cast<size_t>(SH_COEFFS_FOR_DEGREE[packed.shDegree])
                                               : 0;
            if (output.positions.size() != count * 3 ||
                output.scales.size() != count * 3 ||
                output.rotations.size() != count * 4 ||
                output.alphas.size() != count ||
                output.colors.size() != count * 3 ||
                output.sh.size() != count * sh_coefficients * 3) {
                return std::unexpected("SPZ decoded attribute sizes do not match the point count");
            }

            const auto finite = [](const std::span<const float> values) {
                return std::ranges::all_of(values, [](const float value) {
                    return std::isfinite(value);
                });
            };
            if (!finite(output.positions) || !finite(output.scales) ||
                !finite(output.rotations) || !finite(output.alphas) ||
                !finite(output.colors) || !finite(output.sh)) {
                return std::unexpected("SPZ decoded attributes contain a non-finite value");
            }
            return {};
        }

        Tensor gather_live_rows(const Tensor& src, const Tensor& keep, const bool filter) {
            Tensor selected = filter ? src.index_select(0, keep) : src;
            return selected.contiguous().to_pageable_host();
        }

        spz::GaussianCloud convert_to_spz(const SplatData& splat) {
            const bool filter = splat.has_deleted_mask();
            const Tensor keep = filter ? splat.deleted().logical_not() : Tensor{};
            const int sh_degree = splat.get_max_sh_degree();
            const int sh_coeffs = sh_degree > 0 ? SH_COEFFS_FOR_DEGREE[sh_degree] : 0;

            Tensor means;
            Tensor scaling;
            Tensor rotation;
            Tensor opacity;
            Tensor sh0;
            Tensor shN;
            {
                LOG_TIMER_DEBUG("SPZ export: host copies");
                means = gather_live_rows(splat.means(), keep, filter);
                scaling = gather_live_rows(splat.scaling_raw(), keep, filter);
                rotation = gather_live_rows(splat.rotation_raw(), keep, filter);
                opacity = gather_live_rows(splat.opacity_raw(), keep, filter);
                sh0 = gather_live_rows(splat.sh0(), keep, filter);
            }
            if (sh_coeffs > 0 && splat.shN().is_valid() && splat.shN().numel() > 0) {
                LOG_TIMER_DEBUG("SPZ export: sh unpack");
                Tensor decoded = splat.shN_canonical_cpu();
                if (filter && decoded.is_valid() && decoded.numel() > 0) {
                    Tensor keep_for_sh = keep;
                    if (keep_for_sh.device() != decoded.device())
                        keep_for_sh = keep_for_sh.to(decoded.device());
                    decoded = decoded.index_select(0, keep_for_sh);
                }
                shN = decoded.contiguous().to_pageable_host();
            }

            const auto num_points = static_cast<int>(means.size(0));

            spz::GaussianCloud cloud;
            cloud.numPoints = num_points;
            cloud.shDegree = sh_degree;
            cloud.antialiased = false;

            LOG_TIMER_DEBUG("SPZ export: pack");
            cloud.positions.resize(num_points * 3);
            cloud.scales.resize(num_points * 3);
            cloud.rotations.resize(num_points * 4);
            cloud.alphas.resize(num_points);
            cloud.colors.resize(num_points * 3);
            if (sh_coeffs > 0 && shN.is_valid() && shN.numel() > 0) {
                cloud.sh.resize(num_points * sh_coeffs * 3);
            }

            const auto* const means_ptr = static_cast<const float*>(means.data_ptr());
            const auto* const scaling_ptr = static_cast<const float*>(scaling.data_ptr());
            const auto* const rotation_ptr = static_cast<const float*>(rotation.data_ptr());
            const auto* const opacity_ptr = static_cast<const float*>(opacity.data_ptr());
            const auto* const sh0_ptr = static_cast<const float*>(sh0.data_ptr());
            const bool has_sh = sh_coeffs > 0 && shN.is_valid() && shN.numel() > 0;
            const size_t sh_values_per_point = static_cast<size_t>(sh_coeffs) * 3;
            const auto* const shN_ptr = has_sh ? shN.ptr<float>() : nullptr;

            // Keep all row-wise conversion/copy work deterministic while allowing the large
            // host-side buffers (especially SH) to use the available cores.
            parallel_for_chunks(static_cast<size_t>(num_points), [&](size_t begin, size_t end) {
                for (size_t i = begin; i < end; ++i) {
                    std::memcpy(cloud.positions.data() + i * 3,
                                means_ptr + i * 3,
                                3 * sizeof(float));
                    std::memcpy(cloud.scales.data() + i * 3,
                                scaling_ptr + i * 3,
                                3 * sizeof(float));
                    std::memcpy(cloud.colors.data() + i * 3,
                                sh0_ptr + i * 3,
                                3 * sizeof(float));

                    // Rotation: SplatData wxyz -> SPZ xyzw
                    cloud.rotations[i * 4 + 0] = rotation_ptr[i * 4 + 1]; // x
                    cloud.rotations[i * 4 + 1] = rotation_ptr[i * 4 + 2]; // y
                    cloud.rotations[i * 4 + 2] = rotation_ptr[i * 4 + 3]; // z
                    cloud.rotations[i * 4 + 3] = rotation_ptr[i * 4 + 0]; // w

                    cloud.alphas[i] = opacity_ptr[i];
                    if (has_sh) {
                        std::memcpy(cloud.sh.data() + i * sh_values_per_point,
                                    shN_ptr + i * sh_values_per_point,
                                    sh_values_per_point * sizeof(float));
                    }
                }
            });

            return cloud;
        }

        // glTF binary container carrying an SPZ v3 payload via KHR_gaussian_splatting_compression_spz_2
        // (the layout written by the geo-register plugin's 3D Tiles exporter). Implements the Khronos
        // glTF 2.0 and KHR_gaussian_splatting specifications; see THIRD_PARTY_LICENSES.md.
        constexpr uint32_t GLB_MAGIC = 0x46546C67; // "glTF"
        constexpr uint32_t GLB_CHUNK_JSON = 0x4E4F534A;
        constexpr uint32_t GLB_CHUNK_BIN = 0x004E4942;
        constexpr const char* GLB_SPZ_EXTENSION = "KHR_gaussian_splatting_compression_spz_2";
        constexpr const char* GLB_SPZ_POINTER =
            "/extensions/KHR_gaussian_splatting/extensions/KHR_gaussian_splatting_compression_spz_2";

        // 3D Tiles turns glTF Y-up content into Z-up; the node matrix pre-applies the inverse, so the
        // payload stays in the splat's own frame (stored as-is, no RUB conversion).
        const glm::dmat4 GLB_Y_UP_TO_Z_UP(1, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1);

        struct GlbSpz {
            std::vector<uint8_t> spz;
            glm::mat4 transform{1.0f};
            bool linear_color = false;
        };

        // lin_rec709_display -> srgb_rec709_display: SH0 converts exactly, higher bands scale by
        // the sRGB curve slope at the base color (first-order approximation).
        void convert_linear_sh_to_srgb(float* sh0, float* shN, const size_t count, const size_t sh_coeffs) {
            constexpr float SH_C0 = 0.28209479177387814f;
            const auto encode = [](const float c) {
                return c <= 0.0031308f ? 12.92f * c : 1.055f * std::pow(c, 1.0f / 2.4f) - 0.055f;
            };
            const auto slope = [](const float c) {
                return c <= 0.0031308f ? 12.92f : 1.055f / 2.4f * std::pow(c, 1.0f / 2.4f - 1.0f);
            };
            parallel_for_chunks(count, [&](const size_t begin, const size_t end) {
                for (size_t i = begin; i < end; ++i) {
                    for (size_t channel = 0; channel < 3; ++channel) {
                        float& dc = sh0[i * 3 + channel];
                        const float linear = std::clamp(dc * SH_C0 + 0.5f, 0.0f, 1.0f);
                        dc = (encode(linear) - 0.5f) / SH_C0;
                        const float gain = slope(linear);
                        for (size_t k = 0; k < sh_coeffs; ++k) {
                            shN[(i * sh_coeffs + k) * 3 + channel] *= gain;
                        }
                    }
                }
            });
        }

        lfs::Error glb_error(std::string message) {
            return lfs::make_error(lfs::ErrorInit{
                .code = lfs::ErrorCode::DataLoss,
                .domain = lfs::ErrorDomain::IO,
                .detail = std::move(message),
                .detection = LFS_SOURCE_SITE_CURRENT(),
            });
        }

        lfs::Result<glm::dmat4> glb_node_matrix(const nlohmann::json& node) {
            if (node.contains("matrix")) {
                const auto m = node["matrix"].get<std::vector<double>>();
                if (m.size() != 16) {
                    return glb_error("GLB node matrix must have 16 values");
                }
                return glm::dmat4(glm::make_mat4(m.data()));
            }
            const auto t = node.value("translation", std::vector<double>{0, 0, 0});
            const auto r = node.value("rotation", std::vector<double>{0, 0, 0, 1});
            const auto s = node.value("scale", std::vector<double>{1, 1, 1});
            if (t.size() != 3 || r.size() != 4 || s.size() != 3) {
                return glb_error("GLB node translation, rotation and scale must have 3, 4 and 3 values");
            }
            glm::dmat4 matrix = glm::mat4_cast(glm::dquat(r[3], r[0], r[1], r[2]));
            matrix[0] *= s[0];
            matrix[1] *= s[1];
            matrix[2] *= s[2];
            matrix[3] = glm::dvec4(t[0], t[1], t[2], 1.0);
            return matrix;
        }

        lfs::Result<GlbSpz> read_glb_spz(const std::vector<uint8_t>& glb) {
            uint32_t header[3] = {};
            if (glb.size() < sizeof(header)) {
                return glb_error("truncated GLB header");
            }
            std::memcpy(header, glb.data(), sizeof(header));
            if (header[0] != GLB_MAGIC || header[1] != 2) {
                return glb_error("not a glTF 2.0 binary");
            }

            nlohmann::json doc;
            std::span<const uint8_t> bin;
            for (size_t offset = sizeof(header); offset + 8 <= glb.size();) {
                uint32_t chunk[2] = {};
                std::memcpy(chunk, glb.data() + offset, sizeof(chunk));
                offset += sizeof(chunk);
                if (chunk[0] > glb.size() - offset) {
                    return glb_error("truncated GLB chunk");
                }
                if (chunk[1] == GLB_CHUNK_JSON) {
                    doc = nlohmann::json::parse(glb.begin() + offset, glb.begin() + offset + chunk[0]);
                } else if (chunk[1] == GLB_CHUNK_BIN && bin.empty()) {
                    bin = {glb.data() + offset, chunk[0]};
                }
                offset += chunk[0];
            }

            const nlohmann::json::json_pointer spz_pointer(GLB_SPZ_POINTER);
            const auto meshes = doc.value("meshes", nlohmann::json::array());
            for (size_t mesh = 0; mesh < meshes.size(); ++mesh) {
                for (const auto& primitive : meshes[mesh].value("primitives", nlohmann::json::array())) {
                    if (!primitive.contains(spz_pointer)) {
                        continue;
                    }
                    const auto& view = doc.at("bufferViews").at(primitive.at(spz_pointer).at("bufferView").get<size_t>());
                    const auto view_offset = view.value("byteOffset", size_t{0});
                    const auto view_length = view.at("byteLength").get<size_t>();
                    if (view.value("buffer", 0) != 0 || view_offset > bin.size() ||
                        view_length > bin.size() - view_offset) {
                        return glb_error("SPZ buffer view is outside the embedded GLB buffer");
                    }

                    GlbSpz result;
                    result.spz.assign(bin.begin() + view_offset, bin.begin() + view_offset + view_length);
                    result.linear_color = primitive.at(nlohmann::json::json_pointer("/extensions/KHR_gaussian_splatting"))
                                              .value("colorSpace", "") == "lin_rec709_display";
                    for (const auto& node : doc.value("nodes", nlohmann::json::array())) {
                        if (node.value("mesh", -1) == static_cast<int>(mesh)) {
                            auto node_matrix = glb_node_matrix(node);
                            if (!node_matrix) {
                                return std::move(node_matrix).error();
                            }
                            result.transform = glm::mat4(GLB_Y_UP_TO_Z_UP * *node_matrix);
                            break;
                        }
                    }
                    return result;
                }
            }
            return glb_error(std::format("no primitive uses {}", GLB_SPZ_EXTENSION));
        }

        // Recenters positions and picks fractional bits so the extent fits the 24-bit fixed point.
        // Returns the removed center; half_extent receives the recentered bounds.
        glm::dvec3 recenter_for_glb(spz::GaussianCloud& cloud, spz::PackOptions& pack_options,
                                    glm::dvec3& half_extent) {
            glm::dvec3 lo(std::numeric_limits<double>::max());
            glm::dvec3 hi(std::numeric_limits<double>::lowest());
            for (size_t i = 0; i < cloud.positions.size(); i += 3) {
                const glm::dvec3 p(cloud.positions[i], cloud.positions[i + 1], cloud.positions[i + 2]);
                lo = glm::min(lo, p);
                hi = glm::max(hi, p);
            }
            const glm::dvec3 center = (lo + hi) * 0.5;
            half_extent = (hi - lo) * 0.5;
            for (size_t i = 0; i < cloud.positions.size(); ++i) {
                cloud.positions[i] = static_cast<float>(cloud.positions[i] - center[static_cast<int>(i % 3)]);
            }
            // Keep 2^bits * extent below 2^22 (one bit of headroom in the signed 24-bit range).
            const double max_abs = std::max({half_extent.x, half_extent.y, half_extent.z});
            pack_options.fractionalBits = static_cast<uint8_t>(
                max_abs > 0.0 ? std::clamp(std::floor(std::log2(double(1 << 22) / max_abs)), 0.0, 12.0) : 12.0);
            return center;
        }

        std::vector<uint8_t> wrap_spz_in_glb(const std::vector<uint8_t>& spz_data, const spz::GaussianCloud& cloud,
                                             const glm::dvec3& center, const glm::dvec3& half_extent) {
            using nlohmann::json;
            json accessors = json::array();
            json attributes = json::object();
            const auto add_accessor = [&](const std::string& name, const char* type, const int component_type) {
                attributes[name] = accessors.size();
                accessors.push_back({{"componentType", component_type}, {"count", cloud.numPoints}, {"type", type}});
                return &accessors.back();
            };
            auto* position = add_accessor("POSITION", "VEC3", 5126);
            (*position)["min"] = {-half_extent.x, -half_extent.y, -half_extent.z};
            (*position)["max"] = {half_extent.x, half_extent.y, half_extent.z};
            (*add_accessor("COLOR_0", "VEC4", 5121))["normalized"] = true;
            add_accessor("KHR_gaussian_splatting:SCALE", "VEC3", 5126);
            add_accessor("KHR_gaussian_splatting:ROTATION", "VEC4", 5126);
            add_accessor("KHR_gaussian_splatting:OPACITY", "SCALAR", 5126);
            add_accessor("KHR_gaussian_splatting:SH_DEGREE_0_COEF_0", "VEC3", 5126);
            for (int degree = 1; degree <= cloud.shDegree; ++degree) {
                for (int coef = 0; coef < 2 * degree + 1; ++coef) {
                    add_accessor(std::format("KHR_gaussian_splatting:SH_DEGREE_{}_COEF_{}", degree, coef), "VEC3", 5126);
                }
            }

            const glm::dmat4 node = glm::inverse(GLB_Y_UP_TO_Z_UP) * glm::translate(glm::dmat4(1.0), center);
            const size_t bin_length = (spz_data.size() + 3) & ~size_t{3};
            const json spz_extension = {{"kernel", "ellipse"},
                                        {"colorSpace", "srgb_rec709_display"},
                                        {"extensions", {{GLB_SPZ_EXTENSION, {{"bufferView", 0}}}}}};
            const json primitive = {{"mode", 0},
                                    {"attributes", attributes},
                                    {"extensions", {{"KHR_gaussian_splatting", spz_extension}}}};
            const json doc = {
                {"asset", {{"version", "2.0"}, {"generator", "LichtFeld Studio"}}},
                {"extensionsUsed", json::array({"KHR_gaussian_splatting", GLB_SPZ_EXTENSION})},
                {"extensionsRequired", json::array({"KHR_gaussian_splatting", GLB_SPZ_EXTENSION})},
                {"scene", 0},
                {"scenes", json::array({{{"nodes", {0}}}})},
                {"nodes", json::array({{{"mesh", 0},
                                        {"matrix", std::vector<double>(glm::value_ptr(node),
                                                                       glm::value_ptr(node) + 16)}}})},
                {"meshes", json::array({{{"primitives", json::array({primitive})}}})},
                {"buffers", json::array({{{"byteLength", bin_length}}})},
                {"bufferViews", json::array({{{"buffer", 0}, {"byteLength", spz_data.size()}}})},
                {"accessors", accessors}};

            std::string json_text = doc.dump();
            json_text.resize((json_text.size() + 3) & ~size_t{3}, ' ');

            std::vector<uint8_t> glb;
            glb.reserve(28 + json_text.size() + bin_length);
            const auto append_u32 = [&glb](const size_t value) {
                const auto v = static_cast<uint32_t>(value);
                glb.insert(glb.end(), reinterpret_cast<const uint8_t*>(&v), reinterpret_cast<const uint8_t*>(&v) + 4);
            };
            append_u32(GLB_MAGIC);
            append_u32(2);
            append_u32(28 + json_text.size() + bin_length);
            append_u32(json_text.size());
            append_u32(GLB_CHUNK_JSON);
            glb.insert(glb.end(), json_text.begin(), json_text.end());
            append_u32(bin_length);
            append_u32(GLB_CHUNK_BIN);
            glb.insert(glb.end(), spz_data.begin(), spz_data.end());
            glb.resize(glb.size() + bin_length - spz_data.size(), 0);
            return glb;
        }
    } // namespace

    bool is_spz_glb(const std::filesystem::path& filepath) {
        std::ifstream in;
        if (!lfs::core::open_file_for_read(filepath, std::ios::binary, in)) {
            return false;
        }
        uint32_t header[5] = {};
        if (!in.read(reinterpret_cast<char*>(header), sizeof(header)) ||
            header[0] != GLB_MAGIC || header[4] != GLB_CHUNK_JSON || header[3] > (64u << 20)) {
            return false;
        }
        std::string json_text(header[3], '\0');
        return in.read(json_text.data(), static_cast<std::streamsize>(json_text.size())) &&
               json_text.find(GLB_SPZ_EXTENSION) != std::string::npos;
    }

    std::expected<SplatData, std::string> load_spz(const std::filesystem::path& filepath) {
        auto start = std::chrono::high_resolution_clock::now();

        LOG_INFO("Loading SPZ file: {}", lfs::core::path_to_utf8(filepath));

        std::string_view format_name = "SPZ"; // "GLB" once a glTF container is detected
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

            std::vector<uint8_t> data;
            {
                LOG_TIMER_DEBUG("SPZ load: read");
                data.resize(static_cast<size_t>(size));
                in.seekg(0, std::ios::beg);
                if (!in.read(reinterpret_cast<char*>(data.data()),
                             static_cast<std::streamsize>(data.size()))) {
                    return std::unexpected(std::format("Failed to read SPZ file: {}", lfs::core::path_to_utf8(filepath)));
                }
            }

            std::optional<glm::mat4> glb_transform;
            bool glb_linear_color = false;
            if (data.size() >= 4 && std::memcmp(data.data(), "glTF", 4) == 0) {
                format_name = "GLB";
                auto glb = read_glb_spz(data);
                if (!glb) {
                    return std::unexpected(std::format(
                        "Failed to load {} file '{}': {}", format_name, lfs::core::path_to_utf8(filepath), glb.error().detail()));
                }
                data = std::move(glb->spz);
                glb_transform = glb->transform;
                glb_linear_color = glb->linear_color;
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

            // Decode the packed streams first so the LichtFeld-owned pageable tensors can be
            // allocated with their final shapes and used as the unpack destination.
            spz::UnpackOptions options;
            options.to = glb_transform ? spz::CoordinateSystem::RUB : spz::CoordinateSystem::RDF;
            spz::PackedGaussians packed;
            {
                LOG_TIMER_DEBUG("SPZ load: decompress");
                LOG_TIMER_DEBUG("SPZ load: packed decode");
                packed = spz::loadSpzPacked(data);
            }
            if (packed.numPoints <= 0 ||
                static_cast<uint32_t>(packed.numPoints) > spz::kMaxSpzPoints) {
                return std::unexpected(std::format(
                    "Failed to load {} file '{}': header contains invalid point metadata",
                    format_name, lfs::core::path_to_utf8(filepath)));
            }
            if (packed.shDegree < 0 || packed.shDegree > 3) {
                return std::unexpected(std::format(
                    "Failed to load {} file '{}': {}",
                    format_name, lfs::core::path_to_utf8(filepath),
                    "SPZ SH degree-4 files are not supported by LichtFeld yet "
                    "(supported degrees: 0-3)"));
            }

            log_coordinate_system_extension(packed);

            const auto num_points = static_cast<size_t>(packed.numPoints);
            const auto sh_degree = packed.shDegree;
            const auto sh_coeffs = sh_degree > 0
                                       ? static_cast<size_t>(SH_COEFFS_FOR_DEGREE[sh_degree])
                                       : 0;

            Tensor means;
            Tensor sh0;
            Tensor scaling;
            Tensor rotation;
            Tensor opacity;
            Tensor shN;
            {
                LOG_TIMER_DEBUG("SPZ load: host tensors");
                means = Tensor::empty_pageable_host({num_points, 3}, DataType::Float32);
                sh0 = Tensor::empty_pageable_host({num_points, 1, 3}, DataType::Float32);
                scaling = Tensor::empty_pageable_host({num_points, 3}, DataType::Float32);
                rotation = Tensor::empty_pageable_host({num_points, 4}, DataType::Float32);
                opacity = Tensor::empty_pageable_host({num_points, 1}, DataType::Float32);
                if (sh_coeffs > 0) {
                    shN = Tensor::empty_pageable_host({num_points, sh_coeffs, 3}, DataType::Float32);
                }

                // The decoder fills every element. Touch the large pageable destination in
                // parallel first when the allocator left its pages cold, so decode workers do
                // not serialize on first-touch faults.
                LOG_TIMER_DEBUG("SPZ load: host tensors prefault");
                lfs::core::prefault_pageable_host_memory(means.data_ptr(), means.bytes());
                lfs::core::prefault_pageable_host_memory(sh0.data_ptr(), sh0.bytes());
                lfs::core::prefault_pageable_host_memory(scaling.data_ptr(), scaling.bytes());
                lfs::core::prefault_pageable_host_memory(rotation.data_ptr(), rotation.bytes());
                lfs::core::prefault_pageable_host_memory(opacity.data_ptr(), opacity.bytes());
                if (shN.is_valid()) {
                    lfs::core::prefault_pageable_host_memory(shN.data_ptr(), shN.bytes());
                }
            }

            spz::GaussianCloudOutput output{
                std::span<float>(static_cast<float*>(means.data_ptr()), means.numel()),
                std::span<float>(static_cast<float*>(scaling.data_ptr()), scaling.numel()),
                std::span<float>(static_cast<float*>(rotation.data_ptr()), rotation.numel()),
                std::span<float>(static_cast<float*>(opacity.data_ptr()), opacity.numel()),
                std::span<float>(static_cast<float*>(sh0.data_ptr()), sh0.numel()),
                shN.is_valid()
                    ? std::span<float>(static_cast<float*>(shN.data_ptr()), shN.numel())
                    : std::span<float>{}};

            {
                LOG_TIMER_DEBUG("SPZ load: unpackGaussians");
                if (!spz::unpackGaussians(packed, options, output)) {
                    return std::unexpected(std::format(
                        "Failed to load {} file '{}': failed to unpack Gaussian data",
                        format_name, lfs::core::path_to_utf8(filepath)));
                }
            }
            if (auto validation = validate_spz_output(packed, output); !validation) {
                return std::unexpected(std::format(
                    "Failed to load {} file '{}': {}",
                    format_name, lfs::core::path_to_utf8(filepath),
                    validation.error()));
            }

            LOG_DEBUG("SPZ loaded: {} points, SH degree {}", packed.numPoints, packed.shDegree);

            if (glb_linear_color) {
                LOG_WARN("GLB '{}' uses lin_rec709_display colors; converting to sRGB (SH bands approximated)",
                         lfs::core::path_to_utf8(filepath));
                convert_linear_sh_to_srgb(static_cast<float*>(sh0.data_ptr()),
                                          shN.is_valid() ? static_cast<float*>(shN.data_ptr()) : nullptr,
                                          num_points, sh_coeffs);
            }

            SplatData splat;
            {
                LOG_TIMER_DEBUG("SPZ load: upload");
                splat = SplatData(
                    sh_degree,
                    std::move(means),
                    std::move(sh0),
                    std::move(shN),
                    std::move(scaling),
                    std::move(rotation),
                    std::move(opacity),
                    SCENE_SCALE);
            }
            if (glb_transform) {
                lfs::core::transform(splat, *glb_transform);
            }

            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - start);
            LOG_INFO("SPZ loaded: {} gaussians with SH degree {} in {}ms",
                     splat.size(), splat.get_max_sh_degree(), elapsed.count());

            return splat;
        } catch (const std::bad_alloc&) {
            return std::unexpected("SPZ input exceeds available memory");
        } catch (const std::exception& error) {
            return std::unexpected(std::format(
                "Failed to load {} file '{}': {}", format_name, lfs::core::path_to_utf8(filepath), error.what()));
        }
    }

    Result<void> save_spz(const SplatData& splat_data, const SpzSaveOptions& options_in) {
        SpzSaveOptions options = options_in;
        if (options.glb) {
            options.version = 3; // KHR_gaussian_splatting_compression_spz_2 readers expect gzip SPZ
        }
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
        const auto export_count = splat_data.has_deleted_mask()
                                      ? static_cast<size_t>(splat_data.visible_count())
                                      : splat_data.size();
        if (export_count == 0 || export_count > spz::kMaxSpzPoints) {
            return make_error(
                ErrorCode::INVALID_DATASET,
                std::format("SPZ export supports 1..{} visible splats", spz::kMaxSpzPoints),
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
        pack_options.from = options.glb ? spz::CoordinateSystem::RUB : spz::CoordinateSystem::RDF;
        pack_options.version = static_cast<uint32_t>(options.version);
        pack_options.compressionLevel = options.compression_level;

        glm::dvec3 glb_center{0.0};
        glm::dvec3 glb_half_extent{0.0};
        if (options.glb) {
            glb_center = recenter_for_glb(cloud, pack_options, glb_half_extent);
        }

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
        {
            const unsigned hardware_threads = std::thread::hardware_concurrency();
            const unsigned worker_count = hardware_threads == 0 ? 1u : hardware_threads;
            LOG_TIMER_DEBUG(std::format(
                "SPZ export: compress (level {}, workers {})",
                options.compression_level,
                worker_count));
            if (!spz::saveSpz(cloud, pack_options, &data)) {
                return make_error(ErrorCode::WRITE_FAILURE,
                                  "Failed to pack SPZ data", options.output_path);
            }
            if (options.glb) {
                data = wrap_spz_in_glb(data, cloud, glb_center, glb_half_extent);
            }
        }

        if (!report_export_progress(options.progress_callback, 0.9f, "Writing SPZ")) {
            return make_error(ErrorCode::CANCELLED, "SPZ export cancelled", options.output_path);
        }

        {
            LOG_TIMER_DEBUG("SPZ export: write");
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

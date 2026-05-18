
/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#ifdef _WIN32
#define NOMINMAX
#endif

#include "formats/nurec_usdz.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "core/tensor.hpp"
#include "io/atomic_output.hpp"
#include "io/error.hpp"
#include "io/exporter.hpp"
#include <archive.h>
#include <archive_entry.h>
#include <nlohmann/json.hpp>
#include <pxr/base/gf/half.h>
#include <zlib.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <format>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace lfs::io {

    using lfs::core::DataType;
    using lfs::core::Device;
    using lfs::core::SplatData;
    using lfs::core::Tensor;

    namespace {

#ifdef _WIN32
        using ssize_t = std::ptrdiff_t;
#endif

        using Json = nlohmann::ordered_json;

        constexpr int kMaxSupportedShDegree = 3;
        constexpr std::string_view kRootKey = "nre_data";
        constexpr std::string_view kStateDictKey = "state_dict";
        constexpr std::string_view kPositionsKey = ".gaussians_nodes.gaussians.positions";
        constexpr std::string_view kRotationsKey = ".gaussians_nodes.gaussians.rotations";
        constexpr std::string_view kScalesKey = ".gaussians_nodes.gaussians.scales";
        constexpr std::string_view kDensitiesKey = ".gaussians_nodes.gaussians.densities";
        constexpr std::string_view kAlbedoKey = ".gaussians_nodes.gaussians.features_albedo";
        constexpr std::string_view kSpecularKey = ".gaussians_nodes.gaussians.features_specular";
        constexpr std::string_view kExtraSignalKey = ".gaussians_nodes.gaussians.extra_signal";
        constexpr std::string_view kActiveFeaturesKey = ".gaussians_nodes.gaussians.n_active_features";

        constexpr std::string_view kRenderSettingsBlock = R"(        dictionary renderSettings = {
            int "rtx:directLighting:sampledLighting:samplesPerPixel" = 8
            bool "rtx:material:enableRefraction" = 0
            bool "rtx:matteObject:visibility:secondaryRays" = 1
            bool "rtx:post:histogram:enabled" = 0
            bool "rtx:post:registeredCompositing:invertColorCorrection" = 1
            bool "rtx:post:registeredCompositing:invertToneMap" = 1
            int "rtx:post:tonemap:op" = 2
            bool "rtx:raytracing:fractionalCutoutOpacity" = 0
            string "rtx:rendermode" = "RaytracedLighting"
        }
)";

        struct UsdzEntries {
            std::unordered_map<std::string, std::vector<uint8_t>> files;
        };

        std::string normalized_extension(const std::filesystem::path& path) {
            auto extension = path.extension().string();
            std::transform(extension.begin(), extension.end(), extension.begin(), [](const unsigned char c) {
                return static_cast<char>(std::tolower(c));
            });
            return extension;
        }

        std::string format_float(const float value) {
            if (std::abs(value) < 1e-12f) {
                return "0";
            }
            return std::format("{:.9g}", value);
        }

        std::expected<std::vector<uint8_t>, std::string> gzip_compress(const std::vector<uint8_t>& input) {
            z_stream stream{};
            if (deflateInit2(&stream, 0, Z_DEFLATED, 16 + MAX_WBITS, 8, Z_DEFAULT_STRATEGY) != Z_OK) {
                return std::unexpected("Failed to initialize gzip compressor");
            }

            stream.next_in = const_cast<Bytef*>(reinterpret_cast<const Bytef*>(input.data()));
            stream.avail_in = static_cast<uInt>(input.size());

            std::vector<uint8_t> output;
            std::array<uint8_t, 16384> chunk{};

            int status = Z_OK;
            while (status != Z_STREAM_END) {
                stream.next_out = chunk.data();
                stream.avail_out = static_cast<uInt>(chunk.size());
                status = deflate(&stream, Z_FINISH);
                if (status != Z_OK && status != Z_STREAM_END) {
                    deflateEnd(&stream);
                    return std::unexpected(std::format("gzip compression failed with zlib status {}", status));
                }

                const size_t produced = chunk.size() - stream.avail_out;
                output.insert(output.end(), chunk.begin(), chunk.begin() + static_cast<std::ptrdiff_t>(produced));
            }

            deflateEnd(&stream);
            return output;
        }

        std::expected<std::vector<uint8_t>, std::string> gzip_decompress(const std::span<const uint8_t> input) {
            z_stream stream{};
            if (inflateInit2(&stream, 16 + MAX_WBITS) != Z_OK) {
                return std::unexpected("Failed to initialize gzip decompressor");
            }

            stream.next_in = const_cast<Bytef*>(reinterpret_cast<const Bytef*>(input.data()));
            stream.avail_in = static_cast<uInt>(input.size());

            std::vector<uint8_t> output;
            std::array<uint8_t, 16384> chunk{};

            int status = Z_OK;
            while (status != Z_STREAM_END) {
                stream.next_out = chunk.data();
                stream.avail_out = static_cast<uInt>(chunk.size());
                status = inflate(&stream, Z_NO_FLUSH);
                if (status != Z_OK && status != Z_STREAM_END) {
                    inflateEnd(&stream);
                    return std::unexpected(std::format("gzip decompression failed with zlib status {}", status));
                }

                const size_t produced = chunk.size() - stream.avail_out;
                output.insert(output.end(), chunk.begin(), chunk.begin() + static_cast<std::ptrdiff_t>(produced));
            }

            inflateEnd(&stream);
            return output;
        }

        Json::binary_t float_tensor_to_half_bytes(const Tensor& tensor) {
            const auto cpu = tensor.contiguous().to(Device::CPU).to(DataType::Float32);
            const auto* const values = static_cast<const float*>(cpu.data_ptr());

            Json::binary_t bytes;
            bytes.resize(cpu.numel() * sizeof(uint16_t));
            for (size_t index = 0; index < cpu.numel(); ++index) {
                const pxr::GfHalf half(values[index]);
                const uint16_t bits = half.bits();
                bytes[index * 2 + 0] = static_cast<std::uint8_t>(bits & 0xFFu);
                bytes[index * 2 + 1] = static_cast<std::uint8_t>((bits >> 8u) & 0xFFu);
            }
            return bytes;
        }

        Json::binary_t int64_scalar_to_bytes(const int64_t value) {
            Json::binary_t bytes;
            bytes.resize(sizeof(int64_t));
            std::memcpy(bytes.data(), &value, sizeof(int64_t));
            return bytes;
        }

        Json make_shape_json(const lfs::core::TensorShape& shape) {
            Json result = Json::array();
            for (const size_t dim : shape.dims()) {
                result.push_back(static_cast<int64_t>(dim));
            }
            return result;
        }

        Result<void> validate_export_tensors(const SplatData& splat_data, const std::filesystem::path& output_path) {
            if (!splat_data.means().is_valid() || splat_data.means().ndim() != 2 || splat_data.means().size(1) != 3) {
                return make_error(ErrorCode::INTERNAL_ERROR, "NuRec export requires means shaped [N,3]", output_path);
            }
            if (!splat_data.sh0().is_valid() || splat_data.sh0().ndim() != 3 || splat_data.sh0().size(1) != 1 || splat_data.sh0().size(2) != 3) {
                return make_error(ErrorCode::INTERNAL_ERROR, "NuRec export requires SH0 shaped [N,1,3]", output_path);
            }
            if (!splat_data.scaling_raw().is_valid() || splat_data.scaling_raw().ndim() != 2 || splat_data.scaling_raw().size(1) != 3) {
                return make_error(ErrorCode::INTERNAL_ERROR, "NuRec export requires log-scales shaped [N,3]", output_path);
            }
            if (!splat_data.opacity_raw().is_valid() || splat_data.opacity_raw().ndim() != 2 || splat_data.opacity_raw().size(1) != 1) {
                return make_error(ErrorCode::INTERNAL_ERROR, "NuRec export requires opacity_raw shaped [N,1]", output_path);
            }
            return {};
        }

        Json build_nurec_payload(const SplatData& splat_data) {
            const int sh_degree = splat_data.get_max_sh_degree();
            const size_t sh_coeffs = sh_degree > 0 ? static_cast<size_t>((sh_degree + 1) * (sh_degree + 1) - 1) : 0;

            Tensor means = splat_data.means();
            Tensor sh0 = splat_data.sh0();
            Tensor shN = splat_data.shN();
            Tensor scales = splat_data.scaling_raw();
            Tensor densities = splat_data.opacity_raw();
            Tensor rotations = splat_data.get_rotation();

            if (splat_data.has_deleted_mask()) {
                const auto keep_mask = splat_data.deleted().logical_not();
                means = means.index_select(0, keep_mask);
                sh0 = sh0.index_select(0, keep_mask);
                scales = scales.index_select(0, keep_mask);
                densities = densities.index_select(0, keep_mask);
                rotations = rotations.index_select(0, keep_mask);
                if (shN.is_valid()) {
                    shN = shN.index_select(0, keep_mask);
                }
            }

            Json payload = {
                {std::string(kRootKey), {
                                            {"version", "0.2.576"},
                                            {"model", "nre"},
                                            {"config", {
                                                           {"layers", {
                                                                          {"gaussians", {
                                                                                            {"name", "sh-gaussians"},
                                                                                            {"device", "cuda"},
                                                                                            {"density_activation", "sigmoid"},
                                                                                            {"scale_activation", "exp"},
                                                                                            {"rotation_activation", "normalize"},
                                                                                            {"precision", 16},
                                                                                            {"particle", {
                                                                                                             {"density_kernel_planar", false},
                                                                                                             {"density_kernel_degree", 2},
                                                                                                             {"density_kernel_density_clamping", false},
                                                                                                             {"density_kernel_min_response", 0.0113},
                                                                                                             {"radiance_sph_degree", 3},
                                                                                                         }},
                                                                                            {"transmittance_threshold", 0.001},
                                                                                        }},
                                                                      }},
                                                           {"renderer", {
                                                                            {"name", "3dgut-nrend"},
                                                                            {"log_level", 3},
                                                                            {"force_update", false},
                                                                            {"update_step_train_batch_end", false},
                                                                            {"per_ray_features", false},
                                                                            {"global_z_order", false},
                                                                            {"projection", {
                                                                                               {"n_rolling_shutter_iterations", 5},
                                                                                               {"ut_dim", 3},
                                                                                               {"ut_alpha", 1.0},
                                                                                               {"ut_beta", 2.0},
                                                                                               {"ut_kappa", 0.0},
                                                                                               {"ut_require_all_sigma_points", false},
                                                                                               {"image_margin_factor", 0.1},
                                                                                               {"min_projected_ray_radius", 0.5477225575051661},
                                                                                           }},
                                                                            {"culling", {
                                                                                            {"rect_bounding", true},
                                                                                            {"tight_opacity_bounding", true},
                                                                                            {"tile_based", true},
                                                                                            {"near_clip_distance", 0.2},
                                                                                            {"far_clip_distance", 3.402823466e38},
                                                                                        }},
                                                                            {"render", {
                                                                                           {"mode", "kbuffer"},
                                                                                           {"k_buffer_size", 0},
                                                                                       }},
                                                                        }},
                                                           {"name", "gaussians_primitive"},
                                                           {"appearance_embedding", {
                                                                                        {"name", "skip-appearance"},
                                                                                        {"embedding_dim", 0},
                                                                                        {"device", "cuda"},
                                                                                    }},
                                                           {"background", {
                                                                              {"name", "skip-background"},
                                                                              {"device", "cuda"},
                                                                              {"composite_in_linear_space", false},
                                                                          }},
                                                       }},
                                            {std::string(kStateDictKey), {
                                                                             {"._extra_state", {{"obj_track_ids", {{"gaussians", Json::array()}}}}},
                                                                         }},
                                        }},
            };

            auto& state_dict = payload[std::string(kRootKey)][std::string(kStateDictKey)];
            state_dict[std::string(kPositionsKey)] = Json::binary(float_tensor_to_half_bytes(means));
            state_dict[std::string(kRotationsKey)] = Json::binary(float_tensor_to_half_bytes(rotations));
            state_dict[std::string(kScalesKey)] = Json::binary(float_tensor_to_half_bytes(scales));
            state_dict[std::string(kDensitiesKey)] = Json::binary(float_tensor_to_half_bytes(densities));
            state_dict[std::string(kAlbedoKey)] = Json::binary(float_tensor_to_half_bytes(sh0.reshape({static_cast<int>(means.size(0)), 3})));

            if (sh_coeffs > 0 && shN.is_valid() && shN.numel() > 0) {
                state_dict[std::string(kSpecularKey)] = Json::binary(float_tensor_to_half_bytes(shN.reshape({static_cast<int>(means.size(0)), static_cast<int>(sh_coeffs * 3)})));
            } else {
                state_dict[std::string(kSpecularKey)] = Json::binary(Json::binary_t{});
            }

            state_dict[std::string(kExtraSignalKey)] = Json::binary(Json::binary_t{});
            state_dict[std::string(kActiveFeaturesKey)] = Json::binary(int64_scalar_to_bytes(sh_degree));

            state_dict[std::format("{}.shape", kPositionsKey)] = make_shape_json(means.shape());
            state_dict[std::format("{}.shape", kRotationsKey)] = make_shape_json(rotations.shape());
            state_dict[std::format("{}.shape", kScalesKey)] = make_shape_json(scales.shape());
            state_dict[std::format("{}.shape", kDensitiesKey)] = make_shape_json(densities.shape());
            state_dict[std::format("{}.shape", kAlbedoKey)] = Json::array({static_cast<int64_t>(means.size(0)), 3});
            state_dict[std::format("{}.shape", kSpecularKey)] = Json::array({static_cast<int64_t>(means.size(0)), static_cast<int64_t>(sh_coeffs * 3)});
            state_dict[std::format("{}.shape", kExtraSignalKey)] = Json::array({static_cast<int64_t>(means.size(0)), 0});
            state_dict[std::format("{}.shape", kActiveFeaturesKey)] = Json::array();

            return payload;
        }

        std::string render_default_usda() {
            std::ostringstream out;
            out << "#usda 1.0\n";
            out << "(\n";
            out << "    customLayerData = {\n";
            out << kRenderSettingsBlock;
            out << "    }\n";
            out << "    defaultPrim = \"World\"\n";
            out << "    metersPerUnit = 1\n";
            out << "    upAxis = \"Z\"\n";
            out << ")\n\n";
            out << "def Xform \"World\"\n";
            out << "{\n";
            out << "    over \"gauss\" (\n";
            out << "        prepend references = @gauss.usda@\n";
            out << "    )\n";
            out << "    {\n";
            out << "    }\n";
            out << "}\n";
            return out.str();
        }

        std::string render_gauss_usda(const std::string& nurec_filename,
                                      const std::array<float, 3>& min_bounds,
                                      const std::array<float, 3>& max_bounds) {
            std::ostringstream out;
            out << "#usda 1.0\n";
            out << "(\n";
            out << "    customLayerData = {\n";
            out << kRenderSettingsBlock;
            out << "    }\n";
            out << "    defaultPrim = \"World\"\n";
            out << "    metersPerUnit = 1\n";
            out << "    upAxis = \"Z\"\n";
            out << ")\n\n";
            out << "def Xform \"World\"\n";
            out << "{\n";
            out << "    def Volume \"gauss\"\n";
            out << "    {\n";
            out << "        float3[] extent = [(" << format_float(min_bounds[0]) << ", " << format_float(min_bounds[1]) << ", " << format_float(min_bounds[2]) << "), ("
                << format_float(max_bounds[0]) << ", " << format_float(max_bounds[1]) << ", " << format_float(max_bounds[2]) << ")]\n";
            out << "        custom rel field:density = </World/gauss/density_field>\n";
            out << "        custom rel field:emissiveColor = </World/gauss/emissive_color_field>\n";
            out << "        custom float3 omni:nurec:crop:maxBounds = (" << format_float(max_bounds[0]) << ", " << format_float(max_bounds[1]) << ", " << format_float(max_bounds[2]) << ")\n";
            out << "        custom float3 omni:nurec:crop:minBounds = (" << format_float(min_bounds[0]) << ", " << format_float(min_bounds[1]) << ", " << format_float(min_bounds[2]) << ")\n";
            out << "        custom bool omni:nurec:isNuRecVolume = 1\n";
            out << "        custom float3 omni:nurec:offset = (0, 0, 0)\n";
            out << "        custom bool omni:nurec:useProxyTransform = 0\n";
            out << "        custom rel proxy\n";
            out << "        matrix4d xformOp:transform = ( (-1, 0, 0, 0), (0, 0, -1, 0), (0, -1, 0, 0), (0, 0, 0, 1) )\n";
            out << "        uniform token[] xformOpOrder = [\"xformOp:transform\"]\n\n";
            out << "        def OmniNuRecFieldAsset \"density_field\"\n";
            out << "        {\n";
            out << "            custom token fieldDataType = \"float\"\n";
            out << "            custom token fieldName = \"density\"\n";
            out << "            custom token fieldRole = \"density\"\n";
            out << "            custom asset filePath = @./" << nurec_filename << "@\n";
            out << "        }\n\n";
            out << "        def OmniNuRecFieldAsset \"emissive_color_field\"\n";
            out << "        {\n";
            out << "            custom token fieldDataType = \"float3\"\n";
            out << "            custom token fieldName = \"emissiveColor\"\n";
            out << "            custom token fieldRole = \"emissiveColor\"\n";
            out << "            custom asset filePath = @./" << nurec_filename << "@\n";
            out << "            custom float4 omni:nurec:ccmB = (0, 0, 1, 0)\n";
            out << "            custom float4 omni:nurec:ccmG = (0, 1, 0, 0)\n";
            out << "            custom float4 omni:nurec:ccmR = (1, 0, 0, 0)\n";
            out << "        }\n";
            out << "    }\n";
            out << "}\n";
            return out.str();
        }

        std::expected<UsdzEntries, std::string> read_usdz_entries(const std::filesystem::path& filepath) {
            struct archive* archive = archive_read_new();
            if (!archive) {
                return std::unexpected("Failed to allocate libarchive reader");
            }

            archive_read_support_format_zip(archive);
            archive_read_support_filter_all(archive);

#ifdef _WIN32
            const int open_result = archive_read_open_filename_w(archive, filepath.wstring().c_str(), 10240);
#else
            const int open_result = archive_read_open_filename(archive, filepath.c_str(), 10240);
#endif
            if (open_result != ARCHIVE_OK) {
                const std::string error = archive_error_string(archive) ? archive_error_string(archive) : "unknown error";
                archive_read_free(archive);
                return std::unexpected(std::format("Failed to open USDZ archive: {}", error));
            }

            UsdzEntries entries;
            struct archive_entry* entry = nullptr;
            while (archive_read_next_header(archive, &entry) == ARCHIVE_OK) {
                const char* entry_name = archive_entry_pathname(entry);
                if (!entry_name) {
                    archive_read_free(archive);
                    return std::unexpected("USDZ archive entry is missing a path name");
                }

                const auto size = static_cast<size_t>(archive_entry_size(entry));
                std::vector<uint8_t> data(size);
                if (size > 0) {
                    const ssize_t read = archive_read_data(archive, data.data(), size);
                    if (read != static_cast<ssize_t>(size)) {
                        const std::string error = archive_error_string(archive) ? archive_error_string(archive) : "unknown error";
                        archive_read_free(archive);
                        return std::unexpected(std::format("Failed to read USDZ entry '{}': {}", entry_name, error));
                    }
                }
                entries.files.emplace(entry_name, std::move(data));
            }

            archive_read_free(archive);
            return entries;
        }

        class ZipArchiveWriter {
        public:
            explicit ZipArchiveWriter(const std::filesystem::path& output_path) {
                archive_ = archive_write_new();
                if (!archive_) {
                    throw std::runtime_error("Failed to allocate libarchive writer");
                }

                archive_write_add_filter_none(archive_);
                if (archive_write_set_format_zip(archive_) != ARCHIVE_OK) {
                    const std::string error = archive_error_string(archive_) ? archive_error_string(archive_) : "unknown error";
                    archive_write_free(archive_);
                    archive_ = nullptr;
                    throw std::runtime_error(std::format("Failed to initialize zip writer: {}", error));
                }
                archive_write_set_options(archive_, "zip:compression=store");

#ifdef _WIN32
                const int open_result = archive_write_open_filename_w(archive_, output_path.wstring().c_str());
#else
                const int open_result = archive_write_open_filename(archive_, output_path.c_str());
#endif
                if (open_result != ARCHIVE_OK) {
                    const std::string error = archive_error_string(archive_) ? archive_error_string(archive_) : "unknown error";
                    archive_write_free(archive_);
                    archive_ = nullptr;
                    throw std::runtime_error(std::format("Failed to open USDZ output: {}", error));
                }
            }

            ~ZipArchiveWriter() {
                if (archive_) {
                    archive_write_close(archive_);
                    archive_write_free(archive_);
                }
            }

            std::expected<void, std::string> close() {
                if (!archive_) {
                    return {};
                }

                if (archive_write_close(archive_) != ARCHIVE_OK) {
                    const std::string error = archive_error_string(archive_) ? archive_error_string(archive_) : "unknown error";
                    archive_write_free(archive_);
                    archive_ = nullptr;
                    return std::unexpected(std::format("Failed to close USDZ archive: {}", error));
                }

                archive_write_free(archive_);
                archive_ = nullptr;
                return {};
            }

            std::expected<void, std::string> add_file(const std::string& filename, const std::span<const uint8_t> data) {
                auto* entry = archive_entry_new();
                if (!entry) {
                    return std::unexpected("Failed to allocate archive entry");
                }

                archive_entry_set_pathname(entry, filename.c_str());
                archive_entry_set_size(entry, static_cast<la_int64_t>(data.size()));
                archive_entry_set_filetype(entry, AE_IFREG);
                archive_entry_set_perm(entry, 0644);
                archive_entry_set_mtime(entry, 0, 0);

                if (archive_write_header(archive_, entry) != ARCHIVE_OK) {
                    const std::string error = archive_error_string(archive_) ? archive_error_string(archive_) : "unknown error";
                    archive_entry_free(entry);
                    return std::unexpected(std::format("Failed to write USDZ header for '{}': {}", filename, error));
                }

                if (!data.empty()) {
                    const ssize_t written = archive_write_data(archive_, data.data(), data.size());
                    if (written != static_cast<ssize_t>(data.size())) {
                        const std::string error = archive_error_string(archive_) ? archive_error_string(archive_) : "unknown error";
                        archive_entry_free(entry);
                        return std::unexpected(std::format("Failed to write USDZ data for '{}': {}", filename, error));
                    }
                }

                archive_entry_free(entry);
                return {};
            }

        private:
            struct archive* archive_ = nullptr;
        };

        std::expected<std::string, std::string> find_nurec_entry(const UsdzEntries& entries) {
            std::vector<std::string> candidates;
            for (const auto& [name, _] : entries.files) {
                const auto extension = normalized_extension(std::filesystem::path(name));
                if (extension == ".nurec") {
                    candidates.push_back(name);
                }
            }

            if (candidates.empty()) {
                return std::unexpected("USDZ archive does not contain a .nurec payload");
            }
            if (candidates.size() > 1) {
                return std::unexpected("USDZ archive contains multiple .nurec payloads");
            }
            return candidates.front();
        }

        std::expected<Json, std::string> decode_nurec_payload(const UsdzEntries& entries) {
            auto nurec_name = find_nurec_entry(entries);
            if (!nurec_name) {
                return std::unexpected(nurec_name.error());
            }

            const auto it = entries.files.find(*nurec_name);
            if (it == entries.files.end()) {
                return std::unexpected("Resolved .nurec payload is missing from archive map");
            }

            auto decompressed = gzip_decompress(it->second);
            if (!decompressed) {
                return std::unexpected(decompressed.error());
            }

            try {
                return Json::from_msgpack(*decompressed, true, true);
            } catch (const std::exception& e) {
                return std::unexpected(std::format("Failed to decode .nurec msgpack payload: {}", e.what()));
            }
        }

        std::expected<std::vector<size_t>, std::string> get_shape(const Json& state_dict, const std::string_view key) {
            const auto shape_key = std::format("{}.shape", key);
            if (!state_dict.contains(shape_key)) {
                return std::unexpected(std::format("Missing NuRec shape metadata '{}'", shape_key));
            }

            const auto& value = state_dict.at(shape_key);
            if (!value.is_array()) {
                return std::unexpected(std::format("NuRec shape '{}' is not an array", shape_key));
            }

            std::vector<size_t> shape;
            shape.reserve(value.size());
            for (const auto& dim : value) {
                shape.push_back(static_cast<size_t>(dim.get<int64_t>()));
            }
            return shape;
        }

        std::expected<int, std::string> get_active_features(const Json& state_dict) {
            if (!state_dict.contains(std::string(kActiveFeaturesKey))) {
                return std::unexpected("NuRec payload is missing n_active_features");
            }
            const auto& binary = state_dict.at(std::string(kActiveFeaturesKey)).get_binary();
            if (binary.size() != sizeof(int64_t)) {
                return std::unexpected(std::format("NuRec n_active_features payload has {} bytes, expected {}", binary.size(), sizeof(int64_t)));
            }

            int64_t raw = 0;
            std::memcpy(&raw, binary.data(), sizeof(int64_t));
            return static_cast<int>(raw);
        }

        std::expected<std::vector<float>, std::string> decode_half_field(const Json& state_dict,
                                                                         const std::string_view key,
                                                                         const size_t expected_dims) {
            if (!state_dict.contains(std::string(key))) {
                return std::unexpected(std::format("NuRec payload is missing '{}'", key));
            }

            auto shape = get_shape(state_dict, key);
            if (!shape) {
                return std::unexpected(shape.error());
            }
            if (shape->size() != expected_dims) {
                return std::unexpected(std::format("NuRec tensor '{}' has {} dims, expected {}", key, shape->size(), expected_dims));
            }

            size_t element_count = 1;
            for (const size_t dim : *shape) {
                element_count *= dim;
            }

            const auto& binary = state_dict.at(std::string(key)).get_binary();
            if (binary.size() != element_count * sizeof(uint16_t)) {
                return std::unexpected(std::format("NuRec tensor '{}' has {} bytes, expected {}", key, binary.size(), element_count * sizeof(uint16_t)));
            }

            std::vector<float> output(element_count, 0.0f);
            for (size_t index = 0; index < element_count; ++index) {
                const uint16_t bits = static_cast<uint16_t>(binary[index * 2 + 0]) |
                                      (static_cast<uint16_t>(binary[index * 2 + 1]) << 8u);
                pxr::GfHalf half_value;
                half_value.setBits(bits);
                output[index] = static_cast<float>(half_value);
            }
            return output;
        }

        std::expected<SplatData, std::string> build_splat_from_payload(const Json& payload) {
            if (!payload.contains(std::string(kRootKey))) {
                return std::unexpected("NuRec payload is missing top-level nre_data block");
            }
            const auto& root = payload.at(std::string(kRootKey));
            if (!root.contains(std::string(kStateDictKey))) {
                return std::unexpected("NuRec payload is missing state_dict");
            }
            const auto& state_dict = root.at(std::string(kStateDictKey));

            auto active_features = get_active_features(state_dict);
            if (!active_features) {
                return std::unexpected(active_features.error());
            }
            if (*active_features < 0 || *active_features > kMaxSupportedShDegree) {
                return std::unexpected(std::format("NuRec payload uses unsupported SH degree {}", *active_features));
            }

            auto means_shape = get_shape(state_dict, kPositionsKey);
            auto rotations_shape = get_shape(state_dict, kRotationsKey);
            auto scales_shape = get_shape(state_dict, kScalesKey);
            auto densities_shape = get_shape(state_dict, kDensitiesKey);
            auto albedo_shape = get_shape(state_dict, kAlbedoKey);
            auto specular_shape = get_shape(state_dict, kSpecularKey);
            if (!means_shape)
                return std::unexpected(means_shape.error());
            if (!rotations_shape)
                return std::unexpected(rotations_shape.error());
            if (!scales_shape)
                return std::unexpected(scales_shape.error());
            if (!densities_shape)
                return std::unexpected(densities_shape.error());
            if (!albedo_shape)
                return std::unexpected(albedo_shape.error());
            if (!specular_shape)
                return std::unexpected(specular_shape.error());

            if (means_shape->size() != 2 || means_shape->at(1) != 3) {
                return std::unexpected("NuRec positions must be shaped [N,3]");
            }
            const size_t count = means_shape->at(0);
            if (rotations_shape->size() != 2 || rotations_shape->at(0) != count || rotations_shape->at(1) != 4) {
                return std::unexpected("NuRec rotations must be shaped [N,4]");
            }
            if (scales_shape->size() != 2 || scales_shape->at(0) != count || scales_shape->at(1) != 3) {
                return std::unexpected("NuRec scales must be shaped [N,3]");
            }
            if (densities_shape->size() != 2 || densities_shape->at(0) != count || densities_shape->at(1) != 1) {
                return std::unexpected("NuRec densities must be shaped [N,1]");
            }
            if (albedo_shape->size() != 2 || albedo_shape->at(0) != count || albedo_shape->at(1) != 3) {
                return std::unexpected("NuRec features_albedo must be shaped [N,3]");
            }
            if (specular_shape->size() != 2 || specular_shape->at(0) != count) {
                return std::unexpected("NuRec features_specular must be shaped [N,C]");
            }
            if (specular_shape->at(1) % 3 != 0) {
                return std::unexpected("NuRec features_specular column count must be divisible by 3");
            }

            auto means_data = decode_half_field(state_dict, kPositionsKey, 2);
            auto rotations_data = decode_half_field(state_dict, kRotationsKey, 2);
            auto scales_data = decode_half_field(state_dict, kScalesKey, 2);
            auto densities_data = decode_half_field(state_dict, kDensitiesKey, 2);
            auto albedo_data = decode_half_field(state_dict, kAlbedoKey, 2);
            auto specular_data = decode_half_field(state_dict, kSpecularKey, 2);
            if (!means_data)
                return std::unexpected(means_data.error());
            if (!rotations_data)
                return std::unexpected(rotations_data.error());
            if (!scales_data)
                return std::unexpected(scales_data.error());
            if (!densities_data)
                return std::unexpected(densities_data.error());
            if (!albedo_data)
                return std::unexpected(albedo_data.error());
            if (!specular_data)
                return std::unexpected(specular_data.error());

            const size_t specular_coeffs = specular_shape->at(1) / 3;
            Tensor shN;
            if (specular_coeffs > 0) {
                shN = Tensor::from_vector(*specular_data, {count, specular_coeffs, size_t{3}}, Device::CUDA);
            }

            return SplatData(
                *active_features,
                Tensor::from_vector(*means_data, {count, size_t{3}}, Device::CUDA),
                Tensor::from_vector(*albedo_data, {count, size_t{1}, size_t{3}}, Device::CUDA),
                std::move(shN),
                Tensor::from_vector(*scales_data, {count, size_t{3}}, Device::CUDA),
                Tensor::from_vector(*rotations_data, {count, size_t{4}}, Device::CUDA),
                Tensor::from_vector(*densities_data, {count, size_t{1}}, Device::CUDA),
                1.0f);
        }

    } // namespace

    std::expected<bool, std::string> is_nurec_usdz(const std::filesystem::path& filepath) {
        auto entries = read_usdz_entries(filepath);
        if (!entries) {
            return std::unexpected(entries.error());
        }

        size_t match_count = 0;
        for (const auto& [name, _] : entries->files) {
            if (normalized_extension(std::filesystem::path(name)) == ".nurec") {
                ++match_count;
                if (match_count > 1) {
                    return std::unexpected("USDZ archive contains multiple .nurec payloads");
                }
            }
        }

        return match_count == 1;
    }

    std::expected<void, std::string> validate_nurec_usdz(const std::filesystem::path& filepath) {
        auto entries = read_usdz_entries(filepath);
        if (!entries) {
            return std::unexpected(entries.error());
        }
        auto payload = decode_nurec_payload(*entries);
        if (!payload) {
            return std::unexpected(payload.error());
        }
        auto splat = build_splat_from_payload(*payload);
        if (!splat) {
            return std::unexpected(splat.error());
        }
        return {};
    }

    std::expected<SplatData, std::string> load_nurec_usdz(const std::filesystem::path& filepath) {
        auto entries = read_usdz_entries(filepath);
        if (!entries) {
            return std::unexpected(entries.error());
        }
        auto payload = decode_nurec_payload(*entries);
        if (!payload) {
            return std::unexpected(payload.error());
        }
        return build_splat_from_payload(*payload);
    }

    Result<void> save_nurec_usdz(const SplatData& splat_data, const NurecUsdzSaveOptions& options) {
        if (!report_export_progress(options.progress_callback, 0.0f, "Preparing USDZ")) {
            return make_error(ErrorCode::CANCELLED, "NuRec USDZ export cancelled", options.output_path);
        }

        if (splat_data.size() == 0) {
            return make_error(ErrorCode::EMPTY_DATASET, "No splats to write", options.output_path);
        }
        if (splat_data.get_max_sh_degree() < 0 || splat_data.get_max_sh_degree() > kMaxSupportedShDegree) {
            return make_error(ErrorCode::UNSUPPORTED_FORMAT,
                              std::format("NuRec USDZ export supports SH degree 0-{}", kMaxSupportedShDegree),
                              options.output_path);
        }

        const size_t visible_count = splat_data.has_deleted_mask()
                                         ? static_cast<size_t>(splat_data.visible_count())
                                         : splat_data.size();
        if (visible_count == 0) {
            return make_error(ErrorCode::EMPTY_DATASET, "No visible splats to write", options.output_path);
        }

        if (auto validation = validate_export_tensors(splat_data, options.output_path); !validation) {
            return std::unexpected(validation.error());
        }
        if (auto writable = verify_writable(options.output_path); !writable) {
            return std::unexpected(writable.error());
        }
        if (normalized_extension(options.output_path) != ".usdz") {
            return make_error(ErrorCode::UNSUPPORTED_FORMAT,
                              "NuRec USDZ export supports only .usdz output paths",
                              options.output_path);
        }

        if (!report_export_progress(options.progress_callback, 0.15f, "Building NuRec payload")) {
            return make_error(ErrorCode::CANCELLED, "NuRec USDZ export cancelled", options.output_path);
        }

        Json payload = build_nurec_payload(splat_data);

        if (!report_export_progress(options.progress_callback, 0.45f, "Compressing NuRec payload")) {
            return make_error(ErrorCode::CANCELLED, "NuRec USDZ export cancelled", options.output_path);
        }

        std::vector<uint8_t> payload_msgpack = Json::to_msgpack(payload);
        auto nurec_bytes = gzip_compress(payload_msgpack);
        if (!nurec_bytes) {
            return make_error(ErrorCode::ENCODING_FAILED, nurec_bytes.error(), options.output_path);
        }

        if (!report_export_progress(options.progress_callback, 0.65f, "Preparing USDZ bounds")) {
            return make_error(ErrorCode::CANCELLED, "NuRec USDZ export cancelled", options.output_path);
        }

        Tensor means_for_bounds = splat_data.means();
        if (splat_data.has_deleted_mask()) {
            means_for_bounds = means_for_bounds.index_select(0, splat_data.deleted().logical_not());
        }
        const auto means = means_for_bounds.contiguous().to(Device::CPU);
        const auto* const means_ptr = static_cast<const float*>(means.data_ptr());
        const size_t count = static_cast<size_t>(means.size(0));
        std::array<float, 3> min_bounds = {means_ptr[0], means_ptr[1], means_ptr[2]};
        std::array<float, 3> max_bounds = min_bounds;
        for (size_t index = 0; index < count; ++index) {
            for (size_t axis = 0; axis < 3; ++axis) {
                const float value = means_ptr[index * 3 + axis];
                min_bounds[axis] = std::min(min_bounds[axis], value);
                max_bounds[axis] = std::max(max_bounds[axis], value);
            }
        }

        const std::string nurec_filename = std::format("{}.nurec", options.output_path.stem().string());
        const std::string default_usda = render_default_usda();
        const std::string gauss_usda = render_gauss_usda(nurec_filename, min_bounds, max_bounds);

        const std::uintmax_t estimated_size = default_usda.size() + gauss_usda.size() + nurec_bytes->size() + 4096;
        if (auto space_check = check_disk_space(options.output_path, estimated_size, 1.05f); !space_check) {
            return std::unexpected(space_check.error());
        }

        if (auto dir_result = ensure_output_parent_directory(options.output_path); !dir_result) {
            return std::unexpected(dir_result.error());
        }

        if (!report_export_progress(options.progress_callback, 0.8f, "Writing USDZ")) {
            return make_error(ErrorCode::CANCELLED, "NuRec USDZ export cancelled", options.output_path);
        }

        ScopedAtomicOutputFile atomic_output(options.output_path, AtomicOutputTempName::PreserveExtension);
        try {
            ZipArchiveWriter writer(atomic_output.temp_path());
            const std::vector<uint8_t> default_bytes(default_usda.begin(), default_usda.end());
            const std::vector<uint8_t> gauss_bytes(gauss_usda.begin(), gauss_usda.end());
            if (auto result = writer.add_file("default.usda", default_bytes); !result) {
                return make_error(ErrorCode::ARCHIVE_CREATION_FAILED, result.error(), options.output_path);
            }
            if (!report_export_progress(options.progress_callback, 0.85f, "Writing NuRec payload")) {
                return make_error(ErrorCode::CANCELLED, "NuRec USDZ export cancelled", options.output_path);
            }
            if (auto result = writer.add_file(nurec_filename, *nurec_bytes); !result) {
                return make_error(ErrorCode::ARCHIVE_CREATION_FAILED, result.error(), options.output_path);
            }
            if (!report_export_progress(options.progress_callback, 0.95f, "Writing USD metadata")) {
                return make_error(ErrorCode::CANCELLED, "NuRec USDZ export cancelled", options.output_path);
            }
            if (auto result = writer.add_file("gauss.usda", gauss_bytes); !result) {
                return make_error(ErrorCode::ARCHIVE_CREATION_FAILED, result.error(), options.output_path);
            }
            if (auto result = writer.close(); !result) {
                return make_error(ErrorCode::ARCHIVE_CREATION_FAILED, result.error(), options.output_path);
            }
        } catch (const std::exception& e) {
            return make_error(ErrorCode::ARCHIVE_CREATION_FAILED,
                              std::format("Failed to create USDZ archive: {}", e.what()),
                              options.output_path);
        }

        if (!report_export_progress(options.progress_callback, 1.0f, "USDZ export complete")) {
            return make_error(ErrorCode::CANCELLED, "NuRec USDZ export cancelled", options.output_path);
        }

        if (auto commit_result = atomic_output.commit(); !commit_result) {
            return std::unexpected(commit_result.error());
        }

        LOG_INFO("Saved NuRec USDZ file: {}", lfs::core::path_to_utf8(options.output_path));
        return {};
    }

} // namespace lfs::io

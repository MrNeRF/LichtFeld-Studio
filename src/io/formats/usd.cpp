/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "usd.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "core/provenance.hpp"
#include "core/splat_data_transform.hpp"
#include "core/tensor.hpp"
#include "formats/usd_flat/crate.hpp"
#include "formats/usd_flat/usda_read.hpp"
#include "formats/usd_flat/usda_write.hpp"
#include "formats/usd_flat/usdz.hpp"
#include "io/atomic_output.hpp"
#include "io/exporter.hpp"
#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <format>
#include <fstream>
#include <glm/glm.hpp>
#include <optional>
#include <set>
#include <string_view>

namespace lfs::io {

    using lfs::core::DataType;
    using lfs::core::Device;
    using lfs::core::SplatData;
    using lfs::core::Tensor;
    using usd_flat::FlatAttribute;
    using usd_flat::FlatPrim;
    using usd_flat::FlatStage;

    namespace {

        constexpr int MAX_SUPPORTED_SH_DEGREE = 3;
        constexpr float SCENE_SCALE = 0.5f;
        constexpr float MIN_SCALE = 1e-12f;
        constexpr float OPACITY_EPSILON = 1e-6f;
        constexpr float EXTENT_LIMIT = 50000.0f;

        lfs::Error usd_project_error(std::string message) {
            return lfs::make_error(lfs::ErrorInit{
                .code = lfs::ErrorCode::DataLoss,
                .domain = lfs::ErrorDomain::IO,
                .severity = lfs::Severity::Error,
                .retryability = lfs::Retryability::NotRetryable,
                .operation_id = {},
                .user_message = message,
                .detail = std::move(message),
                .detection = LFS_SOURCE_SITE_CURRENT(),
                .fields = {},
                .native = std::nullopt,
            });
        }

        std::string normalized_extension(const std::filesystem::path& path) {
            auto extension = path.extension().string();
            std::transform(extension.begin(), extension.end(), extension.begin(), [](const unsigned char character) {
                return static_cast<char>(std::tolower(character));
            });
            return extension;
        }

        bool is_crate_header(const std::array<char, 8>& header) {
            static constexpr std::array<char, 8> magic = {'P', 'X', 'R', '-', 'U', 'S', 'D', 'C'};
            return header == magic;
        }

        lfs::Result<FlatStage> read_stage(const std::filesystem::path& path) {
            const auto extension = normalized_extension(path);
            if (extension == ".usda") {
                return usd_flat::read_usda(path);
            }
            if (extension == ".usdc") {
                return usd_flat::read_usdc(path);
            }
            if (extension == ".usd") {
                std::ifstream input(path, std::ios::binary);
                std::array<char, 8> header{};
                if (!input || !input.read(header.data(), static_cast<std::streamsize>(header.size()))) {
                    return usd_flat::make_flat_error("Failed to read USD file header: " + lfs::core::path_to_utf8(path));
                }
                if (is_crate_header(header)) {
                    return usd_flat::read_usdc(path);
                }
                if (std::string_view(header.data(), 5) == "#usda") {
                    return usd_flat::read_usda(path);
                }
                return usd_flat::make_flat_error("Unsupported .usd file: expected a crate or #usda header");
            }
            if (extension == ".usdz") {
                return usd_flat::read_usdz(path);
            }
            return usd_flat::make_flat_error("Unsupported USD extension: " + extension);
        }

        std::string flat_error_message(const lfs::Error& error) {
            return std::string(error.user_message());
        }

        std::vector<const FlatPrim*> particle_fields(const FlatStage& stage) {
            std::vector<const FlatPrim*> result;
            for (const auto& prim : stage.prims) {
                if (prim.type_name == "ParticleField" || prim.type_name == "ParticleField3DGaussianSplat") {
                    result.push_back(&prim);
                }
            }
            return result;
        }

        std::string particle_paths(const std::vector<const FlatPrim*>& prims) {
            if (prims.empty()) {
                return "none";
            }
            std::string result;
            for (const auto* prim : prims) {
                if (!result.empty()) {
                    result += ", ";
                }
                result += prim->path;
            }
            return result;
        }

        std::string stage_prim_types(const FlatStage& stage) {
            std::set<std::string> types;
            for (const auto& prim : stage.prims) {
                if (!prim.type_name.empty()) {
                    types.insert(prim.type_name);
                }
            }
            if (types.empty()) {
                return "none";
            }
            std::string result;
            for (const auto& type : types) {
                if (!result.empty()) {
                    result += ", ";
                }
                result += type;
            }
            return result;
        }

        std::expected<const FlatPrim*, std::string> find_particlefield_prim(const FlatStage& stage) {
            const auto candidates = particle_fields(stage);
            const FlatPrim* default_prim = nullptr;
            if (!stage.default_prim.empty()) {
                for (const auto& prim : stage.prims) {
                    if (prim.path == stage.default_prim) {
                        default_prim = &prim;
                        break;
                    }
                }
            }
            if (default_prim) {
                if (default_prim->type_name == "ParticleField" || default_prim->type_name == "ParticleField3DGaussianSplat") {
                    return default_prim;
                }
                std::vector<const FlatPrim*> descendants;
                const std::string prefix = default_prim->path + "/";
                for (const auto* prim : candidates) {
                    if (prim->path.rfind(prefix, 0) == 0) {
                        descendants.push_back(prim);
                    }
                }
                if (descendants.size() == 1) {
                    return descendants.front();
                }
                if (descendants.size() > 1) {
                    return std::unexpected(std::format(
                        "Default prim {} contains multiple OpenUSD ParticleField prims: {}",
                        default_prim->path,
                        particle_paths(descendants)));
                }
            }
            if (candidates.empty()) {
                return std::unexpected(std::format(
                    "No OpenUSD ParticleField prim found in stage. Prim types in stage: {}",
                    stage_prim_types(stage)));
            }
            if (candidates.size() > 1) {
                if (default_prim) {
                    return std::unexpected(std::format(
                        "Default prim {} does not resolve to a unique OpenUSD ParticleField. Candidate ParticleField prims in stage: {}",
                        default_prim->path,
                        particle_paths(candidates)));
                }
                return std::unexpected(std::format(
                    "Stage contains multiple OpenUSD ParticleField prims and no default prim. Candidate ParticleField prims: {}",
                    particle_paths(candidates)));
            }
            return candidates.front();
        }

        const FlatAttribute* attribute(const FlatPrim& prim, const std::string& name) {
            const auto it = prim.attributes.find(name);
            return it == prim.attributes.end() || !it->second.authored ? nullptr : &it->second;
        }

        FlatAttribute* attribute(FlatPrim& prim, const std::string& name) {
            const auto it = prim.attributes.find(name);
            return it == prim.attributes.end() || !it->second.authored ? nullptr : &it->second;
        }

        const FlatAttribute* first_attribute(const FlatPrim& prim, const std::string& float_name, const std::string& half_name) {
            if (const auto* value = attribute(prim, float_name)) {
                return value;
            }
            return attribute(prim, half_name);
        }

        FlatAttribute* first_attribute(FlatPrim& prim, const std::string& float_name, const std::string& half_name) {
            if (auto* value = attribute(prim, float_name)) {
                return value;
            }
            return attribute(prim, half_name);
        }

        int infer_sh_degree(const size_t particle_count, const size_t coefficient_count) {
            if (particle_count == 0 || coefficient_count == 0 || coefficient_count % particle_count != 0) {
                return 0;
            }
            const auto coeffs = coefficient_count / particle_count;
            const int degree = static_cast<int>(std::llround(std::sqrt(static_cast<double>(coeffs)))) - 1;
            return degree >= 0 && static_cast<size_t>((degree + 1) * (degree + 1)) == coeffs ? degree : 0;
        }

        std::vector<float> default_rotations(const size_t count) {
            std::vector<float> values(count * 4, 0.0f);
            for (size_t index = 0; index < count; ++index) {
                values[index * 4] = 1.0f;
            }
            return values;
        }

        std::vector<float> default_scales(const size_t count) { return std::vector<float>(count * 3, 1.0f); }
        std::vector<float> default_opacities(const size_t count) { return std::vector<float>(count, 1.0f); }

        std::expected<SplatData, std::string> load_prim(FlatStage stage, const std::string& prim_path, const bool apply_stage_units) {
            const auto prim_it = std::find_if(stage.prims.begin(), stage.prims.end(), [&](const FlatPrim& candidate) {
                return candidate.path == prim_path;
            });
            if (prim_it == stage.prims.end()) {
                return std::unexpected("USD ParticleField prim disappeared during load: " + prim_path);
            }
            auto& prim = *prim_it;
            auto* positions_attr = first_attribute(prim, "positions", "positionsh");
            if (!positions_attr || positions_attr->values.empty()) {
                return std::unexpected(std::format("USD prim {} does not contain ParticleField positions", prim.path));
            }
            if (positions_attr->components != 3 || positions_attr->values.size() % 3 != 0) {
                return std::unexpected("Malformed USD positions attribute");
            }
            const size_t count = positions_attr->values.size() / 3;
            auto* rotations_attr = first_attribute(prim, "orientations", "orientationsh");
            auto* scales_attr = first_attribute(prim, "scales", "scalesh");
            auto* opacities_attr = first_attribute(prim, "opacities", "opacitiesh");
            if (!rotations_attr) {
                LOG_WARN("USD file omitted orientations; defaulting to identity quaternions");
            }
            if (!scales_attr) {
                LOG_WARN("USD file omitted scales; defaulting to unit scales");
            }
            if (!opacities_attr) {
                LOG_WARN("USD file omitted opacities; defaulting to fully opaque");
            }
            std::vector<float> positions = std::move(positions_attr->values);
            std::vector<float> rotations = rotations_attr ? std::move(rotations_attr->values) : default_rotations(count);
            std::vector<float> scales = scales_attr ? std::move(scales_attr->values) : default_scales(count);
            std::vector<float> opacities = opacities_attr ? std::move(opacities_attr->values) : default_opacities(count);
            if (rotations.size() != count * 4) {
                return std::unexpected("Malformed USD orientations attribute");
            }
            if (scales.size() != count * 3) {
                return std::unexpected("Malformed USD scales attribute");
            }
            if (opacities.size() != count) {
                return std::unexpected("Malformed USD opacities attribute");
            }

            auto* sh_attr = first_attribute(prim, "radiance:sphericalHarmonicsCoefficients", "radiance:sphericalHarmonicsCoefficientsh");
            const bool has_sh = sh_attr && !sh_attr->values.empty();
            int sh_degree = 0;
            auto* degree_attr = attribute(prim, "radiance:sphericalHarmonicsDegree");
            const bool has_degree = degree_attr && !degree_attr->values.empty();
            if (has_degree) {
                sh_degree = static_cast<int>(degree_attr->values.front());
            }
            std::vector<float> coefficients = has_sh ? std::move(sh_attr->values) : std::vector<float>(count * 3, 0.0f);
            if (has_sh) {
                if (sh_attr->components != 3 || coefficients.size() % 3 != 0) {
                    return std::unexpected("Malformed USD SH coefficient attribute");
                }
                const size_t triplets = coefficients.size() / 3;
                const int resolved_degree = has_degree ? sh_degree : infer_sh_degree(count, triplets);
                if (resolved_degree < 0 || resolved_degree > MAX_SUPPORTED_SH_DEGREE) {
                    return std::unexpected(std::format(
                        "Unsupported USD spherical harmonics degree {}. LichtFeld Studio supports degrees 0-{}.",
                        resolved_degree,
                        MAX_SUPPORTED_SH_DEGREE));
                }
                const size_t required = count * static_cast<size_t>((resolved_degree + 1) * (resolved_degree + 1));
                if (triplets < required) {
                    LOG_WARN("USD SH coefficient array is too short for {} gaussians at degree {}; ignoring it and falling back to neutral degree-0 radiance",
                             count,
                             resolved_degree);
                    sh_degree = 0;
                    coefficients.assign(count * 3, 0.0f);
                } else {
                    sh_degree = resolved_degree;
                    if (triplets > required) {
                        LOG_WARN("USD SH coefficient array is longer than expected; truncating to {} coefficients", required);
                        coefficients.resize(required * 3);
                    }
                }
            } else {
                LOG_WARN("USD file omitted SH coefficients; defaulting to neutral degree-0 radiance");
                sh_degree = 0;
                coefficients.assign(count * 3, 0.0f);
            }
            if (sh_degree < 0 || sh_degree > MAX_SUPPORTED_SH_DEGREE) {
                return std::unexpected(std::format(
                    "Unsupported USD spherical harmonics degree {}. LichtFeld Studio supports degrees 0-{}.",
                    sh_degree,
                    MAX_SUPPORTED_SH_DEGREE));
            }
            const size_t sh_count = static_cast<size_t>((sh_degree + 1) * (sh_degree + 1));
            std::vector<float> sh0(count * 3, 0.0f);
            const size_t rest = sh_count - 1;
            std::vector<float> shN(count * rest * 3, 0.0f);
            for (size_t index = 0; index < count; ++index) {
                std::copy_n(coefficients.data() + index * sh_count * 3, 3, sh0.data() + index * 3);
                if (rest != 0) {
                    std::copy_n(coefficients.data() + index * sh_count * 3 + 3, rest * 3, shN.data() + index * rest * 3);
                }
            }
            std::vector<float> scaling_raw(scales.size());
            std::transform(scales.begin(), scales.end(), scaling_raw.begin(), [](const float value) {
                return std::log(std::max(value, MIN_SCALE));
            });
            std::vector<float> opacity_raw(opacities.size());
            std::transform(opacities.begin(), opacities.end(), opacity_raw.begin(), [](const float value) {
                const float clamped = std::clamp(value, OPACITY_EPSILON, 1.0f - OPACITY_EPSILON);
                return std::log(clamped / (1.0f - clamped));
            });
            Tensor shN_tensor = rest == 0 ? Tensor::zeros({count, 0, 3}, Device::CUDA, DataType::Float32)
                                          : Tensor::from_vector(std::move(shN), {count, rest, 3}, Device::CUDA);
            SplatData data(sh_degree,
                           Tensor::from_vector(std::move(positions), {count, 3}, Device::CUDA),
                           Tensor::from_vector(std::move(sh0), {count, 1, 3}, Device::CUDA),
                           std::move(shN_tensor),
                           Tensor::from_vector(std::move(scaling_raw), {count, 3}, Device::CUDA),
                           Tensor::from_vector(std::move(rotations), {count, 4}, Device::CUDA),
                           Tensor::from_vector(std::move(opacity_raw), {count, 1}, Device::CUDA),
                           SCENE_SCALE);

            auto to_glm = [](const std::array<double, 16>& matrix) {
                glm::mat4 result(1.0f);
                for (int row = 0; row < 4; ++row) {
                    for (int column = 0; column < 4; ++column) {
                        result[row][column] = static_cast<float>(matrix[static_cast<std::size_t>(row * 4 + column)]);
                    }
                }
                return result;
            };
            std::vector<const FlatPrim*> ancestors;
            std::string path = prim.path;
            while (!path.empty() && path != "/") {
                const auto slash = path.find_last_of('/');
                const std::string parent = slash == 0 ? "/" : path.substr(0, slash);
                for (const auto& candidate : stage.prims) {
                    if (candidate.path == path) {
                        ancestors.push_back(&candidate);
                        break;
                    }
                }
                path = parent;
            }
            std::array<double, 16> world = usd_flat::identity_matrix();
            for (const auto* ancestor : ancestors) {
                std::array<double, 16> composed{};
                for (int row = 0; row < 4; ++row) {
                    for (int column = 0; column < 4; ++column) {
                        for (int index = 0; index < 4; ++index) {
                            composed[static_cast<std::size_t>(row * 4 + column)] +=
                                world[static_cast<std::size_t>(row * 4 + index)] * ancestor->local_transform[static_cast<std::size_t>(index * 4 + column)];
                        }
                    }
                }
                world = composed;
                if (ancestor->reset_xform_stack) {
                    break;
                }
            }
            if (world != usd_flat::identity_matrix()) {
                lfs::core::transform(data, to_glm(world));
            }
            if (apply_stage_units && std::abs(stage.meters_per_unit - 1.0) >= 1e-9) {
                glm::mat4 scale(1.0f);
                scale[0][0] = scale[1][1] = scale[2][2] = static_cast<float>(stage.meters_per_unit);
                lfs::core::transform(data, scale);
            }
            return data;
        }

        std::expected<void, std::string> validate_prim(const FlatPrim& prim) {
            const auto* positions = first_attribute(prim, "positions", "positionsh");
            if (!positions || positions->values.empty()) {
                return std::unexpected(std::format("USD prim {} does not contain ParticleField positions", prim.path));
            }
            if (positions->components != 3 || positions->values.size() % 3 != 0) {
                return std::unexpected("Malformed USD positions attribute");
            }
            const size_t count = positions->values.size() / 3;
            const auto* rotations = first_attribute(prim, "orientations", "orientationsh");
            const auto* scales = first_attribute(prim, "scales", "scalesh");
            const auto* opacities = first_attribute(prim, "opacities", "opacitiesh");
            if (rotations && rotations->values.size() != count * 4)
                return std::unexpected("Malformed USD orientations attribute");
            if (scales && scales->values.size() != count * 3)
                return std::unexpected("Malformed USD scales attribute");
            if (opacities && opacities->values.size() != count)
                return std::unexpected("Malformed USD opacities attribute");
            if (const auto* degree = attribute(prim, "radiance:sphericalHarmonicsDegree")) {
                if (degree->values.empty() || degree->values.front() < 0 || degree->values.front() > MAX_SUPPORTED_SH_DEGREE) {
                    return std::unexpected("Malformed USD SH degree attribute");
                }
            }
            if (const auto* sh = first_attribute(prim, "radiance:sphericalHarmonicsCoefficients", "radiance:sphericalHarmonicsCoefficientsh")) {
                if (sh->components != 3 || sh->values.size() % 3 != 0)
                    return std::unexpected("Malformed USD SH coefficient attribute");
            }
            return {};
        }

        FlatAttribute make_vec3(const std::string& type, const float* values, const size_t count) {
            return FlatAttribute{type, std::vector<float>(values, values + count * 3), 3, true};
        }

        FlatAttribute make_vec4(const std::string& type, const float* values, const size_t count) {
            return FlatAttribute{type, std::vector<float>(values, values + count * 4), 4, true};
        }

        FlatAttribute make_scalar(const std::string& type, const float* values, const size_t count) {
            return FlatAttribute{type, std::vector<float>(values, values + count), 1, true};
        }

    } // namespace

    std::expected<SplatData, std::string> load_usd(const std::filesystem::path& filepath) {
        LOG_INFO("Loading USD file: {}", lfs::core::path_to_utf8(filepath));
        auto stage = read_stage(filepath);
        if (!stage)
            return std::unexpected(flat_error_message(stage.error()));
        const auto prim = find_particlefield_prim(*stage);
        if (!prim)
            return std::unexpected(std::format("{}: {}", lfs::core::path_to_utf8(filepath), prim.error()));
        auto data = load_prim(std::move(*stage), (*prim)->path, true);
        if (!data)
            return std::unexpected(data.error());
        return std::move(*data);
    }

    lfs::Result<UsdProjectUnitLoad> load_usd_project_units(const std::filesystem::path& filepath) {
        auto stage = read_stage(filepath);
        if (!stage)
            return usd_project_error(flat_error_message(stage.error()));
        const auto prim = find_particlefield_prim(*stage);
        if (!prim)
            return usd_project_error(std::format("{}: {}", lfs::core::path_to_utf8(filepath), prim.error()));
        const auto prim_path = (*prim)->path;
        const double meters_per_unit = stage->meters_per_unit;
        auto data = load_prim(std::move(*stage), prim_path, false);
        if (!data)
            return usd_project_error(data.error());
        if (!std::isfinite(meters_per_unit) || meters_per_unit <= 0.0) {
            return usd_project_error(std::format("USD stage has invalid metersPerUnit {}", meters_per_unit));
        }
        return UsdProjectUnitLoad{.data = std::move(*data), .meters_per_unit = meters_per_unit};
    }

    std::expected<void, std::string> validate_usd(const std::filesystem::path& filepath) {
        auto stage = read_stage(filepath);
        if (!stage)
            return std::unexpected(flat_error_message(stage.error()));
        const auto prim = find_particlefield_prim(*stage);
        if (!prim)
            return std::unexpected(std::format("{}: {}", lfs::core::path_to_utf8(filepath), prim.error()));
        auto result = validate_prim(**prim);
        if (!result)
            return std::unexpected(result.error());
        return {};
    }

    Result<void> save_usd(const SplatData& splat_data, const UsdSaveOptions& options_in) {
        UsdSaveOptions options = options_in;
        if (!options.provenance)
            options.provenance = core::make_minimal_provenance_stamp();
        if (!report_export_progress(options.progress_callback, 0.0f, "Preparing USD"))
            return make_error(ErrorCode::CANCELLED, "USD export cancelled", options.output_path);
        if (splat_data.size() == 0)
            return make_error(ErrorCode::EMPTY_DATASET, "No splats to write", options.output_path);
        if (auto writable = verify_writable(options.output_path); !writable)
            return std::unexpected(writable.error());
        const auto extension = normalized_extension(options.output_path);
        if (extension != ".usd" && extension != ".usda" && extension != ".usdc")
            return make_error(ErrorCode::UNSUPPORTED_FORMAT, "USD export supports .usd, .usda, and .usdc", options.output_path);
        if (!report_export_progress(options.progress_callback, 0.15f, "Preparing USD attributes"))
            return make_error(ErrorCode::CANCELLED, "USD export cancelled", options.output_path);

        const auto means = splat_data.means().contiguous().to(Device::CPU);
        const auto scaling = splat_data.scaling_raw().contiguous().to(Device::CPU);
        const auto rotation = splat_data.rotation_raw().contiguous().to(Device::CPU);
        const auto opacity = splat_data.opacity_raw().contiguous().to(Device::CPU);
        const auto* means_ptr = static_cast<const float*>(means.data_ptr());
        const auto* scaling_ptr = static_cast<const float*>(scaling.data_ptr());
        const auto* rotation_ptr = static_cast<const float*>(rotation.data_ptr());
        const auto* opacity_ptr = static_cast<const float*>(opacity.data_ptr());
        const size_t count = splat_data.size();
        std::vector<float> scales(count * 3);
        for (size_t index = 0; index < scales.size(); ++index)
            scales[index] = std::max(std::exp(scaling_ptr[index]), MIN_SCALE);
        std::vector<float> opacities(count);
        for (size_t index = 0; index < count; ++index) {
            opacities[index] = std::clamp(1.0f / (1.0f + std::exp(-opacity_ptr[index])), OPACITY_EPSILON, 1.0f - OPACITY_EPSILON);
        }
        const int degree = splat_data.get_max_sh_degree();
        const size_t sh_count = static_cast<size_t>((degree + 1) * (degree + 1));
        const size_t rest = sh_count - 1;
        std::vector<float> coefficients(count * sh_count * 3, 0.0f);
        {
            const auto sh0 = splat_data.sh0().contiguous().to(Device::CPU);
            const auto* sh0_ptr = static_cast<const float*>(sh0.data_ptr());
            Tensor shN;
            const float* shN_ptr = nullptr;
            if (rest > 0 && splat_data.shN().is_valid() && splat_data.shN().numel() > 0) {
                shN = splat_data.shN_canonical_cpu().contiguous();
                shN_ptr = static_cast<const float*>(shN.data_ptr());
            }
            for (size_t index = 0; index < count; ++index) {
                std::copy_n(sh0_ptr + index * 3, 3, coefficients.data() + index * sh_count * 3);
                if (shN_ptr && rest > 0)
                    std::copy_n(shN_ptr + index * rest * 3, rest * 3, coefficients.data() + index * sh_count * 3 + 3);
            }
        }
        FlatStage stage;
        stage.default_prim = "/GaussianSplats";
        stage.up_axis = "Y";
        stage.meters_per_unit = 1.0;
        if (options.provenance)
            stage.custom_layer_data["lichtfeld_provenance"] = core::provenance_to_json(*options.provenance);
        FlatPrim prim{"/GaussianSplats", "ParticleField3DGaussianSplat", {}, usd_flat::identity_matrix()};
        prim.attributes.emplace("positions", make_vec3("point3f[]", means_ptr, count));
        prim.attributes.emplace("orientations", make_vec4("quatf[]", rotation_ptr, count));
        prim.attributes.emplace("scales", make_vec3("float3[]", scales.data(), count));
        prim.attributes.emplace("opacities", make_scalar("float[]", opacities.data(), count));
        prim.attributes.emplace("radiance:sphericalHarmonicsDegree", FlatAttribute{"int", {static_cast<float>(degree)}, 1, true});
        prim.attributes.emplace("radiance:sphericalHarmonicsCoefficients", make_vec3("float3[]", coefficients.data(), count * sh_count));
        std::vector<float> extent(6, 0.0f);
        std::fill(extent.begin(), extent.begin() + 3, EXTENT_LIMIT);
        std::fill(extent.begin() + 3, extent.end(), -EXTENT_LIMIT);
        for (size_t index = 0; index < count; ++index)
            for (int axis = 0; axis < 3; ++axis) {
                extent[static_cast<size_t>(axis)] = std::min(extent[static_cast<size_t>(axis)], means_ptr[index * 3 + static_cast<size_t>(axis)]);
                extent[static_cast<size_t>(axis + 3)] = std::max(extent[static_cast<size_t>(axis + 3)], means_ptr[index * 3 + static_cast<size_t>(axis)]);
            }
        for (auto& value : extent)
            value = std::clamp(value, -EXTENT_LIMIT, EXTENT_LIMIT);
        prim.attributes.emplace("extent", FlatAttribute{"float3[]", std::move(extent), 3, true});
        stage.prims.push_back(std::move(prim));
        if (!report_export_progress(options.progress_callback, 0.55f, "Authoring USD stage"))
            return make_error(ErrorCode::CANCELLED, "USD export cancelled", options.output_path);

        ScopedAtomicOutputFile atomic_output(options.output_path, AtomicOutputTempName::PreserveExtension);
        const auto write_result = extension == ".usda" ? usd_flat::write_usda(stage, atomic_output.temp_path()) : usd_flat::write_usdc(stage, atomic_output.temp_path());
        if (!write_result)
            return make_error(ErrorCode::WRITE_FAILURE, flat_error_message(write_result.error()), options.output_path);
        if (!report_export_progress(options.progress_callback, 1.0f, "USD export complete"))
            return make_error(ErrorCode::CANCELLED, "USD export cancelled", options.output_path);
        if (auto commit_result = atomic_output.commit(); !commit_result)
            return std::unexpected(commit_result.error());
        return {};
    }

} // namespace lfs::io

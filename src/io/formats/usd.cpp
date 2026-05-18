/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "usd.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "core/splat_data_transform.hpp"
#include "core/tensor.hpp"
#include "io/atomic_output.hpp"
#include "io/exporter.hpp"
#include <algorithm>
#include <cmath>
#include <format>
#include <fstream>
#include <glm/glm.hpp>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include <pxr/base/gf/half.h>
#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/gf/quatf.h>
#include <pxr/base/gf/quath.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/gf/vec3h.h>
// pxr/base/tf/hashset.h pulls in the deprecated <ext/hash_set> on GCC.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcpp"
#endif
#include <pxr/base/plug/registry.h>
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
#include <pxr/base/tf/diagnostic.h>
#include <pxr/base/tf/diagnosticMgr.h>
#include <pxr/base/tf/errorMark.h>
#include <pxr/base/vt/array.h>
#include <pxr/base/vt/value.h>
#include <pxr/usd/sdf/layer.h>
#include <pxr/usd/sdf/path.h>
#include <pxr/usd/usd/attribute.h>
#include <pxr/usd/usd/prim.h>
#include <pxr/usd/usd/primRange.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdGeom/boundable.h>
#include <pxr/usd/usdGeom/metrics.h>
#include <pxr/usd/usdGeom/xformCache.h>
#include <pxr/usd/usdVol/particleField.h>
#include <pxr/usd/usdVol/particleField3DGaussianSplat.h>
#include <pxr/usd/usdVol/particleFieldOpacityAttributeAPI.h>
#include <pxr/usd/usdVol/particleFieldOrientationAttributeAPI.h>
#include <pxr/usd/usdVol/particleFieldPositionAttributeAPI.h>
#include <pxr/usd/usdVol/particleFieldScaleAttributeAPI.h>
#include <pxr/usd/usdVol/particleFieldSphericalHarmonicsAttributeAPI.h>

namespace lfs::io {

    using lfs::core::DataType;
    using lfs::core::Device;
    using lfs::core::SplatData;
    using lfs::core::Tensor;

    namespace {
        constexpr int MAX_SUPPORTED_SH_DEGREE = 3;
        constexpr float SCENE_SCALE = 0.5f;
        constexpr float MIN_SCALE = 1e-12f;
        constexpr float OPACITY_EPSILON = 1e-6f;
        constexpr float EXTENT_LIMIT = 50000.0f;

        template <typename Vec3T>
        std::vector<float> flatten_vec3_array(const pxr::VtArray<Vec3T>& values) {
            std::vector<float> flattened;
            flattened.reserve(values.size() * 3);
            for (const auto& value : values) {
                flattened.push_back(static_cast<float>(value[0]));
                flattened.push_back(static_cast<float>(value[1]));
                flattened.push_back(static_cast<float>(value[2]));
            }
            return flattened;
        }

        template <typename QuatT>
        std::vector<float> flatten_quaternion_array(const pxr::VtArray<QuatT>& values) {
            std::vector<float> flattened;
            flattened.reserve(values.size() * 4);
            for (const auto& value : values) {
                const auto imaginary = value.GetImaginary();
                flattened.push_back(static_cast<float>(value.GetReal()));
                flattened.push_back(static_cast<float>(imaginary[0]));
                flattened.push_back(static_cast<float>(imaginary[1]));
                flattened.push_back(static_cast<float>(imaginary[2]));
            }
            return flattened;
        }

        template <typename ScalarT>
        std::vector<float> flatten_scalar_array(const pxr::VtArray<ScalarT>& values) {
            std::vector<float> flattened;
            flattened.reserve(values.size());
            for (const auto& value : values) {
                flattened.push_back(static_cast<float>(value));
            }
            return flattened;
        }

        std::optional<std::vector<float>> read_vec3_array(
            const pxr::UsdAttribute& float_attr,
            const pxr::UsdAttribute& half_attr) {

            pxr::VtArray<pxr::GfVec3f> float_values;
            if (float_attr && float_attr.Get(&float_values) && !float_values.empty()) {
                return flatten_vec3_array(float_values);
            }

            pxr::VtArray<pxr::GfVec3h> half_values;
            if (half_attr && half_attr.Get(&half_values) && !half_values.empty()) {
                return flatten_vec3_array(half_values);
            }

            return std::nullopt;
        }

        std::optional<std::vector<float>> read_quaternion_array(
            const pxr::UsdAttribute& float_attr,
            const pxr::UsdAttribute& half_attr) {

            pxr::VtArray<pxr::GfQuatf> float_values;
            if (float_attr && float_attr.Get(&float_values) && !float_values.empty()) {
                return flatten_quaternion_array(float_values);
            }

            pxr::VtArray<pxr::GfQuath> half_values;
            if (half_attr && half_attr.Get(&half_values) && !half_values.empty()) {
                return flatten_quaternion_array(half_values);
            }

            return std::nullopt;
        }

        std::optional<std::vector<float>> read_scalar_array(
            const pxr::UsdAttribute& float_attr,
            const pxr::UsdAttribute& half_attr) {

            pxr::VtArray<float> float_values;
            if (float_attr && float_attr.Get(&float_values) && !float_values.empty()) {
                return flatten_scalar_array(float_values);
            }

            pxr::VtArray<pxr::GfHalf> half_values;
            if (half_attr && half_attr.Get(&half_values) && !half_values.empty()) {
                return flatten_scalar_array(half_values);
            }

            return std::nullopt;
        }

        std::vector<pxr::UsdPrim> collect_particlefield_prims(const pxr::UsdPrim& root) {
            std::vector<pxr::UsdPrim> prims;
            for (const auto& prim : pxr::UsdPrimRange(root)) {
                if (prim.IsA<pxr::UsdVolParticleField>()) {
                    prims.push_back(prim);
                }
            }
            return prims;
        }

        std::string collect_particlefield_paths(const std::vector<pxr::UsdPrim>& prims) {
            if (prims.empty()) {
                return "none";
            }

            std::string result;
            bool first = true;
            for (const auto& prim : prims) {
                if (!first) {
                    result += ", ";
                }
                result += prim.GetPath().GetString();
                first = false;
            }
            return result;
        }

        std::string collect_stage_prim_types(const pxr::UsdStageRefPtr& stage);

        std::expected<pxr::UsdPrim, std::string> find_particlefield_prim(const pxr::UsdStageRefPtr& stage) {
            const pxr::UsdPrim default_prim = stage->GetDefaultPrim();
            std::vector<pxr::UsdPrim> default_candidates;
            if (default_prim) {
                if (default_prim.IsA<pxr::UsdVolParticleField>()) {
                    return default_prim;
                }

                default_candidates = collect_particlefield_prims(default_prim);
                if (default_candidates.size() == 1) {
                    return default_candidates.front();
                }
                if (default_candidates.size() > 1) {
                    return std::unexpected(std::format(
                        "Default prim {} contains multiple OpenUSD ParticleField prims: {}",
                        default_prim.GetPath().GetString(),
                        collect_particlefield_paths(default_candidates)));
                }
            }

            const auto stage_candidates = collect_particlefield_prims(stage->GetPseudoRoot());
            if (stage_candidates.empty()) {
                return std::unexpected(std::format(
                    "No OpenUSD ParticleField prim found in stage. Prim types in stage: {}",
                    collect_stage_prim_types(stage)));
            }

            if (stage_candidates.size() > 1) {
                if (default_prim) {
                    return std::unexpected(std::format(
                        "Default prim {} does not resolve to a unique OpenUSD ParticleField. "
                        "Candidate ParticleField prims in stage: {}",
                        default_prim.GetPath().GetString(),
                        collect_particlefield_paths(stage_candidates)));
                }

                return std::unexpected(std::format(
                    "Stage contains multiple OpenUSD ParticleField prims and no default prim. "
                    "Candidate ParticleField prims: {}",
                    collect_particlefield_paths(stage_candidates)));
            }

            return stage_candidates.front();
        }

        std::string collect_stage_prim_types(const pxr::UsdStageRefPtr& stage) {
            std::set<std::string> prim_types;
            for (const auto& prim : pxr::UsdPrimRange(stage->GetPseudoRoot())) {
                const auto type_name = prim.GetTypeName();
                if (!type_name.IsEmpty()) {
                    prim_types.insert(type_name.GetString());
                }
            }

            if (prim_types.empty()) {
                return "none";
            }

            std::string result;
            bool first = true;
            for (const auto& type_name : prim_types) {
                if (!first) {
                    result += ", ";
                }
                result += type_name;
                first = false;
            }
            return result;
        }

        int infer_sh_degree(const size_t particle_count, const size_t coefficient_count) {
            if (particle_count == 0 || coefficient_count == 0 || coefficient_count % particle_count != 0) {
                return 0;
            }

            const size_t coeffs_per_particle = coefficient_count / particle_count;
            const auto degree_plus_one = static_cast<int>(std::llround(std::sqrt(static_cast<double>(coeffs_per_particle))));
            if (degree_plus_one <= 0) {
                return 0;
            }

            const int inferred = degree_plus_one - 1;
            if (static_cast<size_t>((inferred + 1) * (inferred + 1)) != coeffs_per_particle) {
                return 0;
            }
            return inferred;
        }

        glm::mat4 to_glm_matrix(const pxr::GfMatrix4d& matrix) {
            glm::mat4 result(1.0f);
            for (int row = 0; row < 4; ++row) {
                for (int col = 0; col < 4; ++col) {
                    result[col][row] = static_cast<float>(matrix[row][col]);
                }
            }
            return result;
        }

        void maybe_apply_world_transform(const pxr::UsdPrim& prim, SplatData& splat_data) {
            pxr::UsdGeomXformCache xform_cache;
            const auto world_matrix = xform_cache.GetLocalToWorldTransform(prim);
            if (world_matrix == pxr::GfMatrix4d(1.0)) {
                return;
            }

            LOG_INFO("Applying composed USD transform for prim {}", prim.GetPath().GetString());
            lfs::core::transform(splat_data, to_glm_matrix(world_matrix));
        }

        void maybe_apply_stage_linear_units(const pxr::UsdStageRefPtr& stage, SplatData& splat_data) {
            const double meters_per_unit = pxr::UsdGeomGetStageMetersPerUnit(stage);
            if (std::abs(meters_per_unit - 1.0) < 1e-9) {
                return;
            }

            glm::mat4 scale_matrix(1.0f);
            const float scale = static_cast<float>(meters_per_unit);
            scale_matrix[0][0] = scale;
            scale_matrix[1][1] = scale;
            scale_matrix[2][2] = scale;

            LOG_INFO("Applying USD stage metersPerUnit {} as a uniform import scale", meters_per_unit);
            lfs::core::transform(splat_data, scale_matrix);
        }

        std::vector<float> make_identity_quaternions(const size_t count) {
            std::vector<float> rotations(count * 4, 0.0f);
            for (size_t i = 0; i < count; ++i) {
                rotations[i * 4] = 1.0f;
            }
            return rotations;
        }

        std::vector<float> make_default_scales(const size_t count) {
            return std::vector<float>(count * 3, 1.0f);
        }

        std::vector<float> make_default_opacities(const size_t count) {
            return std::vector<float>(count, 1.0f);
        }

        std::vector<float> make_default_sh0(const size_t count) {
            // Raw SH DC in LichtFeld is converted to RGB as 0.5 + SH_C0 * sh0,
            // so neutral gray corresponds to a zero coefficient.
            return std::vector<float>(count * 3, 0.0f);
        }

        [[nodiscard]] float sigmoid_clamped(const float value) {
            const float opacity = 1.0f / (1.0f + std::exp(-value));
            return std::clamp(opacity, OPACITY_EPSILON, 1.0f - OPACITY_EPSILON);
        }

        [[nodiscard]] std::vector<float> flatten_sh_coefficients(const SplatData& splat_data) {
            const size_t num_gaussians = splat_data.size();
            const int sh_degree = splat_data.get_max_sh_degree();
            const size_t coeffs_per_gaussian = static_cast<size_t>((sh_degree + 1) * (sh_degree + 1));
            const size_t rest_coeffs = coeffs_per_gaussian > 0 ? coeffs_per_gaussian - 1 : 0;

            const auto sh0 = splat_data.sh0().contiguous().to(Device::CPU);
            const auto* const sh0_ptr = static_cast<const float*>(sh0.data_ptr());

            Tensor shN;
            const float* shN_ptr = nullptr;
            if (rest_coeffs > 0 && splat_data.shN().is_valid()) {
                shN = splat_data.shN().contiguous().to(Device::CPU);
                shN_ptr = static_cast<const float*>(shN.data_ptr());
            }

            std::vector<float> coefficients(num_gaussians * coeffs_per_gaussian * 3, 0.0f);
            for (size_t index = 0; index < num_gaussians; ++index) {
                const size_t dst_offset = index * coeffs_per_gaussian * 3;
                std::copy_n(sh0_ptr + static_cast<std::ptrdiff_t>(index * 3),
                            3,
                            coefficients.begin() + static_cast<std::ptrdiff_t>(dst_offset));

                if (rest_coeffs > 0 && shN_ptr) {
                    std::copy_n(shN_ptr + static_cast<std::ptrdiff_t>(index * rest_coeffs * 3),
                                rest_coeffs * 3,
                                coefficients.begin() + static_cast<std::ptrdiff_t>(dst_offset + 3));
                }
            }

            return coefficients;
        }

        [[nodiscard]] pxr::VtArray<pxr::GfVec3f> make_vec3_array(const float* values, const size_t count) {
            pxr::VtArray<pxr::GfVec3f> result;
            result.resize(count);
            for (size_t index = 0; index < count; ++index) {
                result[index] = pxr::GfVec3f(values[index * 3 + 0],
                                             values[index * 3 + 1],
                                             values[index * 3 + 2]);
            }
            return result;
        }

        [[nodiscard]] pxr::VtArray<pxr::GfQuatf> make_quat_array(const float* values, const size_t count) {
            pxr::VtArray<pxr::GfQuatf> result;
            result.resize(count);
            for (size_t index = 0; index < count; ++index) {
                result[index] = pxr::GfQuatf(values[index * 4 + 0],
                                             pxr::GfVec3f(values[index * 4 + 1],
                                                          values[index * 4 + 2],
                                                          values[index * 4 + 3]));
            }
            return result;
        }

        [[nodiscard]] pxr::VtArray<float> make_scalar_array(const float* values, const size_t count) {
            pxr::VtArray<float> result;
            result.resize(count);
            std::copy_n(values, static_cast<std::ptrdiff_t>(count), result.begin());
            return result;
        }

        [[nodiscard]] pxr::VtArray<pxr::GfVec3f> make_extent_array(const float* positions, const size_t count) {
            pxr::GfVec3f extent_min(EXTENT_LIMIT, EXTENT_LIMIT, EXTENT_LIMIT);
            pxr::GfVec3f extent_max(-EXTENT_LIMIT, -EXTENT_LIMIT, -EXTENT_LIMIT);

            for (size_t index = 0; index < count; ++index) {
                const pxr::GfVec3f position(
                    positions[index * 3 + 0],
                    positions[index * 3 + 1],
                    positions[index * 3 + 2]);

                extent_min[0] = std::min(extent_min[0], position[0]);
                extent_min[1] = std::min(extent_min[1], position[1]);
                extent_min[2] = std::min(extent_min[2], position[2]);
                extent_max[0] = std::max(extent_max[0], position[0]);
                extent_max[1] = std::max(extent_max[1], position[1]);
                extent_max[2] = std::max(extent_max[2], position[2]);
            }

            for (int axis = 0; axis < 3; ++axis) {
                extent_min[axis] = std::clamp(extent_min[axis], -EXTENT_LIMIT, EXTENT_LIMIT);
                extent_max[axis] = std::clamp(extent_max[axis], -EXTENT_LIMIT, EXTENT_LIMIT);
            }

            pxr::VtArray<pxr::GfVec3f> extent;
            extent.resize(2);
            extent[0] = extent_min;
            extent[1] = extent_max;
            return extent;
        }

        [[nodiscard]] std::string normalized_extension(const std::filesystem::path& path) {
            auto extension = path.extension().string();
            std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
            return extension;
        }

        std::expected<SplatData, std::string> load_particlefield_prim(const pxr::UsdPrim& prim) {
            std::optional<std::vector<float>> positions;
            std::optional<std::vector<float>> rotations;
            std::optional<std::vector<float>> scales;
            std::optional<std::vector<float>> opacities;
            std::optional<std::vector<float>> sh_coeffs;
            pxr::UsdAttribute sh_degree_attr;

            if (prim.IsA<pxr::UsdVolParticleField3DGaussianSplat>()) {
                const pxr::UsdVolParticleField3DGaussianSplat schema(prim);
                positions = read_vec3_array(schema.GetPositionsAttr(), schema.GetPositionshAttr());
                rotations = read_quaternion_array(schema.GetOrientationsAttr(), schema.GetOrientationshAttr());
                scales = read_vec3_array(schema.GetScalesAttr(), schema.GetScaleshAttr());
                opacities = read_scalar_array(schema.GetOpacitiesAttr(), schema.GetOpacitieshAttr());
                sh_coeffs = read_vec3_array(
                    schema.GetRadianceSphericalHarmonicsCoefficientsAttr(),
                    schema.GetRadianceSphericalHarmonicsCoefficientshAttr());
                sh_degree_attr = schema.GetRadianceSphericalHarmonicsDegreeAttr();
            } else {
                const pxr::UsdVolParticleFieldPositionAttributeAPI position_api(prim);
                const pxr::UsdVolParticleFieldOrientationAttributeAPI orientation_api(prim);
                const pxr::UsdVolParticleFieldScaleAttributeAPI scale_api(prim);
                const pxr::UsdVolParticleFieldOpacityAttributeAPI opacity_api(prim);
                const pxr::UsdVolParticleFieldSphericalHarmonicsAttributeAPI sh_api(prim);

                positions = read_vec3_array(position_api.GetPositionsAttr(), position_api.GetPositionshAttr());
                rotations = read_quaternion_array(
                    orientation_api.GetOrientationsAttr(),
                    orientation_api.GetOrientationshAttr());
                scales = read_vec3_array(scale_api.GetScalesAttr(), scale_api.GetScaleshAttr());
                opacities = read_scalar_array(opacity_api.GetOpacitiesAttr(), opacity_api.GetOpacitieshAttr());
                sh_coeffs = read_vec3_array(
                    sh_api.GetRadianceSphericalHarmonicsCoefficientsAttr(),
                    sh_api.GetRadianceSphericalHarmonicsCoefficientshAttr());
                sh_degree_attr = sh_api.GetRadianceSphericalHarmonicsDegreeAttr();
            }

            if (!positions || positions->empty()) {
                return std::unexpected(std::format(
                    "USD prim {} does not contain ParticleField positions",
                    prim.GetPath().GetString()));
            }

            if (positions->size() % 3 != 0) {
                return std::unexpected("Malformed USD positions attribute");
            }

            const size_t num_gaussians = positions->size() / 3;

            std::vector<float> rotation_values = rotations.value_or(make_identity_quaternions(num_gaussians));
            if (!rotations) {
                LOG_WARN("USD file omitted orientations; defaulting to identity quaternions");
            }

            std::vector<float> scale_values = scales.value_or(make_default_scales(num_gaussians));
            if (!scales) {
                LOG_WARN("USD file omitted scales; defaulting to unit scales");
            }

            std::vector<float> opacity_values = opacities.value_or(make_default_opacities(num_gaussians));
            if (!opacities) {
                LOG_WARN("USD file omitted opacities; defaulting to fully opaque");
            }

            int sh_degree = 0;
            const bool has_authored_sh_degree = sh_degree_attr && sh_degree_attr.HasAuthoredValueOpinion();
            if (has_authored_sh_degree) {
                int authored_degree = 0;
                if (!sh_degree_attr.Get(&authored_degree)) {
                    return std::unexpected("Malformed USD SH degree attribute");
                }
                sh_degree = authored_degree;
            }

            if (rotation_values.size() != num_gaussians * 4) {
                return std::unexpected("Malformed USD orientations attribute");
            }
            if (scale_values.size() != num_gaussians * 3) {
                return std::unexpected("Malformed USD scales attribute");
            }
            if (opacity_values.size() != num_gaussians) {
                return std::unexpected("Malformed USD opacities attribute");
            }

            std::vector<float> sh_coeff_values = make_default_sh0(num_gaussians);
            if (sh_coeffs && !sh_coeffs->empty()) {
                std::vector<float> candidate_sh_coeffs = std::move(*sh_coeffs);
                if (candidate_sh_coeffs.size() % 3 != 0) {
                    return std::unexpected("Malformed USD SH coefficient attribute");
                }

                const size_t authored_coeff_triplets = candidate_sh_coeffs.size() / 3;
                int resolved_sh_degree = sh_degree;
                if (!has_authored_sh_degree) {
                    resolved_sh_degree = infer_sh_degree(num_gaussians, authored_coeff_triplets);
                }

                if (resolved_sh_degree < 0 || resolved_sh_degree > MAX_SUPPORTED_SH_DEGREE) {
                    return std::unexpected(std::format(
                        "Unsupported USD spherical harmonics degree {}. LichtFeld Studio supports degrees 0-{}.",
                        resolved_sh_degree,
                        MAX_SUPPORTED_SH_DEGREE));
                }

                const size_t num_sh_coeffs = static_cast<size_t>((resolved_sh_degree + 1) * (resolved_sh_degree + 1));
                const size_t required_coeff_triplets = num_gaussians * num_sh_coeffs;

                if (authored_coeff_triplets < required_coeff_triplets) {
                    LOG_WARN("USD SH coefficient array is too short for {} gaussians at degree {}; "
                             "ignoring it and falling back to neutral degree-0 radiance",
                             num_gaussians,
                             resolved_sh_degree);
                    sh_degree = 0;
                } else {
                    if (authored_coeff_triplets > required_coeff_triplets) {
                        LOG_WARN("USD SH coefficient array is longer than expected; truncating to {} coefficients",
                                 required_coeff_triplets);
                        candidate_sh_coeffs.resize(required_coeff_triplets * 3);
                    }
                    sh_degree = resolved_sh_degree;
                    sh_coeff_values = std::move(candidate_sh_coeffs);
                }
            } else {
                LOG_WARN("USD file omitted SH coefficients; defaulting to neutral degree-0 radiance");
            }

            const size_t num_sh_coeffs = static_cast<size_t>((sh_degree + 1) * (sh_degree + 1));

            std::vector<float> scaling_raw(scale_values.size());
            std::transform(scale_values.begin(), scale_values.end(), scaling_raw.begin(), [](const float value) {
                return std::log(std::max(value, MIN_SCALE));
            });

            std::vector<float> opacity_raw(opacity_values.size());
            std::transform(opacity_values.begin(), opacity_values.end(), opacity_raw.begin(), [](const float value) {
                const float clamped = std::clamp(value, OPACITY_EPSILON, 1.0f - OPACITY_EPSILON);
                return std::log(clamped / (1.0f - clamped));
            });

            std::vector<float> sh0_values(num_gaussians * 3, 0.0f);
            const size_t rest_coeffs = num_sh_coeffs > 0 ? num_sh_coeffs - 1 : 0;
            std::vector<float> shN_values(rest_coeffs * num_gaussians * 3, 0.0f);

            for (size_t index = 0; index < num_gaussians; ++index) {
                const size_t src_offset = index * num_sh_coeffs * 3;
                std::copy_n(sh_coeff_values.begin() + static_cast<std::ptrdiff_t>(src_offset),
                            3,
                            sh0_values.begin() + static_cast<std::ptrdiff_t>(index * 3));

                if (rest_coeffs > 0) {
                    std::copy_n(sh_coeff_values.begin() + static_cast<std::ptrdiff_t>(src_offset + 3),
                                rest_coeffs * 3,
                                shN_values.begin() + static_cast<std::ptrdiff_t>(index * rest_coeffs * 3));
                }
            }

            Tensor shN_tensor = rest_coeffs > 0
                                    ? Tensor::from_vector(shN_values, {num_gaussians, rest_coeffs, 3}, Device::CUDA)
                                    : Tensor::zeros({num_gaussians, 0, 3}, Device::CUDA, DataType::Float32);

            SplatData splat_data(
                sh_degree,
                Tensor::from_vector(*positions, {num_gaussians, 3}, Device::CUDA),
                Tensor::from_vector(sh0_values, {num_gaussians, 1, 3}, Device::CUDA),
                std::move(shN_tensor),
                Tensor::from_vector(scaling_raw, {num_gaussians, 3}, Device::CUDA),
                Tensor::from_vector(rotation_values, {num_gaussians, 4}, Device::CUDA),
                Tensor::from_vector(opacity_raw, {num_gaussians, 1}, Device::CUDA),
                SCENE_SCALE);

            maybe_apply_world_transform(prim, splat_data);
            maybe_apply_stage_linear_units(prim.GetStage(), splat_data);
            return splat_data;
        }

        std::expected<void, std::string> validate_particlefield_prim(const pxr::UsdPrim& prim) {
            std::optional<std::vector<float>> positions;
            std::optional<std::vector<float>> rotations;
            std::optional<std::vector<float>> scales;
            std::optional<std::vector<float>> opacities;
            std::optional<std::vector<float>> sh_coeffs;
            pxr::UsdAttribute sh_degree_attr;

            if (prim.IsA<pxr::UsdVolParticleField3DGaussianSplat>()) {
                const pxr::UsdVolParticleField3DGaussianSplat schema(prim);
                positions = read_vec3_array(schema.GetPositionsAttr(), schema.GetPositionshAttr());
                rotations = read_quaternion_array(schema.GetOrientationsAttr(), schema.GetOrientationshAttr());
                scales = read_vec3_array(schema.GetScalesAttr(), schema.GetScaleshAttr());
                opacities = read_scalar_array(schema.GetOpacitiesAttr(), schema.GetOpacitieshAttr());
                sh_coeffs = read_vec3_array(
                    schema.GetRadianceSphericalHarmonicsCoefficientsAttr(),
                    schema.GetRadianceSphericalHarmonicsCoefficientshAttr());
                sh_degree_attr = schema.GetRadianceSphericalHarmonicsDegreeAttr();
            } else {
                const pxr::UsdVolParticleFieldPositionAttributeAPI position_api(prim);
                const pxr::UsdVolParticleFieldOrientationAttributeAPI orientation_api(prim);
                const pxr::UsdVolParticleFieldScaleAttributeAPI scale_api(prim);
                const pxr::UsdVolParticleFieldOpacityAttributeAPI opacity_api(prim);
                const pxr::UsdVolParticleFieldSphericalHarmonicsAttributeAPI sh_api(prim);

                positions = read_vec3_array(position_api.GetPositionsAttr(), position_api.GetPositionshAttr());
                rotations = read_quaternion_array(
                    orientation_api.GetOrientationsAttr(),
                    orientation_api.GetOrientationshAttr());
                scales = read_vec3_array(scale_api.GetScalesAttr(), scale_api.GetScaleshAttr());
                opacities = read_scalar_array(opacity_api.GetOpacitiesAttr(), opacity_api.GetOpacitieshAttr());
                sh_coeffs = read_vec3_array(
                    sh_api.GetRadianceSphericalHarmonicsCoefficientsAttr(),
                    sh_api.GetRadianceSphericalHarmonicsCoefficientshAttr());
                sh_degree_attr = sh_api.GetRadianceSphericalHarmonicsDegreeAttr();
            }

            if (!positions || positions->empty()) {
                return std::unexpected(std::format(
                    "USD prim {} does not contain ParticleField positions",
                    prim.GetPath().GetString()));
            }

            if (positions->size() % 3 != 0) {
                return std::unexpected("Malformed USD positions attribute");
            }

            const size_t num_gaussians = positions->size() / 3;
            if (rotations && rotations->size() != num_gaussians * 4) {
                return std::unexpected("Malformed USD orientations attribute");
            }
            if (scales && scales->size() != num_gaussians * 3) {
                return std::unexpected("Malformed USD scales attribute");
            }
            if (opacities && opacities->size() != num_gaussians) {
                return std::unexpected("Malformed USD opacities attribute");
            }

            if (sh_degree_attr && sh_degree_attr.HasAuthoredValueOpinion()) {
                int authored_degree = 0;
                if (!sh_degree_attr.Get(&authored_degree)) {
                    return std::unexpected("Malformed USD SH degree attribute");
                }
                if (authored_degree < 0 || authored_degree > MAX_SUPPORTED_SH_DEGREE) {
                    return std::unexpected(std::format(
                        "Unsupported USD spherical harmonics degree {}. LichtFeld Studio supports degrees 0-{}.",
                        authored_degree,
                        MAX_SUPPORTED_SH_DEGREE));
                }
            }

            if (sh_coeffs && !sh_coeffs->empty() && sh_coeffs->size() % 3 != 0) {
                return std::unexpected("Malformed USD SH coefficient attribute");
            }

            return {};
        }
    } // namespace

    std::expected<SplatData, std::string> load_usd(const std::filesystem::path& filepath) {
        LOG_INFO("Loading USD file: {}", lfs::core::path_to_utf8(filepath));

        const auto stage = pxr::UsdStage::Open(lfs::core::path_to_utf8(filepath));
        if (!stage) {
            return std::unexpected(std::format(
                "Failed to open USD stage: {}",
                lfs::core::path_to_utf8(filepath)));
        }

        const auto particlefield_prim = find_particlefield_prim(stage);
        if (!particlefield_prim) {
            return std::unexpected(std::format(
                "{}: {}",
                lfs::core::path_to_utf8(filepath),
                particlefield_prim.error()));
        }

        LOG_INFO("Found USD gaussian prim: {} ({})",
                 particlefield_prim->GetPath().GetString(),
                 particlefield_prim->GetTypeName().GetString());

        return load_particlefield_prim(*particlefield_prim);
    }

    std::expected<void, std::string> validate_usd(const std::filesystem::path& filepath) {
        LOG_INFO("Validating USD file: {}", lfs::core::path_to_utf8(filepath));

        const auto stage = pxr::UsdStage::Open(lfs::core::path_to_utf8(filepath));
        if (!stage) {
            return std::unexpected(std::format(
                "Failed to open USD stage: {}",
                lfs::core::path_to_utf8(filepath)));
        }

        const auto particlefield_prim = find_particlefield_prim(stage);
        if (!particlefield_prim) {
            return std::unexpected(std::format(
                "{}: {}",
                lfs::core::path_to_utf8(filepath),
                particlefield_prim.error()));
        }

        return validate_particlefield_prim(*particlefield_prim);
    }

    Result<void> save_usd(const SplatData& splat_data, const UsdSaveOptions& options) {
        if (!report_export_progress(options.progress_callback, 0.0f, "Preparing USD")) {
            return make_error(ErrorCode::CANCELLED, "USD export cancelled", options.output_path);
        }

        if (splat_data.size() == 0) {
            return make_error(ErrorCode::EMPTY_DATASET, "No splats to write", options.output_path);
        }

        if (auto writable = verify_writable(options.output_path); !writable) {
            return std::unexpected(writable.error());
        }

        const std::string extension = normalized_extension(options.output_path);
        if (extension != ".usd" && extension != ".usda" && extension != ".usdc") {
            return make_error(ErrorCode::UNSUPPORTED_FORMAT,
                              "USD export supports .usd, .usda, and .usdc",
                              options.output_path);
        }

        LOG_INFO("Saving USD file: {}", lfs::core::path_to_utf8(options.output_path));

        // Pre-flight check: ArResolver requires USD plugins to be registered.
        // If none are registered, the USD plugin path may not be configured.
        const auto plugin_count = pxr::PlugRegistry::GetInstance().GetAllPlugins().size();
        if (plugin_count == 0) {
            LOG_ERROR("[USD] No USD plugins registered — UsdStage::CreateNew will crash fatally");
            return make_error(ErrorCode::WRITE_FAILURE,
                              "USD plugins not registered — ensure the USD plugin path is configured before calling save_usd",
                              options.output_path);
        }

        if (!report_export_progress(options.progress_callback, 0.15f, "Preparing USD attributes")) {
            return make_error(ErrorCode::CANCELLED, "USD export cancelled", options.output_path);
        }

        const auto means = splat_data.means().contiguous().to(Device::CPU);
        const auto scaling = splat_data.scaling_raw().contiguous().to(Device::CPU);
        const auto rotation = splat_data.rotation_raw().contiguous().to(Device::CPU);
        const auto opacity = splat_data.opacity_raw().contiguous().to(Device::CPU);

        const auto* const means_ptr = static_cast<const float*>(means.data_ptr());
        const auto* const scaling_ptr = static_cast<const float*>(scaling.data_ptr());
        const auto* const rotation_ptr = static_cast<const float*>(rotation.data_ptr());
        const auto* const opacity_ptr = static_cast<const float*>(opacity.data_ptr());

        const size_t num_gaussians = splat_data.size();

        std::vector<float> scales_linear(num_gaussians * 3, 1.0f);
        for (size_t index = 0; index < num_gaussians * 3; ++index) {
            scales_linear[index] = std::max(std::exp(scaling_ptr[index]), MIN_SCALE);
        }

        std::vector<float> opacity_linear(num_gaussians, 1.0f);
        for (size_t index = 0; index < num_gaussians; ++index) {
            opacity_linear[index] = sigmoid_clamped(opacity_ptr[index]);
        }

        const auto sh_coefficients = flatten_sh_coefficients(splat_data);

        if (!report_export_progress(options.progress_callback, 0.55f, "Authoring USD stage")) {
            return make_error(ErrorCode::CANCELLED, "USD export cancelled", options.output_path);
        }

        ScopedAtomicOutputFile atomic_output(options.output_path, AtomicOutputTempName::PreserveExtension);

        // TfErrorMark captures OpenUSD diagnostic errors that don't throw.
        pxr::TfErrorMark error_mark;

        pxr::UsdStageRefPtr stage;
        try {
            stage = pxr::UsdStage::CreateNew(lfs::core::path_to_utf8(atomic_output.temp_path()));
        } catch (const std::exception& e) {
            return make_error(ErrorCode::WRITE_FAILURE,
                              std::format("UsdStage::CreateNew threw: {}", e.what()),
                              options.output_path);
        }
        if (!error_mark.IsClean()) {
            for (const auto& err : error_mark) {
                LOG_ERROR("OpenUSD error during stage creation: {} ({}:{})",
                          err.GetCommentary(), err.GetSourceFileName(), err.GetSourceLineNumber());
            }
            error_mark.Clear();
        }
        if (!stage) {
            return make_error(ErrorCode::WRITE_FAILURE,
                              "Failed to create USD stage",
                              options.output_path);
        }

        const pxr::SdfPath prim_path("/GaussianSplats");
        pxr::UsdVolParticleField3DGaussianSplat splat_prim;
        try {
            splat_prim = pxr::UsdVolParticleField3DGaussianSplat::Define(stage, prim_path);
        } catch (const std::exception& e) {
            return make_error(ErrorCode::WRITE_FAILURE,
                              std::format("ParticleField3DGaussianSplat::Define threw: {}", e.what()),
                              options.output_path);
        }
        if (!splat_prim) {
            return make_error(ErrorCode::WRITE_FAILURE,
                              "Failed to define USD gaussian prim",
                              options.output_path);
        }

        try {
            stage->SetDefaultPrim(splat_prim.GetPrim());
            pxr::UsdGeomSetStageUpAxis(stage, pxr::UsdGeomTokens->y);
            pxr::UsdGeomSetStageMetersPerUnit(stage, 1.0);
        } catch (const std::exception& e) {
            return make_error(ErrorCode::WRITE_FAILURE,
                              std::format("Stage metadata threw: {}", e.what()),
                              options.output_path);
        }

        pxr::UsdGeomBoundable boundable(splat_prim.GetPrim());

        try {
            if (!splat_prim.CreatePositionsAttr().Set(make_vec3_array(means_ptr, num_gaussians)) ||
                !splat_prim.CreateOrientationsAttr().Set(make_quat_array(rotation_ptr, num_gaussians)) ||
                !splat_prim.CreateScalesAttr().Set(make_vec3_array(scales_linear.data(), num_gaussians)) ||
                !splat_prim.CreateOpacitiesAttr().Set(make_scalar_array(opacity_linear.data(), num_gaussians)) ||
                !splat_prim.CreateRadianceSphericalHarmonicsDegreeAttr().Set(splat_data.get_max_sh_degree()) ||
                !splat_prim.CreateRadianceSphericalHarmonicsCoefficientsAttr().Set(
                    make_vec3_array(sh_coefficients.data(),
                                    num_gaussians * static_cast<size_t>((splat_data.get_max_sh_degree() + 1) *
                                                                        (splat_data.get_max_sh_degree() + 1)))) ||
                !boundable.CreateExtentAttr().Set(make_extent_array(means_ptr, num_gaussians))) {
                return make_error(ErrorCode::WRITE_FAILURE,
                                  "Failed to author USD gaussian attributes",
                                  options.output_path);
            }
        } catch (const std::exception& e) {
            return make_error(ErrorCode::WRITE_FAILURE,
                              std::format("Authoring attributes threw: {}", e.what()),
                              options.output_path);
        }

        if (!report_export_progress(options.progress_callback, 0.9f, "Saving USD")) {
            return make_error(ErrorCode::CANCELLED, "USD export cancelled", options.output_path);
        }

        try {
            if (const auto root_layer = stage->GetRootLayer(); !root_layer || !root_layer->Save()) {
                return make_error(ErrorCode::WRITE_FAILURE,
                                  "Failed to save USD stage",
                                  options.output_path);
            }
        } catch (const std::exception& e) {
            return make_error(ErrorCode::WRITE_FAILURE,
                              std::format("Layer save threw: {}", e.what()),
                              options.output_path);
        }

        boundable = pxr::UsdGeomBoundable();
        splat_prim = pxr::UsdVolParticleField3DGaussianSplat();
        stage = pxr::UsdStageRefPtr();

        if (!report_export_progress(options.progress_callback, 1.0f, "USD export complete")) {
            return make_error(ErrorCode::CANCELLED, "USD export cancelled", options.output_path);
        }

        if (auto commit_result = atomic_output.commit(); !commit_result) {
            return std::unexpected(commit_result.error());
        }

        return {};
    }

} // namespace lfs::io

/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/splat_data_transform.hpp"
#include "core/assert.hpp"
#include "core/cuda/sh_layout.cuh"
#include "core/logger.hpp"
#include "core/point_cloud.hpp"
#include "core/sh_value_quant.hpp"
#include "core/splat_data.hpp"
#include "core/tensor_backend.hpp"
#include "core/tensor_fused.hpp"
#include "core/tensor_sh.hpp"
#include "core/tensor_splat.hpp"
#include "geometry/bounding_box.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <vector>

namespace lfs::core {

    namespace {

        // Bounds between the 1st and 99th percentile of every axis of [N, 3] positions. Host positions use selection
        // instead of a full sort: the same order statistics without the seconds a sort of millions of points takes.
        void percentile_bounds(const Tensor& means, glm::vec3& min_bounds, glm::vec3& max_bounds, const float padding) {
            LFS_ASSERT(means.ndim() == 2 && means.size(1) == 3 && means.dtype() == DataType::Float32);
            const int64_t n = means.size(0);
            const int64_t lo = n / 100;
            const int64_t hi = n - 1 - lo;
            if (means.device() == Device::CUDA) {
                for (int i = 0; i < 3; ++i) {
                    const auto sorted = means.slice(1, i, i + 1).squeeze(1).sort(0, false).first;
                    min_bounds[i] = sorted[lo].item() - padding;
                    max_bounds[i] = sorted[hi].item() + padding;
                }
                return;
            }
            const auto host = means.contiguous();
            const float* const data = host.ptr<float>();
            // NaN orders last so the comparison stays a strict weak ordering.
            const auto less = [](const float a, const float b) { return a < b || (!std::isnan(a) && std::isnan(b)); };
            std::array<std::vector<float>, 3> columns;
            for (auto& column : columns)
                column.resize(static_cast<size_t>(n));
#pragma omp parallel for num_threads(3) if (n > 100000)
            for (int axis = 0; axis < 3; ++axis) {
                auto& column = columns[static_cast<size_t>(axis)];
                for (int64_t row = 0; row < n; ++row)
                    column[static_cast<size_t>(row)] = data[row * 3 + axis];
                const auto lower = column.begin() + lo;
                const auto upper = column.begin() + hi;
                std::nth_element(column.begin(), lower, column.end(), less);
                std::nth_element(lower + 1, upper, column.end(), less);
                min_bounds[axis] = *lower - padding;
                max_bounds[axis] = *upper + padding;
            }
        }

        constexpr double SH_C1 = 0.48860251190291987;
        constexpr double SH_C2_0 = 1.0925484305920792;
        constexpr double SH_C2_2 = 0.31539156525251999;
        constexpr double SH_C2_3 = 0.54627421529603959;

        constexpr double SH_C3_0 = 0.59004358992664352;
        constexpr double SH_C3_1 = 2.8906114426405538;
        constexpr double SH_C3_2 = 0.45704579946446572;
        constexpr double SH_C3_3 = 0.3731763325901154;
        constexpr double SH_C3_4 = 1.4453057213202769;

        constexpr double SH_SOLVE_EPS = 1e-12;
        constexpr float ROTATION_EPS = 1e-6f;
        constexpr int SH_FIT_SAMPLE_COUNT = 96;

        [[nodiscard]] bool has_significant_rotation(const glm::quat& q) {
            return std::abs(std::abs(q.w) - 1.0f) > ROTATION_EPS ||
                   std::abs(q.x) > ROTATION_EPS ||
                   std::abs(q.y) > ROTATION_EPS ||
                   std::abs(q.z) > ROTATION_EPS;
        }

        [[nodiscard]] int sh_band_offset_in_rest(const int band) {
            // shN omits l=0, so l=1 starts at 0, l=2 at 3, l=3 at 8...
            return band * band - 1;
        }

        [[nodiscard]] std::vector<glm::dvec3> fibonacci_sphere_dirs(const int count) {
            std::vector<glm::dvec3> dirs;
            dirs.reserve(static_cast<size_t>(count));
            constexpr double GOLDEN_ANGLE = 2.39996322972865332;
            for (int i = 0; i < count; ++i) {
                const double t = (static_cast<double>(i) + 0.5) / static_cast<double>(count);
                const double y = 1.0 - 2.0 * t;
                const double r = std::sqrt(std::max(0.0, 1.0 - y * y));
                const double theta = GOLDEN_ANGLE * static_cast<double>(i);
                const double x = std::cos(theta) * r;
                const double z = std::sin(theta) * r;
                dirs.emplace_back(x, y, z);
            }
            return dirs;
        }

        [[nodiscard]] std::vector<double> eval_sh_band_basis(const int band, const glm::dvec3& dir) {
            const double x = dir.x;
            const double y = dir.y;
            const double z = dir.z;
            const double xx = x * x;
            const double yy = y * y;
            const double zz = z * z;

            switch (band) {
            case 0:
                return {0.28209479177387814};
            case 1:
                return {-SH_C1 * y, SH_C1 * z, -SH_C1 * x};
            case 2:
                return {
                    SH_C2_0 * x * y,
                    -SH_C2_0 * y * z,
                    SH_C2_2 * (2.0 * zz - xx - yy),
                    -SH_C2_0 * x * z,
                    SH_C2_3 * (xx - yy)};
            case 3:
                return {
                    SH_C3_0 * y * (-3.0 * xx + yy),
                    SH_C3_1 * x * y * z,
                    SH_C3_2 * y * (xx + yy - 4.0 * zz),
                    SH_C3_3 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy),
                    SH_C3_2 * x * (xx + yy - 4.0 * zz),
                    SH_C3_4 * z * (xx - yy),
                    SH_C3_0 * x * (-xx + 3.0 * yy)};
            default:
                return {};
            }
        }

        [[nodiscard]] bool solve_linear_system(std::vector<double> a, std::vector<double>& b, const int n, const int rhs_cols) {
            for (int col = 0; col < n; ++col) {
                int pivot_row = col;
                double pivot_abs = std::abs(a[col * n + col]);
                for (int row = col + 1; row < n; ++row) {
                    const double candidate = std::abs(a[row * n + col]);
                    if (candidate > pivot_abs) {
                        pivot_abs = candidate;
                        pivot_row = row;
                    }
                }
                if (pivot_abs < SH_SOLVE_EPS) {
                    return false;
                }

                if (pivot_row != col) {
                    for (int k = 0; k < n; ++k) {
                        std::swap(a[col * n + k], a[pivot_row * n + k]);
                    }
                    for (int k = 0; k < rhs_cols; ++k) {
                        std::swap(b[col * rhs_cols + k], b[pivot_row * rhs_cols + k]);
                    }
                }

                const double pivot = a[col * n + col];
                for (int k = col; k < n; ++k) {
                    a[col * n + k] /= pivot;
                }
                for (int k = 0; k < rhs_cols; ++k) {
                    b[col * rhs_cols + k] /= pivot;
                }

                for (int row = 0; row < n; ++row) {
                    if (row == col) {
                        continue;
                    }
                    const double factor = a[row * n + col];
                    if (std::abs(factor) < SH_SOLVE_EPS) {
                        continue;
                    }
                    for (int k = col; k < n; ++k) {
                        a[row * n + k] -= factor * a[col * n + k];
                    }
                    for (int k = 0; k < rhs_cols; ++k) {
                        b[row * rhs_cols + k] -= factor * b[col * rhs_cols + k];
                    }
                }
            }
            return true;
        }

        [[nodiscard]] std::optional<std::vector<float>> compute_sh_coeff_rotation_matrix(
            const glm::mat3& rotation_local_to_world,
            const int band,
            const bool mix_bands = false) {
            if (band < 1 || band > 3) {
                return std::nullopt;
            }

            const int basis_count = mix_bands ? (band + 1) * (band + 1) : 2 * band + 1;
            const auto sample_dirs = fibonacci_sphere_dirs(SH_FIT_SAMPLE_COUNT);

            const glm::dmat3 rot(rotation_local_to_world);
            const glm::dmat3 direction_pull = mix_bands ? glm::transpose(rot) : glm::inverse(rot);
            const auto evaluate_basis = [band, mix_bands](const glm::dvec3& dir) {
                if (!mix_bands)
                    return eval_sh_band_basis(band, dir);
                std::vector<double> result;
                for (int degree = 0; degree <= band; ++degree) {
                    const auto part = eval_sh_band_basis(degree, dir);
                    result.insert(result.end(), part.begin(), part.end());
                }
                return result;
            };

            std::vector<double> wtw(static_cast<size_t>(basis_count * basis_count), 0.0);
            std::vector<double> wtl(static_cast<size_t>(basis_count * basis_count), 0.0);

            for (const auto& world_dir : sample_dirs) {
                const glm::dvec3 pulled = direction_pull * world_dir;
                const glm::dvec3 local_dir = mix_bands ? pulled : glm::normalize(pulled);
                const std::vector<double> basis_world = evaluate_basis(world_dir);
                const std::vector<double> basis_local = evaluate_basis(local_dir);

                for (int r = 0; r < basis_count; ++r) {
                    for (int c = 0; c < basis_count; ++c) {
                        wtw[r * basis_count + c] += basis_world[r] * basis_world[c];
                        wtl[r * basis_count + c] += basis_world[r] * basis_local[c];
                    }
                }
            }

            std::vector<double> rhs = wtl; // Solves for K^T in W * K^T = L
            if (!solve_linear_system(std::move(wtw), rhs, basis_count, basis_count)) {
                return std::nullopt;
            }

            // Coefficient row-vectors transform as c' = c * K, where K = (K^T)^T.
            std::vector<float> coeff_matrix(static_cast<size_t>(basis_count * basis_count), 0.0f);
            for (int r = 0; r < basis_count; ++r) {
                for (int c = 0; c < basis_count; ++c) {
                    coeff_matrix[r * basis_count + c] = static_cast<float>(rhs[c * basis_count + r]);
                }
            }
            return coeff_matrix;
        }

        // Rotates canonical SH coefficients (sh0 [N,1,3] or [N,3], shN [N,K,3]) without
        // writing to the inputs, which may be shared.
        // Every output coefficient is a sum over at most seven inputs of its band, so one kernel reads shN
        // once and writes it once instead of copying, permuting and joining each band.
        constexpr int kMaxBandWidth = 7;

        // Output coefficient k sums terms[k] inputs from first[k]: its band's coefficients, or only itself
        // beyond the transformed bands. identity holds the [K,7] weights that leave every coefficient alone.
        struct ShBandLayout {
            std::vector<int32_t> first;
            std::vector<int32_t> terms;
            std::vector<float> identity;
        };

        ShBandLayout sh_band_layout(const int coefficients, const int bands) {
            ShBandLayout layout{std::vector<int32_t>(coefficients), std::vector<int32_t>(coefficients, 1),
                                std::vector<float>(static_cast<size_t>(coefficients) * kMaxBandWidth, 0.0f)};
            std::iota(layout.first.begin(), layout.first.end(), 0);
            for (int band = 1; band <= bands; ++band) {
                const int offset = sh_band_offset_in_rest(band), width = 2 * band + 1;
                if (offset + width > coefficients)
                    break;
                for (int k = 0; k < width; ++k) {
                    layout.first[offset + k] = offset;
                    layout.terms[offset + k] = width;
                }
            }
            for (int k = 0; k < coefficients; ++k)
                layout.identity[static_cast<size_t>(k) * kMaxBandWidth + (k - layout.first[k])] = 1.0f;
            return layout;
        }

        // out[row,k,c] = sum over j < terms[k] of weights[matrix[row],k,j] * shN[row,first[k]+j,c]. matrix
        // clamps, so a single [1] index serves every row.
        Tensor rotate_sh_rows(const Tensor& shN, const ShBandLayout& layout, const Tensor& weights, const Tensor& matrix) {
            static const auto kernel = [] {
                namespace f = fused;
                f::Builder builder(4);
                const auto coefficients = builder.input(DataType::Float32, 3);
                const auto first = builder.input(DataType::Int32, 1);
                const auto terms = builder.input(DataType::Int32, 1);
                const auto table = builder.input(DataType::Float32, 3);
                const auto index = builder.input(DataType::Int32, 1);
                const auto row = builder.iota(0), k = builder.iota(1), channel = builder.iota(2), j = builder.iota(3);
                const auto value = coefficients.gather({row, first.gather({k}) + j, channel}, f::Bounds::Clamp) *
                                   table.gather({index.gather({row}, f::Bounds::Clamp), k, j});
                // Terms past the band read a clamped neighbour; skip them so a non-finite one cannot leak in.
                builder.output(builder.fold(f::Fold::Sum, f::where(j < terms.gather({k}), value, 0.0f), 3),
                               DataType::Float32);
                return f::Kernel(builder);
            }();
            const auto device = shN.device();
            const size_t count = shN.size(1), rows = shN.size(0), channels = shN.size(2);
            const auto first = Tensor::from_vector(layout.first, {count}, device);
            const auto terms = Tensor::from_vector(layout.terms, {count}, device);
            const auto source = shN.contiguous();
            auto rotated = Tensor::empty(source.shape(), device, DataType::Float32);
            // The kernel domain must stay within int32 elements.
            const size_t chunk = std::max<size_t>(1, size_t(std::numeric_limits<int32_t>::max()) / (count * channels * kMaxBandWidth));
            for (size_t begin = 0; begin < rows; begin += chunk) {
                const size_t end = std::min(rows, begin + chunk);
                kernel({end - begin, count, channels, size_t(kMaxBandWidth)},
                       {source.slice(0, begin, end), first, terms, weights,
                        matrix.size(0) == 1 ? matrix : matrix.slice(0, begin, end)},
                       {rotated.slice(0, begin, end)});
            }
            return rotated;
        }

        [[nodiscard]] bool rotate_sh_fused(Tensor& shN, const int max_band, const glm::mat3& rotation_local_to_world) {
            const auto count = static_cast<int>(shN.size(1));
            const auto layout = sh_band_layout(count, max_band);
            auto weights = layout.identity;
            for (int band = 1; band <= max_band; ++band) {
                const int width = 2 * band + 1;
                const int offset = sh_band_offset_in_rest(band);
                if (offset + width > count)
                    break;
                const auto matrix = compute_sh_coeff_rotation_matrix(rotation_local_to_world, band);
                if (!matrix)
                    return false;
                // out[k] = sum_j in[j] * matrix[j][k] within the band, as the per-band matmul.
                for (int k = 0; k < width; ++k)
                    for (int j = 0; j < kMaxBandWidth; ++j)
                        weights[static_cast<size_t>(offset + k) * kMaxBandWidth + j] =
                            j < width ? (*matrix)[static_cast<size_t>(j) * width + k] : 0.0f;
            }
            const auto device = shN.device();
            shN = rotate_sh_rows(shN, layout, Tensor::from_vector(weights, {1, size_t(count), size_t(kMaxBandWidth)}, device),
                                 Tensor::zeros({1}, device, DataType::Int32));
            return true;
        }

        // means[n] = linear[m] * means[n] + translation[m] for each row's matrix m.
        const fused::Kernel& point_rows_kernel() {
            static const auto kernel = [] {
                namespace f = fused;
                f::Builder builder(3);
                const auto points = builder.input(DataType::Float32, 2);
                const auto linear = builder.input(DataType::Float32, 3);
                const auto translation = builder.input(DataType::Float32, 2);
                const auto matrix = builder.input(DataType::Int32, 1);
                const auto row = builder.iota(0), axis = builder.iota(1), column = builder.iota(2);
                const auto m = matrix.gather({row});
                const auto product = builder.fold(
                    f::Fold::Sum, linear.gather({m, axis, column}) * points.gather({row, column}), 2);
                builder.output(product + translation.gather({m, axis}), DataType::Float32);
                return f::Kernel(builder);
            }();
            return kernel;
        }

        [[nodiscard]] bool rotate_sh_canonical(Tensor& sh0, Tensor& shN, const int max_sh_degree,
                                               const glm::mat3& rotation_local_to_world) {
            const int available_coeffs = shN.is_valid() && shN.ndim() >= 2 ? static_cast<int>(shN.size(1)) : 0;
            if (max_sh_degree <= 0 || available_coeffs <= 0)
                return true;
            if (max_sh_degree > 3)
                return false;

            const int max_band = std::min(3, max_sh_degree);
            const auto device = shN.device();

            const bool orthogonal = std::abs(glm::dot(rotation_local_to_world[0], rotation_local_to_world[1])) <= 1e-6f &&
                                    std::abs(glm::dot(rotation_local_to_world[0], rotation_local_to_world[2])) <= 1e-6f &&
                                    std::abs(glm::dot(rotation_local_to_world[1], rotation_local_to_world[2])) <= 1e-6f;
            if (!orthogonal) {
                // Native rendering does not normalize the pulled direction.
                // Its polynomial remains in bands 0..degree, but shear mixes
                // those bands, including DC. A per-band rotation cannot match.
                const auto matrix = compute_sh_coeff_rotation_matrix(rotation_local_to_world, max_band, true);
                if (!matrix)
                    return false;
                const size_t count = (max_band + 1) * (max_band + 1);
                const bool flat_sh0 = sh0.ndim() == 2;
                const auto coefficients = Tensor::cat({flat_sh0 ? sh0.unsqueeze(1) : sh0, shN}, 1);
                const auto operator_tensor = Tensor::from_vector(*matrix, {count, count}, device);
                const auto transformed = coefficients.permute({2, 0, 1}).matmul(operator_tensor).permute({1, 2, 0});
                sh0 = transformed.slice(1, 0, 1).contiguous();
                if (flat_sh0)
                    sh0 = sh0.squeeze(1);
                shN = transformed.slice(1, 1, count).contiguous();
                return true;
            }

            if (device == Device::GPU)
                return rotate_sh_fused(shN, max_band, rotation_local_to_world);
            std::vector<Tensor> bands;
            int done = 0;
            for (int band = 1; band <= max_band; ++band) {
                const int coeff_count = 2 * band + 1;
                const int offset = sh_band_offset_in_rest(band);
                if (offset + coeff_count > available_coeffs)
                    break;

                const auto coeff_matrix = compute_sh_coeff_rotation_matrix(rotation_local_to_world, band);
                if (!coeff_matrix.has_value())
                    return false;

                const Tensor coeff_matrix_tensor = Tensor::from_vector(
                    coeff_matrix.value(),
                    TensorShape({static_cast<size_t>(coeff_count), static_cast<size_t>(coeff_count)}),
                    device);

                // band coefficients [N, cc, 3] → [3, N, cc]; the matrix broadcasts across channels.
                const Tensor band_coeffs = shN.slice(1, offset, offset + coeff_count).contiguous();
                bands.push_back(band_coeffs.permute({2, 0, 1}).matmul(coeff_matrix_tensor).permute({1, 2, 0}));
                done = offset + coeff_count;
            }
            if (done < available_coeffs)
                bands.push_back(shN.slice(1, done, available_coeffs));
            shN = Tensor::cat(bands, 1).contiguous();
            return true;
        }

        [[nodiscard]] bool rotate_sh_coefficients(SplatData& splat_data, const glm::mat3& rotation_local_to_world) {
            if (!splat_data.shN().is_valid() || splat_data.get_max_sh_degree() <= 0)
                return true;
            // shN is stored swizzled: rotate its canonical [N, K, 3] form, then reswizzle.
            Tensor shN = splat_data.shN_canonical();
            Tensor sh0 = splat_data.sh0_raw();
            if (!rotate_sh_canonical(sh0, shN, splat_data.get_max_sh_degree(), rotation_local_to_world))
                return false;
            splat_data.sh0_raw() = std::move(sh0);
            splat_data.shN_set_from_canonical(shN, splat_data.means().capacity());
            return true;
        }

        struct LinearPart {
            glm::mat3 rotation{1.0f};
            glm::vec3 scale{1.0f};
            bool similarity = false;
            bool changes_sh = false;
        };

        // Steps shared by transform() and transform_canonical(): positions, orientation and
        // scale. Returns the decomposition the SH step needs.
        LinearPart transform_geometry_tensors(Tensor& means, Tensor& rotation, Tensor& scaling,
                                              const glm::mat4& transform_matrix) {
            const int num_points = means.size(0);
            const auto device = means.device();

            // GLM uses column-major storage: mat[col][row], so mat[3] is the translation column.
            // Our tensor MM expects row-major, so we transpose during construction.
            // Final transform: M * p^T where p is [N,4] homogeneous points.
            const std::vector<float> transform_data = {
                transform_matrix[0][0], transform_matrix[1][0], transform_matrix[2][0], transform_matrix[3][0],
                transform_matrix[0][1], transform_matrix[1][1], transform_matrix[2][1], transform_matrix[3][1],
                transform_matrix[0][2], transform_matrix[1][2], transform_matrix[2][2], transform_matrix[3][2],
                transform_matrix[0][3], transform_matrix[1][3], transform_matrix[2][3], transform_matrix[3][3]};

            const auto transform_tensor = Tensor::from_vector(transform_data, TensorShape({4, 4}), device);
            const auto ones = Tensor::ones({static_cast<size_t>(num_points), 1}, device);
            const auto means_homo = means.cat(ones, 1);
            const auto transformed_means = transform_tensor.mm(means_homo.t()).t();

            means = transformed_means.slice(1, 0, 3).contiguous();

            // 2. Extract rotation from transform matrix
            LinearPart part;
            glm::mat3& rot_mat = part.rotation;
            rot_mat = glm::mat3(transform_matrix);
            glm::vec3& scale = part.scale;
            for (int i = 0; i < 3; ++i) {
                scale[i] = glm::length(rot_mat[i]);
                if (scale[i] > 0.0f) {
                    rot_mat[i] /= scale[i];
                }
            }

            // A uniform reflection A=-sR has the same covariance action as sR.
            // Preserve the original splat axes instead of using an SVD whose basis
            // is ambiguous at repeated scales. SH still uses the signed rot_mat below.
            const auto covariance_rotation = glm::determinant(rot_mat) < 0.0f ? -rot_mat : rot_mat;
            glm::quat rotation_quat = glm::quat_cast(covariance_rotation);

            const bool has_rotation = has_significant_rotation(rotation_quat);
            const float largest_scale = std::max({scale.x, scale.y, scale.z});
            const bool similarity = largest_scale > 0.0f &&
                                    std::abs(scale.x - scale.y) <= 1e-6f * largest_scale &&
                                    std::abs(scale.x - scale.z) <= 1e-6f * largest_scale &&
                                    std::abs(glm::dot(rot_mat[0], rot_mat[1])) <= 1e-6f &&
                                    std::abs(glm::dot(rot_mat[0], rot_mat[2])) <= 1e-6f &&
                                    std::abs(glm::dot(rot_mat[1], rot_mat[2])) <= 1e-6f;
            part.similarity = similarity;

            if (!similarity) {
                // Preserve the full affine covariance instead of averaging node
                // scale. Work from the original quaternion and log scales.
                splat_transform::LinearTransform linear;
                for (int i = 0; i < 3; ++i)
                    for (int j = 0; j < 3; ++j)
                        linear.rows[3 * i + j] = transform_matrix[j][i];
                auto scales = scaling.contiguous();
                auto rotations = rotation.contiguous();
                auto out_scales = Tensor::empty(scales.shape(), device, DataType::Float32);
                auto out_rotations = Tensor::empty(rotations.shape(), device, DataType::Float32);
                affine_splat_geometry(linear, scales, rotations, out_scales, out_rotations);
                scaling = std::move(out_scales);
                rotation = std::move(out_rotations);
            }

            // 3. Transform rotations (quaternions) if there's rotation
            if (has_rotation && similarity) {
                std::vector<float> rot_data = {rotation_quat.w, rotation_quat.x, rotation_quat.y, rotation_quat.z};
                auto rot_tensor = Tensor::from_vector(rot_data, TensorShape({4}), device);

                auto q = rotation;
                std::vector<int> expand_shape = {num_points, 4};
                auto q_rot = rot_tensor.unsqueeze(0).expand(std::span<const int>(expand_shape));

                auto w1 = q_rot.slice(1, 0, 1).squeeze(1);
                auto x1 = q_rot.slice(1, 1, 2).squeeze(1);
                auto y1 = q_rot.slice(1, 2, 3).squeeze(1);
                auto z1 = q_rot.slice(1, 3, 4).squeeze(1);

                auto w2 = q.slice(1, 0, 1).squeeze(1);
                auto x2 = q.slice(1, 1, 2).squeeze(1);
                auto y2 = q.slice(1, 2, 3).squeeze(1);
                auto z2 = q.slice(1, 3, 4).squeeze(1);

                auto w_new = w1.mul(w2).sub(x1.mul(x2)).sub(y1.mul(y2)).sub(z1.mul(z2));
                auto x_new = w1.mul(x2).add(x1.mul(w2)).add(y1.mul(z2)).sub(z1.mul(y2));
                auto y_new = w1.mul(y2).sub(x1.mul(z2)).add(y1.mul(w2)).add(z1.mul(x2));
                auto z_new = w1.mul(z2).add(x1.mul(y2)).sub(y1.mul(x2)).add(z1.mul(w2));

                std::vector<Tensor> components = {
                    w_new.unsqueeze(1),
                    x_new.unsqueeze(1),
                    y_new.unsqueeze(1),
                    z_new.unsqueeze(1)};
                rotation = Tensor::cat(components, 1);
            }

            // Match extract_rotation_rows: a degenerate node axis skips the SH
            // direction pull. Compare the matrix itself, not quat_cast(shear).
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    part.changes_sh |= std::abs(rot_mat[i][j] - (i == j ? 1.0f : 0.0f)) > ROTATION_EPS;
            part.changes_sh &= scale.x > 1e-8f && scale.y > 1e-8f && scale.z > 1e-8f;

            // 4. Transform scaling
            if (similarity && (std::abs(scale.x - 1.0f) > 1e-6f ||
                               std::abs(scale.y - 1.0f) > 1e-6f ||
                               std::abs(scale.z - 1.0f) > 1e-6f)) {

                float avg_scale = (scale.x + scale.y + scale.z) / 3.0f;
                scaling = scaling.add(std::log(avg_scale));
            }
            return part;
        }

    } // namespace

    SplatData& transform(SplatData& data, const Tensor& matrices) {
        const size_t count = data.means().size(0);
        const auto index = (Tensor::ones({count}, data.means().device(), DataType::Int32).cumsum(0) - 1).to(DataType::Int32);
        return transform(data, matrices, index);
    }

    void transform_canonical(Tensor& means_in, Tensor& rotation_in, Tensor& scaling_in, Tensor& shN_in,
                             const int degree, const Tensor& matrices, const Tensor& matrix_index) {
        const size_t count = means_in.size(0);
        LFS_ASSERT_MSG(matrices.ndim() == 3 && matrices.size(1) == 4 && matrices.size(2) == 4 &&
                           matrices.dtype() == DataType::Float32 && matrices.device() == means_in.device(),
                       std::format("Splat transforms require Float32 [M,4,4] (shape={}, dtype={}, device={}, data_device={})",
                                   matrices.shape().str(), int(matrices.dtype()), int(matrices.device()), int(means_in.device())));
        LFS_ASSERT_MSG(matrix_index.ndim() == 1 && matrix_index.numel() == count && matrix_index.dtype() == DataType::Int32 &&
                           matrix_index.device() == means_in.device(),
                       std::format("Splat transform indices require Int32 [N] (shape={}, dtype={}, count={})",
                                   matrix_index.shape().str(), int(matrix_index.dtype()), count));
        if (!count)
            return;
        const auto device = means_in.device();
        const size_t matrix_count = matrices.size(0);
        const auto linear = matrices.slice(1, 0, 3).slice(2, 0, 3).contiguous();
        const auto translation = matrices.slice(1, 0, 3).slice(2, 3, 4).squeeze(2).contiguous();
        LFS_ASSERT_MSG(degree <= 3, std::format("Per-splat SH transforms support degrees 0..3 (degree={})", degree));
        // SH rotations depend only on the matrix: fit them once per matrix, not once per splat.
        std::vector<Tensor> band_rotations;
        Tensor valid_rotation;
        if (degree) {
            const auto sample_dirs = fibonacci_sphere_dirs(SH_FIT_SAMPLE_COUNT);
            std::vector<float> directions;
            for (const auto& d : sample_dirs)
                directions.insert(directions.end(), {float(d.x), float(d.y), float(d.z)});
            const auto samples = Tensor::from_vector(directions, {size_t(SH_FIT_SAMPLE_COUNT), 3}, device);
            std::vector<Tensor> projectors;
            for (int band = 1; band <= degree; ++band) {
                const int k = 2 * band + 1;
                std::vector<double> gram(k * k, 0), rhs(k * SH_FIT_SAMPLE_COUNT, 0);
                for (int s = 0; s < SH_FIT_SAMPLE_COUNT; ++s) {
                    const auto basis = eval_sh_band_basis(band, sample_dirs[s]);
                    for (int i = 0; i < k; ++i) {
                        rhs[i * SH_FIT_SAMPLE_COUNT + s] = basis[i];
                        for (int j = 0; j < k; ++j)
                            gram[i * k + j] += basis[i] * basis[j];
                    }
                }
                LFS_ASSERT_MSG(solve_linear_system(gram, rhs, k, SH_FIT_SAMPLE_COUNT), std::format("SH projector solve failed (band={}, samples={})", band, SH_FIT_SAMPLE_COUNT));
                projectors.push_back(Tensor::from_vector(std::vector<float>(rhs.begin(), rhs.end()), {size_t(k), size_t(SH_FIT_SAMPLE_COUNT)}, device));
                band_rotations.push_back(Tensor::empty({matrix_count, size_t(k), size_t(k)}, device));
            }
            const auto norm = (linear * linear).sum(1, true).sqrt();
            const auto rotation = linear / norm.maximum(1e-8f);
            valid_rotation = norm.min(2).gt(1e-8f).unsqueeze(2);
            // Chunk scratch is independent of the number of matrices.
            for (size_t begin = 0; begin < matrix_count; begin += 4096) {
                const size_t end = std::min(matrix_count, begin + 4096);
                const auto pulled = samples.matmul(rotation.slice(0, begin, end));
                const auto x = pulled.slice(2, 0, 1).squeeze(2), y = pulled.slice(2, 1, 2).squeeze(2), z = pulled.slice(2, 2, 3).squeeze(2);
                const auto xx = x * x, yy = y * y, zz = z * z;
                for (int band = 1; band <= degree; ++band) {
                    std::vector<Tensor> basis;
                    if (band == 1)
                        basis = {y * float(-SH_C1), z * float(SH_C1), x * float(-SH_C1)};
                    if (band == 2)
                        basis = {x * y * float(SH_C2_0), y * z * float(-SH_C2_0), (zz * 2 - xx - yy) * float(SH_C2_2), x * z * float(-SH_C2_0), (xx - yy) * float(SH_C2_3)};
                    if (band == 3)
                        basis = {y * (yy - xx * 3) * float(SH_C3_0), x * y * z * float(SH_C3_1), y * (xx + yy - zz * 4) * float(SH_C3_2), z * (zz * 2 - xx * 3 - yy * 3) * float(SH_C3_3), x * (xx + yy - zz * 4) * float(SH_C3_2), z * (xx - yy) * float(SH_C3_4), x * (yy * 3 - xx) * float(SH_C3_0)};
                    band_rotations[band - 1].slice(0, begin, end).copy_from(projectors[band - 1].matmul(Tensor::stack(basis, 2)));
                }
            }
        }
        auto means = Tensor::empty({count, 3}, device), scales = Tensor::empty({count, 3}, device), rotations = Tensor::empty({count, 4}, device);
        const auto& sh = shN_in;
        // Coefficients beyond the transformed bands keep their values.
        const auto needed = static_cast<size_t>((degree + 1) * (degree + 1) - 1);
        if (device == Device::GPU) {
            // Per-row kernels look each row's matrix up instead of gathering [rows,k,k] copies for
            // batched products of tiny matrices.
            const auto linear_rows = linear.contiguous();
            const size_t chunk_rows = size_t(std::numeric_limits<int32_t>::max()) / 9;
            for (size_t begin = 0; begin < count; begin += chunk_rows) {
                const size_t end = std::min(count, begin + chunk_rows), n = end - begin;
                const auto index = matrix_index.slice(0, begin, end);
                point_rows_kernel()({n, 3, 3}, {means_in.slice(0, begin, end), linear_rows, translation, index},
                                    {means.slice(0, begin, end)});
                auto out_s = scales.slice(0, begin, end), out_q = rotations.slice(0, begin, end);
                affine_splat_geometry(linear_rows.index_select(0, index).reshape({int(n), 9}), scaling_in.slice(0, begin, end),
                                      rotation_in.slice(0, begin, end), out_s, out_q);
            }
            Tensor result;
            if (degree) {
                const auto coefficients = static_cast<int>(sh.size(1));
                LFS_ASSERT_MSG(size_t(coefficients) >= needed,
                               std::format("Per-splat SH transforms need {} coefficients for degree {} (got {})", needed, degree, coefficients));
                const auto layout = sh_band_layout(coefficients, degree);
                // Rows of [M,K,7] weights: each band's fitted rotation, padded to the widest band.
                std::vector<Tensor> rows;
                for (int band = 1; band <= degree; ++band) {
                    const int k = 2 * band + 1;
                    const auto& rotation = band_rotations[band - 1];
                    rows.push_back(k == kMaxBandWidth ? rotation
                                                      : Tensor::cat({rotation, Tensor::zeros({matrix_count, size_t(k), size_t(kMaxBandWidth - k)}, device)}, 2));
                }
                const auto identity = Tensor::from_vector(layout.identity, {1, size_t(coefficients), size_t(kMaxBandWidth)}, device);
                // Coefficients beyond the transformed bands keep their values.
                if (const auto extra = size_t(coefficients) - needed)
                    rows.push_back(identity.slice(1, needed, size_t(coefficients)).expand({int(matrix_count), int(extra), kMaxBandWidth}));
                // A matrix with a collapsed axis leaves the coefficients alone, as before.
                const auto weights = Tensor::where(valid_rotation, Tensor::cat(rows, 1), identity).contiguous();
                result = rotate_sh_rows(sh, layout, weights, matrix_index);
            }
            means_in = std::move(means);
            scaling_in = std::move(scales);
            rotation_in = std::move(rotations);
            if (degree)
                shN_in = std::move(result);
            return;
        }
        auto result_sh = !degree ? Tensor{} : sh.size(1) == needed ? Tensor::empty(sh.shape(), device)
                                                                   : sh.clone();
        // Rows gather their matrix; chunks bound the gathered [rows,k,k] SH rotations.
        for (size_t begin = 0; begin < count; begin += size_t{1} << 20) {
            const size_t end = std::min(count, begin + (size_t{1} << 20)), n = end - begin;
            const auto index = matrix_index.slice(0, begin, end);
            const auto row_linear = linear.index_select(0, index);
            means.slice(0, begin, end).copy_from(row_linear.bmm(means_in.slice(0, begin, end).unsqueeze(2)).squeeze(2) + translation.index_select(0, index));
            auto out_s = scales.slice(0, begin, end), out_q = rotations.slice(0, begin, end);
            affine_splat_geometry(row_linear.reshape({int(n), 9}), scaling_in.slice(0, begin, end), rotation_in.slice(0, begin, end), out_s, out_q);
            if (!degree)
                continue;
            const auto row_valid = valid_rotation.index_select(0, index);
            for (int band = 1; band <= degree; ++band) {
                const int offset = sh_band_offset_in_rest(band), k = 2 * band + 1;
                const auto original = sh.slice(0, begin, end).slice(1, offset, offset + k);
                result_sh.slice(0, begin, end).slice(1, offset, offset + k).copy_from(Tensor::where(row_valid, band_rotations[band - 1].index_select(0, index).bmm(original), original));
            }
        }
        means_in = std::move(means);
        scaling_in = std::move(scales);
        rotation_in = std::move(rotations);
        if (degree)
            shN_in = std::move(result_sh);
    }

    SplatData& transform(SplatData& data, const Tensor& matrices, const Tensor& matrix_index) {
        if (!data.means().size(0))
            return data;
        const int degree = data.get_max_sh_degree();
        auto means = data.means(), rotation = data.rotation_raw(), scaling = data.scaling_raw();
        auto shN = degree ? data.shN_canonical().to(means.device()) : Tensor{};
        transform_canonical(means, rotation, scaling, shN, degree, matrices, matrix_index);
        data.means_raw() = std::move(means);
        data.scaling_raw() = std::move(scaling);
        data.rotation_raw() = std::move(rotation);
        if (degree)
            data.shN_set_from_canonical(shN, data.means().size(0));
        return data;
    }

    SplatData& transform(SplatData& splat_data, const glm::mat4& transform_matrix) {
        LOG_TIMER("transform");

        if (!splat_data._means.is_valid() || splat_data._means.size(0) == 0) {
            LOG_WARN("Cannot transform invalid or empty SplatData");
            return splat_data;
        }

        const GpuBackendScope backend_scope(gpu_backend_of(splat_data._means).value_or(default_gpu_backend()));
        const int num_points = splat_data._means.size(0);
        const auto part = transform_geometry_tensors(splat_data._means, splat_data._rotation, splat_data._scaling,
                                                     transform_matrix);
        if (part.changes_sh && !rotate_sh_coefficients(splat_data, part.rotation)) {
            throw std::runtime_error("SH transformation is only supported up to degree 3.");
        }

        splat_data._scene_scale = transformed_scene_scale(splat_data._means, splat_data._scene_scale);

        LOG_DEBUG("Transformed {} gaussians", num_points);
        return splat_data;
    }

    float transformed_scene_scale(const Tensor& means, const float scene_scale) {
        if (!means.is_valid() || means.size(0) == 0)
            return scene_scale;
        const Tensor centre = means.mean({0}, false);
        const Tensor distances = means.sub(centre).norm(2.0f, {1}, false);
        const float median = distances.sort(0, false).first[means.size(0) / 2].item();
        return std::abs(median - scene_scale) > scene_scale * 0.1f ? median : scene_scale;
    }

    float transform_canonical(Tensor& means, Tensor& rotation, Tensor& scaling, Tensor& sh0, Tensor& shN,
                              const int sh_degree, const glm::mat4& transform_matrix) {
        if (!means.is_valid() || means.size(0) == 0)
            return 1.0f;
        const GpuBackendScope backend_scope(gpu_backend_of(means).value_or(default_gpu_backend()));
        const auto part = transform_geometry_tensors(means, rotation, scaling, transform_matrix);
        if (part.changes_sh && !rotate_sh_canonical(sh0, shN, sh_degree, part.rotation))
            throw std::runtime_error("SH transformation is only supported up to degree 3.");
        return part.similarity ? (part.scale.x + part.scale.y + part.scale.z) / 3.0f : 0.0f;
    }

    Tensor compute_cropbox_mask(const Tensor& means,
                                const glm::vec3& crop_min,
                                const glm::vec3& crop_max,
                                const glm::mat4& points_to_cropbox) {
        LFS_ASSERT_MSG(means.is_valid(), "crop box means must be valid");
        LFS_ASSERT_MSG(means.dtype() == DataType::Float32, "crop box means must be Float32");
        LFS_ASSERT_MSG(means.ndim() == 2, "crop box means must be a 2D tensor");
        LFS_ASSERT_MSG(means.size(1) >= 3, "crop box means must have at least three columns");

        const auto rotation = Tensor::from_vector(
            {points_to_cropbox[0][0], points_to_cropbox[1][0], points_to_cropbox[2][0],
             points_to_cropbox[0][1], points_to_cropbox[1][1], points_to_cropbox[2][1],
             points_to_cropbox[0][2], points_to_cropbox[1][2], points_to_cropbox[2][2]},
            {3, 3},
            means.device());
        const auto translation = Tensor::from_vector(
            {points_to_cropbox[3][0], points_to_cropbox[3][1], points_to_cropbox[3][2]},
            {1, 3},
            means.device());

        const auto xyz_means = means.slice(1, 0, 3);
        const auto local_points = xyz_means.mm(rotation.t()).add(translation);

        auto bbox_min_tensor = Tensor::from_vector(
            {crop_min.x, crop_min.y, crop_min.z}, {3}, means.device());
        auto bbox_max_tensor = Tensor::from_vector(
            {crop_max.x, crop_max.y, crop_max.z}, {3}, means.device());

        auto inside_min = local_points.ge(bbox_min_tensor.unsqueeze(0));
        auto inside_max = local_points.le(bbox_max_tensor.unsqueeze(0));

        auto inside_both = inside_min && inside_max;
        std::vector<int> reduce_dims = {1};
        return inside_both.all(std::span<const int>(reduce_dims), false);
    }

    static Tensor compute_cropbox_mask(const Tensor& means,
                                       const lfs::geometry::BoundingBox& bounding_box) {
        const glm::mat4 points_to_cropbox = bounding_box.hasFullTransform()
                                                ? bounding_box.getworld2BBoxMat4()
                                                : bounding_box.getworld2BBox().toMat4();
        return compute_cropbox_mask(
            means,
            bounding_box.getMinBounds(),
            bounding_box.getMaxBounds(),
            points_to_cropbox);
    }

    SplatData crop_by_cropbox(const SplatData& splat_data,
                              const lfs::geometry::BoundingBox& bounding_box,
                              const bool inverse) {
        LOG_TIMER("crop_by_cropbox");

        if (!splat_data._means.is_valid() || splat_data._means.size(0) == 0) {
            LOG_WARN("Cannot crop invalid or empty SplatData");
            return SplatData();
        }

        const int num_points = splat_data._means.size(0);

        auto inside_mask = compute_cropbox_mask(splat_data._means, bounding_box);

        // Invert mask if inverse mode
        auto selection_mask = inverse ? inside_mask.logical_not() : inside_mask;
        const int points_selected = selection_mask.sum_scalar();

        if (points_selected == 0) {
            LOG_WARN("No points selected, returning empty SplatData");
            return SplatData();
        }

        auto indices = selection_mask.nonzero();
        if (indices.ndim() == 2) {
            indices = indices.squeeze(1);
        }

        auto cropped_means = splat_data._means.index_select(0, indices).contiguous();
        auto cropped_sh0 = splat_data._sh0.index_select(0, indices).contiguous();
        Tensor cropped_shN;
        const size_t layout_rest = splat_data.max_sh_coeffs_rest();
        if (splat_data._shN.is_valid() && splat_data._shN.numel() && layout_rest) {
            cropped_shN = Tensor::empty_like(splat_data._shN, {size_t(points_selected), layout_rest, 3}, DataType::Float32);
            sh_codec(splat_data._shN, cropped_shN,
                     {.source_format = sh_storage_format(splat_data._shN, splat_data._shN_value_bounds),
                      .destination_format = ShFormat::Canonical,
                      .source_rows = size_t(num_points),
                      .destination_rows = size_t(points_selected),
                      .count = size_t(points_selected),
                      .source_rest = uint32_t(layout_rest),
                      .destination_rest = uint32_t(layout_rest)},
                     &indices, splat_data.shN_value_quantized() ? &splat_data._shN_value_bounds : nullptr);
        }
        auto cropped_scaling = splat_data._scaling.index_select(0, indices).contiguous();
        auto cropped_rotation = splat_data._rotation.index_select(0, indices).contiguous();
        auto cropped_opacity = splat_data._opacity.index_select(0, indices).contiguous();

        Tensor scene_center = cropped_means.mean({0}, false);
        Tensor dists = cropped_means.sub(scene_center).norm(2.0f, {1}, false);

        float new_scene_scale = splat_data._scene_scale;
        if (points_selected > 1) {
            auto sorted_dists = dists.sort(0, false);
            new_scene_scale = sorted_dists.first[points_selected / 2].item();
        }

        SplatData cropped_splat(
            splat_data._max_sh_degree,
            std::move(cropped_means),
            std::move(cropped_sh0),
            std::move(cropped_shN),
            std::move(cropped_scaling),
            std::move(cropped_rotation),
            std::move(cropped_opacity),
            new_scene_scale);

        cropped_splat.set_active_sh_degree(splat_data._active_sh_degree);

        if (splat_data._densification_info.is_valid() && splat_data._densification_info.size(0) == num_points) {
            cropped_splat._densification_info =
                splat_data._densification_info.index_select(0, indices).contiguous();
        }
        if (splat_data._max_screen_share.is_valid() &&
            splat_data._max_screen_share.ndim() == 1 &&
            splat_data._max_screen_share.size(0) == num_points) {
            cropped_splat._max_screen_share =
                splat_data._max_screen_share.index_select(0, indices).contiguous();
        }

        (void)cropped_splat.apply_shN_value_quant();

        LOG_DEBUG("Cropped SplatData: {} -> {} points (inverse={})", num_points, points_selected, inverse);
        return cropped_splat;
    }

    Tensor soft_crop_by_cropbox(SplatData& splat_data,
                                const lfs::geometry::BoundingBox& bounding_box,
                                const bool inverse) {
        LOG_TIMER("soft_crop_by_cropbox");

        const auto& means = splat_data.means();
        if (!means.is_valid() || means.size(0) == 0) {
            return Tensor();
        }

        const auto inside_mask = compute_cropbox_mask(means, bounding_box);
        const auto delete_mask = inverse ? inside_mask : inside_mask.logical_not();
        const int points_to_delete = delete_mask.sum_scalar();

        if (points_to_delete == 0) {
            return Tensor();
        }

        return splat_data.soft_delete(delete_mask);
    }

    Tensor soft_crop_by_ellipsoid(SplatData& splat_data,
                                  const glm::mat4& transform,
                                  const glm::vec3& radii,
                                  const bool inverse) {
        LOG_TIMER("soft_crop_by_ellipsoid");

        const auto& means = splat_data.means();
        if (!means.is_valid() || means.size(0) == 0) {
            return Tensor();
        }

        const size_t num_points = static_cast<size_t>(means.size(0));
        const auto device = means.device();

        // Build transformation tensor (GLM column-major to row-major)
        const auto transform_tensor = Tensor::from_vector(
            {transform[0][0], transform[1][0], transform[2][0], transform[3][0],
             transform[0][1], transform[1][1], transform[2][1], transform[3][1],
             transform[0][2], transform[1][2], transform[2][2], transform[3][2],
             transform[0][3], transform[1][3], transform[2][3], transform[3][3]},
            {4, 4}, device);

        // Transform to ellipsoid local space
        const auto ones = Tensor::ones({num_points, 1}, device);
        const auto local_pos = transform_tensor.mm(means.cat(ones, 1).t()).t();

        // Compute normalized distances: (x/rx)^2 + (y/ry)^2 + (z/rz)^2
        const auto x = local_pos.slice(1, 0, 1).squeeze(1) / radii.x;
        const auto y = local_pos.slice(1, 1, 2).squeeze(1) / radii.y;
        const auto z = local_pos.slice(1, 2, 3).squeeze(1) / radii.z;

        const auto dist_sq = x * x + y * y + z * z;
        const auto inside_mask = dist_sq <= 1.0f;
        const auto delete_mask = inverse ? inside_mask : inside_mask.logical_not();
        const int points_to_delete = delete_mask.sum_scalar();

        if (points_to_delete == 0) {
            return Tensor();
        }

        return splat_data.soft_delete(delete_mask);
    }

    void random_choose(SplatData& splat_data, int num_required_splat, int seed) {
        LOG_TIMER("random_choose");

        if (!splat_data._means.is_valid() || splat_data._means.size(0) == 0) {
            LOG_WARN("Cannot choose from invalid or empty SplatData");
            return;
        }

        const int num_points = splat_data._means.size(0);

        if (num_required_splat <= 0) {
            LOG_WARN("num_splat must be positive, got {}", num_required_splat);
            return;
        }

        if (num_required_splat >= num_points) {
            LOG_DEBUG("num_splat ({}) >= total points ({}), keeping all data",
                      num_required_splat, num_points);
            return;
        }

        LOG_DEBUG("Randomly selecting {} points from {} total points (seed: {})",
                  num_required_splat, num_points, seed);
        const size_t old_capacity = splat_data._means.is_valid()
                                        ? splat_data._means.capacity()
                                        : static_cast<size_t>(num_points);

        std::vector<int> all_indices(num_points);
        std::iota(all_indices.begin(), all_indices.end(), 0);

        std::mt19937 rng(seed);
        std::shuffle(all_indices.begin(), all_indices.end(), rng);

        std::vector<unsigned char> old_frozen(static_cast<size_t>(num_points), 0);
        if (!splat_data._frozen_ranges.empty()) {
            for (const auto& range : splat_data._frozen_ranges) {
                if (range.count == 0 || range.start >= old_frozen.size()) {
                    continue;
                }
                const size_t end = range.count > old_frozen.size() - range.start
                                       ? old_frozen.size()
                                       : range.start + range.count;
                std::fill(old_frozen.begin() + static_cast<std::ptrdiff_t>(range.start),
                          old_frozen.begin() + static_cast<std::ptrdiff_t>(end),
                          1);
            }
        }
        const size_t frozen_total = std::count(old_frozen.begin(), old_frozen.end(), 1);

        std::vector<int> selected_indices;
        selected_indices.reserve(static_cast<size_t>(num_required_splat));
        if (splat_data._frozen_ranges.empty()) {
            selected_indices.assign(all_indices.begin(), all_indices.begin() + num_required_splat);
        } else {
            std::vector<int> trainable_indices;
            trainable_indices.reserve(all_indices.size());
            for (const int idx : all_indices) {
                if (old_frozen[static_cast<size_t>(idx)]) {
                    if (static_cast<int>(selected_indices.size()) < num_required_splat) {
                        selected_indices.push_back(idx);
                    }
                } else {
                    trainable_indices.push_back(idx);
                }
            }

            if (frozen_total > static_cast<size_t>(num_required_splat)) {
                LOG_WARN("random_choose kept only frozen rows because requested count {} is smaller than frozen count",
                         num_required_splat);
            }
            for (const int idx : trainable_indices) {
                if (static_cast<int>(selected_indices.size()) >= num_required_splat) {
                    break;
                }
                selected_indices.push_back(idx);
            }
        }

        auto indices_tensor = Tensor::from_vector(
            selected_indices,
            TensorShape({static_cast<size_t>(num_required_splat)}),
            splat_data._means.device());

        Tensor shN_selected_swizzled;
        Tensor shN_selected_canonical;
        const auto layout_rest = static_cast<uint32_t>(splat_data.max_sh_coeffs_rest());
        const bool q16_or_f16 =
            splat_data._shN.is_valid() && splat_data._shN.numel() > 0 && layout_rest > 0 &&
            (splat_data._shN.dtype() != DataType::Float32 || splat_data.shN_value_quantized() ||
             splat_data.shN_ieee_f16());
        if (q16_or_f16) {
            shN_selected_canonical = Tensor::empty_like(splat_data._shN, {size_t(num_required_splat), size_t(layout_rest), 3}, DataType::Float32);
            sh_codec(splat_data._shN, shN_selected_canonical,
                     {.source_format = sh_storage_format(splat_data._shN, splat_data._shN_value_bounds), .destination_format = ShFormat::Canonical, .source_rows = size_t(splat_data.size()), .destination_rows = size_t(num_required_splat), .count = size_t(num_required_splat), .source_rest = layout_rest, .destination_rest = layout_rest},
                     &indices_tensor, splat_data.shN_value_quantized() ? &splat_data._shN_value_bounds : nullptr);
        } else if (splat_data._shN.is_valid() && splat_data._shN.numel() > 0 &&
                   layout_rest > 0) {
            shN_selected_swizzled = Tensor::zeros_direct(
                {sh_swizzled_float_count(static_cast<size_t>(num_required_splat), layout_rest)},
                sh_swizzled_float_count(static_cast<size_t>(num_required_splat), layout_rest),
                splat_data._shN.device());
            auto indices_i32 = indices_tensor.dtype() == DataType::Int32
                                   ? indices_tensor
                                   : indices_tensor.to(DataType::Int32);
            sh_codec(splat_data._shN, shN_selected_swizzled,
                     {.source_rows = size_t(splat_data.size()), .destination_rows = size_t(num_required_splat), .count = size_t(num_required_splat), .source_rest = layout_rest, .destination_rest = layout_rest}, &indices_i32);
        }

        splat_data._means = splat_data._means.index_select(0, indices_tensor).contiguous();
        splat_data._sh0 = splat_data._sh0.index_select(0, indices_tensor).contiguous();
        if (shN_selected_canonical.is_valid() && shN_selected_canonical.numel() > 0) {
            splat_data._shN_value_bounds = Tensor{};
            splat_data.shN_set_from_canonical(shN_selected_canonical,
                                              static_cast<size_t>(num_required_splat));
        } else if (shN_selected_swizzled.is_valid() && shN_selected_swizzled.numel() > 0) {
            splat_data._shN = std::move(shN_selected_swizzled);
        }
        (void)splat_data.apply_shN_value_quant();
        splat_data._scaling = splat_data._scaling.index_select(0, indices_tensor).contiguous();
        splat_data._rotation = splat_data._rotation.index_select(0, indices_tensor).contiguous();
        splat_data._opacity = splat_data._opacity.index_select(0, indices_tensor).contiguous();
        if (splat_data._tensor_allocator) {
            const size_t dest_cap = std::max(static_cast<size_t>(num_required_splat), old_capacity);
            const auto home = [&](Tensor& tensor, std::string_view name) {
                if (!tensor.is_valid()) {
                    return;
                }
                Tensor dest = splat_data.allocate_named_param(
                    tensor.shape(), dest_cap, tensor.dtype(), name);
                dest.copy_from(tensor);
                tensor = std::move(dest);
            };
            home(splat_data._means, "SplatData.means");
            home(splat_data._sh0, "SplatData.sh0");
            home(splat_data._scaling, "SplatData.scaling");
            home(splat_data._rotation, "SplatData.rotation");
            home(splat_data._opacity, "SplatData.opacity");
            if (!sh_value_quant::enabled() && splat_data._shN.is_valid() &&
                splat_data._shN.dtype() == DataType::Float32) {
                const auto rest = static_cast<uint32_t>(splat_data.max_sh_coeffs_rest());
                Tensor dest = splat_data.allocate_named_param(
                    splat_data._shN.shape(),
                    sh_swizzled_float_count(dest_cap, rest),
                    DataType::Float32,
                    "SplatData.shN");
                dest.copy_from(splat_data._shN);
                splat_data._shN = std::move(dest);
            }
        }
        if (!splat_data._frozen_ranges.empty()) {
            splat_data.remap_frozen_ranges_after_keep(
                static_cast<size_t>(num_points),
                selected_indices);
        }

        if (splat_data._densification_info.is_valid()) {
            if (splat_data._densification_info.ndim() == 1 &&
                splat_data._densification_info.size(0) == num_points) {
                splat_data._densification_info =
                    splat_data._densification_info.index_select(0, indices_tensor).contiguous();
            } else if (splat_data._densification_info.ndim() == 2 &&
                       splat_data._densification_info.size(1) == num_points) {
                splat_data._densification_info =
                    splat_data._densification_info.index_select(1, indices_tensor).contiguous();
            }
        }
        if (splat_data._max_screen_share.is_valid() &&
            splat_data._max_screen_share.ndim() == 1 &&
            splat_data._max_screen_share.size(0) == num_points) {
            splat_data._max_screen_share =
                splat_data._max_screen_share.index_select(0, indices_tensor).contiguous();
        }

        // keep deleted mask sized to the new N (or invalidate).
        if (splat_data.has_deleted_mask()) {
            if (static_cast<size_t>(splat_data.deleted().numel()) == static_cast<size_t>(num_points)) {
                Tensor gathered = splat_data.deleted()
                                      .index_select(0, indices_tensor)
                                      .contiguous();
                if (gathered.dtype() != DataType::Bool) {
                    gathered = gathered.to(DataType::Bool);
                }
                if (gathered.device() != Device::GPU) {
                    gathered = gathered.gpu();
                }
                gathered.set_name("splat.deleted_mask");
                splat_data.deleted() = std::move(gathered);
                splat_data.refresh_deleted_count();
                splat_data.notify_deleted_mask_changed();
            } else {
                splat_data.reconcile_deleted_mask();
            }
        }

        Tensor scene_center = splat_data._means.mean({0}, false);
        Tensor dists = splat_data._means.sub(scene_center).norm(2.0f, {1}, false);

        float old_scene_scale = splat_data._scene_scale;
        if (num_required_splat > 1) {
            auto sorted_dists = dists.sort(0, false);
            splat_data._scene_scale = sorted_dists.first[num_required_splat / 2].item();
        }

        LOG_DEBUG("Successfully selected {} random splats in-place (scale: {:.4f} -> {:.4f})",
                  num_required_splat, old_scene_scale, splat_data._scene_scale);
    }

    bool compute_bounds(const SplatData& splat_data,
                        glm::vec3& min_bounds,
                        glm::vec3& max_bounds,
                        const float padding,
                        const bool use_percentile) {
        const auto& means = splat_data.means();
        if (!means.is_valid() || means.size(0) == 0) {
            return false;
        }

        // Filter deleted gaussians (index_select preserves [N,3] shape)
        Tensor visible_means = means;
        if (splat_data.has_deleted_mask()) {
            const auto visible_indices = splat_data.deleted().logical_not().nonzero().squeeze(1);
            if (visible_indices.size(0) == 0)
                return false;
            visible_means = means.index_select(0, visible_indices);
        }

        if (visible_means.size(0) == 0) {
            return false;
        }

        const int64_t n = visible_means.size(0);

        if (use_percentile && n > 100) {
            percentile_bounds(visible_means, min_bounds, max_bounds, padding);
        } else {
            for (int i = 0; i < 3; ++i) {
                const auto col = visible_means.slice(1, i, i + 1).squeeze(1);
                min_bounds[i] = col.min().item() - padding;
                max_bounds[i] = col.max().item() + padding;
            }
        }

        return true;
    }

    bool compute_bounds(const PointCloud& point_cloud,
                        glm::vec3& min_bounds,
                        glm::vec3& max_bounds,
                        const float padding,
                        const bool use_percentile) {
        const auto& means = point_cloud.means;
        if (!means.is_valid() || means.size(0) == 0) {
            return false;
        }

        const int64_t n = means.size(0);

        if (use_percentile && n > 100) {
            percentile_bounds(means, min_bounds, max_bounds, padding);
        } else {
            for (int i = 0; i < 3; ++i) {
                const auto col = means.slice(1, i, i + 1).squeeze(1);
                min_bounds[i] = col.min().item() - padding;
                max_bounds[i] = col.max().item() + padding;
            }
        }

        return true;
    }

    void copy_sh_coefficients(const SplatData& model, Tensor& destination,
                              size_t destination_offset, uint32_t destination_rest) {
        const auto count = static_cast<size_t>(model.size());
        const auto rest = static_cast<uint32_t>(model.max_sh_coeffs_rest());
        if (!count || !rest || !destination_rest)
            return;
        sh_codec(model.shN_raw(), destination,
                 {.source_format = sh_storage_format(model.shN_raw(), model.shN_value_bounds()),
                  .source_rows = count,
                  .destination_rows = destination.numel() / (((destination_rest * 3 + 3) / 4) * 4),
                  .count = count,
                  .source_rest = rest,
                  .destination_rest = destination_rest,
                  .destination_offset = destination_offset,
                  .match_cpu_rounding = true},
                 nullptr, model.shN_value_quantized() ? &model.shN_value_bounds() : nullptr);
    }

    SplatData extract_by_mask(const SplatData& splat_data, const Tensor& mask) {
        if (!splat_data._means.is_valid() || splat_data._means.size(0) == 0) {
            return SplatData();
        }
        if (!mask.is_valid() || mask.size(0) != splat_data._means.size(0)) {
            return SplatData();
        }
        std::optional<GpuBackendScope> backend_scope;
        if (const auto backend = gpu_backend_of(splat_data._means))
            backend_scope.emplace(*backend);

        const auto selection_mask = mask.to(DataType::Bool);
        const int count = selection_mask.sum_scalar();
        if (count == 0) {
            return SplatData();
        }

        auto indices = selection_mask.nonzero();
        if (indices.ndim() == 2) {
            indices = indices.squeeze(1);
        }

        Tensor shN_selected, selected_bounds;
        const uint32_t layout_rest = static_cast<uint32_t>(splat_data.max_sh_coeffs_rest());
        const bool resident_output = splat_data._shN.is_valid() && splat_data._shN.device() == Device::GPU;
        const bool quantized_output = resident_output && sh_value_quant::enabled();
        if (splat_data._shN.is_valid() && splat_data._shN.numel() && layout_rest) {
            const auto format = quantized_output ? ShFormat::Q16 : resident_output ? ShFormat::Float32
                                                                                   : ShFormat::Canonical;
            const auto shape = resident_output ? TensorShape({quantized_output ? sh_value_quant::sh_value_u16_count(count, layout_rest)
                                                                               : sh_swizzled_float_count(count, layout_rest)})
                                               : TensorShape({size_t(count), size_t(layout_rest), size_t{3}});
            shN_selected = Tensor::empty(shape, splat_data._shN.device(), quantized_output ? DataType::Float16 : DataType::Float32);
            if (quantized_output)
                selected_bounds = Tensor::empty({sh_value_quant::n_bounds_for_prims(count) * 2}, splat_data._shN.device());
            sh_codec(splat_data._shN, shN_selected,
                     {.source_format = sh_storage_format(splat_data._shN, splat_data._shN_value_bounds), .destination_format = format, .source_rows = size_t(splat_data.size()), .destination_rows = size_t(count), .count = size_t(count), .source_rest = layout_rest, .destination_rest = layout_rest},
                     &indices, splat_data.shN_value_quantized() ? &splat_data._shN_value_bounds : nullptr,
                     quantized_output ? &selected_bounds : nullptr);
        }

        SplatData result(
            splat_data._max_sh_degree,
            splat_data._means.index_select(0, indices).contiguous(),
            splat_data._sh0.index_select(0, indices).contiguous(),
            std::move(shN_selected),
            splat_data._scaling.index_select(0, indices).contiguous(),
            splat_data._rotation.index_select(0, indices).contiguous(),
            splat_data._opacity.index_select(0, indices).contiguous(),
            splat_data._scene_scale, resident_output ? SplatData::ShNLayout::Swizzled : SplatData::ShNLayout::Canonical);
        result.set_active_sh_degree(splat_data._active_sh_degree, std::move(selected_bounds));

        // If the mask keeps every gaussian, preserve the LOD tree unchanged.
        if (count == static_cast<int>(splat_data.size()) && splat_data.lod_tree) {
            result.lod_tree = std::make_unique<lfs::core::SplatLodTree>(*splat_data.lod_tree);
        }

        (void)result.apply_shN_value_quant();
        return result;
    }

} // namespace lfs::core

/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/splat_data_transform.hpp"
#include "core/logger.hpp"
#include "core/point_cloud.hpp"
#include "core/splat_data.hpp"
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

        constexpr double SH_C1 = 0.48860251190291987;
        constexpr double SH_C2_0 = 1.0925484305920792;
        constexpr double SH_C2_1 = 0.94617469575755997;
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
            case 1:
                return {-SH_C1 * y, SH_C1 * z, -SH_C1 * x};
            case 2:
                return {
                    SH_C2_0 * x * y,
                    -SH_C2_0 * y * z,
                    SH_C2_1 * zz - SH_C2_2,
                    -SH_C2_0 * x * z,
                    SH_C2_3 * (xx - yy)};
            case 3:
                return {
                    SH_C3_0 * y * (-3.0 * xx + yy),
                    SH_C3_1 * x * y * z,
                    SH_C3_2 * y * (1.0 - 5.0 * zz),
                    SH_C3_3 * z * (5.0 * zz - 3.0),
                    SH_C3_2 * x * (1.0 - 5.0 * zz),
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
            const int band) {
            if (band < 1 || band > 3) {
                return std::nullopt;
            }

            const int basis_count = 2 * band + 1;
            const auto sample_dirs = fibonacci_sphere_dirs(SH_FIT_SAMPLE_COUNT);

            const glm::dmat3 rot(rotation_local_to_world);
            const glm::dmat3 rot_inv = glm::inverse(rot);

            std::vector<double> wtw(static_cast<size_t>(basis_count * basis_count), 0.0);
            std::vector<double> wtl(static_cast<size_t>(basis_count * basis_count), 0.0);

            for (const auto& world_dir : sample_dirs) {
                const glm::dvec3 local_dir = glm::normalize(rot_inv * world_dir);
                const std::vector<double> basis_world = eval_sh_band_basis(band, world_dir);
                const std::vector<double> basis_local = eval_sh_band_basis(band, local_dir);

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

        [[nodiscard]] bool rotate_sh_coefficients(SplatData& splat_data, const glm::mat3& rotation_local_to_world) {
            if (!splat_data.shN().is_valid() || splat_data.get_max_sh_degree() <= 0) {
                return true;
            }

            const int available_coeffs = splat_data.shN().ndim() >= 2 ? static_cast<int>(splat_data.shN().size(1)) : 0;
            if (available_coeffs <= 0) {
                return true;
            }

            if (splat_data.get_max_sh_degree() > 3) {
                return false;
            }

            const int max_band = std::min(3, splat_data.get_max_sh_degree());
            const auto device = splat_data.shN().device();

            for (int band = 1; band <= max_band; ++band) {
                const int coeff_count = 2 * band + 1;
                const int offset = sh_band_offset_in_rest(band);
                if (offset + coeff_count > available_coeffs) {
                    break;
                }

                const auto coeff_matrix = compute_sh_coeff_rotation_matrix(rotation_local_to_world, band);
                if (!coeff_matrix.has_value()) {
                    return false;
                }

                const Tensor coeff_matrix_tensor = Tensor::from_vector(
                    coeff_matrix.value(),
                    TensorShape({static_cast<size_t>(coeff_count), static_cast<size_t>(coeff_count)}),
                    device);

                const Tensor band_coeffs = splat_data.shN().slice(1, offset, offset + coeff_count).contiguous();
                // band_coeffs: [N, coeff_count, 3] → permute to [3, N, coeff_count]
                // matmul broadcasts coeff_matrix [cc, cc] across batch dim 3
                const Tensor channels_first = band_coeffs.permute({2, 0, 1});
                const Tensor rotated = channels_first.matmul(coeff_matrix_tensor);
                const Tensor rotated_band = rotated.permute({1, 2, 0});
                splat_data.shN().slice(1, offset, offset + coeff_count).copy_from(rotated_band);
            }

            return true;
        }

    } // namespace

    SplatData& transform(SplatData& splat_data, const glm::mat4& transform_matrix) {
        LOG_TIMER("transform");

        if (!splat_data._means.is_valid() || splat_data._means.size(0) == 0) {
            LOG_WARN("Cannot transform invalid or empty SplatData");
            return splat_data;
        }

        const int num_points = splat_data._means.size(0);
        auto device = splat_data._means.device();

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
        const auto means_homo = splat_data._means.cat(ones, 1);
        const auto transformed_means = transform_tensor.mm(means_homo.t()).t();

        splat_data._means = transformed_means.slice(1, 0, 3).contiguous();

        // 2. Extract rotation from transform matrix
        glm::mat3 rot_mat(transform_matrix);
        glm::vec3 scale;
        for (int i = 0; i < 3; ++i) {
            scale[i] = glm::length(rot_mat[i]);
            if (scale[i] > 0.0f) {
                rot_mat[i] /= scale[i];
            }
        }

        glm::quat rotation_quat = glm::quat_cast(rot_mat);

        const bool has_rotation = has_significant_rotation(rotation_quat);

        // 3. Transform rotations (quaternions) and SH orientation if there's rotation
        if (has_rotation) {
            std::vector<float> rot_data = {rotation_quat.w, rotation_quat.x, rotation_quat.y, rotation_quat.z};
            auto rot_tensor = Tensor::from_vector(rot_data, TensorShape({4}), device);

            auto q = splat_data._rotation;
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
            splat_data._rotation = Tensor::cat(components, 1);

            if (!rotate_sh_coefficients(splat_data, rot_mat)) {
                throw std::runtime_error("SH rotation during transform is only supported up to degree 3.");
            }
        }

        // 4. Transform scaling
        if (std::abs(scale.x - 1.0f) > 1e-6f ||
            std::abs(scale.y - 1.0f) > 1e-6f ||
            std::abs(scale.z - 1.0f) > 1e-6f) {

            float avg_scale = (scale.x + scale.y + scale.z) / 3.0f;
            splat_data._scaling = splat_data._scaling.add(std::log(avg_scale));
        }

        // 5. Update scene scale
        Tensor scene_center = splat_data._means.mean({0}, false);
        Tensor dists = splat_data._means.sub(scene_center).norm(2.0f, {1}, false);
        auto sorted_dists = dists.sort(0, false);
        float new_scene_scale = sorted_dists.first[num_points / 2].item();

        if (std::abs(new_scene_scale - splat_data._scene_scale) > splat_data._scene_scale * 0.1f) {
            splat_data._scene_scale = new_scene_scale;
        }

        LOG_DEBUG("Transformed {} gaussians", num_points);
        return splat_data;
    }

    // Helper: compute inside-cropbox mask for given means and bounding box
    static Tensor compute_cropbox_mask(const Tensor& means,
                                       const lfs::geometry::BoundingBox& bounding_box) {
        const auto bbox_min = bounding_box.getMinBounds();
        const auto bbox_max = bounding_box.getMaxBounds();

        const int num_points = means.size(0);

        // Use full mat4 if available (preserves scale), otherwise fall back to EuclideanTransform
        const glm::mat4 world_to_bbox_matrix = bounding_box.hasFullTransform()
                                                   ? bounding_box.getworld2BBoxMat4()
                                                   : bounding_box.getworld2BBox().toMat4();

        const std::vector<float> transform_data = {
            world_to_bbox_matrix[0][0], world_to_bbox_matrix[1][0], world_to_bbox_matrix[2][0], world_to_bbox_matrix[3][0],
            world_to_bbox_matrix[0][1], world_to_bbox_matrix[1][1], world_to_bbox_matrix[2][1], world_to_bbox_matrix[3][1],
            world_to_bbox_matrix[0][2], world_to_bbox_matrix[1][2], world_to_bbox_matrix[2][2], world_to_bbox_matrix[3][2],
            world_to_bbox_matrix[0][3], world_to_bbox_matrix[1][3], world_to_bbox_matrix[2][3], world_to_bbox_matrix[3][3]};
        auto transform_tensor = Tensor::from_vector(
            transform_data,
            TensorShape({4, 4}),
            means.device());

        auto ones = Tensor::ones({static_cast<size_t>(num_points), 1}, means.device());
        auto means_homo = means.cat(ones, 1);

        const auto transformed_points = transform_tensor.mm(means_homo.t()).t();
        const auto local_points = transformed_points.slice(1, 0, 3);

        const std::vector<float> bbox_min_data = {bbox_min.x, bbox_min.y, bbox_min.z};
        const std::vector<float> bbox_max_data = {bbox_max.x, bbox_max.y, bbox_max.z};

        auto bbox_min_tensor = Tensor::from_vector(bbox_min_data, TensorShape({3}), means.device());
        auto bbox_max_tensor = Tensor::from_vector(bbox_max_data, TensorShape({3}), means.device());

        auto inside_min = local_points.ge(bbox_min_tensor.unsqueeze(0));
        auto inside_max = local_points.le(bbox_max_tensor.unsqueeze(0));

        auto inside_both = inside_min && inside_max;
        std::vector<int> reduce_dims = {1};
        return inside_both.all(std::span<const int>(reduce_dims), false);
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
        Tensor cropped_shN = splat_data._shN.is_valid()
                                 ? splat_data._shN.index_select(0, indices).contiguous()
                                 : Tensor{};
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

        std::vector<int> all_indices(num_points);
        std::iota(all_indices.begin(), all_indices.end(), 0);

        std::mt19937 rng(seed);
        std::shuffle(all_indices.begin(), all_indices.end(), rng);

        std::vector<int> selected_indices(all_indices.begin(),
                                          all_indices.begin() + num_required_splat);

        auto indices_tensor = Tensor::from_vector(
            selected_indices,
            TensorShape({static_cast<size_t>(num_required_splat)}),
            splat_data._means.device());

        splat_data._means = splat_data._means.index_select(0, indices_tensor).contiguous();
        splat_data._sh0 = splat_data._sh0.index_select(0, indices_tensor).contiguous();
        if (splat_data._shN.is_valid()) {
            splat_data._shN = splat_data._shN.index_select(0, indices_tensor).contiguous();
        }
        splat_data._scaling = splat_data._scaling.index_select(0, indices_tensor).contiguous();
        splat_data._rotation = splat_data._rotation.index_select(0, indices_tensor).contiguous();
        splat_data._opacity = splat_data._opacity.index_select(0, indices_tensor).contiguous();

        if (splat_data._densification_info.is_valid() && splat_data._densification_info.size(0) == num_points) {
            splat_data._densification_info = splat_data._densification_info.index_select(0, indices_tensor).contiguous();
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
            // Exclude 2% outliers (1% each end)
            const int64_t lo = n / 100;
            const int64_t hi = n - 1 - lo;
            for (int i = 0; i < 3; ++i) {
                const auto sorted = visible_means.slice(1, i, i + 1).squeeze(1).sort(0, false).first;
                min_bounds[i] = sorted[lo].item() - padding;
                max_bounds[i] = sorted[hi].item() + padding;
            }
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
            // Exclude 2% outliers (1% each end)
            const int64_t lo = n / 100;
            const int64_t hi = n - 1 - lo;
            for (int i = 0; i < 3; ++i) {
                const auto sorted = means.slice(1, i, i + 1).squeeze(1).sort(0, false).first;
                min_bounds[i] = sorted[lo].item() - padding;
                max_bounds[i] = sorted[hi].item() + padding;
            }
        } else {
            for (int i = 0; i < 3; ++i) {
                const auto col = means.slice(1, i, i + 1).squeeze(1);
                min_bounds[i] = col.min().item() - padding;
                max_bounds[i] = col.max().item() + padding;
            }
        }

        return true;
    }

    SplatData extract_by_mask(const SplatData& splat_data, const Tensor& mask) {
        if (!splat_data._means.is_valid() || splat_data._means.size(0) == 0) {
            return SplatData();
        }
        if (!mask.is_valid() || mask.size(0) != splat_data._means.size(0)) {
            return SplatData();
        }

        const auto selection_mask = mask.to(DataType::Bool);
        const int count = selection_mask.sum_scalar();
        if (count == 0) {
            return SplatData();
        }

        auto indices = selection_mask.nonzero();
        if (indices.ndim() == 2) {
            indices = indices.squeeze(1);
        }

        Tensor shN_selected = splat_data._shN.is_valid()
                                  ? splat_data._shN.index_select(0, indices).contiguous()
                                  : Tensor{};

        SplatData result(
            splat_data._max_sh_degree,
            splat_data._means.index_select(0, indices).contiguous(),
            splat_data._sh0.index_select(0, indices).contiguous(),
            std::move(shN_selected),
            splat_data._scaling.index_select(0, indices).contiguous(),
            splat_data._rotation.index_select(0, indices).contiguous(),
            splat_data._opacity.index_select(0, indices).contiguous(),
            splat_data._scene_scale);
        result.set_active_sh_degree(splat_data._active_sh_degree);
        return result;
    }

} // namespace lfs::core

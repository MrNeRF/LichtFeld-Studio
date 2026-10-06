/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include <glm/glm.hpp>

namespace lfs::geometry {
    class BoundingBox;
}

namespace lfs::core {

    // Forward declarations
    class SplatData;
    class Tensor;
    struct PointCloud;

    /**
     * @brief Apply a transformation matrix to SplatData
     * @param splat_data The splat data to transform (modified in-place)
     * @param transform_matrix 4x4 transformation matrix
     * @return Reference to the modified splat_data
     */
    LFS_CORE_API SplatData& transform(SplatData& splat_data, const glm::mat4& transform_matrix);
    // Per-splat row-major Float32 [N,4,4] TRS transforms. Shares the affine
    // covariance factorization and SH basis/least-squares convention above.
    LFS_CORE_API SplatData& transform(SplatData& splat_data, const Tensor& transform_matrices);
    // Row i uses transform_matrices[matrix_index[i]]: Float32 [M,4,4] and Int32 [N]. SH rotations are fitted
    // once per matrix, so many rows sharing a matrix cost little more than the copy.
    LFS_CORE_API SplatData& transform(SplatData& splat_data, const Tensor& transform_matrices, const Tensor& matrix_index);

    /**
     * @brief transform() for attributes held with canonical SH (sh0 [N,3] or [N,1,3], shN [N,K,3])
     *
     * Replaces the tensors instead of writing into them, so they may be shared. Leaves the
     * scene scale to the caller: returns the uniform scale of a similarity transform, else 0.
     */
    LFS_CORE_API float transform_canonical(Tensor& means, Tensor& rotation, Tensor& scaling, Tensor& sh0,
                                           Tensor& shN, int sh_degree, const glm::mat4& transform_matrix);
    // Row i uses matrices[matrix_index[i]] (Float32 [M,4,4], Int32 [N]) on canonical tensors, on their own
    // device: means [N,3], rotation [N,4], scaling [N,3] and shN [N,K,3] when sh_degree > 0. SH rotations are
    // fitted once per matrix. Replaces the tensors instead of writing into them.
    LFS_CORE_API void transform_canonical(Tensor& means, Tensor& rotation, Tensor& scaling, Tensor& shN, int sh_degree,
                                          const Tensor& matrices, const Tensor& matrix_index);

    /**
     * @brief The scene scale transform() leaves after moving the means: their median distance from their
     * centre, unless that is within 10% of the current scale. For transforms that are not similarities.
     */
    LFS_CORE_API float transformed_scene_scale(const Tensor& means, float scene_scale);

    LFS_CORE_API Tensor compute_cropbox_mask(const Tensor& means,
                                             const glm::vec3& crop_min,
                                             const glm::vec3& crop_max,
                                             const glm::mat4& points_to_cropbox);

    // Soft crop: mark gaussians as deleted in-place (for undo/redo support)
    // Returns the applied deletion mask
    LFS_CORE_API Tensor soft_crop_by_cropbox(SplatData& splat_data,
                                             const lfs::geometry::BoundingBox& bounding_box,
                                             bool inverse = false);

    // Soft crop by ellipsoid: mark gaussians as deleted if outside ellipsoid
    // transform: world-to-ellipsoid-local transform (combined with node world transform)
    // radii: ellipsoid semi-axes
    LFS_CORE_API Tensor soft_crop_by_ellipsoid(SplatData& splat_data,
                                               const glm::mat4& transform,
                                               const glm::vec3& radii,
                                               bool inverse = false);

    /**
     * @brief Randomly select a subset of splats
     * @param splat_data The splat data to modify (modified in-place)
     * @param num_required_splat Number of splats to keep
     * @param seed Random seed for reproducibility (default: 0)
     */
    LFS_CORE_API void random_choose(SplatData& splat_data, int num_required_splat, int seed = 0);

    /**
     * @brief Compute the axis-aligned bounding box of SplatData
     * @param splat_data The splat data to compute bounds for
     * @param[out] min_bounds Output minimum corner (x, y, z)
     * @param[out] max_bounds Output maximum corner (x, y, z)
     * @param padding Optional padding to add around the bounds (default: 0.0f)
     * @param use_percentile If true, use 1st/99th percentile to exclude outliers (default: false)
     * @return true if bounds were computed successfully, false if splat data is empty/invalid
     */
    LFS_CORE_API bool compute_bounds(const SplatData& splat_data,
                                     glm::vec3& min_bounds,
                                     glm::vec3& max_bounds,
                                     float padding = 0.0f,
                                     bool use_percentile = false);

    /**
     * @brief Compute the axis-aligned bounding box of PointCloud
     * @param point_cloud The point cloud to compute bounds for
     * @param[out] min_bounds Output minimum corner (x, y, z)
     * @param[out] max_bounds Output maximum corner (x, y, z)
     * @param padding Optional padding to add around the bounds (default: 0.0f)
     * @param use_percentile If true, use 1st/99th percentile to exclude outliers (default: false)
     * @return true if bounds were computed successfully, false if point cloud is empty/invalid
     */
    LFS_CORE_API bool compute_bounds(const PointCloud& point_cloud,
                                     glm::vec3& min_bounds,
                                     glm::vec3& max_bounds,
                                     float padding = 0.0f,
                                     bool use_percentile = false);

    // Extract gaussians where mask is non-zero
    LFS_CORE_API SplatData extract_by_mask(const SplatData& splat_data, const Tensor& mask);

    // Copy a model's full SH layout into a float swizzled destination, retaining
    // inactive coefficients and decoding half/q16 storage on its own backend.
    LFS_CORE_API void copy_sh_coefficients(const SplatData& model, Tensor& destination,
                                           size_t destination_offset, uint32_t destination_rest);

} // namespace lfs::core

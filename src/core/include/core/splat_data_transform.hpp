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

    LFS_CORE_API Tensor compute_cropbox_mask(const Tensor& means,
                                             const glm::vec3& crop_min,
                                             const glm::vec3& crop_max,
                                             const glm::mat4& points_to_cropbox);

    /**
     * @brief Crop SplatData by a bounding box (creates new filtered copy)
     * @param splat_data The splat data to crop
     * @param bounding_box The bounding box to crop by
     * @param inverse If true, keep points outside the box instead of inside
     * @return New SplatData containing the selected points
     */
    LFS_CORE_API SplatData crop_by_cropbox(const SplatData& splat_data,
                                           const lfs::geometry::BoundingBox& bounding_box,
                                           bool inverse = false);

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

} // namespace lfs::core

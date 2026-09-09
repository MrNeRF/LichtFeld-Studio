/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/tensor.hpp"
#include <tuple>

namespace lfs::io {

    using lfs::core::Tensor;

    /**
     * @brief K-means over resident vksplat-swizzled shN (SOG path; non-swizzled API removed).
     *
     * @param shN_swizzled 1D swizzled SH-rest tensor
     * @param n_points Number of primitives
     * @param sh_coeffs Active SH-rest coefficient count (3, 8, or 15)
     * @param k Number of clusters
     * @param iterations Maximum iterations
     * @param fast_assignment Use the exact streamed-export SH3 assignment tile
     * @return Tuple of (centroids [k, sh_coeffs * 3], labels [n_points])
     */
    std::tuple<Tensor, Tensor> kmeans_sh_swizzled(
        const Tensor& shN_swizzled,
        int n_points,
        int sh_coeffs,
        int k,
        int iterations = 10,
        bool fast_assignment = false);

    // Internal CUDA launcher: SH3 swizzled rows, float32 centroids/norms and
    // preallocated int32 labels. Both tile shapes use identical FP32 dot products.
    void assign_sh3_labels(const Tensor& shN_swizzled, const Tensor& centroids,
                           const Tensor& centroid_norms, Tensor& labels, bool fast);

} // namespace lfs::io

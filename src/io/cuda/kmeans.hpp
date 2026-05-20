/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/tensor.hpp"
#include <tuple>

namespace lfs::io {

    using lfs::core::Tensor;

    /**
     * @brief K-means clustering using GPU acceleration
     *
     * @param data Input data [n_points, n_dims]
     * @param k Number of clusters
     * @param iterations Maximum iterations
     * @param tolerance Convergence tolerance (unused in GPU implementation)
     * @return Tuple of (centroids [k, n_dims], labels [n_points])
     */
    std::tuple<Tensor, Tensor> kmeans(
        const Tensor& data,
        int k,
        int iterations = 10,
        float tolerance = 1e-4f);

    /**
     * @brief K-means over resident vksplat-swizzled shN without materialising canonical rows.
     *
     * @param shN_swizzled 1D swizzled SH-rest tensor
     * @param n_points Number of primitives
     * @param sh_coeffs Active SH-rest coefficient count (3, 8, or 15)
     * @param k Number of clusters
     * @param iterations Maximum iterations
     * @return Tuple of (centroids [k, sh_coeffs * 3], labels [n_points])
     */
    std::tuple<Tensor, Tensor> kmeans_sh_swizzled(
        const Tensor& shN_swizzled,
        int n_points,
        int sh_coeffs,
        int k,
        int iterations = 10);

    /**
     * @brief 1D k-means using binary search for O(n log k) assignment
     *
     * @param data Input data [n_points] or [n_points, 1]
     * @param k Number of clusters
     * @param iterations Maximum iterations
     * @return Tuple of (sorted centroids [k, 1], labels [n_points])
     */
    std::tuple<Tensor, Tensor> kmeans_1d(
        const Tensor& data,
        int k,
        int iterations = 10);

} // namespace lfs::io

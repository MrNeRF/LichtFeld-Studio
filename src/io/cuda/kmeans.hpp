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
     * @param fast_assignment Use exact screened SH3 assignment for streamed exports
     * @return Tuple of (centroids [k, sh_coeffs * 3], labels [n_points])
     */
    std::tuple<Tensor, Tensor> kmeans_sh_swizzled(
        const Tensor& shN_swizzled,
        int n_points,
        int sh_coeffs,
        int k,
        int iterations = 10,
        bool fast_assignment = false);

    /**
     * @brief e-means (EMA mini-batch k-means, arXiv:2611.09501) over resident swizzled shN.
     *
     * Batch size and decay schedule follow the paper's scaling laws for (n, k, epochs).
     * Ends with an exact full assignment (screened at SH3) followed by a centroid
     * refit, so the palette is the mean of each label's members.
     *
     * @param epochs Number of passes over the shuffled data
     * @return Tuple of (centroids [k, sh_coeffs * 3], labels [n_points])
     */
    std::tuple<Tensor, Tensor> emeans_sh_swizzled(
        const Tensor& shN_swizzled,
        int n_points,
        int sh_coeffs,
        int k,
        int epochs = 10);

    // Mean squared quantization error per SH dimension of a swizzled shN palette assignment.
    double sh_palette_mse(const Tensor& shN_swizzled, int n_points, int sh_coeffs,
                          const Tensor& centroids, const Tensor& labels);

    // Internal CUDA launcher: SH3 swizzled rows, float32 centroids/norms and
    // preallocated int32 labels. Norms must be FP32 squared centroid norms.
    // Screening preserves FP32 winners; have_labels permits valid prior labels
    // as search hints (invalid hints are ignored).
    void assign_sh3_labels(const Tensor& shN_swizzled, const Tensor& centroids,
                           const Tensor& centroid_norms, Tensor& labels, bool fast, bool have_labels = false);

} // namespace lfs::io

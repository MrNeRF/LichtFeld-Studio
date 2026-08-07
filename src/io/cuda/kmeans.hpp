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
     * @return Tuple of (centroids [k, sh_coeffs * 3], labels [n_points])
     */
    std::tuple<Tensor, Tensor> kmeans_sh_swizzled(
        const Tensor& shN_swizzled,
        int n_points,
        int sh_coeffs,
        int k,
        int iterations = 10);

} // namespace lfs::io

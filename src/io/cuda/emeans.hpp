/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/tensor.hpp"
#include <tuple>

namespace lfs::io {

    using lfs::core::Tensor;

    /**
     * @brief e-means (EMA mini-batch k-means, arXiv:2611.09501) over resident swizzled shN.
     *
     * Batch size and decay schedule follow the paper's scaling laws with 10 epochs.
     * Ends with an exact full assignment (screened at SH3) followed by a centroid
     * refit, so the palette is the mean of each label's members.
     *
     * @return Tuple of (centroids [k, sh_coeffs * 3], labels [n_points])
     */
    std::tuple<Tensor, Tensor> emeans_sh_swizzled(
        const Tensor& shN_swizzled,
        int n_points,
        int sh_coeffs,
        int k);

    // Internal CUDA launcher: SH3 swizzled rows, float32 centroids/norms and
    // preallocated int32 labels. Norms must be FP32 squared centroid norms.
    // fast screens candidates in FP16 and preserves FP32 winners; otherwise a
    // brute-force FP32 search. have_labels permits valid prior labels as
    // search hints (invalid hints are ignored).
    void assign_sh3_labels(const Tensor& shN_swizzled, const Tensor& centroids,
                           const Tensor& centroid_norms, Tensor& labels, bool fast, bool have_labels = false);

} // namespace lfs::io

/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/tensor_export.hpp"

namespace lfs::io {
    inline std::tuple<lfs::core::Tensor, lfs::core::Tensor> kmeans_sh_swizzled(
        const lfs::core::Tensor& shN_swizzled, int n_points, int sh_coeffs, int k, int iterations = 10,
        bool fast_assignment = false) {
        return lfs::core::kmeans_sh(shN_swizzled, n_points, sh_coeffs, k, iterations, fast_assignment);
    }

    inline void assign_sh3_labels(const lfs::core::Tensor& shN_swizzled, const lfs::core::Tensor& centroids,
                                  const lfs::core::Tensor& centroid_norms, lfs::core::Tensor& labels, bool fast,
                                  bool have_labels = false) {
        lfs::core::assign_sh3(shN_swizzled, centroids, centroid_norms, labels, fast, have_labels);
    }
} // namespace lfs::io

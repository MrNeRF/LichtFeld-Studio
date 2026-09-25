/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

// CUDA entry points for the export ops. The kernel bodies are the dev
// implementations; only the enclosing namespace differs.
#include "core/decimate/types.hpp"
#include "core/tensor.hpp"

#include <tuple>

namespace lfs::core::export_cuda {
    Tensor morton_sort_indices_for_positions(const Tensor& positions, Tensor* sorted_keys);
    std::tuple<Tensor, Tensor> kmeans_sh_swizzled(const Tensor& shN_swizzled, int n_points, int sh_coeffs, int k,
                                                  int iterations, bool fast_assignment);
    void assign_sh3_labels(const Tensor& shN_swizzled, const Tensor& centroids, const Tensor& centroid_norms,
                           Tensor& labels, bool fast, bool have_labels = false);
    lfs::core::decimate::Candidates gpu_candidates(const lfs::core::decimate::Data& data);
    lfs::core::decimate::Data gpu_merge(const lfs::core::decimate::Data& data, const lfs::core::decimate::Selection& selection);
} // namespace lfs::core::export_cuda

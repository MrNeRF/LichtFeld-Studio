/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/export.hpp"
#include "core/tensor.hpp"

#include <cstdint>
#include <tuple>
#include <vector>

namespace lfs::core {

    // 63-bit Morton order of [N,3] positions. Equal keys keep source order.
    LFS_CORE_API Tensor morton_sort_indices(const Tensor& positions, Tensor* sorted_keys = nullptr);

    // SH palette. CUDA runs dev's kernels. Centroids are [k, coeffs*3], labels [n].
    LFS_CORE_API std::tuple<Tensor, Tensor> kmeans_sh(const Tensor& shN_swizzled, int n_points, int sh_coeffs, int k,
                                                      int iterations = 10, bool fast_assignment = false);

    // Exact (fast == false) or screened (fast == true) SH3 assignment. Screened
    // winners are the FP32 argmin; have_labels keeps a prior label as a hint.
    LFS_CORE_API void assign_sh3(const Tensor& shN_swizzled, const Tensor& centroids, const Tensor& centroid_norms,
                                 Tensor& labels, bool fast, bool have_labels = false);

    // 256-bin scalar Lloyd codebook. Sums are double and reduced in worker order.
    struct ScalarCodebook {
        std::vector<float> centroids;
        std::vector<uint8_t> labels;
    };
    LFS_CORE_API ScalarCodebook cluster_scalar(const float* data, int rows, int columns, int iterations, bool pooled);

    // One decimation generation. idx/cost are n*4 host tables. Merge writes a
    // new row set on the same backend; removed rows are the ones selection drops.
    struct DecimateMerge {
        Tensor position, rotation, scale, opacity, dc, sh;
    };
    LFS_CORE_API void decimate_candidates(const Tensor& position, const Tensor& rotation, const Tensor& scale,
                                          const Tensor& opacity, const Tensor& dc, const Tensor& sh, int rest,
                                          std::vector<uint32_t>& idx, std::vector<float>& cost);
    LFS_CORE_API DecimateMerge decimate_merge(const Tensor& position, const Tensor& rotation, const Tensor& scale,
                                              const Tensor& opacity, const Tensor& dc, const Tensor& sh, int rest,
                                              const std::vector<int>& member_group, const std::vector<uint32_t>& minimum,
                                              const std::vector<uint32_t>& members, const std::vector<uint32_t>& offsets,
                                              size_t removed);
} // namespace lfs::core

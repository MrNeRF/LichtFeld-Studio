/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/tensor.hpp"

#include <cstdint>
#include <cuda_runtime.h>

namespace lfs::training::kernels {

    /// 30-bit Morton argsort of [N, 3] CUDA float means. Returns Int64 [N]
    /// permutation such that dest[i] <- src[perm[i]].
    [[nodiscard]] lfs::core::Tensor launch_morton_permutation(
        const lfs::core::Tensor& means,
        cudaStream_t stream = nullptr);

    /// Decode joint packed rows under source 256-splat bounds, write dest packed
    /// with freshly reduced dest-block bounds. src/dst packed must not alias.
    void launch_joint_permute_contiguous(
        const std::uint8_t* src_packed,
        const float* src_bounds,
        std::uint8_t* dst_packed,
        float* dst_bounds,
        const std::int64_t* perm,
        int n_prims,
        int n_attr,
        int bits,
        cudaStream_t stream = nullptr);

    /// Same for float4-swizzled SH moment cells (8-bit joint).
    void launch_joint_permute_shN(
        const std::uint8_t* src_packed,
        const float* src_bounds,
        std::uint8_t* dst_packed,
        float* dst_bounds,
        const std::int64_t* perm,
        int n_prims,
        int slots_per_primitive,
        int bits,
        cudaStream_t stream = nullptr);

} // namespace lfs::training::kernels

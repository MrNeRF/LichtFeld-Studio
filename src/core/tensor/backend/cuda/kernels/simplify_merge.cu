/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/cuda_error.hpp"
#include "simplify_merge.hpp"

#include <cfloat>

// One thread per group: the moment-matching merge of splat_simplify.cpp (merge_voxel_groups), with the
// same order of accumulation over the members.
namespace lfs::core::tensor_ops {
    namespace {
#include "core/device/simplify_merge.cuh"
    } // namespace

    void launch_simplify_merge(const float* means, const float* scales, const float* rotation, const float* opacity,
                               const float* appearance, const int32_t* offsets, const int32_t* members, float* out_means,
                               float* out_scales, float* out_rotation, float* out_opacity, float* out_appearance,
                               const uint32_t groups, const uint32_t app_dim, const cudaStream_t stream) {
        if (groups == 0)
            return;
        simplify_merge_kernel<<<(groups + kBlock - 1) / kBlock, kBlock, 0, stream>>>(
            means, scales, rotation, opacity, appearance, offsets, members, out_means, out_scales, out_rotation,
            out_opacity, out_appearance, groups, app_dim);
        LFS_CUDA_LAUNCH_CHECK(stream, "tensor.simplify_merge");
    }
} // namespace lfs::core::tensor_ops

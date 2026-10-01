/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor/backend/cuda/kernels/warp_reduce.cuh"

namespace {

    __global__ void consecutive_reductions_kernel(float* output) {
        const float x = static_cast<float>(threadIdx.x + 1);
        const int base = static_cast<int>(blockIdx.x) * 8;

        const float sum_a = lfs::core::warp_ops::block_reduce_sum(x);
        const float sum_b = lfs::core::warp_ops::block_reduce_sum(-2.0f * x);
        const float max_a = lfs::core::warp_ops::block_reduce_max(x);
        const float max_b = lfs::core::warp_ops::block_reduce_max(-x);
        const float min_a = lfs::core::warp_ops::block_reduce_min(x);
        const float min_b = lfs::core::warp_ops::block_reduce_min(-x);
        const float prod_a = lfs::core::warp_ops::block_reduce_prod(1.0f);
        const float prod_b = lfs::core::warp_ops::block_reduce_prod(1.0001f);

        if (threadIdx.x == 0) {
            output[base + 0] = sum_a;
            output[base + 1] = sum_b;
            output[base + 2] = max_a;
            output[base + 3] = max_b;
            output[base + 4] = min_a;
            output[base + 5] = min_b;
            output[base + 6] = prod_a;
            output[base + 7] = prod_b;
        }
    }

} // namespace

void launch_consecutive_tensor_warp_reductions(float* output, int blocks) {
    consecutive_reductions_kernel<<<blocks, 256>>>(output);
}

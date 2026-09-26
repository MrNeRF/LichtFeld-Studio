/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/cuda_error.hpp"
#include "tensor_labels.hpp"

namespace lfs::core::tensor_ops {
    namespace {
        constexpr unsigned kBlockSize = 256;
        template <uint32_t Phase>
        __global__ void updateLabelsKernel(LabelUpdateParams p) {
            const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
            if (row < p.count)
                internal::updateLabel<Phase>(p, row);
        }
    } // namespace
    void launch_update_labels(const LabelUpdateParams& p, cudaStream_t stream) {
        const unsigned blocks = (p.count + kBlockSize - 1) / kBlockSize;
        if (!p.indices) {
            updateLabelsKernel<0><<<blocks, kBlockSize, 0, stream>>>(p);
            LFS_CUDA_LAUNCH_CHECK(stream, "tensor.update_labels");
            return;
        }
        if (p.mode == 2) {
            updateLabelsKernel<1><<<blocks, kBlockSize, 0, stream>>>(p);
            LFS_CUDA_LAUNCH_CHECK(stream, "tensor.update_labels.clear_indexed");
        }
        updateLabelsKernel<2><<<blocks, kBlockSize, 0, stream>>>(p);
        LFS_CUDA_LAUNCH_CHECK(stream, "tensor.update_labels.indexed");
    }
} // namespace lfs::core::tensor_ops

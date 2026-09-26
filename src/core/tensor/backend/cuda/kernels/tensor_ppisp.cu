/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "../../../internal/ppisp.hpp"
#include "core/cuda_error.hpp"
#include "tensor_ppisp.hpp"
namespace lfs::core::tensor_ops {
    namespace {
        __global__ void ppisp_kernel(const float* src, float* dst, int width, int height, PpispParams p) {
            const int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < width * height)
                internal::ppisp_pixel(src, dst, width, height, p, i);
        }
    } // namespace
    void launch_ppisp_apply(const float* input, float* output, int width, int height, const PpispParams& p, cudaStream_t stream) {
        ppisp_kernel<<<(width * height + 255) / 256, 256, 0, stream>>>(input, output, width, height, p);
        LFS_CUDA_LAUNCH_CHECK(stream, "tensor.ppisp_apply");
    }
} // namespace lfs::core::tensor_ops

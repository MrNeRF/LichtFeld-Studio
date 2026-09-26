/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "../../../internal/environment_composite.hpp"
#include "core/cuda_error.hpp"
#include "tensor_environment.hpp"
namespace lfs::core::tensor_ops {
    namespace {
        __global__ void composite_environment_kernel(EnvironmentCompositeParams p, const float* env, const float* rgb,
                                                     const float* alpha, unsigned char* dst) {
            const int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < p.band_width * p.band_height)
                internal::composite_environment_pixel(p, env, rgb, alpha, dst, idx);
        }
    } // namespace
    void launch_environment_composite(const EnvironmentCompositeParams& p, const float* env, const float* rgb,
                                      const float* alpha, unsigned char* dst, cudaStream_t stream) {
        composite_environment_kernel<<<(p.band_width * p.band_height + 255) / 256, 256, 0, stream>>>(p, env, rgb, alpha, dst);
        LFS_CUDA_LAUNCH_CHECK(stream, "tensor.environment_composite");
    }
} // namespace lfs::core::tensor_ops

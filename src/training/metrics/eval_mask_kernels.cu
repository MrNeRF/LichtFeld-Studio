/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "eval_mask_kernels.cuh"

#include "core/tensor/internal/cuda_stream_context.hpp"

#include <cassert>
#include <cstdint>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <nvtx3/nvToolsExt.h>

namespace lfs::training {
    namespace {

        constexpr int BLOCK_DIM = 16;

        __global__ void erode_metrics_mask_kernel(
            const uint8_t* __restrict__ src,
            uint8_t* __restrict__ dst,
            const int width,
            const int height,
            const int radius) {
            const int x = blockIdx.x * BLOCK_DIM + threadIdx.x;
            const int y = blockIdx.y * BLOCK_DIM + threadIdx.y;
            if (x >= width || y >= height)
                return;

            bool keep = true;
            for (int dy = -radius; dy <= radius && keep; ++dy) {
                const int sy = y + dy;
                if (sy < 0 || sy >= height) {
                    keep = false;
                    break;
                }
                for (int dx = -radius; dx <= radius; ++dx) {
                    const int sx = x + dx;
                    if (sx < 0 || sx >= width || src[sy * width + sx] == 0) {
                        keep = false;
                        break;
                    }
                }
            }
            dst[y * width + x] = keep ? 1 : 0;
        }

    } // namespace

    lfs::core::Tensor erode_metrics_mask(
        const lfs::core::Tensor& mask, const int radius, const cudaStream_t stream) {
        assert(mask.is_valid());
        assert(mask.ndim() == 2);
        assert(mask.device() == lfs::core::Device::CUDA);
        assert(mask.dtype() == lfs::core::DataType::UInt8 ||
               mask.dtype() == lfs::core::DataType::Bool);
        assert(radius >= 0);

        nvtxRangePush("erode_metrics_mask");
        const lfs::core::CUDAStreamGuard stream_guard(stream);
        const auto input = mask.contiguous();
        input.sync_to_stream(stream);
        const int height = static_cast<int>(input.shape()[0]);
        const int width = static_cast<int>(input.shape()[1]);
        auto result = lfs::core::Tensor::empty(
            {static_cast<size_t>(height), static_cast<size_t>(width)},
            lfs::core::Device::CUDA, lfs::core::DataType::UInt8);
        const dim3 block(BLOCK_DIM, BLOCK_DIM);
        const dim3 grid((width + BLOCK_DIM - 1) / BLOCK_DIM,
                        (height + BLOCK_DIM - 1) / BLOCK_DIM);
        erode_metrics_mask_kernel<<<grid, block, 0, stream>>>(
            input.ptr<uint8_t>(), result.ptr<uint8_t>(), width, height, radius);
        const cudaError_t error = cudaGetLastError();
        assert(error == cudaSuccess && "erode_metrics_mask_kernel launch failed");
        nvtxRangePop();
        return result;
    }

} // namespace lfs::training

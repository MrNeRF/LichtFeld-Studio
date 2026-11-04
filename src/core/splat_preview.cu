/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/cuda_error.hpp"
#include "core/sh_value_codec.cuh"
#include "splat_preview_cuda.hpp"
#include <algorithm>
#include <cuda_fp16.h>

namespace lfs::core::detail {
    namespace {
        __global__ void gather_sh(const void* source, const float* bounds, bool half,
                                  const int64_t* rows, size_t count, uint32_t layout_cells,
                                  uint32_t selected_cells, float* output) {
            for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
                 i < count * selected_cells; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
                const size_t row = static_cast<size_t>(rows[i / selected_cells]);
                const size_t cell = i % selected_cells;
                const size_t block = row / 32u;
                const size_t lane = row % 32u;
                if (bounds) {
                    const size_t offset = block * layout_cells * 32u + cell * 32u + lane;
                    const auto q = static_cast<const uint16_t*>(source)[offset];
                    const float lo = bounds[(row / 256u) * 2u];
                    const float hi = bounds[(row / 256u) * 2u + 1u];
                    output[i] = sh_value::DeviceCodec16::decode_host_rounded(q, lo, hi);
                } else {
                    const size_t slots = (layout_cells + 3u) / 4u;
                    const size_t offset = (block * slots * 32u + (cell / 4u) * 32u + lane) * 4u + cell % 4u;
                    output[i] = half ? __half2float(static_cast<const __half*>(source)[offset])
                                     : static_cast<const float*>(source)[offset];
                }
            }
        }
    } // namespace

    void gather_preview_sh(const void* source, const float* bounds, bool half,
                           const int64_t* rows, size_t count, uint32_t layout_rest,
                           uint32_t selected_rest, float* output, cudaStream_t stream) {
        constexpr size_t threads = 256;
        const auto blocks = static_cast<unsigned int>(std::min<size_t>(4096, (count * selected_rest * 3u + threads - 1) / threads));
        gather_sh<<<blocks, threads, 0, stream>>>(source, bounds, half, rows, count,
                                                  layout_rest * 3u, selected_rest * 3u, output);
        LFS_CUDA_LAUNCH_CHECK(stream, "core.splat_preview.gather_sh");
    }
} // namespace lfs::core::detail

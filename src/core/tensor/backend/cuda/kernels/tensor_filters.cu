/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/cuda_error.hpp"
#include "tensor_filters.hpp"
namespace lfs::core::tensor_ops {
    namespace {
        template <bool NodeOnly>
        __global__ void filterPointsKernel(internal::PointFilterArgs p) {
            if constexpr (NodeOnly)
                p.flags = LFS_FILTER_NODES;
            constexpr uint32_t elements = NodeOnly ? 4 : 1;
            const uint32_t start = blockIdx.x * blockDim.x * elements + threadIdx.x;
#pragma unroll
            for (uint32_t element = 0; element < elements; ++element) {
                const uint32_t i = start + element * blockDim.x;
                if (i < p.count && p.mask[i] && !internal::pointPassesFilter(p, i))
                    p.mask[i] = 0;
            }
        }
    } // namespace
    void launch_filter_points(const internal::PointFilterArgs& args, cudaStream_t stream) {
        const auto blocks = (args.count + 255) / 256;
        if (args.flags & LFS_FILTER_GEOMETRY)
            filterPointsKernel<false><<<blocks, 256, 0, stream>>>(args);
        else
            filterPointsKernel<true><<<(args.count + 1023) / 1024, 256, 0, stream>>>(args);
        LFS_CUDA_LAUNCH_CHECK(stream, "tensor.filter_points");
    }
} // namespace lfs::core::tensor_ops

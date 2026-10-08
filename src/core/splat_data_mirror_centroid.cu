/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/cuda_error.hpp"
#include "core/tensor.hpp"
#include "core/tensor_cuda_interop.hpp"
#include "splat_data_mirror_centroid.hpp"
#include <algorithm>
#include <cassert>

namespace lfs::core::detail {
    namespace {
        constexpr int THREADS = 256;
        constexpr int MAX_BLOCKS = 256;

        __device__ void reduce_centroid(double (&sum)[4], double (&shared)[4][THREADS]) {
            const int t = threadIdx.x;
            for (int axis = 0; axis < 4; ++axis)
                shared[axis][t] = sum[axis];
            __syncthreads();
            for (int stride = THREADS / 2; stride > 0; stride /= 2) {
                if (t < stride)
                    for (int axis = 0; axis < 4; ++axis)
                        shared[axis][t] += shared[axis][t + stride];
                __syncthreads();
            }
        }

        __global__ void centroid_partials(const float* means, const bool* selected,
                                          const size_t count, double* partials) {
            __shared__ double shared[4][THREADS];
            double sum[4]{};
            for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
                 i += static_cast<size_t>(gridDim.x) * blockDim.x) {
                if (!selected[i])
                    continue;
                for (int axis = 0; axis < 3; ++axis)
                    sum[axis] += static_cast<double>(means[i * 3 + axis]);
                sum[3] += 1.0;
            }
            reduce_centroid(sum, shared);
            if (threadIdx.x == 0)
                for (int axis = 0; axis < 4; ++axis)
                    partials[blockIdx.x * 4 + axis] = shared[axis][0];
        }

        __global__ void centroid_finish(const double* partials, const int blocks, float* output) {
            __shared__ double shared[4][THREADS];
            double sum[4]{};
            if (threadIdx.x < blocks)
                for (int axis = 0; axis < 4; ++axis)
                    sum[axis] = partials[threadIdx.x * 4 + axis];
            reduce_centroid(sum, shared);
            if (threadIdx.x == 0)
                for (int axis = 0; axis < 3; ++axis)
                    output[axis] = shared[3][0] > 0.0
                                       ? static_cast<float>(shared[axis][0] / shared[3][0])
                                       : 0.0f;
        }
    } // namespace

    glm::vec3 selected_centroid_cuda(const Tensor& means, const Tensor& selected) {
        assert(means.dtype() == DataType::Float32 && means.is_contiguous());
        assert(selected.dtype() == DataType::Bool && selected.is_contiguous());
        const auto stream = prepare_inputs_for_stream({&means, &selected});
        const CUDAStreamGuard guard(stream);
        const int blocks = static_cast<int>(std::min((means.size(0) + THREADS - 1) / THREADS,
                                                     static_cast<size_t>(MAX_BLOCKS)));
        auto partials = Tensor::empty({static_cast<size_t>(blocks) * 4 * sizeof(double)}, Device::GPU, DataType::UInt8);
        auto* partial_sums = static_cast<double*>(partials.data_ptr());
        auto output = Tensor::empty({size_t{3}}, Device::GPU, DataType::Float32);
        centroid_partials<<<blocks, THREADS, 0, stream>>>(means.ptr<float>(), selected.ptr<bool>(), means.size(0), partial_sums);
        LFS_CUDA_LAUNCH_CHECK(stream, "core.mirror.centroid_partials");
        centroid_finish<<<1, THREADS, 0, stream>>>(partial_sums, blocks, output.ptr<float>());
        LFS_CUDA_LAUNCH_CHECK(stream, "core.mirror.centroid_finish");
        const auto cpu = output.cpu();
        const auto* values = cpu.ptr<float>();
        return {values[0], values[1], values[2]};
    }
} // namespace lfs::core::detail

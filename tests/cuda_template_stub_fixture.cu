/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor/internal/tensor_broadcast_ops.cuh"
#include "core/tensor/internal/tensor_functors.hpp"
#include <cassert>

namespace cuda_stub_test::LFS_STUB_TEST_VARIANT {

    const void* division_stub() {
        return reinterpret_cast<const void*>(
            lfs::core::tensor_ops::broadcast_2d_flat_kernel<float, float, lfs::core::ops::div_op>);
    }

    __global__ void reference_division(const float* a, const float* b, float* output, const size_t count) {
        const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
        if (i < count)
            output[i] = a[i] / b[i];
    }

    cudaError_t divide(const float* a, const float* b, float* output, float* reference, const size_t count) {
        assert(count > 0 && count % 3 == 0);
        const auto blocks = static_cast<unsigned>((count + 255) / 256);
        lfs::core::tensor_ops::broadcast_2d_flat_kernel<<<blocks, 256>>>(
            a, b, output, count / 3, size_t{3}, 0, 0, lfs::core::ops::div_op{});
        if (const auto status = cudaGetLastError(); status != cudaSuccess)
            return status;
        reference_division<<<blocks, 256>>>(a, b, reference, count);
        return cudaGetLastError();
    }

} // namespace cuda_stub_test::LFS_STUB_TEST_VARIANT

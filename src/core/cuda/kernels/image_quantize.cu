/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/assert.hpp"
#include "core/cuda/image_quantize.hpp"
#include "core/cuda_error.hpp"
#include "core/gpu_backend_fwd.hpp"
#include "core/tensor_cuda_interop.hpp"
#include <cstdint>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

namespace lfs::core::cuda {

    namespace {

        constexpr int kThreads = 256;

        // Rounded multiply and add keep the bytes identical to the tensor expression
        // (image.clamp(0, 1) * 255 + 0.5).to(UInt8), which never contracts to an FMA.
        __global__ void quantize_to_interleaved_bytes_kernel(const float* __restrict__ planes,
                                                             std::uint8_t* __restrict__ pixels,
                                                             const int channels,
                                                             const size_t plane_size) {
            const size_t pixel = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (pixel >= plane_size)
                return;
            for (int c = 0; c < channels; ++c) {
                const float value = fminf(fmaxf(planes[c * plane_size + pixel], 0.0f), 1.0f);
                pixels[pixel * channels + c] = static_cast<std::uint8_t>(__fadd_rn(__fmul_rn(value, 255.0f), 0.5f));
            }
        }

    } // namespace

    Tensor quantize_to_interleaved_bytes(const Tensor& image) {
        LFS_ASSERT(image.device() == Device::CUDA && image.dtype() == DataType::Float32);
        LFS_ASSERT(image.ndim() == 3 && image.size(0) > 0 && image.size(0) <= 4);

        const GpuBackendScope backend_scope(GpuBackend::CUDA);
        const cudaStream_t stream = image.stream();
        const CUDAStreamGuard stream_guard(stream);
        const auto planes = image.contiguous();
        planes.sync_to_stream(stream);
        const size_t channels = planes.size(0);
        const size_t height = planes.size(1);
        const size_t width = planes.size(2);
        auto pixels = Tensor::empty({height, width, channels}, Device::CUDA, DataType::UInt8);
        const size_t plane_size = height * width;
        if (plane_size == 0)
            return pixels;
        const auto blocks = static_cast<unsigned>((plane_size + kThreads - 1) / kThreads);
        nvtxRangePush("quantize_to_interleaved_bytes");
        quantize_to_interleaved_bytes_kernel<<<blocks, kThreads, 0, stream>>>(
            planes.ptr<float>(), pixels.ptr<std::uint8_t>(), static_cast<int>(channels), plane_size);
        LFS_CUDA_CHECK_MSG(cudaGetLastError(), "quantize image to interleaved bytes");
        nvtxRangePop();
        return pixels;
    }

} // namespace lfs::core::cuda

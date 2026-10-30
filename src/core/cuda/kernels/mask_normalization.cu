/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/assert.hpp"
#include "core/cuda/mask_normalization.cuh"
#include "core/cuda/mask_normalization.hpp"
#include "core/cuda_error.hpp"
#include "core/tensor_cuda_interop.hpp"

namespace lfs::core::cuda {
    namespace {
        constexpr unsigned threads = 256;
        template <bool Aligned>
        __global__ void normalize_mask_bytes_kernel(const std::uint8_t* __restrict__ input, float* __restrict__ output, size_t count) {
            const size_t i = (static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 4;
            if (i + 3 < count) {
                uchar4 bytes;
                if constexpr (Aligned)
                    bytes = reinterpret_cast<const uchar4*>(input)[i / 4];
                else
                    bytes = make_uchar4(input[i], input[i + 1], input[i + 2], input[i + 3]);
                reinterpret_cast<float4*>(output)[i / 4] = make_float4(
                    normalize_mask_byte(bytes.x), normalize_mask_byte(bytes.y),
                    normalize_mask_byte(bytes.z), normalize_mask_byte(bytes.w));
            } else {
                for (size_t j = i; j < count; ++j)
                    output[j] = normalize_mask_byte(input[j]);
            }
        }
        __global__ void threshold_mask_kernel(const float* input, std::uint8_t* output, size_t count, float threshold) {
            const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i < count)
                output[i] = mask_value_at_least(input[i], threshold);
        }
        __global__ void normalize_mask_u16_kernel(const std::uint16_t* input, float* output, size_t count) {
            const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i < count)
                output[i] = normalize_mask_u16_value(input[i]);
        }
#include "core/device/mask_grid.cuh"
        __global__ void mask_values_to_bytes_kernel(const float* values, std::uint8_t* bytes, size_t count) {
            const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i < count)
                bytes[i] = mask_grid_byte(values[i]);
        }
    } // namespace

    void normalize_mask_bytes(const Tensor& mask, Tensor& output) {
        LFS_ASSERT(mask.device() == Device::CUDA && mask.dtype() == DataType::UInt8);
        LFS_ASSERT(output.device() == Device::CUDA && output.dtype() == DataType::Float32 && output.shape() == mask.shape());
        const auto stream = mask.stream();
        const CUDAStreamGuard guard(stream);
        const auto bytes = mask.contiguous();
        bytes.sync_to_stream(stream);
        if (bytes.numel()) {
            const auto blocks = static_cast<unsigned>((bytes.numel() + threads * 4 - 1) / (threads * 4));
            if ((reinterpret_cast<std::uintptr_t>(bytes.data_ptr()) & 3u) == 0)
                normalize_mask_bytes_kernel<true><<<blocks, threads, 0, stream>>>(bytes.ptr<std::uint8_t>(), output.ptr<float>(), bytes.numel());
            else
                normalize_mask_bytes_kernel<false><<<blocks, threads, 0, stream>>>(bytes.ptr<std::uint8_t>(), output.ptr<float>(), bytes.numel());
            LFS_CUDA_CHECK_MSG(cudaGetLastError(), "normalize mask bytes");
        }
    }
    Tensor normalize_mask_bytes(const Tensor& mask) {
        auto output = Tensor::empty(mask.shape(), Device::CUDA, DataType::Float32);
        normalize_mask_bytes(mask, output);
        return output;
    }
    Tensor threshold_mask(const Tensor& mask, const float threshold) {
        LFS_ASSERT(mask.device() == Device::CUDA && mask.dtype() == DataType::Float32);
        const auto stream = mask.stream();
        const CUDAStreamGuard guard(stream);
        const auto values = mask.contiguous();
        values.sync_to_stream(stream);
        auto output = Tensor::empty(values.shape(), Device::CUDA, DataType::UInt8);
        if (values.numel()) {
            const auto blocks = static_cast<unsigned>((values.numel() + threads - 1) / threads);
            threshold_mask_kernel<<<blocks, threads, 0, stream>>>(values.ptr<float>(), output.ptr<std::uint8_t>(), values.numel(), threshold);
            LFS_CUDA_CHECK_MSG(cudaGetLastError(), "threshold mask");
        }
        return output;
    }
    void normalize_mask_u16(const std::uint16_t* input, float* output, const size_t count, cudaStream_t stream) {
        if (!count)
            return;
        normalize_mask_u16_kernel<<<static_cast<unsigned>((count + threads - 1) / threads), threads, 0, stream>>>(input, output, count);
        LFS_CUDA_CHECK_MSG(cudaGetLastError(), "normalize 16-bit mask");
    }
    Tensor quantize_mask_to_byte_grid(Tensor& mask, const bool keep_bytes) {
        LFS_ASSERT(mask.device() == Device::CUDA && mask.dtype() == DataType::Float32);
        const auto stream = mask.stream();
        const CUDAStreamGuard guard(stream);
        mask = mask.contiguous();
        mask.sync_to_stream(stream);
        auto bytes = keep_bytes ? Tensor::empty(mask.shape(), Device::CUDA, DataType::UInt8) : Tensor{};
        if (mask.numel()) {
            const auto blocks = static_cast<unsigned>((mask.numel() + threads - 1) / threads);
            if (keep_bytes)
                quantize_mask_grid_kernel<true><<<blocks, threads, 0, stream>>>(mask.ptr<float>(), bytes.ptr<std::uint8_t>(), mask.numel());
            else
                quantize_mask_grid_kernel<false><<<blocks, threads, 0, stream>>>(mask.ptr<float>(), nullptr, mask.numel());
            LFS_CUDA_CHECK_MSG(cudaGetLastError(), "quantize mask cache grid");
        }
        return bytes;
    }
    Tensor mask_values_to_byte_grid(const Tensor& mask) {
        LFS_ASSERT(mask.device() == Device::CUDA && mask.dtype() == DataType::Float32);
        const auto stream = mask.stream();
        const CUDAStreamGuard guard(stream);
        const auto values = mask.contiguous();
        values.sync_to_stream(stream);
        auto bytes = Tensor::empty(values.shape(), Device::CUDA, DataType::UInt8);
        if (values.numel()) {
            const auto blocks = static_cast<unsigned>((values.numel() + threads - 1) / threads);
            mask_values_to_bytes_kernel<<<blocks, threads, 0, stream>>>(values.ptr<float>(), bytes.ptr<std::uint8_t>(), values.numel());
            LFS_CUDA_CHECK_MSG(cudaGetLastError(), "quantize shared mask source");
        }
        return bytes;
    }
} // namespace lfs::core::cuda

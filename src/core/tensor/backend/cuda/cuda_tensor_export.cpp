/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "../native_export.hpp"
#include "core/tensor.hpp"
#include "core/tensor_backend.hpp"
#include "core/tensor_cuda_interop.hpp"
#include <cuda_runtime.h>
namespace lfs::core::internal {
    TensorExportDevice cuda_dlpack_device(const Tensor& tensor) {
        const auto* pointer = tensor.data_ptr();
        if (!pointer)
            return {2, selected_cuda_device()};
        cudaPointerAttributes attributes{};
        if (const auto status = cudaPointerGetAttributes(&attributes, pointer); status != cudaSuccess)
            throw std::runtime_error(std::string("DLPack CUDA device query failed: ") + cudaGetErrorString(status));
        return {2, attributes.device};
    }
    cudaStream_t dlpack_cuda_stream(int64_t stream) {
        if (stream == 0 || stream == 1)
            return nullptr;
        if (stream == 2)
            return cudaStreamPerThread;
        return reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(stream));
    }
    Tensor cuda_prepare_dlpack_export(const Tensor& tensor, std::optional<int64_t> stream) {
        const auto home = tensor.stream();
        if (!stream) {
            if (const auto status = cudaStreamSynchronize(home); status != cudaSuccess)
                throw std::runtime_error(std::string("DLPack CUDA synchronization failed: ") + cudaGetErrorString(status));
        } else if (*stream != -1) {
            const auto consumer = dlpack_cuda_stream(*stream);
            if (consumer != home)
                bridgeStreams(home, consumer);
        }
        return tensor;
    }
} // namespace lfs::core::internal

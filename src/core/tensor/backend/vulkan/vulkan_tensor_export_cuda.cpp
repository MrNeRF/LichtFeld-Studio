/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "../native_export.hpp"
#include "core/tensor.hpp"
#include "core/tensor_backend.hpp"
#include "core/tensor_backend_vulkan.hpp"
#include "core/tensor_cuda_interop.hpp"
#include <cuda_runtime.h>
namespace lfs::core::internal {
    cudaStream_t dlpack_cuda_stream(int64_t);
    TensorExportCapabilities vulkan_dlpack_capabilities() noexcept { return {.dlpack = true}; }
    TensorExportDevice vulkan_dlpack_device(const Tensor&) {
        if (!vulkan_backend_exports_memory())
            throw std::runtime_error("DLPack export of a Vulkan tensor requires CUDA external memory; call .cpu() first");
        return {2, selected_cuda_device()};
    }
    Tensor vulkan_prepare_dlpack_export(const Tensor& tensor, std::optional<int64_t> stream) {
        const auto consumer = !stream || *stream == -1 ? getCurrentCUDAStream() : dlpack_cuda_stream(*stream);
        auto view = cuda_view_of_vulkan_tensor(tensor, consumer);
        if (!view)
            throw std::runtime_error("DLPack export of a Vulkan tensor requires CUDA external memory: " + lfs::format_for_developer(view.error()));
        if (!stream) {
            if (const auto status = cudaStreamSynchronize(consumer); status != cudaSuccess)
                throw std::runtime_error(std::string("DLPack CUDA synchronization failed: ") + cudaGetErrorString(status));
        }
        return std::move(*view);
    }
} // namespace lfs::core::internal

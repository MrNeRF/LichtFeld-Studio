/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "vulkan_external_tensor.hpp"

#include <algorithm>
#include <cuda_runtime.h>
#include <format>

namespace lfs::vis {

    namespace {
        [[nodiscard]] std::size_t rowSize(const lfs::core::TensorShape& shape) {
            if (shape.rank() == 0) {
                return 1;
            }
            std::size_t row_size = 1;
            for (std::size_t i = 1; i < shape.rank(); ++i) {
                row_size *= shape[i];
            }
            return row_size;
        }
    } // namespace

    VulkanExternalTensorStorage::VulkanExternalTensorStorage(
        VulkanContext& context,
        VulkanContext::ExternalBuffer buffer,
        lfs::rendering::CudaVulkanBufferInterop interop,
        const std::size_t bytes)
        : context_(&context),
          buffer_(buffer),
          interop_(std::move(interop)),
          bytes_(bytes) {}

    VulkanExternalTensorStorage::~VulkanExternalTensorStorage() {
        interop_.reset();
        if (context_) {
            context_->destroyExternalBuffer(buffer_);
        }
    }

    std::expected<lfs::core::Tensor, std::string> makeVulkanExternalTensor(
        VulkanContext& context,
        lfs::core::TensorShape shape,
        const lfs::core::DataType dtype,
        const std::size_t capacity,
        const char* const debug_name,
        const cudaStream_t stream) {
        if (!context.externalMemoryInteropEnabled()) {
            return std::unexpected("Vulkan external tensor allocation requires CUDA/Vulkan external-memory interop");
        }
        if (shape.rank() == 0) {
            return std::unexpected("Vulkan external tensor allocation requires a non-scalar tensor shape");
        }

        const std::size_t rows = shape[0];
        const std::size_t cap_rows = std::max(capacity, rows);
        const std::size_t total_elements = cap_rows * rowSize(shape);
        const std::size_t bytes = total_elements * lfs::core::dtype_size(dtype);
        if (bytes == 0) {
            return std::unexpected("Vulkan external tensor allocation requested zero bytes");
        }

        VulkanContext::ExternalBuffer buffer{};
        constexpr VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                             VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                             VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        if (!context.createExternalBuffer(static_cast<VkDeviceSize>(bytes), usage, buffer)) {
            return std::unexpected(std::format("Vulkan external tensor '{}' allocation failed: {}",
                                               debug_name ? debug_name : "<unnamed>",
                                               context.lastError()));
        }

        const auto native = context.releaseExternalBufferNativeHandle(buffer);
        if (!VulkanContext::externalNativeHandleValid(native)) {
            context.destroyExternalBuffer(buffer);
            return std::unexpected(std::format("Vulkan external tensor '{}' returned an invalid native handle",
                                               debug_name ? debug_name : "<unnamed>"));
        }

        lfs::rendering::CudaVulkanBufferInterop interop;
        const lfs::rendering::CudaVulkanExternalBufferImport import{
            .memory_handle = native,
            .allocation_size = static_cast<std::size_t>(buffer.allocation_size),
            .size = static_cast<std::size_t>(buffer.size),
            .dedicated_allocation = context.externalMemoryDedicatedAllocationEnabled(),
        };
        if (!interop.init(import)) {
            const std::string error = interop.lastError();
            context.destroyExternalBuffer(buffer);
            return std::unexpected(std::format("Vulkan external tensor '{}' CUDA import failed: {}",
                                               debug_name ? debug_name : "<unnamed>",
                                               error));
        }

        void* const cuda_ptr = interop.devicePointer();
        if (!cuda_ptr) {
            interop.reset();
            context.destroyExternalBuffer(buffer);
            return std::unexpected(std::format("Vulkan external tensor '{}' mapped to a null CUDA pointer",
                                               debug_name ? debug_name : "<unnamed>"));
        }

        if (const cudaError_t status = cudaMemsetAsync(cuda_ptr, 0, bytes, stream);
            status != cudaSuccess) {
            interop.reset();
            context.destroyExternalBuffer(buffer);
            return std::unexpected(std::format("Vulkan external tensor '{}' zero-fill failed: {} ({})",
                                               debug_name ? debug_name : "<unnamed>",
                                               cudaGetErrorName(status),
                                               cudaGetErrorString(status)));
        }

        auto owner = std::make_shared<VulkanExternalTensorStorage>(
            context, buffer, std::move(interop), bytes);
        auto tensor = lfs::core::Tensor::from_external_owner(
            cuda_ptr,
            std::move(shape),
            lfs::core::Device::CUDA,
            dtype,
            owner,
            cap_rows,
            stream,
            "vulkan_external_buffer");
        return tensor;
    }

} // namespace lfs::vis

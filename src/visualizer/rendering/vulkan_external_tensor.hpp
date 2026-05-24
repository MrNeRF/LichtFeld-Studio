/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/tensor.hpp"
#include "rendering/cuda_vulkan_interop.hpp"
#include "window/vulkan_context.hpp"

#include <expected>
#include <memory>
#include <string>

namespace lfs::vis {

    class VulkanExternalTensorStorage final {
    public:
        VulkanExternalTensorStorage(VulkanContext& context,
                                    VulkanContext::ExternalBuffer buffer,
                                    lfs::rendering::CudaVulkanBufferInterop interop,
                                    std::size_t bytes);
        ~VulkanExternalTensorStorage();

        VulkanExternalTensorStorage(const VulkanExternalTensorStorage&) = delete;
        VulkanExternalTensorStorage& operator=(const VulkanExternalTensorStorage&) = delete;
        VulkanExternalTensorStorage(VulkanExternalTensorStorage&&) = delete;
        VulkanExternalTensorStorage& operator=(VulkanExternalTensorStorage&&) = delete;

        [[nodiscard]] void* cudaPtr() const { return interop_.devicePointer(); }
        [[nodiscard]] VkBuffer vkBuffer() const { return buffer_.buffer; }
        [[nodiscard]] VkDeviceSize vkOffset() const { return 0; }
        [[nodiscard]] std::size_t bytes() const { return bytes_; }

    private:
        VulkanContext* context_ = nullptr;
        VulkanContext::ExternalBuffer buffer_{};
        lfs::rendering::CudaVulkanBufferInterop interop_{};
        std::size_t bytes_ = 0;
    };

    [[nodiscard]] std::expected<lfs::core::Tensor, std::string> makeVulkanExternalTensor(
        VulkanContext& context,
        lfs::core::TensorShape shape,
        lfs::core::DataType dtype,
        std::size_t capacity,
        const char* debug_name,
        cudaStream_t stream = nullptr);

} // namespace lfs::vis

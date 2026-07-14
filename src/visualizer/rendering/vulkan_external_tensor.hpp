/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/splat_data.hpp"
#include "core/splat_exportable_storage.hpp"
#include "core/tensor.hpp"
#include "rendering/cuda_vulkan_interop.hpp"
#include "window/vulkan_context.hpp"

#include <expected>
#include <memory>
#include <string>

namespace lfs::vis {

    class VulkanExternalTensorStorage final {
    public:
        // OWNED variant — this instance owns the Vulkan buffer + CUDA interop.
        // Used by the legacy makeVulkanExternalTensor() path (one tensor per VkBuffer).
        VulkanExternalTensorStorage(VulkanContext& context,
                                    VulkanContext::ExternalBuffer buffer,
                                    lfs::rendering::CudaVulkanBufferInterop interop,
                                    std::size_t bytes,
                                    std::shared_ptr<void> extra_owner = {});

        // SUB-VIEW variant — borrows the VkBuffer and lifetime from `parent` at a
        // fixed (offset, bytes) slice. Tensor::from_external_owner receives the
        // corresponding CUDA data pointer separately.
        VulkanExternalTensorStorage(std::shared_ptr<VulkanExternalTensorStorage> parent,
                                    std::size_t offset,
                                    std::size_t bytes);

        ~VulkanExternalTensorStorage();

        VulkanExternalTensorStorage(const VulkanExternalTensorStorage&) = delete;
        VulkanExternalTensorStorage& operator=(const VulkanExternalTensorStorage&) = delete;
        VulkanExternalTensorStorage(VulkanExternalTensorStorage&&) = delete;
        VulkanExternalTensorStorage& operator=(VulkanExternalTensorStorage&&) = delete;

        [[nodiscard]] VkBuffer vkBuffer() const;
        [[nodiscard]] VkDeviceSize vkBufferSize() const;
        [[nodiscard]] VkDeviceSize vkOffset() const;
        [[nodiscard]] std::size_t bytes() const { return bytes_; }

    private:
        // Owned-variant members (only meaningful when parent_ is nullptr).
        VulkanContext* context_ = nullptr;
        VulkanContext::ExternalBuffer buffer_{};
        lfs::rendering::CudaVulkanBufferInterop interop_{};
        // Sub-view members.
        std::shared_ptr<VulkanExternalTensorStorage> parent_;
        std::size_t offset_ = 0;
        // Common.
        std::size_t bytes_ = 0;
        // Optional lifetime anchor (e.g. CUDA-side ExportableBlock). Released on dtor.
        std::shared_ptr<void> extra_owner_;
    };

    [[nodiscard]] std::expected<lfs::core::Tensor, std::string> makeVulkanExternalTensor(
        VulkanContext& context,
        lfs::core::TensorShape shape,
        lfs::core::DataType dtype,
        std::size_t capacity,
        const char* debug_name,
        cudaStream_t stream = nullptr,
        bool zero_fill = true);

    // Build a SplatTensorAllocator that hands out tensor views into a single
    // CUDA-exportable VMM block imported into Vulkan. Each tensor carries a
    // VulkanExternalTensorStorage sub-view so the existing vksplat fast path
    // (bind external storage directly, no per-frame memcpy) activates. The
    // returned allocator holds shared_ptrs to both the CUDA-side ExportableBlock
    // and the Vulkan-side parent storage; tensors keep them alive via the
    // standard shared_ptr<void> data_owner_ chain.
    [[nodiscard]] std::expected<lfs::core::SplatTensorAllocator, std::string>
    makeSplatExportableInteropAllocator(VulkanContext& context,
                                        const lfs::core::SplatExportableStorage& storage);

    // One-tensor-per-VkBuffer allocator bound to the active window's Vulkan context, matching
    // what the splat renderer binds. Empty when interop is unavailable (headless). Shared by the
    // file loader and in-memory inserts (Python API).
    [[nodiscard]] LFS_VIS_API lfs::core::SplatTensorAllocator makeViewerSplatTensorAllocator();

} // namespace lfs::vis

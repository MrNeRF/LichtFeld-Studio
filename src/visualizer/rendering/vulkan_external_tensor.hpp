/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/sh_value_quant.hpp"
#include "core/splat_data.hpp"
#include "core/splat_exportable_storage.hpp"
#include "core/tensor.hpp"
#include "window/vulkan_context.hpp"

#include <expected>
#include <memory>
#include <string>
#include <string_view>

namespace lfs::vis {

    class VulkanExternalTensorStorage final {
    public:
        // OWNED variant — this instance owns the imported VkBuffer. CUDA storage
        // lives in extra_owner (ExportableBlock). cuda_ptr is registered for
        // diagnostics when non-null.
        VulkanExternalTensorStorage(VulkanContext& context,
                                    VulkanContext::ExternalBuffer buffer,
                                    std::size_t bytes,
                                    std::string debug_label,
                                    std::shared_ptr<void> extra_owner = {},
                                    const void* cuda_ptr = nullptr);

        // SUB-VIEW variant — borrows the VkBuffer and lifetime from `parent` at a
        // fixed (offset, bytes) slice. Tensor::from_external_owner receives the
        // corresponding CUDA data pointer separately.
        VulkanExternalTensorStorage(std::shared_ptr<VulkanExternalTensorStorage> parent,
                                    std::size_t offset,
                                    std::size_t bytes);

        // LIVE-CONTROL sub-view — offset/bytes re-resolved from SplatExportableStorage
        // Control on every query. Matches resolve_exportable_device_ptr for CUDA so a
        // capacity grow cannot leave the viewer binding a pre-grow region while FastGS
        // already reads the live base.
        VulkanExternalTensorStorage(
            std::shared_ptr<VulkanExternalTensorStorage> parent,
            std::shared_ptr<lfs::core::SplatExportableStorage::Control> control,
            lfs::core::SplatExportableStorage::Region region);

        ~VulkanExternalTensorStorage();

        VulkanExternalTensorStorage(const VulkanExternalTensorStorage&) = delete;
        VulkanExternalTensorStorage& operator=(const VulkanExternalTensorStorage&) = delete;
        VulkanExternalTensorStorage(VulkanExternalTensorStorage&&) = delete;
        VulkanExternalTensorStorage& operator=(VulkanExternalTensorStorage&&) = delete;

        [[nodiscard]] VkBuffer vkBuffer() const;
        [[nodiscard]] VkDeviceSize vkBufferSize() const;
        [[nodiscard]] VkDeviceSize vkOffset() const;
        [[nodiscard]] VkDeviceAddress vkDeviceAddress() const;
        [[nodiscard]] std::size_t bytes() const;
        [[nodiscard]] bool bindNewExportableChunks(const lfs::core::ExportableBlock& block);

    private:
        // Owned-variant members (only meaningful when parent_ is nullptr).
        VulkanContext* context_ = nullptr;
        VulkanContext::ExternalBuffer buffer_{};
        const void* registered_cuda_base_ = nullptr;
        // Sub-view members.
        std::shared_ptr<VulkanExternalTensorStorage> parent_;
        std::size_t offset_ = 0;
        // Common.
        std::size_t bytes_ = 0;
        // Optional lifetime anchor (e.g. CUDA-side ExportableBlock). Released on dtor.
        std::shared_ptr<void> extra_owner_;
        // Live-control sub-view (exportable SoA). When set, vkOffset()/bytes()
        // re-resolve through Control rather than the baked offset_/bytes_.
        std::shared_ptr<lfs::core::SplatExportableStorage::Control> live_control_;
        lfs::core::SplatExportableStorage::Region live_region_ =
            lfs::core::SplatExportableStorage::Means;
    };

    [[nodiscard]] std::expected<lfs::core::Tensor, std::string> makeVulkanExternalTensor(
        VulkanContext& context,
        lfs::core::TensorShape shape,
        lfs::core::DataType dtype,
        std::size_t capacity,
        const char* debug_name);

    // Build a SplatTensorAllocator that hands out tensor views into a single
    // CUDA-exportable VMM block imported into Vulkan. Each tensor carries a
    // VulkanExternalTensorStorage sub-view so the existing vksplat fast path
    // (bind external storage directly, no per-frame memcpy) activates. The
    // returned allocator holds shared_ptrs to both the CUDA-side ExportableBlock
    // and the Vulkan-side parent storage; tensors keep them alive via the
    // standard shared_ptr<void> data_owner_ chain.
    [[nodiscard]] std::expected<lfs::core::SplatTensorAllocator, std::string>
    makeSplatExportableInteropAllocator(
        VulkanContext& context,
        const lfs::core::SplatExportableStorage& storage,
        std::shared_ptr<VulkanExternalTensorStorage>* parent_keep = nullptr);

    // Float32 SplatData.shN is a q16 workspace: keep it in pooled CUDA so the viewer
    // never imports a multi-gigabyte float rest buffer just to discard it after encode.
    [[nodiscard]] inline bool keepFloatShNInPooledCuda(const std::string_view name,
                                                       const lfs::core::DataType dtype) {
        return name == "SplatData.shN" &&
               dtype == lfs::core::DataType::Float32 &&
               lfs::core::sh_value_quant::enabled();
    }

    // One-tensor-per-VkBuffer allocator bound to the active window's Vulkan context, matching
    // what the splat renderer binds. Empty when interop is unavailable (headless). Shared by the
    // file loader and in-memory inserts (Python API).
    [[nodiscard]] LFS_VIS_API lfs::core::SplatTensorAllocator makeViewerSplatTensorAllocator();

} // namespace lfs::vis

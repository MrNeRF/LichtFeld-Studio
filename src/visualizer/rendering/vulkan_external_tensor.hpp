/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/error.hpp"
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

#if LFS_BUILD_TRAINER
    class VulkanExternalTensorStorage final {
    public:
        VulkanExternalTensorStorage(VulkanContext& context,
                                    std::shared_ptr<lfs::core::ExportableBlock> block);

        ~VulkanExternalTensorStorage();

        VulkanExternalTensorStorage(const VulkanExternalTensorStorage&) = delete;
        VulkanExternalTensorStorage& operator=(const VulkanExternalTensorStorage&) = delete;
        VulkanExternalTensorStorage(VulkanExternalTensorStorage&&) = delete;
        VulkanExternalTensorStorage& operator=(VulkanExternalTensorStorage&&) = delete;

        [[nodiscard]] bool bindNewExportableChunks(const lfs::core::ExportableBlock& block);

    private:
        VulkanContext* context_ = nullptr;
        lfs::core::Tensor block_tensor_;
        lfs::core::TensorVulkanBuffer buffer_{};
    };

    [[nodiscard]] std::expected<lfs::core::Tensor, std::string> makeVulkanExternalTensor(
        VulkanContext& context,
        lfs::core::TensorShape shape,
        lfs::core::DataType dtype,
        std::size_t capacity,
        const char* debug_name);

    // Training tensors share one CUDA-exportable block. Provenance resolves live
    // region offsets and the parent pins the imported buffer across growth.
    [[nodiscard]] lfs::Result<lfs::core::SplatTensorAllocator>
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

#endif

    // The window context outlives allocators returned for scene loading.
    [[nodiscard]] LFS_VIS_API lfs::core::SplatTensorAllocator makeViewerSplatTensorAllocator(bool preserve_float_shN = false);

} // namespace lfs::vis

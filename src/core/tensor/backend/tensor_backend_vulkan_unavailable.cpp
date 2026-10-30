/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/tensor.hpp"
#include "core/tensor_backend_vulkan.hpp"
#include "core/tensor_vulkan_interop.hpp"
namespace lfs::core {
    void TensorVulkanInterop::upload_host(Tensor& tensor, std::span<const std::byte> bytes) {

        (void)tensor;
        (void)bytes;
        throw TensorError("Vulkan tensor interop is unavailable in this build");
    }
    lfs::Status adopt_vulkan_device(const VulkanDeviceHandles& handles,
                                    const bool subgroup_size_control_enabled) {

        (void)handles;
        (void)subgroup_size_control_enabled;
        return lfs::Status::failure(lfs::make_error(lfs::ErrorInit{
            .code = lfs::ErrorCode::Unsupported,
            .domain = lfs::ErrorDomain::Vulkan,
            .user_message = "this build has no Vulkan tensor backend",
            .detection = LFS_SOURCE_SITE_CURRENT(),
        }));
    }
    bool vulkan_backend_adopted() {

        return false;
    }
    std::optional<TensorVulkanBuffer> tensor_vulkan_buffer(const Tensor& tensor) {

        (void)tensor;
        return std::nullopt;
    }
} // namespace lfs::core

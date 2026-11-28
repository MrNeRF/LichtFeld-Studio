/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/tensor/backend/vulkan/vk_ops_common.hpp"
#include <span>
#include <string_view>
namespace lfs::training::vulkan {
    using core::internal::StorageRef;
    StorageRef ref(const core::Tensor& tensor);
    // A tensor's device address for a kernel binding, guarded by vk::address with the tensor's own extent; 0 for an
    // empty tensor. wide_address is for bindings whose kernel addresses them in 64-bit integer math.
    uint64_t address(const core::Tensor& tensor, core::SourceSite site = LFS_SOURCE_SITE_CURRENT());
    uint64_t wide_address(const core::Tensor& tensor);
    void dispatch(std::string_view module, const void* parameters, size_t bytes,
                  std::span<const StorageRef> reads, std::span<const StorageRef> writes,
                  uint32_t groups, uint32_t specialization = UINT32_MAX,
                  StorageRef indirect = {}, VkDeviceSize indirect_offset = 0);
    template <class T>
    void dispatch(std::string_view module, const T& parameters,
                  std::span<const StorageRef> reads, std::span<const StorageRef> writes,
                  uint32_t groups, uint32_t specialization = UINT32_MAX,
                  StorageRef indirect = {}, VkDeviceSize indirect_offset = 0) {
        dispatch(module, &parameters, sizeof(T), reads, writes, groups, specialization, indirect, indirect_offset);
    }
    uint32_t groups(size_t work);
} // namespace lfs::training::vulkan

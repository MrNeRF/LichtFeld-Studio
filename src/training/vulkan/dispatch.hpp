/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/tensor/backend/vulkan/vk_ops_common.hpp"
#include <span>
#include <string_view>
namespace lfs::training::vulkan {
    using core::internal::StorageRef;
    StorageRef ref(const core::Tensor& tensor);
    uint64_t address(const core::Tensor& tensor);
    void dispatch(std::string_view module, const void* parameters, size_t bytes,
                  std::span<const StorageRef> reads, std::span<const StorageRef> writes,
                  uint32_t groups, uint32_t specialization = UINT32_MAX);
    template <class T>
    void dispatch(std::string_view module, const T& parameters,
                  std::span<const StorageRef> reads, std::span<const StorageRef> writes,
                  uint32_t groups, uint32_t specialization = UINT32_MAX) {
        dispatch(module, &parameters, sizeof(T), reads, writes, groups, specialization);
    }
    uint32_t groups(size_t work);
} // namespace lfs::training::vulkan

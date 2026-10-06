/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/tensor_backend.hpp"
#include "core/tensor_vulkan_interop.hpp"

#include <array>
#include <cstdint>
#include <mutex>

namespace lfs::core {

    // A Vulkan device the application owns, for the Vulkan tensor backend to
    // run on instead of creating its own. Handles are deliberately opaque in
    // this exported header; Vulkan implementation code performs the casts.
    struct VulkanDeviceHandles {
        void* instance = nullptr;
        void* physical_device = nullptr;
        void* device = nullptr;
        void* queue = nullptr;
        uint32_t queue_family = 0;
        std::array<uint32_t, 3> sharing_queue_families{};
        uint32_t sharing_queue_family_count = 0;
        bool shader_atomic_float = false;
        bool memory_budget = false;
        bool shader_float64 = false;
        bool shader_float16 = false;
        bool vulkan_memory_model = false;
        bool vulkan_memory_model_device_scope = false;
        bool cooperative_matrix = false;
        bool external_memory = false;
        bool external_semaphore = false;
        bool metal_objects = false;
        void* consumer_queue = nullptr;
        std::mutex* consumer_queue_mutex = nullptr;
    };

    LFS_CORE_API lfs::Status adopt_vulkan_device(const VulkanDeviceHandles& handles);
    LFS_CORE_API bool vulkan_backend_adopted();
    LFS_CORE_API bool tensor_backend_shares_vulkan_device();
    LFS_CORE_API bool tensor_backend_needs_vulkan_interop();
    LFS_CORE_API lfs::Result<Tensor> cuda_view_of_vulkan_tensor(const Tensor& tensor,
                                                                cudaStream_t stream);
    LFS_CORE_API bool vulkan_backend_exports_memory();

} // namespace lfs::core

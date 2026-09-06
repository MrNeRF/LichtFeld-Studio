/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/error.hpp"
#include "core/gpu_backend_fwd.hpp"

#include <cstdint>
#include <memory>
#include <optional>

namespace lfs::core {

    class MemoryInfo;

    LFS_CORE_API GpuBackend default_gpu_backend();
    LFS_CORE_API lfs::Status set_default_gpu_backend(GpuBackend backend);
    LFS_CORE_API bool gpu_backend_available(GpuBackend backend);

    class LFS_CORE_API GpuBackendScope {
    public:
        explicit GpuBackendScope(GpuBackend backend);
        ~GpuBackendScope();

        GpuBackendScope(const GpuBackendScope&) = delete;
        GpuBackendScope& operator=(const GpuBackendScope&) = delete;
        GpuBackendScope(GpuBackendScope&&) = delete;
        GpuBackendScope& operator=(GpuBackendScope&&) = delete;

    private:
        std::optional<GpuBackend> previous_;
    };

    LFS_CORE_API MemoryInfo gpu_backend_memory_info(GpuBackend backend);
    LFS_CORE_API lfs::Status shutdown_gpu_backend(GpuBackend backend);
    LFS_CORE_API lfs::Status tensor_backend_selftest(GpuBackend backend);

    // A Vulkan device the application owns, for the Vulkan tensor backend to run
    // on instead of creating its own: one device for tensors and rendering. The
    // handles are the dispatchable VkInstance, VkPhysicalDevice, VkDevice and
    // VkQueue; the queue belongs to the backend alone. The device must have
    // shaderInt64, shaderInt16, storageBuffer16BitAccess, storageBuffer8BitAccess,
    // timelineSemaphore, bufferDeviceAddress and synchronization2 enabled; the
    // flags say which optional features are on. The device must outlive the
    // backend: call shutdown_gpu_backend(GpuBackend::Vulkan) before destroying it.
    struct VulkanDeviceHandles {
        void* instance = nullptr;
        void* physical_device = nullptr;
        void* device = nullptr;
        void* queue = nullptr;
        uint32_t queue_family = 0;
        bool shader_atomic_float = false;
        bool memory_budget = false;
        bool shader_float16 = false;
    };

    // Fails when the backend already has a context (adopt before the first
    // Vulkan tensor) or the device lacks a required feature.
    LFS_CORE_API lfs::Status adopt_vulkan_device(const VulkanDeviceHandles& handles);
    LFS_CORE_API bool vulkan_backend_adopted();

    // A Vulkan-backend tensor's storage for a consumer on the same device. The
    // buffer is valid while keep_alive is held; pending_timeline_value is the
    // value of vulkan_backend_timeline() after which every pending write to the
    // tensor is complete (0 when nothing is pending); the query flushes the
    // recorder that owns those writes so the value will be signalled.
    struct TensorVulkanBuffer {
        void* buffer = nullptr;
        uint64_t offset = 0;
        uint64_t device_address = 0;
        uint64_t bytes = 0;
        uint64_t pending_timeline_value = 0;
        std::shared_ptr<void> keep_alive;
    };
    LFS_CORE_API void* vulkan_backend_timeline();
    LFS_CORE_API std::optional<TensorVulkanBuffer> tensor_vulkan_buffer(const Tensor& tensor);

    namespace internal {
        LFS_CORE_API void gpu_backend_reset_for_testing();
    }

} // namespace lfs::core

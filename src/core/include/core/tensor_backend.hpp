/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/cuda_types.hpp"
#include "core/error.hpp"
#include "core/gpu_backend_fwd.hpp"
#include "core/tensor.hpp"
#include "core/tensor_vulkan_interop.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace lfs::core {

    class MemoryInfo;

    // Configure before the first backend use. UI changes apply after restart.
    struct TensorBackendOptions {
        std::string vulkan_device; // Empty selects automatically; otherwise index or UUID.
        int vulkan_validation = 0; // 0: off, 1: API validation, 2: synchronization validation.
        bool force_fp32_half = false;
        bool force_no_atomic_float = false;
    };

    LFS_CORE_API lfs::Status set_tensor_backend_options(const TensorBackendOptions& options);
    LFS_CORE_API TensorBackendOptions tensor_backend_options();

    LFS_CORE_API lfs::Status set_default_gpu_backend(GpuBackend backend);
    LFS_CORE_API bool gpu_backend_available(GpuBackend backend);
    // True while the backend holds device state (a CUDA context, a Vulkan context); never creates it.
    LFS_CORE_API bool gpu_backend_live(GpuBackend backend);

    // Pool counters are optional because collecting them costs extra runtime queries.
    LFS_CORE_API MemoryInfo gpu_backend_memory_info(GpuBackend backend,
                                                    bool include_pool_stats = false);
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
    };

    // Fails when the backend already has a context (adopt before the first
    // Vulkan tensor) or the device lacks a required feature.
    LFS_CORE_API lfs::Status adopt_vulkan_device(const VulkanDeviceHandles& handles);
    LFS_CORE_API bool vulkan_backend_adopted();

    // A Vulkan-backend tensor's storage for a consumer on the same device. The
    // buffer is valid while keep_alive is held; pending_timeline_value is the
    // backend timeline value after which every pending write to the
    // tensor is complete (0 when nothing is pending); the query flushes the
    // recorder that owns those writes so the value will be signalled.
    // A CUDA-tagged tensor aliasing a Vulkan-backend tensor's memory on NVIDIA
    // devices with external memory support. The view is ordered after the
    // tensor's pending Vulkan writes on `stream`; it keeps the Vulkan storage
    // alive; writes through the view are not ordered back into Vulkan work.
    LFS_CORE_API lfs::Result<Tensor> cuda_view_of_vulkan_tensor(const Tensor& tensor,
                                                                cudaStream_t stream);
    LFS_CORE_API bool vulkan_backend_exports_memory();

    namespace internal {
        LFS_CORE_API void gpu_backend_reset_for_testing();
    }

} // namespace lfs::core

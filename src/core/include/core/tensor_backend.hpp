/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/cuda_types.hpp"
#include "core/error.hpp"
#include "core/gpu_backend_fwd.hpp"
#include "core/tensor.hpp"
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace lfs::core {

    class MemoryInfo;

    // Configure before the first backend use. UI changes apply after restart.
    struct TensorBackendOptions {
        std::string vulkan_device;           // Empty selects automatically; otherwise index or UUID.
        std::string rendering_vulkan_device; // Empty follows the tensor GPU for compatibility.
        std::string hip_device;              // Empty selects automatically; otherwise a native device index.
        std::string cuda_device;             // Empty keeps the existing default; otherwise CUDA index or UUID.
        int vulkan_validation = 0;           // 0: off, 1: API validation, 2: synchronization validation.
        bool force_fp32_half = false;
        bool force_no_atomic_float = false;
    };

    // An inactive tensor backend's saved GPU must not steer the viewport.
    inline std::string_view rendering_device_preference(const TensorBackendOptions& options, GpuBackend backend) {
        if (!options.rendering_vulkan_device.empty())
            return options.rendering_vulkan_device;
        return backend == GpuBackend::Vulkan ? std::string_view(options.vulkan_device) : std::string_view{};
    }

    LFS_CORE_API lfs::Status set_tensor_backend_options(const TensorBackendOptions& options);
    LFS_CORE_API TensorBackendOptions tensor_backend_options();
    LFS_CORE_API TensorBackendOptions tensor_backend_options_snapshot();
    // The configured process device is applied at CUDA worker/stream entry points.
    LFS_CORE_API int selected_cuda_device();
    LFS_CORE_API void activate_selected_cuda_device();
    // Installed by the renderer before creating training queues. Foreign GPUs
    // transfer through host memory; they must never import CUDA handles.
    LFS_CORE_API void set_cuda_vulkan_direct_interop(bool enabled);
    LFS_CORE_API bool cuda_vulkan_direct_interop();

    LFS_CORE_API lfs::Status set_default_gpu_backend(GpuBackend backend);
    LFS_CORE_API bool gpu_backend_available(GpuBackend backend);
    // True while the backend holds device state (a CUDA context, a Vulkan context); never creates it.
    LFS_CORE_API bool gpu_backend_live(GpuBackend backend);

    // Pool counters are optional because collecting them costs extra runtime queries.
    LFS_CORE_API MemoryInfo gpu_backend_memory_info(GpuBackend backend,
                                                    bool include_pool_stats = false);
    LFS_CORE_API lfs::Status shutdown_gpu_backend(GpuBackend backend);
    LFS_CORE_API lfs::Status tensor_backend_selftest(GpuBackend backend);

    // Native Metal readers accept resident Metal and exportable MoltenVK storage.
    LFS_CORE_API bool tensor_supports_metal_access(const Tensor& tensor);
    LFS_CORE_API bool tensor_backend_supports_metal_access();

    namespace internal {
        LFS_CORE_API void gpu_backend_reset_for_testing();
    }

} // namespace lfs::core

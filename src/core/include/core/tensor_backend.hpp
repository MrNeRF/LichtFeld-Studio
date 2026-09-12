/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/error.hpp"
#include "core/gpu_backend_fwd.hpp"

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
        bool viewer_vulkan_inputs = false;
    };

    LFS_CORE_API lfs::Status set_tensor_backend_options(const TensorBackendOptions& options);
    LFS_CORE_API TensorBackendOptions tensor_backend_options();

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

    namespace internal {
        LFS_CORE_API void gpu_backend_reset_for_testing();
    }

} // namespace lfs::core

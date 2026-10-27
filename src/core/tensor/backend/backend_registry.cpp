/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gpu_backend_ops.hpp"

#include "../internal/expression_runtime.hpp"
#include "../internal/tensor_impl.hpp"
#include "core/assert.hpp"
#include "core/gpu_device_info.hpp"
#include "core/tensor_backend.hpp"
#ifdef LFS_TENSOR_VULKAN
#include "vulkan/vk_backend_ops.hpp"
#endif
#ifdef LFS_TENSOR_METAL
#include "metal/metal_backend_ops.hpp"
#endif

#include <utility>
#ifdef LFS_TENSOR_HIP
#include "hip/hip_backend_ops.hpp"
#endif

namespace lfs::core::internal {

    GpuBackendOps::~GpuBackendOps() = default;

    std::unique_ptr<BackendUploadBuffer> GpuBackendOps::create_upload_buffer() {
        throw std::runtime_error("Native upload buffers are unsupported on this backend");
    }
    std::unique_ptr<BackendFence> GpuBackendOps::create_fence() {
        throw std::runtime_error("Native fences are unsupported on this backend");
    }
    std::unique_ptr<BackendTimestamps> GpuBackendOps::create_timestamps(size_t) {
        return {};
    }
    std::unique_ptr<BackendReadbackSlot> GpuBackendOps::create_readback_slot(size_t) {
        throw std::runtime_error("Native readback slots are unsupported on this backend");
    }
    BackendQueueLease GpuBackendOps::execution_lease(ExecContext) {
        throw std::runtime_error("Native execution leases are unsupported on this backend");
    }
    StorageRef GpuBackendOps::borrow_native_range(void*, size_t) {
        throw std::runtime_error("Native storage views are unsupported on this backend");
    }
    bool GpuBackendOps::queue_ready(ExecContext) {
        throw std::runtime_error("Native queue polling is unsupported on this backend");
    }
    bool GpuBackendOps::drain_queue_for_destruction() noexcept {
        return false; // Unsupported is never evidence that work has drained.
    }
    void GpuBackendOps::enqueue_host_callback(void (*)(void*), void*, ExecContext) {
        throw std::runtime_error("Native host callbacks are unsupported on this backend");
    }
    void GpuBackendOps::where_scalar(StorageRef, StorageRef, StorageRef, size_t, float, ExecContext) {
        throw std::runtime_error("Native scalar selection is unsupported on this backend");
    }
    ExpressionCacheStats GpuBackendOps::expression_cache_stats() {
        throw std::runtime_error("Expression cache statistics are unsupported on this backend");
    }

    int optional_backend_device_count(const GpuBackend backend) {
#ifdef LFS_TENSOR_HIP
        if (backend == GpuBackend::HIP)
            return hip_device_count(LFS_HIP_RUNTIME_DEFAULT_MODULE);
#endif
        (void)backend;
        return 0;
    }
    std::optional<GpuDeviceInfo> optional_backend_device_info(const GpuBackend backend, int ordinal) {
#ifdef LFS_TENSOR_HIP
        if (backend == GpuBackend::HIP)
            return hip_gpu_device_info(LFS_HIP_RUNTIME_DEFAULT_MODULE, ordinal);
#endif
        (void)backend;
        (void)ordinal;
        return std::nullopt;
    }
    bool optional_backend_live(const GpuBackend backend) noexcept {
#ifdef LFS_TENSOR_HIP
        if (backend == GpuBackend::HIP)
            return hip_backend_is_live();
#endif
        (void)backend;
        return false;
    }
    int select_optional_backend_device(const GpuBackend backend, int requested) {
#ifdef LFS_TENSOR_HIP
        if (backend == GpuBackend::HIP) {
            return select_hip_device(LFS_HIP_RUNTIME_DEFAULT_MODULE, requested);
        }
#endif
        (void)backend;
        (void)requested;
        throw std::runtime_error("GPU backend is not compiled into this build");
    }

    void GpuBackendOps::push_range(const char*) {
        throw std::runtime_error("GPU diagnostic ranges are unsupported on this backend");
    }
    void GpuBackendOps::pop_range() {
        throw std::runtime_error("GPU diagnostic ranges are unsupported on this backend");
    }
    void GpuBackendOps::name_queue(ExecContext, const char*) {}

    GpuBackendOps& backend_ops(const GpuBackend backend) {
        // Share an existing unavailable-backend rejection in standard builds.
        // This keeps the stable HIP identity from adding a dispatch branch on
        // the compiled CUDA/Metal/Vulkan operation paths when HIP is disabled.
        constexpr bool reject_hip_with_cuda = !gpu_backend_compiled(GpuBackend::HIP) && !gpu_backend_compiled(GpuBackend::CUDA);
        if (reject_hip_with_cuda ? (backend != GpuBackend::Vulkan && backend != GpuBackend::Metal) : backend == GpuBackend::CUDA) {
            if (backend == GpuBackend::HIP) {
                LFS_ASSERT_MSG(false, "GPU backend 'HIP' is not compiled into this build");
                std::unreachable();
            }
#if LFS_HAS_CUDA
            // Bind an arbitrary caller thread on its first CUDA operation.
            // Keep driver queries and the selection's atomic load out of the
            // per-operation path; worker/stream scopes also bind explicitly.
            static thread_local CudaBackendOps* const cuda_ops = [] {
                activate_selected_cuda_device();
                static CudaBackendOps* const shared_ops = new CudaBackendOps();
                return shared_ops;
            }();
            return *cuda_ops;
#else
            LFS_ASSERT_MSG(false, "GPU backend 'CUDA' is not compiled into this build");
            std::unreachable();
#endif
        }

        constexpr bool reject_hip_with_metal = !gpu_backend_compiled(GpuBackend::HIP) && !gpu_backend_compiled(GpuBackend::Metal) && !reject_hip_with_cuda;
        if (reject_hip_with_metal ? (backend != GpuBackend::CUDA && backend != GpuBackend::Vulkan) : backend == GpuBackend::Metal) {
            if (backend == GpuBackend::HIP) {
                LFS_ASSERT_MSG(false, "GPU backend 'HIP' is not compiled into this build");
                std::unreachable();
            }
#ifdef LFS_TENSOR_METAL
            if (__builtin_available(macOS 26.0, *)) {
                static MetalBackendOps* const metal_ops = new MetalBackendOps();
                return *metal_ops;
            }
            LFS_ASSERT_MSG(false, "GPU backend 'Metal' needs macOS 26");
#else
            LFS_ASSERT_MSG(false, "GPU backend 'Metal' is not compiled into this build");
#endif
            std::unreachable();
        }

        if ((gpu_backend_compiled(GpuBackend::HIP) ||
             (gpu_backend_compiled(GpuBackend::CUDA) && gpu_backend_compiled(GpuBackend::Metal))) &&
            backend == GpuBackend::HIP) {
#ifdef LFS_TENSOR_HIP
            static HipBackendOps* const hip_ops = [] {
                (void)tensor_backend_options(); // Freeze startup device selection before construction.
                return new HipBackendOps(LFS_HIP_RUNTIME_DEFAULT_MODULE, selected_hip_device());
            }();
            hip_ops->ensure_initialized();
            return *hip_ops;
#else
            LFS_ASSERT_MSG(false, "GPU backend 'HIP' is not compiled into this build");
            std::unreachable();
#endif
        }

#ifdef LFS_TENSOR_VULKAN
        static VulkanBackendOps* const vulkan_ops = new VulkanBackendOps();
        return *vulkan_ops;
#else
        LFS_ASSERT_MSG(false, "GPU backend 'Vulkan' is unavailable");
        std::unreachable();
#endif
    }

    GpuBackendOps& backend_ops_for(const Tensor& tensor) {
        LFS_ASSERT_MSG(tensor.is_valid(), "backend_ops_for requires a valid tensor");
        LFS_ASSERT_MSG(tensor.device() == Device::GPU,
                       "backend_ops_for requires GPU storage");
        return backend_ops(gpu_backend_of(tensor).value());
    }

} // namespace lfs::core::internal

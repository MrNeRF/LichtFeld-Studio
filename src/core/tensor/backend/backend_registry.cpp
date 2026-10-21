/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gpu_backend_ops.hpp"

#include "../internal/tensor_impl.hpp"
#include "core/assert.hpp"
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

    void GpuBackendOps::push_range(const char*) {
        throw std::runtime_error("GPU diagnostic ranges are unsupported on this backend");
    }
    void GpuBackendOps::pop_range() {
        throw std::runtime_error("GPU diagnostic ranges are unsupported on this backend");
    }
    void GpuBackendOps::name_queue(ExecContext, const char*) {}

    GpuBackendOps& backend_ops(const GpuBackend backend) {
#ifdef LFS_TENSOR_HIP
        if (backend == GpuBackend::HIP) {
            static HipBackendOps* const hip_ops = [] {
                (void)tensor_backend_options(); // Freeze startup device selection before construction.
                return new HipBackendOps(LFS_HIP_RUNTIME_DEFAULT_MODULE,selected_hip_device());
            }();
            return *hip_ops;
        }
#endif
        if (backend == GpuBackend::CUDA) {
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

        if (backend == GpuBackend::Metal) {
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

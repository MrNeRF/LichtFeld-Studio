/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gpu_backend_ops.hpp"

#include "../internal/tensor_impl.hpp"
#include "core/assert.hpp"

#include <utility>

namespace lfs::core::internal {

    GpuBackendOps::~GpuBackendOps() = default;

    GpuBackendOps& backend_ops(const GpuBackend backend) {
        if (backend == GpuBackend::CUDA) {
            static CudaBackendOps cuda_ops;
            return cuda_ops;
        }

        LFS_ASSERT_MSG(false, "GPU backend 'Vulkan' is unavailable");
        std::unreachable();
    }

    GpuBackendOps& backend_ops_for(const Tensor& tensor) {
        LFS_ASSERT_MSG(tensor.is_valid(), "backend_ops_for requires a valid tensor");
        LFS_ASSERT_MSG(tensor.device() == Device::CUDA,
                       "backend_ops_for requires GPU storage");
        return backend_ops(gpu_backend_of(tensor).value());
    }

} // namespace lfs::core::internal

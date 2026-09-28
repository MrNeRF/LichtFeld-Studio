/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/shared_image_ops.hpp"
#if LFS_HAS_CUDA
#include "core/cuda/shared_image_cuda.hpp"
#endif

#if LFS_TENSOR_METAL
namespace lfs::core {
    // Null before macOS 26; see tensor/backend/metal/metal_shared_image.mm.
    const gpu_ops::SharedImageOps* metal_shared_image_ops();
} // namespace lfs::core
#endif

namespace lfs::core {
    const gpu_ops::SharedImageOps* shared_image_ops(const GpuBackend backend) {
#if LFS_HAS_CUDA
        if (backend == GpuBackend::CUDA) {
            return &cuda_shared_image_ops();
        }
#endif
#if LFS_TENSOR_METAL
        if (backend == GpuBackend::Metal) {
            return metal_shared_image_ops();
        }
#endif
        return nullptr;
    }
} // namespace lfs::core

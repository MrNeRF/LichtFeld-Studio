/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/shared_image_ops.hpp"
#if LFS_HAS_CUDA
#include "core/cuda/shared_image_cuda.hpp"
#endif

namespace lfs::core {
    const gpu_ops::SharedImageOps* shared_image_ops(const GpuBackend backend) {
#if LFS_HAS_CUDA
        if (backend == GpuBackend::CUDA) {
            return &cuda_shared_image_ops();
        }
#endif
        return nullptr;
    }
} // namespace lfs::core

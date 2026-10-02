/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/splat_block.hpp"

#include <stdexcept>

namespace lfs::core {

    SplatBlockOps& device_splat_block_ops();
#if LFS_HAS_CUDA
    SplatBlockOps& cuda_splat_block_ops();
#endif

    SplatBlockOps& splat_block_ops(const GpuBackend backend) {
        if (backend == GpuBackend::CUDA) {
#if LFS_HAS_CUDA
            return cuda_splat_block_ops();
#else
            throw std::runtime_error("CUDA splat storage is not compiled into this build");
#endif
        }
        return device_splat_block_ops();
    }

} // namespace lfs::core

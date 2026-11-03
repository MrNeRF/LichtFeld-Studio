/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs/training/ops/registry.hpp"

#if defined(LFS_TENSOR_HIP) && LFS_TENSOR_HIP
#include "lfs/training/ops/extra_loss_hip.hpp"
#endif

namespace lfs::training {
    const TrainingOps& hip_training_ops_table() {
        static const TrainingOps table{
            .backend = core::GpuBackend::HIP,
#if defined(LFS_TENSOR_HIP) && LFS_TENSOR_HIP
            .extra_loss = &hip_extra_loss_ops(),
            .shared_image = core::shared_image_ops(core::GpuBackend::HIP),
#endif
        };
        return table;
    }
} // namespace lfs::training

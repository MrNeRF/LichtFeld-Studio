/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs/training/ops/structure_cuda.hpp"
namespace lfs::training {
    const gpu_ops::StructureOps& cuda_structure_ops() {
        static const gpu_ops::StructureOps ops{kernels::cuda_ridge_structure_map,
                                               kernels::cuda_structure_photometric_weight, kernels::cuda_structure_densification_weight,
                                               kernels::cuda_gradient_residual_loss_gradient};
        return ops;
    }
} // namespace lfs::training

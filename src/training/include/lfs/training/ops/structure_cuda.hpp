/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once
#include "lfs/training/ops/structure.hpp"
namespace lfs::training {
    const gpu_ops::StructureOps& cuda_structure_ops();
}
namespace lfs::training::kernels {
    void cuda_ridge_structure_map(const core::Tensor&, core::Tensor&, RidgeWorkspace&);
    void cuda_structure_photometric_weight(const core::Tensor&, const core::Tensor&, core::Tensor&, float, bool);
    void cuda_structure_densification_weight(core::Tensor&, const core::Tensor&, float);
    core::Tensor cuda_gradient_residual_loss_gradient(const core::Tensor&, const core::Tensor&, const core::Tensor&,
                                                      core::Tensor&, float, GradientResidualWorkspace&);
} // namespace lfs::training::kernels

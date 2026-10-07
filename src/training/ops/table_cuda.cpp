/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs/training/ops/blob_cuda.hpp"
#include "lfs/training/ops/registry.hpp"
#include "lfs/training/ops/structure_cuda.hpp"

#include "lfs/training/ops/adam_cuda.hpp"
#include "lfs/training/ops/bilateral_cuda.hpp"
#include "lfs/training/ops/extra_loss_cuda.hpp"
#include "lfs/training/ops/fast_cuda.hpp"
#include "lfs/training/ops/geometry_cuda.hpp"
#include "lfs/training/ops/gsplat_cuda.hpp"
#include "lfs/training/ops/lpips_cuda.hpp"
#include "lfs/training/ops/masks_cuda.hpp"
#include "lfs/training/ops/mcmc_cuda.hpp"
#include "lfs/training/ops/morton_cuda.hpp"
#include "lfs/training/ops/mrnf_cuda.hpp"
#include "lfs/training/ops/photometric_cuda.hpp"
#include "lfs/training/ops/ppisp_cuda.hpp"
#include "lfs/training/ops/refine_cuda.hpp"
#include "lfs/training/ops/session_cuda.hpp"
#include "lfs/training/ops/sh_cuda.hpp"
#include "lfs/training/ops/training_image_cuda.hpp"

namespace lfs::training {

    const TrainingOps& cuda_training_ops_table() {
        static const TrainingOps table{
            .backend = core::GpuBackend::CUDA,
            .photometric = &cuda_photometric_ops(),
            .adam = &cuda_adam_ops(),
            .mcmc = &cuda_mcmc_ops(),
            .mrnf = &cuda_mrnf_ops(),
            .geometry = &cuda_geometry_ops(),
            .fast = &cuda_fast_ops(),
            .morton = &cuda_morton_ops(),
            .masks = &cuda_masks_ops(),
            .extra_loss = &cuda_extra_loss_ops(),
            .bilateral = &cuda_bilateral_ops(),
            .training_image = &cuda_training_image_ops(),
            .sh = &cuda_sh_ops(),
            .ppisp = &cuda_ppisp_ops(),
            .controller = &cuda_controller_ops(),
            .gsplat = &cuda_gsplat_ops(),
            .refine = &cuda_refine_ops(),
            .session = &cuda_session_ops(),
            .shared_image = core::shared_image_ops(core::GpuBackend::CUDA),
            .lpips = &cuda_lpips_ops(),
            .structure = &cuda_structure_ops(),
            .blob = &cuda_blob_ops(),
        };
        return table;
    }

} // namespace lfs::training

/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs/training/ops/registry.hpp"

#if defined(LFS_TENSOR_VULKAN)
#include "lfs/training/ops/adam_vulkan.hpp"
#include "lfs/training/ops/bilateral_vulkan.hpp"
#include "lfs/training/ops/extra_loss_vulkan.hpp"
#include "lfs/training/ops/fast_vulkan.hpp"
#include "lfs/training/ops/geometry_vulkan.hpp"
#include "lfs/training/ops/gsplat_vulkan.hpp"
#include "lfs/training/ops/lpips_vulkan.hpp"
#include "lfs/training/ops/masks_vulkan.hpp"
#include "lfs/training/ops/mcmc_vulkan.hpp"
#include "lfs/training/ops/morton_vulkan.hpp"
#include "lfs/training/ops/mrnf_vulkan.hpp"
#include "lfs/training/ops/photometric_vulkan.hpp"
#include "lfs/training/ops/ppisp_vulkan.hpp"
#include "lfs/training/ops/refine_vulkan.hpp"
#include "lfs/training/ops/session_vulkan.hpp"
#include "lfs/training/ops/sh_vulkan.hpp"
#include "lfs/training/ops/training_image_vulkan.hpp"
#endif

namespace lfs::training {

    const TrainingOps& vulkan_training_ops_table() {
        static const TrainingOps table{
            .backend = core::GpuBackend::Vulkan,
#if defined(LFS_TENSOR_VULKAN)
            .photometric = &vulkan_photometric_ops(),
            .adam = &vulkan_adam_ops(),
            .mcmc = &vulkan_mcmc_ops(),
            .mrnf = &vulkan_mrnf_ops(),
            .geometry = &vulkan_geometry_ops(),
            .fast = &vulkan_fast_ops(),
            .morton = &vulkan_morton_ops(),
            .masks = &vulkan_masks_ops(),
            .extra_loss = &vulkan_extra_loss_ops(),
            .bilateral = &vulkan_bilateral_ops(),
            .training_image = &vulkan_training_image_ops(),
            .sh = &vulkan_sh_ops(),
            .ppisp = &vulkan_ppisp_ops(),
            .controller = &vulkan_controller_ops(),
            .gsplat = &vulkan_gsplat_ops(),
            .refine = &vulkan_refine_ops(),
            .session = &vulkan_session_ops(),
            .shared_image = core::shared_image_ops(core::GpuBackend::Vulkan),
            .lpips = &vulkan_lpips_ops(),
#endif
        };
        return table;
    }

} // namespace lfs::training

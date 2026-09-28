/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "metal_families.hpp"

namespace lfs::training {

    // A family's line stays commented until its port lands.
    const TrainingOps& metal_training_ops_table() {
        static const TrainingOps table{
            .backend = core::GpuBackend::Metal,

            // .photometric = &metal_photometric_ops(),

            // .adam = &metal_adam_ops(),

            // .mcmc = &metal_mcmc_ops(),

            // .mrnf = &metal_mrnf_ops(),

            // .geometry = &metal_geometry_ops(),

            // .fast = &metal_fast_ops(),

            // .morton = &metal_morton_ops(),

            // .masks = &metal_masks_ops(),

            // .extra_loss = &metal_extra_loss_ops(),

            .bilateral = &metal_bilateral_ops(),

            .training_image = &metal_training_image_ops(),

            // .sh = &metal_sh_ops(),

            .ppisp = &metal_ppisp_ops(),

            .controller = &metal_controller_ops(),

            // .gsplat = &metal_gsplat_ops(),

            // .refine = &metal_refine_ops(),

            .session = &metal_session_ops(),

            .shared_image = core::shared_image_ops(core::GpuBackend::Metal),

            .lpips = &metal_lpips_ops(),
        };
        return table;
    }

} // namespace lfs::training

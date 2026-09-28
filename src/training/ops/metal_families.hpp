/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "lfs/training/ops/registry.hpp"

// The Metal ops table's families, one accessor per *_metal.cpp.
namespace lfs::training {
    const lfs::gpu_ops::SessionOps& metal_session_ops();
    const lfs::gpu_ops::TrainingImageOps& metal_training_image_ops();
    const lfs::gpu_ops::AdamOps& metal_adam_ops();
    const lfs::gpu_ops::ShOps& metal_sh_ops();
    const lfs::gpu_ops::MortonOps& metal_morton_ops();
    const lfs::gpu_ops::FastRasterOps& metal_fast_ops();
    const lfs::gpu_ops::GsplatRasterOps& metal_gsplat_ops();
    const lfs::gpu_ops::PhotometricOps& metal_photometric_ops();
    const lfs::gpu_ops::MaskOps& metal_masks_ops();
    const lfs::gpu_ops::ExtraLossOps& metal_extra_loss_ops();
    const lfs::gpu_ops::GeometryLossOps& metal_geometry_ops();
    const lfs::gpu_ops::MrnfOps& metal_mrnf_ops();
    const lfs::gpu_ops::RefineOps& metal_refine_ops();
    const lfs::gpu_ops::McmcOps& metal_mcmc_ops();
    const lfs::gpu_ops::BilateralOps& metal_bilateral_ops();
    const lfs::gpu_ops::PPISPOps& metal_ppisp_ops();
    const lfs::gpu_ops::ControllerOps& metal_controller_ops();
    const lfs::gpu_ops::LpipsOps& metal_lpips_ops();
} // namespace lfs::training

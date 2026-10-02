/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "lfs/training/ops/gsplat.hpp"
namespace lfs::training::vulkan {
    gpu_ops::State gsplat_create();
    gpu_ops::RasterResult gsplat_forward(gpu_ops::GsplatSaved&, const gpu_ops::SplatInputs&,
                                         gpu_ops::In view, gpu_ops::In radial, gpu_ops::In tangential, gpu_ops::In background,
                                         gpu_ops::In background_image, const gpu_ops::GsplatParams&, const gpu_ops::RenderOutputs&);
    void gsplat_release(gpu_ops::GsplatSaved&) noexcept;
} // namespace lfs::training::vulkan

namespace lfs::training {
    const gpu_ops::GsplatRasterOps& vulkan_gsplat_ops();
}

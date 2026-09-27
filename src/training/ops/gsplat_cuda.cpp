/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs/training/ops/gsplat_cuda.hpp"
#include "rasterization/gsplat_rasterizer_cuda.hpp"

namespace lfs::training {
    const lfs::gpu_ops::GsplatRasterOps& cuda_gsplat_ops() {
        static const lfs::gpu_ops::GsplatRasterOps ops{
            .create = gsplat_create,
            .forward = gsplat_forward,
            .backward = gsplat_backward,
            .release = gsplat_release,
            .record_vram = gsplat_record_vram,
            .release_caches = gsplat_release_caches,
        };
        return ops;
    }
} // namespace lfs::training

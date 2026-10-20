/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/error.hpp"
#include "core/tensor.hpp"
#include "lfs/training/ops/gsplat.hpp"
#include <rasterization_api.h>

namespace lfs::training {
    struct GsplatRasterizeContext;

    // Accumulate FP64 removal energy into aligned UInt8 [N, sizeof(double)] storage.
    // The caller zeros these bytes once; each call adds a view without clearing them.
    void accumulate_fast_pop_scores(const fast_lfs::rasterization::ForwardContext& forward_ctx, size_t n,
                                    uint32_t width, uint32_t height, const core::Tensor& bg_color,
                                    const core::Tensor& bg_image, core::Tensor& scores);
    lfs::Status gsplat_accumulate_pop_scores(const GsplatRasterizeContext& ctx, const core::Tensor& bg_color,
                                             const core::Tensor& bg_image, core::Tensor& scores);
    // GsplatRasterOps::pop_scores.
    void gsplat_pop_scores(const lfs::gpu_ops::GsplatSaved&, const core::Tensor& bg_color,
                           const core::Tensor& bg_image, core::Tensor& scores);
} // namespace lfs::training

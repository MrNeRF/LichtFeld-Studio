/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/nn/lpips_dispatch.hpp"
#include "lfs/training/ops/types.hpp"

#include <cstdint>
#include <format>
#include <string>

namespace lfs::gpu_ops {
    using RGBConvParams = core::nn::RGBConvParams;
    using ConvParams = core::nn::Conv2dParams;
    using PoolReduceParams = core::nn::PoolReduceParams;

    // Share the core model's four function-pointer slots without a second inference path.
    struct LpipsOps : core::nn::LpipsDispatch {};

    // A pool reduce scores rows [y0, y1) and columns [x0, x1) of its [N, C, H, W] features, reading
    // the optional float32 weights at (row + weights_y0) * weights_width + col + weights_x0. Returns
    // the violated bound, or an empty string when the region and every weight it reads are in range.
    [[nodiscard]] inline std::string pool_reduce_region_error(const core::Tensor& features, const PoolReduceParams& p) {
        const auto h = static_cast<int64_t>(features.shape()[2]);
        const auto w = static_cast<int64_t>(features.shape()[3]);
        if (p.y0 < 0 || p.x0 < 0 || p.y1 > h || p.x1 > w)
            return std::format("LPIPS pool reduce region must lie inside the {}x{} features (y=[{}, {}), x=[{}, {}))",
                               h, w, p.y0, p.y1, p.x0, p.x1);
        if (!p.weights)
            return {};
        const core::Tensor& weights = *p.weights;
        if (!weights.is_valid() || weights.device() != core::Device::GPU || !weights.is_contiguous() ||
            weights.dtype() != core::DataType::Float32)
            return std::format("LPIPS pool reduce weights must be a contiguous GPU float32 tensor "
                               "(valid={}, device={}, dtype={}, contiguous={})",
                               weights.is_valid(), static_cast<int>(weights.device()), core::dtype_name(weights.dtype()),
                               weights.is_contiguous());
        if (p.y0 >= p.y1 || p.x0 >= p.x1)
            return {};
        const int64_t last_row = p.y1 - 1LL + p.weights_y0;
        const int64_t last_col = p.x1 - 1LL + p.weights_x0;
        if (p.weights_y0 < 0 || p.weights_x0 < 0 || last_col >= p.weights_width ||
            last_row * p.weights_width + last_col >= static_cast<int64_t>(weights.numel()))
            return std::format("LPIPS pool reduce weights must cover the scored region (weights_elements={}, width={}, "
                               "offset=({}, {}), y=[{}, {}), x=[{}, {}))",
                               weights.numel(), p.weights_width, p.weights_y0, p.weights_x0, p.y0, p.y1, p.x0, p.x1);
        return {};
    }
} // namespace lfs::gpu_ops

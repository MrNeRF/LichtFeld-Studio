/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/nn/lpips_dispatch.hpp"
#include "lfs/training/ops/types.hpp"

namespace lfs::gpu_ops {
    using RGBConvParams = core::nn::RGBConvParams;
    using ConvParams = core::nn::Conv2dParams;
    using PoolReduceParams = core::nn::PoolReduceParams;

    // Share the core model's four function-pointer slots without a second inference path.
    struct LpipsOps : core::nn::LpipsDispatch {};
} // namespace lfs::gpu_ops

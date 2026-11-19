/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "lfs/training/ops/types.hpp"

namespace lfs::training {
    struct RayDepthStep {
        core::Tensor state, means, scales, rotations, gradient, camera, proposals;
        core::Tensor frozen, crop, far_mask;
        float lr = 0, mean_lr = 0, far_limit = 0, frozen_scale = 0, crop_scale = 1, median_extent = 0;
    };
    void ray_depth_propose(const RayDepthStep& step);
    void ray_depth_commit(const RayDepthStep& step);
} // namespace lfs::training

/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/cuda/mask_normalization.cuh"
#include "core/device/portability.cuh"
#include "core/normal_prior_transform.hpp"
namespace hip_shared_image {
    using lfs::gpu_ops::NormalPriorTransform;
    constexpr float NORMALIZE_SCALE_U8 = 1.f / 255.f, NORMALIZE_SCALE_U16 = 1.f / 65535.f;
    using namespace lfs::core::cuda;
#include "core/device/image_format.cuh"
#include "core/device/image_sentinel.cuh"
#include "core/device/mask_grid.cuh"
} // namespace hip_shared_image

/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/device/portability.cuh"
#include "shared_image_kernels.hpp"
#include <cassert>
namespace hip_shared_resize {
    namespace cg = cooperative_groups;
    namespace cuda = lfs::core::cuda;
    constexpr int BLOCK_X = 16, BLOCK_Y = 16;
#include "core/device/lanczos.cuh"
#include "core/device/resize_pixels.cuh"
#include "core/device/resize_prior.cuh"
} // namespace hip_shared_resize

/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/tensor_spatial.hpp"

namespace lfs::core {
    struct PointFilterWindow {
        PointProjection projection;
        float near_depth = 0;
        float far_depth = std::numeric_limits<float>::infinity();
        float scale_x = 1, scale_y = 1, offset_x = 0, offset_y = 0;
    };

    struct PointFilter {
        const Tensor* transforms = nullptr;
        const Tensor* indices = nullptr;
        const Tensor* allowed = nullptr;
        const Tensor* box_transform = nullptr;
        const Tensor* box_min = nullptr;
        const Tensor* box_max = nullptr;
        const Tensor* ellipsoid_transform = nullptr;
        const Tensor* ellipsoid_radii = nullptr;
        bool box_inverse = false;
        bool ellipsoid_inverse = false;
        const PointFilterWindow* window = nullptr;
    };

    // Clear Bool/UInt8 [N] mask entries outside the supplied filters, preserving
    // all other byte values. Optional Float32 [N,3] points are required for
    // geometric filters. Row-major Float32 [T,4,4] transforms apply first;
    // Int32 [N] indices choose a transform, clamped to [0,T-1], or zero without
    // indices. Bool/UInt8 [K] allowed flags require indices and reject invalid
    // or disabled entries, including every entry when K == 0.
    // Box and ellipsoid transforms are Float32 column-major 4x4 world-to-local
    // matrices; bounds/radii have three values. Each shape includes its boundary
    // and may be inverted. Supply all tensors for a shape or none of them.
    // Windows are framebuffer-centred, with independent scale and offset axes;
    // projection uses the camera intrinsics, with image-centred orthographic
    // projection. Depth is positive forward distance, or radius for panoramas,
    // inside the inclusive near/far range. Panorama radii <= 1e-6 are rejected.
    // Inputs share the mask's device/backend. Offset/strided views, aliased
    // inputs and queued temporary tensors are supported. GPU work is async.
    LFS_CORE_API void filter_points(Tensor& mask, const Tensor* points, const PointFilter& filter);
} // namespace lfs::core

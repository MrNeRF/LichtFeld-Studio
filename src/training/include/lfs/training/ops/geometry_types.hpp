/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

namespace lfs::training::kernels {
    struct DepthAnchorCandidate {
        bool valid = false;
        float scale = 0.0f;
        float shift = 0.0f;
        float corr = 0.0f;
        int samples = 0;
    };

    // Per-camera alignment of the depth prior against sparse anchor points
    // (COLMAP / init point cloud), fitted once at startup. Keeps the target
    // depth absolute and multi-view consistent instead of chasing the render.
    struct DepthAnchor {
        bool valid = false;
        int model = 0; // 0 = disparity-space fit, 1 = depth-space fit
        float scale = 0.0f;
        float shift = 0.0f;
        float floor = 0.0f;
        float corr = 0.0f;
        int samples = 0;
        DepthAnchorCandidate disparity;
        DepthAnchorCandidate depth;
    };

} // namespace lfs::training::kernels

namespace lfs::gpu_ops {
    struct alignas(8) AnchorSample {
        float x, y;
    };
} // namespace lfs::gpu_ops

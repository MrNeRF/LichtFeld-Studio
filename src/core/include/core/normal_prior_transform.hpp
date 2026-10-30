/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
namespace lfs::gpu_ops {
    // Decode-side transform parameters for normal priors (v = n*0.5 + 0.5 file encoding).
    struct NormalPriorTransform {
        bool srgb = false;
        bool flip_yz = false;
        bool world_to_camera = false;
        float w2c[9] = {};
    };

} // namespace lfs::gpu_ops

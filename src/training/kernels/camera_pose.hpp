/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cuda_runtime.h>

namespace lfs::training::kernels {

    struct CameraPoseAdamStep {
        int row = 0;
        float lr_rotation = 0.0f;    // radians per step
        float lr_translation = 0.0f; // scene units per step
        float beta1 = 0.9f;
        float beta2 = 0.999f;
        float eps = 1e-8f;
    };

    // One Adam step of camera `row`'s offset (offsets, adam_m, adam_v [N, 6], steps [N]) from the rasterizer's gradient of
    // the pose it drew, compose(w2c0[row], offset) (camera_pose_math.hpp). Writes the refined pose to w2c [16] and
    // cam_position [3]. Skips a non-finite gradient.
    void launch_camera_pose_adam_step(
        const float* w2c0,
        float* offsets,
        float* adam_m,
        float* adam_v,
        float* steps,
        const float* grad_w2c,
        const float* grad_cam_position,
        float* w2c,
        float* cam_position,
        const CameraPoseAdamStep& step,
        cudaStream_t stream);

} // namespace lfs::training::kernels

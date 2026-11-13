/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "camera_pose.hpp"
#include "camera_pose_math.hpp"
#include "core/cuda_error.hpp"
#include "kernel_stream.hpp"

namespace lfs::training::kernels {
    namespace {
        __global__ void camera_pose_adam_step_kernel(
            const float* w2c0,
            float* offsets,
            float* adam_m,
            float* adam_v,
            float* steps,
            const float* grad_w2c,
            const float* grad_cam_position,
            float* w2c,
            float* cam_position,
            const CameraPoseAdamStep step) {
            const int row = step.row;
            const float* start = w2c0 + row * 16;
            float* off = offsets + row * 6;
            float grad[6];
            camera_pose::offset_grad(start, off, grad_w2c, grad_cam_position, grad);

            bool finite = true;
            for (int k = 0; k < 6; ++k)
                finite = finite && isfinite(grad[k]);
            if (finite) {
                const float t = steps[row] + 1.0f;
                const float bias1 = 1.0f - powf(step.beta1, t);
                const float bias2 = 1.0f - powf(step.beta2, t);
                for (int k = 0; k < 6; ++k) {
                    const int i = row * 6 + k;
                    const float m = step.beta1 * adam_m[i] + (1.0f - step.beta1) * grad[k];
                    const float v = step.beta2 * adam_v[i] + (1.0f - step.beta2) * grad[k] * grad[k];
                    adam_m[i] = m;
                    adam_v[i] = v;
                    const float lr = k < 3 ? step.lr_rotation : step.lr_translation;
                    off[k] -= lr * (m / bias1) / (sqrtf(v / bias2) + step.eps);
                }
                steps[row] = t;
            }
            camera_pose::compose(start, off, w2c, cam_position);
        }
    } // namespace

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
        cudaStream_t stream) {
        stream = resolve_stream(stream);
        camera_pose_adam_step_kernel<<<1, 1, 0, stream>>>(
            w2c0, offsets, adam_m, adam_v, steps, grad_w2c, grad_cam_position, w2c, cam_position, step);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.camera_pose.adam_step");
    }

} // namespace lfs::training::kernels

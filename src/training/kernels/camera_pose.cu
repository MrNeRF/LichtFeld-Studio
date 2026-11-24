/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "camera_pose.hpp"
#include "core/cuda_error.hpp"
#include "core/tensor_cuda_interop.hpp"
#include "kernel_stream.hpp"
#include "lfs/training/camera_pose_math.hpp"
#include "lfs/training/ops/pose.hpp"

namespace lfs::training::kernels {
    namespace {
        __global__ void camera_pose_adam_step_kernel(
            const float* w2c0,
            float* offsets,
            float* adam_m,
            float* adam_v,
            float* steps,
            const float* draw_to_owner,
            const float* grad_w2c,
            const float* grad_cam_position,
            float* w2c,
            float* cam_position,
            const CameraPoseAdamStep step) {
            const int row = step.row;
            const float* start = w2c0 + row * 16;
            float* off = offsets + row * 6;
            float grad[6];
            float owner_gradient[16];
            camera_pose::rotate_rows<false>(draw_to_owner + 9 * step.draw, grad_w2c, owner_gradient);
            camera_pose::offset_grad(start, off, owner_gradient, grad_cam_position, grad);

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
            float owner_pose[16];
            camera_pose::compose(start, off, owner_pose, cam_position);
            for (int i = 0; i < step.draw_count; ++i)
                camera_pose::rotate_rows<true>(draw_to_owner + 9 * (step.first_draw + i), owner_pose, w2c + 16 * i);
        }
    } // namespace

    void launch_camera_pose_adam_step(
        const float* w2c0,
        float* offsets,
        float* adam_m,
        float* adam_v,
        float* steps,
        const float* draw_to_owner,
        const float* grad_w2c,
        const float* grad_cam_position,
        float* w2c,
        float* cam_position,
        const CameraPoseAdamStep& step,
        cudaStream_t stream) {
        stream = resolve_stream(stream);
        camera_pose_adam_step_kernel<<<1, 1, 0, stream>>>(
            w2c0, offsets, adam_m, adam_v, steps, draw_to_owner, grad_w2c, grad_cam_position, w2c, cam_position, step);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.camera_pose.adam_step");
    }

} // namespace lfs::training::kernels

namespace lfs::training {
    const gpu_ops::PoseOps& cuda_pose_ops() {
        static const gpu_ops::PoseOps ops{[](gpu_ops::PoseTensors& t, gpu_ops::In g, gpu_ops::In c, const gpu_ops::PoseStep& p) {
            const auto stream = core::getCurrentCUDAStream();
            for (const auto* input : std::initializer_list<const core::Tensor*>{&t.start, &t.rotations, &t.offsets, &t.moment1, &t.moment2, &t.steps, &t.poses, &t.centre, &g, &c})
                input->sync_to_stream(stream);
            kernels::launch_camera_pose_adam_step(t.start.ptr<float>(), t.offsets.ptr<float>(), t.moment1.ptr<float>(),
                                                  t.moment2.ptr<float>(), t.steps.ptr<float>(), t.rotations.ptr<float>(), g.ptr<float>(), c.ptr<float>(),
                                                  t.poses.ptr<float>(), t.centre.ptr<float>(), {p.row, p.draw, p.first_draw, p.draw_count, p.lr_rotation, p.lr_translation}, stream);
            for (auto* output : {&t.offsets, &t.moment1, &t.moment2, &t.steps, &t.poses, &t.centre})
                output->set_stream(stream);
        }};
        return ops;
    }
} // namespace lfs::training

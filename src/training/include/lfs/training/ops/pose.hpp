// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "lfs/training/ops/types.hpp"
namespace lfs::gpu_ops {
    struct PoseStep {
        int row, draw, first_draw, draw_count;
        float lr_rotation, lr_translation;
    };
    struct PoseTensors {
        Tensor start, rotations, offsets, moment1, moment2, steps, poses, centre;
    };
    struct PoseOps {
        void (*step)(PoseTensors&, In matrix_gradient, In centre_gradient, const PoseStep&);
    };
} // namespace lfs::gpu_ops
namespace lfs::training {
    const gpu_ops::PoseOps& cuda_pose_ops();
    const gpu_ops::PoseOps& vulkan_pose_ops();
} // namespace lfs::training

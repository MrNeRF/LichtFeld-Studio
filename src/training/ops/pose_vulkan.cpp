// SPDX-License-Identifier: GPL-3.0-or-later
#include "lfs/training/ops/pose.hpp"
#include "vulkan/dispatch.hpp"
namespace lfs::training {
    const gpu_ops::PoseOps& vulkan_pose_ops() {
        static const gpu_ops::PoseOps ops{[](gpu_ops::PoseTensors& t, gpu_ops::In g, gpu_ops::In c, const gpu_ops::PoseStep& step) {
            using vulkan::address;
            using vulkan::ref;
            struct Parameters {
                uint64_t start, rotations, offsets, m, v, steps, poses, centre, grad_pose, grad_centre;
                gpu_ops::PoseStep step;
            };
            const Parameters parameters{address(t.start), address(t.rotations), address(t.offsets), address(t.moment1),
                                        address(t.moment2), address(t.steps), address(t.poses), address(t.centre), address(g), address(c), step};
            const std::array reads{ref(t.start), ref(t.rotations), ref(t.offsets), ref(t.moment1), ref(t.moment2), ref(t.steps), ref(g), ref(c)};
            const std::array writes{ref(t.offsets), ref(t.moment1), ref(t.moment2), ref(t.steps), ref(t.poses), ref(t.centre)};
            vulkan::dispatch("pose", parameters, reads, writes, 1, 0);
        }};
        return ops;
    }
} // namespace lfs::training

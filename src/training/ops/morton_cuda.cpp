/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs/training/ops/morton_cuda.hpp"

#include "core/tensor_cuda_interop.hpp"
#include "kernels/morton_reorder_kernels.hpp"

namespace lfs::training {
    namespace {
        using lfs::gpu_ops::JointCodecParams;
        using lfs::gpu_ops::JointLayout;
        using lfs::gpu_ops::Tensor;

        Tensor permutation(const Tensor& means) {
            return kernels::launch_morton_permutation(means, core::getCurrentCUDAStream());
        }

        void permute_joint(const Tensor& packed, const Tensor& bounds, const Tensor& permutation,
                           Tensor& new_packed, Tensor& new_bounds, const JointCodecParams& p) {
            const auto launch = p.layout == JointLayout::Rows
                                    ? kernels::launch_joint_permute_contiguous
                                    : kernels::launch_joint_permute_shN;
            launch(packed.ptr<std::uint8_t>(), bounds.ptr<float>(),
                   new_packed.ptr<std::uint8_t>(), new_bounds.ptr<float>(), permutation.ptr<std::int64_t>(),
                   p.primitives, p.attributes_or_slots, p.bits, core::getCurrentCUDAStream());
        }

        void permute_joint_grouped(Tensor& packed, const Tensor& bounds, const Tensor& permutation,
                                   Tensor& new_bounds, Tensor& scratch, const JointCodecParams& p) {
            kernels::launch_joint_permute_shN_grouped(
                packed.ptr<std::uint8_t>(), bounds.ptr<float>(), new_bounds.ptr<float>(),
                permutation.ptr<std::int64_t>(), p.primitives, p.attributes_or_slots, p.bits,
                scratch.ptr<std::uint8_t>(), scratch.bytes(), core::getCurrentCUDAStream());
        }

        const lfs::gpu_ops::MortonOps kCudaMortonOps{
            .permutation = permutation,
            .permute_joint = permute_joint,
            .permute_joint_grouped = permute_joint_grouped,
        };
    } // namespace

    const lfs::gpu_ops::MortonOps& cuda_morton_ops() {
        return kCudaMortonOps;
    }
} // namespace lfs::training

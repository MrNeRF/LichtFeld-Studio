/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs/training/ops/refine_cuda.hpp"

#include "core/tensor_cuda_interop.hpp"
#include "kernels/densification_kernels.hpp"
#include "kernels/pruning_kernels.hpp"

namespace lfs::training {
    namespace {
        using lfs::gpu_ops::RefineInputs;
        using lfs::gpu_ops::RefineOutputs;
        using lfs::gpu_ops::Tensor;

        // Preserve the mutable pointer access at the original call sites: even read-only
        // kernel inputs must detach pending lazy snapshots before escaping as raw pointers.
        void split(const RefineOutputs& parents, const RefineOutputs& children, const Tensor& indices) {
            kernels::launch_long_axis_split_gaussians_inplace(
                parents.means.ptr<float>(), parents.rotations.ptr<float>(), parents.scales.ptr<float>(),
                parents.sh0.ptr<float>(), nullptr, parents.opacity.ptr<float>(),
                children.means.ptr<float>(), children.rotations.ptr<float>(), children.scales.ptr<float>(),
                children.sh0.ptr<float>(), nullptr, children.opacity.ptr<float>(),
                const_cast<Tensor&>(indices).ptr<int64_t>(), indices.numel(), 0, core::getCurrentCUDAStream());
        }
        void fill_slots(const Tensor& indices, const RefineInputs& source, const RefineOutputs& destination, Tensor& free_mask) {
            kernels::launch_fill_free_slots_fused(
                const_cast<Tensor&>(indices).ptr<int64_t>(), indices.numel(), const_cast<Tensor&>(source.means).ptr<float>(), const_cast<Tensor&>(source.rotations).ptr<float>(),
                const_cast<Tensor&>(source.scales).ptr<float>(), const_cast<Tensor&>(source.sh0).ptr<float>(), const_cast<Tensor&>(source.opacity).ptr<float>(),
                destination.means.ptr<float>(), destination.rotations.ptr<float>(), destination.scales.ptr<float>(),
                destination.sh0.ptr<float>(), destination.opacity.ptr<float>(), destination.opacity.ndim() == 2 ? 1 : 0,
                free_mask.ptr<bool>(), destination.means.shape()[0], core::getCurrentCUDAStream());
        }
        void counts(const Tensor& bool0, const Tensor& bool1, const Tensor& float0, const Tensor& float1, Tensor& counts4) {
            kernels::launch_packed_refine_counts(
                bool0.is_valid() ? const_cast<Tensor&>(bool0).ptr<bool>() : nullptr, bool0.is_valid() ? bool0.numel() : 0,
                bool1.is_valid() ? const_cast<Tensor&>(bool1).ptr<bool>() : nullptr, bool1.is_valid() ? bool1.numel() : 0,
                float0.is_valid() ? const_cast<Tensor&>(float0).ptr<float>() : nullptr, float0.is_valid() ? float0.numel() : 0,
                float1.is_valid() ? const_cast<Tensor&>(float1).ptr<float>() : nullptr, float1.is_valid() ? float1.numel() : 0,
                counts4.ptr<int64_t>(), core::getCurrentCUDAStream());
        }
        void normalize_positive_median(Tensor& values) {
            kernels::launch_normalize_by_positive_median(values.ptr<float>(), values.numel(), core::getCurrentCUDAStream());
        }
        void clip_scales(Tensor& scales, const Tensor& shares, const Tensor& frozen, float limit) {
            const bool* frozen_data = frozen.is_valid() ? frozen.ptr<bool>() : nullptr;
            kernels::launch_clip_log_scale_by_screen_share(
                scales.ptr<float>(), const_cast<Tensor&>(shares).ptr<float>(), frozen_data,
                frozen.is_valid() ? frozen.numel() : 0, limit, scales.shape()[0], core::getCurrentCUDAStream());
        }
        void oversize_scores(const Tensor& error, const Tensor& shares, const Tensor& frozen, Tensor& scores, float limit) {
            const bool* frozen_data = frozen.is_valid() ? frozen.ptr<bool>() : nullptr;
            kernels::launch_oversize_split_scores(
                const_cast<Tensor&>(error).ptr<float>(), const_cast<Tensor&>(shares).ptr<float>(), frozen_data,
                frozen.is_valid() ? frozen.numel() : 0, scores.ptr<float>(), limit, error.numel(), core::getCurrentCUDAStream());
        }
        void dead_mask(const Tensor& opacity, const Tensor& rotations, Tensor& mask, float minimum_opacity) {
            pruning::launch_compute_dead_mask(const_cast<Tensor&>(opacity).ptr<float>(), rotations.ptr<float>(), mask.ptr<uint8_t>(),
                                              opacity.numel(), minimum_opacity, core::getCurrentCUDAStream());
        }
        void rotation_mask(const Tensor& rotations, Tensor& mask) {
            pruning::launch_compute_near_zero_rotation_mask(rotations.ptr<float>(), mask.ptr<uint8_t>(),
                                                            rotations.shape()[0], core::getCurrentCUDAStream());
        }
        const lfs::gpu_ops::RefineOps kCudaRefineOps{
            .split = split,
            .fill_slots = fill_slots,
            .counts = counts,
            .normalize_positive_median = normalize_positive_median,
            .clip_scales = clip_scales,
            .oversize_scores = oversize_scores,
            .dead_mask = dead_mask,
            .rotation_mask = rotation_mask,
        };
    } // namespace

    const lfs::gpu_ops::RefineOps& cuda_refine_ops() {
        return kCudaRefineOps;
    }
} // namespace lfs::training

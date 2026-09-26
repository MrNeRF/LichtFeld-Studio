/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs/training/ops/adam_cuda.hpp"

#include "adam_api.h"
#include "core/splat_exportable_storage.hpp"
#include "core/tensor_cuda_interop.hpp"

#include <cstdint>
#include <stdexcept>

namespace lfs::training {
    namespace {

        using lfs::gpu_ops::AdamHyper;
        using lfs::gpu_ops::AdamMasks;
        using lfs::gpu_ops::AdamModifiers;
        using lfs::gpu_ops::JointCodecParams;
        using lfs::gpu_ops::JointLayout;
        using lfs::gpu_ops::JointStep;
        using lfs::gpu_ops::ShStepParams;
        using lfs::gpu_ops::Tensor;

        template <typename T>
        [[nodiscard]] const T* optional_ptr(const Tensor& tensor) {
            return tensor.is_valid() && tensor.numel() > 0 ? tensor.ptr<T>() : nullptr;
        }

        [[nodiscard]] int count(const Tensor& tensor) {
            return static_cast<int>(tensor.numel());
        }

        void adam_step_batch(
            const std::span<const JointStep> steps, const AdamMasks& masks,
            const AdamHyper& hyper, const AdamModifiers& modifiers) {
            fast_lfs::optimizer::JointContiguousBatchEntry entries[fast_lfs::optimizer::kMaxContiguousBatch];
            int n_entries = 0;
            for (const JointStep& step : steps) {
                if (!step.parameter.is_valid()) {
                    continue;
                }
                if (n_entries == fast_lfs::optimizer::kMaxContiguousBatch) {
                    throw std::runtime_error("adam_step_batch: too many steps");
                }
                if (step.bits != 16) {
                    throw std::runtime_error("adam_step_batch: moments must be 16-bit");
                }
                auto& entry = entries[n_entries++];
                entry.param = step.parameter.ptr<float>();
                entry.packed = step.packed.ptr<std::uint8_t>();
                entry.bounds = step.bounds.ptr<float>();
                entry.grad = step.gradient.ptr<float>();
                entry.n_prims = step.primitives;
                entry.n_attr = step.attributes;
                entry.lr = step.lr;
                entry.bias_correction1_rcp = step.bc1_rcp;
                entry.bias_correction2_sqrt_rcp = step.bc2_sqrt_rcp;
                entry.apply_mean_step = step.apply_mean_step ? 1 : 0;
                entry.apply_screen_share = step.apply_screen_share ? 1 : 0;
            }
            fast_lfs::optimizer::adam_step_joint_contiguous_batched(
                entries,
                n_entries,
                optional_ptr<bool>(masks.frozen),
                count(masks.frozen),
                modifiers.frozen_lr_scale,
                optional_ptr<bool>(masks.crop_damping),
                count(masks.crop_damping),
                modifiers.cropbox_lr_scale,
                hyper.beta1,
                hyper.beta2,
                hyper.eps,
                lfs::core::getCurrentCUDAStream(),
                optional_ptr<float>(masks.raw_scales),
                count(masks.raw_scales),
                modifiers.median_extent,
                modifiers.r_min,
                modifiers.r_max,
                optional_ptr<bool>(masks.far_mask),
                count(masks.far_mask),
                optional_ptr<float>(masks.screen_share),
                count(masks.screen_share),
                modifiers.screen_share_limit,
                modifiers.screen_share_penalty);
        }

        void adam_step_sh(
            Tensor& parameter, Tensor& packed, Tensor& bounds, Tensor& value_bounds,
            const Tensor& gradient, const AdamMasks& masks,
            const AdamHyper& hyper, const AdamModifiers& modifiers,
            const ShStepParams& params) {
            // Exportable storage can move on growth; resolve through its live block.
            fast_lfs::optimizer::adam_step_shN_joint_from_grad(
                static_cast<float*>(lfs::core::resolve_exportable_device_ptr(parameter)),
                packed.ptr<std::uint8_t>(),
                bounds.ptr<float>(),
                value_bounds.is_valid()
                    ? static_cast<float*>(lfs::core::resolve_exportable_device_ptr(value_bounds))
                    : nullptr,
                gradient.ptr<float>(),
                optional_ptr<bool>(masks.frozen),
                count(masks.frozen),
                modifiers.frozen_lr_scale,
                optional_ptr<bool>(masks.crop_damping),
                count(masks.crop_damping),
                modifiers.cropbox_lr_scale,
                params.primitives,
                params.layout_slots,
                params.active_bases,
                params.value_bits,
                params.value_cells,
                params.step_size,
                hyper.beta1,
                hyper.beta2,
                hyper.eps,
                params.bc2_sqrt_rcp,
                lfs::core::getCurrentCUDAStream());
        }

        void adam_encode_zero(
            Tensor& packed, Tensor& bounds, const Tensor& indices,
            const JointCodecParams& params) {
            if (params.layout == JointLayout::SwizzledSH) {
                fast_lfs::optimizer::joint_encode_zero_shN_at_indices(
                    packed.ptr<std::uint8_t>(),
                    bounds.ptr<float>(),
                    indices.ptr<int64_t>(),
                    count(indices),
                    params.attributes_or_slots,
                    params.bits,
                    params.primitives,
                    lfs::core::getCurrentCUDAStream());
                return;
            }
            fast_lfs::optimizer::joint_encode_zero_rows_at_indices(
                packed.ptr<std::uint8_t>(),
                bounds.ptr<float>(),
                indices.ptr<int64_t>(),
                count(indices),
                params.attributes_or_slots,
                params.bits,
                params.primitives,
                lfs::core::getCurrentCUDAStream());
        }

        const lfs::gpu_ops::AdamOps kCudaAdamOps{
            .step_batch = adam_step_batch,
            .step_sh = adam_step_sh,
            .encode_zero = adam_encode_zero,
        };

    } // namespace

    const lfs::gpu_ops::AdamOps& cuda_adam_ops() {
        return kCudaAdamOps;
    }

} // namespace lfs::training

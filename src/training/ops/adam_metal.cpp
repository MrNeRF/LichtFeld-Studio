/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Metal AdamOps; see ops/adam_cuda.cpp for the reference behaviour.

#include "metal_families.hpp"
#include "metal_kernels.hpp"

#include <algorithm>
#include <format>
#include <span>
#include <stdexcept>
#include <vector>

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
        namespace mk = metal;

        constexpr int kMaxSteps = 6;
        constexpr uint32_t kBlock = 256;

        int count(const Tensor& tensor) { return tensor.is_valid() ? static_cast<int>(tensor.numel()) : 0; }

        uint32_t blocks(const int primitives) { return (static_cast<uint32_t>(primitives) + kBlock - 1) / kBlock; }

        struct RowMasks {
            uint64_t frozen, crop;
            int32_t frozen_count, crop_count;
            float frozen_lr_scale, cropbox_lr_scale;
        };

        RowMasks row_masks(const AdamMasks& masks, const AdamModifiers& modifiers) {
            return {mk::address(masks.frozen), mk::address(masks.crop_damping), count(masks.frozen),
                    count(masks.crop_damping), modifiers.frozen_lr_scale, modifiers.cropbox_lr_scale};
        }

        struct StepEntry {
            uint64_t parameter, packed, bounds, gradient;
            int32_t primitives, attributes;
            float lr, bc1_rcp, bc2_sqrt_rcp;
            uint32_t apply_mean_step, apply_screen_share;
        };

        struct BatchParams {
            StepEntry steps[kMaxSteps];
            RowMasks rows;
            uint64_t raw_scales, far_mask, screen_share;
            int32_t raw_scales_count, far_count, screen_share_count;
            float median_extent, r_min, r_max, screen_share_limit, screen_share_penalty;
            float beta1, beta2, eps;
        };

        void step_batch(const std::span<const JointStep> steps, const AdamMasks& masks, const AdamHyper& hyper,
                        const AdamModifiers& modifiers) {
            BatchParams params{
                .steps = {},
                .rows = row_masks(masks, modifiers),
                .raw_scales = mk::address(masks.raw_scales),
                .far_mask = mk::address(masks.far_mask),
                .screen_share = mk::address(masks.screen_share),
                .raw_scales_count = count(masks.raw_scales),
                .far_count = count(masks.far_mask),
                .screen_share_count = count(masks.screen_share),
                .median_extent = modifiers.median_extent,
                .r_min = modifiers.r_min,
                .r_max = modifiers.r_max,
                .screen_share_limit = modifiers.screen_share_limit,
                .screen_share_penalty = modifiers.screen_share_penalty,
                .beta1 = hyper.beta1,
                .beta2 = hyper.beta2,
                .eps = hyper.eps,
            };
            std::vector<const Tensor*> uses{&masks.frozen, &masks.crop_damping, &masks.raw_scales, &masks.far_mask,
                                            &masks.screen_share};
            int entries = 0;
            int max_primitives = 0;
            for (const JointStep& step : steps) {
                if (!step.parameter.is_valid())
                    continue;
                if (entries == kMaxSteps)
                    throw std::runtime_error(std::format("adam step_batch takes at most {} steps", kMaxSteps));
                if (step.bits != 16)
                    throw std::runtime_error(std::format("adam step_batch moments must be 16-bit, got {}", step.bits));
                if (!step.packed.is_valid() || !step.bounds.is_valid() || !step.gradient.is_valid())
                    throw std::runtime_error(std::format("adam step_batch step {} lacks packed, bounds or gradient",
                                                         entries));
                if (step.primitives <= 0 || step.attributes <= 0)
                    throw std::runtime_error(std::format("adam step_batch step {} has {} primitives x {} attributes",
                                                         entries, step.primitives, step.attributes));
                params.steps[entries++] = {
                    .parameter = mk::address(step.parameter),
                    .packed = mk::address(step.packed),
                    .bounds = mk::address(step.bounds),
                    .gradient = mk::address(step.gradient),
                    .primitives = step.primitives,
                    .attributes = step.attributes,
                    .lr = step.lr,
                    .bc1_rcp = step.bc1_rcp,
                    .bc2_sqrt_rcp = step.bc2_sqrt_rcp,
                    .apply_mean_step = step.apply_mean_step ? 1u : 0u,
                    .apply_screen_share = step.apply_screen_share ? 1u : 0u,
                };
                max_primitives = std::max(max_primitives, step.primitives);
                uses.insert(uses.end(), {&step.parameter, &step.packed, &step.bounds, &step.gradient});
            }
            if (entries == 0)
                return;
            mk::kernels().launch({.function = "adam_step_batch",
                                  .params = std::as_bytes(std::span(&params, 1)),
                                  .uses = uses,
                                  .groups = {blocks(max_primitives), static_cast<uint32_t>(entries), 1},
                                  .group = {kBlock, 1, 1}});
        }

        struct ShParams {
            uint64_t parameter, packed, bounds, value_bounds, gradient;
            RowMasks rows;
            int32_t primitives;
            uint32_t layout_slots, active_slots, value_mode, value_cells;
            float step_size, beta1, beta2, eps, bc2_sqrt_rcp;
        };

        void step_sh(Tensor& parameter, Tensor& packed, Tensor& bounds, Tensor& value_bounds, const Tensor& gradient,
                     const AdamMasks& masks, const AdamHyper& hyper, const AdamModifiers& modifiers,
                     const ShStepParams& p) {
            if (!parameter.is_valid() || !packed.is_valid() || !bounds.is_valid() || !gradient.is_valid())
                throw std::runtime_error("adam step_sh needs parameter, packed moments, bounds and gradient");
            if (p.primitives <= 0 || p.layout_slots <= 0)
                return;
            if (p.active_bases != 4 && p.active_bases != 9 && p.active_bases != 16)
                throw std::runtime_error(std::format("adam step_sh active bases must be 4, 9 or 16, got {}",
                                                     p.active_bases));
            if (p.value_bits != 0 && p.value_bits != 16)
                throw std::runtime_error(std::format("adam step_sh value bits must be 0 or 16, got {}", p.value_bits));
            const bool q16 = p.value_bits == 16 && value_bounds.is_valid() && p.value_cells > 0;
            const ShParams params{
                .parameter = mk::address(parameter),
                .packed = mk::address(packed),
                .bounds = mk::address(bounds),
                .value_bounds = q16 ? mk::address(value_bounds) : 0,
                .gradient = mk::address(gradient),
                .rows = row_masks(masks, modifiers),
                .primitives = p.primitives,
                .layout_slots = static_cast<uint32_t>(p.layout_slots),
                .active_slots = p.active_bases > 9 ? 12u : p.active_bases > 4 ? 6u
                                                                              : 3u,
                .value_mode = q16 ? 2u : p.value_bits == 16 ? 1u
                                                            : 0u,
                .value_cells = q16 ? static_cast<uint32_t>(p.value_cells) : 0u,
                .step_size = p.step_size,
                .beta1 = hyper.beta1,
                .beta2 = hyper.beta2,
                .eps = hyper.eps,
                .bc2_sqrt_rcp = p.bc2_sqrt_rcp,
            };
            mk::launch("adam_step_sh", params,
                       {&parameter, &packed, &bounds, &value_bounds, &gradient, &masks.frozen, &masks.crop_damping},
                       blocks(p.primitives));
        }

        struct ZeroMarkParams {
            uint64_t indices, flags, touched;
            int32_t count, primitives;
        };

        struct ZeroParams {
            uint64_t packed, bounds, flags, touched;
            int32_t primitives;
            uint32_t width, swizzled;
            int32_t bits;
        };

        void encode_zero(Tensor& packed, Tensor& bounds, const Tensor& indices, const JointCodecParams& p) {
            if (!packed.is_valid() || !bounds.is_valid() || !indices.is_valid())
                throw std::runtime_error("adam encode_zero needs packed moments, bounds and indices");
            if (indices.dtype() != core::DataType::Int64)
                throw std::invalid_argument(std::format("adam encode_zero indices must be Int64, got {}",
                                                        core::dtype_name(indices.dtype())));
            const bool swizzled = p.layout == JointLayout::SwizzledSH;
            if (indices.numel() == 0)
                return;
            if (p.attributes_or_slots <= 0) {
                if (swizzled)
                    return;
                throw std::runtime_error(std::format("adam encode_zero needs positive attributes, got {}",
                                                     p.attributes_or_slots));
            }
            if (p.primitives <= 0)
                throw std::runtime_error(std::format("adam encode_zero needs positive primitives, got {}", p.primitives));
            if (p.bits != 8 && p.bits != 16)
                throw std::runtime_error(std::format("adam encode_zero bits must be 8 or 16, got {}", p.bits));

            const uint32_t n_blocks = blocks(p.primitives);
            Tensor flags = Tensor::zeros({static_cast<size_t>(p.primitives)}, core::Device::GPU, core::DataType::UInt8);
            Tensor touched = Tensor::zeros({static_cast<size_t>(n_blocks)}, core::Device::GPU, core::DataType::Int32);
            const ZeroMarkParams mark{mk::address(indices), mk::address(flags), mk::address(touched), count(indices),
                                      p.primitives};
            mk::launch_items("adam_encode_zero_mark", mark, {&indices, &flags, &touched}, indices.numel());
            const ZeroParams params{mk::address(packed), mk::address(bounds), mk::address(flags),
                                    mk::address(touched), p.primitives, static_cast<uint32_t>(p.attributes_or_slots),
                                    swizzled ? 1u : 0u, p.bits};
            mk::launch("adam_encode_zero", params, {&packed, &bounds, &flags, &touched}, n_blocks);
        }

        void validate_far_mask(const bool* pointer) {
            if (pointer == nullptr)
                throw std::invalid_argument("mean-step far mask must not be null");
        }
    } // namespace

    const lfs::gpu_ops::AdamOps& metal_adam_ops() {
        static const lfs::gpu_ops::AdamOps ops{
            .validate_far_mask = validate_far_mask,
            .step_batch = step_batch,
            .step_sh = step_sh,
            .encode_zero = encode_zero,
        };
        return ops;
    }
} // namespace lfs::training

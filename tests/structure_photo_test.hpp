/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "lfs/training/ops/registry.hpp"

namespace lfs::test::structure {
    using core::Tensor;
    struct PhotoWorkspace {
        gpu_ops::PhotoSaved saved;
        Tensor ssim_map, cs_map, gradient, raw_gradient;
    };
    struct PhotoContext {
        float mask_sum_value = 0;
        Tensor gradient, raw_gradient;
    };
    struct Gradients {
        Tensor grad_corrected, grad_raw;
    };
    inline std::pair<Tensor, PhotoContext> evaluate(const Tensor& image, const Tensor& raw, const Tensor& target,
                                                    const Tensor& mask, float weight, PhotoWorkspace& workspace,
                                                    bool padding, float denominator, gpu_ops::PhotoPath path) {
        const auto& ops = *training::training_ops(core::default_gpu_backend()).photometric;
        if (!workspace.saved.backend)
            workspace.saved.backend = ops.create();
        Tensor loss;
        ops.evaluate(workspace.saved, image, raw, target, mask, {.path = path, .ssim_weight = weight, .valid_padding = padding, .denominator = denominator},
                     loss, workspace.gradient, workspace.raw_gradient);
        if (raw.is_valid() && ops.add_raw_gradient) {
            workspace.raw_gradient = Tensor::zeros_like(image);
            ops.add_raw_gradient(workspace.saved, workspace.raw_gradient);
        }
        workspace.ssim_map = workspace.saved.ssim_map;
        workspace.cs_map = workspace.saved.cs_map;
        const float normalization = denominator > 0 ? denominator : (mask.is_valid() ? mask.sum().item<float>() * 3.0f + 1e-8f : 0.0f);
        return {loss, {normalization, workspace.gradient, workspace.raw_gradient}};
    }
    inline auto fused_forward(const Tensor& a, const Tensor& b, float weight, PhotoWorkspace& workspace, bool padding) {
        return evaluate(a, {}, b, {}, weight, workspace, padding, 0, gpu_ops::PhotoPath::Fused);
    }
    inline auto masked_forward(const Tensor& a, const Tensor& b, const Tensor& mask, float weight, PhotoWorkspace& workspace, float denominator) {
        return evaluate(a, {}, b, mask, weight, workspace, true, denominator, gpu_ops::PhotoPath::MaskedFused);
    }
    inline auto decoupled_forward(const Tensor& a, const Tensor& raw, const Tensor& b, float weight, PhotoWorkspace& workspace, bool padding) {
        return evaluate(a, raw, b, {}, weight, workspace, padding, 0, gpu_ops::PhotoPath::Decoupled);
    }
    inline auto masked_decoupled_forward(const Tensor& a, const Tensor& raw, const Tensor& b, const Tensor& mask, float weight, PhotoWorkspace& workspace, float denominator) {
        return evaluate(a, raw, b, mask, weight, workspace, true, denominator, gpu_ops::PhotoPath::MaskedDecoupled);
    }
    inline Tensor fused_backward(const PhotoContext& context, PhotoWorkspace&) { return context.gradient; }
    inline Gradients decoupled_backward(const PhotoContext& context, PhotoWorkspace&) { return {context.gradient, context.raw_gradient}; }
    inline Tensor raw_gradient(const PhotoContext& context) { return context.raw_gradient; }
    inline void accumulate(const Tensor& gradient, Tensor& output) {
        if (gradient.is_valid())
            output.add_(gradient);
    }
} // namespace lfs::test::structure

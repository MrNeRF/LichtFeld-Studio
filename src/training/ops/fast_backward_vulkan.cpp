/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "lfs/training/ops/adam_vulkan.hpp"
#include "lfs/training/ops/fast_vulkan.hpp"
#include "lfs/training/vram_ledger.hpp"
#include "vulkan/fast_state.hpp"
#include <array>

namespace lfs::training::vulkan {
    namespace {
        using namespace gpu_ops;
        using core::Device;
        struct BackPush {
            FastPush forward;
            uint64_t image_gradient, alpha_gradient, depth_gradient, normal_gradient, screen_gradient;
            uint64_t densification, error, edge, scores;
            uint64_t mean_gradient, scale_gradient, rotation_gradient, opacity_gradient, dc_gradient, sh_gradient;
            uint64_t sigmoid, z, u;
            uint32_t densify, stage, slots, sparsity_count;
            float scale_weight, flatten_weight, opacity_weight, rho, grad_loss;
            uint32_t scale_elements, opacity_elements;
        };
        static_assert(sizeof(BackPush) == 512);
        static_assert(offsetof(BackPush, image_gradient) == 320);
        static_assert(offsetof(BackPush, densify) == 464);
        static_assert(offsetof(BackPush, scale_weight) == 480);
        static_assert(offsetof(BackPush, opacity_elements) == 504);
    } // namespace
    void fast_backward(FastSaved& saved, const RenderGradients& gradient, Out densification, In error,
                       In edge, Out scores, const BackwardAdam& adam, DensificationType type) {
        LFS_ASSERT_MSG(saved.backend != nullptr, "Fast backward requires a created state");
        auto& s = static_cast<FastState&>(*saved.backend);
        LFS_ASSERT_MSG(s.live, "Fast backward requires a live forward frame");
        auto& p = s.push;
        if (p.count == 0)
            return;
        for (const Tensor* tensor : std::initializer_list<const Tensor*>{&gradient.image, &gradient.alpha, &gradient.depth, &gradient.normal, &error, &edge, &scores, &densification}) {
            if (tensor->is_valid() && tensor->numel()) {
                LFS_ASSERT_MSG(tensor->dtype() == core::DataType::Float32 && tensor->is_contiguous(), "Fast backward bindings must be contiguous float");
                (void)ref(*tensor);
            }
        }
        const size_t pixels = size_t(p.width) * p.height;
        for (const auto [tensor, expected] : {std::pair{&gradient.image, pixels * 3}, std::pair{&gradient.alpha, pixels}, std::pair{&gradient.depth, pixels}, std::pair{&gradient.normal, pixels * 3}, std::pair{&error, pixels}, std::pair{&edge, pixels}})
            LFS_ASSERT_MSG(!tensor->is_valid() || tensor->numel() == expected, "Fast image gradient shape mismatch");
        // As on CUDA and Metal, statistics that do not cover every splat (the
        // strategy stops refining) are not updated.
        const Tensor no_densification;
        const bool update_densification =
            densification.is_valid() && densification.ndim() == 2 && densification.shape()[1] >= size_t(p.count);
        const Tensor& stats = update_densification ? densification : no_densification;
        LFS_ASSERT_MSG(!scores.is_valid() || scores.numel() >= p.count, "Fast edge scores need one row");
        s.mark(14);
        std::array<Tensor, 6> gradients;
        constexpr std::array<size_t, 5> widths{3, 3, 4, 1, 3};
        for (size_t i = 0; i < 5; ++i)
            gradients[i] = s.temporary(20 + i, size_t(p.count) * widths[i]).reshape(lfs::core::TensorShape{size_t(p.count), widths[i]});
        const uint32_t slots = (p.rest * 3 + 3) / 4;
        gradients[5] = s.temporary(25, size_t((p.count + 31) / 32) * 32 * slots * 4);
        s.mark(15);
        s.mark(10);
        Tensor screen_gradient = s.temporary(26, size_t(p.visible) * 16);
        BackPush b{};
        b.forward = p;
        b.image_gradient = address(gradient.image);
        b.alpha_gradient = address(gradient.alpha);
        b.depth_gradient = address(gradient.depth);
        b.normal_gradient = address(gradient.normal);
        b.screen_gradient = address(screen_gradient);
        b.densification = address(stats);
        b.error = address(error);
        b.edge = address(edge);
        b.scores = address(scores);
        // Match the blend dispatch contract: pixel errors request weighted
        // statistics even when the strategy supplies None (as IGS+ does).
        const auto blend_type = type == DensificationType::MRNF && b.densification != 0
                                    ? DensificationType::MRNF
                                : b.densification != 0 && b.error != 0 ? DensificationType::MCMC
                                                                       : DensificationType::None;
        b.densify = static_cast<uint32_t>(blend_type);
        b.mean_gradient = address(gradients[0]);
        b.scale_gradient = address(gradients[1]);
        b.rotation_gradient = address(gradients[2]);
        b.opacity_gradient = address(gradients[3]);
        b.dc_gradient = address(gradients[4]);
        b.sh_gradient = address(gradients[5]);
        b.slots = slots;
        const auto& scale = adam.groups[static_cast<size_t>(AdamSlot::Scaling)];
        const auto& opacity = adam.groups[static_cast<size_t>(AdamSlot::Opacity)];
        b.scale_weight = scale.enabled ? adam.scale_reg_weight : 0;
        b.flatten_weight = scale.enabled ? adam.flatten_reg_weight : 0;
        b.opacity_weight = opacity.enabled ? adam.opacity_reg_weight : 0;
        b.scale_elements = scale.enabled ? scale.elements : 0;
        b.opacity_elements = opacity.enabled ? opacity.elements : 0;
        if (adam.sparsity_sigmoid.is_valid() && adam.sparsity_z.is_valid() && adam.sparsity_u.is_valid()) {
            b.sigmoid = address(adam.sparsity_sigmoid);
            b.z = address(adam.sparsity_z);
            b.u = address(adam.sparsity_u);
            b.sparsity_count = std::min({adam.sparsity_sigmoid.numel(), adam.sparsity_z.numel(), adam.sparsity_u.numel()});
            b.rho = adam.sparsity_rho;
            b.grad_loss = adam.sparsity_grad_loss;
        }
        auto reads = fast_storage(s);
        for (const Tensor* t : {&gradient.image, &gradient.alpha, &gradient.depth, &gradient.normal, &error, &edge,
                                &adam.sparsity_sigmoid, &adam.sparsity_z, &adam.sparsity_u})
            if (t->is_valid() && t->numel())
                reads.push_back(ref(*t));
        std::vector<StorageRef> writes;
        for (auto& t : gradients)
            if (t.numel())
                writes.push_back(ref(t));
        for (const Tensor* t : std::initializer_list<const Tensor*>{&screen_gradient, &stats, &scores})
            if (t->is_valid() && t->numel()) {
                reads.push_back(ref(*t));
                writes.push_back(ref(*t));
            }
        const auto context = core::internal::acquire_vulkan_context();
        const std::string_view shader = context->caps().shader_atomic_float ? "fast_backward_atomic" : "fast_backward";
        const uint32_t variant = specialization(p, 0) |
                                 (uint32_t(b.depth_gradient != 0) << 18) | (uint32_t(b.normal_gradient != 0) << 19) |
                                 (uint32_t(b.densify != 0 && b.densification != 0) << 20) | (uint32_t(b.scores != 0) << 21);
        if (p.visible) {
            b.stage = 2;
            dispatch(shader, b, reads, writes,
                     std::min((p.count + 31u) / 32u, context->caps().max_workgroup_count[0]), variant | b.stage);
            b.stage = 0;
        }
        if (p.visible)
            dispatch(shader, b, reads, writes, std::min(p.grid_width * p.grid_height, context->caps().max_workgroup_count[0]), variant | b.stage);
        s.mark(11);
        s.mark(12);
        b.stage = 1;
        dispatch(shader, b, reads, writes, std::min((p.count + 127) / 128, context->caps().max_workgroup_count[0]), variant | b.stage);
        // Read every old-state gradient and regularizer before any optimizer write.
        if (adam.scale_reg_loss.is_valid() && b.scale_weight > 0 && b.scale_elements > 0)
            adam.scale_reg_loss.add_(s.inputs[1].exp().sum().unsqueeze(0) * (b.scale_weight / float(b.scale_elements)));
        if (adam.opacity_reg_loss.is_valid() && b.opacity_weight > 0 && b.opacity_elements > 0)
            adam.opacity_reg_loss.add_(s.inputs[3].sigmoid().sum().unsqueeze(0) * (b.opacity_weight / float(b.opacity_elements)));
        const AdamHyper hyper{adam.beta1, adam.beta2, adam.eps};
        for (size_t slot = 0; slot < adam.groups.size(); ++slot) {
            const auto& g = adam.groups[slot];
            if (!g.enabled)
                continue;
            LFS_ASSERT_MSG(g.parameter.is_valid() && g.packed_moments.is_valid() && g.joint_bounds.is_valid(), "Enabled Fast Adam group is missing storage");
            // CUDA has no update for unsupported fused widths.
            if (slot == 5 ? g.joint_bits != 8 : (g.joint_bits != 8 && g.joint_bits != 16))
                continue;
            const AdamMasks masks{g.frozen_mask, g.crop_damping_mask, s.inputs[1], adam.far_mask, g.screen_share};
            const AdamModifiers modifiers{g.frozen_lr_scale, g.cropbox_lr_scale, adam.median_extent, adam.r_min, adam.r_max,
                                          g.screen_share_limit, g.screen_share_penalty};
            if (slot == 5) {
                if (p.active_bases > 1)
                    vulkan_adam_ops().step_sh(g.parameter, g.packed_moments, g.joint_bounds, g.sh_value_bounds,
                                              gradients[slot], masks, hyper, modifiers, {g.primitives, int(slots), int(p.active_bases), g.value_bits, g.value_cells, g.step_size, g.bc2_sqrt_rcp});
            } else {
                LFS_ASSERT_MSG(g.primitives == int(p.count) && g.attributes == int(widths[slot]) && g.elements == int(p.count * widths[slot]), "Fast Adam row binding size mismatch");
                const JointStep step{g.parameter, g.packed_moments, g.joint_bounds, gradients[slot], g.primitives, g.attributes, g.joint_bits,
                                     g.step_size, 1.f, g.bc2_sqrt_rcp, slot == 0 && adam.per_splat_mean_step, g.screen_share.is_valid()};
                vulkan_fused_adam_rows(step, masks, hyper, modifiers);
            }
        }
        s.mark(13);
    }
} // namespace lfs::training::vulkan
namespace lfs::training {
    namespace {
        using namespace gpu_ops;
        State create() { return std::make_unique<vulkan::FastState>(); }
        void release(FastSaved& saved) noexcept {
            if (saved.backend) {
                auto& state = static_cast<vulkan::FastState&>(*saved.backend);
                const bool indirect = state.indirect;
                auto* timer = state.timer;
                auto scratch = std::move(state.scratch);
                auto readback = std::move(state.scalar_readback);
                state = vulkan::FastState{};
                state.timer = timer;
                state.indirect = indirect;
                state.scratch = std::move(scratch);
                state.scalar_readback = std::move(readback);
            }
        }
        void warmup() {
            // Prewarm a visible frame on the caller's tensor execution target.
            FastSaved saved{.backend = create()};
            auto means = Tensor::from_vector(std::vector<float>{0.f, 0.f, 2.f}, {1, 3}, core::Device::GPU);
            auto scales = Tensor::full({1, 3}, -2.f, core::Device::GPU);
            auto rotations = Tensor::from_vector(std::vector<float>{1.f, 0.f, 0.f, 0.f}, {1, 4}, core::Device::GPU);
            auto opacities = Tensor::zeros({1}, core::Device::GPU);
            auto dc = Tensor::zeros({1, 3}, core::Device::GPU);
            auto view = Tensor::eye(4, core::Device::GPU);
            auto camera = Tensor::zeros({3}, core::Device::GPU);
            Tensor absent, image, alpha, depth, normal, share;
            (void)vulkan::fast_forward(saved, {means, scales, rotations, opacities, dc, absent, absent}, view, camera, absent, absent,
                                       {.full_image = {16, 16}, .intrinsics = {16, 16, 8, 8}}, {image, alpha, depth, normal}, share);
            auto gradient = Tensor::zeros({3, 16, 16}, core::Device::GPU);
            const BackwardAdamParam disabled{absent, absent, absent, absent, absent, absent, absent};
            const BackwardAdam adam{{disabled, disabled, disabled, disabled, disabled, disabled}, absent, absent, absent, absent, absent, absent};
            vulkan::fast_backward(saved, {gradient, absent, absent, absent}, absent, absent, absent, absent, adam, DensificationType::None);
        }
        void record_vram(const FastSaved& saved, In image, In alpha, bool backward, size_t primitives) {
            if (!saved.backend)
                return;
            const auto& s = static_cast<const vulkan::FastState&>(*saved.backend);
            constexpr std::string_view scope = "rasterizer.fast";
            record_vram_tensor(scope, "forward.projected", s.projected);
            record_vram_tensor(scope, "forward.tile_ranges", s.ranges);
            record_vram_tensor(scope, "forward.transmittance", s.transmittance);
            record_vram_tensor(scope, "forward.last_contributor", s.last);
            record_vram_tensor(scope, "forward.indices_a", s.values_a);
            record_vram_tensor(scope, "forward.indices_b", s.values_b);
            record_vram_tensor(scope, "forward.visibility", s.visibility);
            record_vram_tensor(scope, "forward.offsets", s.offsets);
            record_vram_tensor(scope, "forward.original_to_work", s.original_to_work);
            record_vram_tensor(scope, "forward.work_to_original", s.work_to_original);
            record_vram_tensor(scope, "forward.counts", s.counts);
            record_vram_tensor(scope, "forward.keys_a", s.keys_a);
            record_vram_tensor(scope, "forward.keys_b", s.keys_b);
            record_vram_tensor(scope, "output.depth", s.depth);
            record_vram_tensor(scope, "output.normal", s.normal);
            record_vram_current(scope, "backward.screen_gradient", backward ? size_t(s.push.visible) * 16 * 4 : 0, true);
            record_vram_current(scope, "backward.parameter_gradients", backward ? primitives * (14 + ((s.push.rest * 3 + 3) / 4) * 4) * 4 : 0, true);
            record_vram_tensor(scope, "output.image", image);
            record_vram_tensor(scope, "output.alpha", alpha);
        }
        void release_caches(FastSaved& saved) noexcept {
            if (!saved.backend)
                return;
            auto& s = static_cast<vulkan::FastState&>(*saved.backend);
            if (!s.live) {
                auto* timer = s.timer;
                s = vulkan::FastState{};
                s.timer = timer;
            }
        }
        const FastRasterOps ops{create, vulkan::fast_forward, vulkan::fast_backward, release, warmup, record_vram, release_caches};
    } // namespace
    void vulkan_fast_set_timer(gpu_ops::FastSaved& saved, core::GpuElapsed* timer) {
        LFS_ASSERT_MSG(saved.backend != nullptr, "Fast timing requires a created state");
        static_cast<vulkan::FastState&>(*saved.backend).timer = timer;
    }
    const gpu_ops::FastRasterOps& vulkan_fast_ops() { return ops; }
} // namespace lfs::training

/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "adam_api.h"
#include "core/cuda/sh_layout.cuh"
#include "core/sh_value_quant.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "lfs/training/joint_adam_codec.hpp"
#include "lfs/training/sh_value_codec.hpp"
#include "lfs/training/sh_value_storage.hpp"
#include "optimizer/adam_optimizer.hpp"

#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

using namespace lfs::core;
using namespace lfs::training;

namespace {

    struct CodecsOnGuard {
        CodecsOnGuard() {
            sh_value::set_sh_value_quant_enabled_for_testing(true);
        }
        ~CodecsOnGuard() {
            sh_value::set_sh_value_quant_enabled_for_testing(std::nullopt);
        }
    };

    SplatData make_mixed_splat(const size_t n, const int sh_degree = 3) {
        const size_t rest = sh_degree > 0
                                ? static_cast<size_t>(sh_degree * (sh_degree + 2))
                                : size_t{0};
        std::vector<float> means(n * 3);
        std::vector<float> sh0(n * 3);
        std::vector<float> scaling(n * 3);
        std::vector<float> rotation(n * 4, 0.0f);
        std::vector<float> opacity(n, 0.0f);
        for (size_t i = 0; i < n; ++i) {
            means[i * 3 + 0] = static_cast<float>((i * 17) % 97) * 0.03f;
            means[i * 3 + 1] = static_cast<float>((i * 13) % 89) * 0.04f;
            means[i * 3 + 2] = static_cast<float>((i * 11) % 83) * 0.05f;
            sh0[i * 3 + 0] = 0.01f * static_cast<float>(i + 1);
            sh0[i * 3 + 1] = 0.02f * static_cast<float>(i + 1);
            sh0[i * 3 + 2] = 0.03f * static_cast<float>(i + 1);
            scaling[i * 3 + 0] = -1.5f - 0.001f * static_cast<float>(i);
            scaling[i * 3 + 1] = -1.6f - 0.001f * static_cast<float>(i);
            scaling[i * 3 + 2] = -1.7f - 0.001f * static_cast<float>(i);
            rotation[i * 4] = 1.0f;
            rotation[i * 4 + 1] = 0.001f * static_cast<float>(i);
            opacity[i] = -1.0f + 0.0005f * static_cast<float>(i);
        }
        Tensor shN = rest == 0
                         ? Tensor::zeros({size_t{0}}, Device::CUDA)
                         : Tensor::zeros({n, rest, size_t{3}}, Device::CUDA);
        if (rest > 0) {
            auto cpu = shN.cpu();
            auto* p = cpu.ptr<float>();
            for (size_t i = 0; i < n * rest * 3; ++i) {
                p[i] = 0.02f * static_cast<float>((i % 11) + 1) *
                       (1.0f + 0.001f * static_cast<float>(i / 3));
            }
            shN = cpu.cuda();
        }
        return SplatData(
            sh_degree,
            Tensor::from_vector(means, {n, size_t{3}}, Device::CUDA),
            Tensor::from_vector(sh0, {n, size_t{1}, size_t{3}}, Device::CUDA),
            std::move(shN),
            Tensor::from_vector(scaling, {n, size_t{3}}, Device::CUDA),
            Tensor::from_vector(rotation, {n, size_t{4}}, Device::CUDA),
            Tensor::from_vector(opacity, {n, size_t{1}}, Device::CUDA),
            1.0f);
    }

    AdamConfig make_cfg(const size_t cap) {
        AdamConfig cfg;
        cfg.lr = 1e-3f;
        cfg.initial_capacity = cap;
        cfg.growth_factor = 1.5f;
        return cfg;
    }

    void copy_tensor_bytes(Tensor& dst, const Tensor& src) {
        ASSERT_TRUE(dst.is_valid());
        ASSERT_TRUE(src.is_valid());
        ASSERT_EQ(dst.numel(), src.numel());
        ASSERT_EQ(dst.dtype(), src.dtype());
        const size_t n_bytes = src.bytes();
        ASSERT_GT(n_bytes, 0u);
        ASSERT_EQ(cudaMemcpy(dst.data_ptr(), src.data_ptr(), n_bytes, cudaMemcpyDeviceToDevice),
                  cudaSuccess);
    }

    std::vector<std::uint8_t> tensor_bytes(const Tensor& t) {
        auto cpu = t.cpu();
        const auto* p = static_cast<const std::uint8_t*>(cpu.data_ptr());
        return {p, p + cpu.bytes()};
    }

} // namespace

TEST(GutShNJointAdam, StandaloneStepMatchesFusedKernelQ16Sh3) {
    CodecsOnGuard guard;
    constexpr size_t n = 300;
    constexpr size_t cap = 512;
    constexpr int sh_degree = 3;
    constexpr int past_warmup = 1001;

    auto splat_fused = make_mixed_splat(n, sh_degree);
    ASSERT_TRUE(sh_value::apply_shN_value_quant(splat_fused));
    ASSERT_TRUE(splat_fused.shN_value_quantized());
    auto splat_step = splat_fused.clone();

    AdamOptimizer opt_fused(splat_fused, make_cfg(cap));
    AdamOptimizer opt_step(splat_step, make_cfg(cap));
    opt_fused.allocate_gradients(cap);
    opt_step.allocate_gradients(cap);

    auto* st_fused = opt_fused.get_state_mutable(ParamType::ShN);
    auto* st_step = opt_step.get_state_mutable(ParamType::ShN);
    ASSERT_NE(st_fused, nullptr);
    ASSERT_NE(st_step, nullptr);
    ASSERT_TRUE(st_fused->is_joint());
    ASSERT_TRUE(st_step->is_joint());
    copy_tensor_bytes(st_step->exp_avg, st_fused->exp_avg);
    copy_tensor_bytes(st_step->joint_bounds, st_fused->joint_bounds);
    copy_tensor_bytes(splat_step.shN(), splat_fused.shN());
    copy_tensor_bytes(splat_step.shN_value_bounds(), splat_fused.shN_value_bounds());

    const auto layout_rest = static_cast<uint32_t>(splat_fused.max_sh_coeffs_rest());
    const size_t float_layout = sh_swizzled_float_count(n, layout_rest);
    ASSERT_EQ(st_fused->size, float_layout);

    auto& grad_fused = opt_fused.get_grad(ParamType::ShN);
    auto& grad_step = opt_step.get_grad(ParamType::ShN);
    ASSERT_EQ(static_cast<size_t>(grad_fused.numel()), float_layout);
    ASSERT_EQ(static_cast<size_t>(grad_step.numel()), float_layout);

    std::vector<float> grads_h(float_layout);
    for (size_t i = 0; i < float_layout; ++i) {
        grads_h[i] = 0.01f * static_cast<float>((i % 17) + 1) *
                     (1.0f + 0.002f * static_cast<float>(i / 4));
    }
    ASSERT_EQ(cudaMemcpy(grad_fused.ptr<float>(), grads_h.data(),
                         float_layout * sizeof(float), cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(grad_step.ptr<float>(), grads_h.data(),
                         float_layout * sizeof(float), cudaMemcpyHostToDevice),
              cudaSuccess);

    auto fused = opt_fused.prepare_fastgs_fused_adam(past_warmup);
    ASSERT_TRUE(fused.enabled);
    ASSERT_TRUE(fused.shN.enabled);
    EXPECT_EQ(fused.shN.sh_value_bits, 16);
    EXPECT_EQ(fused.shN.n_primitives, static_cast<int>(n));
    const int sh_layout_slots = static_cast<int>(sh_float4_slots_for_rest(layout_rest));
    const int active_sh_bases =
        static_cast<int>(splat_fused.active_sh_coeffs_rest() + 1);
    ASSERT_EQ(active_sh_bases, 16);

    fast_lfs::optimizer::adam_step_shN_joint_from_grad(
        fused.shN.param,
        fused.shN.joint_packed,
        fused.shN.joint_bounds,
        fused.shN.sh_value_bounds,
        grad_fused.ptr<float>(),
        fused.shN.frozen_mask,
        fused.shN.frozen_mask_size,
        fused.shN.frozen_lr_scale,
        fused.shN.crop_damping_mask,
        fused.shN.crop_damping_mask_size,
        fused.shN.cropbox_lr_scale,
        fused.shN.n_primitives,
        sh_layout_slots,
        active_sh_bases,
        fused.shN.sh_value_bits,
        fused.shN.sh_value_n_cells,
        fused.shN.step_size,
        fused.beta1,
        fused.beta2,
        fused.eps,
        fused.shN.bias_correction2_sqrt_rcp,
        nullptr);
    opt_fused.commit_fastgs_fused_adam(past_warmup);

    ASSERT_NO_THROW(opt_step.step(past_warmup));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    const auto codes_fused = tensor_bytes(splat_fused.shN());
    const auto codes_step = tensor_bytes(splat_step.shN());
    ASSERT_EQ(codes_fused.size(), codes_step.size());
    EXPECT_EQ(codes_fused, codes_step) << "q16 shN codes must match fused kernel";

    const auto vbounds_fused = tensor_bytes(splat_fused.shN_value_bounds());
    const auto vbounds_step = tensor_bytes(splat_step.shN_value_bounds());
    ASSERT_EQ(vbounds_fused.size(), vbounds_step.size());
    EXPECT_EQ(vbounds_fused, vbounds_step) << "q16 value bounds must match";

    st_fused = opt_fused.get_state_mutable(ParamType::ShN);
    st_step = opt_step.get_state_mutable(ParamType::ShN);
    ASSERT_NE(st_fused, nullptr);
    ASSERT_NE(st_step, nullptr);
    const auto packed_fused = tensor_bytes(st_fused->exp_avg);
    const auto packed_step = tensor_bytes(st_step->exp_avg);
    ASSERT_EQ(packed_fused.size(), packed_step.size());
    EXPECT_EQ(packed_fused, packed_step) << "packed shN moments must match bitwise";

    const auto jbounds_fused = tensor_bytes(st_fused->joint_bounds);
    const auto jbounds_step = tensor_bytes(st_step->joint_bounds);
    ASSERT_EQ(jbounds_fused.size(), jbounds_step.size());
    EXPECT_EQ(jbounds_fused, jbounds_step) << "joint moment bounds must match";
}

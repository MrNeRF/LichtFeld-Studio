/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// mean-abs triad for SH rest formats.
// fp32 float4-swizzle reference vs IEEE f16 cast vs pad-dropped q16 decode.

#include "core/cuda/sh_layout.cuh"
#include "core/sh_value_quant.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "lfs/training/sh_value_codec.hpp"
#include "lfs/training/sh_value_storage.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

using lfs::core::DataType;
using lfs::core::Device;
using lfs::core::SplatData;
using lfs::core::Tensor;
using lfs::core::TensorShape;

namespace {

    constexpr std::size_t kN = 4096;

    [[nodiscard]] SplatData make_random_sh3(std::size_t n, std::uint32_t seed) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
        const auto rest = lfs::core::sh_rest_coefficients_for_degree(3);
        const size_t floats = lfs::core::sh_swizzled_float_count(n, rest);

        Tensor means = Tensor::zeros({n, 3}, Device::CUDA);
        Tensor sh0 = Tensor::zeros({n, 1, 3}, Device::CUDA);
        Tensor shN = Tensor::zeros_direct(TensorShape({floats}), floats, Device::CUDA);
        Tensor scaling = Tensor::zeros({n, 3}, Device::CUDA);
        Tensor rotation = Tensor::zeros({n, 4}, Device::CUDA);
        Tensor opacity = Tensor::zeros({n, 1}, Device::CUDA);

        std::vector<float> host(floats);
        for (size_t i = 0; i < floats; ++i) {
            host[i] = dist(rng);
        }
        EXPECT_EQ(cudaMemcpy(shN.ptr<float>(), host.data(), floats * sizeof(float),
                             cudaMemcpyHostToDevice),
                  cudaSuccess);

        // Identity-ish rotation so the model is valid.
        std::vector<float> rot(n * 4, 0.0f);
        for (size_t i = 0; i < n; ++i)
            rot[i * 4] = 1.0f;
        EXPECT_EQ(cudaMemcpy(rotation.ptr<float>(), rot.data(), rot.size() * sizeof(float),
                             cudaMemcpyHostToDevice),
                  cudaSuccess);

        return SplatData(3, std::move(means), std::move(sh0), std::move(shN),
                         std::move(scaling), std::move(rotation), std::move(opacity),
                         1.0f, SplatData::ShNLayout::Swizzled);
    }

    struct DiffStats {
        double mean_abs = 0;
        double max_abs = 0;
        std::size_t n = 0;
        std::size_t nan_count = 0;
    };

    [[nodiscard]] DiffStats diff_stats(const Tensor& a_cpu, const Tensor& b_cpu) {
        DiffStats s;
        EXPECT_EQ(a_cpu.numel(), b_cpu.numel());
        if (a_cpu.numel() != b_cpu.numel())
            return s;
        const float* pa = a_cpu.ptr<float>();
        const float* pb = b_cpu.ptr<float>();
        s.n = static_cast<std::size_t>(a_cpu.numel());
        double sum = 0;
        for (std::size_t i = 0; i < s.n; ++i) {
            if (!std::isfinite(pa[i]) || !std::isfinite(pb[i])) {
                ++s.nan_count;
                continue;
            }
            const double d = std::abs(static_cast<double>(pa[i]) - static_cast<double>(pb[i]));
            sum += d;
            s.max_abs = std::max(s.max_abs, d);
        }
        const std::size_t finite = s.n - s.nan_count;
        s.mean_abs = finite > 0 ? sum / static_cast<double>(finite) : 0;
        return s;
    }

} // namespace

TEST(ViewerShZerocopyTriad, Fp32VsF16VsQ16MeanAbs) {
    auto splat_fp32 = make_random_sh3(kN, 0xF16A016u);
    const auto ref = splat_fp32.shN_canonical().cpu().contiguous();

    // --- IEEE f16 float4-swizzle (standalone tier-1 path) ---
    auto splat_f16 = make_random_sh3(kN, 0xF16A016u);
    splat_f16.shN() = splat_f16.shN().to(DataType::Float16);
    ASSERT_TRUE(splat_f16.shN_ieee_f16());
    const auto f16_decoded = splat_f16.shN_canonical().cpu().contiguous();
    const DiffStats f16_stats = diff_stats(ref, f16_decoded);

    // --- pad-dropped q16 (training exportable zero-copy path) ---
    lfs::training::sh_value::set_sh_value_quant_enabled_for_testing(true);
    auto splat_q16 = make_random_sh3(kN, 0xF16A016u);
    ASSERT_TRUE(lfs::training::sh_value::apply_shN_value_quant(splat_q16));
    ASSERT_TRUE(splat_q16.shN_value_quantized());
    const auto q16_decoded = splat_q16.shN_canonical().cpu().contiguous();
    const DiffStats q16_stats = diff_stats(ref, q16_decoded);
    lfs::training::sh_value::set_sh_value_quant_enabled_for_testing(std::nullopt);

    // Tight thresholds: f16 is near-exact for this coeff range; q16 is lossy but
    // well under SH contribution to RGB (PSNR>55 on roundtrip tests).
    EXPECT_EQ(f16_stats.nan_count, 0u);
    EXPECT_EQ(q16_stats.nan_count, 0u);
    EXPECT_LT(f16_stats.mean_abs, 5e-4) << "f16 mean_abs=" << f16_stats.mean_abs;
    EXPECT_LT(f16_stats.max_abs, 5e-3) << "f16 max_abs=" << f16_stats.max_abs;
    EXPECT_LT(q16_stats.mean_abs, 5e-3) << "q16 mean_abs=" << q16_stats.mean_abs;
    EXPECT_LT(q16_stats.max_abs, 5e-2) << "q16 max_abs=" << q16_stats.max_abs;
}

/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "checkpoint_fixture.hpp"
#include "core/cuda/sh_layout.cuh"
#include "core/parameters.hpp"
#include "core/sh_value_quant.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "lfs/training/joint_adam_codec.hpp"
#include "lfs/training/sh_value_codec.hpp"
#include "lfs/training/sh_value_storage.hpp"
#include "optimizer/adam_optimizer.hpp"
#include "training/checkpoint.hpp"
#include "training/strategies/improved_gs_plus.hpp"
#include "training/strategies/mcmc.hpp"
#include "training/strategies/mrnf.hpp"

#include "adam_api.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <gtest/gtest.h>
#include <sstream>
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

    SplatData make_sh_splat(const size_t n, const int sh_degree = 3) {
        const size_t rest = sh_degree > 0
                                ? static_cast<size_t>(sh_degree * (sh_degree + 2))
                                : size_t{0};
        std::vector<float> means(n * 3, 0.0f);
        std::vector<float> rotations(n * 4, 0.0f);
        for (size_t i = 0; i < n; ++i) {
            means[i * 3] = static_cast<float>(i) * 0.01f;
            rotations[i * 4] = 1.0f;
        }
        auto shN = rest == 0
                       ? Tensor::zeros({size_t{0}}, Device::CUDA)
                       : Tensor::zeros({n, rest, size_t{3}}, Device::CUDA);
        if (rest > 0) {
            auto cpu = shN.cpu();
            auto* p = cpu.ptr<float>();
            for (size_t i = 0; i < n * rest * 3; ++i)
                p[i] = 0.02f * static_cast<float>((i % 11) + 1);
            shN = cpu.cuda();
        }
        return SplatData(
            sh_degree,
            Tensor::from_vector(means, {n, size_t{3}}, Device::CUDA),
            Tensor::zeros({n, size_t{1}, size_t{3}}, Device::CUDA),
            std::move(shN),
            Tensor::full({n, size_t{3}}, -2.0f, Device::CUDA),
            Tensor::from_vector(rotations, {n, size_t{4}}, Device::CUDA),
            Tensor::full({n, size_t{1}}, 0.5f, Device::CUDA),
            1.0f);
    }

    AdamConfig make_cfg(const size_t cap) {
        AdamConfig cfg;
        cfg.lr = 1e-3f;
        cfg.initial_capacity = cap;
        cfg.growth_factor = 1.5f;
        return cfg;
    }

} // namespace

// ---------------------------------------------------------------------------
// joint shN moments sized from float layout, not q16 cell count
// ---------------------------------------------------------------------------
TEST(DualRepOptimizer, JointShNMomentsUseFloatLayoutNotQ16Cells) {
    CodecsOnGuard guard;
    constexpr size_t n = 32;
    constexpr size_t cap = 64;
    auto splat = make_sh_splat(n, 3);
    ASSERT_TRUE(sh_value::apply_shN_value_quant(splat));
    ASSERT_TRUE(splat.shN_value_quantized());

    const auto layout_rest = static_cast<uint32_t>(splat.max_sh_coeffs_rest());
    const size_t float_layout = sh_swizzled_float_count(n, layout_rest);
    const size_t u16_cells = lfs::core::sh_value_quant::sh_value_u16_count(n, layout_rest);
    ASSERT_GT(float_layout, 0u);
    ASSERT_NE(float_layout, u16_cells)
        << "test requires q16 cell count ≠ float layout (SH3: 45 vs 48 /prim)";

    AdamOptimizer opt(splat, make_cfg(cap));
    opt.allocate_gradients(cap);

    const auto* st = opt.get_state(ParamType::ShN);
    ASSERT_NE(st, nullptr);
    ASSERT_TRUE(st->is_joint());
    EXPECT_EQ(st->size, float_layout)
        << "state.size must be float4-swizzle cells, not q16 cells";
    // Joint packed: float_cells * bpc (8-bit → 2 B/cell)
    const int bpc = joint_adam::bytes_per_cell(st->joint_bits);
    ASSERT_EQ(bpc, 2);
    ASSERT_TRUE(st->exp_avg.is_valid());
    EXPECT_EQ(st->exp_avg.numel(), float_layout * static_cast<size_t>(bpc))
        << "packed moment buffer must cover float layout, not u16 cells";
    EXPECT_GE(st->exp_avg.capacity(),
              sh_swizzled_float_count(cap, layout_rest) * static_cast<size_t>(bpc));
}

// ---------------------------------------------------------------------------
// checkpoint roundtrip AFTER real fused prepare (heals state.size)
// ---------------------------------------------------------------------------
TEST(DualRepOptimizer, CheckpointRoundtripAfterFusedPrepareWithQuantOn) {
    CodecsOnGuard guard;
    constexpr size_t n = 24;
    constexpr size_t max_cap = 48;
    constexpr int sh_degree = 3;

    const auto temp_dir =
        std::filesystem::temp_directory_path() / "lfs_dual_rep_ckpt_after_step";
    std::error_code ec;
    std::filesystem::remove_all(temp_dir, ec);
    std::filesystem::create_directories(temp_dir / "checkpoints");

    param::TrainingParameters params;
    params.dataset.output_path = temp_dir;
    params.optimization.strategy = "mcmc";
    params.optimization.max_cap = static_cast<int>(max_cap);

    auto model = std::make_unique<SplatData>(make_sh_splat(n, sh_degree));
    ASSERT_TRUE(sh_value::apply_shN_value_quant(*model));
    MCMC strategy(*model);
    strategy.initialize(params.optimization);

    auto& opt = strategy.get_optimizer();
    auto* shN_st = opt.get_state_mutable(ParamType::ShN);
    ASSERT_NE(shN_st, nullptr);
    ASSERT_TRUE(shN_st->is_joint());

    // Fused preparation past SH warmup must advance state.size to the float layout.
    constexpr int past_warmup = 1001;
    auto fused = opt.prepare_fastgs_fused_adam(past_warmup);
    EXPECT_TRUE(fused.enabled);
    if (fused.shN.enabled) {
        EXPECT_EQ(fused.shN.sh_value_bits, 16)
            << "joint+q16 must keep value quant on the fused path";
    }
    opt.commit_fastgs_fused_adam(past_warmup);

    const auto layout_rest = static_cast<uint32_t>(model->max_sh_coeffs_rest());
    const size_t float_layout = sh_swizzled_float_count(n, layout_rest);
    shN_st = opt.get_state_mutable(ParamType::ShN);
    ASSERT_NE(shN_st, nullptr);
    EXPECT_EQ(shN_st->size, float_layout);

    ASSERT_TRUE(lfs::test::write_checkpoint_fixture(
                    temp_dir, past_warmup, strategy, params,
                    nullptr, nullptr, nullptr, nullptr)
                    .has_value());

    auto target_model = std::make_unique<SplatData>(make_sh_splat(1, sh_degree));
    MCMC target(*target_model);
    target.initialize(params.optimization);
    auto load_params = params;
    const auto loaded = load_checkpoint(
        lfs::test::checkpoint_fixture_path(temp_dir), target,
        load_params, nullptr, nullptr, nullptr, nullptr);
    ASSERT_TRUE(loaded.has_value()) << loaded.error()
                                    << " — save→load after fused prepare must not throw "
                                       "'state size does not match model'";
    EXPECT_EQ(*loaded, past_warmup);

    const auto* rst = target.get_optimizer().get_state(ParamType::ShN);
    ASSERT_NE(rst, nullptr);
    EXPECT_TRUE(rst->is_joint());
    EXPECT_EQ(rst->size, float_layout);

    std::filesystem::remove_all(temp_dir, ec);
}

TEST(DualRepOptimizer, JointEncodeZeroUnderBoundsExcludingZero) {
    CodecsOnGuard guard;
    constexpr int n = 8;
    constexpr int n_attr = 3;
    constexpr int bits = 16;
    const int bpc = joint_adam::bytes_per_cell(bits);
    ASSERT_EQ(bpc, 4);

    // One block; bounds exclude 0 on both u and log_s.
    std::vector<float> bounds_h = {0.5f, 1.5f, 0.25f, 1.0f}; // umin,umax,smin,smax
    auto bounds = Tensor::from_vector(bounds_h, {size_t{1}, size_t{4}}, Device::CUDA);
    auto packed = Tensor::zeros({static_cast<size_t>(n), static_cast<size_t>(n_attr * bpc)},
                                Device::CUDA, DataType::UInt8);
    // Seed non-zero codes so a no-op would leave garbage.
    {
        auto cpu = packed.cpu();
        auto* b = cpu.ptr<uint8_t>();
        for (size_t i = 0; i < cpu.numel(); ++i)
            b[i] = static_cast<uint8_t>(200);
        packed = cpu.cuda();
    }

    std::vector<int64_t> idx_h = {0, 3, 7};
    auto idx = Tensor::empty({idx_h.size()}, Device::CPU, DataType::Int64);
    std::memcpy(idx.ptr<int64_t>(), idx_h.data(), idx_h.size() * sizeof(int64_t));
    idx = idx.to(Device::CUDA);

    fast_lfs::optimizer::joint_encode_zero_rows_at_indices(
        packed.ptr<uint8_t>(),
        bounds.ptr<float>(),
        idx.ptr<int64_t>(),
        static_cast<int>(idx_h.size()),
        n_attr,
        bits,
        n,
        nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    // Bounds must have been widened to include 0.
    auto bcpu = bounds.cpu();
    const float* bb = bcpu.ptr<float>();
    EXPECT_LE(bb[0], 0.0f) << "umin must include 0";
    EXPECT_GE(bb[1], 0.0f) << "umax must include 0";
    EXPECT_LE(bb[2], 0.0f) << "smin must include 0";
    EXPECT_GE(bb[3], 0.0f) << "smax must include 0";

    // Decode zeroed rows → (m,v) ≈ (0,0).
    auto pcpu = packed.cpu();
    const auto* bytes = pcpu.ptr<uint8_t>();
    for (int64_t prim : idx_h) {
        for (int a = 0; a < n_attr; ++a) {
            float m = 0.0f, v = 0.0f;
            joint_adam::Codec16::decode_g1g2(
                bytes, static_cast<size_t>(prim) * n_attr + static_cast<size_t>(a),
                bb[0], bb[1], bb[2], bb[3], m, v);
            EXPECT_NEAR(m, 0.0f, 1e-6f) << "prim=" << prim << " attr=" << a;
            EXPECT_NEAR(v, 0.0f, 1e-12f) << "prim=" << prim << " attr=" << a;
        }
    }
}

// ---------------------------------------------------------------------------
// joint grow then decode new rows ≈ 0 under live non-zero bounds
// ---------------------------------------------------------------------------
TEST(DualRepOptimizer, JointGrowZeroEncodesNewRows) {
    CodecsOnGuard guard;
    constexpr size_t n0 = 16;
    constexpr size_t n_grow = 4;
    auto splat = make_sh_splat(n0, 0); // degree 0 — exercise contiguous means
    // Rebuild without sh for simpler contiguous joint test
    splat = SplatData(
        0,
        Tensor::randn({n0, 3}, Device::CUDA),
        Tensor::randn({n0, 1, 3}, Device::CUDA),
        Tensor::zeros({size_t{0}}, Device::CUDA),
        Tensor::randn({n0, 3}, Device::CUDA),
        Tensor::randn({n0, 4}, Device::CUDA),
        Tensor::randn({n0, 1}, Device::CUDA),
        1.0f);

    AdamOptimizer opt(splat, make_cfg(64));
    opt.allocate_gradients(64);
    auto* st = opt.get_state_mutable(ParamType::Means);
    ASSERT_NE(st, nullptr);
    ASSERT_TRUE(st->is_joint());

    // Use nonzero block bounds so raw zero codes decode to nonzero values.
    {
        auto b = st->joint_bounds.cpu();
        auto* p = b.ptr<float>();
        for (size_t i = 0; i < b.shape()[0]; ++i) {
            p[i * 4 + 0] = 0.5f;
            p[i * 4 + 1] = 1.5f;
            p[i * 4 + 2] = 0.25f;
            p[i * 4 + 3] = 1.0f;
        }
        st->joint_bounds = b.cuda();
    }

    // Grow params + state
    for (auto t : {ParamType::Means, ParamType::Sh0, ParamType::Scaling,
                   ParamType::Rotation, ParamType::Opacity}) {
        auto& p = (t == ParamType::Means)      ? splat.means()
                  : (t == ParamType::Sh0)      ? splat.sh0()
                  : (t == ParamType::Scaling)  ? splat.scaling_raw()
                  : (t == ParamType::Rotation) ? splat.rotation_raw()
                                               : splat.opacity_raw();
        p.reserve(n0 + n_grow + 8);
        p.append_zeros(n_grow);
        opt.extend_state_for_new_params(t, n_grow);
    }
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    st = opt.get_state_mutable(ParamType::Means);
    ASSERT_NE(st, nullptr);
    EXPECT_EQ(st->size, n0 + n_grow);

    auto bcpu = st->joint_bounds.cpu();
    auto pcpu = st->exp_avg.cpu();
    const float* bb = bcpu.ptr<float>();
    const auto* bytes = pcpu.ptr<uint8_t>();
    const int bpc = joint_adam::bytes_per_cell(st->joint_bits);
    const int n_attr = static_cast<int>(pcpu.shape()[1] / static_cast<size_t>(bpc));

    for (size_t prim = n0; prim < n0 + n_grow; ++prim) {
        const size_t bidx = prim / 256;
        for (int a = 0; a < n_attr; ++a) {
            float m = 0.0f, v = 0.0f;
            joint_adam::Codec16::decode_g1g2(
                bytes, prim * static_cast<size_t>(n_attr) + static_cast<size_t>(a),
                bb[bidx * 4 + 0], bb[bidx * 4 + 1], bb[bidx * 4 + 2], bb[bidx * 4 + 3],
                m, v);
            EXPECT_NEAR(m, 0.0f, 1e-5f) << "new prim " << prim;
            EXPECT_NEAR(v, 0.0f, 1e-12f) << "new prim " << prim;
        }
    }
}

// ---------------------------------------------------------------------------
// joint add_new_params_gather(ShN) must grow moment tensor
// ---------------------------------------------------------------------------
TEST(DualRepOptimizer, JointAddNewParamsGatherShNGrowsMoments) {
    CodecsOnGuard guard;
    // Cross a reorder block boundary (R=32) so float_layout actually grows.
    constexpr size_t n0 = 30;
    constexpr size_t n_new = 8;
    auto splat = make_sh_splat(n0, 3);
    ASSERT_TRUE(sh_value::apply_shN_value_quant(splat));

    AdamOptimizer opt(splat, make_cfg(64));
    opt.allocate_gradients(64);
    auto* st = opt.get_state_mutable(ParamType::ShN);
    ASSERT_NE(st, nullptr);
    ASSERT_TRUE(st->is_joint());
    const size_t packed_before = st->exp_avg.numel();
    const size_t size_before = st->size;

    // Densify-style: expand model first, then gather-duplicate parents into new slots.
    // Strategies dequant before shN mutation.
    ASSERT_TRUE(sh_value::ensure_shN_fp32_for_mutation(splat));

    // Grow contiguous params to new_N (as strategies do before shN gather).
    auto indices = Tensor::arange(0.0f, static_cast<float>(n_new), 1.0f)
                       .to(DataType::Int64)
                       .to(Device::CUDA);
    // Manually bump splat size bookkeeping via means append (SplatData size tracks means).
    splat.means().reserve(n0 + n_new + 8);
    splat.means().append_zeros(n_new);
    splat.sh0().reserve(n0 + n_new + 8);
    splat.sh0().append_zeros(n_new);
    splat.scaling_raw().reserve(n0 + n_new + 8);
    splat.scaling_raw().append_zeros(n_new);
    splat.rotation_raw().reserve(n0 + n_new + 8);
    splat.rotation_raw().append_zeros(n_new);
    splat.opacity_raw().reserve(n0 + n_new + 8);
    splat.opacity_raw().append_zeros(n_new);

    // SplatData::size() follows means rows.
    ASSERT_EQ(static_cast<size_t>(splat.size()), n0 + n_new);

    opt.add_new_params_gather(ParamType::ShN, indices);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    st = opt.get_state_mutable(ParamType::ShN);
    ASSERT_NE(st, nullptr);
    const auto layout_rest = static_cast<uint32_t>(splat.max_sh_coeffs_rest());
    const size_t expected_floats = sh_swizzled_float_count(n0 + n_new, layout_rest);
    EXPECT_EQ(st->size, expected_floats)
        << "joint gather must advance shN moment size (was " << size_before << ")";
    EXPECT_GT(st->exp_avg.numel(), packed_before)
        << "joint gather must grow packed moment buffer";
}

// ---------------------------------------------------------------------------
// n_primitives set; N%256≠0 prepare does not set q16 OOB conditions
// ---------------------------------------------------------------------------
TEST(DualRepOptimizer, FusedPrepareSetsNPrimitivesForOverhangGuard) {
    CodecsOnGuard guard;
    constexpr size_t n = 300; // not divisible by 256 → grid overhang
    auto splat = make_sh_splat(n, 1);
    ASSERT_TRUE(sh_value::apply_shN_value_quant(splat));

    AdamOptimizer opt(splat, make_cfg(512));
    opt.allocate_gradients(512);
    auto fused = opt.prepare_fastgs_fused_adam(1001);
    EXPECT_EQ(fused.shN.n_primitives, static_cast<int>(n))
        << "kernel overhang guard needs live primitive count";
    EXPECT_EQ(fused.means.n_primitives, static_cast<int>(n));
}

TEST(DualRepOptimizer, JointBoundsLoadAcceptsOversizedTable) {
    CodecsOnGuard guard;
    constexpr size_t n = 10;
    auto splat = make_sh_splat(n, 0);
    splat = SplatData(
        0,
        Tensor::randn({n, 3}, Device::CUDA),
        Tensor::randn({n, 1, 3}, Device::CUDA),
        Tensor::zeros({size_t{0}}, Device::CUDA),
        Tensor::randn({n, 3}, Device::CUDA),
        Tensor::randn({n, 4}, Device::CUDA),
        Tensor::randn({n, 1}, Device::CUDA),
        1.0f);

    AdamOptimizer opt(splat, make_cfg(32));
    opt.allocate_gradients(32);
    auto* st = opt.get_state_mutable(ParamType::Means);
    ASSERT_NE(st, nullptr);
    ASSERT_TRUE(st->is_joint());

    // Artificially grow bounds table past live N (compaction leftover).
    const size_t live_nb = joint_adam::n_bounds_for_prims(n);
    ensure_joint_bounds_capacity(st->joint_bounds, n + 200, n + 200, Device::CUDA, false);
    EXPECT_GE(st->joint_bounds.shape()[0], live_nb);

    std::stringstream ss;
    opt.serialize(ss);
    AdamOptimizer loaded(splat, make_cfg(32));
    loaded.allocate_gradients(32);
    ASSERT_NO_THROW(loaded.deserialize(ss))
        << "oversized joint bounds table must not fail strict-equality load";
}

// ---------------------------------------------------------------------------
// Strategy suites with BOTH codecs ON (the gap that let this cluster survive)
// ---------------------------------------------------------------------------
TEST(DualRepOptimizer, MCMC_InitializeWithBothCodecsOn) {
    CodecsOnGuard guard;
    auto splat = make_sh_splat(20, 3);
    MCMC strategy(splat);
    param::OptimizationParameters opt_params;
    opt_params.iterations = 100;
    opt_params.max_cap = 40;
    ASSERT_NO_THROW(strategy.initialize(opt_params));
    EXPECT_TRUE(strategy.get_model().shN_value_quantized() ||
                strategy.get_model().shN_raw().dtype() == DataType::Float16);
    const auto* st = strategy.get_optimizer().get_state(ParamType::ShN);
    ASSERT_NE(st, nullptr);
    EXPECT_TRUE(st->is_joint());
    const auto layout_rest =
        static_cast<uint32_t>(strategy.get_model().max_sh_coeffs_rest());
    EXPECT_EQ(st->size,
              sh_swizzled_float_count(static_cast<size_t>(strategy.get_model().size()),
                                      layout_rest));
}

TEST(DualRepOptimizer, MRNF_InitializeWithBothCodecsOn) {
    CodecsOnGuard guard;
    auto splat = make_sh_splat(12, 3);
    MRNF strategy(splat);
    param::OptimizationParameters opt_params;
    opt_params.iterations = 100;
    opt_params.max_cap = 32;
    opt_params.sh_degree_interval = 10000;
    ASSERT_NO_THROW(strategy.initialize(opt_params));
    const auto* st = strategy.get_optimizer().get_state(ParamType::Means);
    ASSERT_NE(st, nullptr);
    EXPECT_TRUE(st->is_joint());
}

TEST(DualRepOptimizer, MCMC_QuantOn_MidRunSaveLoadResume) {
    CodecsOnGuard guard;
    constexpr size_t n0 = 48;
    constexpr size_t max_cap = 128;
    constexpr int sh_degree = 3;
    constexpr int mid_iter = 1200;
    constexpr int resume_to = 2000;

    const auto temp_dir =
        std::filesystem::temp_directory_path() / "lfs_dual_rep_mcmc_midrun_resume";
    std::error_code ec;
    std::filesystem::remove_all(temp_dir, ec);
    std::filesystem::create_directories(temp_dir / "checkpoints");

    param::TrainingParameters params;
    params.dataset.output_path = temp_dir;
    params.optimization.strategy = "mcmc";
    params.optimization.max_cap = static_cast<int>(max_cap);
    params.optimization.iterations = resume_to;
    params.optimization.sh_degree = sh_degree;

    auto model = std::make_unique<SplatData>(make_sh_splat(n0, sh_degree));
    ASSERT_TRUE(sh_value::apply_shN_value_quant(*model));
    ASSERT_TRUE(model->shN_value_quantized());

    MCMC strategy(*model);
    strategy.initialize(params.optimization);
    auto& opt = strategy.get_optimizer();

    // shN step_count advances only after SH warmup.
    for (int it = 1; it <= mid_iter; ++it) {
        auto fused = opt.prepare_fastgs_fused_adam(it);
        EXPECT_TRUE(fused.enabled);
        opt.commit_fastgs_fused_adam(it);
    }

    auto* shN_st = opt.get_state_mutable(ParamType::ShN);
    ASSERT_NE(shN_st, nullptr);
    ASSERT_TRUE(shN_st->is_joint());
    const size_t n_live = static_cast<size_t>(strategy.get_model().size());
    const auto layout_rest =
        static_cast<uint32_t>(strategy.get_model().max_sh_coeffs_rest());
    EXPECT_EQ(shN_st->size, sh_swizzled_float_count(n_live, layout_rest));
    EXPECT_EQ(shN_st->step_count, mid_iter - 1000)
        << "shN step_count must not inflate during warmup";

    ASSERT_TRUE(lfs::test::write_checkpoint_fixture(
                    temp_dir, mid_iter, strategy, params,
                    nullptr, nullptr, nullptr, nullptr)
                    .has_value());

    auto target_model = std::make_unique<SplatData>(make_sh_splat(1, sh_degree));
    MCMC target(*target_model);
    target.initialize(params.optimization);
    auto load_params = params;
    const auto loaded = load_checkpoint(
        lfs::test::checkpoint_fixture_path(temp_dir), target,
        load_params, nullptr, nullptr, nullptr, nullptr);
    ASSERT_TRUE(loaded.has_value()) << loaded.error()
                                    << " — quantized mid-run resume must load";
    EXPECT_EQ(*loaded, mid_iter);
    EXPECT_EQ(static_cast<size_t>(target.get_model().size()), n_live);
    EXPECT_TRUE(target.get_model().shN_value_quantized() ||
                target.get_model().shN_raw().dtype() == DataType::Float16);

    auto& ropt = target.get_optimizer();
    for (int it = mid_iter + 1; it <= resume_to; ++it) {
        auto fused = ropt.prepare_fastgs_fused_adam(it);
        EXPECT_TRUE(fused.enabled);
        if (fused.shN.enabled) {
            EXPECT_EQ(fused.shN.sh_value_bits, 16)
                << "joint+q16 must stay on after mid-run resume";
        }
        ropt.commit_fastgs_fused_adam(it);
    }
    const auto* rst = ropt.get_state(ParamType::ShN);
    ASSERT_NE(rst, nullptr);
    EXPECT_TRUE(rst->is_joint());
    EXPECT_EQ(rst->size, sh_swizzled_float_count(n_live, layout_rest));
    EXPECT_EQ(rst->step_count, (mid_iter - 1000) + (resume_to - mid_iter));

    std::filesystem::remove_all(temp_dir, ec);
}

TEST(DualRepOptimizer, ShortBoundsFailsLoud) {
    CodecsOnGuard guard;
    auto splat = make_sh_splat(300, 3); // >256 so n_bounds=2 → need 4 floats
    ASSERT_TRUE(sh_value::apply_shN_value_quant(splat));
    // Truncate bounds well below n_bounds_for_prims(N)*2.
    splat.shN_value_bounds() = Tensor::zeros({size_t{2}}, Device::CUDA);
    EXPECT_THROW((void)sh_value::ensure_shN_fp32_for_mutation(splat), std::runtime_error);
    // Invalid bounds also refuse.
    splat.shN_value_bounds() = Tensor{};
    EXPECT_THROW((void)sh_value::ensure_shN_fp32_for_mutation(splat), std::runtime_error);
}

TEST(DualRepOptimizer, IGSPlus_QuantOnDensifyAndPrune) {
    CodecsOnGuard guard;
    constexpr size_t n = 32;
    constexpr size_t max_cap = 96;
    auto model = std::make_unique<SplatData>(make_sh_splat(n, 3));
    ASSERT_TRUE(sh_value::apply_shN_value_quant(*model));
    ASSERT_TRUE(model->shN_value_quantized());

    ImprovedGSPlus strategy(*model);
    param::OptimizationParameters opt_params =
        param::OptimizationParameters::igs_plus_defaults();
    opt_params.iterations = 200;
    opt_params.max_cap = static_cast<int>(max_cap);
    opt_params.start_refine = 0;
    opt_params.stop_refine = 150;
    opt_params.refine_every = 10;
    ASSERT_NO_THROW(strategy.initialize(opt_params));

    // Seed non-trivial joint moments so prune must clear them.
    auto* means_st = strategy.get_optimizer().get_state_mutable(ParamType::Means);
    ASSERT_NE(means_st, nullptr);
    ASSERT_TRUE(means_st->is_joint());
    {
        auto packed_cpu = means_st->exp_avg.cpu();
        auto* bytes = packed_cpu.ptr<uint8_t>();
        for (size_t i = 0; i < packed_cpu.numel(); ++i)
            bytes[i] = static_cast<uint8_t>((i * 13 + 7) & 0xff);
        means_st->exp_avg = packed_cpu.cuda();
    }

    // Densify via LAS_densify with uniform scores (must not abort on q16).
    auto scores = Tensor::ones({n}, Device::CUDA);
    ASSERT_NO_THROW(strategy.densify_for_test(scores, 8));
    EXPECT_GE(static_cast<size_t>(strategy.get_model().size()), n);

    // Soft-prune a few rows — joint moments must reset (no early-return).
    auto prune = Tensor::zeros({static_cast<size_t>(strategy.get_model().size())},
                               Device::CUDA, DataType::Bool);
    {
        auto cpu = prune.cpu();
        auto* p = cpu.ptr<bool>();
        p[0] = true;
        p[1] = true;
        p[2] = true;
        prune = cpu.cuda();
    }
    ASSERT_NO_THROW(strategy.remove_gaussians(prune));

    means_st = strategy.get_optimizer().get_state_mutable(ParamType::Means);
    ASSERT_NE(means_st, nullptr);
    EXPECT_TRUE(means_st->is_joint());
}

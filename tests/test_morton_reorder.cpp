/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/error.hpp"
#include "core/event_bridge/control_boundary.hpp"
#include "core/parameters.hpp"
#include "core/scene.hpp"
#include "core/sh_value_quant.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "lfs/training/joint_adam_codec.hpp"
#include "lfs/training/morton_reorder.hpp"
#include "lfs/training/sh_value_codec.hpp"
#include "lfs/training/sh_value_storage.hpp"
#include "optimizer/adam_optimizer.hpp"
#include "training/trainer.hpp"
#include "training/training_setup.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <gtest/gtest.h>
#include <string>
#include <tuple>
#include <vector>

using namespace lfs::core;
using namespace lfs::training;

namespace {

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

    void fill_joint_contiguous(
        AdamParamState& state,
        const size_t n,
        const int n_attr,
        const auto& m_of,
        const auto& v_of) {
        using C = joint_adam::Codec16;
        ASSERT_TRUE(state.is_joint());
        ASSERT_EQ(state.joint_bits, 16);
        const size_t nb = joint_adam::n_bounds_for_prims(n);
        std::vector<std::uint8_t> packed(n * static_cast<size_t>(n_attr) * 4, 0);
        std::vector<float> bounds(nb * 4, 0.0f);
        for (size_t b = 0; b < nb; ++b) {
            const size_t begin = b * 256;
            const size_t end = std::min(n, begin + 256);
            float mm[4];
            std::vector<float> bm;
            std::vector<float> bv;
            bm.reserve((end - begin) * static_cast<size_t>(n_attr));
            bv.reserve(bm.capacity());
            for (size_t i = begin; i < end; ++i) {
                for (int a = 0; a < n_attr; ++a) {
                    bm.push_back(m_of(i, a));
                    bv.push_back(v_of(i, a));
                }
            }
            C::reduce_bounds(bm.data(), bv.data(), bm.size(), mm);
            bounds[b * 4 + 0] = mm[0];
            bounds[b * 4 + 1] = mm[1];
            bounds[b * 4 + 2] = mm[2];
            bounds[b * 4 + 3] = mm[3];
            for (size_t i = begin; i < end; ++i) {
                for (int a = 0; a < n_attr; ++a) {
                    C::encode_g1g2(
                        packed.data(),
                        i * static_cast<size_t>(n_attr) + static_cast<size_t>(a),
                        m_of(i, a), v_of(i, a), mm[0], mm[1], mm[2], mm[3]);
                }
            }
        }
        ASSERT_EQ(cudaMemcpy(state.exp_avg.ptr<std::uint8_t>(), packed.data(),
                             packed.size(), cudaMemcpyHostToDevice),
                  cudaSuccess);
        ASSERT_EQ(cudaMemcpy(state.joint_bounds.ptr<float>(), bounds.data(),
                             bounds.size() * sizeof(float), cudaMemcpyHostToDevice),
                  cudaSuccess);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    }

    void decode_joint_contiguous_host(
        const AdamParamState& state,
        const size_t n,
        const int n_attr,
        std::vector<float>& m,
        std::vector<float>& v) {
        using C = joint_adam::Codec16;
        m.assign(n * static_cast<size_t>(n_attr), 0.0f);
        v.assign(n * static_cast<size_t>(n_attr), 0.0f);
        std::vector<std::uint8_t> packed(n * static_cast<size_t>(n_attr) * 4);
        const size_t nb = joint_adam::n_bounds_for_prims(n);
        std::vector<float> bounds(nb * 4);
        ASSERT_EQ(cudaMemcpy(packed.data(), state.exp_avg.ptr<std::uint8_t>(),
                             packed.size(), cudaMemcpyDeviceToHost),
                  cudaSuccess);
        ASSERT_EQ(cudaMemcpy(bounds.data(), state.joint_bounds.ptr<float>(),
                             bounds.size() * sizeof(float), cudaMemcpyDeviceToHost),
                  cudaSuccess);
        for (size_t i = 0; i < n; ++i) {
            const float* mm = bounds.data() + 4 * (i / 256);
            for (int a = 0; a < n_attr; ++a) {
                const size_t cell = i * static_cast<size_t>(n_attr) + static_cast<size_t>(a);
                C::decode_g1g2(packed.data(), cell, mm[0], mm[1], mm[2], mm[3],
                               m[cell], v[cell]);
            }
        }
    }

    [[nodiscard]] double max_abs_diff(const Tensor& a, const Tensor& b) {
        auto ac = a.cpu().contiguous();
        auto bc = b.cpu().contiguous();
        EXPECT_EQ(ac.numel(), bc.numel());
        const auto* pa = ac.ptr<float>();
        const auto* pb = bc.ptr<float>();
        double m = 0.0;
        for (size_t i = 0; i < ac.numel(); ++i) {
            m = std::max(m, std::abs(static_cast<double>(pa[i]) - static_cast<double>(pb[i])));
        }
        return m;
    }

} // namespace

TEST(MortonReorderTest, CadenceMatchesStopRefineAndZeroDisables) {
    EXPECT_FALSE(morton::should_reorder(0, 5000, 25000));
    EXPECT_FALSE(morton::should_reorder(1, 0, 25000));
    EXPECT_TRUE(morton::should_reorder(5000, 5000, 25000));
    EXPECT_TRUE(morton::should_reorder(25000, 5000, 25000));
    EXPECT_FALSE(morton::should_reorder(25001, 5000, 25000));
    EXPECT_FALSE(morton::should_reorder(4999, 5000, 25000));
}

TEST(MortonReorderTest, PreservesPerRowAttributesAndAdamMoments) {
    sh_value::set_sh_value_quant_enabled_for_testing(true);
    constexpr size_t n = 2048;
    auto splat = make_mixed_splat(n, 3);
    ASSERT_TRUE(sh_value::apply_shN_value_quant(splat));
    ASSERT_TRUE(splat.shN_value_quantized());

    splat._densification_info = Tensor::zeros({size_t{2}, n}, Device::CUDA);
    {
        auto cpu = splat._densification_info.cpu();
        auto* p = cpu.ptr<float>();
        for (size_t i = 0; i < n; ++i) {
            p[i] = static_cast<float>(i);
            p[n + i] = static_cast<float>(i) * 0.5f;
        }
        splat._densification_info = cpu.cuda();
    }

    AdamConfig cfg;
    cfg.initial_capacity = n;
    AdamOptimizer opt(splat, cfg);
    opt.allocate_gradients(n);

    auto* means_state = opt.get_state_mutable(ParamType::Means);
    ASSERT_NE(means_state, nullptr);
    fill_joint_contiguous(
        *means_state, n, 3,
        [](size_t i, int a) {
            return 0.01f * static_cast<float>(i + 1) * static_cast<float>(a + 1);
        },
        [](size_t i, int a) {
            return 1.0e-4f * static_cast<float>(i + 1) * static_cast<float>(a + 1);
        });

    std::vector<float> m_before;
    std::vector<float> v_before;
    decode_joint_contiguous_host(*means_state, n, 3, m_before, v_before);

    const auto means_before = splat.means().cpu().contiguous();
    const auto sh0_before = splat.sh0().cpu().contiguous();
    const auto scale_before = splat.scaling_raw().cpu().contiguous();
    const auto rot_before = splat.rotation_raw().cpu().contiguous();
    const auto opa_before = splat.opacity_raw().cpu().contiguous();
    const auto shN_before = splat.shN_canonical().cpu().contiguous();
    const auto dens_before = splat._densification_info.cpu().contiguous();

    const auto result = morton::apply_morton_reorder(splat, &opt);
    ASSERT_TRUE(result.applied);
    ASSERT_TRUE(result.permutation.is_valid());
    ASSERT_EQ(result.permutation.numel(), n);

    const auto perm = result.permutation.cpu().contiguous();
    const auto* ip = perm.ptr<std::int64_t>();
    bool moved = false;
    for (size_t i = 0; i < n; ++i) {
        if (ip[i] != static_cast<std::int64_t>(i)) {
            moved = true;
            break;
        }
    }
    EXPECT_TRUE(moved) << "synthetic positions should not already be Morton-sorted";

    auto gather_row = [](const Tensor& src, size_t row, size_t width) {
        std::vector<float> out(width);
        const auto* p = src.ptr<float>();
        std::memcpy(out.data(), p + row * width, width * sizeof(float));
        return out;
    };

    const auto means_after = splat.means().cpu().contiguous();
    const auto sh0_after = splat.sh0().cpu().contiguous();
    const auto scale_after = splat.scaling_raw().cpu().contiguous();
    const auto rot_after = splat.rotation_raw().cpu().contiguous();
    const auto opa_after = splat.opacity_raw().cpu().contiguous();
    const auto dens_after = splat._densification_info.cpu().contiguous();
    const auto shN_after = splat.shN_canonical().cpu().contiguous();

    for (size_t i = 0; i < n; ++i) {
        const auto src = static_cast<size_t>(ip[i]);
        EXPECT_EQ(gather_row(means_after, i, 3), gather_row(means_before, src, 3));
        EXPECT_EQ(gather_row(sh0_after, i, 3), gather_row(sh0_before, src, 3));
        EXPECT_EQ(gather_row(scale_after, i, 3), gather_row(scale_before, src, 3));
        EXPECT_EQ(gather_row(rot_after, i, 4), gather_row(rot_before, src, 4));
        EXPECT_EQ(gather_row(opa_after, i, 1), gather_row(opa_before, src, 1));
        EXPECT_FLOAT_EQ(dens_after.ptr<float>()[i], dens_before.ptr<float>()[src]);
        EXPECT_FLOAT_EQ(dens_after.ptr<float>()[n + i], dens_before.ptr<float>()[n + src]);
        const auto* shN_a = shN_after.ptr<float>() + i * 15 * 3;
        const auto* shN_b = shN_before.ptr<float>() + src * 15 * 3;
        double mse = 0.0;
        for (size_t k = 0; k < 45; ++k) {
            const double e = static_cast<double>(shN_a[k]) - static_cast<double>(shN_b[k]);
            mse += e * e;
        }
        mse /= 45.0;
        EXPECT_LT(mse, 1e-6) << "decoded shN row " << i << " src " << src;
    }

    std::vector<float> m_after;
    std::vector<float> v_after;
    const auto* means_state_after = opt.get_state(ParamType::Means);
    ASSERT_NE(means_state_after, nullptr);
    decode_joint_contiguous_host(*means_state_after, n, 3, m_after, v_after);
    for (size_t i = 0; i < n; ++i) {
        const auto src = static_cast<size_t>(ip[i]);
        for (int a = 0; a < 3; ++a) {
            const size_t di = i * 3 + static_cast<size_t>(a);
            const size_t si = src * 3 + static_cast<size_t>(a);
            const float m_tol = std::max(0.02f, 2.0e-4f * std::abs(m_before[si]));
            const float v_tol = std::max(0.01f, 2.0e-4f * std::abs(v_before[si]));
            EXPECT_NEAR(m_after[di], m_before[si], m_tol) << "m row " << i;
            EXPECT_NEAR(v_after[di], v_before[si], v_tol) << "v row " << i;
        }
    }

    ASSERT_TRUE(splat.shN_value_quantized());
    sh_value::set_sh_value_quant_enabled_for_testing(std::nullopt);
}

TEST(MortonReorderTest, FrozenRangesSkipLeavesRowsUntouched) {
    sh_value::set_sh_value_quant_enabled_for_testing(true);
    constexpr size_t n = 512;
    auto splat = make_mixed_splat(n, 3);
    ASSERT_TRUE(sh_value::apply_shN_value_quant(splat));
    splat.set_frozen_ranges({{.start = 0, .count = 32}});

    const auto means_before = splat.means().cpu().contiguous();
    const auto shN_before = splat.shN_canonical().cpu().contiguous();

    const auto result = morton::apply_morton_reorder(splat, nullptr);
    EXPECT_FALSE(result.applied);
    EXPECT_FALSE(result.permutation.is_valid());
    EXPECT_LT(max_abs_diff(means_before, splat.means()), 1e-12);
    EXPECT_LT(max_abs_diff(shN_before, splat.shN_canonical()), 1e-12);

    sh_value::set_sh_value_quant_enabled_for_testing(std::nullopt);
}

TEST(MortonReorderTest, TrainingLossStaysContinuousAndCountMatches) {
    const auto data_path = std::filesystem::path(TEST_DATA_DIR) / "bicycle";
    if (!std::filesystem::exists(data_path / "sparse" / "0" / "cameras.bin")) {
        GTEST_SKIP() << "bicycle dataset not available";
    }

    auto run = [&](const size_t interval)
        -> std::tuple<std::vector<float>, std::vector<std::size_t>, std::size_t> {
        lfs::core::param::TrainingParameters params;
        params.dataset.data_path = data_path;
        params.dataset.images = "images_4";
        params.dataset.output_path = std::filesystem::temp_directory_path() /
                                     ("lfs_morton_reorder_" + std::to_string(interval));
        std::error_code ec;
        std::filesystem::create_directories(params.dataset.output_path, ec);
        params.optimization.iterations = 8;
        params.optimization.strategy = "mcmc";
        params.optimization.sh_degree = 1;
        params.optimization.headless = true;
        params.optimization.max_cap = 100000;
        params.optimization.refine_every = 100;
        params.optimization.start_refine = 0;
        params.optimization.stop_refine = 8;
        params.optimization.morton_reorder_interval = interval;
        params.optimization.save_steps = {};
        params.optimization.eval_steps = {};

        std::vector<float> losses;
        std::vector<std::size_t> counts;
        auto& boundary = ControlBoundary::instance();
        const auto handle = boundary.register_callback(
            ControlHook::PostStep, [&](const HookContext& ctx) {
                losses.push_back(ctx.loss);
                counts.push_back(ctx.num_gaussians);
            });

        Scene scene;
        const auto loaded = loadTrainingDataIntoScene(params, scene);
        if (!loaded) {
            ADD_FAILURE() << loaded.error();
            boundary.unregister_callback(ControlHook::PostStep, handle);
            return {std::vector<float>{}, std::vector<std::size_t>{}, std::size_t{0}};
        }
        const auto inited = initializeTrainingModel(params, scene);
        if (!inited) {
            ADD_FAILURE() << inited.error();
            boundary.unregister_callback(ControlHook::PostStep, handle);
            return {std::vector<float>{}, std::vector<std::size_t>{}, std::size_t{0}};
        }
        Trainer trainer(scene);
        const auto init = trainer.initialize(params);
        if (!init) {
            ADD_FAILURE() << init.error();
            boundary.unregister_callback(ControlHook::PostStep, handle);
            return {std::vector<float>{}, std::vector<std::size_t>{}, std::size_t{0}};
        }
        const auto trained = trainer.train();
        if (!trained) {
            ADD_FAILURE() << lfs::format_for_developer(trained.error());
            trainer.shutdown();
            boundary.unregister_callback(ControlHook::PostStep, handle);
            return {std::vector<float>{}, std::vector<std::size_t>{}, std::size_t{0}};
        }
        const std::size_t n_end = trainer.get_strategy().get_model().size();
        trainer.shutdown();
        boundary.unregister_callback(ControlHook::PostStep, handle);
        return std::tuple{losses, counts, n_end};
    };

    const auto [off_losses, off_counts, off_n] = run(0);
    const auto [on_losses, on_counts, on_n] = run(4);

    ASSERT_FALSE(off_losses.empty());
    ASSERT_EQ(off_n, on_n);
    ASSERT_EQ(off_counts.size(), on_counts.size());
    for (size_t i = 0; i < off_counts.size(); ++i) {
        EXPECT_EQ(off_counts[i], on_counts[i]) << "iter slot " << i;
    }

    ASSERT_GE(on_losses.size(), 4u);
    const size_t reorder_slot = 3; // 1-based iter 4, 0-based if every PostStep fired
    // Locate the first recorded iter near the reorder step by using the longest
    // adjacent jump as a spike detector, not as a required failure.
    std::vector<float> deltas;
    for (size_t i = 1; i < on_losses.size(); ++i) {
        deltas.push_back(std::abs(on_losses[i] - on_losses[i - 1]));
    }
    std::vector<float> sorted = deltas;
    std::sort(sorted.begin(), sorted.end());
    const float median = sorted[sorted.size() / 2];
    const float cap = std::max(median * 25.0f, 0.5f);
    for (size_t i = 0; i < deltas.size(); ++i) {
        EXPECT_LT(deltas[i], cap)
            << "loss spike at step " << (i + 1) << " delta=" << deltas[i]
            << " median=" << median;
    }
    (void)reorder_slot;

    const size_t cmp = std::min(off_losses.size(), on_losses.size());
    for (size_t i = 0; i < cmp; ++i) {
        const float denom = std::max(std::abs(off_losses[i]), 1e-4f);
        EXPECT_LT(std::abs(on_losses[i] - off_losses[i]) / denom, 0.25f)
            << "loss diverged at sample " << i << " off=" << off_losses[i]
            << " on=" << on_losses[i];
    }
}

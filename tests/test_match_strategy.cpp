/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/parameters.hpp"
#include "core/splat_data.hpp"
#include "lfs/training/match_geometry.cuh"
#include "training/kernels/match_kernels.hpp"
#include "training/strategies/match.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <set>
#include <vector>

using namespace lfs::core;
using namespace lfs::training;

namespace {

    float logit(const float p) { return std::log(p / (1.0f - p)); }
    float sigmoid(const float x) { return 1.0f / (1.0f + std::exp(-x)); }

    struct SplatInit {
        int n = 16;
        float opacity = 0.5f;
        float log_scale = -2.0f;
    };

    SplatData make_splats(const SplatInit& init) {
        const auto n = static_cast<size_t>(init.n);
        std::vector<float> means(n * 3), sh0(n * 3, 0.1f), shN(n * 45, 0.0f), scaling(n * 3, init.log_scale),
            rotation(n * 4, 0.0f), opacity(n, logit(init.opacity));
        for (size_t i = 0; i < n; ++i) {
            means[i * 3 + 0] = static_cast<float>(i);
            means[i * 3 + 2] = 5.0f;
            rotation[i * 4] = 1.0f;
            sh0[i * 3] = static_cast<float>(i);
        }
        return SplatData(3, Tensor::from_vector(means, TensorShape({n, 3}), Device::CUDA),
                         Tensor::from_vector(sh0, TensorShape({n, 3}), Device::CUDA),
                         Tensor::from_vector(shN, TensorShape({n, 15, 3}), Device::CUDA),
                         Tensor::from_vector(scaling, TensorShape({n, 3}), Device::CUDA),
                         Tensor::from_vector(rotation, TensorShape({n, 4}), Device::CUDA),
                         Tensor::from_vector(opacity, TensorShape({n, 1}), Device::CUDA), 1.0f);
    }

    param::OptimizationParameters match_params(const int cap) {
        auto params = param::OptimizationParameters::match_defaults();
        params.max_cap = cap;
        params.use_exposure_correction = false;
        return params;
    }

    std::vector<float> host(const Tensor& t) { return t.cpu().contiguous().to_vector(); }

    // Writes the leading values.size() floats of a CUDA tensor in place (views and preallocated rows alike).
    void set_rows(const Tensor& row, const std::vector<float>& values) {
        ASSERT_GE(row.numel(), values.size());
        ASSERT_EQ(cudaMemcpy(const_cast<Tensor&>(row).ptr<float>(), values.data(), values.size() * sizeof(float),
                             cudaMemcpyHostToDevice),
                  cudaSuccess);
    }

} // namespace

TEST(MatchStrategyTest, RefineScheduleFollowsTheConfiguredWindow) {
    auto splats = make_splats({});
    Match strategy(splats);
    strategy.initialize(match_params(64));
    // The hooks of iteration i see step i - 2; refines at steps 600, 700, ..., 27,400.
    EXPECT_FALSE(strategy.refine_step(500));
    EXPECT_TRUE(strategy.refine_step(600));
    EXPECT_FALSE(strategy.refine_step(650));
    EXPECT_TRUE(strategy.refine_step(27'400));
    EXPECT_FALSE(strategy.refine_step(27'500));
    EXPECT_TRUE(strategy.is_refining(602));
    EXPECT_FALSE(strategy.is_refining(601));

    EXPECT_FLOAT_EQ(strategy.split_opacity_share(0), 0.5f);
    EXPECT_FLOAT_EQ(strategy.split_opacity_share(7'500), 0.525f);
    EXPECT_FLOAT_EQ(strategy.split_opacity_share(15'000), 0.55f);
    EXPECT_FLOAT_EQ(strategy.split_opacity_share(29'000), 0.55f);

    EXPECT_NEAR(strategy.walk_amplitude(0), 0.05 * 160.0 * std::pow(0.01, 0.5 / 30'000.0), 1e-4);
    EXPECT_NEAR(strategy.walk_amplitude(29'999), 0.05 * 1.6, 1e-4);
}

TEST(MatchStrategyTest, LearningRatesFollowTheExponentialSchedules) {
    auto splats = make_splats({});
    Match strategy(splats);
    strategy.initialize(match_params(64));
    auto& optimizer = strategy.get_optimizer();
    EXPECT_NEAR(optimizer.get_param_lr(ParamType::Means), 2.56e-4, 1e-9);
    EXPECT_NEAR(optimizer.get_param_lr(ParamType::Scaling), 0.02, 1e-7);
    EXPECT_NEAR(optimizer.get_param_lr(ParamType::Opacity), 0.025, 1e-7);
    EXPECT_NEAR(optimizer.get_param_lr(ParamType::Sh0), 0.005, 1e-7);
    EXPECT_NEAR(optimizer.get_param_lr(ParamType::ShN), 0.00025, 1e-8);
    RenderOutput out;
    strategy.post_backward(15'001, out); // iteration 15,001 renders step 15,000
    EXPECT_NEAR(optimizer.get_param_lr(ParamType::Means), 2.56e-4 * std::pow(3.2e-6 / 2.56e-4, 0.5), 1e-9);
    EXPECT_NEAR(optimizer.get_param_lr(ParamType::Scaling), 0.02 * std::pow(0.25, 0.5), 1e-7);
    // Full SH degree from step 3.
    EXPECT_EQ(splats.get_active_sh_degree(), 3);
}

TEST(MatchStrategyTest, ScoreIsTheRunningMeanOfTheErrorPowerOverScoringSteps) {
    auto splats = make_splats({.n = 4});
    Match strategy(splats);
    auto params = match_params(64);
    params.match_walk = 0.0f;
    strategy.initialize(params);
    RenderOutput out;

    const auto feed = [&](const std::vector<float>& error, const std::vector<float>& visible,
                          const std::vector<float>& peak, const std::vector<float>& radius, const int iter) {
        auto rows = strategy.explore_scratch(iter);
        set_rows(rows.first, visible);
        set_rows(rows.second, peak);
        set_rows(strategy.fused_geometry_hooks(iter).view_radius, radius);
        set_rows(splats._densification_info, error);
        strategy.post_backward(iter, out);
    };
    // Splat 0: errors 32 then 1 -> scores 2 and 1 -> mean 1.5. Splat 1: visible with zero error, skipped.
    // Splat 2: not visible. Splat 3: oversize, radius 0.6 = twice the limit.
    feed({32.0f, 0.0f, 5.0f, 1.0f}, {1, 1, 0, 1}, {0.5f, 0.2f, 0.0f, 0.9f}, {0.01f, 0.01f, 0.0f, 0.6f}, 3);
    feed({1.0f, 0.0f, 5.0f, 1.0f}, {1, 1, 0, 1}, {0.5f, 0.0f, 0.0f, 0.9f}, {0.01f, 0.01f, 0.0f, 0.6f}, 4);

    const auto score = host(strategy.score_rows());
    const auto count = host(strategy.score_count_rows());
    EXPECT_NEAR(score[0], 1.5f, 1e-5f);
    EXPECT_FLOAT_EQ(count[0], 2.0f);
    EXPECT_FLOAT_EQ(score[1], 0.0f);
    EXPECT_FLOAT_EQ(count[1], 0.0f);
    EXPECT_FLOAT_EQ(count[2], 0.0f);
    EXPECT_NEAR(score[3], 1.0f, 1e-6f);
    const auto oversize = host(strategy.oversize_rows());
    EXPECT_NEAR(oversize[3], 2.0f, 1e-5f);
    EXPECT_FLOAT_EQ(oversize[0], 0.0f);
    // Unrendered streaks: splat 1 rendered on the first step only, splat 2 never.
    const auto streak = host(strategy.streak_rows());
    EXPECT_FLOAT_EQ(streak[0], 0.0f);
    EXPECT_FLOAT_EQ(streak[1], 1.0f);
    EXPECT_FLOAT_EQ(streak[2], 2.0f);
    // Peak: running maximum with a half-life of 800 steps, starting at the opacity 0.5.
    const auto peak = host(strategy.peak_rows());
    const float decay = std::pow(0.5f, 1.0f / 800.0f);
    EXPECT_NEAR(peak[3], 0.9f, 1e-6f);
    EXPECT_NEAR(peak[1], 0.5f * decay * decay, 1e-6f);
    // The step inputs are consumed.
    for (const float v : host(splats._densification_info))
        EXPECT_FLOAT_EQ(v, 0.0f);
    for (const float v : host(strategy.step_peak_rows()))
        EXPECT_FLOAT_EQ(v, 0.0f);
    for (const float v : host(strategy.visible_rows()))
        EXPECT_FLOAT_EQ(v, 0.0f);
}

TEST(MatchStrategyTest, SizeClipRunsEveryHundredSteps) {
    auto splats = make_splats({.n = 3});
    Match strategy(splats);
    strategy.initialize(match_params(64));
    RenderOutput out;
    const auto clip_at = [&](const int iter) {
        set_rows(strategy.fused_geometry_hooks(iter).view_radius, {0.3f, 0.45f, 3.0f});
        const auto before = host(splats.scaling_raw());
        strategy.post_backward(iter, out);
        const auto after = host(splats.scaling_raw());
        return std::vector<float>{before[0] - after[0], before[3] - after[3], before[6] - after[6]};
    };
    const auto none = clip_at(150); // step 148
    for (const float d : none)
        EXPECT_FLOAT_EQ(d, 0.0f);
    const auto clipped = clip_at(202); // step 200
    EXPECT_FLOAT_EQ(clipped[0], 0.0f);
    EXPECT_NEAR(clipped[1], std::log(1.5f), 1e-5f);
    EXPECT_NEAR(clipped[2], std::log(1.5f), 1e-5f); // capped at 1.5
}

TEST(MatchStrategyTest, DrawIsWithoutReplacementAndProportional) {
    const size_t n = 64;
    std::vector<float> score(n, 0.0f);
    std::vector<uint8_t> mask(n, 1);
    for (size_t i = 0; i < 40; ++i)
        score[i] = 1.0f + static_cast<float>(i % 4);
    mask[3] = 0;
    auto d_score = Tensor::from_vector(score, TensorShape({n}), Device::CUDA);
    auto d_mask = Tensor::from_vector(std::vector<int>(mask.begin(), mask.end()), TensorShape({n}), Device::CUDA)
                      .to(DataType::UInt8);
    auto out = Tensor::zeros({n}, Device::CUDA, DataType::Int64);

    // Asking for more than the rows of positive weight yields those rows, each once.
    const size_t drawn = match::draw_without_replacement(d_score.ptr<float>(), nullptr, d_mask.ptr<uint8_t>(), n, n,
                                                         7, out.ptr<int64_t>());
    EXPECT_EQ(drawn, 39u);
    auto rows = out.slice(0, 0, drawn).cpu().to(DataType::Float32).to_vector();
    std::set<int> unique(rows.begin(), rows.end());
    EXPECT_EQ(unique.size(), 39u);
    EXPECT_FALSE(unique.contains(3));
    for (const int r : unique)
        EXPECT_LT(r, 40);

    // A single draw picks row i with probability w_i / sum(w): rows of weight 4 four times as often as weight 1.
    int heavy = 0, light = 0;
    for (uint64_t seed = 0; seed < 4000; ++seed) {
        ASSERT_EQ(match::draw_without_replacement(d_score.ptr<float>(), nullptr, nullptr, n, 1, seed,
                                                  out.ptr<int64_t>()),
                  1u);
        const int row = static_cast<int>(out.slice(0, 0, 1).cpu().to(DataType::Float32).to_vector()[0]);
        heavy += row % 4 == 3;
        light += row % 4 == 0;
    }
    EXPECT_NEAR(static_cast<double>(heavy) / light, 4.0, 0.6);

    // The oversize weight multiplies the score by (oversize + 0.001): a zero score is never drawn.
    std::vector<float> over(n, 0.0f);
    over[50] = 100.0f; // score 0
    over[10] = 100.0f;
    auto d_over = Tensor::from_vector(over, TensorShape({n}), Device::CUDA);
    int tens = 0;
    for (uint64_t seed = 0; seed < 200; ++seed) {
        match::draw_without_replacement(d_score.ptr<float>(), d_over.ptr<float>(), nullptr, n, 1, seed,
                                        out.ptr<int64_t>());
        const int row = static_cast<int>(out.slice(0, 0, 1).cpu().to(DataType::Float32).to_vector()[0]);
        EXPECT_NE(row, 50);
        tens += row == 10;
    }
    EXPECT_GT(tens, 190);
}

TEST(MatchStrategyTest, LongAxisSplitPlacesBothHalvesAndDimsBoth) {
    auto splats = make_splats({.n = 2, .opacity = 0.6f});
    Match strategy(splats);
    strategy.initialize(match_params(64));
    // Splat 0: long axis y after a 90 degree turn about z of a splat long in x.
    auto scales = host(splats.scaling_raw());
    scales[0] = std::log(2.0f);
    scales[1] = std::log(1.0f);
    scales[2] = std::log(0.5f);
    set_rows(splats.scaling_raw(), scales);
    auto rot = host(splats.rotation_raw());
    rot[0] = std::cos(0.25f * 3.14159265f);
    rot[3] = std::sin(0.25f * 3.14159265f);
    set_rows(splats.rotation_raw(), rot);
    set_rows(strategy.peak_rows(), {0.8f, 0.3f});

    match::SplitRows rows;
    rows.means = splats.means().ptr<float>();
    rows.log_scales = splats.scaling_raw().ptr<float>();
    rows.raw_rotations = splats.rotation_raw().ptr<float>();
    rows.raw_opacities = splats.opacity_raw().ptr<float>();
    rows.sh0 = splats.sh0().ptr<float>();
    auto score = strategy.score_rows(), count = strategy.score_count_rows(), over = strategy.oversize_rows(),
         peak = strategy.peak_rows(), streak = strategy.streak_rows();
    set_rows(score, {3.0f, 0.0f});
    set_rows(streak, {5.0f, 9.0f});
    rows.score = score.ptr<float>();
    rows.score_count = count.ptr<float>();
    rows.oversize = over.ptr<float>();
    rows.peak = peak.ptr<float>();
    rows.streak = streak.ptr<float>();
    auto src = Tensor::from_vector(std::vector<int>{0}, TensorShape({1}), Device::CUDA).to(DataType::Int64);
    auto dst = Tensor::from_vector(std::vector<int>{1}, TensorShape({1}), Device::CUDA).to(DataType::Int64);
    match::launch_long_axis_split(rows, src.ptr<int64_t>(), dst.ptr<int64_t>(), 1, 0.5f);

    const auto means = host(splats.means());
    // Offset 0.5 x sigma_long = 1 along the rotated long axis (y).
    EXPECT_NEAR(means[0], 0.0f, 1e-5f);
    EXPECT_NEAR(means[1], -1.0f, 1e-5f);
    EXPECT_NEAR(means[3], 0.0f, 1e-5f);
    EXPECT_NEAR(means[4], 1.0f, 1e-5f);
    EXPECT_NEAR(means[5], 5.0f, 1e-5f);
    const auto s = host(splats.scaling_raw());
    for (const int base : {0, 3}) {
        EXPECT_NEAR(s[base + 0], std::log(2.0f * 0.5f), 1e-5f);
        EXPECT_NEAR(s[base + 1], std::log(1.0f * 0.85f), 1e-5f);
        EXPECT_NEAR(s[base + 2], std::log(0.5f * 0.85f), 1e-5f);
    }
    const auto op = host(splats.opacity_raw());
    EXPECT_NEAR(sigmoid(op[0]), 0.3f, 1e-5f);
    EXPECT_NEAR(sigmoid(op[1]), 0.3f, 1e-5f);
    const auto q = host(splats.rotation_raw());
    for (int c = 0; c < 4; ++c)
        EXPECT_FLOAT_EQ(q[4 + c], q[c]);
    EXPECT_FLOAT_EQ(host(splats.sh0())[3], 0.0f);
    const auto p = host(peak);
    EXPECT_NEAR(p[0], 0.4f, 1e-6f);
    EXPECT_NEAR(p[1], 0.4f, 1e-6f);
    const auto st = host(streak);
    EXPECT_FLOAT_EQ(st[0], 0.0f);
    EXPECT_FLOAT_EQ(st[1], 0.0f);
    EXPECT_FLOAT_EQ(host(score)[1], 3.0f);
}

TEST(MatchStrategyTest, DeadSplatsAreRevivedWithinTheBudget) {
    auto splats = make_splats({.n = 20});
    Match strategy(splats);
    strategy.initialize(match_params(64));
    auto op = host(splats.opacity_raw());
    for (int i = 0; i < 5; ++i)
        op[i] = logit(0.001f); // dead by opacity
    set_rows(splats.opacity_raw(), op);
    std::vector<float> score(20, 1.0f);
    set_rows(strategy.score_rows(), score);

    // Budget max(1, floor(0.1 x 20)) = 2 revivals; growth to floor(1.05 x 20) = 21 finds no budget left.
    const auto counts = strategy.refine(600);
    EXPECT_EQ(counts.dead, 5u);
    EXPECT_EQ(counts.revived, 2u);
    EXPECT_EQ(counts.born + counts.oversize_born, 0u);
    EXPECT_EQ(splats.size(), 20);
    const auto after = host(splats.opacity_raw());
    // The two lowest dead slots took half a live opacity (0.5 x 0.5); the rest stay dead and untouched.
    EXPECT_NEAR(sigmoid(after[0]), 0.25f, 1e-5f);
    EXPECT_NEAR(sigmoid(after[1]), 0.25f, 1e-5f);
    for (int i = 2; i < 5; ++i)
        EXPECT_FLOAT_EQ(after[i], op[i]);
    int dimmed = 0;
    for (int i = 5; i < 20; ++i)
        dimmed += std::abs(sigmoid(after[i]) - 0.25f) < 1e-5f;
    EXPECT_EQ(dimmed, 2);
    // Scores reset after the refine.
    for (const float v : host(strategy.score_rows()))
        EXPECT_FLOAT_EQ(v, 0.0f);
}

TEST(MatchStrategyTest, GrowthAppendsSplitChildrenUpToTheGrowthFactor) {
    auto splats = make_splats({.n = 100});
    Match strategy(splats);
    strategy.initialize(match_params(1000));
    std::vector<float> score(100, 1.0f), over(100, 0.0f);
    over[7] = 50.0f;
    set_rows(strategy.score_rows(), score);
    set_rows(strategy.oversize_rows(), over);

    const auto counts = strategy.refine(600);
    EXPECT_EQ(counts.dead, 0u);
    EXPECT_EQ(counts.oversize_born, 1u); // round(0.15 x 5)
    EXPECT_EQ(counts.born, 4u);
    EXPECT_EQ(splats.size(), 105);
    // The oversize draw (weight 50 x 1 against 99 x 0.001) took splat 7: the first child is its copy. Every
    // child has half its donor's opacity; the regular draw may pick a fresh child, which halves it again.
    EXPECT_FLOAT_EQ(host(splats.sh0())[100 * 3], 7.0f);
    const auto op = host(splats.opacity_raw());
    for (int i = 100; i < 105; ++i) {
        const float o = sigmoid(op[i]);
        EXPECT_TRUE(std::abs(o - 0.25f) < 1e-5f || std::abs(o - 0.125f) < 1e-5f) << i << ": " << o;
    }
    EXPECT_LT(sigmoid(op[7]), 0.25f + 1e-5f);
    EXPECT_EQ(strategy.score_rows().numel(), 105u);
    EXPECT_EQ(strategy.get_optimizer().get_state(ParamType::Means)->size, 105u);
}

TEST(MatchStrategyTest, GrowthStopsAtTheCap) {
    auto splats = make_splats({.n = 100});
    Match strategy(splats);
    strategy.initialize(match_params(102));
    set_rows(strategy.score_rows(), std::vector<float>(100, 1.0f));
    const auto counts = strategy.refine(600);
    EXPECT_EQ(counts.born + counts.oversize_born, 2u);
    EXPECT_EQ(splats.size(), 102);
}

// The position step is a share of the camera distance, shaped by the splat: two splats of different size at
// the same distance take the same step along their long axis; the same splat twice as far takes twice the step.
TEST(MatchGeometryTest, AngularMeanStepScalesWithDistanceNotSize) {
    namespace mg = lfs::training::match_geometry;
    const float q[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    const float g[3] = {0.3f, 0.0f, 0.0f};
    const float opacity = 0.8f;
    const float lr = 2.56e-4f;
    // First bias-corrected Adam step: lr x m_hat / sqrt(v_hat) = lr x u / |0.6 r g|.
    const auto first_step = [&](const float sigma_long, const float distance) {
        const float log_scales[3] = {std::log(sigma_long), std::log(0.2f * sigma_long), std::log(0.1f * sigma_long)};
        float u[3];
        mg::angular_first_moment(q, log_scales, opacity, g, u);
        const float r = mg::view_radius(log_scales[0], opacity, distance);
        return lr * u[0] / std::sqrt(g[0] * g[0] * mg::angular_second_moment_scale(r));
    };
    const float small_near = first_step(0.01f, 10.0f);
    const float big_near = first_step(0.08f, 10.0f);
    const float small_far = first_step(0.01f, 20.0f);
    EXPECT_NEAR(big_near / small_near, 1.0f, 1e-3f);
    EXPECT_NEAR(small_far / small_near, 2.0f, 1e-3f);
    // About 3.3 lr x distance far away.
    EXPECT_NEAR(small_near / (lr * 10.0f), 2.0f / 0.6f, 0.01f);
    // The view radius is about rho / (2 d) far away and 1 when the camera is inside the footprint.
    const float rho = 0.01f * mg::footprint_extent(opacity, 1.0f);
    EXPECT_NEAR(mg::view_radius(std::log(0.01f), opacity, 10.0f), rho / 20.0f, 1e-7f);
    EXPECT_NEAR(mg::view_radius(std::log(10.0f), opacity, 1.0f), 1.0f, 1e-6f);
    // Hinge: one learning-rate step per octave over the limit.
    EXPECT_FLOAT_EQ(mg::hinge_push(0.6f, 0.3f, 1.0f), 1.0f);
    EXPECT_FLOAT_EQ(mg::hinge_push(0.2f, 0.3f, 1.0f), 0.0f);
}

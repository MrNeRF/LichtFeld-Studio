/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/camera.hpp"
#include "core/cuda/memory_arena.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "lfs/training/morton_reorder.hpp"
#include "lfs/training/row_bias_correction.cuh"
#include "training/optimizer/adam_optimizer.hpp"
#include "training/rasterization/fast_rasterizer.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <memory>
#include <sstream>
#include <vector>

using namespace lfs::core;
using namespace lfs::training;

namespace {

    constexpr float kLr = 1.0e-3f;
    constexpr float kGrad = 0.5f;
    constexpr double kBeta1 = 0.9;
    constexpr double kBeta2 = 0.999;

    // Deterministic splats, so two runs start from identical bytes.
    std::unique_ptr<SplatData> make_splat(const size_t n, const int degree) {
        std::vector<float> means(n * 3), sh0(n * 3), scaling(n * 3), rotation(n * 4, 0.0f), opacity(n);
        std::uint32_t state = 12345u;
        const auto next = [&state] {
            state = state * 1664525u + 1013904223u;
            return static_cast<float>(state >> 8) / static_cast<float>(1u << 24);
        };
        for (size_t i = 0; i < n; ++i) {
            for (int a = 0; a < 3; ++a) {
                means[i * 3 + a] = next() * 4.0f - 2.0f;
                sh0[i * 3 + a] = next() - 0.5f;
                scaling[i * 3 + a] = -3.0f + next();
            }
            rotation[i * 4] = 1.0f;
            rotation[i * 4 + 1] = 0.1f * next();
            opacity[i] = next() * 2.0f - 1.0f;
        }
        const size_t rest = degree > 0 ? static_cast<size_t>(degree * (degree + 2)) : 0;
        std::vector<float> shN(n * rest * 3);
        for (auto& value : shN)
            value = 0.1f * (next() - 0.5f);
        auto splat = std::make_unique<SplatData>(
            degree,
            Tensor::from_vector(means, TensorShape({n, 3}), Device::CUDA),
            Tensor::from_vector(sh0, TensorShape({n, 1, 3}), Device::CUDA),
            Tensor::from_vector(shN, TensorShape({n, rest, 3}), Device::CUDA),
            Tensor::from_vector(scaling, TensorShape({n, 3}), Device::CUDA),
            Tensor::from_vector(rotation, TensorShape({n, 4}), Device::CUDA),
            Tensor::from_vector(opacity, TensorShape({n, 1}), Device::CUDA),
            1.0f);
        if (degree > 0)
            splat->set_active_sh_degree(degree);
        return splat;
    }

    std::vector<float> host_floats(const Tensor& tensor) {
        const auto host = tensor.cpu().contiguous();
        const auto* p = host.ptr<float>();
        return {p, p + host.numel()};
    }

    std::vector<std::int32_t> host_steps(const AdamOptimizer& opt) {
        if (!opt.row_steps().is_valid())
            return {};
        const auto host = opt.row_steps().cpu().contiguous();
        const auto* p = host.ptr<std::int32_t>();
        return {p, p + host.numel()};
    }

    bool same_bits(const float a, const float b) { return std::memcmp(&a, &b, sizeof(float)) == 0; }

    // One splat model and its optimizer, stepped with a constant gradient on every group.
    struct AdamRun {
        std::unique_ptr<SplatData> splat;
        std::unique_ptr<AdamOptimizer> opt;

        AdamRun(const size_t n, const int degree, const bool per_splat) : splat(make_splat(n, degree)) {
            AdamConfig cfg;
            cfg.lr = kLr;
            cfg.beta1 = kBeta1;
            cfg.beta2 = kBeta2;
            cfg.eps = 1e-15;
            opt = std::make_unique<AdamOptimizer>(*splat, cfg);
            opt->allocate_gradients();
            opt->set_per_splat_bias_correction(per_splat);
        }

        bool has(const ParamType type) const {
            const auto* state = opt->get_state(type);
            return state != nullptr && state->exp_avg.is_valid() && state->size > 0;
        }

        void set_group_steps(const int64_t steps) {
            for (const auto type : AdamOptimizer::all_param_types())
                if (auto* state = opt->get_state_mutable(type))
                    state->step_count = steps;
        }

        void step(const int iteration, const float grad = kGrad) {
            for (const auto type : AdamOptimizer::all_param_types())
                if (has(type))
                    opt->get_grad(type).fill_(grad);
            opt->step(iteration);
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        }

        void reset(const std::vector<int64_t>& rows) {
            for (const auto type : AdamOptimizer::all_param_types())
                if (has(type))
                    opt->reset_state_at_indices(type, rows);
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        }

        // Canonical per-row parameters of every group, in all_param_types order.
        std::vector<std::vector<float>> params() const {
            return {host_floats(splat->means()), host_floats(splat->sh0()),
                    splat->shN().is_valid() && splat->shN().numel() > 0 ? host_floats(splat->shN_canonical())
                                                                        : std::vector<float>{},
                    host_floats(splat->scaling_raw()), host_floats(splat->rotation_raw()),
                    host_floats(splat->opacity_raw())};
        }
    };

    const char* group_name(const size_t g) {
        static const char* names[] = {"means", "sh0", "shN", "scaling", "rotation", "opacity"};
        return names[g];
    }

    // |update| / lr of a row on its k-th step since a reset when the group correction for group
    // step `step` is applied instead of the row's own (constant gradient).
    double group_corrected_ratio(const int k, const int step) {
        return (1.0 - std::pow(kBeta1, k)) / std::sqrt(1.0 - std::pow(kBeta2, k)) *
               std::sqrt(1.0 - std::pow(kBeta2, step)) / (1.0 - std::pow(kBeta1, step));
    }

} // namespace

// Off, the optimizer applies the group correction; at the group's first step that is the
// textbook step (|update| = lr for a constant gradient). Counters that track the group step
// exactly fall back to the same arithmetic, bit for bit.
TEST(PerSplatBiasCorrection, CountersAtTheGroupStepKeepTheGroupArithmetic) {
    constexpr size_t n = 300;
    AdamRun off(n, 1, false);
    AdamRun on(n, 1, true);
    on.opt->restart_row_steps();
    EXPECT_EQ(off.opt->row_steps().is_valid(), false);

    const auto start = off.params();
    for (int iteration = 1001; iteration <= 1006; ++iteration) {
        const float grad = kGrad * (1.0f + 0.25f * static_cast<float>(iteration % 3));
        off.step(iteration, grad);
        on.step(iteration, grad);
        if (iteration == 1001) {
            const auto after = off.params();
            for (size_t g = 0; g < after.size(); ++g)
                for (size_t i = 0; i < after[g].size(); ++i)
                    ASSERT_NEAR(start[g][i] - after[g][i], kLr, kLr * 1e-4) << group_name(g) << " " << i;
        }
    }
    const auto a = off.params();
    const auto b = on.params();
    for (size_t g = 0; g < a.size(); ++g) {
        ASSERT_EQ(a[g].size(), b[g].size());
        for (size_t i = 0; i < a[g].size(); ++i)
            ASSERT_TRUE(same_bits(a[g][i], b[g][i])) << group_name(g) << " " << i << ": " << a[g][i] << " vs " << b[g][i];
    }
    for (const auto steps : host_steps(*on.opt))
        EXPECT_EQ(steps, 6);
}

// A row reset at global step 5000 takes the textbook first Adam step (|update| = lr) and keeps
// taking it under a constant gradient; off, the same row overshoots by the group correction.
// Every untouched row is bit-identical between off and on.
TEST(PerSplatBiasCorrection, RowResetAtStep5000TakesTheTextbookStep) {
    constexpr size_t n = 300;
    const std::vector<int64_t> reset_rows{7, 260};
    AdamRun off(n, 1, false);
    AdamRun on(n, 1, true);
    off.set_group_steps(4996);
    on.set_group_steps(4996);
    for (const auto steps : host_steps(*on.opt))
        ASSERT_EQ(steps, kRowStepSaturation);

    for (int iteration = 4997; iteration <= 4999; ++iteration) {
        off.step(iteration);
        on.step(iteration);
    }
    off.reset(reset_rows);
    on.reset(reset_rows);

    for (int k = 1; k <= 3; ++k) {
        const auto off_before = off.params();
        const auto on_before = on.params();
        off.step(4999 + k);
        on.step(4999 + k);
        const auto off_after = off.params();
        const auto on_after = on.params();
        for (size_t g = 0; g < off_after.size(); ++g) {
            const size_t width = off_after[g].size() / n;
            for (size_t row = 0; row < n; ++row) {
                const bool reset = row == reset_rows[0] || row == reset_rows[1];
                for (size_t c = 0; c < width; ++c) {
                    const size_t i = row * width + c;
                    if (!reset) {
                        ASSERT_TRUE(same_bits(off_after[g][i], on_after[g][i]))
                            << group_name(g) << " row " << row << " step " << k;
                        continue;
                    }
                    const double on_step = on_before[g][i] - on_after[g][i];
                    const double off_step = off_before[g][i] - off_after[g][i];
                    EXPECT_NEAR(on_step, kLr, kLr * 1e-4) << group_name(g) << " row " << row << " step " << k;
                    EXPECT_NEAR(off_step, kLr * group_corrected_ratio(k, 4999 + k), kLr * 1e-3)
                        << group_name(g) << " row " << row << " step " << k;
                }
            }
        }
    }
    const auto steps = host_steps(*on.opt);
    ASSERT_EQ(steps.size(), n);
    for (size_t row = 0; row < n; ++row) {
        const bool reset = row == reset_rows[0] || row == reset_rows[1];
        EXPECT_EQ(steps[row], reset ? 3 : kRowStepSaturation) << row;
    }
}

// The fused rasterizer-backward Adam applies the same per-row correction. A zero image gradient
// plus the log-scale regulariser gives every scale a constant gradient.
TEST(PerSplatBiasCorrection, FusedBackwardRowResetTakesTheTextbookStep) {
    constexpr int n = 64;
    constexpr int64_t reset_row = 7;
    const auto camera_for = [] {
        std::vector<float> r{1, 0, 0, 0, 1, 0, 0, 0, 1};
        std::vector<float> t{0, 0, 4};
        return Camera(Tensor::from_vector(r, TensorShape({3, 3}), Device::CUDA),
                      Tensor::from_vector(t, TensorShape({3}), Device::CUDA),
                      100.f, 100.f, 32.f, 32.f, Tensor(), Tensor(), CameraModelType::PINHOLE, "test", "",
                      std::filesystem::path{}, 64, 64, 0);
    };
    auto camera = camera_for();
    auto bg = Tensor::zeros({3}, Device::CUDA);

    struct FusedRun {
        std::unique_ptr<SplatData> splat;
        std::unique_ptr<AdamOptimizer> opt;
    };
    const auto make_run = [](const bool per_splat) {
        FusedRun run;
        std::vector<float> means(n * 3, 0.0f), scaling(n * 3), rotation(n * 4, 0.0f), opacity(n);
        for (int i = 0; i < n; ++i) {
            means[i * 3] = static_cast<float>(i % 5) * 0.3f - 0.6f;
            means[i * 3 + 1] = static_cast<float>(i / 5) * 0.3f - 0.6f;
            scaling[i * 3] = -2.0f + 0.01f * static_cast<float>(i);
            scaling[i * 3 + 1] = -1.8f;
            scaling[i * 3 + 2] = -2.2f;
            rotation[i * 4] = 1.0f;
            opacity[i] = 1.5f;
        }
        run.splat = std::make_unique<SplatData>(
            0, Tensor::from_vector(means, TensorShape({n, 3}), Device::CUDA),
            Tensor::full({static_cast<size_t>(n), 1, 3}, 0.5f, Device::CUDA),
            Tensor::zeros({static_cast<size_t>(n), 0, 3}, Device::CUDA),
            Tensor::from_vector(scaling, TensorShape({n, 3}), Device::CUDA),
            Tensor::from_vector(rotation, TensorShape({n, 4}), Device::CUDA),
            Tensor::from_vector(opacity, TensorShape({n}), Device::CUDA), 1.0f);
        AdamConfig cfg{.lr = kLr, .beta1 = kBeta1, .beta2 = kBeta2, .eps = 1e-15};
        run.opt = std::make_unique<AdamOptimizer>(*run.splat, cfg);
        run.opt->allocate_gradients();
        run.opt->zero_grad(0);
        run.opt->set_per_splat_bias_correction(per_splat);
        for (const auto type : AdamOptimizer::all_param_types())
            if (auto* state = run.opt->get_state_mutable(type))
                state->step_count = 4996;
        return run;
    };
    auto off = make_run(false);
    auto on = make_run(true);

    const auto fused_step = [&](FusedRun& run, const int iteration) {
        auto fwd = fast_rasterize_forward(camera, *run.splat, bg, 0, 0, 0, 0, false);
        ASSERT_TRUE(fwd.has_value()) << std::string(fwd.error().user_message());
        FastGSFusedExtraGradients extra;
        extra.scale_reg_weight = 0.01f;
        extra.scale_reg_log = true;
        const auto grad_image = Tensor::zeros_like(fwd->first.image);
        fast_rasterize_backward(fwd->second, grad_image, *run.splat, *run.opt, {}, {},
                                DensificationType::None, iteration, extra);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    };

    for (int iteration = 4997; iteration <= 4999; ++iteration) {
        fused_step(off, iteration);
        fused_step(on, iteration);
    }
    for (auto* run : {&off, &on})
        for (const auto type : AdamOptimizer::all_param_types())
            if (const auto* state = run->opt->get_state(type); state != nullptr && state->exp_avg.is_valid())
                run->opt->reset_state_at_indices(type, std::vector<int64_t>{reset_row});

    for (int k = 1; k <= 2; ++k) {
        const auto off_before = host_floats(off.splat->scaling_raw());
        const auto on_before = host_floats(on.splat->scaling_raw());
        fused_step(off, 4999 + k);
        fused_step(on, 4999 + k);
        const auto off_after = host_floats(off.splat->scaling_raw());
        const auto on_after = host_floats(on.splat->scaling_raw());
        for (int row = 0; row < n; ++row) {
            for (int c = 0; c < 3; ++c) {
                const size_t i = static_cast<size_t>(row) * 3 + c;
                if (row != reset_row) {
                    ASSERT_TRUE(same_bits(off_after[i], on_after[i])) << "row " << row << " step " << k;
                    continue;
                }
                EXPECT_NEAR(on_before[i] - on_after[i], kLr, kLr * 1e-4) << "step " << k << " axis " << c;
                EXPECT_NEAR(off_before[i] - off_after[i], kLr * group_corrected_ratio(k, 4999 + k), kLr * 1e-3)
                    << "step " << k << " axis " << c;
            }
        }
    }
    const auto steps = host_steps(*on.opt);
    ASSERT_EQ(steps.size(), static_cast<size_t>(n));
    for (int row = 0; row < n; ++row)
        EXPECT_EQ(steps[row], row == reset_row ? 2 : kRowStepSaturation) << row;
    GlobalArenaManager::instance().get_arena().full_reset();
}

// Counter lifecycle: start saturated, restart at reset (vector and tensor paths), advance once per
// step, hold for a frozen row at scale 0, saturate, start at 0 for appended rows, and survive a
// checkpoint round trip; with the switch off the stream stays at version 3.
TEST(PerSplatBiasCorrection, CountersFollowResetAppendFreezeSaturationAndCheckpoint) {
    constexpr size_t n = 300;
    constexpr size_t added = 10;
    AdamRun run(n, 1, true);
    run.splat->reserve_capacity(n + 64);
    auto steps = host_steps(*run.opt);
    ASSERT_EQ(steps.size(), n);
    for (const auto s : steps)
        EXPECT_EQ(s, kRowStepSaturation);

    run.step(1001);
    run.opt->reset_state_at_indices(ParamType::Means, std::vector<int64_t>{3});
    run.opt->reset_state_at_indices(ParamType::Opacity,
                                    Tensor::from_vector(std::vector<int>{5}, TensorShape({1}), Device::CUDA)
                                        .to(DataType::Int64));
    run.opt->restart_row_steps_at(
        Tensor::from_vector(std::vector<int>{9}, TensorShape({1}), Device::CUDA).to(DataType::Int64));
    steps = host_steps(*run.opt);
    EXPECT_EQ(steps[3], 0);
    EXPECT_EQ(steps[5], 0);
    EXPECT_EQ(steps[9], 0);
    EXPECT_EQ(steps[4], kRowStepSaturation);

    run.step(1002);
    steps = host_steps(*run.opt);
    EXPECT_EQ(steps[3], 1);
    EXPECT_EQ(steps[5], 1);

    // Saturation.
    {
        const std::int32_t near = kRowStepSaturation - 1;
        ASSERT_EQ(cudaMemcpy(run.opt->row_steps_mutable().ptr<std::int32_t>() + 11, &near, sizeof(near),
                             cudaMemcpyHostToDevice),
                  cudaSuccess);
    }
    run.step(1003);
    run.step(1004);
    steps = host_steps(*run.opt);
    EXPECT_EQ(steps[11], kRowStepSaturation);
    EXPECT_EQ(steps[3], 3);

    // A frozen row at lr scale 0 neither moves nor counts a step.
    {
        std::vector<bool> frozen(n, false);
        frozen[3] = true;
        run.opt->set_frozen_mask(Tensor::from_vector(frozen, TensorShape({n}), Device::CUDA));
        run.opt->set_frozen_lr_scale(0.0f);
        run.step(1005);
        steps = host_steps(*run.opt);
        EXPECT_EQ(steps[3], 3);
        EXPECT_EQ(steps[5], 4);
        run.opt->set_frozen_mask(Tensor());
    }

    // Appended rows start at 0.
    const auto sources = Tensor::arange(0.0f, static_cast<float>(added), 1.0f).to(DataType::Int32).cuda();
    for (const auto type : AdamOptimizer::all_param_types())
        run.opt->add_new_params_gather(type, sources);
    ASSERT_EQ(run.splat->size(), n + added);
    steps = host_steps(*run.opt);
    ASSERT_EQ(steps.size(), n + added);
    for (size_t row = n; row < n + added; ++row)
        EXPECT_EQ(steps[row], 0) << row;
    run.step(1006);
    steps = host_steps(*run.opt);
    for (size_t row = n; row < n + added; ++row)
        EXPECT_EQ(steps[row], 1) << row;

    // Checkpoint round trip (version 4 carries the counters).
    std::stringstream stream(std::ios::in | std::ios::out | std::ios::binary);
    run.opt->serialize(stream);
    {
        std::uint32_t header[2] = {};
        std::memcpy(header, stream.str().data(), sizeof(header));
        EXPECT_EQ(header[1], 4u);
    }
    AdamOptimizer resumed(*run.splat, run.opt->get_config());
    ASSERT_NO_THROW(resumed.deserialize(stream));
    resumed.set_per_splat_bias_correction(true);
    EXPECT_EQ(host_steps(resumed), steps);

    // Off keeps today's version 3 stream; a v3 stream loads saturated counters.
    AdamRun off(n, 1, false);
    std::stringstream off_stream(std::ios::in | std::ios::out | std::ios::binary);
    off.opt->serialize(off_stream);
    {
        std::uint32_t header[2] = {};
        std::memcpy(header, off_stream.str().data(), sizeof(header));
        EXPECT_EQ(header[1], 3u);
    }
    AdamOptimizer from_v3(*off.splat, off.opt->get_config());
    ASSERT_NO_THROW(from_v3.deserialize(off_stream));
    from_v3.set_per_splat_bias_correction(true);
    for (const auto s : host_steps(from_v3))
        EXPECT_EQ(s, kRowStepSaturation);
}

// Morton reorder permutes the counters with the rows.
TEST(PerSplatBiasCorrection, MortonReorderPermutesTheCounters) {
    constexpr size_t n = 2048;
    AdamRun run(n, 0, true);
    std::vector<std::int32_t> values(n);
    for (size_t i = 0; i < n; ++i)
        values[i] = static_cast<std::int32_t>(i * 7 % 1000);
    ASSERT_EQ(cudaMemcpy(run.opt->row_steps_mutable().ptr<std::int32_t>(), values.data(), n * sizeof(std::int32_t),
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    const auto result = morton::apply_morton_reorder(*run.splat, run.opt.get());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_TRUE(result.applied);
    const auto perm = result.permutation.cpu().contiguous();
    const auto* p = perm.ptr<std::int64_t>();
    const auto steps = host_steps(*run.opt);
    ASSERT_EQ(steps.size(), n);
    bool moved = false;
    for (size_t i = 0; i < n; ++i) {
        EXPECT_EQ(steps[i], values[static_cast<size_t>(p[i])]) << i;
        moved |= p[i] != static_cast<std::int64_t>(i);
    }
    EXPECT_TRUE(moved);
}

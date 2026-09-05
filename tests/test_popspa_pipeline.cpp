/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/camera.hpp"
#include "core/cuda_error.hpp"
#include "core/parameters.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "lfs/training/live_model_mutation_guard.hpp"
#include "training/components/popspa_controller.hpp"
#include "training/rasterization/fast_rasterizer.hpp"
#include "training/rasterization/gsplat_rasterizer.hpp"
#include "training/strategies/postprocess_compaction.hpp"
#include "training/strategies/strategy_factory.hpp"
#include "training/strategies/strategy_utils.hpp"
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <gtest/gtest.h>
#include <memory>
#include <numeric>
#include <vector>

namespace {
    using namespace lfs::core;
    using namespace lfs::training;
    constexpr size_t kInitial = 8, kIntermediate = 5, kFinal = 3;
    constexpr int kWidth = 8, kHeight = 8;

    SplatData pipeline_model() {
        std::vector<float> means(kInitial * 3), sh0(kInitial * 3), quats(kInitial * 4, 0.f);
        for (size_t i = 0; i < kInitial; ++i) {
            means[i * 3] = -0.175f + 0.05f * float(i);
            means[i * 3 + 1] = (i % 2 ? 0.02f : -0.02f);
            means[i * 3 + 2] = 0.04f * float(i);
            quats[i * 4] = 1.f;
            sh0[i * 3] = 0.15f + 0.07f * float(i);
            sh0[i * 3 + 1] = -0.1f + 0.02f * float(i);
            sh0[i * 3 + 2] = 0.25f - 0.03f * float(i);
        }
        SplatData model(1,
                        Tensor::from_vector(means, {kInitial, 3}, Device::CUDA),
                        Tensor::from_vector(sh0, {kInitial, 1, 3}, Device::CUDA),
                        Tensor::full({kInitial, 3, 3}, 0.015f, Device::CUDA),
                        Tensor::full({kInitial, 3}, -1.5f, Device::CUDA),
                        Tensor::from_vector(quats, {kInitial, 4}, Device::CUDA),
                        Tensor::full({kInitial, 1}, -0.4f, Device::CUDA), 1.f);
        model.set_active_sh_degree(1);
        model.set_frozen_ranges({{kInitial - 1, 1}});
        return model;
    }

    std::unique_ptr<Camera> pipeline_camera(int view) {
        auto R = Tensor::from_vector({1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f}, {3, 3}, Device::CUDA);
        auto T = Tensor::from_vector({float(view) * 0.015f, 0.f, 3.f}, {3}, Device::CUDA);
        return std::make_unique<Camera>(R, T, 35.f, 35.f, 4.f, 4.f, Tensor{}, Tensor{},
                                        lfs::core::CameraModelType::PINHOLE, "pipeline", "",
                                        std::filesystem::path{}, kWidth, kHeight, view);
    }

    std::unique_ptr<AdamOptimizer> pipeline_optimizer(SplatData& model) {
        AdamConfig config;
        config.eps = 1e-15;
        config.initial_capacity = model.size();
        config.param_lrs = {{"means", 1.6e-6}, {"scaling", 5e-3}, {"rotation", 1e-3}, {"opacity", 5e-2}, {"sh0", 2.5e-3}, {"shN", 2.5e-3 / 20}};
        auto optimizer = std::make_unique<AdamOptimizer>(model, config);
        optimizer->allocate_gradients(model.size());
        for (const auto type : {ParamType::Means, ParamType::Sh0, ParamType::ShN,
                                ParamType::Scaling, ParamType::Rotation, ParamType::Opacity})
            (void)optimizer->get_grad(type);
        apply_frozen_ranges_to_optimizer(model, *optimizer, 0.f);
        return optimizer;
    }

    void expect_finite_and_frozen(const SplatData& model, const std::array<float, 3>& original_frozen_mean) {
        for (const auto* tensor : {&model.means(), &model.sh0(), &model.scaling_raw(), &model.rotation_raw(), &model.opacity_raw()})
            for (const float value : tensor->to_vector())
                EXPECT_TRUE(std::isfinite(value));
        for (const float value : model.shN_canonical().to_vector())
            EXPECT_TRUE(std::isfinite(value));
        ASSERT_EQ(model.frozen_ranges().size(), 1);
        ASSERT_EQ(model.frozen_ranges().front().count, 1);
        const size_t frozen = model.frozen_ranges().front().start;
        ASSERT_LT(frozen, model.size());
        const auto means = model.means().to_vector();
        for (size_t axis = 0; axis < 3; ++axis)
            EXPECT_EQ(means[frozen * 3 + axis], original_frozen_mean[axis]);
        EXPECT_EQ(model.opacity_raw().to_vector()[frozen], -0.4f);
    }

    class POPSpaPipeline : public ::testing::TestWithParam<bool> {};

    void run_pipeline(bool fast, uint32_t steps, bool zero_photometric_gradient = false) {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
            GTEST_SKIP() << "CUDA device required";
        // Standalone fixture owns its model; mirror the trainer's exclusive mutation scope.
        LiveModelMutationGuard mutation("POPSpa pipeline test");
        auto model = pipeline_model();
        const auto original_means = model.means().to_vector();
        const std::array<float, 3> frozen_mean{original_means[21], original_means[22], original_means[23]};
        auto params = param::OptimizationParameters::mrnf_defaults();
        params.strategy = "mcmc";
        params.max_cap = kInitial;
        auto created = StrategyFactory::instance().create(params.strategy, model);
        ASSERT_TRUE(created.has_value()) << created.error();
        auto strategy = std::move(*created);
        strategy->initialize(params);
        auto first_camera = pipeline_camera(0), second_camera = pipeline_camera(1);
        const std::array<Camera*, 2> cameras{first_camera.get(), second_camera.get()};
        auto background = Tensor::from_vector({0.12f, 0.18f, 0.24f}, {3}, Device::CUDA);
        auto image_background = Tensor::full({3, kHeight, kWidth}, 0.08f, Device::CUDA);
        POPSpaController controller;
        const POPSpaController::Config config{
            .target_count = kFinal,
            .first_prune_count = kIntermediate,
            .sparsify_steps = steps,
            .refine_steps = steps,
            .rho = 5e-4f,
            .projection_interval = 1,
            .erank_weight = 0.01f,
            .thin_scale_weight = 1.f,
            .erank_epsilon = 1e-7f,
            .camera_fingerprint = 12345};
        ASSERT_TRUE(controller.initialize(config, model.opacity_raw(), cameras.size(), {},
                                          make_frozen_mask(model, model.size(), Device::CUDA)));
        std::unique_ptr<AdamOptimizer> optimizer;
        std::array<int, 2> scored_views{}, optimized_steps{};
        int prunes = 0, resets = 0;
        bool trainable_parameters_changed = false;
        for (int guard = 0; controller.phase() != POPSpaPhase::Complete && guard < 32; ++guard) {
            const auto phase = controller.phase();
            if (phase == POPSpaPhase::FirstScore || phase == POPSpaPhase::SecondScore) {
                const size_t view = controller.score_view();
                const auto bg_image = view == 1 ? image_background : Tensor{};
                if (fast) {
                    auto rendered = fast_rasterize_forward(*cameras.at(view), model, background, 0, 0, 0, 0, false, bg_image);
                    ASSERT_TRUE(rendered.has_value());
                    ASSERT_TRUE(fast_accumulate_pop_scores(rendered->second, controller.scores()));
                } else {
                    auto rendered = gsplat_rasterize_forward(*cameras.at(view), model, background, 0, 0, 0, 0,
                                                             1.f, false, GsplatRenderMode::RGB, true, bg_image);
                    ASSERT_TRUE(rendered.has_value()) << rendered.error();
                    const auto scored = gsplat_accumulate_pop_scores(rendered->second, controller.scores());
                    GlobalArenaManager::instance().get_arena().end_frame(rendered->second.frame_id, rendered->second.stream);
                    ASSERT_TRUE(scored);
                }
                const auto host = controller.scores().cpu();
                std::vector<double> scores(model.size());
                std::memcpy(scores.data(), host.data_ptr(), scores.size() * sizeof(double));
                for (const double score : scores) {
                    EXPECT_TRUE(std::isfinite(score));
                    EXPECT_GE(score, 0.0);
                }
                EXPECT_GT(std::accumulate(scores.begin(), scores.end(), 0.0), 0.0);
                ++scored_views[phase == POPSpaPhase::FirstScore ? 0 : 1];
                ASSERT_TRUE(controller.finish_score_view());
            } else if (phase == POPSpaPhase::FirstPrune || phase == POPSpaPhase::FinalPrune) {
                if (optimizer)
                    EXPECT_EQ(optimizer->get_step_count(ParamType::Opacity), steps);
                auto keep = controller.prune_indices();
                ASSERT_TRUE(keep.has_value());
                const auto status = compact_gaussians_for_postprocess(*strategy, params, *keep);
                ASSERT_TRUE(status) << (status ? "" : lfs::format_for_developer(status.error()));
                ASSERT_EQ(model.size(), phase == POPSpaPhase::FirstPrune ? kIntermediate : kFinal);
                ASSERT_TRUE(controller.accept_prune(model.opacity_raw(), {}, make_frozen_mask(model, model.size(), Device::CUDA)));
                optimizer = pipeline_optimizer(model);
                ++resets;
                ++prunes;
                for (auto type : {ParamType::Means, ParamType::Sh0, ParamType::ShN, ParamType::Scaling, ParamType::Rotation, ParamType::Opacity}) {
                    EXPECT_EQ(optimizer->get_step_count(type), 0);
                    EXPECT_EQ(strategy->get_optimizer().get_step_count(type), 0);
                }
                expect_finite_and_frozen(model, frozen_mean);
            } else {
                ASSERT_TRUE(optimizer);
                const size_t view = controller.phase_step() % cameras.size();
                // Post-training refinement updates resident SH even for short base runs.
                const int optimizer_iteration = 1 + static_cast<int>(controller.phase_step());
                const auto before = model.opacity_raw().to_vector();
                optimizer->zero_grad(optimizer_iteration);
                // Exact derivative of mean squared RGB error against a constant target.
                // An empty training view has zero photo gradients, but must still
                // commit the shape/ADMM and Adam step to keep resume counts exact.
                if (!zero_photometric_gradient && fast) {
                    auto rendered = fast_rasterize_forward(*cameras[view], model, background);
                    ASSERT_TRUE(rendered.has_value());
                    auto gradient = (rendered->second.image - 0.22f) * (2.f / (3 * kWidth * kHeight));
                    fast_rasterize_backward(rendered->second, gradient, model, *optimizer, {}, {}, DensificationType::None,
                                            optimizer_iteration, {}, {}, {}, true);
                } else if (!zero_photometric_gradient) {
                    auto rendered = gsplat_rasterize_forward(*cameras[view], model, background, 0, 0, 0, 0,
                                                             1.f, false, GsplatRenderMode::RGB, true);
                    ASSERT_TRUE(rendered.has_value()) << rendered.error();
                    auto gradient = (rendered->first.image - 0.22f) * (2.f / (3 * kWidth * kHeight));
                    auto alpha_gradient = Tensor::zeros({1, kHeight, kWidth}, Device::CUDA);
                    gsplat_rasterize_backward(rendered->second, gradient, alpha_gradient, model, *optimizer);
                }
                auto loss = controller.regularization(model.opacity_raw(), model.scaling_raw(),
                                                      optimizer->get_grad(ParamType::Opacity), optimizer->get_grad(ParamType::Scaling));
                ASSERT_TRUE(loss.has_value());
                EXPECT_TRUE(std::isfinite(loss->item<float>()));
                ASSERT_TRUE(controller.after_backward(model.opacity_raw()));
                if (phase == POPSpaPhase::Sparsify) {
                    size_t support = 0;
                    for (const float proxy : controller.proxy().to_vector()) {
                        EXPECT_TRUE(std::isfinite(proxy));
                        support += proxy != 0.f;
                    }
                    EXPECT_EQ(support, kFinal);
                }
                optimizer->step(optimizer_iteration, false);
                ++optimized_steps[phase == POPSpaPhase::Sparsify ? 0 : 1];
                EXPECT_EQ(optimizer->get_step_count(ParamType::Opacity), controller.phase_step() + 1);
                EXPECT_EQ(optimizer->get_step_count(ParamType::ShN), controller.phase_step() + 1);
                const auto after = model.opacity_raw().to_vector();
                for (size_t i = 0; i < after.size(); ++i)
                    trainable_parameters_changed |= before[i] != after[i];
                expect_finite_and_frozen(model, frozen_mean);
                ASSERT_TRUE(controller.finish_optimization_step());
            }
        }
        EXPECT_EQ(controller.phase(), POPSpaPhase::Complete);
        EXPECT_EQ(model.size(), kFinal);
        EXPECT_EQ(prunes, 2);
        EXPECT_EQ(resets, 2);
        EXPECT_EQ(scored_views, (std::array<int, 2>{2, 2}));
        EXPECT_EQ(optimized_steps, (std::array<int, 2>{int(steps), int(steps)}));
        EXPECT_EQ(trainable_parameters_changed, steps != 0);
        ASSERT_TRUE(optimizer);
        EXPECT_EQ(optimizer->get_step_count(ParamType::Opacity), steps);
        expect_finite_and_frozen(model, frozen_mean);
        LFS_CUDA_CHECK(cudaDeviceSynchronize());
    }

    TEST_P(POPSpaPipeline, ScoresPrunesSparsifiesAndRecovers) { run_pipeline(GetParam(), 2); }
    TEST_P(POPSpaPipeline, ZeroPhotoGradientsStillCompleteEveryOptimizationStep) { run_pipeline(GetParam(), 2, true); }
    TEST_P(POPSpaPipeline, ZeroLengthOptimizationStillRunsBothScoreAndPrunePasses) { run_pipeline(GetParam(), 0); }
    INSTANTIATE_TEST_SUITE_P(Renderers, POPSpaPipeline, ::testing::Values(false, true),
                             [](const auto& info) { return info.param ? "FastGS" : "Gsplat"; });
} // namespace

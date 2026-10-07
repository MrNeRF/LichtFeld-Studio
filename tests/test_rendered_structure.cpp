/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/camera.hpp"
#include "core/image_io.hpp"
#include "core/parameters.hpp"
#include "core/scene.hpp"
#include "cuda_backend_test.hpp"
#include "fast_raster_test_helpers.hpp"
#include "io/argument_parser.hpp"
#include "io/project_document.hpp"
#include "lfs/training/ops/registry.hpp"
#include "licht_test_support.hpp"
#include "training/kernels/mrnf_kernels.hpp"
#include "training/kernels/thin_structure.hpp"
#include "training/strategies/mrnf.hpp"
#include "training/trainer.hpp"
#include "training/training_setup.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <filesystem>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <sstream>
#include <vector>

using namespace lfs::core;
using namespace lfs::training;

namespace {
    bool image_available() {
        return std::getenv("LFS_THIN_STRUCTURE_TEST_IMAGE") && std::getenv("LFS_THIN_STRUCTURE_TEST_OUTPUT");
    }

    Tensor crop() {
        auto [data, width, height, channels] = load_image(std::getenv("LFS_THIN_STRUCTURE_TEST_IMAGE"));
        const std::unique_ptr<void, decltype(&free_image)> owner(data, free_image);
        if (!data || width < 712 || height < 32 || channels != 3)
            throw std::runtime_error("A real RGB crop is required");
        std::vector<float> pixels(3 * 32 * 32);
        for (int c = 0; c < 3; ++c)
            for (int y = 0; y < 32; ++y)
                for (int x = 0; x < 32; ++x)
                    pixels[c * 1024 + y * 32 + x] = data[(y * width + 680 + x) * 3 + c] / 255.0f;
        return Tensor::from_vector(pixels, {3, 32, 32}, Device::GPU);
    }

    void exact(const Tensor& actual, const Tensor& expected) {
        const auto a = actual.cpu().to_vector(), b = expected.cpu().to_vector();
        ASSERT_EQ(a.size(), b.size());
        EXPECT_EQ(std::memcmp(a.data(), b.data(), a.size() * sizeof(float)), 0);
    }

    void load_model(Scene& scene) {
        using namespace lfs::io::project;
        using namespace lfs::test::licht;
        auto document = require_result(ProjectDocument::open(std::getenv("LFS_SPLIT_TEST_MODEL")));
        static_cast<void>(require_result(document.hydrate(scene)));
    }
} // namespace

TEST(RenderedStructure, ZeroGainAndRealMapNormalization) {
    if (!image_available())
        GTEST_SKIP() << "set real-image and output paths";
    const auto image = crop();
    const auto original = image.slice(0, 0, 1).squeeze(0).contiguous();
    auto error = original.clone();
    kernels::structure_densification_weight(error, {}, 0);
    exact(error, original);
    kernels::RidgeWorkspace workspace;
    auto structure = Tensor::empty({32, 32}, Device::GPU);
    kernels::ridge_structure_map(image, structure, workspace);
    auto values = original.cpu().to_vector();
    auto map = structure.cpu().to_vector();
    values[17] = 0;
    error = Tensor::from_vector(values, {32, 32}, Device::GPU);
    kernels::structure_densification_weight(error, structure, 1);
    const auto weighted = error.cpu().to_vector();
    double mean = 0;
    for (size_t i = 0; i < values.size(); ++i) {
        const float expected = values[i] * (1 + std::min(std::max(map[i], 0.0f), 4.0f));
        EXPECT_NEAR(weighted[i], expected, 2e-7f);
        mean += expected / values.size();
    }
    const auto normalized = (error / error.mean()).cpu().to_vector();
    for (size_t i = 0; i < values.size(); ++i)
        EXPECT_NEAR(normalized[i], weighted[i] / mean, 2e-6);
    EXPECT_EQ(weighted[17], 0);
    auto zero = Tensor::zeros_like(structure);
    error = original.clone();
    kernels::structure_densification_weight(error, zero, 4);
    exact(error, original);
}

TEST(RenderedStructure, DecayGatesOnlyOpacityAndPreservesFrozenRows) {
    if (!std::getenv("LFS_SPLIT_TEST_MODEL"))
        GTEST_SKIP() << "set real-model path";
    Scene scene;
    load_model(scene);
    const auto* model = scene.getCombinedModel();
    ASSERT_NE(model, nullptr);
    constexpr size_t n = 64;
    ASSERT_GE(model->size(), n);
    const int rows = static_cast<int>(model->size());
    auto opacity = model->opacity_raw().reshape({rows}).slice(0, 0, n).contiguous();
    auto scale = model->scaling_raw().slice(0, 0, n).contiguous();
    auto baseline = opacity.clone(), gated = opacity.clone();
    auto baseline_scale = scale.clone(), gated_scale = scale.clone();
    std::vector<float> support(n);
    std::vector<bool> frozen(n);
    for (size_t i = 0; i < n; ++i) {
        support[i] = i % 3 ? 1 : 0;
        frozen[i] = i % 7 == 0;
    }
    auto counts = Tensor::from_vector(support, {n}, Device::GPU);
    auto frozen_tensor = Tensor::from_vector(frozen, {n}, Device::GPU);
    training_ops(default_gpu_backend()).mrnf->decay(baseline, baseline_scale, frozen_tensor, {.opacity_decay = .01f, .scale_decay = .02f, .train_t = .2f});
    training_ops(default_gpu_backend()).mrnf->decay(gated, gated_scale, frozen_tensor, {.opacity_decay = .01f, .scale_decay = .02f, .train_t = .2f, .rendered_count = counts});
    exact(gated_scale, baseline_scale);
    const auto original = opacity.cpu().to_vector(), all = baseline.cpu().to_vector(), selective = gated.cpu().to_vector();
    for (size_t i = 0; i < n; ++i)
        EXPECT_EQ(selective[i], support[i] > 0 && !frozen[i] ? all[i] : original[i]);
    gated = opacity.clone();
    gated_scale = scale.clone();
    counts = Tensor::ones({n}, Device::GPU);
    training_ops(default_gpu_backend()).mrnf->decay(gated, gated_scale, frozen_tensor, {.opacity_decay = .01f, .scale_decay = .02f, .train_t = .2f, .rendered_count = counts});
    exact(gated, baseline);
    exact(gated_scale, baseline_scale);
}

TEST(RenderedStructure, SupportedProjectResumeKeepsSupportAndFiniteLoss) {
    const auto* on = std::getenv("LFS_RENDERED_TEST_PROJECT_ON");
    const auto* off = std::getenv("LFS_RENDERED_TEST_PROJECT_OFF");
    if (!on || !off)
        GTEST_SKIP() << "set saved real-training project paths";
    using namespace lfs::io::project;
    using namespace lfs::test::licht;
    for (const auto* path : {off, on}) {
        SCOPED_TRACE(path);
        Scene scene;
        auto document = require_result(ProjectDocument::open(path));
        const auto hydration = require_result(document.hydrate(scene));
        ASSERT_TRUE(hydration.checkpoint_header);
        ASSERT_TRUE(hydration.checkpoint_uuid);
        ASSERT_TRUE(hydration.checkpoint_params);
        const auto iteration = hydration.checkpoint_header->iteration;
        ASSERT_GE(iteration, 700);
        auto params = *hydration.checkpoint_params;
        ASSERT_EQ(params.optimization.opacity_decay_rendered_only, path == on);
        params.optimization.iterations = iteration + 25;
        params.cli_iterations_set = true;
        params.optimization.enable_eval = false;
        params.save_project_at_iteration.reset();
        params.save_project_path.clear();
        const auto installed = installTrainerFromProjectCheckpoint(scene, document, *hydration.checkpoint_uuid,
                                                                   params, path, iteration);
        ASSERT_TRUE(installed) << installed.error();
        auto& trainer = *installed->trainer;
        const auto support = trainer.get_strategy().rendered_support_counts();
        if (params.optimization.opacity_decay_rendered_only) {
            ASSERT_EQ(support.numel(), trainer.get_strategy().get_model().size());
            const auto values = support.cpu().to_vector();
            EXPECT_TRUE(std::all_of(values.begin(), values.end(), [](float value) { return value > 0 && std::isfinite(value); }));
        } else {
            EXPECT_FALSE(support.is_valid());
        }
        const auto trained = trainer.train();
        ASSERT_TRUE(trained) << lfs::format_for_developer(trained.error());
        EXPECT_EQ(trainer.get_current_iteration(), iteration + 25);
        RecordProperty(path == on ? "enabled_resume_iteration" : "disabled_resume_iteration",
                       trainer.get_current_iteration());
        EXPECT_TRUE(std::isfinite(trainer.get_current_loss()));
    }
}

class RenderedStructureNative : public lfs::test::CudaBackendTest {};
TEST_F(RenderedStructureNative, NativeRasterizedSupportAccumulatesWithoutImageGradient) {
    if (!std::getenv("LFS_SPLIT_TEST_MODEL"))
        GTEST_SKIP() << "set real-model path";
    Scene scene;
    load_model(scene);
    const auto cameras = scene.getAllCameras();
    ASSERT_FALSE(cameras.empty());
    auto model = scene.getCombinedModel()->clone();
    auto camera = cameras.front();
    camera->load_image_size();
    auto background = Tensor::zeros({3}, Device::GPU);
    auto counts = Tensor::zeros({model.size()}, Device::GPU);
    AdamConfig config{.lr = 0, .beta1 = .9, .beta2 = .999, .eps = 1e-15};
    AdamOptimizer optimizer(model, config);
    optimizer.allocate_gradients();
    FastGSFusedExtraGradients extra;
    extra.rendered_count = counts.ptr<float>();
    std::vector<float> expected(model.size(), 0);
    for (int step = 1; step <= 2; ++step) {
        optimizer.zero_grad(step);
        auto forward = fast_rasterize_forward(*camera, model, background, 680, 0, 32, 32, false);
        ASSERT_TRUE(forward) << lfs::format_for_developer(forward.error());
        std::vector<unsigned> work(model.size());
        ASSERT_EQ(cudaMemcpyAsync(work.data(), forward->second.forward_ctx.primitive_work_indices, work.size() * sizeof(unsigned), cudaMemcpyDeviceToHost, forward->second.forward_ctx.completion_stream), cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(forward->second.forward_ctx.completion_stream), cudaSuccess);
        for (size_t i = 0; i < work.size(); ++i)
            expected[i] += work[i] != 0xffffffffu ? 1.f : 0.f;
        auto gradient = Tensor::zeros_like(forward->first.image);
        fast_rasterize_backward(forward->second, gradient, model, optimizer, {}, {}, DensificationType::None, step, extra);
        EXPECT_EQ(counts.cpu().to_vector(), expected);
    }
    EXPECT_GT(counts.sum().item<float>(), 0);
}

TEST(RenderedStructure, ParameterJsonCliBoundsAndProjectRoundTrip) {
    if (!image_available())
        GTEST_SKIP() << "set real-image and output paths";
    param::OptimizationParameters generic;
    EXPECT_FALSE(generic.opacity_decay_rendered_only);
    EXPECT_EQ(generic.densify_structure_weight, 0);
    for (const auto* strategy : {"mrnf", "mcmc", "igs+"}) {
        auto params = param::OptimizationParameters::defaults_for_strategy(strategy);
        const bool mrnf = param::is_mrnf_strategy(strategy);
        EXPECT_EQ(params.opacity_decay_rendered_only, mrnf);
        EXPECT_EQ(params.densify_structure_weight, mrnf ? 1 : 0);
        for (float value : {0.f, 1.f, 4.f}) {
            params.densify_structure_weight = value;
            params.opacity_decay_rendered_only = !mrnf;
            auto restored = param::OptimizationParameters::from_json(params.to_json());
            EXPECT_EQ(restored.opacity_decay_rendered_only, !mrnf);
            EXPECT_EQ(restored.densify_structure_weight, value);
            EXPECT_TRUE(params.validate().empty());
        }
        for (float value : {-.1f, 4.1f, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
            params.densify_structure_weight = value;
            EXPECT_FALSE(params.validate().empty());
        }
    }
    const auto* output = std::getenv("LFS_THIN_STRUCTURE_TEST_OUTPUT");
    const auto data = std::filesystem::path(std::getenv("LFS_THIN_STRUCTURE_TEST_IMAGE")).parent_path().parent_path().string();
    for (const auto* flag : {"--opacity-decay-rendered-only", "--no-opacity-decay-rendered-only"}) {
        const char* argv[] = {"LichtFeld-Studio", "--headless", "--data-path", data.c_str(), "--output-path", output, "--log-file", "/dev/null", "--strategy", "mrnf", "--densify-structure-weight", "0", flag};
        const auto parsed = lfs::io::args::parse_args_and_params(static_cast<int>(std::size(argv)), argv);
        ASSERT_TRUE(parsed) << parsed.error();
        param::TrainingParameters target;
        target.optimization.densify_structure_weight = 3;
        target.optimization.opacity_decay_rendered_only = std::string_view(flag) != "--opacity-decay-rendered-only";
        param::apply_explicit_training_overrides(target, (*parsed)->overrides);
        EXPECT_EQ(target.optimization.densify_structure_weight, 0);
        EXPECT_EQ(target.optimization.opacity_decay_rendered_only, std::string_view(flag) == "--opacity-decay-rendered-only");
    }
    using namespace lfs::io::project;
    using namespace lfs::test::licht;
    auto document = require_result(ProjectDocument::create(fixed_uuid(4247), 1'700'000'000'000'000'000));
    auto params = require_result(document.parameters().snapshot());
    params.mrnf_current.opacity_decay_rendered_only = params.mrnf_session.opacity_decay_rendered_only = false;
    params.mrnf_current.densify_structure_weight = params.mrnf_session.densify_structure_weight = 2.3f;
    require_status(document.edit_parameters().set_snapshot(params));
    ProjectDocumentSaveOptions options;
    options.file_uuid = fixed_uuid(4248);
    options.disk_reserve_bytes = 0;
    const auto path = std::filesystem::path(output) / (generate_uuid_v4().to_string() + ".licht");
    std::filesystem::create_directories(path.parent_path());
    static_cast<void>(require_result(document.save(path, options)));
    auto restored = require_result(ProjectDocument::open(path));
    auto resumed = require_result(restored.parameters().snapshot()).active_optimization();
    EXPECT_FALSE(resumed.opacity_decay_rendered_only);
    EXPECT_FLOAT_EQ(resumed.densify_structure_weight, 2.3f);
    std::filesystem::remove(path);
}

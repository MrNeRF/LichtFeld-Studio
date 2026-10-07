/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/image_io.hpp"
#include "core/parameters.hpp"
#include "core/scene.hpp"
#include "io/argument_parser.hpp"
#include "io/project_document.hpp"
#include "licht_test_support.hpp"
#include "training/kernels/gradient_residual.hpp"
#include "training/trainer.hpp"
#include "training/training_setup.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <vector>

using namespace lfs::core;
using namespace lfs::training::kernels;

namespace {
    constexpr int SIDE = 32;
    constexpr int PIXELS = SIDE * SIDE;

    bool has_real_data() {
        return std::getenv("LFS_THIN_STRUCTURE_TEST_IMAGE") && std::getenv("LFS_THIN_STRUCTURE_TEST_OUTPUT");
    }

    Tensor real_crop(int offset = 680) {
        auto [data, width, height, channels] = load_image(std::getenv("LFS_THIN_STRUCTURE_TEST_IMAGE"));
        const std::unique_ptr<void, decltype(&free_image)> owner(data, free_image);
        if (!data || width < offset + SIDE || height < SIDE || channels != 3)
            throw std::runtime_error("A real RGB crop is required");
        std::vector<float> values(3 * PIXELS);
        for (int c = 0; c < 3; ++c)
            for (int y = 0; y < SIDE; ++y)
                for (int x = 0; x < SIDE; ++x)
                    values[c * PIXELS + y * SIDE + x] = data[(y * width + offset + x) * 3 + c] / 255.0f;
        return Tensor::from_vector(values, {3, SIDE, SIDE}, Device::GPU);
    }

    void exact(const Tensor& actual, const Tensor& expected) {
        ASSERT_EQ(actual.is_valid(), expected.is_valid());
        if (actual.is_valid()) {
            const auto values = actual.cpu().to_vector();
            const auto reference = expected.cpu().to_vector();
            ASSERT_EQ(values.size(), reference.size());
            EXPECT_EQ(std::memcmp(values.data(), reference.data(), values.size() * sizeof(float)), 0);
        }
    }

    double cpu_loss(const std::vector<float>& image, const std::vector<float>& target,
                    const std::vector<float>& mask, float weight) {
        double sum = 0, denominator = 0;
        for (int y = 1; y < SIDE - 1; ++y)
            for (int x = 1; x < SIDE - 1; ++x) {
                double gx = 0, gy = 0;
                bool valid = true;
                for (int dy = -1; dy <= 1; ++dy)
                    for (int dx = -1; dx <= 1; ++dx) {
                        const int i = (y + dy) * SIDE + x + dx;
                        valid = valid && (mask.empty() || mask[i] > 0);
                        const double r = .2126 * (image[i] - target[i]) +
                                         .7152 * (image[PIXELS + i] - target[PIXELS + i]) +
                                         .0722 * (image[2 * PIXELS + i] - target[2 * PIXELS + i]);
                        gx += dx * (dy == 0 ? 2 : 1) * .125 * r;
                        gy += dy * (dx == 0 ? 2 : 1) * .125 * r;
                    }
                if (valid) {
                    const double confidence = mask.empty() ? 1 : mask[y * SIDE + x];
                    sum += confidence * (std::hypot(gx, double(GRADIENT_LOSS_EPSILON)) +
                                         std::hypot(gy, double(GRADIENT_LOSS_EPSILON)) - 2 * GRADIENT_LOSS_EPSILON);
                    denominator += 2 * confidence;
                }
            }
        return denominator > 0 ? weight * sum / denominator : 0;
    }

    float value(const Tensor& image, const Tensor& target, const Tensor& mask,
                GradientResidualWorkspace& workspace, float weight = 1.8f) {
        auto gradient = Tensor::zeros_like(image);
        return gradient_residual_loss_gradient(image, target, mask, gradient, weight, workspace).item<float>();
    }

    void real_scene(Scene& scene) {
        param::TrainingParameters params;
        params.dataset.data_path = std::filesystem::path(std::getenv("LFS_THIN_STRUCTURE_TEST_IMAGE")).parent_path().parent_path();
        params.dataset.output_path = std::getenv("LFS_THIN_STRUCTURE_TEST_OUTPUT");
        const auto loaded = lfs::training::loadTrainingDataIntoScene(params, scene);
        if (!loaded)
            throw std::runtime_error(loaded.error());
    }
} // namespace

namespace lfs::training {
    struct TrainerGradientResidualTestAccess {
        static bool allocated(const Trainer& trainer) {
            return trainer.gradient_residual_workspace_.partial.is_valid() ||
                   trainer.gradient_residual_workspace_.totals.is_valid() ||
                   trainer.gradient_residual_workspace_.loss.is_valid();
        }
        struct FinishedLoss {
            Tensor loss;
            Tensor grad_corrected;
            Tensor grad_raw;
        };
        // Adds the raw-render terms the training step applies after the appearance backward.
        template <typename Result>
        static std::expected<FinishedLoss, std::string> finish(Trainer& trainer, const Result& result,
                                                               const param::OptimizationParameters& params, bool decoupled) {
            if (!result)
                return std::unexpected(result.error());
            FinishedLoss finished{.loss = result->loss, .grad_corrected = result->grad_corrected};
            if (!decoupled && !result->grad_raw.is_valid() && !result->raw_residual)
                return finished;
            finished.grad_raw = result->grad_raw.is_valid() ? result->grad_raw.clone() : Tensor::zeros_like(result->grad_corrected);
            if (trainer.training_ops_->photometric->add_raw_gradient)
                trainer.training_ops_->photometric->add_raw_gradient(trainer.photo_saved_, finished.grad_raw);
            if (result->raw_residual)
                finished.loss = finished.loss + gradient_residual_loss_gradient(
                                                    result->raw_residual->raw, result->raw_residual->target,
                                                    result->raw_residual->pixel_weight, finished.grad_raw,
                                                    params.gradient_loss_weight, trainer.gradient_residual_workspace_);
            return finished;
        }
        static auto loss(Trainer& trainer, const Tensor& image, const Tensor& target,
                         const param::OptimizationParameters& params, const Tensor& raw, int iteration) {
            trainer.training_ops_ = &training_ops(core::default_gpu_backend());
            if (!trainer.photo_saved_.backend)
                trainer.photo_saved_.backend = trainer.training_ops_->photometric->create();
            return finish(trainer, trainer.compute_photometric_loss_with_gradient(image, target, params, raw, iteration),
                          params, raw.is_valid() && params.lambda_dssim > 0.f);
        }
        static auto masked(Trainer& trainer, const Tensor& image, const Tensor& target,
                           const Tensor& mask, const param::OptimizationParameters& params,
                           const Tensor& raw, const Tensor& ridge, int iteration) {
            trainer.training_ops_ = &training_ops(core::default_gpu_backend());
            if (!trainer.photo_saved_.backend)
                trainer.photo_saved_.backend = trainer.training_ops_->photometric->create();
            return finish(trainer,
                          trainer.compute_photometric_loss_with_mask(image, target, mask, {}, {}, params, raw, ridge,
                                                                     iteration),
                          params, raw.is_valid() && params.lambda_dssim > 0.f);
        }
    };
} // namespace lfs::training

TEST(GradientResidual, ZeroWeightIdenticalInputAndEmptyDomains) {
    if (!has_real_data())
        GTEST_SKIP() << "set real-image and output paths";
    auto image = real_crop(), target = real_crop(681), gradient = image.clone();
    const auto original = gradient.clone();
    GradientResidualWorkspace workspace;
    EXPECT_FALSE(gradient_residual_loss_gradient(image, target, {}, gradient, 0, workspace).is_valid());
    exact(gradient, original);
    EXPECT_FALSE(workspace.partial.is_valid());
    EXPECT_FALSE(workspace.totals.is_valid());
    EXPECT_FALSE(workspace.loss.is_valid());
    for (const Tensor& mask : {Tensor{}, Tensor::zeros({SIDE, SIDE}, Device::GPU)}) {
        auto zero = Tensor::zeros_like(image);
        auto loss = gradient_residual_loss_gradient(image, image, mask, zero, 1.8f, workspace);
        EXPECT_EQ(loss.item<float>(), 0);
        EXPECT_EQ(zero.abs().max().item<float>(), 0);
    }
    auto tiny = image.slice(1, 0, 1).slice(2, 0, 2).contiguous();
    auto zero = Tensor::zeros_like(tiny);
    EXPECT_EQ(gradient_residual_loss_gradient(tiny, tiny, {}, zero, 1.8f, workspace).item<float>(), 0);
    EXPECT_EQ(zero.abs().max().item<float>(), 0);
}

TEST(GradientResidual, RealCropMatchesSignedCpuReferenceAndByteTargets) {
    if (!has_real_data())
        GTEST_SKIP() << "set real-image and output paths";
    const auto image = real_crop(), target = real_crop(681);
    auto mask = Tensor::full({SIDE, SIDE}, .4f, Device::GPU);
    mask.slice(0, 10, 13).slice(1, 11, 14).fill_(0);
    GradientResidualWorkspace workspace;
    const auto weights = mask.cpu().to_vector();
    const double reference = cpu_loss(image.cpu().to_vector(), target.cpu().to_vector(), weights, 1.8f);
    EXPECT_NEAR(value(image, target, mask, workspace), reference, 2e-7);
    const auto byte_target = (target * 255.0f).to(DataType::UInt8);
    const auto rounded_target = byte_target.to(DataType::Float32) / 255.0f;
    EXPECT_NEAR(value(image, byte_target, mask, workspace), value(image, rounded_target, mask, workspace), 2e-7);
    const auto opposite = target * 2.0f - image;
    EXPECT_GT(value(opposite, image, {}, workspace), value(target, image, {}, workspace));
    const auto binary = mask > 0.0f;
    EXPECT_NEAR(value(image, target, binary, workspace), value(image, target, binary.to(DataType::UInt8), workspace), 1e-7);
}

TEST(GradientResidual, FiniteDifferencesWithSoftMaskAndBoundaryGather) {
    if (!has_real_data())
        GTEST_SKIP() << "set real-image and output paths";
    const auto image = real_crop(), target = real_crop(681);
    const auto pixels = image.cpu().to_vector();
    auto mask = Tensor::full({SIDE, SIDE}, .7f, Device::GPU);
    mask.slice(0, 10, 13).slice(1, 11, 14).fill_(0);
    GradientResidualWorkspace workspace;
    auto gradient = Tensor::zeros_like(image);
    static_cast<void>(gradient_residual_loss_gradient(image, target, mask, gradient, 1.8f, workspace));
    const auto analytical = gradient.cpu().to_vector();
    constexpr float epsilon = .0005f;
    for (const size_t index : {size_t{0}, size_t{16 * SIDE + 16}, size_t{PIXELS + 14 * SIDE + 17}, size_t{2 * PIXELS + 18 * SIDE + 15}, size_t{11 * SIDE + 12}}) {
        auto plus = pixels, minus = pixels;
        plus[index] += epsilon;
        minus[index] -= epsilon;
        const auto a = Tensor::from_vector(plus, image.shape(), Device::GPU);
        const auto b = Tensor::from_vector(minus, image.shape(), Device::GPU);
        const float numerical = (value(a, target, mask, workspace) - value(b, target, mask, workspace)) / (2 * epsilon);
        EXPECT_NEAR(analytical[index], numerical, 4e-6f + .01f * std::abs(numerical));
    }
    EXPECT_EQ(analytical[11 * SIDE + 12], 0);
}

TEST(GradientResidual, TrainerOffPreStartAndExposureRouting) {
    if (!has_real_data())
        GTEST_SKIP() << "set real-image and output paths";
    using lfs::training::TrainerGradientResidualTestAccess;
    auto image = real_crop(), target = real_crop(681), raw = real_crop(682);
    Scene scene;
    real_scene(scene);
    lfs::training::Trainer trainer(scene);
    for (const float lambda : {0.0f, .22f}) {
        for (const Tensor& raw_input : {Tensor{}, raw}) {
            param::OptimizationParameters params;
            params.lambda_dssim = lambda;
            const auto baseline = TrainerGradientResidualTestAccess::loss(trainer, image, target, params, raw_input, 2000);
            ASSERT_TRUE(baseline);
            const auto base_loss = baseline->loss.clone(), base_corrected = baseline->grad_corrected.clone();
            const auto base_raw = baseline->grad_raw.is_valid() ? baseline->grad_raw.clone() : Tensor{};
            params.gradient_loss_weight = 1.8f;
            const auto before = TrainerGradientResidualTestAccess::loss(trainer, image, target, params, raw_input, 1999);
            ASSERT_TRUE(before);
            exact(before->loss, base_loss);
            exact(before->grad_corrected, base_corrected);
            exact(before->grad_raw, base_raw);
            if (lambda == 0 && !raw_input.is_valid())
                EXPECT_FALSE(TrainerGradientResidualTestAccess::allocated(trainer));
            const auto active = TrainerGradientResidualTestAccess::loss(trainer, image, target, params, raw_input, 2000);
            ASSERT_TRUE(active);
            auto expected = Tensor::zeros_like(image);
            GradientResidualWorkspace workspace;
            const auto term = gradient_residual_loss_gradient(raw_input.is_valid() ? raw_input : image, target, {}, expected, 1.8f, workspace);
            EXPECT_NEAR(active->loss.item<float>(), base_loss.item<float>() + term.item<float>(), 1e-6);
            const auto actual = (raw_input.is_valid() ? active->grad_raw : active->grad_corrected).cpu().to_vector();
            const auto original = (raw_input.is_valid() ? base_raw : base_corrected).is_valid() ? (raw_input.is_valid() ? base_raw : base_corrected).cpu().to_vector() : std::vector<float>(3 * PIXELS, 0);
            const auto derivative = expected.cpu().to_vector();
            for (size_t i = 0; i < actual.size(); ++i)
                EXPECT_NEAR(actual[i], original[i] + derivative[i], 2e-7);
            if (raw_input.is_valid())
                exact(active->grad_corrected, base_corrected);
            else
                EXPECT_FALSE(active->grad_raw.is_valid());
        }
    }
}

TEST(GradientResidual, MaskedTermUsesBaseConfidenceWithRidgeAndAlphaModes) {
    if (!has_real_data())
        GTEST_SKIP() << "set real-image and output paths";
    using lfs::training::TrainerGradientResidualTestAccess;
    const auto image = real_crop(), target = real_crop(681), raw = real_crop(682);
    auto mask = Tensor::full({SIDE, SIDE}, .3f, Device::GPU);
    mask.slice(0, 12, 15).slice(1, 12, 15).fill_(0);
    Scene scene;
    real_scene(scene);
    lfs::training::Trainer trainer(scene);
    for (const auto mode : {param::MaskMode::Segment, param::MaskMode::AlphaConsistent}) {
        for (const Tensor& ridge : {Tensor{}, Tensor::ones({SIDE, SIDE}, Device::GPU)}) {
            param::OptimizationParameters params;
            params.mask_mode = mode;
            params.thin_structure_weight = .5f;
            const auto baseline = TrainerGradientResidualTestAccess::masked(trainer, image, target, mask, params, raw, ridge, 2000);
            ASSERT_TRUE(baseline);
            const auto corrected = baseline->grad_corrected.clone(), base_raw = baseline->grad_raw.clone();
            const float old_loss = baseline->loss.item<float>();
            params.gradient_loss_weight = 1.8f;
            const auto active = TrainerGradientResidualTestAccess::masked(trainer, image, target, mask, params, raw, ridge, 2000);
            ASSERT_TRUE(active);
            exact(active->grad_corrected, corrected);
            auto expected = Tensor::zeros_like(raw);
            GradientResidualWorkspace workspace;
            const auto term = gradient_residual_loss_gradient(raw, target, mode == param::MaskMode::Segment ? mask : Tensor{}, expected, 1.8f, workspace);
            EXPECT_NEAR(active->loss.item<float>(), old_loss + term.item<float>(), 1e-6);
            const auto actual = active->grad_raw.cpu().to_vector(), base = base_raw.cpu().to_vector(), extra = expected.cpu().to_vector();
            for (size_t i = 0; i < actual.size(); ++i)
                EXPECT_NEAR(actual[i], base[i] + extra[i], 2e-7);
        }
    }
}

TEST(GradientResidual, ParameterJsonCliBoundsAndProjectRoundTrip) {
    if (!has_real_data())
        GTEST_SKIP() << "set real-image and output paths";
    for (const auto* strategy : {"mrnf", "mcmc", "igs+"}) {
        auto params = param::OptimizationParameters::defaults_for_strategy(strategy);
        EXPECT_FLOAT_EQ(params.gradient_loss_weight, param::is_mrnf_strategy(strategy) ? 1.8f : 0);
        for (const float value : {0.0f, 1.8f, 8.0f}) {
            params.gradient_loss_weight = value;
            EXPECT_FLOAT_EQ(param::OptimizationParameters::from_json(params.to_json()).gradient_loss_weight, value);
            EXPECT_TRUE(params.validate().empty());
        }
        for (const float invalid : {-.1f, 8.1f, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
            params.gradient_loss_weight = invalid;
            EXPECT_FALSE(params.validate().empty());
        }
    }
    const auto* output = std::getenv("LFS_THIN_STRUCTURE_TEST_OUTPUT");
    const auto data = std::filesystem::path(std::getenv("LFS_THIN_STRUCTURE_TEST_IMAGE")).parent_path().parent_path().string();
    for (const auto* number : {"0", "1.8", "8"}) {
        const char* argv[] = {"LichtFeld-Studio", "--headless", "--data-path", data.c_str(), "--output-path", output, "--log-file", "/dev/null", "--gradient-loss-weight", number};
        const auto parsed = lfs::io::args::parse_args_and_params(static_cast<int>(std::size(argv)), argv);
        ASSERT_TRUE(parsed) << parsed.error();
        param::TrainingParameters target;
        target.optimization.gradient_loss_weight = 2;
        param::apply_explicit_training_overrides(target, (*parsed)->overrides);
        EXPECT_FLOAT_EQ(target.optimization.gradient_loss_weight, std::stof(number));
    }
    using namespace lfs::io::project;
    using namespace lfs::test::licht;
    auto document = require_result(ProjectDocument::create(fixed_uuid(4237), 1'700'000'000'000'000'000));
    auto params = require_result(document.parameters().snapshot());
    params.mrnf_current.gradient_loss_weight = 2.3f;
    params.mrnf_session.gradient_loss_weight = 2.3f;
    require_status(document.edit_parameters().set_snapshot(params));
    ProjectDocumentSaveOptions options;
    options.file_uuid = fixed_uuid(4238);
    options.disk_reserve_bytes = 0;
    const auto path = std::filesystem::path(output) / (generate_uuid_v4().to_string() + ".licht");
    std::filesystem::create_directories(path.parent_path());
    static_cast<void>(require_result(document.save(path, options)));
    const auto restored = require_result(ProjectDocument::open(path));
    EXPECT_FLOAT_EQ(require_result(restored.parameters().snapshot()).active_optimization().gradient_loss_weight, 2.3f);
    std::filesystem::remove(path);
}

/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/argument_parser.hpp"
#include "core/image_io.hpp"
#include "core/parameters.hpp"
#include "core/scene.hpp"
#include "core/tensor.hpp"
#include "io/project_document.hpp"
#include "io/project_operations.hpp"
#include "lfs/kernels/ssim.cuh"
#include "licht_test_support.hpp"
#include "training/kernels/densification_kernels.hpp"
#include "training/kernels/image_kernels.hpp"
#include "training/kernels/thin_structure.hpp"
#include "training/trainer.hpp"
#include "training/training_setup.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <nlohmann/json.hpp>
#include <vector>

namespace {
    using namespace lfs::core;
    using namespace lfs::training::kernels;

    struct RealCrop {
        std::vector<float> pixels;
        Tensor tensor;
    };

    RealCrop read_crop(const int offset_x = 680, const int offset_y = 0, const int side = 32) {
        const char* path = std::getenv("LFS_THIN_STRUCTURE_TEST_IMAGE");
        if (!path)
            throw std::runtime_error("LFS_THIN_STRUCTURE_TEST_IMAGE must point to a real RGB image");
        auto [data, width, height, channels] = load_image(std::filesystem::path(path));
        if (!data || width < offset_x + side || height < offset_y + side || channels != 3)
            throw std::runtime_error("Real RGB image cannot supply the requested crop");
        std::vector<float> pixels(3 * side * side);
        for (int c = 0; c < 3; ++c)
            for (int y = 0; y < side; ++y)
                for (int x = 0; x < side; ++x)
                    pixels[c * side * side + y * side + x] =
                        data[((offset_y + y) * width + offset_x + x) * 3 + c] / 255.0f;
        free_image(data);
        auto tensor = Tensor::from_vector(pixels, TensorShape{3, static_cast<size_t>(side), static_cast<size_t>(side)}, Device::CUDA);
        return {std::move(pixels), std::move(tensor)};
    }

    void expect_close(const Tensor& left, const Tensor& right, const float tolerance) {
        ASSERT_EQ(left.numel(), right.numel());
        const auto a = left.to(Device::CPU).to_vector();
        const auto b = right.to(Device::CPU).to_vector();
        float max_difference = 0.0f;
        for (size_t i = 0; i < a.size(); ++i)
            max_difference = std::max(max_difference, std::abs(a[i] - b[i]));
        EXPECT_LE(max_difference, tolerance);
    }
} // namespace

TEST(ThinStructure, RidgeResponsePrefersRealLineCrop) {
    if (!std::getenv("LFS_THIN_STRUCTURE_TEST_IMAGE") || !std::getenv("LFS_THIN_STRUCTURE_TEST_OUTPUT"))
        GTEST_SKIP() << "set LFS_THIN_STRUCTURE_TEST_IMAGE and LFS_THIN_STRUCTURE_TEST_OUTPUT to run";
    auto line = read_crop(680, 0, 96);
    auto flat = read_crop(960, 288, 96);
    const auto combined = Tensor::cat({line.tensor, flat.tensor}, 2);
    RidgeWorkspace workspace;
    auto ridge = Tensor::empty({96, 192}, Device::CUDA);
    ridge_structure_map(combined, ridge, workspace);
    const auto response = ridge.to(Device::CPU).to_vector();
    double line_sum = 0.0, flat_sum = 0.0;
    for (int y = 10; y < 86; ++y)
        for (int x = 10; x < 86; ++x) {
            line_sum += response[y * 192 + x];
            flat_sum += response[y * 192 + 96 + x];
        }
    EXPECT_GT(line_sum, 1.5 * flat_sum);
}

namespace lfs::training {
    struct TrainerThinStructureTestAccess {
        static auto map(Trainer& trainer, const core::Tensor& image, float weight) {
            return trainer.get_thin_structure_map(0, image, weight);
        }
        static bool allocated(const Trainer& trainer) {
            return trainer.thin_structure_workspace_.luminance.is_valid() ||
                   trainer.thin_structure_weight_buffer_.is_valid() ||
                   !trainer.thin_structure_cache_.empty() || trainer.thin_structure_map_computations_ != 0;
        }
        static auto plain(Trainer& trainer, const core::Tensor& image, const core::Tensor& target,
                          const core::param::OptimizationParameters& params, const core::Tensor& raw) {
            return trainer.compute_photometric_loss_with_gradient(image, target, params, raw);
        }
        static auto weighted(Trainer& trainer, const core::Tensor& image, const core::Tensor& target,
                             const core::param::OptimizationParameters& params, const core::Tensor& raw,
                             const core::Tensor& structure) {
            return trainer.compute_photometric_loss_with_mask(image, target, {}, {}, {}, params, raw, structure);
        }
    };
} // namespace lfs::training

namespace {
    void load_real_scene(lfs::core::Scene& scene) {
        const auto* path = std::getenv("LFS_THIN_STRUCTURE_TEST_IMAGE");
        const auto* output = std::getenv("LFS_THIN_STRUCTURE_TEST_OUTPUT");
        if (!path || !output)
            throw std::runtime_error("Real image and output paths are required");
        param::TrainingParameters params;
        params.dataset.data_path = std::filesystem::path(path).parent_path().parent_path();
        params.dataset.output_path = output;
        const auto loaded = lfs::training::loadTrainingDataIntoScene(params, scene);
        if (!loaded)
            throw std::runtime_error(loaded.error());
    }

    float weighted_value(const Tensor& prediction, const Tensor& target, const Tensor& weight,
                         MaskedFusedL1SSIMWorkspace& workspace, float denominator) {
        auto [loss, ctx] = masked_fused_l1_ssim_forward(prediction, target, weight, 0.22f, workspace, denominator);
        return loss.item<float>() * ctx.mask_sum_value / denominator;
    }
} // namespace

TEST(ThinStructure, ZeroWeightBypassesMapAndPreservesLossAndGradient) {
    if (!std::getenv("LFS_THIN_STRUCTURE_TEST_IMAGE") || !std::getenv("LFS_THIN_STRUCTURE_TEST_OUTPUT"))
        GTEST_SKIP() << "set LFS_THIN_STRUCTURE_TEST_IMAGE and LFS_THIN_STRUCTURE_TEST_OUTPUT to run";
    using lfs::training::TrainerThinStructureTestAccess;
    auto image = read_crop(), target = read_crop(681), raw = read_crop(682);
    Scene scene;
    load_real_scene(scene);
    lfs::training::Trainer trainer(scene);
    param::OptimizationParameters params;
    for (const auto& raw_input : {Tensor{}, raw.tensor}) {
        const auto baseline = TrainerThinStructureTestAccess::plain(trainer, image.tensor, target.tensor, params, raw_input);
        ASSERT_TRUE(baseline);
        const auto baseline_loss = baseline->loss.item<float>();
        const auto corrected = baseline->grad_corrected.clone();
        const auto raw_gradient = baseline->grad_raw.is_valid() ? baseline->grad_raw.clone() : Tensor{};
        const auto map = TrainerThinStructureTestAccess::map(trainer, target.tensor, params.thin_structure_weight);
        EXPECT_FALSE(map.is_valid());
        EXPECT_FALSE(TrainerThinStructureTestAccess::allocated(trainer));
        const auto result = TrainerThinStructureTestAccess::weighted(trainer, image.tensor, target.tensor, params, raw_input, map);
        ASSERT_TRUE(result);
        EXPECT_EQ(result->loss.item<float>(), baseline_loss);
        expect_close(result->grad_corrected, corrected, 0.0f);
        EXPECT_EQ(result->grad_raw.is_valid(), raw_gradient.is_valid());
        if (raw_gradient.is_valid())
            expect_close(result->grad_raw, raw_gradient, 0.0f);
    }
}

TEST(ThinStructure, ConstantResponseScalesFusedAndBothDecoupledOutputs) {
    if (!std::getenv("LFS_THIN_STRUCTURE_TEST_IMAGE") || !std::getenv("LFS_THIN_STRUCTURE_TEST_OUTPUT"))
        GTEST_SKIP() << "set LFS_THIN_STRUCTURE_TEST_IMAGE and LFS_THIN_STRUCTURE_TEST_OUTPUT to run";
    auto image = read_crop(), target = read_crop(681), raw = read_crop(682);
    for (const float response : {0.0f, 1.0f}) {
        SCOPED_TRACE(response);
        auto structure = Tensor::full({32, 32}, response, Device::CUDA);
        Tensor weight;
        structure_photometric_weight(structure, {}, weight, 2.0f, true);
        const float scale = 1.0f + 2.0f * response;
        const float denominator = structure_base_denominator({}, 32, 32, true);
        FusedL1SSIMWorkspace plain_ws;
        MaskedFusedL1SSIMWorkspace weighted_ws;
        auto [plain_loss, plain_ctx] = fused_l1_ssim_forward(image.tensor, target.tensor, 0.22f, plain_ws, true);
        const float baseline = plain_loss.item<float>();
        auto plain_gradient = fused_l1_ssim_backward(plain_ctx, plain_ws).clone();
        auto [loss, ctx] = masked_fused_l1_ssim_forward(image.tensor, target.tensor, weight, 0.22f, weighted_ws, denominator);
        const float value = loss.item<float>() * ctx.mask_sum_value / denominator;
        ctx.mask_sum_value = denominator;
        if (response == 0.0f)
            EXPECT_EQ(value, baseline);
        EXPECT_NEAR(value, scale * baseline, 2e-6f);
        expect_close(plain_ws.cs_map, weighted_ws.cs_map, 0.0f);
        expect_close(plain_ws.ssim_map, weighted_ws.ssim_map, 0.0f);
        expect_close(masked_fused_l1_ssim_backward(ctx, weighted_ws), plain_gradient * scale, response == 0.0f ? 0.0f : 5e-7f);
        DecoupledFusedL1SSIMWorkspace dec_ws;
        MaskedDecoupledFusedL1SSIMWorkspace weighted_dec_ws;
        auto [dec_loss, dec_ctx] = decoupled_fused_l1_ssim_forward(image.tensor, raw.tensor, target.tensor, 0.22f, dec_ws, true);
        auto dec_gradient = decoupled_fused_l1_ssim_backward(dec_ctx, dec_ws);
        auto [weighted_loss, weighted_ctx] = masked_decoupled_fused_l1_ssim_forward(image.tensor, raw.tensor, target.tensor, weight, 0.22f, weighted_dec_ws, denominator);
        EXPECT_NEAR(weighted_loss.item<float>() * weighted_ctx.mask_sum_value / denominator, dec_loss.item<float>() * scale, 2e-6f);
        if (response == 0.0f)
            EXPECT_EQ(weighted_loss.item<float>() * weighted_ctx.mask_sum_value / denominator, dec_loss.item<float>());
        expect_close(dec_ws.cs_map, weighted_dec_ws.cs_map, 0.0f);
        weighted_ctx.mask_sum_value = denominator;
        auto gradients = masked_decoupled_fused_l1_ssim_backward(weighted_ctx, weighted_dec_ws);
        expect_close(gradients.grad_corrected, dec_gradient.grad_corrected * scale, response == 0.0f ? 0.0f : 5e-7f);
        expect_close(gradients.grad_raw, dec_gradient.grad_raw * scale, response == 0.0f ? 0.0f : 5e-7f);
    }
}

TEST(ThinStructure, WeightedFusedGradientMatchesFiniteDifference) {
    if (!std::getenv("LFS_THIN_STRUCTURE_TEST_IMAGE") || !std::getenv("LFS_THIN_STRUCTURE_TEST_OUTPUT"))
        GTEST_SKIP() << "set LFS_THIN_STRUCTURE_TEST_IMAGE and LFS_THIN_STRUCTURE_TEST_OUTPUT to run";
    auto image = read_crop(), target = read_crop(681);
    RidgeWorkspace ridge_ws;
    auto structure = Tensor::empty({32, 32}, Device::CUDA);
    ridge_structure_map(target.tensor, structure, ridge_ws);
    Tensor weight;
    structure_photometric_weight(structure, {}, weight, 1.0f, true);
    const float denominator = structure_base_denominator({}, 32, 32, true);
    MaskedFusedL1SSIMWorkspace workspace;
    auto [loss, ctx] = masked_fused_l1_ssim_forward(image.tensor, target.tensor, weight, 0.22f, workspace, denominator);
    ctx.mask_sum_value = denominator;
    const auto gradient = masked_fused_l1_ssim_backward(ctx, workspace).cpu().to_vector();
    constexpr float epsilon = 0.002f;
    for (const size_t index : {size_t{16 * 32 + 16}, size_t{32 * 32 + 14 * 32 + 17}, size_t{2 * 32 * 32 + 18 * 32 + 15}}) {
        auto plus = image.pixels, minus = image.pixels;
        plus[index] += epsilon;
        minus[index] -= epsilon;
        const auto a = Tensor::from_vector(plus, image.tensor.shape(), Device::CUDA);
        const auto b = Tensor::from_vector(minus, image.tensor.shape(), Device::CUDA);
        const float lp = weighted_value(a, target.tensor, weight, workspace, denominator);
        const float lm = weighted_value(b, target.tensor, weight, workspace, denominator);
        const float numerical = (lp - lm) / (2.0f * epsilon);
        EXPECT_NEAR(gradient[index], numerical, 4e-5f + 0.02f * std::abs(numerical));
    }
}

TEST(ThinStructure, RidgeShapeNormalizationAndFlatGuard) {
    if (!std::getenv("LFS_THIN_STRUCTURE_TEST_IMAGE") || !std::getenv("LFS_THIN_STRUCTURE_TEST_OUTPUT"))
        GTEST_SKIP() << "set LFS_THIN_STRUCTURE_TEST_IMAGE and LFS_THIN_STRUCTURE_TEST_OUTPUT to run";
    auto image = read_crop(680, 0, 96);
    RidgeWorkspace workspace;
    auto map = Tensor::empty({96, 96}, Device::CUDA);
    ridge_structure_map(image.tensor, map, workspace);
    EXPECT_EQ(map.shape(), (TensorShape{96, 96}));
    double sum = 0.0;
    for (const auto value : map.cpu().to_vector()) {
        ASSERT_TRUE(std::isfinite(value));
        ASSERT_GE(value, 0.0f);
        sum += value;
    }
    EXPECT_NEAR(sum / map.numel(), 1.0, 2e-5);
    auto flat = Tensor::full(image.tensor.shape(), image.pixels.front(), Device::CUDA);
    ridge_structure_map(flat, map, workspace);
    EXPECT_EQ(map.abs().max().item<float>(), 0.0f);
}

TEST(ThinStructure, ParameterDefaultsJsonCliAndBounds) {
    if (!std::getenv("LFS_THIN_STRUCTURE_TEST_IMAGE") || !std::getenv("LFS_THIN_STRUCTURE_TEST_OUTPUT"))
        GTEST_SKIP() << "set LFS_THIN_STRUCTURE_TEST_IMAGE and LFS_THIN_STRUCTURE_TEST_OUTPUT to run";
    for (const auto* strategy : {"mrnf", "mcmc", "igs+"}) {
        auto params = param::OptimizationParameters::defaults_for_strategy(strategy);
        const float expected_default = param::is_mrnf_strategy(strategy) ? 0.5f : 0.0f;
        EXPECT_EQ(params.thin_structure_weight, expected_default);
        EXPECT_EQ(params.to_json().at("thin_structure_weight").get<float>(), expected_default);
        params.thin_structure_weight = 1.0f;
        EXPECT_EQ(param::OptimizationParameters::from_json(params.to_json()).thin_structure_weight, 1.0f);
        for (const float invalid : {-0.1f, 4.1f, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
            params.thin_structure_weight = invalid;
            EXPECT_FALSE(params.validate().empty());
        }
        for (const float valid : {0.0f, 1.0f, 4.0f}) {
            params.thin_structure_weight = valid;
            EXPECT_TRUE(params.validate().empty()) << params.validate();
        }
    }
    const auto* image = std::getenv("LFS_THIN_STRUCTURE_TEST_IMAGE");
    const auto* output = std::getenv("LFS_THIN_STRUCTURE_TEST_OUTPUT");
    const auto data = std::filesystem::path(image).parent_path().parent_path().string();
    for (const auto* value : {"0", "1", "4"}) {
        const char* argv[] = {"LichtFeld-Studio", "--headless", "--data-path", data.c_str(), "--output-path", output,
                              "--log-file", "/dev/null", "--thin-structure-weight", value};
        auto parsed = lfs::core::args::parse_args_and_params(static_cast<int>(std::size(argv)), argv);
        ASSERT_TRUE(parsed) << parsed.error();
        param::TrainingParameters target;
        target.optimization.thin_structure_weight = 2.0f;
        param::apply_explicit_training_overrides(target, (*parsed)->overrides);
        EXPECT_FLOAT_EQ(target.optimization.thin_structure_weight, std::stof(value));
    }
}

TEST(ThinStructure, ProjectParameterSaveAndReload) {
    if (!std::getenv("LFS_THIN_STRUCTURE_TEST_IMAGE") || !std::getenv("LFS_THIN_STRUCTURE_TEST_OUTPUT"))
        GTEST_SKIP() << "set LFS_THIN_STRUCTURE_TEST_IMAGE and LFS_THIN_STRUCTURE_TEST_OUTPUT to run";
    using namespace lfs::io::project;
    using namespace lfs::test::licht;
    const auto* output = std::getenv("LFS_THIN_STRUCTURE_TEST_OUTPUT");
    auto document = require_result(ProjectDocument::create(fixed_uuid(3217), 1'700'000'000'000'000'000));
    auto params = require_result(document.parameters().snapshot());
    params.mrnf_current.thin_structure_weight = 1.0f;
    params.mrnf_session.thin_structure_weight = 1.0f;
    require_status(document.edit_parameters().set_snapshot(params));
    ProjectDocumentSaveOptions options;
    options.file_uuid = fixed_uuid(3218);
    options.disk_reserve_bytes = 0;
    const auto path = std::filesystem::path(output) / (generate_uuid_v4().to_string() + ".licht");
    std::filesystem::create_directories(path.parent_path());
    static_cast<void>(require_result(document.save(path, options)));
    auto restored = require_result(ProjectDocument::open(path));
    EXPECT_EQ(require_result(restored.parameters().snapshot()).active_optimization().thin_structure_weight, 1.0f);
    std::filesystem::remove(path);
}

TEST(ThinStructure, BaseMaskCompositionAndUnitWeightL1Gradient) {
    if (!std::getenv("LFS_THIN_STRUCTURE_TEST_IMAGE") || !std::getenv("LFS_THIN_STRUCTURE_TEST_OUTPUT"))
        GTEST_SKIP() << "set LFS_THIN_STRUCTURE_TEST_IMAGE and LFS_THIN_STRUCTURE_TEST_OUTPUT to run";
    auto image = read_crop(), target = read_crop(681);
    auto base = image.tensor.slice(0, 1, 2).squeeze(0).contiguous();
    auto response = image.tensor.slice(0, 0, 1).squeeze(0).contiguous();
    Tensor weight;
    structure_photometric_weight(response, base, weight, 1.0f, false);
    const auto weights = weight.cpu().to_vector();
    const auto masks = base.cpu().to_vector();
    const auto values = response.cpu().to_vector();
    for (size_t i = 0; i < weights.size(); ++i)
        EXPECT_FLOAT_EQ(weights[i], masks[i] * (1.0f + std::min(values[i], 4.0f)));
    const float denominator = structure_base_denominator(base, 32, 32, true);
    EXPECT_NEAR(denominator, 3.0f * base.sum().item<float>(), 1e-5f);

    std::vector<float> response_values(32 * 32, 0.0f);
    for (size_t i = response_values.size() / 2; i < response_values.size(); ++i)
        response_values[i] = 2.0f;
    response = Tensor::from_vector(response_values, {32, 32}, Device::CUDA);
    structure_photometric_weight(response, {}, weight, 1.0f, false);
    FusedL1SSIMWorkspace plain_ws;
    MaskedFusedL1SSIMWorkspace weighted_ws;
    auto [baseline_loss, baseline_context] = fused_l1_ssim_forward(image.tensor, target.tensor, 0.0f, plain_ws, false);
    const auto plain = fused_l1_ssim_backward(baseline_context, plain_ws).cpu().to_vector();
    auto [loss, ctx] = masked_fused_l1_ssim_forward(image.tensor, target.tensor, weight, 0.0f, weighted_ws, structure_base_denominator({}, 32, 32, false));
    ctx.mask_sum_value = structure_base_denominator({}, 32, 32, false);
    const auto gradient = masked_fused_l1_ssim_backward(ctx, weighted_ws).cpu().to_vector();
    for (int c = 0; c < 3; ++c)
        for (size_t i = 0; i < response_values.size() / 2; ++i)
            EXPECT_EQ(gradient[c * response_values.size() + i], plain[c * response_values.size() + i]);
}

namespace {
    Tensor read_full_real_image() {
        auto [data, width, height, channels] = load_image(std::getenv("LFS_THIN_STRUCTURE_TEST_IMAGE"));
        const std::unique_ptr<void, decltype(&free_image)> owner(data, free_image);
        if (!data || width <= 0 || height <= 0 || channels != 3)
            throw std::runtime_error("A real RGB image is required");
        return Tensor::from_blob(data, TensorShape{static_cast<size_t>(height), static_cast<size_t>(width), 3},
                                 Device::CPU, DataType::UInt8)
            .permute({2, 0, 1})
            .contiguous()
            .to(Device::CUDA);
    }
} // namespace

TEST(ThinStructure, RidgeBoundaryShapesMatchReference) {
    const auto* image_path = std::getenv("LFS_THIN_STRUCTURE_TEST_IMAGE");
    const auto* reference_path = std::getenv("LFS_THIN_STRUCTURE_REFERENCE_PATH");
    const auto* output_path = std::getenv("LFS_THIN_STRUCTURE_TEST_OUTPUT");
    if (!image_path || !reference_path || !output_path)
        GTEST_SKIP() << "set real-image, reference-map and output paths to run";
    const auto full_image = read_full_real_image();
    const bool write_reference = std::getenv("LFS_THIN_STRUCTURE_WRITE_REFERENCE") != nullptr;
    const std::filesystem::path reference_root(reference_path);
    if (write_reference)
        std::filesystem::create_directories(reference_root);
    const std::array<TensorShape, 9> shapes{
        TensorShape{1, 1}, TensorShape{2, 3}, TensorShape{5, 7}, TensorShape{11, 11},
        TensorShape{31, 33}, TensorShape{32, 128}, TensorShape{33, 129}, TensorShape{127, 131},
        TensorShape{full_image.shape()[1], full_image.shape()[2]}};
    nlohmann::json results = nlohmann::json::array();
    for (const auto dtype : {DataType::UInt8, DataType::Float32}) {
        for (const auto& shape : shapes) {
            if (std::getenv("LFS_THIN_STRUCTURE_SMALL_CASES") && shape[0] == full_image.shape()[1] && shape[1] == full_image.shape()[2])
                continue;
            const auto name = std::format("{}_{}x{}", dtype == DataType::UInt8 ? "u8" : "f32", shape[0], shape[1]);
            SCOPED_TRACE(name);
            ASSERT_LE(shape[0], full_image.shape()[1]);
            ASSERT_LE(shape[1], full_image.shape()[2]);
            auto image = full_image.slice(1, 0, static_cast<int>(shape[0]))
                             .slice(2, 0, static_cast<int>(shape[1]))
                             .contiguous();
            if (dtype == DataType::Float32)
                image = image.to(DataType::Float32) / 255.0f;
            RidgeWorkspace workspace;
            auto map = Tensor::empty(shape, Device::CUDA);
            ridge_structure_map(image, map, workspace);
            const auto actual = map.cpu().to_vector();
            const auto path = reference_root / (name + ".bin");
            const auto bytes = static_cast<std::streamsize>(actual.size() * sizeof(float));
            if (write_reference) {
                std::ofstream file(path, std::ios::binary);
                ASSERT_TRUE(file);
                file.write(reinterpret_cast<const char*>(actual.data()), bytes);
                ASSERT_TRUE(file);
            } else {
                ASSERT_TRUE(std::filesystem::is_regular_file(path));
                ASSERT_EQ(std::filesystem::file_size(path), actual.size() * sizeof(float));
                std::vector<float> reference(actual.size());
                std::ifstream file(path, std::ios::binary);
                file.read(reinterpret_cast<char*>(reference.data()), bytes);
                ASSERT_TRUE(file);
                float max_error = 0.0f;
                for (size_t i = 0; i < actual.size(); ++i) {
                    ASSERT_TRUE(std::isfinite(actual[i]));
                    ASSERT_GE(actual[i], 0.0f);
                    max_error = std::max(max_error, std::abs(actual[i] - reference[i]));
                }
                const bool byte_identical = std::memcmp(actual.data(), reference.data(), static_cast<size_t>(bytes)) == 0;
                EXPECT_LE(max_error, 1e-6f);
                results.push_back({{"case", name}, {"byte_identical", byte_identical}, {"max_error", max_error}});
            }
        }
    }
    std::filesystem::create_directories(output_path);
    std::ofstream result(std::filesystem::path(output_path) / "detector_reference_comparison.json");
    ASSERT_TRUE(result);
    result << results.dump(2);
    ASSERT_TRUE(result);
}

TEST(ThinStructure, DetectorTimingOnRealImage) {
    const auto* image_path = std::getenv("LFS_THIN_STRUCTURE_TEST_IMAGE");
    const auto* output_path = std::getenv("LFS_THIN_STRUCTURE_TEST_OUTPUT");
    if (!image_path || !output_path || !std::getenv("LFS_THIN_STRUCTURE_MEASURE"))
        GTEST_SKIP() << "set real-image, output paths and LFS_THIN_STRUCTURE_MEASURE to run timing";
    const auto byte_image = read_full_real_image();
    nlohmann::json results = nlohmann::json::array();
    for (const auto dtype : {DataType::UInt8, DataType::Float32}) {
        auto image = dtype == DataType::UInt8 ? byte_image : byte_image.to(DataType::Float32) / 255.0f;
        RidgeWorkspace workspace;
        auto map = Tensor::empty({image.shape()[1], image.shape()[2]}, Device::CUDA);
        for (int i = 0; i < 10; ++i)
            ridge_structure_map(image, map, workspace);
        ASSERT_EQ(cudaStreamSynchronize(image.stream()), cudaSuccess);
        cudaEvent_t start = nullptr, end = nullptr;
        ASSERT_EQ(cudaEventCreate(&start), cudaSuccess);
        ASSERT_EQ(cudaEventCreate(&end), cudaSuccess);
        std::vector<float> samples;
        for (int i = 0; i < 100; ++i) {
            ASSERT_EQ(cudaEventRecord(start, image.stream()), cudaSuccess);
            ridge_structure_map(image, map, workspace);
            ASSERT_EQ(cudaEventRecord(end, image.stream()), cudaSuccess);
            ASSERT_EQ(cudaEventSynchronize(end), cudaSuccess);
            float elapsed = 0.0f;
            ASSERT_EQ(cudaEventElapsedTime(&elapsed, start, end), cudaSuccess);
            samples.push_back(elapsed);
        }
        ASSERT_EQ(cudaEventDestroy(start), cudaSuccess);
        ASSERT_EQ(cudaEventDestroy(end), cudaSuccess);
        std::sort(samples.begin(), samples.end());
        float sum = 0.0f;
        for (size_t i = 10; i < 90; ++i)
            sum += samples[i];
        results.push_back({{"dtype", dtype == DataType::UInt8 ? "u8" : "f32"},
                           {"height", image.shape()[1]},
                           {"width", image.shape()[2]},
                           {"samples", samples},
                           {"trimmed_mean_ms", sum / 80.0f}});
    }
    std::filesystem::create_directories(output_path);
    std::ofstream result(std::filesystem::path(output_path) / "detector_timing.json");
    ASSERT_TRUE(result);
    result << results.dump(2);
    ASSERT_TRUE(result);
}

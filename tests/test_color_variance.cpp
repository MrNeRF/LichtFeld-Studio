/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/camera.hpp"
#include "core/cuda/sh_layout.cuh"
#include "core/parameters.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "training/checkpoint.hpp"
#include "training/optimizer/adam_optimizer.hpp"
#include "training/rasterization/fast_rasterizer.hpp"
#include "training/strategies/strategy_factory.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <gtest/gtest.h>
#include <limits>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace {
    using namespace lfs::core;
    using namespace lfs::training;
    constexpr int width = 19, height = 21, pixels = width * height;
    constexpr double sh0_basis = 0.28209479177387814;
    constexpr double sh1_basis = 0.4886025119029199;

    struct Scene {
        std::vector<float> means{0.08f, -0.05f, 0.0f, -0.12f, 0.10f, 0.8f};
        std::vector<float> scales{-1.5f, -1.2f, -1.7f, -1.0f, -1.5f, -1.3f};
        std::vector<float> rotations{1.0f, 0.1f, 0.2f, -0.1f, 0.9f, -0.2f, 0.1f, 0.15f};
        std::vector<float> opacity{-0.2f, 0.6f};
        std::vector<float> sh0{0.4f, -0.5f, 0.9f, -0.3f, 0.8f, 0.2f};
        std::vector<float> shN = std::vector<float>(18, 0.03f);
    };

    std::vector<float> targets() {
        std::vector<float> result(3 * pixels);
        for (size_t i = 0; i < result.size(); ++i)
            result[i] = static_cast<float>((i * 37 + 19) % 256) / 255.0f;
        return result;
    }

    // Independent double-precision forward oracle. It evaluates the weighted
    // objective directly in front-to-back order; no backward recurrence is used.
    double reference(const Scene& scene, const std::vector<float>& target, double weight, bool photometric = false) {
        struct Primitive {
            double x, y, a, b, c, opacity;
            std::array<double, 3> color;
        };
        std::array<Primitive, 2> prims;
        for (int i = 0; i < 2; ++i) {
            const double x = scene.means[i * 3], y = scene.means[i * 3 + 1], z = scene.means[i * 3 + 2] + 3.0;
            const double qr = scene.rotations[i * 4], qx = scene.rotations[i * 4 + 1];
            const double qy = scene.rotations[i * 4 + 2], qz = scene.rotations[i * 4 + 3];
            const double s = 2.0 / (qr * qr + qx * qx + qy * qy + qz * qz);
            const double R[3][3] = {{1 - s * (qy * qy + qz * qz), s * (qx * qy - qr * qz), s * (qr * qy + qx * qz)},
                                    {s * (qr * qz + qx * qy), 1 - s * (qx * qx + qz * qz), s * (qy * qz - qr * qx)},
                                    {s * (qx * qz - qr * qy), s * (qr * qx + qy * qz), 1 - s * (qx * qx + qy * qy)}};
            double cov[3][3]{};
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c)
                    for (int k = 0; k < 3; ++k)
                        cov[r][c] += R[r][k] * std::exp(2.0 * scene.scales[i * 3 + k]) * R[c][k];
            const double J[2][3] = {{18.0 / z, 0, -18.0 * x / (z * z)}, {0, 19.0 / z, -19.0 * y / (z * z)}};
            double projected[2][2]{{0.3, 0}, {0, 0.3}};
            for (int r = 0; r < 2; ++r)
                for (int c = 0; c < 2; ++c)
                    for (int j = 0; j < 3; ++j)
                        for (int k = 0; k < 3; ++k)
                            projected[r][c] += J[r][j] * cov[j][k] * J[c][k];
            const double det = projected[0][0] * projected[1][1] - projected[0][1] * projected[0][1];
            auto& p = prims[i];
            p = {18.0 * x / z + width * 0.5, 19.0 * y / z + height * 0.5, projected[1][1] / det, -projected[0][1] / det, projected[0][0] / det, 1.0 / (1.0 + std::exp(-scene.opacity[i])), {}};
            const double norm = std::sqrt(x * x + y * y + z * z);
            const double basis[3] = {-sh1_basis * y / norm, sh1_basis * z / norm, -sh1_basis * x / norm};
            for (int ch = 0; ch < 3; ++ch) {
                double color = 0.5 + sh0_basis * scene.sh0[i * 3 + ch];
                for (int k = 0; k < 3; ++k)
                    color += basis[k] * scene.shN[i * 9 + k * 3 + ch];
                p.color[ch] = std::clamp(color, 0.0, 4.0);
            }
        }
        double loss = 0.0;
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                const int pix = y * width + x;
                double T = 1.0;
                std::array<double, 3> image{};
                for (const auto& p : prims) {
                    const double dx = p.x - (x + 0.5), dy = p.y - (y + 0.5);
                    const double alpha = std::min(0.999, p.opacity * std::exp(-0.5 * (p.a * dx * dx + 2 * p.b * dx * dy + p.c * dy * dy)));
                    if (alpha < 1.0 / 255.0)
                        continue;
                    for (int ch = 0; ch < 3; ++ch) {
                        const double residual = p.color[ch] - target[ch * pixels + pix];
                        loss += weight / pixels * T * alpha * residual * residual;
                        image[ch] += T * alpha * p.color[ch];
                    }
                    T *= 1.0 - alpha;
                    if (T < 1e-4)
                        break;
                }
                if (photometric) {
                    for (int ch = 0; ch < 3; ++ch)
                        loss += 0.002 * (ch + 1) * (image[ch] + T * (0.1 + ch * 0.2));
                }
            }
        }
        return loss;
    }

    Tensor cuda_tensor(const std::vector<float>& data, TensorShape shape) {
        return Tensor::from_blob(const_cast<float*>(data.data()), shape, Device::CPU, DataType::Float32).cuda();
    }

    std::unique_ptr<SplatData> make_model(const Scene& scene) {
        auto model = std::make_unique<SplatData>(
            1, cuda_tensor(scene.means, {2, 3}), cuda_tensor(scene.sh0, {2, 1, 3}),
            cuda_tensor(scene.shN, {2, 3, 3}), cuda_tensor(scene.scales, {2, 3}),
            cuda_tensor(scene.rotations, {2, 4}), cuda_tensor(scene.opacity, {2, 1}), 1.0f);
        model->set_active_sh_degree(1);
        return model;
    }

    Camera make_camera() {
        auto R = cuda_tensor({1, 0, 0, 0, 1, 0, 0, 0, 1}, {3, 3});
        auto T = cuda_tensor({0, 0, 3}, {3});
        return Camera(R, T, 18.f, 19.f, width * 0.5f, height * 0.5f, Tensor{}, Tensor{},
                      CameraModelType::PINHOLE, "test", "", std::filesystem::path{}, width, height, 0);
    }

    struct Capture {
        float loss;
        std::array<std::vector<float>, 6> gradients;
    };

    Capture capture(const Scene& scene, float weight = 0.7f, bool u8 = false, bool photo = false, bool omit_disabled_target = false) {
        auto cam = make_camera();
        auto owned_model = make_model(scene);
        auto& model = *owned_model;
        AdamOptimizer opt(model, AdamConfig{.lr = 0.0f});
        opt.allocate_gradients();
        opt.zero_grad(1);
        auto bg = cuda_tensor({0.1f, 0.3f, 0.5f}, {3});
        auto fwd = fast_rasterize_forward(cam, model, bg, 0, 0, 0, 0, false, {}, false, false);
        if (!fwd)
            throw std::runtime_error(std::string(fwd.error().user_message()));
        auto target = targets();
        Tensor gt;
        if (u8) {
            std::vector<std::uint8_t> bytes(target.size());
            for (size_t i = 0; i < bytes.size(); ++i)
                bytes[i] = static_cast<std::uint8_t>(std::lround(target[i] * 255));
            gt = Tensor::from_blob(bytes.data(), {3, height, width}, Device::CPU, DataType::UInt8).cuda();
        } else {
            gt = cuda_tensor(target, {3, height, width});
        }
        auto loss = Tensor::zeros({1}, Device::CUDA);
        std::vector<float> image_grad(3 * pixels, 0.0f);
        if (photo)
            for (int ch = 0; ch < 3; ++ch)
                std::fill_n(image_grad.begin() + ch * pixels, pixels, 0.002f * (ch + 1));
        auto grad = cuda_tensor(image_grad, {3, height, width});
        // A disabled loss must not validate or read its unused target.
        FastGSColorVarianceLoss variance{weight == 0.0f && omit_disabled_target ? Tensor{} : gt, loss, weight};
        fast_rasterize_backward(fwd->second, grad, model, opt, {}, {}, DensificationType::None,
                                1, {}, {}, {}, true, variance);
        Capture result{loss.cpu().item<float>(), {}};
        constexpr ParamType types[]{ParamType::Means, ParamType::Scaling, ParamType::Rotation,
                                    ParamType::Opacity, ParamType::Sh0, ParamType::ShN};
        for (int k = 0; k < 6; ++k) {
            auto cpu = opt.get_grad(types[k]).cpu();
            if (types[k] != ParamType::ShN) {
                result.gradients[k].assign(cpu.ptr<float>(), cpu.ptr<float>() + cpu.numel());
            } else {
                for (unsigned p = 0; p < 2; ++p)
                    for (unsigned j = 0; j < 9; ++j) {
                        const size_t offset = (j / 4 * kShReorderSize + p) * 4 + j % 4;
                        result.gradients[k].push_back(cpu.ptr<float>()[offset]);
                    }
            }
        }
        return result;
    }

    class ColorVarianceCuda : public ::testing::Test {
        void SetUp() override {
            int devices = 0;
            if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
                GTEST_SKIP() << "CUDA device required";
        }
    };

    TEST_F(ColorVarianceCuda, LossAndEveryParameterGradientMatchIndependentReference) {
        for (const bool photo : {false, true}) {
            Scene scene;
            const auto actual = capture(scene, 0.7f, false, photo);
            EXPECT_NEAR(actual.loss, reference(scene, targets(), 0.7), 2e-6);
            std::array<std::vector<float>*, 6> parameters{&scene.means, &scene.scales, &scene.rotations,
                                                          &scene.opacity, &scene.sh0, &scene.shN};
            for (size_t k = 0; k < parameters.size(); ++k) {
                for (size_t i = 0; i < parameters[k]->size(); ++i) {
                    SCOPED_TRACE(::testing::Message() << "photo=" << photo << " group=" << k << " index=" << i);
                    auto& value = (*parameters[k])[i];
                    const float old = value;
                    value = old + 1e-4f;
                    const double plus = reference(scene, targets(), 0.7, photo);
                    const float step_plus = value;
                    value = old - 1e-4f;
                    const double minus = reference(scene, targets(), 0.7, photo);
                    const float step_minus = value;
                    value = old;
                    const double expected = (plus - minus) / (step_plus - step_minus);
                    EXPECT_NEAR(actual.gradients[k][i], expected, 2e-5 + 1e-3 * std::abs(expected));
                }
            }
        }
    }

    TEST_F(ColorVarianceCuda, UInt8AndNormalizedFloatTargetsAgree) {
        const auto a = capture(Scene{});
        const auto b = capture(Scene{}, 0.7f, true);
        EXPECT_NEAR(a.loss, b.loss, 1e-7);
        for (size_t k = 0; k < a.gradients.size(); ++k)
            for (size_t j = 0; j < a.gradients[k].size(); ++j)
                EXPECT_NEAR(a.gradients[k][j], b.gradients[k][j], 1e-6);
    }

    TEST_F(ColorVarianceCuda, DisabledLossIgnoresTargetAndPreservesPhotometricGradients) {
        const auto a = capture(Scene{}, 0.0f, false, true);
        const auto b = capture(Scene{}, 0.0f, false, true, true);
        EXPECT_EQ(a.loss, 0.0f);
        EXPECT_EQ(b.loss, 0.0f);
        for (size_t k = 0; k < a.gradients.size(); ++k)
            for (size_t j = 0; j < a.gradients[k].size(); ++j)
                EXPECT_NEAR(a.gradients[k][j], b.gradients[k][j], 1e-7);
    }

    TEST_F(ColorVarianceCuda, ClampedFragmentColorsHaveNoDirectColorDerivative) {
        Scene scene;
        scene.sh0[0] = -10;
        scene.sh0[1] = 20;
        const auto actual = capture(scene);
        EXPECT_NEAR(actual.loss, reference(scene, targets(), 0.7), 2e-6);
        EXPECT_EQ(actual.gradients[4][0], 0.0f);
        EXPECT_EQ(actual.gradients[4][1], 0.0f);
        for (size_t i : {0u, 1u, 3u, 4u, 6u, 7u})
            EXPECT_EQ(actual.gradients[5][i], 0.0f);
    }

    TEST_F(ColorVarianceCuda, InvisibleSplatsAndBackgroundHaveZeroVarianceLoss) {
        Scene scene;
        scene.opacity = {-30, -30};
        const auto actual = capture(scene);
        EXPECT_EQ(actual.loss, 0.0f);
        for (const auto& gradient : actual.gradients)
            for (const float value : gradient)
                EXPECT_EQ(value, 0.0f);
    }

    TEST_F(ColorVarianceCuda, SaturatedAlphaMatchesForwardTruncation) {
        Scene scene;
        scene.means = {0, 0, 0, 0, 0, 0.001f};
        scene.opacity = {20, 20};
        scene.scales.assign(6, -0.5f);
        const auto actual = capture(scene);
        EXPECT_NEAR(actual.loss, reference(scene, targets(), 0.7), 2e-6);
        for (float gradient : actual.gradients[3])
            EXPECT_NEAR(gradient, 0.0f, 1e-7);
    }

    TEST_F(ColorVarianceCuda, FusedAdamUsesLossAndRespectsFrozenRows) {
        for (const float weight : {0.0f, 0.7f}) {
            Scene scene;
            auto model = make_model(scene);
            auto cam = make_camera();
            AdamOptimizer optimizer(*model, AdamConfig{.lr = 0.001f});
            optimizer.allocate_gradients();
            optimizer.zero_grad(1);
            std::array<std::uint8_t, 2> frozen{1, 0};
            optimizer.set_frozen_mask(Tensor::from_blob(frozen.data(), {2}, Device::CPU, DataType::Bool).cuda());
            auto bg = cuda_tensor({0.1f, 0.3f, 0.5f}, {3});
            auto fwd = fast_rasterize_forward(cam, *model, bg, 0, 0, 0, 0, false, {}, false, false);
            ASSERT_TRUE(fwd.has_value());
            auto gt = cuda_tensor(targets(), {3, height, width});
            auto loss = Tensor::zeros({1}, Device::CUDA);
            auto grad = Tensor::zeros({3, height, width}, Device::CUDA);
            fast_rasterize_backward(fwd->second, grad, *model, optimizer, {}, {}, DensificationType::None,
                                    1001, {}, {}, {}, false, {gt, loss, weight});
            auto opacity = model->opacity_raw().cpu();
            EXPECT_EQ(opacity.ptr<float>()[0], scene.opacity[0]);
            if (weight == 0.0f) {
                EXPECT_EQ(opacity.ptr<float>()[1], scene.opacity[1]);
                EXPECT_EQ(loss.cpu().item<float>(), 0.0f);
            } else {
                EXPECT_NE(opacity.ptr<float>()[1], scene.opacity[1]);
                EXPECT_GT(loss.cpu().item<float>(), 0.0f);
            }
            EXPECT_EQ(optimizer.get_state(ParamType::Opacity)->step_count, 1);
        }
    }

    TEST_F(ColorVarianceCuda, CheckpointResumeKeepsLossAndWarmupForEveryStrategy) {
        for (const auto* name : {"mcmc", "mrnf", "igs+"}) {
            SCOPED_TRACE(name);
            auto model = make_model(Scene{});
            auto source = StrategyFactory::instance().create(name, *model);
            ASSERT_TRUE(source.has_value());
            param::TrainingParameters params;
            params.optimization = param::OptimizationParameters::defaults_for_strategy(name);
            auto& p = params.optimization;
            p.max_cap = 1000;
            p.sh_degree = 1;
            p.use_exposure_correction = false;
            p.color_variance_weight = 1.0f;
            p.color_variance_warmup_iters = 3;
            (*source)->initialize(p);
            std::stringstream buffer(std::ios::in | std::ios::out | std::ios::binary);
            const auto saved = serialize_checkpoint(buffer, 2, **source, params, nullptr, nullptr, nullptr);
            ASSERT_TRUE(saved.has_value());
            const auto bytes = buffer.str().size();
            buffer.seekg(0);
            auto destination_model = make_model(Scene{});
            auto destination = StrategyFactory::instance().create(name, *destination_model);
            ASSERT_TRUE(destination.has_value());
            (*destination)->initialize(p);
            auto restored = params;
            restored.optimization.color_variance_weight = 0.0f;
            restored.optimization.color_variance_warmup_iters = 0;
            const auto loaded = load_checkpoint(buffer, bytes, **destination, restored, nullptr, nullptr, nullptr);
            ASSERT_TRUE(loaded.has_value()) << loaded.error();
            EXPECT_EQ(*loaded, 2);
            EXPECT_EQ(restored.optimization.color_variance_weight, 1.0f);
            EXPECT_EQ(restored.optimization.color_variance_warmup_iters, 3u);
            EXPECT_EQ(restored.optimization.color_variance_weight_at(3), 0.0f);
            EXPECT_EQ(restored.optimization.color_variance_weight_at(4), 1.0f);
        }
    }

    TEST(ColorVarianceParameters, AllStrategiesRoundTripAndScaleWarmup) {
        for (const auto* strategy : {"mcmc", "mrnf", "igs+"}) {
            auto p = param::OptimizationParameters::defaults_for_strategy(strategy);
            EXPECT_EQ(p.color_variance_weight, 0.0f);
            p.use_exposure_correction = false;
            p.color_variance_weight = 1.0f;
            p.color_variance_warmup_iters = 15'000;
            EXPECT_TRUE(p.validate().empty()) << p.validate();
            EXPECT_EQ(p.color_variance_weight_at(15'000), 0.0f);
            EXPECT_EQ(p.color_variance_weight_at(15'001), 1.0f);
            auto restored = param::OptimizationParameters::from_json(p.to_json());
            EXPECT_EQ(restored.color_variance_weight, 1.0f);
            EXPECT_EQ(restored.color_variance_warmup_iters, 15'000u);
            restored.scale_steps(2.0f);
            EXPECT_EQ(restored.color_variance_warmup_iters, 30'000u);
            EXPECT_EQ(restored.color_variance_weight_at(30'000), 0.0f);
            EXPECT_EQ(restored.color_variance_weight_at(30'001), 1.0f);
            auto legacy = p.to_json();
            legacy.erase("color_variance_weight");
            legacy.erase("color_variance_warmup_iters");
            const auto old = param::OptimizationParameters::from_json(legacy);
            EXPECT_EQ(old.color_variance_weight, 0.0f);
            EXPECT_EQ(old.color_variance_warmup_iters, 0u);
        }
    }

    TEST(ColorVarianceParameters, RejectsInvalidWeightsAndUnsupportedModes) {
        auto p = param::OptimizationParameters::mcmc_defaults();
        for (float value : {-1.0f, 9.0f, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
            p.color_variance_weight = value;
            EXPECT_FALSE(p.validate().empty());
        }
        p.color_variance_weight = 1.0f;
        EXPECT_TRUE(p.validate().empty());
        for (bool* flag : {&p.gut, &p.mip_filter, &p.use_ppisp, &p.use_bilateral_grid, &p.use_exposure_correction, &p.enable_sparsity}) {
            *flag = true;
            EXPECT_FALSE(p.validate().empty());
            *flag = false;
        }
        p.mask_mode = param::MaskMode::Ignore;
        EXPECT_FALSE(p.validate().empty());
        p.mask_mode = param::MaskMode::None;
        p.color_variance_warmup_iters = p.iterations;
        EXPECT_FALSE(p.validate().empty());
        p.color_variance_weight = 0.0f;
        EXPECT_TRUE(p.validate().empty());
    }
} // namespace

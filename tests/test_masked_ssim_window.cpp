/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/camera.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "lfs/kernels/ssim.cuh"
#include "masked_ssim_reference.hpp"
#include "training/kernels/thin_structure.hpp"
#include "training/losses/mask_loss.hpp"
#include "training/rasterization/fast_rasterizer.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>
#include <vector>

namespace {
    using namespace lfs::core;
    using namespace lfs::training::kernels;

    constexpr int H = 37;
    constexpr int W = 43;
    constexpr int P = H * W;

    struct Inputs {
        std::vector<float> prediction, corrected, target, weight;
        Inputs() : prediction(3 * P), corrected(3 * P), target(3 * P), weight(P, 1.0f) {
            std::mt19937 rng(1011);
            std::uniform_real_distribution<float> uniform(0.1f, 0.9f);
            for (int i = 0; i < 3 * P; ++i) {
                prediction[i] = uniform(rng);
                corrected[i] = uniform(rng);
                target[i] = uniform(rng);
            }
            for (int y = 11; y < 26; ++y)
                for (int x = 13; x < 30; ++x)
                    weight[y * W + x] = 0.0f;
        }
    };

    struct Result {
        float loss;
        std::vector<float> gradient, raw_gradient;
    };

    Result evaluate(const Inputs& input, bool decoupled, float denominator = 0.0f,
                    bool byte_mask = false, float ssim_weight = 0.2f) {
        const auto prediction = Tensor::from_vector(input.prediction, {3, H, W}, Device::CUDA);
        const auto target = Tensor::from_vector(input.target, {3, H, W}, Device::CUDA);
        auto mask = Tensor::from_vector(input.weight, {H, W}, Device::CUDA);
        if (byte_mask)
            mask = mask.to(DataType::UInt8);
        if (decoupled) {
            const auto corrected = Tensor::from_vector(input.corrected, {3, H, W}, Device::CUDA);
            MaskedDecoupledFusedL1SSIMWorkspace workspace;
            const auto [loss, ctx] = masked_decoupled_fused_l1_ssim_forward(
                corrected, prediction, target, mask, ssim_weight, workspace, denominator);
            const auto grads = masked_decoupled_fused_l1_ssim_backward(ctx, workspace);
            auto raw = Tensor::zeros(grads.grad_corrected.shape(), Device::CUDA);
            accumulate_decoupled_raw_gradient(decoupled_raw_gradient(ctx), raw);
            return {loss.item<float>(), grads.grad_corrected.cpu().to_vector(), raw.cpu().to_vector()};
        }
        MaskedFusedL1SSIMWorkspace workspace;
        const auto [loss, ctx] = masked_fused_l1_ssim_forward(
            prediction, target, mask, ssim_weight, workspace, denominator);
        const auto gradient = masked_fused_l1_ssim_backward(ctx, workspace);
        return {loss.item<float>(), gradient.cpu().to_vector(), {}};
    }

    // Optional raw evidence for the generated regression figures. No private
    // scene, filename or output path is embedded in the test.
    void save_panel(const std::string& name, const std::vector<float>& data,
                    int width, int height, int channels) {
        const char* directory = std::getenv("LFS_MASK_FIGURE_DIR");
        if (!directory)
            return;
        const std::filesystem::path root(directory);
        std::filesystem::create_directories(root);
        std::ofstream output(root / (name + ".f32"), std::ios::binary);
        output.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
        ASSERT_TRUE(output.good());
        std::ofstream shape(root / (name + ".json"));
        shape << "{\"width\":" << width << ",\"height\":" << height
              << ",\"channels\":" << channels << "}";
        ASSERT_TRUE(shape.good());
    }

    class MaskedSSIMWindowTest : public ::testing::Test {
    protected:
        void SetUp() override {
            int count = 0;
            if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0)
                GTEST_SKIP() << "No CUDA device available";
        }
    };

    void expect_finite(const Result& result) {
        EXPECT_TRUE(std::isfinite(result.loss));
        for (const auto* gradient : {&result.gradient, &result.raw_gradient}) {
            for (size_t i = 0; i < gradient->size(); ++i) {
                ASSERT_TRUE(std::isfinite((*gradient)[i])) << "gradient index=" << i;
            }
        }
    }

    void expect_ignored_gradients_zero(const Result& result, const Inputs& input) {
        expect_finite(result);
        float max_ignored = 0.0f;
        for (int i = 0; i < 3 * P; ++i) {
            if (input.weight[i % P] == 0.0f) {
                max_ignored = std::max(max_ignored, std::abs(result.gradient[i]));
                if (!result.raw_gradient.empty())
                    max_ignored = std::max(max_ignored, std::abs(result.raw_gradient[i]));
            }
        }
        EXPECT_EQ(max_ignored, 0.0f);
    }

    void expect_identical(const Result& a, const Result& b) {
        EXPECT_EQ(a.loss, b.loss);
        EXPECT_EQ(a.gradient, b.gradient);
        EXPECT_EQ(a.raw_gradient, b.raw_gradient);
    }
} // namespace

TEST_F(MaskedSSIMWindowTest, IgnoredTargetCannotChangeLossOrAnyGradient) {
    for (bool decoupled : {false, true}) {
        for (int weights = 0; weights < 4; ++weights) {
            SCOPED_TRACE(::testing::Message() << "decoupled=" << decoupled << " weights=" << weights);
            Inputs input;
            if (weights == 1) {
                for (int y = 9; y < 28; ++y)
                    for (int x = 11; x < 32; ++x)
                        if (input.weight[y * W + x] != 0.0f)
                            input.weight[y * W + x] = 0.25f;
            } else if (weights == 2) {
                // Exercise the trainer's user-mask, crop ROI and thin-structure composition.
                auto user = Tensor::from_vector(input.weight, {H, W}, Device::CUDA);
                auto roi = Tensor::full({H, W}, 0.7f, Device::CUDA);
                roi.slice(1, 0, 3).zero_();
                lfs::training::losses::MaskPreprocessWorkspace preprocess;
                const auto base = lfs::training::losses::fuse_photometric_mask_weight(
                    preprocess, user, roi, false, false);
                auto structure = Tensor::full({H, W}, 0.4f, Device::CUDA);
                Tensor weight;
                structure_photometric_weight(structure, base, weight, 2.0f, false);
                input.weight = weight.cpu().to_vector();
            }
            const float denominator = weights == 2 ? 3.0f * P : 0.0f;
            const auto clean = evaluate(input, decoupled, denominator, weights == 3);
            expect_ignored_gradients_zero(clean, input);
            for (int i = 0; i < 3 * P; ++i)
                if (input.weight[i % P] == 0.0f)
                    input.target[i] = i < P ? 1.0f : 0.0f;
            const auto painted = evaluate(input, decoupled, denominator, weights == 3);
            expect_identical(clean, painted);
            expect_ignored_gradients_zero(painted, input);
            if (!decoupled && weights == 0) {
                save_panel("gradient", painted.gradient, W, H, 3);
                save_panel("gradient_prediction", input.prediction, W, H, 3);
                save_panel("gradient_target", input.target, W, H, 3);
                save_panel("gradient_mask", input.weight, W, H, 1);
            }
        }
    }
}

TEST_F(MaskedSSIMWindowTest, IgnoredPredictionCannotChangeLossOrKeptGradients) {
    for (bool decoupled : {false, true}) {
        SCOPED_TRACE(decoupled);
        Inputs input;
        const auto before = evaluate(input, decoupled);
        for (int i = 0; i < 3 * P; ++i) {
            if (input.weight[i % P] == 0.0f) {
                input.prediction[i] = 0.97f;
                input.corrected[i] = 0.03f;
            }
        }
        expect_identical(before, evaluate(input, decoupled));
    }
}

TEST_F(MaskedSSIMWindowTest, SoftWindowGradientMatchesFiniteDifference) {
    for (bool decoupled : {false, true}) {
        Inputs input;
        const int pixel = 18 * W + 12;
        input.weight[pixel] = 0.37f;
        const auto analytic = evaluate(input, decoupled, 0.0f, false, 0.7f);
        for (bool raw : {false, true}) {
            if (!decoupled && raw)
                continue;
            auto& values = decoupled && !raw ? input.corrected : input.prediction;
            const float original = values[pixel];
            constexpr float epsilon = 0.002f;
            values[pixel] = original + epsilon;
            const float plus = evaluate(input, decoupled, 0.0f, false, 0.7f).loss;
            values[pixel] = original - epsilon;
            const float minus = evaluate(input, decoupled, 0.0f, false, 0.7f).loss;
            values[pixel] = original;
            const float numerical = (plus - minus) / (2.0f * epsilon);
            const float gradient = raw ? analytic.raw_gradient[pixel] : analytic.gradient[pixel];
            EXPECT_NEAR(gradient, numerical, 2.0e-5f);
        }
    }
}

TEST_F(MaskedSSIMWindowTest, FractionalWeightsApproachZeroContinuously) {
    for (bool decoupled : {false, true}) {
        Inputs input;
        const auto zero = evaluate(input, decoupled);
        const int pixel = 18 * W + 13;
        for (float weight : {1.0e-3f, 1.0e-5f, 1.0e-7f}) {
            input.weight[pixel] = weight;
            const auto fractional = evaluate(input, decoupled);
            EXPECT_NEAR(fractional.loss, zero.loss, 2.0e-6f);
            EXPECT_LE(std::abs(fractional.gradient[pixel]), weight * 0.01f);
            if (decoupled)
                EXPECT_LE(std::abs(fractional.raw_gradient[pixel]), weight * 0.01f);
        }
    }
}

TEST_F(MaskedSSIMWindowTest, SparseSupportRemainsFinite) {
    for (bool decoupled : {false, true}) {
        for (float value : {0.5f, 3.0f}) {
            SCOPED_TRACE(::testing::Message() << "decoupled=" << decoupled << " value=" << value);
            Inputs input;
            std::fill(input.weight.begin(), input.weight.end(), 0.0f);
            input.weight[18 * W + 21] = 1.0f;
            std::fill(input.prediction.begin(), input.prediction.end(), value);
            std::fill(input.corrected.begin(), input.corrected.end(), 0.0f);
            std::fill(input.target.begin(), input.target.end(), 0.0f);
            const auto result = evaluate(input, decoupled, 0.0f, false, 1.0f);
            expect_finite(result);
            expect_ignored_gradients_zero(result, input);
        }
    }
}

TEST_F(MaskedSSIMWindowTest, TinyWindowSupportApproachesUnsupported) {
    for (bool decoupled : {false, true}) {
        Inputs input;
        std::fill(input.weight.begin(), input.weight.end(), 0.0f);
        const auto zero = evaluate(input, decoupled, 0.0f, false, 1.0f);
        expect_finite(zero);
        for (float weight : {1.0e-32f, 1.0e-35f, 1.0e-39f}) {
            SCOPED_TRACE(::testing::Message() << "decoupled=" << decoupled << " weight=" << weight);
            // Interior support cannot borrow mass from the zero-padding convention.
            for (int y = 10; y < H - 10; ++y)
                for (int x = 10; x < W - 10; ++x)
                    input.weight[y * W + x] = weight;
            const auto result = evaluate(input, decoupled, 0.0f, false, 1.0f);
            expect_finite(result);
            expect_ignored_gradients_zero(result, input);
            EXPECT_NEAR(result.loss, zero.loss, 1.0e-10f);
            for (const auto* gradient : {&result.gradient, &result.raw_gradient})
                for (float value : *gradient)
                    EXPECT_NEAR(value, 0.0f, 1.0e-10f);
        }
    }
}

TEST_F(MaskedSSIMWindowTest, DuplicatePoseMaskedSquareDoesNotMoveTrainedSurface) {
    using namespace lfs::training;
    constexpr int H = 128, W = 160, P = H * W;
    constexpr int left = 56, right = 104, top = 40, bottom = 88;
    constexpr int iterations = 300;
    constexpr size_t count = 192;
    const auto make_scene = [] {
        std::vector<float> means(count * 3), rotations(count * 4, 0.0f), colors(count * 3);
        std::mt19937 rng(1011);
        std::uniform_real_distribution<float> color(-1.1f, 1.1f);
        for (size_t i = 0; i < count; ++i) {
            means[i * 3] = (static_cast<float>(i % 16) - 7.5f) * 0.18f;
            means[i * 3 + 1] = (static_cast<float>(i / 16) - 5.5f) * 0.18f;
            rotations[i * 4] = 1.0f;
            for (int c = 0; c < 3; ++c)
                colors[i * 3 + c] = color(rng);
        }
        return SplatData(0,
                         Tensor::from_vector(means, {count, 3}, Device::CUDA),
                         Tensor::from_vector(colors, {count, 1, 3}, Device::CUDA),
                         Tensor::zeros({count, 0, 3}, Device::CUDA),
                         Tensor::full({count, 3}, -2.4f, Device::CUDA),
                         Tensor::from_vector(rotations, {count, 4}, Device::CUDA),
                         Tensor::full({count}, 2.0f, Device::CUDA), 1.0f);
    };
    auto rotation = Tensor::from_vector({1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f}, {3, 3}, Device::CUDA);
    auto translation = Tensor::from_vector({0.f, 0.f, 4.f}, {3}, Device::CUDA);
    Camera camera(rotation, translation, 200.f, 200.f, W * 0.5f, H * 0.5f,
                  {}, {}, CameraModelType::PINHOLE, "generated", "", {}, W, H, 0);
    auto bg = Tensor::full({3}, 0.2f, Device::CUDA);
    auto scene = make_scene();
    const auto clean = fast_rasterize(camera, scene, bg).image.clone();
    auto painted = clean.clone();
    painted.slice(0, 0, 1).slice(1, top, bottom).slice(2, left, right).fill_(1.0f);
    painted.slice(0, 1, 3).slice(1, top, bottom).slice(2, left, right).zero_();
    auto hole = Tensor::ones({H, W}, Device::CUDA);
    hole.slice(0, top, bottom).slice(1, left, right).zero_();

    // With fixed geometry, alpha compositing is linear in SH0 colors. Build its
    // sparse Jacobian once using actual renders, then reduce in a fixed CPU order.
    // This test-only SGD path avoids atomic sums and Adam amplifying roundoff at
    // the representable optimum; production rendering and both losses are used.
    std::vector<std::vector<std::pair<int, float>>> color_jacobian(count);
    auto basis_model = make_scene();
    basis_model.sh0().zero_();
    const auto basis_zero = fast_rasterize(camera, basis_model, bg).image.cpu().to_vector();
    for (size_t n = 0; n < count; ++n) {
        basis_model.sh0().zero_();
        basis_model.sh0().slice(0, n, n + 1).slice(2, 0, 1).fill_(1.0f);
        const auto basis = fast_rasterize(camera, basis_model, bg).image.cpu().to_vector();
        for (int pixel = 0; pixel < P; ++pixel) {
            const float derivative = basis[pixel] - basis_zero[pixel];
            if (derivative != 0.0f)
                color_jacobian[n].emplace_back(pixel, derivative);
        }
    }
    const auto train = [&](bool duplicate, bool red_square) {
        Tensor::manual_seed(1011);
        auto model = make_scene();
        auto colors = model.sh0().cpu().to_vector();
        MaskedFusedL1SSIMWorkspace workspace;
        FusedL1SSIMWorkspace unmasked_workspace;
        for (int iteration = 0; iteration < iterations; ++iteration) {
            const auto rendered = fast_rasterize(camera, model, bg).image;
            const bool second = duplicate && iteration % 2 != 0;
            Tensor gradient;
            if (second) {
                const auto [loss, ctx] = masked_fused_l1_ssim_forward(
                    rendered, red_square ? painted : clean, hole, 1.0f, workspace);
                gradient = masked_fused_l1_ssim_backward(ctx, workspace);
            } else {
                const auto [loss, ctx] = fused_l1_ssim_forward(
                    rendered, clean, 1.0f, unmasked_workspace, false);
                gradient = fused_l1_ssim_backward(ctx, unmasked_workspace);
            }
            const auto pixels = gradient.cpu().to_vector();
            for (size_t n = 0; n < count; ++n) {
                for (int c = 0; c < 3; ++c) {
                    double derivative = 0.0;
                    for (const auto& [pixel, weight] : color_jacobian[n])
                        derivative += static_cast<double>(weight) * pixels[c * P + pixel];
                    colors[n * 3 + c] -= static_cast<float>(100.0 * derivative);
                }
            }
            model.sh0() = Tensor::from_vector(colors, {count, 1, 3}, Device::CUDA);
        }
        return fast_rasterize(camera, model, bg).image.cpu().to_vector();
    };
    const auto a = train(false, false);
    const auto ab_clean = train(true, false);
    const auto ab_repeat = train(true, false);
    const auto ab_painted = train(true, true);
    constexpr float repeat_tolerance = 2.0e-5f;
    float max_repeat_difference = 0.0f, max_painted_difference = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        max_repeat_difference = std::max(max_repeat_difference, std::abs(ab_clean[i] - ab_repeat[i]));
        max_painted_difference = std::max(max_painted_difference, std::abs(ab_clean[i] - ab_painted[i]));
    }
    EXPECT_LE(max_repeat_difference, repeat_tolerance);
    EXPECT_LE(max_painted_difference, repeat_tolerance);
    double band = 0.0, far = 0.0, control_band = 0.0, inside_band = 0.0, outside_band = 0.0;
    int band_count = 0, far_count = 0, inside_count = 0, outside_count = 0;
    for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                const int i = c * P + y * W + x;
                const bool outer = x >= left - 8 && x < right + 8 && y >= top - 8 && y < bottom + 8;
                const bool inner = x >= left + 8 && x < right - 8 && y >= top + 8 && y < bottom - 8;
                if (outer && !inner) {
                    band += std::abs(a[i] - ab_painted[i]);
                    control_band += std::abs(a[i] - ab_clean[i]);
                    ++band_count;
                    if (x >= left && x < right && y >= top && y < bottom) {
                        inside_band += std::abs(a[i] - ab_painted[i]);
                        ++inside_count;
                    } else {
                        outside_band += std::abs(a[i] - ab_painted[i]);
                        ++outside_count;
                    }
                } else if (x < left - 16 || x >= right + 16 || y < top - 16 || y >= bottom + 16) {
                    far += std::abs(a[i] - ab_painted[i]);
                    ++far_count;
                }
            }
        }
    }
    band /= band_count;
    control_band /= band_count;
    far /= far_count;
    std::cout << "masked duplicate: band=" << band << " far=" << far
              << " inside8=" << inside_band / inside_count
              << " outside8=" << outside_band / outside_count
              << " clean_duplicate_band=" << control_band << '\n';
    EXPECT_LE(band, far + control_band + 2.0e-5);
    std::cout << "SGD control: repeat_max=" << max_repeat_difference
              << " painted_clean_max=" << max_painted_difference
              << " tolerance=" << repeat_tolerance << '\n';
    save_panel("synthetic_repeat", ab_repeat, W, H, 3);
    save_panel("synthetic_photo_a", clean.cpu().to_vector(), W, H, 3);
    save_panel("synthetic_photo_b", painted.cpu().to_vector(), W, H, 3);
    save_panel("synthetic_mask", hole.cpu().to_vector(), W, H, 1);
    save_panel("synthetic_reference", a, W, H, 3);
    save_panel("synthetic_clean_duplicate", ab_clean, W, H, 3);
    save_panel("synthetic_masked", ab_painted, W, H, 3);
}

TEST_F(MaskedSSIMWindowTest, WeightedWindowsMatchIndependentReference) {
    Inputs input;
    for (int i = 0; i < P; ++i)
        if (input.weight[i] != 0.0f)
            input.weight[i] = 0.2f + (i % 7) * 0.3f;
    const auto actual = evaluate(input, false);
    const auto [loss, gradient] = masked_ssim_reference(
        Tensor::from_vector(input.prediction, {3, H, W}, Device::CUDA),
        Tensor::from_vector(input.target, {3, H, W}, Device::CUDA),
        Tensor::from_vector(input.weight, {H, W}, Device::CUDA), 0.2f);
    EXPECT_NEAR(actual.loss, loss, 2.0e-6f);
    const auto reference = gradient.cpu().to_vector();
    float max_error = 0.0f;
    for (size_t i = 0; i < reference.size(); ++i)
        max_error = std::max(max_error, std::abs(reference[i] - actual.gradient[i]));
    EXPECT_LT(max_error, 2.0e-6f);
}

// Opt-in CUDA-event benchmark for interleaved before/after loss measurements.
TEST_F(MaskedSSIMWindowTest, DISABLED_LossBenchmark) {
    constexpr int height = 1080, width = 1920;
    constexpr int warmup = 30, repeats = 200;
    std::vector<float> pixels(3 * height * width), targets(pixels.size());
    for (size_t i = 0; i < pixels.size(); ++i) {
        pixels[i] = 0.1f + (i % 127) * 0.006f;
        targets[i] = 0.1f + (i % 137) * 0.005f;
    }
    const auto prediction = Tensor::from_vector(pixels, {3, height, width}, Device::CUDA);
    const auto target = Tensor::from_vector(targets, {3, height, width}, Device::CUDA);
    const auto corrected = prediction * 0.9f;
    auto mask = Tensor::ones({height, width}, Device::CUDA);
    mask.slice(0, height / 3, 2 * height / 3).slice(1, width / 3, 2 * width / 3).zero_();
    const float denominator = mask.sum().item<float>() * 3;
    for (bool masked : {false, true}) {
        for (bool decoupled : {false, true}) {
            LossWorkspaceArena arena;
            auto raw_gradient = Tensor::zeros({1, 3, height, width}, Device::CUDA);
            Tensor last_loss, last_gradient;
            const auto step = [&] {
                if (masked && decoupled) {
                    auto& ws = arena.masked_decoupled();
                    const auto [loss, ctx] = masked_decoupled_fused_l1_ssim_forward(
                        corrected, prediction, target, mask, 0.2f, ws, denominator);
                    const auto grads = masked_decoupled_fused_l1_ssim_backward(ctx, ws);
                    raw_gradient.zero_();
                    accumulate_decoupled_raw_gradient(decoupled_raw_gradient(ctx), raw_gradient);
                    last_loss = loss;
                    last_gradient = grads.grad_corrected;
                } else if (masked) {
                    auto& ws = arena.masked_fused();
                    const auto [loss, ctx] = masked_fused_l1_ssim_forward(prediction, target, mask, 0.2f, ws, denominator);
                    last_loss = loss;
                    last_gradient = masked_fused_l1_ssim_backward(ctx, ws);
                } else if (decoupled) {
                    auto& ws = arena.decoupled();
                    const auto [loss, ctx] = decoupled_fused_l1_ssim_forward(corrected, prediction, target, 0.2f, ws, false);
                    const auto grads = decoupled_fused_l1_ssim_backward(ctx, ws);
                    raw_gradient.zero_();
                    accumulate_decoupled_raw_gradient(decoupled_raw_gradient(ctx), raw_gradient);
                    last_loss = loss;
                    last_gradient = grads.grad_corrected;
                } else {
                    auto& ws = arena.fused();
                    const auto [loss, ctx] = fused_l1_ssim_forward(prediction, target, 0.2f, ws, false);
                    last_loss = loss;
                    last_gradient = fused_l1_ssim_backward(ctx, ws);
                }
            };
            for (int i = 0; i < warmup; ++i)
                step();
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
            cudaEvent_t begin, end;
            ASSERT_EQ(cudaEventCreate(&begin), cudaSuccess);
            ASSERT_EQ(cudaEventCreate(&end), cudaSuccess);
            ASSERT_EQ(cudaEventRecord(begin, prediction.stream()), cudaSuccess);
            for (int i = 0; i < repeats; ++i)
                step();
            ASSERT_EQ(cudaEventRecord(end, prediction.stream()), cudaSuccess);
            ASSERT_EQ(cudaEventSynchronize(end), cudaSuccess);
            float ms = 0.0f;
            ASSERT_EQ(cudaEventElapsedTime(&ms, begin, end), cudaSuccess);
            ASSERT_EQ(cudaEventDestroy(begin), cudaSuccess);
            ASSERT_EQ(cudaEventDestroy(end), cudaSuccess);
            uint64_t hash = 1469598103934665603ULL;
            for (const auto& tensor : {last_loss, last_gradient, raw_gradient}) {
                const auto values = tensor.cpu().to_vector();
                const auto* bytes = reinterpret_cast<const unsigned char*>(values.data());
                for (size_t i = 0; i < values.size() * sizeof(float); ++i)
                    hash = (hash ^ bytes[i]) * 1099511628211ULL;
            }
            std::cout << "LOSS_BENCH masked=" << masked << " decoupled=" << decoupled
                      << " ms=" << ms / repeats << " workspace_bytes=" << arena.allocated_bytes()
                      << " hash=" << hash << '\n';
        }
    }
}

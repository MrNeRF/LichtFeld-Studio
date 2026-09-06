/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/camera.hpp"
#include "core/cuda_error.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "core/tensor/internal/cuda_stream_context.hpp"
#include "training/rasterization/fast_rasterizer.hpp"
#include "training/rasterization/gsplat_rasterizer.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <gtest/gtest.h>
#include <vector>

namespace {
    using namespace lfs::core;
    using namespace lfs::training;
    using RGB = std::array<double, 3>;

    struct TestStream {
        cudaStream_t stream = nullptr;
        TestStream() { LFS_CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)); }
        ~TestStream() { LFS_CUDA_CHECK(cudaStreamDestroy(stream)); }
    };

    // Independent leave-one-out compositor on the renderer's accepted sequence.
    // The cutoff is held fixed, as prescribed by ordered GaussianPOP.
    RGB composite(const std::vector<float>& alphas, const std::vector<RGB>& colors,
                  const std::vector<size_t>& accepted, RGB background, size_t omit) {
        RGB result{};
        double T = 1;
        for (const auto i : accepted) {
            if (i == omit)
                continue;
            for (int c = 0; c < 3; ++c)
                result[c] += T * alphas[i] * colors[i][c];
            T *= 1.f - alphas[i];
        }
        for (int c = 0; c < 3; ++c)
            result[c] += T * background[c];
        return result;
    }

    void check_scores(bool fast, size_t n, float opacity, bool image_background, bool capture_gradients = false) {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
            GTEST_SKIP();
        auto R = Tensor::from_vector({1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f}, {3, 3}, Device::CUDA);
        auto T = Tensor::from_vector({0.f, 0.f, 3.f}, {3}, Device::CUDA);
        // GUT requires all UT sigma points to lie within the image margin.
        // At f=100 these broad splats project sigma points outside the 1x1 sensor
        // and are culled. f=20 keeps them inside while preserving alpha on the
        // central ray and enough projected area to pass opacity-aware culling.
        Camera camera(R, T, 20.f, 20.f, 0.5f, 0.5f, {}, {}, lfs::core::CameraModelType::PINHOLE,
                      "pop", "", std::filesystem::path{}, 1, 1, 0);
        std::vector<float> means(n * 3, 0.f), sh(n * 3), quats(n * 4, 0.f);
        std::vector<float> alphas(n);
        std::vector<RGB> colors(n);
        for (size_t i = 0; i < n; ++i) {
            means[i * 3 + 2] = float(i) * 0.001f;
            quats[i * 4] = 1.f;
            for (int c = 0; c < 3; ++c) {
                sh[i * 3 + c] = float((i + c) % 7) * 0.2f;
                colors[i][c] = 0.5f + 0.28209479177387814f * sh[i * 3 + c];
            }
            alphas[i] = std::min(0.999f, 1.f / (1.f + std::exp(-opacity)));
        }
        // An invisible row must retain exactly zero energy.
        means[(n - 1) * 3 + 2] = -10.f;
        auto means_t = Tensor::from_blob(means.data(), {n, 3}, Device::CPU, DataType::Float32).to(Device::CUDA);
        auto sh_t = Tensor::from_blob(sh.data(), {n, 1, 3}, Device::CPU, DataType::Float32).to(Device::CUDA);
        auto q_t = Tensor::from_blob(quats.data(), {n, 4}, Device::CPU, DataType::Float32).to(Device::CUDA);
        auto shn = Tensor::zeros({n, 0, 3}, Device::CUDA);
        auto scales = Tensor::full({n, 3}, -1.f, Device::CUDA);
        auto opacities = Tensor::full({n}, opacity, Device::CUDA);
        SplatData model(0, means_t, sh_t, shn, scales, q_t, opacities, 1.f);
        auto bg = Tensor::from_vector({0.17f, 0.37f, 0.73f}, {3}, Device::CUDA);
        auto bg_image = image_background ? Tensor::from_vector({0.83f, 0.29f, 0.11f}, {3, 1, 1}, Device::CUDA) : Tensor{};
        const RGB background = image_background ? RGB{0.83f, 0.29f, 0.11f} : RGB{0.17f, 0.37f, 0.73f};
        auto scores = Tensor::zeros({n, sizeof(double)}, Device::CUDA, DataType::UInt8);
        if (fast) {
            auto render = fast_rasterize_forward(camera, model, bg, 0, 0, 0, 0, false, bg_image);
            ASSERT_TRUE(render.has_value());
            ASSERT_TRUE(fast_accumulate_pop_scores(render->second, scores).has_value());
            ASSERT_TRUE(fast_accumulate_pop_scores(render->second, scores).has_value());
            if (capture_gradients) {
                TestStream gradient_stream;
                AdamOptimizer optimizer(model, AdamConfig{});
                {
                    CUDAStreamGuard stream_guard(gradient_stream.stream);
                    optimizer.allocate_gradients();
                    for (auto type : {ParamType::Means, ParamType::Sh0, ParamType::Scaling, ParamType::Rotation, ParamType::Opacity})
                        (void)optimizer.get_grad(type);
                    optimizer.zero_grad(0);
                }
                const auto execution_stream = render->second.forward_ctx.stream;
                ASSERT_NE(gradient_stream.stream, execution_stream);
                const auto before = model.opacity_raw().to(Device::CPU);
                auto gradient = Tensor::ones({3, 1, 1}, Device::CUDA);
                fast_rasterize_backward(render->second, gradient, model, optimizer, {}, {}, DensificationType::None,
                                        0, {}, {}, {}, true);
                for (auto type : {ParamType::Means, ParamType::Sh0, ParamType::Scaling, ParamType::Rotation, ParamType::Opacity})
                    EXPECT_EQ(optimizer.get_grad(type).stream(), execution_stream);
                const auto after = model.opacity_raw().to(Device::CPU);
                EXPECT_EQ(std::memcmp(before.data_ptr(), after.data_ptr(), n * sizeof(float)), 0);
                EXPECT_EQ(optimizer.get_state(ParamType::Opacity)->step_count, 0);
                const auto sh_grad = optimizer.get_grad(ParamType::Sh0).to(Device::CPU);
                // At the central pixel, d(sum RGB)/dSH0_red = alpha * C0.
                EXPECT_NEAR(sh_grad.ptr<float>()[0], alphas[0] * 0.28209479177387814f, 1.e-5f);
            }
        } else {
            auto render = gsplat_rasterize_forward(camera, model, bg, 0, 0, 0, 0, 1.f, false, GsplatRenderMode::RGB, true, bg_image);
            ASSERT_TRUE(render.has_value()) << render.error();
            // Prove the oracle's assumed contributor population reached the
            // rasterizer, including all >256 overlaps in the long-list case.
            EXPECT_EQ(render->second.n_isects, alphas[0] >= 1.f / 255.f ? static_cast<int32_t>(n - 1) : 0);
            ASSERT_TRUE(gsplat_accumulate_pop_scores(render->second, scores).has_value());
            ASSERT_TRUE(gsplat_accumulate_pop_scores(render->second, scores).has_value());
            GlobalArenaManager::instance().get_arena().end_frame(render->second.frame_id, render->second.stream);
        }
        LFS_CUDA_CHECK(cudaDeviceSynchronize());
        const auto cpu = scores.to(Device::CPU);
        std::vector<double> actual(n);
        std::memcpy(actual.data(), cpu.data_ptr(), n * sizeof(double));
        std::vector<size_t> accepted;
        float transmittance = 1;
        for (size_t i = 0; i + 1 < n; ++i) {
            if (alphas[i] < 1.f / 255.f)
                continue;
            const float next = transmittance * (1.f - alphas[i]);
            if (!fast && next <= 1.e-4f)
                break;
            accepted.push_back(i);
            transmittance = next;
            if (fast && next < 1.e-4f)
                break;
        }
        const auto full = composite(alphas, colors, accepted, background, n);
        for (size_t i = 0; i < n; ++i) {
            const auto removed = composite(alphas, colors, accepted, background, i);
            double expected = 0;
            for (int c = 0; c < 3; ++c)
                expected += 2 * (full[c] - removed[c]) * (full[c] - removed[c]);
            EXPECT_NEAR(actual[i], expected, 2.e-7 + expected * 2.e-4) << "row=" << i << " fast=" << fast;
        }
        EXPECT_EQ(actual.back(), 0.0);
        if (n > 256 && opacity < -4.f)
            EXPECT_GT(actual[270], 0.0);
    }

    void check_distorted_gut_leave_one_out(bool fisheye) {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
            GTEST_SKIP() << "CUDA device required";
        constexpr size_t n = 4, width = 16, height = 12, pixels = width * height;
        auto R = Tensor::from_vector({1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f}, {3, 3}, Device::CUDA);
        auto T = Tensor::zeros({3}, Device::CUDA);
        auto radial = fisheye
                          ? Tensor::from_vector({0.08f, -0.02f, 0.004f, -0.0003f}, {4}, Device::CPU)
                          : Tensor::from_vector({0.12f, -0.025f, 0.003f, 0.f, 0.f, 0.f}, {6}, Device::CPU);
        auto tangential = fisheye ? Tensor{} : Tensor::from_vector({0.008f, -0.006f}, {2}, Device::CPU);
        Camera camera(R, T, 10.f, 9.f, 7.2f, 5.1f, radial, tangential,
                      fisheye ? lfs::core::CameraModelType::FISHEYE : lfs::core::CameraModelType::PINHOLE,
                      "distorted-pop", "", std::filesystem::path{}, width, height, 0);
        const auto make_model = [](size_t omitted) {
            // With four alphas bounded by 0.12, T stays above 0.88^4 > 0.59.
            // Removing a row therefore cannot reveal contributors beyond a cutoff.
            std::vector<float> opacities(n, std::log(0.12f / 0.88f));
            if (omitted < n)
                opacities[omitted] = -30.f;
            return SplatData(0,
                             Tensor::from_vector({-0.65f, -0.3f, 2.8f, 0.45f, 0.4f, 3.0f,
                                                  -0.25f, 0.5f, 3.3f, 0.7f, -0.4f, 3.6f},
                                                 {n, 3}, Device::CUDA),
                             Tensor::from_vector({0.2f, 0.5f, 0.3f, 0.5f, 0.1f, 0.4f,
                                                  0.3f, 0.6f, 0.1f, 0.6f, 0.3f, 0.5f},
                                                 {n, 1, 3}, Device::CUDA),
                             Tensor::zeros({n, 0, 3}, Device::CUDA),
                             Tensor::from_vector({-1.1f, -1.4f, -1.7f, -1.4f, -1.0f, -1.5f,
                                                  -1.2f, -1.6f, -1.0f, -1.5f, -1.1f, -1.3f},
                                                 {n, 3}, Device::CUDA),
                             Tensor::from_vector({1.f, 0.f, 0.f, 0.f, 0.98f, 0.f, 0.f, 0.2f,
                                                  0.98f, 0.2f, 0.f, 0.f, 0.98f, 0.f, 0.2f, 0.f},
                                                 {n, 4}, Device::CUDA),
                             Tensor::from_vector(opacities, {n, 1}, Device::CUDA), 1.f);
        };
        std::vector<float> background_values(3 * pixels);
        for (size_t c = 0; c < 3; ++c)
            for (size_t y = 0; y < height; ++y)
                for (size_t x = 0; x < width; ++x)
                    background_values[c * pixels + y * width + x] =
                        0.04f + 0.07f * float(c) + 0.08f * float(x) / width + 0.05f * float(y) / height;
        auto background = Tensor::zeros({3}, Device::CUDA);
        auto background_image = Tensor::from_vector(background_values, {3, height, width}, Device::CUDA);
        auto scores = Tensor::zeros({n, sizeof(double)}, Device::CUDA, DataType::UInt8);
        auto full_model = make_model(n);
        std::vector<float> full_image;
        {
            auto full = gsplat_rasterize_forward(camera, full_model, background, 0, 0, 0, 0,
                                                 1.f, false, GsplatRenderMode::RGB, true, background_image);
            ASSERT_TRUE(full.has_value()) << full.error();
            EXPECT_NE(full->second.radial_ptr, nullptr);
            if (!fisheye)
                EXPECT_NE(full->second.tangential_ptr, nullptr);
            const auto scored = gsplat_accumulate_pop_scores(full->second, scores);
            full_image = full->first.image.to_vector();
            GlobalArenaManager::instance().get_arena().end_frame(full->second.frame_id, full->second.stream);
            ASSERT_TRUE(scored) << (scored ? "" : lfs::format_for_developer(scored.error()));
        }
        const auto host_scores = scores.cpu();
        std::array<double, n> actual{};
        std::memcpy(actual.data(), host_scores.data_ptr(), sizeof(actual));
        for (size_t omitted = 0; omitted < n; ++omitted) {
            SCOPED_TRACE(omitted);
            auto removed_model = make_model(omitted);
            auto removed = gsplat_rasterize_forward(camera, removed_model, background, 0, 0, 0, 0,
                                                    1.f, false, GsplatRenderMode::RGB, true, background_image);
            ASSERT_TRUE(removed.has_value()) << removed.error();
            const auto removed_image = removed->first.image.to_vector();
            GlobalArenaManager::instance().get_arena().end_frame(removed->second.frame_id, removed->second.stream);
            ASSERT_EQ(full_image.size(), 3 * pixels);
            ASSERT_EQ(removed_image.size(), full_image.size());
            double expected = 0;
            for (size_t i = 0; i < full_image.size(); ++i) {
                const double delta = double(full_image[i]) - double(removed_image[i]);
                expected += delta * delta;
            }
            EXPECT_GT(expected, 1.e-6) << "off-axis Gaussian must contribute to the image";
            EXPECT_NEAR(actual[omitted], expected, 1.e-7 + 2.e-4 * expected);
        }
    }
} // namespace

TEST(PopScoreBackends, SolidAndPerPixelBackground) {
    for (bool fast : {false, true})
        for (bool image : {false, true})
            check_scores(fast, 5, -0.4f, image);
}
TEST(PopScoreBackends, InclusiveFastAndExclusiveGsplatEarlyStop) {
    for (bool fast : {false, true})
        check_scores(fast, 30, 1.5f, true);
}
TEST(PopScoreBackends, MoreThan256OrderedOverlaps) {
    for (bool fast : {false, true})
        check_scores(fast, 302, -5.f, false);
}
TEST(PopScoreBackends, BelowAlphaThreshold) {
    for (bool fast : {false, true})
        check_scores(fast, 5, -8.f, false);
}
TEST(PopScoreBackends, DistortedOpenCvPinholeMatchesLeaveOneOutRenders) {
    check_distorted_gut_leave_one_out(false);
}
TEST(PopScoreBackends, DistortedFisheyeMatchesLeaveOneOutRenders) {
    check_distorted_gut_leave_one_out(true);
}
TEST(PopScoreBackends, DeferredFastBackwardCapturesWithoutAdamUpdate) {
    check_scores(true, 5, -0.4f, false, true);
}
TEST(PopScoreBackends, DeferredFastBackwardWithNoVisibleContributorsProducesZeroGradients) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
        GTEST_SKIP() << "CUDA device required";
    auto R = Tensor::from_vector({1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f}, {3, 3}, Device::CUDA);
    auto T = Tensor::zeros({3}, Device::CUDA);
    Camera camera(R, T, 100.f, 100.f, 0.5f, 0.5f, {}, {}, lfs::core::CameraModelType::PINHOLE,
                  "empty-pop", "", std::filesystem::path{}, 1, 1, 0);
    constexpr size_t n = 3;
    SplatData model(1,
                    Tensor::from_vector({0.f, 0.f, -3.f, 0.1f, 0.f, -4.f, -0.1f, 0.f, -5.f}, {n, 3}, Device::CUDA),
                    Tensor::full({n, 1, 3}, 0.1f, Device::CUDA),
                    Tensor::full({n, 3, 3}, 0.03f, Device::CUDA),
                    Tensor::full({n, 3}, -2.f, Device::CUDA),
                    Tensor::from_vector({1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f}, {n, 4}, Device::CUDA),
                    Tensor::full({n, 1}, 0.2f, Device::CUDA), 1.f);
    model.set_active_sh_degree(1);
    const std::array<const Tensor*, 6> parameters{
        &model.means(), &model.sh0(), &model.shN(), &model.scaling_raw(), &model.rotation_raw(), &model.opacity_raw()};
    const std::array<ParamType, 6> types{
        ParamType::Means, ParamType::Sh0, ParamType::ShN, ParamType::Scaling, ParamType::Rotation, ParamType::Opacity};
    std::array<std::vector<float>, 6> before;
    for (size_t i = 0; i < parameters.size(); ++i)
        before[i] = parameters[i]->to_vector();
    AdamOptimizer optimizer(model, AdamConfig{});
    optimizer.allocate_gradients();
    for (const auto type : types)
        (void)optimizer.get_grad(type);
    optimizer.zero_grad(1);
    auto background = Tensor::from_vector({0.2f, 0.3f, 0.4f}, {3}, Device::CUDA);
    auto rendered = fast_rasterize_forward(camera, model, background);
    ASSERT_TRUE(rendered.has_value());
    ASSERT_EQ(rendered->second.forward_ctx.n_instances, 0);
    ASSERT_EQ(rendered->second.forward_ctx.n_visible, 0);
    auto image_gradient = Tensor::ones({3, 1, 1}, Device::CUDA);
    fast_rasterize_backward(rendered->second, image_gradient, model, optimizer, {}, {}, DensificationType::None,
                            1, {}, {}, {}, true);
    LFS_CUDA_CHECK(cudaDeviceSynchronize());
    for (size_t i = 0; i < types.size(); ++i) {
        SCOPED_TRACE(i);
        EXPECT_EQ(optimizer.get_step_count(types[i]), 0);
        for (const float value : optimizer.get_grad(types[i]).to_vector())
            EXPECT_EQ(value, 0.f);
        EXPECT_EQ(parameters[i]->to_vector(), before[i]);
    }
}
TEST(PopScoreBackends, RejectsWrongScoreStorageBeforeDispatch) {
    auto invalid_scores = Tensor::zeros({2, 8}, Device::CPU, DataType::UInt8);
    FastRasterizeContext fast;
    GsplatRasterizeContext gsplat;
    EXPECT_FALSE(fast_accumulate_pop_scores(fast, invalid_scores).has_value());
    EXPECT_FALSE(gsplat_accumulate_pop_scores(gsplat, invalid_scores).has_value());
}

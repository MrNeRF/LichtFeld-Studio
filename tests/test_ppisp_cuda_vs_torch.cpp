/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <gtest/gtest.h>
#include <limits>
#include <random>
#include <torch/torch.h>

#include "lfs/kernels/ppisp.cuh"

namespace {

    constexpr float FORWARD_ATOL = 5e-6f;
    constexpr float BACKWARD_ATOL = 1e-4f;
    constexpr int DEFAULT_SEED = 42;

    const auto GPU_F32 = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);

    // clang-format off
const torch::Tensor COLOR_PINV_BLOCK_DIAG = torch::tensor({
    0.0480542f, -0.0043631f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    -0.0043631f, 0.0481283f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0580570f, -0.0179872f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, -0.0179872f, 0.0431061f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0433336f, -0.0180537f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, -0.0180537f, 0.0580500f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0128369f, -0.0034654f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, -0.0034654f, 0.0128158f,
}, torch::kFloat32).reshape({8, 8});
    // clang-format on

    torch::Tensor computeHomography(const torch::Tensor& color_params, const int frame_idx) {
        const auto cp = color_params[frame_idx];
        const auto block_diag = COLOR_PINV_BLOCK_DIAG.to(cp.device());
        const auto offsets = torch::matmul(cp, block_diag);

        const auto bd = offsets.slice(0, 0, 2);
        const auto rd = offsets.slice(0, 2, 4);
        const auto gd = offsets.slice(0, 4, 6);
        const auto nd = offsets.slice(0, 6, 8);

        const auto opts = torch::TensorOptions().device(cp.device()).dtype(torch::kFloat32);
        const auto s_b = torch::tensor({0.0f, 0.0f, 1.0f}, opts);
        const auto s_r = torch::tensor({1.0f, 0.0f, 1.0f}, opts);
        const auto s_g = torch::tensor({0.0f, 1.0f, 1.0f}, opts);
        const auto s_gray = torch::tensor({1.0f / 3.0f, 1.0f / 3.0f, 1.0f}, opts);

        const auto one = torch::ones({1}, opts).squeeze();
        const auto t_b = torch::stack({s_b[0] + bd[0], s_b[1] + bd[1], one});
        const auto t_r = torch::stack({s_r[0] + rd[0], s_r[1] + rd[1], one});
        const auto t_g = torch::stack({s_g[0] + gd[0], s_g[1] + gd[1], one});
        const auto t_gray = torch::stack({s_gray[0] + nd[0], s_gray[1] + nd[1], one});

        const auto T = torch::stack({t_b, t_r, t_g}, 1);

        const auto zero = torch::zeros({1}, opts).squeeze();
        const auto skew = torch::stack({torch::stack({zero, -t_gray[2], t_gray[1]}),
                                        torch::stack({t_gray[2], zero, -t_gray[0]}),
                                        torch::stack({-t_gray[1], t_gray[0], zero})});

        const auto M = torch::matmul(skew, T);
        const auto r0 = M[0], r1 = M[1], r2 = M[2];
        const auto lam01 = torch::cross(r0, r1);
        const auto lam02 = torch::cross(r0, r2);
        const auto lam12 = torch::cross(r1, r2);

        const auto n01 = (lam01 * lam01).sum();
        const auto n02 = (lam02 * lam02).sum();
        const auto n12 = (lam12 * lam12).sum();

        const auto lam =
            torch::where(n01 >= n02, torch::where(n01 >= n12, lam01, lam12), torch::where(n02 >= n12, lam02, lam12));

        const auto S_inv = torch::tensor({{-1.0f, -1.0f, 1.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}}, opts);
        auto H = torch::matmul(torch::matmul(T, torch::diag(lam)), S_inv);
        return H / (H[2][2] + 1e-10f);
    }

    torch::Tensor computeVignettingFalloff(const torch::Tensor& vig_params, const int height, const int width,
                                           const torch::Device& device) {
        const auto opts = torch::TensorOptions().device(device).dtype(torch::kFloat32);
        const auto y_coords = torch::arange(height, opts).unsqueeze(1).expand({height, width});
        const auto x_coords = torch::arange(width, opts).unsqueeze(0).expand({height, width});

        const float res_max = static_cast<float>(std::max(width, height));
        const auto u = (x_coords + 0.5f - width * 0.5f) / res_max;
        const auto v = (y_coords + 0.5f - height * 0.5f) / res_max;

        const auto optical_center = vig_params.slice(0, 0, 2);
        const auto alphas = vig_params.slice(0, 2, 5);

        const auto delta_u = u - optical_center[0];
        const auto delta_v = v - optical_center[1];
        const auto r2 = delta_u * delta_u + delta_v * delta_v;

        auto falloff = torch::ones_like(r2);
        auto r2_pow = r2.clone();
        for (int i = 0; i < 3; ++i) {
            falloff = falloff + alphas[i] * r2_pow;
            r2_pow = r2_pow * r2;
        }
        return falloff.clamp(0.0f, 1.0f);
    }

    torch::Tensor ppispApplyTorch(const torch::Tensor& exposure_params, const torch::Tensor& vignetting_params,
                                  const torch::Tensor& color_params, const torch::Tensor& crf_params,
                                  const torch::Tensor& rgb_in, const int camera_idx, const int frame_idx) {
        const int height = rgb_in.size(1);
        const int width = rgb_in.size(2);
        const auto opts = torch::TensorOptions().device(rgb_in.device()).dtype(torch::kFloat32);
        auto rgb = rgb_in;

        if (frame_idx >= 0) {
            rgb = rgb * torch::pow(torch::tensor(2.0f, opts), exposure_params[frame_idx]);
        }

        if (camera_idx >= 0) {
            std::vector<torch::Tensor> channels;
            channels.reserve(3);
            for (int ch = 0; ch < 3; ++ch) {
                const auto falloff = computeVignettingFalloff(vignetting_params[camera_idx][ch], height, width, rgb.device());
                channels.push_back(rgb[ch] * falloff);
            }
            rgb = torch::stack(channels, 0);
        }

        if (frame_idx >= 0) {
            const auto H = computeHomography(color_params, frame_idx);
            const auto r = rgb[0], g = rgb[1], b = rgb[2];
            const auto intensity = r + g + b;

            auto rgi = torch::stack({r, g, intensity}, 0).reshape({3, -1});
            rgi = torch::matmul(H, rgi).reshape({3, height, width});

            const auto scale = intensity / (rgi[2] + 1e-5f);
            rgi = rgi * scale.unsqueeze(0);

            const auto r_out = rgi[0], g_out = rgi[1];
            rgb = torch::stack({r_out, g_out, rgi[2] - r_out - g_out}, 0);
        }

        if (camera_idx >= 0) {
            rgb = rgb.clamp(0.0f, 1.0f);
            const auto eps = torch::tensor(1e-6f, opts);

            std::vector<torch::Tensor> channels;
            channels.reserve(3);
            for (int ch = 0; ch < 3; ++ch) {
                const auto crf = crf_params[camera_idx][ch];
                const auto toe = 0.3f + torch::nn::functional::softplus(crf[0]);
                const auto shoulder = 0.3f + torch::nn::functional::softplus(crf[1]);
                const auto gamma = 0.1f + torch::nn::functional::softplus(crf[2]);
                const auto center = torch::sigmoid(crf[3]);

                const auto lerp_val = toe + center * (shoulder - toe);
                const auto a = (shoulder * center) / lerp_val;
                const auto b_coef = 1.0f - a;

                const auto x = rgb[ch];
                const auto y_low = a * torch::pow((x / center).clamp_min(eps), toe);
                const auto y_high = 1.0f - b_coef * torch::pow(((1.0f - x) / (1.0f - center)).clamp_min(eps), shoulder);
                const auto y = torch::where(x <= center, y_low, y_high);
                channels.push_back(torch::pow(y.clamp_min(eps), gamma));
            }
            rgb = torch::stack(channels, 0);
        }

        return rgb;
    }

    struct TestParams {
        torch::Tensor exposure;
        torch::Tensor vignetting;
        torch::Tensor color;
        torch::Tensor crf;
    };

    TestParams createParams(const int num_cameras, const int num_frames, const int seed) {
        torch::manual_seed(seed);
        return {
            torch::empty({num_frames}, GPU_F32).uniform_(-0.5f, 0.5f),
            torch::empty({num_cameras, 3, 5}, GPU_F32).uniform_(-0.1f, 0.1f),
            torch::empty({num_frames, 8}, GPU_F32).uniform_(-0.1f, 0.1f),
            torch::empty({num_cameras, 3, 4}, GPU_F32).uniform_(-0.5f, 0.5f),
        };
    }

    torch::Tensor createImage(const int height, const int width, const int seed) {
        torch::manual_seed(seed);
        return torch::empty({3, height, width}, GPU_F32).uniform_(0.1f, 0.9f);
    }

    torch::Tensor runCudaForward(const TestParams& p, const torch::Tensor& rgb_in, const int height, const int width,
                                 const int camera_idx, const int frame_idx) {
        auto rgb_out = torch::empty_like(rgb_in);
        lfs::training::kernels::launch_ppisp_forward_chw(
            p.exposure.data_ptr<float>(), p.vignetting.data_ptr<float>(), p.color.data_ptr<float>(),
            p.crf.data_ptr<float>(), rgb_in.data_ptr<float>(), rgb_out.data_ptr<float>(), height, width,
            static_cast<int>(p.vignetting.size(0)), static_cast<int>(p.exposure.size(0)), camera_idx, frame_idx, nullptr);
        cudaDeviceSynchronize();
        return rgb_out;
    }

    struct BackwardGrads {
        torch::Tensor rgb_in;
        torch::Tensor exposure;
        torch::Tensor vignetting;
        torch::Tensor color;
        torch::Tensor crf;
    };

    BackwardGrads runCudaBackward(const TestParams& p, const torch::Tensor& rgb_in, const torch::Tensor& grad_out,
                                  const int height, const int width, const int camera_idx, const int frame_idx) {
        BackwardGrads g{
            torch::zeros_like(rgb_in),
            torch::zeros_like(p.exposure),
            torch::zeros_like(p.vignetting),
            torch::zeros_like(p.color),
            torch::zeros_like(p.crf),
        };

        lfs::training::kernels::launch_ppisp_backward_chw(
            p.exposure.data_ptr<float>(), p.vignetting.data_ptr<float>(), p.color.data_ptr<float>(),
            p.crf.data_ptr<float>(), rgb_in.data_ptr<float>(), grad_out.data_ptr<float>(), g.exposure.data_ptr<float>(),
            g.vignetting.data_ptr<float>(), g.color.data_ptr<float>(), g.crf.data_ptr<float>(), g.rgb_in.data_ptr<float>(),
            height, width, static_cast<int>(p.vignetting.size(0)), static_cast<int>(p.exposure.size(0)), camera_idx,
            frame_idx, nullptr);
        cudaDeviceSynchronize();
        return g;
    }

    class PPISPCudaVsTorchTest : public ::testing::Test {};

    TEST_F(PPISPCudaVsTorchTest, ForwardBasic) {
        const auto params = createParams(2, 5, DEFAULT_SEED);
        const auto rgb_in = createImage(64, 64, DEFAULT_SEED);

        const auto rgb_cuda = runCudaForward(params, rgb_in, 64, 64, 0, 0);
        const auto rgb_torch = ppispApplyTorch(params.exposure, params.vignetting, params.color, params.crf, rgb_in, 0, 0);

        const auto max_diff = torch::abs(rgb_cuda - rgb_torch).max().item<float>();
        EXPECT_LT(max_diff, FORWARD_ATOL);
    }

    TEST_F(PPISPCudaVsTorchTest, ForwardMultipleIterations) {
        for (int i = 0; i < 10; ++i) {
            const auto params = createParams(2, 5, DEFAULT_SEED + i);
            const auto rgb_in = createImage(64, 64, 100 + i);

            std::mt19937 gen(i);
            const int camera_idx = std::uniform_int_distribution<>(0, 1)(gen);
            const int frame_idx = std::uniform_int_distribution<>(0, 4)(gen);

            const auto rgb_cuda = runCudaForward(params, rgb_in, 64, 64, camera_idx, frame_idx);
            const auto rgb_torch =
                ppispApplyTorch(params.exposure, params.vignetting, params.color, params.crf, rgb_in, camera_idx, frame_idx);

            EXPECT_LT(torch::abs(rgb_cuda - rgb_torch).max().item<float>(), FORWARD_ATOL);
        }
    }

    TEST_F(PPISPCudaVsTorchTest, ForwardDifferentImageSizes) {
        const std::vector<std::pair<int, int>> sizes = {{32, 32}, {64, 64}, {128, 128}, {64, 128}, {256, 256}};
        const auto params = createParams(2, 5, DEFAULT_SEED);

        for (const auto& [h, w] : sizes) {
            const auto rgb_in = createImage(h, w, DEFAULT_SEED);
            const auto rgb_cuda = runCudaForward(params, rgb_in, h, w, 0, 0);
            const auto rgb_torch = ppispApplyTorch(params.exposure, params.vignetting, params.color, params.crf, rgb_in, 0, 0);
            EXPECT_LT(torch::abs(rgb_cuda - rgb_torch).max().item<float>(), FORWARD_ATOL);
        }
    }

    TEST_F(PPISPCudaVsTorchTest, ForwardIdentityParams) {
        auto params = createParams(2, 5, DEFAULT_SEED);
        params.exposure.zero_();
        params.vignetting.zero_();
        params.color.zero_();
        params.crf.zero_();

        const auto rgb_in = createImage(64, 64, DEFAULT_SEED);
        const auto rgb_cuda = runCudaForward(params, rgb_in, 64, 64, 0, 0);
        const auto rgb_torch = ppispApplyTorch(params.exposure, params.vignetting, params.color, params.crf, rgb_in, 0, 0);

        EXPECT_LT(torch::abs(rgb_cuda - rgb_torch).max().item<float>(), FORWARD_ATOL);
    }

    TEST_F(PPISPCudaVsTorchTest, ForwardNoCameraEffects) {
        const auto params = createParams(2, 5, DEFAULT_SEED);
        const auto rgb_in = createImage(64, 64, DEFAULT_SEED);

        const auto rgb_cuda = runCudaForward(params, rgb_in, 64, 64, -1, 0);
        const auto rgb_torch = ppispApplyTorch(params.exposure, params.vignetting, params.color, params.crf, rgb_in, -1, 0);

        EXPECT_LT(torch::abs(rgb_cuda - rgb_torch).max().item<float>(), FORWARD_ATOL);
    }

    TEST_F(PPISPCudaVsTorchTest, ForwardNoFrameEffects) {
        const auto params = createParams(2, 5, DEFAULT_SEED);
        const auto rgb_in = createImage(64, 64, DEFAULT_SEED);

        const auto rgb_cuda = runCudaForward(params, rgb_in, 64, 64, 0, -1);
        const auto rgb_torch = ppispApplyTorch(params.exposure, params.vignetting, params.color, params.crf, rgb_in, 0, -1);

        EXPECT_LT(torch::abs(rgb_cuda - rgb_torch).max().item<float>(), FORWARD_ATOL);
    }

    TEST_F(PPISPCudaVsTorchTest, BackwardBasic) {
        const auto params_cuda = createParams(2, 5, DEFAULT_SEED);
        auto params_torch = createParams(2, 5, DEFAULT_SEED);
        params_torch.exposure.set_requires_grad(true);
        params_torch.vignetting.set_requires_grad(true);
        params_torch.color.set_requires_grad(true);
        params_torch.crf.set_requires_grad(true);

        const auto rgb_cuda_in = createImage(64, 64, DEFAULT_SEED);
        auto rgb_torch_in = rgb_cuda_in.clone().set_requires_grad(true);

        const auto output =
            ppispApplyTorch(params_torch.exposure, params_torch.vignetting, params_torch.color, params_torch.crf, rgb_torch_in, 0, 0);

        torch::manual_seed(123);
        const auto grad_out = torch::randn_like(output);
        output.backward(grad_out);

        const auto g = runCudaBackward(params_cuda, rgb_cuda_in, grad_out, 64, 64, 0, 0);

        EXPECT_LE(torch::abs(g.rgb_in - rgb_torch_in.grad()).max().item<float>(), BACKWARD_ATOL);
        EXPECT_LE(torch::abs(g.exposure - params_torch.exposure.grad()).max().item<float>(), BACKWARD_ATOL);
        EXPECT_LE(torch::abs(g.vignetting - params_torch.vignetting.grad()).max().item<float>(), BACKWARD_ATOL);
        EXPECT_LE(torch::abs(g.color - params_torch.color.grad()).max().item<float>(), BACKWARD_ATOL);
        EXPECT_LE(torch::abs(g.crf - params_torch.crf.grad()).max().item<float>(), BACKWARD_ATOL);
    }

    TEST_F(PPISPCudaVsTorchTest, BackwardNoCameraEffects) {
        const auto params_cuda = createParams(2, 5, DEFAULT_SEED);
        auto params_torch = createParams(2, 5, DEFAULT_SEED);
        params_torch.exposure.set_requires_grad(true);
        params_torch.color.set_requires_grad(true);

        const auto rgb_cuda_in = createImage(64, 64, DEFAULT_SEED);
        auto rgb_torch_in = rgb_cuda_in.clone().set_requires_grad(true);

        const auto output =
            ppispApplyTorch(params_torch.exposure, params_torch.vignetting, params_torch.color, params_torch.crf, rgb_torch_in, -1, 0);

        torch::manual_seed(123);
        const auto grad_out = torch::randn_like(output);
        output.backward(grad_out);

        const auto g = runCudaBackward(params_cuda, rgb_cuda_in, grad_out, 64, 64, -1, 0);

        EXPECT_LE(torch::abs(g.rgb_in - rgb_torch_in.grad()).max().item<float>(), BACKWARD_ATOL);
        EXPECT_LE(torch::abs(g.exposure - params_torch.exposure.grad()).max().item<float>(), BACKWARD_ATOL);
        EXPECT_LE(torch::abs(g.color - params_torch.color.grad()).max().item<float>(), BACKWARD_ATOL);
    }

    TEST_F(PPISPCudaVsTorchTest, BackwardNoFrameEffects) {
        const auto params_cuda = createParams(2, 5, DEFAULT_SEED);
        auto params_torch = createParams(2, 5, DEFAULT_SEED);
        params_torch.vignetting.set_requires_grad(true);
        params_torch.crf.set_requires_grad(true);

        const auto rgb_cuda_in = createImage(64, 64, DEFAULT_SEED);
        auto rgb_torch_in = rgb_cuda_in.clone().set_requires_grad(true);

        const auto output =
            ppispApplyTorch(params_torch.exposure, params_torch.vignetting, params_torch.color, params_torch.crf, rgb_torch_in, 0, -1);

        torch::manual_seed(123);
        const auto grad_out = torch::randn_like(output);
        output.backward(grad_out);

        const auto g = runCudaBackward(params_cuda, rgb_cuda_in, grad_out, 64, 64, 0, -1);

        EXPECT_LE(torch::abs(g.rgb_in - rgb_torch_in.grad()).max().item<float>(), BACKWARD_ATOL);
        EXPECT_LE(torch::abs(g.vignetting - params_torch.vignetting.grad()).max().item<float>(), BACKWARD_ATOL);
        EXPECT_LE(torch::abs(g.crf - params_torch.crf.grad()).max().item<float>(), BACKWARD_ATOL);
    }

    TEST_F(PPISPCudaVsTorchTest, LargeImage) {
        const auto params = createParams(2, 5, DEFAULT_SEED);
        const auto rgb_in = createImage(256, 256, DEFAULT_SEED);

        const auto rgb_cuda = runCudaForward(params, rgb_in, 256, 256, 0, 0);
        const auto rgb_torch = ppispApplyTorch(params.exposure, params.vignetting, params.color, params.crf, rgb_in, 0, 0);

        EXPECT_LT(torch::abs(rgb_cuda - rgb_torch).max().item<float>(), FORWARD_ATOL);
    }

    TEST_F(PPISPCudaVsTorchTest, ManyCamerasFrames) {
        constexpr int NUM_CAMERAS = 10;
        constexpr int NUM_FRAMES = 50;

        const auto params = createParams(NUM_CAMERAS, NUM_FRAMES, DEFAULT_SEED);
        const auto rgb_in = createImage(64, 64, DEFAULT_SEED);

        std::mt19937 gen(DEFAULT_SEED);
        const int camera_idx = std::uniform_int_distribution<>(0, NUM_CAMERAS - 1)(gen);
        const int frame_idx = std::uniform_int_distribution<>(0, NUM_FRAMES - 1)(gen);

        const auto rgb_cuda = runCudaForward(params, rgb_in, 64, 64, camera_idx, frame_idx);
        const auto rgb_torch =
            ppispApplyTorch(params.exposure, params.vignetting, params.color, params.crf, rgb_in, camera_idx, frame_idx);

        EXPECT_LT(torch::abs(rgb_cuda - rgb_torch).max().item<float>(), FORWARD_ATOL);
    }

    TEST_F(PPISPCudaVsTorchTest, ExtremeAndNonFiniteParametersStayFinite) {
        auto params = createParams(1, 1, DEFAULT_SEED);
        params.exposure.fill_(1000.0f);
        params.vignetting.fill_(std::numeric_limits<float>::infinity());
        params.color.fill_(std::numeric_limits<float>::quiet_NaN());
        params.crf.select(2, 0).fill_(1000.0f);
        params.crf.select(2, 1).fill_(-1000.0f);
        params.crf.select(2, 2).fill_(1000.0f);
        params.crf.select(2, 3).fill_(1000.0f);
        params.crf.select(1, 1).select(1, 3).fill_(-1000.0f);
        params.crf.select(1, 2).select(1, 3).fill_(std::numeric_limits<float>::quiet_NaN());

        const auto rgb_in = createImage(8, 8, DEFAULT_SEED);
        const auto output = runCudaForward(params, rgb_in, 8, 8, 0, 0);
        EXPECT_TRUE(torch::isfinite(output).all().item<bool>());

        const auto grads = runCudaBackward(params, rgb_in, torch::ones_like(output), 8, 8, 0, 0);
        EXPECT_TRUE(torch::isfinite(grads.rgb_in).all().item<bool>());
        EXPECT_TRUE(torch::isfinite(grads.exposure).all().item<bool>());
        EXPECT_TRUE(torch::isfinite(grads.vignetting).all().item<bool>());
        EXPECT_TRUE(torch::isfinite(grads.color).all().item<bool>());
        EXPECT_TRUE(torch::isfinite(grads.crf).all().item<bool>());
        EXPECT_EQ(grads.exposure.item<float>(), 0.0f);
    }

    TEST_F(PPISPCudaVsTorchTest, AdamSanitizesNonFiniteParametersGradientsAndState) {
        auto params = torch::tensor(
            {std::numeric_limits<float>::quiet_NaN(), 1.0f}, GPU_F32);
        auto exp_avg = torch::tensor(
            {std::numeric_limits<float>::infinity(), 0.0f}, GPU_F32);
        auto exp_avg_sq = torch::tensor({-1.0f, 0.0f}, GPU_F32);
        auto grad = torch::tensor(
            {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity()}, GPU_F32);

        lfs::training::kernels::launch_ppisp_adam_update(
            params.data_ptr<float>(), exp_avg.data_ptr<float>(), exp_avg_sq.data_ptr<float>(),
            grad.data_ptr<float>(), 2, 1.0e-3f, 0.9f, 0.999f, 10.0f, 10.0f, 1.0e-8f, nullptr);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

        EXPECT_TRUE(torch::isfinite(params).all().item<bool>());
        EXPECT_TRUE(torch::isfinite(exp_avg).all().item<bool>());
        EXPECT_TRUE(torch::isfinite(exp_avg_sq).all().item<bool>());
        EXPECT_EQ(params[0].item<float>(), 0.0f);
        EXPECT_EQ(params[1].item<float>(), 1.0f);
        EXPECT_EQ(exp_avg.abs().max().item<float>(), 0.0f);
        EXPECT_EQ(exp_avg_sq.abs().max().item<float>(), 0.0f);
    }

} // namespace

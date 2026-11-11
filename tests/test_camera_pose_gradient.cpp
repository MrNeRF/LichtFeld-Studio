/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// The FastGS camera gradient (grad_w2c, grad_cam_position) chained to a pose offset against finite differences of
// renders from the offset camera.

#include "core/camera.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "training/kernels/camera_pose_math.hpp"
#include "training/optimizer/adam_optimizer.hpp"
#include "training/rasterization/fast_rasterizer.hpp"

#include <array>
#include <cmath>
#include <cuda_runtime.h>
#include <filesystem>
#include <gtest/gtest.h>
#include <numbers>
#include <random>
#include <vector>

namespace {

    using namespace lfs::core;
    using namespace lfs::training;
    namespace cp = lfs::training::camera_pose;
    using Pose = std::array<float, 16>;

    constexpr int kWidth = 64;
    constexpr int kHeight = 48;

    bool cuda_available() {
        int count = 0;
        return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
    }

    // Eight splats of different size, orientation and view-dependent colour, about three units in front.
    SplatData make_scene() {
        constexpr size_t n = 8;
        std::mt19937 rng(7);
        std::uniform_real_distribution<float> u(-1.0f, 1.0f);
        std::vector<float> means, sh0, shN, scales, rotations, opacities;
        for (size_t i = 0; i < n; ++i) {
            means.insert(means.end(), {0.6f * u(rng), 0.45f * u(rng), 0.5f * u(rng)});
            sh0.insert(sh0.end(), {u(rng), u(rng), u(rng)});
            for (int k = 0; k < 9; ++k)
                shN.push_back(0.4f * u(rng));
            scales.insert(scales.end(), {-2.2f + 0.4f * u(rng), -2.0f + 0.4f * u(rng), -2.6f + 0.3f * u(rng)});
            rotations.insert(rotations.end(), {1.0f, 0.4f * u(rng), 0.4f * u(rng), 0.4f * u(rng)});
            opacities.push_back(0.5f + u(rng));
        }
        return SplatData(1,
                         Tensor::from_vector(means, {n, size_t{3}}, Device::CUDA),
                         Tensor::from_vector(sh0, {n, size_t{1}, size_t{3}}, Device::CUDA),
                         Tensor::from_vector(shN, {n, size_t{3}, size_t{3}}, Device::CUDA),
                         Tensor::from_vector(scales, {n, size_t{3}}, Device::CUDA),
                         Tensor::from_vector(rotations, {n, size_t{4}}, Device::CUDA),
                         Tensor::from_vector(opacities, {n, size_t{1}}, Device::CUDA),
                         1.0f);
    }

    Pose start_pose() {
        const float phi[3] = {0.05f, -0.08f, 0.03f};
        float E[9];
        cp::exp_so3(phi, E);
        return {E[0], E[1], E[2], 0.1f, E[3], E[4], E[5], -0.05f, E[6], E[7], E[8], 3.0f, 0.0f, 0.0f, 0.0f, 1.0f};
    }

    Pose offset_pose(const std::array<float, 6>& off) {
        const Pose start = start_pose();
        Pose w2c;
        float centre[3];
        cp::compose(start.data(), off.data(), w2c.data(), centre);
        return w2c;
    }

    Camera make_camera(const Pose& w2c) {
        auto R = Tensor::from_vector({w2c[0], w2c[1], w2c[2], w2c[4], w2c[5], w2c[6], w2c[8], w2c[9], w2c[10]},
                                     {3, 3}, Device::CPU);
        auto T = Tensor::from_vector({w2c[3], w2c[7], w2c[11]}, {3}, Device::CPU);
        return Camera(R, T, 70.0f, 70.0f, kWidth / 2.0f, kHeight / 2.0f, Tensor(), Tensor(), CameraModelType::PINHOLE,
                      "pose_gradient", "", std::filesystem::path{}, kWidth, kHeight, 0);
    }

    // Low-frequency loss weights [3, H, W]. With per-pixel random weights the finite difference is dominated by
    // pixels crossing the 1/255 alpha cutoff, which the rasterizer's gradient leaves out by design.
    std::vector<float> weights() {
        std::vector<float> w(3 * kWidth * kHeight);
        for (int c = 0; c < 3; ++c)
            for (int y = 0; y < kHeight; ++y)
                for (int x = 0; x < kWidth; ++x)
                    w[(c * kHeight + y) * kWidth + x] =
                        static_cast<float>(std::sin(2.0 * std::numbers::pi * (x + 0.5) / kWidth + 1.3 * c) *
                                           std::cos(2.0 * std::numbers::pi * (y + 0.5) / kHeight - 0.7 * c));
        return w;
    }

    double dot(const Tensor& image, const std::vector<float>& w) {
        const auto values = image.cpu().contiguous().to_vector();
        double sum = 0.0;
        for (size_t i = 0; i < values.size(); ++i)
            sum += double(values[i]) * w[i];
        return sum;
    }

    // L(offset) = <image, image_w> + <normal, normal_w> rendered from compose(start, offset).
    double render_loss(const std::array<float, 6>& off, const std::vector<float>& image_w,
                       const std::vector<float>& normal_w) {
        auto model = make_scene();
        auto camera = make_camera(offset_pose(off));
        auto bg = Tensor::full({3}, 0.3f, Device::CUDA);
        const bool normals = !normal_w.empty();
        auto forward = fast_rasterize_forward(camera, model, bg, 0, 0, 0, 0, false, {}, normals, true, 1.0f, false);
        EXPECT_TRUE(forward.has_value());
        if (!forward)
            return 0.0;
        double loss = image_w.empty() ? 0.0 : dot(forward->first.image, image_w);
        if (normals)
            loss += dot(forward->first.normal, normal_w);
        forward->second.release_forward_context();
        return loss;
    }

    void expect_matches_finite_differences(const std::array<float, 6>& analytic, const std::vector<float>& image_w,
                                           const std::vector<float>& normal_w) {
        constexpr float h = 1e-3f;
        std::array<double, 6> numeric{};
        double scale = 0.0;
        for (int k = 0; k < 6; ++k) {
            std::array<float, 6> plus{}, minus{};
            plus[k] = h;
            minus[k] = -h;
            numeric[k] = (render_loss(plus, image_w, normal_w) - render_loss(minus, image_w, normal_w)) / (2.0 * h);
            scale = std::max(scale, std::abs(numeric[k]));
        }
        ASSERT_GT(scale, 1e-3) << "the test scene must respond to the camera pose";
        for (int k = 0; k < 6; ++k)
            EXPECT_NEAR(analytic[k], numeric[k], 0.03 * scale + 0.03 * std::abs(numeric[k])) << "component " << k;
    }

    std::array<float, 6> offset_gradient(const Tensor& grad_w2c, const Tensor& grad_cam_position) {
        EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        const auto g_w2c = grad_w2c.cpu().to_vector();
        const auto g_c = grad_cam_position.cpu().to_vector();
        const Pose start = start_pose();
        const std::array<float, 6> zero{};
        std::array<float, 6> out{};
        cp::offset_grad(start.data(), zero.data(), g_w2c.data(), g_c.data(), out.data());
        return out;
    }

} // namespace

TEST(CameraPoseGradientTest, FrozenImageGradientMatchesFiniteDifferences) {
    if (!cuda_available())
        GTEST_SKIP() << "CUDA not available";
    const auto image_w = weights();
    auto model = make_scene();
    const auto means_before = model.means().cpu().to_vector();
    const auto opacity_before = model.opacity_raw().cpu().to_vector();

    auto camera = make_camera(start_pose());
    auto bg = Tensor::full({3}, 0.3f, Device::CUDA);
    auto forward = fast_rasterize_forward(camera, model, bg, 0, 0, 0, 0, false, {}, false, true, 1.0f, false);
    ASSERT_TRUE(forward.has_value()) << lfs::format_for_developer(forward.error());
    auto grad_image = Tensor::from_vector(image_w, {size_t{3}, size_t{kHeight}, size_t{kWidth}}, Device::CUDA);
    auto grad_w2c = Tensor::zeros({16}, Device::CUDA);
    auto grad_cam_position = Tensor::zeros({3}, Device::CUDA);
    fast_rasterize_camera_backward(forward->second, grad_image, model, grad_w2c.ptr<float>(),
                                   grad_cam_position.ptr<float>());
    const auto analytic = offset_gradient(grad_w2c, grad_cam_position);

    EXPECT_EQ(model.means().cpu().to_vector(), means_before) << "the camera backward must not move the model";
    EXPECT_EQ(model.opacity_raw().cpu().to_vector(), opacity_before);
    expect_matches_finite_differences(analytic, image_w, {});
}

TEST(CameraPoseGradientTest, TrainingBackwardNormalGradientMatchesFiniteDifferences) {
    if (!cuda_available())
        GTEST_SKIP() << "CUDA not available";
    const auto normal_w = weights();
    auto model = make_scene();
    auto camera = make_camera(start_pose());
    auto bg = Tensor::full({3}, 0.3f, Device::CUDA);
    auto forward = fast_rasterize_forward(camera, model, bg, 0, 0, 0, 0, false, {}, true, true, 1.0f, false);
    ASSERT_TRUE(forward.has_value()) << lfs::format_for_developer(forward.error());

    AdamConfig cfg{.lr = 0.001f, .beta1 = 0.9, .beta2 = 0.999, .eps = 1e-15};
    AdamOptimizer optimizer(model, cfg);
    optimizer.allocate_gradients();
    optimizer.zero_grad(0);
    auto grad_w2c = Tensor::zeros({16}, Device::CUDA);
    auto grad_cam_position = Tensor::zeros({3}, Device::CUDA);
    FastGSFusedExtraGradients extra;
    extra.grad_w2c = grad_w2c.ptr<float>();
    extra.grad_cam_position = grad_cam_position.ptr<float>();
    auto grad_normal = Tensor::from_vector(normal_w, {size_t{3}, size_t{kHeight}, size_t{kWidth}}, Device::CUDA);
    fast_rasterize_backward(forward->second, Tensor::zeros_like(forward->first.image), model, optimizer, {}, {},
                            DensificationType::None, 1, extra, {}, grad_normal);
    const auto analytic = offset_gradient(grad_w2c, grad_cam_position);

    expect_matches_finite_differences(analytic, {}, normal_w);
}

/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/camera.hpp"
#include "core/cuda/memory_arena.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "io/formats/ply.hpp"
#include "rasterization/fastgs/utils/utils.h"
#include "training/optimizer/adam_optimizer.hpp"
#include "training/rasterization/fast_rasterizer.hpp"
#include <cstdint>
#include <cuda_runtime.h>
#include <filesystem>
#include <gtest/gtest.h>
#include <limits>
#include <random>
#include <stdexcept>
#include <torch/torch.h>

using namespace lfs::training;
using namespace lfs::core;

namespace {
    constexpr const char* GARDEN_PATH = "data/garden";
    constexpr int W = 640, H = 480;
    constexpr float FX = 500.0f, FY = 500.0f;

    const AdamParamState& adam_state(const AdamOptimizer& opt, ParamType type) {
        const auto* state = opt.get_state(type);
        if (!state || !state->exp_avg.is_valid()) {
            throw std::runtime_error("Missing Adam moment state");
        }
        return *state;
    }

    Tensor adam_moment(const AdamOptimizer& opt, ParamType type) {
        const auto& state = adam_state(opt, type);
        if (state.exp_avg.dtype() == DataType::Float32) {
            return state.exp_avg;
        }

        auto q_cpu = state.exp_avg.to(Device::CPU);
        auto scale_cpu = state.exp_avg_scale.to(Device::CPU);
        const auto& shape = state.exp_avg.shape();
        const size_t rows = shape[0];
        const size_t row_size = rows == 0 ? 0 : state.exp_avg.numel() / rows;
        const auto* q = q_cpu.ptr<std::uint8_t>();
        const auto* scales = scale_cpu.ptr<float>();
        std::vector<float> dequant(state.exp_avg.numel(), 0.0f);
        for (size_t row = 0; row < rows; ++row) {
            const float scale = scales[row];
            for (size_t col = 0; col < row_size; ++col) {
                const size_t idx = row * row_size + col;
                dequant[idx] = scale == 0.0f ? 0.0f : (static_cast<int>(q[idx]) - 128) * scale;
            }
        }
        return Tensor::from_blob(dequant.data(), shape, Device::CPU, DataType::Float32).clone().to(state.exp_avg.device());
    }

    void expect_adam_state_finite(const AdamOptimizer& opt, ParamType type) {
        const auto& state = adam_state(opt, type);
        if (state.exp_avg_scale.is_valid()) {
            auto scales = state.exp_avg_scale.to(Device::CPU);
            auto* ptr = scales.ptr<float>();
            for (size_t i = 0; i < scales.numel(); ++i) {
                EXPECT_TRUE(std::isfinite(ptr[i]));
            }
        }
        if (state.exp_avg_sq_scale.is_valid()) {
            auto scales = state.exp_avg_sq_scale.to(Device::CPU);
            auto* ptr = scales.ptr<float>();
            for (size_t i = 0; i < scales.numel(); ++i) {
                EXPECT_TRUE(std::isfinite(ptr[i]));
            }
        }
    }

    Tensor recovered_fused_grad(const AdamOptimizer& opt, ParamType type, float beta1 = 0.9f) {
        return adam_moment(opt, type).mul(1.0f / (1.0f - beta1));
    }
} // namespace

TEST(FastGSOverflowGuards, RejectsInstanceCountsBeyondIntRange) {
    const uint64_t max_int = static_cast<uint64_t>(std::numeric_limits<int>::max());
    EXPECT_EQ(checked_fastgs_instance_count(max_int, 1, 1), std::numeric_limits<int>::max());
    EXPECT_THROW(
        checked_fastgs_instance_count(max_int + 1, 595037, 11907),
        std::overflow_error);
}

class FastGSKernelTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!std::filesystem::exists(GARDEN_PATH)) {
            GTEST_SKIP() << "Garden dataset not found";
        }

        std::string ply = std::string(GARDEN_PATH) + "/point_cloud/iteration_7000/point_cloud.ply";
        if (std::filesystem::exists(ply)) {
            load_ply(ply);
        } else {
            create_synthetic_data();
        }

        auto R = Tensor::eye(3, Device::CUDA);
        std::vector<float> t_data{0, 0, 5};
        auto T = Tensor::from_blob(t_data.data(), {3}, Device::CPU, DataType::Float32).to(Device::CUDA);
        camera_ = std::make_unique<Camera>(R, T, FX, FY, W / 2.0f, H / 2.0f,
                                           Tensor(), Tensor(), CameraModelType::PINHOLE,
                                           "test", "", std::filesystem::path{}, W, H, 0);
        bg_ = Tensor::zeros({3}, Device::CUDA);
    }

    void TearDown() override {
        splat_.reset();
        camera_.reset();
        GlobalArenaManager::instance().get_arena().full_reset();
    }

    void load_ply(const std::string& path) {
        auto result = lfs::io::load_ply(path);
        if (!result) {
            create_synthetic_data();
            return;
        }
        n_ = std::min(result->means().shape()[0], size_t(10000));
        means_ = result->means().slice(0, 0, n_).contiguous().to(Device::CUDA);
        init_params();
    }

    void create_synthetic_data() {
        n_ = 10000;
        std::vector<float> data(n_ * 3);
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> xy(-5, 5), z(-1, 3);
        for (size_t i = 0; i < n_; ++i) {
            data[i * 3] = xy(gen);
            data[i * 3 + 1] = xy(gen);
            data[i * 3 + 2] = z(gen);
        }
        means_ = Tensor::from_blob(data.data(), {n_, 3}, Device::CPU, DataType::Float32).to(Device::CUDA);
        init_params();
    }

    void init_params() {
        sh0_ = Tensor::randn({n_, 1, 3}, Device::CUDA).mul(0.5f);
        shN_ = Tensor::zeros({n_, 0, 3}, Device::CUDA);
        scaling_ = Tensor::randn({n_, 3}, Device::CUDA).mul(0.3f).sub(3.5f);
        rotation_ = Tensor::randn({n_, 4}, Device::CUDA);
        rotation_ = rotation_ / rotation_.pow(2.0f).sum(-1, true).sqrt();
        opacity_ = Tensor::randn({n_}, Device::CUDA).mul(2.0f);
        splat_ = std::make_unique<SplatData>(0, means_, sh0_, shN_, scaling_, rotation_, opacity_, 1.0f);
    }

    std::unique_ptr<AdamOptimizer> make_optimizer() {
        AdamConfig cfg{.lr = 0.001f, .beta1 = 0.9, .beta2 = 0.999, .eps = 1e-15};
        auto opt = std::make_unique<AdamOptimizer>(*splat_, cfg);
        opt->allocate_gradients();
        return opt;
    }

    auto forward() { return fast_rasterize_forward(*camera_, *splat_, bg_, 0, 0, 0, 0, false); }

    size_t n_ = 0;
    std::unique_ptr<SplatData> splat_;
    std::unique_ptr<Camera> camera_;
    Tensor means_, sh0_, shN_, scaling_, rotation_, opacity_, bg_;
};

// Forward kernels
TEST_F(FastGSKernelTest, Forward_Preprocess) {
    auto r = forward();
    ASSERT_TRUE(r.has_value()) << r.error();
    EXPECT_GT(r->second.forward_ctx.n_instances, 0);
}

TEST_F(FastGSKernelTest, Forward_TileDepthOrdering) {
    auto r = forward();
    ASSERT_TRUE(r.has_value());
    EXPECT_TRUE(r->first.image.is_valid());
}

TEST_F(FastGSKernelTest, Forward_Instances) {
    auto r = forward();
    ASSERT_TRUE(r.has_value());
    EXPECT_GT(r->second.forward_ctx.n_instances, 0);
}

TEST_F(FastGSKernelTest, Forward_TileState) {
    auto r = forward();
    ASSERT_TRUE(r.has_value());
    EXPECT_GT(r->second.forward_ctx.per_tile_buffers_size, 0);
}

TEST_F(FastGSKernelTest, Forward_Blend) {
    auto r = forward();
    ASSERT_TRUE(r.has_value());
    auto& out = r->first;

    EXPECT_EQ(out.image.ndim(), 3);
    EXPECT_EQ(out.image.shape()[0], 3);
    EXPECT_EQ(out.image.shape()[1], static_cast<size_t>(H));
    EXPECT_EQ(out.image.shape()[2], static_cast<size_t>(W));

    float alpha_min = out.alpha.min().item<float>();
    float alpha_max = out.alpha.max().item<float>();
    EXPECT_GE(alpha_min, 0.0f);
    EXPECT_LE(alpha_max, 1.0f);
    EXPECT_GT(out.image.std().item<float>(), 0.0f);
}

TEST_F(FastGSKernelTest, Forward_Full) {
    auto r = forward();
    ASSERT_TRUE(r.has_value()) << r.error();
    EXPECT_EQ(r->first.width, W);
    EXPECT_EQ(r->first.height, H);
}

// Backward kernels
TEST_F(FastGSKernelTest, Backward_Blend) {
    auto r = forward();
    ASSERT_TRUE(r.has_value());

    auto opt = make_optimizer();
    opt->zero_grad(0);
    fast_rasterize_backward(r->second, Tensor::ones_like(r->first.image),
                            *splat_, *opt, Tensor::zeros_like(r->first.alpha));

    EXPECT_GT(adam_moment(*opt, ParamType::Means).pow(2.0f).sum().item<float>(), 0.0f);
    EXPECT_GT(adam_moment(*opt, ParamType::Scaling).pow(2.0f).sum().item<float>(), 0.0f);
}

TEST(FastGSDepthGradientTest, BackwardDepthMatchesLibtorchAutogradForCenteredSplat) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA not available";
    }

    std::vector<float> means_data{0.0f, 0.0f, 1.0f};
    auto means = Tensor::from_blob(means_data.data(), {1, 3}, Device::CPU, DataType::Float32).to(Device::CUDA);
    auto sh0 = Tensor::zeros({1, 1, 3}, Device::CUDA);
    auto shN = Tensor::zeros({1, 0, 3}, Device::CUDA);
    auto scaling = Tensor::full({1, 3}, -1.5f, Device::CUDA);
    std::vector<float> rotation_data{1.0f, 0.0f, 0.0f, 0.0f};
    auto rotation = Tensor::from_blob(rotation_data.data(), {1, 4}, Device::CPU, DataType::Float32).to(Device::CUDA);

    const float opacity_value = 0.3f;
    const float raw_opacity_value = std::log(opacity_value / (1.0f - opacity_value));
    auto opacity = Tensor::full({1}, raw_opacity_value, Device::CUDA);
    auto splat = SplatData(0, means, sh0, shN, scaling, rotation, opacity, 1.0f);

    auto R = Tensor::eye(3, Device::CUDA);
    std::vector<float> t_data{0.0f, 0.0f, 4.0f};
    auto T = Tensor::from_blob(t_data.data(), {3}, Device::CPU, DataType::Float32).to(Device::CUDA);
    auto camera = Camera(R, T, 1.0f, 1.0f, 0.5f, 0.5f,
                         Tensor(), Tensor(), CameraModelType::PINHOLE,
                         "depth_grad", "", std::filesystem::path{}, 1, 1, 0);
    auto bg = Tensor::zeros({3}, Device::CUDA);

    auto forward = fast_rasterize_forward(camera, splat, bg, 0, 0, 0, 0, false);
    ASSERT_TRUE(forward.has_value()) << forward.error();
    ASSERT_EQ(forward->first.depth.numel(), 1);
    EXPECT_GT(forward->first.depth.item<float>(), 0.0f);

    AdamConfig cfg{.lr = 0.001f, .beta1 = 0.9, .beta2 = 0.999, .eps = 1e-15};
    AdamOptimizer opt(splat, cfg);
    opt.allocate_gradients();
    opt.zero_grad(0);

    const float upstream_depth_grad = 1.7f;
    auto grad_image = Tensor::zeros_like(forward->first.image);
    auto grad_depth = Tensor::full({1, 1}, upstream_depth_grad, Device::CUDA);
    fast_rasterize_backward(
        forward->second,
        grad_image,
        splat,
        opt,
        {},
        {},
        DensificationType::None,
        1,
        {},
        grad_depth);

    const auto mean_grad = recovered_fused_grad(opt, ParamType::Means).to(Device::CPU);
    const auto opacity_grad = recovered_fused_grad(opt, ParamType::Opacity).to(Device::CPU);

    auto raw_opacity_ag = torch::tensor({raw_opacity_value}, torch::dtype(torch::kFloat32).device(torch::kCUDA))
                              .set_requires_grad(true);
    auto depth_ag = torch::tensor({means_data[2] + t_data[2]}, torch::dtype(torch::kFloat32).device(torch::kCUDA))
                        .set_requires_grad(true);
    auto depth_out = torch::sigmoid(raw_opacity_ag) * depth_ag;
    auto loss = depth_out * upstream_depth_grad;
    loss.backward();

    const float expected_opacity_grad = raw_opacity_ag.grad().item<float>();
    const float expected_depth_grad = depth_ag.grad().item<float>();
    const float actual_opacity_grad = opacity_grad.ptr<float>()[0];
    const float actual_mean_z_grad = mean_grad.ptr<float>()[2];

    EXPECT_NEAR(actual_opacity_grad, expected_opacity_grad, 1.0e-4f)
        << "raw opacity depth gradient should match libtorch autograd";
    EXPECT_NEAR(actual_mean_z_grad, expected_depth_grad, 1.0e-4f)
        << "mean z depth gradient should match libtorch autograd";
    EXPECT_NEAR(mean_grad.ptr<float>()[0], 0.0f, 1.0e-5f);
    EXPECT_NEAR(mean_grad.ptr<float>()[1], 0.0f, 1.0e-5f);
}

TEST(FastGSDepthGradientTest, BackwardDepthMatchesLibtorchAutogradForOverlappingSplats) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA not available";
    }

    std::vector<float> means_data{
        0.0f, 0.0f, 0.5f,
        0.0f, 0.0f, 1.5f};
    auto means = Tensor::from_blob(means_data.data(), {2, 3}, Device::CPU, DataType::Float32).to(Device::CUDA);
    auto sh0 = Tensor::zeros({2, 1, 3}, Device::CUDA);
    auto shN = Tensor::zeros({2, 0, 3}, Device::CUDA);
    auto scaling = Tensor::full({2, 3}, -1.5f, Device::CUDA);
    std::vector<float> rotation_data{
        1.0f, 0.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f, 0.0f};
    auto rotation = Tensor::from_blob(rotation_data.data(), {2, 4}, Device::CPU, DataType::Float32).to(Device::CUDA);

    const std::vector<float> opacity_values{0.25f, 0.4f};
    std::vector<float> raw_opacity_values{
        std::log(opacity_values[0] / (1.0f - opacity_values[0])),
        std::log(opacity_values[1] / (1.0f - opacity_values[1]))};
    auto opacity = Tensor::from_blob(raw_opacity_values.data(), {2}, Device::CPU, DataType::Float32).to(Device::CUDA);
    auto splat = SplatData(0, means, sh0, shN, scaling, rotation, opacity, 1.0f);

    auto R = Tensor::eye(3, Device::CUDA);
    std::vector<float> t_data{0.0f, 0.0f, 4.0f};
    auto T = Tensor::from_blob(t_data.data(), {3}, Device::CPU, DataType::Float32).to(Device::CUDA);
    auto camera = Camera(R, T, 1.0f, 1.0f, 0.5f, 0.5f,
                         Tensor(), Tensor(), CameraModelType::PINHOLE,
                         "depth_grad_overlap", "", std::filesystem::path{}, 1, 1, 0);
    auto bg = Tensor::zeros({3}, Device::CUDA);

    auto forward = fast_rasterize_forward(camera, splat, bg, 0, 0, 0, 0, false);
    ASSERT_TRUE(forward.has_value()) << forward.error();
    ASSERT_EQ(forward->first.depth.numel(), 1);

    const float depth0 = means_data[2] + t_data[2];
    const float depth1 = means_data[5] + t_data[2];
    const float expected_forward_depth =
        opacity_values[0] * depth0 +
        (1.0f - opacity_values[0]) * opacity_values[1] * depth1;
    EXPECT_NEAR(forward->first.depth.item<float>(), expected_forward_depth, 1.0e-4f)
        << "test setup should render the nearer splat first";

    AdamConfig cfg{.lr = 0.001f, .beta1 = 0.9, .beta2 = 0.999, .eps = 1e-15};
    AdamOptimizer opt(splat, cfg);
    opt.allocate_gradients();
    opt.zero_grad(0);

    const float upstream_depth_grad = 1.3f;
    auto grad_image = Tensor::zeros_like(forward->first.image);
    auto grad_depth = Tensor::full({1, 1}, upstream_depth_grad, Device::CUDA);
    fast_rasterize_backward(
        forward->second,
        grad_image,
        splat,
        opt,
        {},
        {},
        DensificationType::None,
        1,
        {},
        grad_depth);

    const auto mean_grad = recovered_fused_grad(opt, ParamType::Means).to(Device::CPU);
    const auto opacity_grad = recovered_fused_grad(opt, ParamType::Opacity).to(Device::CPU);

    const auto opts = torch::dtype(torch::kFloat32).device(torch::kCUDA);
    auto raw_opacity_ag = torch::tensor(raw_opacity_values, opts).clone().set_requires_grad(true);
    auto depth_ag = torch::tensor({depth0, depth1}, opts).clone().set_requires_grad(true);
    const auto alpha = torch::sigmoid(raw_opacity_ag);
    const auto depth_out =
        alpha.select(0, 0) * depth_ag.select(0, 0) +
        (1.0f - alpha.select(0, 0)) * alpha.select(0, 1) * depth_ag.select(0, 1);
    const auto loss = depth_out * upstream_depth_grad;
    loss.backward();

    const auto expected_opacity_grad = raw_opacity_ag.grad().to(torch::kCPU);
    const auto expected_depth_grad = depth_ag.grad().to(torch::kCPU);
    const float* actual_opacity_grad = opacity_grad.ptr<float>();
    const float* actual_mean_grad = mean_grad.ptr<float>();

    for (int i = 0; i < 2; ++i) {
        EXPECT_NEAR(actual_opacity_grad[i], expected_opacity_grad[i].item<float>(), 1.0e-4f)
            << "raw opacity depth gradient mismatch for splat " << i;
        EXPECT_NEAR(actual_mean_grad[i * 3 + 2], expected_depth_grad[i].item<float>(), 1.0e-4f)
            << "mean z depth gradient mismatch for splat " << i;
        EXPECT_NEAR(actual_mean_grad[i * 3], 0.0f, 1.0e-5f);
        EXPECT_NEAR(actual_mean_grad[i * 3 + 1], 0.0f, 1.0e-5f);
    }
}

namespace {
    struct NormalChannelScene {
        std::vector<float> means_data{0.0f, 0.0f, 1.0f};
        // Distinct raw log-scales with z clearly smallest so the normal axis
        // (argmin variance) is stable under the perturbations below.
        std::vector<float> scaling_data{-1.0f, -1.5f, -3.0f};
        float opacity_value = 0.3f;
        std::vector<float> t_data{0.0f, 0.0f, 4.0f};

        SplatData make_splat(const std::vector<float>& rotation_data) const {
            auto means = Tensor::from_blob(const_cast<float*>(means_data.data()), {1, 3}, Device::CPU, DataType::Float32).to(Device::CUDA);
            auto sh0 = Tensor::zeros({1, 1, 3}, Device::CUDA);
            auto shN = Tensor::zeros({1, 0, 3}, Device::CUDA);
            auto scaling = Tensor::from_blob(const_cast<float*>(scaling_data.data()), {1, 3}, Device::CPU, DataType::Float32).to(Device::CUDA);
            auto rotation = Tensor::from_blob(const_cast<float*>(rotation_data.data()), {1, 4}, Device::CPU, DataType::Float32).to(Device::CUDA);
            const float raw_opacity = std::log(opacity_value / (1.0f - opacity_value));
            auto opacity = Tensor::full({1}, raw_opacity, Device::CUDA);
            return SplatData(0, means, sh0, shN, scaling, rotation, opacity, 1.0f);
        }

        Camera make_camera() const {
            auto R = Tensor::eye(3, Device::CUDA);
            auto T = Tensor::from_blob(const_cast<float*>(t_data.data()), {3}, Device::CPU, DataType::Float32).to(Device::CUDA);
            return Camera(R, T, 1.0f, 1.0f, 0.5f, 0.5f,
                          Tensor(), Tensor(), CameraModelType::PINHOLE,
                          "normal_grad", "", std::filesystem::path{}, 1, 1, 0);
        }
    };
} // namespace

TEST(FastGSNormalChannelTest, RendersCameraSpaceNormalForCenteredSplat) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA not available";
    }

    NormalChannelScene scene;
    const std::vector<float> identity_quat{1.0f, 0.0f, 0.0f, 0.0f};
    auto splat = scene.make_splat(identity_quat);
    auto camera = scene.make_camera();
    auto bg = Tensor::zeros({3}, Device::CUDA);

    auto without_normal = fast_rasterize_forward(camera, splat, bg, 0, 0, 0, 0, false);
    ASSERT_TRUE(without_normal.has_value()) << without_normal.error();
    EXPECT_FALSE(without_normal->first.normal.is_valid())
        << "normal channel must stay off unless requested";
    without_normal->second.release_forward_context();

    auto forward = fast_rasterize_forward(camera, splat, bg, 0, 0, 0, 0, false, Tensor{}, true);
    ASSERT_TRUE(forward.has_value()) << forward.error();
    ASSERT_TRUE(forward->first.normal.is_valid());
    ASSERT_EQ(forward->first.normal.numel(), 3);

    // Identity rotation, z the smallest axis: world normal is -z (oriented
    // toward the camera at -4z), identity w2c keeps it in place, and the
    // pixel-centered splat blends with weight alpha.
    const auto normal_cpu = forward->first.normal.to(Device::CPU);
    const float* n = normal_cpu.ptr<float>();
    EXPECT_NEAR(n[0], 0.0f, 1.0e-5f);
    EXPECT_NEAR(n[1], 0.0f, 1.0e-5f);
    EXPECT_NEAR(n[2], -scene.opacity_value, 1.0e-4f);
    forward->second.release_forward_context();
}

TEST(FastGSNormalChannelTest, BackwardNormalRotationGradientMatchesFiniteDifferences) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA not available";
    }

    NormalChannelScene scene;
    // Generic quaternion away from symmetry; keeps the smallest-axis column
    // pointed toward the camera so the orientation sign stays fixed.
    const std::vector<float> base_quat{0.95f, 0.15f, -0.1f, 0.05f};
    const std::vector<float> upstream{0.7f, -0.4f, 1.1f};
    auto camera = scene.make_camera();
    auto bg = Tensor::zeros({3}, Device::CUDA);

    const auto render_loss = [&](const std::vector<float>& quat) {
        auto splat = scene.make_splat(quat);
        auto forward = fast_rasterize_forward(camera, splat, bg, 0, 0, 0, 0, false, Tensor{}, true);
        if (!forward.has_value()) {
            throw std::runtime_error(forward.error());
        }
        const auto normal_cpu = forward->first.normal.to(Device::CPU);
        const float* n = normal_cpu.ptr<float>();
        const float loss = upstream[0] * n[0] + upstream[1] * n[1] + upstream[2] * n[2];
        forward->second.release_forward_context();
        return loss;
    };

    auto splat = scene.make_splat(base_quat);
    auto forward = fast_rasterize_forward(camera, splat, bg, 0, 0, 0, 0, false, Tensor{}, true);
    ASSERT_TRUE(forward.has_value()) << forward.error();

    AdamConfig cfg{.lr = 0.001f, .beta1 = 0.9, .beta2 = 0.999, .eps = 1e-15};
    AdamOptimizer opt(splat, cfg);
    opt.allocate_gradients();
    opt.zero_grad(0);

    std::vector<float> upstream_data = upstream;
    auto grad_image = Tensor::zeros_like(forward->first.image);
    auto grad_normal = Tensor::from_blob(upstream_data.data(), {3, 1, 1}, Device::CPU, DataType::Float32).to(Device::CUDA);
    fast_rasterize_backward(
        forward->second,
        grad_image,
        splat,
        opt,
        {},
        {},
        DensificationType::None,
        1,
        {},
        {},
        grad_normal);

    const auto rotation_grad = recovered_fused_grad(opt, ParamType::Rotation).to(Device::CPU);
    const float* actual = rotation_grad.ptr<float>();

    // The splat sits exactly on the pixel center, so the blend weight has zero
    // derivative w.r.t. rotation there and central differences through the full
    // forward isolate exactly the detached-weight value path the kernel emits.
    const float h = 2.0e-2f;
    for (int c = 0; c < 4; ++c) {
        std::vector<float> plus = base_quat;
        std::vector<float> minus = base_quat;
        plus[c] += h;
        minus[c] -= h;
        const float expected = (render_loss(plus) - render_loss(minus)) / (2.0f * h);
        EXPECT_NEAR(actual[c], expected, std::max(2.0e-3f, std::abs(expected) * 2.0e-2f))
            << "rotation gradient mismatch for quaternion component " << c;
    }
}

TEST_F(FastGSKernelTest, Backward_Preprocess) {
    auto r = forward();
    ASSERT_TRUE(r.has_value());

    auto opt = make_optimizer();
    opt->zero_grad(0);
    fast_rasterize_backward(r->second, Tensor::randn_like(r->first.image).mul(0.1f),
                            *splat_, *opt, Tensor::randn_like(r->first.alpha).mul(0.1f));

    EXPECT_TRUE(adam_moment(*opt, ParamType::Means).is_valid());
    EXPECT_TRUE(adam_moment(*opt, ParamType::Rotation).is_valid());
}

TEST_F(FastGSKernelTest, Backward_Full) {
    auto r = forward();
    ASSERT_TRUE(r.has_value());

    auto target = Tensor::randn_like(r->first.image).mul(0.5f).add(0.5f);
    auto grad = (r->first.image - target).mul(2.0f / static_cast<float>(r->first.image.numel()));

    auto opt = make_optimizer();
    opt->zero_grad(0);
    ASSERT_NO_THROW(fast_rasterize_backward(r->second, grad, *splat_, *opt, {}));
}

// Optimizer kernels
TEST_F(FastGSKernelTest, Optimizer_AdamStep) {
    auto opt = make_optimizer();
    opt->zero_grad(0);
    opt->get_grad(ParamType::Scaling).fill_(0.01f);

    auto before = splat_->scaling_raw().clone();
    opt->step(1);
    auto diff = (splat_->scaling_raw() - before).abs().sum().item<float>();

    EXPECT_GT(diff, 0.0f);
}

TEST_F(FastGSKernelTest, Optimizer_ZeroRows) {
    auto opt = make_optimizer();
    opt->get_grad(ParamType::Means).fill_(1.0f);
    opt->step(1);

    std::vector<int64_t> indices = {0, 1, 2, 3, 4};
    ASSERT_NO_THROW(opt->reset_state_at_indices(ParamType::Means, indices));
}

// Numerical tests
TEST_F(FastGSKernelTest, Numerical_Deterministic) {
    auto r1 = forward();
    ASSERT_TRUE(r1.has_value());
    auto image1 = r1->first.image.clone();
    r1->second.release_forward_context();

    auto r2 = forward();
    ASSERT_TRUE(r2.has_value());

    float diff = (image1 - r2->first.image).abs().max().item<float>();
    EXPECT_LT(diff, 1e-5f);
}

TEST_F(FastGSKernelTest, Numerical_GradientFinite) {
    auto r = forward();
    ASSERT_TRUE(r.has_value());

    auto opt = make_optimizer();
    opt->zero_grad(0);
    fast_rasterize_backward(r->second, Tensor::randn_like(r->first.image), *splat_, *opt, {});

    auto check = [](const Tensor& t) {
        auto cpu = t.to(Device::CPU);
        auto p = cpu.ptr<float>();
        for (size_t i = 0; i < std::min(t.numel(), size_t(1000)); ++i) {
            EXPECT_TRUE(std::isfinite(p[i]));
        }
    };

    check(adam_moment(*opt, ParamType::Means));
    check(adam_moment(*opt, ParamType::Scaling));
    check(adam_moment(*opt, ParamType::Rotation));
    check(adam_moment(*opt, ParamType::Opacity));
    check(adam_moment(*opt, ParamType::Sh0));
}

// Edge cases
TEST_F(FastGSKernelTest, EdgeCase_SingleGaussian) {
    n_ = 1;
    means_ = Tensor::zeros({1, 3}, Device::CUDA);
    sh0_ = Tensor::zeros({1, 1, 3}, Device::CUDA);
    shN_ = Tensor::zeros({1, 0, 3}, Device::CUDA);
    scaling_ = Tensor::full({1, 3}, -5.0f, Device::CUDA);
    rotation_ = Tensor::zeros({1, 4}, Device::CUDA);
    rotation_.slice(1, 0, 1).fill_(1.0f);
    opacity_ = Tensor::zeros({1}, Device::CUDA);
    splat_ = std::make_unique<SplatData>(0, means_, sh0_, shN_, scaling_, rotation_, opacity_, 1.0f);

    auto r = forward();
    ASSERT_TRUE(r.has_value());
}

TEST_F(FastGSKernelTest, EdgeCase_LargeGaussians) {
    scaling_.fill_(2.0f);
    splat_ = std::make_unique<SplatData>(0, means_, sh0_, shN_, scaling_, rotation_, opacity_, 1.0f);

    auto r = forward();
    ASSERT_TRUE(r.has_value());
    EXPECT_GT(r->second.forward_ctx.n_instances, static_cast<int>(n_));
}

TEST_F(FastGSKernelTest, EdgeCase_CameraBehind) {
    std::vector<float> t_data{0, 0, -10};
    auto T = Tensor::from_blob(t_data.data(), {3}, Device::CPU, DataType::Float32).to(Device::CUDA);
    camera_ = std::make_unique<Camera>(Tensor::eye(3, Device::CUDA), T, FX, FY, W / 2.0f, H / 2.0f,
                                       Tensor(), Tensor(), CameraModelType::PINHOLE,
                                       "behind", "", std::filesystem::path{}, W, H, 0);

    auto r = forward();
    ASSERT_TRUE(r.has_value());
}

// Tiled rendering
TEST_F(FastGSKernelTest, TiledRendering_Single) {
    auto r = fast_rasterize_forward(*camera_, *splat_, bg_, 100, 100, 256, 256, false);
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(r->first.width, 256);
    EXPECT_EQ(r->first.height, 256);
}

TEST_F(FastGSKernelTest, TiledRendering_Consistency) {
    auto full = forward();
    ASSERT_TRUE(full.has_value());
    auto region = full->first.image.slice(1, 50, 200).slice(2, 100, 300).clone();
    full->second.release_forward_context();

    auto tile = fast_rasterize_forward(*camera_, *splat_, bg_, 100, 50, 200, 150, false);
    ASSERT_TRUE(tile.has_value());

    float diff = (tile->first.image - region).abs().max().item<float>();
    EXPECT_LT(diff, 0.01f);
}

// =============================================================================
// Numerical gradient verification using finite differences
// =============================================================================

namespace {

    torch::Tensor to_torch(const Tensor& t) {
        auto cpu = t.to(Device::CPU);
        std::vector<int64_t> shape;
        for (size_t i = 0; i < t.ndim(); ++i)
            shape.push_back(t.shape()[i]);
        return torch::from_blob(cpu.ptr<float>(), shape, torch::kFloat32).clone().to(torch::kCUDA);
    }

} // namespace

class FastGSGradientTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Small scene for numerical gradient verification
        n_ = 32;
        std::mt19937 gen(123);
        std::uniform_real_distribution<float> pos(-2, 2);

        std::vector<float> means_data(n_ * 3);
        for (size_t i = 0; i < n_ * 3; ++i)
            means_data[i] = pos(gen);
        means_ = Tensor::from_blob(means_data.data(), {n_, 3}, Device::CPU, DataType::Float32).to(Device::CUDA);

        sh0_ = Tensor::randn({n_, 1, 3}, Device::CUDA).mul(0.3f);
        shN_ = Tensor::zeros({n_, 0, 3}, Device::CUDA);
        scaling_ = Tensor::randn({n_, 3}, Device::CUDA).mul(0.2f).sub(3.0f);
        rotation_ = Tensor::randn({n_, 4}, Device::CUDA);
        rotation_ = rotation_ / rotation_.pow(2.0f).sum(-1, true).sqrt();
        opacity_ = Tensor::randn({n_}, Device::CUDA);

        auto R = Tensor::eye(3, Device::CUDA);
        std::vector<float> t_data{0, 0, 4};
        auto T = Tensor::from_blob(t_data.data(), {3}, Device::CPU, DataType::Float32).to(Device::CUDA);
        camera_ = std::make_unique<Camera>(R, T, 200.0f, 200.0f, 64.0f, 64.0f,
                                           Tensor(), Tensor(), CameraModelType::PINHOLE,
                                           "test", "", std::filesystem::path{}, 128, 128, 0);
        bg_ = Tensor::zeros({3}, Device::CUDA);
    }

    void TearDown() override {
        GlobalArenaManager::instance().get_arena().full_reset();
    }

    float compute_loss(const Tensor& means, const Tensor& scaling, const Tensor& rotation,
                       const Tensor& opacity, const Tensor& sh0) {
        auto splat = std::make_unique<SplatData>(0, means, sh0, shN_, scaling, rotation, opacity, 1.0f);
        auto r = fast_rasterize_forward(*camera_, *splat, bg_, 0, 0, 0, 0, false);
        if (!r)
            return 0.0f;
        return r->first.image.pow(2.0f).sum().item<float>();
    }

    Tensor numerical_grad(ParamType param, float eps = 1e-3f) {
        Tensor orig;
        switch (param) {
        case ParamType::Means: orig = means_.clone(); break;
        case ParamType::Scaling: orig = scaling_.clone(); break;
        case ParamType::Rotation: orig = rotation_.clone(); break;
        case ParamType::Opacity: orig = opacity_.clone(); break;
        case ParamType::Sh0: orig = sh0_.clone(); break;
        default: return {};
        }

        Tensor grad = Tensor::zeros_like(orig);
        auto orig_cpu = orig.to(Device::CPU);
        auto grad_cpu = grad.to(Device::CPU);
        float* o_ptr = orig_cpu.ptr<float>();
        float* g_ptr = grad_cpu.ptr<float>();

        for (size_t i = 0; i < orig.numel(); ++i) {
            // Perturb +eps
            auto perturbed = orig_cpu.clone();
            perturbed.ptr<float>()[i] += eps;
            set_param(param, perturbed.to(Device::CUDA));
            float loss_plus = compute_loss(means_, scaling_, rotation_, opacity_, sh0_);

            // Perturb -eps
            perturbed.ptr<float>()[i] = o_ptr[i] - eps;
            set_param(param, perturbed.to(Device::CUDA));
            float loss_minus = compute_loss(means_, scaling_, rotation_, opacity_, sh0_);

            g_ptr[i] = (loss_plus - loss_minus) / (2.0f * eps);
        }

        set_param(param, orig);
        return grad_cpu.to(Device::CUDA);
    }

    void set_param(ParamType param, const Tensor& val) {
        switch (param) {
        case ParamType::Means: means_ = val; break;
        case ParamType::Scaling: scaling_ = val; break;
        case ParamType::Rotation: rotation_ = val; break;
        case ParamType::Opacity: opacity_ = val; break;
        case ParamType::Sh0: sh0_ = val; break;
        default: break;
        }
    }

    Tensor analytical_grad(ParamType param) {
        auto splat = std::make_unique<SplatData>(0, means_, sh0_, shN_, scaling_, rotation_, opacity_, 1.0f);
        auto r = fast_rasterize_forward(*camera_, *splat, bg_, 0, 0, 0, 0, false);
        if (!r)
            return {};

        AdamConfig cfg{.lr = 0.001f, .beta1 = 0.9, .beta2 = 0.999, .eps = 1e-15};
        auto opt = std::make_unique<AdamOptimizer>(*splat, cfg);
        opt->allocate_gradients();
        opt->zero_grad(0);

        auto grad_out = r->first.image.mul(2.0f);
        fast_rasterize_backward(r->second, grad_out, *splat, *opt, {}, {}, DensificationType::None, 1);

        return recovered_fused_grad(*opt, param).clone();
    }

    size_t n_;
    Tensor means_, sh0_, shN_, scaling_, rotation_, opacity_, bg_;
    std::unique_ptr<Camera> camera_;
};

namespace {
    void print_grad_stats(const char* name, const Tensor& num, const Tensor& ana) {
        auto n_cpu = num.to(Device::CPU);
        auto a_cpu = ana.to(Device::CPU);
        float* n = n_cpu.ptr<float>();
        float* a = a_cpu.ptr<float>();

        float max_err = 0, sum_err = 0, num_norm = 0, ana_norm = 0, dot = 0;
        for (size_t i = 0; i < num.numel(); ++i) {
            max_err = std::max(max_err, std::abs(n[i] - a[i]));
            sum_err += std::abs(n[i] - a[i]);
            num_norm += n[i] * n[i];
            ana_norm += a[i] * a[i];
            dot += n[i] * a[i];
        }
        float cos_sim = dot / (std::sqrt(num_norm) * std::sqrt(ana_norm) + 1e-8f);
        printf("  %-10s num_norm=%.4f ana_norm=%.4f max_err=%.5f mean_err=%.5f cos_sim=%.4f\n",
               name, std::sqrt(num_norm), std::sqrt(ana_norm), max_err, sum_err / num.numel(), cos_sim);
    }
} // namespace

TEST_F(FastGSGradientTest, Numerical_Means) {
    auto num = numerical_grad(ParamType::Means);
    auto ana = analytical_grad(ParamType::Means);
    ASSERT_TRUE(num.is_valid() && ana.is_valid());
    print_grad_stats("Means", num, ana);

    auto num_cpu = num.to(Device::CPU);
    auto ana_cpu = ana.to(Device::CPU);
    float* n_ptr = num_cpu.ptr<float>();
    float* a_ptr = ana_cpu.ptr<float>();

    int mismatches = 0;
    for (size_t i = 0; i < num.numel(); ++i) {
        float err = std::abs(n_ptr[i] - a_ptr[i]);
        float rel = err / (std::abs(n_ptr[i]) + 1e-6f);
        if (rel > 0.2f && err > 1e-3f)
            ++mismatches;
    }
    EXPECT_LT(mismatches, static_cast<int>(num.numel() * 0.15f));
}

TEST_F(FastGSGradientTest, Numerical_Scaling) {
    auto num = numerical_grad(ParamType::Scaling);
    auto ana = analytical_grad(ParamType::Scaling);
    ASSERT_TRUE(num.is_valid() && ana.is_valid());
    print_grad_stats("Scaling", num, ana);

    auto diff = (num - ana).abs();
    float mean_err = diff.mean().item<float>();
    EXPECT_LT(mean_err, 1.0f);
}

TEST_F(FastGSGradientTest, Numerical_Opacity) {
    auto num = numerical_grad(ParamType::Opacity);
    auto ana = analytical_grad(ParamType::Opacity);
    ASSERT_TRUE(num.is_valid() && ana.is_valid());
    print_grad_stats("Opacity", num, ana);

    auto num_cpu = num.to(Device::CPU);
    auto ana_cpu = ana.to(Device::CPU);
    float* n_ptr = num_cpu.ptr<float>();
    float* a_ptr = ana_cpu.ptr<float>();

    int mismatches = 0;
    for (size_t i = 0; i < num.numel(); ++i) {
        float err = std::abs(n_ptr[i] - a_ptr[i]);
        float rel = err / (std::abs(n_ptr[i]) + 1e-6f);
        if (rel > 0.1f && err > 1e-4f)
            ++mismatches;
    }
    EXPECT_LT(mismatches, static_cast<int>(num.numel() * 0.1f));
}

TEST_F(FastGSGradientTest, Numerical_Sh0) {
    auto num = numerical_grad(ParamType::Sh0);
    auto ana = analytical_grad(ParamType::Sh0);
    ASSERT_TRUE(num.is_valid() && ana.is_valid());
    print_grad_stats("Sh0", num, ana);

    auto diff = (num - ana).abs();
    float mean_err = diff.mean().item<float>();
    EXPECT_LT(mean_err, 1.0f);
}

TEST_F(FastGSGradientTest, GradientDirection) {
    // Verify gradient descent decreases loss
    auto splat = std::make_unique<SplatData>(0, means_, sh0_, shN_, scaling_, rotation_, opacity_, 1.0f);
    auto r = fast_rasterize_forward(*camera_, *splat, bg_, 0, 0, 0, 0, false);
    ASSERT_TRUE(r.has_value());

    float loss_before = r->first.image.pow(2.0f).sum().item<float>();

    AdamConfig cfg{.lr = 0.01f, .beta1 = 0.9, .beta2 = 0.999, .eps = 1e-15};
    auto opt = std::make_unique<AdamOptimizer>(*splat, cfg);
    opt->allocate_gradients();
    opt->zero_grad(0);

    auto grad_out = r->first.image.mul(2.0f);
    fast_rasterize_backward(r->second, grad_out, *splat, *opt, {}, {}, DensificationType::None, 1);

    r = fast_rasterize_forward(*camera_, *splat, bg_, 0, 0, 0, 0, false);
    ASSERT_TRUE(r.has_value());
    float loss_after = r->first.image.pow(2.0f).sum().item<float>();

    EXPECT_LT(loss_after, loss_before);
}

// =============================================================================
// Dense single-tile gradient test. This exercises the tile backward path with
// many splats contributing to the same pixels.
// =============================================================================

class FastGSDenseTileGradientTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create 128 gaussians concentrated in a SINGLE 16x16 tile
        // This ensures many gaussians end up in the same tile.
        n_ = 128;
        std::mt19937 gen(456);
        // Very small spread - all gaussians within ~1 pixel of each other
        std::uniform_real_distribution<float> tiny_offset(-0.01f, 0.01f);

        // Gaussians at (0, 0, z_i) with z spaced to give stable depth ordering
        // With camera at z=5, focal=100, this projects to pixel ~(32, 32) center of 64x64 image
        std::vector<float> means_data(n_ * 3);
        for (size_t i = 0; i < n_; ++i) {
            means_data[i * 3] = tiny_offset(gen);                  // x: tiny spread
            means_data[i * 3 + 1] = tiny_offset(gen);              // y: tiny spread
            means_data[i * 3 + 2] = static_cast<float>(i) * 0.02f; // z: stable ordering
        }
        means_ = Tensor::from_blob(means_data.data(), {n_, 3}, Device::CPU, DataType::Float32).to(Device::CUDA);

        sh0_ = Tensor::randn({n_, 1, 3}, Device::CUDA).mul(0.3f);
        shN_ = Tensor::zeros({n_, 0, 3}, Device::CUDA);
        // Very small gaussians so they all project to the same tile (scale exp(-5) ≈ 0.007)
        scaling_ = Tensor::full({n_, 3}, -5.0f, Device::CUDA);
        rotation_ = Tensor::zeros({n_, 4}, Device::CUDA);
        rotation_.slice(1, 0, 1).fill_(1.0f);               // Identity rotation (w=1, x=y=z=0)
        opacity_ = Tensor::full({n_}, -3.0f, Device::CUDA); // sigmoid(-3) ≈ 0.047, all contribute

        // Camera looking at origin from z=5
        auto R = Tensor::eye(3, Device::CUDA);
        std::vector<float> t_data{0, 0, 5};
        auto T = Tensor::from_blob(t_data.data(), {3}, Device::CPU, DataType::Float32).to(Device::CUDA);
        // 64x64 image, focal length 100 -> gaussians at (0,0) project to center (32,32)
        // which is in tile (32/16, 32/16) = tile (2, 2)
        camera_ = std::make_unique<Camera>(R, T, 100.0f, 100.0f, 32.0f, 32.0f,
                                           Tensor(), Tensor(), CameraModelType::PINHOLE,
                                           "test", "", std::filesystem::path{}, 64, 64, 0);
        bg_ = Tensor::zeros({3}, Device::CUDA);
    }

    void TearDown() override {
        GlobalArenaManager::instance().get_arena().full_reset();
    }

    float compute_loss(const Tensor& means, const Tensor& scaling, const Tensor& rotation,
                       const Tensor& opacity, const Tensor& sh0) {
        auto splat = std::make_unique<SplatData>(0, means, sh0, shN_, scaling, rotation, opacity, 1.0f);
        auto r = fast_rasterize_forward(*camera_, *splat, bg_, 0, 0, 0, 0, false);
        if (!r)
            return 0.0f;
        return r->first.image.pow(2.0f).sum().item<float>();
    }

    Tensor numerical_grad(ParamType param, float eps = 1e-3f) {
        Tensor orig;
        switch (param) {
        case ParamType::Means: orig = means_.clone(); break;
        case ParamType::Scaling: orig = scaling_.clone(); break;
        case ParamType::Opacity: orig = opacity_.clone(); break;
        case ParamType::Sh0: orig = sh0_.clone(); break;
        default: return {};
        }

        Tensor grad = Tensor::zeros_like(orig);
        auto orig_cpu = orig.to(Device::CPU);
        auto grad_cpu = grad.to(Device::CPU);
        float* o_ptr = orig_cpu.ptr<float>();
        float* g_ptr = grad_cpu.ptr<float>();

        for (size_t i = 0; i < orig.numel(); ++i) {
            auto perturbed = orig_cpu.clone();
            perturbed.ptr<float>()[i] += eps;
            set_param(param, perturbed.to(Device::CUDA));
            float loss_plus = compute_loss(means_, scaling_, rotation_, opacity_, sh0_);

            perturbed.ptr<float>()[i] = o_ptr[i] - eps;
            set_param(param, perturbed.to(Device::CUDA));
            float loss_minus = compute_loss(means_, scaling_, rotation_, opacity_, sh0_);

            g_ptr[i] = (loss_plus - loss_minus) / (2.0f * eps);
        }

        set_param(param, orig);
        return grad_cpu.to(Device::CUDA);
    }

    void set_param(ParamType param, const Tensor& val) {
        switch (param) {
        case ParamType::Means: means_ = val; break;
        case ParamType::Scaling: scaling_ = val; break;
        case ParamType::Opacity: opacity_ = val; break;
        case ParamType::Sh0: sh0_ = val; break;
        default: break;
        }
    }

    Tensor analytical_grad(ParamType param) {
        auto splat = std::make_unique<SplatData>(0, means_, sh0_, shN_, scaling_, rotation_, opacity_, 1.0f);
        auto r = fast_rasterize_forward(*camera_, *splat, bg_, 0, 0, 0, 0, false);
        if (!r)
            return {};

        AdamConfig cfg{.lr = 0.001f, .beta1 = 0.9, .beta2 = 0.999, .eps = 1e-15};
        auto opt = std::make_unique<AdamOptimizer>(*splat, cfg);
        opt->allocate_gradients();
        opt->zero_grad(0);

        auto grad_out = r->first.image.mul(2.0f);
        fast_rasterize_backward(r->second, grad_out, *splat, *opt, {}, {}, DensificationType::None, 1);

        return recovered_fused_grad(*opt, param).clone();
    }

    size_t n_;
    Tensor means_, sh0_, shN_, scaling_, rotation_, opacity_, bg_;
    std::unique_ptr<Camera> camera_;
};

TEST_F(FastGSDenseTileGradientTest, VerifyDenseTileInstances) {
    // Verify this setup actually produces a dense tile workload.
    auto splat = std::make_unique<SplatData>(0, means_, sh0_, shN_, scaling_, rotation_, opacity_, 1.0f);
    auto r = fast_rasterize_forward(*camera_, *splat, bg_, 0, 0, 0, 0, false);
    ASSERT_TRUE(r.has_value());

    printf("  n_instances=%d\n", r->second.forward_ctx.n_instances);

    EXPECT_GT(r->second.forward_ctx.n_instances, 100);
}

TEST_F(FastGSDenseTileGradientTest, Numerical_Means_DenseTile) {
    auto num = numerical_grad(ParamType::Means);
    auto ana = analytical_grad(ParamType::Means);
    ASSERT_TRUE(num.is_valid() && ana.is_valid());

    auto n_cpu = num.to(Device::CPU);
    auto a_cpu = ana.to(Device::CPU);
    float* n = n_cpu.ptr<float>();
    float* a = a_cpu.ptr<float>();

    float max_err = 0, sum_err = 0, num_norm = 0, ana_norm = 0, dot = 0;
    for (size_t i = 0; i < num.numel(); ++i) {
        max_err = std::max(max_err, std::abs(n[i] - a[i]));
        sum_err += std::abs(n[i] - a[i]);
        num_norm += n[i] * n[i];
        ana_norm += a[i] * a[i];
        dot += n[i] * a[i];
    }
    float cos_sim = dot / (std::sqrt(num_norm) * std::sqrt(ana_norm) + 1e-8f);
    printf("  DenseTile Means: num_norm=%.4f ana_norm=%.4f max_err=%.5f mean_err=%.5f cos_sim=%.4f\n",
           std::sqrt(num_norm), std::sqrt(ana_norm), max_err, sum_err / num.numel(), cos_sim);

    EXPECT_GT(cos_sim, 0.80f) << "Gradient direction mismatch in dense tile backward";

    float mean_err = sum_err / num.numel();
    EXPECT_LT(mean_err, 2.0f) << "Mean gradient error too high";
}

TEST_F(FastGSDenseTileGradientTest, Numerical_Opacity_DenseTile) {
    auto num = numerical_grad(ParamType::Opacity);
    auto ana = analytical_grad(ParamType::Opacity);
    ASSERT_TRUE(num.is_valid() && ana.is_valid());

    auto n_cpu = num.to(Device::CPU);
    auto a_cpu = ana.to(Device::CPU);
    float* n = n_cpu.ptr<float>();
    float* a = a_cpu.ptr<float>();

    float max_err = 0, sum_err = 0, num_norm = 0, ana_norm = 0, dot = 0;
    for (size_t i = 0; i < num.numel(); ++i) {
        max_err = std::max(max_err, std::abs(n[i] - a[i]));
        sum_err += std::abs(n[i] - a[i]);
        num_norm += n[i] * n[i];
        ana_norm += a[i] * a[i];
        dot += n[i] * a[i];
    }
    float cos_sim = dot / (std::sqrt(num_norm) * std::sqrt(ana_norm) + 1e-8f);
    printf("  DenseTile Opacity: num_norm=%.4f ana_norm=%.4f max_err=%.5f mean_err=%.5f cos_sim=%.4f\n",
           std::sqrt(num_norm), std::sqrt(ana_norm), max_err, sum_err / num.numel(), cos_sim);

    EXPECT_GT(cos_sim, 0.95f) << "Gradient direction mismatch in dense tile backward";
}

TEST_F(FastGSDenseTileGradientTest, GradientDescent_DenseTile) {
    // Verify gradient descent actually reduces loss with many gaussians per tile
    auto splat = std::make_unique<SplatData>(0, means_, sh0_, shN_, scaling_, rotation_, opacity_, 1.0f);
    auto r = fast_rasterize_forward(*camera_, *splat, bg_, 0, 0, 0, 0, false);
    ASSERT_TRUE(r.has_value());

    float loss_before = r->first.image.pow(2.0f).sum().item<float>();
    printf("  Loss before: %.4f\n", loss_before);
    r->second.release_forward_context();

    AdamConfig cfg{.lr = 0.01f, .beta1 = 0.9, .beta2 = 0.999, .eps = 1e-15};
    auto opt = std::make_unique<AdamOptimizer>(*splat, cfg);
    opt->allocate_gradients();

    // Do several gradient descent steps
    for (int step = 0; step < 10; ++step) {
        r = fast_rasterize_forward(*camera_, *splat, bg_, 0, 0, 0, 0, false);
        ASSERT_TRUE(r.has_value());

        opt->zero_grad(0);
        auto grad_out = r->first.image.mul(2.0f);
        fast_rasterize_backward(r->second, grad_out, *splat, *opt, {}, {}, DensificationType::None, step + 1);
    }

    r = fast_rasterize_forward(*camera_, *splat, bg_, 0, 0, 0, 0, false);
    ASSERT_TRUE(r.has_value());
    float loss_after = r->first.image.pow(2.0f).sum().item<float>();
    printf("  Loss after 10 steps: %.4f (reduction: %.2f%%)\n",
           loss_after, (loss_before - loss_after) / loss_before * 100.0f);

    EXPECT_LT(loss_after, loss_before) << "Gradient descent should reduce loss";
    // Expect at least 10% reduction with 10 steps
    EXPECT_LT(loss_after, loss_before * 0.9f) << "Loss reduction too small - gradients may be wrong";
}

/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/camera.hpp"
#include "core/cuda/memory_arena.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "io/formats/ply.hpp"
#include "training/optimizer/adam_optimizer.hpp"
#include "training/rasterization/fast_rasterizer.hpp"
#include <cuda_runtime.h>
#include <filesystem>
#include <gtest/gtest.h>
#include <random>
#include <torch/torch.h>

using namespace lfs::training;
using namespace lfs::core;

namespace {
    constexpr const char* GARDEN_PATH = "data/garden";
    constexpr int W = 640, H = 480;
    constexpr float FX = 500.0f, FY = 500.0f;
} // namespace

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
    EXPECT_GT(r->second.forward_ctx.n_visible_primitives, 0);
    EXPECT_LE(r->second.forward_ctx.n_visible_primitives, static_cast<int>(n_));
}

TEST_F(FastGSKernelTest, Forward_DepthOrdering) {
    auto r = forward();
    ASSERT_TRUE(r.has_value());
    EXPECT_TRUE(r->first.image.is_valid());
}

TEST_F(FastGSKernelTest, Forward_Instances) {
    auto r = forward();
    ASSERT_TRUE(r.has_value());
    EXPECT_GT(r->second.forward_ctx.n_instances, 0);
}

TEST_F(FastGSKernelTest, Forward_Buckets) {
    auto r = forward();
    ASSERT_TRUE(r.has_value());
    EXPECT_GE(r->second.forward_ctx.n_buckets, 0);
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

    EXPECT_GT(opt->get_grad(ParamType::Means).pow(2.0f).sum().item<float>(), 0.0f);
    EXPECT_GT(opt->get_grad(ParamType::Scaling).pow(2.0f).sum().item<float>(), 0.0f);
}

TEST_F(FastGSKernelTest, Backward_Preprocess) {
    auto r = forward();
    ASSERT_TRUE(r.has_value());

    auto opt = make_optimizer();
    opt->zero_grad(0);
    fast_rasterize_backward(r->second, Tensor::randn_like(r->first.image).mul(0.1f),
                            *splat_, *opt, Tensor::randn_like(r->first.alpha).mul(0.1f));

    EXPECT_TRUE(opt->get_grad(ParamType::Means).is_valid());
    EXPECT_TRUE(opt->get_grad(ParamType::Rotation).is_valid());
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
    auto r2 = forward();
    ASSERT_TRUE(r1.has_value() && r2.has_value());

    float diff = (r1->first.image - r2->first.image).abs().max().item<float>();
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

    check(opt->get_grad(ParamType::Means));
    check(opt->get_grad(ParamType::Scaling));
    check(opt->get_grad(ParamType::Rotation));
    check(opt->get_grad(ParamType::Opacity));
    check(opt->get_grad(ParamType::Sh0));
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

    auto tile = fast_rasterize_forward(*camera_, *splat_, bg_, 100, 50, 200, 150, false);
    ASSERT_TRUE(tile.has_value());

    auto region = full->first.image.slice(1, 50, 200).slice(2, 100, 300);
    float diff = (tile->first.image - region).abs().max().item<float>();
    EXPECT_LT(diff, 0.01f);
}

// Performance
TEST_F(FastGSKernelTest, Performance_Forward) {
    for (int i = 0; i < 3; ++i)
        forward();
    cudaDeviceSynchronize();

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; ++i)
        forward();
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / 10;
    EXPECT_LT(ms, 10.0);
}

TEST_F(FastGSKernelTest, Performance_Backward) {
    auto r = forward();
    ASSERT_TRUE(r.has_value());

    auto grad = Tensor::randn_like(r->first.image);
    auto opt = make_optimizer();

    for (int i = 0; i < 3; ++i) {
        opt->zero_grad(0);
        fast_rasterize_backward(r->second, grad, *splat_, *opt, {});
    }
    cudaDeviceSynchronize();

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; ++i) {
        auto fwd = forward();
        opt->zero_grad(0);
        fast_rasterize_backward(fwd->second, grad, *splat_, *opt, {});
    }
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / 10;
    EXPECT_LT(ms, 20.0);
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
        fast_rasterize_backward(r->second, grad_out, *splat, *opt, {});

        return opt->get_grad(param).clone();
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
    fast_rasterize_backward(r->second, grad_out, *splat, *opt, {});
    opt->step(1);

    r = fast_rasterize_forward(*camera_, *splat, bg_, 0, 0, 0, 0, false);
    ASSERT_TRUE(r.has_value());
    float loss_after = r->first.image.pow(2.0f).sum().item<float>();

    EXPECT_LT(loss_after, loss_before);
}

// =============================================================================
// Test for checkpoint_interval > 32 (sub-bucket gradient correctness)
// This tests the case where a single bucket contains more than 32 gaussians
// =============================================================================

class FastGSMultiBucketGradientTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create 128 gaussians concentrated in a SINGLE 16x16 tile
        // This ensures many gaussians (>64) end up in the same tile, triggering
        // multiple sub-buckets per bucket when checkpoint_interval=64
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
        fast_rasterize_backward(r->second, grad_out, *splat, *opt, {});

        return opt->get_grad(param).clone();
    }

    size_t n_;
    Tensor means_, sh0_, shN_, scaling_, rotation_, opacity_, bg_;
    std::unique_ptr<Camera> camera_;
};

TEST_F(FastGSMultiBucketGradientTest, VerifyBucketCount) {
    // Verify we actually have multiple sub-buckets (>64 instances per tile)
    auto splat = std::make_unique<SplatData>(0, means_, sh0_, shN_, scaling_, rotation_, opacity_, 1.0f);
    auto r = fast_rasterize_forward(*camera_, *splat, bg_, 0, 0, 0, 0, false);
    ASSERT_TRUE(r.has_value());

    // With 128 gaussians concentrated in single tile, we should have:
    // - ~128 visible (all in front of camera)
    // - ~128 instances (each gaussian in 1 tile)
    // - 2 buckets (128/64 = 2 with checkpoint_interval=64)
    printf("  n_visible=%d, n_instances=%d, n_buckets=%d\n",
           r->second.forward_ctx.n_visible_primitives,
           r->second.forward_ctx.n_instances,
           r->second.forward_ctx.n_buckets);

    // Expect at least 100 visible (most should be visible since they're in front of camera)
    EXPECT_GT(r->second.forward_ctx.n_visible_primitives, 100);
    // Expect at least 2 buckets (which means sub_bucket > 0 will be triggered)
    // With 128 instances in 1-2 tiles and checkpoint_interval=64, we need >= 2 buckets
    EXPECT_GE(r->second.forward_ctx.n_buckets, 2) << "Need at least 2 buckets to test sub-bucket logic";
}

TEST_F(FastGSMultiBucketGradientTest, Numerical_Means_MultiBucket) {
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
    printf("  MultiBucket Means: num_norm=%.4f ana_norm=%.4f max_err=%.5f mean_err=%.5f cos_sim=%.4f\n",
           std::sqrt(num_norm), std::sqrt(ana_norm), max_err, sum_err / num.numel(), cos_sim);

    EXPECT_GT(cos_sim, 0.80f) << "Gradient direction mismatch - possible sub-bucket bug";

    float mean_err = sum_err / num.numel();
    EXPECT_LT(mean_err, 2.0f) << "Mean gradient error too high";
}

TEST_F(FastGSMultiBucketGradientTest, Numerical_Opacity_MultiBucket) {
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
    printf("  MultiBucket Opacity: num_norm=%.4f ana_norm=%.4f max_err=%.5f mean_err=%.5f cos_sim=%.4f\n",
           std::sqrt(num_norm), std::sqrt(ana_norm), max_err, sum_err / num.numel(), cos_sim);

    EXPECT_GT(cos_sim, 0.95f) << "Gradient direction mismatch - possible sub-bucket bug";
}

TEST_F(FastGSMultiBucketGradientTest, GradientDescent_MultiBucket) {
    // Verify gradient descent actually reduces loss with many gaussians per tile
    auto splat = std::make_unique<SplatData>(0, means_, sh0_, shN_, scaling_, rotation_, opacity_, 1.0f);
    auto r = fast_rasterize_forward(*camera_, *splat, bg_, 0, 0, 0, 0, false);
    ASSERT_TRUE(r.has_value());

    float loss_before = r->first.image.pow(2.0f).sum().item<float>();
    printf("  Loss before: %.4f\n", loss_before);

    AdamConfig cfg{.lr = 0.01f, .beta1 = 0.9, .beta2 = 0.999, .eps = 1e-15};
    auto opt = std::make_unique<AdamOptimizer>(*splat, cfg);
    opt->allocate_gradients();

    // Do several gradient descent steps
    for (int step = 0; step < 10; ++step) {
        r = fast_rasterize_forward(*camera_, *splat, bg_, 0, 0, 0, 0, false);
        ASSERT_TRUE(r.has_value());

        opt->zero_grad(0);
        auto grad_out = r->first.image.mul(2.0f);
        fast_rasterize_backward(r->second, grad_out, *splat, *opt, {});
        opt->step(step + 1);
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

/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/camera.hpp"
#include "core/cuda/memory_arena.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "training/kernels/grad_alpha.hpp"
#include "training/optimizer/adam_optimizer.hpp"
#include "training/rasterization/fast_rasterizer.hpp"

#include <cmath>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <memory>
#include <vector>

using namespace lfs::training;
using namespace lfs::core;

namespace {

    Camera make_camera(int w, int h) {
        std::vector<float> R_data = {1, 0, 0, 0, 1, 0, 0, 0, 1};
        std::vector<float> T_data = {0, 0, 4};
        auto R = Tensor::from_blob(R_data.data(), {3, 3}, Device::CPU, DataType::Float32).to(Device::CUDA);
        auto T = Tensor::from_blob(T_data.data(), {3}, Device::CPU, DataType::Float32).to(Device::CUDA);
        return Camera(R, T, /*fx=*/100.f, /*fy=*/100.f, /*cx=*/w * 0.5f, /*cy=*/h * 0.5f,
                      Tensor(), Tensor(), CameraModelType::PINHOLE, "test", "",
                      std::filesystem::path{}, w, h, 0);
    }

    std::unique_ptr<SplatData> make_splat(int n) {
        auto means = Tensor::zeros({static_cast<size_t>(n), 3}, Device::CUDA);
        if (n > 0) {
            auto cpu = means.to(Device::CPU);
            float* p = cpu.ptr<float>();
            for (int i = 0; i < n; ++i) {
                p[i * 3 + 0] = (i % 5) * 0.3f - 0.6f;
                p[i * 3 + 1] = (i / 5) * 0.3f - 0.6f;
                p[i * 3 + 2] = 0.0f;
            }
            means = cpu.to(Device::CUDA);
        }
        auto sh0 = Tensor::full({static_cast<size_t>(n), 1, 3}, 0.5f, Device::CUDA);
        auto shN = Tensor::zeros({static_cast<size_t>(n), 0, 3}, Device::CUDA);
        auto scaling = Tensor::full({static_cast<size_t>(n), 3}, -2.0f, Device::CUDA);
        std::vector<float> rot(static_cast<size_t>(n) * 4, 0.f);
        for (int i = 0; i < n; ++i) {
            rot[static_cast<size_t>(i) * 4] = 1.f;
        }
        auto rotation = Tensor::from_blob(rot.data(), {static_cast<size_t>(n), 4}, Device::CPU, DataType::Float32)
                            .to(Device::CUDA);
        auto opacity = Tensor::full({static_cast<size_t>(n)}, 2.0f, Device::CUDA);
        return std::make_unique<SplatData>(0, means, sh0, shN, scaling, rotation, opacity, 1.0f);
    }

    void cleanup_arena() {
        GlobalArenaManager::instance().get_arena().full_reset();
    }

    float max_abs_diff(const Tensor& a, const Tensor& b) {
        auto ac = a.cpu().contiguous();
        auto bc = b.cpu().contiguous();
        EXPECT_EQ(ac.numel(), bc.numel());
        const float* pa = ac.ptr<float>();
        const float* pb = bc.ptr<float>();
        float m = 0.f;
        for (size_t i = 0; i < ac.numel(); ++i) {
            m = std::max(m, std::fabs(pa[i] - pb[i]));
        }
        return m;
    }

} // namespace

class FusedBgBlendTest : public ::testing::Test {
protected:
    void SetUp() override {
        black_bg_ = Tensor::zeros({3}, Device::CUDA);
        std::vector<float> bg_host = {0.2f, 0.4f, 0.6f};
        color_bg_ = Tensor::from_blob(bg_host.data(), {3}, Device::CPU, DataType::Float32)
                        .to(Device::CUDA)
                        .contiguous();
        camera_ = std::make_unique<Camera>(make_camera(32, 32));
        splat_ = make_splat(16);
    }

    void TearDown() override {
        splat_.reset();
        camera_.reset();
        cleanup_arena();
    }

    Tensor black_bg_;
    Tensor color_bg_;
    std::unique_ptr<Camera> camera_;
    std::unique_ptr<SplatData> splat_;
};

// Forward: fused path (bg in blend_cu) must match black-bg raw + external compose.
TEST_F(FusedBgBlendTest, ForwardBlendedMatchesExternalCompose) {
    auto raw_fwd = fast_rasterize_forward(*camera_, *splat_, black_bg_, 0, 0, 0, 0, false);
    ASSERT_TRUE(raw_fwd.has_value()) << std::string(raw_fwd.error().user_message());
    auto raw_image = raw_fwd->first.image.clone();
    auto alpha = raw_fwd->first.alpha.clone();
    const int H = static_cast<int>(raw_image.shape()[1]);
    const int W = static_cast<int>(raw_image.shape()[2]);
    auto expected = Tensor::empty_like(raw_image);
    kernels::launch_fused_background_blend(
        raw_image.ptr<float>(),
        alpha.ptr<float>(),
        color_bg_.ptr<float>(),
        expected.ptr<float>(),
        H, W,
        nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    raw_fwd->second.release_forward_context();

    auto fused_fwd = fast_rasterize_forward(*camera_, *splat_, color_bg_, 0, 0, 0, 0, false);
    ASSERT_TRUE(fused_fwd.has_value()) << std::string(fused_fwd.error().user_message());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    const float diff = max_abs_diff(expected, fused_fwd->first.image);
    EXPECT_LT(diff, 1e-6f) << "blended pixel max abs diff=" << diff;
    fused_fwd->second.release_forward_context();
}

// Backward: grad_alpha and param updates bit-equal / <1e-6 whether or not the
// context image is "raw" (unblend is dead — blend_backward ignores image).
TEST_F(FusedBgBlendTest, BackwardGradsMatchWithBlendedImage) {
    constexpr float kLr = 0.01f;

    auto run_step = [&](Tensor& bg) -> std::unique_ptr<SplatData> {
        auto splat = make_splat(16);
        AdamConfig cfg{.lr = kLr, .beta1 = 0.9, .beta2 = 0.999, .eps = 1e-15};
        AdamOptimizer opt(*splat, cfg);
        opt.allocate_gradients();
        opt.zero_grad(0);

        auto fwd = fast_rasterize_forward(*camera_, *splat, bg, 0, 0, 0, 0, false);
        EXPECT_TRUE(fwd.has_value());
        if (!fwd.has_value()) {
            return nullptr;
        }
        // Synthetic photometric grad: constant 1 on all channels.
        auto grad_out = Tensor::full_like(fwd->first.image, 1.0f);
        fast_rasterize_backward(fwd->second, grad_out, *splat, opt, {}, {},
                                DensificationType::None, /*iteration=*/1);
        EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        fwd->second.release_forward_context();
        return splat;
    };

    // Color bg path (blended image in ctx, no unblend after 1.4).
    auto splat_color = run_step(color_bg_);
    // Black bg: no unblend needed either; used only as a second trajectory check.
    auto splat_black = run_step(black_bg_);
    ASSERT_NE(splat_color, nullptr);
    ASSERT_NE(splat_black, nullptr);

    // Sanity: different bgs produce different param updates (grad_alpha differs).
    const float means_diff = max_abs_diff(splat_color->means(), splat_black->means());
    EXPECT_GT(means_diff, 0.0f) << "bg must affect gradients via grad_alpha";

    // Reproducibility: same bg twice → bit-equal params.
    auto splat_color2 = run_step(color_bg_);
    ASSERT_NE(splat_color2, nullptr);
    EXPECT_LT(max_abs_diff(splat_color->means(), splat_color2->means()), 1e-6f);
    EXPECT_LT(max_abs_diff(splat_color->scaling_raw(), splat_color2->scaling_raw()), 1e-6f);
    EXPECT_LT(max_abs_diff(splat_color->opacity_raw(), splat_color2->opacity_raw()), 1e-6f);
}

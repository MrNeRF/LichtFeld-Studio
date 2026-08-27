/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/camera.hpp"
#include "core/cuda/memory_arena.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "training/rasterization/fast_rasterizer.hpp"
#include "training/rasterization/fastgs/rasterization/include/forward.h"

#include <cmath>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <memory>
#include <vector>

using namespace lfs::training;
using namespace lfs::core;
using fast_lfs::rasterization::set_blend_batch_size_for_testing;
using fast_lfs::rasterization::set_warp_cull_mode_for_testing;

namespace {

    Camera make_camera(int w, int h) {
        std::vector<float> R_data = {1, 0, 0, 0, 1, 0, 0, 0, 1};
        std::vector<float> T_data = {0, 0, 4};
        auto R = Tensor::from_blob(R_data.data(), {3, 3}, Device::CPU, DataType::Float32).to(Device::CUDA);
        auto T = Tensor::from_blob(T_data.data(), {3}, Device::CPU, DataType::Float32).to(Device::CUDA);
        return Camera(R, T, /*fx=*/120.f, /*fy=*/120.f, /*cx=*/w * 0.5f, /*cy=*/h * 0.5f,
                      Tensor(), Tensor(), CameraModelType::PINHOLE, "test", "",
                      std::filesystem::path{}, w, h, 0);
    }

    // Small synthetic cloud: compact Gaussians so some tiles see only a subset.
    std::unique_ptr<SplatData> make_synthetic_splat(int n) {
        auto means = Tensor::zeros({static_cast<size_t>(n), 3}, Device::CUDA);
        if (n > 0) {
            auto cpu = means.to(Device::CPU);
            float* p = cpu.ptr<float>();
            for (int i = 0; i < n; ++i) {
                // Spread across the frustum so sub-tile cull has work to do.
                p[i * 3 + 0] = ((i % 8) - 3.5f) * 0.35f;
                p[i * 3 + 1] = ((i / 8) % 8 - 3.5f) * 0.35f;
                p[i * 3 + 2] = 0.0f;
            }
            means = cpu.to(Device::CUDA);
        }
        auto sh0 = Tensor::full({static_cast<size_t>(n), 1, 3}, 0.55f, Device::CUDA);
        auto shN = Tensor::zeros({static_cast<size_t>(n), 0, 3}, Device::CUDA);
        // Modest scales → small on-screen footprints (partial tile coverage).
        auto scaling = Tensor::full({static_cast<size_t>(n), 3}, -1.8f, Device::CUDA);
        std::vector<float> rot(static_cast<size_t>(n) * 4, 0.f);
        for (int i = 0; i < n; ++i) {
            rot[static_cast<size_t>(i) * 4] = 1.f;
        }
        auto rotation = Tensor::from_blob(rot.data(), {static_cast<size_t>(n), 4}, Device::CPU, DataType::Float32)
                            .to(Device::CUDA);
        auto opacity = Tensor::full({static_cast<size_t>(n)}, 2.5f, Device::CUDA);
        return std::make_unique<SplatData>(0, means, sh0, shN, scaling, rotation, opacity, 1.0f);
    }

    // Denser / larger footprints — closer to a real-scene stress case.
    std::unique_ptr<SplatData> make_dense_splat(int n) {
        auto means = Tensor::zeros({static_cast<size_t>(n), 3}, Device::CUDA);
        if (n > 0) {
            auto cpu = means.to(Device::CPU);
            float* p = cpu.ptr<float>();
            for (int i = 0; i < n; ++i) {
                const float u = static_cast<float>(i % 16) / 15.f;
                const float v = static_cast<float>((i / 16) % 16) / 15.f;
                p[i * 3 + 0] = (u - 0.5f) * 2.5f;
                p[i * 3 + 1] = (v - 0.5f) * 2.5f;
                p[i * 3 + 2] = ((i / 256) % 3) * 0.15f;
            }
            means = cpu.to(Device::CUDA);
        }
        auto sh0 = Tensor::full({static_cast<size_t>(n), 1, 3}, 0.4f, Device::CUDA);
        auto shN = Tensor::zeros({static_cast<size_t>(n), 0, 3}, Device::CUDA);
        auto scaling = Tensor::full({static_cast<size_t>(n), 3}, -1.2f, Device::CUDA);
        std::vector<float> rot(static_cast<size_t>(n) * 4, 0.f);
        for (int i = 0; i < n; ++i) {
            rot[static_cast<size_t>(i) * 4] = 1.f;
            rot[static_cast<size_t>(i) * 4 + 1] = 0.1f * ((i % 5) - 2);
        }
        auto rotation = Tensor::from_blob(rot.data(), {static_cast<size_t>(n), 4}, Device::CPU, DataType::Float32)
                            .to(Device::CUDA);
        auto opacity = Tensor::full({static_cast<size_t>(n)}, 1.8f, Device::CUDA);
        return std::make_unique<SplatData>(0, means, sh0, shN, scaling, rotation, opacity, 1.0f);
    }

    void cleanup_arena() {
        GlobalArenaManager::instance().get_arena().full_reset();
    }

    Tensor render_image(Camera& cam, SplatData& splat, Tensor& bg, int cull_mode, int batch = 0) {
        set_warp_cull_mode_for_testing(cull_mode);
        set_blend_batch_size_for_testing(batch);
        auto r = fast_rasterize_forward(cam, splat, bg, 0, 0, 0, 0, false);
        EXPECT_TRUE(r.has_value()) << (r.has_value() ? "" : lfs::format_for_developer(r.error()));
        if (!r.has_value()) {
            return Tensor();
        }
        Tensor img = r->first.image.to(Device::CPU).contiguous();
        r->second.release_forward_context();
        EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        return img;
    }

    bool images_bit_identical(const Tensor& a, const Tensor& b) {
        if (a.numel() != b.numel() || a.numel() == 0) {
            return false;
        }
        const float* pa = a.ptr<float>();
        const float* pb = b.ptr<float>();
        for (size_t i = 0; i < a.numel(); ++i) {
            // Bit-identical: exact float equality (also catches NaN mismatches).
            if (pa[i] != pb[i]) {
                return false;
            }
        }
        return true;
    }

    size_t count_diffs(const Tensor& a, const Tensor& b) {
        if (a.numel() != b.numel()) {
            return a.numel() + b.numel();
        }
        size_t n = 0;
        const float* pa = a.ptr<float>();
        const float* pb = b.ptr<float>();
        for (size_t i = 0; i < a.numel(); ++i) {
            if (pa[i] != pb[i]) {
                ++n;
            }
        }
        return n;
    }

} // namespace

class WarpCullBlendTest : public ::testing::Test {
protected:
    void SetUp() override {
        bg_ = Tensor::zeros({3}, Device::CUDA);
        camera_ = std::make_unique<Camera>(make_camera(64, 64));
        synthetic_ = make_synthetic_splat(48);
        dense_ = make_dense_splat(512);
        // Default production mode after each test.
        set_warp_cull_mode_for_testing(0);
        set_blend_batch_size_for_testing(0);
    }

    void TearDown() override {
        set_warp_cull_mode_for_testing(0);
        set_blend_batch_size_for_testing(0);
        synthetic_.reset();
        dense_.reset();
        camera_.reset();
        cleanup_arena();
    }

    Tensor bg_;
    std::unique_ptr<Camera> camera_;
    std::unique_ptr<SplatData> synthetic_;
    std::unique_ptr<SplatData> dense_;
};

// A deliberately wrong mask must not match the cull-disabled reference.
TEST_F(WarpCullBlendTest, WrongMaskDiffersFromReference) {
    auto ref = render_image(*camera_, *synthetic_, bg_, /*cull_mode=*/1);
    auto wrong = render_image(*camera_, *synthetic_, bg_, /*cull_mode=*/2);
    ASSERT_GT(ref.numel(), 0u);
    ASSERT_EQ(ref.numel(), wrong.numel());
    // Deliberately-wrong empty mask skips all splat evals → solid background (zeros).
    // Reference has visible content. They must differ.
    EXPECT_FALSE(images_bit_identical(ref, wrong))
        << "wrong empty mask must diverge from reference (test sensitivity)";
    EXPECT_GT(count_diffs(ref, wrong), 0u);
}

// ---------------------------------------------------------------------------
// Enabled warp cull is bit-identical to cull-disabled reference (synthetic).
// ---------------------------------------------------------------------------
TEST_F(WarpCullBlendTest, EnabledMatchesReference_Synthetic) {
    auto ref = render_image(*camera_, *synthetic_, bg_, /*cull_mode=*/1);
    auto cull = render_image(*camera_, *synthetic_, bg_, /*cull_mode=*/0);
    ASSERT_GT(ref.numel(), 0u);
    ASSERT_EQ(ref.numel(), cull.numel());
    EXPECT_TRUE(images_bit_identical(ref, cull))
        << "warp cull vs disabled: " << count_diffs(ref, cull) << " differing floats";
}

// ---------------------------------------------------------------------------
// Same identity on denser scene (partial + full tile coverage mix).
// ---------------------------------------------------------------------------
TEST_F(WarpCullBlendTest, EnabledMatchesReference_Dense) {
    auto ref = render_image(*camera_, *dense_, bg_, /*cull_mode=*/1);
    auto cull = render_image(*camera_, *dense_, bg_, /*cull_mode=*/0);
    ASSERT_GT(ref.numel(), 0u);
    ASSERT_EQ(ref.numel(), cull.numel());
    EXPECT_TRUE(images_bit_identical(ref, cull))
        << "dense warp cull vs disabled: " << count_diffs(ref, cull) << " differing floats";
}

// ---------------------------------------------------------------------------
// Batch-size sweep identity: every legal batch size matches default (256 path).
// ---------------------------------------------------------------------------
TEST_F(WarpCullBlendTest, BatchSizeSweepBitIdentical) {
    auto ref = render_image(*camera_, *dense_, bg_, /*cull_mode=*/0, /*batch=*/0);
    ASSERT_GT(ref.numel(), 0u);
    for (int b = 32; b <= 256; b += 32) {
        auto img = render_image(*camera_, *dense_, bg_, /*cull_mode=*/0, b);
        ASSERT_EQ(img.numel(), ref.numel()) << "batch=" << b;
        EXPECT_TRUE(images_bit_identical(ref, img))
            << "batch " << b << " differs in " << count_diffs(ref, img) << " floats";
    }
}

// ---------------------------------------------------------------------------
// Determinism: two enabled-cull renders are bit-identical.
// ---------------------------------------------------------------------------
TEST_F(WarpCullBlendTest, EnabledIsDeterministic) {
    auto a = render_image(*camera_, *dense_, bg_, /*cull_mode=*/0);
    auto b = render_image(*camera_, *dense_, bg_, /*cull_mode=*/0);
    ASSERT_GT(a.numel(), 0u);
    EXPECT_TRUE(images_bit_identical(a, b));
}

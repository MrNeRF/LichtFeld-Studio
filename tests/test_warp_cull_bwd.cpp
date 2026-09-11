/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/camera.hpp"
#include "core/cuda/memory_arena.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "training/optimizer/adam_optimizer.hpp"
#include "training/rasterization/fast_rasterizer.hpp"
#include "training/rasterization/fastgs/rasterization/include/forward.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <fstream>
#include <gtest/gtest.h>
#include <memory>
#include <sstream>
#include <string>
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

    // Deterministic synthetic cloud (same layout as WarpCullBlendTest).
    std::unique_ptr<SplatData> make_synthetic_splat(int n, uint32_t seed = 0) {
        (void)seed;
        auto means = Tensor::zeros({static_cast<size_t>(n), 3}, Device::CUDA);
        if (n > 0) {
            auto cpu = means.to(Device::CPU);
            float* p = cpu.ptr<float>();
            for (int i = 0; i < n; ++i) {
                p[i * 3 + 0] = ((i % 8) - 3.5f) * 0.35f;
                p[i * 3 + 1] = ((i / 8) % 8 - 3.5f) * 0.35f;
                p[i * 3 + 2] = 0.0f;
            }
            means = cpu.to(Device::CUDA);
        }
        auto sh0 = Tensor::full({static_cast<size_t>(n), 1, 3}, 0.55f, Device::CUDA);
        auto shN = Tensor::zeros({static_cast<size_t>(n), 0, 3}, Device::CUDA);
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

    struct ParamSnapshot {
        Tensor means;
        Tensor opacity;
        Tensor sh0;
        Tensor scaling;
        bool ok = false;
    };

    // One forward (cull ON) + backward at the given bwd cull mode + fused Adam step.
    // Returns post-step parameters (fused path applies the step inside backward).
    ParamSnapshot run_one_step(Camera& cam, std::unique_ptr<SplatData> splat, Tensor& bg, int bwd_cull_mode) {
        ParamSnapshot out{};
        set_warp_cull_mode_for_testing(0); // forward production cull
        set_blend_batch_size_for_testing(0);
        auto r = fast_rasterize_forward(cam, *splat, bg, 0, 0, 0, 0, false);
        EXPECT_TRUE(r.has_value()) << (r.has_value() ? "" : lfs::format_for_developer(r.error()));
        if (!r.has_value()) {
            return out;
        }

        AdamConfig cfg{.lr = 0.01f, .beta1 = 0.9, .beta2 = 0.999, .eps = 1e-15};
        auto opt = std::make_unique<AdamOptimizer>(*splat, cfg);
        opt->allocate_gradients();
        opt->zero_grad(0);

        set_warp_cull_mode_for_testing(bwd_cull_mode);
        auto grad_out = r->first.image.mul(2.0f);
        fast_rasterize_backward(r->second, grad_out, *splat, *opt, {}, {}, DensificationType::None, 1);
        set_warp_cull_mode_for_testing(0);

        out.means = splat->means().to(Device::CPU).contiguous().clone();
        out.opacity = splat->opacity_raw().to(Device::CPU).contiguous().clone();
        out.sh0 = splat->sh0().to(Device::CPU).contiguous().clone();
        out.scaling = splat->scaling_raw().to(Device::CPU).contiguous().clone();
        out.ok = true;

        r->second.release_forward_context();
        EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        return out;
    }

    float max_abs_diff(const Tensor& a, const Tensor& b) {
        if (!a.is_valid() || !b.is_valid() || a.numel() != b.numel() || a.numel() == 0) {
            return 1e30f;
        }
        const float* pa = a.ptr<float>();
        const float* pb = b.ptr<float>();
        float m = 0.f;
        for (size_t i = 0; i < a.numel(); ++i) {
            m = std::max(m, std::abs(pa[i] - pb[i]));
        }
        return m;
    }

    std::string load_kernels_backward_source() {
        const char* candidates[] = {
            "src/training/rasterization/fastgs/rasterization/include/kernels_backward.cuh",
            "../src/training/rasterization/fastgs/rasterization/include/kernels_backward.cuh",
            "../../src/training/rasterization/fastgs/rasterization/include/kernels_backward.cuh",
            "/home/gauss/projects/LichtFeld-Studio/src/training/rasterization/fastgs/rasterization/include/kernels_backward.cuh",
        };
        for (const char* path : candidates) {
            std::ifstream in(path);
            if (!in)
                continue;
            std::ostringstream ss;
            ss << in.rdbuf();
            return ss.str();
        }
        return {};
    }

} // namespace

class WarpCullBwdTest : public ::testing::Test {
protected:
    void SetUp() override {
        cleanup_arena();
        set_warp_cull_mode_for_testing(0);
        set_blend_batch_size_for_testing(0);
        camera_ = std::make_unique<Camera>(make_camera(64, 64));
        bg_ = Tensor::zeros({3}, Device::CUDA);
    }
    void TearDown() override {
        set_warp_cull_mode_for_testing(0);
        set_blend_batch_size_for_testing(0);
        cleanup_arena();
    }

    std::unique_ptr<Camera> camera_;
    Tensor bg_;
};

// ---------------------------------------------------------------------------
// Sync-count: reverse-walk body must have zero block fences.
// ---------------------------------------------------------------------------
TEST_F(WarpCullBwdTest, ReverseWalkHasZeroBlockFences) {
    const std::string src = load_kernels_backward_source();
    ASSERT_FALSE(src.empty()) << "could not locate kernels_backward.cuh";

    const std::string begin_mark = "WARP_BWD_WALK_BEGIN";
    const std::string end_mark = "WARP_BWD_WALK_END";
    const auto b = src.find(begin_mark);
    const auto e = src.find(end_mark);
    ASSERT_NE(b, std::string::npos) << "missing " << begin_mark << " marker";
    ASSERT_NE(e, std::string::npos) << "missing " << end_mark << " marker";
    ASSERT_LT(b, e);

    const std::string walk = src.substr(b, e - b);
    EXPECT_EQ(walk.find("block.sync()"), std::string::npos)
        << "block.sync() found inside WARP_BWD_WALK (must be 0)";
    EXPECT_EQ(walk.find("__syncthreads"), std::string::npos)
        << "__syncthreads found inside WARP_BWD_WALK (must be 0)";
    EXPECT_EQ(walk.find("for (int diagonal"), std::string::npos)
        << "legacy diagonal reverse walk still present";
}

// A deliberately empty mask must differ from the cull-disabled reference.
TEST_F(WarpCullBwdTest, WrongMaskDiffersFromReference) {
    auto ref = run_one_step(*camera_, make_synthetic_splat(48), bg_, /*bwd_cull_mode=*/1);
    auto wrong = run_one_step(*camera_, make_synthetic_splat(48), bg_, /*bwd_cull_mode=*/2);
    ASSERT_TRUE(ref.ok && wrong.ok);

    const float max_means = max_abs_diff(ref.means, wrong.means);
    const float max_opac = max_abs_diff(ref.opacity, wrong.opacity);
    // Empty cull mask suppresses blend grads → params must diverge after the step.
    EXPECT_GT(max_means + max_opac, 1e-5f)
        << "wrong empty mask should change post-step params (test sensitivity); "
        << "max_means=" << max_means << " max_opac=" << max_opac;
}

// ---------------------------------------------------------------------------
// Grad-epsilon goldens: enabled cull vs reference (cull off) within 1e-6.
// ---------------------------------------------------------------------------
TEST_F(WarpCullBwdTest, EnabledMatchesReference_Synthetic) {
    auto ref = run_one_step(*camera_, make_synthetic_splat(48), bg_, /*bwd_cull_mode=*/1);
    auto on = run_one_step(*camera_, make_synthetic_splat(48), bg_, /*bwd_cull_mode=*/0);
    ASSERT_TRUE(ref.ok && on.ok);

    constexpr float kTol = 1e-6f;
    EXPECT_LE(max_abs_diff(ref.means, on.means), kTol) << "means";
    EXPECT_LE(max_abs_diff(ref.opacity, on.opacity), kTol) << "opacity";
    EXPECT_LE(max_abs_diff(ref.sh0, on.sh0), kTol) << "sh0";
    EXPECT_LE(max_abs_diff(ref.scaling, on.scaling), kTol) << "scaling";
}

TEST_F(WarpCullBwdTest, EnabledMatchesReference_Dense) {
    auto ref = run_one_step(*camera_, make_dense_splat(256), bg_, /*bwd_cull_mode=*/1);
    auto on = run_one_step(*camera_, make_dense_splat(256), bg_, /*bwd_cull_mode=*/0);
    ASSERT_TRUE(ref.ok && on.ok);

    constexpr float kTol = 1e-6f;
    EXPECT_LE(max_abs_diff(ref.means, on.means), kTol) << "means";
    EXPECT_LE(max_abs_diff(ref.opacity, on.opacity), kTol) << "opacity";
    EXPECT_LE(max_abs_diff(ref.sh0, on.sh0), kTol) << "sh0";
    EXPECT_LE(max_abs_diff(ref.scaling, on.scaling), kTol) << "scaling";
}

// ---------------------------------------------------------------------------
// Determinism: two enabled-cull steps match.
// ---------------------------------------------------------------------------
TEST_F(WarpCullBwdTest, EnabledIsDeterministic) {
    auto a = run_one_step(*camera_, make_dense_splat(256), bg_, /*bwd_cull_mode=*/0);
    auto b = run_one_step(*camera_, make_dense_splat(256), bg_, /*bwd_cull_mode=*/0);
    ASSERT_TRUE(a.ok && b.ok);
    // Multi-warp atomicAdd order can inject sub-ULP noise; stay within golden tol.
    constexpr float kTol = 1e-6f;
    EXPECT_LE(max_abs_diff(a.means, b.means), kTol);
    EXPECT_LE(max_abs_diff(a.opacity, b.opacity), kTol);
}

namespace {

    bool tensors_bit_identical(const Tensor& a, const Tensor& b) {
        if (!a.is_valid() || !b.is_valid() || a.numel() != b.numel()) {
            return false;
        }
        if (a.numel() == 0) {
            return true;
        }
        const auto* pa = reinterpret_cast<const std::uint8_t*>(a.ptr<float>());
        const auto* pb = reinterpret_cast<const std::uint8_t*>(b.ptr<float>());
        return std::memcmp(pa, pb, a.numel() * sizeof(float)) == 0;
    }

    struct Warp8x4Snapshot {
        Tensor image;
        Tensor alpha;
        Tensor depth;
        Tensor normal;
        Tensor means;
        Tensor opacity;
        Tensor sh0;
        Tensor scaling;
        Tensor densification;
        bool ok = false;
    };

    Warp8x4Snapshot run_8x4_step() {
        Warp8x4Snapshot out{};
        Camera cam = make_camera(8, 4);
        auto splat = make_synthetic_splat(12);
        splat->_densification_info = Tensor::zeros({2, static_cast<size_t>(12)}, Device::CUDA);
        Tensor bg = Tensor::zeros({3}, Device::CUDA);

        set_warp_cull_mode_for_testing(0);
        set_blend_batch_size_for_testing(0);
        auto r = fast_rasterize_forward(cam, *splat, bg, 0, 0, 0, 0, false, {}, true);
        if (!r.has_value()) {
            ADD_FAILURE() << lfs::format_for_developer(r.error());
            return out;
        }

        AdamConfig cfg{.lr = 0.01f, .beta1 = 0.9, .beta2 = 0.999, .eps = 1e-15};
        auto opt = std::make_unique<AdamOptimizer>(*splat, cfg);
        opt->allocate_gradients();
        opt->zero_grad(0);

        auto grad_out = r->first.image.mul(2.0f);
        auto grad_depth = Tensor::ones_like(r->first.depth);
        auto grad_normal = Tensor::ones_like(r->first.normal);
        auto pixel_error = Tensor::ones({4, 8}, Device::CUDA);
        fast_rasterize_backward(
            r->second,
            grad_out,
            *splat,
            *opt,
            {},
            pixel_error,
            DensificationType::MCMC,
            1,
            {},
            grad_depth,
            grad_normal);

        out.image = r->first.image.to(Device::CPU).contiguous().clone();
        out.alpha = r->first.alpha.to(Device::CPU).contiguous().clone();
        out.depth = r->first.depth.to(Device::CPU).contiguous().clone();
        out.normal = r->first.normal.to(Device::CPU).contiguous().clone();
        out.means = splat->means().to(Device::CPU).contiguous().clone();
        out.opacity = splat->opacity_raw().to(Device::CPU).contiguous().clone();
        out.sh0 = splat->sh0().to(Device::CPU).contiguous().clone();
        out.scaling = splat->scaling_raw().to(Device::CPU).contiguous().clone();
        out.densification = splat->_densification_info.to(Device::CPU).contiguous().clone();
        out.ok = true;

        r->second.release_forward_context();
        EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        return out;
    }

} // namespace

// 8x4 image = one warp sub-tile, so blend_backward atomics have a single writer
// per splat/field and are bit-deterministic (dense tiles are not).
TEST_F(WarpCullBwdTest, SingleWarp8x4_BitIdenticalAcrossRuns) {
    auto a = run_8x4_step();
    auto b = run_8x4_step();
    ASSERT_TRUE(a.ok && b.ok);
    EXPECT_TRUE(tensors_bit_identical(a.image, b.image)) << "image";
    EXPECT_TRUE(tensors_bit_identical(a.alpha, b.alpha)) << "alpha";
    EXPECT_TRUE(tensors_bit_identical(a.depth, b.depth)) << "depth";
    EXPECT_TRUE(tensors_bit_identical(a.normal, b.normal)) << "normal";
    EXPECT_TRUE(tensors_bit_identical(a.means, b.means)) << "means";
    EXPECT_TRUE(tensors_bit_identical(a.opacity, b.opacity)) << "opacity";
    EXPECT_TRUE(tensors_bit_identical(a.sh0, b.sh0)) << "sh0";
    EXPECT_TRUE(tensors_bit_identical(a.scaling, b.scaling)) << "scaling";
    EXPECT_TRUE(tensors_bit_identical(a.densification, b.densification)) << "densification";
}

/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Gsplat / 3DGUT path: forward smoke + persistent high-water isect buffers.
 */

#include "core/alloc_counter.hpp"
#include "core/camera.hpp"
#include "core/cuda/memory_arena.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "lfs/training/sh_value_codec.hpp"
#include "lfs/training/sh_value_storage.hpp"
#include "optimizer/adam_optimizer.hpp"
#include "training/rasterization/gsplat/Common.h"
#include "training/rasterization/gsplat/Ops.h"
#include "training/rasterization/gsplat_rasterizer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace lfs::training;
using namespace lfs::core;

namespace {

    void release_ctx_arena(GsplatRasterizeContext& ctx) {
        // Isect pointers are TLS high-water — never cudaFree them.
        ctx.isect_ids_ptr = nullptr;
        ctx.flatten_ids_ptr = nullptr;
        GlobalArenaManager::instance().get_arena().end_frame(ctx.frame_id, ctx.stream);
    }

    Camera make_camera(int w, int h) {
        // Qualify CameraModelType: gsplat Common.h also defines a global enum
        // of the same name (Cameras.cuh compat), which shadows lfs::core's.
        std::vector<float> R_data = {1, 0, 0, 0, 1, 0, 0, 0, 1};
        std::vector<float> T_data = {0, 0, 3};
        auto R = Tensor::from_blob(R_data.data(), {3, 3}, Device::CPU, DataType::Float32).to(Device::CUDA);
        auto T = Tensor::from_blob(T_data.data(), {3}, Device::CPU, DataType::Float32).to(Device::CUDA);
        return Camera(
            R, T,
            500.f, 500.f,
            static_cast<float>(w) * 0.5f, static_cast<float>(h) * 0.5f,
            Tensor(), Tensor(),
            lfs::core::CameraModelType::PINHOLE,
            "test",
            "",
            std::filesystem::path{},
            w, h,
            0);
    }

    // London OPENCV_FISHEYE (COLMAP model 5, 3504x2336) scaled to (w,h).
    Camera make_fisheye_camera(int w, int h) {
        std::vector<float> R_data = {1, 0, 0, 0, 1, 0, 0, 0, 1};
        std::vector<float> T_data = {0, 0, 3};
        auto R = Tensor::from_blob(R_data.data(), {3, 3}, Device::CPU, DataType::Float32).to(Device::CUDA);
        auto T = Tensor::from_blob(T_data.data(), {3}, Device::CPU, DataType::Float32).to(Device::CUDA);
        const float fx = 1212.4493198497419f * (static_cast<float>(w) / 3504.f);
        const float fy = 1212.5428599857478f * (static_cast<float>(h) / 2336.f);
        auto radial = Tensor::from_vector(
            {0.03556234340872849f, 0.007709264733622488f,
             0.0006652110074087792f, -0.0003144582282973898f},
            {4}, Device::CPU);
        return Camera(
            R, T,
            fx, fy,
            static_cast<float>(w) * 0.5f, static_cast<float>(h) * 0.5f,
            radial, Tensor(),
            lfs::core::CameraModelType::FISHEYE,
            "test_fisheye",
            "",
            std::filesystem::path{},
            w, h,
            0);
    }

    std::unique_ptr<SplatData> make_visible_splat(int n) {
        auto means = Tensor::zeros({static_cast<size_t>(n), 3}, Device::CUDA);
        if (n > 0) {
            auto cpu = means.to(Device::CPU);
            float* p = cpu.ptr<float>();
            for (int i = 0; i < n; ++i) {
                p[i * 3 + 0] = (i % 5) * 0.3f - 0.6f;
                p[i * 3 + 1] = (i / 5) * 0.3f - 0.6f;
                p[i * 3 + 2] = 0.0f; // in front of camera at z=3
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

    // Deterministic ~50k-splat GUT fixture for gradient / image parity.
    std::unique_ptr<SplatData> make_parity_splat(int n, uint32_t seed) {
        std::vector<float> means(static_cast<size_t>(n) * 3);
        std::vector<float> sh0(static_cast<size_t>(n) * 3);
        std::vector<float> scaling(static_cast<size_t>(n) * 3);
        std::vector<float> rotation(static_cast<size_t>(n) * 4);
        std::vector<float> opacity(static_cast<size_t>(n));
        uint32_t s = seed;
        auto rnd = [&]() {
            s = s * 1664525u + 1013904223u;
            return static_cast<float>(s >> 8) * (1.0f / 16777216.0f);
        };
        for (int i = 0; i < n; ++i) {
            means[static_cast<size_t>(i) * 3 + 0] = rnd() * 2.4f - 1.2f;
            means[static_cast<size_t>(i) * 3 + 1] = rnd() * 2.4f - 1.2f;
            means[static_cast<size_t>(i) * 3 + 2] = rnd() * 1.0f - 0.5f;
            sh0[static_cast<size_t>(i) * 3 + 0] = rnd() * 0.8f + 0.1f;
            sh0[static_cast<size_t>(i) * 3 + 1] = rnd() * 0.8f + 0.1f;
            sh0[static_cast<size_t>(i) * 3 + 2] = rnd() * 0.8f + 0.1f;
            scaling[static_cast<size_t>(i) * 3 + 0] = -3.2f + rnd() * 0.6f;
            scaling[static_cast<size_t>(i) * 3 + 1] = -3.2f + rnd() * 0.6f;
            scaling[static_cast<size_t>(i) * 3 + 2] = -3.2f + rnd() * 0.6f;
            rotation[static_cast<size_t>(i) * 4 + 0] = 0.7f + rnd();
            rotation[static_cast<size_t>(i) * 4 + 1] = rnd() * 0.4f - 0.2f;
            rotation[static_cast<size_t>(i) * 4 + 2] = rnd() * 0.4f - 0.2f;
            rotation[static_cast<size_t>(i) * 4 + 3] = rnd() * 0.4f - 0.2f;
            opacity[static_cast<size_t>(i)] = 0.5f + rnd() * 2.0f;
        }
        auto means_t = Tensor::from_blob(means.data(), {static_cast<size_t>(n), 3}, Device::CPU, DataType::Float32)
                           .to(Device::CUDA);
        auto sh0_t = Tensor::from_blob(sh0.data(), {static_cast<size_t>(n), 1, 3}, Device::CPU, DataType::Float32)
                         .to(Device::CUDA);
        auto shN_t = Tensor::zeros({static_cast<size_t>(n), 0, 3}, Device::CUDA);
        auto scaling_t = Tensor::from_blob(scaling.data(), {static_cast<size_t>(n), 3}, Device::CPU, DataType::Float32)
                             .to(Device::CUDA);
        auto rotation_t = Tensor::from_blob(rotation.data(), {static_cast<size_t>(n), 4}, Device::CPU, DataType::Float32)
                              .to(Device::CUDA);
        auto opacity_t = Tensor::from_blob(opacity.data(), {static_cast<size_t>(n)}, Device::CPU, DataType::Float32)
                             .to(Device::CUDA);
        return std::make_unique<SplatData>(0, means_t, sh0_t, shN_t, scaling_t, rotation_t, opacity_t, 1.0f);
    }

    void write_float_bin(const std::filesystem::path& path, const Tensor& t) {
        auto cpu = t.cpu();
        std::ofstream out(path, std::ios::binary);
        ASSERT_TRUE(out) << "failed to write " << path;
        const auto n = static_cast<size_t>(cpu.numel());
        out.write(reinterpret_cast<const char*>(cpu.ptr<float>()),
                  static_cast<std::streamsize>(n * sizeof(float)));
        ASSERT_TRUE(out) << "failed to write " << path;
    }

    std::vector<float> read_float_bin(const std::filesystem::path& path) {
        std::ifstream in(path, std::ios::binary | std::ios::ate);
        EXPECT_TRUE(in) << "failed to read " << path;
        if (!in) {
            return {};
        }
        const auto bytes = static_cast<size_t>(in.tellg());
        in.seekg(0);
        std::vector<float> v(bytes / sizeof(float));
        in.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(bytes));
        return v;
    }

    float max_rel_diff(const Tensor& a, const std::vector<float>& b) {
        auto cpu = a.cpu();
        const auto n = static_cast<size_t>(cpu.numel());
        EXPECT_EQ(n, b.size());
        if (n != b.size()) {
            return 1.0f;
        }
        const float* p = cpu.ptr<float>();
        float max_abs = 0.0f;
        float max_diff = 0.0f;
        for (size_t i = 0; i < n; ++i) {
            max_abs = std::max(max_abs, std::abs(p[i]));
            max_diff = std::max(max_diff, std::abs(p[i] - b[i]));
        }
        const float denom = std::max(max_abs, 1e-8f);
        return max_diff / denom;
    }

    float max_rel_diff_tensors(const Tensor& a, const Tensor& b) {
        auto ac = a.cpu();
        auto bc = b.cpu();
        const auto n = static_cast<size_t>(ac.numel());
        EXPECT_EQ(n, static_cast<size_t>(bc.numel()));
        const float* pa = ac.ptr<float>();
        const float* pb = bc.ptr<float>();
        float max_abs = 0.0f;
        float max_diff = 0.0f;
        for (size_t i = 0; i < n; ++i) {
            max_abs = std::max(max_abs, std::abs(pa[i]));
            max_diff = std::max(max_diff, std::abs(pa[i] - pb[i]));
        }
        return max_diff / std::max(max_abs, 1e-8f);
    }

    float max_abs_diff_tensors(const Tensor& a, const Tensor& b) {
        auto ac = a.cpu();
        auto bc = b.cpu();
        const auto n = static_cast<size_t>(ac.numel());
        EXPECT_EQ(n, static_cast<size_t>(bc.numel()));
        const float* pa = ac.ptr<float>();
        const float* pb = bc.ptr<float>();
        float max_diff = 0.0f;
        for (size_t i = 0; i < n; ++i) {
            max_diff = std::max(max_diff, std::abs(pa[i] - pb[i]));
        }
        return max_diff;
    }

} // namespace

class GsplatRasterizerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create minimal test data
        const size_t N = 100; // Number of Gaussians
        const int sh_degree = 0;

        // Create random Gaussian parameters
        means_ = Tensor::randn({N, 3}, Device::CUDA, DataType::Float32);
        sh0_ = Tensor::randn({N, 1, 3}, Device::CUDA, DataType::Float32);            // sh0 is [N, 1, 3]
        shN_ = Tensor::zeros({N, 0, 3}, Device::CUDA, DataType::Float32);            // No higher SH for degree 0
        scaling_ = Tensor::randn({N, 3}, Device::CUDA, DataType::Float32).mul(0.1f); // Small scales
        rotation_ = Tensor::randn({N, 4}, Device::CUDA, DataType::Float32);
        opacity_ = Tensor::randn({N}, Device::CUDA, DataType::Float32);

        // Create SplatData
        splat_data_ = std::make_unique<SplatData>(
            sh_degree,
            means_,
            sh0_,
            shN_,
            scaling_,
            rotation_,
            opacity_,
            1.0f // scene_scale
        );

        // Create camera
        auto R = Tensor::eye(3, Device::CUDA);
        auto T = Tensor::zeros({3}, Device::CUDA, DataType::Float32);

        // Set camera at z=3 looking at origin
        std::vector<float> T_data = {0.0f, 0.0f, 3.0f};
        T = Tensor::from_blob(T_data.data(), {3}, Device::CPU, DataType::Float32).to(Device::CUDA);

        camera_ = std::make_unique<Camera>(
            R, T,
            500.0f, 500.0f, // focal_x, focal_y
            320.0f, 240.0f, // center_x, center_y
            Tensor(),       // radial_distortion
            Tensor(),       // tangential_distortion
            lfs::core::CameraModelType::PINHOLE,
            "test_image",
            "",
            std::filesystem::path{}, // mask_path
            640, 480,                // camera_width, camera_height (constructor sets image_width/height too)
            0                        // uid
        );

        // Background color
        bg_color_ = Tensor::zeros({3}, Device::CUDA, DataType::Float32);
        bg_color_.fill_(0.5f); // Gray background
    }

    void TearDown() override {
#if LFS_CUDA_FAILURE_INJECTION_ENABLED
        gsplat_lfs::set_cuda_allocation_failure_for_testing(false);
#endif
        (void)gsplat_lfs::release_intersect_thread_local_cache();
        (void)release_gsplat_rasterizer_thread_local_caches();
        GlobalArenaManager::instance().get_arena().full_reset();
    }

    std::unique_ptr<SplatData> splat_data_;
    std::unique_ptr<Camera> camera_;
    Tensor means_, sh0_, shN_, scaling_, rotation_, opacity_;
    Tensor bg_color_;
};

#if LFS_CUDA_FAILURE_INJECTION_ENABLED
TEST_F(GsplatRasterizerTest, CudaAllocationFailureAbortsAndRecovers) {
    gsplat_lfs::set_cuda_allocation_failure_for_testing(true);
    EXPECT_THROW(
        (void)gsplat_rasterize_forward(
            *camera_, *splat_data_, bg_color_,
            0, 0, 0, 0, 1.0f, false, GsplatRenderMode::RGB),
        std::runtime_error);

    gsplat_lfs::set_cuda_allocation_failure_for_testing(false);
    auto result = gsplat_rasterize_forward(
        *camera_, *splat_data_, bg_color_,
        0, 0, 0, 0, 1.0f, false, GsplatRenderMode::RGB);
    ASSERT_TRUE(result.has_value());

    auto& ctx = result->second;
    release_ctx_arena(ctx);
}
#endif

TEST_F(GsplatRasterizerTest, ForwardPassBasic) {
    // Just test that forward pass doesn't crash
    auto result = gsplat_rasterize_forward(
        *camera_, *splat_data_, bg_color_,
        0, 0, 0, 0, 1.0f, false, GsplatRenderMode::RGB);

    ASSERT_TRUE(result.has_value()) << "Forward pass failed: " << result.error();

    auto& [render_output, ctx] = result.value();

    // Check output dimensions
    EXPECT_EQ(render_output.width, 640);
    EXPECT_EQ(render_output.height, 480);
    EXPECT_TRUE(render_output.image.is_valid());
    EXPECT_EQ(render_output.image.shape()[0], 3); // CHW format
    EXPECT_EQ(render_output.image.shape()[1], 480);
    EXPECT_EQ(render_output.image.shape()[2], 640);

    // Check alpha
    EXPECT_TRUE(render_output.alpha.is_valid());
    EXPECT_EQ(render_output.alpha.shape()[0], 1);
    EXPECT_EQ(render_output.alpha.shape()[1], 480);
    EXPECT_EQ(render_output.alpha.shape()[2], 640);

    std::cout << "Forward pass succeeded!" << std::endl;
    std::cout << "  Image shape: [" << render_output.image.shape()[0] << ", "
              << render_output.image.shape()[1] << ", "
              << render_output.image.shape()[2] << "]" << std::endl;

    release_ctx_arena(ctx);
}

TEST_F(GsplatRasterizerTest, InferenceWrapper) {
    // Test the convenience wrapper
    EXPECT_NO_THROW({
        auto output = gsplat_rasterize(*camera_, *splat_data_, bg_color_);
        EXPECT_TRUE(output.image.is_valid());
    });
}

// The gsplat intersection buffers are grow-only thread-local storage; a second
// same-size forward must not allocate them again.
TEST_F(GsplatRasterizerTest, SteadyStateSecondForwardHasZeroIsectAllocs) {
    // Visible fixture so n_isects > 0 and the isect/sort path runs.
    auto camera = make_camera(64, 64);
    auto splat = make_visible_splat(32);
    auto bg = Tensor::zeros({3}, Device::CUDA);

    auto run_once = [&]() {
        auto r = gsplat_rasterize_forward(
            camera, *splat, bg, 0, 0, 0, 0, 1.0f, false, GsplatRenderMode::RGB,
            /*use_gut=*/true);
        ASSERT_TRUE(r.has_value()) << r.error();
        ASSERT_GT(r->second.n_isects, 0)
            << "fixture must produce intersections so the isect path runs";
        release_ctx_arena(r->second);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    };

    // Warmup: arena + TLS image caches + first isect/sort/CUB growth.
    run_once();
    // Second pass at same size — may still finish residual growth; absorb it.
    {
        const auto snap = alloc_counter::snapshot();
        run_once();
        (void)alloc_counter::delta_since(snap);
    }

    // Steady-state third forward: high-water pools must issue 0 driver allocs.
    const auto snap2 = alloc_counter::snapshot();
    run_once();
    const auto delta2 = alloc_counter::delta_since(snap2);
    EXPECT_EQ(delta2, 0u)
        << "steady-state gsplat forward at fixed size must issue 0 real device "
           "allocs (isect_ids, flatten_ids, sort pairs, CUB WS, cum_tiles are "
           "grow-only). Observed delta="
        << delta2;
}

TEST_F(GsplatRasterizerTest, GutModeSteadyStateAllocs) {
    auto camera = make_camera(128, 128);
    auto splat = make_visible_splat(256);
    auto bg = Tensor::zeros({3}, Device::CUDA);

    constexpr int kWarmup = 5;
    constexpr int kIters = 30;

    for (int i = 0; i < kWarmup; ++i) {
        auto r = gsplat_rasterize_forward(
            camera, *splat, bg, 0, 0, 0, 0, 1.0f, false, GsplatRenderMode::RGB, true);
        ASSERT_TRUE(r.has_value()) << r.error();
        release_ctx_arena(r->second);
    }
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::uint64_t allocs = 0;
    for (int i = 0; i < kIters; ++i) {
        const auto snap = alloc_counter::snapshot();
        auto r = gsplat_rasterize_forward(
            camera, *splat, bg, 0, 0, 0, 0, 1.0f, false, GsplatRenderMode::RGB, true);
        ASSERT_TRUE(r.has_value()) << r.error();
        ASSERT_GT(r->second.n_isects, 0);
        release_ctx_arena(r->second);
        allocs += alloc_counter::delta_since(snap);
    }
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    const double allocs_per = static_cast<double>(allocs) / static_cast<double>(kIters);

    // Steady high-water: average allocs per forward should be ~0.
    EXPECT_LT(allocs_per, 0.5)
        << "gut steady-state should not touch the driver every forward";
}

// gut/gsplat forward+backward with default quant ON + sh_degree>0.
// Saves dequant temp in ctx so backward does not dtype-abort on q16 codes.
TEST(GsplatRasterizerQuantTest, GutForwardBackwardWithDefaultQuantAndShDegree) {
    // Default flags: quant ON (no force-off).
    lfs::training::sh_value::set_sh_value_quant_enabled_for_testing(true);

    auto camera = make_camera(64, 64);
    constexpr size_t n = 24;
    constexpr int sh_degree = 3;
    constexpr size_t rest = 15;

    std::vector<float> means(n * 3, 0.0f);
    std::vector<float> rots(n * 4, 0.0f);
    for (size_t i = 0; i < n; ++i) {
        means[i * 3 + 0] = (static_cast<float>(i % 5) * 0.3f) - 0.6f;
        means[i * 3 + 1] = (static_cast<float>(i / 5) * 0.3f) - 0.6f;
        means[i * 3 + 2] = 0.0f;
        rots[i * 4] = 1.0f;
    }
    auto shN = Tensor::full({n, rest, size_t{3}}, 0.05f, Device::CUDA);
    auto splat = SplatData(
        sh_degree,
        Tensor::from_vector(means, {n, size_t{3}}, Device::CUDA),
        Tensor::full({n, size_t{1}, size_t{3}}, 0.5f, Device::CUDA),
        std::move(shN),
        Tensor::full({n, size_t{3}}, -2.0f, Device::CUDA),
        Tensor::from_vector(rots, {n, size_t{4}}, Device::CUDA),
        Tensor::full({n, size_t{1}}, 2.0f, Device::CUDA),
        1.0f);
    ASSERT_TRUE(lfs::training::sh_value::apply_shN_value_quant(splat));
    ASSERT_TRUE(splat.shN_value_quantized());

    AdamConfig cfg;
    cfg.lr = 1e-3f;
    cfg.initial_capacity = n * 2;
    AdamOptimizer opt(splat, cfg);
    opt.allocate_gradients(n * 2);

    auto bg = Tensor::zeros({3}, Device::CUDA);
    auto result = gsplat_rasterize_forward(
        camera, splat, bg, 0, 0, 0, 0, 1.0f, false, GsplatRenderMode::RGB,
        /*use_gut=*/true);
    ASSERT_TRUE(result.has_value()) << result.error();
    auto& [output, ctx] = *result;
    // ctx must hold float dequant, not raw Float16 codes.
    ASSERT_TRUE(ctx.shN.is_valid());
    EXPECT_EQ(ctx.shN.dtype(), DataType::Float32)
        << "backward requires float dequant temp in ctx under q16";

    auto grad_image = Tensor::ones_like(output.image);
    auto grad_alpha = Tensor::zeros_like(output.alpha);
    ASSERT_NO_THROW({
        gsplat_rasterize_backward(ctx, grad_image, grad_alpha, splat, opt, Tensor{});
    });

    // Arena cleanup
    ctx.isect_ids_ptr = nullptr;
    ctx.flatten_ids_ptr = nullptr;
    GlobalArenaManager::instance().get_arena().end_frame(ctx.frame_id, ctx.stream);

    lfs::training::sh_value::set_sh_value_quant_enabled_for_testing(std::nullopt);
}

TEST(GsplatRasterizerQuantTest, RejectsFloat16ShRestWithoutQ16Bounds) {
    auto camera = make_camera(32, 32);
    constexpr size_t n = 4;
    constexpr size_t rest = 3;

    std::vector<float> rotations(n * 4, 0.0f);
    for (size_t i = 0; i < n; ++i) {
        rotations[i * 4] = 1.0f;
    }
    auto splat = SplatData(
        1,
        Tensor::zeros({n, size_t{3}}, Device::CUDA),
        Tensor::full({n, size_t{1}, size_t{3}}, 0.5f, Device::CUDA),
        Tensor::full({n, rest, size_t{3}}, 0.05f, Device::CUDA),
        Tensor::full({n, size_t{3}}, -2.0f, Device::CUDA),
        Tensor::from_vector(rotations, {n, size_t{4}}, Device::CUDA),
        Tensor::full({n, size_t{1}}, 2.0f, Device::CUDA),
        1.0f);
    // Canonical construction currently accepts Float32 only. Convert the valid
    // resident float4 swizzle afterward to model the viewer's IEEE-f16 storage
    // without q16 bounds.
    splat.shN() = splat.shN().to(DataType::Float16);
    ASSERT_EQ(splat.shN().dtype(), DataType::Float16);
    ASSERT_FALSE(splat.shN_value_quantized());

    auto background = Tensor::zeros({3}, Device::CUDA);
    try {
        (void)gsplat_rasterize_forward(
            camera, splat, background, 0, 0, 0, 0, 1.0f, false,
            GsplatRenderMode::RGB, /*use_gut=*/true);
        FAIL() << "gsplat accepted non-q16 Float16 SH-rest";
    } catch (const std::runtime_error& error) {
        EXPECT_NE(std::string_view(error.what()).find("unsupported SH-rest storage"),
                  std::string_view::npos);
    }
}

TEST_F(GsplatRasterizerTest, ForwardWritesChwAndBackwardIsStable) {
    auto camera = make_camera(32, 32);
    auto splat = make_visible_splat(16);
    auto bg = Tensor::zeros({3}, Device::CUDA);

    AdamConfig cfg;
    cfg.lr = 1e-3f;
    cfg.initial_capacity = 32;
    AdamOptimizer opt(*splat, cfg);
    opt.allocate_gradients(32);

    auto run_once = [&]() {
        auto r = gsplat_rasterize_forward(
            camera, *splat, bg, 0, 0, 0, 0, 1.0f, false, GsplatRenderMode::RGB,
            /*use_gut=*/true);
        EXPECT_TRUE(r.has_value()) << r.error();
        auto output = std::move(r->first);
        auto ctx = std::move(r->second);
        EXPECT_TRUE(output.image.is_valid());
        EXPECT_EQ(output.image.ndim(), 3);
        EXPECT_EQ(output.image.shape()[0], 3u);
        auto grad_image = Tensor::ones_like(output.image);
        auto grad_alpha = Tensor::zeros_like(output.alpha);
        gsplat_rasterize_backward(ctx, grad_image, grad_alpha, *splat, opt, Tensor{});
        return output.image.clone();
    };

    auto image0 = run_once();
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    auto means_g = opt.get_grad(ParamType::Means).clone().cpu();
    auto scale_g = opt.get_grad(ParamType::Scaling).clone().cpu();
    auto quat_g = opt.get_grad(ParamType::Rotation).clone().cpu();
    auto opa_g = opt.get_grad(ParamType::Opacity).clone().cpu();
    float max_abs = 0.0f;
    auto bump = [&](const Tensor& t) {
        const auto* p = t.ptr<float>();
        for (size_t i = 0; i < static_cast<size_t>(t.numel()); ++i) {
            max_abs = std::max(max_abs, std::abs(p[i]));
            EXPECT_TRUE(std::isfinite(p[i]));
        }
    };
    bump(means_g);
    bump(scale_g);
    bump(quat_g);
    bump(opa_g);
    EXPECT_GT(max_abs, 0.0f);

    opt.zero_grad(1);
    auto image1 = run_once();
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    auto means_g2 = opt.get_grad(ParamType::Means).cpu();
    const auto* a = means_g.ptr<float>();
    const auto* b = means_g2.ptr<float>();
    float max_rel = 0.0f;
    for (size_t i = 0; i < static_cast<size_t>(means_g.numel()); ++i) {
        const float denom = std::max(max_abs, 1e-8f);
        max_rel = std::max(max_rel, std::abs(a[i] - b[i]) / denom);
    }
    EXPECT_LT(max_rel, 1e-5f);

    auto img0 = image0.cpu();
    auto img1 = image1.cpu();
    ASSERT_EQ(img0.numel(), img1.numel());
    const auto* p0 = img0.ptr<float>();
    const auto* p1 = img1.ptr<float>();
    for (size_t i = 0; i < static_cast<size_t>(img0.numel()); ++i) {
        EXPECT_EQ(p0[i], p1[i]);
        EXPECT_TRUE(std::isfinite(p0[i]));
    }
}

void run_gut_from_world_parity(Camera& camera, const char* dump_env, const char* ref_env) {
    constexpr int kN = 50000;
    auto splat = make_parity_splat(kN, 0xC0FFEE01u);
    auto bg = Tensor::zeros({3}, Device::CUDA);
    bg.fill_(0.25f);

    AdamConfig cfg;
    cfg.lr = 1e-3f;
    cfg.initial_capacity = static_cast<size_t>(kN);
    AdamOptimizer opt(*splat, cfg);
    opt.allocate_gradients(static_cast<size_t>(kN));

    Tensor image0, image1;
    auto run_once = [&](Tensor& image_out) {
        auto r = gsplat_rasterize_forward(
            camera, *splat, bg, 0, 0, 0, 0, 1.0f, false, GsplatRenderMode::RGB,
            /*use_gut=*/true);
        ASSERT_TRUE(r.has_value()) << r.error();
        auto output = std::move(r->first);
        auto ctx = std::move(r->second);
        ASSERT_GT(ctx.n_isects, 0);
        auto grad_image = Tensor::ones_like(output.image);
        auto grad_alpha = Tensor::zeros_like(output.alpha);
        gsplat_rasterize_backward(ctx, grad_image, grad_alpha, *splat, opt, Tensor{});
        image_out = output.image.clone();
        ctx.isect_ids_ptr = nullptr;
        ctx.flatten_ids_ptr = nullptr;
    };

    run_once(image0);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    auto means_g = opt.get_grad(ParamType::Means).clone();
    auto scale_g = opt.get_grad(ParamType::Scaling).clone();
    auto quat_g = opt.get_grad(ParamType::Rotation).clone();
    auto opa_g = opt.get_grad(ParamType::Opacity).clone();
    auto color_g = opt.get_grad(ParamType::Sh0).clone();

    opt.zero_grad(1);
    run_once(image1);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    EXPECT_EQ(max_abs_diff_tensors(image0, image1), 0.0f);
    EXPECT_LT(max_rel_diff_tensors(means_g, opt.get_grad(ParamType::Means)), 1e-4f);
    EXPECT_LT(max_rel_diff_tensors(scale_g, opt.get_grad(ParamType::Scaling)), 1e-4f);
    EXPECT_LT(max_rel_diff_tensors(quat_g, opt.get_grad(ParamType::Rotation)), 1e-4f);
    EXPECT_LT(max_rel_diff_tensors(opa_g, opt.get_grad(ParamType::Opacity)), 1e-4f);
    EXPECT_LT(max_rel_diff_tensors(color_g, opt.get_grad(ParamType::Sh0)), 1e-4f);

    if (const char* dump_dir = std::getenv(dump_env)) {
        std::filesystem::create_directories(dump_dir);
        write_float_bin(std::filesystem::path(dump_dir) / "v_means.bin", means_g);
        write_float_bin(std::filesystem::path(dump_dir) / "v_scales.bin", scale_g);
        write_float_bin(std::filesystem::path(dump_dir) / "v_quats.bin", quat_g);
        write_float_bin(std::filesystem::path(dump_dir) / "v_opacities.bin", opa_g);
        write_float_bin(std::filesystem::path(dump_dir) / "v_colors.bin", color_g);
        write_float_bin(std::filesystem::path(dump_dir) / "render.bin", image0);
        std::cout << dump_env << " wrote tensors to " << dump_dir << std::endl;
    }
    if (const char* ref_dir = std::getenv(ref_env)) {
        const auto rel_means = max_rel_diff(means_g, read_float_bin(std::filesystem::path(ref_dir) / "v_means.bin"));
        const auto rel_scales = max_rel_diff(scale_g, read_float_bin(std::filesystem::path(ref_dir) / "v_scales.bin"));
        const auto rel_quats = max_rel_diff(quat_g, read_float_bin(std::filesystem::path(ref_dir) / "v_quats.bin"));
        const auto rel_opa = max_rel_diff(opa_g, read_float_bin(std::filesystem::path(ref_dir) / "v_opacities.bin"));
        const auto rel_color = max_rel_diff(color_g, read_float_bin(std::filesystem::path(ref_dir) / "v_colors.bin"));
        auto render_ref = read_float_bin(std::filesystem::path(ref_dir) / "render.bin");
        auto render_cpu = image0.cpu();
        float render_max_abs = 0.0f;
        const auto nimg = static_cast<size_t>(render_cpu.numel());
        ASSERT_EQ(nimg, render_ref.size());
        const float* pi = render_cpu.ptr<float>();
        bool render_bit_identical = true;
        for (size_t i = 0; i < nimg; ++i) {
            render_max_abs = std::max(render_max_abs, std::abs(pi[i] - render_ref[i]));
            render_bit_identical = render_bit_identical && (pi[i] == render_ref[i]);
        }
        std::cout << ref_env << " max|a-b|/max|a| means=" << rel_means
                  << " scales=" << rel_scales << " quats=" << rel_quats
                  << " opacities=" << rel_opa << " colors=" << rel_color
                  << " render_max_abs=" << render_max_abs
                  << " render_bit_identical=" << (render_bit_identical ? "yes" : "no")
                  << std::endl;
        EXPECT_LT(rel_means, 1e-4f);
        EXPECT_LT(rel_scales, 1e-4f);
        EXPECT_LT(rel_quats, 1e-4f);
        EXPECT_LT(rel_opa, 1e-4f);
        EXPECT_LT(rel_color, 1e-4f);
        EXPECT_LE(render_max_abs, 1e-6f);
        EXPECT_TRUE(render_bit_identical);
    }
}

TEST_F(GsplatRasterizerTest, GutFromWorldGradParity) {
    constexpr int kW = 96;
    constexpr int kH = 64;
    auto camera = make_camera(kW, kH);
    run_gut_from_world_parity(camera, "GUT_GRAD_DUMP", "GUT_GRAD_REF");
}

TEST_F(GsplatRasterizerTest, GutFromWorldFisheyeGradParity) {
    constexpr int kW = 96;
    constexpr int kH = 64;
    auto camera = make_fisheye_camera(kW, kH);
    run_gut_from_world_parity(camera, "GUT_FISHEYE_GRAD_DUMP", "GUT_FISHEYE_GRAD_REF");
}

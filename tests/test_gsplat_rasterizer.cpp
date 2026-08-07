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
#include "training/rasterization/gsplat/Common.h"
#include "training/rasterization/gsplat/Ops.h"
#include "training/rasterization/gsplat_rasterizer.hpp"

#include <chrono>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <iostream>
#include <stdexcept>
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

// ---------------------------------------------------------------------------
// Worker R — persistent high-water isect buffers (gut/gsplat path).
// Pre-change: every forward cudaMalloc'd isect_ids + flatten_ids and callers
// cudaFree'd them (trainer cleanup + backward). Post-change: grow-only TLS.
// ---------------------------------------------------------------------------
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

// Small gut-mode microbench: median ms/forward after warmup (record in PROGRESS).
TEST_F(GsplatRasterizerTest, GutModeSmokeBench) {
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

    const auto t0 = std::chrono::steady_clock::now();
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
    const auto t1 = std::chrono::steady_clock::now();
    const double ms_total =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double ms_per = ms_total / static_cast<double>(kIters);
    const double allocs_per = static_cast<double>(allocs) / static_cast<double>(kIters);

    std::cout << "GUT_SMOKE_BENCH ms/forward=" << ms_per
              << " allocs/forward=" << allocs_per
              << " iters=" << kIters
              << " N=256 128x128" << std::endl;

    // Steady high-water: average allocs per forward should be ~0.
    EXPECT_LT(allocs_per, 0.5)
        << "gut steady-state should not touch the driver every forward";
}

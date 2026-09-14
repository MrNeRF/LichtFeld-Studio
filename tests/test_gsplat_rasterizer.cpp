/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Gsplat / 3DGUT path: forward smoke + persistent high-water isect buffers.
 */

#include "core/alloc_counter.hpp"
#include "core/camera.hpp"
#include "core/cuda/memory_arena.hpp"
#include "core/cuda/vmm_device_buffer.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "core/tensor/internal/cuda_stream_context.hpp"
#include "core/tensor/internal/memory_pool.hpp"
#include "lfs/training/sh_value_codec.hpp"
#include "lfs/training/sh_value_storage.hpp"
#include "optimizer/adam_optimizer.hpp"
#include "training/kernels/depth_loss.hpp"
#include "training/kernels/normal_consistency_loss.hpp"
#include "training/rasterization/fast_rasterizer.hpp"
#include "training/rasterization/gsplat/Common.h"
#include "training/rasterization/gsplat/GeometryFeatures.h"
#include "training/rasterization/gsplat/Ops.h"
#include "training/rasterization/gsplat_rasterizer.hpp"

#include <algorithm>
#include <array>
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

TEST(VmmDeviceBufferTest, GrowsInPlace) {
    constexpr size_t kMiB = 1024u * 1024u;
    constexpr size_t kFirstCommit = 256u * kMiB;
    constexpr size_t kSecondCommit = 512u * kMiB;
    constexpr size_t kKeptPrefix = 128u * kMiB;
    auto result = VmmDeviceBuffer::create(1024u * kMiB,
                                          "test.gsplat.vmm_intersection");
    ASSERT_TRUE(result) << format_for_developer(result.error());
    auto buffer = std::move(*result);
    const size_t granularity = buffer.granularity_bytes();

    ASSERT_TRUE(buffer.commit(kFirstCommit));
    void* const initial_address = buffer.data();
    EXPECT_EQ(buffer.committed_bytes(), kFirstCommit);

    ASSERT_TRUE(buffer.commit(kSecondCommit));
    EXPECT_EQ(buffer.data(), initial_address);
    EXPECT_EQ(buffer.committed_bytes(), kSecondCommit);

    ASSERT_EQ(cudaMemset(buffer.data(), 0x5a, kKeptPrefix), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    size_t free_before = 0;
    size_t total_bytes = 0;
    ASSERT_EQ(cudaMemGetInfo(&free_before, &total_bytes), cudaSuccess);
    ASSERT_TRUE(buffer.decommit_tail(kKeptPrefix));
    EXPECT_EQ(buffer.data(), initial_address);
    EXPECT_GE(buffer.committed_bytes(), kKeptPrefix);
    EXPECT_EQ(buffer.committed_bytes() % granularity, 0u);

    std::array<unsigned char, 4> kept_prefix{};
    ASSERT_EQ(cudaMemcpy(kept_prefix.data(), buffer.data(), kept_prefix.size(), cudaMemcpyDeviceToHost),
              cudaSuccess);
    EXPECT_EQ(kept_prefix, (std::array<unsigned char, 4>{0x5a, 0x5a, 0x5a, 0x5a}));

    size_t free_after = 0;
    ASSERT_EQ(cudaMemGetInfo(&free_after, &total_bytes), cudaSuccess);
    ASSERT_TRUE(buffer.commit(kSecondCommit));
    EXPECT_EQ(buffer.data(), initial_address);
    EXPECT_EQ(buffer.committed_bytes(), kSecondCommit);
    ASSERT_EQ(cudaMemcpy(kept_prefix.data(), buffer.data(), kept_prefix.size(), cudaMemcpyDeviceToHost),
              cudaSuccess);
    EXPECT_EQ(kept_prefix, (std::array<unsigned char, 4>{0x5a, 0x5a, 0x5a, 0x5a}));

    if (free_after < free_before + 256u * kMiB) {
        GTEST_SKIP() << "CUDA free-memory jitter masked VMM decommit reclaim: before="
                     << free_before << " after=" << free_after;
    }
}

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

TEST(GsplatRasterizerEdgeScores, GutFusedScoresRespectEdgeMapAndCameraModel) {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        GTEST_SKIP() << "CUDA device unavailable";
    }

    auto run = [](Camera camera, const Tensor& edge_map) {
        auto splat = make_visible_splat(32);
        AdamConfig cfg;
        cfg.lr = 1e-3f;
        cfg.initial_capacity = 64;
        AdamOptimizer optimizer(*splat, cfg);
        optimizer.allocate_gradients(64);
        auto background = Tensor::zeros({3}, Device::CUDA);
        auto scores = Tensor::zeros({32}, Device::CUDA);

        auto result = gsplat_rasterize_forward(
            camera, *splat, background, 0, 0, 0, 0, 1.0f, false,
            GsplatRenderMode::RGB, true);
        EXPECT_TRUE(result.has_value()) << result.error();
        if (!result) {
            return scores.cpu();
        }
        auto output = std::move(result->first);
        auto context = std::move(result->second);
        if (context.n_isects == 0) {
            ADD_FAILURE() << "fixture must produce intersections";
            return scores.cpu();
        }
        auto grad_image = Tensor::ones_like(output.image);
        auto grad_alpha = Tensor::zeros_like(output.alpha);
        gsplat_rasterize_backward(
            context, grad_image, grad_alpha, *splat, optimizer, Tensor{}, edge_map, scores);
        return scores.cpu();
    };

    const auto pinhole_ones = run(make_camera(64, 64), Tensor::ones({64, 64}, Device::CUDA));
    bool has_positive = false;
    for (size_t i = 0; i < static_cast<size_t>(pinhole_ones.numel()); ++i) {
        const float value = pinhole_ones.ptr<float>()[i];
        ASSERT_TRUE(std::isfinite(value));
        ASSERT_GE(value, 0.0f);
        has_positive |= value > 0.0f;
    }
    EXPECT_TRUE(has_positive);

    const auto pinhole_zeros = run(make_camera(64, 64), Tensor::zeros({64, 64}, Device::CUDA));
    for (size_t i = 0; i < static_cast<size_t>(pinhole_zeros.numel()); ++i) {
        EXPECT_FLOAT_EQ(pinhole_zeros.ptr<float>()[i], 0.0f);
    }

    const auto fisheye_ones = run(make_fisheye_camera(64, 64), Tensor::ones({64, 64}, Device::CUDA));
    bool fisheye_has_positive = false;
    for (size_t i = 0; i < static_cast<size_t>(fisheye_ones.numel()); ++i) {
        const float value = fisheye_ones.ptr<float>()[i];
        ASSERT_TRUE(std::isfinite(value));
        ASSERT_GE(value, 0.0f);
        fisheye_has_positive |= value > 0.0f;
    }
    EXPECT_TRUE(fisheye_has_positive);
}

TEST(GsplatRasterizerEdgeScores, GutAndFastGsHavePositiveScoreCorrelation) {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        GTEST_SKIP() << "CUDA device unavailable";
    }

    constexpr int width = 64;
    constexpr int height = 64;
    constexpr int count = 32;
    auto camera = make_camera(width, height);
    const auto edge_map = Tensor::ones({height, width}, Device::CUDA);

    auto gut_model = make_visible_splat(count);
    AdamConfig gut_config;
    gut_config.lr = 1e-3f;
    gut_config.initial_capacity = 64;
    AdamOptimizer gut_optimizer(*gut_model, gut_config);
    gut_optimizer.allocate_gradients(64);
    auto background = Tensor::zeros({3}, Device::CUDA);
    auto gut_scores = Tensor::zeros({count}, Device::CUDA);
    auto gut_result = gsplat_rasterize_forward(
        camera, *gut_model, background, 0, 0, 0, 0, 1.0f, false,
        GsplatRenderMode::RGB, true);
    ASSERT_TRUE(gut_result.has_value()) << gut_result.error();
    auto gut_output = std::move(gut_result->first);
    auto gut_context = std::move(gut_result->second);
    gsplat_rasterize_backward(
        gut_context, Tensor::ones_like(gut_output.image), Tensor::zeros_like(gut_output.alpha),
        *gut_model, gut_optimizer, Tensor{}, edge_map, gut_scores);

    auto fast_model = make_visible_splat(count);
    AdamOptimizer fast_optimizer(*fast_model, gut_config);
    fast_optimizer.allocate_gradients(64);
    auto fast_result = fast_rasterize_forward(
        camera, *fast_model, background, 0, 0, 0, 0, false);
    ASSERT_TRUE(fast_result.has_value()) << lfs::format_for_developer(fast_result.error());
    auto fast_output = std::move(fast_result->first);
    auto fast_context = std::move(fast_result->second);
    auto fast_scores = Tensor::zeros({count}, Device::CUDA);
    FastGSFusedExtraGradients fused;
    fused.edge_weight_map = edge_map.ptr<float>();
    fused.edge_score_out = fast_scores.ptr<float>();
    fast_rasterize_backward(
        fast_context, Tensor::ones_like(fast_output.image), *fast_model, fast_optimizer,
        Tensor{}, Tensor{}, DensificationType::None, 0, fused);

    const auto gut_cpu = gut_scores.cpu();
    const auto fast_cpu = fast_scores.cpu();
    float gut_mean = 0.0f;
    float fast_mean = 0.0f;
    for (int i = 0; i < count; ++i) {
        gut_mean += gut_cpu.ptr<float>()[i];
        fast_mean += fast_cpu.ptr<float>()[i];
    }
    gut_mean /= static_cast<float>(count);
    fast_mean /= static_cast<float>(count);
    float covariance = 0.0f;
    float gut_variance = 0.0f;
    float fast_variance = 0.0f;
    for (int i = 0; i < count; ++i) {
        const float gut_delta = gut_cpu.ptr<float>()[i] - gut_mean;
        const float fast_delta = fast_cpu.ptr<float>()[i] - fast_mean;
        covariance += gut_delta * fast_delta;
        gut_variance += gut_delta * gut_delta;
        fast_variance += fast_delta * fast_delta;
    }
    ASSERT_GT(gut_variance, 0.0f);
    ASSERT_GT(fast_variance, 0.0f);
    const float correlation = covariance / std::sqrt(gut_variance * fast_variance);
    EXPECT_GT(correlation, 0.75f);
}

TEST(GsplatRasterizerErrors, GutArenaExhaustionPreservesTypedResourceError) {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        GTEST_SKIP() << "CUDA device unavailable";
    }

    lfs::core::RasterizerMemoryArena::Config config;
    config.virtual_size = 64ULL << 20;
    config.max_physical = 4ULL << 10;
    config.granularity = 4ULL << 10;
    config.enable_vmm = false;
    auto& manager = lfs::core::GlobalArenaManager::instance();
    manager.reconfigure_for_testing(config);

    auto camera = make_camera(64, 64);
    auto splat = make_visible_splat(32);
    auto background = Tensor::zeros({3}, Device::CUDA);
    bool caught_typed_resource_error = false;
    try {
        (void)gsplat_rasterize_forward(
            camera, *splat, background, 0, 0, 0, 0, 1.0f, false,
            GsplatRenderMode::RGB, true);
    } catch (const lfs::Exception& exception) {
        caught_typed_resource_error =
            exception.error().code() == lfs::ErrorCode::ResourceExhausted &&
            exception.error().detail().find("gsplat forward arena allocation failed") !=
                std::string::npos;
    }
    manager.reset();
    EXPECT_TRUE(caught_typed_resource_error);
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

// All supervision channels must describe the same camera-space geometry as the
// rays used by GUT, including cameras whose rear hemisphere has negative Z.
TEST_F(GsplatRasterizerTest, GeometryChannelsAndGradientsMatchEveryCameraModel) {
    struct ForwardStream {
        cudaStream_t stream = nullptr;
        ForwardStream() {
            if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess)
                throw std::runtime_error("failed to create test stream");
        }
        ~ForwardStream() {
            release_gsplat_rasterizer_thread_local_caches();
            gsplat_lfs::release_intersect_thread_local_cache();
            CudaMemoryPool::instance().release_stream(stream);
            cudaStreamDestroy(stream);
        }
    } forward_stream;
    for (const auto model : {lfs::core::CameraModelType::PINHOLE,
                             lfs::core::CameraModelType::FISHEYE,
                             lfs::core::CameraModelType::EQUIRECTANGULAR,
                             lfs::core::CameraModelType::THIN_PRISM_FISHEYE}) {
        SCOPED_TRACE(static_cast<int>(model));
        const size_t w = 64, h = 48;
        auto R = Tensor::from_vector({1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f}, {3, 3}, Device::CUDA);
        auto T = Tensor::zeros({3}, Device::CUDA);
        Camera camera(R, T, 28.f, 28.f, w / 2.f, h / 2.f,
                      Tensor::zeros({4}, Device::CPU), Tensor::zeros({4}, Device::CPU),
                      model, "geometry", "", {}, w, h, 0);
        auto means = Tensor::from_vector({.4f, .2f, 3.f}, {1, 3}, Device::CUDA);
        auto rotation = Tensor::from_vector({1.f, .1f, .2f, .05f}, {1, 4}, Device::CUDA);
        auto scales = Tensor::from_vector({-.3f, -.5f, -1.5f}, {1, 3}, Device::CUDA);
        SplatData splat(0, means, Tensor::full({1, 1, 3}, .5f, Device::CUDA),
                        Tensor::zeros({1, 0, 3}, Device::CUDA), scales, rotation,
                        Tensor::full({1}, 2.f, Device::CUDA), 1.f);
        auto bg = Tensor::full({3}, .25f, Device::CUDA);
        auto rgb = gsplat_rasterize(camera, splat, bg, 1.f, false, GsplatRenderMode::RGB, true).image.cpu();
        auto plain_depth = gsplat_rasterize(camera, splat, bg, 1.f, false, GsplatRenderMode::RGB_D, true);
        auto depth_cpu = plain_depth.depth.cpu();
        auto alpha_cpu = plain_depth.alpha.cpu();
        auto image_cpu = plain_depth.image.cpu();
        const float expected = model == lfs::core::CameraModelType::PINHOLE ? 3.f : std::sqrt(9.2f);
        int covered = 0;
        for (size_t i = 0; i < w * h; ++i) {
            if (alpha_cpu.ptr<float>()[i] > .1f) {
                ++covered;
                EXPECT_NEAR(depth_cpu.ptr<float>()[i] / alpha_cpu.ptr<float>()[i], expected, 2e-5f);
            }
            for (int c = 0; c < 3; ++c)
                EXPECT_NEAR(image_cpu.ptr<float>()[c * w * h + i], rgb.ptr<float>()[c * w * h + i], 2e-5f);
        }
        EXPECT_GT(covered, 10);
        AdamConfig cfg;
        cfg.initial_capacity = 1;
        AdamOptimizer opt(splat, cfg);
        opt.allocate_gradients(1);
        auto r = [&] {
            CUDAStreamGuard guard(forward_stream.stream);
            return gsplat_rasterize_forward(camera, splat, bg, 0, 0, 0, 0, 1.f, false, GsplatRenderMode::RGB_D_N, true);
        }();
        ASSERT_TRUE(r.has_value());
        std::vector<float> depth_weights(w * h);
        for (size_t i = 0; i < w * h; ++i)
            depth_weights[i] = alpha_cpu.ptr<float>()[i] > .2f ? .3f / (w * h) : 0.f;
        auto gd = Tensor::from_vector(depth_weights, {h, w}, Device::CUDA);
        std::vector<float> normal_weights(3 * w * h);
        for (size_t i = 0; i < w * h; ++i) {
            normal_weights[i] = depth_weights[i] * (2.f / 3.f);
            normal_weights[w * h + i] = depth_weights[i] * (-4.f / 3.f);
            normal_weights[2 * w * h + i] = depth_weights[i] / 3.f;
        }
        auto gn = Tensor::from_vector(normal_weights, {3, h, w}, Device::CUDA);
        gsplat_rasterize_backward(r->second, Tensor::zeros_like(r->first.image),
                                  Tensor::zeros_like(r->first.alpha), splat, opt, {}, {}, {}, gd, gn);
        auto gm = opt.get_grad(ParamType::Means).cpu();
        auto gq = opt.get_grad(ParamType::Rotation).cpu();
        auto objective = [&]() {
            auto out = gsplat_rasterize(camera, splat, bg, 1.f, false, GsplatRenderMode::RGB_D_N, true);
            auto d = out.depth.cpu(), n = out.normal.cpu();
            double sum = 0;
            for (size_t i = 0; i < w * h; ++i)
                sum += depth_weights[i] * d.ptr<float>()[i] + normal_weights[i] * n.ptr<float>()[i] + normal_weights[w * h + i] * n.ptr<float>()[w * h + i] + normal_weights[2 * w * h + i] * n.ptr<float>()[2 * w * h + i];
            return sum;
        };
        auto check = [&](Tensor& param, const Tensor& grad) {
            auto host = param.cpu();
            for (size_t j = 0; j < param.numel(); ++j) {
                const float original = host.ptr<float>()[j], eps = 1e-3f;
                float v = original + eps;
                ASSERT_EQ(cudaMemcpy(param.ptr<float>() + j, &v, sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
                const double plus = objective();
                v = original - eps;
                ASSERT_EQ(cudaMemcpy(param.ptr<float>() + j, &v, sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
                const double minus = objective();
                ASSERT_EQ(cudaMemcpy(param.ptr<float>() + j, &original, sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
                const double numeric = (plus - minus) / (2 * eps);
                EXPECT_NEAR(grad.ptr<float>()[j], numeric, 2e-4 + .03 * std::abs(numeric)) << "component " << j;
            }
        };
        check(means, gm);
        check(rotation, gq);
    }
}

TEST_F(GsplatRasterizerTest, GeometryAnchorsAndDepthNormalsFollowCameraRays) {
    constexpr size_t w = 64, h = 48, pixels = w * h;
    auto view = Tensor::from_vector({1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f}, {16}, Device::CUDA);
    for (const auto model : {::CameraModelType::PINHOLE, ::CameraModelType::FISHEYE,
                             ::CameraModelType::EQUIRECTANGULAR, ::CameraModelType::THIN_PRISM_FISHEYE}) {
        SCOPED_TRACE(static_cast<int>(model));
        const bool equi = model == ::CameraModelType::EQUIRECTANGULAR;
        auto K = Tensor::from_vector({equi ? float(w) : 28.f, 0.f, equi ? 0.f : w / 2.f,
                                      0.f, equi ? float(h) : 28.f, equi ? 0.f : h / 2.f, 0.f, 0.f, 1.f},
                                     {9}, Device::CUDA);
        auto radial = Tensor::from_vector({.02f, -.003f, 0.f, 0.f, 0.f, 0.f}, {6}, Device::CUDA);
        auto rays = Tensor::empty({h, w, 3}, Device::CUDA);
        gsplat_lfs::geometry_camera_rays(rays.ptr<float>(), w, h, K.ptr<float>(), model, radial.ptr<float>(), nullptr, nullptr, nullptr);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        const auto host = rays.cpu();
        std::vector<float> points(3 * pixels), prior(pixels), normals(3 * pixels), depths(pixels, 2.f);
        size_t valid = 0;
        for (size_t i = 0; i < pixels; ++i) {
            const float d = 2.f + .001f * i;
            prior[i] = 2.f * d + 1.f;
            if (std::isfinite(host.ptr<float>()[3 * i]))
                ++valid;
            for (int c = 0; c < 3; ++c) {
                points[3 * i + c] = d * host.ptr<float>()[3 * i + c];
                normals[c * pixels + i] = model == ::CameraModelType::PINHOLE ? (c == 2 ? -1.f : 0.f) : -host.ptr<float>()[3 * i + c];
            }
        }
        auto pts = Tensor::from_vector(points, {pixels, 3}, Device::CUDA);
        auto pr = Tensor::from_vector(prior, {h, w}, Device::CUDA);
        kernels::DepthCameraProjection projection;
        projection.model = static_cast<int>(model);
        projection.radial[0] = .02f;
        projection.radial[1] = -.003f;
        const float lo[3] = {-100, -100, -100}, hi[3] = {100, 100, 100};
        auto pairs = kernels::collect_depth_anchor_samples(pts.ptr<float>(), pixels, view.ptr<float>(), 28, 28, w / 2.f, h / 2.f,
                                                           pr.ptr<float>(), w, h, .01f, lo, hi, nullptr, projection);
        EXPECT_EQ(pairs.size(), valid);
        EXPECT_GT(pairs.size(), pixels / 2);
        for (auto pair : pairs)
            EXPECT_NEAR(pair.y, (pair.x - 1.f) / 2.f, 2e-5f);
        const auto fit = kernels::fit_depth_anchor_from_samples(pairs);
        ASSERT_TRUE(fit.depth.valid);
        EXPECT_NEAR(fit.depth.scale, .5f, 2e-4f);
        EXPECT_NEAR(fit.depth.shift, -.5f, 2e-4f);
        auto normal = Tensor::from_vector(normals, {3, h, w}, Device::CUDA);
        auto depth = Tensor::from_vector(depths, {h, w}, Device::CUDA);
        auto alpha = Tensor::ones({h, w}, Device::CUDA);
        auto gn = Tensor::zeros_like(normal), gd = Tensor::zeros_like(depth), ga = Tensor::zeros_like(depth);
        auto loss = Tensor::zeros({1}, Device::CUDA);
        auto partials = Tensor::empty({kernels::normal_consistency_partial_count(pixels)}, Device::CUDA);
        kernels::launch_normal_consistency_loss(normal.ptr<float>(), depth.ptr<float>(), alpha.ptr<float>(),
                                                gn.ptr<float>(), gd.ptr<float>(), ga.ptr<float>(), loss.ptr<float>(), partials.ptr<float>(),
                                                w, h, 28, 28, w / 2.f, h / 2.f, 1.f, nullptr, nullptr, rays.ptr<float>(), equi);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        EXPECT_NEAR(loss.cpu().ptr<float>()[0], 0.f, 2e-4f);
        const auto stats = partials.cpu();
        EXPECT_EQ(stats.ptr<float>()[kernels::normal_consistency_slots::kValid], 1.f);
        if (equi)
            EXPECT_EQ(stats.ptr<float>()[kernels::normal_consistency_slots::kCount], float(w * (h - 2)));
    }
}

TEST_F(GsplatRasterizerTest, ExpectedRadialDepthBehindCameraHasCorrectGradients) {
    for (const auto model : {lfs::core::CameraModelType::EQUIRECTANGULAR, lfs::core::CameraModelType::FISHEYE, lfs::core::CameraModelType::THIN_PRISM_FISHEYE}) {
        SCOPED_TRACE(static_cast<int>(model));
        auto R = Tensor::from_vector({1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f}, {3, 3}, Device::CUDA);
        Camera camera(R, Tensor::zeros({3}, Device::CUDA), 16, 16, 32, 24, {}, {},
                      model, "rear", "", {}, 64, 48, 0);
        auto splat = make_visible_splat(1);
        const float pos[3] = {3.f, .2f, -.4f};
        ASSERT_EQ(cudaMemcpy(splat->means().ptr<float>(), pos, sizeof(pos), cudaMemcpyHostToDevice), cudaSuccess);
        const float scales[3] = {-.3f, -.5f, -1.5f};
        ASSERT_EQ(cudaMemcpy(splat->scaling_raw().ptr<float>(), scales, sizeof(scales), cudaMemcpyHostToDevice), cudaSuccess);
        auto bg = Tensor::full({3}, .7f, Device::CUDA);
        for (const auto mode : {GsplatRenderMode::ED, GsplatRenderMode::RGB_ED}) {
            AdamConfig cfg;
            cfg.initial_capacity = 1;
            AdamOptimizer opt(*splat, cfg);
            opt.allocate_gradients(1);
            auto r = gsplat_rasterize_forward(camera, *splat, bg, 0, 0, 0, 0, 1.f, false, mode, true);
            ASSERT_TRUE(r.has_value());
            auto alpha = r->first.alpha.cpu(), depth = r->first.depth.cpu();
            std::vector<float> weights(64 * 48, 0.f);
            int count = 0;
            for (size_t i = 0; i < weights.size(); ++i)
                if (alpha.ptr<float>()[i] > .2f) {
                    weights[i] = 1.f;
                    ++count;
                    EXPECT_NEAR(depth.ptr<float>()[i], std::sqrt(9.2f), 2e-5f);
                }
            if (count <= 10) {
                release_ctx_arena(r->second);
                FAIL() << "Insufficient rear-hemisphere coverage: " << count;
            }
            for (auto& weight : weights)
                weight /= count;
            auto gd = Tensor::from_vector(weights, {48, 64}, Device::CUDA);
            gsplat_rasterize_backward(r->second, {}, {}, *splat, opt, {}, {}, {}, gd, {}, .6f);
            auto gm = opt.get_grad(ParamType::Means).cpu();
            for (int c = 0; c < 3; ++c)
                EXPECT_NEAR(gm.ptr<float>()[c], pos[c] / std::sqrt(9.2f), 2e-4f);
            auto go = opt.get_grad(ParamType::Opacity).cpu();
            EXPECT_NEAR(go.ptr<float>()[0], 0.f, 1e-5f);
            auto gs = opt.get_grad(ParamType::Scaling).cpu();
            EXPECT_NEAR(gs.ptr<float>()[0], 0.f, 1e-5f);
            EXPECT_NEAR(gs.ptr<float>()[1], 0.f, 1e-5f);
            EXPECT_NEAR(gs.ptr<float>()[2], .6f * std::exp(-1.5f), 1e-5f);
        }
    }
}

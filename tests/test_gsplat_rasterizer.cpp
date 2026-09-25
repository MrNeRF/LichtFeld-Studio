/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Gsplat / 3DGUT path: forward smoke + persistent high-water isect buffers.
 */

#include "core/alloc_counter.hpp"
#include "core/camera.hpp"
#include "core/cuda/memory_arena.hpp"
#include "core/cuda/vmm_device_buffer.hpp"
#include "core/environment.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "core/tensor_cuda_interop.hpp"
#include "core/tensor_upload.hpp"
#include "cuda_backend_test.hpp"
#include "lfs/training/sh_value_codec.hpp"
#include "lfs/training/sh_value_storage.hpp"
#include "optimizer/adam_optimizer.hpp"
#include "training/components/ppisp.hpp"
#include "training/kernels/densification_kernels.hpp"
#include "training/rasterization/fast_rasterizer.hpp"
#include "training/rasterization/gsplat/Common.h"
#include "training/rasterization/gsplat/IntersectionCount.h"
#include "training/rasterization/gsplat/Ops.h"
#include "training/rasterization/gsplat_rasterizer.hpp"
#include "training/strategies/mrnf.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace tensor_hardening {
    cudaError_t launch_delay_kernel(cudaStream_t stream, uint64_t cycles);
}

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
        auto R = Tensor::from_blob(R_data.data(), {3, 3}, Device::CPU, DataType::Float32).to(Device::GPU);
        auto T = Tensor::from_blob(T_data.data(), {3}, Device::CPU, DataType::Float32).to(Device::GPU);
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
        auto R = Tensor::from_blob(R_data.data(), {3, 3}, Device::CPU, DataType::Float32).to(Device::GPU);
        auto T = Tensor::from_blob(T_data.data(), {3}, Device::CPU, DataType::Float32).to(Device::GPU);
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
        auto means = Tensor::zeros({static_cast<size_t>(n), 3}, Device::GPU);
        if (n > 0) {
            auto cpu = means.to(Device::CPU);
            float* p = cpu.ptr<float>();
            for (int i = 0; i < n; ++i) {
                p[i * 3 + 0] = (i % 5) * 0.3f - 0.6f;
                p[i * 3 + 1] = (i / 5) * 0.3f - 0.6f;
                p[i * 3 + 2] = 0.0f; // in front of camera at z=3
            }
            means = cpu.to(Device::GPU);
        }
        auto sh0 = Tensor::full({static_cast<size_t>(n), 1, 3}, 0.5f, Device::GPU);
        auto shN = Tensor::zeros({static_cast<size_t>(n), 0, 3}, Device::GPU);
        auto scaling = Tensor::full({static_cast<size_t>(n), 3}, -2.0f, Device::GPU);
        std::vector<float> rot(static_cast<size_t>(n) * 4, 0.f);
        for (int i = 0; i < n; ++i) {
            rot[static_cast<size_t>(i) * 4] = 1.f;
        }
        auto rotation = Tensor::from_blob(rot.data(), {static_cast<size_t>(n), 4}, Device::CPU, DataType::Float32)
                            .to(Device::GPU);
        auto opacity = Tensor::full({static_cast<size_t>(n)}, 2.0f, Device::GPU);
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
                           .to(Device::GPU);
        auto sh0_t = Tensor::from_blob(sh0.data(), {static_cast<size_t>(n), 1, 3}, Device::CPU, DataType::Float32)
                         .to(Device::GPU);
        auto shN_t = Tensor::zeros({static_cast<size_t>(n), 0, 3}, Device::GPU);
        auto scaling_t = Tensor::from_blob(scaling.data(), {static_cast<size_t>(n), 3}, Device::CPU, DataType::Float32)
                             .to(Device::GPU);
        auto rotation_t = Tensor::from_blob(rotation.data(), {static_cast<size_t>(n), 4}, Device::CPU, DataType::Float32)
                              .to(Device::GPU);
        auto opacity_t = Tensor::from_blob(opacity.data(), {static_cast<size_t>(n)}, Device::CPU, DataType::Float32)
                             .to(Device::GPU);
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

class GsplatRasterizerTest : public lfs::test::CudaBackendTest {
protected:
    void SetUp() override {
        LFS_CUDA_BACKEND_OR_RETURN();
        // Create minimal test data
        const size_t N = 100; // Number of Gaussians
        const int sh_degree = 0;

        // Create random Gaussian parameters
        means_ = Tensor::randn({N, 3}, Device::GPU, DataType::Float32);
        sh0_ = Tensor::randn({N, 1, 3}, Device::GPU, DataType::Float32);            // sh0 is [N, 1, 3]
        shN_ = Tensor::zeros({N, 0, 3}, Device::GPU, DataType::Float32);            // No higher SH for degree 0
        scaling_ = Tensor::randn({N, 3}, Device::GPU, DataType::Float32).mul(0.1f); // Small scales
        rotation_ = Tensor::randn({N, 4}, Device::GPU, DataType::Float32);
        opacity_ = Tensor::randn({N}, Device::GPU, DataType::Float32);

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
        auto R = Tensor::eye(3, Device::GPU);
        auto T = Tensor::zeros({3}, Device::GPU, DataType::Float32);

        // Set camera at z=3 looking at origin
        std::vector<float> T_data = {0.0f, 0.0f, 3.0f};
        T = Tensor::from_blob(T_data.data(), {3}, Device::CPU, DataType::Float32).to(Device::GPU);

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
        bg_color_ = Tensor::zeros({3}, Device::GPU, DataType::Float32);
        bg_color_.fill_(0.5f); // Gray background
    }

    void TearDown() override {
        if (IsSkipped())
            return;
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

class VmmDeviceBufferTest : public lfs::test::CudaBackendTest {};

TEST_F(VmmDeviceBufferTest, GrowsInPlace) {
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

class GsplatRasterizerPPISP : public lfs::test::CudaBackendTest {};

TEST_F(GsplatRasterizerPPISP, NegativeShRadianceDoesNotCreateBrightPixels) {
    constexpr int width = 32;
    constexpr int height = 32;
    auto camera = make_camera(width, height);
    auto model = make_visible_splat(1);
    model->means_raw().fill_(0.0f);
    auto background = Tensor::zeros({3}, Device::CUDA);
    PPISP ppisp(100);
    ppisp.register_frame(0, 0);
    ppisp.finalize();

    for (const auto& color : {std::array{-0.20f, 0.05f, 0.05f},
                              std::array{-0.10f, -0.10f, 0.05f}}) {
        const std::vector<float> sh{
            (color[0] - 0.5f) / 0.28209479177387814f,
            (color[1] - 0.5f) / 0.28209479177387814f,
            (color[2] - 0.5f) / 0.28209479177387814f};
        model->sh0_raw() = Tensor::from_vector(sh, {1, 1, 3}, Device::CUDA);
        auto result = gsplat_rasterize_forward(
            camera, *model, background,
            0, 0, 0, 0, 1.0f, false, GsplatRenderMode::RGB, true);
        ASSERT_TRUE(result.has_value()) << result.error();
        auto raw = result->first.image.clone();
        release_ctx_arena(result->second);
        ASSERT_LT(raw.min().item<float>(), -0.01f) << "Fixture must reach PPISP with negative radiance";
        const auto corrected = ppisp.apply(raw, 0, 0).cpu();
        const auto* pixels = corrected.ptr<float>();
        for (int c = 0; c < 3; ++c) {
            for (int p = 0; p < width * height; ++p) {
                ASSERT_TRUE(std::isfinite(pixels[c * width * height + p]));
                ASSERT_LE(pixels[c * width * height + p], std::max(color[c], 0.0f) + 1e-4f)
                    << "channel=" << c << ", pixel=" << p;
            }
        }
    }
}

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
    auto bg = Tensor::zeros({3}, Device::GPU);

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
    auto bg = Tensor::zeros({3}, Device::GPU);

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
class GsplatRasterizerQuantTest : public lfs::test::CudaBackendTest {};

TEST_F(GsplatRasterizerQuantTest, GutForwardBackwardWithDefaultQuantAndShDegree) {
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
    auto shN = Tensor::full({n, rest, size_t{3}}, 0.05f, Device::GPU);
    auto splat = SplatData(
        sh_degree,
        Tensor::from_vector(means, {n, size_t{3}}, Device::GPU),
        Tensor::full({n, size_t{1}, size_t{3}}, 0.5f, Device::GPU),
        std::move(shN),
        Tensor::full({n, size_t{3}}, -2.0f, Device::GPU),
        Tensor::from_vector(rots, {n, size_t{4}}, Device::GPU),
        Tensor::full({n, size_t{1}}, 2.0f, Device::GPU),
        1.0f);
    ASSERT_TRUE(lfs::training::sh_value::apply_shN_value_quant(splat));
    ASSERT_TRUE(splat.shN_value_quantized());

    AdamConfig cfg;
    cfg.lr = 1e-3f;
    cfg.initial_capacity = n * 2;
    AdamOptimizer opt(splat, cfg);
    opt.allocate_gradients(n * 2);

    auto bg = Tensor::zeros({3}, Device::GPU);
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

TEST_F(GsplatRasterizerQuantTest, RejectsFloat16ShRestWithoutQ16Bounds) {
    auto camera = make_camera(32, 32);
    constexpr size_t n = 4;
    constexpr size_t rest = 3;

    std::vector<float> rotations(n * 4, 0.0f);
    for (size_t i = 0; i < n; ++i) {
        rotations[i * 4] = 1.0f;
    }
    auto splat = SplatData(
        1,
        Tensor::zeros({n, size_t{3}}, Device::GPU),
        Tensor::full({n, size_t{1}, size_t{3}}, 0.5f, Device::GPU),
        Tensor::full({n, rest, size_t{3}}, 0.05f, Device::GPU),
        Tensor::full({n, size_t{3}}, -2.0f, Device::GPU),
        Tensor::from_vector(rotations, {n, size_t{4}}, Device::GPU),
        Tensor::full({n, size_t{1}}, 2.0f, Device::GPU),
        1.0f);
    // Canonical construction currently accepts Float32 only. Convert the valid
    // resident float4 swizzle afterward to model the viewer's IEEE-f16 storage
    // without q16 bounds.
    splat.shN() = splat.shN().to(DataType::Float16);
    ASSERT_EQ(splat.shN().dtype(), DataType::Float16);
    ASSERT_FALSE(splat.shN_value_quantized());

    auto background = Tensor::zeros({3}, Device::GPU);
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

class GsplatRasterizerEdgeScores : public lfs::test::CudaBackendTest {};

TEST_F(GsplatRasterizerEdgeScores, GutFusedScoresRespectEdgeMapAndCameraModel) {

    auto run = [](Camera camera, const Tensor& edge_map) {
        auto splat = make_visible_splat(32);
        AdamConfig cfg;
        cfg.lr = 1e-3f;
        cfg.initial_capacity = 64;
        AdamOptimizer optimizer(*splat, cfg);
        optimizer.allocate_gradients(64);
        auto background = Tensor::zeros({3}, Device::GPU);
        auto scores = Tensor::zeros({32}, Device::GPU);

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

    const auto pinhole_ones = run(make_camera(64, 64), Tensor::ones({64, 64}, Device::GPU));
    bool has_positive = false;
    for (size_t i = 0; i < static_cast<size_t>(pinhole_ones.numel()); ++i) {
        const float value = pinhole_ones.ptr<float>()[i];
        ASSERT_TRUE(std::isfinite(value));
        ASSERT_GE(value, 0.0f);
        has_positive |= value > 0.0f;
    }
    EXPECT_TRUE(has_positive);

    const auto pinhole_zeros = run(make_camera(64, 64), Tensor::zeros({64, 64}, Device::GPU));
    for (size_t i = 0; i < static_cast<size_t>(pinhole_zeros.numel()); ++i) {
        EXPECT_FLOAT_EQ(pinhole_zeros.ptr<float>()[i], 0.0f);
    }

    const auto fisheye_ones = run(make_fisheye_camera(64, 64), Tensor::ones({64, 64}, Device::GPU));
    bool fisheye_has_positive = false;
    for (size_t i = 0; i < static_cast<size_t>(fisheye_ones.numel()); ++i) {
        const float value = fisheye_ones.ptr<float>()[i];
        ASSERT_TRUE(std::isfinite(value));
        ASSERT_GE(value, 0.0f);
        fisheye_has_positive |= value > 0.0f;
    }
    EXPECT_TRUE(fisheye_has_positive);
}

TEST_F(GsplatRasterizerEdgeScores, GutAndFastGsHavePositiveScoreCorrelation) {

    constexpr int width = 64;
    constexpr int height = 64;
    constexpr int count = 32;
    auto camera = make_camera(width, height);
    const auto edge_map = Tensor::ones({height, width}, Device::GPU);

    auto gut_model = make_visible_splat(count);
    AdamConfig gut_config;
    gut_config.lr = 1e-3f;
    gut_config.initial_capacity = 64;
    AdamOptimizer gut_optimizer(*gut_model, gut_config);
    gut_optimizer.allocate_gradients(64);
    auto background = Tensor::zeros({3}, Device::GPU);
    auto gut_scores = Tensor::zeros({count}, Device::GPU);
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
    auto fast_scores = Tensor::zeros({count}, Device::GPU);
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

// These count checks do not allocate tensors or call CUDA.
TEST(GsplatIntersectionCount, AcceptsNonnegativeAggregateCounts) {
    for (const int64_t count : {int64_t{0}, int64_t{1}, gsplat_lfs::kMaxIntersectionCount,
                                gsplat_lfs::kMaxIntersectionCount + 1, int64_t{2241098778},
                                std::numeric_limits<int64_t>::max()}) {
        EXPECT_TRUE(gsplat_lfs::validate_intersection_count(count)) << count;
    }
}

TEST(GsplatIntersectionCount, RejectsNegativeCount) {
    for (const int64_t count : {int64_t{-1}, std::numeric_limits<int64_t>::min()}) {
        const auto status = gsplat_lfs::validate_intersection_count(count);
        ASSERT_FALSE(status);
        EXPECT_EQ(status.error().code(), lfs::ErrorCode::Internal);
    }
}

TEST(GsplatIntersectionCount, RoundedCapacityCannotOverflowSignedSortCount) {
    const size_t limit = static_cast<size_t>(gsplat_lfs::kMaxIntersectionCount);
    EXPECT_EQ(gsplat_lfs::intersection_sort_capacity(0, limit), 0u);
    EXPECT_EQ(gsplat_lfs::intersection_sort_capacity(1024, limit), 1024u);
    EXPECT_EQ(gsplat_lfs::intersection_sort_capacity(limit, limit), limit);
    EXPECT_EQ(gsplat_lfs::intersection_sort_capacity(limit + 1, limit), limit);
    EXPECT_EQ(gsplat_lfs::intersection_sort_capacity(limit + 65536, limit), limit);
    EXPECT_EQ(gsplat_lfs::intersection_sort_capacity(limit + 65536, 1024), 1024u);
    EXPECT_EQ(gsplat_lfs::intersection_sort_capacity(512, 1024), 512u);
    EXPECT_EQ(gsplat_lfs::intersection_sort_capacity(limit + 65536, limit + 65536), limit);
}

class GsplatRasterizerTestPositive : public lfs::test::CudaBackendTest {};

TEST_F(GsplatRasterizerTestPositive, AggregateIntersectionsRenderOnColdAndWarmCache) {
    struct CacheCleanup {
        ~CacheCleanup() { (void)gsplat_lfs::release_intersect_thread_local_cache(); }
    } cleanup;

    // Preserve #2185's 32769-splat / 256x256-tile fixture and exact pair count.
    // Project coincident splats across the full grid, then exercise the production
    // partitioner and renderer. 64 copies already saturate every pixel, providing
    // an independent unbatched reference for the large frame.
    constexpr uint32_t n = 32769;
    constexpr uint32_t tiles = 256;
    constexpr int64_t count = int64_t{n} * tiles * tiles;
    static_assert(count == 2147549184LL);
    auto make_model = [](size_t size) {
        auto rotations = Tensor::zeros({size, 4}, Device::CPU);
        for (size_t i = 0; i < size; ++i)
            rotations.ptr<float>()[4 * i] = 1.f;
        return SplatData(0, Tensor::zeros({size, 3}, Device::CUDA),
                         Tensor::full({size, 1, 3}, 0.5f, Device::CUDA),
                         Tensor::zeros({size, 0, 3}, Device::CUDA),
                         Tensor::zeros({size, 3}, Device::CUDA),
                         rotations.to(Device::CUDA), Tensor::full({size}, 2.f, Device::CUDA), 1.f);
    };
    auto model = make_model(n);
    auto reference_model = make_model(64);
    auto R = Tensor::eye(3, Device::CUDA);
    auto T = Tensor::from_vector({0.f, 0.f, 3.f}, {3}, Device::CUDA);
    Camera camera(R, T, 5000.f, 5000.f, 2048.f, 2048.f, Tensor(), Tensor(),
                  lfs::core::CameraModelType::PINHOLE, "aggregate_2185", "",
                  std::filesystem::path{}, 4096, 4096, 0);
    auto background = Tensor::full({3}, 0.25f, Device::CUDA);
    auto render = [&](SplatData& splats) {
        return gsplat_rasterize_forward(camera, splats, background,
                                        0, 0, 0, 0, 1.f, false, GsplatRenderMode::RGB, true);
    };
    auto reference = render(reference_model);
    ASSERT_TRUE(reference.has_value()) << reference.error();
    ASSERT_TRUE(reference->second.batches.empty());
    auto reference_image = reference->first.image.clone();
    auto reference_alpha = reference->first.alpha.clone();
    auto reference_image_cpu = reference_image.cpu();
    auto reference_alpha_cpu = reference_alpha.cpu();
    bool saturated_reference = true;
    for (size_t i = 0; i < reference_alpha_cpu.numel(); ++i) {
        const float alpha = reference_alpha_cpu.ptr<float>()[i];
        saturated_reference = saturated_reference && std::isfinite(alpha) && alpha > 0.999f && alpha < 1.f;
    }
    ASSERT_TRUE(saturated_reference);
    release_ctx_arena(reference->second);
    reference = {};

    for (const bool warm : {false, true}) {
        SCOPED_TRACE(warm ? "warm cache" : "cold cache");
        ASSERT_TRUE(gsplat_lfs::release_intersect_thread_local_cache());
        if (warm) {
            auto seed = render(reference_model);
            ASSERT_TRUE(seed.has_value()) << seed.error();
            ASSERT_TRUE(seed->second.batches.empty());
            release_ctx_arena(seed->second);
        }
        auto result = render(model);
        ASSERT_TRUE(result.has_value()) << result.error();
        EXPECT_EQ(result->second.n_isects, count);
        ASSERT_GT(result->second.batches.size(), 1u);
        int64_t sum = 0;
        uint32_t next_tile = 0;
        for (const auto& batch : result->second.batches) {
            EXPECT_EQ(batch.tiles.begin, next_tile);
            EXPECT_GT(batch.tiles.end, batch.tiles.begin);
            EXPECT_EQ(batch.count, int64_t{n} * (batch.tiles.end - batch.tiles.begin));
            EXPECT_LE(batch.count, gsplat_lfs::kMaxIntersectionCount);
            sum += batch.count;
            next_tile = batch.tiles.end;
        }
        EXPECT_EQ(sum, count);
        EXPECT_EQ(next_tile, tiles * tiles);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        auto image_cpu = result->first.image.cpu();
        auto alpha_cpu = result->first.alpha.cpu();
        EXPECT_EQ(std::memcmp(reference_image_cpu.ptr<float>(), image_cpu.ptr<float>(),
                              image_cpu.numel() * sizeof(float)),
                  0);
        EXPECT_EQ(std::memcmp(reference_alpha_cpu.ptr<float>(), alpha_cpu.ptr<float>(),
                              alpha_cpu.numel() * sizeof(float)),
                  0);
        std::cout << "#2185 " << (warm ? "warm" : "cold") << " pairs=" << sum
                  << " batches=" << result->second.batches.size()
                  << " RGB/alpha match unbatched reference" << std::endl;
        release_ctx_arena(result->second);
    }
}

class GsplatRasterizerErrors : public lfs::test::CudaBackendTest {};

TEST_F(GsplatRasterizerErrors, GutArenaExhaustionPreservesTypedResourceError) {

    lfs::core::RasterizerMemoryArena::Config config;
    config.virtual_size = 64ULL << 20;
    config.max_physical = 4ULL << 10;
    config.granularity = 4ULL << 10;
    config.enable_vmm = false;
    auto& manager = lfs::core::GlobalArenaManager::instance();
    manager.reconfigure_for_testing(config);

    auto camera = make_camera(64, 64);
    auto splat = make_visible_splat(32);
    auto background = Tensor::zeros({3}, Device::GPU);
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
    auto bg = Tensor::zeros({3}, Device::GPU);

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
    splat->_max_screen_share = Tensor::zeros({static_cast<size_t>(kN)}, Device::GPU);
    auto bg = Tensor::zeros({3}, Device::GPU);
    bg.fill_(0.25f);

    AdamConfig cfg;
    cfg.lr = 1e-3f;
    cfg.initial_capacity = static_cast<size_t>(kN);
    AdamOptimizer opt(*splat, cfg);
    opt.allocate_gradients(static_cast<size_t>(kN));

    Tensor image0, image1, alpha0, alpha1;
    auto run_once = [&](Tensor& image_out, Tensor& alpha_out) {
        auto r = gsplat_rasterize_forward(
            camera, *splat, bg, 0, 0, 0, 0, 1.0f, false, GsplatRenderMode::RGB,
            /*use_gut=*/true);
        ASSERT_TRUE(r.has_value()) << r.error();
        auto output = std::move(r->first);
        auto ctx = std::move(r->second);
        ASSERT_GT(ctx.n_isects, 0);
        std::cout << "pairs=" << ctx.n_isects << " batches=" << std::max(size_t{1}, ctx.batches.size()) << std::endl;
        if (std::getenv("LFS_GSPLAT_PAIR_BUDGET"))
            EXPECT_GT(ctx.batches.size(), 1u);
        auto grad_image = Tensor::ones_like(output.image);
        auto grad_alpha = Tensor::zeros_like(output.alpha);
        gsplat_rasterize_backward(ctx, grad_image, grad_alpha, *splat, opt, Tensor{});
        EXPECT_EQ(splat->_max_screen_share.max().item<float>(), 0.f);
        image_out = output.image.clone();
        alpha_out = output.alpha.clone();
        ctx.isect_ids_ptr = nullptr;
        ctx.flatten_ids_ptr = nullptr;
    };

    run_once(image0, alpha0);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    auto means_g = opt.get_grad(ParamType::Means).clone();
    auto scale_g = opt.get_grad(ParamType::Scaling).clone();
    auto quat_g = opt.get_grad(ParamType::Rotation).clone();
    auto opa_g = opt.get_grad(ParamType::Opacity).clone();
    auto color_g = opt.get_grad(ParamType::Sh0).clone();

    opt.zero_grad(1);
    run_once(image1, alpha1);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    EXPECT_EQ(max_abs_diff_tensors(image0, image1), 0.0f);
    EXPECT_EQ(max_abs_diff_tensors(alpha0, alpha1), 0.0f);
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
        write_float_bin(std::filesystem::path(dump_dir) / "alpha.bin", alpha0);
        std::cout << dump_env << " wrote tensors to " << dump_dir << std::endl;
    }
    if (const char* ref_dir = std::getenv(ref_env)) {
        const auto rel_means = max_rel_diff(means_g, read_float_bin(std::filesystem::path(ref_dir) / "v_means.bin"));
        const auto rel_scales = max_rel_diff(scale_g, read_float_bin(std::filesystem::path(ref_dir) / "v_scales.bin"));
        const auto rel_quats = max_rel_diff(quat_g, read_float_bin(std::filesystem::path(ref_dir) / "v_quats.bin"));
        const auto rel_opa = max_rel_diff(opa_g, read_float_bin(std::filesystem::path(ref_dir) / "v_opacities.bin"));
        const auto rel_color = max_rel_diff(color_g, read_float_bin(std::filesystem::path(ref_dir) / "v_colors.bin"));
        auto alpha_ref = read_float_bin(std::filesystem::path(ref_dir) / "alpha.bin");
        auto alpha_cpu = alpha0.cpu();
        ASSERT_EQ(alpha_ref.size(), alpha_cpu.numel());
        EXPECT_EQ(std::memcmp(alpha_ref.data(), alpha_cpu.ptr<float>(), alpha_ref.size() * sizeof(float)), 0);
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
        EXPECT_EQ(render_max_abs, 0.f);
        EXPECT_EQ(std::memcmp(render_ref.data(), pi, nimg * sizeof(float)), 0);
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

namespace {
    Camera make_extra_parity_camera(lfs::core::CameraModelType model, int w = 96, int h = 64) {
        auto R = Tensor::from_vector({1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f}, {3, 3}, Device::CUDA);
        auto T = Tensor::from_vector({0.f, 0.f, 3.f}, {3}, Device::CUDA);
        Tensor radial, tangential;
        if (model == lfs::core::CameraModelType::PINHOLE) {
            radial = Tensor::from_vector({0.03f, -0.01f, 0.002f, 0.f, 0.f, 0.f}, {6}, Device::CPU);
            tangential = Tensor::from_vector({0.001f, -0.002f}, {2}, Device::CPU);
        } else if (model == lfs::core::CameraModelType::THIN_PRISM_FISHEYE) {
            radial = Tensor::from_vector({0.035f, 0.007f, 0.0006f, -0.0003f}, {4}, Device::CPU);
            tangential = Tensor::from_vector({0.001f, -0.002f, 0.0003f, -0.0004f}, {4}, Device::CPU);
        }
        return Camera(R, T, 40.f, 40.f, w * 0.5f, h * 0.5f,
                      radial, tangential, model, "parity", "", std::filesystem::path{}, w, h, 0);
    }
} // namespace

TEST_F(GsplatRasterizerTest, GutFromWorldDistortedGradParity) {
    auto camera = make_extra_parity_camera(lfs::core::CameraModelType::PINHOLE);
    run_gut_from_world_parity(camera, "GUT_DISTORTED_GRAD_DUMP", "GUT_DISTORTED_GRAD_REF");
}
TEST_F(GsplatRasterizerTest, GutFromWorldEquirectangularGradParity) {
    auto camera = make_extra_parity_camera(lfs::core::CameraModelType::EQUIRECTANGULAR);
    run_gut_from_world_parity(camera, "GUT_EQUIRECTANGULAR_GRAD_DUMP", "GUT_EQUIRECTANGULAR_GRAD_REF");
}
TEST_F(GsplatRasterizerTest, GutFromWorldThinPrismGradParity) {
    auto camera = make_extra_parity_camera(lfs::core::CameraModelType::THIN_PRISM_FISHEYE);
    run_gut_from_world_parity(camera, "GUT_THIN_PRISM_GRAD_DUMP", "GUT_THIN_PRISM_GRAD_REF");
}

// Opt-in because this emits 2.4986 billion pairs, bounded by the tile batches.
TEST_F(GsplatRasterizerTest, AggregateOverflowTrainingStep) {
    if (!std::getenv("GUT_LARGE_FRAME"))
        GTEST_SKIP();
    constexpr size_t n = 2600000;
    auto R = Tensor::from_vector({1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f}, {3, 3}, Device::CUDA);
    auto T = Tensor::zeros({3}, Device::CUDA);
    Camera camera(R, T, 1920.f, 1920.f, 1920.f, 1080.f, Tensor(), Tensor(),
                  lfs::core::CameraModelType::PINHOLE, "aggregate", "", std::filesystem::path{}, 3840, 2160, 0);
    auto means = Tensor::zeros({n, 3}, Device::CPU);
    auto rotations = Tensor::zeros({n, 4}, Device::CPU);
    for (size_t i = 0; i < n; ++i) {
        means.ptr<float>()[3 * i + 2] = 5.f;
        rotations.ptr<float>()[4 * i] = 1.f;
    }
    SplatData model(0, means.to(Device::CUDA), Tensor::full({n, 1, 3}, 0.5f, Device::CUDA),
                    Tensor::zeros({n, 0, 3}, Device::CUDA), Tensor::full({n, 3}, std::log(0.2f), Device::CUDA),
                    rotations.to(Device::CUDA), Tensor::zeros({n}, Device::CUDA), 1.f);
    AdamConfig cfg;
    cfg.initial_capacity = n;
    AdamOptimizer opt(model, cfg);
    opt.allocate_gradients(n);
    auto background = Tensor::zeros({3}, Device::CUDA);
    auto r = gsplat_rasterize_forward(camera, model, background,
                                      0, 0, 0, 0, 1.f, false, GsplatRenderMode::RGB, true);
    ASSERT_TRUE(r.has_value()) << r.error();
    EXPECT_EQ(static_cast<int64_t>(r->second.n_isects), 2498600000LL);
    ASSERT_GT(r->second.batches.size(), 1u);
    int64_t sum = 0, largest = 0;
    for (const auto& batch : r->second.batches) {
        sum += batch.count;
        largest = std::max(largest, batch.count);
    }
    EXPECT_EQ(sum, 2498600000LL);
    std::cout << "large batches=" << r->second.batches.size() << " max_pairs=" << largest << std::endl;
    auto before = model.sh0().clone();
    gsplat_rasterize_backward(r->second, Tensor::ones_like(r->first.image),
                              Tensor::zeros_like(r->first.alpha), model, opt, Tensor{});
    opt.step(1);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    auto after = model.sh0();
    EXPECT_GT(max_abs_diff_tensors(before, after), 0.f);
    std::cout << "Aggregate frame: forward/backward/Adam step complete, pairs=2498600000" << std::endl;
}

TEST_F(GsplatRasterizerTest, TileRangesPreserveCountsAndStableDepthOrder) {
    constexpr uint32_t n = 64, tw = 7, th = 5;
    std::vector<float> means(n * 2);
    std::vector<int32_t> radii(n * 2);
    for (uint32_t i = 0; i < n; ++i) {
        means[2 * i] = float((i * 17) % 140);
        means[2 * i + 1] = float((i * 23) % 100);
        radii[2 * i] = i % 9 == 0 ? 0 : 8 + (i * 11) % 80;
        radii[2 * i + 1] = 4 + (i * 7) % 70;
    }
    auto m = Tensor::from_blob(means.data(), {n, 2}, Device::CPU, DataType::Float32).to(Device::CUDA);
    auto r = Tensor::from_blob(radii.data(), {n, 2}, Device::CPU, DataType::Int32).to(Device::CUDA);
    auto d = Tensor::ones({n}, Device::CUDA); // Equal depth: stable tie order matters.
    auto counts = Tensor::empty({n}, Device::CUDA, DataType::Int32);
    auto offsets = Tensor::empty({tw * th + 1}, Device::CUDA, DataType::Int32);
    auto intersect = [&](gsplat_lfs::TileRange range) {
        return gsplat_lfs::intersect_tile(m.ptr<float>(), r.ptr<int32_t>(), d.ptr<float>(),
                                          nullptr, nullptr, 1, n, 16, tw, th, true, counts.ptr<int32_t>(), nullptr,
                                          offsets.ptr<int32_t>(), range);
    };
    const auto full = intersect({});
    ASSERT_GT(full.n_sort, 0);
    std::vector<int64_t> keys(full.n_isects);
    std::vector<int32_t> ids(full.n_isects), totals(n, 0), expected_counts(n);
    ASSERT_EQ(cudaMemcpy(keys.data(), full.isect_ids, keys.size() * sizeof(int64_t), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(ids.data(), full.flatten_ids, ids.size() * sizeof(int32_t), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(expected_counts.data(), counts.ptr<int32_t>(), n * sizeof(int32_t), cudaMemcpyDeviceToHost), cudaSuccess);
    int64_t sum = 0;
    // Includes partial rows, empty tiles, one-tile ranges and the last tile.
    for (uint32_t begin = 0; begin < tw * th;) {
        const uint32_t end = std::min(tw * th, begin + 1 + begin % 6);
        const auto batch = intersect({begin, end});
        std::vector<int64_t> batch_keys(batch.n_isects);
        std::vector<int32_t> batch_ids(batch.n_isects), batch_counts(n);
        ASSERT_EQ(cudaMemcpy(batch_keys.data(), batch.isect_ids, batch_keys.size() * sizeof(int64_t), cudaMemcpyDeviceToHost), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(batch_ids.data(), batch.flatten_ids, batch_ids.size() * sizeof(int32_t), cudaMemcpyDeviceToHost), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(batch_counts.data(), counts.ptr<int32_t>(), n * sizeof(int32_t), cudaMemcpyDeviceToHost), cudaSuccess);
        std::vector<int64_t> wanted_keys;
        std::vector<int32_t> wanted_ids;
        for (size_t j = 0; j < keys.size(); ++j) {
            const auto tile = uint32_t(uint64_t(keys[j]) >> 32);
            if (tile >= begin && tile < end) {
                wanted_keys.push_back(keys[j]);
                wanted_ids.push_back(ids[j]);
            }
        }
        EXPECT_EQ(batch_keys, wanted_keys);
        EXPECT_EQ(batch_ids, wanted_ids);
        for (size_t j = 0; j < n; ++j)
            totals[j] += batch_counts[j];
        sum += batch.n_isects;
        begin = end;
    }
    EXPECT_EQ(totals, expected_counts);
    EXPECT_EQ(sum, full.n_isects);
}

TEST_F(GsplatRasterizerTest, TileBatchesPreserveShRestAlphaAndDensificationGradients) {
    constexpr size_t n = 50000;
    auto seed = make_parity_splat(n, 0xC0FFEE01u);
    SplatData model(3, seed->means_raw(), seed->sh0_raw(),
                    Tensor::full({n, 15, 3}, 0.025f, Device::CUDA),
                    seed->scaling_raw(), seed->rotation_raw(), seed->opacity_raw(), 1.f);
    model.set_active_sh_degree(3);
    auto camera = make_extra_parity_camera(lfs::core::CameraModelType::THIN_PRISM_FISHEYE, 99, 67);
    auto bg = Tensor::full({3}, 0.25f, Device::CUDA);
    AdamConfig config;
    config.initial_capacity = n;
    AdamOptimizer opt(model, config);
    opt.allocate_gradients(n);
    auto error_map = Tensor::full({67, 99}, 0.2f, Device::CUDA);
    auto edge_map = Tensor::full({67, 99}, 0.3f, Device::CUDA);
    auto scores = Tensor::zeros({n}, Device::CUDA);
    const auto prior = lfs::core::environment::value("LFS_GSPLAT_PAIR_BUDGET");
    struct RestoreBudget {
        std::optional<std::string> prior;
        ~RestoreBudget() {
            (void)lfs::core::environment::set_value("LFS_GSPLAT_PAIR_BUDGET", prior.value_or(""));
        }
    } restore{prior};
    Tensor reference_image, reference_alpha, reference_scores, reference_densification;
    std::vector<Tensor> gradients;
    for (int arm = 0; arm < 2; ++arm) {
        ASSERT_TRUE(gsplat_lfs::release_intersect_thread_local_cache());
        ASSERT_TRUE(lfs::core::environment::set_value("LFS_GSPLAT_PAIR_BUDGET", arm ? "1" : ""));
        opt.zero_grad(1);
        scores.fill_(0.f);
        model._densification_info = Tensor::zeros({2, n}, Device::CUDA);
        auto result = gsplat_rasterize_forward(camera, model, bg, 0, 0, 0, 0, 1.f, false, GsplatRenderMode::RGB, true);
        ASSERT_TRUE(result.has_value()) << result.error();
        EXPECT_EQ(result->second.batches.empty(), arm == 0);
        // Nonzero opacity-output derivatives exercise the alpha/background terms.
        gsplat_rasterize_backward(result->second, Tensor::full(result->first.image.shape(), 0.7f, Device::CUDA),
                                  Tensor::full(result->first.alpha.shape(), 0.4f, Device::CUDA), model, opt,
                                  error_map, edge_map, scores);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        if (arm == 0) {
            reference_image = result->first.image.clone();
            reference_alpha = result->first.alpha.clone();
            reference_scores = scores.clone();
            reference_densification = model._densification_info.clone();
            for (auto param : AdamOptimizer::all_param_types())
                gradients.push_back(opt.get_grad(param).clone());
        } else {
            EXPECT_EQ(max_abs_diff_tensors(reference_image, result->first.image), 0.f);
            EXPECT_EQ(max_abs_diff_tensors(reference_alpha, result->first.alpha), 0.f);
            EXPECT_LT(max_rel_diff_tensors(reference_scores, scores), 1e-4f);
            EXPECT_LT(max_rel_diff_tensors(reference_densification, model._densification_info), 1e-4f);
            size_t i = 0;
            for (auto param : AdamOptimizer::all_param_types()) {
                EXPECT_LT(max_rel_diff_tensors(gradients[i++], opt.get_grad(param)), 1e-4f);
            }
        }
    }
}

// Regression for issue #2189. Run with and without the existing pair-budget
// test override to cover both the single-list and tile replay dispatch paths.
class GutScreenShare : public lfs::test::CudaBackendTest, public ::testing::WithParamInterface<int> {};

TEST_P(GutScreenShare, PublishedOncePerFrameAndConstrainsOnlyWhenEnabled) {
    using Model = lfs::core::CameraModelType;
    const int variant = GetParam();
    auto camera = variant == 0 ? make_camera(96, 64)
                               : make_extra_parity_camera(variant == 1   ? Model::PINHOLE
                                                          : variant == 2 ? Model::FISHEYE
                                                          : variant == 3 ? Model::EQUIRECTANGULAR
                                                                         : Model::THIN_PRISM_FISHEYE);
    constexpr size_t n = 50000;
    auto model = make_parity_splat(n, 0xC0FFEE01u);
    model->_max_screen_share = Tensor::zeros({n}, Device::CUDA);
    auto bg = Tensor::full({3}, .25f, Device::CUDA);
    AdamConfig config;
    config.initial_capacity = n;
    AdamOptimizer opt(*model, config);
    opt.allocate_gradients(n);
    opt.set_collect_projected_screen_share(true);
    // Statistics must not depend on hinge strength: clipping and splitting
    // still need them when the hinge is disabled.
    opt.set_screen_share_cap(model->_max_screen_share.ptr<float>(), n, .0001f, 0.f);
    auto r = gsplat_rasterize_forward(camera, *model, bg, 0, 0, 0, 0, 1.f, false, GsplatRenderMode::RGB, true);
    ASSERT_TRUE(r.has_value()) << r.error();
    if (lfs::core::environment::value("LFS_GSPLAT_PAIR_BUDGET") == "1")
        EXPECT_GT(r->second.batches.size(), 1u);
    EXPECT_EQ(model->_max_screen_share.max().item<float>(), 0.f);
    auto reference = r->first.image.clone();
    gsplat_rasterize_backward(r->second, Tensor::ones_like(r->first.image), Tensor{}, *model, opt);
    auto shares = model->_max_screen_share.clone();
    ASSERT_GT(shares.max().item<float>(), 0.f) << "GUT must publish screen-share measurements";
    EXPECT_LE(shares.max().item<float>(), 1.f);
    auto again = gsplat_rasterize_forward(camera, *model, bg, 0, 0, 0, 0, 1.f, false, GsplatRenderMode::RGB, true);
    ASSERT_TRUE(again.has_value());
    EXPECT_EQ(max_abs_diff_tensors(reference, again->first.image), 0.f);
    gsplat_rasterize_backward(again->second, Tensor::ones_like(again->first.image), Tensor{}, *model, opt);
    EXPECT_EQ(max_abs_diff_tensors(shares, model->_max_screen_share), 0.f)
        << "Revisiting the same splat in tiles/frames must not sum its area";

    // With no reconstruction gradient, an unset limit is a strict Adam no-op.
    auto before = model->scaling_raw().clone();
    opt.zero_grad(1);
    opt.set_screen_share_cap(nullptr, 0, 0.f, 0.f);
    opt.step(1);
    EXPECT_EQ(max_abs_diff_tensors(before, model->scaling_raw()), 0.f);
    // The configured limit must actually alter oversized splats via production Adam.
    opt.set_screen_share_cap(model->_max_screen_share.ptr<float>(), n, .0001f, 1.f);
    opt.step(2);
    EXPECT_GT(max_abs_diff_tensors(before, model->scaling_raw()), 0.f);
    auto a = before.cpu(), b = model->scaling_raw().cpu(), h = shares.cpu();
    size_t oversized = 0;
    for (size_t i = 0; i < n; ++i) {
        if (h.ptr<float>()[i] > .0001f) {
            ++oversized;
            EXPECT_LT(b.ptr<float>()[3 * i], a.ptr<float>()[3 * i]);
        } else {
            EXPECT_FLOAT_EQ(b.ptr<float>()[3 * i], a.ptr<float>()[3 * i]);
        }
    }
    EXPECT_GT(oversized, 0u);
}

INSTANTIATE_TEST_SUITE_P(CameraModels, GutScreenShare, ::testing::Values(0, 1, 2, 3, 4));

TEST_P(GutScreenShare, MatchesClippedProjectionAndKeepsRefinementWindowMaximum) {
    using Model = lfs::core::CameraModelType;
    const int variant = GetParam();
    auto camera = variant == 0 ? make_camera(96, 64)
                               : make_extra_parity_camera(variant == 1   ? Model::PINHOLE
                                                          : variant == 2 ? Model::FISHEYE
                                                          : variant == 3 ? Model::EQUIRECTANGULAR
                                                                         : Model::THIN_PRISM_FISHEYE);
    constexpr size_t n = 50000;
    auto model = make_parity_splat(n, 0xC0FFEE01u);
    model->_max_screen_share = Tensor::zeros({n}, Device::CUDA);
    auto bg = Tensor::full({3}, .25f, Device::CUDA);
    AdamConfig config;
    config.initial_capacity = n;
    AdamOptimizer opt(*model, config);
    opt.allocate_gradients(n);
    opt.set_collect_projected_screen_share(true);
    opt.set_screen_share_cap(model->_max_screen_share.ptr<float>(), n, .1f, 0.f);
    Tensor first;
    for (int frame = 0; frame < 3; ++frame) {
        if (frame == 1)
            model->scaling_raw().add_(-1.f);
        if (frame == 2)
            model->_max_screen_share.zero_();
        auto prior = model->_max_screen_share.cpu();
        auto r = gsplat_rasterize_forward(camera, *model, bg, 0, 0, 0, 0, 1.f, false, GsplatRenderMode::RGB, true);
        ASSERT_TRUE(r.has_value()) << r.error();
        std::vector<int32_t> radii(2 * n);
        std::vector<float> means2d(2 * n);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(radii.data(), r->second.radii_ptr, 2 * n * sizeof(int32_t), cudaMemcpyDeviceToHost), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(means2d.data(), r->second.means2d_ptr, 2 * n * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
        gsplat_rasterize_backward(r->second, Tensor::ones_like(r->first.image), Tensor{}, *model, opt);
        auto actual = model->_max_screen_share.cpu();
        for (size_t i = 0; i < n; ++i) {
            float expected = prior.ptr<float>()[i];
            if (radii[2 * i] > 0 && radii[2 * i + 1] > 0) {
                const float x = means2d[2 * i], y = means2d[2 * i + 1];
                const float dx = std::max(0.f, std::min(96.f, x + radii[2 * i]) - std::max(0.f, x - radii[2 * i]));
                const float dy = std::max(0.f, std::min(64.f, y + radii[2 * i + 1]) - std::max(0.f, y - radii[2 * i + 1]));
                expected = std::max(expected, (dx / 96.f) * (dy / 64.f));
            }
            EXPECT_NEAR(actual.ptr<float>()[i], expected, 1e-7f) << "splat=" << i << " frame=" << frame;
        }
        if (frame == 0)
            first = model->_max_screen_share.clone();
        // Shrinking 3D scales can also move the unscented projected center;
        // the per-splat CPU comparison above verifies the running maximum.
        if (frame == 2)
            EXPECT_LT(model->_max_screen_share.max().item<float>(), first.max().item<float>());
    }
}

TEST_P(GutScreenShare, MrnfPreservesGrowthCoverageAndConstrainsMatureSplats) {
    using Model = lfs::core::CameraModelType;
    const int variant = GetParam();
    auto camera = variant == 0 ? make_camera(96, 64)
                               : make_extra_parity_camera(variant == 1   ? Model::PINHOLE
                                                          : variant == 2 ? Model::FISHEYE
                                                          : variant == 3 ? Model::EQUIRECTANGULAR
                                                                         : Model::THIN_PRISM_FISHEYE);
    constexpr size_t n = 50000;
    for (bool enabled : {false, true}) {
        for (bool mature : {false, true}) {
            auto model = make_parity_splat(n, 0xC0FFEE01u);
            MRNF strategy(*model);
            auto params = lfs::core::param::OptimizationParameters::mrnf_defaults();
            params.gut = true;
            params.sh_degree = 0;
            params.max_cap = n;
            params.background_improvements = false;
            params.use_edge_map = false;
            params.start_refine = 2000;
            params.refine_every = 250;
            params.grow_until_iter = 15000;
            params.stop_refine = 28500;
            params.iterations = 30000;
            params.max_screen_share = enabled ? .0001f : 0.f;
            params.screen_share_penalty = 1.f;
            params.means_noise_weight = 0.f;
            params.scale_decay = 0.f;
            params.opacity_decay = 0.f;
            strategy.initialize(params);
            auto& opt = strategy.get_optimizer();
            opt.allocate_gradients(n);
            const int iteration = mature ? 15001 : 100;
            auto bg = Tensor::full({3}, .25f, Device::CUDA);
            auto r = gsplat_rasterize_forward(camera, *model, bg, 0, 0, 0, 0, 1.f, false, GsplatRenderMode::RGB, true);
            ASSERT_TRUE(r.has_value()) << r.error();
            auto before = model->scaling_raw().clone();
            gsplat_rasterize_backward(r->second, Tensor::zeros_like(r->first.image), Tensor{}, *model, opt);
            if (enabled)
                EXPECT_GT(model->_max_screen_share.max().item<float>(), params.max_screen_share);
            opt.zero_grad(iteration);
            // Exercise the actual strategy publication/schedule, not a test
            // manually binding the cap directly to Adam.
            strategy.step(iteration);
            const float diff = max_abs_diff_tensors(before, model->scaling_raw());
            if (enabled && mature)
                EXPECT_GT(diff, 0.f);
            else
                EXPECT_EQ(diff, 0.f);
        }
    }
}

TEST_P(GutScreenShare, MrnfClipsAfterGrowthAndLeavesUnsetLimitUnchanged) {
    using Model = lfs::core::CameraModelType;
    const int variant = GetParam();
    auto camera = variant == 0 ? make_camera(96, 64)
                               : make_extra_parity_camera(variant == 1   ? Model::PINHOLE
                                                          : variant == 2 ? Model::FISHEYE
                                                          : variant == 3 ? Model::EQUIRECTANGULAR
                                                                         : Model::THIN_PRISM_FISHEYE);
    constexpr size_t n = 50000;
    Tensor unset_scales[2];
    for (bool enabled : {false, true}) {
        for (bool mature : {false, true}) {
            auto model = make_parity_splat(n, 0xC0FFEE01u);
            MRNF strategy(*model);
            auto params = lfs::core::param::OptimizationParameters::mrnf_defaults();
            params.gut = true;
            params.sh_degree = 0;
            params.max_cap = n; // isolate clipping: no spare growth budget
            params.background_improvements = false;
            params.use_edge_map = false;
            params.start_refine = 2000;
            params.refine_every = 250;
            params.grow_until_iter = 15000;
            params.stop_refine = 28500;
            params.iterations = 30000;
            params.max_screen_share = enabled ? .0001f : 0.f;
            params.screen_share_penalty = 0.f;
            params.means_noise_weight = 0.f;
            params.scale_decay = 0.f;
            params.opacity_decay = 0.f;
            strategy.initialize(params);
            auto& opt = strategy.get_optimizer();
            opt.allocate_gradients(n);
            const int iteration = mature ? 15000 : 2250;
            auto bg = Tensor::full({3}, .25f, Device::CUDA);
            auto r = gsplat_rasterize_forward(camera, *model, bg, 0, 0, 0, 0, 1.f, false, GsplatRenderMode::RGB, true);
            ASSERT_TRUE(r.has_value()) << r.error();
            auto before = model->scaling_raw().clone();
            gsplat_rasterize_backward(r->second, Tensor::zeros_like(r->first.image), Tensor{}, *model, opt);
            if (enabled)
                EXPECT_GT(model->_max_screen_share.max().item<float>(), params.max_screen_share);
            strategy.post_backward(iteration, r->first);
            ASSERT_EQ(model->size(), n);
            const float diff = max_abs_diff_tensors(before, model->scaling_raw());
            if (!enabled) {
                // Existing decay performs an exp/log round trip even at zero
                // strength. Record that baseline to compare growth exactly.
                EXPECT_LT(diff, 1e-6f);
                unset_scales[mature] = model->scaling_raw().clone();
            } else if (mature) {
                EXPECT_GT(max_abs_diff_tensors(unset_scales[mature], model->scaling_raw()), 1e-3f);
            } else {
                EXPECT_EQ(max_abs_diff_tensors(unset_scales[mature], model->scaling_raw()), 0.f);
            }
        }
    }
}

class GutScreenShareGeometry : public lfs::test::CudaBackendTest {};

TEST_F(GutScreenShareGeometry, ClippingVisibilityWindowResetAndNonDefaultStream) {
    auto radii = Tensor::from_vector(std::vector<int32_t>{10, 10, 10, 10, 0, 10, 100, 100}, {4, 2}, Device::CUDA);
    auto centers = Tensor::from_vector({5.f, 5.f, 95.f, 45.f, 50.f, 25.f, 50.f, 25.f}, {4, 2}, Device::CUDA);
    auto shares = Tensor::from_vector({.1f, 0.f, .3f, 0.f}, {4}, Device::CUDA);
    struct Stream {
        cudaStream_t value = nullptr;
        ~Stream() {
            if (value)
                cudaStreamDestroy(value);
        }
    } stream;
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream.value, cudaStreamNonBlocking), cudaSuccess);
    auto record = [&] {
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        kernels::launch_accumulate_projected_screen_share(radii.ptr<int32_t>(), centers.ptr<float>(), shares.ptr<float>(), 4, 100, 50, stream.value);
        ASSERT_EQ(cudaStreamSynchronize(stream.value), cudaSuccess);
    };
    record();
    auto h = shares.cpu();
    EXPECT_FLOAT_EQ(h.ptr<float>()[0], .1f);
    EXPECT_NEAR(h.ptr<float>()[1], 225.f / 5000.f, 1e-7f);
    EXPECT_FLOAT_EQ(h.ptr<float>()[2], .3f); // invisible retains earlier views
    EXPECT_FLOAT_EQ(h.ptr<float>()[3], 1.f);
    auto previous = shares.clone();
    radii = Tensor::from_vector(std::vector<int32_t>{1, 1, 1, 1, 0, 1, 1, 1}, {4, 2}, Device::CUDA);
    record();
    EXPECT_EQ(max_abs_diff_tensors(previous, shares), 0.f);
    shares.zero_();
    record();
    h = shares.cpu();
    for (size_t i : {0u, 1u, 3u})
        EXPECT_NEAR(h.ptr<float>()[i], 4.f / 5000.f, 1e-8f);
    EXPECT_EQ(h.ptr<float>()[2], 0.f);
}

class GutScreenShareStrategy : public lfs::test::CudaBackendTest {};

TEST_F(GutScreenShareStrategy, RendererSwitchStartsANewMeasurementWindow) {
    auto model = make_parity_splat(100, 42);
    MRNF strategy(*model);
    auto params = lfs::core::param::OptimizationParameters::mrnf_defaults();
    params.gut = true;
    params.max_cap = 100;
    params.sh_degree = 0;
    params.background_improvements = false;
    strategy.initialize(params);
    ASSERT_TRUE(strategy.get_optimizer().collect_projected_screen_share());
    model->_max_screen_share.fill_(.5f);
    params.gut = false;
    strategy.set_optimization_params(params);
    EXPECT_FALSE(strategy.get_optimizer().collect_projected_screen_share());
    EXPECT_EQ(model->_max_screen_share.max().item<float>(), 0.f);
    model->_max_screen_share.fill_(.5f);
    params.gut = true;
    strategy.set_optimization_params(params);
    EXPECT_TRUE(strategy.get_optimizer().collect_projected_screen_share());
    EXPECT_EQ(model->_max_screen_share.max().item<float>(), 0.f);
    params.max_screen_share = 0.f;
    strategy.set_optimization_params(params);
    EXPECT_FALSE(strategy.get_optimizer().collect_projected_screen_share());
}

TEST_F(GutScreenShareStrategy, MatureRefinementsReduceActualProjectedAreaBelowLimit) {
    auto camera = make_camera(96, 64);
    for (bool enabled : {false, true}) {
        auto model = make_visible_splat(1);
        model->means().zero_();
        // Front-facing surface: isolate clipping of the projected axes.
        model->scaling_raw() = Tensor::from_vector({-2.f, -2.f, -6.f}, {1, 3}, Device::CUDA);
        MRNF strategy(*model);
        auto params = lfs::core::param::OptimizationParameters::mrnf_defaults();
        params.gut = true;
        params.sh_degree = 0;
        params.max_cap = 1;
        params.background_improvements = false;
        params.use_edge_map = false;
        params.start_refine = 2000;
        params.refine_every = 250;
        params.grow_until_iter = 15000;
        params.stop_refine = 28500;
        params.iterations = 30000;
        params.max_screen_share = enabled ? .1f : 0.f;
        params.screen_share_penalty = 0.f;
        params.means_noise_weight = 0.f;
        params.scale_decay = 0.f;
        params.opacity_decay = 0.f;
        strategy.initialize(params);
        auto& opt = strategy.get_optimizer();
        opt.allocate_gradients(1);
        auto bg = Tensor::zeros({3}, Device::CUDA);
        float initial_area = 0.f;
        float final_area = 0.f;
        for (int window = 0; window <= 52; ++window) {
            auto r = gsplat_rasterize_forward(camera, *model, bg, 0, 0, 0, 0, 1.f, false, GsplatRenderMode::RGB, true);
            ASSERT_TRUE(r.has_value()) << r.error();
            int32_t radii[2]{};
            float center[2]{};
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
            ASSERT_EQ(cudaMemcpy(radii, r->second.radii_ptr, sizeof(radii), cudaMemcpyDeviceToHost), cudaSuccess);
            ASSERT_EQ(cudaMemcpy(center, r->second.means2d_ptr, sizeof(center), cudaMemcpyDeviceToHost), cudaSuccess);
            const float dx = std::max(0.f, std::min(96.f, center[0] + radii[0]) - std::max(0.f, center[0] - radii[0]));
            const float dy = std::max(0.f, std::min(64.f, center[1] + radii[1]) - std::max(0.f, center[1] - radii[1]));
            final_area = radii[0] > 0 && radii[1] > 0 ? dx * dy / (96.f * 64.f) : 0.f;
            if (window == 0)
                initial_area = final_area;
            gsplat_rasterize_backward(r->second, Tensor::zeros_like(r->first.image), Tensor{}, *model, opt);
            if (window < 52)
                strategy.post_backward(15000 + 250 * window, r->first);
        }
        std::cout << "GUT projected-area constraint: enabled=" << enabled
                  << " initial=" << initial_area << " final=" << final_area << '\n';
        EXPECT_GT(initial_area, .1f);
        if (enabled) {
            EXPECT_GT(final_area, 0.f); // Constrain it without deleting it.
            EXPECT_LE(final_area, .1f);
        } else {
            EXPECT_EQ(final_area, initial_area);
        }
        ASSERT_EQ(model->size(), 1);
    }
}

TEST_F(GsplatRasterizerTest, BackwardUsesCurrentQueueAndJoinsForwardStorage) {
    TensorWorkQueue producer(GpuBackend::CUDA);
    TensorWorkQueue consumer(GpuBackend::CUDA);
    TensorWorkQueue::Scope scope(producer);
    auto camera = make_camera(32, 32);
    auto splat = make_visible_splat(16);
    auto bg = Tensor::zeros({3}, Device::GPU);
    AdamConfig config;
    config.initial_capacity = 32;
    AdamOptimizer optimizer(*splat, config);
    optimizer.allocate_gradients(32);
    for (int repeat = 0; repeat < 3; ++repeat) {
        auto result = gsplat_rasterize_forward(camera, *splat, bg);
        ASSERT_TRUE(result.has_value()) << result.error();
        auto gradient = Tensor::ones_like(result->first.image);
        auto alpha_gradient = Tensor::zeros_like(result->first.alpha);
        {
            TensorWorkQueue::Scope backward_scope(consumer);
            gsplat_rasterize_backward(result->second, gradient, alpha_gradient,
                                      *splat, optimizer, Tensor{});
            EXPECT_EQ(optimizer.get_grad(ParamType::Means).stream(), consumer.native_handle());
        }
        optimizer.zero_grad(repeat);
    }
    consumer.wait();
    producer.wait();
    release_gsplat_rasterizer_thread_local_caches();
}

TEST_F(GsplatRasterizerTest, BackwardJoinsAuxiliaryProducersAndOutputs) {
    TensorWorkQueue forward(GpuBackend::CUDA), producer(GpuBackend::CUDA), backward(GpuBackend::CUDA);
    TensorWorkQueue::Scope scope(forward);
    for (const bool use_error : {false, true}) {
        for (int delayed = 0; delayed < 4; ++delayed) {
            if (!use_error && delayed == 0)
                continue;
            SCOPED_TRACE(::testing::Message() << "error=" << use_error << " delayed=" << delayed);
            auto camera = make_camera(64, 64);
            auto splat = make_visible_splat(32);
            splat->_densification_info = Tensor::zeros({2, 32}, Device::GPU);
            AdamConfig config;
            config.initial_capacity = 64;
            AdamOptimizer optimizer(*splat, config);
            optimizer.allocate_gradients(64);
            auto bg = Tensor::zeros({3}, Device::GPU);
            auto error = Tensor::ones({64, 64}, Device::GPU);
            auto edge = Tensor::ones({64, 64}, Device::GPU);
            auto scores = Tensor::zeros({32}, Device::GPU);
            auto result = gsplat_rasterize_forward(camera, *splat, bg, 0, 0, 0, 0,
                                                   1.f, false, GsplatRenderMode::RGB, true);
            ASSERT_TRUE(result.has_value()) << result.error();
            auto gradient = Tensor::ones_like(result->first.image);
            auto alpha_gradient = Tensor::zeros_like(result->first.alpha);
            Tensor* target = std::array<Tensor*, 4>{&error, &edge, &scores, &splat->_densification_info}[delayed];
            target->fill_(delayed < 2 ? 0.f : -100.f);
            forward.wait();
            {
                TensorWorkQueue::Scope producer_scope(producer);
                target->set_stream(static_cast<cudaStream_t>(producer.native_handle()));
                ASSERT_EQ(tensor_hardening::launch_delay_kernel(
                              static_cast<cudaStream_t>(producer.native_handle()), 150000000),
                          cudaSuccess);
                target->fill_(delayed < 2 ? 1.f : 0.f);
            }
            {
                TensorWorkQueue::Scope backward_scope(backward);
                gsplat_rasterize_backward(result->second, gradient, alpha_gradient, *splat, optimizer,
                                          use_error ? error : Tensor{}, edge, scores);
                EXPECT_EQ(splat->_densification_info.stream(), backward.native_handle());
                const auto score_cpu = scores.cpu();
                const auto info_cpu = splat->_densification_info.cpu();
                EXPECT_GT(*std::max_element(score_cpu.ptr<float>(), score_cpu.ptr<float>() + score_cpu.numel()), 0.f);
                EXPECT_GT(*std::max_element(info_cpu.ptr<float>(), info_cpu.ptr<float>() + info_cpu.numel()), 0.f);
                EXPECT_GE(*std::min_element(score_cpu.ptr<float>(), score_cpu.ptr<float>() + score_cpu.numel()), 0.f);
                EXPECT_GE(*std::min_element(info_cpu.ptr<float>(), info_cpu.ptr<float>() + info_cpu.numel()), 0.f);
            }
            producer.wait();
            backward.wait();
        }
    }
    forward.wait();
    release_gsplat_rasterizer_thread_local_caches();
}

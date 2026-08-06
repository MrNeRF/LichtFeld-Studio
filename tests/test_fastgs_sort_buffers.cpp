/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Phase 1.1 — Persistent high-water sort buffers in FastGS forward.
 * Phase 1.2 — Remove n_instances hard sync (async + capacity path).
 */

#include "core/alloc_counter.hpp"
#include "core/camera.hpp"
#include "core/cuda/memory_arena.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "training/rasterization/fast_rasterizer.hpp"
#include "training/rasterization/fastgs/rasterization/include/forward.h"

#include <cmath>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
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
        // Place a few gaussians in front of the camera so n_instances > 0.
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
            rot[static_cast<size_t>(i) * 4] = 1.f; // identity quat
        }
        auto rotation = Tensor::from_blob(rot.data(), {static_cast<size_t>(n), 4}, Device::CPU, DataType::Float32)
                            .to(Device::CUDA);
        auto opacity = Tensor::full({static_cast<size_t>(n)}, 2.0f, Device::CUDA);
        return std::make_unique<SplatData>(0, means, sh0, shN, scaling, rotation, opacity, 1.0f);
    }

    void cleanup_arena() {
        GlobalArenaManager::instance().get_arena().full_reset();
    }

} // namespace

class FastGSSortBufferTest : public ::testing::Test {
protected:
    void SetUp() override {
        bg_ = Tensor::zeros({3}, Device::CUDA);
        camera_ = std::make_unique<Camera>(make_camera(64, 64));
        splat_ = make_splat(32);
    }

    void TearDown() override {
        splat_.reset();
        camera_.reset();
        cleanup_arena();
    }

    Tensor bg_;
    std::unique_ptr<Camera> camera_;
    std::unique_ptr<SplatData> splat_;
};

// ---------------------------------------------------------------------------
// Task 1.1 — second same-size forward must issue 0 real driver allocs.
// Baseline (pre-1.1): ~5 cudaMallocAsync for sort keys×2, indices×2, CUB WS.
// ---------------------------------------------------------------------------
TEST_F(FastGSSortBufferTest, SteadyStateSecondForwardHasZeroSortAllocs) {
    // Warmup: arena + thread-local image buffers + first sort-buffer growth.
    {
        auto warm = fast_rasterize_forward(*camera_, *splat_, bg_, 0, 0, 0, 0, false);
        ASSERT_TRUE(warm.has_value()) << lfs::format_for_developer(warm.error());
        ASSERT_GT(warm->second.forward_ctx.n_instances, 0)
            << "fixture must produce visible instances so the sort path runs";
        // Explicit release before next forward (also happens in context dtor).
        warm->second.release_forward_context();
    }
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    // First measured forward at steady size — may still grow if warm n_instances
    // differed; with fixed scene it should already be at high-water after warm.
    {
        const auto snap = alloc_counter::snapshot();
        auto r1 = fast_rasterize_forward(*camera_, *splat_, bg_, 0, 0, 0, 0, false);
        ASSERT_TRUE(r1.has_value()) << lfs::format_for_developer(r1.error());
        r1->second.release_forward_context();
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        (void)alloc_counter::delta_since(snap);
    }

    // Second consecutive same-size forward: sort path must not touch the driver.
    const auto snap2 = alloc_counter::snapshot();
    auto r2 = fast_rasterize_forward(*camera_, *splat_, bg_, 0, 0, 0, 0, false);
    ASSERT_TRUE(r2.has_value()) << lfs::format_for_developer(r2.error());
    ASSERT_GT(r2->second.forward_ctx.n_instances, 0);
    r2->second.release_forward_context();
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    const auto delta2 = alloc_counter::delta_since(snap2);
    EXPECT_EQ(delta2, 0u)
        << "steady-state FastGS forward at fixed size must issue 0 real device "
           "allocs (sort keys×2, indices×2, CUB workspace should be grow-only "
           "persistent). Observed delta="
        << delta2;
}

// ---------------------------------------------------------------------------
// Task 1.2 — async n_instances path matches sync path (pixel golden).
// First forward forces mid-pipeline sync (empty capacity); second uses steady
// async/capacity path. Images must be bit-identical.
// ---------------------------------------------------------------------------
TEST_F(FastGSSortBufferTest, AsyncPathMatchesSyncPathPixels) {
    using fast_lfs::rasterization::n_instances_fallback_sync_count;
    using fast_lfs::rasterization::reset_n_instances_fallback_sync_count;
    using fast_lfs::rasterization::reset_sort_capacity_for_testing;
    using fast_lfs::rasterization::set_force_n_instances_sync_for_testing;

    reset_sort_capacity_for_testing();
    reset_n_instances_fallback_sync_count();
    set_force_n_instances_sync_for_testing(false);

    // Sync-path render (capacity empty → fallback sync).
    Tensor image_sync;
    {
        const auto before = n_instances_fallback_sync_count();
        auto r = fast_rasterize_forward(*camera_, *splat_, bg_, 0, 0, 0, 0, false);
        ASSERT_TRUE(r.has_value()) << lfs::format_for_developer(r.error());
        ASSERT_GT(n_instances_fallback_sync_count(), before)
            << "first forward after capacity reset must take the sync fallback";
        image_sync = r->first.image.to(Device::CPU).contiguous();
        r->second.release_forward_context();
    }
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    // Steady async path (capacity warm).
    Tensor image_async;
    {
        const auto before = n_instances_fallback_sync_count();
        auto r = fast_rasterize_forward(*camera_, *splat_, bg_, 0, 0, 0, 0, false);
        ASSERT_TRUE(r.has_value()) << lfs::format_for_developer(r.error());
        EXPECT_EQ(n_instances_fallback_sync_count(), before)
            << "steady same-size forward must not mid-pipeline-sync for n_instances";
        image_async = r->first.image.to(Device::CPU).contiguous();
        r->second.release_forward_context();
    }

    ASSERT_EQ(image_sync.numel(), image_async.numel());
    const float* a = image_sync.ptr<float>();
    const float* b = image_async.ptr<float>();
    for (size_t i = 0; i < image_sync.numel(); ++i) {
        ASSERT_EQ(a[i], b[i]) << "pixel mismatch at i=" << i
                              << " sync=" << a[i] << " async=" << b[i];
    }
}

// ---------------------------------------------------------------------------
// Task 1.2 — capacity growth / fallback path fires when high-water is reset.
// ---------------------------------------------------------------------------
TEST_F(FastGSSortBufferTest, CapacityGrowthTakesFallbackSync) {
    using fast_lfs::rasterization::n_instances_fallback_sync_count;
    using fast_lfs::rasterization::reset_n_instances_fallback_sync_count;
    using fast_lfs::rasterization::reset_sort_capacity_for_testing;
    using fast_lfs::rasterization::set_force_n_instances_sync_for_testing;

    set_force_n_instances_sync_for_testing(false);
    reset_sort_capacity_for_testing();
    reset_n_instances_fallback_sync_count();

    // Warm capacity with the default 32-gaussian scene.
    {
        auto r = fast_rasterize_forward(*camera_, *splat_, bg_, 0, 0, 0, 0, false);
        ASSERT_TRUE(r.has_value());
        r->second.release_forward_context();
    }
    const auto after_warm = n_instances_fallback_sync_count();
    ASSERT_GE(after_warm, 1u);

    // Steady re-render: no additional fallback.
    {
        auto r = fast_rasterize_forward(*camera_, *splat_, bg_, 0, 0, 0, 0, false);
        ASSERT_TRUE(r.has_value());
        r->second.release_forward_context();
    }
    EXPECT_EQ(n_instances_fallback_sync_count(), after_warm);

    // Force capacity drop → next forward must fall back and grow.
    reset_sort_capacity_for_testing();
    {
        const auto before = n_instances_fallback_sync_count();
        auto r = fast_rasterize_forward(*camera_, *splat_, bg_, 0, 0, 0, 0, false);
        ASSERT_TRUE(r.has_value());
        EXPECT_GT(n_instances_fallback_sync_count(), before)
            << "capacity reset must force a mid-pipeline sync fallback to grow";
        r->second.release_forward_context();
    }

    set_force_n_instances_sync_for_testing(false);
}

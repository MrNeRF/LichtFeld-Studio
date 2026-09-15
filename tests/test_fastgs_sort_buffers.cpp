/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * FastGS sort storage is exact per-frame arena storage.
 */

#include "core/camera.hpp"
#include "core/cuda/memory_arena.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "training/rasterization/fast_rasterizer.hpp"
#include "training/rasterization/fastgs/rasterization/include/forward.h"
#include "training/rasterization/fastgs/rasterization/include/rasterization_api.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <thread>
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
        release_fastgs_sort_workspace_buffers();
        cleanup_arena();
    }

    Tensor bg_;
    std::unique_ptr<Camera> camera_;
    std::unique_ptr<SplatData> splat_;
};

TEST_F(FastGSSortBufferTest, SortWorkspaceIsExactAndArenaOwned) {
    using fast_lfs::rasterization::sort_workspace_allocated_bytes;
    using fast_lfs::rasterization::sort_workspace_required_bytes;

    auto warm = fast_rasterize_forward(*camera_, *splat_, bg_, 0, 0, 0, 0, false);
    ASSERT_TRUE(warm.has_value()) << lfs::format_for_developer(warm.error());
    ASSERT_GT(warm->second.forward_ctx.n_instances, 0)
        << "fixture must produce visible instances so the sort path runs";
    const auto sort_bytes = warm->second.forward_ctx.per_instance_sort_total_size;
    ASSERT_GT(sort_bytes, 0u);
    EXPECT_EQ(sort_workspace_required_bytes(), 0u);
    EXPECT_EQ(sort_workspace_allocated_bytes(), 0u);

    const auto frame_buffers = GlobalArenaManager::instance().get_arena().get_frame_buffers(
        warm->second.forward_ctx.frame_id);
    EXPECT_TRUE(std::any_of(frame_buffers.begin(), frame_buffers.end(),
                            [sort_bytes](const auto& buffer) {
                                return buffer.size >= sort_bytes;
                            }))
        << "FastGS sort storage must be recorded in the arena frame";
    warm->second.release_forward_context();
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
}

TEST_F(FastGSSortBufferTest, ExactPathMatchesRepeatedPathPixels) {
    Tensor image_sync;
    {
        auto r = fast_rasterize_forward(*camera_, *splat_, bg_, 0, 0, 0, 0, false);
        ASSERT_TRUE(r.has_value()) << lfs::format_for_developer(r.error());
        image_sync = r->first.image.to(Device::CPU).contiguous();
        r->second.release_forward_context();
    }
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    Tensor image_async;
    {
        auto r = fast_rasterize_forward(*camera_, *splat_, bg_, 0, 0, 0, 0, false);
        ASSERT_TRUE(r.has_value()) << lfs::format_for_developer(r.error());
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

TEST_F(FastGSSortBufferTest, RepeatedFramesUseTheCurrentArenaFrame) {
    auto first = fast_rasterize_forward(*camera_, *splat_, bg_, 0, 0, 0, 0, false);
    ASSERT_TRUE(first.has_value());
    const auto first_frame = first->second.forward_ctx.frame_id;
    const auto first_sort = first->second.forward_ctx.per_instance_sort_total_size;
    first->second.release_forward_context();

    auto second = fast_rasterize_forward(*camera_, *splat_, bg_, 0, 0, 0, 0, false);
    ASSERT_TRUE(second.has_value());
    EXPECT_NE(second->second.forward_ctx.frame_id, first_frame);
    EXPECT_EQ(second->second.forward_ctx.per_instance_sort_total_size, first_sort);
    second->second.release_forward_context();
}

// cudaPointerGetAttributes preflight is debug-only.
// Release (NDEBUG) builds must not call it per-step; after N frames the
// instrumented counter stays 0.
TEST_F(FastGSSortBufferTest, ReleasePreflightPointerAttrsAreSkipped) {
    using fast_lfs::rasterization::preflight_pointer_attr_call_count;
    using fast_lfs::rasterization::reset_preflight_pointer_attr_call_count;

    reset_preflight_pointer_attr_call_count();
    constexpr int kFrames = 8;
    for (int i = 0; i < kFrames; ++i) {
        auto r = fast_rasterize_forward(*camera_, *splat_, bg_, 0, 0, 0, 0, false);
        ASSERT_TRUE(r.has_value()) << lfs::format_for_developer(r.error());
        r->second.release_forward_context();
    }
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    const auto calls = preflight_pointer_attr_call_count();
#ifdef NDEBUG
    EXPECT_EQ(calls, 0u)
        << "Release/NDEBUG builds must not call cudaPointerGetAttributes per "
           "forward (observed "
        << calls << " after " << kFrames << " frames)";
#else
    // Debug builds keep full preflight (~10 attrs × frames).
    EXPECT_GE(calls, static_cast<std::uint64_t>(kFrames) * 8u)
        << "Debug builds should still run pointer attribute preflight";
#endif
}

// ---------------------------------------------------------------------------
// VRAM audit — arena-owned sort storage + raster output buffers release on join.
// Spawn N worker threads that each run a few FastGS forwards, explicitly
// release the remaining renderer caches, then join.
// cudaMemGetInfo free must return near the pre-spawn baseline.
// ---------------------------------------------------------------------------
TEST(FastGSThreadLocalCacheTest, SpawnRenderJoinReturnsVram) {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        GTEST_SKIP() << "CUDA device unavailable";
    }

    // Warm primary thread caches so first-touch noise is outside the measurement.
    {
        auto cam = make_camera(64, 64);
        auto splat = make_splat(64);
        auto bg = Tensor::zeros({3}, Device::CUDA);
        auto r = fast_rasterize_forward(cam, *splat, bg, 0, 0, 0, 0, false);
        ASSERT_TRUE(r.has_value()) << lfs::format_for_developer(r.error());
        r->second.release_forward_context();
        release_fast_rasterizer_thread_local_caches();
        release_fastgs_sort_workspace_buffers();
        cleanup_arena();
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    }

    std::size_t free_before = 0, total = 0;
    ASSERT_EQ(cudaMemGetInfo(&free_before, &total), cudaSuccess);

    constexpr int kThreads = 4;
    constexpr int kForwardsPerThread = 3;
    std::atomic<int> failures{0};

    auto worker = [&]() {
        if (cudaSetDevice(0) != cudaSuccess) {
            failures.fetch_add(1);
            return;
        }
        try {
            auto cam = make_camera(96, 96);
            auto splat = make_splat(128);
            auto bg = Tensor::zeros({3}, Device::CUDA);
            for (int i = 0; i < kForwardsPerThread; ++i) {
                auto r = fast_rasterize_forward(cam, *splat, bg, 0, 0, 0, 0, false);
                if (!r.has_value()) {
                    failures.fetch_add(1);
                    return;
                }
                r->second.release_forward_context();
            }
            // Explicit TLS release (mirrors training-thread shutdown). Without
            // this, join relies solely on TLS destructors — which must also free.
            release_fast_rasterizer_thread_local_caches();
            release_fastgs_sort_workspace_buffers();
            if (cudaDeviceSynchronize() != cudaSuccess) {
                failures.fetch_add(1);
            }
        } catch (...) {
            failures.fetch_add(1);
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back(worker);
    }
    for (auto& th : threads) {
        th.join();
    }
    ASSERT_EQ(failures.load(), 0) << "one or more worker threads failed";

    cleanup_arena();
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::size_t free_after = 0;
    ASSERT_EQ(cudaMemGetInfo(&free_after, &total), cudaSuccess);

    // Each worker builds arena sort storage plus image TLS. If either leaks,
    // free drops by multiple MiB * kThreads.
    constexpr std::size_t kSlack = 32ull << 20; // 32 MiB driver/fragmentation slack
    EXPECT_GE(free_after + kSlack, free_before)
        << "FastGS sort/raster storage leaked across spawn-render-join "
        << "free_before=" << free_before << " free_after=" << free_after
        << " delta_MiB="
        << (static_cast<long long>(free_before) - static_cast<long long>(free_after)) /
               (1024 * 1024);
}

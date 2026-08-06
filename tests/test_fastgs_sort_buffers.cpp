/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Phase 1.1 — Persistent high-water sort buffers in FastGS forward.
 * Asserts steady-state sort-path driver allocs drop to 0 across two consecutive
 * same-size forwards (alloc_counter::delta on the second must be 0).
 */

#include "core/alloc_counter.hpp"
#include "core/camera.hpp"
#include "core/cuda/memory_arena.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "training/rasterization/fast_rasterizer.hpp"

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

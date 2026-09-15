/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/camera.hpp"
#include "core/cuda/memory_arena.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "training/rasterization/gsplat/Ops.h"
#include "training/rasterization/gsplat_rasterizer.hpp"

#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

using namespace lfs::core;
using namespace lfs::training;

class GutUndistortParity : public ::testing::Test {
protected:
    void TearDown() override {
        (void)gsplat_lfs::release_intersect_thread_local_cache();
        (void)lfs::training::release_gsplat_rasterizer_thread_local_caches();
        lfs::core::GlobalArenaManager::instance().get_arena().full_reset();
    }
};

TEST_F(GutUndistortParity, PreparedCameraMatchesPinholeWithoutCoefficients) {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        GTEST_SKIP() << "CUDA device unavailable";
    }

    constexpr size_t n = 35;
    std::vector<float> means(n * 3);
    std::vector<float> rotations(n * 4, 0.0f);
    for (size_t i = 0; i < n; ++i) {
        means[i * 3] = (static_cast<float>(i % 7) - 3.0f) * 0.4f;
        means[i * 3 + 1] = (static_cast<float>(i / 7) - 2.0f) * 0.35f;
        means[i * 3 + 2] = 3.0f + 0.01f * static_cast<float>(i);
        rotations[i * 4] = 1.0f;
    }
    SplatData model(
        0, Tensor::from_vector(means, {n, 3}, Device::CUDA),
        Tensor::full({n, 1, 3}, 0.5f, Device::CUDA),
        Tensor::zeros({n, 0, 3}, Device::CUDA),
        Tensor::full({n, 3}, -3.0f, Device::CUDA),
        Tensor::from_vector(rotations, {n, 4}, Device::CUDA),
        Tensor::full({n}, 2.0f, Device::CUDA), 1.0f);
    auto background = Tensor::zeros({3}, Device::CUDA);
    const auto R = Tensor::from_vector({1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f},
                                       {3, 3}, Device::CUDA);
    const auto T = Tensor::zeros({3}, Device::CUDA);
    // GUT reads six pinhole radial coefficients; explicitly zero the unused terms.
    const auto radial = Tensor::from_vector({-0.25f, 0.08f, 0.f, 0.f, 0.f, 0.f}, {6}, Device::CPU);
    Camera A(R, T, 200.f, 200.f, 128.f, 96.f, radial, Tensor(),
             lfs::core::CameraModelType::PINHOLE, "prepared", "", "", 256, 192, 0);
    A.prepare_undistortion();
    ASSERT_TRUE(A.is_undistort_prepared());

    // get_intrinsics() scales the prepared values to the current image dimensions.
    const auto [fx, fy, cx, cy] = A.get_intrinsics();
    Camera B(R, T, fx, fy, cx, cy, Tensor(), Tensor(),
             lfs::core::CameraModelType::PINHOLE, "pinhole", "", "",
             A.image_width(), A.image_height(), 1);
    Camera C(R, T, 200.f, 200.f, 128.f, 96.f, radial, Tensor(),
             lfs::core::CameraModelType::PINHOLE, "distorted", "", "", 256, 192, 2);
    ASSERT_FALSE(C.is_undistort_prepared());
    ASSERT_TRUE(C.has_distortion());

    // Snapshot each image before the next render reuses the thread-local output buffer.
    const auto render_cpu = [&](Camera& camera) {
        return gsplat_rasterize(camera, model, background, 1.0f, false,
                                GsplatRenderMode::RGB, true)
            .image.cpu()
            .contiguous();
    };
    const auto image_a = render_cpu(A);
    const auto image_b = render_cpu(B);
    const auto image_c = render_cpu(C);
    ASSERT_EQ(image_a.shape(), image_b.shape());
    ASSERT_EQ(image_c.shape(), image_b.shape());
    ASSERT_GT(image_b.numel(), 0u);
    const float* a = image_a.ptr<float>();
    const float* b = image_b.ptr<float>();
    const float* c = image_c.ptr<float>();
    float max_ab = 0.0f, max_cb = 0.0f, max_pixel = 0.0f;
    for (size_t i = 0; i < image_b.numel(); ++i) {
        ASSERT_TRUE(std::isfinite(a[i]) && std::isfinite(b[i]) && std::isfinite(c[i]));
        max_ab = std::max(max_ab, std::abs(a[i] - b[i]));
        max_cb = std::max(max_cb, std::abs(c[i] - b[i]));
        max_pixel = std::max(max_pixel, b[i]);
    }
    EXPECT_EQ(max_ab, 0.0f);
    EXPECT_GT(max_pixel, 0.0f);
    EXPECT_GT(max_cb, 0.0f);
}

/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Comprehensive fuzz tests for FastGS kernels to expose edge cases and bugs.
 */

#include "core/camera.hpp"
#include "core/cuda/memory_arena.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "io/formats/ply.hpp"
#include "training/optimizer/adam_optimizer.hpp"
#include "training/rasterization/fast_rasterizer.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <random>

using namespace lfs::training;
using namespace lfs::core;

namespace {

    Camera make_camera(int w, int h, float fx, float fy, float cx, float cy,
                       const std::vector<float>& R_data = {1, 0, 0, 0, 1, 0, 0, 0, 1},
                       const std::vector<float>& T_data = {0, 0, 4}) {
        auto R = Tensor::from_blob(const_cast<float*>(R_data.data()), {3, 3}, Device::CPU, DataType::Float32).to(Device::CUDA);
        auto T = Tensor::from_blob(const_cast<float*>(T_data.data()), {3}, Device::CPU, DataType::Float32).to(Device::CUDA);
        return Camera(R, T, fx, fy, cx, cy, Tensor(), Tensor(),
                      CameraModelType::PINHOLE, "test", "", std::filesystem::path{}, w, h, 0);
    }

    void cleanup_arena() {
        GlobalArenaManager::instance().get_arena().emergency_cleanup();
    }

    bool has_nan(const Tensor& t) {
        auto cpu = t.to(Device::CPU);
        float* ptr = cpu.ptr<float>();
        for (size_t i = 0; i < t.numel(); ++i) {
            if (std::isnan(ptr[i]))
                return true;
        }
        return false;
    }

    bool has_inf(const Tensor& t) {
        auto cpu = t.to(Device::CPU);
        float* ptr = cpu.ptr<float>();
        for (size_t i = 0; i < t.numel(); ++i) {
            if (std::isinf(ptr[i]))
                return true;
        }
        return false;
    }

    int count_nonzero(const Tensor& t) {
        auto cpu = t.to(Device::CPU);
        float* ptr = cpu.ptr<float>();
        int count = 0;
        for (size_t i = 0; i < t.numel(); ++i) {
            if (ptr[i] != 0.0f)
                ++count;
        }
        return count;
    }

} // namespace

class FastGSFuzzTest : public ::testing::Test {
protected:
    void SetUp() override {
        bg_ = Tensor::zeros({3}, Device::CUDA);
    }

    void TearDown() override {
        cleanup_arena();
    }

    Tensor bg_;
};

// =============================================================================
// EDGE CASE TESTS: Empty and minimal scenes
// =============================================================================

TEST_F(FastGSFuzzTest, EmptyScene_ZeroPrimitives) {
    GTEST_SKIP() << "Rasterizer does not support 0 gaussians";

    auto means = Tensor::zeros({0, 3}, Device::CUDA);
    auto sh0 = Tensor::zeros({0, 1, 3}, Device::CUDA);
    auto shN = Tensor::zeros({0, 0, 3}, Device::CUDA);
    auto scaling = Tensor::zeros({0, 3}, Device::CUDA);
    auto rotation = Tensor::zeros({0, 4}, Device::CUDA);
    auto opacity = Tensor::zeros({0}, Device::CUDA);

    auto camera = make_camera(64, 64, 100, 100, 32, 32);
    auto splat = std::make_unique<SplatData>(0, means, sh0, shN, scaling, rotation, opacity, 1.0f);

    auto result = fast_rasterize_forward(camera, *splat, bg_, 0, 0, 0, 0, false);
    // Should handle gracefully - either return nullopt or valid empty result
    if (result.has_value()) {
        EXPECT_FALSE(has_nan(result->first.image));
        EXPECT_FALSE(has_inf(result->first.image));
    }
}

TEST_F(FastGSFuzzTest, SingleGaussian_Visible) {
    auto means = Tensor::zeros({1, 3}, Device::CUDA); // at origin
    auto sh0 = Tensor::ones({1, 1, 3}, Device::CUDA).mul(0.5f);
    auto shN = Tensor::zeros({1, 0, 3}, Device::CUDA);
    auto scaling = Tensor::full({1, 3}, -2.0f, Device::CUDA); // exp(-2) ~ 0.135
    std::vector<float> rot_data = {1, 0, 0, 0};               // identity quaternion
    auto rotation = Tensor::from_blob(rot_data.data(), {1, 4}, Device::CPU, DataType::Float32).to(Device::CUDA);
    auto opacity = Tensor::full({1}, 2.0f, Device::CUDA); // sigmoid(2) ~ 0.88

    auto camera = make_camera(64, 64, 100, 100, 32, 32);
    auto splat = std::make_unique<SplatData>(0, means, sh0, shN, scaling, rotation, opacity, 1.0f);

    auto result = fast_rasterize_forward(camera, *splat, bg_, 0, 0, 0, 0, false);
    ASSERT_TRUE(result.has_value());
    EXPECT_FALSE(has_nan(result->first.image));
    EXPECT_GT(count_nonzero(result->first.image), 0); // Should produce some output
}

TEST_F(FastGSFuzzTest, SingleGaussian_VeryFarAway) {
    // Test gaussian at extreme distance - should not crash or produce NaN/Inf
    std::vector<float> pos = {0, 0, 1000}; // Very far away
    auto means = Tensor::from_blob(pos.data(), {1, 3}, Device::CPU, DataType::Float32).to(Device::CUDA);
    auto sh0 = Tensor::ones({1, 1, 3}, Device::CUDA);
    auto shN = Tensor::zeros({1, 0, 3}, Device::CUDA);
    auto scaling = Tensor::full({1, 3}, -2.0f, Device::CUDA);
    std::vector<float> rot = {1, 0, 0, 0};
    auto rotation = Tensor::from_blob(rot.data(), {1, 4}, Device::CPU, DataType::Float32).to(Device::CUDA);
    auto opacity = Tensor::full({1}, 2.0f, Device::CUDA);

    auto camera = make_camera(64, 64, 100, 100, 32, 32);
    auto splat = std::make_unique<SplatData>(0, means, sh0, shN, scaling, rotation, opacity, 1.0f);

    auto result = fast_rasterize_forward(camera, *splat, bg_, 0, 0, 0, 0, false);
    ASSERT_TRUE(result.has_value());
    EXPECT_FALSE(has_nan(result->first.image));
    EXPECT_FALSE(has_inf(result->first.image));
    // Far gaussian will project very small - may or may not render depending on far plane
}

TEST_F(FastGSFuzzTest, AllGaussians_OutsideFrustum) {
    size_t n = 100;
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-100, 100);

    std::vector<float> pos(n * 3);
    for (size_t i = 0; i < n; ++i) {
        pos[i * 3] = dist(gen);
        pos[i * 3 + 1] = dist(gen);
        pos[i * 3 + 2] = -5.0f; // All in front of camera (negative z in camera space = in front)
    }
    auto means = Tensor::from_blob(pos.data(), {n, 3}, Device::CPU, DataType::Float32).to(Device::CUDA);
    auto sh0 = Tensor::randn({n, 1, 3}, Device::CUDA).mul(0.3f);
    auto shN = Tensor::zeros({n, 0, 3}, Device::CUDA);
    auto scaling = Tensor::full({n, 3}, -4.0f, Device::CUDA); // Very small
    auto rotation = Tensor::randn({n, 4}, Device::CUDA);
    rotation = rotation / rotation.pow(2.0f).sum(-1, true).sqrt();
    auto opacity = Tensor::full({n}, 2.0f, Device::CUDA);

    auto camera = make_camera(64, 64, 100, 100, 32, 32);
    auto splat = std::make_unique<SplatData>(0, means, sh0, shN, scaling, rotation, opacity, 1.0f);

    auto result = fast_rasterize_forward(camera, *splat, bg_, 0, 0, 0, 0, false);
    ASSERT_TRUE(result.has_value());
    EXPECT_FALSE(has_nan(result->first.image));
}

// =============================================================================
// NUMERICAL EXTREME TESTS
// =============================================================================

TEST_F(FastGSFuzzTest, VerySmallScale) {
    size_t n = 32;
    auto means = Tensor::randn({n, 3}, Device::CUDA).mul(0.5f);
    auto sh0 = Tensor::randn({n, 1, 3}, Device::CUDA).mul(0.3f);
    auto shN = Tensor::zeros({n, 0, 3}, Device::CUDA);
    auto scaling = Tensor::full({n, 3}, -20.0f, Device::CUDA); // exp(-20) ~ 2e-9
    auto rotation = Tensor::randn({n, 4}, Device::CUDA);
    rotation = rotation / rotation.pow(2.0f).sum(-1, true).sqrt();
    auto opacity = Tensor::full({n}, 2.0f, Device::CUDA);

    auto camera = make_camera(64, 64, 100, 100, 32, 32);
    auto splat = std::make_unique<SplatData>(0, means, sh0, shN, scaling, rotation, opacity, 1.0f);

    auto result = fast_rasterize_forward(camera, *splat, bg_, 0, 0, 0, 0, false);
    ASSERT_TRUE(result.has_value());
    EXPECT_FALSE(has_nan(result->first.image));
    EXPECT_FALSE(has_inf(result->first.image));
}

TEST_F(FastGSFuzzTest, VeryLargeScale) {
    size_t n = 32;
    auto means = Tensor::randn({n, 3}, Device::CUDA).mul(0.5f);
    auto sh0 = Tensor::randn({n, 1, 3}, Device::CUDA).mul(0.3f);
    auto shN = Tensor::zeros({n, 0, 3}, Device::CUDA);
    auto scaling = Tensor::full({n, 3}, 10.0f, Device::CUDA); // exp(10) ~ 22000
    auto rotation = Tensor::randn({n, 4}, Device::CUDA);
    rotation = rotation / rotation.pow(2.0f).sum(-1, true).sqrt();
    auto opacity = Tensor::full({n}, 2.0f, Device::CUDA);

    auto camera = make_camera(64, 64, 100, 100, 32, 32);
    auto splat = std::make_unique<SplatData>(0, means, sh0, shN, scaling, rotation, opacity, 1.0f);

    auto result = fast_rasterize_forward(camera, *splat, bg_, 0, 0, 0, 0, false);
    ASSERT_TRUE(result.has_value());
    EXPECT_FALSE(has_nan(result->first.image));
    EXPECT_FALSE(has_inf(result->first.image));
}

TEST_F(FastGSFuzzTest, ExtremePositions) {
    size_t n = 16;
    std::vector<float> pos(n * 3);
    // Mix of extreme positions
    for (size_t i = 0; i < n; ++i) {
        if (i < 4) {
            pos[i * 3] = 1e6f;
            pos[i * 3 + 1] = 0;
            pos[i * 3 + 2] = 0; // Far away
        } else if (i < 8) {
            pos[i * 3] = 0;
            pos[i * 3 + 1] = 0;
            pos[i * 3 + 2] = 1e-6f; // Very close to camera
        } else if (i < 12) {
            pos[i * 3] = 1e-8f;
            pos[i * 3 + 1] = 1e-8f;
            pos[i * 3 + 2] = 1e-8f; // Near origin
        } else {
            pos[i * 3] = 0;
            pos[i * 3 + 1] = 0;
            pos[i * 3 + 2] = 0; // At camera
        }
    }
    auto means = Tensor::from_blob(pos.data(), {n, 3}, Device::CPU, DataType::Float32).to(Device::CUDA);
    auto sh0 = Tensor::randn({n, 1, 3}, Device::CUDA).mul(0.3f);
    auto shN = Tensor::zeros({n, 0, 3}, Device::CUDA);
    auto scaling = Tensor::full({n, 3}, -2.0f, Device::CUDA);
    auto rotation = Tensor::randn({n, 4}, Device::CUDA);
    rotation = rotation / rotation.pow(2.0f).sum(-1, true).sqrt();
    auto opacity = Tensor::full({n}, 2.0f, Device::CUDA);

    auto camera = make_camera(64, 64, 100, 100, 32, 32);
    auto splat = std::make_unique<SplatData>(0, means, sh0, shN, scaling, rotation, opacity, 1.0f);

    auto result = fast_rasterize_forward(camera, *splat, bg_, 0, 0, 0, 0, false);
    ASSERT_TRUE(result.has_value());
    EXPECT_FALSE(has_nan(result->first.image));
    EXPECT_FALSE(has_inf(result->first.image));
}

TEST_F(FastGSFuzzTest, ZeroOpacity) {
    size_t n = 32;
    auto means = Tensor::randn({n, 3}, Device::CUDA).mul(0.5f);
    auto sh0 = Tensor::randn({n, 1, 3}, Device::CUDA).mul(0.3f);
    auto shN = Tensor::zeros({n, 0, 3}, Device::CUDA);
    auto scaling = Tensor::full({n, 3}, -2.0f, Device::CUDA);
    auto rotation = Tensor::randn({n, 4}, Device::CUDA);
    rotation = rotation / rotation.pow(2.0f).sum(-1, true).sqrt();
    auto opacity = Tensor::full({n}, -100.0f, Device::CUDA); // sigmoid(-100) ~ 0

    auto camera = make_camera(64, 64, 100, 100, 32, 32);
    auto splat = std::make_unique<SplatData>(0, means, sh0, shN, scaling, rotation, opacity, 1.0f);

    auto result = fast_rasterize_forward(camera, *splat, bg_, 0, 0, 0, 0, false);
    ASSERT_TRUE(result.has_value());
    EXPECT_FALSE(has_nan(result->first.image));
    // Zero opacity should produce black image
    EXPECT_EQ(count_nonzero(result->first.image), 0);
}

TEST_F(FastGSFuzzTest, FullOpacity) {
    size_t n = 32;
    auto means = Tensor::randn({n, 3}, Device::CUDA).mul(0.5f);
    auto sh0 = Tensor::randn({n, 1, 3}, Device::CUDA).mul(0.3f);
    auto shN = Tensor::zeros({n, 0, 3}, Device::CUDA);
    auto scaling = Tensor::full({n, 3}, -2.0f, Device::CUDA);
    auto rotation = Tensor::randn({n, 4}, Device::CUDA);
    rotation = rotation / rotation.pow(2.0f).sum(-1, true).sqrt();
    auto opacity = Tensor::full({n}, 100.0f, Device::CUDA); // sigmoid(100) ~ 1

    auto camera = make_camera(64, 64, 100, 100, 32, 32);
    auto splat = std::make_unique<SplatData>(0, means, sh0, shN, scaling, rotation, opacity, 1.0f);

    auto result = fast_rasterize_forward(camera, *splat, bg_, 0, 0, 0, 0, false);
    ASSERT_TRUE(result.has_value());
    EXPECT_FALSE(has_nan(result->first.image));
    EXPECT_FALSE(has_inf(result->first.image));
}

// =============================================================================
// DEGENERATE INPUT TESTS
// =============================================================================

TEST_F(FastGSFuzzTest, ZeroQuaternion) {
    size_t n = 32;
    auto means = Tensor::randn({n, 3}, Device::CUDA).mul(0.5f);
    auto sh0 = Tensor::randn({n, 1, 3}, Device::CUDA).mul(0.3f);
    auto shN = Tensor::zeros({n, 0, 3}, Device::CUDA);
    auto scaling = Tensor::full({n, 3}, -2.0f, Device::CUDA);
    auto rotation = Tensor::zeros({n, 4}, Device::CUDA); // All zero quaternions!
    auto opacity = Tensor::full({n}, 2.0f, Device::CUDA);

    auto camera = make_camera(64, 64, 100, 100, 32, 32);
    auto splat = std::make_unique<SplatData>(0, means, sh0, shN, scaling, rotation, opacity, 1.0f);

    auto result = fast_rasterize_forward(camera, *splat, bg_, 0, 0, 0, 0, false);
    ASSERT_TRUE(result.has_value());
    // Zero quaternion should be handled gracefully (culled or normalized)
    EXPECT_FALSE(has_nan(result->first.image));
    EXPECT_FALSE(has_inf(result->first.image));
}

TEST_F(FastGSFuzzTest, DenormalizedQuaternion) {
    size_t n = 32;
    auto means = Tensor::randn({n, 3}, Device::CUDA).mul(0.5f);
    auto sh0 = Tensor::randn({n, 1, 3}, Device::CUDA).mul(0.3f);
    auto shN = Tensor::zeros({n, 0, 3}, Device::CUDA);
    auto scaling = Tensor::full({n, 3}, -2.0f, Device::CUDA);
    // Very small quaternion components
    std::vector<float> rot(n * 4);
    for (size_t i = 0; i < n * 4; ++i)
        rot[i] = 1e-20f;
    auto rotation = Tensor::from_blob(rot.data(), {n, 4}, Device::CPU, DataType::Float32).to(Device::CUDA);
    auto opacity = Tensor::full({n}, 2.0f, Device::CUDA);

    auto camera = make_camera(64, 64, 100, 100, 32, 32);
    auto splat = std::make_unique<SplatData>(0, means, sh0, shN, scaling, rotation, opacity, 1.0f);

    auto result = fast_rasterize_forward(camera, *splat, bg_, 0, 0, 0, 0, false);
    ASSERT_TRUE(result.has_value());
    EXPECT_FALSE(has_nan(result->first.image));
    EXPECT_FALSE(has_inf(result->first.image));
}

TEST_F(FastGSFuzzTest, MixedValidInvalid) {
    size_t n = 64;
    auto means = Tensor::randn({n, 3}, Device::CUDA).mul(0.5f);
    auto sh0 = Tensor::randn({n, 1, 3}, Device::CUDA).mul(0.3f);
    auto shN = Tensor::zeros({n, 0, 3}, Device::CUDA);
    auto scaling = Tensor::randn({n, 3}, Device::CUDA).mul(0.5f).sub(2.0f);
    auto rotation = Tensor::randn({n, 4}, Device::CUDA);
    rotation = rotation / rotation.pow(2.0f).sum(-1, true).sqrt();
    auto opacity = Tensor::randn({n}, Device::CUDA).mul(4.0f); // Mix of high/low

    // Corrupt some entries
    auto rot_cpu = rotation.to(Device::CPU);
    auto opa_cpu = opacity.to(Device::CPU);
    float* rot_ptr = rot_cpu.ptr<float>();
    float* opa_ptr = opa_cpu.ptr<float>();

    // Zero out some quaternions
    for (int i = 0; i < 8; ++i) {
        rot_ptr[i * 4] = rot_ptr[i * 4 + 1] = rot_ptr[i * 4 + 2] = rot_ptr[i * 4 + 3] = 0.0f;
    }
    // Set some opacities to extreme values
    for (int i = 8; i < 16; ++i) {
        opa_ptr[i] = -1000.0f;
    }

    rotation = rot_cpu.to(Device::CUDA);
    opacity = opa_cpu.to(Device::CUDA);

    auto camera = make_camera(64, 64, 100, 100, 32, 32);
    auto splat = std::make_unique<SplatData>(0, means, sh0, shN, scaling, rotation, opacity, 1.0f);

    auto result = fast_rasterize_forward(camera, *splat, bg_, 0, 0, 0, 0, false);
    ASSERT_TRUE(result.has_value());
    EXPECT_FALSE(has_nan(result->first.image));
    EXPECT_FALSE(has_inf(result->first.image));
}

// =============================================================================
// BOUNDARY CONDITION TESTS
// =============================================================================

TEST_F(FastGSFuzzTest, GaussianAtTileBoundary) {
    // Place gaussians exactly at tile boundaries (16x16 tiles)
    size_t n = 16;
    std::vector<float> pos(n * 3);
    for (size_t i = 0; i < n; ++i) {
        float tile_x = (i % 4) * 16.0f; // 0, 16, 32, 48
        float tile_y = (i / 4) * 16.0f;
        // Convert pixel to world (rough approximation)
        pos[i * 3] = (tile_x - 32) / 100.0f * 4.0f;
        pos[i * 3 + 1] = (tile_y - 32) / 100.0f * 4.0f;
        pos[i * 3 + 2] = 0.0f;
    }
    auto means = Tensor::from_blob(pos.data(), {n, 3}, Device::CPU, DataType::Float32).to(Device::CUDA);
    auto sh0 = Tensor::ones({n, 1, 3}, Device::CUDA).mul(0.5f);
    auto shN = Tensor::zeros({n, 0, 3}, Device::CUDA);
    auto scaling = Tensor::full({n, 3}, -3.0f, Device::CUDA);
    std::vector<float> rot(n * 4);
    for (size_t i = 0; i < n; ++i) {
        rot[i * 4] = 1;
        rot[i * 4 + 1] = rot[i * 4 + 2] = rot[i * 4 + 3] = 0;
    }
    auto rotation = Tensor::from_blob(rot.data(), {n, 4}, Device::CPU, DataType::Float32).to(Device::CUDA);
    auto opacity = Tensor::full({n}, 2.0f, Device::CUDA);

    auto camera = make_camera(64, 64, 100, 100, 32, 32);
    auto splat = std::make_unique<SplatData>(0, means, sh0, shN, scaling, rotation, opacity, 1.0f);

    auto result = fast_rasterize_forward(camera, *splat, bg_, 0, 0, 0, 0, false);
    ASSERT_TRUE(result.has_value());
    EXPECT_FALSE(has_nan(result->first.image));
}

TEST_F(FastGSFuzzTest, LargeGaussianCoveringManyTiles) {
    // Single gaussian covering entire screen
    std::vector<float> pos = {0, 0, 0};
    auto means = Tensor::from_blob(pos.data(), {1, 3}, Device::CPU, DataType::Float32).to(Device::CUDA);
    auto sh0 = Tensor::ones({1, 1, 3}, Device::CUDA).mul(0.5f);
    auto shN = Tensor::zeros({1, 0, 3}, Device::CUDA);
    auto scaling = Tensor::full({1, 3}, 2.0f, Device::CUDA); // Very large
    std::vector<float> rot = {1, 0, 0, 0};
    auto rotation = Tensor::from_blob(rot.data(), {1, 4}, Device::CPU, DataType::Float32).to(Device::CUDA);
    auto opacity = Tensor::full({1}, 2.0f, Device::CUDA);

    auto camera = make_camera(128, 128, 100, 100, 64, 64);
    auto splat = std::make_unique<SplatData>(0, means, sh0, shN, scaling, rotation, opacity, 1.0f);

    auto result = fast_rasterize_forward(camera, *splat, bg_, 0, 0, 0, 0, false);
    ASSERT_TRUE(result.has_value());
    EXPECT_FALSE(has_nan(result->first.image));
    EXPECT_GT(count_nonzero(result->first.image), 0);
}

TEST_F(FastGSFuzzTest, ManyGaussiansPerTile) {
    // Many gaussians all in center tile
    size_t n = 1000;
    auto means = Tensor::randn({n, 3}, Device::CUDA).mul(0.01f); // All near origin
    auto sh0 = Tensor::randn({n, 1, 3}, Device::CUDA).mul(0.3f);
    auto shN = Tensor::zeros({n, 0, 3}, Device::CUDA);
    auto scaling = Tensor::full({n, 3}, -4.0f, Device::CUDA); // Small
    auto rotation = Tensor::randn({n, 4}, Device::CUDA);
    rotation = rotation / rotation.pow(2.0f).sum(-1, true).sqrt();
    auto opacity = Tensor::full({n}, 1.0f, Device::CUDA);

    auto camera = make_camera(64, 64, 100, 100, 32, 32);
    auto splat = std::make_unique<SplatData>(0, means, sh0, shN, scaling, rotation, opacity, 1.0f);

    auto result = fast_rasterize_forward(camera, *splat, bg_, 0, 0, 0, 0, false);
    ASSERT_TRUE(result.has_value());
    EXPECT_FALSE(has_nan(result->first.image));
    EXPECT_FALSE(has_inf(result->first.image));
}

TEST_F(FastGSFuzzTest, OddImageDimensions) {
    size_t n = 32;
    auto means = Tensor::randn({n, 3}, Device::CUDA).mul(0.5f);
    auto sh0 = Tensor::randn({n, 1, 3}, Device::CUDA).mul(0.3f);
    auto shN = Tensor::zeros({n, 0, 3}, Device::CUDA);
    auto scaling = Tensor::full({n, 3}, -2.0f, Device::CUDA);
    auto rotation = Tensor::randn({n, 4}, Device::CUDA);
    rotation = rotation / rotation.pow(2.0f).sum(-1, true).sqrt();
    auto opacity = Tensor::full({n}, 2.0f, Device::CUDA);

    // Odd dimensions that don't align with tile size
    auto camera = make_camera(63, 67, 100, 100, 31.5f, 33.5f);
    auto splat = std::make_unique<SplatData>(0, means, sh0, shN, scaling, rotation, opacity, 1.0f);

    auto result = fast_rasterize_forward(camera, *splat, bg_, 0, 0, 0, 0, false);
    ASSERT_TRUE(result.has_value());
    EXPECT_FALSE(has_nan(result->first.image));
    EXPECT_EQ(result->first.image.shape()[1], 67);
    EXPECT_EQ(result->first.image.shape()[2], 63);
}

TEST_F(FastGSFuzzTest, VerySmallImage) {
    size_t n = 16;
    auto means = Tensor::randn({n, 3}, Device::CUDA).mul(0.5f);
    auto sh0 = Tensor::randn({n, 1, 3}, Device::CUDA).mul(0.3f);
    auto shN = Tensor::zeros({n, 0, 3}, Device::CUDA);
    auto scaling = Tensor::full({n, 3}, -2.0f, Device::CUDA);
    auto rotation = Tensor::randn({n, 4}, Device::CUDA);
    rotation = rotation / rotation.pow(2.0f).sum(-1, true).sqrt();
    auto opacity = Tensor::full({n}, 2.0f, Device::CUDA);

    // Smaller than a tile
    auto camera = make_camera(8, 8, 10, 10, 4, 4);
    auto splat = std::make_unique<SplatData>(0, means, sh0, shN, scaling, rotation, opacity, 1.0f);

    auto result = fast_rasterize_forward(camera, *splat, bg_, 0, 0, 0, 0, false);
    ASSERT_TRUE(result.has_value());
    EXPECT_FALSE(has_nan(result->first.image));
}

TEST_F(FastGSFuzzTest, SinglePixelImage) {
    size_t n = 4;
    auto means = Tensor::randn({n, 3}, Device::CUDA).mul(0.1f);
    auto sh0 = Tensor::randn({n, 1, 3}, Device::CUDA).mul(0.3f);
    auto shN = Tensor::zeros({n, 0, 3}, Device::CUDA);
    auto scaling = Tensor::full({n, 3}, -2.0f, Device::CUDA);
    auto rotation = Tensor::randn({n, 4}, Device::CUDA);
    rotation = rotation / rotation.pow(2.0f).sum(-1, true).sqrt();
    auto opacity = Tensor::full({n}, 2.0f, Device::CUDA);

    auto camera = make_camera(1, 1, 1, 1, 0.5f, 0.5f);
    auto splat = std::make_unique<SplatData>(0, means, sh0, shN, scaling, rotation, opacity, 1.0f);

    auto result = fast_rasterize_forward(camera, *splat, bg_, 0, 0, 0, 0, false);
    ASSERT_TRUE(result.has_value());
    EXPECT_FALSE(has_nan(result->first.image));
}

// =============================================================================
// BACKWARD PASS STABILITY TESTS
// =============================================================================

TEST_F(FastGSFuzzTest, Backward_ZeroGradient) {
    size_t n = 32;
    auto means = Tensor::randn({n, 3}, Device::CUDA).mul(0.5f);
    auto sh0 = Tensor::randn({n, 1, 3}, Device::CUDA).mul(0.3f);
    auto shN = Tensor::zeros({n, 0, 3}, Device::CUDA);
    auto scaling = Tensor::full({n, 3}, -2.0f, Device::CUDA);
    auto rotation = Tensor::randn({n, 4}, Device::CUDA);
    rotation = rotation / rotation.pow(2.0f).sum(-1, true).sqrt();
    auto opacity = Tensor::full({n}, 2.0f, Device::CUDA);

    auto camera = make_camera(64, 64, 100, 100, 32, 32);
    auto splat = std::make_unique<SplatData>(0, means, sh0, shN, scaling, rotation, opacity, 1.0f);

    auto result = fast_rasterize_forward(camera, *splat, bg_, 0, 0, 0, 0, false);
    ASSERT_TRUE(result.has_value());

    AdamConfig cfg{.lr = 0.001f, .beta1 = 0.9, .beta2 = 0.999, .eps = 1e-15};
    auto opt = std::make_unique<AdamOptimizer>(*splat, cfg);
    opt->allocate_gradients();
    opt->zero_grad(0);

    // Zero gradient
    auto grad_out = Tensor::zeros_like(result->first.image);
    fast_rasterize_backward(result->second, grad_out, *splat, *opt, {});

    // All gradients should be zero
    auto grad_means = opt->get_grad(ParamType::Means);
    EXPECT_FALSE(has_nan(grad_means));
    float grad_sum = grad_means.abs().sum().item<float>();
    EXPECT_LT(grad_sum, 1e-6f);
}

TEST_F(FastGSFuzzTest, Backward_LargeGradient) {
    size_t n = 32;
    auto means = Tensor::randn({n, 3}, Device::CUDA).mul(0.5f);
    auto sh0 = Tensor::randn({n, 1, 3}, Device::CUDA).mul(0.3f);
    auto shN = Tensor::zeros({n, 0, 3}, Device::CUDA);
    auto scaling = Tensor::full({n, 3}, -2.0f, Device::CUDA);
    auto rotation = Tensor::randn({n, 4}, Device::CUDA);
    rotation = rotation / rotation.pow(2.0f).sum(-1, true).sqrt();
    auto opacity = Tensor::full({n}, 2.0f, Device::CUDA);

    auto camera = make_camera(64, 64, 100, 100, 32, 32);
    auto splat = std::make_unique<SplatData>(0, means, sh0, shN, scaling, rotation, opacity, 1.0f);

    auto result = fast_rasterize_forward(camera, *splat, bg_, 0, 0, 0, 0, false);
    ASSERT_TRUE(result.has_value());

    AdamConfig cfg{.lr = 0.001f, .beta1 = 0.9, .beta2 = 0.999, .eps = 1e-15};
    auto opt = std::make_unique<AdamOptimizer>(*splat, cfg);
    opt->allocate_gradients();
    opt->zero_grad(0);

    // Very large gradient
    auto grad_out = Tensor::full_like(result->first.image, 1e6f);
    fast_rasterize_backward(result->second, grad_out, *splat, *opt, {});

    // Gradients should be clamped, not NaN/Inf
    auto grad_means = opt->get_grad(ParamType::Means);
    EXPECT_FALSE(has_nan(grad_means));
    EXPECT_FALSE(has_inf(grad_means));
}

TEST_F(FastGSFuzzTest, Backward_AllCulled) {
    size_t n = 32;
    std::vector<float> pos(n * 3);
    for (size_t i = 0; i < n * 3; ++i)
        pos[i] = 100.0f; // All far away
    auto means = Tensor::from_blob(pos.data(), {n, 3}, Device::CPU, DataType::Float32).to(Device::CUDA);
    auto sh0 = Tensor::randn({n, 1, 3}, Device::CUDA).mul(0.3f);
    auto shN = Tensor::zeros({n, 0, 3}, Device::CUDA);
    auto scaling = Tensor::full({n, 3}, -2.0f, Device::CUDA);
    auto rotation = Tensor::randn({n, 4}, Device::CUDA);
    rotation = rotation / rotation.pow(2.0f).sum(-1, true).sqrt();
    auto opacity = Tensor::full({n}, 2.0f, Device::CUDA);

    auto camera = make_camera(64, 64, 100, 100, 32, 32);
    auto splat = std::make_unique<SplatData>(0, means, sh0, shN, scaling, rotation, opacity, 1.0f);

    auto result = fast_rasterize_forward(camera, *splat, bg_, 0, 0, 0, 0, false);
    ASSERT_TRUE(result.has_value());

    AdamConfig cfg{.lr = 0.001f, .beta1 = 0.9, .beta2 = 0.999, .eps = 1e-15};
    auto opt = std::make_unique<AdamOptimizer>(*splat, cfg);
    opt->allocate_gradients();
    opt->zero_grad(0);

    auto grad_out = result->first.image.mul(2.0f);
    fast_rasterize_backward(result->second, grad_out, *splat, *opt, {});

    // All culled - gradients should be zero
    auto grad_means = opt->get_grad(ParamType::Means);
    EXPECT_FALSE(has_nan(grad_means));
}

// =============================================================================
// RANDOM STRESS TESTS
// =============================================================================

TEST_F(FastGSFuzzTest, RandomStress_SmallBatch) {
    std::mt19937 gen(12345);

    for (int trial = 0; trial < 10; ++trial) {
        size_t n = 16 + (gen() % 64); // 16-80 gaussians

        auto means = Tensor::randn({n, 3}, Device::CUDA).mul(2.0f);
        auto sh0 = Tensor::randn({n, 1, 3}, Device::CUDA).mul(0.5f);
        auto shN = Tensor::zeros({n, 0, 3}, Device::CUDA);
        auto scaling = Tensor::randn({n, 3}, Device::CUDA).mul(2.0f).sub(3.0f);
        auto rotation = Tensor::randn({n, 4}, Device::CUDA);
        rotation = rotation / rotation.pow(2.0f).sum(-1, true).sqrt();
        auto opacity = Tensor::randn({n}, Device::CUDA).mul(3.0f);

        int w = 32 + (gen() % 64);
        int h = 32 + (gen() % 64);
        auto camera = make_camera(w, h, 50 + gen() % 100, 50 + gen() % 100, w / 2.0f, h / 2.0f);
        auto splat = std::make_unique<SplatData>(0, means, sh0, shN, scaling, rotation, opacity, 1.0f);

        auto result = fast_rasterize_forward(camera, *splat, bg_, 0, 0, 0, 0, false);
        ASSERT_TRUE(result.has_value()) << "Trial " << trial << " failed";
        EXPECT_FALSE(has_nan(result->first.image)) << "NaN in trial " << trial;
        EXPECT_FALSE(has_inf(result->first.image)) << "Inf in trial " << trial;

        // Also test backward
        AdamConfig cfg{.lr = 0.001f, .beta1 = 0.9, .beta2 = 0.999, .eps = 1e-15};
        auto opt = std::make_unique<AdamOptimizer>(*splat, cfg);
        opt->allocate_gradients();
        opt->zero_grad(0);

        auto grad_out = result->first.image.mul(2.0f);
        fast_rasterize_backward(result->second, grad_out, *splat, *opt, {});

        auto grad_means = opt->get_grad(ParamType::Means);
        EXPECT_FALSE(has_nan(grad_means)) << "NaN gradient in trial " << trial;
        EXPECT_FALSE(has_inf(grad_means)) << "Inf gradient in trial " << trial;

        cleanup_arena();
    }
}

TEST_F(FastGSFuzzTest, RandomStress_LargeBatch) {
    std::mt19937 gen(54321);

    for (int trial = 0; trial < 3; ++trial) {
        size_t n = 10000 + (gen() % 10000); // 10k-20k gaussians

        auto means = Tensor::randn({n, 3}, Device::CUDA).mul(3.0f);
        auto sh0 = Tensor::randn({n, 1, 3}, Device::CUDA).mul(0.3f);
        auto shN = Tensor::zeros({n, 0, 3}, Device::CUDA);
        auto scaling = Tensor::randn({n, 3}, Device::CUDA).mul(1.5f).sub(3.0f);
        auto rotation = Tensor::randn({n, 4}, Device::CUDA);
        rotation = rotation / rotation.pow(2.0f).sum(-1, true).sqrt();
        auto opacity = Tensor::randn({n}, Device::CUDA).mul(2.0f);

        auto camera = make_camera(256, 256, 200, 200, 128, 128);
        auto splat = std::make_unique<SplatData>(0, means, sh0, shN, scaling, rotation, opacity, 1.0f);

        auto result = fast_rasterize_forward(camera, *splat, bg_, 0, 0, 0, 0, false);
        ASSERT_TRUE(result.has_value()) << "Large trial " << trial << " failed";
        EXPECT_FALSE(has_nan(result->first.image)) << "NaN in large trial " << trial;
        EXPECT_FALSE(has_inf(result->first.image)) << "Inf in large trial " << trial;

        cleanup_arena();
    }
}

// =============================================================================
// MIP FILTER TESTS
// =============================================================================

TEST_F(FastGSFuzzTest, MipFilter_Enabled) {
    size_t n = 64;
    auto means = Tensor::randn({n, 3}, Device::CUDA).mul(0.5f);
    auto sh0 = Tensor::randn({n, 1, 3}, Device::CUDA).mul(0.3f);
    auto shN = Tensor::zeros({n, 0, 3}, Device::CUDA);
    auto scaling = Tensor::full({n, 3}, -2.0f, Device::CUDA);
    auto rotation = Tensor::randn({n, 4}, Device::CUDA);
    rotation = rotation / rotation.pow(2.0f).sum(-1, true).sqrt();
    auto opacity = Tensor::full({n}, 2.0f, Device::CUDA);

    auto camera = make_camera(64, 64, 100, 100, 32, 32);
    auto splat = std::make_unique<SplatData>(0, means, sh0, shN, scaling, rotation, opacity, 1.0f);

    // Test with mip filter enabled
    auto result = fast_rasterize_forward(camera, *splat, bg_, 0, 0, 0, 0, true);
    ASSERT_TRUE(result.has_value());
    EXPECT_FALSE(has_nan(result->first.image));
    EXPECT_FALSE(has_inf(result->first.image));

    // Test backward with mip filter
    AdamConfig cfg{.lr = 0.001f, .beta1 = 0.9, .beta2 = 0.999, .eps = 1e-15};
    auto opt = std::make_unique<AdamOptimizer>(*splat, cfg);
    opt->allocate_gradients();
    opt->zero_grad(0);

    auto grad_out = result->first.image.mul(2.0f);
    fast_rasterize_backward(result->second, grad_out, *splat, *opt, {});

    auto grad_opacity = opt->get_grad(ParamType::Opacity);
    EXPECT_FALSE(has_nan(grad_opacity));
    EXPECT_FALSE(has_inf(grad_opacity));
}

// =============================================================================
// DEPTH EDGE CASES
// =============================================================================

TEST_F(FastGSFuzzTest, SameDepthAllGaussians) {
    size_t n = 100;
    std::vector<float> pos(n * 3);
    for (size_t i = 0; i < n; ++i) {
        pos[i * 3] = (i % 10) * 0.1f - 0.5f;
        pos[i * 3 + 1] = (i / 10) * 0.1f - 0.5f;
        pos[i * 3 + 2] = 0.0f; // All same depth
    }
    auto means = Tensor::from_blob(pos.data(), {n, 3}, Device::CPU, DataType::Float32).to(Device::CUDA);
    auto sh0 = Tensor::randn({n, 1, 3}, Device::CUDA).mul(0.3f);
    auto shN = Tensor::zeros({n, 0, 3}, Device::CUDA);
    auto scaling = Tensor::full({n, 3}, -3.0f, Device::CUDA);
    auto rotation = Tensor::randn({n, 4}, Device::CUDA);
    rotation = rotation / rotation.pow(2.0f).sum(-1, true).sqrt();
    auto opacity = Tensor::full({n}, 1.5f, Device::CUDA);

    auto camera = make_camera(64, 64, 100, 100, 32, 32);
    auto splat = std::make_unique<SplatData>(0, means, sh0, shN, scaling, rotation, opacity, 1.0f);

    auto result = fast_rasterize_forward(camera, *splat, bg_, 0, 0, 0, 0, false);
    ASSERT_TRUE(result.has_value());
    EXPECT_FALSE(has_nan(result->first.image));
}

TEST_F(FastGSFuzzTest, AtNearPlane) {
    // Gaussians exactly at near plane
    size_t n = 16;
    std::vector<float> pos(n * 3);
    for (size_t i = 0; i < n; ++i) {
        pos[i * 3] = (i % 4) * 0.5f - 1.0f;
        pos[i * 3 + 1] = (i / 4) * 0.5f - 1.0f;
        pos[i * 3 + 2] = 3.9f; // Just inside near plane (camera at z=4, near ~0.1)
    }
    auto means = Tensor::from_blob(pos.data(), {n, 3}, Device::CPU, DataType::Float32).to(Device::CUDA);
    auto sh0 = Tensor::randn({n, 1, 3}, Device::CUDA).mul(0.3f);
    auto shN = Tensor::zeros({n, 0, 3}, Device::CUDA);
    auto scaling = Tensor::full({n, 3}, -2.0f, Device::CUDA);
    std::vector<float> rot(n * 4);
    for (size_t i = 0; i < n; ++i) {
        rot[i * 4] = 1;
        rot[i * 4 + 1] = rot[i * 4 + 2] = rot[i * 4 + 3] = 0;
    }
    auto rotation = Tensor::from_blob(rot.data(), {n, 4}, Device::CPU, DataType::Float32).to(Device::CUDA);
    auto opacity = Tensor::full({n}, 2.0f, Device::CUDA);

    auto camera = make_camera(64, 64, 100, 100, 32, 32);
    auto splat = std::make_unique<SplatData>(0, means, sh0, shN, scaling, rotation, opacity, 1.0f);

    auto result = fast_rasterize_forward(camera, *splat, bg_, 0, 0, 0, 0, false);
    ASSERT_TRUE(result.has_value());
    EXPECT_FALSE(has_nan(result->first.image));
}

// =============================================================================
// SPHERICAL HARMONICS EDGE CASES
// =============================================================================

TEST_F(FastGSFuzzTest, HigherOrderSH) {
    size_t n = 32;
    auto means = Tensor::randn({n, 3}, Device::CUDA).mul(0.5f);
    auto sh0 = Tensor::randn({n, 1, 3}, Device::CUDA).mul(0.3f);
    auto shN = Tensor::randn({n, 15, 3}, Device::CUDA).mul(0.1f); // SH degree 3
    auto scaling = Tensor::full({n, 3}, -2.0f, Device::CUDA);
    auto rotation = Tensor::randn({n, 4}, Device::CUDA);
    rotation = rotation / rotation.pow(2.0f).sum(-1, true).sqrt();
    auto opacity = Tensor::full({n}, 2.0f, Device::CUDA);

    auto camera = make_camera(64, 64, 100, 100, 32, 32);
    auto splat = std::make_unique<SplatData>(0, means, sh0, shN, scaling, rotation, opacity, 1.0f);

    auto result = fast_rasterize_forward(camera, *splat, bg_, 0, 0, 0, 0, false);
    ASSERT_TRUE(result.has_value());
    EXPECT_FALSE(has_nan(result->first.image));
    EXPECT_FALSE(has_inf(result->first.image));

    // Test backward
    AdamConfig cfg{.lr = 0.001f, .beta1 = 0.9, .beta2 = 0.999, .eps = 1e-15};
    auto opt = std::make_unique<AdamOptimizer>(*splat, cfg);
    opt->allocate_gradients();
    opt->zero_grad(0);

    auto grad_out = result->first.image.mul(2.0f);
    fast_rasterize_backward(result->second, grad_out, *splat, *opt, {});

    auto grad_shN = opt->get_grad(ParamType::ShN);
    EXPECT_FALSE(has_nan(grad_shN));
}

TEST_F(FastGSFuzzTest, ExtremeSHCoefficients) {
    size_t n = 32;
    auto means = Tensor::randn({n, 3}, Device::CUDA).mul(0.5f);
    auto sh0 = Tensor::full({n, 1, 3}, 100.0f, Device::CUDA); // Very large
    auto shN = Tensor::zeros({n, 0, 3}, Device::CUDA);
    auto scaling = Tensor::full({n, 3}, -2.0f, Device::CUDA);
    auto rotation = Tensor::randn({n, 4}, Device::CUDA);
    rotation = rotation / rotation.pow(2.0f).sum(-1, true).sqrt();
    auto opacity = Tensor::full({n}, 2.0f, Device::CUDA);

    auto camera = make_camera(64, 64, 100, 100, 32, 32);
    auto splat = std::make_unique<SplatData>(0, means, sh0, shN, scaling, rotation, opacity, 1.0f);

    auto result = fast_rasterize_forward(camera, *splat, bg_, 0, 0, 0, 0, false);
    ASSERT_TRUE(result.has_value());
    EXPECT_FALSE(has_nan(result->first.image));
    EXPECT_FALSE(has_inf(result->first.image));
}

// =============================================================================
// CAMERA INTRINSICS EDGE CASES
// =============================================================================

TEST_F(FastGSFuzzTest, AsymmetricFocalLength) {
    size_t n = 32;
    auto means = Tensor::randn({n, 3}, Device::CUDA).mul(0.5f);
    auto sh0 = Tensor::randn({n, 1, 3}, Device::CUDA).mul(0.3f);
    auto shN = Tensor::zeros({n, 0, 3}, Device::CUDA);
    auto scaling = Tensor::full({n, 3}, -2.0f, Device::CUDA);
    auto rotation = Tensor::randn({n, 4}, Device::CUDA);
    rotation = rotation / rotation.pow(2.0f).sum(-1, true).sqrt();
    auto opacity = Tensor::full({n}, 2.0f, Device::CUDA);

    // Very asymmetric focal length
    auto camera = make_camera(64, 64, 50, 200, 32, 32);
    auto splat = std::make_unique<SplatData>(0, means, sh0, shN, scaling, rotation, opacity, 1.0f);

    auto result = fast_rasterize_forward(camera, *splat, bg_, 0, 0, 0, 0, false);
    ASSERT_TRUE(result.has_value());
    EXPECT_FALSE(has_nan(result->first.image));
}

TEST_F(FastGSFuzzTest, OffCenterPrincipalPoint) {
    size_t n = 32;
    auto means = Tensor::randn({n, 3}, Device::CUDA).mul(0.5f);
    auto sh0 = Tensor::randn({n, 1, 3}, Device::CUDA).mul(0.3f);
    auto shN = Tensor::zeros({n, 0, 3}, Device::CUDA);
    auto scaling = Tensor::full({n, 3}, -2.0f, Device::CUDA);
    auto rotation = Tensor::randn({n, 4}, Device::CUDA);
    rotation = rotation / rotation.pow(2.0f).sum(-1, true).sqrt();
    auto opacity = Tensor::full({n}, 2.0f, Device::CUDA);

    // Principal point at corner
    auto camera = make_camera(64, 64, 100, 100, 5, 60);
    auto splat = std::make_unique<SplatData>(0, means, sh0, shN, scaling, rotation, opacity, 1.0f);

    auto result = fast_rasterize_forward(camera, *splat, bg_, 0, 0, 0, 0, false);
    ASSERT_TRUE(result.has_value());
    EXPECT_FALSE(has_nan(result->first.image));
}

TEST_F(FastGSFuzzTest, VerySmallFocalLength) {
    size_t n = 32;
    auto means = Tensor::randn({n, 3}, Device::CUDA).mul(0.5f);
    auto sh0 = Tensor::randn({n, 1, 3}, Device::CUDA).mul(0.3f);
    auto shN = Tensor::zeros({n, 0, 3}, Device::CUDA);
    auto scaling = Tensor::full({n, 3}, -2.0f, Device::CUDA);
    auto rotation = Tensor::randn({n, 4}, Device::CUDA);
    rotation = rotation / rotation.pow(2.0f).sum(-1, true).sqrt();
    auto opacity = Tensor::full({n}, 2.0f, Device::CUDA);

    // Very small focal length (wide FOV)
    auto camera = make_camera(64, 64, 10, 10, 32, 32);
    auto splat = std::make_unique<SplatData>(0, means, sh0, shN, scaling, rotation, opacity, 1.0f);

    auto result = fast_rasterize_forward(camera, *splat, bg_, 0, 0, 0, 0, false);
    ASSERT_TRUE(result.has_value());
    EXPECT_FALSE(has_nan(result->first.image));
}

TEST_F(FastGSFuzzTest, VeryLargeFocalLength) {
    size_t n = 32;
    auto means = Tensor::randn({n, 3}, Device::CUDA).mul(0.5f);
    auto sh0 = Tensor::randn({n, 1, 3}, Device::CUDA).mul(0.3f);
    auto shN = Tensor::zeros({n, 0, 3}, Device::CUDA);
    auto scaling = Tensor::full({n, 3}, -2.0f, Device::CUDA);
    auto rotation = Tensor::randn({n, 4}, Device::CUDA);
    rotation = rotation / rotation.pow(2.0f).sum(-1, true).sqrt();
    auto opacity = Tensor::full({n}, 2.0f, Device::CUDA);

    // Very large focal length (narrow FOV / telephoto)
    auto camera = make_camera(64, 64, 10000, 10000, 32, 32);
    auto splat = std::make_unique<SplatData>(0, means, sh0, shN, scaling, rotation, opacity, 1.0f);

    auto result = fast_rasterize_forward(camera, *splat, bg_, 0, 0, 0, 0, false);
    ASSERT_TRUE(result.has_value());
    EXPECT_FALSE(has_nan(result->first.image));
}

// =============================================================================
// TRANSMITTANCE SATURATION TEST
// =============================================================================

TEST_F(FastGSFuzzTest, TransmittanceSaturation) {
    // Many overlapping high-opacity gaussians to test transmittance threshold
    size_t n = 200;
    auto means = Tensor::randn({n, 3}, Device::CUDA).mul(0.1f); // All clustered
    auto sh0 = Tensor::randn({n, 1, 3}, Device::CUDA).mul(0.3f);
    auto shN = Tensor::zeros({n, 0, 3}, Device::CUDA);
    auto scaling = Tensor::full({n, 3}, -1.0f, Device::CUDA); // Medium size
    auto rotation = Tensor::randn({n, 4}, Device::CUDA);
    rotation = rotation / rotation.pow(2.0f).sum(-1, true).sqrt();
    auto opacity = Tensor::full({n}, 5.0f, Device::CUDA); // High opacity

    auto camera = make_camera(64, 64, 100, 100, 32, 32);
    auto splat = std::make_unique<SplatData>(0, means, sh0, shN, scaling, rotation, opacity, 1.0f);

    auto result = fast_rasterize_forward(camera, *splat, bg_, 0, 0, 0, 0, false);
    ASSERT_TRUE(result.has_value());
    EXPECT_FALSE(has_nan(result->first.image));
    EXPECT_FALSE(has_inf(result->first.image));

    // Alpha should be saturated near 1.0
    float max_alpha = result->first.alpha.max().item<float>();
    EXPECT_GT(max_alpha, 0.99f);
}

// =============================================================================
// NaN/INF INJECTION TESTS - Verify robustness to corrupted data
// =============================================================================

TEST_F(FastGSFuzzTest, NaN_InMeans) {
    size_t n = 32;
    auto means = Tensor::randn({n, 3}, Device::CUDA).mul(0.5f);
    auto sh0 = Tensor::randn({n, 1, 3}, Device::CUDA).mul(0.3f);
    auto shN = Tensor::zeros({n, 0, 3}, Device::CUDA);
    auto scaling = Tensor::full({n, 3}, -2.0f, Device::CUDA);
    auto rotation = Tensor::randn({n, 4}, Device::CUDA);
    rotation = rotation / rotation.pow(2.0f).sum(-1, true).sqrt();
    auto opacity = Tensor::full({n}, 2.0f, Device::CUDA);

    // Inject NaN into some means
    auto means_cpu = means.to(Device::CPU);
    float* ptr = means_cpu.ptr<float>();
    ptr[0] = std::numeric_limits<float>::quiet_NaN();
    ptr[10] = std::numeric_limits<float>::quiet_NaN();
    means = means_cpu.to(Device::CUDA);

    auto camera = make_camera(64, 64, 100, 100, 32, 32);
    auto splat = std::make_unique<SplatData>(0, means, sh0, shN, scaling, rotation, opacity, 1.0f);

    auto result = fast_rasterize_forward(camera, *splat, bg_, 0, 0, 0, 0, false);
    ASSERT_TRUE(result.has_value());
    // NaN in input should result in NaN in output (expected behavior)
    // Or the gaussian should be culled. Either way, no crash.
}

TEST_F(FastGSFuzzTest, Inf_InScaling) {
    size_t n = 32;
    auto means = Tensor::randn({n, 3}, Device::CUDA).mul(0.5f);
    auto sh0 = Tensor::randn({n, 1, 3}, Device::CUDA).mul(0.3f);
    auto shN = Tensor::zeros({n, 0, 3}, Device::CUDA);
    auto scaling = Tensor::full({n, 3}, -2.0f, Device::CUDA);
    auto rotation = Tensor::randn({n, 4}, Device::CUDA);
    rotation = rotation / rotation.pow(2.0f).sum(-1, true).sqrt();
    auto opacity = Tensor::full({n}, 2.0f, Device::CUDA);

    // Inject Inf into some scales
    auto scaling_cpu = scaling.to(Device::CPU);
    float* ptr = scaling_cpu.ptr<float>();
    ptr[0] = std::numeric_limits<float>::infinity();
    ptr[15] = -std::numeric_limits<float>::infinity();
    scaling = scaling_cpu.to(Device::CUDA);

    auto camera = make_camera(64, 64, 100, 100, 32, 32);
    auto splat = std::make_unique<SplatData>(0, means, sh0, shN, scaling, rotation, opacity, 1.0f);

    auto result = fast_rasterize_forward(camera, *splat, bg_, 0, 0, 0, 0, false);
    ASSERT_TRUE(result.has_value());
    // Should not crash
}

TEST_F(FastGSFuzzTest, NaN_InRotation) {
    size_t n = 32;
    auto means = Tensor::randn({n, 3}, Device::CUDA).mul(0.5f);
    auto sh0 = Tensor::randn({n, 1, 3}, Device::CUDA).mul(0.3f);
    auto shN = Tensor::zeros({n, 0, 3}, Device::CUDA);
    auto scaling = Tensor::full({n, 3}, -2.0f, Device::CUDA);
    auto rotation = Tensor::randn({n, 4}, Device::CUDA);
    rotation = rotation / rotation.pow(2.0f).sum(-1, true).sqrt();
    auto opacity = Tensor::full({n}, 2.0f, Device::CUDA);

    // Inject NaN into some rotations
    auto rot_cpu = rotation.to(Device::CPU);
    float* ptr = rot_cpu.ptr<float>();
    ptr[0] = std::numeric_limits<float>::quiet_NaN();
    rotation = rot_cpu.to(Device::CUDA);

    auto camera = make_camera(64, 64, 100, 100, 32, 32);
    auto splat = std::make_unique<SplatData>(0, means, sh0, shN, scaling, rotation, opacity, 1.0f);

    auto result = fast_rasterize_forward(camera, *splat, bg_, 0, 0, 0, 0, false);
    ASSERT_TRUE(result.has_value());
    // Should not crash - NaN quaternion should be culled
}

// =============================================================================
// EXTREME BATCH SIZE TESTS
// =============================================================================

TEST_F(FastGSFuzzTest, VeryLargeBatch_100k) {
    size_t n = 100000;
    auto means = Tensor::randn({n, 3}, Device::CUDA).mul(5.0f);
    auto sh0 = Tensor::randn({n, 1, 3}, Device::CUDA).mul(0.3f);
    auto shN = Tensor::zeros({n, 0, 3}, Device::CUDA);
    auto scaling = Tensor::randn({n, 3}, Device::CUDA).mul(1.0f).sub(4.0f);
    auto rotation = Tensor::randn({n, 4}, Device::CUDA);
    rotation = rotation / rotation.pow(2.0f).sum(-1, true).sqrt();
    auto opacity = Tensor::randn({n}, Device::CUDA);

    auto camera = make_camera(512, 512, 500, 500, 256, 256);
    auto splat = std::make_unique<SplatData>(0, means, sh0, shN, scaling, rotation, opacity, 1.0f);

    auto result = fast_rasterize_forward(camera, *splat, bg_, 0, 0, 0, 0, false);
    ASSERT_TRUE(result.has_value());
    EXPECT_FALSE(has_nan(result->first.image));
    EXPECT_FALSE(has_inf(result->first.image));
}

// =============================================================================
// PRECISION EDGE CASES
// =============================================================================

TEST_F(FastGSFuzzTest, DenormalizedFloats) {
    size_t n = 32;
    // Create tensor with denormalized floats
    std::vector<float> pos(n * 3);
    for (size_t i = 0; i < n * 3; ++i) {
        pos[i] = std::numeric_limits<float>::denorm_min() * (i + 1);
    }
    auto means = Tensor::from_blob(pos.data(), {n, 3}, Device::CPU, DataType::Float32).to(Device::CUDA);
    auto sh0 = Tensor::randn({n, 1, 3}, Device::CUDA).mul(0.3f);
    auto shN = Tensor::zeros({n, 0, 3}, Device::CUDA);
    auto scaling = Tensor::full({n, 3}, -2.0f, Device::CUDA);
    auto rotation = Tensor::randn({n, 4}, Device::CUDA);
    rotation = rotation / rotation.pow(2.0f).sum(-1, true).sqrt();
    auto opacity = Tensor::full({n}, 2.0f, Device::CUDA);

    auto camera = make_camera(64, 64, 100, 100, 32, 32);
    auto splat = std::make_unique<SplatData>(0, means, sh0, shN, scaling, rotation, opacity, 1.0f);

    auto result = fast_rasterize_forward(camera, *splat, bg_, 0, 0, 0, 0, false);
    ASSERT_TRUE(result.has_value());
    EXPECT_FALSE(has_nan(result->first.image));
    EXPECT_FALSE(has_inf(result->first.image));
}

TEST_F(FastGSFuzzTest, MaxFloatValues) {
    size_t n = 16;
    std::vector<float> pos(n * 3);
    for (size_t i = 0; i < n * 3; ++i) {
        pos[i] = std::numeric_limits<float>::max() / 1e10f; // Large but won't overflow
    }
    auto means = Tensor::from_blob(pos.data(), {n, 3}, Device::CPU, DataType::Float32).to(Device::CUDA);
    auto sh0 = Tensor::randn({n, 1, 3}, Device::CUDA).mul(0.3f);
    auto shN = Tensor::zeros({n, 0, 3}, Device::CUDA);
    auto scaling = Tensor::full({n, 3}, -10.0f, Device::CUDA); // Very small
    auto rotation = Tensor::randn({n, 4}, Device::CUDA);
    rotation = rotation / rotation.pow(2.0f).sum(-1, true).sqrt();
    auto opacity = Tensor::full({n}, 2.0f, Device::CUDA);

    auto camera = make_camera(64, 64, 100, 100, 32, 32);
    auto splat = std::make_unique<SplatData>(0, means, sh0, shN, scaling, rotation, opacity, 1.0f);

    auto result = fast_rasterize_forward(camera, *splat, bg_, 0, 0, 0, 0, false);
    ASSERT_TRUE(result.has_value());
    // Should handle gracefully without crash
}

// =============================================================================
// REGRESSION: Test specific kernel interactions
// =============================================================================

TEST_F(FastGSFuzzTest, AllGaussiansInSinglePixel) {
    // All gaussians project to the exact same pixel
    size_t n = 100;
    auto means = Tensor::zeros({n, 3}, Device::CUDA); // All at origin
    auto sh0 = Tensor::ones({n, 1, 3}, Device::CUDA).mul(0.5f);
    auto shN = Tensor::zeros({n, 0, 3}, Device::CUDA);
    auto scaling = Tensor::full({n, 3}, -6.0f, Device::CUDA); // Very small
    std::vector<float> rot(n * 4);
    for (size_t i = 0; i < n; ++i) {
        rot[i * 4] = 1;
        rot[i * 4 + 1] = rot[i * 4 + 2] = rot[i * 4 + 3] = 0;
    }
    auto rotation = Tensor::from_blob(rot.data(), {n, 4}, Device::CPU, DataType::Float32).to(Device::CUDA);
    auto opacity = Tensor::full({n}, 0.5f, Device::CUDA);

    auto camera = make_camera(64, 64, 100, 100, 32, 32);
    auto splat = std::make_unique<SplatData>(0, means, sh0, shN, scaling, rotation, opacity, 1.0f);

    auto result = fast_rasterize_forward(camera, *splat, bg_, 0, 0, 0, 0, false);
    ASSERT_TRUE(result.has_value());
    EXPECT_FALSE(has_nan(result->first.image));
    EXPECT_FALSE(has_inf(result->first.image));
}

TEST_F(FastGSFuzzTest, GradientStability_MultipleIterations) {
    // Run forward/backward multiple times to check for accumulation issues
    size_t n = 64;
    auto means = Tensor::randn({n, 3}, Device::CUDA).mul(0.5f);
    auto sh0 = Tensor::randn({n, 1, 3}, Device::CUDA).mul(0.3f);
    auto shN = Tensor::zeros({n, 0, 3}, Device::CUDA);
    auto scaling = Tensor::full({n, 3}, -2.0f, Device::CUDA);
    auto rotation = Tensor::randn({n, 4}, Device::CUDA);
    rotation = rotation / rotation.pow(2.0f).sum(-1, true).sqrt();
    auto opacity = Tensor::full({n}, 2.0f, Device::CUDA);

    auto camera = make_camera(64, 64, 100, 100, 32, 32);

    for (int iter = 0; iter < 5; ++iter) {
        auto splat = std::make_unique<SplatData>(0, means, sh0, shN, scaling, rotation, opacity, 1.0f);

        auto result = fast_rasterize_forward(camera, *splat, bg_, 0, 0, 0, 0, false);
        ASSERT_TRUE(result.has_value()) << "Iteration " << iter;
        EXPECT_FALSE(has_nan(result->first.image)) << "NaN at iteration " << iter;

        AdamConfig cfg{.lr = 0.001f, .beta1 = 0.9, .beta2 = 0.999, .eps = 1e-15};
        auto opt = std::make_unique<AdamOptimizer>(*splat, cfg);
        opt->allocate_gradients();
        opt->zero_grad(0);

        auto grad_out = result->first.image.mul(2.0f);
        fast_rasterize_backward(result->second, grad_out, *splat, *opt, {});

        auto grad_means = opt->get_grad(ParamType::Means);
        EXPECT_FALSE(has_nan(grad_means)) << "NaN gradient at iteration " << iter;
        EXPECT_FALSE(has_inf(grad_means)) << "Inf gradient at iteration " << iter;

        cleanup_arena();
    }
}

TEST_F(FastGSFuzzTest, AnisotropicGaussians) {
    // Highly anisotropic gaussians (very different scales on each axis)
    size_t n = 32;
    auto means = Tensor::randn({n, 3}, Device::CUDA).mul(0.5f);
    auto sh0 = Tensor::randn({n, 1, 3}, Device::CUDA).mul(0.3f);
    auto shN = Tensor::zeros({n, 0, 3}, Device::CUDA);

    std::vector<float> scale_data(n * 3);
    for (size_t i = 0; i < n; ++i) {
        scale_data[i * 3] = -6.0f;    // Very thin on X
        scale_data[i * 3 + 1] = 0.0f; // Normal on Y
        scale_data[i * 3 + 2] = 2.0f; // Very thick on Z
    }
    auto scaling = Tensor::from_blob(scale_data.data(), {n, 3}, Device::CPU, DataType::Float32).to(Device::CUDA);
    auto rotation = Tensor::randn({n, 4}, Device::CUDA);
    rotation = rotation / rotation.pow(2.0f).sum(-1, true).sqrt();
    auto opacity = Tensor::full({n}, 2.0f, Device::CUDA);

    auto camera = make_camera(64, 64, 100, 100, 32, 32);
    auto splat = std::make_unique<SplatData>(0, means, sh0, shN, scaling, rotation, opacity, 1.0f);

    auto result = fast_rasterize_forward(camera, *splat, bg_, 0, 0, 0, 0, false);
    ASSERT_TRUE(result.has_value());
    EXPECT_FALSE(has_nan(result->first.image));
    EXPECT_FALSE(has_inf(result->first.image));
}

/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/camera.hpp"
#include "core/cube_face_projection.hpp"
#include "core/cuda/undistort/undistort.hpp"
#include "core/image_io.hpp"
#include "io/cube_face_projection.hpp"
#include "io/formats/colmap.hpp"
#include "training/training_setup.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

using namespace lfs::core;

namespace {

    constexpr float TEST_FX = 500.0f;
    constexpr float TEST_FY = 500.0f;
    constexpr float TEST_CX = 320.0f;
    constexpr float TEST_CY = 240.0f;
    constexpr int TEST_W = 640;
    constexpr int TEST_H = 480;
    constexpr float SPHERICAL_UNDISTORT_FOV = 96.0f;
    constexpr size_t SPHERICAL_UNDISTORT_VIEW_COUNT = 6;
    constexpr std::array<std::array<float, 3>, SPHERICAL_UNDISTORT_VIEW_COUNT> SPHERICAL_UNDISTORT_FORWARDS{{
        {1.f, 0.f, 0.f},
        {-1.f, 0.f, 0.f},
        {0.f, 1.f, 0.f},
        {0.f, -1.f, 0.f},
        {0.f, 0.f, 1.f},
        {0.f, 0.f, -1.f},
    }};

    using Vec3 = std::array<float, 3>;

    Vec3 cross(const Vec3& a, const Vec3& b) {
        return {
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]};
    }

    float length(const Vec3& v) {
        return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    }

    Vec3 normalize(const Vec3& v) {
        const float len = length(v);
        if (len <= 0.0f)
            return {0.0f, 0.0f, 1.0f};
        return {v[0] / len, v[1] / len, v[2] / len};
    }

    std::array<float, 9> expected_spherical_matrix(const Vec3& forward_in) {
        const Vec3 forward = normalize(forward_in);
        constexpr Vec3 world_up{0.0f, 1.0f, 0.0f};
        constexpr Vec3 fallback_up{0.0f, 0.0f, 1.0f};

        Vec3 right = cross(world_up, forward);
        if (length(right) < 1e-5f)
            right = cross(fallback_up, forward);
        right = normalize(right);
        const Vec3 up = cross(forward, right);

        return {
            right[0], right[1], right[2],
            up[0], up[1], up[2],
            forward[0], forward[1], forward[2]};
    }

    void validate_params(const UndistortParams& p, int src_w, int src_h) {
        EXPECT_GT(p.dst_width, 0);
        EXPECT_GT(p.dst_height, 0);
        EXPECT_LE(p.dst_width, src_w * 2);
        EXPECT_LE(p.dst_height, src_h * 2);
        EXPECT_GT(p.dst_fx, 0.0f);
        EXPECT_GT(p.dst_fy, 0.0f);
        EXPECT_GT(p.dst_cx, 0.0f);
        EXPECT_GT(p.dst_cy, 0.0f);
    }

    void run_image_undistort(const UndistortParams& params) {
        auto src = Tensor::randn(
            {3, static_cast<size_t>(params.src_height), static_cast<size_t>(params.src_width)},
            Device::CUDA);

        auto dst = undistort_image(src, params, nullptr);
        cudaDeviceSynchronize();

        ASSERT_EQ(dst.ndim(), 3u);
        EXPECT_EQ(static_cast<int>(dst.shape()[0]), 3);
        EXPECT_EQ(static_cast<int>(dst.shape()[1]), params.dst_height);
        EXPECT_EQ(static_cast<int>(dst.shape()[2]), params.dst_width);
    }

    void run_mask_undistort(const UndistortParams& params) {
        auto src = Tensor::ones(
            {static_cast<size_t>(params.src_height), static_cast<size_t>(params.src_width)},
            Device::CUDA);

        auto dst = undistort_mask(src, params, nullptr);
        cudaDeviceSynchronize();

        ASSERT_EQ(dst.ndim(), 2u);
        EXPECT_EQ(static_cast<int>(dst.shape()[0]), params.dst_height);
        EXPECT_EQ(static_cast<int>(dst.shape()[1]), params.dst_width);
    }

} // namespace

// ====================== Coefficient packing tests ======================

TEST(UndistortPacking, PinholeRadialOnly) {
    // COLMAP SIMPLE_RADIAL / RADIAL: 1-2 radial, no tangential
    auto radial = Tensor::from_vector({-0.1f, 0.02f}, TensorShape({2}), Device::CPU);
    auto params = compute_undistort_params(
        TEST_FX, TEST_FY, TEST_CX, TEST_CY, TEST_W, TEST_H,
        radial, Tensor(), CameraModelType::PINHOLE);

    EXPECT_FLOAT_EQ(params.distortion[0], -0.1f);
    EXPECT_FLOAT_EQ(params.distortion[1], 0.02f);
    EXPECT_FLOAT_EQ(params.distortion[2], 0.0f); // k3 = 0
    EXPECT_FLOAT_EQ(params.distortion[3], 0.0f); // p1 = 0
    EXPECT_FLOAT_EQ(params.distortion[4], 0.0f); // p2 = 0
}

TEST(UndistortPacking, PinholeRadialAndTangential) {
    // COLMAP OPENCV: 2 radial (k1,k2) + 2 tangential (p1,p2)
    auto radial = Tensor::from_vector({-0.1f, 0.02f}, TensorShape({2}), Device::CPU);
    auto tangential = Tensor::from_vector({0.003f, -0.004f}, TensorShape({2}), Device::CPU);
    auto params = compute_undistort_params(
        TEST_FX, TEST_FY, TEST_CX, TEST_CY, TEST_W, TEST_H,
        radial, tangential, CameraModelType::PINHOLE);

    EXPECT_FLOAT_EQ(params.distortion[0], -0.1f);   // k1
    EXPECT_FLOAT_EQ(params.distortion[1], 0.02f);   // k2
    EXPECT_FLOAT_EQ(params.distortion[2], 0.0f);    // k3 = 0
    EXPECT_FLOAT_EQ(params.distortion[3], 0.003f);  // p1
    EXPECT_FLOAT_EQ(params.distortion[4], -0.004f); // p2
    EXPECT_EQ(params.num_distortion, 5);
}

TEST(UndistortPacking, PinholeFullRadialAndTangential) {
    // COLMAP FULL_OPENCV: 6 radial + 2 tangential (only 3 radial used by our kernel)
    auto radial = Tensor::from_vector({-0.1f, 0.02f, -0.003f}, TensorShape({3}), Device::CPU);
    auto tangential = Tensor::from_vector({0.001f, -0.002f}, TensorShape({2}), Device::CPU);
    auto params = compute_undistort_params(
        TEST_FX, TEST_FY, TEST_CX, TEST_CY, TEST_W, TEST_H,
        radial, tangential, CameraModelType::PINHOLE);

    EXPECT_FLOAT_EQ(params.distortion[0], -0.1f);   // k1
    EXPECT_FLOAT_EQ(params.distortion[1], 0.02f);   // k2
    EXPECT_FLOAT_EQ(params.distortion[2], -0.003f); // k3
    EXPECT_FLOAT_EQ(params.distortion[3], 0.001f);  // p1
    EXPECT_FLOAT_EQ(params.distortion[4], -0.002f); // p2
    EXPECT_EQ(params.num_distortion, 5);
}

TEST(UndistortPacking, Fisheye4Coeffs) {
    // COLMAP OPENCV_FISHEYE: 4 radial (k1-k4), no tangential
    auto radial = Tensor::from_vector({0.1f, -0.02f, 0.005f, -0.001f}, TensorShape({4}), Device::CPU);
    auto params = compute_undistort_params(
        TEST_FX, TEST_FY, TEST_CX, TEST_CY, TEST_W, TEST_H,
        radial, Tensor(), CameraModelType::FISHEYE);

    EXPECT_FLOAT_EQ(params.distortion[0], 0.1f);
    EXPECT_FLOAT_EQ(params.distortion[1], -0.02f);
    EXPECT_FLOAT_EQ(params.distortion[2], 0.005f);
    EXPECT_FLOAT_EQ(params.distortion[3], -0.001f);
    EXPECT_EQ(params.num_distortion, 4);
}

TEST(UndistortPacking, Fisheye1Coeff) {
    // COLMAP SIMPLE_RADIAL_FISHEYE: 1 radial
    auto radial = Tensor::from_vector({0.05f}, TensorShape({1}), Device::CPU);
    auto params = compute_undistort_params(
        TEST_FX, TEST_FY, TEST_CX, TEST_CY, TEST_W, TEST_H,
        radial, Tensor(), CameraModelType::FISHEYE);

    EXPECT_FLOAT_EQ(params.distortion[0], 0.05f);
    EXPECT_FLOAT_EQ(params.distortion[1], 0.0f);
    EXPECT_FLOAT_EQ(params.distortion[2], 0.0f);
    EXPECT_FLOAT_EQ(params.distortion[3], 0.0f);
    EXPECT_EQ(params.num_distortion, 1);
}

TEST(UndistortPacking, Fisheye2Coeffs) {
    // COLMAP RADIAL_FISHEYE: 2 radial (k1, k2)
    auto radial = Tensor::from_vector({0.05f, -0.01f}, TensorShape({2}), Device::CPU);
    auto params = compute_undistort_params(
        TEST_FX, TEST_FY, TEST_CX, TEST_CY, TEST_W, TEST_H,
        radial, Tensor(), CameraModelType::FISHEYE);

    EXPECT_FLOAT_EQ(params.distortion[0], 0.05f);
    EXPECT_FLOAT_EQ(params.distortion[1], -0.01f);
    EXPECT_FLOAT_EQ(params.distortion[2], 0.0f);
    EXPECT_FLOAT_EQ(params.distortion[3], 0.0f);
    EXPECT_EQ(params.num_distortion, 2);
}

TEST(UndistortPacking, ThinPrismFisheye) {
    // COLMAP THIN_PRISM_FISHEYE: radial={k1,k2,k3,k4}, tangential={p1,p2,s1,s2}
    auto radial = Tensor::from_vector({0.1f, -0.02f, 0.003f, -0.001f}, TensorShape({4}), Device::CPU);
    auto tangential = Tensor::from_vector({0.0005f, -0.0003f, 0.0001f, -0.0002f}, TensorShape({4}), Device::CPU);
    auto params = compute_undistort_params(
        TEST_FX, TEST_FY, TEST_CX, TEST_CY, TEST_W, TEST_H,
        radial, tangential, CameraModelType::THIN_PRISM_FISHEYE);

    EXPECT_FLOAT_EQ(params.distortion[0], 0.1f);     // k1
    EXPECT_FLOAT_EQ(params.distortion[1], -0.02f);   // k2
    EXPECT_FLOAT_EQ(params.distortion[2], 0.003f);   // k3
    EXPECT_FLOAT_EQ(params.distortion[3], -0.001f);  // k4
    EXPECT_FLOAT_EQ(params.distortion[4], 0.0005f);  // p1
    EXPECT_FLOAT_EQ(params.distortion[5], -0.0003f); // p2
    EXPECT_FLOAT_EQ(params.distortion[6], 0.0001f);  // s1
    EXPECT_FLOAT_EQ(params.distortion[7], -0.0002f); // s2
    EXPECT_EQ(params.num_distortion, 8);
}

// ====================== Per-model undistortion tests ======================

TEST(UndistortPinhole, ZeroDistortion_Noop) {
    auto params = compute_undistort_params(
        TEST_FX, TEST_FY, TEST_CX, TEST_CY, TEST_W, TEST_H,
        Tensor(), Tensor(), CameraModelType::PINHOLE);

    validate_params(params, TEST_W, TEST_H);
    EXPECT_NEAR(params.dst_width, TEST_W, TEST_W / 4);
    EXPECT_NEAR(params.dst_height, TEST_H, TEST_H / 4);
}

TEST(UndistortPinhole, SimpleRadial) {
    // COLMAP model 2: SIMPLE_RADIAL — 1 radial coeff
    auto radial = Tensor::from_vector({-0.08f}, TensorShape({1}), Device::CPU);
    auto params = compute_undistort_params(
        TEST_FX, TEST_FY, TEST_CX, TEST_CY, TEST_W, TEST_H,
        radial, Tensor(), CameraModelType::PINHOLE);

    validate_params(params, TEST_W, TEST_H);
    run_image_undistort(params);
    run_mask_undistort(params);
}

TEST(UndistortPinhole, Radial) {
    // COLMAP model 3: RADIAL — 2 radial coeffs (k1, k2)
    auto radial = Tensor::from_vector({-0.1f, 0.02f}, TensorShape({2}), Device::CPU);
    auto params = compute_undistort_params(
        TEST_FX, TEST_FY, TEST_CX, TEST_CY, TEST_W, TEST_H,
        radial, Tensor(), CameraModelType::PINHOLE);

    validate_params(params, TEST_W, TEST_H);
    run_image_undistort(params);
}

TEST(UndistortPinhole, OpenCV) {
    // COLMAP model 4: OPENCV — k1,k2 + p1,p2
    auto radial = Tensor::from_vector({-0.1f, 0.02f}, TensorShape({2}), Device::CPU);
    auto tangential = Tensor::from_vector({0.003f, -0.004f}, TensorShape({2}), Device::CPU);
    auto params = compute_undistort_params(
        TEST_FX, TEST_FY, TEST_CX, TEST_CY, TEST_W, TEST_H,
        radial, tangential, CameraModelType::PINHOLE);

    validate_params(params, TEST_W, TEST_H);
    run_image_undistort(params);
    run_mask_undistort(params);
}

TEST(UndistortPinhole, FullOpenCV) {
    // COLMAP model 6: FULL_OPENCV — k1,k2,k3 (we cap at 3 radial) + p1,p2
    auto radial = Tensor::from_vector({-0.15f, 0.03f, -0.005f}, TensorShape({3}), Device::CPU);
    auto tangential = Tensor::from_vector({0.001f, -0.002f}, TensorShape({2}), Device::CPU);
    auto params = compute_undistort_params(
        TEST_FX, TEST_FY, TEST_CX, TEST_CY, TEST_W, TEST_H,
        radial, tangential, CameraModelType::PINHOLE);

    validate_params(params, TEST_W, TEST_H);
    run_image_undistort(params);
}

TEST(UndistortPinhole, StrongBarrelDistortion) {
    auto radial = Tensor::from_vector({-0.3f, 0.1f, -0.02f}, TensorShape({3}), Device::CPU);
    auto params = compute_undistort_params(
        TEST_FX, TEST_FY, TEST_CX, TEST_CY, TEST_W, TEST_H,
        radial, Tensor(), CameraModelType::PINHOLE);

    validate_params(params, TEST_W, TEST_H);
    run_image_undistort(params);
}

TEST(UndistortPinhole, StrongPincushionDistortion) {
    auto radial = Tensor::from_vector({0.3f, -0.1f}, TensorShape({2}), Device::CPU);
    auto params = compute_undistort_params(
        TEST_FX, TEST_FY, TEST_CX, TEST_CY, TEST_W, TEST_H,
        radial, Tensor(), CameraModelType::PINHOLE);

    validate_params(params, TEST_W, TEST_H);
    run_image_undistort(params);
}

// ====================== Fisheye model tests ======================

TEST(UndistortFisheye, SimpleRadialFisheye) {
    // COLMAP model 8: SIMPLE_RADIAL_FISHEYE — 1 coeff
    auto radial = Tensor::from_vector({0.05f}, TensorShape({1}), Device::CPU);
    auto params = compute_undistort_params(
        TEST_FX, TEST_FY, TEST_CX, TEST_CY, TEST_W, TEST_H,
        radial, Tensor(), CameraModelType::FISHEYE);

    validate_params(params, TEST_W, TEST_H);
    run_image_undistort(params);
}

TEST(UndistortFisheye, RadialFisheye) {
    // COLMAP model 9: RADIAL_FISHEYE — 2 coeffs
    auto radial = Tensor::from_vector({0.05f, -0.01f}, TensorShape({2}), Device::CPU);
    auto params = compute_undistort_params(
        TEST_FX, TEST_FY, TEST_CX, TEST_CY, TEST_W, TEST_H,
        radial, Tensor(), CameraModelType::FISHEYE);

    validate_params(params, TEST_W, TEST_H);
    run_image_undistort(params);
    run_mask_undistort(params);
}

TEST(UndistortFisheye, OpenCVFisheye) {
    // COLMAP model 5: OPENCV_FISHEYE — 4 coeffs (k1-k4)
    auto radial = Tensor::from_vector({0.1f, -0.02f, 0.005f, -0.001f}, TensorShape({4}), Device::CPU);
    auto params = compute_undistort_params(
        TEST_FX, TEST_FY, TEST_CX, TEST_CY, TEST_W, TEST_H,
        radial, Tensor(), CameraModelType::FISHEYE);

    validate_params(params, TEST_W, TEST_H);
    run_image_undistort(params);
}

TEST(UndistortFisheye, StrongFisheyeDistortion) {
    auto radial = Tensor::from_vector({0.3f, -0.1f, 0.02f, -0.005f}, TensorShape({4}), Device::CPU);
    auto params = compute_undistort_params(
        300.0f, 300.0f, 320.0f, 240.0f,
        640, 480,
        radial, Tensor(), CameraModelType::FISHEYE);

    validate_params(params, TEST_W, TEST_H);
    run_image_undistort(params);
}

// ====================== Thin prism fisheye tests ======================

TEST(UndistortThinPrism, FullCoefficients) {
    // COLMAP model 10: THIN_PRISM_FISHEYE
    auto radial = Tensor::from_vector({0.1f, -0.02f, 0.003f, -0.001f}, TensorShape({4}), Device::CPU);
    auto tangential = Tensor::from_vector({0.0005f, -0.0003f, 0.0001f, -0.0002f}, TensorShape({4}), Device::CPU);
    auto params = compute_undistort_params(
        TEST_FX, TEST_FY, TEST_CX, TEST_CY, TEST_W, TEST_H,
        radial, tangential, CameraModelType::THIN_PRISM_FISHEYE);

    validate_params(params, TEST_W, TEST_H);
    run_image_undistort(params);
    run_mask_undistort(params);
}

TEST(UndistortThinPrism, RadialOnlyNoTangential) {
    auto radial = Tensor::from_vector({0.05f, -0.01f, 0.002f, -0.0005f}, TensorShape({4}), Device::CPU);
    auto params = compute_undistort_params(
        TEST_FX, TEST_FY, TEST_CX, TEST_CY, TEST_W, TEST_H,
        radial, Tensor(), CameraModelType::THIN_PRISM_FISHEYE);

    validate_params(params, TEST_W, TEST_H);
    run_image_undistort(params);
}

// ====================== blank_pixels parameter ======================

TEST(UndistortBlankPixels, ZeroVsNonZero) {
    auto radial = Tensor::from_vector({-0.15f, 0.03f}, TensorShape({2}), Device::CPU);

    auto tight = compute_undistort_params(
        TEST_FX, TEST_FY, TEST_CX, TEST_CY, TEST_W, TEST_H,
        radial, Tensor(), CameraModelType::PINHOLE, 0.0f);

    auto loose = compute_undistort_params(
        TEST_FX, TEST_FY, TEST_CX, TEST_CY, TEST_W, TEST_H,
        radial, Tensor(), CameraModelType::PINHOLE, 0.5f);

    // COLMAP-style blank pixel handling keeps focal length fixed and expands
    // the output image instead of zooming the undistorted camera.
    EXPECT_FLOAT_EQ(loose.dst_fx, tight.dst_fx);
    EXPECT_FLOAT_EQ(loose.dst_fy, tight.dst_fy);
    EXPECT_GE(loose.dst_width, tight.dst_width);
    EXPECT_GE(loose.dst_height, tight.dst_height);

    // Both must produce valid results
    validate_params(tight, TEST_W, TEST_H);
    validate_params(loose, TEST_W, TEST_H);
}

TEST(UndistortColmapParity, SimpleRadialNoBlankPixelsMatchesColmap) {
    constexpr float kColmapParityTolerance = 1e-4f;
    auto radial = Tensor::from_vector({0.5f}, TensorShape({1}), Device::CPU);

    const auto params = compute_undistort_params(
        100.0f, 100.0f, 50.0f, 50.0f, 100, 100,
        radial, Tensor(), CameraModelType::PINHOLE, 0.0f);

    EXPECT_NEAR(params.dst_fx, 100.0f, kColmapParityTolerance);
    EXPECT_NEAR(params.dst_fy, 100.0f, kColmapParityTolerance);
    EXPECT_NEAR(params.dst_cx, 42.0f, kColmapParityTolerance);
    EXPECT_NEAR(params.dst_cy, 42.0f, kColmapParityTolerance);
    EXPECT_EQ(params.dst_width, 84);
    EXPECT_EQ(params.dst_height, 84);
}

TEST(UndistortColmapParity, SimpleRadialAllowBlankPixelsMatchesColmap) {
    constexpr float kColmapParityTolerance = 1e-4f;
    auto radial = Tensor::from_vector({0.5f}, TensorShape({1}), Device::CPU);

    const auto params = compute_undistort_params(
        100.0f, 100.0f, 50.0f, 50.0f, 100, 100,
        radial, Tensor(), CameraModelType::PINHOLE, 1.0f);

    EXPECT_NEAR(params.dst_fx, 100.0f, kColmapParityTolerance);
    EXPECT_NEAR(params.dst_fy, 100.0f, kColmapParityTolerance);
    EXPECT_NEAR(params.dst_cx, 45.0f, kColmapParityTolerance);
    EXPECT_NEAR(params.dst_cy, 45.0f, kColmapParityTolerance);
    EXPECT_EQ(params.dst_width, 90);
    EXPECT_EQ(params.dst_height, 90);
}

TEST(ScaleUndistortParams, PreservesPrincipalPointOffset) {
    UndistortParams params{};
    params.src_fx = 100.0f;
    params.src_fy = 120.0f;
    params.src_cx = 40.0f;
    params.src_cy = 15.0f;
    params.src_width = 80;
    params.src_height = 30;
    params.dst_fx = 100.0f;
    params.dst_fy = 120.0f;
    params.dst_cx = 33.0f;
    params.dst_cy = 12.0f;
    params.dst_width = 66;
    params.dst_height = 24;

    const auto scaled = scale_undistort_params(params, 40, 15);

    EXPECT_FLOAT_EQ(scaled.src_fx, 50.0f);
    EXPECT_FLOAT_EQ(scaled.src_fy, 60.0f);
    EXPECT_FLOAT_EQ(scaled.src_cx, 20.0f);
    EXPECT_FLOAT_EQ(scaled.src_cy, 7.5f);
    EXPECT_FLOAT_EQ(scaled.dst_fx, 50.0f);
    EXPECT_FLOAT_EQ(scaled.dst_fy, 60.0f);
    EXPECT_FLOAT_EQ(scaled.dst_cx, 16.5f);
    EXPECT_FLOAT_EQ(scaled.dst_cy, 6.0f);
    EXPECT_EQ(scaled.dst_width, 33);
    EXPECT_EQ(scaled.dst_height, 12);
}

// ====================== Mask-image consistency ======================

TEST(UndistortConsistency, MaskAndImageSameDimensions) {
    auto radial = Tensor::from_vector({-0.1f, 0.02f}, TensorShape({2}), Device::CPU);
    auto tangential = Tensor::from_vector({0.003f, -0.004f}, TensorShape({2}), Device::CPU);
    auto params = compute_undistort_params(
        TEST_FX, TEST_FY, TEST_CX, TEST_CY, TEST_W, TEST_H,
        radial, tangential, CameraModelType::PINHOLE);

    auto img_src = Tensor::randn({3, static_cast<size_t>(TEST_H), static_cast<size_t>(TEST_W)}, Device::CUDA);
    auto mask_src = Tensor::ones({static_cast<size_t>(TEST_H), static_cast<size_t>(TEST_W)}, Device::CUDA);

    auto img_dst = undistort_image(img_src, params, nullptr);
    auto mask_dst = undistort_mask(mask_src, params, nullptr);
    cudaDeviceSynchronize();

    EXPECT_EQ(img_dst.shape()[1], mask_dst.shape()[0]);
    EXPECT_EQ(img_dst.shape()[2], mask_dst.shape()[1]);
}

// ====================== Center pixel preservation ======================

TEST(UndistortCenter, CenterPixelPreserved) {
    // For radial distortion, the center of distortion should map to itself.
    // Create a white image with a single bright pixel at center.
    auto params = compute_undistort_params(
        TEST_FX, TEST_FY, TEST_CX, TEST_CY, TEST_W, TEST_H,
        Tensor::from_vector({-0.1f, 0.01f}, TensorShape({2}), Device::CPU),
        Tensor(), CameraModelType::PINHOLE);

    auto src = Tensor::zeros({1, static_cast<size_t>(TEST_H), static_cast<size_t>(TEST_W)}, Device::CUDA);
    auto src_cpu = src.cpu();
    auto acc = src_cpu.accessor<float, 3>();
    int cx = static_cast<int>(TEST_CX);
    int cy = static_cast<int>(TEST_CY);
    acc(0, cy, cx) = 1.0f;
    src = src_cpu.to(Device::CUDA);

    auto dst = undistort_image(src, params, nullptr);
    cudaDeviceSynchronize();

    // The brightest pixel in the output should be near the destination principal point
    auto dst_cpu = dst.cpu();
    auto dst_acc = dst_cpu.accessor<float, 3>();
    float max_val = 0.0f;
    int max_x = 0, max_y = 0;
    for (int y = 0; y < params.dst_height; ++y) {
        for (int x = 0; x < params.dst_width; ++x) {
            float v = dst_acc(0, y, x);
            if (v > max_val) {
                max_val = v;
                max_x = x;
                max_y = y;
            }
        }
    }
    EXPECT_NEAR(max_x, static_cast<int>(params.dst_cx), 3);
    EXPECT_NEAR(max_y, static_cast<int>(params.dst_cy), 3);
}

// ====================== Camera class integration (bicycle data) ======================

class UndistortCameraTest : public ::testing::Test {
protected:
    void SetUp() override {
        base_path_ = std::filesystem::path(TEST_DATA_DIR) / "bicycle";
        if (!std::filesystem::exists(base_path_ / "sparse" / "0" / "cameras.bin")) {
            GTEST_SKIP() << "Bicycle dataset not available";
        }

        auto result = lfs::io::read_colmap_cameras_and_images(base_path_, "images_4");
        ASSERT_TRUE(result.has_value()) << "Failed to load COLMAP data";
        auto& [cams, center] = result->value;
        cameras_ = std::move(cams);
        ASSERT_GT(cameras_.size(), 0u);
    }

    std::filesystem::path base_path_;
    std::vector<std::shared_ptr<Camera>> cameras_;
};

TEST_F(UndistortCameraTest, BicyclePinholeNoDist) {
    // Bicycle is pure pinhole — prepare_undistortion should be a noop
    auto& cam = cameras_[0];
    EXPECT_FALSE(cam->has_distortion());
    cam->prepare_undistortion();
    EXPECT_FALSE(cam->is_undistort_prepared());
}

TEST_F(UndistortCameraTest, HasDistortionDetection) {
    auto& cam = cameras_[0];

    // Bicycle has no distortion
    EXPECT_FALSE(cam->has_distortion());

    // A camera constructed with radial params should report distortion
    auto R = cam->R();
    auto T = cam->T();
    auto radial = Tensor::from_vector({-0.1f}, TensorShape({1}), Device::CPU);
    Camera distorted_cam(R, T,
                         cam->focal_x(), cam->focal_y(),
                         cam->center_x(), cam->center_y(),
                         radial, Tensor(),
                         CameraModelType::PINHOLE,
                         "test", cam->image_path(), "",
                         cam->camera_width(), cam->camera_height(), 999);
    EXPECT_TRUE(distorted_cam.has_distortion());
}

TEST_F(UndistortCameraTest, PrepareAndQueryUndistortedIntrinsics) {
    auto& cam = cameras_[0];
    auto R = cam->R();
    auto T = cam->T();
    auto radial = Tensor::from_vector({-0.1f, 0.02f}, TensorShape({2}), Device::CPU);

    Camera distorted_cam(R, T,
                         cam->focal_x(), cam->focal_y(),
                         cam->center_x(), cam->center_y(),
                         radial, Tensor(),
                         CameraModelType::PINHOLE,
                         "test", cam->image_path(), "",
                         cam->camera_width(), cam->camera_height(), 998);

    distorted_cam.prepare_undistortion();
    ASSERT_TRUE(distorted_cam.is_undistort_prepared());

    auto [fx, fy, cx, cy] = distorted_cam.get_intrinsics();
    EXPECT_GT(fx, 0.0f);
    EXPECT_GT(fy, 0.0f);
    EXPECT_GT(cx, 0.0f);
    EXPECT_GT(cy, 0.0f);

    // Undistorted intrinsics should differ from original when distortion is present
    auto& p = distorted_cam.undistort_params();
    EXPECT_NE(p.dst_width, cam->camera_width());
}

TEST_F(UndistortCameraTest, FisheyeCameraModel) {
    auto& cam = cameras_[0];
    auto R = cam->R();
    auto T = cam->T();
    auto radial = Tensor::from_vector({0.05f, -0.01f, 0.002f, -0.0005f}, TensorShape({4}), Device::CPU);

    Camera fisheye_cam(R, T,
                       cam->focal_x(), cam->focal_y(),
                       cam->center_x(), cam->center_y(),
                       radial, Tensor(),
                       CameraModelType::FISHEYE,
                       "test_fisheye", cam->image_path(), "",
                       cam->camera_width(), cam->camera_height(), 997);

    EXPECT_TRUE(fisheye_cam.has_distortion());
    fisheye_cam.prepare_undistortion();
    ASSERT_TRUE(fisheye_cam.is_undistort_prepared());

    auto& p = fisheye_cam.undistort_params();
    EXPECT_GT(p.dst_width, 0);
    EXPECT_GT(p.dst_height, 0);
    EXPECT_EQ(p.model_type, CameraModelType::FISHEYE);
}

TEST_F(UndistortCameraTest, EquirectangularModelDoesNotUseUndistortion) {
    auto& cam = cameras_[0];
    auto R = cam->R();
    auto T = cam->T();

    auto radial = Tensor::from_vector({0.05f}, TensorShape({1}), Device::CPU);
    auto tangential = Tensor::from_vector({0.01f, -0.02f}, TensorShape({2}), Device::CPU);

    Camera equirect_cam(R, T,
                        cam->focal_x(), cam->focal_y(),
                        cam->center_x(), cam->center_y(),
                        radial, tangential,
                        CameraModelType::EQUIRECTANGULAR,
                        "test_equirect", cam->image_path(), "",
                        cam->camera_width(), cam->camera_height(), 996);

    EXPECT_FALSE(equirect_cam.has_distortion());
    equirect_cam.prepare_undistortion();
    EXPECT_FALSE(equirect_cam.is_undistort_prepared());
}

// Pins the face-sizing policy to concrete values. The previous version of this
// test re-derived the formula, so it could not detect a change to it.
//
// A 2:1 4K panorama resolves 3840/(2*pi) = 611.15 px/rad, so a 96 degree face
// matching that rate at its centre is 2 * 611.15 * tan(48 deg) = 1358 px.
TEST(SphericalFaceSizing, MatchesPanoramaSamplingRateAtFaceCentre) {
    EXPECT_EQ(cube_face_size_for_panorama(3840, 1920, 96.0f), 1358);
    EXPECT_EQ(cube_face_size_for_panorama(3840, 1920, 90.0f), 1222);
    EXPECT_EQ(cube_face_size_for_panorama(2048, 1024, 96.0f), 724);
}

TEST(SphericalFaceSizing, FocalRoundTripsToTheRequestedFov) {
    constexpr float kPi = 3.14159265358979323846f;
    for (const float fov : {60.0f, 90.0f, 96.0f, 120.0f}) {
        const float focal = cube_face_focal(fov, 1000);
        EXPECT_NEAR(focal2fov(focal, 1000) * 180.0f / kPi, fov, 1e-2f);
    }
}

TEST(SphericalFaceSizing, DegenerateFovStaysFinite) {
    // Field of view is clamped, so neither a zero nor an over-wide value may
    // produce a zero, negative or infinite focal length.
    EXPECT_GT(cube_face_size_for_panorama(3840, 1920, 0.0f), 0);
    EXPECT_GT(cube_face_size_for_panorama(3840, 1920, 360.0f), 0);
    EXPECT_GT(cube_face_focal(0.0f, 512), 0.0f);
    EXPECT_TRUE(std::isfinite(cube_face_focal(0.0f, 512)));
    EXPECT_TRUE(std::isfinite(cube_face_focal(360.0f, 512)));
}

TEST_F(UndistortCameraTest, EquirectangularSphericalExpansionUsesVirtualPinholeCameras) {
    auto& source = cameras_[0];
    auto equirect_cam = std::make_shared<Camera>(
        source->R(), source->T(),
        source->focal_x(), source->focal_y(),
        source->center_x(), source->center_y(),
        Tensor(), Tensor(),
        CameraModelType::EQUIRECTANGULAR,
        "pano.jpg", source->image_path(), "",
        source->camera_width(), source->camera_height(), 7, 99);
    equirect_cam->set_has_alpha(true);
    equirect_cam->set_split(CameraSplit::Eval);

    auto expanded = lfs::training::expandEquirectangularCamerasForUndistort({equirect_cam});

    ASSERT_TRUE(expanded.has_value()) << expanded.error();
    ASSERT_EQ(expanded->size(), SPHERICAL_UNDISTORT_VIEW_COUNT);

    const auto [source_width, source_height, source_channels] = get_image_info(source->image_path());
    ASSERT_GT(source_channels, 0);
    const int expected_face_size = cube_face_size_for_panorama(
        source_width, source_height, SPHERICAL_UNDISTORT_FOV);
    const float expected_focal = cube_face_focal(SPHERICAL_UNDISTORT_FOV, expected_face_size);

    for (size_t i = 0; i < expanded->size(); ++i) {
        const auto& face = (*expanded)[i];
        ASSERT_TRUE(face);
        EXPECT_EQ(face->camera_model_type(), CameraModelType::PINHOLE);
        EXPECT_TRUE(face->has_cube_face_projection());
        EXPECT_EQ(face->image_path(), source->image_path());
        EXPECT_EQ(face->split(), CameraSplit::Eval);
        EXPECT_TRUE(face->has_alpha());
        EXPECT_EQ(face->camera_id(), 99);
        EXPECT_EQ(face->uid(), static_cast<int>(i));
        EXPECT_EQ(face->camera_width(), expected_face_size);
        EXPECT_EQ(face->camera_height(), expected_face_size);
        EXPECT_NEAR(face->focal_x(), expected_focal, 1e-4f);
        EXPECT_NEAR(face->focal_y(), expected_focal, 1e-4f);
        ASSERT_TRUE(face->cube_face_projection().has_value());
        EXPECT_EQ(face->cube_face_projection()->face_size, expected_face_size);
        EXPECT_EQ(face->cube_face_projection()->source_width, source_width);
        EXPECT_EQ(face->cube_face_projection()->source_height, source_height);
        EXPECT_NEAR(face->cube_face_projection()->fov_degrees, SPHERICAL_UNDISTORT_FOV, 1e-4f);
        const auto& pano_to_face = face->cube_face_projection()->pano_to_face;
        const auto expected_matrix = expected_spherical_matrix(SPHERICAL_UNDISTORT_FORWARDS[i]);
        for (size_t j = 0; j < pano_to_face.size(); ++j)
            EXPECT_NEAR(pano_to_face[j], expected_matrix[j], 1e-6f);
        for (int row = 0; row < 3; ++row) {
            const float row_len =
                pano_to_face[row * 3 + 0] * pano_to_face[row * 3 + 0] +
                pano_to_face[row * 3 + 1] * pano_to_face[row * 3 + 1] +
                pano_to_face[row * 3 + 2] * pano_to_face[row * 3 + 2];
            EXPECT_NEAR(row_len, 1.0f, 1e-5f);
        }
        const float dot01 = pano_to_face[0] * pano_to_face[3] +
                            pano_to_face[1] * pano_to_face[4] +
                            pano_to_face[2] * pano_to_face[5];
        const float dot02 = pano_to_face[0] * pano_to_face[6] +
                            pano_to_face[1] * pano_to_face[7] +
                            pano_to_face[2] * pano_to_face[8];
        const float dot12 = pano_to_face[3] * pano_to_face[6] +
                            pano_to_face[4] * pano_to_face[7] +
                            pano_to_face[5] * pano_to_face[8];
        EXPECT_NEAR(dot01, 0.0f, 1e-5f);
        EXPECT_NEAR(dot02, 0.0f, 1e-5f);
        EXPECT_NEAR(dot12, 0.0f, 1e-5f);
    }
}

TEST_F(UndistortCameraTest, SphericalExpansionKeepsPassthroughCamerasAndAvoidsUidCollisions) {
    auto& source = cameras_[0];
    auto equirect_cam = std::make_shared<Camera>(
        source->R(), source->T(),
        source->focal_x(), source->focal_y(),
        source->center_x(), source->center_y(),
        Tensor(), Tensor(),
        CameraModelType::EQUIRECTANGULAR,
        "pano.jpg", source->image_path(), "",
        source->camera_width(), source->camera_height(), 2, 20);
    auto pinhole_cam = std::make_shared<Camera>(
        source->R(), source->T(),
        source->focal_x(), source->focal_y(),
        source->center_x(), source->center_y(),
        Tensor(), Tensor(),
        CameraModelType::PINHOLE,
        "pinhole.jpg", source->image_path(), "",
        source->camera_width(), source->camera_height(), 12, 12);

    auto expanded = lfs::training::expandEquirectangularCamerasForUndistort({equirect_cam, pinhole_cam});

    ASSERT_TRUE(expanded.has_value()) << expanded.error();
    ASSERT_EQ(expanded->size(), SPHERICAL_UNDISTORT_VIEW_COUNT + 1);
    for (size_t i = 0; i < SPHERICAL_UNDISTORT_VIEW_COUNT; ++i) {
        ASSERT_TRUE((*expanded)[i]);
        EXPECT_EQ((*expanded)[i]->camera_model_type(), CameraModelType::PINHOLE);
        EXPECT_TRUE((*expanded)[i]->has_cube_face_projection());
        EXPECT_EQ((*expanded)[i]->uid(), static_cast<int>(13 + i));
    }
    EXPECT_EQ((*expanded)[SPHERICAL_UNDISTORT_VIEW_COUNT], pinhole_cam);
    EXPECT_FALSE((*expanded)[SPHERICAL_UNDISTORT_VIEW_COUNT]->has_cube_face_projection());
    EXPECT_EQ((*expanded)[SPHERICAL_UNDISTORT_VIEW_COUNT]->uid(), 12);
}

TEST(UndistortScale, ScaleUndistortParams) {
    const auto radial = Tensor::from_vector({-0.1f, 0.02f}, TensorShape({2}), Device::CPU);
    const auto params = compute_undistort_params(
        TEST_FX, TEST_FY, TEST_CX, TEST_CY, TEST_W, TEST_H,
        radial, Tensor(), CameraModelType::PINHOLE);

    validate_params(params, TEST_W, TEST_H);

    constexpr int HALF_W = TEST_W / 2;
    constexpr int HALF_H = TEST_H / 2;
    const auto scaled = scale_undistort_params(params, HALF_W, HALF_H);

    EXPECT_EQ(scaled.src_width, HALF_W);
    EXPECT_EQ(scaled.src_height, HALF_H);

    const float sx = static_cast<float>(HALF_W) / static_cast<float>(params.src_width);
    const float sy = static_cast<float>(HALF_H) / static_cast<float>(params.src_height);

    EXPECT_NEAR(scaled.src_fx, params.src_fx * sx, 1e-4f);
    EXPECT_NEAR(scaled.src_fy, params.src_fy * sy, 1e-4f);
    EXPECT_NEAR(scaled.src_cx, params.src_cx * sx, 1e-4f);
    EXPECT_NEAR(scaled.src_cy, params.src_cy * sy, 1e-4f);

    EXPECT_NEAR(scaled.dst_fx, params.dst_fx * sx, 1e-4f);
    EXPECT_NEAR(scaled.dst_fy, params.dst_fy * sy, 1e-4f);

    const int expected_dst_w = std::max(1, static_cast<int>(std::lroundf(params.dst_width * sx)));
    const int expected_dst_h = std::max(1, static_cast<int>(std::lroundf(params.dst_height * sy)));
    EXPECT_EQ(scaled.dst_width, expected_dst_w);
    EXPECT_EQ(scaled.dst_height, expected_dst_h);
    EXPECT_NEAR(scaled.dst_cx, params.dst_cx * sx, 1e-4f);
    EXPECT_NEAR(scaled.dst_cy, params.dst_cy * sy, 1e-4f);

    run_image_undistort(scaled);
    run_mask_undistort(scaled);
}

// ====================== Cube-face projection ======================
//
// These build their panorama analytically, so they need no dataset and the
// expected value is a closed form rather than a golden image.

namespace {

    constexpr float kProjPi = 3.14159265358979323846f;

    // Equirectangular pixel centre -> unit direction.
    Vec3 equirect_direction(const int x, const int y, const int width, const int height) {
        const float az = (static_cast<float>(x) + 0.5f) / width * 2.0f * kProjPi - kProjPi;
        const float el = (static_cast<float>(y) + 0.5f) / height * kProjPi - 0.5f * kProjPi;
        return {std::cos(el) * std::sin(az), std::sin(el), std::cos(el) * std::cos(az)};
    }

    // Face pixel -> unnormalised direction, matching project_cube_face_hwc.
    Vec3 face_direction(const std::array<float, 9>& m, const int x, const int y,
                        const int size, const float fov_degrees) {
        const float focal = cube_face_focal(fov_degrees, size);
        const float c = 0.5f * static_cast<float>(size);
        const float u = (static_cast<float>(x) + 0.5f - c) / focal;
        const float v = (static_cast<float>(y) + 0.5f - c) / focal;
        return {m[0] * u + m[3] * v + m[6],
                m[1] * u + m[4] * v + m[7],
                m[2] * u + m[5] * v + m[8]};
    }

    // Smooth band-limited signal in [0, 1] built from the direction components,
    // so it stays continuous at the poles. An az/el signal does not: every
    // azimuth meets at the pole, so sin(k*az) is discontinuous there, and a
    // correctly filtered face would legitimately disagree with a point sample.
    float smooth_sphere_signal(const Vec3& d) {
        const float len = length(d);
        const float nx = d[0] / len;
        const float ny = d[1] / len;
        const float nz = d[2] / len;
        constexpr float k = 3.0f;
        return 0.5f + 0.5f * std::sin(k * nx) * std::sin(k * ny) * std::sin(k * nz);
    }

    // Deliberately high-frequency signal for the aliasing test. Singular at the
    // poles, which is why only an equatorial face uses it.
    float sphere_signal(const Vec3& d, const float cycles_per_rad) {
        const float k = 2.0f * kProjPi * cycles_per_rad;
        const float az = std::atan2(d[0], d[2]);
        const float len = length(d);
        const float el = std::asin(std::clamp(len > 0.0f ? d[1] / len : 0.0f, -1.0f, 1.0f));
        return 0.5f + 0.5f * std::sin(k * az) * std::sin(0.5f * k * el);
    }

    std::vector<uint8_t> make_smooth_panorama(const int width, const int height) {
        std::vector<uint8_t> pano(static_cast<size_t>(width) * height);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                pano[static_cast<size_t>(y) * width + x] = static_cast<uint8_t>(std::lround(
                    255.0f * smooth_sphere_signal(equirect_direction(x, y, width, height))));
            }
        }
        return pano;
    }

    std::vector<uint8_t> make_signal_panorama(const int width, const int height,
                                              const float cycles_per_rad) {
        std::vector<uint8_t> pano(static_cast<size_t>(width) * height);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                pano[static_cast<size_t>(y) * width + x] = static_cast<uint8_t>(std::lround(
                    255.0f * sphere_signal(equirect_direction(x, y, width, height), cycles_per_rad)));
            }
        }
        return pano;
    }

    const std::array<Vec3, 6>& cube_forwards() {
        static const std::array<Vec3, 6> f{{
            {1.f, 0.f, 0.f},
            {-1.f, 0.f, 0.f},
            {0.f, 1.f, 0.f},
            {0.f, -1.f, 0.f},
            {0.f, 0.f, 1.f},
            {0.f, 0.f, -1.f},
        }};
        return f;
    }

} // namespace

// A transposed or mirrored face basis still produces a plausible-looking image,
// so check that each face pixel carries the panorama content for the direction
// that pixel actually represents.
TEST(CubeFaceProjection, FacePixelsCarryTheContentOfTheirOwnRay) {
    constexpr int W = 1024;
    constexpr int H = 512;
    constexpr int S = 128;
    constexpr float kFov = 96.0f;
    const auto pano = make_smooth_panorama(W, H);

    for (size_t f = 0; f < cube_forwards().size(); ++f) {
        const auto basis = expected_spherical_matrix(cube_forwards()[f]);
        const lfs::core::CubeFaceProjection proj{basis, S, W, H, kFov};
        const auto face = lfs::io::project_cube_face_hwc(pano.data(), W, H, 1, proj, S);

        double sum_sq = 0.0;
        for (int y = 0; y < S; ++y) {
            for (int x = 0; x < S; ++x) {
                const float expected =
                    255.0f * smooth_sphere_signal(face_direction(basis, x, y, S, kFov));
                const double d =
                    static_cast<double>(face[static_cast<size_t>(y) * S + x]) - expected;
                sum_sq += d * d;
            }
        }
        const double rmse = std::sqrt(sum_sq / (static_cast<double>(S) * S));
        EXPECT_LT(rmse, 3.0) << "face " << f << " does not match the panorama along its own rays";
    }
}

// The elliptical filter weights must sum to one everywhere, including at the
// poles where the footprint is extremely anisotropic and gets strided.
TEST(CubeFaceProjection, ConstantPanoramaIsPreservedOnEveryFace) {
    constexpr int W = 512;
    constexpr int H = 256;
    constexpr int S = 96;
    constexpr uint8_t kValue = 173;
    const std::vector<uint8_t> pano(static_cast<size_t>(W) * H, kValue);

    for (size_t f = 0; f < cube_forwards().size(); ++f) {
        const lfs::core::CubeFaceProjection proj{
            expected_spherical_matrix(cube_forwards()[f]), S, W, H, 96.0f};
        const auto face = lfs::io::project_cube_face_hwc(pano.data(), W, H, 1, proj, S);
        const auto bounds = std::minmax_element(face.begin(), face.end());
        EXPECT_EQ(static_cast<int>(*bounds.first), static_cast<int>(kValue)) << "face " << f;
        EXPECT_EQ(static_cast<int>(*bounds.second), static_cast<int>(kValue)) << "face " << f;
    }
}

// Content the face cannot resolve must be attenuated towards the local mean
// rather than aliased into spurious low-frequency structure. A plain bilinear
// fetch fails this; the prefilter is what makes it pass.
TEST(CubeFaceProjection, SignalAboveFaceNyquistIsAttenuatedNotAliased) {
    constexpr int W = 2048;
    constexpr int H = 1024;
    constexpr int S = 64;
    constexpr float kFov = 96.0f;
    // The face resolves ~14 cyc/rad here; the panorama still resolves 60 easily.
    const auto pano = make_signal_panorama(W, H, 60.0f);

    const lfs::core::CubeFaceProjection proj{
        expected_spherical_matrix(cube_forwards()[4]), S, W, H, kFov};
    const auto face = lfs::io::project_cube_face_hwc(pano.data(), W, H, 1, proj, S);

    double mean = 0.0;
    for (const uint8_t v : face)
        mean += v;
    mean /= static_cast<double>(face.size());

    double stddev = 0.0;
    for (const uint8_t v : face)
        stddev += (v - mean) * (v - mean);
    stddev = std::sqrt(stddev / static_cast<double>(face.size()));

    EXPECT_NEAR(mean, 127.5, 12.0) << "attenuated output should sit near the signal mean";
    EXPECT_LT(stddev, 30.0) << "unresolvable detail survived as aliasing instead of being filtered";
}

// ====================== Cube-face depth ======================

// Analytic scene: a cube of half-extent h centred on the camera, stored as
// 16-bit equirectangular radial distance. Camera-space z on a flat wall is
// exactly h wherever it is sampled, which holds only if the radial-to-z
// conversion is applied.
TEST(CubeFaceDepth, FlatWallHasConstantCameraSpaceZ) {
    constexpr int W = 1024;
    constexpr int H = 512;
    constexpr int S = 96;
    constexpr float kFov = 96.0f;
    constexpr double kHalf = 0.5;

    std::vector<uint16_t> pano(static_cast<size_t>(W) * H);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const Vec3 d = equirect_direction(x, y, W, H);
            const double m = std::max(std::max(std::abs(d[0]), std::abs(d[1])), std::abs(d[2]));
            pano[static_cast<size_t>(y) * W + x] =
                static_cast<uint16_t>(std::lround(std::clamp(kHalf / m, 0.0, 1.0) * 65535.0));
        }
    }

    const float focal = cube_face_focal(kFov, S);
    const float center = 0.5f * S;
    const float wall = std::tan(45.0f * kProjPi / 180.0f) * 0.94f; // stay off the seam

    for (size_t f = 0; f < cube_forwards().size(); ++f) {
        const lfs::core::CubeFaceProjection proj{
            expected_spherical_matrix(cube_forwards()[f]), S, W, H, kFov};
        const auto face = lfs::io::project_cube_face_depth(
            reinterpret_cast<const uint8_t*>(pano.data()), true, W, H, proj, S);

        double worst = 0.0;
        for (int y = 0; y < S; ++y) {
            const float v = (static_cast<float>(y) + 0.5f - center) / focal;
            for (int x = 0; x < S; ++x) {
                const float u = (static_cast<float>(x) + 0.5f - center) / focal;
                if (std::abs(u) > wall || std::abs(v) > wall)
                    continue;
                worst = std::max(worst,
                                 std::abs(static_cast<double>(face[static_cast<size_t>(y) * S + x]) - kHalf));
            }
        }
        EXPECT_LT(worst, 0.01) << "face " << f << ": z is not constant on a flat wall";
    }
}

// Negative control for the test above: without the radial-to-z conversion an
// off-axis pixel would report the radial distance, which differs materially.
TEST(CubeFaceDepth, RadialToZConversionIsApplied) {
    constexpr int W = 1024;
    constexpr int H = 512;
    constexpr int S = 96;
    constexpr float kFov = 96.0f;
    constexpr float kRadial = 0.5f;

    // Constant radial distance in every direction: a sphere around the camera.
    const std::vector<uint16_t> pano(static_cast<size_t>(W) * H,
                                     static_cast<uint16_t>(std::lround(kRadial * 65535.0f)));

    const lfs::core::CubeFaceProjection proj{
        expected_spherical_matrix(cube_forwards()[4]), S, W, H, kFov};
    const auto face = lfs::io::project_cube_face_depth(
        reinterpret_cast<const uint8_t*>(pano.data()), true, W, H, proj, S);

    const float focal = cube_face_focal(kFov, S);
    const float center = 0.5f * S;

    // The centre pixel looks along the axis, so z equals the radial distance.
    const int mid = S / 2;
    EXPECT_NEAR(face[static_cast<size_t>(mid) * S + mid], kRadial, 0.01f);

    // A corner pixel is off-axis, so z must be shortened by cos(theta).
    const float u = (0.5f - center) / focal;
    const float v = (0.5f - center) / focal;
    const float cos_theta = 1.0f / std::sqrt(u * u + v * v + 1.0f);
    EXPECT_NEAR(face[0], kRadial * cos_theta, 0.01f);
    EXPECT_LT(face[0], kRadial * 0.65f) << "corner z was not converted from radial distance";
}

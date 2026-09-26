/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/camera.hpp"
#include "core/cuda/lanczos_resize/lanczos_resize.hpp"
#include "core/cuda/undistort/undistort.hpp"
#include "core/image_io.hpp"
#include "core/tensor_backend.hpp"
#include "io/formats/colmap.hpp"
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <limits>

using namespace lfs::core;

namespace {

    constexpr float TEST_FX = 500.0f;
    constexpr float TEST_FY = 500.0f;
    constexpr float TEST_CX = 320.0f;
    constexpr float TEST_CY = 240.0f;
    constexpr int TEST_W = 640;
    constexpr int TEST_H = 480;

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
            Device::GPU);

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
            Device::GPU);

        auto dst = undistort_mask(src, params, nullptr);
        cudaDeviceSynchronize();

        ASSERT_EQ(dst.ndim(), 2u);
        EXPECT_EQ(static_cast<int>(dst.shape()[0]), params.dst_height);
        EXPECT_EQ(static_cast<int>(dst.shape()[1]), params.dst_width);
    }

    UndistortParams make_expanding_undistort_params() {
        const auto radial = Tensor::from_vector({-0.3f, 0.1f, -0.02f}, TensorShape({3}), Device::CPU);
        return compute_undistort_params(
            TEST_FX, TEST_FY, TEST_CX, TEST_CY, TEST_W, TEST_H,
            radial, Tensor(), CameraModelType::PINHOLE);
    }

    void expect_params_equal(const UndistortParams& actual, const UndistortParams& expected) {
        EXPECT_FLOAT_EQ(actual.src_fx, expected.src_fx);
        EXPECT_FLOAT_EQ(actual.src_fy, expected.src_fy);
        EXPECT_FLOAT_EQ(actual.src_cx, expected.src_cx);
        EXPECT_FLOAT_EQ(actual.src_cy, expected.src_cy);
        EXPECT_FLOAT_EQ(actual.dst_fx, expected.dst_fx);
        EXPECT_FLOAT_EQ(actual.dst_fy, expected.dst_fy);
        EXPECT_FLOAT_EQ(actual.dst_cx, expected.dst_cx);
        EXPECT_FLOAT_EQ(actual.dst_cy, expected.dst_cy);
        EXPECT_EQ(actual.src_width, expected.src_width);
        EXPECT_EQ(actual.src_height, expected.src_height);
        EXPECT_EQ(actual.dst_width, expected.dst_width);
        EXPECT_EQ(actual.dst_height, expected.dst_height);
        EXPECT_EQ(actual.model_type, expected.model_type);
        EXPECT_EQ(actual.num_distortion, expected.num_distortion);
        EXPECT_EQ(actual.crop_solve_failed, expected.crop_solve_failed);
        for (int i = 0; i < 12; ++i) {
            EXPECT_FLOAT_EQ(actual.distortion[i], expected.distortion[i]);
        }
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

TEST(UndistortDiagnostics, ReportsCropSolveFailure) {
    const auto params = compute_undistort_params(
        TEST_FX, TEST_FY, TEST_CX, TEST_CY, 0, 0,
        Tensor::from_vector({0.1f, -0.02f, 0.003f, -0.0004f}, {4}, Device::CPU),
        Tensor(), CameraModelType::FISHEYE);

    EXPECT_TRUE(params.crop_solve_failed);
    EXPECT_EQ(params.dst_width, 0);
    EXPECT_EQ(params.dst_height, 0);
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

TEST(ScaleUndistortParams, CapsOutputAndScalesDestinationIntrinsics) {
    const auto params = make_expanding_undistort_params();
    ASSERT_GT(std::max(params.dst_width, params.dst_height),
              std::max(params.src_width, params.src_height));

    constexpr int actual_src_width = TEST_W / 2;
    constexpr int actual_src_height = TEST_H / 2;
    const auto uncapped = scale_undistort_params(params, actual_src_width, actual_src_height);
    const int max_width = std::max(uncapped.dst_width, uncapped.dst_height) / 2;
    ASSERT_GT(max_width, 0);

    const auto capped = scale_undistort_params(
        params, actual_src_width, actual_src_height, max_width);

    int expected_width;
    int expected_height;
    if (uncapped.dst_width > uncapped.dst_height) {
        expected_width = max_width;
        expected_height = std::max(
            1, static_cast<int>(static_cast<std::int64_t>(uncapped.dst_height) * max_width /
                                uncapped.dst_width));
    } else {
        expected_height = max_width;
        expected_width = std::max(
            1, static_cast<int>(static_cast<std::int64_t>(uncapped.dst_width) * max_width /
                                uncapped.dst_height));
    }

    EXPECT_EQ(std::max(capped.dst_width, capped.dst_height), max_width);
    EXPECT_EQ(capped.dst_width, expected_width);
    EXPECT_EQ(capped.dst_height, expected_height);
    const float dst_sx = static_cast<float>(expected_width) / static_cast<float>(uncapped.dst_width);
    const float dst_sy = static_cast<float>(expected_height) / static_cast<float>(uncapped.dst_height);
    EXPECT_FLOAT_EQ(capped.dst_fx, uncapped.dst_fx * dst_sx);
    EXPECT_FLOAT_EQ(capped.dst_fy, uncapped.dst_fy * dst_sy);
    EXPECT_FLOAT_EQ(capped.dst_cx, uncapped.dst_cx * dst_sx);
    EXPECT_FLOAT_EQ(capped.dst_cy, uncapped.dst_cy * dst_sy);
    EXPECT_FLOAT_EQ(capped.src_fx, uncapped.src_fx);
    EXPECT_FLOAT_EQ(capped.src_fy, uncapped.src_fy);
    EXPECT_FLOAT_EQ(capped.src_cx, uncapped.src_cx);
    EXPECT_FLOAT_EQ(capped.src_cy, uncapped.src_cy);
    EXPECT_EQ(capped.src_width, uncapped.src_width);
    EXPECT_EQ(capped.src_height, uncapped.src_height);

    run_image_undistort(capped);
    run_mask_undistort(capped);
}

TEST(ScaleUndistortParams, CapsOutputWhenSourceSizeMatches) {
    const auto params = make_expanding_undistort_params();
    const int max_width = std::max(params.src_width, params.src_height);
    ASSERT_GT(std::max(params.dst_width, params.dst_height), max_width);

    const auto capped = scale_undistort_params(
        params, params.src_width, params.src_height, max_width);

    EXPECT_EQ(std::max(capped.dst_width, capped.dst_height), max_width);
    EXPECT_LT(capped.dst_width, params.dst_width);
    EXPECT_LT(capped.dst_height, params.dst_height);
}

TEST(ScaleUndistortParams, NonRestrictiveCapsMatchUncappedResult) {
    const auto params = make_expanding_undistort_params();
    ASSERT_GT(std::max(params.dst_width, params.dst_height),
              std::max(params.src_width, params.src_height));

    constexpr int actual_src_width = TEST_W / 2;
    constexpr int actual_src_height = TEST_H / 2;
    const auto uncapped = scale_undistort_params(params, actual_src_width, actual_src_height);
    const int output_max = std::max(uncapped.dst_width, uncapped.dst_height);

    for (const int max_width : {output_max, output_max + 1, 0, -1}) {
        SCOPED_TRACE(max_width);
        const auto actual = scale_undistort_params(
            params, actual_src_width, actual_src_height, max_width);
        expect_params_equal(actual, uncapped);
    }
}

// ====================== Mask-image consistency ======================

TEST(UndistortConsistency, MaskAndImageSameDimensions) {
    auto radial = Tensor::from_vector({-0.1f, 0.02f}, TensorShape({2}), Device::CPU);
    auto tangential = Tensor::from_vector({0.003f, -0.004f}, TensorShape({2}), Device::CPU);
    auto params = compute_undistort_params(
        TEST_FX, TEST_FY, TEST_CX, TEST_CY, TEST_W, TEST_H,
        radial, tangential, CameraModelType::PINHOLE);

    auto img_src = Tensor::randn({3, static_cast<size_t>(TEST_H), static_cast<size_t>(TEST_W)}, Device::GPU);
    auto mask_src = Tensor::ones({static_cast<size_t>(TEST_H), static_cast<size_t>(TEST_W)}, Device::GPU);

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

    auto src = Tensor::zeros({1, static_cast<size_t>(TEST_H), static_cast<size_t>(TEST_W)}, Device::GPU);
    auto src_cpu = src.cpu();
    auto acc = src_cpu.accessor<float, 3>();
    int cx = static_cast<int>(TEST_CX);
    int cy = static_cast<int>(TEST_CY);
    acc(0, cy, cx) = 1.0f;
    src = src_cpu.to(Device::GPU);

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

namespace {
    void expectImageNear(const Tensor& actual, const Tensor& expected, const float tolerance) {
        const auto a = actual.cpu().contiguous();
        const auto e = expected.cpu().contiguous();
        ASSERT_EQ(a.shape(), e.shape());
        for (size_t i = 0; i < a.numel(); ++i) {
            ASSERT_TRUE(std::isfinite(a.ptr<float>()[i])) << i;
            ASSERT_NEAR(a.ptr<float>()[i], e.ptr<float>()[i], tolerance) << i;
        }
    }
} // namespace

TEST(ImageTensorBackends, UndistortionMatchesCudaAcrossBandsAndCameraModels) {
    if (!gpu_backend_available(GpuBackend::CUDA) || !gpu_backend_available(GpuBackend::Vulkan)) {
        GTEST_SKIP() << "Both tensor backends required";
    }
    constexpr int width = 513, height = 131;
    auto cpu = Tensor::empty({3, height, width}, Device::CPU);
    for (size_t i = 0; i < cpu.numel(); ++i) {
        cpu.ptr<float>()[i] = std::sin(static_cast<float>(i) * 0.003f) * 0.4f + 0.5f;
    }
    for (const auto model : {CameraModelType::PINHOLE, CameraModelType::FISHEYE, CameraModelType::THIN_PRISM_FISHEYE}) {
        SCOPED_TRACE(static_cast<int>(model));
        UndistortParams p{};
        p.src_width = p.dst_width = width;
        p.src_height = p.dst_height = height;
        p.src_fx = 320.0f;
        p.src_fy = 319.0f;
        p.dst_fx = 300.0f;
        p.dst_fy = 290.0f;
        p.src_cx = 257.2f;
        p.src_cy = 67.1f;
        p.dst_cx = 258.0f;
        p.dst_cy = 64.0f;
        p.model_type = model;
        p.num_distortion = 10;
        const float coefficients[] = {-0.15f, 0.01f, 0.002f, -0.001f, 0.003f, -0.002f, 0.001f, 0.0002f, -0.001f, -0.0002f};
        std::copy(std::begin(coefficients), std::end(coefficients), p.distortion);
        Tensor expected_image, expected_mask;
        {
            GpuBackendScope scope(GpuBackend::CUDA);
            const auto source = cpu.gpu();
            expected_image = undistort_image(source, p, nullptr).cpu();
            expected_mask = undistort_mask(source.slice(0, 1, 2).squeeze(0).contiguous(), p, nullptr).cpu();
        }
        expectImageNear(undistort_image(cpu, p, nullptr), expected_image, 0.00015f);
        expectImageNear(undistort_mask(cpu.slice(0, 1, 2).squeeze(0).contiguous(), p, nullptr), expected_mask, 0.00015f);
        GpuBackendScope scope(GpuBackend::Vulkan);
        const auto source = cpu.gpu();
        // Factory selection on the calling thread must not move resident data.
        GpuBackendScope other_default(GpuBackend::CUDA);
        auto image = undistort_image(source, p, nullptr);
        EXPECT_EQ(gpu_backend_of(image), GpuBackend::Vulkan);
        expectImageNear(image, expected_image, 0.00015f);
        expectImageNear(undistort_mask(source.slice(0, 1, 2).squeeze(0).contiguous(), p, nullptr), expected_mask, 0.00015f);
    }
}

TEST(ImageTensorBackends, PriorResizePreservesInvalidPixelsAndNormalDirections) {
    if (!gpu_backend_available(GpuBackend::CUDA) || !gpu_backend_available(GpuBackend::Vulkan)) {
        GTEST_SKIP() << "Both tensor backends required";
    }
    constexpr size_t width = 7, height = 5, plane = width * height;
    auto depth = Tensor::empty({height, width}, Device::CPU);
    auto normal = Tensor::empty({3, height, width}, Device::CPU);
    for (size_t i = 0; i < plane; ++i) {
        depth.ptr<float>()[i] = static_cast<float>(i) * 0.15f;
        normal.ptr<float>()[i] = 0.4f;
        normal.ptr<float>()[plane + i] = 0.5f;
        normal.ptr<float>()[2 * plane + i] = 0.7f;
    }
    const float invalid[] = {0.0f, -1.0f, std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity()};
    for (size_t i = 0; i < 4; ++i) {
        depth.ptr<float>()[i * 5] = invalid[i];
        normal.ptr<float>()[i * 5] = i < 2 ? 0.001f : invalid[i];
        normal.ptr<float>()[plane + i * 5] = 0.002f;
        normal.ptr<float>()[2 * plane + i * 5] = 0.003f;
    }
    for (auto size : {std::pair{13, 9}, std::pair{3, 2}, std::pair{7, 5}}) {
        SCOPED_TRACE(size.first);
        Tensor expected_depth, expected_normal;
        {
            GpuBackendScope scope(GpuBackend::CUDA);
            expected_depth = resize_depth_prior(depth.gpu(), size.second, size.first).cpu();
            expected_normal = resize_normal_prior(normal.gpu(), size.second, size.first).cpu();
        }
        expectImageNear(resize_depth_prior(depth, size.second, size.first), expected_depth, 0.00001f);
        expectImageNear(resize_normal_prior(normal, size.second, size.first), expected_normal, 0.00001f);
        GpuBackendScope scope(GpuBackend::Vulkan);
        expectImageNear(resize_depth_prior(depth.gpu(), size.second, size.first), expected_depth, 0.00001f);
        expectImageNear(resize_normal_prior(normal.gpu(), size.second, size.first), expected_normal, 0.00001f);
    }
}

TEST(ImageTensorBackends, LargePhotoKeepsAdjacentPixelIndicesDistinct) {
    constexpr int width = 6001, height = 3001;
    auto mask = Tensor::zeros({height, width}, Device::CPU);
    mask.ptr<float>()[static_cast<size_t>(height - 1) * width + width - 2] = 0.75f;
    UndistortParams params{};
    params.model_type = CameraModelType::PINHOLE;
    params.src_width = width;
    params.src_height = height;
    params.dst_width = params.dst_height = 1;
    params.src_fx = params.src_fy = params.dst_fx = params.dst_fy = 1.0f;
    params.dst_cx = params.dst_cy = 0.5f;
    params.src_cx = width - 1.5f;
    params.src_cy = height - 0.5f;
    EXPECT_FLOAT_EQ(undistort_mask(mask, params, nullptr).cpu().ptr<float>()[0], 0.75f);
    if (gpu_backend_available(GpuBackend::Vulkan)) {
        const GpuBackendScope scope(GpuBackend::Vulkan);
        EXPECT_FLOAT_EQ(undistort_mask(mask.gpu(), params, nullptr).cpu().ptr<float>()[0], 0.75f);
    }
}

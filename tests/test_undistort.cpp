/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/camera.hpp"
#include "core/cuda/undistort/undistort.hpp"
#include "core/image_io.hpp"
#include "io/formats/colmap.hpp"
#include <cuda_runtime.h>
#include <gtest/gtest.h>

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

    // With blank_pixels > 0, the focal length should be smaller (zoomed out)
    EXPECT_LT(loose.dst_fx, tight.dst_fx);
    EXPECT_LT(loose.dst_fy, tight.dst_fy);

    // Both must produce valid results
    validate_params(tight, TEST_W, TEST_H);
    validate_params(loose, TEST_W, TEST_H);
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
        auto& [cams, center] = *result;
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

TEST_F(UndistortCameraTest, NonPinholeModelDetectsDistortion) {
    auto& cam = cameras_[0];
    auto R = cam->R();
    auto T = cam->T();

    // Even with no distortion coefficients, a non-pinhole model means distortion
    Camera equirect_cam(R, T,
                        cam->focal_x(), cam->focal_y(),
                        cam->center_x(), cam->center_y(),
                        Tensor(), Tensor(),
                        CameraModelType::EQUIRECTANGULAR,
                        "test_equirect", cam->image_path(), "",
                        cam->camera_width(), cam->camera_height(), 996);

    EXPECT_TRUE(equirect_cam.has_distortion());
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
    EXPECT_FLOAT_EQ(scaled.dst_cx, scaled.dst_width * 0.5f);
    EXPECT_FLOAT_EQ(scaled.dst_cy, scaled.dst_height * 0.5f);

    run_image_undistort(scaled);
    run_mask_undistort(scaled);
}

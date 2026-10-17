/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// The branch-aware undistort inverse of #3610 on the portable paths: image_warp.slang on Vulkan and
// Metal, which also serves the crop and CPU images.

#include "core/tensor.hpp"
#include "core/tensor_backend.hpp"
#include "core/tensor_image.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <gtest/gtest.h>
#include <utility>
#include <vector>

namespace {
    using namespace lfs::core;

    std::pair<float, float> distort(const float x, const float y, const UndistortParams& params) {
        const float r2 = x * x + y * y;
        if (params.model_type == CameraModelType::PINHOLE) {
            const float numerator = 1.0f + params.distortion[0] * r2 + params.distortion[1] * r2 * r2 +
                                    params.distortion[2] * r2 * r2 * r2;
            const float denominator = params.num_distortion >= 6
                                          ? 1.0f + params.distortion[3] * r2 + params.distortion[4] * r2 * r2 +
                                                params.distortion[5] * r2 * r2 * r2
                                          : 1.0f;
            const float radial = numerator / denominator;
            const int offset = params.num_distortion >= 6 ? 6 : 3;
            const float p1 = params.distortion[offset];
            const float p2 = params.distortion[offset + 1];
            return {x * radial + 2.0f * p1 * x * y + p2 * (r2 + 2.0f * x * x),
                    y * radial + p1 * (r2 + 2.0f * y * y) + 2.0f * p2 * x * y};
        }
        const float r = std::sqrt(r2);
        if (r < 1e-8f)
            return {x, y};
        const float theta = std::atan(r);
        const float uu = x * theta / r;
        const float vv = y * theta / r;
        const float t2 = theta * theta;
        const float t4 = t2 * t2;
        const float radial = params.distortion[0] * t2 + params.distortion[1] * t4 +
                             params.distortion[2] * t4 * t2 + params.distortion[3] * t4 * t4;
        float dx = uu + uu * radial;
        float dy = vv + vv * radial;
        if (params.model_type == CameraModelType::THIN_PRISM_FISHEYE) {
            const float p1 = params.distortion[4];
            const float p2 = params.distortion[5];
            dx += 2.0f * p1 * uu * vv + p2 * (t2 + 2.0f * uu * uu) + params.distortion[6] * t2 + params.distortion[7] * t4;
            dy += 2.0f * p2 * uu * vv + p1 * (t2 + 2.0f * vv * vv) + params.distortion[8] * t2 + params.distortion[9] * t4;
        }
        return {dx, dy};
    }

    // Cameras from #3610: a fisheye whose frame corners lie past the radial fold, the folded
    // OPENCV c0148, and the near-180-degree thin-prism c0250.
    UndistortParams fisheye_fold() {
        return compute_undistort_params(
            1453.56848f, 1452.57385f, 1920.0f, 1440.0f, 3840, 2880,
            Tensor::from_vector({0.159315839f, 0.134339839f, -0.0869080648f, -0.00134019367f}, {4}, Device::CPU),
            Tensor(), CameraModelType::FISHEYE);
    }

    UndistortParams pinhole_fold() {
        return compute_undistort_params(
            2051.9519f, 2052.85376f, 341.5f, 513.0f, 683, 1026,
            Tensor::from_vector({0.476160347f, -15.2258253f}, {2}, Device::CPU),
            Tensor::from_vector({0.0117965238f, -0.0146329654f}, {2}, Device::CPU), CameraModelType::PINHOLE);
    }

    UndistortParams thin_prism_wide() {
        return compute_undistort_params(
            1188.64233f, 1188.78235f, 1920.0f, 1920.0f, 3840, 3840,
            Tensor::from_vector({-0.00961296912f, -0.00112110213f, -0.00146211369f, -0.000204068245f}, {4}, Device::CPU),
            Tensor::from_vector({0.000781554671f, 0.000780893257f, -0.0034037116f, -0.00189556787f}, {4}, Device::CPU),
            CameraModelType::THIN_PRISM_FISHEYE);
    }

    UndistortParams full_thin_prism() {
        return compute_undistort_params(
            1000.0f, 1000.0f, 2000.0f, 2000.0f, 4000, 4000,
            Tensor::from_vector({0.1f, -0.02f, 0.003f, -0.001f}, {4}, Device::CPU),
            Tensor::from_vector({0.002f, -0.0015f, 0.003f, -0.002f}, {4}, Device::CPU),
            CameraModelType::THIN_PRISM_FISHEYE);
    }

    void expect_crop_border_inside(const UndistortParams& params) {
        ASSERT_FALSE(params.crop_solve_failed);
        for (int edge = 0; edge < 4; ++edge) {
            const int count = edge < 2 ? params.dst_height : params.dst_width;
            for (int i = 0; i < count; ++i) {
                const float px = edge < 2 ? (edge == 0 ? 0.5f : params.dst_width - 0.5f) : i + 0.5f;
                const float py = edge >= 2 ? (edge == 2 ? 0.5f : params.dst_height - 0.5f) : i + 0.5f;
                const float x = (px - params.dst_cx) / params.dst_fx;
                const float y = (py - params.dst_cy) / params.dst_fy;
                const auto [dx, dy] = distort(x, y, params);
                const auto [ex, ey] = distort(x * 1.0001f, y * 1.0001f, params);
                const float sx = dx * params.src_fx + params.src_cx;
                const float sy = dy * params.src_fy + params.src_cy;
                ASSERT_TRUE(std::isfinite(sx) && std::isfinite(sy) && sx >= 0.0f && sx <= params.src_width &&
                            sy >= 0.0f && sy <= params.src_height && (ex - dx) * x + (ey - dy) * y > 0.0f)
                    << "edge " << edge << " pixel " << i << " source " << sx << ", " << sy;
            }
        }
    }

    TEST(UndistortCropPortable, FisheyeFoldKeepsClimbingFieldOfView) {
        const auto params = fisheye_fold();
        EXPECT_EQ(params.dst_width, 5388);
        EXPECT_EQ(params.dst_height, 3360);
        expect_crop_border_inside(params);
        const auto capped = compute_undistort_grid(params, 1, 3840);
        EXPECT_EQ(capped.width, 3840);
        EXPECT_EQ(capped.height, 2395);
    }

    TEST(UndistortCropPortable, PinholeFoldKeepsBorderInsideSource) {
        expect_crop_border_inside(pinhole_fold());
    }

    TEST(UndistortCropPortable, ThinPrismWithoutFrameBorderRaysReachesClamp) {
        const auto params = thin_prism_wide();
        EXPECT_EQ(params.dst_width, 7680);
        EXPECT_EQ(params.dst_height, 7680);
        expect_crop_border_inside(params);
    }

    class UndistortInversePortable : public testing::TestWithParam<GpuBackend> {
    protected:
        void SetUp() override {
            if (!gpu_backend_available(GetParam()))
                GTEST_SKIP() << "Backend unavailable";
        }

        // One-pixel sample map centred on (x, y) through the backend's image warp.
        bool point(UndistortParams params, const float x, const float y, float& ux, float& uy) const {
            const GpuBackendScope scope(GetParam());
            params.src_cx -= x - 0.5f;
            params.src_cy -= y - 0.5f;
            params.src_width = params.src_height = 1;
            const auto sample = inverse_distortion_sample_map(params, nullptr).cpu().contiguous();
            ux = sample.ptr<float>()[0];
            uy = sample.ptr<float>()[1];
            return std::isfinite(ux) && std::isfinite(uy);
        }
    };

    // Catches an inverse that solves rays past the first radial fold, as the old seed did at 1.63 fx.
    TEST_P(UndistortInversePortable, FisheyeFoldRejectsRaysPastTheFold) {
        const auto params = fisheye_fold();
        const auto [dx, dy] = distort(std::tan(1.35f), 0.0f, params);
        const float x = dx * params.src_fx + params.src_cx;
        const float y = dy * params.src_fy + params.src_cy;
        const float fold_x = params.src_cx + 1.63f * params.src_fx;
        float ux, uy;
        ASSERT_TRUE(point(params, x, y, ux, uy));
        EXPECT_NEAR(std::atan(std::hypot(ux, uy)), 1.35f, 1e-4f);
        EXPECT_FALSE(point(params, fold_x, params.src_cy, ux, uy));
    }

    // Catches the shared inverse rejecting or mis-solving wide thin-prism rays.
    TEST_P(UndistortInversePortable, ThinPrismWideRaysRoundTrip) {
        const auto params = full_thin_prism();
        for (const float theta : {1.0f, 1.3f, 1.45f, 1.5f, 1.54f}) {
            for (const float azimuth : {0.0f, 1.1f, 2.5f, 4.2f}) {
                const float r = std::tan(theta);
                const auto [dx, dy] = distort(r * std::cos(azimuth), r * std::sin(azimuth), params);
                const float x = dx * params.src_fx + params.src_cx;
                const float y = dy * params.src_fy + params.src_cy;
                float nx, ny;
                ASSERT_TRUE(point(params, x, y, nx, ny)) << "theta " << theta << " azimuth " << azimuth;
                const auto [rx, ry] = distort(nx, ny, params);
                EXPECT_NEAR(rx * params.src_fx + params.src_cx, x, 1.0e-2f) << "theta " << theta << " azimuth " << azimuth;
                EXPECT_NEAR(ry * params.src_fy + params.src_cy, y, 1.0e-2f) << "theta " << theta << " azimuth " << azimuth;
            }
        }
    }

    // Catches sample-map rays that do not project back onto their pixels, or rays past the fold of a
    // radial-only lens.
    TEST_P(UndistortInversePortable, SampleMapRaysProjectBackOntoTheirPixels) {
        for (const auto& params : {fisheye_fold(), pinhole_fold(), thin_prism_wide()}) {
            auto map = params;
            map.src_width = 48;
            map.src_height = 36;
            map.src_fx *= 48.0f / params.src_width;
            map.src_fy *= 36.0f / params.src_height;
            map.src_cx *= 48.0f / params.src_width;
            map.src_cy *= 36.0f / params.src_height;
            const auto branch = undistort_radial_branch(map);
            const GpuBackendScope scope(GetParam());
            const auto samples = inverse_distortion_sample_map(map, nullptr).cpu().contiguous();
            const float* values = samples.ptr<float>();
            int valid = 0;
            for (int y = 0; y < map.src_height; ++y) {
                for (int x = 0; x < map.src_width; ++x) {
                    const float gx = values[2 * (y * map.src_width + x)];
                    const float gy = values[2 * (y * map.src_width + x) + 1];
                    const float xd = (x + 0.5f - map.src_cx) / map.src_fx;
                    const float yd = (y + 0.5f - map.src_cy) / map.src_fy;
                    const bool ray = std::isfinite(gx) && std::isfinite(gy);
                    const auto where = std::format("model {} pixel {},{}", int(map.model_type), x, y);
                    const float rd = std::hypot(xd, yd);
                    if (map.model_type == CameraModelType::FISHEYE && std::abs(rd - branch.maximum) > 1e-4f * branch.maximum)
                        EXPECT_EQ(ray, rd < branch.maximum) << where;
                    if (!ray)
                        continue;
                    ++valid;
                    const auto [dx, dy] = distort(gx, gy, map);
                    EXPECT_LT(std::hypot((dx - xd) * map.src_fx, (dy - yd) * map.src_fy), 1e-2f)
                        << where << " ray " << gx << "," << gy;
                }
            }
            EXPECT_GT(valid, map.src_width * map.src_height / 4) << "model " << int(map.model_type);
        }
    }

    // Catches CPU images losing the GPU inverse: their inverse warps round-trip through it.
    TEST_P(UndistortInversePortable, CpuImagesMatchTheGpuInverse) {
        auto params = fisheye_fold();
        params = prepare_undistort_params(params, 96, 72, 1, 64);
        std::vector<float> values(size_t(3) * params.dst_height * params.dst_width);
        for (size_t i = 0; i < values.size(); ++i)
            values[i] = float(i % 29) / 29.0f;
        const auto host = Tensor::from_vector(values, {3, size_t(params.dst_height), size_t(params.dst_width)}, Device::CPU);
        const GpuBackendScope scope(GetParam());
        Tensor gpu_validity, cpu_validity;
        const auto gpu = distort_image_to_source(host.to(Device::GPU), params, gpu_validity, nullptr).cpu();
        const auto cpu = distort_image_to_source(host, params, cpu_validity, nullptr);
        ASSERT_EQ(cpu.device(), Device::CPU);
        ASSERT_EQ(cpu_validity.device(), Device::CPU);
        EXPECT_EQ(cpu.to_vector(), gpu.to_vector());
        EXPECT_EQ(cpu_validity.to(DataType::Float32).to_vector(), gpu_validity.cpu().to(DataType::Float32).to_vector());
        const auto mask = host.slice(0, 0, 1).squeeze(0).contiguous();
        EXPECT_EQ(distort_mask_to_source_area(mask, params, nullptr).to_vector(),
                  distort_mask_to_source_area(mask.to(Device::GPU), params, nullptr).cpu().to_vector());
    }

    INSTANTIATE_TEST_SUITE_P(Backends, UndistortInversePortable, testing::Values(GpuBackend::Vulkan, GpuBackend::Metal),
                             [](const auto& info) { return info.param == GpuBackend::Metal ? "Metal" : "Vulkan"; });
} // namespace

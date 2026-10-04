/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/camera.hpp"
#include "core/cuda/lanczos_resize/lanczos_resize.hpp"
#include "core/cuda/undistort/undistort.hpp"
#include "core/image_io.hpp"
#include "core/tensor_backend.hpp"
#include "io/formats/colmap.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <iostream>
#include <limits>
#include <numbers>
#include <random>
#include <utility>
#include <vector>

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

        auto dst = undistort_mask_area(src, params, nullptr);
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

    std::pair<float, float> direct_full_opencv_distortion(
        const float x,
        const float y,
        const std::array<float, 6>& radial,
        const std::array<float, 2>& tangential) {
        const float r2 = x * x + y * y;
        const float r4 = r2 * r2;
        const float r6 = r4 * r2;
        const float numerator =
            1.0f + radial[0] * r2 + radial[1] * r4 + radial[2] * r6;
        const float denominator =
            1.0f + radial[3] * r2 + radial[4] * r4 + radial[5] * r6;
        const float scale = numerator / denominator;
        const float p1 = tangential[0];
        const float p2 = tangential[1];
        return {
            x * scale + 2.0f * p1 * x * y + p2 * (r2 + 2.0f * x * x),
            y * scale + p1 * (r2 + 2.0f * y * y) + 2.0f * p2 * x * y};
    }

    // COLMAP ThinPrismFisheyeCameraModel::ImgFromCam, extra params {k1, k2, p1, p2, k3, k4, sx1, sy1}.
    std::pair<double, double> colmap_thin_prism_fisheye_reference(
        const double u, const double v, const std::array<double, 8>& extra) {
        const double r = std::sqrt(u * u + v * v);
        double uu = u;
        double vv = v;
        if (r > std::numeric_limits<double>::epsilon()) {
            const double theta = std::atan(r);
            uu = theta * u / r;
            vv = theta * v / r;
        }
        const auto [k1, k2, p1, p2, k3, k4, sx1, sy1] = extra;
        const double u2 = uu * uu;
        const double uv = uu * vv;
        const double v2 = vv * vv;
        const double r2 = u2 + v2;
        const double r4 = r2 * r2;
        const double r6 = r4 * r2;
        const double r8 = r6 * r2;
        const double radial = k1 * r2 + k2 * r4 + k3 * r6 + k4 * r8;
        return {uu + uu * radial + 2.0 * p1 * uv + p2 * (r2 + 2.0 * u2) + sx1 * r2,
                vv + vv * radial + 2.0 * p2 * uv + p1 * (r2 + 2.0 * v2) + sy1 * r2};
    }

    std::pair<float, float> distort_test_coordinate(
        const float x, const float y, const UndistortParams& params) {
        const float r2 = x * x + y * y;
        if (params.model_type == CameraModelType::PINHOLE) {
            const float numerator = 1.0f + params.distortion[0] * r2 +
                                    params.distortion[1] * r2 * r2 +
                                    params.distortion[2] * r2 * r2 * r2;
            const float denominator = params.num_distortion >= 6
                                          ? 1.0f + params.distortion[3] * r2 +
                                                params.distortion[4] * r2 * r2 +
                                                params.distortion[5] * r2 * r2 * r2
                                          : 1.0f;
            const float radial = numerator / denominator;
            const int tangential_offset = params.num_distortion >= 6 ? 6 : 3;
            const float p1 = params.distortion[tangential_offset];
            const float p2 = params.distortion[tangential_offset + 1];
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
            dx += 2.0f * p1 * uu * vv + p2 * (t2 + 2.0f * uu * uu) +
                  params.distortion[6] * t2 + params.distortion[7] * t4;
            dy += 2.0f * p2 * uu * vv + p1 * (t2 + 2.0f * vv * vv) +
                  params.distortion[8] * t2 + params.distortion[9] * t4;
        }
        return {dx, dy};
    }

    constexpr std::array<float, 4> FULL_THIN_PRISM_RADIAL = {0.1f, -0.02f, 0.003f, -0.001f};
    constexpr std::array<float, 4> FULL_THIN_PRISM_TANGENTIAL = {0.002f, -0.0015f, 0.003f, -0.002f};

    UndistortParams make_full_thin_prism_params(
        const float fx, const float fy, const float cx, const float cy,
        const int width, const int height) {
        const auto radial = Tensor::from_vector(
            std::vector<float>(FULL_THIN_PRISM_RADIAL.begin(), FULL_THIN_PRISM_RADIAL.end()),
            TensorShape({4}), Device::CPU);
        const auto tangential = Tensor::from_vector(
            std::vector<float>(FULL_THIN_PRISM_TANGENTIAL.begin(), FULL_THIN_PRISM_TANGENTIAL.end()),
            TensorShape({4}), Device::CPU);
        return compute_undistort_params(
            fx, fy, cx, cy, width, height, radial, tangential, CameraModelType::THIN_PRISM_FISHEYE);
    }

    std::array<double, 8> full_thin_prism_colmap_extra() {
        const auto& k = FULL_THIN_PRISM_RADIAL;
        const auto& t = FULL_THIN_PRISM_TANGENTIAL;
        return {k[0], k[1], t[0], t[1], k[2], k[3], t[2], t[3]};
    }

    // Normalized points at incidence angles up to 1.45 rad in several azimuths.
    std::vector<std::pair<float, float>> wide_angle_sample_points() {
        std::vector<std::pair<float, float>> points;
        for (const float theta : {0.3f, 0.8f, 1.2f, 1.45f}) {
            const float r = std::tan(theta);
            for (const float azimuth : {0.0f, 1.1f, 2.5f, 4.2f})
                points.emplace_back(r * std::cos(azimuth), r * std::sin(azimuth));
        }
        return points;
    }

    int source_column_at_angle(const UndistortParams& params, const float theta) {
        float dx, dy;
        distort_normalized_point(params, std::tan(theta), 0.0f, dx, dy);
        return static_cast<int>(std::floor(dx * params.src_fx + params.src_cx));
    }

    // Round trip through the GPU inverse for every source column between incidence angles 1.3 and
    // 1.5 rad on a row near the optical axis. Undistorted focal lengths at or above the source keep
    // the Lanczos ramp phase error (up to 0.018 px) from being amplified on the way back.
    void expect_thin_prism_wide_angle_round_trip(const std::array<float, 10>& packed) {
        UndistortParams params{};
        params.model_type = CameraModelType::THIN_PRISM_FISHEYE;
        params.src_width = 1792;
        params.src_height = 8;
        params.src_fx = params.src_fy = 1000.0f;
        params.src_cx = 32.0f;
        params.src_cy = 4.0f;
        params.dst_width = 14336;
        params.dst_height = 48;
        params.dst_fx = 1000.0f;
        params.dst_fy = 2000.0f;
        params.dst_cx = 8.0f;
        params.dst_cy = 24.0f;
        std::copy(packed.begin(), packed.end(), params.distortion);
        params.num_distortion = static_cast<int>(packed.size());

        const size_t plane = static_cast<size_t>(params.dst_width) * params.dst_height;
        std::vector<float> coordinates(3 * plane, 0.0f);
        for (int y = 0; y < params.dst_height; ++y) {
            for (int x = 0; x < params.dst_width; ++x) {
                const size_t i = static_cast<size_t>(y) * params.dst_width + x;
                coordinates[i] = static_cast<float>(x);
                coordinates[plane + i] = static_cast<float>(y);
            }
        }
        auto input = Tensor::from_vector(
            coordinates, TensorShape({3, 48, 14336}), Device::GPU);
        Tensor validity;
        const auto output = distort_image_to_source(input, params, validity, nullptr);
        ASSERT_EQ(output.shape(), (TensorShape({3, 8, 1792})));
        ASSERT_EQ(validity.shape(), (TensorShape({8, 1792})));

        const auto output_cpu = output.cpu().contiguous();
        const auto validity_cpu = validity.cpu().contiguous();
        const float* const mapped_x = output_cpu.ptr<float>();
        const float* const mapped_y = output_cpu.ptr<float>() + 8 * 1792;
        const uint8_t* const valid = validity_cpu.ptr<uint8_t>();

        constexpr int row = 3;
        const int first_column = source_column_at_angle(params, 1.3f);
        const int last_column = source_column_at_angle(params, 1.5f);
        ASSERT_LT(first_column, last_column);
        ASSERT_LT(last_column, params.src_width);
        for (int x = first_column; x <= last_column; ++x) {
            const size_t i = static_cast<size_t>(row) * 1792 + x;
            ASSERT_EQ(valid[i], 1) << "column " << x;
            const float ux = (mapped_x[i] + 0.5f - params.dst_cx) / params.dst_fx;
            const float uy = (mapped_y[i] + 0.5f - params.dst_cy) / params.dst_fy;
            float dx, dy;
            distort_normalized_point(params, ux, uy, dx, dy);
            EXPECT_NEAR(dx * params.src_fx + params.src_cx, x + 0.5f, 1.0e-2f) << "column " << x;
            EXPECT_NEAR(dy * params.src_fy + params.src_cy, row + 0.5f, 1.0e-2f) << "column " << x;
            if (x == last_column)
                EXPECT_GT(std::atan(std::hypot(ux, uy)), 1.49f);
        }
    }

    double lanczos3(const double x) {
        const double absolute_x = std::abs(x);
        if (absolute_x >= 3.0)
            return 0.0;
        if (absolute_x < 1.0e-12)
            return 1.0;
        const double pi_x = std::numbers::pi * x;
        return std::sin(pi_x) / pi_x * std::sin(pi_x / 3.0) / (pi_x / 3.0);
    }

    float reconstruct_lanczos3(
        const std::vector<float>& source,
        const int width,
        const int height,
        const double sample_x,
        const double sample_y) {
        const int begin_x = static_cast<int>(std::ceil(sample_x - 3.0));
        const int end_x = static_cast<int>(std::floor(sample_x + 3.0));
        const int begin_y = static_cast<int>(std::ceil(sample_y - 3.0));
        const int end_y = static_cast<int>(std::floor(sample_y + 3.0));
        double value = 0.0;
        double weight_sum = 0.0;
        for (int y = begin_y; y <= end_y; ++y) {
            const double wy = lanczos3(sample_y - static_cast<double>(y));
            for (int x = begin_x; x <= end_x; ++x) {
                const double weight = wy * lanczos3(sample_x - static_cast<double>(x));
                if (x < 0 || x >= width || y < 0 || y >= height)
                    continue;
                value += weight * source[static_cast<size_t>(y) * width + x];
                weight_sum += weight;
            }
        }
        return std::abs(weight_sum) > 1.0e-12
                   ? static_cast<float>(value / weight_sum)
                   : 0.0f;
    }

    std::vector<float> oracle_undistort_lanczos3_8x8(
        const std::vector<float>& source,
        const UndistortParams& params) {
        constexpr int quadrature = 8;
        std::vector<float> result(
            static_cast<size_t>(params.dst_width) * params.dst_height, 0.0f);
        for (int oy = 0; oy < params.dst_height; ++oy) {
            for (int ox = 0; ox < params.dst_width; ++ox) {
                double value = 0.0;
                for (int qy = 0; qy < quadrature; ++qy) {
                    for (int qx = 0; qx < quadrature; ++qx) {
                        const double px = ox + (qx + 0.5) / quadrature;
                        const double py = oy + (qy + 0.5) / quadrature;
                        const float nx = static_cast<float>((px - params.dst_cx) / params.dst_fx);
                        const float ny = static_cast<float>((py - params.dst_cy) / params.dst_fy);
                        const auto [dx, dy] = distort_test_coordinate(nx, ny, params);
                        const double sx = dx * params.src_fx + params.src_cx - 0.5;
                        const double sy = dy * params.src_fy + params.src_cy - 0.5;
                        value += reconstruct_lanczos3(
                            source, params.src_width, params.src_height, sx, sy);
                    }
                }
                result[static_cast<size_t>(oy) * params.dst_width + ox] =
                    static_cast<float>(value / (quadrature * quadrature));
            }
        }
        return result;
    }

    double laplacian_energy(
        const float* const image, const int width, const int height) {
        double energy = 0.0;
        size_t samples = 0;
        for (int y = 1; y + 1 < height; ++y) {
            for (int x = 1; x + 1 < width; ++x) {
                const size_t index = static_cast<size_t>(y) * width + x;
                const double laplacian =
                    -4.0 * image[index] + image[index - 1] + image[index + 1] +
                    image[index - width] + image[index + width];
                energy += laplacian * laplacian;
                ++samples;
            }
        }
        return energy / static_cast<double>(samples);
    }

    void expect_inverse_round_trip(const CameraModelType model) {
        auto radial = Tensor::from_vector(
            model == CameraModelType::PINHOLE
                ? std::vector<float>{-0.08f, 0.01f}
                : std::vector<float>{0.01f, -0.001f, 0.0001f, 0.0f},
            TensorShape({model == CameraModelType::PINHOLE ? 2u : 4u}), Device::CPU);
        Tensor tangential;
        if (model == CameraModelType::PINHOLE) {
            tangential = Tensor::from_vector({0.001f, -0.0008f}, TensorShape({2}), Device::CPU);
        } else if (model == CameraModelType::THIN_PRISM_FISHEYE) {
            tangential = Tensor::from_vector(
                {0.001f, -0.0008f, 0.0001f, -0.0001f, 0.00001f, -0.00001f},
                TensorShape({6}), Device::CPU);
        }
        const auto params = compute_undistort_params(
            450.0f, 455.0f, 160.0f, 120.0f, 320, 240,
            radial, tangential, model);

        const size_t plane = static_cast<size_t>(params.dst_width) * params.dst_height;
        std::vector<float> coordinates(3 * plane, 0.0f);
        for (int y = 0; y < params.dst_height; ++y) {
            for (int x = 0; x < params.dst_width; ++x) {
                const size_t i = static_cast<size_t>(y) * params.dst_width + x;
                coordinates[i] = static_cast<float>(x);
                coordinates[plane + i] = static_cast<float>(y);
            }
        }
        auto input = Tensor::from_vector(
            coordinates,
            TensorShape({3, static_cast<size_t>(params.dst_height),
                         static_cast<size_t>(params.dst_width)}),
            Device::GPU);
        Tensor validity;
        const auto output = distort_image_to_source(input, params, validity, nullptr);
        ASSERT_EQ(output.shape(), (TensorShape({3, 240, 320})));
        ASSERT_EQ(validity.shape(), (TensorShape({240, 320})));
        ASSERT_EQ(validity.dtype(), DataType::UInt8);

        const auto output_cpu = output.cpu().contiguous();
        const auto validity_cpu = validity.cpu().contiguous();
        const float* const mapped_x = output_cpu.ptr<float>();
        const float* const mapped_y = output_cpu.ptr<float>() + 320 * 240;
        const uint8_t* const valid = validity_cpu.ptr<uint8_t>();
        int checked = 0;
        for (int y = 4; y < 236; y += 11) {
            for (int x = 4; x < 316; x += 11) {
                const size_t i = static_cast<size_t>(y) * 320 + x;
                if (valid[i] == 0)
                    continue;
                const float ux = (mapped_x[i] + 0.5f - params.dst_cx) / params.dst_fx;
                const float uy = (mapped_y[i] + 0.5f - params.dst_cy) / params.dst_fy;
                const auto [dx, dy] = distort_test_coordinate(ux, uy, params);
                const float round_trip_x = dx * params.src_fx + params.src_cx;
                const float round_trip_y = dy * params.src_fy + params.src_cy;
                // Area sampling returns the footprint centroid, which differs from the centre
                // point by a second-order term; 1e-2 px matches the inverse acceptance.
                EXPECT_LE(std::hypot(round_trip_x - (x + 0.5f),
                                     round_trip_y - (y + 0.5f)),
                          1.0e-2f);
                ++checked;
            }
        }
        EXPECT_GT(checked, 100);
    }

} // namespace

// Catches a fixed sample lattice aliasing under strong minification: with eight samples per axis over
// 32 source pixels, every sample lands on the same phase of a four-pixel stripe pattern.
TEST(UndistortResampling, StrongMinificationAveragesTheWholeFootprint) {
    constexpr int source_width = 512;
    constexpr int source_height = 64;
    constexpr int factor = 32;
    UndistortParams params{};
    params.model_type = CameraModelType::PINHOLE;
    params.src_width = source_width;
    params.src_height = source_height;
    params.src_fx = params.src_fy = 400.0f;
    params.src_cx = 0.5f * source_width;
    params.src_cy = 0.5f * source_height;
    params.dst_width = source_width / factor;
    params.dst_height = source_height / factor;
    params.dst_fx = params.dst_fy = params.src_fx / factor;
    params.dst_cx = params.src_cx / factor;
    params.dst_cy = params.src_cy / factor;

    std::vector<float> stripes(static_cast<size_t>(source_width) * source_height);
    for (size_t i = 0; i < stripes.size(); ++i)
        stripes[i] = (i % 4 == 1 || i % 4 == 2) ? 1.0f : 0.0f;
    const auto input = Tensor::from_vector(
        stripes, {1, static_cast<size_t>(source_height), static_cast<size_t>(source_width)}, Device::GPU);
    const auto output = undistort_image(input, params, nullptr).cpu().contiguous();
    ASSERT_EQ(output.numel(), static_cast<size_t>(params.dst_width * params.dst_height));
    const float* const values = output.ptr<float>();
    for (size_t i = 0; i < output.numel(); ++i)
        EXPECT_NEAR(values[i], 0.5f, 0.02f) << i;
}
TEST(UndistortResampling, MatchesEightByEightOracle) {
    const auto native = compute_undistort_params(
        187.5f, 187.5f, 96.0f, 54.0f, 192, 108,
        Tensor::from_vector({0.10f, -0.33f, 0.85f}, {3}, Device::CPU),
        Tensor::from_vector({0.001f, -0.001f}, {2}, Device::CPU),
        CameraModelType::PINHOLE, 0.0f);
    const auto params = prepare_undistort_params(native, 192, 108, 1, 120);

    std::vector<float> source(static_cast<size_t>(params.src_width) * params.src_height);
    for (int y = 0; y < params.src_height; ++y) {
        for (int x = 0; x < params.src_width; ++x) {
            const float value = 0.5f + 0.22f * std::sin(2.0f * std::numbers::pi_v<float> * 0.18f * x) +
                                0.17f * std::cos(2.0f * std::numbers::pi_v<float> * 0.11f * y);
            source[static_cast<size_t>(y) * params.src_width + x] = std::clamp(value, 0.0f, 1.0f);
        }
    }
    const auto reference = oracle_undistort_lanczos3_8x8(source, params);
    const auto input = Tensor::from_vector(
        source, {1, static_cast<size_t>(params.src_height), static_cast<size_t>(params.src_width)},
        Device::GPU);
    const auto output = undistort_image(input, params, nullptr).cpu().contiguous();
    const float* const actual = output.ptr<float>();
    double squared_error = 0.0;
    for (size_t i = 0; i < reference.size(); ++i) {
        const double difference = static_cast<double>(actual[i]) - reference[i];
        squared_error += difference * difference;
    }
    const double psnr = -10.0 * std::log10(squared_error / static_cast<double>(reference.size()));
    const double hf_ratio = laplacian_energy(actual, params.dst_width, params.dst_height) /
                            laplacian_energy(reference.data(), params.dst_width, params.dst_height);
    EXPECT_GE(psnr, 58.0);
    EXPECT_GE(hf_ratio, 0.95);
    EXPECT_LE(hf_ratio, 1.10);
}

TEST(UndistortResampling, CpuLanczosPreservesZerosAndNearIntegerBorderWeights) {
    const std::vector<float> source = {0.2f, 0.3f, 0.4f,
                                       0.8f, 0.7f, 0.6f,
                                       0.4f, 0.5f, 0.6f};
    const auto input = Tensor::from_vector(source, {1, 3, 3}, Device::CPU);
    UndistortParams p{};
    p.model_type = CameraModelType::PINHOLE;
    p.src_width = p.src_height = 3;
    p.dst_width = p.dst_height = 1;
    // Keep the entire output footprint at the same float source coordinate.
    p.src_fx = p.src_fy = 1.0e-9f;
    p.dst_fx = p.dst_fy = 1.0f;
    p.dst_cx = p.dst_cy = 0.5f;
    p.src_cx = 1.5f;
    for (const float y : {-1.0f, -1.00000286102294921875f, -0.99999713897705078125f}) {
        SCOPED_TRACE(y);
        p.src_cy = y + 0.5f;
        const float value = undistort_image(input, p, nullptr).item<float>();
        if (y == -1.0f) {
            // All in-bounds taps are sinc zeros; use the nearest border pixel.
            EXPECT_NEAR(value, source[1], 1.0e-6f);
        } else {
            EXPECT_NEAR(value, reconstruct_lanczos3(source, 3, 3, 1.0, y), 1.0e-6f);
        }
    }
}

TEST(UndistortResampling, PreservesConstantWithDistortionAndBorderRenormalization) {
    const auto native = compute_undistort_params(
        62.5f, 62.5f, 32.0f, 20.0f, 64, 40,
        Tensor::from_vector({0.10f, -0.33f, 0.85f}, {3}, Device::CPU),
        Tensor::from_vector({0.001f, -0.001f}, {2}, Device::CPU),
        CameraModelType::PINHOLE, 0.0f);
    const auto params = prepare_undistort_params(native, 64, 40, 1, 40);
    const auto source = Tensor::ones({3, 40, 64}, Device::GPU) * 0.625f;
    const auto output = undistort_image(source, params, nullptr);
    EXPECT_LE((output - 0.625f).abs().max().item<float>(), 2.0e-5f);
}

TEST(UndistortResampling, IdentityIsExact) {
    UndistortParams params{};
    params.src_width = params.dst_width = 37;
    params.src_height = params.dst_height = 23;
    params.src_fx = params.dst_fx = 31.25f;
    params.src_fy = params.dst_fy = 30.75f;
    params.src_cx = params.dst_cx = 18.25f;
    params.src_cy = params.dst_cy = 11.75f;
    params.model_type = CameraModelType::PINHOLE;
    const auto source = Tensor::randn({3, 23, 37}, Device::GPU);
    const auto output = undistort_image(source, params, nullptr);
    const auto difference = (source - output).abs().max().item<float>();
    EXPECT_EQ(difference, 0.0f);
}

TEST(UndistortGrid, UsesFinalIntegerDimensionsForIntrinsics) {
    UndistortParams params{};
    params.src_width = 8192;
    params.src_height = 4320;
    params.src_fx = params.dst_fx = 8000.0f;
    params.src_fy = params.dst_fy = 8000.0f;
    params.src_cx = params.dst_cx = 4096.0f;
    params.src_cy = params.dst_cy = 2160.0f;
    params.dst_width = 7963;
    params.dst_height = 4217;
    const auto grid = compute_undistort_grid(params, 1, 5120);
    EXPECT_EQ(grid.width, 5120);
    EXPECT_EQ(grid.height, 2711);
    const auto scaled = prepare_undistort_params(params, 8192, 4320, 1, 5120);
    EXPECT_FLOAT_EQ(scaled.dst_fx, params.dst_fx * 5120.0f / 7963.0f);
    EXPECT_FLOAT_EQ(scaled.dst_fy, params.dst_fy * 2711.0f / 4217.0f);
    EXPECT_FLOAT_EQ(scaled.dst_cx, params.dst_cx * 5120.0f / 7963.0f);
    EXPECT_FLOAT_EQ(scaled.dst_cy, params.dst_cy * 2711.0f / 4217.0f);
}

TEST(UndistortSidecars, ShareGridPreserveRangesAndNormalizeVectors) {
    const auto native = compute_undistort_params(
        62.5f, 62.5f, 32.0f, 20.0f, 64, 40,
        Tensor::from_vector({0.10f, -0.33f, 0.85f}, {3}, Device::CPU),
        Tensor::from_vector({0.001f, -0.001f}, {2}, Device::CPU),
        CameraModelType::PINHOLE, 0.0f);
    const auto params = prepare_undistort_params(native, 64, 40, 1, 40);

    std::vector<float> mask_values(64 * 40);
    std::vector<float> depth_values(64 * 40);
    std::vector<float> normal_values(3 * 64 * 40, 0.0f);
    for (int y = 0; y < 40; ++y) {
        for (int x = 0; x < 64; ++x) {
            const size_t index = static_cast<size_t>(y) * 64 + x;
            mask_values[index] = x < 32 ? 0.0f : 1.0f;
            depth_values[index] = x < 32 ? 1.0f : 10.0f;
            normal_values[index] = x < 32 ? 1.0f : 0.0f;
            normal_values[2 * 64 * 40 + index] = x < 32 ? 0.0f : 1.0f;
        }
    }
    const auto mask = undistort_mask_area(
        Tensor::from_vector(mask_values, {40, 64}, Device::GPU), params, nullptr);
    const auto depth = undistort_depth_area(
        Tensor::from_vector(depth_values, {40, 64}, Device::GPU), params, nullptr);
    const auto normal = undistort_normal_area(
        Tensor::from_vector(normal_values, {3, 40, 64}, Device::GPU), params, nullptr);

    const TensorShape scalar_shape = {
        static_cast<size_t>(params.dst_height), static_cast<size_t>(params.dst_width)};
    EXPECT_EQ(mask.shape(), scalar_shape);
    EXPECT_EQ(depth.shape(), scalar_shape);
    EXPECT_EQ(normal.shape(), (TensorShape({3, scalar_shape[0], scalar_shape[1]})));
    EXPECT_GE(mask.min().item<float>(), 0.0f);
    EXPECT_LE(mask.max().item<float>(), 1.0f);
    EXPECT_GE(depth.min().item<float>(), 1.0f);
    EXPECT_LE(depth.max().item<float>(), 10.0f);

    const auto normal_cpu = normal.cpu().contiguous();
    const float* const values = normal_cpu.ptr<float>();
    const size_t plane = scalar_shape[0] * scalar_shape[1];
    for (size_t index = 0; index < plane; ++index) {
        const float length = std::sqrt(
            values[index] * values[index] +
            values[plane + index] * values[plane + index] +
            values[2 * plane + index] * values[2 * plane + index]);
        EXPECT_NEAR(length, 1.0f, 2.0e-5f);
    }
}

TEST(UndistortInverse, HighQualityValidityRequiresFullLanczosCoverage) {
    UndistortParams params{};
    params.src_fx = params.src_fy = 32.0f;
    params.src_cx = params.src_cy = 32.0f;
    params.dst_fx = params.dst_fy = 32.0f;
    params.dst_cx = params.dst_cy = 16.0f;
    params.src_width = params.src_height = 64;
    params.dst_width = params.dst_height = 32;
    params.model_type = CameraModelType::PINHOLE;

    Tensor validity;
    const auto output = distort_image_to_source(
        Tensor::ones({3, 32, 32}, Device::GPU), params, validity, nullptr);
    EXPECT_EQ(output.shape(), (TensorShape({3, 64, 64})));
    const auto mask_cpu = validity.cpu().contiguous();
    const uint8_t* const mask = mask_cpu.ptr<uint8_t>();
    EXPECT_EQ(mask[32 * 64 + 18], 0);
    EXPECT_EQ(mask[32 * 64 + 19], 1);
    EXPECT_EQ(mask[32 * 64 + 44], 1);
    EXPECT_EQ(mask[32 * 64 + 45], 0);
}

TEST(UndistortInverse, PinholeRoundTrip) {
    expect_inverse_round_trip(CameraModelType::PINHOLE);
}

TEST(UndistortInverse, FisheyeRoundTrip) {
    expect_inverse_round_trip(CameraModelType::FISHEYE);
}

TEST(UndistortInverse, ThinPrismFisheyeRoundTrip) {
    expect_inverse_round_trip(CameraModelType::THIN_PRISM_FISHEYE);
}

// Catches an inverse that rejects a valid thin-prism ray because the radial-only seed exceeds the
// fisheye angle limit once the prism term has pushed the distorted radius outward.
TEST(UndistortInverse, ThinPrismWideAngleRayRecoveredPastRadialSeedLimit) {
    expect_thin_prism_wide_angle_round_trip({0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.1f, 0.0f, 0.0f, 0.0f});
}

// Catches a seed or Newton solve that cannot invert the full COLMAP thin-prism model at wide angles.
TEST(UndistortInverse, ThinPrismFullCoefficientsWideAngleRoundTrip) {
    const auto& k = FULL_THIN_PRISM_RADIAL;
    const auto& t = FULL_THIN_PRISM_TANGENTIAL;
    expect_thin_prism_wide_angle_round_trip({k[0], k[1], k[2], k[3], t[0], t[1], t[2], 0.0f, t[3], 0.0f});
}

// Catches a CPU inverse (used by the crop solve) that rejects or mis-solves wide thin-prism rays.
TEST(UndistortInverse, CpuThinPrismWideRaysRoundTrip) {
    const auto params = make_full_thin_prism_params(1000.0f, 1000.0f, 2000.0f, 2000.0f, 4000, 4000);
    for (const float theta : {1.0f, 1.3f, 1.45f, 1.5f, 1.54f}) {
        for (const float azimuth : {0.0f, 1.1f, 2.5f, 4.2f}) {
            const float r = std::tan(theta);
            float dx, dy;
            distort_normalized_point(params, r * std::cos(azimuth), r * std::sin(azimuth), dx, dy);
            const float image_x = dx * params.src_fx + params.src_cx;
            const float image_y = dy * params.src_fy + params.src_cy;
            float nx, ny;
            ASSERT_TRUE(undistort_image_point(params, image_x, image_y, nx, ny))
                << "theta " << theta << " azimuth " << azimuth;
            float rx, ry;
            distort_normalized_point(params, nx, ny, rx, ry);
            EXPECT_NEAR(rx * params.src_fx + params.src_cx, image_x, 1.0e-2f)
                << "theta " << theta << " azimuth " << azimuth;
            EXPECT_NEAR(ry * params.src_fy + params.src_cy, image_y, 1.0e-2f)
                << "theta " << theta << " azimuth " << azimuth;
        }
    }
}

// The mesh evaluation mask samples coverage at these points; each must map back onto its
// exact pixel centre through the forward model.
TEST(UndistortInverse, SampleMapInvertsPixelCentres) {
    auto radial = Tensor::from_vector({-0.08f, 0.01f}, TensorShape({2}), Device::CPU);
    auto tangential = Tensor::from_vector({0.001f, -0.0008f}, TensorShape({2}), Device::CPU);
    const auto params = compute_undistort_params(
        450.0f, 455.0f, 160.0f, 120.0f, 320, 240, radial, tangential, CameraModelType::PINHOLE);
    const auto samples = inverse_distortion_sample_map(params, nullptr).cpu().contiguous();
    ASSERT_EQ(samples.shape(), (TensorShape({240, 320, 2})));
    const float* const map = samples.ptr<float>();
    int checked = 0;
    for (int y = 0; y < 240; y += 7) {
        for (int x = 0; x < 320; x += 7) {
            const size_t i = static_cast<size_t>(y) * 320 + x;
            if (!std::isfinite(map[2 * i]))
                continue;
            const auto [dx, dy] = distort_test_coordinate(map[2 * i], map[2 * i + 1], params);
            EXPECT_LE(std::hypot(dx * params.src_fx + params.src_cx - (x + 0.5f),
                                 dy * params.src_fy + params.src_cy - (y + 0.5f)),
                      1.0e-2f);
            ++checked;
        }
    }
    EXPECT_GT(checked, 1000);
}

TEST(UndistortInverse, ValidityMaskExcludesOutOfFrameSamples) {
    UndistortParams params{};
    params.src_fx = params.src_fy = 32.0f;
    params.src_cx = params.src_cy = 32.0f;
    params.dst_fx = params.dst_fy = 32.0f;
    params.dst_cx = params.dst_cy = 16.0f;
    params.src_width = params.src_height = 64;
    params.dst_width = params.dst_height = 32;
    params.model_type = CameraModelType::PINHOLE;

    auto input = Tensor::ones({3, 32, 32}, Device::GPU);
    Tensor validity;
    const auto output = distort_image_to_source(input, params, validity, nullptr);
    ASSERT_EQ(output.shape(), (TensorShape({3, 64, 64})));
    const auto mask_cpu = validity.cpu().contiguous();
    const uint8_t* const mask = mask_cpu.ptr<uint8_t>();
    EXPECT_EQ(mask[32 * 64 + 15], 0);
    EXPECT_EQ(mask[32 * 64 + 32], 1);
    EXPECT_EQ(mask[32 * 64 + 48], 0);
    for (int x = 0; x < 64; ++x) {
        if (mask[32 * 64 + x] != 0) {
            EXPECT_GE(x, 16);
            EXPECT_LE(x, 47);
        }
    }
}

TEST(UndistortInverse, HighResolutionValidityHasNoInteriorHoles) {
    constexpr int SOURCE_WIDTH = 8192;
    constexpr int SOURCE_HEIGHT = 4320;
    constexpr int EVAL_WIDTH = 5120;
    constexpr int EVAL_HEIGHT = 2700;
    auto radial = Tensor::from_vector({0.1f, -0.3f, 0.8f}, TensorShape({3}), Device::CPU);
    auto tangential = Tensor::from_vector({-4.0e-4f, -1.3e-3f}, TensorShape({2}), Device::CPU);
    const auto full = compute_undistort_params(
        8000.0f, 8000.0f, SOURCE_WIDTH / 2.0f, SOURCE_HEIGHT / 2.0f, SOURCE_WIDTH, SOURCE_HEIGHT,
        radial, tangential, CameraModelType::PINHOLE);
    const auto params = prepare_undistort_params(full, EVAL_WIDTH, EVAL_HEIGHT, 1, EVAL_WIDTH);

    const auto input = Tensor::ones(
        {3, static_cast<size_t>(params.dst_height), static_cast<size_t>(params.dst_width)}, Device::GPU);
    Tensor validity;
    distort_image_to_source(input, params, validity, nullptr);
    ASSERT_EQ(validity.shape(), (TensorShape({EVAL_HEIGHT, EVAL_WIDTH})));
    const auto mask_cpu = validity.cpu().contiguous();
    const uint8_t* const mask = mask_cpu.ptr<uint8_t>();

    // Invalid pixels must all connect to the frame border; an isolated invalid pixel is a valid
    // ray that the inverse solve rejected (the float32 residual floor at this scale is ~6e-4 px).
    std::vector<uint8_t> reached(static_cast<size_t>(EVAL_WIDTH) * EVAL_HEIGHT, 0);
    std::vector<int> frontier;
    const auto visit = [&](const int x, const int y) {
        const size_t i = static_cast<size_t>(y) * EVAL_WIDTH + x;
        if (mask[i] == 0 && reached[i] == 0) {
            reached[i] = 1;
            frontier.push_back(static_cast<int>(i));
        }
    };
    for (int x = 0; x < EVAL_WIDTH; ++x) {
        visit(x, 0);
        visit(x, EVAL_HEIGHT - 1);
    }
    for (int y = 0; y < EVAL_HEIGHT; ++y) {
        visit(0, y);
        visit(EVAL_WIDTH - 1, y);
    }
    while (!frontier.empty()) {
        const int i = frontier.back();
        frontier.pop_back();
        const int x = i % EVAL_WIDTH;
        const int y = i / EVAL_WIDTH;
        if (x > 0)
            visit(x - 1, y);
        if (x + 1 < EVAL_WIDTH)
            visit(x + 1, y);
        if (y > 0)
            visit(x, y - 1);
        if (y + 1 < EVAL_HEIGHT)
            visit(x, y + 1);
    }
    int interior_holes = 0;
    for (size_t i = 0; i < reached.size(); ++i)
        interior_holes += (mask[i] == 0 && reached[i] == 0) ? 1 : 0;
    EXPECT_EQ(interior_holes, 0);
    EXPECT_EQ(mask[(EVAL_HEIGHT / 2) * EVAL_WIDTH + EVAL_WIDTH / 2], 1);
}

// Wide-angle fisheye rays map to huge pinhole radii (r = tan(theta)); a solver that walks there
// in the image plane with bounded steps rejects them although they land inside the frame.
TEST(UndistortInverse, WideAngleFisheyeRaysAreInverted) {
    constexpr int HEIGHT = 16;
    constexpr int ROW = 8;
    UndistortParams params{};
    params.model_type = CameraModelType::FISHEYE;
    params.src_fx = params.src_fy = 100.0f;
    params.src_cx = 2048.0f;
    params.src_cy = ROW + 0.5f;
    params.src_width = 4096;
    params.src_height = HEIGHT;
    params.dst_cx = 1024.0f;
    params.dst_cy = HEIGHT / 2.0f;
    params.dst_width = 2048;
    params.dst_height = HEIGHT;

    std::vector<float> coordinates(3 * HEIGHT * 2048);
    for (int y = 0; y < HEIGHT; ++y)
        for (int x = 0; x < 2048; ++x)
            coordinates[static_cast<size_t>(y) * 2048 + x] = static_cast<float>(x);
    const auto input = Tensor::from_vector(
        coordinates, TensorShape({3, static_cast<size_t>(HEIGHT), 2048}), Device::GPU);

    for (const auto& [focal, source_x] : {std::pair{100.0f, 2193}, std::pair{10.0f, 2203}}) {
        params.dst_fx = params.dst_fy = focal;
        Tensor validity;
        const auto output = distort_image_to_source(input, params, validity, nullptr);

        // The warp integrates the pixel footprint with 8x8 quadrature; near 90 degrees the mean
        // of tan(theta) over the footprint is far from tan(theta) at the centre.
        double expected_x = 0.0;
        for (int qy = 0; qy < 8; ++qy) {
            for (int qx = 0; qx < 8; ++qx) {
                const double xd = (source_x + (qx + 0.5) / 8.0 - params.src_cx) / params.src_fx;
                const double yd = (ROW + (qy + 0.5) / 8.0 - params.src_cy) / params.src_fy;
                const double theta = std::hypot(xd, yd);
                expected_x += xd * std::tan(theta) / theta * focal + params.dst_cx - 0.5;
            }
        }
        expected_x /= 64.0;
        ASSERT_LT(expected_x, params.dst_width - 1.0);
        const auto valid = validity.cpu().contiguous();
        const auto mapped = output.cpu().contiguous();
        const size_t index = static_cast<size_t>(ROW) * params.src_width + source_x;
        EXPECT_EQ(valid.ptr<uint8_t>()[index], 1) << "focal " << focal;
        EXPECT_NEAR(mapped.ptr<float>()[index], expected_x, 0.05) << "focal " << focal;
    }
}

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
    // COLMAP FULL_OPENCV: numerator k1-k3, denominator k4-k6, then p1-p2.
    auto radial = Tensor::from_vector(
        {-0.1f, 0.02f, -0.003f, 0.001f, -0.0005f, 0.0002f},
        TensorShape({6}), Device::CPU);
    auto tangential = Tensor::from_vector({0.001f, -0.002f}, TensorShape({2}), Device::CPU);
    auto params = compute_undistort_params(
        TEST_FX, TEST_FY, TEST_CX, TEST_CY, TEST_W, TEST_H,
        radial, tangential, CameraModelType::PINHOLE);

    EXPECT_FLOAT_EQ(params.distortion[0], -0.1f);    // k1
    EXPECT_FLOAT_EQ(params.distortion[1], 0.02f);    // k2
    EXPECT_FLOAT_EQ(params.distortion[2], -0.003f);  // k3
    EXPECT_FLOAT_EQ(params.distortion[3], 0.001f);   // k4
    EXPECT_FLOAT_EQ(params.distortion[4], -0.0005f); // k5
    EXPECT_FLOAT_EQ(params.distortion[5], 0.0002f);  // k6
    EXPECT_FLOAT_EQ(params.distortion[6], 0.001f);   // p1
    EXPECT_FLOAT_EQ(params.distortion[7], -0.002f);  // p2
    EXPECT_EQ(params.num_distortion, 8);
}

TEST(UndistortFullOpenCV, ForwardMatchesDirectColmapFormula) {
    std::mt19937 generator(0x46554c4cU);
    std::uniform_real_distribution<float> numerator_distribution(-0.04f, 0.04f);
    std::uniform_real_distribution<float> denominator_distribution(-0.01f, 0.01f);
    std::uniform_real_distribution<float> tangential_distribution(-0.002f, 0.002f);
    std::uniform_real_distribution<float> point_distribution(-0.7f, 0.7f);

    for (int coefficient_set = 0; coefficient_set < 24; ++coefficient_set) {
        std::array<float, 6> radial{};
        for (int i = 0; i < 3; ++i)
            radial[i] = numerator_distribution(generator);
        for (int i = 3; i < 6; ++i)
            radial[i] = denominator_distribution(generator);
        const std::array<float, 2> tangential = {
            tangential_distribution(generator), tangential_distribution(generator)};
        const auto params = compute_undistort_params(
            TEST_FX, TEST_FY, TEST_CX, TEST_CY, TEST_W, TEST_H,
            Tensor::from_vector(
                std::vector<float>(radial.begin(), radial.end()), {6}, Device::CPU),
            Tensor::from_vector(
                std::vector<float>(tangential.begin(), tangential.end()), {2}, Device::CPU),
            CameraModelType::PINHOLE);

        for (int point = 0; point < 32; ++point) {
            const float x = point_distribution(generator);
            const float y = point_distribution(generator);
            const auto [expected_x, expected_y] =
                direct_full_opencv_distortion(x, y, radial, tangential);
            float actual_x = 0.0f;
            float actual_y = 0.0f;
            distort_normalized_point(params, x, y, actual_x, actual_y);
            // The shared host implementation can fuse operations on native CPUs.
            EXPECT_FLOAT_EQ(actual_x, expected_x);
            EXPECT_FLOAT_EQ(actual_y, expected_y);
        }
    }
}

TEST(UndistortFullOpenCV, InverseRoundTripsWithinOneThousandthPixel) {
    const std::array<float, 6> radial = {
        -0.12f, 0.035f, -0.004f, 0.018f, -0.003f, 0.0004f};
    const std::array<float, 2> tangential = {0.0015f, -0.001f};
    const auto params = compute_undistort_params(
        TEST_FX, TEST_FY, TEST_CX, TEST_CY, TEST_W, TEST_H,
        Tensor::from_vector(
            std::vector<float>(radial.begin(), radial.end()), {6}, Device::CPU),
        Tensor::from_vector(
            std::vector<float>(tangential.begin(), tangential.end()), {2}, Device::CPU),
        CameraModelType::PINHOLE);

    for (int yi = -7; yi <= 7; ++yi) {
        for (int xi = -9; xi <= 9; ++xi) {
            const float x = static_cast<float>(xi) * 0.065f;
            const float y = static_cast<float>(yi) * 0.065f;
            float distorted_x = 0.0f;
            float distorted_y = 0.0f;
            distort_normalized_point(
                params, x, y, distorted_x, distorted_y);

            float recovered_x = 0.0f;
            float recovered_y = 0.0f;
            ASSERT_TRUE(undistort_image_point(
                params,
                distorted_x * TEST_FX + TEST_CX,
                distorted_y * TEST_FY + TEST_CY,
                recovered_x, recovered_y));
            EXPECT_NEAR(recovered_x * TEST_FX + TEST_CX,
                        x * TEST_FX + TEST_CX, 1.0e-3f);
            EXPECT_NEAR(recovered_y * TEST_FY + TEST_CY,
                        y * TEST_FY + TEST_CY, 1.0e-3f);
        }
    }
}

TEST(UndistortFullOpenCV, ThreeCoefficientCaseIsBitIdentical) {
    const std::vector<float> numerator = {-0.12f, 0.035f, -0.004f};
    const auto polynomial = compute_undistort_params(
        TEST_FX, TEST_FY, TEST_CX, TEST_CY, TEST_W, TEST_H,
        Tensor::from_vector(numerator, {3}, Device::CPU), Tensor(),
        CameraModelType::PINHOLE);
    const auto rational = compute_undistort_params(
        TEST_FX, TEST_FY, TEST_CX, TEST_CY, TEST_W, TEST_H,
        Tensor::from_vector(
            {numerator[0], numerator[1], numerator[2], 0.0f, 0.0f, 0.0f},
            {6}, Device::CPU),
        Tensor(), CameraModelType::PINHOLE);

    EXPECT_EQ(polynomial.dst_width, rational.dst_width);
    EXPECT_EQ(polynomial.dst_height, rational.dst_height);
    EXPECT_EQ(std::bit_cast<std::uint32_t>(polynomial.dst_cx),
              std::bit_cast<std::uint32_t>(rational.dst_cx));
    EXPECT_EQ(std::bit_cast<std::uint32_t>(polynomial.dst_cy),
              std::bit_cast<std::uint32_t>(rational.dst_cy));

    for (int yi = -7; yi <= 7; ++yi) {
        for (int xi = -9; xi <= 9; ++xi) {
            const float x = static_cast<float>(xi) * 0.065f;
            const float y = static_cast<float>(yi) * 0.065f;
            float polynomial_x = 0.0f;
            float polynomial_y = 0.0f;
            float rational_x = 0.0f;
            float rational_y = 0.0f;
            distort_normalized_point(
                polynomial, x, y, polynomial_x, polynomial_y);
            distort_normalized_point(
                rational, x, y, rational_x, rational_y);
            EXPECT_EQ(std::bit_cast<std::uint32_t>(polynomial_x),
                      std::bit_cast<std::uint32_t>(rational_x));
            EXPECT_EQ(std::bit_cast<std::uint32_t>(polynomial_y),
                      std::bit_cast<std::uint32_t>(rational_y));
        }
    }

    auto source = Tensor::randn(
        {3, static_cast<size_t>(TEST_H), static_cast<size_t>(TEST_W)},
        Device::GPU);
    const auto polynomial_image = undistort_image(source, polynomial, nullptr).cpu();
    const auto rational_image = undistort_image(source, rational, nullptr).cpu();
    EXPECT_EQ(polynomial_image.to_vector(), rational_image.to_vector());
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
    // COLMAP THIN_PRISM_FISHEYE: radial={k1,k2,k3,k4}, tangential={p1,p2,sx1,sy1}
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
    EXPECT_FLOAT_EQ(params.distortion[6], 0.0001f);  // sx1
    EXPECT_FLOAT_EQ(params.distortion[7], 0.0f);     // sx2
    EXPECT_FLOAT_EQ(params.distortion[8], -0.0002f); // sy1
    EXPECT_FLOAT_EQ(params.distortion[9], 0.0f);     // sy2
    EXPECT_EQ(params.num_distortion, 9);
}

// Catches a packing that stores COLMAP's sy1 in the x-axis r^4 prism slot instead of the y-axis
// r^2 slot: the y displacement of a pure-sy1 camera then vanishes.
TEST(UndistortPacking, ThinPrismPureSy1MatchesColmapReference) {
    constexpr float sy1 = 0.05f;
    auto radial = Tensor::from_vector({0.0f, 0.0f, 0.0f, 0.0f}, TensorShape({4}), Device::CPU);
    auto tangential = Tensor::from_vector({0.0f, 0.0f, 0.0f, sy1}, TensorShape({4}), Device::CPU);
    const auto params = compute_undistort_params(
        TEST_FX, TEST_FY, TEST_CX, TEST_CY, TEST_W, TEST_H,
        radial, tangential, CameraModelType::THIN_PRISM_FISHEYE);
    const std::array<double, 8> colmap_extra = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, sy1};

    for (const auto [x, y] : {std::pair{0.5f, 0.0f}, std::pair{0.0f, 0.4f}, std::pair{-0.3f, 0.25f}}) {
        float dx, dy;
        distort_normalized_point(params, x, y, dx, dy);
        const auto [rx, ry] = colmap_thin_prism_fisheye_reference(x, y, colmap_extra);
        EXPECT_NEAR(dx, rx, 1.0e-6) << "(" << x << ", " << y << ")";
        EXPECT_NEAR(dy, ry, 1.0e-6) << "(" << x << ", " << y << ")";
    }

    float dx, dy;
    distort_normalized_point(params, 0.5f, 0.0f, dx, dy);
    const double theta = std::atan(0.5);
    EXPECT_NEAR(dx, theta, 1.0e-6);
    EXPECT_NEAR(dy, sy1 * theta * theta, 1.0e-6);
}

// Catches a forward model that composes radial, tangential and prism terms sequentially instead of
// evaluating all of them on the theta-scaled point as COLMAP does; at 1.45 rad the sequential
// variant is off by about a millipixel-per-unit-focal (1e-3 normalized).
TEST(UndistortThinPrism, FullCoefficientsMatchColmapReference) {
    const auto params = make_full_thin_prism_params(
        TEST_FX, TEST_FY, TEST_CX, TEST_CY, TEST_W, TEST_H);
    const auto colmap_extra = full_thin_prism_colmap_extra();
    for (const auto [x, y] : wide_angle_sample_points()) {
        float dx, dy;
        distort_normalized_point(params, x, y, dx, dy);
        const auto [rx, ry] = colmap_thin_prism_fisheye_reference(x, y, colmap_extra);
        EXPECT_NEAR(dx, rx, 5.0e-6) << "(" << x << ", " << y << ")";
        EXPECT_NEAR(dy, ry, 5.0e-6) << "(" << x << ", " << y << ")";
    }
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
    // COLMAP model 6: FULL_OPENCV rational radial model + p1,p2.
    auto radial = Tensor::from_vector(
        {-0.15f, 0.03f, -0.005f, 0.02f, -0.004f, 0.0005f},
        TensorShape({6}), Device::CPU);
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

    const auto scaled = prepare_undistort_params(params, 40, 15, 2, 0);

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
    const auto uncapped = prepare_undistort_params(params, actual_src_width, actual_src_height, 2, 0);
    const int max_width = std::max(uncapped.dst_width, uncapped.dst_height) / 2;
    ASSERT_GT(max_width, 0);

    const auto capped = prepare_undistort_params(
        params, actual_src_width, actual_src_height, 2, max_width);

    EXPECT_EQ(std::max(capped.dst_width, capped.dst_height), max_width);
    const float dst_sx = static_cast<float>(capped.dst_width) / static_cast<float>(params.dst_width);
    const float dst_sy = static_cast<float>(capped.dst_height) / static_cast<float>(params.dst_height);
    EXPECT_FLOAT_EQ(capped.dst_fx, params.dst_fx * dst_sx);
    EXPECT_FLOAT_EQ(capped.dst_fy, params.dst_fy * dst_sy);
    EXPECT_FLOAT_EQ(capped.dst_cx, params.dst_cx * dst_sx);
    EXPECT_FLOAT_EQ(capped.dst_cy, params.dst_cy * dst_sy);
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

    const auto capped = prepare_undistort_params(
        params, params.src_width, params.src_height, 1, max_width);

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
    const auto uncapped = prepare_undistort_params(params, actual_src_width, actual_src_height, 2, 0);
    const int output_max = std::max(uncapped.dst_width, uncapped.dst_height);

    for (const int max_width : {output_max, output_max + 1, 0, -1}) {
        SCOPED_TRACE(max_width);
        const auto actual = prepare_undistort_params(
            params, actual_src_width, actual_src_height, 2, max_width);
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
    auto mask_dst = undistort_mask_area(mask_src, params, nullptr);
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
    const auto scaled = prepare_undistort_params(params, HALF_W, HALF_H, 2, 0);

    EXPECT_EQ(scaled.src_width, HALF_W);
    EXPECT_EQ(scaled.src_height, HALF_H);

    const float sx = static_cast<float>(HALF_W) / static_cast<float>(params.src_width);
    const float sy = static_cast<float>(HALF_H) / static_cast<float>(params.src_height);

    EXPECT_NEAR(scaled.src_fx, params.src_fx * sx, 1e-4f);
    EXPECT_NEAR(scaled.src_fy, params.src_fy * sy, 1e-4f);
    EXPECT_NEAR(scaled.src_cx, params.src_cx * sx, 1e-4f);
    EXPECT_NEAR(scaled.src_cy, params.src_cy * sy, 1e-4f);

    const int expected_dst_w = std::max(1, static_cast<int>(std::lround(params.dst_width * 0.5)));
    const int expected_dst_h = std::max(1, static_cast<int>(std::lround(params.dst_height * 0.5)));
    EXPECT_EQ(scaled.dst_width, expected_dst_w);
    EXPECT_EQ(scaled.dst_height, expected_dst_h);
    const float dst_sx = static_cast<float>(expected_dst_w) / static_cast<float>(params.dst_width);
    const float dst_sy = static_cast<float>(expected_dst_h) / static_cast<float>(params.dst_height);
    EXPECT_NEAR(scaled.dst_fx, params.dst_fx * dst_sx, 1e-4f);
    EXPECT_NEAR(scaled.dst_fy, params.dst_fy * dst_sy, 1e-4f);
    EXPECT_NEAR(scaled.dst_cx, params.dst_cx * dst_sx, 1e-4f);
    EXPECT_NEAR(scaled.dst_cy, params.dst_cy * dst_sy, 1e-4f);

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
    // Area quadrature integrates the selected pixel's bilinear tent, with
    // the bottom image edge clamped for the positive half of its footprint.
    constexpr float expected = 0.75f * 0.75f * 0.875f;
    EXPECT_FLOAT_EQ(undistort_mask_area(mask, params, nullptr).cpu().ptr<float>()[0], expected);
    if (gpu_backend_available(GpuBackend::CUDA)) {
        const GpuBackendScope scope(GpuBackend::CUDA);
        EXPECT_FLOAT_EQ(undistort_mask_area(mask.gpu(), params, nullptr).cpu().ptr<float>()[0], expected);
    }
    if (gpu_backend_available(GpuBackend::Vulkan)) {
        const GpuBackendScope scope(GpuBackend::Vulkan);
        EXPECT_FLOAT_EQ(undistort_mask_area(mask.gpu(), params, nullptr).cpu().ptr<float>()[0], expected);
    }
}

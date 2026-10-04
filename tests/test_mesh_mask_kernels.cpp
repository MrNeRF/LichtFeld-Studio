/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include "core/tensor_backend.hpp"
#include "core/tensor_image.hpp"
#include "training/metrics/mesh_mask.hpp"

#include "cuda_backend_test.hpp"
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numbers>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace {

    using lfs::core::Device;
    using lfs::core::Tensor;
    using lfs::core::TensorShape;
    using lfs::training::MeshMaskCamera;

    struct Mesh {
        std::vector<float> vertices;
        std::vector<int32_t> indices;
    };

    struct Vec3 {
        double x;
        double y;
        double z;
    };

    Vec3 operator-(const Vec3 a, const Vec3 b) {
        return {a.x - b.x, a.y - b.y, a.z - b.z};
    }

    Vec3 cross(const Vec3 a, const Vec3 b) {
        return {
            a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x};
    }

    double dot(const Vec3 a, const Vec3 b) {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }

    Vec3 camera_point(const Mesh& mesh, const int32_t index,
                      const MeshMaskCamera& camera) {
        const float* const point = mesh.vertices.data() + 3 * index;
        return {
            camera.world_to_camera[0] * point[0] +
                camera.world_to_camera[1] * point[1] +
                camera.world_to_camera[2] * point[2] +
                camera.world_to_camera[3],
            camera.world_to_camera[4] * point[0] +
                camera.world_to_camera[5] * point[1] +
                camera.world_to_camera[6] * point[2] +
                camera.world_to_camera[7],
            camera.world_to_camera[8] * point[0] +
                camera.world_to_camera[9] * point[1] +
                camera.world_to_camera[10] * point[2] +
                camera.world_to_camera[11]};
    }

    bool ray_hits_triangle(const double x, const double y, const Vec3 a,
                           const Vec3 b, const Vec3 c, const double z_near) {
        const Vec3 direction{x, y, 1.0};
        const Vec3 edge1 = b - a;
        const Vec3 edge2 = c - a;
        const Vec3 p = cross(direction, edge2);
        const double determinant = dot(edge1, p);
        if (std::abs(determinant) < 1.0e-14)
            return false;
        const double inverse_determinant = 1.0 / determinant;
        const Vec3 from_a{-a.x, -a.y, -a.z};
        const double u = dot(from_a, p) * inverse_determinant;
        const Vec3 q = cross(from_a, edge1);
        const double v = dot(direction, q) * inverse_determinant;
        const double t = dot(edge2, q) * inverse_determinant;
        constexpr double EDGE_EPSILON = 1.0e-12;
        return u >= -EDGE_EPSILON && v >= -EDGE_EPSILON &&
               u + v <= 1.0 + EDGE_EPSILON && t >= z_near;
    }

    bool reference_covered(const Mesh& mesh, const MeshMaskCamera& camera,
                           const double x, const double y,
                           const double z_near) {
        for (size_t face = 0; face < mesh.indices.size() / 3; ++face) {
            const Vec3 a = camera_point(mesh, mesh.indices[3 * face], camera);
            const Vec3 b = camera_point(mesh, mesh.indices[3 * face + 1], camera);
            const Vec3 c = camera_point(mesh, mesh.indices[3 * face + 2], camera);
            if (ray_hits_triangle(x, y, a, b, c, z_near))
                return true;
        }
        return false;
    }

    std::vector<uint8_t> rasterize(
        const Mesh& mesh, const MeshMaskCamera& camera, const float z_near,
        const Tensor* sample_map = nullptr,
        const lfs::core::UndistortParams* distortion = nullptr) {
        const auto vertices = Tensor::from_vector(
            mesh.vertices, TensorShape({mesh.vertices.size() / 3, 3}), Device::GPU);
        const auto indices = Tensor::from_vector(
            mesh.indices, TensorShape({mesh.indices.size() / 3, 3}), Device::GPU);
        Tensor result;
        if (sample_map && distortion) {
            result = lfs::training::rasterize_mesh_coverage(
                vertices, indices, camera, *sample_map, *distortion, z_near);
        } else {
            result = lfs::training::rasterize_mesh_coverage(
                vertices, indices, camera, z_near);
        }
        const auto cpu = result.cpu().contiguous();
        return {cpu.ptr<uint8_t>(), cpu.ptr<uint8_t>() + cpu.numel()};
    }

    Mesh quad(const double min_x, const double min_y,
              const double max_x, const double max_y,
              const double z) {
        return {
            {
                static_cast<float>(min_x * z),
                static_cast<float>(min_y * z),
                static_cast<float>(z),
                static_cast<float>(max_x * z),
                static_cast<float>(min_y * z),
                static_cast<float>(z),
                static_cast<float>(max_x * z),
                static_cast<float>(max_y * z),
                static_cast<float>(z),
                static_cast<float>(min_x * z),
                static_cast<float>(max_y * z),
                static_cast<float>(z),
            },
            {0, 1, 2, 0, 2, 3}};
    }

    Mesh cube(const float lo, const float hi) {
        return {{lo, lo, lo, hi, lo, lo, hi, hi, lo, lo, hi, lo, lo, lo, hi, hi, lo, hi, hi, hi, hi, lo, hi, hi},
                {0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4,
                 3, 7, 6, 3, 6, 2, 0, 4, 7, 0, 7, 3, 1, 2, 6, 1, 6, 5}};
    }

    MeshMaskCamera camera(const int width, const int height,
                          const float fx, const float fy) {
        MeshMaskCamera result;
        result.fx = fx;
        result.fy = fy;
        result.cx = width * 0.5f;
        result.cy = height * 0.5f;
        result.width = width;
        result.height = height;
        return result;
    }

    std::pair<double, double> distort_pinhole(
        const double x, const double y,
        const lfs::core::UndistortParams& params) {
        const double r2 = x * x + y * y;
        const double r4 = r2 * r2;
        const double r6 = r4 * r2;
        const double radial = 1.0 + params.distortion[0] * r2 +
                              params.distortion[1] * r4 +
                              params.distortion[2] * r6;
        return {
            x * radial + 2.0 * params.distortion[3] * x * y +
                params.distortion[4] * (r2 + 2.0 * x * x),
            y * radial + params.distortion[3] * (r2 + 2.0 * y * y) +
                2.0 * params.distortion[4] * x * y};
    }

    std::pair<double, double> inverse_distortion_reference(
        const double xd, const double yd,
        const lfs::core::UndistortParams& params) {
        double x = xd;
        double y = yd;
        for (int iteration = 0; iteration < 30; ++iteration) {
            const auto [value_x, value_y] = distort_pinhole(x, y, params);
            const double residual_x = value_x - xd;
            const double residual_y = value_y - yd;
            if (std::hypot(residual_x * params.src_fx,
                           residual_y * params.src_fy) < 1.0e-10)
                break;
            constexpr double STEP = 1.0e-6;
            const auto [xp_x, xp_y] = distort_pinhole(x + STEP, y, params);
            const auto [xm_x, xm_y] = distort_pinhole(x - STEP, y, params);
            const auto [yp_x, yp_y] = distort_pinhole(x, y + STEP, params);
            const auto [ym_x, ym_y] = distort_pinhole(x, y - STEP, params);
            const double j00 = (xp_x - xm_x) / (2.0 * STEP);
            const double j10 = (xp_y - xm_y) / (2.0 * STEP);
            const double j01 = (yp_x - ym_x) / (2.0 * STEP);
            const double j11 = (yp_y - ym_y) / (2.0 * STEP);
            const double determinant = j00 * j11 - j01 * j10;
            x -= (j11 * residual_x - j01 * residual_y) / determinant;
            y -= (-j10 * residual_x + j00 * residual_y) / determinant;
        }
        return {x, y};
    }

    double edge_distance_pixels(const Mesh& mesh, const MeshMaskCamera& camera,
                                const double x, const double y) {
        double closest = std::numeric_limits<double>::infinity();
        for (size_t face = 0; face < mesh.indices.size() / 3; ++face) {
            std::array<Vec3, 3> points{
                camera_point(mesh, mesh.indices[3 * face], camera),
                camera_point(mesh, mesh.indices[3 * face + 1], camera),
                camera_point(mesh, mesh.indices[3 * face + 2], camera)};
            for (int edge = 0; edge < 3; ++edge) {
                const Vec3 a = points[edge];
                const Vec3 b = points[(edge + 1) % 3];
                const double ax = a.x / a.z;
                const double ay = a.y / a.z;
                const double bx = b.x / b.z;
                const double by = b.y / b.z;
                const double vx = (bx - ax) * camera.fx;
                const double vy = (by - ay) * camera.fy;
                const double wx = (x - ax) * camera.fx;
                const double wy = (y - ay) * camera.fy;
                const double length_squared = vx * vx + vy * vy;
                const double t = length_squared > 0.0
                                     ? std::clamp((wx * vx + wy * vy) /
                                                      length_squared,
                                                  0.0, 1.0)
                                     : 0.0;
                closest = std::min(
                    closest,
                    std::hypot(wx - t * vx, wy - t * vy));
            }
        }
        return closest;
    }

    Mesh subdivided_sphere(const int latitude_cells,
                           const int longitude_cells) {
        Mesh mesh;
        const int columns = longitude_cells + 1;
        mesh.vertices.reserve(
            static_cast<size_t>(latitude_cells + 1) * columns * 3);
        mesh.indices.reserve(
            static_cast<size_t>(latitude_cells) * longitude_cells * 6);
        for (int latitude = 0; latitude <= latitude_cells; ++latitude) {
            const double phi = std::numbers::pi * latitude / latitude_cells;
            const double sin_phi = std::sin(phi);
            const double cos_phi = std::cos(phi);
            for (int longitude = 0; longitude <= longitude_cells; ++longitude) {
                const double theta = 2.0 * std::numbers::pi * longitude /
                                     longitude_cells;
                mesh.vertices.push_back(static_cast<float>(sin_phi * std::cos(theta)));
                mesh.vertices.push_back(static_cast<float>(sin_phi * std::sin(theta)));
                mesh.vertices.push_back(static_cast<float>(3.0 + cos_phi));
            }
        }
        for (int latitude = 0; latitude < latitude_cells; ++latitude) {
            for (int longitude = 0; longitude < longitude_cells; ++longitude) {
                const int32_t first = latitude * columns + longitude;
                const int32_t second = first + columns;
                mesh.indices.insert(
                    mesh.indices.end(),
                    {first, second, first + 1,
                     first + 1, second, second + 1});
            }
        }
        return mesh;
    }

    // Each backend's rasterizer must match the double-precision reference; the trainer picks
    // the rasterizer of the default backend.
    class MeshMaskRasterizerTest : public ::testing::TestWithParam<lfs::core::GpuBackend> {
    protected:
        void SetUp() override {
            if (!lfs::core::gpu_backend_available(GetParam()))
                GTEST_SKIP() << lfs::core::gpu_backend_name(GetParam()) << " unavailable";
            backend_.emplace(GetParam());
            ASSERT_TRUE(backend_->switched());
        }

    private:
        std::optional<lfs::test::DefaultGpuBackendForTesting> backend_;
    };

    INSTANTIATE_TEST_SUITE_P(Backends, MeshMaskRasterizerTest,
                             ::testing::Values(lfs::core::GpuBackend::CUDA, lfs::core::GpuBackend::Vulkan,
                                               lfs::core::GpuBackend::Metal),
                             [](const auto& info) { return std::string(lfs::core::gpu_backend_name(info.param)); });

} // namespace

// Catches corner sampling or output scaling that changes the covered pixel rectangle.
TEST_P(MeshMaskRasterizerTest, PixelCenterQuadMatchesAtTwoResolutions) {
    const Mesh mesh = quad(-0.5625, -0.4375, 0.4375, 0.3125, 2.0);
    for (const auto [width, height, focal] : {
             std::tuple{16, 12, 8.0f}, std::tuple{32, 24, 16.0f}}) {
        const auto cam = camera(width, height, focal, focal);
        const auto mask = rasterize(mesh, cam, 0.01f);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                const double nx = (x + 0.5 - cam.cx) / cam.fx;
                const double ny = (y + 0.5 - cam.cy) / cam.fy;
                const bool expected = nx >= -0.5625 && nx <= 0.4375 &&
                                      ny >= -0.4375 && ny <= 0.3125;
                EXPECT_EQ(mask[static_cast<size_t>(y) * width + x], expected)
                    << "pixel " << x << ',' << y << " at " << width << 'x' << height;
            }
        }
    }
}

// Catches exclusive shared-edge tests that leave cracks between adjacent faces.
TEST_P(MeshMaskRasterizerTest, WatertightCubeHasNoDiagonalCracks) {
    Mesh mesh = cube(-1.0f, 1.0f);
    for (size_t vertex = 0; vertex < mesh.vertices.size() / 3; ++vertex)
        mesh.vertices[3 * vertex + 2] += 3.0f;
    const auto cam = camera(64, 64, 32.0f, 32.0f);
    const auto mask = rasterize(mesh, cam, 0.01f);
    for (int y = 16; y < 48; ++y) {
        for (int x = 16; x < 48; ++x)
            EXPECT_EQ(mask[static_cast<size_t>(y) * 64 + x], 1) << x << ',' << y;
    }
    EXPECT_EQ(mask[0], 0);
}

// Catches projection without near-plane and one-frame guard-band clipping.
TEST_P(MeshMaskRasterizerTest, NearPlaneGuardBandAndBehindCameraMatchReference) {
    const auto cam = camera(96, 72, 55.0f, 55.0f);
    constexpr float Z_NEAR = 0.1f;
    const std::array meshes{
        Mesh{{-0.04f, -0.03f, 0.05f,
              0.16f, -0.12f, 0.2f,
              0.0f, 0.18f, 0.2f},
             {0, 1, 2}},
        Mesh{{0.0f, -0.001f, 0.10001f,
              -100.0f, 60.0f, 100.0f,
              100.0f, 60.0f, 100.0f},
             {0, 1, 2}},
        Mesh{{-1.0f, -1.0f, -2.0f,
              1.0f, -1.0f, -2.0f,
              0.0f, 1.0f, -2.0f},
             {0, 1, 2}},
    };
    for (const auto& mesh : meshes) {
        const auto mask = rasterize(mesh, cam, Z_NEAR);
        for (int y = 0; y < cam.height; ++y) {
            for (int x = 0; x < cam.width; ++x) {
                const double nx = (x + 0.5 - cam.cx) / cam.fx;
                const double ny = (y + 0.5 - cam.cy) / cam.fy;
                EXPECT_EQ(
                    mask[static_cast<size_t>(y) * cam.width + x],
                    reference_covered(mesh, cam, nx, ny, Z_NEAR))
                    << "pixel " << x << ',' << y;
            }
        }
    }
}

// Catches back-face culling, which is invalid for silhouette union coverage.
TEST_P(MeshMaskRasterizerTest, CameraInsideClosedCubeCoversEveryPixel) {
    const Mesh mesh = cube(-1.0f, 1.0f);
    const auto cam = camera(80, 60, 30.0f, 30.0f);
    const auto mask = rasterize(mesh, cam, 1.0e-4f);
    EXPECT_EQ(std::count(mask.begin(), mask.end(), uint8_t{1}), mask.size());
}

// Catches a second inverse solve or a non-conservative distorted triangle bound.
TEST_P(MeshMaskRasterizerTest, DistortedSamplesMatchDoublePrecisionReference) {
    lfs::core::UndistortParams params{};
    params.src_fx = 70.0f;
    params.src_fy = 69.0f;
    params.src_cx = 48.0f;
    params.src_cy = 36.0f;
    params.dst_fx = 70.0f;
    params.dst_fy = 69.0f;
    params.dst_cx = 48.0f;
    params.dst_cy = 36.0f;
    params.src_width = params.dst_width = 96;
    params.src_height = params.dst_height = 72;
    params.model_type = lfs::core::CameraModelType::PINHOLE;
    params.distortion[0] = -0.18f;
    params.distortion[1] = 0.035f;
    params.distortion[2] = -0.004f;
    params.distortion[3] = 0.0015f;
    params.distortion[4] = -0.0008f;
    params.num_distortion = 5;
    const Mesh mesh = quad(-0.45, -0.30, 0.38, 0.34, 2.0);
    auto cam = camera(96, 72, params.src_fx, params.src_fy);
    const auto samples = lfs::core::inverse_distortion_sample_map(params, nullptr);
    const auto mask = rasterize(mesh, cam, 0.01f, &samples, &params);

    int compared = 0;
    for (int y = 0; y < cam.height; ++y) {
        for (int x = 0; x < cam.width; ++x) {
            const double xd = (x + 0.5 - params.src_cx) / params.src_fx;
            const double yd = (y + 0.5 - params.src_cy) / params.src_fy;
            const auto [ux, uy] = inverse_distortion_reference(xd, yd, params);
            if (edge_distance_pixels(mesh, cam, ux, uy) <= 1.0e-3)
                continue;
            ++compared;
            EXPECT_EQ(
                mask[static_cast<size_t>(y) * cam.width + x],
                reference_covered(mesh, cam, ux, uy, 0.01))
                << "distorted pixel " << x << ',' << y;
        }
    }
    EXPECT_GT(compared, 6800);
}

// Catches clipping against a guard band narrower than the inverse samples: rays from 72 to 87
// degrees off-axis land far outside every frame of a wide equidistant fisheye.
TEST_P(MeshMaskRasterizerTest, WideFisheyeRaysBeyondTheFrameAreCovered) {
    lfs::core::UndistortParams params{};
    params.model_type = lfs::core::CameraModelType::FISHEYE;
    params.num_distortion = 4;
    params.src_width = params.dst_width = 96;
    params.src_height = params.dst_height = 72;
    params.src_fx = params.src_fy = params.dst_fx = params.dst_fy = 31.0f;
    params.src_cx = params.dst_cx = 48.0f;
    params.src_cy = params.dst_cy = 36.0f;
    const Mesh mesh = quad(3.0, -20.0, 20.0, 20.0, 1.0);
    const auto cam = camera(96, 72, params.src_fx, params.src_fy);
    const auto samples = lfs::core::inverse_distortion_sample_map(params, nullptr);
    const auto mask = rasterize(mesh, cam, 0.01f, &samples, &params);

    int beyond_frame = 0;
    for (int y = 0; y < cam.height; ++y) {
        for (int x = 0; x < cam.width; ++x) {
            const double xd = (x + 0.5 - params.src_cx) / params.src_fx;
            const double yd = (y + 0.5 - params.src_cy) / params.src_fy;
            const double theta = std::hypot(xd, yd);
            if (theta >= 1.55)
                continue;
            const double scale = std::tan(theta) / theta;
            const double ux = xd * scale;
            const double uy = yd * scale;
            if (std::abs(ux - 3.0) < 1.0e-2 || std::abs(ux - 20.0) < 1.0e-2 || std::abs(std::abs(uy) - 20.0) < 1.0e-2)
                continue;
            const bool expected = ux >= 3.0 && ux <= 20.0 && std::abs(uy) <= 20.0;
            EXPECT_EQ(mask[static_cast<size_t>(y) * cam.width + x], expected) << x << ',' << y;
            beyond_frame += expected && ux > 2.0 * cam.width / cam.fx;
        }
    }
    EXPECT_GT(beyond_frame, 10);
}

// Catches dense-mesh queue loss and shared-edge holes.
TEST_P(MeshMaskRasterizerTest, TwoMillionFaceSphereHasNoHolesAt5K) {
    constexpr int WIDTH = 5120;
    constexpr int HEIGHT = 2700;
    const Mesh mesh = subdivided_sphere(1000, 1000);
    ASSERT_EQ(mesh.indices.size() / 3, 2'000'000u);
    const auto cam = camera(WIDTH, HEIGHT, 2400.0f, 2400.0f);
    const auto vertices = Tensor::from_vector(
        mesh.vertices, TensorShape({mesh.vertices.size() / 3, 3}), Device::GPU);
    const auto indices = Tensor::from_vector(
        mesh.indices, TensorShape({mesh.indices.size() / 3, 3}), Device::GPU);
    const auto mask =
        lfs::training::rasterize_mesh_coverage(vertices, indices, cam, 1.0e-4f).cpu().contiguous();
    const auto* const bytes = mask.ptr<uint8_t>();
    const double interior_radius = 2400.0 / std::sqrt(8.0) - 2.0;
    size_t missing = 0;
    int first_missing_x = -1;
    int first_missing_y = -1;
    for (int y = HEIGHT / 2 - 850; y <= HEIGHT / 2 + 850; ++y) {
        for (int x = WIDTH / 2 - 850; x <= WIDTH / 2 + 850; ++x) {
            const double dx = x + 0.5 - cam.cx;
            const double dy = y + 0.5 - cam.cy;
            if (dx * dx + dy * dy > interior_radius * interior_radius ||
                bytes[static_cast<size_t>(y) * WIDTH + x] != 0)
                continue;
            if (missing++ == 0) {
                first_missing_x = x;
                first_missing_y = y;
            }
        }
    }
    EXPECT_EQ(missing, 0) << first_missing_x << ',' << first_missing_y;
}

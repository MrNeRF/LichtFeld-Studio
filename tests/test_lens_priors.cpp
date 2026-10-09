/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/camera_types.h"
#include "core/tensor.hpp"
#include "preprocessing/lens_priors.hpp"

#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <numbers>
#include <vector>

namespace {

    using lfs::preprocessing::LensCamera;
    using lfs::preprocessing::LensFace;
    using lfs::preprocessing::LensFaceInference;
    using lfs::preprocessing::LensFaceOutputs;

    constexpr float kRadius = 3.0f;
    constexpr int kFaceSize = 112;

    // Equidistant lens whose image corners see `corner_angle` off axis.
    LensCamera fisheye(const int size, const double corner_angle) {
        LensCamera camera;
        camera.projection.model = static_cast<int>(lfs::core::CameraModelType::FISHEYE);
        camera.width = camera.height = size;
        camera.fx = camera.fy = static_cast<float>(0.5 * size * std::numbers::sqrt2 / corner_angle);
        camera.cx = camera.cy = 0.5f * size;
        return camera;
    }

    LensCamera panorama(const int width) {
        LensCamera camera;
        camera.projection.model = static_cast<int>(lfs::core::CameraModelType::EQUIRECTANGULAR);
        camera.width = width;
        camera.height = width / 2;
        return camera;
    }

    // Every face looks at the inside of a sphere around the camera; face k answers with its scale off by
    // 1 + 0.15 k, as separate monocular estimates do.
    LensFaceInference sphere_inside(const std::vector<LensFace>& faces, int& calls) {
        return [&faces, &calls](const std::vector<float>&, const int size) {
            const float focal = faces.at(static_cast<size_t>(calls)).focal;
            const float scale = 1.0f + 0.15f * static_cast<float>(calls++);
            const float half = 0.5f * static_cast<float>(size);
            LensFaceOutputs out;
            out.points.resize(static_cast<size_t>(size) * size * 3);
            out.normals.resize(out.points.size());
            out.mask.assign(static_cast<size_t>(size) * size, 1.0f);
            for (int y = 0; y < size; ++y) {
                for (int x = 0; x < size; ++x) {
                    const float r[3] = {(x + 0.5f - half) / focal, (y + 0.5f - half) / focal, 1.0f};
                    const float len = std::sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
                    for (int c = 0; c < 3; ++c) {
                        out.points[(static_cast<size_t>(y) * size + x) * 3 + c] = scale * kRadius * r[c] / len;
                        out.normals[(static_cast<size_t>(y) * size + x) * 3 + c] = -r[c] / len;
                    }
                }
            }
            return out;
        };
    }

    struct SphereCheck {
        double coverage = 0.0;
        double distance_error_p99 = 0.0;
        double normal_dot_p01 = 0.0;
        int faces = 0;
    };

    SphereCheck gather_sphere(const LensCamera& camera) {
        const auto faces = lfs::preprocessing::plan_lens_faces(camera, kFaceSize);
        int calls = 0;
        const std::vector<float> image(static_cast<size_t>(camera.width) * camera.height * 3, 0.5f);
        const auto priors = lfs::preprocessing::estimate_lens_priors(image, camera, camera, kFaceSize,
                                                                     sphere_inside(faces, calls));
        auto rays_gpu = lfs::core::Tensor::empty(
            {static_cast<size_t>(camera.height), static_cast<size_t>(camera.width), 3}, lfs::core::Device::GPU);
        lfs::preprocessing::lens_rays(camera, rays_gpu);
        const auto rays = rays_gpu.to_vector();

        size_t valid = 0;
        std::vector<double> errors;
        std::vector<double> dots;
        for (size_t p = 0; p < priors.distance.size(); ++p) {
            if (!std::isfinite(rays[3 * p]))
                continue;
            ++valid;
            if (!(priors.distance[p] > 0.0f))
                continue;
            errors.push_back(std::abs(priors.distance[p] - kRadius) / kRadius);
            double dot = 0.0;
            for (int c = 0; c < 3; ++c)
                dot += priors.normal[3 * p + c] * rays[3 * p + c];
            dots.push_back(dot);
        }
        SphereCheck check;
        check.faces = priors.faces;
        check.coverage = valid ? static_cast<double>(errors.size()) / static_cast<double>(valid) : 0.0;
        if (!errors.empty()) {
            std::sort(errors.begin(), errors.end());
            std::sort(dots.begin(), dots.end(), std::greater<>());
            check.distance_error_p99 = errors[errors.size() * 99 / 100];
            check.normal_dot_p01 = dots[dots.size() * 99 / 100];
        }
        return check;
    }

} // namespace

// Catches faces gathered with the wrong rotation (normals and distances land on other pixels), camera Z stored
// instead of the distance along the ray, faces left at their own scale, and lens regions no face covers.
TEST(LensPriors, WideFisheyeRingRecoversRayDistanceAndNormals) {
    const auto check = gather_sphere(fisheye(256, 95.0 * std::numbers::pi / 180.0));
    EXPECT_EQ(check.faces, 9);
    EXPECT_GT(check.coverage, 0.995);
    EXPECT_LT(check.distance_error_p99, 0.01);
    EXPECT_LT(check.normal_dot_p01, -0.999);
}

// A 3:2 full-frame fisheye reaches far past the front face sideways but barely up and down: faces there would
// be mostly fill around a thin strip of image.
TEST(LensPriors, FullFrameFisheyeSkipsDirectionsTheImageDoesNotReach) {
    LensCamera camera = fisheye(300, 88.0 * std::numbers::pi / 180.0);
    camera.height = 200;
    camera.cy = 100.0f;
    const auto faces = lfs::preprocessing::plan_lens_faces(camera, kFaceSize);
    EXPECT_LT(faces.size(), 9u);
    const auto check = gather_sphere(camera);
    EXPECT_GT(check.coverage, 0.995);
    EXPECT_LT(check.distance_error_p99, 0.01);
}

TEST(LensPriors, ModerateFisheyeUsesOneFace) {
    const auto check = gather_sphere(fisheye(256, 40.0 * std::numbers::pi / 180.0));
    EXPECT_EQ(check.faces, 1);
    EXPECT_GT(check.coverage, 0.995);
    EXPECT_LT(check.distance_error_p99, 0.01);
    EXPECT_LT(check.normal_dot_p01, -0.999);
}

TEST(LensPriors, PanoramaCubeCoversEveryDirection) {
    const auto check = gather_sphere(panorama(256));
    EXPECT_EQ(check.faces, 6);
    EXPECT_GT(check.coverage, 0.995);
    EXPECT_LT(check.distance_error_p99, 0.01);
    EXPECT_LT(check.normal_dot_p01, -0.999);
}

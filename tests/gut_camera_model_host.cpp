/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Builds without CUDA evaluate GUT's camera models on the host, from the source the device runs:
// src/training/rasterization/gsplat/Cameras.cuh, as gut_camera_model_cuda.cu uses it, with the CUDA
// annotations defined away and CUDA's float max and min, which the header calls unqualified.

#include "gut_camera_model_cuda.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#define __host__
#define __device__
#define __forceinline__ inline

inline float max(const float a, const float b) { return fmaxf(a, b); }
inline float min(const float a, const float b) { return fminf(a, b); }

// The header's nvcc pragmas mean nothing to a host compiler.
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-pragmas"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunknown-pragmas"
#elif defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4068)
#endif
#include "training/rasterization/gsplat/Cameras.cuh"
#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#elif defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace gut_camera_model_test {
    namespace {
        ThinPrismFisheyeCameraModel<>::Parameters parameters_of(const ThinPrismCamera& camera) {
            ThinPrismFisheyeCameraModel<>::Parameters parameters{};
            parameters.resolution = {static_cast<uint32_t>(camera.resolution[0]),
                                     static_cast<uint32_t>(camera.resolution[1])};
            parameters.focal_length = camera.focal;
            parameters.principal_point = camera.principal;
            parameters.radial_coeffs = camera.radial;
            parameters.thin_prism_coeffs = camera.thin_prism;
            return parameters;
        }

        OpenCVFisheyeCameraModel<>::Parameters parameters_of(const OpenCvFisheyeCamera& camera) {
            OpenCVFisheyeCameraModel<>::Parameters parameters{};
            parameters.resolution = {static_cast<uint32_t>(camera.resolution[0]),
                                     static_cast<uint32_t>(camera.resolution[1])};
            parameters.focal_length = camera.focal;
            parameters.principal_point = camera.principal;
            parameters.radial_coeffs = camera.radial;
            return parameters;
        }

        template <typename Model>
        std::vector<float> project_rays(const Model& model, const std::vector<float>& rays) {
            const size_t count = rays.size() / 3;
            std::vector<float> points(3 * count);
            for (size_t i = 0; i < count; ++i) {
                const auto result = model.camera_ray_to_image_point(glm::fvec3{rays[3 * i], rays[3 * i + 1], rays[3 * i + 2]}, 0.f);
                points[3 * i] = result.imagePoint.x;
                points[3 * i + 1] = result.imagePoint.y;
                points[3 * i + 2] = result.valid_flag ? 1.f : 0.f;
            }
            return points;
        }

        template <typename Model>
        std::vector<float> unproject_points(const Model& model, const std::vector<float>& points) {
            const size_t count = points.size() / 2;
            std::vector<float> rays(4 * count);
            for (size_t i = 0; i < count; ++i) {
                const auto result = model.image_point_to_camera_ray(glm::fvec2{points[2 * i], points[2 * i + 1]});
                rays[4 * i] = result.ray_dir.x;
                rays[4 * i + 1] = result.ray_dir.y;
                rays[4 * i + 2] = result.ray_dir.z;
                rays[4 * i + 3] = result.valid_flag ? 1.f : 0.f;
            }
            return rays;
        }
    } // namespace

    std::vector<float> project(const ThinPrismCamera& camera, const std::vector<float>& rays) {
        return project_rays(ThinPrismFisheyeCameraModel<>(parameters_of(camera)), rays);
    }

    std::vector<float> unproject(const ThinPrismCamera& camera, const std::vector<float>& points) {
        return unproject_points(ThinPrismFisheyeCameraModel<>(parameters_of(camera)), points);
    }

    std::vector<float> project(const OpenCvFisheyeCamera& camera, const std::vector<float>& rays) {
        return project_rays(OpenCVFisheyeCameraModel<>(parameters_of(camera)), rays);
    }

    std::vector<float> unproject(const OpenCvFisheyeCamera& camera, const std::vector<float>& points) {
        return unproject_points(OpenCVFisheyeCameraModel<>(parameters_of(camera)), points);
    }

} // namespace gut_camera_model_test

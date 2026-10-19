/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "preprocessing/lens_priors.hpp"

#include "lens_views.hpp"

#include "core/camera_types.h"
#include "core/tensor.hpp"
#include "core/tensor/internal/cuda_stream_context.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <string>

namespace lfs::preprocessing {
    namespace {

        using lfs::core::Device;
        using lfs::core::Tensor;
        using lfs::core::TensorShape;
        using Vec3 = std::array<double, 3>;

        constexpr double kDegree = std::numbers::pi / 180.0;
        constexpr int kPlanningSide = 128;
        constexpr double kSingleFaceLimit = 50.0 * kDegree;
        constexpr double kPanoramaLimit = 100.0 * kDegree;
        constexpr double kFaceHalfFov = 50.0 * kDegree;
        constexpr double kMinSingleFaceHalfFov = 15.0 * kDegree;
        constexpr int kRingFaces = 8;
        constexpr double kRingMargin = 5.0 * kDegree;
        constexpr double kMaxTilt = 60.0 * kDegree;
        constexpr std::size_t kMinOverlapSamples = 64;

        [[nodiscard]] Vec3 cross(const Vec3& a, const Vec3& b) {
            return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
        }

        [[nodiscard]] double norm(const Vec3& v) {
            return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        }

        [[nodiscard]] Vec3 normalized(const Vec3& v) {
            const double n = norm(v);
            return {v[0] / n, v[1] / n, v[2] / n};
        }

        [[nodiscard]] LensFace make_face(const Vec3& axis, const double half_fov, const int size) {
            const Vec3 z = normalized(axis);
            Vec3 x = cross({0.0, 1.0, 0.0}, z);
            x = norm(x) < 1.0e-6 ? Vec3{1.0, 0.0, 0.0} : normalized(x);
            const Vec3 y = cross(z, x);
            LensFace face;
            for (int c = 0; c < 3; ++c) {
                face.rotation[c] = static_cast<float>(x[c]);
                face.rotation[3 + c] = static_cast<float>(y[c]);
                face.rotation[6 + c] = static_cast<float>(z[c]);
            }
            face.focal = static_cast<float>(0.5 * size / std::tan(half_fov));
            face.size = size;
            return face;
        }

        // Unit rays of the lens on a coarse grid; NaN where it holds none.
        [[nodiscard]] std::vector<float> coarse_rays(const LensCamera& camera) {
            const double scale = std::min(1.0, static_cast<double>(kPlanningSide) /
                                                   std::max(camera.width, camera.height));
            LensCamera coarse = camera;
            coarse.width = std::max(1, static_cast<int>(std::lround(camera.width * scale)));
            coarse.height = std::max(1, static_cast<int>(std::lround(camera.height * scale)));
            coarse.fx = static_cast<float>(camera.fx * coarse.width / camera.width);
            coarse.fy = static_cast<float>(camera.fy * coarse.height / camera.height);
            coarse.cx = static_cast<float>(camera.cx * coarse.width / camera.width);
            coarse.cy = static_cast<float>(camera.cy * coarse.height / camera.height);

            auto rays = Tensor::empty({static_cast<size_t>(coarse.height), static_cast<size_t>(coarse.width), 3},
                                      Device::CUDA);
            lens_rays(coarse, rays.ptr<float>(), lfs::core::getCurrentCUDAStream());
            return rays.to_vector();
        }

        // Faces in gather order; every face after the first overlaps one before it.
        [[nodiscard]] std::vector<float> align_face_scales(const std::vector<float>& distance,
                                                           const std::vector<float>& weight,
                                                           const std::size_t faces,
                                                           const std::size_t pixels) {
            std::vector<float> scales(faces, 1.0f);
            std::vector<double> merged(pixels, 0.0);
            std::vector<double> merged_weight(pixels, 0.0);
            std::vector<float> ratios;
            for (std::size_t k = 0; k < faces; ++k) {
                const float* d = distance.data() + k * pixels;
                const float* w = weight.data() + k * pixels;
                if (k > 0) {
                    ratios.clear();
                    for (std::size_t p = 0; p < pixels; ++p) {
                        if (merged_weight[p] > 0.0 && d[p] > 0.0f && std::isfinite(d[p]))
                            ratios.push_back(static_cast<float>(merged[p] / d[p]));
                    }
                    if (ratios.size() >= kMinOverlapSamples) {
                        const auto mid = ratios.begin() + static_cast<std::ptrdiff_t>(ratios.size() / 2);
                        std::nth_element(ratios.begin(), mid, ratios.end());
                        scales[k] = *mid;
                    }
                }
                for (std::size_t p = 0; p < pixels; ++p) {
                    if (!(d[p] > 0.0f) || !std::isfinite(d[p]))
                        continue;
                    const double total = merged_weight[p] + w[p];
                    merged[p] = (merged[p] * merged_weight[p] + scales[k] * d[p] * w[p]) / total;
                    merged_weight[p] = total;
                }
            }
            return scales;
        }

    } // namespace

    std::vector<LensFace> plan_lens_faces(const LensCamera& camera, const int face_size) {
        const bool panorama = camera.projection.model == static_cast<int>(lfs::core::CameraModelType::EQUIRECTANGULAR);
        // How far off axis the image reaches in each direction around the axis.
        std::array<double, kRingFaces> reach{};
        double max_angle = panorama ? std::numbers::pi : 0.0;
        if (!panorama) {
            const auto rays = coarse_rays(camera);
            for (std::size_t i = 0; i + 2 < rays.size(); i += 3) {
                if (!std::isfinite(rays[i + 2]))
                    continue;
                const double angle = std::acos(std::clamp(static_cast<double>(rays[i + 2]), -1.0, 1.0));
                const double azimuth = std::atan2(static_cast<double>(rays[i + 1]), static_cast<double>(rays[i]));
                const auto sector = static_cast<std::size_t>(
                                        std::lround(azimuth / (2.0 * std::numbers::pi / kRingFaces)) + kRingFaces) %
                                    kRingFaces;
                reach[sector] = std::max(reach[sector], angle);
                max_angle = std::max(max_angle, angle);
            }
        }
        if (!(max_angle > 0.0))
            throw std::runtime_error("The lens has no valid ray to plan faces for");

        std::vector<LensFace> faces;
        if (max_angle > kPanoramaLimit) {
            for (const Vec3& axis : {Vec3{0, 0, 1}, Vec3{1, 0, 0}, Vec3{-1, 0, 0}, Vec3{0, 1, 0}, Vec3{0, -1, 0},
                                     Vec3{0, 0, -1}})
                faces.push_back(make_face(axis, kFaceHalfFov, face_size));
            return faces;
        }
        if (max_angle <= kSingleFaceLimit) {
            faces.push_back(make_face({0, 0, 1}, std::max(max_angle * 1.03, kMinSingleFaceHalfFov), face_size));
            return faces;
        }
        // A face only where the image reaches past the front face, tilted just far enough to hold its edge,
        // so no face is mostly fill around a narrow strip of image.
        faces.push_back(make_face({0, 0, 1}, kFaceHalfFov, face_size));
        for (int k = 0; k < kRingFaces; ++k) {
            if (reach[static_cast<std::size_t>(k)] <= kFaceHalfFov)
                continue;
            const double tilt = std::min(reach[static_cast<std::size_t>(k)] - kFaceHalfFov + kRingMargin, kMaxTilt);
            const double azimuth = 2.0 * std::numbers::pi * k / kRingFaces;
            faces.push_back(make_face({std::sin(tilt) * std::cos(azimuth), std::sin(tilt) * std::sin(azimuth),
                                       std::cos(tilt)},
                                      kFaceHalfFov, face_size));
        }
        return faces;
    }

    LensPriors estimate_lens_priors(const std::vector<float>& source_rgb,
                                    const LensCamera& source,
                                    const LensCamera& grid,
                                    const int face_size,
                                    const LensFaceInference& infer) {
        assert(source_rgb.size() == static_cast<std::size_t>(source.width) * source.height * 3);
        const auto faces = plan_lens_faces(source, face_size);
        const cudaStream_t stream = lfs::core::getCurrentCUDAStream();
        const std::size_t face_count = faces.size();
        const std::size_t pixels = static_cast<std::size_t>(grid.width) * grid.height;
        const std::size_t face_pixels = static_cast<std::size_t>(face_size) * face_size;
        const auto size = static_cast<std::size_t>(face_size);

        const auto lens = Tensor::from_vector(
            source_rgb, TensorShape({static_cast<size_t>(source.height), static_cast<size_t>(source.width), 3}),
            Device::CUDA);
        auto face_rgb = Tensor::empty({size, size, 3}, Device::CUDA);
        auto distance = Tensor::empty({face_count, pixels}, Device::CUDA);
        auto normal = Tensor::empty({face_count, pixels, 3}, Device::CUDA);
        auto weight = Tensor::empty({face_count, pixels}, Device::CUDA);

        for (std::size_t k = 0; k < face_count; ++k) {
            sample_lens_face(lens.ptr<float>(), source, faces[k], face_rgb.ptr<float>(), stream);
            const auto outputs = infer(face_rgb.to_vector(), face_size);
            if (outputs.points.size() != face_pixels * 3 || outputs.normals.size() != face_pixels * 3 ||
                outputs.mask.size() != face_pixels)
                throw std::runtime_error("Face estimate has " + std::to_string(outputs.mask.size()) +
                                         " pixels, expected " + std::to_string(face_pixels));
            const auto points = Tensor::from_vector(outputs.points, TensorShape({size, size, 3}), Device::CUDA);
            const auto normals = Tensor::from_vector(outputs.normals, TensorShape({size, size, 3}), Device::CUDA);
            const auto mask = Tensor::from_vector(outputs.mask, TensorShape({size, size}), Device::CUDA);
            project_face_to_lens(
                grid, faces[k], points.ptr<float>(), normals.ptr<float>(), mask.ptr<float>(),
                distance.ptr<float>() + k * pixels, normal.ptr<float>() + k * pixels * 3,
                weight.ptr<float>() + k * pixels, stream);
        }

        const auto scales = Tensor::from_vector(
            align_face_scales(distance.to_vector(), weight.to_vector(), face_count, pixels),
            TensorShape({face_count}), Device::CUDA);
        auto out_distance = Tensor::empty({pixels}, Device::CUDA);
        auto out_normal = Tensor::empty({pixels, 3}, Device::CUDA);
        blend_lens_faces(distance.ptr<float>(), normal.ptr<float>(), weight.ptr<float>(),
                         scales.ptr<float>(), static_cast<int>(face_count),
                         static_cast<int>(pixels), out_distance.ptr<float>(),
                         out_normal.ptr<float>(), stream);
        return LensPriors{
            .width = grid.width,
            .height = grid.height,
            .distance = out_distance.to_vector(),
            .normal = out_normal.to_vector(),
            .faces = static_cast<int>(face_count),
        };
    }

} // namespace lfs::preprocessing

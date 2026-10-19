/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lens_views.hpp"

#include "core/cuda_error.hpp"
#include "training/rasterization/gsplat/Cameras.cuh"

#include <cmath>

namespace lfs::preprocessing {
    namespace {

        constexpr int kThreads = 256;
        // Cross-fade band as a fraction of the face's half size.
        constexpr float kBorderBand = 0.15f;

        [[nodiscard]] unsigned int blocks_for(const size_t count) {
            return static_cast<unsigned int>((count + kThreads - 1) / kThreads);
        }

        template <typename Parameters>
        [[nodiscard]] Parameters lens_parameters(const LensCamera& camera) {
            Parameters p{};
            p.resolution = {static_cast<uint32_t>(camera.width), static_cast<uint32_t>(camera.height)};
            p.shutter_type = ShutterType::GLOBAL;
            return p;
        }

        // Builds the rasterizer's model for the lens once on the host; the fisheye models solve their
        // maximum angle in the constructor.
        template <typename Fn>
        void with_lens_model(const LensCamera& camera, Fn&& fn) {
            const auto& projection = camera.projection;
            switch (projection.model) {
            case CameraModelType::EQUIRECTANGULAR:
                fn(EquirectangularCameraModel(lens_parameters<EquirectangularCameraModel::Parameters>(camera)));
                return;
            case CameraModelType::FISHEYE: {
                auto p = lens_parameters<OpenCVFisheyeCameraModel<>::Parameters>(camera);
                p.focal_length = {camera.fx, camera.fy};
                p.principal_point = {camera.cx, camera.cy};
                for (int i = 0; i < 4; ++i)
                    p.radial_coeffs[i] = projection.radial[i];
                fn(OpenCVFisheyeCameraModel<>(p));
                return;
            }
            case CameraModelType::THIN_PRISM_FISHEYE: {
                auto p = lens_parameters<ThinPrismFisheyeCameraModel<>::Parameters>(camera);
                p.focal_length = {camera.fx, camera.fy};
                p.principal_point = {camera.cx, camera.cy};
                for (int i = 0; i < 4; ++i) {
                    p.radial_coeffs[i] = projection.radial[i];
                    p.thin_prism_coeffs[i] = projection.thin_prism[i];
                }
                fn(ThinPrismFisheyeCameraModel<>(p));
                return;
            }
            default: {
                auto p = lens_parameters<OpenCVPinholeCameraModel<>::Parameters>(camera);
                p.focal_length = {camera.fx, camera.fy};
                p.principal_point = {camera.cx, camera.cy};
                for (int i = 0; i < 6; ++i)
                    p.radial_coeffs[i] = projection.radial[i];
                for (int i = 0; i < 2; ++i)
                    p.tangential_coeffs[i] = projection.tangential[i];
                fn(OpenCVPinholeCameraModel<>(p));
                return;
            }
            }
        }

        __device__ glm::fvec3 to_face(const LensFace& face, const glm::fvec3& ray) {
            const float* r = face.rotation;
            return {r[0] * ray.x + r[1] * ray.y + r[2] * ray.z,
                    r[3] * ray.x + r[4] * ray.y + r[5] * ray.z,
                    r[6] * ray.x + r[7] * ray.y + r[8] * ray.z};
        }

        __device__ glm::fvec3 to_camera(const LensFace& face, const glm::fvec3& ray) {
            const float* r = face.rotation;
            return {r[0] * ray.x + r[3] * ray.y + r[6] * ray.z,
                    r[1] * ray.x + r[4] * ray.y + r[7] * ray.z,
                    r[2] * ray.x + r[5] * ray.y + r[8] * ray.z};
        }

        template <class Model>
        __device__ bool pixel_ray(const Model& model, const float x, const float y, glm::fvec3& ray) {
            const auto camera_ray = model.image_point_to_camera_ray(glm::fvec2{x, y});
            const float len = glm::length(camera_ray.ray_dir);
            if (!camera_ray.valid_flag || !(len > 0.0f) || !isfinite(len))
                return false;
            ray = camera_ray.ray_dir / len;
            return true;
        }

        template <class Model>
        __global__ void lens_rays_kernel(const Model model, const int width, const int height, float* rays) {
            const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i >= static_cast<size_t>(width) * height)
                return;
            glm::fvec3 ray;
            const bool valid = pixel_ray(model, static_cast<float>(i % width) + 0.5f,
                                         static_cast<float>(i / width) + 0.5f, ray);
            for (int c = 0; c < 3; ++c)
                rays[3 * i + c] = valid ? ray[c] : nanf("");
        }

        template <class Model>
        __global__ void sample_face_kernel(const Model model, const float* __restrict__ lens_rgb,
                                           const int width, const int height, const LensFace face,
                                           float* __restrict__ face_rgb) {
            const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            const int size = face.size;
            if (i >= static_cast<size_t>(size) * size)
                return;
            const float half = 0.5f * static_cast<float>(size);
            const glm::fvec3 face_ray{(static_cast<float>(i % size) + 0.5f - half) / face.focal,
                                      (static_cast<float>(i / size) + 0.5f - half) / face.focal, 1.0f};
            const auto projected = model.camera_ray_to_image_point(to_camera(face, face_ray), 0.0f);
            const float sx = projected.imagePoint.x - 0.5f;
            const float sy = projected.imagePoint.y - 0.5f;
            float rgb[3] = {0.5f, 0.5f, 0.5f};
            if (projected.valid_flag && sx >= 0.0f && sy >= 0.0f && sx <= width - 1.0f && sy <= height - 1.0f) {
                const int x0 = min(static_cast<int>(sx), width - 2 > 0 ? width - 2 : 0);
                const int y0 = min(static_cast<int>(sy), height - 2 > 0 ? height - 2 : 0);
                const int x1 = min(x0 + 1, width - 1);
                const int y1 = min(y0 + 1, height - 1);
                const float fx = sx - x0;
                const float fy = sy - y0;
                for (int c = 0; c < 3; ++c) {
                    const float top = lens_rgb[(static_cast<size_t>(y0) * width + x0) * 3 + c] * (1.0f - fx) +
                                      lens_rgb[(static_cast<size_t>(y0) * width + x1) * 3 + c] * fx;
                    const float bottom = lens_rgb[(static_cast<size_t>(y1) * width + x0) * 3 + c] * (1.0f - fx) +
                                         lens_rgb[(static_cast<size_t>(y1) * width + x1) * 3 + c] * fx;
                    rgb[c] = top * (1.0f - fy) + bottom * fy;
                }
            }
            for (int c = 0; c < 3; ++c)
                face_rgb[3 * i + c] = rgb[c];
        }

        template <class Model>
        __global__ void project_face_kernel(const Model model, const int width, const int height,
                                            const LensFace face, const float* __restrict__ face_points,
                                            const float* __restrict__ face_normals,
                                            const float* __restrict__ face_mask, float* __restrict__ distance,
                                            float* __restrict__ normal, float* __restrict__ weight) {
            const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i >= static_cast<size_t>(width) * height)
                return;
            distance[i] = nanf("");
            weight[i] = 0.0f;
            for (int c = 0; c < 3; ++c)
                normal[3 * i + c] = 0.0f;

            glm::fvec3 ray;
            if (!pixel_ray(model, static_cast<float>(i % width) + 0.5f, static_cast<float>(i / width) + 0.5f, ray))
                return;
            const glm::fvec3 face_ray = to_face(face, ray);
            if (face_ray.z <= 1.0e-4f)
                return;
            const int size = face.size;
            const float half = 0.5f * static_cast<float>(size);
            const float u = face.focal * face_ray.x / face_ray.z + half;
            const float v = face.focal * face_ray.y / face_ray.z + half;
            const float edge = fminf(fminf(u, size - u), fminf(v, size - v)) / half;
            if (!(edge > 0.0f))
                return;

            const float sx = u - 0.5f;
            const float sy = v - 0.5f;
            const int x0 = static_cast<int>(floorf(sx));
            const int y0 = static_cast<int>(floorf(sy));
            const float fx = sx - x0;
            const float fy = sy - y0;
            float z_sum = 0.0f;
            float tap_sum = 0.0f;
            glm::fvec3 n_sum{0.0f};
            for (int dy = 0; dy < 2; ++dy) {
                for (int dx = 0; dx < 2; ++dx) {
                    const int x = x0 + dx;
                    const int y = y0 + dy;
                    if (x < 0 || y < 0 || x >= size || y >= size)
                        continue;
                    const size_t t = static_cast<size_t>(y) * size + x;
                    const float z = face_points[3 * t + 2];
                    if (!(face_mask[t] >= 0.5f) || !(z > 0.0f) || !isfinite(z))
                        continue;
                    const float w = (dx ? fx : 1.0f - fx) * (dy ? fy : 1.0f - fy) + 1.0e-6f;
                    z_sum += w * z;
                    tap_sum += w;
                    n_sum += w * glm::fvec3{face_normals[3 * t], face_normals[3 * t + 1], face_normals[3 * t + 2]};
                }
            }
            if (tap_sum <= 0.0f)
                return;
            const float t = fminf(edge / kBorderBand, 1.0f);
            distance[i] = z_sum / tap_sum / face_ray.z;
            weight[i] = fmaxf(t * t * (3.0f - 2.0f * t), 1.0e-4f);
            const float n_len = glm::length(n_sum);
            if (n_len > 0.0f) {
                const glm::fvec3 n = to_camera(face, n_sum / n_len);
                for (int c = 0; c < 3; ++c)
                    normal[3 * i + c] = n[c];
            }
        }

        __global__ void blend_kernel(const float* __restrict__ distance, const float* __restrict__ normal,
                                     const float* __restrict__ weight, const float* __restrict__ scales,
                                     const int faces, const int pixels, float* __restrict__ out_distance,
                                     float* __restrict__ out_normal) {
            const int p = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            if (p >= pixels)
                return;
            float weight_sum = 0.0f;
            float distance_sum = 0.0f;
            glm::fvec3 n{0.0f};
            for (int k = 0; k < faces; ++k) {
                const size_t j = static_cast<size_t>(k) * pixels + p;
                const float d = distance[j];
                if (!(d > 0.0f) || !isfinite(d))
                    continue;
                const float w = weight[j];
                weight_sum += w;
                distance_sum += w * scales[k] * d;
                n += w * glm::fvec3{normal[3 * j], normal[3 * j + 1], normal[3 * j + 2]};
            }
            const float n_len = glm::length(n);
            out_distance[p] = weight_sum > 0.0f ? distance_sum / weight_sum : 0.0f;
            for (int c = 0; c < 3; ++c)
                out_normal[3 * static_cast<size_t>(p) + c] = weight_sum > 0.0f && n_len > 0.0f ? n[c] / n_len : 0.0f;
        }

    } // namespace

    void lens_rays(const LensCamera& camera, float* rays, cudaStream_t stream) {
        const size_t pixels = static_cast<size_t>(camera.width) * camera.height;
        if (!pixels)
            return;
        with_lens_model(camera, [&](const auto& model) {
            lens_rays_kernel<<<blocks_for(pixels), kThreads, 0, stream>>>(model, camera.width, camera.height, rays);
        });
        LFS_CUDA_LAUNCH_CHECK(stream, "lens.rays");
    }

    void sample_lens_face(const float* lens_rgb, const LensCamera& camera, const LensFace& face,
                          float* face_rgb, cudaStream_t stream) {
        const size_t pixels = static_cast<size_t>(face.size) * face.size;
        if (!pixels)
            return;
        with_lens_model(camera, [&](const auto& model) {
            sample_face_kernel<<<blocks_for(pixels), kThreads, 0, stream>>>(
                model, lens_rgb, camera.width, camera.height, face, face_rgb);
        });
        LFS_CUDA_LAUNCH_CHECK(stream, "lens.sample_face");
    }

    void project_face_to_lens(const LensCamera& camera, const LensFace& face,
                              const float* face_points, const float* face_normals, const float* face_mask,
                              float* distance, float* normal, float* weight, cudaStream_t stream) {
        const size_t pixels = static_cast<size_t>(camera.width) * camera.height;
        if (!pixels)
            return;
        with_lens_model(camera, [&](const auto& model) {
            project_face_kernel<<<blocks_for(pixels), kThreads, 0, stream>>>(
                model, camera.width, camera.height, face, face_points, face_normals, face_mask,
                distance, normal, weight);
        });
        LFS_CUDA_LAUNCH_CHECK(stream, "lens.project_face");
    }

    void blend_lens_faces(const float* distance, const float* normal, const float* weight,
                          const float* scales, const int faces, const int pixels,
                          float* out_distance, float* out_normal, cudaStream_t stream) {
        if (pixels <= 0)
            return;
        blend_kernel<<<blocks_for(static_cast<size_t>(pixels)), kThreads, 0, stream>>>(
            distance, normal, weight, scales, faces, pixels, out_distance, out_normal);
        LFS_CUDA_LAUNCH_CHECK(stream, "lens.blend");
    }

} // namespace lfs::preprocessing

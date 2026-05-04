/* Derived from Mesh2Splat by Electronic Arts Inc.
 * Original: Copyright (c) 2025 Electronic Arts Inc. All rights reserved.
 * Licensed under BSD 3-Clause (see THIRD_PARTY_LICENSES.md)
 *
 * Modifications: Copyright (c) 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/logger.hpp"
#include "core/material.hpp"
#include "core/mesh_data.hpp"
#include "core/tensor.hpp"
#include "rendering/mesh2splat.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <expected>
#include <format>
#include <limits>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

namespace lfs::rendering {

    using core::DataType;
    using core::Device;
    using core::Mesh2SplatOptions;
    using core::Mesh2SplatProgressCallback;
    using core::MeshData;
    using core::SplatData;
    using core::Submesh;
    using core::Tensor;
    using core::TextureImage;

    namespace {
        constexpr float SH_C0 = 0.28209479177387814f;

        struct FaceData {
            glm::vec3 position[3]{};
            glm::vec3 normal[3]{};
            glm::vec2 uv[3]{};
            glm::vec4 color[3]{glm::vec4(1.0f), glm::vec4(1.0f), glm::vec4(1.0f)};
            glm::vec2 ortho_uv[3]{};
            size_t material_index = 0;
            glm::vec3 face_normal{0.0f, 1.0f, 0.0f};
            glm::quat rotation{1.0f, 0.0f, 0.0f, 0.0f};
            glm::vec3 packed_scale{1.0f, 1.0f, 1.0e-7f};
        };

        [[nodiscard]] bool reportProgress(const Mesh2SplatProgressCallback& progress,
                                          const float pct,
                                          const std::string& stage) {
            return progress ? progress(pct, stage) : true;
        }

        [[nodiscard]] glm::vec3 safeNormalize(const glm::vec3& value,
                                              const glm::vec3& fallback) {
            const float len = glm::length(value);
            return len > 1.0e-8f ? value / len : fallback;
        }

        [[nodiscard]] glm::vec3 computeFaceNormal(const glm::vec3& v0,
                                                  const glm::vec3& v1,
                                                  const glm::vec3& v2) {
            return safeNormalize(glm::cross(v1 - v0, v2 - v0), glm::vec3(0.0f, 1.0f, 0.0f));
        }

        [[nodiscard]] glm::vec3 srgbToLinear(const glm::vec3& value) {
            return glm::pow(glm::clamp(value, glm::vec3(0.0f), glm::vec3(1.0f)), glm::vec3(2.2f));
        }

        [[nodiscard]] glm::vec4 fetchTexel(const TextureImage& image, const int x, const int y) {
            if (image.width <= 0 || image.height <= 0 || image.channels <= 0 || image.pixels.empty()) {
                return glm::vec4(1.0f);
            }

            const int px = std::clamp(x, 0, image.width - 1);
            const int py = std::clamp(y, 0, image.height - 1);
            const size_t index =
                (static_cast<size_t>(py) * static_cast<size_t>(image.width) + static_cast<size_t>(px)) *
                static_cast<size_t>(image.channels);

            const auto read = [&](const int channel, const float fallback) {
                return channel < image.channels
                           ? static_cast<float>(image.pixels[index + static_cast<size_t>(channel)]) / 255.0f
                           : fallback;
            };

            return {
                read(0, 1.0f),
                read(1, read(0, 1.0f)),
                read(2, read(0, 1.0f)),
                read(3, 1.0f),
            };
        }

        [[nodiscard]] glm::vec4 sampleTextureBilinear(const TextureImage& image, float u, float v) {
            if (image.width <= 0 || image.height <= 0 || image.channels <= 0 || image.pixels.empty()) {
                return glm::vec4(1.0f);
            }

            u -= std::floor(u);
            v -= std::floor(v);

            const float x = u * static_cast<float>(image.width - 1);
            const float y = v * static_cast<float>(image.height - 1);
            const int x0 = std::clamp(static_cast<int>(std::floor(x)), 0, image.width - 1);
            const int y0 = std::clamp(static_cast<int>(std::floor(y)), 0, image.height - 1);
            const int x1 = (x0 + 1) % image.width;
            const int y1 = (y0 + 1) % image.height;
            const float tx = x - static_cast<float>(x0);
            const float ty = y - static_cast<float>(y0);

            const glm::vec4 top = glm::mix(fetchTexel(image, x0, y0), fetchTexel(image, x1, y0), tx);
            const glm::vec4 bottom = glm::mix(fetchTexel(image, x0, y1), fetchTexel(image, x1, y1), tx);
            return glm::mix(top, bottom, ty);
        }

        [[nodiscard]] glm::vec4 materialColorAt(const MeshData& mesh,
                                                const size_t material_index,
                                                const glm::vec2& uv,
                                                const glm::vec4& vertex_color) {
            glm::vec4 material_factor(1.0f);
            glm::vec4 color(1.0f);

            if (material_index < mesh.materials.size()) {
                const auto& material = mesh.materials[material_index];
                material_factor = material.base_color;

                if (material.has_albedo_texture() &&
                    material.albedo_tex > 0 &&
                    material.albedo_tex <= mesh.texture_images.size()) {
                    const auto& image = mesh.texture_images[material.albedo_tex - 1u];
                    color = sampleTextureBilinear(image, uv.x, uv.y);
                    color = glm::vec4(srgbToLinear(glm::vec3(color)), color.a);
                } else {
                    color = vertex_color;
                }
            } else {
                color = vertex_color;
            }

            const glm::vec4 albedo = color * material_factor;
            const glm::vec3 encoded = glm::pow(
                glm::clamp(glm::vec3(albedo), glm::vec3(0.0f), glm::vec3(1.0f)),
                glm::vec3(1.0f / 2.2f));
            return glm::vec4(encoded, albedo.a);
        }

        [[nodiscard]] glm::vec2 projectForDominantAxis(const glm::vec3& position,
                                                       const glm::vec3& bbox_min,
                                                       const glm::vec3& bbox_max,
                                                       const glm::vec3& normal,
                                                       float& range_out) {
            const glm::vec3 abs_normal = glm::abs(normal);
            if (abs_normal.x > abs_normal.y && abs_normal.x > abs_normal.z) {
                const float range_y = bbox_max.y - bbox_min.y;
                const float range_z = bbox_max.z - bbox_min.z;
                range_out = std::max(range_y, range_z);
                return {position.y - bbox_min.y, position.z - bbox_min.z};
            }
            if (abs_normal.y > abs_normal.z) {
                const float range_x = bbox_max.x - bbox_min.x;
                const float range_z = bbox_max.z - bbox_min.z;
                range_out = std::max(range_x, range_z);
                return {position.x - bbox_min.x, position.z - bbox_min.z};
            }

            const float range_x = bbox_max.x - bbox_min.x;
            const float range_y = bbox_max.y - bbox_min.y;
            range_out = std::max(range_x, range_y);
            return {position.x - bbox_min.x, position.y - bbox_min.y};
        }

        void finishFaceBasisAndScale(FaceData& face,
                                     const glm::vec3& bbox_min,
                                     const glm::vec3& bbox_max) {
            glm::vec3 edge1 = face.position[1] - face.position[0];
            glm::vec3 edge2 = face.position[2] - face.position[0];
            const glm::vec3 edge3 = face.position[2] - face.position[1];

            const float l1 = glm::dot(edge1, edge1);
            const float l2 = glm::dot(edge2, edge2);
            const float l3 = glm::dot(edge3, edge3);
            if (l2 > l1 && l2 > l3) {
                std::swap(edge1, edge2);
            } else if (l3 > l1 && l3 > l2) {
                glm::vec3 tmp = edge1;
                edge1 = edge3;
                edge2 = tmp;
            }

            const glm::vec3 x_axis = safeNormalize(edge1, glm::vec3(1.0f, 0.0f, 0.0f));
            const glm::vec3 z_axis = safeNormalize(glm::cross(x_axis, edge2), face.face_normal);
            const glm::vec3 y_axis = safeNormalize(glm::cross(z_axis, x_axis), glm::vec3(0.0f, 1.0f, 0.0f));

            const glm::mat3 rotation_matrix(x_axis, y_axis, z_axis);
            face.rotation = glm::normalize(glm::quat_cast(rotation_matrix));

            float range = 1.0f;
            for (int corner = 0; corner < 3; ++corner) {
                face.ortho_uv[corner] = projectForDominantAxis(
                    face.position[corner], bbox_min, bbox_max, face.face_normal, range);
            }
            range = std::max(range, 1.0e-6f);
            for (int corner = 0; corner < 3; ++corner) {
                face.ortho_uv[corner] /= range;
            }

            const glm::vec2 duv1 = face.ortho_uv[1] - face.ortho_uv[0];
            const glm::vec2 duv2 = face.ortho_uv[2] - face.ortho_uv[0];
            const float det = duv1.x * duv2.y - duv1.y * duv2.x;

            if (std::abs(det) > 1.0e-10f) {
                const glm::vec3 e1 = face.position[1] - face.position[0];
                const glm::vec3 e2 = face.position[2] - face.position[0];
                const glm::vec3 ju = (e1 * duv2.y - e2 * duv1.y) / det;
                const glm::vec3 jv = (-e1 * duv2.x + e2 * duv1.x) / det;
                face.packed_scale = {glm::length(ju), glm::length(jv), 1.0e-7f};
            } else {
                face.packed_scale = glm::vec3(range, range, 1.0e-7f);
            }
        }

        [[nodiscard]] std::expected<std::vector<FaceData>, std::string>
        extractFaces(const MeshData& mesh, glm::vec3& global_min, glm::vec3& global_max) {
            if (!mesh.vertices.is_valid() || mesh.vertex_count() == 0) {
                return std::unexpected("Mesh has no vertices");
            }
            if (!mesh.indices.is_valid() || mesh.face_count() == 0) {
                return std::unexpected("Mesh has no faces");
            }
            if (mesh.vertices.dtype() != DataType::Float32 || mesh.indices.dtype() != DataType::Int32) {
                return std::unexpected("Mesh vertices must be Float32 and indices must be Int32");
            }

            Tensor vertices_cpu = mesh.vertices.device() == Device::CPU
                                      ? mesh.vertices.contiguous()
                                      : mesh.vertices.to(Device::CPU).contiguous();
            Tensor indices_cpu = mesh.indices.device() == Device::CPU
                                     ? mesh.indices.contiguous()
                                     : mesh.indices.to(Device::CPU).contiguous();

            const auto vertex_count = static_cast<int64_t>(vertices_cpu.shape()[0]);
            const auto face_count = static_cast<size_t>(indices_cpu.shape()[0]);
            const float* vertices = vertices_cpu.ptr<float>();
            const int32_t* indices = indices_cpu.ptr<int32_t>();

            Tensor normals_cpu;
            const float* normals = nullptr;
            if (mesh.has_normals()) {
                normals_cpu = mesh.normals.device() == Device::CPU
                                  ? mesh.normals.contiguous()
                                  : mesh.normals.to(Device::CPU).contiguous();
                if (normals_cpu.shape()[0] == vertices_cpu.shape()[0]) {
                    normals = normals_cpu.ptr<float>();
                }
            }

            Tensor texcoords_cpu;
            const float* texcoords = nullptr;
            if (mesh.has_texcoords()) {
                texcoords_cpu = mesh.texcoords.device() == Device::CPU
                                    ? mesh.texcoords.contiguous()
                                    : mesh.texcoords.to(Device::CPU).contiguous();
                if (texcoords_cpu.shape()[0] == vertices_cpu.shape()[0]) {
                    texcoords = texcoords_cpu.ptr<float>();
                }
            }

            Tensor colors_cpu;
            const float* colors = nullptr;
            if (mesh.has_colors()) {
                colors_cpu = mesh.colors.device() == Device::CPU
                                 ? mesh.colors.contiguous()
                                 : mesh.colors.to(Device::CPU).contiguous();
                if (colors_cpu.shape()[0] == vertices_cpu.shape()[0]) {
                    colors = colors_cpu.ptr<float>();
                }
            }

            std::vector<Submesh> submeshes = mesh.submeshes;
            if (submeshes.empty()) {
                submeshes.push_back({0, face_count * 3u, 0});
            }

            std::vector<FaceData> faces;
            faces.reserve(face_count);
            global_min = glm::vec3(std::numeric_limits<float>::max());
            global_max = glm::vec3(std::numeric_limits<float>::lowest());

            for (const auto& submesh : submeshes) {
                if (submesh.index_count % 3u != 0u ||
                    submesh.start_index + submesh.index_count > face_count * 3u) {
                    return std::unexpected("Mesh submesh index range is invalid");
                }

                const size_t submesh_face_count = submesh.index_count / 3u;
                for (size_t face_index = 0; face_index < submesh_face_count; ++face_index) {
                    const size_t flat_index = submesh.start_index + face_index * 3u;
                    const int32_t i0 = indices[flat_index + 0u];
                    const int32_t i1 = indices[flat_index + 1u];
                    const int32_t i2 = indices[flat_index + 2u];
                    if (i0 < 0 || i1 < 0 || i2 < 0 ||
                        i0 >= vertex_count || i1 >= vertex_count || i2 >= vertex_count) {
                        return std::unexpected("Mesh contains an out-of-range triangle index");
                    }

                    const int32_t face_indices[3] = {i0, i1, i2};
                    FaceData face;
                    face.material_index = submesh.material_index;

                    for (int corner = 0; corner < 3; ++corner) {
                        const int32_t vertex_index = face_indices[corner];
                        face.position[corner] = {
                            vertices[vertex_index * 3 + 0],
                            vertices[vertex_index * 3 + 1],
                            vertices[vertex_index * 3 + 2],
                        };
                        global_min = glm::min(global_min, face.position[corner]);
                        global_max = glm::max(global_max, face.position[corner]);

                        face.uv[corner] = texcoords
                                              ? glm::vec2(
                                                    texcoords[vertex_index * 2 + 0],
                                                    texcoords[vertex_index * 2 + 1])
                                              : glm::vec2(0.0f);

                        face.color[corner] = colors
                                                 ? glm::vec4(
                                                       colors[vertex_index * 4 + 0],
                                                       colors[vertex_index * 4 + 1],
                                                       colors[vertex_index * 4 + 2],
                                                       colors[vertex_index * 4 + 3])
                                                 : glm::vec4(1.0f);
                    }

                    face.face_normal = computeFaceNormal(face.position[0], face.position[1], face.position[2]);
                    for (int corner = 0; corner < 3; ++corner) {
                        face.normal[corner] = normals
                                                  ? safeNormalize(glm::vec3(
                                                                      normals[face_indices[corner] * 3 + 0],
                                                                      normals[face_indices[corner] * 3 + 1],
                                                                      normals[face_indices[corner] * 3 + 2]),
                                                                  face.face_normal)
                                                  : face.face_normal;
                    }
                    faces.push_back(face);
                }
            }

            return faces;
        }

        [[nodiscard]] glm::vec3 interpolate(const glm::vec3 (&values)[3], const glm::vec3& barycentric) {
            return values[0] * barycentric.x + values[1] * barycentric.y + values[2] * barycentric.z;
        }

        [[nodiscard]] glm::vec2 interpolate(const glm::vec2 (&values)[3], const glm::vec3& barycentric) {
            return values[0] * barycentric.x + values[1] * barycentric.y + values[2] * barycentric.z;
        }

        [[nodiscard]] glm::vec4 interpolate(const glm::vec4 (&values)[3], const glm::vec3& barycentric) {
            return values[0] * barycentric.x + values[1] * barycentric.y + values[2] * barycentric.z;
        }

        struct GaussianSample {
            glm::vec3 position;
            glm::vec4 color;
            glm::quat rotation;
            glm::vec3 scale;
        };

        // Scanline-rasterize one face into the [0, res] x [0, res] orthogonal-UV grid.
        // For each pixel sample (px+0.5, py+0.5) inside the triangle, emit one Gaussian.
        // Mirrors master's GS+FS pipeline (one fragment = one Gaussian).
        void rasterizeFace(const FaceData& face,
                           const MeshData& mesh,
                           const int resolution,
                           std::vector<GaussianSample>& out) {
            const float res_f = static_cast<float>(resolution);
            const glm::vec2 p0 = face.ortho_uv[0] * res_f;
            const glm::vec2 p1 = face.ortho_uv[1] * res_f;
            const glm::vec2 p2 = face.ortho_uv[2] * res_f;

            const float det = (p1.x - p0.x) * (p2.y - p0.y) - (p1.y - p0.y) * (p2.x - p0.x);
            if (std::abs(det) < 1.0e-10f) {
                return;
            }
            const float inv_det = 1.0f / det;

            const float min_x = std::min({p0.x, p1.x, p2.x});
            const float min_y = std::min({p0.y, p1.y, p2.y});
            const float max_x = std::max({p0.x, p1.x, p2.x});
            const float max_y = std::max({p0.y, p1.y, p2.y});
            const int x0 = std::max(0, static_cast<int>(std::floor(min_x)));
            const int y0 = std::max(0, static_cast<int>(std::floor(min_y)));
            const int x1 = std::min(resolution - 1, static_cast<int>(std::ceil(max_x)));
            const int y1 = std::min(resolution - 1, static_cast<int>(std::ceil(max_y)));

            for (int py = y0; py <= y1; ++py) {
                for (int px = x0; px <= x1; ++px) {
                    const glm::vec2 sample(static_cast<float>(px) + 0.5f,
                                           static_cast<float>(py) + 0.5f);
                    const float w1 = ((sample.x - p0.x) * (p2.y - p0.y) -
                                      (sample.y - p0.y) * (p2.x - p0.x)) *
                                     inv_det;
                    const float w2 = ((p1.x - p0.x) * (sample.y - p0.y) -
                                      (p1.y - p0.y) * (sample.x - p0.x)) *
                                     inv_det;
                    const float w0 = 1.0f - w1 - w2;
                    if (w0 < 0.0f || w1 < 0.0f || w2 < 0.0f) {
                        continue;
                    }
                    const glm::vec3 bary{w0, w1, w2};
                    const glm::vec3 position = interpolate(face.position, bary);
                    const glm::vec2 uv = interpolate(face.uv, bary);
                    const glm::vec4 vertex_color = interpolate(face.color, bary);
                    const glm::vec4 color = materialColorAt(mesh, face.material_index, uv, vertex_color);
                    out.push_back({position, color, face.rotation, face.packed_scale});
                }
            }
        }

    } // namespace

    std::expected<std::unique_ptr<SplatData>, std::string>
    mesh_to_splat(const MeshData& mesh,
                  const Mesh2SplatOptions& options,
                  Mesh2SplatProgressCallback progress) {
        if (options.resolution_target < Mesh2SplatOptions::kMinResolution) {
            return std::unexpected(std::format(
                "Mesh2Splat resolution must be at least {}", Mesh2SplatOptions::kMinResolution));
        }
        if (options.sigma <= 0.0f) {
            return std::unexpected("Mesh2Splat sigma must be positive");
        }

        if (!reportProgress(progress, 0.0f, "Preparing mesh data")) {
            return std::unexpected("Cancelled");
        }

        glm::vec3 global_min;
        glm::vec3 global_max;
        auto faces_result = extractFaces(mesh, global_min, global_max);
        if (!faces_result) {
            return std::unexpected(faces_result.error());
        }
        auto faces = std::move(*faces_result);
        if (faces.empty()) {
            return std::unexpected("Mesh contains no convertible triangles");
        }

        const glm::vec3 extent = global_max - global_min;
        const float scene_scale = glm::length(extent) * 0.5f;
        if (scene_scale <= 1.0e-8f) {
            return std::unexpected("Degenerate mesh: zero bounding box extent");
        }

        if (!reportProgress(progress, 0.15f, "Projecting triangles")) {
            return std::unexpected("Cancelled");
        }

        for (auto& face : faces) {
            finishFaceBasisAndScale(face, global_min, global_max);
        }

        LOG_INFO("mesh2splat: CPU rasterizer converting {} triangles "
                 "(resolution={}, bbox=[{:.2f},{:.2f},{:.2f}]-[{:.2f},{:.2f},{:.2f}])",
                 faces.size(), options.resolution_target,
                 global_min.x, global_min.y, global_min.z,
                 global_max.x, global_max.y, global_max.z);

        std::vector<GaussianSample> samples;
        samples.reserve(static_cast<size_t>(options.resolution_target) *
                        static_cast<size_t>(options.resolution_target));

        size_t processed_faces = 0;
        for (const auto& face : faces) {
            rasterizeFace(face, mesh, options.resolution_target, samples);
            ++processed_faces;
            if ((processed_faces & 0x3ffu) == 0u) {
                const float pct = 0.2f + 0.6f * static_cast<float>(processed_faces) /
                                             static_cast<float>(faces.size());
                if (!reportProgress(progress, pct, "Rasterizing mesh")) {
                    return std::unexpected("Cancelled");
                }
            }
        }

        const size_t gaussian_count = samples.size();
        if (gaussian_count == 0u) {
            return std::unexpected("Conversion produced zero gaussians");
        }

        LOG_INFO("mesh2splat: produced {} gaussians (resolution={})",
                 gaussian_count, options.resolution_target);

        std::vector<float> means(gaussian_count * 3u);
        std::vector<float> scaling_raw(gaussian_count * 3u);
        std::vector<float> rotation_raw(gaussian_count * 4u);
        std::vector<float> opacity_raw(gaussian_count);
        std::vector<float> sh0(gaussian_count * 3u);

        const float scale_multiplier = options.sigma / static_cast<float>(options.resolution_target);
        const float opacity_logit = -std::log(1.0f / 0.999f - 1.0f);

        for (size_t i = 0; i < gaussian_count; ++i) {
            const auto& s = samples[i];
            means[i * 3 + 0] = s.position.x;
            means[i * 3 + 1] = s.position.y;
            means[i * 3 + 2] = s.position.z;

            const glm::vec3 ls = glm::max(s.scale * scale_multiplier, glm::vec3(1.0e-8f));
            scaling_raw[i * 3 + 0] = std::log(ls.x);
            scaling_raw[i * 3 + 1] = std::log(ls.y);
            scaling_raw[i * 3 + 2] = std::log(ls.z);

            const glm::quat r = glm::normalize(s.rotation);
            rotation_raw[i * 4 + 0] = r.w;
            rotation_raw[i * 4 + 1] = r.x;
            rotation_raw[i * 4 + 2] = r.y;
            rotation_raw[i * 4 + 3] = r.z;

            opacity_raw[i] = opacity_logit;

            sh0[i * 3 + 0] = (s.color.r - 0.5f) / SH_C0;
            sh0[i * 3 + 1] = (s.color.g - 0.5f) / SH_C0;
            sh0[i * 3 + 2] = (s.color.b - 0.5f) / SH_C0;
        }

        if (!reportProgress(progress, 0.85f, "Building SplatData")) {
            return std::unexpected("Cancelled");
        }

        Tensor means_tensor = Tensor::from_vector(means, {gaussian_count, 3}, Device::CPU).cuda();
        Tensor scaling_tensor = Tensor::from_vector(scaling_raw, {gaussian_count, 3}, Device::CPU).cuda();
        Tensor rotation_tensor = Tensor::from_vector(rotation_raw, {gaussian_count, 4}, Device::CPU).cuda();
        Tensor opacity_tensor = Tensor::from_vector(opacity_raw, {gaussian_count, 1}, Device::CPU).cuda();
        Tensor sh0_tensor = Tensor::from_vector(sh0, {gaussian_count, 1, 3}, Device::CPU).cuda();
        Tensor shn_tensor = Tensor::zeros({gaussian_count, 0, 3}, Device::CUDA);

        auto splat = std::make_unique<SplatData>(
            0,
            std::move(means_tensor),
            std::move(sh0_tensor),
            std::move(shn_tensor),
            std::move(scaling_tensor),
            std::move(rotation_tensor),
            std::move(opacity_tensor),
            scene_scale);

        if (!reportProgress(progress, 1.0f, "Complete")) {
            return std::unexpected("Cancelled");
        }

        return splat;
    }

} // namespace lfs::rendering

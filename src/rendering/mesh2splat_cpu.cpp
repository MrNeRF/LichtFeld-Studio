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

#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>

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
            glm::vec2 uv[3]{};
            glm::vec4 color[3]{glm::vec4(1.0f), glm::vec4(1.0f), glm::vec4(1.0f)};
            glm::vec2 ortho_uv[3]{};
            size_t material_index = 0;
            glm::vec3 face_normal{0.0f, 1.0f, 0.0f};
            glm::quat rotation{1.0f, 0.0f, 0.0f, 0.0f};
            glm::vec3 packed_scale{1.0f, 1.0f, 1.0e-7f};

            glm::vec4 base_color{1.0f};
            const TextureImage* albedo_texture = nullptr;
            glm::vec4 premul_color[3]{glm::vec4(1.0f), glm::vec4(1.0f), glm::vec4(1.0f)};
            bool uniform_color = false;
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
                    faces.push_back(face);
                }
            }

            return faces;
        }

        void prepareMaterialBindings(const MeshData& mesh, std::vector<FaceData>& faces) {
            for (auto& face : faces) {
                face.base_color = glm::vec4(1.0f);
                face.albedo_texture = nullptr;
                if (face.material_index < mesh.materials.size()) {
                    const auto& material = mesh.materials[face.material_index];
                    face.base_color = material.base_color;
                    if (material.has_albedo_texture() &&
                        material.albedo_tex > 0 &&
                        material.albedo_tex <= mesh.texture_images.size()) {
                        const auto& image = mesh.texture_images[material.albedo_tex - 1u];
                        if (image.width > 0 && image.height > 0 && image.channels > 0 &&
                            !image.pixels.empty()) {
                            face.albedo_texture = &image;
                        }
                    }
                }
                for (int corner = 0; corner < 3; ++corner) {
                    face.premul_color[corner] = face.color[corner] * face.base_color;
                }
                face.uniform_color = face.premul_color[0] == face.premul_color[1] &&
                                     face.premul_color[1] == face.premul_color[2];
            }
        }

        struct GaussianSample {
            glm::vec3 position;
            glm::vec3 color_encoded;
            std::uint32_t face_index;
        };

        [[nodiscard]] inline glm::vec3 gammaEncode(glm::vec3 c) {
            c = glm::clamp(c, glm::vec3(0.0f), glm::vec3(1.0f));
            constexpr float inv_gamma = 1.0f / 2.2f;
            return {std::pow(c.x, inv_gamma), std::pow(c.y, inv_gamma), std::pow(c.z, inv_gamma)};
        }

        // Bilinear sampler operating directly on raw image data with hoisted invariants.
        // Returns gamma-encoded color (sRGB texture → linear via pow(2.2) → mul base_color → encode).
        [[nodiscard]] inline glm::vec4 sampleAlbedoLinear(const TextureImage& img, float u, float v) {
            u -= std::floor(u);
            v -= std::floor(v);
            const int w = img.width;
            const int h = img.height;
            const int ch = img.channels;
            const float x = u * static_cast<float>(w - 1);
            const float y = v * static_cast<float>(h - 1);
            const int x0 = static_cast<int>(x);
            const int y0 = static_cast<int>(y);
            const int x1 = (x0 + 1) % w;
            const int y1 = (y0 + 1) % h;
            const float tx = x - static_cast<float>(x0);
            const float ty = y - static_cast<float>(y0);
            constexpr float inv255 = 1.0f / 255.0f;
            const std::uint8_t* base = img.pixels.data();
            const auto fetch = [&](int px, int py) -> glm::vec4 {
                const std::size_t idx = (static_cast<std::size_t>(py) * static_cast<std::size_t>(w) +
                                         static_cast<std::size_t>(px)) *
                                        static_cast<std::size_t>(ch);
                if (ch >= 4) {
                    return {base[idx + 0] * inv255, base[idx + 1] * inv255,
                            base[idx + 2] * inv255, base[idx + 3] * inv255};
                }
                if (ch == 3) {
                    return {base[idx + 0] * inv255, base[idx + 1] * inv255,
                            base[idx + 2] * inv255, 1.0f};
                }
                if (ch == 2) {
                    const float r = base[idx + 0] * inv255;
                    return {r, r, r, base[idx + 1] * inv255};
                }
                const float r = base[idx + 0] * inv255;
                return {r, r, r, 1.0f};
            };
            const glm::vec4 c00 = fetch(x0, y0);
            const glm::vec4 c10 = fetch(x1, y0);
            const glm::vec4 c01 = fetch(x0, y1);
            const glm::vec4 c11 = fetch(x1, y1);
            const glm::vec4 top = glm::mix(c00, c10, tx);
            const glm::vec4 bot = glm::mix(c01, c11, tx);
            glm::vec4 c = glm::mix(top, bot, ty);
            // Approximate sRGB → linear (master uses GL_SRGB8 hardware path, same exponent).
            c = glm::vec4(glm::pow(glm::clamp(glm::vec3(c), 0.0f, 1.0f), glm::vec3(2.2f)), c.a);
            return c;
        }

        // Scanline-rasterize one face. One Gaussian per pixel covered, matching master's
        // GS+FS one-fragment-per-Gaussian semantics. Uses incremental edge functions
        // (constant per-pixel deltas) and a per-face material fast path that skips the
        // texture sample + sRGB→linear when the face is untextured + uniform-vertex-color.
        void rasterizeFace(const FaceData& face,
                           const std::uint32_t face_index,
                           const int resolution,
                           std::vector<GaussianSample>& out) {
            const float res_f = static_cast<float>(resolution);
            const glm::vec2 p0 = face.ortho_uv[0] * res_f;
            const glm::vec2 p1 = face.ortho_uv[1] * res_f;
            const glm::vec2 p2 = face.ortho_uv[2] * res_f;

            const float dx10 = p1.x - p0.x;
            const float dy10 = p1.y - p0.y;
            const float dx20 = p2.x - p0.x;
            const float dy20 = p2.y - p0.y;
            const float det = dx10 * dy20 - dy10 * dx20;
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
            if (x0 > x1 || y0 > y1) {
                return;
            }

            // Incremental edge functions: per-pixel deltas are constants.
            const float w1_dx = dy20 * inv_det;
            const float w1_dy = -dx20 * inv_det;
            const float w2_dx = -dy10 * inv_det;
            const float w2_dy = dx10 * inv_det;

            const float sx0 = static_cast<float>(x0) + 0.5f - p0.x;
            const float sy0 = static_cast<float>(y0) + 0.5f - p0.y;
            float w1_row = sx0 * w1_dx + sy0 * w1_dy;
            float w2_row = sx0 * w2_dx + sy0 * w2_dy;

            const std::size_t bbox_pixels = static_cast<std::size_t>(x1 - x0 + 1) *
                                            static_cast<std::size_t>(y1 - y0 + 1);
            if (out.capacity() - out.size() < bbox_pixels) {
                out.reserve(out.size() + std::max(bbox_pixels, out.capacity()));
            }

            // Manually unswitched per-face: hoists the texture/uniform/per-pixel-color
            // decision out of the per-pixel hot loop so each variant stays branchless inside.
            if (face.albedo_texture != nullptr) {
                const TextureImage& tex_img = *face.albedo_texture;
                for (int py = y0; py <= y1; ++py) {
                    float w1 = w1_row;
                    float w2 = w2_row;
                    for (int px = x0; px <= x1; ++px, w1 += w1_dx, w2 += w2_dx) {
                        const float w0 = 1.0f - w1 - w2;
                        if (w0 < 0.0f || w1 < 0.0f || w2 < 0.0f) {
                            continue;
                        }
                        const glm::vec3 position = face.position[0] * w0 +
                                                   face.position[1] * w1 +
                                                   face.position[2] * w2;
                        const glm::vec2 uv = face.uv[0] * w0 + face.uv[1] * w1 + face.uv[2] * w2;
                        const glm::vec4 tex = sampleAlbedoLinear(tex_img, uv.x, uv.y);
                        const glm::vec3 encoded = gammaEncode(glm::vec3(tex * face.base_color));
                        out.push_back({position, encoded, face_index});
                    }
                    w1_row += w1_dy;
                    w2_row += w2_dy;
                }
            } else if (face.uniform_color) {
                const glm::vec3 uniform_encoded = gammaEncode(glm::vec3(face.premul_color[0]));
                for (int py = y0; py <= y1; ++py) {
                    float w1 = w1_row;
                    float w2 = w2_row;
                    for (int px = x0; px <= x1; ++px, w1 += w1_dx, w2 += w2_dx) {
                        const float w0 = 1.0f - w1 - w2;
                        if (w0 < 0.0f || w1 < 0.0f || w2 < 0.0f) {
                            continue;
                        }
                        const glm::vec3 position = face.position[0] * w0 +
                                                   face.position[1] * w1 +
                                                   face.position[2] * w2;
                        out.push_back({position, uniform_encoded, face_index});
                    }
                    w1_row += w1_dy;
                    w2_row += w2_dy;
                }
            } else {
                for (int py = y0; py <= y1; ++py) {
                    float w1 = w1_row;
                    float w2 = w2_row;
                    for (int px = x0; px <= x1; ++px, w1 += w1_dx, w2 += w2_dx) {
                        const float w0 = 1.0f - w1 - w2;
                        if (w0 < 0.0f || w1 < 0.0f || w2 < 0.0f) {
                            continue;
                        }
                        const glm::vec3 position = face.position[0] * w0 +
                                                   face.position[1] * w1 +
                                                   face.position[2] * w2;
                        const glm::vec4 albedo = face.premul_color[0] * w0 +
                                                 face.premul_color[1] * w1 +
                                                 face.premul_color[2] * w2;
                        const glm::vec3 encoded = gammaEncode(glm::vec3(albedo));
                        out.push_back({position, encoded, face_index});
                    }
                    w1_row += w1_dy;
                    w2_row += w2_dy;
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

        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, faces.size()),
            [&](const tbb::blocked_range<size_t>& r) {
                for (size_t i = r.begin(); i != r.end(); ++i) {
                    finishFaceBasisAndScale(faces[i], global_min, global_max);
                }
            });

        prepareMaterialBindings(mesh, faces);

        LOG_INFO("mesh2splat: CPU rasterizer converting {} triangles "
                 "(resolution={}, bbox=[{:.2f},{:.2f},{:.2f}]-[{:.2f},{:.2f},{:.2f}])",
                 faces.size(), options.resolution_target,
                 global_min.x, global_min.y, global_min.z,
                 global_max.x, global_max.y, global_max.z);

        tbb::enumerable_thread_specific<std::vector<GaussianSample>> buckets;

        if (!reportProgress(progress, 0.2f, "Rasterizing mesh")) {
            return std::unexpected("Cancelled");
        }

        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, faces.size()),
            [&](const tbb::blocked_range<size_t>& r) {
                auto& b = buckets.local();
                for (size_t i = r.begin(); i != r.end(); ++i) {
                    rasterizeFace(faces[i], static_cast<std::uint32_t>(i),
                                  options.resolution_target, b);
                }
            });

        std::vector<const std::vector<GaussianSample>*> bucket_ptrs;
        std::vector<std::size_t> bucket_offsets;
        std::size_t gaussian_count = 0;
        bucket_ptrs.reserve(buckets.size());
        bucket_offsets.reserve(buckets.size());
        for (const auto& b : buckets) {
            bucket_ptrs.push_back(&b);
            bucket_offsets.push_back(gaussian_count);
            gaussian_count += b.size();
        }

        if (gaussian_count == 0u) {
            return std::unexpected("Conversion produced zero gaussians");
        }

        LOG_INFO("mesh2splat: produced {} gaussians (resolution={})",
                 gaussian_count, options.resolution_target);

        if (!reportProgress(progress, 0.85f, "Building SplatData")) {
            return std::unexpected("Cancelled");
        }

        Tensor means_tensor = Tensor::empty({gaussian_count, 3}, Device::CPU);
        Tensor scaling_tensor = Tensor::empty({gaussian_count, 3}, Device::CPU);
        Tensor rotation_tensor = Tensor::empty({gaussian_count, 4}, Device::CPU);
        Tensor opacity_tensor = Tensor::empty({gaussian_count, 1}, Device::CPU);
        Tensor sh0_tensor = Tensor::empty({gaussian_count, 1, 3}, Device::CPU);

        float* means_ptr = means_tensor.ptr<float>();
        float* scale_ptr = scaling_tensor.ptr<float>();
        float* rot_ptr = rotation_tensor.ptr<float>();
        float* opacity_ptr = opacity_tensor.ptr<float>();
        float* sh0_ptr = sh0_tensor.ptr<float>();

        const float scale_multiplier = options.sigma / static_cast<float>(options.resolution_target);
        const float opacity_logit = -std::log(1.0f / 0.999f - 1.0f);

        // Pre-normalize per-face quaternion once, log-scale once. Saves one normalize +
        // three logs per pixel vs. doing it inside the per-sample loop.
        std::vector<glm::vec4> face_rot_wxyz(faces.size());
        std::vector<glm::vec3> face_log_scale(faces.size());
        tbb::parallel_for(
            tbb::blocked_range<std::size_t>(0, faces.size()),
            [&](const tbb::blocked_range<std::size_t>& r) {
                for (std::size_t i = r.begin(); i != r.end(); ++i) {
                    const glm::quat q = glm::normalize(faces[i].rotation);
                    face_rot_wxyz[i] = glm::vec4(q.w, q.x, q.y, q.z);
                    const glm::vec3 ls = glm::max(faces[i].packed_scale * scale_multiplier,
                                                  glm::vec3(1.0e-8f));
                    face_log_scale[i] = glm::vec3(std::log(ls.x), std::log(ls.y), std::log(ls.z));
                }
            });

        tbb::parallel_for(
            tbb::blocked_range<std::size_t>(0, bucket_ptrs.size()),
            [&](const tbb::blocked_range<std::size_t>& r) {
                for (std::size_t b = r.begin(); b != r.end(); ++b) {
                    const auto& bucket = *bucket_ptrs[b];
                    const std::size_t base = bucket_offsets[b];
                    const std::size_t n = bucket.size();
                    for (std::size_t i = 0; i < n; ++i) {
                        const std::size_t out = base + i;
                        const auto& s = bucket[i];
                        means_ptr[out * 3 + 0] = s.position.x;
                        means_ptr[out * 3 + 1] = s.position.y;
                        means_ptr[out * 3 + 2] = s.position.z;

                        const glm::vec3& ls = face_log_scale[s.face_index];
                        scale_ptr[out * 3 + 0] = ls.x;
                        scale_ptr[out * 3 + 1] = ls.y;
                        scale_ptr[out * 3 + 2] = ls.z;

                        const glm::vec4& q = face_rot_wxyz[s.face_index];
                        rot_ptr[out * 4 + 0] = q.x;
                        rot_ptr[out * 4 + 1] = q.y;
                        rot_ptr[out * 4 + 2] = q.z;
                        rot_ptr[out * 4 + 3] = q.w;

                        opacity_ptr[out] = opacity_logit;

                        sh0_ptr[out * 3 + 0] = (s.color_encoded.x - 0.5f) / SH_C0;
                        sh0_ptr[out * 3 + 1] = (s.color_encoded.y - 0.5f) / SH_C0;
                        sh0_ptr[out * 3 + 2] = (s.color_encoded.z - 0.5f) / SH_C0;
                    }
                }
            });

        means_tensor = means_tensor.cuda();
        scaling_tensor = scaling_tensor.cuda();
        rotation_tensor = rotation_tensor.cuda();
        opacity_tensor = opacity_tensor.cuda();
        sh0_tensor = sh0_tensor.cuda();
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

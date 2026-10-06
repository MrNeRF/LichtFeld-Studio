/* Derived from Mesh2Splat by Electronic Arts Inc.
 * Original: Copyright (c) 2025 Electronic Arts Inc. All rights reserved.
 * Licensed under BSD 3-Clause (see THIRD_PARTY_LICENSES.md)
 *
 * Modifications: Copyright (c) 2025-2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "mesh2splat_shared.hpp"
#include "core/logger.hpp"
#include "core/mesh_data.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <format>
#include <optional>
#include <string>

namespace lfs::rendering::mesh2splat_detail {

    using core::DataType;
    using core::Device;
    using core::Mesh2SplatOptions;
    using core::MeshData;
    using core::SplatData;
    using core::Submesh;
    using core::Tensor;
    using core::TextureImage;

    lfs::Error conversion_error(const lfs::ErrorCode code, std::string detail,
                                const core::SourceSite detection) {
        return lfs::make_error({.code = code,
                                .domain = lfs::ErrorDomain::Rendering,
                                .detail = std::move(detail),
                                .detection = detection});
    }

    namespace {

        constexpr float SH_C0 = 0.28209479177387814f;

        [[nodiscard]] glm::vec3 compute_face_normal(const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2) {
            const glm::vec3 n = glm::cross(v1 - v0, v2 - v0);
            const float len = glm::length(n);
            return len > 1e-8f ? n / len : glm::vec3(0.0f, 1.0f, 0.0f);
        }

        [[nodiscard]] glm::vec4 compute_face_tangent(const glm::vec3& v0,
                                                     const glm::vec3& v1,
                                                     const glm::vec3& v2,
                                                     const glm::vec2& uv0,
                                                     const glm::vec2& uv1,
                                                     const glm::vec2& uv2) {
            const glm::vec3 dv1 = v1 - v0;
            const glm::vec3 dv2 = v2 - v0;
            const glm::vec2 duv1 = uv1 - uv0;
            const glm::vec2 duv2 = uv2 - uv0;
            const float det = duv1.x * duv2.y - duv1.y * duv2.x;
            if (std::abs(det) < 1e-8f) {
                return glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);
            }
            const float r = 1.0f / det;
            return glm::vec4(glm::normalize((dv1 * duv2.y - dv2 * duv1.y) * r), 1.0f);
        }

        [[nodiscard]] lfs::Result<std::vector<SubmeshGeometry>> extract_geometry(const MeshData& mesh) {
            if (!mesh.vertices.is_valid() || mesh.vertices.ndim() != 2 ||
                mesh.vertices.shape()[1] != 3 || mesh.vertices.dtype() != DataType::Float32) {
                return conversion_error(ErrorCode::InvalidArgument, "Mesh vertices must be a Float32 [V, 3] tensor");
            }
            if (!mesh.indices.is_valid() || mesh.indices.ndim() != 2 ||
                mesh.indices.shape()[1] != 3 || mesh.indices.dtype() != DataType::Int32) {
                return conversion_error(ErrorCode::InvalidArgument, "Mesh indices must be an Int32 [F, 3] tensor");
            }
            if (mesh.vertices.shape()[0] == 0) {
                return conversion_error(ErrorCode::InvalidArgument, "Mesh has no vertices");
            }
            if (mesh.indices.shape()[0] == 0) {
                return conversion_error(ErrorCode::InvalidArgument, "Mesh has no faces");
            }

            const auto V = static_cast<int64_t>(mesh.vertices.shape()[0]);
            const auto validate_attribute = [V](const Tensor& tensor,
                                                const int64_t width,
                                                const char* const label) -> std::optional<std::string> {
                if (!tensor.is_valid() || tensor.numel() == 0) {
                    return std::nullopt;
                }
                if (tensor.dtype() != DataType::Float32 || tensor.ndim() != 2 ||
                    tensor.shape()[0] != V || tensor.shape()[1] != width) {
                    return std::format("Mesh {} must be a Float32 [V, {}] tensor", label, width);
                }
                return std::nullopt;
            };
            if (auto error = validate_attribute(mesh.normals, 3, "normals")) {
                return conversion_error(ErrorCode::InvalidArgument, std::move(*error));
            }
            if (auto error = validate_attribute(mesh.tangents, 4, "tangents")) {
                return conversion_error(ErrorCode::InvalidArgument, std::move(*error));
            }
            if (auto error = validate_attribute(mesh.texcoords, 2, "texture coordinates")) {
                return conversion_error(ErrorCode::InvalidArgument, std::move(*error));
            }
            if (auto error = validate_attribute(mesh.colors, 4, "colors")) {
                return conversion_error(ErrorCode::InvalidArgument, std::move(*error));
            }

            auto verts_cpu = (mesh.vertices.device() == Device::CPU
                                  ? mesh.vertices
                                  : mesh.vertices.to(Device::CPU))
                                 .contiguous();
            auto idx_cpu = (mesh.indices.device() == Device::CPU
                                ? mesh.indices
                                : mesh.indices.to(Device::CPU))
                               .contiguous();

            const float* verts_ptr = verts_cpu.ptr<float>();
            const int32_t* idx_ptr = idx_cpu.ptr<int32_t>();
            const auto F = static_cast<int64_t>(idx_cpu.shape()[0]);

            const float* normals_ptr = nullptr;
            Tensor normals_cpu;
            if (mesh.has_normals()) {
                normals_cpu = (mesh.normals.device() == Device::CPU
                                   ? mesh.normals
                                   : mesh.normals.to(Device::CPU))
                                  .contiguous();
                normals_ptr = normals_cpu.ptr<float>();
            }

            const float* tangents_ptr = nullptr;
            Tensor tangents_cpu;
            if (mesh.has_tangents()) {
                tangents_cpu = (mesh.tangents.device() == Device::CPU
                                    ? mesh.tangents
                                    : mesh.tangents.to(Device::CPU))
                                   .contiguous();
                tangents_ptr = tangents_cpu.ptr<float>();
            }

            const float* texcoords_ptr = nullptr;
            Tensor texcoords_cpu;
            if (mesh.has_texcoords()) {
                texcoords_cpu = (mesh.texcoords.device() == Device::CPU
                                     ? mesh.texcoords
                                     : mesh.texcoords.to(Device::CPU))
                                    .contiguous();
                texcoords_ptr = texcoords_cpu.ptr<float>();
            }

            const float* colors_ptr = nullptr;
            Tensor colors_cpu;
            if (mesh.has_colors()) {
                colors_cpu = (mesh.colors.device() == Device::CPU
                                  ? mesh.colors
                                  : mesh.colors.to(Device::CPU))
                                 .contiguous();
                colors_ptr = colors_cpu.ptr<float>();
            }

            std::vector<Submesh> submeshes;
            if (mesh.submeshes.empty()) {
                submeshes.push_back({0, static_cast<size_t>(F) * 3, 0});
            } else {
                submeshes = mesh.submeshes;
            }

            std::vector<SubmeshGeometry> result;
            result.reserve(submeshes.size());
            size_t skipped_degenerate_faces = 0;
            const size_t total_index_count = idx_cpu.numel();

            for (const auto& sub : submeshes) {
                if (sub.index_count % 3 != 0) {
                    return conversion_error(ErrorCode::InvalidArgument, "Mesh submesh index count is not divisible by three");
                }
                if (sub.start_index > total_index_count ||
                    sub.index_count > total_index_count - sub.start_index) {
                    return conversion_error(ErrorCode::InvalidArgument, "Mesh submesh index range exceeds the index tensor");
                }
                SubmeshGeometry geo;
                geo.material_index = sub.material_index;
                const size_t face_count = sub.index_count / 3;
                geo.vertices.reserve(sub.index_count);

                for (size_t f = 0; f < face_count; ++f) {
                    const size_t base = sub.start_index + f * 3;
                    const int32_t indices[3] = {idx_ptr[base + 0], idx_ptr[base + 1], idx_ptr[base + 2]};
                    if (indices[0] < 0 || indices[0] >= V ||
                        indices[1] < 0 || indices[1] >= V ||
                        indices[2] < 0 || indices[2] >= V) {
                        return conversion_error(ErrorCode::InvalidArgument, "Mesh face contains an out-of-range vertex index");
                    }

                    glm::vec3 pos[3];
                    glm::vec3 nrm[3];
                    glm::vec4 tan[3];
                    glm::vec2 uv[3];
                    glm::vec4 col[3] = {glm::vec4(1.0f), glm::vec4(1.0f), glm::vec4(1.0f)};

                    for (int k = 0; k < 3; ++k) {
                        const int32_t vi = indices[k];
                        pos[k] = {verts_ptr[vi * 3 + 0], verts_ptr[vi * 3 + 1], verts_ptr[vi * 3 + 2]};
                        if (!std::isfinite(pos[k].x) || !std::isfinite(pos[k].y) || !std::isfinite(pos[k].z)) {
                            return conversion_error(ErrorCode::InvalidArgument, "Mesh contains a non-finite vertex position");
                        }
                    }

                    const glm::dvec3 edge01 = glm::dvec3(pos[1]) - glm::dvec3(pos[0]);
                    const glm::dvec3 edge02 = glm::dvec3(pos[2]) - glm::dvec3(pos[0]);
                    const glm::dvec3 edge12 = glm::dvec3(pos[2]) - glm::dvec3(pos[1]);
                    const double max_edge_sq = std::max({glm::dot(edge01, edge01),
                                                         glm::dot(edge02, edge02),
                                                         glm::dot(edge12, edge12)});
                    const glm::dvec3 area = glm::cross(edge01, edge02);
                    const double area_sq = glm::dot(area, area);
                    if (max_edge_sq == 0.0 || area_sq <= max_edge_sq * max_edge_sq * 1e-12) {
                        ++skipped_degenerate_faces;
                        continue;
                    }

                    for (int k = 0; k < 3; ++k) {
                        const int32_t vi = indices[k];
                        geo.bbox_min = glm::min(geo.bbox_min, pos[k]);
                        geo.bbox_max = glm::max(geo.bbox_max, pos[k]);

                        uv[k] = texcoords_ptr ? glm::vec2(texcoords_ptr[vi * 2 + 0], texcoords_ptr[vi * 2 + 1]) : glm::vec2(0.0f);
                        if (normals_ptr)
                            nrm[k] = {normals_ptr[vi * 3 + 0], normals_ptr[vi * 3 + 1], normals_ptr[vi * 3 + 2]};
                        if (tangents_ptr)
                            tan[k] = {tangents_ptr[vi * 4 + 0], tangents_ptr[vi * 4 + 1], tangents_ptr[vi * 4 + 2], tangents_ptr[vi * 4 + 3]};
                        if (colors_ptr)
                            col[k] = {colors_ptr[vi * 4 + 0], colors_ptr[vi * 4 + 1], colors_ptr[vi * 4 + 2], colors_ptr[vi * 4 + 3]};
                        if ((texcoords_ptr && (!std::isfinite(uv[k].x) || !std::isfinite(uv[k].y))) ||
                            (normals_ptr && (!std::isfinite(nrm[k].x) || !std::isfinite(nrm[k].y) || !std::isfinite(nrm[k].z))) ||
                            (tangents_ptr && (!std::isfinite(tan[k].x) || !std::isfinite(tan[k].y) ||
                                              !std::isfinite(tan[k].z) || !std::isfinite(tan[k].w))) ||
                            (colors_ptr && (!std::isfinite(col[k].x) || !std::isfinite(col[k].y) ||
                                            !std::isfinite(col[k].z) || !std::isfinite(col[k].w)))) {
                            return conversion_error(ErrorCode::InvalidArgument, "Mesh contains non-finite vertex attributes");
                        }
                    }

                    const glm::vec3 face_normal = compute_face_normal(pos[0], pos[1], pos[2]);
                    if (!normals_ptr) {
                        nrm[0] = nrm[1] = nrm[2] = face_normal;
                    } else {
                        for (int k = 0; k < 3; ++k) {
                            if (glm::dot(nrm[k], nrm[k]) <= 1e-20f) {
                                nrm[k] = face_normal;
                            }
                        }
                    }
                    if (!tangents_ptr) {
                        const glm::vec4 ft = compute_face_tangent(pos[0], pos[1], pos[2], uv[0], uv[1], uv[2]);
                        tan[0] = tan[1] = tan[2] = ft;
                    } else {
                        const glm::vec4 face_tangent = compute_face_tangent(
                            pos[0], pos[1], pos[2], uv[0], uv[1], uv[2]);
                        for (int k = 0; k < 3; ++k) {
                            const glm::vec3 tangent(tan[k]);
                            if (glm::dot(tangent, tangent) <= 1e-20f) {
                                tan[k] = face_tangent;
                            }
                        }
                    }

                    for (int k = 0; k < 3; ++k) {
                        geo.vertices.push_back(PerVertexData{pos[k], nrm[k], tan[k], uv[k], glm::vec2(0.0f), glm::vec3(0.0f), col[k]});
                    }
                }

                if (!geo.vertices.empty())
                    result.push_back(std::move(geo));
            }

            if (skipped_degenerate_faces > 0) {
                LOG_WARN("mesh2splat: skipped {} degenerate mesh faces", skipped_degenerate_faces);
            }
            return result;
        }

    } // namespace

    lfs::Result<PreparedConversion> prepare_conversion(const MeshData& mesh,
                                                       const Mesh2SplatOptions& options,
                                                       const core::Mesh2SplatProgressCallback& progress) {
        if (options.resolution_target < Mesh2SplatOptions::kMinResolution)
            return conversion_error(ErrorCode::InvalidArgument,
                                    std::format("Mesh2Splat resolution must be at least {}", Mesh2SplatOptions::kMinResolution));
        if (!std::isfinite(options.sigma) || options.sigma <= 0.0f)
            return conversion_error(ErrorCode::InvalidArgument, "Mesh2Splat sigma must be positive");

        if (progress && !progress(0.0f, "Preparing mesh data"))
            return conversion_error(ErrorCode::Cancelled, "Cancelled");

        auto extracted_geometry = extract_geometry(mesh);
        if (!extracted_geometry)
            return std::move(extracted_geometry).error();
        PreparedConversion prepared;
        prepared.submeshes = std::move(*extracted_geometry);
        const auto& submesh_geometries = prepared.submeshes;
        if (submesh_geometries.empty())
            return conversion_error(ErrorCode::InvalidArgument, "No geometry extracted");
        if (submesh_geometries.size() > std::numeric_limits<uint32_t>::max() / 3ull)
            return conversion_error(ErrorCode::ResourceExhausted, "Mesh2Splat submesh count exceeds Vulkan descriptor limits");

        size_t total_vertices = 0;
        for (const auto& geo : submesh_geometries) {
            if (geo.vertices.size() > std::numeric_limits<uint32_t>::max())
                return conversion_error(ErrorCode::ResourceExhausted, "Mesh2Splat submesh vertex count exceeds Vulkan draw limits");
            prepared.global_min = glm::min(prepared.global_min, geo.bbox_min);
            prepared.global_max = glm::max(prepared.global_max, geo.bbox_max);
            if (geo.vertices.size() > std::numeric_limits<size_t>::max() - total_vertices)
                return conversion_error(ErrorCode::ResourceExhausted, "Mesh2Splat triangle count exceeds host size range");
            total_vertices += geo.vertices.size();
        }

        prepared.scene_scale = glm::length(prepared.global_max - prepared.global_min) * 0.5f;
        if (!std::isfinite(prepared.scene_scale) || prepared.scene_scale <= 0.0f)
            return conversion_error(ErrorCode::InvalidArgument, "Degenerate mesh: invalid bounding box extent");

        const int res = options.resolution_target;
        prepared.triangle_count = static_cast<uint64_t>(total_vertices / 3);
        constexpr uint64_t max_entries = std::numeric_limits<uint32_t>::max();
        const uint64_t res64 = static_cast<uint64_t>(res);
        if (res64 > max_entries / 6ull / res64 || prepared.triangle_count > max_entries / 2ull)
            return conversion_error(ErrorCode::ResourceExhausted, "Mesh2Splat output capacity exceeds Vulkan uint32 counter range");
        const uint64_t pixel_based = res64 * res64 * 6ull;
        const uint64_t triangle_based = prepared.triangle_count * 2ull;
        prepared.output_capacity = static_cast<uint32_t>(std::max(pixel_based, triangle_based));
        return prepared;
    }

    lfs::Result<std::vector<uint8_t>> to_rgba8(const TextureImage& img) {
        if (img.width <= 0 || img.height <= 0 || img.channels <= 0 ||
            img.channels > 4 || img.pixels.empty()) {
            return conversion_error(ErrorCode::InvalidArgument, "Invalid Mesh2Splat texture image");
        }
        const size_t width = static_cast<size_t>(img.width);
        const size_t height = static_cast<size_t>(img.height);
        const size_t channels = static_cast<size_t>(img.channels);
        if (width > std::numeric_limits<size_t>::max() / height ||
            width * height > std::numeric_limits<size_t>::max() / channels ||
            width * height > std::numeric_limits<size_t>::max() / 4) {
            return conversion_error(ErrorCode::ResourceExhausted, "Mesh2Splat texture dimensions overflow host storage");
        }
        const size_t required_bytes = width * height * channels;
        if (img.pixels.size() < required_bytes) {
            return conversion_error(ErrorCode::InvalidArgument, "Mesh2Splat texture pixel storage is truncated");
        }

        std::vector<uint8_t> rgba(static_cast<size_t>(img.width) * img.height * 4, 255);
        for (int y = 0; y < img.height; ++y) {
            for (int x = 0; x < img.width; ++x) {
                const size_t src = (static_cast<size_t>(y) * img.width + x) * img.channels;
                const size_t dst = (static_cast<size_t>(y) * img.width + x) * 4;
                rgba[dst + 0] = img.channels >= 1 ? img.pixels[src + 0] : 255;
                rgba[dst + 1] = img.channels >= 2 ? img.pixels[src + 1] : 0;
                rgba[dst + 2] = img.channels >= 3 ? img.pixels[src + 2] : 0;
                rgba[dst + 3] = img.channels >= 4 ? img.pixels[src + 3] : 255;
            }
        }
        return rgba;
    }

    std::unique_ptr<SplatData> build_splat_data(const std::vector<GaussianVertex>& data,
                                                float scale_multiplier,
                                                float scene_scale) {
        const auto N = data.size();
        assert(N > 0);

        auto means = Tensor::empty({N, 3}, Device::CPU);
        auto scaling_raw = Tensor::empty({N, 3}, Device::CPU);
        auto rotation_raw = Tensor::empty({N, 4}, Device::CPU);
        auto opacity_raw = Tensor::empty({N, 1}, Device::CPU);
        auto sh0 = Tensor::empty({N, 1, 3}, Device::CPU);

        float* m_ptr = means.ptr<float>();
        float* s_ptr = scaling_raw.ptr<float>();
        float* r_ptr = rotation_raw.ptr<float>();
        float* o_ptr = opacity_raw.ptr<float>();
        float* c_ptr = sh0.ptr<float>();
        const float opacity_logit = -std::log(1.0f / 0.999f - 1.0f);

        for (size_t i = 0; i < N; ++i) {
            const auto& g = data[i];
            m_ptr[i * 3 + 0] = g.position.x;
            m_ptr[i * 3 + 1] = g.position.y;
            m_ptr[i * 3 + 2] = g.position.z;

            glm::vec3 scale(g.scale.x, g.scale.y, g.scale.z);
            scale *= scale_multiplier;
            scale = glm::max(scale, glm::vec3(1e-8f));
            s_ptr[i * 3 + 0] = std::log(scale.x);
            s_ptr[i * 3 + 1] = std::log(scale.y);
            s_ptr[i * 3 + 2] = std::log(scale.z);

            r_ptr[i * 4 + 0] = g.rotation.x;
            r_ptr[i * 4 + 1] = g.rotation.y;
            r_ptr[i * 4 + 2] = g.rotation.z;
            r_ptr[i * 4 + 3] = g.rotation.w;

            o_ptr[i] = opacity_logit;
            c_ptr[i * 3 + 0] = (g.color.x - 0.5f) / SH_C0;
            c_ptr[i * 3 + 1] = (g.color.y - 0.5f) / SH_C0;
            c_ptr[i * 3 + 2] = (g.color.z - 0.5f) / SH_C0;
        }

        means = means.to(Device::GPU);
        scaling_raw = scaling_raw.to(Device::GPU);
        rotation_raw = rotation_raw.to(Device::GPU);
        opacity_raw = opacity_raw.to(Device::GPU);
        sh0 = sh0.to(Device::GPU);
        auto shN = Tensor::zeros({N, 0, 3}, Device::GPU);

        return std::make_unique<SplatData>(0,
                                           std::move(means),
                                           std::move(sh0),
                                           std::move(shN),
                                           std::move(scaling_raw),
                                           std::move(rotation_raw),
                                           std::move(opacity_raw),
                                           scene_scale);
    }

} // namespace lfs::rendering::mesh2splat_detail

/* Derived from Mesh2Splat by Electronic Arts Inc.
 * Original: Copyright (c) 2025 Electronic Arts Inc. All rights reserved.
 * Licensed under BSD 3-Clause (see THIRD_PARTY_LICENSES.md)
 *
 * Modifications: Copyright (c) 2025-2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "rendering/mesh2splat.hpp"
#include "core/logger.hpp"
#include "core/mesh_data.hpp"
#include "core/tensor.hpp"

#include <glslang/Public/ResourceLimits.h>
#include <glslang/Public/ShaderLang.h>
#include <glslang/SPIRV/GlslangToSpv.h>
#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstring>
#include <expected>
#include <format>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include <glm/glm.hpp>

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
        constexpr VkFormat kFramebufferFormat = VK_FORMAT_R8G8B8A8_UNORM;
        constexpr VkFormat kTextureFormatLinear = VK_FORMAT_R8G8B8A8_UNORM;
        constexpr VkFormat kTextureFormatSrgb = VK_FORMAT_R8G8B8A8_SRGB;

        struct GaussianVertex {
            glm::vec4 position;
            glm::vec4 color;
            glm::vec4 scale;
            glm::vec4 normal;
            glm::vec4 rotation;
            glm::vec4 pbr;
        };
        static_assert(sizeof(GaussianVertex) == 6 * sizeof(glm::vec4));

        struct PerVertexData {
            glm::vec3 position;
            glm::vec3 normal;
            glm::vec4 tangent;
            glm::vec2 uv;
            glm::vec2 normalized_uv;
            glm::vec3 scale;
            glm::vec4 color;
        };
        static_assert(sizeof(PerVertexData) == 21 * sizeof(float));

        struct SubmeshGeometry {
            std::vector<PerVertexData> vertices;
            glm::vec3 bbox_min{std::numeric_limits<float>::max()};
            glm::vec3 bbox_max{std::numeric_limits<float>::lowest()};
            size_t material_index = 0;
        };

        struct PushConstants {
            glm::vec4 material_factor{1.0f};
            glm::vec4 bbox_min_metallic{0.0f};
            glm::vec4 bbox_max_roughness{0.0f};
            glm::ivec4 flags{0};
            glm::uvec4 limits{0};
        };
        static_assert(sizeof(PushConstants) <= 128);

        constexpr std::string_view kVertexShader = R"GLSL(
#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec4 tangent;
layout(location = 3) in vec2 uv;
layout(location = 4) in vec2 normalizedUv;
layout(location = 5) in vec3 scale;
layout(location = 6) in vec4 vertexColor;

layout(location = 0) out vec3 vPosition;
layout(location = 1) out vec3 vNormal;
layout(location = 2) out vec4 vTangent;
layout(location = 3) out vec2 vUv;
layout(location = 4) out vec2 vNormalizedUv;
layout(location = 5) out vec3 vScale;
layout(location = 6) out vec4 vVertexColor;

void main() {
    vPosition = position;
    vNormal = normal;
    vTangent = tangent;
    vUv = uv;
    vNormalizedUv = normalizedUv;
    vScale = scale;
    vVertexColor = vertexColor;
    gl_Position = vec4(position, 1.0);
}
)GLSL";

        constexpr std::string_view kGeometryShader = R"GLSL(
#version 450

layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;

layout(push_constant) uniform PushConstants {
    vec4 materialFactor;
    vec4 bboxMinMetallic;
    vec4 bboxMaxRoughness;
    ivec4 flags;
    uvec4 limits;
} pc;

layout(location = 0) in vec3 vPosition[];
layout(location = 1) in vec3 vNormal[];
layout(location = 2) in vec4 vTangent[];
layout(location = 3) in vec2 vUv[];
layout(location = 4) in vec2 vNormalizedUv[];
layout(location = 5) in vec3 vScale[];
layout(location = 6) in vec4 vVertexColor[];

layout(location = 0) out vec3 Position;
layout(location = 1) flat out vec3 Scale;
layout(location = 2) out vec2 UV;
layout(location = 3) out vec4 Tangent;
layout(location = 4) out vec4 VertexColor;
layout(location = 5) out vec3 Normal;
layout(location = 6) flat out vec4 Quaternion;

void transpose2x3(in mat2x3 m, out mat3x2 outputMat) {
    outputMat[0][0] = m[0][0];
    outputMat[1][0] = m[0][1];
    outputMat[2][0] = m[0][2];
    outputMat[0][1] = m[1][0];
    outputMat[1][1] = m[1][1];
    outputMat[2][1] = m[1][2];
}

vec4 quat_cast(mat3 m) {
    float fourXSquaredMinus1 = m[0][0] - m[1][1] - m[2][2];
    float fourYSquaredMinus1 = m[1][1] - m[0][0] - m[2][2];
    float fourZSquaredMinus1 = m[2][2] - m[0][0] - m[1][1];
    float fourWSquaredMinus1 = m[0][0] + m[1][1] + m[2][2];

    int biggestIndex = 0;
    float fourBiggestSquaredMinus1 = fourWSquaredMinus1;
    if (fourXSquaredMinus1 > fourBiggestSquaredMinus1) {
        fourBiggestSquaredMinus1 = fourXSquaredMinus1;
        biggestIndex = 1;
    }
    if (fourYSquaredMinus1 > fourBiggestSquaredMinus1) {
        fourBiggestSquaredMinus1 = fourYSquaredMinus1;
        biggestIndex = 2;
    }
    if (fourZSquaredMinus1 > fourBiggestSquaredMinus1) {
        fourBiggestSquaredMinus1 = fourZSquaredMinus1;
        biggestIndex = 3;
    }

    float biggestVal = sqrt(fourBiggestSquaredMinus1 + 1.0f) * 0.5f;
    float mult = 0.25f / biggestVal;
    vec4 q;

    if (biggestIndex == 0) {
        q.w = biggestVal;
        q.x = (m[1][2] - m[2][1]) * mult;
        q.y = (m[2][0] - m[0][2]) * mult;
        q.z = (m[0][1] - m[1][0]) * mult;
    } else if (biggestIndex == 1) {
        q.w = (m[1][2] - m[2][1]) * mult;
        q.x = biggestVal;
        q.y = (m[0][1] + m[1][0]) * mult;
        q.z = (m[2][0] + m[0][2]) * mult;
    } else if (biggestIndex == 2) {
        q.w = (m[2][0] - m[0][2]) * mult;
        q.x = (m[0][1] + m[1][0]) * mult;
        q.y = biggestVal;
        q.z = (m[1][2] + m[2][1]) * mult;
    } else {
        q.w = (m[0][1] - m[1][0]) * mult;
        q.x = (m[2][0] + m[0][2]) * mult;
        q.y = (m[1][2] + m[2][1]) * mult;
        q.z = biggestVal;
    }

    return q;
}

mat2 inverse2x2(mat2 m) {
    float determinant = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    if (determinant == 0.0) {
        return mat2(0.0);
    }
    float invDet = 1.0 / determinant;
    mat2 inverse;
    inverse[0][0] = m[1][1] * invDet;
    inverse[1][0] = -m[1][0] * invDet;
    inverse[0][1] = -m[0][1] * invDet;
    inverse[1][1] = m[0][0] * invDet;
    return inverse;
}

mat2x3 multiplyMat2x3WithMat2x2(mat2x3 matA, mat2 matB) {
    mat2x3 result;
    result[0][0] = matA[0][0] * matB[0][0] + matA[1][0] * matB[0][1];
    result[1][0] = matA[0][0] * matB[1][0] + matA[1][0] * matB[1][1];
    result[0][1] = matA[0][1] * matB[0][0] + matA[1][1] * matB[0][1];
    result[1][1] = matA[0][1] * matB[1][0] + matA[1][1] * matB[1][1];
    result[0][2] = matA[0][2] * matB[0][0] + matA[1][2] * matB[0][1];
    result[1][2] = matA[0][2] * matB[1][0] + matA[1][2] * matB[1][1];
    return result;
}

mat2x3 computeUv3DJacobian(vec3 verticesTriangle3D[3], vec2 verticesTriangleUV[3]) {
    vec3 pos0 = verticesTriangle3D[0];
    vec3 pos1 = verticesTriangle3D[1];
    vec3 pos2 = verticesTriangle3D[2];
    vec2 uv0 = verticesTriangleUV[0];
    vec2 uv1 = verticesTriangleUV[1];
    vec2 uv2 = verticesTriangleUV[2];

    mat2 UVMatrix;
    UVMatrix[0][0] = uv1.x - uv0.x;
    UVMatrix[1][0] = uv2.x - uv0.x;
    UVMatrix[0][1] = uv1.y - uv0.y;
    UVMatrix[1][1] = uv2.y - uv0.y;

    mat2x3 VMatrix;
    VMatrix[0][0] = pos1.x - pos0.x;
    VMatrix[1][0] = pos2.x - pos0.x;
    VMatrix[0][1] = pos1.y - pos0.y;
    VMatrix[1][1] = pos2.y - pos0.y;
    VMatrix[0][2] = pos1.z - pos0.z;
    VMatrix[1][2] = pos2.z - pos0.z;

    return multiplyMat2x3WithMat2x2(VMatrix, inverse2x2(UVMatrix));
}

void main() {
    vec3 edge1 = vPosition[1] - vPosition[0];
    vec3 edge2 = vPosition[2] - vPosition[0];
    vec3 edge3 = vPosition[2] - vPosition[1];

    vec3 u_bboxMin = pc.bboxMinMetallic.xyz;
    vec3 u_bboxMax = pc.bboxMaxRoughness.xyz;

    if (length(edge2) > length(edge1) && length(edge2) > length(edge3)) {
        vec3 temp = edge1;
        edge1 = edge2;
        edge2 = temp;
    } else if (length(edge3) > length(edge1) && length(edge3) > length(edge2)) {
        vec3 temp = edge1;
        edge1 = edge3;
        edge3 = temp;
    }

    edge1 = normalize(edge1);
    vec3 normal = normalize(cross(edge1, edge2));

    float absX = abs(normal.x);
    float absY = abs(normal.y);
    float absZ = abs(normal.z);

    vec2 orthogonalUvs[3];
    for (int i = 0; i < 3; i++) {
        float u, v;
        vec3 pos = vPosition[i];

        if (absX > absY && absX > absZ) {
            float rangeY = u_bboxMax.y - u_bboxMin.y;
            float rangeZ = u_bboxMax.z - u_bboxMin.z;
            float range = max(rangeY, rangeZ);
            u = (pos.y - u_bboxMin.y) / range;
            v = (pos.z - u_bboxMin.z) / range;
        } else if (absY > absZ) {
            float rangeX = u_bboxMax.x - u_bboxMin.x;
            float rangeZ = u_bboxMax.z - u_bboxMin.z;
            float range = max(rangeX, rangeZ);
            u = (pos.x - u_bboxMin.x) / range;
            v = (pos.z - u_bboxMin.z) / range;
        } else {
            float rangeX = u_bboxMax.x - u_bboxMin.x;
            float rangeY = u_bboxMax.y - u_bboxMin.y;
            float range = max(rangeX, rangeY);
            u = (pos.x - u_bboxMin.x) / range;
            v = (pos.y - u_bboxMin.y) / range;
        }

        orthogonalUvs[i] = vec2(u, v);
    }

    vec3 xAxis = edge1;
    vec3 yAxis = normalize(cross(normal, xAxis));
    vec3 zAxis = normal;

    mat3 rotationMatrix = mat3(xAxis, yAxis, zAxis);
    vec4 q = quat_cast(rotationMatrix);
    vec4 quaternion = vec4(q.w, q.x, q.y, q.z);

    vec3 true_vertices3D[3] = {vPosition[0], vPosition[1], vPosition[2]};
    vec2 true_normalized_vertices2D[3] = {orthogonalUvs[0], orthogonalUvs[1], orthogonalUvs[2]};
    mat2x3 J = computeUv3DJacobian(true_vertices3D, true_normalized_vertices2D);

    mat3x2 J_T;
    transpose2x3(J, J_T);

    vec3 Ju = vec3(J_T[0][0], J_T[1][0], J_T[2][0]);
    vec3 Jv = vec3(J_T[0][1], J_T[1][1], J_T[2][1]);

    Scale = vec3(length(Ju), length(Jv), 1e-7);

    for (int i = 0; i < 3; i++) {
        Tangent = vTangent[i];
        Position = vPosition[i];
        Normal = vNormal[i];
        UV = vUv[i];
        VertexColor = vVertexColor[i];
        Quaternion = quaternion;
        gl_Position = vec4(orthogonalUvs[i] * 2.0 - 1.0, 0.0, 1.0);
        EmitVertex();
    }
    EndPrimitive();
}
)GLSL";

        constexpr std::string_view kFragmentShader = R"GLSL(
#version 450

layout(set = 0, binding = 1, std430) buffer CounterBuffer {
    uint g_validCounter;
};

layout(set = 0, binding = 2) uniform sampler2D albedoTexture;
layout(set = 0, binding = 3) uniform sampler2D normalTexture;
layout(set = 0, binding = 4) uniform sampler2D metallicRoughnessTexture;

struct GaussianVertex {
    vec4 position;
    vec4 color;
    vec4 scale;
    vec4 normal;
    vec4 rotation;
    vec4 pbr;
};

layout(set = 0, binding = 0, std430) buffer GaussianBuffer {
    GaussianVertex vertices[];
} gaussianBuffer;

layout(push_constant) uniform PushConstants {
    vec4 materialFactor;
    vec4 bboxMinMetallic;
    vec4 bboxMaxRoughness;
    ivec4 flags;
    uvec4 limits;
} pc;

layout(location = 0) in vec3 Position;
layout(location = 1) flat in vec3 Scale;
layout(location = 2) in vec2 UV;
layout(location = 3) in vec4 Tangent;
layout(location = 4) in vec4 VertexColor;
layout(location = 5) in vec3 Normal;
layout(location = 6) flat in vec4 Quaternion;

void main() {
    uint index = atomicAdd(g_validCounter, 1u);
    if (index >= pc.limits.x) {
        return;
    }

    vec4 out_Color = vec4(1.0);
    if (pc.flags.x == 1) {
        out_Color = texture(albedoTexture, UV);
    } else if (pc.flags.w == 1) {
        out_Color = VertexColor;
    }

    vec3 out_Normal;
    if (pc.flags.y == 1) {
        vec3 normalMap_normal = texture(normalTexture, UV).xyz;
        vec3 retrievedNormal = normalize(normalMap_normal * 2.0 - 1.0);
        vec3 bitangent = normalize(cross(Normal, Tangent.xyz)) * Tangent.w;
        mat3 TBN = mat3(Tangent.xyz, bitangent, normalize(Normal));
        out_Normal = normalize(TBN * retrievedNormal);
    } else {
        out_Normal = normalize(Normal);
    }

    float metallic = pc.bboxMinMetallic.w;
    float roughness = pc.bboxMaxRoughness.w;
    if (pc.flags.z == 1) {
        vec3 orm = texture(metallicRoughnessTexture, UV).rgb;
        roughness *= orm.g;
        metallic *= orm.b;
    }
    roughness = max(roughness, 0.04);

    vec3 albedo = (out_Color * pc.materialFactor).rgb;
    vec3 color = pow(clamp(albedo, 0.0, 1.0), vec3(1.0 / 2.2));

    gaussianBuffer.vertices[index].position = vec4(Position.xyz, 1.0);
    gaussianBuffer.vertices[index].color = vec4(color, out_Color.a);
    gaussianBuffer.vertices[index].scale = vec4(Scale, 0.0);
    gaussianBuffer.vertices[index].normal = vec4(out_Normal, 0.0);
    gaussianBuffer.vertices[index].rotation = Quaternion;
    gaussianBuffer.vertices[index].pbr = vec4(metallic, roughness, 0.0, 1.0);
}
)GLSL";

        [[nodiscard]] const char* vkResultName(const VkResult result) {
            switch (result) {
            case VK_SUCCESS: return "VK_SUCCESS";
            case VK_NOT_READY: return "VK_NOT_READY";
            case VK_TIMEOUT: return "VK_TIMEOUT";
            case VK_EVENT_SET: return "VK_EVENT_SET";
            case VK_EVENT_RESET: return "VK_EVENT_RESET";
            case VK_INCOMPLETE: return "VK_INCOMPLETE";
            case VK_ERROR_OUT_OF_HOST_MEMORY: return "VK_ERROR_OUT_OF_HOST_MEMORY";
            case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
            case VK_ERROR_INITIALIZATION_FAILED: return "VK_ERROR_INITIALIZATION_FAILED";
            case VK_ERROR_DEVICE_LOST: return "VK_ERROR_DEVICE_LOST";
            case VK_ERROR_MEMORY_MAP_FAILED: return "VK_ERROR_MEMORY_MAP_FAILED";
            case VK_ERROR_LAYER_NOT_PRESENT: return "VK_ERROR_LAYER_NOT_PRESENT";
            case VK_ERROR_EXTENSION_NOT_PRESENT: return "VK_ERROR_EXTENSION_NOT_PRESENT";
            case VK_ERROR_FEATURE_NOT_PRESENT: return "VK_ERROR_FEATURE_NOT_PRESENT";
            case VK_ERROR_INCOMPATIBLE_DRIVER: return "VK_ERROR_INCOMPATIBLE_DRIVER";
            case VK_ERROR_TOO_MANY_OBJECTS: return "VK_ERROR_TOO_MANY_OBJECTS";
            case VK_ERROR_FORMAT_NOT_SUPPORTED: return "VK_ERROR_FORMAT_NOT_SUPPORTED";
            default: return "VK_ERROR_UNKNOWN";
            }
        }

        [[nodiscard]] std::string vkError(const char* op, const VkResult result) {
            return std::format("{} failed: {}", op, vkResultName(result));
        }

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

        [[nodiscard]] std::vector<SubmeshGeometry> extract_geometry(const MeshData& mesh) {
            assert(mesh.vertices.is_valid());
            assert(mesh.indices.is_valid());
            assert(mesh.vertices.dtype() == DataType::Float32);
            assert(mesh.indices.dtype() == DataType::Int32);

            auto verts_cpu = mesh.vertices.device() == Device::CPU ? mesh.vertices : mesh.vertices.to(Device::CPU);
            auto idx_cpu = mesh.indices.device() == Device::CPU ? mesh.indices : mesh.indices.to(Device::CPU);

            const float* verts_ptr = verts_cpu.ptr<float>();
            const int32_t* idx_ptr = idx_cpu.ptr<int32_t>();
            const auto V = static_cast<int64_t>(verts_cpu.shape()[0]);
            const auto F = static_cast<int64_t>(idx_cpu.shape()[0]);

            const float* normals_ptr = nullptr;
            Tensor normals_cpu;
            if (mesh.has_normals()) {
                normals_cpu = mesh.normals.device() == Device::CPU ? mesh.normals : mesh.normals.to(Device::CPU);
                normals_ptr = normals_cpu.ptr<float>();
                assert(normals_cpu.shape()[0] == verts_cpu.shape()[0]);
            }

            const float* tangents_ptr = nullptr;
            Tensor tangents_cpu;
            if (mesh.has_tangents()) {
                tangents_cpu = mesh.tangents.device() == Device::CPU ? mesh.tangents : mesh.tangents.to(Device::CPU);
                tangents_ptr = tangents_cpu.ptr<float>();
                assert(tangents_cpu.shape()[0] == verts_cpu.shape()[0]);
            }

            const float* texcoords_ptr = nullptr;
            Tensor texcoords_cpu;
            if (mesh.has_texcoords()) {
                texcoords_cpu = mesh.texcoords.device() == Device::CPU ? mesh.texcoords : mesh.texcoords.to(Device::CPU);
                texcoords_ptr = texcoords_cpu.ptr<float>();
                assert(texcoords_cpu.shape()[0] == verts_cpu.shape()[0]);
            }

            const float* colors_ptr = nullptr;
            Tensor colors_cpu;
            if (mesh.has_colors()) {
                colors_cpu = mesh.colors.device() == Device::CPU ? mesh.colors : mesh.colors.to(Device::CPU);
                colors_ptr = colors_cpu.ptr<float>();
                assert(colors_cpu.shape()[0] == verts_cpu.shape()[0]);
            }

            std::vector<Submesh> submeshes;
            if (mesh.submeshes.empty()) {
                submeshes.push_back({0, static_cast<size_t>(F) * 3, 0});
            } else {
                submeshes = mesh.submeshes;
            }

            std::vector<SubmeshGeometry> result;
            result.reserve(submeshes.size());

            for (const auto& sub : submeshes) {
                SubmeshGeometry geo;
                geo.material_index = sub.material_index;
                assert(sub.index_count % 3 == 0);
                const size_t face_count = sub.index_count / 3;
                geo.vertices.reserve(sub.index_count);

                for (size_t f = 0; f < face_count; ++f) {
                    const size_t base = sub.start_index + f * 3;
                    const int32_t indices[3] = {idx_ptr[base + 0], idx_ptr[base + 1], idx_ptr[base + 2]};
                    assert(indices[0] >= 0 && indices[0] < V);
                    assert(indices[1] >= 0 && indices[1] < V);
                    assert(indices[2] >= 0 && indices[2] < V);

                    glm::vec3 pos[3];
                    glm::vec3 nrm[3];
                    glm::vec4 tan[3];
                    glm::vec2 uv[3];
                    glm::vec4 col[3] = {glm::vec4(1.0f), glm::vec4(1.0f), glm::vec4(1.0f)};

                    for (int k = 0; k < 3; ++k) {
                        const int32_t vi = indices[k];
                        pos[k] = {verts_ptr[vi * 3 + 0], verts_ptr[vi * 3 + 1], verts_ptr[vi * 3 + 2]};
                        geo.bbox_min = glm::min(geo.bbox_min, pos[k]);
                        geo.bbox_max = glm::max(geo.bbox_max, pos[k]);

                        uv[k] = texcoords_ptr ? glm::vec2(texcoords_ptr[vi * 2 + 0], texcoords_ptr[vi * 2 + 1]) : glm::vec2(0.0f);
                        if (normals_ptr)
                            nrm[k] = {normals_ptr[vi * 3 + 0], normals_ptr[vi * 3 + 1], normals_ptr[vi * 3 + 2]};
                        if (tangents_ptr)
                            tan[k] = {tangents_ptr[vi * 4 + 0], tangents_ptr[vi * 4 + 1], tangents_ptr[vi * 4 + 2], tangents_ptr[vi * 4 + 3]};
                        if (colors_ptr)
                            col[k] = {colors_ptr[vi * 4 + 0], colors_ptr[vi * 4 + 1], colors_ptr[vi * 4 + 2], colors_ptr[vi * 4 + 3]};
                    }

                    if (!normals_ptr) {
                        const glm::vec3 fn = compute_face_normal(pos[0], pos[1], pos[2]);
                        nrm[0] = nrm[1] = nrm[2] = fn;
                    }
                    if (!tangents_ptr) {
                        const glm::vec4 ft = compute_face_tangent(pos[0], pos[1], pos[2], uv[0], uv[1], uv[2]);
                        tan[0] = tan[1] = tan[2] = ft;
                    }

                    for (int k = 0; k < 3; ++k) {
                        geo.vertices.push_back(PerVertexData{pos[k], nrm[k], tan[k], uv[k], glm::vec2(0.0f), glm::vec3(0.0f), col[k]});
                    }
                }

                if (!geo.vertices.empty())
                    result.push_back(std::move(geo));
            }

            return result;
        }

        class GlslangLifetime {
        public:
            GlslangLifetime() { glslang::InitializeProcess(); }
            ~GlslangLifetime() { glslang::FinalizeProcess(); }
            GlslangLifetime(const GlslangLifetime&) = delete;
            GlslangLifetime& operator=(const GlslangLifetime&) = delete;
        };

        [[nodiscard]] std::expected<std::vector<uint32_t>, std::string> compile_shader(std::string_view source,
                                                                                        EShLanguage stage,
                                                                                        const char* label) {
            static std::mutex mutex;
            std::lock_guard lock(mutex);
            static GlslangLifetime lifetime;

            const std::string source_copy(source);
            const char* source_ptr = source_copy.c_str();
            glslang::TShader shader(stage);
            shader.setStrings(&source_ptr, 1);
            shader.setEnvInput(glslang::EShSourceGlsl, stage, glslang::EShClientVulkan, 100);
            shader.setEnvClient(glslang::EShClientVulkan, glslang::EShTargetVulkan_1_0);
            shader.setEnvTarget(glslang::EShTargetSpv, glslang::EShTargetSpv_1_0);

            constexpr auto messages = static_cast<EShMessages>(EShMsgSpvRules | EShMsgVulkanRules);
            if (!shader.parse(GetDefaultResources(), 450, false, messages)) {
                return std::unexpected(std::format("mesh2splat {} shader compile failed: {} {}", label, shader.getInfoLog(), shader.getInfoDebugLog()));
            }

            glslang::TProgram program;
            program.addShader(&shader);
            if (!program.link(messages)) {
                return std::unexpected(std::format("mesh2splat {} shader link failed: {} {}", label, program.getInfoLog(), program.getInfoDebugLog()));
            }

            std::vector<uint32_t> spirv;
            glslang::GlslangToSpv(*program.getIntermediate(stage), spirv);
            return spirv;
        }

        struct Buffer {
            VkDevice device = VK_NULL_HANDLE;
            VkBuffer buffer = VK_NULL_HANDLE;
            VkDeviceMemory memory = VK_NULL_HANDLE;
            VkDeviceSize size = 0;

            ~Buffer() { reset(); }
            Buffer() = default;
            Buffer(const Buffer&) = delete;
            Buffer& operator=(const Buffer&) = delete;
            Buffer(Buffer&& other) noexcept { *this = std::move(other); }
            Buffer& operator=(Buffer&& other) noexcept {
                if (this == &other)
                    return *this;
                reset();
                device = other.device;
                buffer = other.buffer;
                memory = other.memory;
                size = other.size;
                other.device = VK_NULL_HANDLE;
                other.buffer = VK_NULL_HANDLE;
                other.memory = VK_NULL_HANDLE;
                other.size = 0;
                return *this;
            }
            void reset() {
                if (buffer)
                    vkDestroyBuffer(device, buffer, nullptr);
                if (memory)
                    vkFreeMemory(device, memory, nullptr);
                buffer = VK_NULL_HANDLE;
                memory = VK_NULL_HANDLE;
                size = 0;
            }
        };

        struct Image {
            VkDevice device = VK_NULL_HANDLE;
            VkImage image = VK_NULL_HANDLE;
            VkDeviceMemory memory = VK_NULL_HANDLE;
            VkImageView view = VK_NULL_HANDLE;
            VkFormat format = VK_FORMAT_UNDEFINED;
            uint32_t width = 0;
            uint32_t height = 0;

            ~Image() { reset(); }
            Image() = default;
            Image(const Image&) = delete;
            Image& operator=(const Image&) = delete;
            Image(Image&& other) noexcept { *this = std::move(other); }
            Image& operator=(Image&& other) noexcept {
                if (this == &other)
                    return *this;
                reset();
                device = other.device;
                image = other.image;
                memory = other.memory;
                view = other.view;
                format = other.format;
                width = other.width;
                height = other.height;
                other.device = VK_NULL_HANDLE;
                other.image = VK_NULL_HANDLE;
                other.memory = VK_NULL_HANDLE;
                other.view = VK_NULL_HANDLE;
                other.format = VK_FORMAT_UNDEFINED;
                other.width = 0;
                other.height = 0;
                return *this;
            }
            void reset() {
                if (view)
                    vkDestroyImageView(device, view, nullptr);
                if (image)
                    vkDestroyImage(device, image, nullptr);
                if (memory)
                    vkFreeMemory(device, memory, nullptr);
                image = VK_NULL_HANDLE;
                memory = VK_NULL_HANDLE;
                view = VK_NULL_HANDLE;
            }
        };

        struct ShaderModule {
            VkDevice device = VK_NULL_HANDLE;
            VkShaderModule module = VK_NULL_HANDLE;
            ~ShaderModule() {
                if (module)
                    vkDestroyShaderModule(device, module, nullptr);
            }
            ShaderModule() = default;
            ShaderModule(const ShaderModule&) = delete;
            ShaderModule& operator=(const ShaderModule&) = delete;
            ShaderModule(ShaderModule&& other) noexcept { *this = std::move(other); }
            ShaderModule& operator=(ShaderModule&& other) noexcept {
                if (this == &other)
                    return *this;
                if (module)
                    vkDestroyShaderModule(device, module, nullptr);
                device = other.device;
                module = other.module;
                other.device = VK_NULL_HANDLE;
                other.module = VK_NULL_HANDLE;
                return *this;
            }
        };

        class VulkanMesh2SplatContext {
        public:
            ~VulkanMesh2SplatContext() { shutdown(); }

            [[nodiscard]] bool ensure(std::string& error) {
                if (device_)
                    return true;
                return createInstance(error) && pickPhysicalDevice(error) && createDevice(error) && createCommandPool(error);
            }

            [[nodiscard]] VkDevice device() const { return device_; }
            [[nodiscard]] VkPhysicalDevice physicalDevice() const { return physical_device_; }
            [[nodiscard]] VkQueue queue() const { return queue_; }

            [[nodiscard]] uint32_t findMemoryType(uint32_t type_filter,
                                                   VkMemoryPropertyFlags properties,
                                                   std::string& error) const {
                VkPhysicalDeviceMemoryProperties mem_properties{};
                vkGetPhysicalDeviceMemoryProperties(physical_device_, &mem_properties);
                for (uint32_t i = 0; i < mem_properties.memoryTypeCount; ++i) {
                    if ((type_filter & (1u << i)) && (mem_properties.memoryTypes[i].propertyFlags & properties) == properties)
                        return i;
                }
                error = "No compatible Vulkan memory type for Mesh2Splat";
                return std::numeric_limits<uint32_t>::max();
            }

            [[nodiscard]] std::expected<Buffer, std::string> createBuffer(VkDeviceSize size,
                                                                           VkBufferUsageFlags usage,
                                                                           VkMemoryPropertyFlags properties) const {
                Buffer out;
                out.device = device_;
                out.size = size;

                VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
                info.size = size;
                info.usage = usage;
                info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
                VkResult result = vkCreateBuffer(device_, &info, nullptr, &out.buffer);
                if (result != VK_SUCCESS)
                    return std::unexpected(vkError("vkCreateBuffer", result));

                VkMemoryRequirements req{};
                vkGetBufferMemoryRequirements(device_, out.buffer, &req);
                std::string error;
                const uint32_t mem_type = findMemoryType(req.memoryTypeBits, properties, error);
                if (mem_type == std::numeric_limits<uint32_t>::max())
                    return std::unexpected(error);

                VkMemoryAllocateInfo alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
                alloc.allocationSize = req.size;
                alloc.memoryTypeIndex = mem_type;
                result = vkAllocateMemory(device_, &alloc, nullptr, &out.memory);
                if (result != VK_SUCCESS)
                    return std::unexpected(vkError("vkAllocateMemory(buffer)", result));
                result = vkBindBufferMemory(device_, out.buffer, out.memory, 0);
                if (result != VK_SUCCESS)
                    return std::unexpected(vkError("vkBindBufferMemory", result));
                return out;
            }

            [[nodiscard]] std::expected<Image, std::string> createImage(uint32_t width,
                                                                         uint32_t height,
                                                                         VkFormat format,
                                                                         VkImageUsageFlags usage) const {
                Image out;
                out.device = device_;
                out.format = format;
                out.width = width;
                out.height = height;

                VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
                info.imageType = VK_IMAGE_TYPE_2D;
                info.format = format;
                info.extent = {width, height, 1};
                info.mipLevels = 1;
                info.arrayLayers = 1;
                info.samples = VK_SAMPLE_COUNT_1_BIT;
                info.tiling = VK_IMAGE_TILING_OPTIMAL;
                info.usage = usage;
                info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
                info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                VkResult result = vkCreateImage(device_, &info, nullptr, &out.image);
                if (result != VK_SUCCESS)
                    return std::unexpected(vkError("vkCreateImage", result));

                VkMemoryRequirements req{};
                vkGetImageMemoryRequirements(device_, out.image, &req);
                std::string error;
                const uint32_t mem_type = findMemoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, error);
                if (mem_type == std::numeric_limits<uint32_t>::max())
                    return std::unexpected(error);

                VkMemoryAllocateInfo alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
                alloc.allocationSize = req.size;
                alloc.memoryTypeIndex = mem_type;
                result = vkAllocateMemory(device_, &alloc, nullptr, &out.memory);
                if (result != VK_SUCCESS)
                    return std::unexpected(vkError("vkAllocateMemory(image)", result));
                result = vkBindImageMemory(device_, out.image, out.memory, 0);
                if (result != VK_SUCCESS)
                    return std::unexpected(vkError("vkBindImageMemory", result));

                VkImageViewCreateInfo view_info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
                view_info.image = out.image;
                view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
                view_info.format = format;
                view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                view_info.subresourceRange.baseMipLevel = 0;
                view_info.subresourceRange.levelCount = 1;
                view_info.subresourceRange.baseArrayLayer = 0;
                view_info.subresourceRange.layerCount = 1;
                result = vkCreateImageView(device_, &view_info, nullptr, &out.view);
                if (result != VK_SUCCESS)
                    return std::unexpected(vkError("vkCreateImageView", result));
                return out;
            }

            [[nodiscard]] std::expected<VkCommandBuffer, std::string> beginCommands() const {
                VkCommandBufferAllocateInfo alloc{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
                alloc.commandPool = command_pool_;
                alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
                alloc.commandBufferCount = 1;
                VkCommandBuffer cmd = VK_NULL_HANDLE;
                VkResult result = vkAllocateCommandBuffers(device_, &alloc, &cmd);
                if (result != VK_SUCCESS)
                    return std::unexpected(vkError("vkAllocateCommandBuffers", result));
                VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
                begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
                result = vkBeginCommandBuffer(cmd, &begin);
                if (result != VK_SUCCESS) {
                    vkFreeCommandBuffers(device_, command_pool_, 1, &cmd);
                    return std::unexpected(vkError("vkBeginCommandBuffer", result));
                }
                return cmd;
            }

            [[nodiscard]] std::optional<std::string> submitAndWait(VkCommandBuffer cmd) const {
                VkResult result = vkEndCommandBuffer(cmd);
                if (result != VK_SUCCESS) {
                    vkFreeCommandBuffers(device_, command_pool_, 1, &cmd);
                    return vkError("vkEndCommandBuffer", result);
                }
                VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
                VkFence fence = VK_NULL_HANDLE;
                result = vkCreateFence(device_, &fence_info, nullptr, &fence);
                if (result != VK_SUCCESS) {
                    vkFreeCommandBuffers(device_, command_pool_, 1, &cmd);
                    return vkError("vkCreateFence", result);
                }
                VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
                submit.commandBufferCount = 1;
                submit.pCommandBuffers = &cmd;
                result = vkQueueSubmit(queue_, 1, &submit, fence);
                if (result == VK_SUCCESS)
                    result = vkWaitForFences(device_, 1, &fence, VK_TRUE, UINT64_MAX);
                vkDestroyFence(device_, fence, nullptr);
                vkFreeCommandBuffers(device_, command_pool_, 1, &cmd);
                if (result != VK_SUCCESS)
                    return vkError("vkQueueSubmit/vkWaitForFences", result);
                return std::nullopt;
            }

            [[nodiscard]] std::optional<std::string> uploadBuffer(const void* data, VkDeviceSize size, Buffer& dst) const {
                auto staging = createBuffer(size,
                                            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
                if (!staging)
                    return staging.error();
                void* mapped = nullptr;
                VkResult result = vkMapMemory(device_, staging->memory, 0, size, 0, &mapped);
                if (result != VK_SUCCESS)
                    return vkError("vkMapMemory(upload)", result);
                std::memcpy(mapped, data, static_cast<size_t>(size));
                vkUnmapMemory(device_, staging->memory);

                auto cmd = beginCommands();
                if (!cmd)
                    return cmd.error();
                VkBufferCopy copy{};
                copy.size = size;
                vkCmdCopyBuffer(*cmd, staging->buffer, dst.buffer, 1, &copy);
                return submitAndWait(*cmd);
            }

        private:
            bool createInstance(std::string& error) {
                VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
                app.pApplicationName = "LichtFeld Mesh2Splat";
                app.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
                app.pEngineName = "LichtFeld Studio";
                app.engineVersion = VK_MAKE_VERSION(1, 0, 0);
                app.apiVersion = VK_API_VERSION_1_0;

                VkInstanceCreateInfo info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
                info.pApplicationInfo = &app;
                const VkResult result = vkCreateInstance(&info, nullptr, &instance_);
                if (result != VK_SUCCESS) {
                    error = vkError("vkCreateInstance", result);
                    return false;
                }
                return true;
            }

            bool pickPhysicalDevice(std::string& error) {
                uint32_t count = 0;
                VkResult result = vkEnumeratePhysicalDevices(instance_, &count, nullptr);
                if (result != VK_SUCCESS || count == 0) {
                    error = result == VK_SUCCESS ? "No Vulkan physical devices available" : vkError("vkEnumeratePhysicalDevices", result);
                    return false;
                }
                std::vector<VkPhysicalDevice> devices(count);
                result = vkEnumeratePhysicalDevices(instance_, &count, devices.data());
                if (result != VK_SUCCESS) {
                    error = vkError("vkEnumeratePhysicalDevices", result);
                    return false;
                }

                auto score_device = [&](VkPhysicalDevice candidate, uint32_t& family) -> int {
                    VkPhysicalDeviceFeatures features{};
                    vkGetPhysicalDeviceFeatures(candidate, &features);
                    if (!features.geometryShader || !features.fragmentStoresAndAtomics)
                        return -1;

                    uint32_t family_count = 0;
                    vkGetPhysicalDeviceQueueFamilyProperties(candidate, &family_count, nullptr);
                    std::vector<VkQueueFamilyProperties> families(family_count);
                    vkGetPhysicalDeviceQueueFamilyProperties(candidate, &family_count, families.data());
                    for (uint32_t i = 0; i < family_count; ++i) {
                        if (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                            family = i;
                            VkPhysicalDeviceProperties props{};
                            vkGetPhysicalDeviceProperties(candidate, &props);
                            return props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? 100 : 10;
                        }
                    }
                    return -1;
                };

                int best_score = -1;
                for (VkPhysicalDevice candidate : devices) {
                    uint32_t family = 0;
                    const int score = score_device(candidate, family);
                    if (score > best_score) {
                        best_score = score;
                        physical_device_ = candidate;
                        graphics_family_ = family;
                    }
                }

                if (!physical_device_) {
                    error = "No Vulkan device supports graphics, geometry shader, and fragment storage writes required by Mesh2Splat";
                    return false;
                }
                return true;
            }

            bool createDevice(std::string& error) {
                constexpr float priority = 1.0f;
                VkDeviceQueueCreateInfo queue_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
                queue_info.queueFamilyIndex = graphics_family_;
                queue_info.queueCount = 1;
                queue_info.pQueuePriorities = &priority;

                VkPhysicalDeviceFeatures features{};
                features.geometryShader = VK_TRUE;
                features.fragmentStoresAndAtomics = VK_TRUE;
                features.shaderStorageImageWriteWithoutFormat = VK_FALSE;

                VkDeviceCreateInfo info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
                info.queueCreateInfoCount = 1;
                info.pQueueCreateInfos = &queue_info;
                info.pEnabledFeatures = &features;
                const VkResult result = vkCreateDevice(physical_device_, &info, nullptr, &device_);
                if (result != VK_SUCCESS) {
                    error = vkError("vkCreateDevice", result);
                    return false;
                }
                vkGetDeviceQueue(device_, graphics_family_, 0, &queue_);
                return true;
            }

            bool createCommandPool(std::string& error) {
                VkCommandPoolCreateInfo info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
                info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
                info.queueFamilyIndex = graphics_family_;
                const VkResult result = vkCreateCommandPool(device_, &info, nullptr, &command_pool_);
                if (result != VK_SUCCESS) {
                    error = vkError("vkCreateCommandPool", result);
                    return false;
                }
                return true;
            }

            void shutdown() {
                if (device_)
                    vkDeviceWaitIdle(device_);
                if (command_pool_)
                    vkDestroyCommandPool(device_, command_pool_, nullptr);
                if (device_)
                    vkDestroyDevice(device_, nullptr);
                if (instance_)
                    vkDestroyInstance(instance_, nullptr);
                command_pool_ = VK_NULL_HANDLE;
                device_ = VK_NULL_HANDLE;
                physical_device_ = VK_NULL_HANDLE;
                instance_ = VK_NULL_HANDLE;
            }

            VkInstance instance_ = VK_NULL_HANDLE;
            VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
            VkDevice device_ = VK_NULL_HANDLE;
            VkQueue queue_ = VK_NULL_HANDLE;
            VkCommandPool command_pool_ = VK_NULL_HANDLE;
            uint32_t graphics_family_ = 0;
        };

        VulkanMesh2SplatContext& vulkan_context() {
            static VulkanMesh2SplatContext context;
            return context;
        }

        [[nodiscard]] std::optional<std::string> map_write(VkDevice device, Buffer& buffer, const void* data, VkDeviceSize size) {
            void* mapped = nullptr;
            const VkResult result = vkMapMemory(device, buffer.memory, 0, size, 0, &mapped);
            if (result != VK_SUCCESS)
                return vkError("vkMapMemory(write)", result);
            std::memcpy(mapped, data, static_cast<size_t>(size));
            vkUnmapMemory(device, buffer.memory);
            return std::nullopt;
        }

        [[nodiscard]] std::expected<ShaderModule, std::string> create_shader_module(VkDevice device,
                                                                                     std::string_view source,
                                                                                     EShLanguage stage,
                                                                                     const char* label) {
            auto spirv = compile_shader(source, stage, label);
            if (!spirv)
                return std::unexpected(spirv.error());

            ShaderModule module;
            module.device = device;
            VkShaderModuleCreateInfo info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
            info.codeSize = spirv->size() * sizeof(uint32_t);
            info.pCode = spirv->data();
            const VkResult result = vkCreateShaderModule(device, &info, nullptr, &module.module);
            if (result != VK_SUCCESS)
                return std::unexpected(vkError("vkCreateShaderModule", result));
            return module;
        }

        [[nodiscard]] std::vector<uint8_t> to_rgba8(const TextureImage& img) {
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

        [[nodiscard]] std::expected<Image, std::string> upload_texture(VulkanMesh2SplatContext& ctx,
                                                                        const TextureImage& img,
                                                                        bool srgb) {
            if (img.width <= 0 || img.height <= 0 || img.pixels.empty())
                return std::unexpected("Invalid Mesh2Splat texture image");

            const VkFormat format = srgb && img.channels >= 3 ? kTextureFormatSrgb : kTextureFormatLinear;
            auto image = ctx.createImage(static_cast<uint32_t>(img.width),
                                         static_cast<uint32_t>(img.height),
                                         format,
                                         VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
            if (!image)
                return std::unexpected(image.error());

            const std::vector<uint8_t> rgba = to_rgba8(img);
            auto staging = ctx.createBuffer(rgba.size(),
                                            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            if (!staging)
                return std::unexpected(staging.error());
            if (auto error = map_write(ctx.device(), *staging, rgba.data(), rgba.size()))
                return std::unexpected(*error);

            auto cmd = ctx.beginCommands();
            if (!cmd)
                return std::unexpected(cmd.error());

            VkImageMemoryBarrier to_transfer{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            to_transfer.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            to_transfer.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            to_transfer.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            to_transfer.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            to_transfer.image = image->image;
            to_transfer.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            to_transfer.subresourceRange.levelCount = 1;
            to_transfer.subresourceRange.layerCount = 1;
            to_transfer.srcAccessMask = 0;
            to_transfer.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            vkCmdPipelineBarrier(*cmd,
                                 VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 0,
                                 0,
                                 nullptr,
                                 0,
                                 nullptr,
                                 1,
                                 &to_transfer);

            VkBufferImageCopy copy{};
            copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            copy.imageSubresource.layerCount = 1;
            copy.imageExtent = {static_cast<uint32_t>(img.width), static_cast<uint32_t>(img.height), 1};
            vkCmdCopyBufferToImage(*cmd, staging->buffer, image->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);

            VkImageMemoryBarrier to_sample{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            to_sample.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            to_sample.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            to_sample.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            to_sample.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            to_sample.image = image->image;
            to_sample.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            to_sample.subresourceRange.levelCount = 1;
            to_sample.subresourceRange.layerCount = 1;
            to_sample.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            to_sample.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(*cmd,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                                 0,
                                 0,
                                 nullptr,
                                 0,
                                 nullptr,
                                 1,
                                 &to_sample);

            if (auto error = ctx.submitAndWait(*cmd))
                return std::unexpected(*error);
            return std::move(*image);
        }

        [[nodiscard]] std::expected<Image, std::string> create_dummy_texture(VulkanMesh2SplatContext& ctx) {
            TextureImage img;
            img.width = 1;
            img.height = 1;
            img.channels = 4;
            img.pixels = {255, 255, 255, 255};
            return upload_texture(ctx, img, false);
        }

        struct MaterialTextures {
            Image albedo;
            Image normal;
            Image metallic_roughness;
            bool has_albedo = false;
            bool has_normal = false;
            bool has_metallic_roughness = false;
        };

        [[nodiscard]] std::expected<std::vector<MaterialTextures>, std::string> upload_material_textures(VulkanMesh2SplatContext& ctx,
                                                                                                          const MeshData& mesh,
                                                                                                          const Image& dummy) {
            (void)dummy;
            std::vector<MaterialTextures> textures(mesh.materials.size());
            for (size_t i = 0; i < mesh.materials.size(); ++i) {
                const auto& mat = mesh.materials[i];
                if (mat.has_albedo_texture() && mat.albedo_tex > 0 && mat.albedo_tex <= mesh.texture_images.size()) {
                    const auto& img = mesh.texture_images[mat.albedo_tex - 1];
                    if (!img.pixels.empty()) {
                        auto tex = upload_texture(ctx, img, true);
                        if (!tex)
                            return std::unexpected(tex.error());
                        textures[i].albedo = std::move(*tex);
                        textures[i].has_albedo = true;
                    }
                }
                if (mat.has_normal_texture() && mat.normal_tex > 0 && mat.normal_tex <= mesh.texture_images.size()) {
                    const auto& img = mesh.texture_images[mat.normal_tex - 1];
                    if (!img.pixels.empty()) {
                        auto tex = upload_texture(ctx, img, false);
                        if (!tex)
                            return std::unexpected(tex.error());
                        textures[i].normal = std::move(*tex);
                        textures[i].has_normal = true;
                    }
                }
                if (mat.has_metallic_roughness_texture() && mat.metallic_roughness_tex > 0 && mat.metallic_roughness_tex <= mesh.texture_images.size()) {
                    const auto& img = mesh.texture_images[mat.metallic_roughness_tex - 1];
                    if (!img.pixels.empty()) {
                        auto tex = upload_texture(ctx, img, false);
                        if (!tex)
                            return std::unexpected(tex.error());
                        textures[i].metallic_roughness = std::move(*tex);
                        textures[i].has_metallic_roughness = true;
                    }
                }
            }
            return textures;
        }

        [[nodiscard]] std::unique_ptr<SplatData> build_splat_data(const std::vector<GaussianVertex>& data,
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

            means = means.to(Device::CUDA);
            scaling_raw = scaling_raw.to(Device::CUDA);
            rotation_raw = rotation_raw.to(Device::CUDA);
            opacity_raw = opacity_raw.to(Device::CUDA);
            sh0 = sh0.to(Device::CUDA);
            auto shN = Tensor::zeros({N, 0, 3}, Device::CUDA);

            return std::make_unique<SplatData>(0,
                                               std::move(means),
                                               std::move(sh0),
                                               std::move(shN),
                                               std::move(scaling_raw),
                                               std::move(rotation_raw),
                                               std::move(opacity_raw),
                                               scene_scale);
        }

        [[nodiscard]] std::expected<VkRenderPass, std::string> create_render_pass(VkDevice device) {
            VkAttachmentDescription color{};
            color.format = kFramebufferFormat;
            color.samples = VK_SAMPLE_COUNT_1_BIT;
            color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            color.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            color.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            color.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            color.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            color.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

            VkAttachmentReference ref{};
            ref.attachment = 0;
            ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

            VkSubpassDescription subpass{};
            subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
            subpass.colorAttachmentCount = 1;
            subpass.pColorAttachments = &ref;

            VkRenderPassCreateInfo info{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
            info.attachmentCount = 1;
            info.pAttachments = &color;
            info.subpassCount = 1;
            info.pSubpasses = &subpass;

            VkRenderPass render_pass = VK_NULL_HANDLE;
            const VkResult result = vkCreateRenderPass(device, &info, nullptr, &render_pass);
            if (result != VK_SUCCESS)
                return std::unexpected(vkError("vkCreateRenderPass", result));
            return render_pass;
        }

    } // namespace

    std::expected<std::unique_ptr<SplatData>, std::string>
    mesh_to_splat(const MeshData& mesh,
                  const Mesh2SplatOptions& options,
                  Mesh2SplatProgressCallback progress) {
        auto report = [&](float pct, const std::string& stage) -> bool {
            return progress ? progress(pct, stage) : true;
        };

        if (!mesh.vertices.is_valid() || mesh.vertex_count() == 0)
            return std::unexpected("Mesh has no vertices");
        if (!mesh.indices.is_valid() || mesh.face_count() == 0)
            return std::unexpected("Mesh has no faces");
        if (options.resolution_target < Mesh2SplatOptions::kMinResolution)
            return std::unexpected(std::format("Mesh2Splat resolution must be at least {}", Mesh2SplatOptions::kMinResolution));
        if (options.sigma <= 0.0f)
            return std::unexpected("Mesh2Splat sigma must be positive");

        if (!report(0.0f, "Preparing mesh data"))
            return std::unexpected("Cancelled");

        auto submesh_geometries = extract_geometry(mesh);
        if (submesh_geometries.empty())
            return std::unexpected("No geometry extracted");

        glm::vec3 global_min(std::numeric_limits<float>::max());
        glm::vec3 global_max(std::numeric_limits<float>::lowest());
        size_t total_vertices = 0;
        for (const auto& geo : submesh_geometries) {
            global_min = glm::min(global_min, geo.bbox_min);
            global_max = glm::max(global_max, geo.bbox_max);
            total_vertices += geo.vertices.size();
        }

        const float scene_scale = glm::length(global_max - global_min) * 0.5f;
        if (scene_scale <= 0.0f)
            return std::unexpected("Degenerate mesh: zero bounding box extent");

        const int res = options.resolution_target;
        const auto triangle_count = static_cast<uint64_t>(total_vertices / 3);
        const uint64_t pixel_based = static_cast<uint64_t>(res) * static_cast<uint64_t>(res) * 6ull;
        const uint64_t triangle_based = triangle_count * 2ull;
        const uint64_t ssbo_entries64 = std::max(pixel_based, triangle_based);
        if (ssbo_entries64 > std::numeric_limits<uint32_t>::max())
            return std::unexpected("Mesh2Splat output capacity exceeds Vulkan uint32 counter range");
        const uint32_t ssbo_entries = static_cast<uint32_t>(ssbo_entries64);
        const VkDeviceSize ssbo_size = static_cast<VkDeviceSize>(ssbo_entries64 * sizeof(GaussianVertex));

        LOG_INFO("mesh2splat: Vulkan converter, {} submeshes, {} triangles, resolution={}",
                 submesh_geometries.size(), triangle_count, res);

        if (!report(0.15f, "Initializing Vulkan converter"))
            return std::unexpected("Cancelled");

        auto& ctx = vulkan_context();
        std::string error;
        if (!ctx.ensure(error))
            return std::unexpected(error);
        const VkDevice device = ctx.device();

        if (!report(0.2f, "Compiling Vulkan shaders"))
            return std::unexpected("Cancelled");

        auto vs = create_shader_module(device, kVertexShader, EShLangVertex, "vertex");
        if (!vs)
            return std::unexpected(vs.error());
        auto gs = create_shader_module(device, kGeometryShader, EShLangGeometry, "geometry");
        if (!gs)
            return std::unexpected(gs.error());
        auto fs = create_shader_module(device, kFragmentShader, EShLangFragment, "fragment");
        if (!fs)
            return std::unexpected(fs.error());

        auto output_buffer = ctx.createBuffer(ssbo_size,
                                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (!output_buffer)
            return std::unexpected(output_buffer.error());
        auto counter_buffer = ctx.createBuffer(sizeof(uint32_t),
                                               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (!counter_buffer)
            return std::unexpected(counter_buffer.error());
        auto counter_readback = ctx.createBuffer(sizeof(uint32_t),
                                                 VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (!counter_readback)
            return std::unexpected(counter_readback.error());

        auto framebuffer_image = ctx.createImage(static_cast<uint32_t>(res),
                                                 static_cast<uint32_t>(res),
                                                 kFramebufferFormat,
                                                 VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
        if (!framebuffer_image)
            return std::unexpected(framebuffer_image.error());

        auto render_pass_expected = create_render_pass(device);
        if (!render_pass_expected)
            return std::unexpected(render_pass_expected.error());
        VkRenderPass render_pass = *render_pass_expected;

        VkFramebuffer framebuffer = VK_NULL_HANDLE;
        VkFramebufferCreateInfo fb_info{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        fb_info.renderPass = render_pass;
        fb_info.attachmentCount = 1;
        fb_info.pAttachments = &framebuffer_image->view;
        fb_info.width = static_cast<uint32_t>(res);
        fb_info.height = static_cast<uint32_t>(res);
        fb_info.layers = 1;
        VkResult result = vkCreateFramebuffer(device, &fb_info, nullptr, &framebuffer);
        if (result != VK_SUCCESS) {
            vkDestroyRenderPass(device, render_pass, nullptr);
            return std::unexpected(vkError("vkCreateFramebuffer", result));
        }

        VkDescriptorSetLayoutBinding bindings[5]{};
        bindings[0] = {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
        bindings[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
        bindings[2] = {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
        bindings[3] = {3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
        bindings[4] = {4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo dsl_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        dsl_info.bindingCount = static_cast<uint32_t>(std::size(bindings));
        dsl_info.pBindings = bindings;
        VkDescriptorSetLayout descriptor_layout = VK_NULL_HANDLE;
        result = vkCreateDescriptorSetLayout(device, &dsl_info, nullptr, &descriptor_layout);
        if (result != VK_SUCCESS) {
            vkDestroyFramebuffer(device, framebuffer, nullptr);
            vkDestroyRenderPass(device, render_pass, nullptr);
            return std::unexpected(vkError("vkCreateDescriptorSetLayout", result));
        }

        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_GEOMETRY_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        push_range.offset = 0;
        push_range.size = sizeof(PushConstants);
        VkPipelineLayoutCreateInfo pipeline_layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pipeline_layout_info.setLayoutCount = 1;
        pipeline_layout_info.pSetLayouts = &descriptor_layout;
        pipeline_layout_info.pushConstantRangeCount = 1;
        pipeline_layout_info.pPushConstantRanges = &push_range;
        VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
        result = vkCreatePipelineLayout(device, &pipeline_layout_info, nullptr, &pipeline_layout);
        if (result != VK_SUCCESS) {
            vkDestroyDescriptorSetLayout(device, descriptor_layout, nullptr);
            vkDestroyFramebuffer(device, framebuffer, nullptr);
            vkDestroyRenderPass(device, render_pass, nullptr);
            return std::unexpected(vkError("vkCreatePipelineLayout", result));
        }

        VkPipelineShaderStageCreateInfo stages[3]{};
        stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, vs->module, "main", nullptr};
        stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_GEOMETRY_BIT, gs->module, "main", nullptr};
        stages[2] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, fs->module, "main", nullptr};

        VkVertexInputBindingDescription vertex_binding{};
        vertex_binding.binding = 0;
        vertex_binding.stride = sizeof(PerVertexData);
        vertex_binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        std::array<VkVertexInputAttributeDescription, 7> attrs{};
        attrs[0] = {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(PerVertexData, position)};
        attrs[1] = {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(PerVertexData, normal)};
        attrs[2] = {2, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(PerVertexData, tangent)};
        attrs[3] = {3, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(PerVertexData, uv)};
        attrs[4] = {4, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(PerVertexData, normalized_uv)};
        attrs[5] = {5, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(PerVertexData, scale)};
        attrs[6] = {6, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(PerVertexData, color)};
        VkPipelineVertexInputStateCreateInfo vertex_input{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        vertex_input.vertexBindingDescriptionCount = 1;
        vertex_input.pVertexBindingDescriptions = &vertex_binding;
        vertex_input.vertexAttributeDescriptionCount = static_cast<uint32_t>(attrs.size());
        vertex_input.pVertexAttributeDescriptions = attrs.data();

        VkPipelineInputAssemblyStateCreateInfo input_assembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkViewport viewport{};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = static_cast<float>(res);
        viewport.height = static_cast<float>(res);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        VkRect2D scissor{{0, 0}, {static_cast<uint32_t>(res), static_cast<uint32_t>(res)}};
        VkPipelineViewportStateCreateInfo viewport_state{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        viewport_state.viewportCount = 1;
        viewport_state.pViewports = &viewport;
        viewport_state.scissorCount = 1;
        viewport_state.pScissors = &scissor;

        VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        raster.polygonMode = VK_POLYGON_MODE_FILL;
        raster.cullMode = VK_CULL_MODE_NONE;
        raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        raster.lineWidth = 1.0f;

        VkPipelineMultisampleStateCreateInfo msaa{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        msaa.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineColorBlendAttachmentState blend_attachment{};
        blend_attachment.colorWriteMask = 0;
        VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        blend.attachmentCount = 1;
        blend.pAttachments = &blend_attachment;

        VkGraphicsPipelineCreateInfo pipeline_info{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        pipeline_info.stageCount = static_cast<uint32_t>(std::size(stages));
        pipeline_info.pStages = stages;
        pipeline_info.pVertexInputState = &vertex_input;
        pipeline_info.pInputAssemblyState = &input_assembly;
        pipeline_info.pViewportState = &viewport_state;
        pipeline_info.pRasterizationState = &raster;
        pipeline_info.pMultisampleState = &msaa;
        pipeline_info.pColorBlendState = &blend;
        pipeline_info.layout = pipeline_layout;
        pipeline_info.renderPass = render_pass;
        VkPipeline pipeline = VK_NULL_HANDLE;
        result = vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline);
        if (result != VK_SUCCESS) {
            vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
            vkDestroyDescriptorSetLayout(device, descriptor_layout, nullptr);
            vkDestroyFramebuffer(device, framebuffer, nullptr);
            vkDestroyRenderPass(device, render_pass, nullptr);
            return std::unexpected(vkError("vkCreateGraphicsPipelines", result));
        }

        VkSamplerCreateInfo sampler_info{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        sampler_info.magFilter = VK_FILTER_LINEAR;
        sampler_info.minFilter = VK_FILTER_LINEAR;
        sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sampler_info.maxLod = 1.0f;
        VkSampler sampler = VK_NULL_HANDLE;
        result = vkCreateSampler(device, &sampler_info, nullptr, &sampler);
        if (result != VK_SUCCESS) {
            vkDestroyPipeline(device, pipeline, nullptr);
            vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
            vkDestroyDescriptorSetLayout(device, descriptor_layout, nullptr);
            vkDestroyFramebuffer(device, framebuffer, nullptr);
            vkDestroyRenderPass(device, render_pass, nullptr);
            return std::unexpected(vkError("vkCreateSampler", result));
        }

        auto dummy_texture = create_dummy_texture(ctx);
        if (!dummy_texture)
            return std::unexpected(dummy_texture.error());
        auto material_textures = upload_material_textures(ctx, mesh, *dummy_texture);
        if (!material_textures)
            return std::unexpected(material_textures.error());

        std::vector<Buffer> vertex_buffers;
        vertex_buffers.reserve(submesh_geometries.size());
        for (const auto& geo : submesh_geometries) {
            const VkDeviceSize bytes = static_cast<VkDeviceSize>(geo.vertices.size() * sizeof(PerVertexData));
            auto vb = ctx.createBuffer(bytes,
                                       VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            if (!vb)
                return std::unexpected(vb.error());
            if (auto upload_error = ctx.uploadBuffer(geo.vertices.data(), bytes, *vb))
                return std::unexpected(*upload_error);
            vertex_buffers.push_back(std::move(*vb));
        }

        VkDescriptorPoolSize pool_sizes[2]{};
        pool_sizes[0] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, static_cast<uint32_t>(submesh_geometries.size() * 2)};
        pool_sizes[1] = {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, static_cast<uint32_t>(submesh_geometries.size() * 3)};
        VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pool_info.maxSets = static_cast<uint32_t>(submesh_geometries.size());
        pool_info.poolSizeCount = 2;
        pool_info.pPoolSizes = pool_sizes;
        VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
        result = vkCreateDescriptorPool(device, &pool_info, nullptr, &descriptor_pool);
        if (result != VK_SUCCESS)
            return std::unexpected(vkError("vkCreateDescriptorPool", result));

        std::vector<VkDescriptorSetLayout> layouts(submesh_geometries.size(), descriptor_layout);
        std::vector<VkDescriptorSet> descriptor_sets(submesh_geometries.size(), VK_NULL_HANDLE);
        VkDescriptorSetAllocateInfo alloc_sets{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        alloc_sets.descriptorPool = descriptor_pool;
        alloc_sets.descriptorSetCount = static_cast<uint32_t>(descriptor_sets.size());
        alloc_sets.pSetLayouts = layouts.data();
        result = vkAllocateDescriptorSets(device, &alloc_sets, descriptor_sets.data());
        if (result != VK_SUCCESS)
            return std::unexpected(vkError("vkAllocateDescriptorSets", result));

        VkDescriptorBufferInfo output_info{output_buffer->buffer, 0, ssbo_size};
        VkDescriptorBufferInfo counter_info{counter_buffer->buffer, 0, sizeof(uint32_t)};
        for (size_t i = 0; i < submesh_geometries.size(); ++i) {
            const size_t mat_idx = submesh_geometries[i].material_index;
            const MaterialTextures* mt = mat_idx < material_textures->size() ? &(*material_textures)[mat_idx] : nullptr;
            const VkImageView albedo_view = mt && mt->has_albedo ? mt->albedo.view : dummy_texture->view;
            const VkImageView normal_view = mt && mt->has_normal ? mt->normal.view : dummy_texture->view;
            const VkImageView mr_view = mt && mt->has_metallic_roughness ? mt->metallic_roughness.view : dummy_texture->view;
            VkDescriptorImageInfo image_infos[3] = {
                {sampler, albedo_view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
                {sampler, normal_view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
                {sampler, mr_view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
            };
            VkWriteDescriptorSet writes[5]{};
            writes[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptor_sets[i], 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &output_info, nullptr};
            writes[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptor_sets[i], 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &counter_info, nullptr};
            writes[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptor_sets[i], 2, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &image_infos[0], nullptr, nullptr};
            writes[3] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptor_sets[i], 3, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &image_infos[1], nullptr, nullptr};
            writes[4] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptor_sets[i], 4, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &image_infos[2], nullptr, nullptr};
            vkUpdateDescriptorSets(device, static_cast<uint32_t>(std::size(writes)), writes, 0, nullptr);
        }

        if (!report(0.3f, "Converting mesh to splats"))
            return std::unexpected("Cancelled");

        auto cmd = ctx.beginCommands();
        if (!cmd)
            return std::unexpected(cmd.error());
        vkCmdFillBuffer(*cmd, counter_buffer->buffer, 0, sizeof(uint32_t), 0);
        VkBufferMemoryBarrier counter_reset_barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        counter_reset_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        counter_reset_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        counter_reset_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        counter_reset_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        counter_reset_barrier.buffer = counter_buffer->buffer;
        counter_reset_barrier.size = sizeof(uint32_t);
        vkCmdPipelineBarrier(*cmd,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                             0,
                             0,
                             nullptr,
                             1,
                             &counter_reset_barrier,
                             0,
                             nullptr);

        VkClearValue clear{};
        VkRenderPassBeginInfo rp_begin{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        rp_begin.renderPass = render_pass;
        rp_begin.framebuffer = framebuffer;
        rp_begin.renderArea.extent = {static_cast<uint32_t>(res), static_cast<uint32_t>(res)};
        rp_begin.clearValueCount = 1;
        rp_begin.pClearValues = &clear;
        vkCmdBeginRenderPass(*cmd, &rp_begin, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(*cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

        for (size_t i = 0; i < submesh_geometries.size(); ++i) {
            const auto& geo = submesh_geometries[i];
            PushConstants pc{};
            pc.material_factor = glm::vec4(1.0f);
            pc.bbox_min_metallic = glm::vec4(global_min, 0.0f);
            pc.bbox_max_roughness = glm::vec4(global_max, 1.0f);
            pc.flags.w = mesh.has_colors() ? 1 : 0;
            if (geo.material_index < mesh.materials.size()) {
                const auto& mat = mesh.materials[geo.material_index];
                pc.material_factor = mat.base_color;
                pc.bbox_min_metallic.w = mat.metallic;
                pc.bbox_max_roughness.w = mat.roughness;
                if (geo.material_index < material_textures->size()) {
                    const auto& mt = (*material_textures)[geo.material_index];
                    pc.flags.x = mt.has_albedo ? 1 : 0;
                    pc.flags.y = mt.has_normal ? 1 : 0;
                    pc.flags.z = mt.has_metallic_roughness ? 1 : 0;
                }
            }
            pc.limits.x = ssbo_entries;

            VkDeviceSize offset = 0;
            vkCmdBindVertexBuffers(*cmd, 0, 1, &vertex_buffers[i].buffer, &offset);
            vkCmdBindDescriptorSets(*cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 0, 1, &descriptor_sets[i], 0, nullptr);
            vkCmdPushConstants(*cmd,
                               pipeline_layout,
                               VK_SHADER_STAGE_GEOMETRY_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                               0,
                               sizeof(PushConstants),
                               &pc);
            vkCmdDraw(*cmd, static_cast<uint32_t>(geo.vertices.size()), 1, 0, 0);
        }
        vkCmdEndRenderPass(*cmd);

        VkBufferMemoryBarrier shader_to_copy[2]{};
        shader_to_copy[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        shader_to_copy[0].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        shader_to_copy[0].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        shader_to_copy[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        shader_to_copy[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        shader_to_copy[0].buffer = counter_buffer->buffer;
        shader_to_copy[0].size = sizeof(uint32_t);
        shader_to_copy[1] = shader_to_copy[0];
        shader_to_copy[1].buffer = output_buffer->buffer;
        shader_to_copy[1].size = ssbo_size;
        vkCmdPipelineBarrier(*cmd,
                             VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0,
                             0,
                             nullptr,
                             2,
                             shader_to_copy,
                             0,
                             nullptr);
        VkBufferCopy counter_copy{0, 0, sizeof(uint32_t)};
        vkCmdCopyBuffer(*cmd, counter_buffer->buffer, counter_readback->buffer, 1, &counter_copy);
        if (auto submit_error = ctx.submitAndWait(*cmd))
            return std::unexpected(*submit_error);

        uint32_t num_gaussians = 0;
        void* mapped_counter = nullptr;
        result = vkMapMemory(device, counter_readback->memory, 0, sizeof(uint32_t), 0, &mapped_counter);
        if (result != VK_SUCCESS)
            return std::unexpected(vkError("vkMapMemory(counter)", result));
        std::memcpy(&num_gaussians, mapped_counter, sizeof(uint32_t));
        vkUnmapMemory(device, counter_readback->memory);

        if (num_gaussians == 0)
            return std::unexpected("Conversion produced zero gaussians");
        if (num_gaussians > ssbo_entries) {
            LOG_WARN("mesh2splat: atomic counter ({}) exceeds SSBO capacity ({}), clamping", num_gaussians, ssbo_entries);
            num_gaussians = ssbo_entries;
        }

        if (!report(0.85f, "Reading back data"))
            return std::unexpected("Cancelled");

        const VkDeviceSize readback_size = static_cast<VkDeviceSize>(num_gaussians) * sizeof(GaussianVertex);
        auto output_readback = ctx.createBuffer(readback_size,
                                                VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (!output_readback)
            return std::unexpected(output_readback.error());
        auto read_cmd = ctx.beginCommands();
        if (!read_cmd)
            return std::unexpected(read_cmd.error());
        VkBufferCopy output_copy{0, 0, readback_size};
        vkCmdCopyBuffer(*read_cmd, output_buffer->buffer, output_readback->buffer, 1, &output_copy);
        if (auto submit_error = ctx.submitAndWait(*read_cmd))
            return std::unexpected(*submit_error);

        std::vector<GaussianVertex> gpu_data(num_gaussians);
        void* mapped_output = nullptr;
        result = vkMapMemory(device, output_readback->memory, 0, readback_size, 0, &mapped_output);
        if (result != VK_SUCCESS)
            return std::unexpected(vkError("vkMapMemory(output)", result));
        std::memcpy(gpu_data.data(), mapped_output, static_cast<size_t>(readback_size));
        vkUnmapMemory(device, output_readback->memory);

        if (!report(0.9f, "Building SplatData"))
            return std::unexpected("Cancelled");

        LOG_INFO("mesh2splat: produced {} gaussians (resolution={})", num_gaussians, options.resolution_target);
        auto splat = build_splat_data(gpu_data, options.sigma / static_cast<float>(res), scene_scale);

        vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
        vkDestroySampler(device, sampler, nullptr);
        vkDestroyPipeline(device, pipeline, nullptr);
        vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
        vkDestroyDescriptorSetLayout(device, descriptor_layout, nullptr);
        vkDestroyFramebuffer(device, framebuffer, nullptr);
        vkDestroyRenderPass(device, render_pass, nullptr);

        if (!report(1.0f, "Complete"))
            return std::unexpected("Cancelled");
        return splat;
    }

} // namespace lfs::rendering

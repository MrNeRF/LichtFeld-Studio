/* Derived from Mesh2Splat by Electronic Arts Inc.
 * Original: Copyright (c) 2025 Electronic Arts Inc. All rights reserved.
 * Licensed under BSD 3-Clause (see THIRD_PARTY_LICENSES.md)
 *
 * Modifications: Copyright (c) 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "rendering/mesh2splat.hpp"
#include "core/executable_path.hpp"
#include "core/logger.hpp"
#include "core/mesh_data.hpp"
#include "core/tensor.hpp"

// clang-format off
#include <glad/glad.h>
// clang-format on

#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

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

        // Mirrors the GaussianVertex struct in converterFS.glsl SSBO layout
        struct GaussianVertex {
            glm::vec4 position;
            glm::vec4 color;
            glm::vec4 scale;
            glm::vec4 normal;
            glm::vec4 rotation;
            glm::vec4 pbr;
        };
        static_assert(sizeof(GaussianVertex) == 6 * sizeof(glm::vec4));

        // VBO layout matching the conversion vertex shader (17 floats per vertex)
        struct PerVertexData {
            glm::vec3 position;      // location 0
            glm::vec3 normal;        // location 1
            glm::vec4 tangent;       // location 2
            glm::vec2 uv;            // location 3
            glm::vec2 normalized_uv; // location 4
            glm::vec3 scale;         // location 5
            glm::vec4 color;         // location 6
        };
        static_assert(sizeof(PerVertexData) == 21 * sizeof(float));

        struct SubmeshGeometry {
            std::vector<PerVertexData> vertices;
            glm::vec3 bbox_min{std::numeric_limits<float>::max()};
            glm::vec3 bbox_max{std::numeric_limits<float>::lowest()};
            size_t material_index = 0;
        };

        // RAII wrapper for a set of GL objects that need cleanup
        struct GlCleanup {
            GLuint program = 0;
            GLuint fbo = 0;
            GLuint rbo = 0;
            GLuint ssbo = 0;
            GLuint atomic_counter = 0;
            std::vector<GLuint> textures;

            ~GlCleanup() {
                for (auto tex : textures) {
                    if (tex)
                        glDeleteTextures(1, &tex);
                }
                if (atomic_counter)
                    glDeleteBuffers(1, &atomic_counter);
                if (ssbo)
                    glDeleteBuffers(1, &ssbo);
                if (rbo)
                    glDeleteRenderbuffers(1, &rbo);
                if (fbo)
                    glDeleteFramebuffers(1, &fbo);
                if (program)
                    glDeleteProgram(program);
            }

            GlCleanup() = default;
            GlCleanup(const GlCleanup&) = delete;
            GlCleanup& operator=(const GlCleanup&) = delete;
        };

        // Minimal GL state save/restore (cannot use lfs::rendering::GLStateGuard from lfs_core)
        struct GlStateSave {
            GLint viewport[4];
            GLint prev_program;
            GLint prev_fbo;
            GLint prev_vao;
            GLint prev_active_texture;
            GLboolean depth_test;
            GLboolean blend;
            GLboolean cull_face;

            GlStateSave() {
                glGetIntegerv(GL_VIEWPORT, viewport);
                glGetIntegerv(GL_CURRENT_PROGRAM, &prev_program);
                glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &prev_fbo);
                glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &prev_vao);
                glGetIntegerv(GL_ACTIVE_TEXTURE, &prev_active_texture);
                depth_test = glIsEnabled(GL_DEPTH_TEST);
                blend = glIsEnabled(GL_BLEND);
                cull_face = glIsEnabled(GL_CULL_FACE);
            }

            ~GlStateSave() {
                glViewport(viewport[0], viewport[1], viewport[2], viewport[3]);
                glUseProgram(prev_program);
                glBindFramebuffer(GL_FRAMEBUFFER, prev_fbo);
                glBindVertexArray(prev_vao);
                glActiveTexture(prev_active_texture);
                if (depth_test)
                    glEnable(GL_DEPTH_TEST);
                else
                    glDisable(GL_DEPTH_TEST);
                if (blend)
                    glEnable(GL_BLEND);
                else
                    glDisable(GL_BLEND);
                if (cull_face)
                    glEnable(GL_CULL_FACE);
                else
                    glDisable(GL_CULL_FACE);
            }

            GlStateSave(const GlStateSave&) = delete;
            GlStateSave& operator=(const GlStateSave&) = delete;
        };

        // ========================================================================
        // Geometry extraction
        // ========================================================================

        glm::vec3 compute_face_normal(const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2) {
            glm::vec3 edge1 = v1 - v0;
            glm::vec3 edge2 = v2 - v0;
            glm::vec3 n = glm::cross(edge1, edge2);
            float len = glm::length(n);
            return len > 1e-8f ? n / len : glm::vec3(0.0f, 1.0f, 0.0f);
        }

        glm::vec4 compute_face_tangent(const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2,
                                       const glm::vec2& uv0, const glm::vec2& uv1, const glm::vec2& uv2) {
            glm::vec3 dv1 = v1 - v0;
            glm::vec3 dv2 = v2 - v0;
            glm::vec2 duv1 = uv1 - uv0;
            glm::vec2 duv2 = uv2 - uv0;

            float det = duv1.x * duv2.y - duv1.y * duv2.x;
            if (std::abs(det) < 1e-8f) {
                return glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);
            }

            float r = 1.0f / det;
            glm::vec3 t = glm::normalize((dv1 * duv2.y - dv2 * duv1.y) * r);
            return glm::vec4(t, 1.0f);
        }

        std::vector<SubmeshGeometry> extract_geometry(const MeshData& mesh) {
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

            // Build submesh list; treat whole mesh as one if no submeshes defined
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

                for (size_t f = 0; f < face_count; f++) {
                    const size_t base = sub.start_index + f * 3;
                    int32_t i0 = idx_ptr[base + 0];
                    int32_t i1 = idx_ptr[base + 1];
                    int32_t i2 = idx_ptr[base + 2];
                    assert(i0 >= 0 && i0 < V);
                    assert(i1 >= 0 && i1 < V);
                    assert(i2 >= 0 && i2 < V);

                    const int32_t indices[3] = {i0, i1, i2};

                    glm::vec3 pos[3];
                    glm::vec3 nrm[3];
                    glm::vec4 tan[3];
                    glm::vec2 uv[3];
                    glm::vec4 col[3] = {glm::vec4(1.0f), glm::vec4(1.0f), glm::vec4(1.0f)};

                    for (int k = 0; k < 3; k++) {
                        int32_t vi = indices[k];
                        pos[k] = {verts_ptr[vi * 3], verts_ptr[vi * 3 + 1], verts_ptr[vi * 3 + 2]};

                        geo.bbox_min = glm::min(geo.bbox_min, pos[k]);
                        geo.bbox_max = glm::max(geo.bbox_max, pos[k]);

                        if (texcoords_ptr) {
                            uv[k] = {texcoords_ptr[vi * 2], texcoords_ptr[vi * 2 + 1]};
                        } else {
                            uv[k] = {0.0f, 0.0f};
                        }

                        if (normals_ptr) {
                            nrm[k] = {normals_ptr[vi * 3], normals_ptr[vi * 3 + 1], normals_ptr[vi * 3 + 2]};
                        }

                        if (tangents_ptr) {
                            tan[k] = {tangents_ptr[vi * 4], tangents_ptr[vi * 4 + 1],
                                      tangents_ptr[vi * 4 + 2], tangents_ptr[vi * 4 + 3]};
                        }

                        if (colors_ptr) {
                            col[k] = {colors_ptr[vi * 4], colors_ptr[vi * 4 + 1],
                                      colors_ptr[vi * 4 + 2], colors_ptr[vi * 4 + 3]};
                        }
                    }

                    // Compute face normal if mesh has no normals
                    if (!normals_ptr) {
                        glm::vec3 fn = compute_face_normal(pos[0], pos[1], pos[2]);
                        nrm[0] = nrm[1] = nrm[2] = fn;
                    }

                    // Compute face tangent if mesh has no tangents
                    if (!tangents_ptr) {
                        glm::vec4 ft = compute_face_tangent(pos[0], pos[1], pos[2], uv[0], uv[1], uv[2]);
                        tan[0] = tan[1] = tan[2] = ft;
                    }

                    for (int k = 0; k < 3; k++) {
                        PerVertexData vtx;
                        vtx.position = pos[k];
                        vtx.normal = nrm[k];
                        vtx.tangent = tan[k];
                        vtx.uv = uv[k];
                        vtx.normalized_uv = {0.0f, 0.0f}; // GS ignores this; uses triplanar projection
                        vtx.scale = {0.0f, 0.0f, 0.0f};   // unused by GS
                        vtx.color = col[k];
                        geo.vertices.push_back(vtx);
                    }
                }

                if (!geo.vertices.empty()) {
                    result.push_back(std::move(geo));
                }
            }

            return result;
        }

        // ========================================================================
        // Shader compilation
        // ========================================================================

        std::string read_file_contents(const std::filesystem::path& path) {
            std::ifstream file(path);
            if (!file.is_open()) {
                return {};
            }
            std::stringstream ss;
            ss << file.rdbuf();
            return ss.str();
        }

        GLuint compile_shader_stage(const std::string& source, GLenum type, const char* label) {
            GLuint shader = glCreateShader(type);
            assert(shader != 0);

            const char* src = source.c_str();
            glShaderSource(shader, 1, &src, nullptr);
            glCompileShader(shader);

            GLint status;
            glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
            if (status != GL_TRUE) {
                char log[2048];
                GLsizei len;
                glGetShaderInfoLog(shader, sizeof(log), &len, log);
                LOG_ERROR("mesh2splat {} shader compile error: {}", label, log);
                glDeleteShader(shader);
                return 0;
            }

            return shader;
        }

        GLuint create_conversion_program() {
            auto shader_dir = core::getShadersDir() / "mesh2splat";

            auto vs_source = read_file_contents(shader_dir / "converterVS.glsl");
            auto gs_source = read_file_contents(shader_dir / "converterGS.glsl");
            auto fs_source = read_file_contents(shader_dir / "converterFS.glsl");

            if (vs_source.empty() || gs_source.empty() || fs_source.empty()) {
                LOG_ERROR("mesh2splat: failed to read shader files from {}", shader_dir.string());
                return 0;
            }

            GLuint vs = compile_shader_stage(vs_source, GL_VERTEX_SHADER, "vertex");
            if (!vs)
                return 0;

            GLuint gs = compile_shader_stage(gs_source, GL_GEOMETRY_SHADER, "geometry");
            if (!gs) {
                glDeleteShader(vs);
                return 0;
            }

            GLuint fs = compile_shader_stage(fs_source, GL_FRAGMENT_SHADER, "fragment");
            if (!fs) {
                glDeleteShader(vs);
                glDeleteShader(gs);
                return 0;
            }

            GLuint program = glCreateProgram();
            glAttachShader(program, vs);
            glAttachShader(program, gs);
            glAttachShader(program, fs);
            glLinkProgram(program);

            // Shaders can be deleted after linking
            glDeleteShader(vs);
            glDeleteShader(gs);
            glDeleteShader(fs);

            GLint status;
            glGetProgramiv(program, GL_LINK_STATUS, &status);
            if (status != GL_TRUE) {
                char log[2048];
                GLsizei len;
                glGetProgramInfoLog(program, sizeof(log), &len, log);
                LOG_ERROR("mesh2splat program link error: {}", log);
                glDeleteProgram(program);
                return 0;
            }

            return program;
        }

        // ========================================================================
        // Texture upload
        // ========================================================================

        GLuint upload_texture(const TextureImage& img) {
            assert(img.width > 0 && img.height > 0);
            assert(!img.pixels.empty());

            GLenum format;
            GLint internal_format;
            switch (img.channels) {
            case 1:
                format = GL_RED;
                internal_format = GL_R8;
                break;
            case 2:
                format = GL_RG;
                internal_format = GL_RG8;
                break;
            case 3:
                format = GL_RGB;
                internal_format = GL_RGB8;
                break;
            case 4:
                format = GL_RGBA;
                internal_format = GL_RGBA8;
                break;
            default: return 0;
            }

            GLuint tex;
            glGenTextures(1, &tex);
            glBindTexture(GL_TEXTURE_2D, tex);
            glTexImage2D(GL_TEXTURE_2D, 0, internal_format,
                         img.width, img.height, 0, format, GL_UNSIGNED_BYTE, img.pixels.data());
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
            glBindTexture(GL_TEXTURE_2D, 0);

            return tex;
        }

        // ========================================================================
        // FBO creation
        // ========================================================================

        bool create_fbo(int width, int height, GLuint& out_fbo, GLuint& out_rbo) {
            glGenFramebuffers(1, &out_fbo);
            glBindFramebuffer(GL_FRAMEBUFFER, out_fbo);

            glGenRenderbuffers(1, &out_rbo);
            glBindRenderbuffer(GL_RENDERBUFFER, out_rbo);
            glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA32F, width, height);
            glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, out_rbo);

            GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
            glBindFramebuffer(GL_FRAMEBUFFER, 0);

            return status == GL_FRAMEBUFFER_COMPLETE;
        }

        // ========================================================================
        // SSBO readback → SplatData conversion
        // ========================================================================

        std::unique_ptr<SplatData> build_splat_data(const std::vector<GaussianVertex>& data,
                                                    float scale_multiplier,
                                                    float scene_scale) {
            const auto N = static_cast<size_t>(data.size());
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

            // unsigmoid(0.999) — fully opaque but avoids infinity
            const float opacity_logit = -std::log(1.0f / 0.999f - 1.0f);

            for (size_t i = 0; i < N; i++) {
                const auto& g = data[i];

                // Position
                m_ptr[i * 3 + 0] = g.position.x;
                m_ptr[i * 3 + 1] = g.position.y;
                m_ptr[i * 3 + 2] = g.position.z;

                // Scale: log(linear_scale * sigma / resolution_target)
                glm::vec3 ls(g.scale.x, g.scale.y, g.scale.z);
                ls *= scale_multiplier;
                ls = glm::max(ls, glm::vec3(1e-8f));
                s_ptr[i * 3 + 0] = std::log(ls.x);
                s_ptr[i * 3 + 1] = std::log(ls.y);
                s_ptr[i * 3 + 2] = std::log(ls.z);

                // Rotation: SSBO stores (w, x, y, z), SplatData stores (w, x, y, z) at [0,1,2,3]
                r_ptr[i * 4 + 0] = g.rotation.x;
                r_ptr[i * 4 + 1] = g.rotation.y;
                r_ptr[i * 4 + 2] = g.rotation.z;
                r_ptr[i * 4 + 3] = g.rotation.w;

                // Opacity (fully opaque)
                o_ptr[i] = opacity_logit;

                // SH DC: (color_linear - 0.5) / SH_C0
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

            return std::make_unique<SplatData>(
                0, // sh_degree
                std::move(means),
                std::move(sh0),
                std::move(shN),
                std::move(scaling_raw),
                std::move(rotation_raw),
                std::move(opacity_raw),
                scene_scale);
        }

        // ========================================================================
        // Bind material textures for a submesh
        // ========================================================================

        void bind_submesh_textures(GLuint program, const MeshData& mesh, size_t material_index,
                                   GlCleanup& cleanup) {
            GLint has_albedo_loc = glGetUniformLocation(program, "hasAlbedoMap");
            GLint has_normal_loc = glGetUniformLocation(program, "hasNormalMap");
            GLint has_mr_loc = glGetUniformLocation(program, "hasMetallicRoughnessMap");

            if (has_albedo_loc >= 0)
                glUniform1i(has_albedo_loc, 0);
            if (has_normal_loc >= 0)
                glUniform1i(has_normal_loc, 0);
            if (has_mr_loc >= 0)
                glUniform1i(has_mr_loc, 0);

            if (material_index >= mesh.materials.size()) {
                LOG_DEBUG("mesh2splat: no material at index {}", material_index);
                return;
            }
            const auto& mat = mesh.materials[material_index];
            LOG_DEBUG("mesh2splat: material '{}' base_color=({},{},{},{}), albedo_tex={}, albedo_path='{}'",
                      mat.name, mat.base_color.r, mat.base_color.g, mat.base_color.b, mat.base_color.a,
                      mat.albedo_tex, mat.albedo_tex_path);

            // Upload all textures before binding — upload_texture() binds/unbinds
            // on the active texture unit, which would clobber earlier bindings.
            GLuint albedo_gl = 0, normal_gl = 0, mr_gl = 0;

            if (mat.has_albedo_texture() && mat.albedo_tex > 0 &&
                mat.albedo_tex <= mesh.texture_images.size()) {
                const auto& img = mesh.texture_images[mat.albedo_tex - 1];
                if (!img.pixels.empty()) {
                    LOG_DEBUG("mesh2splat: uploading albedo texture {}x{} ({} ch, {} bytes)",
                              img.width, img.height, img.channels, img.pixels.size());
                    albedo_gl = upload_texture(img);
                    if (albedo_gl)
                        cleanup.textures.push_back(albedo_gl);
                }
            }

            if (mat.has_normal_texture() && mat.normal_tex > 0 &&
                mat.normal_tex <= mesh.texture_images.size()) {
                const auto& img = mesh.texture_images[mat.normal_tex - 1];
                if (!img.pixels.empty()) {
                    normal_gl = upload_texture(img);
                    if (normal_gl)
                        cleanup.textures.push_back(normal_gl);
                }
            }

            if (mat.has_metallic_roughness_texture() && mat.metallic_roughness_tex > 0 &&
                mat.metallic_roughness_tex <= mesh.texture_images.size()) {
                const auto& img = mesh.texture_images[mat.metallic_roughness_tex - 1];
                if (!img.pixels.empty()) {
                    mr_gl = upload_texture(img);
                    if (mr_gl)
                        cleanup.textures.push_back(mr_gl);
                }
            }

            // Bind to texture units after all uploads are complete
            if (albedo_gl) {
                glActiveTexture(GL_TEXTURE0);
                glBindTexture(GL_TEXTURE_2D, albedo_gl);
                GLint loc = glGetUniformLocation(program, "albedoTexture");
                if (loc >= 0)
                    glUniform1i(loc, 0);
                if (has_albedo_loc >= 0)
                    glUniform1i(has_albedo_loc, 1);
            }

            if (normal_gl) {
                glActiveTexture(GL_TEXTURE1);
                glBindTexture(GL_TEXTURE_2D, normal_gl);
                GLint loc = glGetUniformLocation(program, "normalTexture");
                if (loc >= 0)
                    glUniform1i(loc, 1);
                if (has_normal_loc >= 0)
                    glUniform1i(has_normal_loc, 1);
            }

            if (mr_gl) {
                glActiveTexture(GL_TEXTURE2);
                glBindTexture(GL_TEXTURE_2D, mr_gl);
                GLint loc = glGetUniformLocation(program, "metallicRoughnessTexture");
                if (loc >= 0)
                    glUniform1i(loc, 2);
                if (has_mr_loc >= 0)
                    glUniform1i(has_mr_loc, 1);
            }
        }

    } // anonymous namespace

    // ============================================================================
    // Public API
    // ============================================================================

    std::expected<std::unique_ptr<SplatData>, std::string>
    mesh_to_splat(const MeshData& mesh,
                  const Mesh2SplatOptions& options,
                  Mesh2SplatProgressCallback progress) {

        auto report = [&](float pct, const std::string& stage) -> bool {
            if (progress)
                return progress(pct, stage);
            return true;
        };

        // Validate inputs
        if (!mesh.vertices.is_valid() || mesh.vertex_count() == 0)
            return std::unexpected("Mesh has no vertices");
        if (!mesh.indices.is_valid() || mesh.face_count() == 0)
            return std::unexpected("Mesh has no faces");
        assert(options.resolution_target > 0);
        assert(options.sigma > 0.0f);

        // Phase 1: Extract per-face geometry from indexed mesh
        if (!report(0.0f, "Preparing mesh data"))
            return std::unexpected("Cancelled");

        auto submesh_geometries = extract_geometry(mesh);
        if (submesh_geometries.empty())
            return std::unexpected("No geometry extracted");

        // Compute global bounding box and total vertex count
        glm::vec3 global_min(std::numeric_limits<float>::max());
        glm::vec3 global_max(std::numeric_limits<float>::lowest());
        size_t total_vertices = 0;
        for (const auto& geo : submesh_geometries) {
            global_min = glm::min(global_min, geo.bbox_min);
            global_max = glm::max(global_max, geo.bbox_max);
            total_vertices += geo.vertices.size();
        }
        const float scene_scale = glm::length(global_max - global_min) * 0.5f;
        assert(scene_scale > 0.0f);

        const int res = options.resolution_target;

        LOG_INFO("mesh2splat: {} submeshes, {} triangles, resolution={}, bbox=[{:.2f},{:.2f},{:.2f}]-[{:.2f},{:.2f},{:.2f}]",
                 submesh_geometries.size(), total_vertices / 3, res,
                 global_min.x, global_min.y, global_min.z,
                 global_max.x, global_max.y, global_max.z);

        // Phase 3: GL pipeline — save state, compile shaders, run conversion
        if (!report(0.2f, "Compiling shaders"))
            return std::unexpected("Cancelled");

        GlStateSave state_save;
        GlCleanup cleanup;

        cleanup.program = create_conversion_program();
        if (!cleanup.program)
            return std::unexpected("Failed to compile mesh2splat shaders");

        // SSBO for gaussian output. Each rasterized fragment writes one GaussianVertex.
        // For sparse meshes (few large triangles), res^2 * 6 is sufficient.
        // For dense meshes (many small triangles), fragments scale with triangle count.
        const auto triangle_count = static_cast<GLsizeiptr>(total_vertices / 3);
        const GLsizeiptr pixel_based = static_cast<GLsizeiptr>(res) * res * 6;
        const GLsizeiptr triangle_based = triangle_count * 2;
        const GLsizeiptr ssbo_entries = std::max(pixel_based, triangle_based);
        const GLsizeiptr ssbo_size = ssbo_entries * static_cast<GLsizeiptr>(sizeof(GaussianVertex));

        glGenBuffers(1, &cleanup.ssbo);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, cleanup.ssbo);
        glBufferData(GL_SHADER_STORAGE_BUFFER, ssbo_size, nullptr, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

        // Create atomic counter
        glGenBuffers(1, &cleanup.atomic_counter);
        glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, cleanup.atomic_counter);
        GLuint zero = 0;
        glBufferData(GL_ATOMIC_COUNTER_BUFFER, sizeof(GLuint), &zero, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, 0);

        // Create FBO at full resolution
        if (!create_fbo(res, res, cleanup.fbo, cleanup.rbo)) {
            return std::unexpected("Failed to create framebuffer");
        }

        // Set up GL state for conversion
        glBindFramebuffer(GL_FRAMEBUFFER, cleanup.fbo);
        glViewport(0, 0, res, res);
        glDisable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glDisable(GL_CULL_FACE);

        glUseProgram(cleanup.program);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, cleanup.ssbo);
        glBindBufferBase(GL_ATOMIC_COUNTER_BUFFER, 1, cleanup.atomic_counter);

        // Phase 4: Process each submesh
        if (!report(0.3f, "Converting mesh to splats"))
            return std::unexpected("Cancelled");

        for (size_t si = 0; si < submesh_geometries.size(); si++) {
            const auto& geo = submesh_geometries[si];

            // Upload VBO
            GLuint vao, vbo;
            glGenVertexArrays(1, &vao);
            glGenBuffers(1, &vbo);

            glBindVertexArray(vao);
            glBindBuffer(GL_ARRAY_BUFFER, vbo);
            glBufferData(GL_ARRAY_BUFFER,
                         static_cast<GLsizeiptr>(geo.vertices.size() * sizeof(PerVertexData)),
                         geo.vertices.data(), GL_STATIC_DRAW);

            constexpr GLsizei stride = sizeof(PerVertexData);

            // location 0: position (vec3)
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride,
                                  reinterpret_cast<void*>(offsetof(PerVertexData, position)));
            glEnableVertexAttribArray(0);

            // location 1: normal (vec3)
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride,
                                  reinterpret_cast<void*>(offsetof(PerVertexData, normal)));
            glEnableVertexAttribArray(1);

            // location 2: tangent (vec4)
            glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, stride,
                                  reinterpret_cast<void*>(offsetof(PerVertexData, tangent)));
            glEnableVertexAttribArray(2);

            // location 3: uv (vec2)
            glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, stride,
                                  reinterpret_cast<void*>(offsetof(PerVertexData, uv)));
            glEnableVertexAttribArray(3);

            // location 4: normalizedUv (vec2)
            glVertexAttribPointer(4, 2, GL_FLOAT, GL_FALSE, stride,
                                  reinterpret_cast<void*>(offsetof(PerVertexData, normalized_uv)));
            glEnableVertexAttribArray(4);

            // location 5: scale (vec3)
            glVertexAttribPointer(5, 3, GL_FLOAT, GL_FALSE, stride,
                                  reinterpret_cast<void*>(offsetof(PerVertexData, scale)));
            glEnableVertexAttribArray(5);

            // location 6: vertex color (vec4)
            glVertexAttribPointer(6, 4, GL_FLOAT, GL_FALSE, stride,
                                  reinterpret_cast<void*>(offsetof(PerVertexData, color)));
            glEnableVertexAttribArray(6);

            // Set uniforms
            glm::vec4 material_factor(1.0f);
            float metallic_factor = 0.0f;
            float roughness_factor = 1.0f;
            if (geo.material_index < mesh.materials.size()) {
                const auto& mat = mesh.materials[geo.material_index];
                material_factor = mat.base_color;
                metallic_factor = mat.metallic;
                roughness_factor = mat.roughness;
            }

            GLint loc_material = glGetUniformLocation(cleanup.program, "u_materialFactor");
            GLint loc_metallic = glGetUniformLocation(cleanup.program, "u_metallicFactor");
            GLint loc_roughness = glGetUniformLocation(cleanup.program, "u_roughnessFactor");
            GLint loc_light_dir = glGetUniformLocation(cleanup.program, "u_lightDir");
            GLint loc_light_int = glGetUniformLocation(cleanup.program, "u_lightIntensity");
            GLint loc_ambient = glGetUniformLocation(cleanup.program, "u_ambient");
            GLint loc_bbox_min = glGetUniformLocation(cleanup.program, "u_bboxMin");
            GLint loc_bbox_max = glGetUniformLocation(cleanup.program, "u_bboxMax");
            GLint loc_has_vtx_colors = glGetUniformLocation(cleanup.program, "hasVertexColors");

            if (loc_material >= 0)
                glUniform4fv(loc_material, 1, glm::value_ptr(material_factor));
            if (loc_metallic >= 0)
                glUniform1f(loc_metallic, metallic_factor);
            if (loc_roughness >= 0)
                glUniform1f(loc_roughness, roughness_factor);
            if (loc_light_dir >= 0)
                glUniform3fv(loc_light_dir, 1, glm::value_ptr(options.light_dir));
            if (loc_light_int >= 0)
                glUniform1f(loc_light_int, options.light_intensity);
            if (loc_ambient >= 0)
                glUniform1f(loc_ambient, options.ambient);
            if (loc_has_vtx_colors >= 0)
                glUniform1i(loc_has_vtx_colors, mesh.has_colors() ? 1 : 0);
            // Use GLOBAL bbox so all submeshes share the same orthogonal UV space.
            // The GS maps positions to FBO pixels via triplanar projection using bbox.
            // Shared bbox = shared UV space = no duplicate gaussians across submeshes.
            if (loc_bbox_min >= 0)
                glUniform3fv(loc_bbox_min, 1, glm::value_ptr(global_min));
            if (loc_bbox_max >= 0)
                glUniform3fv(loc_bbox_max, 1, glm::value_ptr(global_max));

            // Bind textures for this submesh
            bind_submesh_textures(cleanup.program, mesh, geo.material_index, cleanup);

            LOG_DEBUG("mesh2splat: submesh[{}] material_factor=({},{},{},{}), vertices={}, "
                      "uniform_locs: material={}, bbox_min={}, bbox_max={}",
                      si, material_factor.x, material_factor.y, material_factor.z, material_factor.w,
                      geo.vertices.size(), loc_material, loc_bbox_min, loc_bbox_max);

            // Draw
            GLenum err_before = glGetError(); // clear any pre-existing errors
            (void)err_before;
            glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(geo.vertices.size()));
            GLenum err_after = glGetError();
            if (err_after != GL_NO_ERROR)
                LOG_ERROR("mesh2splat: GL error after draw: 0x{:X}", err_after);

            // Cleanup per-submesh resources
            glDeleteBuffers(1, &vbo);
            glDeleteVertexArrays(1, &vao);

            const float pct = 0.3f + 0.5f * static_cast<float>(si + 1) /
                                         static_cast<float>(submesh_geometries.size());
            if (!report(pct, "Converting submesh"))
                return std::unexpected("Cancelled");
        }

        // Phase 5: Read back results
        glFinish();

        glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, cleanup.atomic_counter);
        uint32_t num_gaussians = 0;
        glGetBufferSubData(GL_ATOMIC_COUNTER_BUFFER, 0, sizeof(uint32_t), &num_gaussians);
        glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, 0);

        if (num_gaussians == 0) {
            return std::unexpected("Conversion produced zero gaussians");
        }

        const uint32_t max_gaussians = static_cast<uint32_t>(ssbo_size / sizeof(GaussianVertex));
        if (num_gaussians > max_gaussians) {
            LOG_WARN("mesh2splat: atomic counter ({}) exceeds SSBO capacity ({}), clamping",
                     num_gaussians, max_gaussians);
            num_gaussians = max_gaussians;
        }

        LOG_INFO("mesh2splat: produced {} gaussians (resolution={})", num_gaussians, options.resolution_target);

        if (!report(0.85f, "Reading back data"))
            return std::unexpected("Cancelled");

        std::vector<GaussianVertex> gpu_data(num_gaussians);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, cleanup.ssbo);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                           static_cast<GLsizeiptr>(num_gaussians * sizeof(GaussianVertex)),
                           gpu_data.data());
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

        // Phase 6: Convert to SplatData
        if (!report(0.9f, "Building SplatData"))
            return std::unexpected("Cancelled");

        const float scale_multiplier = options.sigma / static_cast<float>(res);
        LOG_DEBUG("mesh2splat: scale_multiplier={:.6f} (sigma={}, res={})",
                  scale_multiplier, options.sigma, res);
        auto splat = build_splat_data(gpu_data, scale_multiplier, scene_scale);

        // Log SplatData statistics
        {
            auto sh0_cpu = splat->sh0_raw().to(Device::CPU);
            auto scale_cpu = splat->scaling_raw().to(Device::CPU);
            LOG_DEBUG("mesh2splat: SplatData sh0  min=({:.3f},{:.3f},{:.3f}) max=({:.3f},{:.3f},{:.3f})",
                      sh0_cpu.slice(2, 0, 1).min().item(), sh0_cpu.slice(2, 1, 2).min().item(),
                      sh0_cpu.slice(2, 2, 3).min().item(),
                      sh0_cpu.slice(2, 0, 1).max().item(), sh0_cpu.slice(2, 1, 2).max().item(),
                      sh0_cpu.slice(2, 2, 3).max().item());
            LOG_DEBUG("mesh2splat: SplatData scale(log) min=({:.3f},{:.3f},{:.3f}) max=({:.3f},{:.3f},{:.3f})",
                      scale_cpu.slice(1, 0, 1).min().item(), scale_cpu.slice(1, 1, 2).min().item(),
                      scale_cpu.slice(1, 2, 3).min().item(),
                      scale_cpu.slice(1, 0, 1).max().item(), scale_cpu.slice(1, 1, 2).max().item(),
                      scale_cpu.slice(1, 2, 3).max().item());
        }

        if (!report(1.0f, "Complete"))
            return std::unexpected("Cancelled");

        return splat;
    }

} // namespace lfs::rendering

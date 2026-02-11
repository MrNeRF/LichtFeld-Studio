/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "mesh_renderer.hpp"
#include "core/logger.hpp"
#include "core/mesh_data.hpp"
#include <algorithm>
#include <cassert>
#include <glad/glad.h>
#include <glm/gtc/matrix_inverse.hpp>

namespace lfs::rendering {

    namespace {
        Texture create_gl_texture(const lfs::core::TextureImage& img) {
            assert(!img.pixels.empty());
            assert(img.width > 0 && img.height > 0);

            GLuint tex;
            glGenTextures(1, &tex);
            glBindTexture(GL_TEXTURE_2D, tex);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, img.width, img.height, 0,
                         GL_RGBA, GL_UNSIGNED_BYTE, img.pixels.data());
            glGenerateMipmap(GL_TEXTURE_2D);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
            glBindTexture(GL_TEXTURE_2D, 0);
            return Texture(tex);
        }

        glm::mat4 compute_light_vp(const lfs::core::MeshData& mesh, const glm::mat4& model,
                                   const glm::vec3& light_dir) {
            auto cpu_verts = mesh.vertices.to(lfs::core::Device::CPU).contiguous();
            auto vacc = cpu_verts.accessor<float, 2>();
            const int64_t nv = mesh.vertex_count();

            glm::vec3 aabb_min(std::numeric_limits<float>::max());
            glm::vec3 aabb_max(std::numeric_limits<float>::lowest());
            for (int64_t i = 0; i < nv; ++i) {
                glm::vec3 p = glm::vec3(model * glm::vec4(vacc(i, 0), vacc(i, 1), vacc(i, 2), 1.0f));
                aabb_min = glm::min(aabb_min, p);
                aabb_max = glm::max(aabb_max, p);
            }

            const glm::vec3 center = (aabb_min + aabb_max) * 0.5f;
            const float radius = glm::length(aabb_max - aabb_min) * 0.5f;
            const glm::vec3 dir = glm::normalize(light_dir);
            const glm::vec3 eye = center + dir * radius * 2.0f;

            glm::vec3 up(0.0f, 1.0f, 0.0f);
            if (std::abs(glm::dot(dir, up)) > 0.99f)
                up = glm::vec3(0.0f, 0.0f, 1.0f);

            const glm::mat4 light_view = glm::lookAt(eye, center, up);
            const glm::mat4 light_proj = glm::ortho(-radius, radius, -radius, radius,
                                                    0.01f, radius * 4.0f);
            return light_proj * light_view;
        }
    } // namespace

    Result<void> MeshRenderer::initialize() {
        if (initialized_)
            return {};

        auto pbr_result = load_shader("mesh_pbr", "mesh_pbr.vert", "mesh_pbr.frag", false);
        if (!pbr_result) {
            LOG_ERROR("Failed to load PBR shader: {}", pbr_result.error().what());
            return std::unexpected(pbr_result.error().what());
        }
        pbr_shader_ = std::move(*pbr_result);

        auto wire_result = load_shader("mesh_wireframe", "mesh_wireframe.vert", "mesh_wireframe.frag", false);
        if (!wire_result) {
            LOG_ERROR("Failed to load wireframe shader: {}", wire_result.error().what());
            return std::unexpected(wire_result.error().what());
        }
        wireframe_shader_ = std::move(*wire_result);

        auto shadow_result = load_shader("shadow_depth", "shadow_depth.vert", "shadow_depth.frag", false);
        if (!shadow_result) {
            LOG_ERROR("Failed to load shadow shader: {}", shadow_result.error().what());
            return std::unexpected(shadow_result.error().what());
        }
        shadow_shader_ = std::move(*shadow_result);

        auto vao_result = create_vao();
        if (!vao_result)
            return std::unexpected(vao_result.error());
        vao_ = std::move(*vao_result);

        auto pos_result = create_vbo();
        auto norm_result = create_vbo();
        auto tang_result = create_vbo();
        auto tc_result = create_vbo();
        auto col_result = create_vbo();
        auto ebo_result = create_vbo();
        if (!pos_result || !norm_result || !tang_result || !tc_result || !col_result || !ebo_result)
            return std::unexpected("Failed to create VBOs");

        vbo_positions_ = std::move(*pos_result);
        vbo_normals_ = std::move(*norm_result);
        vbo_tangents_ = std::move(*tang_result);
        vbo_texcoords_ = std::move(*tc_result);
        vbo_colors_ = std::move(*col_result);
        ebo_ = std::move(*ebo_result);

        glBindVertexArray(vao_.get());

        glBindBuffer(GL_ARRAY_BUFFER, vbo_positions_.get());
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), nullptr);
        glEnableVertexAttribArray(0);

        glBindBuffer(GL_ARRAY_BUFFER, vbo_normals_.get());
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), nullptr);
        glEnableVertexAttribArray(1);

        glBindBuffer(GL_ARRAY_BUFFER, vbo_tangents_.get());
        glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), nullptr);
        glEnableVertexAttribArray(2);

        glBindBuffer(GL_ARRAY_BUFFER, vbo_texcoords_.get());
        glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), nullptr);
        glEnableVertexAttribArray(3);

        glBindBuffer(GL_ARRAY_BUFFER, vbo_colors_.get());
        glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), nullptr);
        glEnableVertexAttribArray(4);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_.get());

        glBindVertexArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

        initialized_ = true;
        return {};
    }

    Result<void> MeshRenderer::setupFBO(int width, int height) {
        assert(width > 0 && height > 0);

        GLuint color_tex;
        glGenTextures(1, &color_tex);
        glBindTexture(GL_TEXTURE_2D, color_tex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        color_texture_ = Texture(color_tex);

        GLuint depth_tex;
        glGenTextures(1, &depth_tex);
        glBindTexture(GL_TEXTURE_2D, depth_tex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, width, height, 0,
                     GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_NONE);
        depth_texture_ = Texture(depth_tex);

        GLuint fbo;
        glGenFramebuffers(1, &fbo);
        fbo_ = FBO(fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo_.get());
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, color_texture_.get(), 0);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depth_texture_.get(), 0);

        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            return std::unexpected("Mesh FBO incomplete");
        }

        fbo_width_ = width;
        fbo_height_ = height;

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        return {};
    }

    Result<void> MeshRenderer::setupShadowFBO(int resolution) {
        assert(resolution > 0);

        GLuint depth_tex;
        glGenTextures(1, &depth_tex);
        glBindTexture(GL_TEXTURE_2D, depth_tex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, resolution, resolution, 0,
                     GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
        const float border_color[] = {1.0f, 1.0f, 1.0f, 1.0f};
        glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, border_color);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);
        shadow_depth_texture_ = Texture(depth_tex);

        GLuint fbo;
        glGenFramebuffers(1, &fbo);
        shadow_fbo_ = FBO(fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, shadow_fbo_.get());
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, shadow_depth_texture_.get(), 0);
        glDrawBuffer(GL_NONE);
        glReadBuffer(GL_NONE);

        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            return std::unexpected("Shadow FBO incomplete");
        }

        shadow_map_resolution_ = resolution;

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        return {};
    }

    void MeshRenderer::resize(int width, int height) {
        if (width == fbo_width_ && height == fbo_height_)
            return;

        const auto result = setupFBO(width, height);
        if (!result)
            LOG_ERROR("Failed to resize mesh FBO: {}", result.error());
    }

    Result<void> MeshRenderer::uploadMeshData(const lfs::core::MeshData& mesh) {
        if (mesh.vertex_count() == uploaded_vertex_count_ &&
            mesh.face_count() == uploaded_face_count_ &&
            mesh.generation() == uploaded_generation_) {
            return {};
        }

        const auto cpu_verts = mesh.vertices.to(lfs::core::Device::CPU).contiguous();
        glBindBuffer(GL_ARRAY_BUFFER, vbo_positions_.get());
        glBufferData(GL_ARRAY_BUFFER,
                     cpu_verts.numel() * sizeof(float),
                     cpu_verts.ptr<float>(), GL_DYNAMIC_DRAW);

        if (mesh.has_normals()) {
            const auto cpu_normals = mesh.normals.to(lfs::core::Device::CPU).contiguous();
            glBindBuffer(GL_ARRAY_BUFFER, vbo_normals_.get());
            glBufferData(GL_ARRAY_BUFFER,
                         cpu_normals.numel() * sizeof(float),
                         cpu_normals.ptr<float>(), GL_DYNAMIC_DRAW);
        } else {
            glBindBuffer(GL_ARRAY_BUFFER, vbo_normals_.get());
            glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW);
        }

        if (mesh.has_tangents()) {
            const auto cpu_tangents = mesh.tangents.to(lfs::core::Device::CPU).contiguous();
            glBindBuffer(GL_ARRAY_BUFFER, vbo_tangents_.get());
            glBufferData(GL_ARRAY_BUFFER,
                         cpu_tangents.numel() * sizeof(float),
                         cpu_tangents.ptr<float>(), GL_DYNAMIC_DRAW);
        } else {
            glBindBuffer(GL_ARRAY_BUFFER, vbo_tangents_.get());
            glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW);
        }

        if (mesh.has_texcoords()) {
            const auto cpu_tc = mesh.texcoords.to(lfs::core::Device::CPU).contiguous();
            glBindBuffer(GL_ARRAY_BUFFER, vbo_texcoords_.get());
            glBufferData(GL_ARRAY_BUFFER,
                         cpu_tc.numel() * sizeof(float),
                         cpu_tc.ptr<float>(), GL_DYNAMIC_DRAW);
        } else {
            glBindBuffer(GL_ARRAY_BUFFER, vbo_texcoords_.get());
            glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW);
        }

        if (mesh.has_colors()) {
            const auto cpu_colors = mesh.colors.to(lfs::core::Device::CPU).contiguous();
            glBindBuffer(GL_ARRAY_BUFFER, vbo_colors_.get());
            glBufferData(GL_ARRAY_BUFFER,
                         cpu_colors.numel() * sizeof(float),
                         cpu_colors.ptr<float>(), GL_DYNAMIC_DRAW);
        } else {
            glBindBuffer(GL_ARRAY_BUFFER, vbo_colors_.get());
            glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW);
        }

        const auto cpu_idx = mesh.indices.to(lfs::core::Device::CPU).contiguous();
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_.get());
        glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                     cpu_idx.numel() * sizeof(int32_t),
                     cpu_idx.ptr<int32_t>(), GL_DYNAMIC_DRAW);

        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

        uploaded_vertex_count_ = mesh.vertex_count();
        uploaded_face_count_ = mesh.face_count();
        uploaded_generation_ = mesh.generation();

        return {};
    }

    void MeshRenderer::uploadTextures(const lfs::core::MeshData& mesh) {
        if (mesh.generation() == uploaded_texture_generation_ && !material_textures_.empty())
            return;

        material_textures_.clear();

        for (size_t mat_idx = 0; mat_idx < mesh.materials.size(); ++mat_idx) {
            const auto& mat = mesh.materials[mat_idx];
            GLMaterialTextures gl_tex;

            if (mat.albedo_tex > 0 && mat.albedo_tex <= mesh.texture_images.size()) {
                gl_tex.albedo = create_gl_texture(mesh.texture_images[mat.albedo_tex - 1]);
            }
            if (mat.normal_tex > 0 && mat.normal_tex <= mesh.texture_images.size()) {
                gl_tex.normal = create_gl_texture(mesh.texture_images[mat.normal_tex - 1]);
            }
            if (mat.metallic_roughness_tex > 0 && mat.metallic_roughness_tex <= mesh.texture_images.size()) {
                gl_tex.metallic_roughness = create_gl_texture(mesh.texture_images[mat.metallic_roughness_tex - 1]);
            }

            if (gl_tex.albedo.get() || gl_tex.normal.get() || gl_tex.metallic_roughness.get()) {
                material_textures_[mat_idx] = std::move(gl_tex);
            }
        }

        uploaded_texture_generation_ = mesh.generation();
        LOG_INFO("Uploaded {} material texture sets", material_textures_.size());
    }

    void MeshRenderer::bindMaterial(const lfs::core::Material& mat, size_t mat_idx, bool has_texcoords) {
        pbr_shader_->set_uniform("u_base_color", glm::vec4(mat.base_color));
        pbr_shader_->set_uniform("u_metallic", mat.metallic);
        pbr_shader_->set_uniform("u_roughness", mat.roughness);
        pbr_shader_->set_uniform("u_emissive", glm::vec3(mat.emissive));

        bool has_albedo = false;
        bool has_normal = false;
        bool has_mr = false;

        if (has_texcoords) {
            auto it = material_textures_.find(mat_idx);
            if (it != material_textures_.end()) {
                const auto& gl_tex = it->second;
                if (gl_tex.albedo.get()) {
                    glActiveTexture(GL_TEXTURE0);
                    glBindTexture(GL_TEXTURE_2D, gl_tex.albedo.get());
                    pbr_shader_->set_uniform("u_albedo_tex", 0);
                    has_albedo = true;
                }
                if (gl_tex.normal.get()) {
                    glActiveTexture(GL_TEXTURE1);
                    glBindTexture(GL_TEXTURE_2D, gl_tex.normal.get());
                    pbr_shader_->set_uniform("u_normal_tex", 1);
                    has_normal = true;
                }
                if (gl_tex.metallic_roughness.get()) {
                    glActiveTexture(GL_TEXTURE2);
                    glBindTexture(GL_TEXTURE_2D, gl_tex.metallic_roughness.get());
                    pbr_shader_->set_uniform("u_metallic_roughness_tex", 2);
                    has_mr = true;
                }
            }
        }

        pbr_shader_->set_uniform("u_has_albedo_tex", static_cast<int>(has_albedo));
        pbr_shader_->set_uniform("u_has_normal_tex", static_cast<int>(has_normal));
        pbr_shader_->set_uniform("u_has_metallic_roughness_tex", static_cast<int>(has_mr));
    }

    Result<void> MeshRenderer::render(const lfs::core::MeshData& mesh,
                                      const glm::mat4& model,
                                      const glm::mat4& view,
                                      const glm::mat4& projection,
                                      const glm::vec3& camera_pos,
                                      const MeshRenderOptions& opts,
                                      bool use_fbo,
                                      bool clear_fbo) {
        if (!initialized_)
            return std::unexpected("MeshRenderer not initialized");

        if (mesh.vertex_count() == 0 || mesh.face_count() == 0)
            return {};

        auto upload_result = uploadMeshData(mesh);
        if (!upload_result)
            return upload_result;

        if (!mesh.texture_images.empty())
            uploadTextures(mesh);

        glBindVertexArray(vao_.get());
        const auto enable_attrib = [](GLuint loc, bool has_data) {
            if (has_data)
                glEnableVertexAttribArray(loc);
            else
                glDisableVertexAttribArray(loc);
        };
        enable_attrib(0, true);
        enable_attrib(1, mesh.has_normals());
        enable_attrib(2, mesh.has_tangents());
        enable_attrib(3, mesh.has_texcoords());
        enable_attrib(4, mesh.has_colors());

        const glm::vec3 headlight_dir = glm::normalize(camera_pos);

        glm::mat4 light_vp(1.0f);
        if (opts.shadow_enabled && shadow_shader_.valid()) {
            const int res = opts.shadow_map_resolution;
            if (shadow_map_resolution_ != res)
                setupShadowFBO(res);

            if (shadow_fbo_.get()) {
                light_vp = compute_light_vp(mesh, model, headlight_dir);

                glBindFramebuffer(GL_FRAMEBUFFER, shadow_fbo_.get());
                glViewport(0, 0, shadow_map_resolution_, shadow_map_resolution_);
                glClear(GL_DEPTH_BUFFER_BIT);

                glEnable(GL_DEPTH_TEST);
                glEnable(GL_CULL_FACE);
                glCullFace(GL_FRONT);
                glEnable(GL_POLYGON_OFFSET_FILL);
                glPolygonOffset(1.1f, 4.0f);

                {
                    ShaderScope scope(shadow_shader_);
                    const glm::mat4 shadow_mvp = light_vp * model;
                    shadow_shader_->set_uniform("u_mvp", shadow_mvp);

                    glDrawElements(GL_TRIANGLES,
                                   static_cast<GLsizei>(mesh.face_count() * 3),
                                   GL_UNSIGNED_INT, nullptr);
                }

                glDisable(GL_POLYGON_OFFSET_FILL);
                glCullFace(GL_BACK);
                glBindFramebuffer(GL_FRAMEBUFFER, 0);
            }
        }

        if (use_fbo) {
            glBindFramebuffer(GL_FRAMEBUFFER, fbo_.get());
            glViewport(0, 0, fbo_width_, fbo_height_);

            if (clear_fbo) {
                glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            }
        }

        const GLboolean blend_was_enabled = glIsEnabled(GL_BLEND);
        glDisable(GL_BLEND);
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LESS);

        if (opts.backface_culling) {
            glEnable(GL_CULL_FACE);
            glCullFace(GL_BACK);
        } else {
            glDisable(GL_CULL_FACE);
        }

        {
            ShaderScope scope(pbr_shader_);

            const glm::mat3 normal_matrix = glm::inverseTranspose(glm::mat3(model));

            pbr_shader_->set_uniform("u_model", model);
            pbr_shader_->set_uniform("u_view", view);
            pbr_shader_->set_uniform("u_projection", projection);
            pbr_shader_->set_uniform("u_normal_matrix", normal_matrix);
            pbr_shader_->set_uniform("u_camera_pos", camera_pos);
            pbr_shader_->set_uniform("u_light_dir", headlight_dir);
            pbr_shader_->set_uniform("u_light_intensity", opts.light_intensity);
            pbr_shader_->set_uniform("u_ambient", opts.ambient);
            pbr_shader_->set_uniform("u_has_vertex_colors", static_cast<int>(mesh.has_colors()));

            const bool shadow_active = opts.shadow_enabled && shadow_fbo_.get() && shadow_depth_texture_.get();
            pbr_shader_->set_uniform("u_shadow_enabled", static_cast<int>(shadow_active));
            if (shadow_active) {
                pbr_shader_->set_uniform("u_light_vp", light_vp);
                glActiveTexture(GL_TEXTURE3);
                glBindTexture(GL_TEXTURE_2D, shadow_depth_texture_.get());
                pbr_shader_->set_uniform("u_shadow_map", 3);
            }

            if (mesh.submeshes.empty()) {
                const auto& mat = mesh.materials.empty() ? lfs::core::Material{} : mesh.materials[0];
                bindMaterial(mat, 0, mesh.has_texcoords());

                glDrawElements(GL_TRIANGLES,
                               static_cast<GLsizei>(mesh.face_count() * 3),
                               GL_UNSIGNED_INT, nullptr);
            } else {
                for (const auto& sub : mesh.submeshes) {
                    assert(sub.material_index < mesh.materials.size() || mesh.materials.empty());
                    const auto& mat = mesh.materials.empty()
                                          ? lfs::core::Material{}
                                          : mesh.materials[std::min(sub.material_index, mesh.materials.size() - 1)];

                    bindMaterial(mat, sub.material_index, mesh.has_texcoords());

                    if (mat.double_sided)
                        glDisable(GL_CULL_FACE);
                    else if (opts.backface_culling)
                        glEnable(GL_CULL_FACE);

                    const auto byte_offset = static_cast<GLintptr>(sub.start_index * sizeof(uint32_t));
                    glDrawElements(GL_TRIANGLES,
                                   static_cast<GLsizei>(sub.index_count),
                                   GL_UNSIGNED_INT,
                                   reinterpret_cast<const void*>(byte_offset));
                }
            }
        }

        if (opts.wireframe_overlay) {
            ShaderScope scope(wireframe_shader_);

            const glm::mat4 mvp = projection * view * model;
            wireframe_shader_->set_uniform("u_mvp", mvp);
            wireframe_shader_->set_uniform("u_color", opts.wireframe_color);

            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
            glLineWidth(std::min(opts.wireframe_width, 10.0f));
            glEnable(GL_POLYGON_OFFSET_LINE);
            glPolygonOffset(-1.0f, -1.0f);

            glDrawElements(GL_TRIANGLES,
                           static_cast<GLsizei>(mesh.face_count() * 3),
                           GL_UNSIGNED_INT, nullptr);

            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
            glDisable(GL_POLYGON_OFFSET_LINE);
            glLineWidth(1.0f);
        }

        glBindVertexArray(0);
        glDisable(GL_CULL_FACE);

        if (blend_was_enabled)
            glEnable(GL_BLEND);

        if (use_fbo) {
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
        }

        return {};
    }

    void MeshRenderer::blitToScreen(const glm::ivec2& dst_pos, const glm::ivec2& dst_size) {
        assert(fbo_.get() != 0);

        glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo_.get());
        glReadBuffer(GL_COLOR_ATTACHMENT0);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);

        glBlitFramebuffer(0, 0, fbo_width_, fbo_height_,
                          dst_pos.x, dst_pos.y,
                          dst_pos.x + dst_size.x, dst_pos.y + dst_size.y,
                          GL_COLOR_BUFFER_BIT, GL_LINEAR);

        glBlitFramebuffer(0, 0, fbo_width_, fbo_height_,
                          dst_pos.x, dst_pos.y,
                          dst_pos.x + dst_size.x, dst_pos.y + dst_size.y,
                          GL_DEPTH_BUFFER_BIT, GL_NEAREST);

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

} // namespace lfs::rendering
